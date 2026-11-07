// ZenInfer D3D12 backend: Q8_0 batched matrix product, integer domain (prefill).
//
// No dequantisation of weights. The raw Q8_0 bytes are copied into shared memory as
// they are; activations are converted to int8 once per 32-element block; every
// multiply-accumulate runs in dot4add_i8packed. The only float arithmetic left is the
// block epilogue: one scale pair per output element per block, i.e. 32 float
// multiply-adds per 1024 integer multiply-accumulates.
//
// Measured chain that led here, zen_d3d12_gemm_bench, one layer's four matmuls:
//   q8_0_gemm_t.hlsl      fp32, dequant in the inner loop, scalar shared reads   4.42 TFLOP/s
//   q8_0_gemm_dp4a.hlsl   dp4a, one warp per row, host-quantised activations     0.98 TFLOP/s
//   this kernel, 64x64 tile, 4x4 accumulators, K step = 32 columns               5.37 TFLOP/s
//   this kernel, 64x64 tile, 4x4 accumulators, K step = 128 columns              4.54 TFLOP/s
// The 128-column K step amortised three barriers over four times the work and came
// out *slower*: it needed 23.5 KB of shared memory per workgroup, which cost about
// four times the residency. So this kernel is latency-bound, not barrier-bound, and
// the lever is work per thread at a small footprint - hence an 8-row x 4-token micro
// tile (32 accumulators, 8.2 KB shared) on a 128-row x 64-token tile.
//
// Plain Direct3D 12: [numthreads(16,16)], no wave intrinsics, no lane or wave-size
// assumptions. Shader Model 6.4 is required only for dot4add_i8packed; this shader
// alone is built at cs_6_4 and a device below that simply fails to build the PSO, so
// the caller keeps the fp32 tile.
//
// Accuracy: activations are int8 with one scale per 32-element block (max|x|/127), the
// same granularity the weight blocks already use, so the error is bounded by the int8
// activation step rather than by the reduction. Not bit-parity with the fp32 path.

cbuffer GemmParams : register(b0)
{
    uint rows;
    uint blocks_per_row;
    uint token_count;
    uint tokens_per_tile;
};

ByteAddressBuffer         qbuf   : register(t0);  // 32 raw bytes per Q8_0 block
StructuredBuffer<float>   sbuf   : register(t1);  // one fp32 scale per weight block
ByteAddressBuffer         xraw   : register(t2);  // [token][columns] float32
StructuredBuffer<float>   xscale : register(t3);  // unused here, keeps the root layout
RWStructuredBuffer<float> yvec   : register(u0);  // [token][rows]

#define TM 64u                     // tokens per workgroup tile
#define RM 128u                    // rows per workgroup tile
#define KC 32u                     // columns per K step: exactly one Q8_0 block
#define KG (KC / 4u)               // 8 four-byte groups: one dp4a each
#define WSTRIDE (RM + 4u)          // 132 uints per group row
#define XSTRIDE (TM + 4u)          // 68 uints per group row

groupshared uint  ws[KG][WSTRIDE];   // raw weight groups [group][row]
groupshared uint  xs[KG][XSTRIDE];   // int8 activation groups [group][token]
groupshared float wsc[RM];           // weight block scale per row
groupshared float xsc[TM];           // activation block scale per token
groupshared float xamax[4][TM];      // partial |x| max, four threads per token

[numthreads(16, 16, 1)]
void GemmQ8_0I8(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    // SV_GroupThreadID is uint3. Declaring it as a scalar uint silently binds only .x,
    // so in a 2-D group every thread would read tid 0..15, ty would always be zero and
    // three quarters of the tile would never be written - which is exactly how this
    // kernel first failed: rows 8..127 came back zero and the uninitialised half of
    // the scale table came back as Inf. Flatten it explicitly.
    const uint tid = gtid.x + gtid.y * 16u;
    const uint tx = tid & 15u;              // token group: four tokens
    const uint ty = tid >> 4u;              // row group: eight rows
    const uint t_base = tx * 4u;
    const uint r_base = ty * 8u;

    const uint row0 = gid.x * RM;
    const uint tok0 = gid.y * TM;
    const uint cols = blocks_per_row * 32u;

    // Weights: thread t stages row (t >> 1), 16 bytes of it - two threads cover one
    // 32-byte block, so the load is a 16-byte vector and consecutive threads read
    // consecutive 16-byte slices.
    const uint w_srow = tid >> 1u;          // 0..127
    const uint w_soff = (tid & 1u) * 16u;   // 0 or 16 bytes
    const uint w_g0 = w_soff >> 2u;         // group 0 or 4
    // Activations: thread t stages token (t >> 2), 8 columns of it - four threads
    // cover one 32-float row.
    const uint x_stok = tid >> 2u;          // 0..63
    const uint x_scol = (tid & 3u) * 8u;    // 0, 8, 16, 24
    const uint x_g0 = x_scol >> 2u;         // 0, 2, 4, 6

    const uint w_row = row0 + w_srow;
    const uint x_tok = tok0 + x_stok;
    const bool w_ok = w_row < rows;
    const bool x_ok = x_tok < token_count;
    const uint x_row_va = (x_tok * cols) * 4u;

    float acc[8][4];
    [unroll]
    for (uint i = 0; i < 8; ++i)
    {
        [unroll]
        for (uint j = 0; j < 4; ++j)
        {
            acc[i][j] = 0.0f;
        }
    }

    for (uint b = 0; b < blocks_per_row; ++b)
    {
        // ---- weights: raw bytes straight into shared, group-major
        uint4 wp = uint4(0u, 0u, 0u, 0u);
        float wscale = 0.0f;
        // Probe 3: drop the global weight/activation loads (lane-varying values instead,
        // so nothing folds) while keeping the barriers, quantise, epilogue and stores.
#if I8_ABLATE_NO_GLOBAL
        wp = uint4(0x01010101u * (b + 1u) + tid, 0x02020202u ^ b, 0x03030303u + tid, 0x04040404u * b);
        wscale = 0.5f;
#else
        if (w_ok)
        {
            wp = qbuf.Load<uint4>((w_row * blocks_per_row + b) * 32u + w_soff);
            wscale = sbuf[w_row * blocks_per_row + b];
        }
#endif
        ws[w_g0 + 0u][w_srow] = wp.x;
        ws[w_g0 + 1u][w_srow] = wp.y;
        ws[w_g0 + 2u][w_srow] = wp.z;
        ws[w_g0 + 3u][w_srow] = wp.w;
        wsc[w_srow] = wscale;

        // ---- activations, pass 1: per-thread partial |x| maximum
        float amax = 0.0f;
        if (x_ok)
        {
#if I8_ABLATE_NO_GLOBAL
            const float4 a = float4(0.25f + b, 0.5f, 0.75f, 1.0f);
            const float4 c = float4(1.25f, 1.5f, 1.75f, 2.0f + tid);
#else
            const uint xb = x_row_va + (b * 32u + x_scol) * 4u;
            const float4 a = asfloat(xraw.Load<float4>(xb));
            const float4 c = asfloat(xraw.Load<float4>(xb + 16u));
#endif
            const float4 v4[2] = {a, c};
            [unroll]
            for (uint q = 0; q < 2; ++q)
            {
                [unroll]
                for (uint j = 0; j < 4; ++j)
                {
                    const float v = v4[q][j];
                    const float m = v < 0.0f ? -v : v;
                    amax = m > amax ? m : amax;
                }
            }
        }
        xamax[tid & 3u][x_stok] = amax;
        GroupMemoryBarrierWithGroupSync();

        // ---- pass 2: reduce the four partial maxima, quantise, pack into groups.
        // Re-reading the slab is a cache hit and keeps eight floats out of the live set
        // across the barrier.
        const float block_amax = max(max(xamax[0][x_stok], xamax[1][x_stok]),
                                     max(xamax[2][x_stok], xamax[3][x_stok]));
        const float inv = block_amax > 0.0f ? 127.0f / block_amax : 0.0f;
        xsc[x_stok] = block_amax > 0.0f ? block_amax / 127.0f : 0.0f;

        float xv[8];
        [unroll]
        for (uint j = 0; j < 8; ++j)
        {
            xv[j] = 0.0f;
        }
        if (x_ok)
        {
            const uint xb = x_row_va + (b * 32u + x_scol) * 4u;
            const float4 a = asfloat(xraw.Load<float4>(xb));
            const float4 c = asfloat(xraw.Load<float4>(xb + 16u));
            xv[0] = a.x; xv[1] = a.y; xv[2] = a.z; xv[3] = a.w;
            xv[4] = c.x; xv[5] = c.y; xv[6] = c.z; xv[7] = c.w;
        }

        uint xq0 = 0u;
        uint xq1 = 0u;
        [unroll]
        for (uint j = 0; j < 4; ++j)
        {
            const float v0 = min(max(xv[j] * inv, -127.0f), 127.0f);
            const float v1 = min(max(xv[j + 4u] * inv, -127.0f), 127.0f);
            xq0 |= ((uint)((int)round(v0) & 0xFF)) << (j * 8u);
            xq1 |= ((uint)((int)round(v1) & 0xFF)) << (j * 8u);
        }
        xs[x_g0 + 0u][x_stok] = xq0;
        xs[x_g0 + 1u][x_stok] = xq1;

        GroupMemoryBarrierWithGroupSync();

        // ---- this block's 32 columns into the 8x4 micro-tile, all in int32
        int dot[8][4];
        [unroll]
        for (uint i = 0; i < 8; ++i)
        {
            [unroll]
            for (uint j = 0; j < 4; ++j)
            {
                dot[i][j] = 0;
            }
        }
// Measurement trap: I8_ABLATE_NO_LDS=1 replaces the per-group shared reads with
// lane-varying values computed in registers, keeping every dot4add and the dependency
// structure intact. If the kernel gets much faster the inner loop is bound by shared
// memory issue bandwidth; if it barely moves, dot4add issue is the limit. Compiled out
// (#if 0) in the shipped kernel.
#ifndef I8_ABLATE_NO_LDS
#define I8_ABLATE_NO_LDS 0
#define I8_ABLATE_FEWER_DOTS 0
#define I8_ABLATE_NO_GLOBAL 0
#endif

        [unroll]
        for (uint g = 0; g < KG; ++g)
        {
#if I8_ABLATE_NO_LDS
            const int x0 = (int)(0x02020202u * (g + 1u) + tid);
            const int x1 = (int)(0x03030303u ^ (g * 5u + tid));
            const int x2 = (int)(0x04040404u + g * 9u + tid);
            const int x3 = (int)(0x05050505u ^ (g + tid));
#else
            const int x0 = (int)xs[g][t_base + 0u];
            const int x1 = (int)xs[g][t_base + 1u];
            const int x2 = (int)xs[g][t_base + 2u];
            const int x3 = (int)xs[g][t_base + 3u];
#endif
            [unroll]
            for (uint i = 0; i < 8; ++i)
            {
#if I8_ABLATE_NO_LDS
                const int w = (int)(0x01010101u * (g + 2u) + i);
#else
                const int w = (int)ws[g][r_base + i];
#endif
                dot[i][0] = dot4add_i8packed(w, x0, dot[i][0]);
                // Probe 2: quarter the dot4add issue count, keep the rest of the block
                // (barriers, epilogue, staging) identical, to split dot issue from the
                // fixed per-block cost. Wrong results on purpose; never shipped.
#if !I8_ABLATE_FEWER_DOTS
                dot[i][1] = dot4add_i8packed(w, x1, dot[i][1]);
                dot[i][2] = dot4add_i8packed(w, x2, dot[i][2]);
                dot[i][3] = dot4add_i8packed(w, x3, dot[i][3]);
#endif
            }
        }

        // ---- one scale pair per output element per block
        [unroll]
        for (uint i = 0; i < 8; ++i)
        {
            const float wsc_i = wsc[r_base + i];
            acc[i][0] += (float)dot[i][0] * (wsc_i * xsc[t_base + 0u]);
            acc[i][1] += (float)dot[i][1] * (wsc_i * xsc[t_base + 1u]);
            acc[i][2] += (float)dot[i][2] * (wsc_i * xsc[t_base + 2u]);
            acc[i][3] += (float)dot[i][3] * (wsc_i * xsc[t_base + 3u]);
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // acc[i][j] is row r_base+i, token t_base+j.
    [unroll]
    for (uint i = 0; i < 8; ++i)
    {
        const uint row = row0 + r_base + i;
        if (row >= rows)
        {
            continue;
        }
        [unroll]
        for (uint j = 0; j < 4; ++j)
        {
            const uint tok = tok0 + t_base + j;
            if (tok < token_count)
            {
                yvec[tok * rows + row] = acc[i][j];
            }
        }
    }
}
