// ZenInfer D3D12 backend: Q8_0 batched matrix product, shared-memory tiled (prefill).
//
// Same math and the same root bindings as q8_0_gemm.hlsl (fp32 activations, one
// fp32 scale per 32-element weight block), but a different work decomposition.
//
// Why this exists
// ---------------
// q8_0_gemm.hlsl gives one warp to one output row and one accumulator per token:
//
//     acc[32];                                   // 32 tokens per lane, 1 row
//     for (block ...) {
//         q = dequant(load 4 bytes of this row); // one 4-byte global load per lane
//         for (t = 0; t < 32; ++t)
//             acc[t] += q * xs[t][col];          // 1 shared load per FFMA
//     }
//
// Two things cap it. Each FFMA pays a shared-memory read, so the inner loop issues
// 1 LDS + 1 FFMA and can never exceed 50% FFMA issue; and the weight load for every
// 32-element block sits in the dependency chain of the following 32 FFMAs, so a
// ~300-cycle memory latency is hidden by at most 32 cycles of work per wave. Measured
// with zen_d3d12_gemm_bench at the real prefill shapes: 3.15 TFLOP/s, i.e. ~8% of this
// part's fp32 rate, and the whole prefill is dominated by it.
//
// This kernel is a conventional register tile instead: one workgroup computes a
// 64-row x 64-token output tile, each thread owns a 4x4 micro-tile, and every value it
// needs is staged through shared memory once per K step. Per K step a thread issues 4
// activation reads + 4 weight reads (broadcast: every thread in a 16-thread group
// reads the same 4 weights) against 16 FFMAs, and the weight load latency is amortised
// over 16 tokens x 4 rows instead of one row.
//
// The K order per output element is still ascending block index, and the scale is
// folded into the weight during staging (instead of multiplying the product by it per
// block as v1 did), so results agree with the CPU reference to the same tolerance the
// parity test enforces rather than bit-for-bit with v1.

cbuffer GemmParams : register(b0)
{
    uint rows;
    uint blocks_per_row;
    uint token_count;
    uint tokens_per_tile;
};

ByteAddressBuffer         qbuf   : register(t0);  // 32 bytes per Q8_0 block
StructuredBuffer<float>   sbuf   : register(t1);  // one fp32 scale per block
ByteAddressBuffer         xraw   : register(t2);  // [token][columns] float32
StructuredBuffer<float>   xscale : register(t3);  // unused here, keeps the root layout
RWStructuredBuffer<float> yvec   : register(u0);  // [token][rows]

#define TM 64u              // tokens per workgroup tile
#define RM 64u              // rows per workgroup tile
#define KC 32u              // columns per K step: exactly one Q8_0 weight block
// Strides are padded to keep the compute loop's shared reads conflict-free: weights
// step by one bank per row, activations by one bank per token.
#define WSTRIDE (KC + 1u)   // 33 floats per row
#define XSTRIDE (TM + 4u)   // 68 floats per K row; multiple of 4 so a token's 4 floats
                            // are 16-byte aligned and load as one LDS.128

groupshared float ws[RM][WSTRIDE];   // dequantised weight block [row][k]
groupshared float xs[KC][XSTRIDE];   // activation block [k][token]

float signed_byte(uint packed, uint byte_index)
{
    const uint b = (packed >> (byte_index * 8u)) & 0xFFu;
    return (float)b - 256.0f * (float)(b >> 7u);
}

[numthreads(256, 1, 1)]
void GemmQ8_0T(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID)
{
    const uint row0 = gid.x * RM;
    const uint tok0 = gid.y * TM;
    const uint cols = blocks_per_row * 32u;

    // Micro-tile: 4 rows x 4 tokens per thread. rm picks the row quarter (16 threads
    // share it, so their weight reads broadcast), tn picks the token quarter.
    const uint rm = tid >> 4u;
    const uint tn = tid & 15u;
    const uint r_base = rm * 4u;
    const uint t_base = tn * 4u;

    float acc[4][4];
    [unroll]
    for (uint i = 0; i < 4; ++i)
    {
        [unroll]
        for (uint j = 0; j < 4; ++j)
        {
            acc[i][j] = 0.0f;
        }
    }

    // Staging assignment: thread t stages row (t >> 2) and its 8 consecutive columns,
    // which is 8 of the 32 bytes of one block, and token (t >> 2) with 8 columns, two
    // float4s. Four consecutive threads therefore cover a whole 32-byte block and a
    // whole 32-float activation row, so both loads are coalesced.
    const uint s_rc = tid >> 2u;             // 0..63: row for weights, token for acts
    const uint s_col = (tid & 3u) * 8u;      // 0, 8, 16, 24
    const bool s_row_ok = (row0 + s_rc) < rows;
    const bool s_tok_ok = (tok0 + s_rc) < token_count;

    for (uint b = 0; b < blocks_per_row; ++b)
    {
        // ---- stage the weight block (dequantised, per-block scale folded in)
        float wv[8];
        float wscale = 0.0f;
        if (s_row_ok)
        {
            const uint wrow = row0 + s_rc;
            const uint wbase = (wrow * blocks_per_row + b) * 32u + s_col;
            const uint2 p = qbuf.Load<uint2>(wbase);
            wscale = sbuf[wrow * blocks_per_row + b];
            [unroll]
            for (uint j = 0; j < 4; ++j)
            {
                wv[j] = signed_byte(p.x, j);
                wv[j + 4u] = signed_byte(p.y, j);
            }
        }
        else
        {
            [unroll]
            for (uint j = 0; j < 8; ++j)
            {
                wv[j] = 0.0f;
            }
        }
        [unroll]
        for (uint j = 0; j < 8; ++j)
        {
            ws[s_rc][s_col + j] = wv[j] * wscale;
        }

        // ---- stage the activation block
        float xv[8];
        if (s_tok_ok)
        {
            const uint xbase = ((tok0 + s_rc) * cols + b * 32u + s_col) * 4u;
            const float4 a = asfloat(xraw.Load<float4>(xbase));
            const float4 c = asfloat(xraw.Load<float4>(xbase + 16u));
            xv[0] = a.x; xv[1] = a.y; xv[2] = a.z; xv[3] = a.w;
            xv[4] = c.x; xv[5] = c.y; xv[6] = c.z; xv[7] = c.w;
        }
        else
        {
            [unroll]
            for (uint j = 0; j < 8; ++j)
            {
                xv[j] = 0.0f;
            }
        }
        [unroll]
        for (uint j = 0; j < 8; ++j)
        {
            xs[s_col + j][s_rc] = xv[j];
        }

        GroupMemoryBarrierWithGroupSync();

        // ---- accumulate this block's 32 columns into the 4x4 micro-tile
        // Fully unrolled: measured at 5.5 TFLOP/s against 5.2 for a [loop] version, so
        // the 16 accumulators do not spill and the extra scheduling freedom wins.
        for (uint k = 0; k < KC; ++k)
        {
            const float x0 = xs[k][t_base + 0u];
            const float x1 = xs[k][t_base + 1u];
            const float x2 = xs[k][t_base + 2u];
            const float x3 = xs[k][t_base + 3u];
            const float w0 = ws[r_base + 0u][k];
            const float w1 = ws[r_base + 1u][k];
            const float w2 = ws[r_base + 2u][k];
            const float w3 = ws[r_base + 3u][k];
            // acc[i][j] is (row r_base+i, token t_base+j): row i multiplies the four
            // tokens. Keeping the row as the first index is what makes the epilogue's
            // acc[i][j] -> yvec[tok*rows + row] mapping the same order.
            acc[0][0] += x0 * w0; acc[0][1] += x1 * w0; acc[0][2] += x2 * w0; acc[0][3] += x3 * w0;
            acc[1][0] += x0 * w1; acc[1][1] += x1 * w1; acc[1][2] += x2 * w1; acc[1][3] += x3 * w1;
            acc[2][0] += x0 * w2; acc[2][1] += x1 * w2; acc[2][2] += x2 * w2; acc[2][3] += x3 * w2;
            acc[3][0] += x0 * w3; acc[3][1] += x1 * w3; acc[3][2] += x2 * w3; acc[3][3] += x3 * w3;
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // acc[i][j] is row r_base+i, token t_base+j.
    [unroll]
    for (uint i = 0; i < 4; ++i)
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
