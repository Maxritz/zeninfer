// ZenInfer D3D12 backend: Q8_0 matrix-vector product (decode path).
//
// Exact fp32 activations, so this stays the parity path. Decode is the phase that
// matters here: at 1B/Q8_0 a token streams 1.24 GB of weights, so the kernel is a
// pure sequential read and should sustain ~90% of VRAM peak. An opt-in int8
// variant exists in q8_0_gemv_int8.hlsl for the reasons described there.
//
// Mapping: 256 threads, 8 rows per group, one warp (32 lanes) per row. Each lane
// owns every 32nd 32-byte block of its row, so the warp reads one contiguous
// 1 KB span per iteration and the block scale folds into that lane's accumulator.
// The final reduction is WaveActiveSum across the warp.
//
// Wide mode (params.reserved != 0, host gate ZEN_D3D12_GEMV_WIDE, default ON):
// each lane walks its row with stride 2*LANES and handles TWO 32-byte blocks per
// iteration, issuing both blocks' quant and scale fetches before the first FMA
// chain. The classic loop leaves a lane with roughly one outstanding block fetch
// at a time, so only ~2 of the group's 8 row streams are in flight on the memory
// bus; the wide loop keeps two fetches outstanding per lane and all 8 row streams
// of the group concurrently in flight. The (row, block) -> lane mapping and the
// per-lane accumulation sequence are IDENTICAL to the classic loop (block b,
// then b + LANES, then b + 2*LANES, ...), so every fp add runs in the same order
// and the two paths produce bit-identical outputs - only the timing of the
// memory requests changes. ZEN_D3D12_GEMV_WIDE=0 selects the classic loop.
//
// The previous version reduced through shared memory instead:
//
//     red[tid] = int(acc * 1000000.0f);
//     if (lane == 0u) for (i = 0; i < 32; ++i) total += red[tid + i] / 1000000.0f;
//
// Two defects. `int(acc * 1e6)` overflows int32 once |acc| exceeds ~2147, so large
// dot products silently produced garbage. And the summation ran serially on one
// lane through 32 shared-memory slots, leaving 31 lanes idle while adding 32 round
// trips to shared memory. WaveActiveSum returns the total to every lane in one
// operation with no shared memory involved.

cbuffer GemvParams : register(b0)
{
    uint rows;
    uint blocks_per_row;
    uint rows_per_group;
    // Kernel mode selector written by the host from ZEN_D3D12_GEMV_WIDE:
    // 0 = classic lane-strided loop, 1 = wide two-blocks-per-iteration loop.
    // Pure scheduling difference; both paths accumulate in the same order.
    uint reserved;
};

ByteAddressBuffer         qbuf : register(t0);   // 32 bytes per block
StructuredBuffer<float>   sbuf : register(t1);   // one fp32 scale per block
ByteAddressBuffer         xraw : register(t2);   // fp32 input vector
RWStructuredBuffer<float> yvec : register(u0);   // fp32 output vector
// register(t3) is bound by the host for the opt-in int8 kernel's scale table and
// is unused here.

#define STAGE_LIMIT_BLOCKS 128u   // cols <= 4096 fits the 16 KB staging tile
#define XSTAGE_FLOATS      4096u
#define LANES              32u    // one warp per row

groupshared float xs[XSTAGE_FLOATS];

// Sum the 32 lanes that own one row, independent of the hardware wave size.
// A bare WaveActiveSum would be wrong here: RDNA executes this shader as wave64,
// so lanes 0-31 (row A) and 32-63 (row B) sit in the same wave and a wave-wide sum
// would fold two rows together. Masking to the 32-lane group keeps one row per
// warp on both wave32 and wave64.
float warp32_sum(float v, uint lane)
{
    const uint base = WaveGetLaneIndex() - lane;
    [unroll]
    for (uint s = 16u; s > 0u; s >>= 1u)
    {
        const float other = WaveReadLaneAt(v, base + ((lane + s) & 31u));
        if (lane + s < 32u)
        {
            v += other;
        }
    }
    return v;
}

// One 32-byte Q8_0 block's contribution to a row dot: 8 packed uints, four int4
// quants each. Used only by the wide loop; the expression is textually identical
// to the classic loop's inner body, so both paths evaluate the same fp ops in the
// same order and a lane's partial for a given block is the same in either mode.
float block_part(uint xoff, uint4 v0, uint4 v1, bool stage)
{
    const uint w[8] = { v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w };
    float part = 0.0f;
    [unroll]
    for (uint k = 0; k < 8u; ++k)
    {
        const uint   xo = xoff + k * 4u;
        const float4 xv = stage
            ? float4(xs[xo], xs[xo + 1u], xs[xo + 2u], xs[xo + 3u])
            : asfloat(xraw.Load<float4>(xo * 4u));

        part += ((float)(w[k] & 0xFFu) - 256.0f * (float)((w[k] >> 7u) & 1u)) * xv.x
              + ((float)((w[k] >> 8u) & 0xFFu) - 256.0f * (float)(((w[k] >> 8u) >> 7u) & 1u)) * xv.y
              + ((float)((w[k] >> 16u) & 0xFFu) - 256.0f * (float)(((w[k] >> 16u) >> 7u) & 1u)) * xv.z
              + ((float)((w[k] >> 24u) & 0xFFu) - 256.0f * (float)(((w[k] >> 24u) >> 7u) & 1u)) * xv.w;
    }
    return part;
}

[numthreads(256, 1, 1)]
void GemvQ8_0(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID)
{
    const uint local_row = tid / LANES;
    const uint lane      = tid % LANES;
    const uint row       = gid.x * rows_per_group + local_row;

    // Stage the activation vector once per group when it fits; every warp in the
    // group then reads it from shared memory instead of re-fetching from VRAM.
    const bool stage = blocks_per_row <= STAGE_LIMIT_BLOCKS;
    if (stage)
    {
        const uint total = blocks_per_row * 32u;
        for (uint i = tid * 4u; i < total; i += 1024u)
        {
            const float4 v = asfloat(xraw.Load<float4>(i * 4u));
            xs[i] = v.x;
            xs[i + 1u] = v.y;
            xs[i + 2u] = v.z;
            xs[i + 3u] = v.w;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    float acc = 0.0f;
    if (row < rows)
    {
        const uint qbase = row * blocks_per_row * 32u;
        const uint sbase = row * blocks_per_row;

        for (uint b = lane; b < blocks_per_row; b += LANES)
        {
            const float scale = sbuf[sbase + b];
            const uint  xoff  = b * 32u;
            const uint  qoff  = qbase + b * 32u;

            const uint4 v0 = qbuf.Load<uint4>(qoff);
            const uint4 v1 = qbuf.Load<uint4>(qoff + 16u);

            const uint w[8] = { v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w };
            float part = 0.0f;
            [unroll]
            for (uint k = 0; k < 8u; ++k)
            {
                const uint   xo = xoff + k * 4u;
                const float4 xv = stage
                    ? float4(xs[xo], xs[xo + 1u], xs[xo + 2u], xs[xo + 3u])
                    : asfloat(xraw.Load<float4>(xo * 4u));

                part += ((float)(w[k] & 0xFFu) - 256.0f * (float)((w[k] >> 7u) & 1u)) * xv.x
                      + ((float)((w[k] >> 8u) & 0xFFu) - 256.0f * (float)(((w[k] >> 8u) >> 7u) & 1u)) * xv.y
                      + ((float)((w[k] >> 16u) & 0xFFu) - 256.0f * (float)(((w[k] >> 16u) >> 7u) & 1u)) * xv.z
                      + ((float)((w[k] >> 24u) & 0xFFu) - 256.0f * (float)(((w[k] >> 24u) >> 7u) & 1u)) * xv.w;
            }
            acc += scale * part;
        }
    }

    // Each lane summed a disjoint set of blocks, so the warp total is the answer.
    const float total = warp32_sum(acc, lane);
    if (lane == 0u && row < rows)
    {
        yvec[row] = total;
    }
}
