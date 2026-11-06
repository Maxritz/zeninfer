// ZenInfer D3D12 backend: Q8_0 batched matrix product v2 (prefill path).
//
// Same math as q8_0_gemm.hlsl (fp32 activations, parity): one warp owns one
// output row, lanes stride the 32 elements of each weight block, per-block
// scale folds into the lane accumulator, single warp sum at the end. K order
// per output is therefore identical to v1, so outputs are bit-identical.
//
// What changed (occupancy + issue pressure, not math):
//
//   * numthreads 1024 -> 256, ROWS_PER_GROUP 32 -> 8. A 1024-thread group
//     leaves little headroom for register pressure (acc[32] per thread) and
//     makes the per-chunk barrier synchronize 32 warps. 8 warps per group is
//     the same shape the decode GEMV uses and leaves room for several
//     resident groups per CU.
//   * BLOCK_T 32 -> 16 (tokens per tile). Halves the accumulator (32 -> 16
//     registers) which was the occupancy limiter, at the cost of 8 passes over
//     the weights for a 128-token prefill instead of 4. Prefill is
//     instruction-bound (cutting weight traffic 4x moved it 6%), so the extra
//     passes are cheap and the occupancy wins.
//   * CHUNK_BLOCKS 8 -> 16 (512 columns per stage, same 32 KB shared):
//     fewer chunks per tile (4 instead of 8 for 2048 cols), so fewer barrier
//     pairs per output. Barrier count per call is unchanged overall
//     (tgroups double, chunks halve), but each barrier now syncs 8 warps
//     instead of 32.
//   * Activation staging loads float4 (8.1k scalars -> 2k vector loads) and
//     uses shift-based index math (CHUNK_COLS is a power of two).

cbuffer GemmParams : register(b0)
{
    uint rows;
    uint blocks_per_row;
    uint token_count;
    uint tokens_per_tile;
};

ByteAddressBuffer         qbuf   : register(t0);  // 32 bytes per block
StructuredBuffer<float>   sbuf   : register(t1);  // one fp32 scale per block
ByteAddressBuffer         xraw   : register(t2);  // [token][columns] float32
StructuredBuffer<float>   xscale : register(t3);  // unused here, keeps the root layout
RWStructuredBuffer<float> yvec   : register(u0);  // [token][rows]

#define ROWS_PER_GROUP 8u     // one warp per row, 8 warps per 256-thread group
#define BLOCK_T        16u    // tokens accumulated per thread block
#define CHUNK_BLOCKS   16u    // 32-byte weight blocks staged per pass
#define CHUNK_COLS     (CHUNK_BLOCKS * 32u)

// BLOCK_T x CHUNK_COLS float32 activations (16 x 512 x 4 B = 32 KB).
groupshared float xs[BLOCK_T][CHUNK_COLS];

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

float signed_byte(uint packed, uint byte_index)
{
    const uint b = (packed >> (byte_index * 8u)) & 0xFFu;
    return (float)b - 256.0f * (float)(b >> 7u);
}

[numthreads(256, 1, 1)]
void GemmQ8_0V2(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID)
{
    const uint warp = tid >> 5u;          // row inside the 8-row block
    const uint lane = tid & 31u;          // element inside every 32-wide block
    const uint row  = gid.x * ROWS_PER_GROUP + warp;
    const uint word_i = lane >> 2u;
    const bool active = row < rows;

    const uint chunks  = (blocks_per_row + CHUNK_BLOCKS - 1u) / CHUNK_BLOCKS;
    const uint tgroups = (token_count + BLOCK_T - 1u) / BLOCK_T;

    for (uint tg = 0; tg < tgroups; ++tg)
    {
        const uint tbase = tg * BLOCK_T;

        float acc[BLOCK_T];
        [unroll]
        for (uint t = 0; t < BLOCK_T; ++t)
        {
            acc[t] = 0.0f;
        }

        for (uint chunk = 0; chunk < chunks; ++chunk)
        {
            const uint col0 = chunk * CHUNK_COLS;

            // Stage this token group's activation slab once, shared by all 8 rows.
            // Vectorized float4 loads; CHUNK_COLS=512 so the (t, c) split is a
            // shift, not a division.
            for (uint i = tid; i < BLOCK_T * (CHUNK_COLS / 4u); i += 256u)
            {
                const uint t   = i >> 7u;               // /128 float4 per token row
                const uint c4  = i & 127u;              // %128
                const uint tok = tbase + t;
                float4 v = float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (tok < token_count && col0 + c4 * 4u < blocks_per_row * 32u)
                {
                    v = asfloat(xraw.Load<float4>((tok * blocks_per_row * 32u + col0 + c4 * 4u) * 4u));
                }
                const uint c = c4 * 4u;
                xs[t][c] = v.x;
                xs[t][c + 1u] = v.y;
                xs[t][c + 2u] = v.z;
                xs[t][c + 3u] = v.w;
            }
            GroupMemoryBarrierWithGroupSync();

            if (active)
            {
                const uint b0 = col0 / 32u;
                const uint b_end = min(blocks_per_row, b0 + CHUNK_BLOCKS);
                for (uint b = b0; b < b_end; ++b)
                {
                    const uint  qoff  = (row * blocks_per_row + b) * 32u + word_i * 4u;
                    const uint  packed = qbuf.Load<uint>(qoff);
                    const float scale = sbuf[row * blocks_per_row + b];
                    const float q     = signed_byte(packed, lane & 3u);
                    const uint  xc    = (b - b0) * 32u + lane;

                    [unroll]
                    for (uint t = 0; t < BLOCK_T; ++t)
                    {
                        acc[t] += (q * xs[t][xc]) * scale;
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }

        [unroll]
        for (uint t = 0; t < BLOCK_T; ++t)
        {
            const uint tok = tbase + t;
            if (tok >= token_count) break;
            const float total = warp32_sum(acc[t], lane);
            if (lane == 0u && active)
            {
                yvec[tok * rows + row] = total;
            }
        }
    }
}
