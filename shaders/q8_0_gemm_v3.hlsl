// ZenInfer D3D12 backend: Q8_0 batched matrix product v3 (verify path), fp32 activations.
//
// Same math as q8_0_gemm.hlsl (fp32 activations, parity): one warp owns one
// output row, lanes stride the 32 elements of each weight block, per-block
// scale folds into each lane's accumulator, single warp sum at the end. K
// order per output is therefore identical to v1, so outputs are
// bit-identical (chunks outer, blocks inner, same multiply order).
//
// What changed (speculative verify only, not math):
//
//   * BLOCK_T 32 -> 8. The v1 tile wastes 87.5% of its token lanes at
//     T=4 (verify with proposal_count=4) and 75% at T=8. With 8 tokens
//     per tile, T=4 still wastes half a tile but T=8 is exact, and the
//     per-thread accumulator drops from 32 floats to 8, which is also
//     the occupancy win v2 chased. CHUNK_BLOCKS stays 8 (256 columns
//     per stage) and ROWS_PER_GROUP stays 32 with 1024 threads, so
//     groupshared is 8x256x4 = 8 KB vs 32 KB in v1 and every barrier
//     syncs the same 32 warps. Staging keeps the v1 scalar path
//     (no float4 vectorization) so the only delta vs v1 is the tile
//     width; the K traversal, scale fold and warp reduction are
//     untouched, which is what keeps the v3 result within 2e-7 of v1.
//   * At large T (prefill 128) v3 makes 16 passes over the weights
//     instead of 4. Prefill is instruction-bound so the extra traffic
//     moves prefill little, but v3 is not the prefill default: it is
//     selected only behind ZEN_D3D12_GEMM_V3=1 for the verify path
//     where T is 4..8 and the v1 tile never fills. Default stays v1.

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

#define ROWS_PER_GROUP 32u    // one warp per row
#define BLOCK_T         8u    // tokens accumulated per thread block (verify tile)
#define CHUNK_BLOCKS    8u    // 32-byte weight blocks staged per pass
#define CHUNK_COLS     (CHUNK_BLOCKS * 32u)

// BLOCK_T x CHUNK_COLS float32 activations (8 x 256 x 4 B = 8 KB).
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

[numthreads(1024, 1, 1)]
void GemmQ8_0V3(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID)
{
    const uint warp = tid >> 5u;          // row inside the 32-row block
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

            // Stage this token group's activation slab once, shared by all 32 rows.
            for (uint i = tid; i < BLOCK_T * CHUNK_COLS; i += 1024u)
            {
                const uint t   = i / CHUNK_COLS;
                const uint c   = i % CHUNK_COLS;
                const uint tok = tbase + t;
                float v = 0.0f;
                if (tok < token_count && col0 + c < blocks_per_row * 32u)
                {
                    v = asfloat(xraw.Load<float>((tok * blocks_per_row * 32u + col0 + c) * 4u));
                }
                xs[t][c] = v;
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
