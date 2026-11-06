// ZenInfer D3D12 backend: Q8_0 batched matrix product (prefill path), fp32 activations.
//
// This is the parity path: activations stay float32, so the result matches the CPU
// reference to rounding. The faster DP4A variant lives in q8_0_gemm_dp4a.hlsl and is
// opt-in because it quantises activations to int8.
//
// What changed from the previous version
// --------------------------------------
// The old kernel nested the token loop OUTSIDE the weight loop:
//
//     for (tile = 0; tile < tokens/8; ++tile)   // 8 tokens
//         for (b = 0; b < blocks; ++b)
//             q = qbuf.Load(...)                // weights re-read every tile
//
// A 128-token prefill therefore re-read the whole 1.3 GB weight matrix 16 times
// (~21 GB). Measured, it cost 9.7 ms per token at 64 tokens against 11.8 ms per
// token for decode - no batching benefit at all - while all 112 GEMMs were
// confirmed to be running on the GPU (ZEN_QGEMM_TRACE showed gpu_ok=112).
//
// Now a thread block accumulates 32 tokens at once, so each weight block is
// fetched once per 32-token group instead of once per 8. A 128-token prefill
// makes 4 passes over the weights instead of 16.
//
// Layout
// ------
// One warp owns one output row; its 32 lanes span one 32-element weight block, so
// the per-block scale folds into each lane's accumulator and the reduction is a
// single warp sum at the end.

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
#define BLOCK_T        32u    // tokens accumulated per thread block
#define CHUNK_BLOCKS    8u    // 32-byte weight blocks staged per pass
#define CHUNK_COLS     (CHUNK_BLOCKS * 32u)

// 32 tokens x 256 columns of float32 activations.
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
void GemmQ8_0(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID)
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
