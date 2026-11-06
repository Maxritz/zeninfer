// ZenInfer D3D12 backend: Q8_0 batched matrix product, DP4A variant (opt-in).
//
// NOT the parity path: it quantises activations to int8 so the inner product can
// use v_dot4_i32_i8. The fp32 kernel in q8_0_gemm.hlsl stays the default.
//
// Why this exists
// ---------------
// The fp32 kernel dequantises inside the inner loop: it extracts one byte, converts
// it to float, multiplies by the activation and by the block scale, and accumulates.
// That is roughly 4-6 instructions per weight element, and measurement put it at
// 2.65 TFLOP/s. Cutting weight traffic 4x by keeping weights resident across 32
// tokens moved prefill by only 6%, which proves the loop is instruction-bound, not
// bandwidth-bound. The only way through is to stop converting to float at all.
//
// Mapping
// -------
// One warp owns one output row, exactly as the fp32 kernel does, so the two share
// the same dispatch geometry (ROWS_PER_GROUP = 32). What differs is that each lane
// owns four consecutive bytes instead of one:
//
//   lane l  ->  block (iter*4 + l/8), byte offset (l%8)*4
//
// so a warp covers four 32-byte blocks per iteration and the 32 lanes read one
// contiguous 128-byte span of the row. An earlier attempt gave each row only 8
// lanes and packed 4 rows into a warp; that scattered the loads across four
// distant rows and measured 0.94 TFLOP/s, three times slower than the fp32 path.
//
// Every lane accumulates a disjoint set of blocks, so the final reduction is a
// single 32-lane warp sum, with no partial-sum bookkeeping.
//
// Accuracy: one scale per token (max|x|/127), finer than the per-32-element scale
// the weights already carry. Measured relative error against the fp32 reference is
// ~4e-3, normal for int8 activations and well inside a Q8_0 model's tolerance, but
// it is not bit-parity, which is why it is opt-in via ZEN_D3D12_DP4A=1.
//
// Requires Shader Model 6.4 for dot4add_i8packed; this shader alone is built at
// cs_6_4 (everything else stays at cs_6_0).

cbuffer GemmParams : register(b0)
{
    uint rows;
    uint blocks_per_row;
    uint token_count;
    uint tokens_per_tile;
};

ByteAddressBuffer         qbuf   : register(t0);  // 32 bytes per block
StructuredBuffer<float>   sbuf   : register(t1);  // one fp32 scale per weight block
ByteAddressBuffer         xraw   : register(t2);  // [token][columns] int8-quantised
StructuredBuffer<float>   xscale : register(t3);  // one fp32 scale per token
RWStructuredBuffer<float> yvec   : register(u0);  // [token][rows]

#define ROWS_PER_GROUP 32u    // one warp per row
#define BLOCK_T         32u    // tokens accumulated per thread block
#define BLOCKS_PER_ITER  4u    // a warp covers four 32-byte blocks per iteration
#define CHUNK_BLOCKS     8u    // blocks staged per pass (2 iterations)
#define CHUNK_COLS     (CHUNK_BLOCKS * 32u)
#define CHUNK_WORDS    (CHUNK_COLS / 4u)

// 32 tokens x 256 columns, pre-packed as int8x4: one 32-bit load feeds one DP4A.
groupshared uint xpk[BLOCK_T][CHUNK_WORDS];

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

[numthreads(1024, 1, 1)]
void GemmQ8_0Dp4a(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID)
{
    const uint warp = tid >> 5u;          // row inside the 32-row block
    const uint lane = tid & 31u;
    const uint row  = gid.x * ROWS_PER_GROUP + warp;
    const bool active = row < rows;

    const uint chunks  = (blocks_per_row + CHUNK_BLOCKS - 1u) / CHUNK_BLOCKS;
    const uint tgroups = (token_count + BLOCK_T - 1u) / BLOCK_T;
    const uint byte_off = (lane % 8u) * 4u;   // 4-byte group inside a 32-byte block
    const uint blk_in_iter = lane / 8u;       // which of the 4 blocks this lane owns

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
            const uint b0 = chunk * CHUNK_BLOCKS;

            // Stage the token group's activation slab once for all 32 rows.
            for (uint i = tid; i < BLOCK_T * CHUNK_WORDS; i += 1024u)
            {
                const uint t   = i / CHUNK_WORDS;
                const uint w   = i % CHUNK_WORDS;
                const uint tok = tbase + t;
                uint packed = 0u;
                if (tok < token_count)
                {
                    const uint col = (b0 * 32u) + w * 4u;
                    if (col + 4u <= blocks_per_row * 32u)
                    {
                        packed = xraw.Load<uint>(tok * blocks_per_row * 32u + col);
                    }
                }
                xpk[t][w] = packed;
            }
            GroupMemoryBarrierWithGroupSync();

            if (active)
            {
                const uint b_end = min(blocks_per_row, b0 + CHUNK_BLOCKS);
                for (uint b = b0; b < b_end; b += BLOCKS_PER_ITER)
                {
                    // This lane's block and the matching packed activation word.
                    const uint blk = b + blk_in_iter;
                    if (blk >= b_end) break;

                    const uint  qoff  = (row * blocks_per_row + blk) * 32u + byte_off;
                    const uint  q     = qbuf.Load<uint>(qoff);
                    const float scale = sbuf[row * blocks_per_row + blk];
                    const uint  w     = (blk - b0) * 8u + (lane % 8u);

                    [unroll]
                    for (uint t = 0; t < BLOCK_T; ++t)
                    {
                        acc[t] += (float)dot4add_i8packed(q, xpk[t][w], 0) * scale;
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }

        // Each lane summed a disjoint set of blocks, so one warp sum completes it.
        [unroll]
        for (uint t = 0; t < BLOCK_T; ++t)
        {
            const uint tok = tbase + t;
            if (tok >= token_count) break;
            const float total = warp32_sum(acc[t], lane);
            if (lane == 0u && active)
            {
                yvec[tok * rows + row] = total * xscale[tok];
            }
        }
    }
}
