// ZenInfer D3D12 backend: Q8_0 batched matrix product, FP32 activations, SCALE-HOISTED.
// Experimental (ZEN_D3D12_GEMM_SCALEHOIST=1). Same math as q8_0_gemm.hlsl (parity path).
// What changes: scale multiplied ONCE per 32-block (hoisted out of 32-token loop),
// accumulation uses mad() instead of (q*xs)*scale. Removes 31 redundant scale
// multiplies per block and gives the compiler an FMA opportunity.
// Bindings match q8_0_gemm.hlsl. Dispatch identical to v1.

cbuffer GemmParams : register(b0)
{
    uint rows;
    uint blocks_per_row;
    uint token_count;
    uint tokens_per_tile;
};

ByteAddressBuffer         qbuf : register(t0);
StructuredBuffer<float>   sbuf : register(t1);
ByteAddressBuffer         xraw : register(t2);
RWStructuredBuffer<float> yvec : register(u0);

#define ROWS_PER_GROUP 32u
#define BLOCK_T        32u
#define CHUNK_BLOCKS    8u
#define CHUNK_COLS     (CHUNK_BLOCKS * 32u)

groupshared float xs[BLOCK_T][CHUNK_COLS];

float warp32_sum(float v, uint lane) {
    const uint base = WaveGetLaneIndex() - lane;
    [unroll]
    for (uint s = 16u; s > 0u; s >>= 1u) {
        const float other = WaveReadLaneAt(v, base + ((lane + s) & 31u));
        if (lane + s < 32u) {
            v += other;
        }
    }
    return v;
}

float signed_byte(uint packed, uint byte_index) {
    const uint b = (packed >> (byte_index * 8u)) & 0xFFu;
    return (float)b - 256.0f * (float)(b >> 7u);
}

[numthreads(1024, 1, 1)]
void GemvQ8_0SH(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID)
{
    const uint warp = tid >> 5u;
    const uint lane = tid & 31u;
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
        for (uint t = 0; t < BLOCK_T; ++t) acc[t] = 0.0f;

        for (uint chunk = 0; chunk < chunks; ++chunk)
        {
            const uint col0 = chunk * CHUNK_COLS;

            for (uint i = tid; i < BLOCK_T * CHUNK_COLS; i += 1024u) {
                const uint t   = i / CHUNK_COLS;
                const uint c   = i % CHUNK_COLS;
                const uint tok = tbase + t;
                float v = 0.0f;
                if (tok < token_count && col0 + c < blocks_per_row * 32u) {
                    v = asfloat(xraw.Load<float>((tok * blocks_per_row * 32u + col0 + c) * 4u));
                }
                xs[t][c] = v;
            }
            GroupMemoryBarrierWithGroupSync();

            if (active) {
                const uint b0 = col0 / 32u;
                const uint b_end = min(blocks_per_row, b0 + CHUNK_BLOCKS);
                for (uint b = b0; b < b_end; ++b) {
                    const uint  qoff  = (row * blocks_per_row + b) * 32u + word_i * 4u;
                    const uint  packed = qbuf.Load<uint>(qoff);
                    const float scale = sbuf[row * blocks_per_row + b];
                    const float q_scaled = signed_byte(packed, lane & 3u) * scale;
                    const uint  xc    = (b - b0) * 32u + lane;

                    [unroll]
                    for (uint t = 0; t < BLOCK_T; ++t) {
                        acc[t] = mad(q_scaled, xs[t][xc], acc[t]);
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }

        [unroll]
        for (uint t = 0; t < BLOCK_T; ++t) {
            const uint tok = tbase + t;
            if (tok >= token_count) break;
            const float total = warp32_sum(acc[t], lane);
            if (lane == 0u && active) {
                yvec[tok * rows + row] = total;
            }
        }
    }
}
