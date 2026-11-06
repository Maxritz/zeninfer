// ZenInfer D3D12 backend: Q8_0 batched matrix product, FP32 activations, REGISTER-TILED (fused dequant).
//
// Experimental (ZEN_D3D12_GEMM_FUSED=1). Same math as q8_0_gemm.hlsl (parity path):
// activations stay float32, output matches CPU reference to rounding. What changes
// is the tile shape and the inner loop to eliminate redundant LDS traffic:
//
//   * 4 rows x 4 tokens per workgroup (16 (row,token) pairs), 2 warps of 32
//     threads. Each warp owns 8 (row,token) pairs (4 rows x 2 tokens); each
//     thread owns 4 consecutive K elements (loaded as one uint).
//   * Activation tile xs[4][256] (4 tokens, 256 cols) = 4 KB LDS. Higher
//     occupancy vs v1's 32 KB. Weight reuse 4x (tokens), activation reuse 4x (rows).
//   * Block scale folded out of the inner loop: scale multiplied once per
//     32-block, not per element.
//   * Wider weight loads: qbuf.Load<uint> (4 bytes) covers 4 K elements.
//
// FUSED DEQUANT: each weight block is read ONCE (qbuf.Load<uint>), dequantized
// in registers (scale * int8 -> float), multiplied by activation, accumulated.
// No scratch buffer, no second pass.
//
// Dispatch geometry: (ceil(rows/4), ceil(tokens/4), 1). Output: yvec[token*rows+row].

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
RWStructuredBuffer<float> yvec   : register(u0);  // [token][rows]

#define ROWS_PER_WG 4u
#define TOKENS_PER_WG 4u
#define K_PER_LANE 4u
#define LANES_PER_ROW 8u
#define TOTAL_THREADS 64u
#define CHUNK_BLOCKS 8u
#define CHUNK_COLS (CHUNK_BLOCKS * 32u)

groupshared float xs[TOKENS_PER_WG][CHUNK_COLS];

float warp8_sum(float v, uint lane) {
    const uint base = WaveGetLaneIndex() - (lane & 7u);
    [unroll]
    for (uint s = 4u; s > 0u; s >>= 1u) {
        const float other = WaveReadLaneAt(v, base + ((lane + s) & 7u));
        if ((lane & 7u) + s < 8u) {
            v += other;
        }
    }
    return v;
}

float signed_byte(uint packed, uint byte_index) {
    const uint b = (packed >> (byte_index * 8u)) & 0xFFu;
    return (float)b - 256.0f * (float)(b >> 7u);
}

[numthreads(TOTAL_THREADS, 1, 1)]
void GemmQ8_0Fused(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID)
{
    const uint wave = tid >> 5u;
    const uint lane = tid & 31u;
    const uint row_in_wg = lane >> 3u;
    const uint k_lane = lane & 7u;
    const uint row = gid.x * ROWS_PER_WG + row_in_wg;
    const uint token_base = gid.y * TOKENS_PER_WG;
    const uint token0 = token_base + wave * 2u;
    const uint token1 = token0 + 1u;

    const bool active = row < rows && token0 < token_count;

    const uint chunks = (blocks_per_row + CHUNK_BLOCKS - 1u) / CHUNK_BLOCKS;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint chunk = 0; chunk < chunks; ++chunk)
    {
        const uint col0 = chunk * CHUNK_COLS;
        const uint b0 = col0 / 32u;
        const uint b_end = min(blocks_per_row, b0 + CHUNK_BLOCKS);

        // Stage 2 tokens' activations into xs (this wave's tokens).
        for (uint ti = 0u; ti < 2u; ++ti) {
            const uint t = wave * 2u + ti;
            const uint tok = token_base + t;
            for (uint c4 = k_lane; c4 < CHUNK_COLS / 4u; c4 += LANES_PER_ROW) {
                const uint c = c4 * 4u;
                float4 v = float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (tok < token_count && col0 + c < blocks_per_row * 32u) {
                    v = asfloat(xraw.Load<float4>((tok * blocks_per_row * 32u + col0 + c) * 4u));
                }
                xs[t][c] = v.x;
                xs[t][c + 1u] = v.y;
                xs[t][c + 2u] = v.z;
                xs[t][c + 3u] = v.w;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        if (active)
        {
            const uint k_base = k_lane * K_PER_LANE;

            for (uint b = b0; b < b_end; ++b)
            {
                const uint qoff = (row * blocks_per_row + b) * 32u + k_base;
                const uint packed = qbuf.Load<uint>(qoff);
                const float scale = sbuf[row * blocks_per_row + b];
                const uint xc0 = (b - b0) * 32u + k_base;

                const float w0 = signed_byte(packed, 0u);
                const float w1 = signed_byte(packed, 1u);
                const float w2 = signed_byte(packed, 2u);
                const float w3 = signed_byte(packed, 3u);

                if (token0 < token_count) {
                    acc0 += (w0 * xs[wave * 2u][xc0] + w1 * xs[wave * 2u][xc0 + 1u] +
                             w2 * xs[wave * 2u][xc0 + 2u] + w3 * xs[wave * 2u][xc0 + 3u]) * scale;
                }
                if (token1 < token_count) {
                    acc1 += (w0 * xs[wave * 2u + 1u][xc0] + w1 * xs[wave * 2u + 1u][xc0 + 1u] +
                             w2 * xs[wave * 2u + 1u][xc0 + 2u] + w3 * xs[wave * 2u + 1u][xc0 + 3u]) * scale;
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (active) {
        const float total0 = warp8_sum(acc0, lane);
        const float total1 = warp8_sum(acc1, lane);
        if (k_lane == 0u) {
            if (token0 < token_count) {
                yvec[token0 * rows + row] = total0;
            }
            if (token1 < token_count) {
                yvec[token1 * rows + row] = total1;
            }
        }
    }
}
