// ZenInfer D3D12 backend: Q8_0 batched matrix product, FP32 activations, REGISTER-TILED.
// Experimental (ZEN_D3D12_GEMM_RT=1). Same math as q8_0_gemm.hlsl (parity path).
// Activations stay float32, output matches CPU reference to rounding.
//
// Tile: 4 rows x 8 tokens per workgroup, 4 warps of 32 threads.
// K step = one Q8_0 block (32 values). Scale folded out of inner loop.
// Chunk-based staging: 8 blocks (256 cols) per chunk, 2 barriers per chunk.
// LDS: xs[8][256] = 8 KB. Occupancy: 8 warps/SIMD (vs v1's 2).
//
// Bindings match q8_0_gemm.hlsl. Dispatch: (ceil(rows/4), ceil(token_count/8), 1).

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

#define ROWS_PER_WG 4u
#define TOKENS_PER_WG 8u
#define K_PER_LANE 4u
#define LANES_PER_ROW 8u
#define THREADS_PER_WAVE 32u
#define WAVES_PER_WG 4u
#define TOTAL_THREADS 128u
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
void GemmQ8_0RT(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID)
{
    const uint wave = tid >> 5u;          // 0..3 (which token-pair group)
    const uint lane = tid & 31u;          // 0..31 within wave
    const uint row_in_wg = lane >> 3u;    // 0..3 (row within workgroup)
    const uint k_lane = lane & 7u;        // 0..7 (K lane within row)
    const uint row = gid.x * ROWS_PER_WG + row_in_wg;
    const uint token_base = gid.y * TOKENS_PER_WG;
    // Wave w handles tokens w*2 and w*2+1.
    const uint token0 = token_base + wave * 2u;
    const uint token1 = token0 + 1u;

    const bool active = row < rows && token0 < token_count;

    const uint chunks = (blocks_per_row + CHUNK_BLOCKS - 1u) / CHUNK_BLOCKS;

    // Accumulators: 2 tokens x 1 row = 2 per thread.
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint chunk = 0; chunk < chunks; ++chunk)
    {
        const uint col0 = chunk * CHUNK_COLS;
        const uint b0 = col0 / 32u;
        const uint b_end = min(blocks_per_row, b0 + CHUNK_BLOCKS);

        // Stage 2 tokens' activations (this wave's tokens) into xs.
        for (uint i = wave * 2u; i < wave * 2u + 2u; ++i) {
            const uint t = i;
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
