// ZenInfer D3D12 backend: Q4_K batched matrix product, FP32 activations.
// Experimental (ZEN_D3D12_Q4K=1). Fused dequant: Q4_K weights dequantized
// in registers, never spilled to scratch. Same math as CPU dequantize() in
// src/quant.cpp (k_scale_min + d/dmin fp16 scales).
//
// Q4_K super-block: 256 weights, 144 bytes.
//   d (fp16), dmin (fp16), scales[12], quants[128]
// Dequant: sub-block j (0-7, 32 weights each), scale = d*sc[j], min = dmin*m[j]
//   weight[i] = scale * nibble - min
// Nibbles: weights 0-127 use low nibbles, 128-255 use high nibbles.
//
// Layout: 1 warp per row, 32 tokens per tile. Each lane handles 8 K per
// 256-weight block. Scale folded out of inner loop.
//
// Bindings match q8_0_gemm.hlsl. Dispatch: (ceil(rows/32), ceil(token_count/32), 1).

cbuffer GemmParams : register(b0)
{
    uint rows;
    uint blocks_per_row;   // 256-weight blocks per row (cols/256)
    uint token_count;
    uint tokens_per_tile;
};

ByteAddressBuffer         qbuf : register(t0);
StructuredBuffer<float>   sbuf : register(t1);   // unused, keeps root layout
ByteAddressBuffer         xraw : register(t2);
RWStructuredBuffer<float> yvec : register(u0);

#define ROWS_PER_GROUP 32u
#define BLOCK_T        32u
#define WEIGHTS_PER_BLOCK 256u

float h2f(uint h) {
    uint sign = (h >> 15u) & 1u;
    uint exp  = (h >> 10u) & 0x1Fu;
    uint mant = h & 0x3FFu;
    if (exp == 0u) return 0.0f;
    if (exp == 31u) return sign ? -1e30f : 1e30f;
    uint bits = (sign << 31u) | ((exp + 112u) << 23u) | (mant << 13u);
    return asfloat(bits);
}

// k_scale_min for sub-block j (0-7) within a 256-block.
// scales is the 12-byte packed array at scales_base.
uint2 get_scale_min(uint j, ByteAddressBuffer qbuf, uint scales_base) {
    uint sc, m;
    if (j < 4u) {
        sc = qbuf.Load<uint>(scales_base + j) & 0x3Fu;
        m  = qbuf.Load<uint>(scales_base + j + 4u) & 0x3Fu;
    } else {
        sc = (qbuf.Load<uint>(scales_base + j + 4u) & 0x0Fu) |
             ((qbuf.Load<uint>(scales_base + j - 4u) >> 6u) << 4u);
        m  = (qbuf.Load<uint>(scales_base + j + 4u) >> 4u) |
             ((qbuf.Load<uint>(scales_base + j) >> 6u) << 4u);
    }
    return uint2(sc, m);
}

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

[numthreads(1024, 1, 1)]
void GemmQ4_K(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID)
{
    const uint warp = tid >> 5u;
    const uint lane = tid & 31u;
    const uint row  = gid.x * ROWS_PER_GROUP + warp;
    const bool active = row < rows;

    const uint cols = blocks_per_row * WEIGHTS_PER_BLOCK;
    const uint tgroups = (token_count + BLOCK_T - 1u) / BLOCK_T;

    for (uint tg = 0; tg < tgroups; ++tg)
    {
        const uint tbase = tg * BLOCK_T;

        float acc[BLOCK_T];
        [unroll]
        for (uint t = 0; t < BLOCK_T; ++t) acc[t] = 0.0f;

        for (uint blk = 0; blk < blocks_per_row; ++blk)
        {
            uint block_base = (row * blocks_per_row + blk) * 144u;
            uint d_half = qbuf.Load<uint>(block_base) & 0xFFFFu;
            uint dmin_half = (qbuf.Load<uint>(block_base + 2u)) & 0xFFFFu;
            float d = h2f(d_half);
            float dmin = h2f(dmin_half);
            uint scales_base = block_base + 4u;
            uint quants_base = block_base + 16u;

            // Each lane handles 8 K elements: lane*8 .. lane*8+7
            // These span sub-blocks: lane*8/32 = lane/4 (sub 0-7)
            for (uint j = 0; j < 8u; ++j) {
                uint widx = lane * 8u + j;
                uint sub = widx / 32u;
                uint2 sm = get_scale_min(sub, qbuf, scales_base);
                float scale = d * (float)sm.x;
                float minv = dmin * (float)sm.y;

                // Load nibble
                uint nibble;
                uint byte_idx;
                if (widx < 128u) {
                    byte_idx = widx;
                    uint packed = qbuf.Load<uint>(quants_base + byte_idx / 2u);
                    nibble = (byte_idx % 2u == 0u) ? (packed & 0x0Fu) : ((packed >> 4u) & 0x0Fu);
                } else {
                    byte_idx = widx - 128u;
                    uint packed = qbuf.Load<uint>(quants_base + byte_idx / 2u);
                    nibble = (byte_idx % 2u == 0u) ? (packed & 0x0Fu) : ((packed >> 4u) & 0x0Fu);
                }
                float w = scale * (float)nibble - minv;

                uint k_global = blk * WEIGHTS_PER_BLOCK + widx;
                if (k_global >= cols) continue;

                [unroll]
                for (uint t = 0; t < BLOCK_T; ++t) {
                    uint tok = tbase + t;
                    if (tok >= token_count) break;
                    float x = asfloat(xraw.Load<uint>((tok * cols + k_global) * 4u));
                    acc[t] += w * x;
                }
            }
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
