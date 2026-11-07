// ZenInfer D3D12 backend: multi-token FlashAttention for prefill.
//
// Why this kernel exists
// ----------------------
// flash_attention.hlsl maps one wave to one (query token, query head) pair and
// walks the visible window one key at a time. A whole wave therefore produces a
// single score per iteration, and every query token re-reads the entire K/V
// window: at 512 tokens and 512 KB of K/V per layer that is ~4.3 GB of L2
// traffic per layer. Measured on an RX 9070 XT with Qwen3-4B (32 heads, 8 KV
// heads, head_dim 128) the dispatch cost ~1.3 ms per layer, which is within a
// few percent of that traffic divided by the card's measured ~3.2 TB/s L2
// bandwidth - it is a bandwidth and dependency-chain problem, not an arithmetic
// one.
//
// This kernel keeps the lane-per-dimension mapping (so a K row is still one
// coalesced transaction per wave and there is no transpose) and instead has a
// workgroup carry FTOK consecutive query tokens at once:
//
//   * every K/V row is fetched once and consumed by FTOK tokens, cutting the
//     traffic by FTOK (~0.54 GB per layer at FTOK = 8);
//   * the FTOK dot products, softmax updates and V accumulations are
//     independent, so the compiler interleaves FTOK chains. The legacy kernel's
//     inner loop is one serial chain: wave reduction -> exp -> exp -> rescale,
//     each step waiting on the previous one. That chain, not the arithmetic, is
//     what the per-iteration latency was made of.
//
// Measured limits (do not re-derive these the hard way): a groupshared version
// that stages K/V tiles and serves 32 tokens per workgroup from one fetch - i.e.
// 4x less traffic again - measures 1795 us/layer, and double-buffering its fill
// (so the next tile's rows are in flight during the current tile's arithmetic)
// changes nothing at 1792 us. Cutting K/V traffic therefore does not pay here:
// what this kernel spends is the per-key dependent chain (dot -> WaveActiveSum ->
// exp -> rescale), and widening a workgroup costs occupancy faster than it saves
// bytes. Keep the one-wave mapping and spend the effort on fewer reductions per
// key, not on staging.
//
// The arithmetic per token is unchanged and in the same order as the legacy
// kernel - partial products in lane order, one WaveActiveSum, the same online
// rescaling - so a token's result is bit-identical to what
// flash_attention.hlsl produces for it. zen_d3d12_flash_parity verifies that
// against a CPU reference, and the model-level token ids are checked against
// the oracles.
//
// Causal mask: the block covers positions start_pos+q0 .. +ntok-1, and key j is
// visible to token t only when j <= start_pos+q0+t. Masked lanes are pushed to
// -FLT_MAX, which leaves the running max untouched, makes the rescale
// exp(0) = 1 and the probability exp(-FLT_MAX - m) = 0, so the masked key
// contributes nothing without a branch.
//
// Gate: ZEN_D3D12_FLASH_T=0 restores flash_attention.hlsl. The host also
// requires head_dim <= 128 (FTD * 64), kv_heads <= heads and heads % kv_heads
// == 0.

#define FTOK     8      // query tokens per workgroup
#define FTD      4      // dimensions per lane (4 * 32 >= 128, covers wave32)
#define FNEG     (-3.402823466e+38f)

cbuffer FlashAttnParams : register(b0)
{
    uint  tokens;          // query tokens in this prefill batch
    uint  heads;           // query heads
    uint  kv_heads;        // key/value heads (GQA)
    uint  head_dim;        // head dimension
    float inv_scale;       // 1 / sqrt(head_dim)
    uint  kv_cache_base;   // element offset of this layer inside the KV cache
    uint  kv_cache_stride; // elements between consecutive tokens (kv_heads * head_dim)
    uint  valid_kv_tokens; // absolute count of populated KV positions
    uint  start_pos;       // absolute position of query token 0
    uint  first_pos;       // absolute first KV position to attend to
    uint  pad0;
    uint  pad1;
};

StructuredBuffer<float> qbuf : register(t0);     // [tokens, heads, head_dim]
StructuredBuffer<float> k_cache : register(t1);  // flat KV cache
StructuredBuffer<float> v_cache : register(t2);
RWStructuredBuffer<float> outbuf : register(u0); // [tokens, heads, head_dim]

[numthreads(64, 1, 1)]
void FlashAttentionTile(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID)
{
    // One wave per group; the second half of the 64-thread group on a wave32
    // device exits. The host only selects this kernel when the reported wave
    // size is 32..64, and every loop below is sized from WaveGetLaneCount().
    const uint lanes = WaveGetLaneCount();
    if ((tid / lanes) != 0) {
        return;
    }
    const uint q0 = gid.y * FTOK;
    if (gid.x >= heads || q0 >= tokens) {
        return;
    }

    const uint lane = WaveGetLaneIndex();
    const uint h = gid.x;
    const uint group = max(1u, heads / kv_heads);
    const uint kv_head = h / group;
    const uint head_stride = heads * head_dim;
    const uint h_base = h * head_dim;
    const uint ntok = min(FTOK, tokens - q0);
    const uint pos0 = start_pos + q0;
    // Exclusive end of the KV range this block can attend to.
    const uint j_end = min(pos0 + ntok, valid_kv_tokens);

    // One accumulator chain per token in the block.
    float qv[FTOK][FTD];
    float acc[FTOK][FTD];
    float m[FTOK];
    float l[FTOK];
    [unroll]
    for (uint t0 = 0u; t0 < FTOK; ++t0) {
        m[t0] = FNEG;
        l[t0] = 0.0f;
        [unroll]
        for (uint i0 = 0u; i0 < FTD; ++i0) {
            const uint d = lane + i0 * lanes;
            qv[t0][i0] = (t0 < ntok && d < head_dim)
                             ? qbuf[(q0 + t0) * head_stride + h_base + d]
                             : 0.0f;
            acc[t0][i0] = 0.0f;
        }
    }

    for (uint j = first_pos; j < j_end; ++j) {
        const uint base = kv_cache_base + j * kv_cache_stride + kv_head * head_dim;

        // K and V rows are read once and reused by every token in the block.
        float kv[FTD];
        float vv[FTD];
        [unroll]
        for (uint i1 = 0u; i1 < FTD; ++i1) {
            const uint d = lane + i1 * lanes;
            if (d < head_dim) {
                kv[i1] = k_cache[base + d];
                vv[i1] = v_cache[base + d];
            } else {
                kv[i1] = 0.0f;
                vv[i1] = 0.0f;
            }
        }

        [unroll]
        for (uint t1 = 0u; t1 < FTOK; ++t1) {
            if (t1 < ntok) {
                float partial = 0.0f;
                [unroll]
                for (uint i2 = 0u; i2 < FTD; ++i2) {
                    partial += qv[t1][i2] * kv[i2];
                }
                // WaveActiveSum broadcasts, so s (and therefore the causal test)
                // is uniform across the wave.
                float s = WaveActiveSum(partial) * inv_scale;
                // s is uniform across the wave, so every branch below is
                // wave-uniform and the two exp() cases are exclusive.
                if (j > pos0 + t1) {
                    s = FNEG;                       // causal mask: contributes nothing
                }
                // Legacy form evaluates exp(m - nm) and exp(s - nm) every key.
                // Two of those are exact no-ops in the common cases, and skipping
                // them is bit-identical: when the running max is not beaten
                // nm == m, so exp(m - nm) is exactly 1.0f; when it is beaten,
                // nm == s, so exp(s - nm) is exactly 1.0f. Only the first key of
                // a token (m = -FLT_MAX) keeps the rescale, which underflows to 0
                // exactly as before.
                float corr = 1.0f;
                float p = 0.0f;
                if (s != FNEG) {
                    if (s > m[t1]) {
                        corr = exp(m[t1] - s);
                        p = 1.0f;
                        m[t1] = s;
                    } else {
                        p = exp(s - m[t1]);
                    }
                }
                // p is uniform across lanes, so the denominator needs no reduction.
                l[t1] = l[t1] * corr + p;
                [unroll]
                for (uint i3 = 0u; i3 < FTD; ++i3) {
                    acc[t1][i3] = acc[t1][i3] * corr + p * vv[i3];
                }
            }
        }
    }

    [unroll]
    for (uint t2 = 0u; t2 < FTOK; ++t2) {
        if (t2 < ntok) {
            const float inv = (l[t2] > 0.0f) ? (1.0f / l[t2]) : 0.0f;
            [unroll]
            for (uint i4 = 0u; i4 < FTD; ++i4) {
                const uint d = lane + i4 * lanes;
                if (d < head_dim) {
                    outbuf[(q0 + t2) * head_stride + h_base + d] = acc[t2][i4] * inv;
                }
            }
        }
    }
}
