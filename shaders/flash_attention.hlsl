// ZenInfer D3D12 backend: FlashAttention for prefill with KV cache support.
//
// Mapping
// -------
// One wave handles exactly one (query token, query head) pair. Lanes stride
// across head_dim, so a K or V row is fetched as one coalesced transaction per
// lane group. Every accumulator is a register: 4 floats of Q, 4 of V output,
// plus the running softmax max and denominator. There is no groupshared
// staging and no per-thread array larger than 4 elements.
//
// That is the whole point of this layout. The previous version kept
// acc[4][128] = 512 floats per thread. 512 > 255 registers, so the compiler
// demoted the array to local memory and the inner loop did ~8k iterations of
// read-modify-write against a 2 KB per-thread scratch region. The dispatch
// never retired inside the Windows TDR watchdog and the device was removed
// with DXGI_ERROR_DEVICE_HUNG. Nothing about the old shape was salvageable, so
// the kernel is rewritten rather than tuned.
//
// Softmax
// -------
// Standard online form: keep m (running max) and l (running denominator),
// rescale the V accumulator by exp(m_old - m_new) whenever the max moves.
// WaveActiveSum/WaveActiveMax deliver the reduction to every lane, so m and l
// are implicitly broadcast and there is no lane-0 handoff to get wrong.
//
// K/V source: KV cache [layers][context][kv_heads][head_dim], flat, with
// kv_cache_base selecting the layer. Positions are absolute, so a batch that
// starts at start_pos attends to [first_pos, start_pos + q_rel].

#define MAX_D_PER_LANE 4

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
    uint  first_pos;       // absolute first KV position to attend to (window / cache start)
    uint  pad0;
    uint  pad1;
};

StructuredBuffer<float> qbuf : register(t0);     // [tokens, heads, head_dim]
StructuredBuffer<float> k_cache : register(t1);  // flat KV cache
StructuredBuffer<float> v_cache : register(t2);
RWStructuredBuffer<float> outbuf : register(u0); // [tokens, heads, head_dim]

[numthreads(64, 1, 1)]
void FlashAttentionWithCache(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID)
{
    // One wave per (query token, head). A 64-thread group is a single wave on
    // wave64 hardware and two waves on wave32; the second wave exits at once so
    // the token/head mapping never depends on the target's wave size.
    const uint lanes = WaveGetLaneCount();
    if ((tid / lanes) != 0) {
        return;
    }
    if (gid.y >= tokens || gid.x >= heads) {
        return;
    }

    const uint lane = WaveGetLaneIndex();
    const uint q_rel = gid.y;
    const uint h = gid.x;

    const uint q_abs = start_pos + q_rel;
    const uint group = heads / kv_heads;
    const uint kv_head = h / group;
    const uint head_stride = heads * head_dim;
    const uint h_base = h * head_dim;

    float qv[MAX_D_PER_LANE];
    float acc[MAX_D_PER_LANE];
    [unroll]
    for (uint i = 0; i < MAX_D_PER_LANE; ++i) {
        qv[i] = 0.0f;
        acc[i] = 0.0f;
    }
    [unroll]
    for (uint iq = 0; iq < MAX_D_PER_LANE; ++iq) {
        const uint d = lane + iq * lanes;
        if (d < head_dim) {
            qv[iq] = qbuf[q_rel * head_stride + h_base + d];
        }
    }

    float m = -3.402823466e+38f;
    float l = 0.0f;

    const uint j_end = min(q_abs + 1u, valid_kv_tokens);
    for (uint j = first_pos; j < j_end; ++j) {
        const uint base = kv_cache_base + j * kv_cache_stride + kv_head * head_dim;

        float partial = 0.0f;
        [unroll]
        for (uint ik = 0; ik < MAX_D_PER_LANE; ++ik) {
            const uint d = lane + ik * lanes;
            if (d < head_dim) {
                partial += qv[ik] * k_cache[base + d];
            }
        }

        // Uniform control flow here: every lane reaches both wave intrinsics.
        const float s = WaveActiveSum(partial) * inv_scale;
        const float new_m = max(m, s);
        const float corr = exp(m - new_m); // 0 on the first iteration (m = -FLT_MAX)
        const float p = exp(s - new_m);

        l = l * corr + p;
        [unroll]
        for (uint iv = 0; iv < MAX_D_PER_LANE; ++iv) {
            const uint d = lane + iv * lanes;
            if (d < head_dim) {
                acc[iv] = acc[iv] * corr + p * v_cache[base + d];
            }
        }
        m = new_m;
    }

    const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    [unroll]
    for (uint io = 0; io < MAX_D_PER_LANE; ++io) {
        const uint d = lane + io * lanes;
        if (d < head_dim) {
            outbuf[q_rel * head_stride + h_base + d] = acc[io] * inv_l;
        }
    }
}
