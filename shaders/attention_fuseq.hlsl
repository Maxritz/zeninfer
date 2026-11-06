// ZenInfer D3D12 backend: single-token (decode) grouped-query attention,
// fused with int8 activation quantize of the output (decode DP4A).
//
// Fused producer variant of attention.hlsl for the DP4A decode path: the
// attention dispatch also quantises its output row (heads*head_dim elements)
// into the DP4A staging streams (qact int8 acts at u3 + per-32-block scales
// at u4), so the separate quantize_acts dispatch for the attn-output matvec
// input can be skipped when ZEN_D3D12_DP4A_FUSEQ=1. The fp32 output is
// still written (u0) alongside the int8 form.
//
// Quantize is one lane (lane 0) per head after the weighted-sum stores, over
// the just-written fp32 values read back after a device barrier - exactly the
// vector quantize_acts would have read. Only correct when head_dim % 32 == 0
// (blocks then align with heads; the host refuses the fused path otherwise).
// The quantize math mirrors quantize_acts.hlsl bit for bit (d = amax/127,
// clamp, half-away lround).

#define ATTN_MAX_HEAD_DIM 128

cbuffer AttnParams : register(b0)
{
    uint heads;        // query heads
    uint kv_heads;     // key/value heads (GQA group = heads / kv_heads)
    uint head_dim;     // elements per head
    uint context;      // allocated positions in the cache
    uint first_pos;    // first visible position (sliding window start)
    uint pos;          // position of the query token, inclusive
    uint scores_off;   // element offset of the score/weight scratch in the UAV
    uint out_off;      // element offset of the output in the UAV
    float inv_scale;   // 1 / sqrt(head_dim)
    float pad0;
    float pad1;
    float pad2;
    uint kv_base;      // element offset of this layer's k slice in the k cache
    uint v_base;       // element offset of this layer's v slice in the v cache
};

ByteAddressBuffer            qraw : register(t0);   // [heads][head_dim]
RWByteAddressBuffer          kraw : register(u1);   // [context][kv_heads][head_dim]
RWByteAddressBuffer          vraw : register(u2);   // [context][kv_heads][head_dim]
RWStructuredBuffer<float>    outv : register(u0);
RWByteAddressBuffer          qactv : register(u3);  // int8 acts (fused quantize)
RWStructuredBuffer<float>    qscale : register(u4); // one fp32 scale per block

#define WARPS_PER_GROUP 1u
#define LANES 32u

float warp_sum(float v, uint lane)
{
    const uint base = WaveGetLaneIndex() - lane;
    [unroll]
    for (uint s = LANES / 2u; s > 0u; s >>= 1u)
    {
        const float other = WaveReadLaneAt(v, base + ((lane + s) & (LANES - 1u)));
        if (lane + s < LANES)
        {
            v += other;
        }
    }
    return v;
}

int QuantizeScalar(float x, float inv)
{
    const float v = clamp(x * inv, -127.0f, 127.0f);
    return (int)(v >= 0.0f ? floor(v + 0.5f) : ceil(v - 0.5f));
}

[numthreads(LANES, 1, 1)]
void AttentionFused(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID)
{
    const uint head = gid.x * WARPS_PER_GROUP + (tid / LANES);
    const uint lane = tid % LANES;
    const bool active = head < heads;
    const uint h = active ? head : 0u;

    const uint group = max(1u, heads / kv_heads);
    const uint kh = h / group;
    const uint row_stride = kv_heads * head_dim;
    const uint hbase = h * head_dim;

    // ---- 1. scores ---------------------------------------------------------
    for (uint t = first_pos; t <= pos; ++t)
    {
        const uint kbase = kv_base + t * row_stride + kh * head_dim;
        float acc = 0.0f;
        for (uint c = lane; c < head_dim; c += LANES)
        {
            acc += asfloat(qraw.Load((hbase + c) * 4u)) * asfloat(kraw.Load((kbase + c) * 4u));
        }
        acc = warp_sum(acc, lane);
        if (active && lane == 0u)
        {
            outv[scores_off + h * context + t] = acc * inv_scale;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // ---- 2. softmax --------------------------------------------------------
    if (active && lane == 0u)
    {
        float mx = -3.402823466e+38f;
        for (uint t2 = first_pos; t2 <= pos; ++t2)
        {
            mx = max(mx, outv[scores_off + h * context + t2]);
        }
        float sum = 0.0f;
        for (uint t3 = first_pos; t3 <= pos; ++t3)
        {
            const float e = exp(outv[scores_off + h * context + t3] - mx);
            outv[scores_off + h * context + t3] = e;
            sum += e;
        }
        const float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
        for (uint t4 = first_pos; t4 <= pos; ++t4)
        {
            outv[scores_off + h * context + t4] *= inv;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // ---- 3. weighted sum ---------------------------------------------------
    if (!active)
    {
        return;
    }
    for (uint d = lane; d < head_dim; d += LANES)
    {
        float acc = 0.0f;
        for (uint t5 = first_pos; t5 <= pos; ++t5)
        {
            const float w = outv[scores_off + h * context + t5];
            acc += w * asfloat(vraw.Load((v_base + t5 * row_stride + kh * head_dim + d) * 4u));
        }
        outv[out_off + hbase + d] = acc;
    }

    // ---- 4. fused quantize (lane 0, one head at a time) --------------------
    // Every lane in the active group hits the barrier (inactive groups
    // returned above together, so no group is split). Lane 0 then reads its
    // head's just-written fp32 row and quantises each 32-element block.
    DeviceMemoryBarrierWithGroupSync();
    if (lane != 0u || (head_dim % 32u) != 0u)
    {
        return;
    }
    const uint blocks_per_head = head_dim / 32u;
    for (uint bh = 0u; bh < blocks_per_head; ++bh)
    {
        const uint gb = h * blocks_per_head + bh;
        const uint hb = bh * 32u;
        float v[32];
        float am = 0.0f;
        for (uint k = 0u; k < 32u; ++k)
        {
            v[k] = outv[out_off + hbase + hb + k];
            am = max(am, abs(v[k]));
        }
        const float d = (am > 0.0f) ? (am / 127.0f) : 0.0f;
        const float inv = (d > 0.0f) ? (1.0f / d) : 0.0f;
        qscale[gb] = d;
        const uint byte_base = gb * 32u;
        for (uint p = 0u; p < 8u; ++p)
        {
            int packed = 0;
            packed |= (uint)(QuantizeScalar(v[p * 4u + 0u], inv) & 0xff) << 0;
            packed |= (uint)(QuantizeScalar(v[p * 4u + 1u], inv) & 0xff) << 8;
            packed |= (uint)(QuantizeScalar(v[p * 4u + 2u], inv) & 0xff) << 16;
            packed |= (uint)(QuantizeScalar(v[p * 4u + 3u], inv) & 0xff) << 24;
            qactv.Store<uint>(byte_base + p * 4u, packed);
        }
    }
}
