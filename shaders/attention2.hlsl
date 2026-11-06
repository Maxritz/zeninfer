// ZenInfer D3D12 backend: single-token (decode) GQA attention, 2 heads per
// 64-thread group (wave64). Same algorithm, same arithmetic, same memory layout
// as attention.hlsl; only the thread->head mapping changes:
//
//   attention.hlsl:  [numthreads(32)] one head per 32-thread group.
//                    On wave64 hardware two groups share a wave, so half the
//                    wave lanes idle, and the 32-lane reduction needs a rebased
//                    shuffle (WaveReadLaneAt over a lane-rebased base).
//   attention2.hlsl: [numthreads(64)] two heads per group: lanes 0..31 do head
//                    2*gid, lanes 32..63 do head 2*gid+1. The group exactly
//                    fills one wave64, so no wave slots idle.
//   The reduction is deliberately NOT WaveActiveSum: on wave64 that sums all
//   64 lanes, folding the two heads into one score (the same bug class as the
//   q8_0_gemv wave64 bug). The per-head shuffle uses the same rebased
//   WaveReadLaneAt tree over 32 lanes as attention.hlsl, so for each head the
//   floating-point sum is performed in the identical order and the kernel is
//   bit-identical to the 1-head version for every head.
//   Softmax stays on the leader lane of each 32-lane half (lane 0 and lane
//   32 run concurrently inside the wave); both heads sync together at the two
//   group barriers, which is safe because their score/output regions are
//   disjoint.
//
// KV cache layout, kv_base/v_base layer-slice offsets, and the UAV bindings
// are the same as attention.hlsl. The caller dispatches grid = ceil(heads/2);
// a trailing odd head runs lanes 32..63 clamped to head 0 and writes nothing
// (same discipline as attention.hlsl's out-of-range clamp).
//
// Selected by ZEN_D3D12_ATTN2=1 (default off). ZEN_D3D12_ATTN2_MINPOS=N (default
// 0) optionally gates on context: pos < N uses attention.hlsl.
//
// Deliberately unsupported (the caller keeps those layers on the CPU path): ALiBi.

#define ATTN2_MAX_HEAD_DIM 128

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

#define LANES2 32u
#define THREADS2 64u

// Identical to attention.hlsl's warp_sum: a 32-lane butterfly reduction over
// the caller's 32-lane half. base is lane-rebased, so on wave64 hardware the
// two halves never fold into each other, and on wave32 hardware the two halves
// are separate waves and each reduces its own head.
float warp_sum2(float v, uint lane)
{
    const uint base = WaveGetLaneIndex() - lane;
    [unroll]
    for (uint s = LANES2 / 2u; s > 0u; s >>= 1u)
    {
        const float other = WaveReadLaneAt(v, base + ((lane + s) & (LANES2 - 1u)));
        if (lane + s < LANES2)
        {
            v += other;
        }
    }
    return v;
}

[numthreads(THREADS2, 1, 1)]
void Attention2(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID)
{
    const uint head = gid.x * 2u + (tid / LANES2);
    const uint lane = tid % LANES2;
    // No early return: every lane must reach the shuffles and the block
    // barriers, or the reduction reads a lane that already exited. Out-of-range
    // heads clamp to head 0 and simply do not write.
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
        for (uint c = lane; c < head_dim; c += LANES2)
        {
            acc += asfloat(qraw.Load((hbase + c) * 4u)) * asfloat(kraw.Load((kbase + c) * 4u));
        }
        acc = warp_sum2(acc, lane);
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
    for (uint d = lane; d < head_dim; d += LANES2)
    {
        float acc = 0.0f;
        for (uint t5 = first_pos; t5 <= pos; ++t5)
        {
            const float w = outv[scores_off + h * context + t5];
            acc += w * asfloat(vraw.Load((v_base + t5 * row_stride + kh * head_dim + d) * 4u));
        }
        outv[out_off + hbase + d] = acc;
    }
}
