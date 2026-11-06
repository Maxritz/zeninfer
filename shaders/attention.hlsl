// ZenInfer D3D12 backend: single-token (decode) grouped-query attention.
//
// One WARP per query head. The old version ran one 128-thread workgroup per head and
// did a full block reduction for every cached position: 7 GroupMemoryBarrierWithGroupSync
// calls per position, so a 256-token context cost ~1800 barriers per head, serially.
// Attention was the only O(context) term in the layer and it dominated.
//
// Now the per-position dot is a warp reduction (5 shuffles, zero barriers) and the whole
// kernel contains no block barrier at all. Four warps share a workgroup, so occupancy is
// unchanged; each warp is fully independent, which also means an early return on
// head >= heads is safe (no barrier can be skipped by part of a group).
//
// The KV cache is [context][kv_heads][head_dim], which is what makes the append a single
// contiguous copy per token. A [kv_head][context][head_dim] layout would interleave it.
//
// K and V are read as UAVs (RWByteAddressBuffer, u1/u2), not SRVs. The decode pipeline
// writes the current token's k row (rope-k) and v row (the v matvec) straight into the
// caches through compute UAV writes, so keeping the caches in UNORDERED_ACCESS permanently
// removes both CopyBufferRegions and all four per-layer state transitions on the 256MB
// cache buffers - the copy-visibility window that dominated the per-layer time. The price
// is two UAV barriers per layer: one ordering the compute writes before this dispatch's
// reads, one ordering these reads before the next layer's writes. Loads through
// RWByteAddressBuffer are bit-identical to ByteAddressBuffer loads.
//
// kv_base/v_base carry the element offset of this layer's [context][kv_heads][head_dim]
// slice inside the whole-cache buffer, so the binding is always the 256B-aligned buffer
// base (root UAV addresses must be 256B aligned; a per-layer binding offset would not be
// for every shape).
//
// Constraints enforced by the caller: head_dim <= 128 and grid.x * 4 >= heads. Scores and
// the output are disjoint regions of the one UAV slot the root signature exposes.
//
// Deliberately unsupported (the caller keeps those layers on the CPU path): ALiBi.

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

#define WARPS_PER_GROUP 1u
#define LANES 32u

// Reduction over exactly 32 lanes. This GPU runs wave64, so WaveActiveSum (or any
// wave-wide read) would fold two adjacent logical warps - two different heads - into
// one sum. The lane index is therefore rebased onto the caller's 32-lane group and the
// partial lanes past the end discard their wrapped value.
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

// One 32-thread group per head. Packing four heads into a 128-thread group made the
// kernel occupy a quarter of the CUs (8 groups instead of 32), which cost more than the
// barriers it removed: measured 120.1 tok/s against 122.1 for the old block-reduction
// kernel. One group per head restores the CU count while keeping the shuffle reduction.
[numthreads(LANES, 1, 1)]
void Attention(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID)
{
    const uint head = gid.x * WARPS_PER_GROUP + (tid / LANES);
    const uint lane = tid % LANES;
    // No early return: every lane must reach the shuffles and the block barriers, or
    // the reduction reads a lane that already exited. Out-of-range warps clamp to head
    // 0 and simply do not write.
    const bool active = head < heads;
    const uint h = active ? head : 0u;

    const uint group = max(1u, heads / kv_heads);
    const uint kh = h / group;
    const uint row_stride = kv_heads * head_dim;
    const uint hbase = h * head_dim;

    // ---- 1. scores ---------------------------------------------------------
    // Lane-strided over head_dim so the whole head fits in a few registers; the
    // reduction is a shuffle tree, so no barrier and no groupshared tile.
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
    // On lane 0: one exp per position, which is cheaper than another reduction round.
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
}
