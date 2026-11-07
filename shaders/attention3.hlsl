// ZenInfer D3D12 backend: single-token (decode) GQA attention, KV window split
// across the waves of one workgroup.
//
// Why this kernel exists
// ----------------------
// attention.hlsl (and attention2.hlsl) give every query head exactly one wave and
// walk the visible KV window in a strictly serial loop: load the K row, reduce the
// dot over the wave, publish the score through global memory, then run the softmax
// as three more serial passes over the same global scratch from a single lane.
// Every position therefore costs a dependent global-memory round trip and there is
// only one wave per CU to hide it with. Measured on an RX 9070 XT with
// Llama-3.2-1B (32 heads, 8 KV heads, head_dim 64): 1.35 us per visible position
// per layer, ~0.02 us of which is the actual KV traffic at peak bandwidth, and a
// measured cost that grows linearly with position (80 us per layer at pos=64,
// 146 us at pos=113). At an 8k context that is ~177 ms per token of attention
// alone, and it was already 1.3 ms/token at a 64-token context.
//
// This kernel keeps the same arithmetic (online softmax) but removes the serial
// structure:
//   * THREADS3 threads per head = several waves per workgroup, and wave w handles
//     positions first_pos + w, + waves, ... so the window is covered in parallel.
//   * Each wave keeps its own running max/denominator and V accumulator in
//     registers (one element per lane when head_dim <= lanes), so a position costs
//     no global round trip at all: the score reduction is a wave sum.
//   * The per-wave partial softmax states are merged once, through groupshared
//     memory, at the end. Merging online-softmax states is exact softmax up to
//     floating-point association; no scratch buffer and no global barrier.
// Grid is `heads` (one workgroup per head). K and V are read as UAVs exactly as in
// attention.hlsl (rwbyteaddress, whole-buffer base, layer slice in the cbuffer), so
// callers keep the caches in UNORDERED_ACCESS and need no extra transition.
//
// The scores scratch region of the UAV (scores_off) is unused: the old kernel
// published scores there only because it had no other way to carry them between
// passes.
//
// Measured result (RX 9070 XT, Llama-3.2-1B Q8_0, zen_d3d12_metrics 3 runs each):
//   per-layer attention, pos=64:  80 us -> 15.1 us (at 256 threads; 1024 is faster
//   still on the whole-step figure below)
//   per-layer attention slope:    1.35 us/position -> 0.14 us/position
//   decode, 64-token context:     5.46 -> 3.50 ms/token (183 -> 286 tok/s)
//   decode, 512-token context:    9.18 -> 3.54 ms/token (109 -> 283 tok/s)
// Token ids are unchanged against the llama.cpp oracles, and zen_d3d12_attn_parity
// keeps max_abs_err <= 1.8e-07 over its shape sweep.
//
// Constraints enforced by the caller: head_dim <= 128 (MAXD3 * 64) and grid.x =
// heads. Deliberately unsupported: ALiBi (those layers stay on the CPU path).

#define ATTN3_MAX_HEAD_DIM 128
#define ATTN3_THREADS 1024u           // 16 waves on this wave64 part
#define ATTN3_MAX_WAVES 32u           // 1024 threads / 32 lanes, the wave32 case
#define ATTN3_MAXD 4u                 // 4 * 64 = 256 >= 128, covers both wave sizes

// The wave count per head is the tuning knob: more waves is more positions in
// flight per workgroup and more waves resident per CU, at the cost of a wider
// final merge (ATTN3_MAX_WAVES entries). Measured on the RX 9070 XT with
// Llama-3.2-1B at pos=64, decode ms/token: 256 threads 3.83, 512 3.69, 1024 3.63.
// The curve is flat enough that 1024 is the pick; the old one-wave serial kernel
// was 5.60 ms/token at the same shape.

cbuffer AttnParams : register(b0)
{
    uint heads;        // query heads
    uint kv_heads;     // key/value heads (GQA group = heads / kv_heads)
    uint head_dim;     // elements per head
    uint context;      // allocated positions in the cache
    uint first_pos;    // first visible position (sliding window start)
    uint pos;          // position of the query token, inclusive
    uint scores_off;   // unused here (kept for cbuffer compatibility)
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

groupshared float sh_m[ATTN3_MAX_WAVES];
groupshared float sh_l[ATTN3_MAX_WAVES];
groupshared float sh_acc[ATTN3_MAX_WAVES][ATTN3_MAX_HEAD_DIM];

[numthreads(ATTN3_THREADS, 1, 1)]
void Attention3(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID)
{
    if (gid.x >= heads) {
        return;                        // uniform per group, so no barrier is skipped
    }

    const uint lanes = WaveGetLaneCount();
    const uint waves = ATTN3_THREADS / lanes;
    const uint wave = tid / lanes;
    const uint lane = WaveGetLaneIndex();

    const uint h = gid.x;
    const uint group = max(1u, heads / kv_heads);
    const uint kh = h / group;
    const uint row_stride = kv_heads * head_dim;
    const uint hbase = h * head_dim;

    // Q stays in registers for the whole kernel: one element per lane, plus the
    // extra round at head_dim > lanes.
    float qv[ATTN3_MAXD];
    float acc[ATTN3_MAXD];
    [unroll]
    for (uint i = 0u; i < ATTN3_MAXD; ++i) {
        qv[i] = 0.0f;
        acc[i] = 0.0f;
    }
    [unroll]
    for (uint iq = 0u; iq < ATTN3_MAXD; ++iq) {
        const uint d = lane + iq * lanes;
        if (d < head_dim) {
            qv[iq] = asfloat(qraw.Load((hbase + d) * 4u));
        }
    }

    // Online softmax per wave. WaveActiveSum reduces across the whole wave, which
    // is exactly the head_dim slice this wave owns, so the dot needs no barrier.
    float m = -3.402823466e+38f;
    float l = 0.0f;
    for (uint t = first_pos + wave; t <= pos; t += waves) {
        const uint kb = kv_base + t * row_stride + kh * head_dim;
        const uint vb = v_base + t * row_stride + kh * head_dim;

        float partial = 0.0f;
        [unroll]
        for (uint ik = 0u; ik < ATTN3_MAXD; ++ik) {
            const uint d = lane + ik * lanes;
            if (d < head_dim) {
                partial += qv[ik] * asfloat(kraw.Load((kb + d) * 4u));
            }
        }
        const float s = WaveActiveSum(partial) * inv_scale;

        const float nm = max(m, s);
        const float corr = exp(m - nm);   // 0 on the first iteration (m = -FLT_MAX)
        const float p = exp(s - nm);
        l = l * corr + p;
        [unroll]
        for (uint iv = 0u; iv < ATTN3_MAXD; ++iv) {
            const uint d = lane + iv * lanes;
            if (d < head_dim) {
                acc[iv] = acc[iv] * corr + p * asfloat(vraw.Load((vb + d) * 4u));
            }
        }
        m = nm;
    }

    // Publish this wave's partial state, then let wave 0 combine them.
    if (lane == 0u) {
        sh_m[wave] = m;
        sh_l[wave] = l;
    }
    [unroll]
    for (uint ia = 0u; ia < ATTN3_MAXD; ++ia) {
        const uint d = lane + ia * lanes;
        if (d < head_dim) {
            sh_acc[wave][d] = acc[ia];
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (wave == 0u) {
        float gm = -3.402823466e+38f;
        for (uint w = 0u; w < waves; ++w) {
            gm = max(gm, sh_m[w]);
        }
        float gl = 0.0f;
        for (uint w2 = 0u; w2 < waves; ++w2) {
            gl += sh_l[w2] * exp(sh_m[w2] - gm);   // idle waves have l = 0, m = -inf
        }
        const float inv = (gl > 0.0f) ? (1.0f / gl) : 0.0f;
        [unroll]
        for (uint io = 0u; io < ATTN3_MAXD; ++io) {
            const uint d = lane + io * lanes;
            if (d < head_dim) {
                float a = 0.0f;
                for (uint w3 = 0u; w3 < waves; ++w3) {
                    a += sh_acc[w3][d] * exp(sh_m[w3] - gm);
                }
                outv[out_off + hbase + d] = a * inv;
            }
        }
    }
}
