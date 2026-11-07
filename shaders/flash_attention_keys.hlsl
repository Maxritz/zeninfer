// ZenInfer D3D12 backend: prefill FlashAttention with one lane per key.
//
// Why this kernel exists
// ----------------------
// Both earlier kernels put head_dim across the lanes, so a wave produces one
// score per iteration and therefore needs one WaveActiveSum per key. That
// reduction sits in the middle of a dependency chain - dot, reduce, exp,
// rescale, accumulate - so the wave spends most of each key waiting on it.
// Measured ablation at 512 tokens: the per-key chain, not the bytes, is what the
// dispatch costs (cutting K/V traffic 4x with a groupshared tile made it slower,
// 1795 us/layer against 1330).
//
// This kernel turns the mapping around: one lane owns one key. The lane holds a
// whole key row in its own registers from the groupshared tile and finishes the
// entire 128-wide dot product in-lane, so the score for 32 keys appears with no
// cross-lane work at all. What is left across lanes is the tile's softmax
// statistics: one WaveActiveMax and one WaveActiveSum per tile of 32 keys - 1/32
// of the reductions per key - and one rescale of the accumulator per tile rather
// than per key.
//
// The trade this makes is that a key row is read transposed (lane j wants all
// 128 dimensions of key j), which no coalesced load can serve, so each tile's K
// rows are staged into groupshared memory first (coalesced, one 512-byte row per
// lane pass) and read back with a row stride padded to 33 float4 so the
// transposed reads stay bank-conflict-free. V needs no staging: it is read with
// dimensions across the lanes, exactly as before.
//
// Workgroup layout: 32 lanes per query head, four heads per workgroup (four waves on
// a wave32 device, two on a wave64 one - the mapping is per 32-lane block either way),
// and those four heads are the GQA group of a single KV head, so one staged K tile
// serves all four. That is what keeps the shared-memory footprint at one tile per four
// waves (19 KB/group, 3 groups per 64 KB CU = 12 waves) instead of 17 KB per wave.
// The host selects this kernel only when heads / kv_heads == WAVES and head_dim is
// exactly 128; everything else keeps the 8-token kernel
// (flash_attention_tile.hlsl) or the legacy one.
//
// Causal mask: lane j owns key tbase + lane and contributes nothing unless
// first_pos <= key < j_end, where j_end = min(pos0 + 1, valid_kv_tokens). Masked
// lanes get -FLT_MAX, whose exp underflows to 0, so the V accumulation stays
// exact without a branch.
//
// Numerics: the max is taken over 32 keys at once and the accumulator is rescaled
// once per tile, so results differ from the per-key recurrence in the last bits
// (they are, if anything, closer to the exact softmax). The model-level token ids
// and zen_d3d12_flash_parity are the checks; both are run for every change here.

#define WAVES     4u          // query heads per workgroup (host-gated: heads / kv_heads)
#define NTHREADS  (WAVES * 32u)
#define KT        32u         // keys per tile: one per lane
#define KROW      (KT + 1u)   // padded float4 row stride: 33 * 16 B = 528 B
#define FP4       32u         // float4 lanes per 128-wide row
#define FNEG      (-3.402823466e+38f)

cbuffer FlashAttnParams : register(b0)
{
    uint  tokens;          // query tokens in this prefill batch
    uint  heads;           // query heads
    uint  kv_heads;        // key/value heads (GQA)
    uint  head_dim;        // head dimension (always 128 here)
    float inv_scale;       // 1 / sqrt(head_dim)
    uint  kv_cache_base;   // element offset of this layer inside the KV cache
    uint  kv_cache_stride; // elements between consecutive tokens (kv_heads * head_dim)
    uint  valid_kv_tokens; // absolute count of populated KV positions
    uint  start_pos;       // absolute position of query token 0
    uint  first_pos;       // absolute first KV position to attend to
    uint  pad0;
    uint  pad1;
};

// The root descriptors carry no format or stride, so viewing each float buffer as
// float4 is a pure addressing change. Every offset below is a multiple of four
// because head_dim is 128 and the KV cache stride is a multiple of it.
StructuredBuffer<float4> q4buf : register(t0);   // [tokens, heads, 32]
StructuredBuffer<float4> k4buf : register(t1);   // flat KV cache
StructuredBuffer<float4> v4buf : register(t2);
RWStructuredBuffer<float4> o4buf : register(u0); // [tokens, heads, 32]

groupshared float4 k_tile[KT][KROW];
groupshared float4 q_row[WAVES][KT];

// Reduce within the 32-lane block that owns one head. The wave on this class of
// hardware may be 32 or 64 lanes wide (the device reports min 32, max 64), and a
// whole-wave reduction would then mix two heads' scores, so the tree is explicit:
// XOR masks below 32 never leave a 32-lane block.
float block_max(float v) {
    const uint l = WaveGetLaneIndex();
    [unroll]
    for (uint m = 1u; m < 32u; m <<= 1u) {
        v = max(v, WaveReadLaneAt(v, l ^ m));
    }
    return v;
}

float block_sum(float v) {
    const uint l = WaveGetLaneIndex();
    [unroll]
    for (uint m = 1u; m < 32u; m <<= 1u) {
        v += WaveReadLaneAt(v, l ^ m);
    }
    return v;
}

[numthreads(NTHREADS, 1, 1)]
void FlashAttentionKeys(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID)
{
    // 32 lanes per head: lane is the key index and wave selects the head, so every
    // cross-lane op below stays inside one head's 32 scores.
    const uint lane = tid & 31u;
    const uint wave = tid >> 5u;
    const uint h = gid.x * WAVES + wave;
    const uint token = gid.y;

    const uint gqa = max(1u, heads / kv_heads);
    const uint kv_head = h / gqa;
    const uint pos0 = start_pos + token;
    // Row of this (token, head) in the Q/O buffers, in float4 units.
    const uint row4 = (token * heads + h) * FP4;

    q_row[wave][lane] = q4buf[row4 + lane];

    const uint stride4 = kv_cache_stride >> 2;
    const uint base4 = (kv_cache_base >> 2) + kv_head * FP4;
    const uint j_end = min(pos0 + 1u, valid_kv_tokens);

    float m = FNEG;
    float l = 0.0f;
    float4 acc = float4(0.0f, 0.0f, 0.0f, 0.0f);

    // Tiles are aligned to KT so that lane j's key is always tbase + lane.
    for (uint tbase = (first_pos / KT) * KT; tbase < j_end; tbase += KT) {
        // Stage the tile: one pass of one row per wave-width of threads, so each
        // load is a 512-byte coalesced fetch of a whole key row.
        [unroll]
        for (uint p = 0u; p < (KT * FP4) / NTHREADS; ++p) {
            const uint u = tid + p * NTHREADS;
            const uint row = u >> 5u;
            const uint c = u & 31u;
            const uint j = tbase + row;
            float4 kv = float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (j < j_end) {
                kv = k4buf[base4 + j * stride4 + c];
            }
            k_tile[row][c] = kv;
        }
        GroupMemoryBarrierWithGroupSync();

        // This lane's key: a full 128-wide dot with no cross-lane reduction.
        const uint jk = tbase + lane;
        float s0 = 0.0f;
        float s1 = 0.0f;
        float s2 = 0.0f;
        float s3 = 0.0f;
        [unroll]
        for (uint c = 0u; c < FP4; ++c) {
            // q_row is read by every lane of the wave from one address (broadcast);
            // k_tile[lane] is this lane's own row, 33 float4 apart, which lands on
            // a distinct 16-byte bank group for each of the eight lanes in a phase.
            const float4 qv = q_row[wave][c];
            const float4 kv = k_tile[lane][c];
            s0 += qv.x * kv.x;
            s1 += qv.y * kv.y;
            s2 += qv.z * kv.z;
            s3 += qv.w * kv.w;
        }
        float s = ((s0 + s1) + (s2 + s3)) * inv_scale;
        // j_end already bounds the causal end (j_end <= pos0 + 1), so this one test
        // covers both the sliding-window start and the causal mask.
        if (jk < first_pos || jk >= j_end) {
            s = FNEG;
        }

        // One max and one sum per 32 keys, both wave-uniform.
        const float smax = block_max(s);
        if (smax != FNEG) {
            const float mn = max(m, smax);
            // The first tile has m == FNEG, and exp(-FLT_MAX) underflows to 0, so
            // the initial rescale is exact without a special case.
            const float corr = exp(m - mn);
            const float p = exp(s - mn);   // masked lanes: exp(-FLT_MAX) == 0

            l = l * corr + block_sum(p);
            acc *= corr;
            // Broadcast each key's probability instead of staging P: a shuffle is
            // cheaper than a groupshared round trip plus another barrier. The index
            // is absolute within the wave, so it is anchored to this head's 32-lane
            // block - on a 64-lane wave lanes 0..31 belong to another head.
            const uint blk = WaveGetLaneIndex() & ~31u;
            [loop]
            for (uint jj = 0u; jj < KT; ++jj) {
                const float pj = WaveReadLaneAt(p, blk + jj);
                // Keys past the tile end carry p == 0, so re-reading the last valid
                // key keeps the read in bounds without changing the sum. A one-deep
                // prefetch of this row was measured and is slower by ~3%: the V loads
                // are already overlapped, and the extra registers cost occupancy.
                const uint jr = min(tbase + jj, j_end - 1u);
                acc += pj * v4buf[base4 + jr * stride4 + lane];
            }
            m = mn;
        }
        // The next tile overwrites k_tile, so no wave may still be reading it.
        GroupMemoryBarrierWithGroupSync();
    }

    const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    o4buf[row4 + lane] = acc * inv;
}
