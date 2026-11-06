// ZenInfer D3D12 backend: Q8_0 matrix-vector product, DP4A variant (decode path).
//
// NOT the parity path: activations are quantised to int8 per 32-element block so
// the inner product can use v_dot4 (dot4add_i8packed). One weight row is fully
// consumed per output element (decode is T=1), so the whole row is streamed: the
// loop is bounded by how fast blocks leave VRAM, with no per-element scalar
// dequant anywhere.
//
// Why this exists
// ---------------
// The fp32 kernel (q8_0_gemv.hlsl) extracts every byte, converts it to float,
// sign-extends it, and FMA's it: at 1.24 GB of weights per token that is
// ~1.24 billion convoluted conversions before any bandwidth is even touched.
// The weights are already int8 in memory in 32-byte blocks, the activations
// quantise to int8 with one scale per block, so the dot is an integer
// instruction by construction.
//
// Mapping: 256 threads, rows_per_group rows per group, one warp (32 lanes) per
// row. Lane l owns blocks (iter*16 + l/2), a uint4 (16 bytes, half of a block)
// at (l&1)*16 within the block, so a warp covers 16 complete 32-byte blocks per
// iteration and the 32 lanes read 16 coalesced 16-byte spans in one pass. Each
// lane does four dot4 ops per iteration; the warp reduces once per row. Each
// iteration's integer partial is converted with that block's own weight scale
// and activation scale (scales differ per block, so they cannot be folded over
// the whole-row integer accumulator), then reduced.
//
// Buffers: qbuf/sbuf are the repacked (aligned) quant/scale streams; xraw is
// the int8 activation stream, uint32-packed; xscale is one fp32 scale per block.
// Both activation buffers are bound as UAVs (u1/u2) so the host can keep the
// staging resource in UNORDERED_ACCESS permanently and order the quantizer's
// writes with a single UAV barrier (no state transitions). The kernel stages
// both into shared memory once per group; every warp in the group then reads
// them from LDS instead of re-fetching the full activation vector from VRAM
// once per row (without staging a 2048-col matvec re-reads 4-6 MB of
// activations, comparable to the weight stream itself).
//
// Accuracy: activations quantise per 32-element block (max/127) against the
// weights' own per-block scales, matching the DP4A prefill path (~4e-3
// relative, inside the Q8_0 model's tolerance) but not bit-parity with the
// fp32 kernel, hence opt-in via ZEN_D3D12_DP4A=1.
//
// Requires Shader Model 6.4 for dot4add_i8packed.

cbuffer GemvParams : register(b0)
{
    uint rows;            // output rows
    uint blocks_per_row;  // 32-element blocks per row (columns / 32)
    uint rows_per_group;  // warps (rows) scheduled per thread group
    uint reserved;
};

ByteAddressBuffer         qbuf   : register(t0);  // 32 bytes per block
StructuredBuffer<float>   sbuf   : register(t1);  // one fp32 scale per block
RWByteAddressBuffer       xraw   : register(u1);  // int8 activations
RWStructuredBuffer<float> xscale : register(u2);  // one fp32 scale per block
RWStructuredBuffer<float> yvec   : register(u0);  // fp32 output row

#define LANES             32u   // one warp per row
#define BLOCKS_PER_ITER   16u   // 16 blocks per warp-iteration (2 uint4 lanes per 32B block)
#define STAGE_MAX_BLOCKS  256u  // cols <= 8192 stages in shared memory
#define XSTAGE_U4         512u  // 8192 int8 elems as uint4 chunks
#define XSTAGE_SCSS       256u  // one fp32 scale per block

groupshared uint4 qa[XSTAGE_U4];   // staged int8 activations (8 KB at max cols)
groupshared float xs[XSTAGE_SCSS]; // staged per-block activation scales (1 KB)

float warp32_sum(float v, uint lane)
{
    const uint base = WaveGetLaneIndex() - lane;
    [unroll]
    for (uint s = 16u; s > 0u; s >>= 1u)
    {
        const float other = WaveReadLaneAt(v, base + ((lane + s) & 31u));
        if (lane + s < 32u)
        {
            v += other;
        }
    }
    return v;
}

[numthreads(256, 1, 1)]
void GemvQ8_0Dp4a(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID)
{
    const uint local_row = tid / LANES;
    const uint lane      = tid % LANES;
    const uint row       = gid.x * rows_per_group + local_row;

    // Stage the int8 activation vector + its scale table once per group. The
    // staging must run before the row early-out: rows in a partial group never
    // compute but still have to fill their chunks (and the group barrier below
    // must see every thread). Every warp in the group then reads activations
    // from LDS; without this each row re-fetches the whole vector from VRAM.
    const bool staged = blocks_per_row <= STAGE_MAX_BLOCKS;
    if (staged)
    {
        const uint chunks = blocks_per_row * 2u;  // uint4 halves per block
        for (uint i = tid; i < chunks; i += 256u)
        {
            qa[i] = xraw.Load<uint4>(i * 16u);
        }
        for (uint b = tid; b < blocks_per_row; b += 256u)
        {
            xs[b] = xscale[b];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (row >= rows)
    {
        return;
    }

    const uint qbase = row * blocks_per_row * 32u;
    const uint sbase = row * blocks_per_row;

    // Vectorized loads: each lane pulls a uint4 (16B, half a 32-byte block), so
    // two lanes cover one block and the warp covers 16 blocks per iteration.
    // This mirrors the fp32 kernel's float4 load width — the previous scalar
    // u32 mapping (4 blocks per iteration, 4 bytes per lane) was 4x weaker on
    // memory parallelism and measurably slower than fp32 despite doing less work.
    const uint blk_in_iter = lane >> 1u;    // which of the 16 blocks this lane owns
    const uint byte_off    = (lane & 1u) * 16u;

    float facc = 0.0f;
    for (uint b = 0; b < blocks_per_row; b += BLOCKS_PER_ITER)
    {
        const uint iter_block = b + blk_in_iter;
        if (iter_block >= blocks_per_row)
        {
            continue;
        }
        const uint4 q = qbuf.Load<uint4>(qbase + iter_block * 32u + byte_off);
        const uint4 a = staged
            ? qa[iter_block * 2u + (lane & 1u)]
            : xraw.Load<uint4>(iter_block * 32u + byte_off);
        const int partial =
            dot4add_i8packed(q.x, a.x, 0) + dot4add_i8packed(q.y, a.y, 0) +
            dot4add_i8packed(q.z, a.z, 0) + dot4add_i8packed(q.w, a.w, 0);
        const float as = staged ? xs[iter_block] : xscale[iter_block];
        facc += (float)partial * sbuf[sbase + iter_block] * as;
    }

    const float total = warp32_sum(facc, lane);
    if (lane == 0u)
    {
        yvec[row] = total;
    }
}
