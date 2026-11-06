// ZenInfer D3D12 backend: fp32 activation -> int8 (decode DP4A path).
//
// Quantises one activation vector into 32-element blocks, exactly mirroring
// quantise_acts_q8_0 (kernels/avx2.cpp): per block, d = amax/127, each element
// becomes round(clamp(x/d, -127, 127)), and the block scale d is stored. The
// int8-activation DP4A kernels (q8_0_gemv_dp4a, q8_0_gemm_dp4a) consume this
// form, so the decode path becomes pure integer FMA per block.
//
// One thread per block: 32-element loads, a per-thread max over the 32
// components (the thread owns the whole block, so no cross-thread or shuffle
// reduction is needed), then the quantise + stores. No transcendentals,
// so the rounding matches the CPU reference bit-for-bit.

int QuantizeScalar(float x, float inv)
{
    const float v = clamp(x * inv, -127.0f, 127.0f);
    // Half-to-nearest-away-from-zero, matching std::lround (HLSL round() rounds
    // half to even, which differs at exact .5 boundaries).
    return (int)(v >= 0.0f ? floor(v + 0.5f) : ceil(v - 0.5f));
}

cbuffer QuantParams : register(b0)
{
    uint blocks;     // columns / 32
    uint columns;    // element count of the source vector
};

ByteAddressBuffer         xraw    : register(t0);  // fp32 source vector
RWStructuredBuffer<float> sscale  : register(u1);  // one fp32 scale per block (rw, so u1/UAV root slot)
RWByteAddressBuffer       qout    : register(u0);  // int8 block stream (bytes)

[numthreads(256, 1, 1)]
void QuantizeActs(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID)
{
    const uint b = gid.x * 256u + tid;
    if (b >= blocks)
    {
        return;
    }

    const uint base = b * 32u;

    float4 v0 = asfloat(xraw.Load<float4>(base * 4u));
    float4 v1 = asfloat(xraw.Load<float4>((base + 4u) * 4u));
    float4 v2 = asfloat(xraw.Load<float4>((base + 8u) * 4u));
    float4 v3 = asfloat(xraw.Load<float4>((base + 12u) * 4u));
    float4 v4 = asfloat(xraw.Load<float4>((base + 16u) * 4u));
    float4 v5 = asfloat(xraw.Load<float4>((base + 20u) * 4u));
    float4 v6 = asfloat(xraw.Load<float4>((base + 24u) * 4u));
    float4 v7 = asfloat(xraw.Load<float4>((base + 28u) * 4u));

    // Per-thread max over the 32 components (the thread owns the whole block,
    // so no cross-thread reduction is needed). First pass: the model's
    // activation magnitudes are the reference amax, so recompute anyway.
    float am = max(max(abs(v0.x), abs(v0.y)), max(abs(v0.z), abs(v0.w)));
    am = max(am, max(max(abs(v1.x), abs(v1.y)), max(abs(v1.z), abs(v1.w))));
    am = max(am, max(max(abs(v2.x), abs(v2.y)), max(abs(v2.z), abs(v2.w))));
    am = max(am, max(max(abs(v3.x), abs(v3.y)), max(abs(v3.z), abs(v3.w))));
    am = max(am, max(max(abs(v4.x), abs(v4.y)), max(abs(v4.z), abs(v4.w))));
    am = max(am, max(max(abs(v5.x), abs(v5.y)), max(abs(v5.z), abs(v5.w))));
    am = max(am, max(max(abs(v6.x), abs(v6.y)), max(abs(v6.z), abs(v6.w))));
    am = max(am, max(max(abs(v7.x), abs(v7.y)), max(abs(v7.z), abs(v7.w))));

    // Half-to-nearest-away-from-zero, matching std::lround (HLSL round() rounds
    // half to even, which differs at exact .5 boundaries).
    const float d = (am > 0.0f) ? (am / 127.0f) : 0.0f;
    const float inv = (d > 0.0f) ? (1.0f / d) : 0.0f;
    sscale[b] = d;

    int packed = 0;
    packed |= (uint)(QuantizeScalar(v0.x, inv) & 0xff) << 0;
    packed |= (uint)(QuantizeScalar(v0.y, inv) & 0xff) << 8;
    packed |= (uint)(QuantizeScalar(v0.z, inv) & 0xff) << 16;
    packed |= (uint)(QuantizeScalar(v0.w, inv) & 0xff) << 24;
    qout.Store<uint>(base, packed);
    packed = 0;
    packed |= (uint)(QuantizeScalar(v1.x, inv) & 0xff) << 0;
    packed |= (uint)(QuantizeScalar(v1.y, inv) & 0xff) << 8;
    packed |= (uint)(QuantizeScalar(v1.z, inv) & 0xff) << 16;
    packed |= (uint)(QuantizeScalar(v1.w, inv) & 0xff) << 24;
    qout.Store<uint>(base + 4u, packed);
    packed = 0;
    packed |= (uint)(QuantizeScalar(v2.x, inv) & 0xff) << 0;
    packed |= (uint)(QuantizeScalar(v2.y, inv) & 0xff) << 8;
    packed |= (uint)(QuantizeScalar(v2.z, inv) & 0xff) << 16;
    packed |= (uint)(QuantizeScalar(v2.w, inv) & 0xff) << 24;
    qout.Store<uint>(base + 8u, packed);
    packed = 0;
    packed |= (uint)(QuantizeScalar(v3.x, inv) & 0xff) << 0;
    packed |= (uint)(QuantizeScalar(v3.y, inv) & 0xff) << 8;
    packed |= (uint)(QuantizeScalar(v3.z, inv) & 0xff) << 16;
    packed |= (uint)(QuantizeScalar(v3.w, inv) & 0xff) << 24;
    qout.Store<uint>(base + 12u, packed);
    packed = 0;
    packed |= (uint)(QuantizeScalar(v4.x, inv) & 0xff) << 0;
    packed |= (uint)(QuantizeScalar(v4.y, inv) & 0xff) << 8;
    packed |= (uint)(QuantizeScalar(v4.z, inv) & 0xff) << 16;
    packed |= (uint)(QuantizeScalar(v4.w, inv) & 0xff) << 24;
    qout.Store<uint>(base + 16u, packed);
    packed = 0;
    packed |= (uint)(QuantizeScalar(v5.x, inv) & 0xff) << 0;
    packed |= (uint)(QuantizeScalar(v5.y, inv) & 0xff) << 8;
    packed |= (uint)(QuantizeScalar(v5.z, inv) & 0xff) << 16;
    packed |= (uint)(QuantizeScalar(v5.w, inv) & 0xff) << 24;
    qout.Store<uint>(base + 20u, packed);
    packed = 0;
    packed |= (uint)(QuantizeScalar(v6.x, inv) & 0xff) << 0;
    packed |= (uint)(QuantizeScalar(v6.y, inv) & 0xff) << 8;
    packed |= (uint)(QuantizeScalar(v6.z, inv) & 0xff) << 16;
    packed |= (uint)(QuantizeScalar(v6.w, inv) & 0xff) << 24;
    qout.Store<uint>(base + 24u, packed);
    packed = 0;
    packed |= (uint)(QuantizeScalar(v7.x, inv) & 0xff) << 0;
    packed |= (uint)(QuantizeScalar(v7.y, inv) & 0xff) << 8;
    packed |= (uint)(QuantizeScalar(v7.z, inv) & 0xff) << 16;
    packed |= (uint)(QuantizeScalar(v7.w, inv) & 0xff) << 24;
    qout.Store<uint>(base + 28u, packed);
}
