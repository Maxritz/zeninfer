// ZenInfer D3D12 backend: RMSNorm + int8 activation quantize, fused (decode DP4A).
//
// Fused producer variant of rmsnorm.hlsl for the DP4A decode path: the RMSNorm
// dispatch also quantises its output vector into the DP4A staging streams
// (qact int8 acts at u3 + per-32-block scales at u4), so the separate
// quantize_acts dispatch per site can be skipped when ZEN_D3D12_DP4A_FUSEQ=1.
// The fp32 output is still written (u0) alongside the int8 form, so the fp32
// GEMV inputs and all resource-state transitions stay valid.
//
// Quantize math is bit-identical to quantize_acts.hlsl: per 32-element block,
// d = amax/127, x -> round(clamp(x/d, -127, 127)) half-away-from-zero
// (std::lround), scale d stored. The amax is taken over the values just
// written to outv (read back after a device barrier), which is exactly the
// vector quantize_acts would have read, so the int8 stream and scales match
// the unfused path bit for bit.
//
// Only row 0 is quantised (the decode pipeline/legacy paths dispatch rows=1).
// Rows past 0 write fp32 only, keeping multi-row use of this PSO safe.

cbuffer RmsParams : register(b0)
{
    uint  count;      // elements per row
    uint  rows;
    float epsilon;
    uint  pad;
};

ByteAddressBuffer         xraw : register(t0);    // [row][count]
StructuredBuffer<float>   wvec : register(t1);    // [count]
RWStructuredBuffer<float> outv : register(u0);    // [row][count] fp32
RWByteAddressBuffer       qactv : register(u3);   // int8 acts (fused quantize)
RWStructuredBuffer<float> qscale : register(u4);  // one fp32 scale per block

groupshared float partial[256];

int QuantizeScalar(float x, float inv)
{
    const float v = clamp(x * inv, -127.0f, 127.0f);
    return (int)(v >= 0.0f ? floor(v + 0.5f) : ceil(v - 0.5f));
}

[numthreads(256, 1, 1)]
void RmsNormFused(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID)
{
    const uint row = gid.x;
    const uint base = row * count;

    float ss = 0.0f;
    for (uint i = tid; i < count; i += 256u)
    {
        const float v = asfloat(xraw.Load((base + i) * 4u));
        ss += v * v;
    }

    partial[tid] = ss;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint s = 128u; s > 0u; s >>= 1u)
    {
        if (tid < s)
        {
            partial[tid] += partial[tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    const float scale = rsqrt(partial[0] / (float)count + epsilon);
    for (uint i = tid; i < count; i += 256u)
    {
        const float v = asfloat(xraw.Load((base + i) * 4u));
        outv[base + i] = v * scale * wvec[i];
    }

    // Fused quantize: one thread per 32-element block of row 0, exactly
    // mirroring quantize_acts.hlsl (thread per block, per-thread max over the
    // 32 components). Read back the just-written fp32 row so amax sees the
    // stored values. DeviceMemoryBarrier makes the group's UAV writes visible
    // before the readback. Every thread reaches the barrier (the early-out is
    // AFTER it), matching the reduction's barrier discipline above.
    DeviceMemoryBarrierWithGroupSync();
    if (row != 0u)
    {
        return;
    }
    const uint blocks = count / 32u;
    if (tid >= blocks)
    {
        return;
    }
    const uint b = tid;
    const uint bbase = b * 32u;

    float v[32];
    float am = 0.0f;
    for (uint k = 0u; k < 32u; ++k)
    {
        v[k] = outv[base + bbase + k];
        am = max(am, abs(v[k]));
    }
    const float d = (am > 0.0f) ? (am / 127.0f) : 0.0f;
    const float inv = (d > 0.0f) ? (1.0f / d) : 0.0f;
    qscale[b] = d;

    for (uint p = 0u; p < 8u; ++p)
    {
        int packed = 0;
        packed |= (uint)(QuantizeScalar(v[p * 4u + 0u], inv) & 0xff) << 0;
        packed |= (uint)(QuantizeScalar(v[p * 4u + 1u], inv) & 0xff) << 8;
        packed |= (uint)(QuantizeScalar(v[p * 4u + 2u], inv) & 0xff) << 16;
        packed |= (uint)(QuantizeScalar(v[p * 4u + 3u], inv) & 0xff) << 24;
        qactv.Store<uint>(bbase + p * 4u, packed);
    }
}
