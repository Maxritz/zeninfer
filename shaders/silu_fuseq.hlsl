// ZenInfer D3D12 backend: SiLU(gate)*up + int8 activation quantize, fused
// (decode DP4A).
//
// Fused producer variant of silu_mul.hlsl for the DP4A decode path: the SiLU
// dispatch also quantises its output (ffn_rows elements) into the DP4A
// staging streams (qact int8 acts at u3 + per-32-block scales at u4), so the
// separate quantize_acts dispatch for the down matvec input can be skipped
// when ZEN_D3D12_DP4A_FUSEQ=1. The fp32 output is still written (u0)
// alongside the int8 form.
//
// Restructured to one thread per 32-element block (the unfused kernel is one
// thread per element): each thread computes the 32 SiLU values, writes them
// as fp32, tracks the block amax over those same register values, and stores
// the packed int8 block + scale. The SiLU expression is character-identical
// to silu_mul.hlsl, and the quantize math mirrors quantize_acts.hlsl
// (d = amax/127, clamp, half-away lround), so both streams are bit-identical
// to the unfused two-kernel path.

cbuffer SiluParams : register(b0)
{
    uint count;
    uint pad0;
    uint pad1;
    uint pad2;
};

StructuredBuffer<float>   gate_vec : register(t0);
StructuredBuffer<float>   up_vec   : register(t1);
RWStructuredBuffer<float> out_vec  : register(u0);
RWByteAddressBuffer       qactv : register(u3);    // int8 acts (fused quantize)
RWStructuredBuffer<float> qscale : register(u4);   // one fp32 scale per block

int QuantizeScalar(float x, float inv)
{
    const float v = clamp(x * inv, -127.0f, 127.0f);
    return (int)(v >= 0.0f ? floor(v + 0.5f) : ceil(v - 0.5f));
}

[numthreads(256, 1, 1)]
void SiluMulFused(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID)
{
    const uint b = gid.x * 256u + tid;
    const uint blocks = count / 32u;
    if (b >= blocks)
    {
        return;
    }
    const uint base = b * 32u;

    float v[32];
    float am = 0.0f;
    for (uint k = 0u; k < 32u; ++k)
    {
        const float g = gate_vec[base + k];
        const float u = up_vec[base + k];
        const float val = (g / (1.0f + exp(-g))) * u;
        v[k] = val;
        out_vec[base + k] = val;
        am = max(am, abs(val));
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
        qactv.Store<uint>(base + p * 4u, packed);
    }
}
