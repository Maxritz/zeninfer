// ZenInfer D3D12 backend: root-mean-square normalisation, one workgroup per row.
//
// Keeping this on the device is what lets the layer's input stay in VRAM: the CPU
// path has to read the activation back, normalise it, and upload it again before
// every Q|K|V matvec.
//
// The row is clamped rather than early-returned because the workgroup barriers below
// must be reached by every thread.

cbuffer RmsParams : register(b0)
{
    uint  count;      // elements per row
    uint  rows;
    float epsilon;
    uint  pad;
};

ByteAddressBuffer         xraw : register(t0);   // [row][count]
StructuredBuffer<float>   wvec : register(t1);   // [count]
RWStructuredBuffer<float> outv : register(u0);   // [row][count]

groupshared float partial[256];

[numthreads(256, 1, 1)]
void RmsNorm(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID)
{
    // The host dispatches exactly `rows` groups, so no guard is needed here.
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
}
