// ZenInfer D3D12 backend: per-head RMS normalisation of Q or K (Qwen3 family).
//
// Qwen3 divides every head of q and k by its own RMS before RoPE. The CPU path does
// this with qk_rms_norm(), which is rms_norm() restarted at each head boundary; this
// kernel is the same arithmetic, one workgroup per head, so a head's sum of squares
// never leaves the group.
//
// Everything is a UAV (u0 output, u1 input) rather than an SRV read plus a UAV write,
// for the same reason the DP4A staging buffer is: the caller applies this in place -
// u0 and u1 point at the SAME buffer - and a resource cannot be bound as an SRV and a
// UAV at once. The kernel is written so that aliasing is benign: element i is read and
// then written only by the thread that owns i, so no thread ever reads a value another
// thread has rewritten.
//
// The weight row is `count` floats and comes from an upload heap, which is permanently
// GENERIC_READ and can never be transitioned to UNORDERED_ACCESS. It is therefore read
// through a root SRV (t0): root descriptors are raw GPU addresses, so no state is
// implied for the buffer they point into.

cbuffer QkNormParams : register(b0)
{
    uint  count;      // elements per head (= head_dim)
    uint  rows;       // heads in this tensor
    float epsilon;
    uint  pad;
};

StructuredBuffer<float>   wvec : register(t0);   // [count], the q_norm or k_norm row
RWStructuredBuffer<float> xio  : register(u1);   // [rows][count], in place source
RWStructuredBuffer<float> outv : register(u0);   // [rows][count]

// head_dim is at most 128 in this pipeline (attention3 enforces the same bound), so a
// plain 128-wide group covers a head in one pass and the two strided loops below are
// single-iteration for every supported shape.
groupshared float partial[128];

[numthreads(128, 1, 1)]
void QkNorm(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID)
{
    // The host dispatches exactly `rows` groups, so no guard is needed here.
    const uint row = gid.x;
    const uint base = row * count;

    float ss = 0.0f;
    for (uint i = tid; i < count; i += 128u)
    {
        const float v = xio[base + i];
        ss += v * v;
    }

    partial[tid] = ss;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint s = 64u; s > 0u; s >>= 1u)
    {
        if (tid < s)
        {
            partial[tid] += partial[tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // rms_norm on the CPU accumulates in double; the device path accumulates in float
    // exactly like the layer-level rmsnorm kernel this file mirrors, and does the final
    // division and rsqrt in float, matching rms_norm()'s float `inv`.
    const float scale = rsqrt(partial[0] / (float)count + epsilon);
    for (uint i = tid; i < count; i += 128u)
    {
        outv[base + i] = xio[base + i] * scale * wvec[i];
    }
}
