// ZenInfer D3D12 backend: element-wise residual add, out[i] = a[i] + b[i].
//
// Kept on the device so a layer's activations can accumulate across ops without
// round-tripping through the host between them. This is the residual connection in
// both the attention half (`+ attn_out`) and the FFN half (`+ down`).
//
// Note: an earlier version of this file computed (a + b) - row_mean and assigned
// (rather than accumulated) into the reduction partial, which is not a residual add
// and could never match a reference. The elementwise form below has no reduction at
// all, so there is no shared memory and no barrier.

cbuffer AddnParams : register(b0)
{
    uint count;   // total element count
    uint pad0;
    uint pad1;
    uint pad2;
};

// Byte-address loads for the first operand, matching rmsnorm.hlsl: root SRV bindings
// carry no structure stride, so the leading operand is read by explicit byte offset.
ByteAddressBuffer         araw : register(t0);
StructuredBuffer<float>   bvec : register(t1);
RWStructuredBuffer<float> outv : register(u0);

[numthreads(256, 1, 1)]
void Addn(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID)
{
    const uint i = gid.x * 256u + tid;
    if (i >= count)
    {
        return;
    }
    outv[i] = asfloat(araw.Load(i * 4u)) + bvec[i];
}
