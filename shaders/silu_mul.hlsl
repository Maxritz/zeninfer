// ZenInfer D3D12 backend: SiLU(gate) * up, evaluated on the device.
//
// The FFN activation normally round-trips to the CPU between gate|up and ffn_down,
// which costs a full submission (~70us of fixed latency on this stack). Keeping it
// in VRAM lets gate, up, this kernel and ffn_down share one command list.

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

[numthreads(256, 1, 1)]
void SiluMul(uint3 id : SV_DispatchThreadID)
{
    const uint i = id.x;
    if (i >= count) {
        return;
    }
    const float g = gate_vec[i];
    const float u = up_vec[i];
    out_vec[i] = (g / (1.0f + exp(-g))) * u;
}
