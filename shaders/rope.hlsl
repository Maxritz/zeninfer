// ZenInfer D3D12 backend: rotary position embedding over q or k.
//
// The cos/sin table is built on the HOST and uploaded, deliberately. The angle for
// dimension i is pos * base^(-2i/rotary_dim) with four scaling variants (linear,
// yarn, llama3, dynamic), and the CPU path evaluates that with std::pow/cos/sin in
// double-promoted arithmetic. Recomputing it on the device with powf/cosf/sinf would
// produce last-bit differences against the reference and could flip sampled ids on a
// knife-edge logit, so the kernel only applies a table it was handed. It therefore
// performs the rotation and nothing else: two multiplies and two adds per pair.
//
// Dimension layout is selected by `pad0`: 0 = NeoX half-split (qwen family),
// element i pairs with i + pairs; 1 = llama.cpp NORM rope (llama family),
// consecutive elements pair, (2i, 2i+1). Both rotate as (a*cos - b*sin,
// a*sin + b*cos) with the same angle table, so the layout only changes which
// elements form a pair.

cbuffer RopeParams : register(b0)
{
    uint heads;       // attention heads in this tensor (H for q, KH for k)
    uint head_dim;    // stride between heads, in floats
    uint pairs;       // rotary_dim / 2, i.e. pairs per head. NOT named `half`: that
                      // is a reserved HLSL type and shadowing it breaks the file.
    uint pad0;        // pairing layout: 0 = NeoX half-split (i with i+pairs),
                      // 1 = llama.cpp NORM rope (2i with 2i+1). See below.
};

ByteAddressBuffer         xraw : register(t0);   // source tensor
StructuredBuffer<float>   cosv : register(t1);   // [half]
StructuredBuffer<float>   sinv : register(t2);   // [half]
RWStructuredBuffer<float> outv : register(u0);   // destination tensor

// One thread per output element, covering the whole head rather than only the rotated
// pairs: dimensions at or past rotary_dim (2 * pairs) are copied through unchanged.
// The previous version launched heads*pairs threads and left those dimensions unwritten,
// so a partial-rotary model read back whatever the destination buffer already held.
//
// `pad0` picks the pairing, mirroring the CPU `rope`:
//   0 (NeoX half-split): j < pairs is the first half of pair i = j, whose second
//     half sits at i + pairs; pairs <= j < 2*pairs is that second half.
//   1 (llama NORM): consecutive elements pair; even j is the first element of
//     pair i = j/2, odd j the second.
[numthreads(256, 1, 1)]
void Rope(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID)
{
    const uint idx = gid.x * 256u + tid;
    const uint total = heads * head_dim;
    if (idx >= total)
    {
        return;
    }

    const uint h = idx / head_dim;
    const uint j = idx - h * head_dim;
    const uint base = h * head_dim;

    if (j >= pairs * 2u)
    {
        outv[idx] = asfloat(xraw.Load(idx * 4u));
        return;
    }

    uint i;       // angle index into cos/sin
    uint first;   // 1 when this element is the first of its pair
    float a, b;
    if (pad0 == 0u)
    {
        // NeoX half-split (qwen family): element i pairs with i + pairs.
        i = (j < pairs) ? j : (j - pairs);
        first = (j < pairs) ? 1u : 0u;
        a = asfloat(xraw.Load((base + i) * 4u));
        b = asfloat(xraw.Load((base + i + pairs) * 4u));
    }
    else
    {
        // llama.cpp NORM (llama family): consecutive elements pair, (2i, 2i+1).
        i = j >> 1;
        first = 1u - (j & 1u);
        a = asfloat(xraw.Load((base + (j & ~1u)) * 4u));
        b = asfloat(xraw.Load((base + (j | 1u)) * 4u));
    }
    const float c = cosv[i];
    const float s = sinv[i];

    outv[idx] = (first != 0u) ? (a * c - b * s) : (a * s + b * c);
}
