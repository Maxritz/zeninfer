// ZenInfer D3D12 backend: batched rotary position embedding for PREFILL (T > 1).
//
// Decode uses shaders/rope.hlsl (single token, one (cos,sin) table). Prefill needs
// the same rotation applied to [tokens, heads, head_dim] with ONE table PER TOKEN:
// token t's table is the (pos = start_pos + t) table, [tokens * pairs] floats.
// This kernel is a strict generalisation of rope.hlsl:
//   threads   = tokens * heads * head_dim (one per output element)
//   token     = idx / (heads * head_dim)
//   cos/sin   = cosv[token * pairs + i] (same host-evaluated table values)
//
// The angle table is still built on the HOST (std::pow/cos/sin in double-promoted
// arithmetic, via rope_table) and uploaded; the kernel only rotates (two multiplies
// and two adds per pair), so results match CPU within the same FMA-contraction
// bound as the decode kernel (max_abs_err ~1e-7, see zen_d3d12_rope_parity).
// `pad0` selects the pairing exactly as rope.hlsl:
//   0 = NeoX half-split (qwen family), element i pairs with i + pairs
//   1 = llama.cpp NORM rope (llama family), consecutive elements (2i, 2i+1).
// Dimensions at or past rotary_dim (2 * pairs) are copied through unchanged.
// Decode is untouched: this file does not replace rope.hlsl.

cbuffer RopePrefillParams : register(b0)
{
    uint heads;       // attention heads in this tensor (H for q, KH for k)
    uint head_dim;    // stride between heads, in floats
    uint pairs;       // rotary_dim / 2, pairs per head (not `half`: reserved HLSL type)
    uint pad0;        // pairing layout: 0 = NeoX half-split, 1 = llama NORM rope
    uint tokens;      // number of tokens in this prefill batch (T)
    uint pad1;
    uint pad2;
    uint pad3;
};

ByteAddressBuffer         xraw : register(t0);   // source [tokens, heads, head_dim]
StructuredBuffer<float>   cosv : register(t1);   // [tokens * pairs], token-major
StructuredBuffer<float>   sinv : register(t2);   // [tokens * pairs], token-major
RWStructuredBuffer<float> outv : register(u0);   // destination, same shape as source

[numthreads(256, 1, 1)]
void RopePrefill(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID)
{
    const uint idx = gid.x * 256u + tid;
    const uint per_token = heads * head_dim;
    const uint total = tokens * per_token;
    if (idx >= total)
    {
        return;
    }

    const uint token = idx / per_token;
    const uint inner = idx - token * per_token;
    const uint h = inner / head_dim;
    const uint j = inner - h * head_dim;
    const uint base = idx - j; // token*per_token + h*head_dim: head's first element

    if (j >= pairs * 2u)
    {
        outv[idx] = asfloat(xraw.Load(idx * 4u));
        return;
    }

    uint i;       // angle index into this token's cos/sin slice
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
    const uint tab = token * pairs + i;
    const float c = cosv[tab];
    const float s = sinv[tab];

    outv[idx] = (first != 0u) ? (a * c - b * s) : (a * s + b * c);
}
