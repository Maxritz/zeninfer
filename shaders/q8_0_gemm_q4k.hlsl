// Q4_K / Q5_K matrix product with FP32 activations. A 32-lane logical
// warp owns one output row; each lane processes one element of each 32-value
// sub-block. Four tokens share every decoded weight. Raw GGML blocks are
// 144/176 bytes, both multiples of four, so ByteAddressBuffer loads are aligned.
// The host dispatches ceil(rows/8) x ceil(tokens/4) groups.
cbuffer Params : register(b0) {
    uint rows;
    uint blocks_per_row; // 256 values each
    uint tokens;
    uint is_q5;
};
ByteAddressBuffer weights : register(t0);
ByteAddressBuffer input_data : register(t2);
RWStructuredBuffer<float> output_data : register(u0);

uint load_byte(uint offset) {
    return (weights.Load(offset & ~3u) >> ((offset & 3u) * 8u)) & 255u;
}
uint2 scale_min(uint base, uint j) {
    if (j < 4u) return uint2(load_byte(base+j) & 63u, load_byte(base+j+4u) & 63u);
    const uint s = load_byte(base+j+4u);
    return uint2((s & 15u) | ((load_byte(base+j-4u) >> 6u) << 4u),
                 (s >> 4u) | ((load_byte(base+j) >> 6u) << 4u));
}
float sum32(float v, uint lane) {
    const uint base = WaveGetLaneIndex() - lane;
    [unroll] for (uint s=16u; s>0u; s>>=1u) {
        const float other = WaveReadLaneAt(v, base + ((lane+s) & 31u));
        if (lane+s < 32u) v += other;
    }
    return v;
}
[numthreads(256,1,1)]
void GemmQ4_K(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID) {
    const uint lane = tid & 31u;
    const uint row = gid.x*8u + (tid >> 5u);
    const uint token_base = gid.y*4u;
    const uint cols = blocks_per_row*256u;
    const uint block_bytes = is_q5 != 0u ? 176u : 144u;
    float a0=0.0f, a1=0.0f, a2=0.0f, a3=0.0f;
    if (row < rows) {
        for (uint b=0u; b<blocks_per_row; ++b) {
            const uint base = (row*blocks_per_row+b)*block_bytes;
            const uint halves = weights.Load(base);
            const float d = f16tof32(halves & 65535u);
            const float dmin = f16tof32(halves >> 16u);
            [unroll] for (uint j=0u; j<8u; ++j) {
                const uint2 sm = scale_min(base+4u,j);
                const uint qbase = base + (is_q5 != 0u ? 48u : 16u);
                const uint packed = load_byte(qbase + (j >> 1u)*32u + lane);
                uint q = (packed >> ((j & 1u)*4u)) & 15u;
                if (is_q5 != 0u) q |= ((load_byte(base+16u+lane) >> j) & 1u) << 4u;
                const float w = (d*(float)sm.x)*(float)q - dmin*(float)sm.y;
                const uint c = b*256u+j*32u+lane;
                if (token_base     < tokens) a0 += w*asfloat(input_data.Load(((token_base    )*cols+c)*4u));
                if (token_base+1u < tokens) a1 += w*asfloat(input_data.Load(((token_base+1u)*cols+c)*4u));
                if (token_base+2u < tokens) a2 += w*asfloat(input_data.Load(((token_base+2u)*cols+c)*4u));
                if (token_base+3u < tokens) a3 += w*asfloat(input_data.Load(((token_base+3u)*cols+c)*4u));
            }
        }
    }
    // All lanes participate even for out-of-range rows on wave32 or wave64.
    a0=sum32(a0,lane); a1=sum32(a1,lane); a2=sum32(a2,lane); a3=sum32(a3,lane);
    if (lane == 0u && row < rows) {
        if (token_base    < tokens) output_data[(token_base   )*rows+row]=a0;
        if (token_base+1u < tokens) output_data[(token_base+1u)*rows+row]=a1;
        if (token_base+2u < tokens) output_data[(token_base+2u)*rows+row]=a2;
        if (token_base+3u < tokens) output_data[(token_base+3u)*rows+row]=a3;
    }
}
