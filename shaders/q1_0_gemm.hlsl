// Q1_0 matrix product with FP32 activations. A 32-lane logical warp owns
// one output row; each lane decodes 4 of the 128 signs in a GGML Q1_0 block
// (18 bytes: fp16 scale + 128 packed 1-bit signs). Four tokens share every
// decoded weight. The host dispatches ceil(rows/8) x ceil(tokens/4) groups.
cbuffer Params : register(b0) {
    uint rows;
    uint blocks_per_row; // 128 values each
    uint tokens;
    uint reserved;
};
ByteAddressBuffer weights : register(t0);
ByteAddressBuffer input_data : register(t2);
RWStructuredBuffer<float> output_data : register(u0);

uint load_byte(uint offset) {
    return (weights.Load(offset & ~3u) >> ((offset & 3u) * 8u)) & 255u;
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
void GemmQ1_0(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID) {
    const uint lane = tid & 31u;
    const uint row = gid.x*8u + (tid >> 5u);
    const uint token_base = gid.y*4u;
    const uint cols = blocks_per_row*128u;
    float a0=0.0f, a1=0.0f, a2=0.0f, a3=0.0f;
    if (row < rows) {
        for (uint b=0u; b<blocks_per_row; ++b) {
            const uint base = (row*blocks_per_row+b)*24u; // upload pads 18B blocks to 24B for 4-byte load alignment
            const float d = f16tof32(weights.Load(base) & 65535u);
            // Elements lane, lane+32, lane+64, lane+96; bit j of byte j/8.
            [unroll] for (uint k=0u; k<4u; ++k) {
                const uint e = lane + k*32u;
                const uint bit = (load_byte(base + 2u + (e >> 3u)) >> (e & 7u)) & 1u;
                const float w = bit != 0u ? d : -d;
                const uint c = b*128u + e;
                if (token_base     < tokens) a0 += w*asfloat(input_data.Load(((token_base    )*cols+c)*4u));
                if (token_base+1u < tokens) a1 += w*asfloat(input_data.Load(((token_base+1u)*cols+c)*4u));
                if (token_base+2u < tokens) a2 += w*asfloat(input_data.Load(((token_base+2u)*cols+c)*4u));
                if (token_base+3u < tokens) a3 += w*asfloat(input_data.Load(((token_base+3u)*cols+c)*4u));
            }
        }
    }
    a0=sum32(a0,lane); a1=sum32(a1,lane); a2=sum32(a2,lane); a3=sum32(a3,lane);
    if (lane == 0u && row < rows) {
        if (token_base    < tokens) output_data[(token_base   )*rows+row]=a0;
        if (token_base+1u < tokens) output_data[(token_base+1u)*rows+row]=a1;
        if (token_base+2u < tokens) output_data[(token_base+2u)*rows+row]=a2;
        if (token_base+3u < tokens) output_data[(token_base+3u)*rows+row]=a3;
    }
}
