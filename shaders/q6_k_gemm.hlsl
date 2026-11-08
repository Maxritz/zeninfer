// Q6_K matrix product with FP32 activations. A 32-lane logical warp owns
// one output row; each lane decodes 8 of the 256 values in a GGML Q6_K
// block (210 bytes: 128B ql + 64B qh + 16B int8 group scales + fp16 d).
// Decode follows src/quant.cpp (CPU reference): per 128-half, lane l emits
// base+l+{0,32,64,96} with 6-bit indices from ql/qh and the half's superscale
// group (sc[is+{0,2,4,6}], is=l/16). Four tokens share every decoded weight.
// The host dispatches ceil(rows/8) x ceil(tokens/4) groups.
cbuffer Params : register(b0) {
    uint rows;
    uint blocks_per_row; // 256 values each
    uint tokens;
    uint reserved;
};
ByteAddressBuffer weights : register(t0);
ByteAddressBuffer input_data : register(t2);
RWStructuredBuffer<float> output_data : register(u0);

uint load_byte(uint offset) {
    return (weights.Load(offset & ~3u) >> ((offset & 3u) * 8u)) & 255u;
}
int load_i8(uint offset) {
    return (int)(load_byte(offset) & 255u) - ( (load_byte(offset) & 128u) ? 256 : 0 );
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
void GemmQ6_K(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID) {
    const uint lane = tid & 31u;
    const uint row = gid.x*8u + (tid >> 5u);
    const uint token_base = gid.y*4u;
    const uint cols = blocks_per_row*256u;
    float a0=0.0f, a1=0.0f, a2=0.0f, a3=0.0f;
    if (row < rows) {
        for (uint b=0u; b<blocks_per_row; ++b) {
            const uint base = (row*blocks_per_row+b)*216u; // upload pads 210B blocks to 216B for 4-byte load alignment
            const float d = f16tof32((weights.Load(base+208u) >> 0u) & 65535u);
            [unroll] for (uint h=0u; h<2u; ++h) {
                const uint ql = base + h*64u;
                const uint qh = base + 128u + h*32u;
                const uint sc = base + 192u + h*8u;
                const uint l = lane;
                const uint is = l >> 4u;
                const uint qlo = load_byte(ql + l);
                const uint qlo2 = load_byte(ql + 32u + l);
                const uint qhi = load_byte(qh + l);
                const int q1 = (int)((qlo & 15u) | ((qhi & 3u) << 4u)) - 32;
                const int q2 = (int)((qlo2 & 15u) | (((qhi >> 2u) & 3u) << 4u)) - 32;
                const int q3 = (int)((qlo >> 4u) | (((qhi >> 4u) & 3u) << 4u)) - 32;
                const int q4 = (int)((qlo2 >> 4u) | (((qhi >> 6u) & 3u) << 4u)) - 32;
                const int s0 = load_i8(sc + is + 0u);
                const int s1 = load_i8(sc + is + 2u);
                const int s2 = load_i8(sc + is + 4u);
                const int s3 = load_i8(sc + is + 6u);
                const float w[4] = { d*(float)s0*(float)q1, d*(float)s1*(float)q2,
                                     d*(float)s2*(float)q3, d*(float)s3*(float)q4 };
                const uint cbase = b*256u + h*128u + lane;
                [unroll] for (uint k=0u; k<4u; ++k) {
                    const uint c = cbase + k*32u;
                    const float wv = w[k];
                    if (token_base     < tokens) a0 += wv*asfloat(input_data.Load(((token_base    )*cols+c)*4u));
                    if (token_base+1u < tokens) a1 += wv*asfloat(input_data.Load(((token_base+1u)*cols+c)*4u));
                    if (token_base+2u < tokens) a2 += wv*asfloat(input_data.Load(((token_base+2u)*cols+c)*4u));
                    if (token_base+3u < tokens) a3 += wv*asfloat(input_data.Load(((token_base+3u)*cols+c)*4u));
                }
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
