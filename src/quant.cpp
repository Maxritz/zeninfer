#include "zen/quant.h"
#include "zen/compute_pool.h"
#include "zen/diagnostics.h"
#include "zen/kernels.h"
#include "zen/resource_governor.h"
#include "iq_tables.h"
#include <thread>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace zen::quant {
namespace {
std::uint16_t read16(const std::uint8_t* p) noexcept {
    return std::uint16_t(p[0]) | (std::uint16_t(p[1]) << 8);
}
float half_to_float(std::uint16_t h) noexcept {
    const std::uint32_t sign=std::uint32_t(h&0x8000)<<16;
    std::uint32_t exp=(h>>10)&31, mant=h&1023, out{};
    if(exp==0){
        if(mant==0) out=sign;
        else { int e=-14; while((mant&1024)==0){mant<<=1;--e;} mant&=1023; out=sign|(std::uint32_t(e+127)<<23)|(mant<<13); }
    } else if(exp==31) out=sign|0x7f800000|(mant<<13);
    else out=sign|((exp+112)<<23)|(mant<<13);
    float f; std::memcpy(&f,&out,sizeof(f)); return f;
}
std::uint16_t float_to_half(float value) noexcept {
    std::uint32_t bits{}; std::memcpy(&bits,&value,sizeof(bits));
    const std::uint16_t sign=static_cast<std::uint16_t>((bits>>16)&0x8000u);
    const std::uint32_t exp=(bits>>23)&0xffu, mant=bits&0x7fffffu;
    if(exp==0xffu) {
        if(mant==0) return static_cast<std::uint16_t>(sign|0x7c00u);
        std::uint16_t payload=static_cast<std::uint16_t>(mant>>13);
        if(payload==0) payload=1;
        return static_cast<std::uint16_t>(sign|0x7c00u|payload);
    }
    if(exp==0) return sign; // Every binary32 subnormal is below the binary16 rounding range.
    const int e=static_cast<int>(exp)-127;
    if(e>15) return static_cast<std::uint16_t>(sign|0x7c00u);
    const std::uint32_t significand=0x800000u|mant;
    auto rounded_shift=[](std::uint32_t x,unsigned shift) noexcept -> std::uint32_t {
        if(shift==0) return x;
        if(shift>=32) return 0;
        const std::uint32_t q=x>>shift;
        const std::uint32_t mask=(std::uint32_t(1)<<shift)-1u;
        const std::uint32_t rem=x&mask, halfway=std::uint32_t(1)<<(shift-1);
        return q + (rem>halfway || (rem==halfway && (q&1u)) ? 1u : 0u);
    };
    if(e>=-14) {
        std::uint32_t hm=rounded_shift(mant,13);
        int he=e+15;
        if(hm==0x400u) { hm=0; ++he; }
        if(he>=31) return static_cast<std::uint16_t>(sign|0x7c00u);
        return static_cast<std::uint16_t>(sign|(std::uint16_t(he)<<10)|std::uint16_t(hm));
    }
    const unsigned shift=static_cast<unsigned>(-e-1); // 24 bits to 10-bit subnormal fraction.
    const std::uint32_t sub=rounded_shift(significand,shift);
    return static_cast<std::uint16_t>(sign|static_cast<std::uint16_t>(sub));
}
}

/**
 * Helper used by the K‑quant dequantizers.
 *
 * The 12‑byte packed scale array stores two 6‑bit values per scale index:
 *   - for j < 4: low nibble of q[j] (bits 0‑5) is the scale, high nibble of q[j+4] is the min.
 *   - for j >= 4: low nibble of q[j+4] (bits 0‑3) + high two bits of q[j-4] (bits 6‑7) form the scale;
 *     high nibble of q[j+4] (bits 4‑7) + low two bits of q[j] (bits 6‑7) form the min.
 */
void k_scale_min(unsigned j,const std::uint8_t* q,std::uint8_t& scale,std::uint8_t& min) noexcept {
    if(j<4){
        scale=std::uint8_t(q[j]&63u);
        min=std::uint8_t(q[j+4]&63u);
    }else{
        scale=std::uint8_t((q[j+4]&0x0fu)|((q[j-4]>>6)<<4));
        min=std::uint8_t((q[j+4]>>4)|((q[j]>>6)<<4));
    }
}

// Helper: exact E8M0 to float semantics (same as GGML_E8M0_TO_FP32_HALF)
static inline float ggml_e8m0_to_fp32_half(uint8_t x) noexcept {
    uint32_t bits;
    if (x < 2) {
        bits = 0x00200000 << x; // 2^(-128) * x
    } else {
        bits = (uint32_t)(x - 1) << 23; // 0.5 * 2^(x-127)
    }
    float result;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
}

// Doubled E2M1 values for MXFP4, verbatim from ggml-common.h (kvalues_fp4).
// Code 0-7 = +0..+6 (doubled), code 8-15 = -0..-6 (doubled); -0 maps to 0.
static constexpr std::int8_t kvalues_mxfp4[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};

// ---------------------------------------------------------------------------
// ROCmFP4 (AMD Strix Halo / RDNA4 community format, ciru-ai/ROCmFPX fork).
//
// Two GGUF types share the Codebook10 value set:
//   type 100 q4_0_rocmfp4      { uint8_t qs[16]; uint8_t e[2]; } = 18 B
//   type 101 q4_0_rocmfp4_fast { uint8_t qs[16]; uint8_t e;   } = 17 B
// Codes map through kvalues_rocmfp4 (largest level retuned 12 -> 10 versus
// MXFP4). Scales are FINITE UNSIGNED E4M3 (UE4M3, bias 7) bytes: values 1..126
// decode to (8+M)*2^(E-10), byte 0 is subnormal M*2^-10, byte 0x7F is 0. The
// decoder consumes the "half" table because the codebook levels are the
// doubled E2M1 magnitudes.
// ---------------------------------------------------------------------------

// Codebook10, verbatim from rocmfp4.c (ciru-ai/ROCmFPX).
static constexpr std::int8_t kvalues_rocmfp4[16] = {
    0, 1, 2, 3, 4, 6, 8, 10, 0, -1, -2, -3, -4, -6, -8, -10
};

// UE4M3 (unsigned E4M3, bias 7) scale byte -> half-scale float.
// Bytes 0x00..0x7E are finite; 0x7F decodes to 0. Semantics mirror
// rocmfp4_scale_ue4m3_half from the fork: subnormal M*2^-10 for E==0,
// then (8+M)*2^(E-10) per exponent (i.e. (1+M/8) * 2^(E-7) in fp32 terms).
static inline float rocmfp4_ue4m3_to_fp32_half(std::uint8_t e) noexcept {
    if (e >= 0x7f) return 0.0f;
    const unsigned M = e & 7u;
    const unsigned E = e >> 3;
    if (E == 0) {
        // Subnormal in E4M3 terms: M * 2^-10. In fp32 this is a normal float:
        // 0 -> 0, M>0 -> exp field 127-10+floor(log2 M), mantissa fraction.
        // Cheapest exact route: ldexp on the integer.
        return std::ldexp(static_cast<float>(M), -10);
    }
    // (8 + M) * 2^(E-10) = (1 + M/8) * 2^(E-7).
    // fp32: exponent field 127+(E-7), mantissa M<<20.
    const std::uint32_t b = static_cast<std::uint32_t>(127 + E - 7) << 23
                          | static_cast<std::uint32_t>(M) << 20;
    float out;
    std::memcpy(&out, &b, sizeof(out));
    return out;
}

const std::vector<Format>& registry() {
    static const std::vector<Format> f=[] {
        std::vector<Format> v{
            {"F32",Encoding::fp32,32,1,4,true,"IEEE-754 binary32"},
            {"F16",Encoding::fp16,16,1,2,true,"IEEE-754 binary16"},
            {"BF16",Encoding::bf16,16,1,2,true,"IEEE-754 bfloat16"},
            {"Q4_0",Encoding::q4_0,4,32,18,true,"GGML Q4_0 block: fp16 scale plus 16 packed bytes"},
            {"Q4_1",Encoding::q4_1,4,32,20,true,"GGML Q4_1 block: fp16 scale, fp16 minimum and 16 packed bytes"},
            {"Q5_0",Encoding::q5_0,5,32,22,true,"GGML Q5_0 block"},
            {"Q5_1",Encoding::q5_1,5,32,24,true,"GGML Q5_1 block: fp16 scale, fp16 minimum, 32 high bits and 16 packed bytes"},
            {"Q8_0",Encoding::q8_0,8,32,34,true,"GGML Q8_0 block: fp16 scale plus 32 signed bytes"},
            {"Q3_K",Encoding::q3_k,3,256,110,true,"GGML Q3_K super-block"},
            {"Q6_K",Encoding::q6_k,6,256,210,true,"GGML Q6_K super-block"},
            {"IQ4_NL",Encoding::iq4_nl,4,32,18,true,"GGML non-linear 4-bit lookup quantisation"},
            {"TERNARY2",Encoding::ternary,2,4,1,true,"Generic 2-bit codes: 00=-1, 01=0, 10=+1, 11=reserved"},
            {"TQ1_0",Encoding::tq1_0,2,256,54,true,"GGML ternary base-3 packing: 256 values per block"},
            {"TQ2_0",Encoding::tq2_0,2,256,66,true,"GGML ternary 2-bit packing: 256 values per block"}
        };
        for(unsigned b=1;b<=18;++b) {
            const unsigned bytes=(32*b+7)/8;
            v.push_back({"Q"+std::to_string(b)+"_SIGNED",Encoding::signed_uniform,
                         static_cast<std::uint8_t>(b),32,static_cast<std::uint16_t>(bytes),true,
                         "Generic signed bit-packed block; one caller-supplied scale"});
        }
        v.push_back({"Q2_K",Encoding::q2_k,2,256,84,true,"GGML Q2_K super-block: 16 packed scales/mins, 2-bit quants"});
        v.push_back({"Q4_K",Encoding::q4_k,4,256,144,true,"GGML Q4_K super-block: fp16 d/dmin, 12-byte 6-bit scales/mins, 4-bit quants"});
        v.push_back({"Q5_K",Encoding::q5_k,5,256,176,true,"GGML Q5_K super-block: Q4_K plus a 32-byte high-bit plane"});
        v.push_back({"Q8_K",Encoding::q8_k,8,256,292,true,"GGML Q8_K super-block: f32 delta plus 256 signed quants"});
        v.push_back({"IQ1_S",Encoding::iq1_s,1,256,50,true,"GGML IQ1_S: 1.5625 bpw grid super-block"});
        v.push_back({"IQ1_M",Encoding::iq1_m,1,256,56,true,"GGML IQ1_M: 1.75 bpw grid super-block"});
        v.push_back({"IQ2_XXS",Encoding::iq2_xxs,2,256,66,true,"GGML IQ2_XXS: 2.0625 bpw grid super-block"});
        v.push_back({"IQ2_XS",Encoding::iq2_xs,2,256,74,true,"GGML IQ2_XS: 2.3125 bpw grid super-block"});
        v.push_back({"IQ2_S",Encoding::iq2_s,2,256,82,true,"GGML IQ2_S: 2.5625 bpw grid super-block"});
        v.push_back({"IQ3_XXS",Encoding::iq3_xxs,3,256,98,true,"GGML IQ3_XXS: 3.0625 bpw grid super-block"});
        v.push_back({"IQ3_S",Encoding::iq3_s,3,256,110,true,"GGML IQ3_S: 3.3125 bpw grid super-block"});
        v.push_back({"IQ4_XS",Encoding::iq4_xs,4,256,136,true,"GGML IQ4_XS: non-linear 4-bit with per-32 scales"});
        v.push_back({"MXFP4",Encoding::mxfp4,4,32,17,true,"GGML MXFP4: E8M0 scale plus 16 packed 4-bit e2m1-doubled nibbles"});
        v.push_back({"Q4_0_ROCMFP4",Encoding::rocmfp4,4,32,18,true,"ROCmFP4 dual-scale: two UE4M3 half-block scales plus 16 packed Codebook10 nibbles (ciru-ai/ROCmFPX)"});
        v.push_back({"Q4_0_ROCMFP4_FAST",Encoding::rocmfp4_fast,4,32,17,true,"ROCmFP4 fast: one UE4M3 scale plus 16 packed Codebook10 nibbles (ciru-ai/ROCmFPX)"});
        v.push_back({"PTQ1_0",Encoding::ptq1_0,2,128,28,true,"Prism PTQ1_0: TQ1_0 base-3 trit packing at group size 128, one f16 scale per 128 weights (PrismML fork)"});
        return v;
    }();
    return f;
}
const Format* find(std::string_view n) noexcept {
    for(const auto& f:registry()) if(f.name==n) return &f;
    return nullptr;
}
bool decode_fp16(const std::uint16_t* src,float* dst,std::size_t n) noexcept {
    if(!src||!dst)return false;
    for(std::size_t i=0;i<n;++i)dst[i]=half_to_float(src[i]);
    return true;
}
bool encode_fp16(const float* src,std::uint16_t* dst,std::size_t n) noexcept {
    if(!src||!dst)return false;
    for(std::size_t i=0;i<n;++i)dst[i]=float_to_half(src[i]);
    return true;
}
bool dequantize(const Format& f,const std::uint8_t* src,std::size_t n,float* dst,float scale) noexcept {
    if(!src||!dst||!std::isfinite(scale))return false;
    switch(f.encoding) {
    case Encoding::fp32:
        if(f.bits!=32)return false;
        for(std::size_t i=0;i<n;++i)std::memcpy(dst+i,src+i*4,4);
        return true;
    case Encoding::fp16:
        for(std::size_t i=0;i<n;++i)dst[i]=half_to_float(read16(src+i*2));
        return true;
    case Encoding::bf16:
        // bfloat16 is the high half of a binary32: widen the 16 bits and
        // reinterpret, so no rounding step is needed.
        for(std::size_t i=0;i<n;++i){
            const std::uint32_t bits=std::uint32_t(read16(src+i*2))<<16;
            float v;std::memcpy(&v,&bits,sizeof(v));dst[i]=v;
        }
        return true;
    case Encoding::q4_1:
        if(n%32)return false;
        for(std::size_t b=0;b<n/32;++b){
            const auto* p=src+b*20; const float d=half_to_float(read16(p)),m=half_to_float(read16(p+2));
            for(unsigned j=0;j<16;++j){const auto q=p[4+j];
                dst[b*32+j]=(int(q&15u))*d+m;dst[b*32+j+16]=(int(q>>4))*d+m;}
        }
        return true;
    case Encoding::q5_1:
        if(n%32)return false;
        for(std::size_t b=0;b<n/32;++b){
            const auto* p=src+b*24; const float d=half_to_float(read16(p)),m=half_to_float(read16(p+2));
            const std::uint32_t qh=std::uint32_t(p[4])|(std::uint32_t(p[5])<<8)|(std::uint32_t(p[6])<<16)|(std::uint32_t(p[7])<<24);
            for(unsigned j=0;j<16;++j){const auto lo=p[8+j];
                const int hi0=int((qh>>j)&1u)<<4, hi1=int((qh>>(j+16))&1u)<<4;
                dst[b*32+j]=(int(lo&15u)+hi0)*d+m;
                dst[b*32+j+16]=(int(lo>>4)+hi1)*d+m;
            }
        }
        return true;
    case Encoding::q4_0:
        if(n%32)return false;
        for(std::size_t b=0;b<n/32;++b){
            const auto* p=src+b*18; const float d=half_to_float(read16(p));
            for(unsigned j=0;j<16;++j){const auto q=p[2+j];dst[b*32+j]=(int(q&15)-8)*d;dst[b*32+j+16]=(int(q>>4)-8)*d;}
        }
        return true;
    case Encoding::q5_0:
        if(n%32)return false;
        for(std::size_t b=0;b<n/32;++b){
            const auto* p=src+b*22; const float d=half_to_float(read16(p));
            const std::uint32_t qh=std::uint32_t(p[2])|(std::uint32_t(p[3])<<8)|(std::uint32_t(p[4])<<16)|(std::uint32_t(p[5])<<24);
            for(unsigned j=0;j<16;++j){const auto lo=p[6+j];
                const int hi0=int((qh>>j)&1u)<<4, hi1=int((qh>>(j+16))&1u)<<4;
                dst[b*32+j]=(int(lo&15u)+hi0-16)*d;
                dst[b*32+j+16]=(int(lo>>4)+hi1-16)*d;
            }
        }
        return true;
    case Encoding::q8_0:
        if(n%32)return false;
        for(std::size_t b=0;b<n/32;++b){
            const auto*p=src+b*34;const float d=half_to_float(read16(p));
            for(unsigned j=0;j<32;++j)dst[b*32+j]=static_cast<std::int8_t>(p[2+j])*d;
        }
        return true;
    case Encoding::q2_k:
        if(n%256)return false;
        for(std::size_t b=0;b<n/256;++b){
            const auto* p=src+b*84;
            const float d=half_to_float(read16(p+80)), dmin=half_to_float(read16(p+82));
            const std::uint8_t* scales=p; const std::uint8_t* q=p+16;
            unsigned is=0, out=b*256;
            for(unsigned j=0;j<256;j+=128){
                unsigned shift=0;
                for(std::size_t l=0;l<4;++l){
                    std::uint8_t sc=scales[is++];
                    const float dl=d*float(sc&15u), ml=dmin*float(sc>>4);
                    for(unsigned k=0;k<16;++k)dst[out++]=dl*float((q[k]>>shift)&3u)-ml;
                    sc=scales[is++];
                    const float dh=d*float(sc&15u), mh=dmin*float(sc>>4);
                    for(unsigned k=0;k<16;++k)dst[out++]=dh*float((q[16+k]>>shift)&3u)-mh;
                    shift+=2;
                }
                q+=32;
            }
        }
        return true;
    case Encoding::q4_k:
        if(n%256)return false;
        for(std::size_t b=0;b<n/256;++b){
            const auto* p=src+b*144;
            const float d=half_to_float(read16(p)), dmin=half_to_float(read16(p+2));
            const std::uint8_t* scales=p+4; const std::uint8_t* q=p+16;
            unsigned is=0, out=b*256;
            for(unsigned j=0;j<256;j+=64){
                std::uint8_t sc{}, m{};
                k_scale_min(is+0,scales,sc,m);
                const float d1=d*float(sc), m1=dmin*float(m);
                k_scale_min(is+1,scales,sc,m);
                const float d2=d*float(sc), m2=dmin*float(m);
                for(unsigned l=0;l<32;++l)dst[out++]=d1*float(q[l]&15u)-m1;
                for(unsigned l=0;l<32;++l)dst[out++]=d2*float(q[l]>>4)-m2;
                q+=32; is+=2;
            }
        }
        return true;
    case Encoding::q5_k:
        if(n%256)return false;
        for(std::size_t b=0;b<n/256;++b){
            const auto* p=src+b*176;
            const float d=half_to_float(read16(p)), dmin=half_to_float(read16(p+2));
            const std::uint8_t* scales=p+4; const std::uint8_t* qh=p+16; const std::uint8_t* ql=p+48;
            unsigned is=0, out=b*256; std::uint8_t u1=1, u2=2;
            for(unsigned j=0;j<256;j+=64){
                std::uint8_t sc{}, m{};
                k_scale_min(is+0,scales,sc,m);
                const float d1=d*float(sc), m1=dmin*float(m);
                k_scale_min(is+1,scales,sc,m);
                const float d2=d*float(sc), m2=dmin*float(m);
                for(unsigned l=0;l<32;++l)dst[out++]=d1*float((ql[l]&15u)+((qh[l]&u1)?16u:0u))-m1;
                for(unsigned l=0;l<32;++l)dst[out++]=d2*float((ql[l]>>4)+((qh[l]&u2)?16u:0u))-m2;
                ql+=32; is+=2; u1=static_cast<std::uint8_t>(u1<<2); u2=static_cast<std::uint8_t>(u2<<2);
            }
        }
        return true;
    case Encoding::q3_k:
        if(n%256)return false;
        for(std::size_t b=0;b<n/256;++b){
            const auto* p=src+b*110; const auto* hm=p; const auto* q=p+32;
            const float dall=half_to_float(read16(p+108));
            std::uint32_t aux[4]{}; std::memcpy(aux,p+96,12); const std::uint32_t tmp=aux[2];
            aux[2]=((aux[0]>>4)&0x0f0f0f0fu)|(((tmp>>4)&0x03030303u)<<4);
            aux[3]=((aux[1]>>4)&0x0f0f0f0fu)|(((tmp>>6)&0x03030303u)<<4);
            aux[0]=(aux[0]&0x0f0f0f0fu)|(((tmp>>0)&0x03030303u)<<4);
            aux[1]=(aux[1]&0x0f0f0f0fu)|(((tmp>>2)&0x03030303u)<<4);
            const auto* scales=reinterpret_cast<const std::int8_t*>(aux);
            unsigned is=0, mask=1, out=0;
            for(unsigned n128=0;n128<256;n128+=128){ unsigned shift=0;
                for(unsigned j=0;j<4;++j){
                    float dl=dall*float(scales[is++]-32);
                    for(unsigned l=0;l<16;++l)dst[b*256+out++]=dl*(int((q[l]>>shift)&3u)-((hm[l]&mask)?0:4));
                    dl=dall*float(scales[is++]-32);
                    for(unsigned l=0;l<16;++l)dst[b*256+out++]=dl*(int((q[l+16]>>shift)&3u)-((hm[l+16]&mask)?0:4));
                    shift+=2; mask<<=1;
                } q+=32;
            }
        }
        return true;
    case Encoding::q6_k:
        if(n%256)return false;
        for(std::size_t b=0;b<n/256;++b){
            const auto* p=src+b*210; const auto* ql=p; const auto* qh=p+128;
            const auto* sc=reinterpret_cast<const std::int8_t*>(p+192); const float d=half_to_float(read16(p+208));
            for(unsigned n128=0;n128<256;n128+=128){
                for(unsigned l=0;l<32;++l){ const unsigned is=l/16;
                    const int q1=int((ql[l]&15u)|(((qh[l]>>0)&3u)<<4))-32;
                    const int q2=int((ql[l+32]&15u)|(((qh[l]>>2)&3u)<<4))-32;
                    const int q3=int((ql[l]>>4)|(((qh[l]>>4)&3u)<<4))-32;
                    const int q4=int((ql[l+32]>>4)|(((qh[l]>>6)&3u)<<4))-32;
                    const std::size_t base=b*256+(n128)+l;
                    dst[base+0]=d*sc[is+0]*q1; dst[base+32]=d*sc[is+2]*q2;
                    dst[base+64]=d*sc[is+4]*q3; dst[base+96]=d*sc[is+6]*q4;
                } ql+=64; qh+=32; sc+=8;
            }
        }
        return true;
    case Encoding::iq4_nl: {
        static constexpr std::int8_t lut[16]={-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
        if(n%32)return false;
        for(std::size_t b=0;b<n/32;++b){ const auto* p=src+b*18; const float d=half_to_float(read16(p));
            for(unsigned j=0;j<16;++j){dst[b*32+j]=d*lut[p[2+j]&15u];dst[b*32+j+16]=d*lut[p[2+j]>>4];}
        } return true;
    }
    case Encoding::q8_k:
        // GGML Q8_K: f32 delta, 256 signed quants, 16 per-16 sums (the sums are
        // a dot-product acceleration hint and are not needed to decode). 292 B.
        if(n%256)return false;
        for(std::size_t b=0;b<n/256;++b){
            const auto* p=src+b*292; float d;std::memcpy(&d,p,4);
            for(unsigned j=0;j<256;++j)dst[b*256+j]=d*float(static_cast<std::int8_t>(p[4+j]));
        }
        return true;
    case Encoding::tq2_0:
        // GGML TQ2_0: 64 bytes of 2-bit ternary codes, code c maps to (c-1)*d
        // with c in {0,1,2,3} -> {-d,0,+d,+2d}; f16 delta at offset 64. 66 B.
        if(n%256)return false;
        for(std::size_t b=0;b<n/256;++b){
            const auto* p=src+b*66; const float d=half_to_float(read16(p+64)); const auto* qs=p;
            unsigned out=0;
            for(unsigned j=0;j<64;j+=32)for(std::size_t l=0;l<4;++l)for(unsigned m=0;m<32;++m)
                dst[b*256+out++]=float(int((qs[j+m]>>(2*l))&3u)-1)*d;
        }
        return true;
    case Encoding::tq1_0: {
        // GGML TQ1_0: 48 bytes of base-3 packed quants (5 digits per byte,
        // least-significant digit first) plus 4 bytes holding the last 16
        // values at 4 digits per byte; digit d maps to (d-1)*d_scale. 54 B.
        if(n%256)return false;
        static constexpr unsigned pow3[6]={1,3,9,27,81,243};
        for(std::size_t b=0;b<n/256;++b){
            const auto* p=src+b*54; const float d=half_to_float(read16(p+52));
            const auto* qs=p; const auto* qh=p+48;
            auto digit=[&](std::uint8_t byte,unsigned k){
                // The upstream digit extraction truncates byte*3^k to 8 bits
                // before scaling; the truncation is part of the format.
                const std::uint8_t q=static_cast<std::uint8_t>(byte*pow3[k]);
                return float(int(static_cast<unsigned>(q)*3u>>8)-1)*d;
            };
            unsigned out=0;
            for(unsigned nn=0;nn<5;++nn)for(unsigned m=0;m<32;++m)dst[b*256+out++]=digit(qs[m],nn);
            for(unsigned nn=0;nn<5;++nn)for(unsigned m=32;m<48;++m)dst[b*256+out++]=digit(qs[m],nn);
            for(unsigned nn=0;nn<4;++nn)for(unsigned j=0;j<4;++j)dst[b*256+out++]=digit(qh[j],nn);
        }
        return true;
    }
    case Encoding::iq4_xs: {
        // GGML IQ4_XS: 128 bytes of non-linear 4-bit codes (same 16-entry LUT
        // as IQ4_NL) with an 8-level scale per 32 values. Scale packing: low
        // 4 bits from scales_l (two per byte), high 2 bits from scales_h,
        // stored biased by 32. 136 B.
        if(n%256)return false;
        static constexpr std::int8_t lut[16]={-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
        for(std::size_t b=0;b<n/256;++b){
            const auto* p=src+b*136; const float d=half_to_float(read16(p));
            const std::uint16_t scales_h=read16(p+2); const auto* scales_l=p+4; const auto* qs=p+8;
            unsigned out=0;
            for(std::size_t ib=0;ib<8;++ib){
                const unsigned ls=((scales_l[ib/2]>>(4*(ib%2)))&0xfu)|(((scales_h>>(2*ib))&3u)<<4);
                const float dl=d*(float(ls)-32.0f);
                for(unsigned j=0;j<16;++j){dst[b*256+out+j]=dl*float(lut[qs[j]&15u]);dst[b*256+out+16+j]=dl*float(lut[qs[j]>>4]);}
                out+=32; qs+=16;
            }
        }
        return true;
    }
    case Encoding::iq2_xxs: {
        // GGML IQ2_XXS: 32 u16 words per super-block. Each 8-byte group holds
        // four grid indices (bytes 0-3) and a 28-bit sign field plus a 4-bit
        // scale (bytes 4-7); grid values are 8 bytes of a u64 codebook entry.
        // 66 B.
        if(n%256)return false;
        for(std::size_t b=0;b<n/256;++b){
            const auto* p=src+b*66; const float d=half_to_float(read16(p)); const auto* qs=p+2;
            unsigned out=0;
            for(std::size_t ib=0;ib<8;++ib){
                std::uint32_t aux[2];std::memcpy(aux,qs+4*ib,8);
                const auto* aux8=reinterpret_cast<const std::uint8_t*>(aux);
                const float db=d*(0.5f+float(aux[1]>>28))*0.25f;
                for(std::size_t l=0;l<4;++l){
                    const auto* grid=reinterpret_cast<const std::uint8_t*>(&detail::iq2xxs_grid[aux8[l]]);
                    const std::uint8_t signs=detail::ksigns_iq2xs[(aux[1]>>(7*l))&127u];
                    for(unsigned j=0;j<8;++j)dst[b*256+out+j]=db*float(grid[j])*((signs&detail::kmask_iq2xs[j])?-1.0f:1.0f);
                    out+=8;
                }
            }
        }
        return true;
    }
    case Encoding::iq2_xs: {
        // GGML IQ2_XS: 32 u16 words, each split into a 9-bit grid index and a
        // 7-bit sign field; one 8-bit scale per 32 values (two 4-bit halves).
        // 74 B.
        if(n%256)return false;
        for(std::size_t b=0;b<n/256;++b){
            const auto* p=src+b*74; const float d=half_to_float(read16(p));
            const auto* qs=p+2; const auto* scales=p+66;
            unsigned out=0;
            for(std::size_t ib=0;ib<8;++ib){
                const float db[2]={d*(0.5f+float(scales[ib]&0xf))*0.25f,d*(0.5f+float(scales[ib]>>4))*0.25f};
                for(std::size_t l=0;l<4;++l){
                    const std::uint16_t idx=read16(qs+2*(4*ib+l));
                    const auto* grid=reinterpret_cast<const std::uint8_t*>(&detail::iq2xs_grid[idx&511u]);
                    const std::uint8_t signs=detail::ksigns_iq2xs[idx>>9];
                    for(unsigned j=0;j<8;++j)dst[b*256+out+j]=db[l/2]*float(grid[j])*((signs&detail::kmask_iq2xs[j])?-1.0f:1.0f);
                    out+=8;
                }
            }
        }
        return true;
    }
    case Encoding::iq2_s: {
        // GGML IQ2_S: 64 bytes of low grid bits, 8 bytes of high bits (two per
        // 32 values, combined with (8-2l) shifts into a 10-bit index), 32 sign
        // bytes and 8 scale bytes. 82 B.
        if(n%256)return false;
        for(std::size_t b=0;b<n/256;++b){
            const auto* p=src+b*82; const float d=half_to_float(read16(p));
            const auto* qs=p+2; const auto* qh=qs+64; const auto* scales=qh+8; const auto* signs=qs+32;
            unsigned out=0;
            for(std::size_t ib=0;ib<8;++ib){
                const float db[2]={d*(0.5f+float(scales[ib]&0xf))*0.25f,d*(0.5f+float(scales[ib]>>4))*0.25f};
                for(std::size_t l=0;l<4;++l){
                    const auto* grid=reinterpret_cast<const std::uint8_t*>(&detail::iq2s_grid[qs[l]|((qh[ib]<<(8-2*l))&0x300u)]);
                    const std::uint8_t sg=signs[4*ib+l];
                    for(unsigned j=0;j<8;++j)dst[b*256+out+j]=db[l/2]*float(grid[j])*((sg&detail::kmask_iq2xs[j])?-1.0f:1.0f);
                    out+=8;
                }
                qs+=4;
            }
        }
        return true;
    }
    case Encoding::iq3_xxs: {
        // GGML IQ3_XXS: 64 bytes of 8-bit grid indices (two 4-value u32 grid
        // entries per 8 outputs) followed by 32 bytes holding a 4-bit scale
        // and four 7-bit sign fields per 32 values. 98 B.
        if(n%256)return false;
        for(std::size_t b=0;b<n/256;++b){
            const auto* p=src+b*98; const float d=half_to_float(read16(p));
            const auto* qs=p+2; const auto* ssh=qs+64;
            unsigned out=0;
            for(std::size_t ib=0;ib<8;++ib){
                std::uint32_t aux;std::memcpy(&aux,ssh+4*ib,4);
                const float db=d*(0.5f+float(aux>>28))*0.5f;
                for(std::size_t l=0;l<4;++l){
                    const std::uint8_t signs=detail::ksigns_iq2xs[(aux>>(7*l))&127u];
                    const auto* g1=reinterpret_cast<const std::uint8_t*>(&detail::iq3xxs_grid[qs[2*l]]);
                    const auto* g2=reinterpret_cast<const std::uint8_t*>(&detail::iq3xxs_grid[qs[2*l+1]]);
                    for(unsigned j=0;j<4;++j){
                        dst[b*256+out+j]=db*float(g1[j])*((signs&detail::kmask_iq2xs[j])?-1.0f:1.0f);
                        dst[b*256+out+4+j]=db*float(g2[j])*((signs&detail::kmask_iq2xs[j+4])?-1.0f:1.0f);
                    }
                    out+=8;
                }
                qs+=8;
            }
        }
        return true;
    }
    case Encoding::iq3_s: {
        // GGML IQ3_S: 64 bytes of 8-bit grid indices with a 9th bit interleaved
        // from qh, 32 sign bytes and 4 scale bytes (one 4-bit scale per 64
        // values, scaled by (1+2*s)). 110 B.
        if(n%256)return false;
        for(std::size_t b=0;b<n/256;++b){
            const auto* p=src+b*110; const float d=half_to_float(read16(p));
            const auto* qs=p+2; const auto* qh=qs+64; const auto* signs=qh+8; const auto* scales=signs+32;
            unsigned out=0;
            for(std::size_t ib=0;ib<8;ib+=2){
                const float db1=d*(1.0f+2.0f*float(scales[ib/2]&0xf));
                const float db2=d*(1.0f+2.0f*float(scales[ib/2]>>4));
                for(std::size_t l=0;l<4;++l){
                    const auto* g1=reinterpret_cast<const std::uint8_t*>(&detail::iq3s_grid[qs[2*l]|((qh[0]<<(8-2*l))&0x100u)]);
                    const auto* g2=reinterpret_cast<const std::uint8_t*>(&detail::iq3s_grid[qs[2*l+1]|((qh[0]<<(7-2*l))&0x100u)]);
                    const std::uint8_t sg=signs[l];
                    for(unsigned j=0;j<4;++j){
                        dst[b*256+out+j]=db1*float(g1[j])*((sg&detail::kmask_iq2xs[j])?-1.0f:1.0f);
                        dst[b*256+out+4+j]=db1*float(g2[j])*((sg&detail::kmask_iq2xs[j+4])?-1.0f:1.0f);
                    }
                    out+=8;
                }
                qs+=8;signs+=4;
                for(std::size_t l=0;l<4;++l){
                    const auto* g1=reinterpret_cast<const std::uint8_t*>(&detail::iq3s_grid[qs[2*l]|((qh[1]<<(8-2*l))&0x100u)]);
                    const auto* g2=reinterpret_cast<const std::uint8_t*>(&detail::iq3s_grid[qs[2*l+1]|((qh[1]<<(7-2*l))&0x100u)]);
                    const std::uint8_t sg=signs[l];
                    for(unsigned j=0;j<4;++j){
                        dst[b*256+out+j]=db2*float(g1[j])*((sg&detail::kmask_iq2xs[j])?-1.0f:1.0f);
                        dst[b*256+out+4+j]=db2*float(g2[j])*((sg&detail::kmask_iq2xs[j+4])?-1.0f:1.0f);
                    }
                    out+=8;
                }
                qh+=2;qs+=8;signs+=4;
            }
        }
        return true;
    }
    case Encoding::iq1_s: {
        // GGML IQ1_S: 32 bytes of low grid bits plus 16 bytes of u16 words
        // (3 high index bits, 3 scale bits, 1 sign bit per 32 values). Grid
        // values are signed bytes offset by +-1/8. 50 B.
        if(n%256)return false;
        constexpr float delta=0.125f;
        for(std::size_t b=0;b<n/256;++b){
            const auto* p=src+b*50; const float d=half_to_float(read16(p)); const auto* qs=p+2;
            unsigned out=0;
            for(std::size_t ib=0;ib<8;++ib){
                const std::uint16_t h=read16(p+34+2*ib);
                const float dl=d*(2.0f*float((h>>12)&7u)+1.0f);
                const float dd=(h&0x8000u)?-delta:delta;
                for(std::size_t l=0;l<4;++l){
                    const auto* grid=reinterpret_cast<const std::int8_t*>(&detail::iq1s_grid[qs[l]|(((h>>(3*l))&7u)<<8)]);
                    for(unsigned j=0;j<8;++j)dst[b*256+out+j]=dl*(float(grid[j])+dd);
                    out+=8;
                }
                qs+=4;
            }
        }
        return true;
    }
    case Encoding::iq1_m: {
        // GGML IQ1_M: 32 bytes of low grid bits, 16 bytes of high bits and 8
        // bytes of packed scales: a f16 super-scale assembled from four nibbles
        // plus two 3-bit sub-scales and two sign bits per 32 values. 56 B.
        if(n%256)return false;
        constexpr float delta=0.125f;
        for(std::size_t b=0;b<n/256;++b){
            const auto* p=src+b*56; const auto* qs=p; const auto* qh=p+32; const auto* sc=p+48;
            const std::uint16_t scale=static_cast<std::uint16_t>((read16(sc)>>12)|((read16(sc+2)>>8)&0xf0u)
                                            |((read16(sc+4)>>4)&0xf00u)|(read16(sc+6)&0xf000u));
            const float d=half_to_float(scale);
            unsigned out=0;
            for(std::size_t ib=0;ib<8;++ib){
                const std::uint16_t s16=read16(sc+2*(ib/2));
                const float dl1=d*(2.0f*float((s16>>(6*(ib%2)))&7u)+1.0f);
                const float dl2=d*(2.0f*float((s16>>(6*(ib%2)+3))&7u)+1.0f);
                const std::uint16_t h0=read16(qh+2*ib);
                const std::uint16_t h1=static_cast<std::uint16_t>(h0>>8); // second byte of the same pair
                const unsigned idx[4]={qs[0]|((h0<<8)&0x700u),qs[1]|((h0<<4)&0x700u),qs[2]|((h1<<8)&0x700u),qs[3]|((h1<<4)&0x700u)};
                const float dd[4]={(h0&0x08u)?-delta:delta,(h0&0x80u)?-delta:delta,(h1&0x08u)?-delta:delta,(h1&0x80u)?-delta:delta};
                for(unsigned l=0;l<2;++l){
                    const auto* grid=reinterpret_cast<const std::int8_t*>(&detail::iq1s_grid[idx[l]]);
                    for(unsigned j=0;j<8;++j)dst[b*256+out+j]=dl1*(float(grid[j])+dd[l]);
                    out+=8;
                }
                for(unsigned l=2;l<4;++l){
                    const auto* grid=reinterpret_cast<const std::int8_t*>(&detail::iq1s_grid[idx[l]]);
                    for(unsigned j=0;j<8;++j)dst[b*256+out+j]=dl2*(float(grid[j])+dd[l]);
                    out+=8;
                }
                qs+=4;
            }
        }
        return true;
    }
    case Encoding::mxfp4: {
        // GGML MXFP4: E8M0 scale plus 16 packed 4-bit e2m1-doubled nibbles
        // (block_mxfp4 = { uint8_t e; uint8_t qs[16]; }, 17 B). The scale uses
        // exact GGML_E8M0_TO_FP32_HALF semantics; kvalues_mxfp4 is the shared
        // doubled-e2m1 table.
        if(n%32)return false;
        for(std::size_t b=0;b<n/32;++b){
            const auto* p=src+b*17;
            const float d=ggml_e8m0_to_fp32_half(p[0]);
            for(unsigned j=0;j<16;++j){
                dst[b*32+j]=d*float(kvalues_mxfp4[p[1+j]&15u]);
                dst[b*32+16+j]=d*float(kvalues_mxfp4[p[1+j]>>4]);
            }
        }
        return true;
    }
    case Encoding::rocmfp4: {
        // ROCmFP4 dual-scale (GGUF type 100, block_rocmfp4 = { qs[16]; e[2]; },
        // 18 B). Codebook10 levels; UE4M3 scale per 16-weight half.
        if(n%32)return false;
        for(std::size_t b=0;b<n/32;++b){
            const auto* p=src+b*18;
            const float d0=rocmfp4_ue4m3_to_fp32_half(p[16]);
            const float d1=rocmfp4_ue4m3_to_fp32_half(p[17]);
            for(unsigned j=0;j<16;++j){
                dst[b*32+j]=d0*float(kvalues_rocmfp4[p[j]&15u]);
                dst[b*32+16+j]=d1*float(kvalues_rocmfp4[p[j]>>4]);
            }
        }
        return true;
    }
    case Encoding::rocmfp4_fast: {
        // ROCmFP4 fast (GGUF type 101, block_rocmfp4_fast = { qs[16]; e; },
        // 17 B). One UE4M3 scale for the whole 32-weight block.
        if(n%32)return false;
        for(std::size_t b=0;b<n/32;++b){
            const auto* p=src+b*17;
            const float d=rocmfp4_ue4m3_to_fp32_half(p[16]);
            for(unsigned j=0;j<16;++j){
                dst[b*32+j]=d*float(kvalues_rocmfp4[p[j]&15u]);
                dst[b*32+16+j]=d*float(kvalues_rocmfp4[p[j]>>4]);
            }
        }
        return true;
    }
    case Encoding::ptq1_0: {
        // Prism PTQ1_0 (GGUF type 143, block_ptq1_0 = { qs[24]; qh[2]; f16 d; },
        // 28 B): the upstream TQ1_0 base-3 trit packing applied at group size
        // 128. 24 qs bytes carry 5 trits each (120 values), 2 qh bytes carry 4
        // trits each (8 values). Digit decode identical to TQ1_0: q = byte*3^n
        // truncated to 8 bits, xi = (q*3)>>8, value = (xi-1)*d.
        if(n%128)return false;
        static constexpr unsigned pow3[6]={1,3,9,27,81,243};
        for(std::size_t b=0;b<n/128;++b){
            const auto* p=src+b*28;
            const float d=half_to_float(read16(p+26));
            const auto* qs=p; const auto* qh=p+24;
            auto digit=[&](std::uint8_t byte,unsigned k){
                const std::uint8_t q=static_cast<std::uint8_t>(byte*pow3[k]);
                return float(int(static_cast<unsigned>(q)*3u>>8)-1)*d;
            };
            unsigned out=0;
            for(unsigned nn=0;nn<5;++nn)for(unsigned m=0;m<24;++m)dst[b*128+out++]=digit(qs[m],nn);
            for(unsigned nn=0;nn<4;++nn)for(unsigned j=0;j<2;++j)dst[b*128+out++]=digit(qh[j],nn);
        }
        return true;
    }
    case Encoding::ternary:
        if(n%4)return false;
        for(std::size_t i=0;i<n;++i){
            const unsigned code=(src[i/4]>>((i%4)*2))&3u;
            if(code==3)return false;
            dst[i]=(code==0?-1.0f:code==2?1.0f:0.0f)*scale;
        }
        return true;
    case Encoding::signed_uniform: {
        const unsigned bits=f.bits;
        if(bits<1||bits>18||f.block_elements==0)return false;
        std::size_t bit=0;
        for(std::size_t i=0;i<n;++i){
            std::uint32_t raw=0;
            for(unsigned k=0;k<bits;++k,++bit)
                if(src[bit/8]&(1u<<(bit%8))) raw|=1u<<k;
            const std::int32_t signed_value=(raw&(1u<<(bits-1)))
                ?static_cast<std::int32_t>(raw|(~0u<<bits)):static_cast<std::int32_t>(raw);
            dst[i]=static_cast<float>(signed_value)*scale;
        }
        return true;
    }
    }
    return false;
}
std::size_t row_bytes(const Format& f,std::size_t columns) noexcept {
    auto blocks=[&](std::size_t block,std::size_t bytes)->std::size_t{
        if(columns%block)return 0;
        const auto n=columns/block;
        return n>std::numeric_limits<std::size_t>::max()/bytes?0:n*bytes;
    };
    switch(f.encoding){
    case Encoding::fp32:return columns>std::numeric_limits<std::size_t>::max()/4?0:columns*4;
    case Encoding::fp16:case Encoding::bf16:return columns>std::numeric_limits<std::size_t>::max()/2?0:columns*2;
    case Encoding::q4_0:return blocks(32,18);
    case Encoding::q4_1:return blocks(32,20);
    case Encoding::q5_0:return blocks(32,22);
    case Encoding::q5_1:return blocks(32,24);
    case Encoding::q8_0:return blocks(32,34);
    case Encoding::q3_k:return blocks(256,110);
    case Encoding::q2_k:return blocks(256,84);
    case Encoding::q4_k:return blocks(256,144);
    case Encoding::q5_k:return blocks(256,176);
    case Encoding::q8_k:return blocks(256,292);
    case Encoding::iq1_s:return blocks(256,50);
    case Encoding::iq1_m:return blocks(256,56);
    case Encoding::iq2_xxs:return blocks(256,66);
    case Encoding::iq2_xs:return blocks(256,74);
    case Encoding::iq2_s:return blocks(256,82);
    case Encoding::iq3_xxs:return blocks(256,98);
    case Encoding::iq3_s:return blocks(256,110);
    case Encoding::iq4_xs:return blocks(256,136);
    case Encoding::mxfp4:return blocks(32,17);
    case Encoding::rocmfp4:return blocks(32,18);
    case Encoding::rocmfp4_fast:return blocks(32,17);
    case Encoding::ptq1_0:return blocks(128,28);
    case Encoding::q6_k:return blocks(256,210);
    case Encoding::iq4_nl:return blocks(32,18);
    case Encoding::ternary:return columns>(std::numeric_limits<std::size_t>::max()-7)/2?0:(columns*2+7)/8;
    case Encoding::tq1_0:return blocks(256,54);
    case Encoding::tq2_0:return blocks(256,66);
    case Encoding::signed_uniform:return f.bits&&f.bits<=18&&columns<=(std::numeric_limits<std::size_t>::max()-7)/f.bits?(columns*f.bits+7)/8:0;
    }
    return 0;
}
bool gemv(const Format& f,const std::uint8_t*w,const float*x,float*y,
          std::size_t rows,std::size_t cols,float scale,unsigned requested_workers) {
    zen::diag::Scope trace("quant.gemv",zen::diag::Phase::compute);
    zen::diag::breadcrumb(300,static_cast<std::int64_t>(rows),static_cast<std::int64_t>(cols));
    if(!w||!x||!y||!rows||!cols||!std::isfinite(scale))return false;
    const auto rb=row_bytes(f,cols);if(!rb)return false;
#ifdef ZEN_HAS_AVX2_KERNELS
    if(f.encoding==Encoding::q8_0 && scale==1.0f && zen::cpu_has_avx2_fma()) {
        zen::ResourceGovernor governor;const unsigned workers=requested_workers?requested_workers:governor.current_policy().worker_threads;
        if(rows>=128&&workers>1){
            const unsigned count=std::min<unsigned>(workers,static_cast<unsigned>(rows));
            std::atomic<bool> success{true};
            zen::ComputePool::instance().parallel_for(rows,count,[&](std::size_t first,std::size_t last){
                if(!zen::gemv_q8_0_avx2(w+first*rb,x,y+first,last-first,cols))success.store(false,std::memory_order_relaxed);
            });
            return success.load();
        }
        return zen::gemv_q8_0_avx2(w,x,y,rows,cols);
    }
#endif
    zen::ResourceGovernor governor;const unsigned workers=requested_workers?requested_workers:governor.current_policy().worker_threads;
    const unsigned count=(rows>=64&&workers>1)?std::min<unsigned>(workers,static_cast<unsigned>(rows)):1;
    std::atomic<bool> ok{true};
    auto work=[&](std::size_t first,std::size_t last){
        const bool block_quant=f.encoding!=Encoding::fp32&&f.encoding!=Encoding::fp16&&f.encoding!=Encoding::bf16;
        const std::size_t block=block_quant?f.block_elements:cols;
        const std::size_t block_bytes=block_quant?f.block_bytes:rb;
        if(!block||!block_bytes){ok=false;return;}
        std::vector<float> scratch(block);[[maybe_unused]] const bool simd=
#ifdef ZEN_HAS_AVX2_KERNELS
            zen::cpu_has_avx2_fma();
#else
            false;
#endif
        for(std::size_t r=first;r<last&&ok.load(std::memory_order_relaxed);++r){double total=0.0;
            for(std::size_t off=0;off<cols;off+=block){const auto count=std::min(block,cols-off);if(!dequantize(f,w+r*rb+(off/block)*block_bytes,count,scratch.data(),scale)){ok=false;break;}
#ifdef ZEN_HAS_AVX2_KERNELS
                if(simd){float part{};zen::gemv_avx2(scratch.data(),x+off,&part,1,count);total+=part;continue;}
#endif
                for(std::size_t c=0;c<count;++c)total+=double(scratch[c])*x[off+c];
            }
            y[r]=static_cast<float>(total);
        }
    };
    if(count==1){work(0,rows);return ok.load();}
    zen::ComputePool::instance().parallel_for(rows,count,work);
    return ok.load();
}
bool gemm(const Format& f,const std::uint8_t*w,const float*input,float*output,
          std::size_t tokens,std::size_t rows,std::size_t cols,float scale,unsigned requested_workers){
    zen::diag::Scope trace("quant.gemm",zen::diag::Phase::compute);
    if(!w||!input||!output||!tokens||!rows||!cols||!std::isfinite(scale))return false;
    const auto rb=row_bytes(f,cols);if(!rb)return false;
    zen::ResourceGovernor governor;const unsigned workers=requested_workers?requested_workers:governor.current_policy().worker_threads;
    const unsigned count=(rows>=32&&workers>1)?std::min<unsigned>(workers,static_cast<unsigned>(rows)):1;
    std::atomic<bool> ok{true};
    auto work=[&](std::size_t first,std::size_t last){
        const bool block_quant=f.encoding!=Encoding::fp32&&f.encoding!=Encoding::fp16&&f.encoding!=Encoding::bf16;
        const std::size_t block=block_quant?f.block_elements:cols;
        const std::size_t block_bytes=block_quant?f.block_bytes:rb;
        if(!block||!block_bytes){ok=false;return;}
        std::vector<float> scratch(block);[[maybe_unused]] const bool simd=
#ifdef ZEN_HAS_AVX2_KERNELS
            zen::cpu_has_avx2_fma();
#else
            false;
#endif
        for(std::size_t r=first;r<last&&ok.load(std::memory_order_relaxed);++r){for(std::size_t t=0;t<tokens;++t)output[t*rows+r]=0.0f;
            for(std::size_t off=0;off<cols;off+=block){const auto count=std::min(block,cols-off);if(!dequantize(f,w+r*rb+(off/block)*block_bytes,count,scratch.data(),scale)){ok=false;break;}
                for(std::size_t t=0;t<tokens;++t){const float*x=input+t*cols+off;float*y=output+t*rows+r;
#ifdef ZEN_HAS_AVX2_KERNELS
                    if(simd){float part{};zen::gemv_avx2(scratch.data(),x,&part,1,count);*y+=part;continue;}
#endif
                    double sum=0;for(std::size_t c=0;c<count;++c)sum+=double(scratch[c])*x[c];*y+=static_cast<float>(sum);
                }
            }
        }
    };
    if(count==1){work(0,rows);return ok.load();}
    zen::ComputePool::instance().parallel_for(rows,count,work);
    return ok.load();
}

}
