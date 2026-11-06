#include "zen/kernels.h"
#include <immintrin.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <limits>
#include <array>
#include <vector>

namespace zen {
// A named, explicitly aligned wrapper avoids putting an attributed compiler
// vector type directly in a standard-library template.
struct alignas(32) VectorAccumulator { __m256 value; };
void gemv_avx2(const float* a, const float* x, float* y,
               std::size_t m, std::size_t k) {
    for (std::size_t i=0; i<m; ++i) {
        const float* row = a + i*k;
        std::size_t p = 0;
        __m256 s0 = _mm256_setzero_ps();
        __m256 s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps();
        __m256 s3 = _mm256_setzero_ps();
        for (; p + 31 < k; p += 32) {
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(row+p),      _mm256_loadu_ps(x+p),      s0);
            s1 = _mm256_fmadd_ps(_mm256_loadu_ps(row+p+8),    _mm256_loadu_ps(x+p+8),    s1);
            s2 = _mm256_fmadd_ps(_mm256_loadu_ps(row+p+16),   _mm256_loadu_ps(x+p+16),   s2);
            s3 = _mm256_fmadd_ps(_mm256_loadu_ps(row+p+24),   _mm256_loadu_ps(x+p+24),   s3);
        }
        __m256 sum = _mm256_add_ps(_mm256_add_ps(s0,s1), _mm256_add_ps(s2,s3));
        for (; p + 7 < k; p += 8)
            sum = _mm256_fmadd_ps(_mm256_loadu_ps(row+p), _mm256_loadu_ps(x+p), sum);
        alignas(32) float lanes[8];
        _mm256_store_ps(lanes, sum);
        float total = lanes[0]+lanes[1]+lanes[2]+lanes[3]+lanes[4]+lanes[5]+lanes[6]+lanes[7];
        for (; p<k; ++p) total += row[p]*x[p];
        y[i] = total;
    }
}

// 4x8 register micro-kernel. B is traversed contiguously; each loaded B vector
// is reused across four rows of A. Edge tiles fall back to scalar code.
void gemm_avx2(const float* a, const float* b, float* c,
               std::size_t m, std::size_t n, std::size_t k) {
    constexpr std::size_t MR=4, NR=8;
    for (std::size_t i0=0; i0<m; i0+=MR) {
        for (std::size_t j0=0; j0<n; j0+=NR) {
            const std::size_t rows = (m-i0 < MR) ? (m-i0) : MR;
            const std::size_t cols = (n-j0 < NR) ? (n-j0) : NR;
            if (rows != MR || cols != NR) {
                for (std::size_t ii=0; ii<rows; ++ii)
                    for (std::size_t jj=0; jj<cols; ++jj) {
                        float sum=0.0f;
                        for (std::size_t p=0; p<k; ++p)
                            sum += a[(i0+ii)*k+p]*b[p*n+(j0+jj)];
                        c[(i0+ii)*n+(j0+jj)] = sum;
                    }
                continue;
            }
            __m256 c0=_mm256_setzero_ps(), c1=_mm256_setzero_ps();
            __m256 c2=_mm256_setzero_ps(), c3=_mm256_setzero_ps();
            for (std::size_t p=0; p<k; ++p) {
                const __m256 bv = _mm256_loadu_ps(b+p*n+j0);
                c0 = _mm256_fmadd_ps(_mm256_set1_ps(a[(i0+0)*k+p]), bv, c0);
                c1 = _mm256_fmadd_ps(_mm256_set1_ps(a[(i0+1)*k+p]), bv, c1);
                c2 = _mm256_fmadd_ps(_mm256_set1_ps(a[(i0+2)*k+p]), bv, c2);
                c3 = _mm256_fmadd_ps(_mm256_set1_ps(a[(i0+3)*k+p]), bv, c3);
            }
            _mm256_storeu_ps(c+(i0+0)*n+j0, c0);
            _mm256_storeu_ps(c+(i0+1)*n+j0, c1);
            _mm256_storeu_ps(c+(i0+2)*n+j0, c2);
            _mm256_storeu_ps(c+(i0+3)*n+j0, c3);
        }
    }
}
}


namespace zen {
namespace {
float q8_half_to_float(std::uint16_t h) noexcept {
    const std::uint32_t sign=std::uint32_t(h&0x8000u)<<16;
    std::uint32_t exp=(h>>10)&31u, mant=h&1023u, out{};
    if(exp==0){
        if(mant==0) out=sign;
        else { int e=-14; while((mant&1024u)==0){mant<<=1;--e;} mant&=1023u; out=sign|(std::uint32_t(e+127)<<23)|(mant<<13); }
    } else if(exp==31) out=sign|0x7f800000u|(mant<<13);
    else out=sign|((exp+112u)<<23)|(mant<<13);
    float f; std::memcpy(&f,&out,sizeof(f)); return f;
}
}
// Sum the eight lanes. The previous version stored to a stack array and summed
// scalars once per 32-element block, which cost more than the dot product itself.
static inline float hsum_ps256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

// NVFP4 uses UE4M3 scales, interpreted with the upstream half-scale
// convention (including the high-bit cases). Build the 256-value table once.
static const std::array<float,256>& nvfp4_scale_table(){
    static const auto values=[](){std::array<float,256> out{};for(unsigned x=0;x<256;++x){
        if(x==0||x==0x7f){out[x]=0.f;continue;}
        const unsigned exp=(x>>3)&15u,mant=x&7u;
        out[x]=(exp==0?std::ldexp(float(mant),-9):std::ldexp(1.f+float(mant)/8.f,int(exp)-7))*0.5f;
    }return out;}();return values;
}
static const std::array<float,256>& f8_e4m3_table(){
    static const auto values=[](){std::array<float,256> out{};for(unsigned x=0;x<256;++x){
        const unsigned a=x&127u,exp=(a>>3)&15u,mant=a&7u;
        if(a==127){out[x]=std::numeric_limits<float>::quiet_NaN();continue;}
        const float v=exp==0?std::ldexp(float(mant),-9):std::ldexp(1.f+float(mant)/8.f,int(exp)-7);
        out[x]=(x&128u)?-v:v;
    }return out;}();return values;
}
// Eight decoded elements from compact Q1_0/Q2_0, NVFP4, FP8 and Q8_1 layouts.
// The callers only pass whole, block-aligned panels (Kc=256).
static inline __m256 new_quant_decode8(const std::uint8_t* p, std::size_t offset, NewQuantKind kind) {
    if(kind==NewQuantKind::f8_e4m3){
        std::uint64_t bytes;std::memcpy(&bytes,p+offset,8);
        const auto indices=_mm256_cvtepu8_epi32(_mm_cvtsi64_si128(static_cast<long long>(bytes)));
        return _mm256_i32gather_ps(f8_e4m3_table().data(),indices,4);
    }
    if(kind==NewQuantKind::nvfp4){
        const unsigned group=offset/16, lane=offset%16;
        std::uint64_t bytes;std::memcpy(&bytes,p+4+group*8+(lane%8),8);
        __m128i indices=_mm_cvtsi64_si128(static_cast<long long>(bytes));
        if(lane>=8)indices=_mm_srli_epi16(indices,4);
        indices=_mm_and_si128(indices,_mm_set1_epi8(15));
        const __m128i table=_mm_setr_epi8(0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12);
        const auto q=_mm256_cvtepi8_epi32(_mm_shuffle_epi8(table,indices));
        return _mm256_mul_ps(_mm256_cvtepi32_ps(q),_mm256_set1_ps(nvfp4_scale_table()[p[group]]));
    }
    float d;
    if(kind==NewQuantKind::q8_1_f32){std::uint32_t bits=std::uint32_t(p[0])|(std::uint32_t(p[1])<<8)|(std::uint32_t(p[2])<<16)|(std::uint32_t(p[3])<<24);std::memcpy(&d,&bits,4);}
    else d=q8_half_to_float(std::uint16_t(p[0])|(std::uint16_t(p[1])<<8));
    __m256i q;
    if(kind==NewQuantKind::q1_0){
        const int byte=p[2+offset/8];
        const __m256i v=_mm256_set1_epi32(byte);
        const __m256i shifts=_mm256_setr_epi32(0,1,2,3,4,5,6,7);
        q=_mm256_sub_epi32(_mm256_slli_epi32(_mm256_and_si256(_mm256_srlv_epi32(v,shifts),_mm256_set1_epi32(1)),1),_mm256_set1_epi32(1));
    }else if(kind==NewQuantKind::q2_0){
        const int a=p[2+offset/4], b=p[3+offset/4];
        const __m256i v=_mm256_setr_epi32(a,a,a,a,b,b,b,b);
        const __m256i shifts=_mm256_setr_epi32(0,2,4,6,0,2,4,6);
        q=_mm256_sub_epi32(_mm256_and_si256(_mm256_srlv_epi32(v,shifts),_mm256_set1_epi32(3)),_mm256_set1_epi32(1));
    }else{
        std::uint64_t bytes;std::memcpy(&bytes,p+(kind==NewQuantKind::q8_1_f32?8:4)+offset,8);
        q=_mm256_cvtepi8_epi32(_mm_cvtsi64_si128(static_cast<long long>(bytes)));
    }
    return _mm256_mul_ps(_mm256_cvtepi32_ps(q),_mm256_set1_ps(d));
}
static inline std::size_t new_quant_elements(NewQuantKind kind){return kind==NewQuantKind::f8_e4m3?1:kind==NewQuantKind::q1_0?128:kind==NewQuantKind::q2_0||kind==NewQuantKind::nvfp4?64:32;}
static inline std::size_t new_quant_bytes(NewQuantKind kind){return kind==NewQuantKind::f8_e4m3?1:kind==NewQuantKind::q1_0||kind==NewQuantKind::q2_0?18:kind==NewQuantKind::q8_1||kind==NewQuantKind::nvfp4?36:40;}

bool gemv_new_quant_avx2(const std::uint8_t* weights,const float* x,float* y,
                          std::size_t rows,std::size_t columns,NewQuantKind kind){
    if(kind!=NewQuantKind::q1_0 && kind!=NewQuantKind::q2_0 && kind!=NewQuantKind::q8_1 &&
       kind!=NewQuantKind::q8_1_f32 && kind!=NewQuantKind::nvfp4 && kind!=NewQuantKind::f8_e4m3)return false;
    const auto block=new_quant_elements(kind), bytes=new_quant_bytes(kind);
    if(!weights||!x||!y||!rows||!columns||columns%block)return false;
    if(kind==NewQuantKind::f8_e4m3){
        const auto& table=f8_e4m3_table();
        for(std::size_t r=0;r<rows;++r){
            __m256 acc=_mm256_setzero_ps();std::size_t c=0;
            const auto* row=weights+r*columns;
            for(;c+8<=columns;c+=8)acc=_mm256_fmadd_ps(new_quant_decode8(row,c,kind),_mm256_loadu_ps(x+c),acc);
            float sum=hsum_ps256(acc);for(;c<columns;++c)sum+=table[row[c]]*x[c];y[r]=sum;
        }
        return true;
    }
    const auto blocks=columns/block;
    for(std::size_t r=0;r<rows;++r){
        __m256 acc=_mm256_setzero_ps();
        for(std::size_t b=0;b<blocks;++b){
            const auto* p=weights+(r*blocks+b)*bytes;
            for(unsigned j=0;j<block;j+=8)acc=_mm256_fmadd_ps(new_quant_decode8(p,j,kind),_mm256_loadu_ps(x+b*block+j),acc);
        }
        y[r]=hsum_ps256(acc);
    }
    return true;
}

// Decode K-quant nibbles directly into vector registers.  Unlike the generic
// path this does not materialize a 256-float scratch block or horizontally
// reduce after every block. Q5_K adds a shared high-bit plane to the Q4 layout.
bool gemv_q4k_q5k_avx2(const std::uint8_t* weights, const float* x, float* y,
                        std::size_t rows, std::size_t columns, bool q5) {
    if (!weights || !x || !y || !rows || !columns || columns % 256) return false;
    const std::size_t block_bytes = q5 ? 176 : 144;
    const std::size_t blocks = columns / 256;
    const __m256i mask15 = _mm256_set1_epi32(15);
    for (std::size_t r = 0; r < rows; ++r) {
        __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(),
                         _mm256_setzero_ps(), _mm256_setzero_ps()};
        for (std::size_t b = 0; b < blocks; ++b) {
            const std::uint8_t* p = weights + (r * blocks + b) * block_bytes;
            const float d = q8_half_to_float(std::uint16_t(p[0]) | (std::uint16_t(p[1]) << 8));
            const float dm = q8_half_to_float(std::uint16_t(p[2]) | (std::uint16_t(p[3]) << 8));
            const std::uint8_t* scales = p + 4;
            const std::uint8_t* qh = p + 16;
            const std::uint8_t* ql = p + (q5 ? 48 : 16);
            for (unsigned group = 0; group < 8; ++group) {
                unsigned sc, mn;
                if (group < 4) { sc = scales[group] & 63u; mn = scales[group + 4] & 63u; }
                else {
                    sc = (scales[group + 4] & 15u) | ((scales[group - 4] >> 6) << 4);
                    mn = (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4);
                }
                const __m256 ds = _mm256_set1_ps(d * float(sc));
                const __m256 ms = _mm256_set1_ps(dm * float(mn));
                const unsigned shift = (group & 1u) * 4;
                const unsigned high_bit = group;
                for (unsigned chunk = 0; chunk < 4; ++chunk) {
                    std::uint64_t bytes;
                    std::memcpy(&bytes, ql + (group / 2) * 32 + chunk * 8, 8);
                    __m256i q = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(static_cast<long long>(bytes)));
                    if (shift) q = _mm256_srli_epi32(q, 4);
                    q = _mm256_and_si256(q, mask15);
                    if (q5) {
                        std::memcpy(&bytes, qh + chunk * 8, 8);
                        __m256i h = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(static_cast<long long>(bytes)));
                        h = _mm256_and_si256(_mm256_srlv_epi32(h, _mm256_set1_epi32(high_bit)), _mm256_set1_epi32(1));
                        q = _mm256_or_si256(q, _mm256_slli_epi32(h, 4));
                    }
                    const __m256 w = _mm256_sub_ps(_mm256_mul_ps(ds, _mm256_cvtepi32_ps(q)), ms);
                    const __m256 xv = _mm256_loadu_ps(x + b * 256 + group * 32 + chunk * 8);
                    acc[chunk] = _mm256_fmadd_ps(w, xv, acc[chunk]);
                }
            }
        }
        y[r] = hsum_ps256(_mm256_add_ps(_mm256_add_ps(acc[0], acc[1]),
                                        _mm256_add_ps(acc[2], acc[3])));
    }
    return true;
}

// Decode eight Q6_K values at the same group/lane offset. A 16-value
// scale group is never crossed by a chunk. Keep the same (d * scale) * q
// ordering as the scalar GGML decoder.
static inline __m256 q6k_decode8(const std::uint8_t* p, unsigned half,
                                  unsigned group, unsigned lane, float d) {
    std::uint64_t low_bytes, high_bytes;
    std::memcpy(&low_bytes, p + half * 64 + (group % 2) * 32 + lane, 8);
    std::memcpy(&high_bytes, p + 128 + half * 32 + lane, 8);
    __m256i lo = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(static_cast<long long>(low_bytes)));
    __m256i hi = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(static_cast<long long>(high_bytes)));
    if (group >= 2) lo = _mm256_srli_epi32(lo, 4);
    hi = _mm256_srlv_epi32(hi, _mm256_set1_epi32(group * 2));
    __m256i q = _mm256_or_si256(_mm256_and_si256(lo, _mm256_set1_epi32(15)),
        _mm256_slli_epi32(_mm256_and_si256(hi, _mm256_set1_epi32(3)), 4));
    q = _mm256_sub_epi32(q, _mm256_set1_epi32(32));
    const auto* scales = reinterpret_cast<const std::int8_t*>(p + 192);
    const float ds = d * float(scales[half * 8 + group * 2 + lane / 16]);
    return _mm256_mul_ps(_mm256_set1_ps(ds), _mm256_cvtepi32_ps(q));
}

bool gemv_q6k_avx2(const std::uint8_t* weights, const float* x, float* y,
                   std::size_t rows, std::size_t columns) {
    if (!weights || !x || !y || !rows || !columns || columns % 256) return false;
    const std::size_t blocks = columns / 256;
    for (std::size_t r = 0; r < rows; ++r) {
        __m256 sums[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(),
                          _mm256_setzero_ps(), _mm256_setzero_ps()};
        for (std::size_t b = 0; b < blocks; ++b) {
            const auto* p = weights + (r * blocks + b) * 210;
            const float d = q8_half_to_float(std::uint16_t(p[208]) | (std::uint16_t(p[209]) << 8));
            for (unsigned half = 0; half < 2; ++half)
                for (unsigned group = 0; group < 4; ++group)
                    for (unsigned lane = 0; lane < 32; lane += 8) {
                        const __m256 w = q6k_decode8(p, half, group, lane, d);
                        const __m256 xv = _mm256_loadu_ps(x + b * 256 + half * 128 + group * 32 + lane);
                        sums[lane / 8] = _mm256_fmadd_ps(w, xv, sums[lane / 8]);
                    }
        }
        y[r] = hsum_ps256(_mm256_add_ps(_mm256_add_ps(sums[0], sums[1]),
                                         _mm256_add_ps(sums[2], sums[3])));
    }
    return true;
}

// Unpack GGML Q3_K's sixteen six-bit scale fields, as in the scalar
// decoder. The resulting bytes are 0..63, with 32 subtracted per group.
static inline void q3k_scales(const std::uint8_t* p, std::uint8_t* scales) {
    std::uint32_t aux[4]{};std::memcpy(aux,p+96,12);const std::uint32_t tmp=aux[2];
    aux[2]=((aux[0]>>4)&0x0f0f0f0fu)|(((tmp>>4)&0x03030303u)<<4);
    aux[3]=((aux[1]>>4)&0x0f0f0f0fu)|(((tmp>>6)&0x03030303u)<<4);
    aux[0]=(aux[0]&0x0f0f0f0fu)|(((tmp>>0)&0x03030303u)<<4);
    aux[1]=(aux[1]&0x0f0f0f0fu)|(((tmp>>2)&0x03030303u)<<4);
    std::memcpy(scales,aux,16);
}
static inline __m256 q3k_decode8(const std::uint8_t* p, const std::uint8_t* scales,
                                  unsigned half, unsigned group, unsigned lane, float d) {
    std::uint64_t low,high;
    std::memcpy(&low,p+32+half*32+lane,8);std::memcpy(&high,p+lane,8);
    __m256i q=_mm256_cvtepu8_epi32(_mm_cvtsi64_si128(static_cast<long long>(low)));
    q=_mm256_srlv_epi32(q,_mm256_set1_epi32(group*2));
    q=_mm256_and_si256(q,_mm256_set1_epi32(3));
    __m256i h=_mm256_cvtepu8_epi32(_mm_cvtsi64_si128(static_cast<long long>(high)));
    h=_mm256_srlv_epi32(h,_mm256_set1_epi32(half*4+group));
    h=_mm256_and_si256(h,_mm256_set1_epi32(1));
    q=_mm256_sub_epi32(q,_mm256_slli_epi32(_mm256_sub_epi32(_mm256_set1_epi32(1),h),2));
    const float ds=d*float(int(scales[half*8+group*2+lane/16])-32);
    return _mm256_mul_ps(_mm256_set1_ps(ds),_mm256_cvtepi32_ps(q));
}
bool gemv_q3k_avx2(const std::uint8_t* weights,const float* x,float* y,
                   std::size_t rows,std::size_t columns){
    if(!weights||!x||!y||!rows||!columns||columns%256)return false;
    const std::size_t blocks=columns/256;
    for(std::size_t r=0;r<rows;++r){
        __m256 sums[4]={_mm256_setzero_ps(),_mm256_setzero_ps(),_mm256_setzero_ps(),_mm256_setzero_ps()};
        for(std::size_t b=0;b<blocks;++b){
            const auto*p=weights+(r*blocks+b)*110;std::uint8_t scales[16];q3k_scales(p,scales);
            const float d=q8_half_to_float(std::uint16_t(p[108])|(std::uint16_t(p[109])<<8));
            for(unsigned h=0;h<2;++h)for(unsigned g=0;g<4;++g)for(unsigned l=0;l<32;l+=8){
                const auto w=q3k_decode8(p,scales,h,g,l,d);
                const auto xv=_mm256_loadu_ps(x+b*256+h*128+g*32+l);
                sums[l/8]=_mm256_fmadd_ps(w,xv,sums[l/8]);
            }
        }
        y[r]=hsum_ps256(_mm256_add_ps(_mm256_add_ps(sums[0],sums[1]),_mm256_add_ps(sums[2],sums[3])));
    }
    return true;
}

// TQ2_0 block: two groups of 32 packed bytes; each group emits four
// planes of 32 values, with codes 0..3 mapping to -1..2.
static inline __m256 tq2_decode8(const std::uint8_t* p, unsigned group,
                                  unsigned plane, unsigned lane, float d) {
    std::uint64_t bytes; std::memcpy(&bytes, p + group * 32 + lane, 8);
    __m256i q = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(static_cast<long long>(bytes)));
    q = _mm256_srlv_epi32(q, _mm256_set1_epi32(plane * 2));
    q = _mm256_sub_epi32(_mm256_and_si256(q, _mm256_set1_epi32(3)), _mm256_set1_epi32(1));
    return _mm256_mul_ps(_mm256_cvtepi32_ps(q), _mm256_set1_ps(d));
}

bool gemv_tq2_avx2(const std::uint8_t* weights, const float* x, float* y,
                   std::size_t rows, std::size_t columns) {
    if (!weights || !x || !y || !rows || !columns || columns % 256) return false;
    const std::size_t blocks=columns/256;
    for(std::size_t r=0;r<rows;++r){
        __m256 sums[4]={_mm256_setzero_ps(),_mm256_setzero_ps(),_mm256_setzero_ps(),_mm256_setzero_ps()};
        for(std::size_t b=0;b<blocks;++b){
            const auto* p=weights+(r*blocks+b)*66;
            const float d=q8_half_to_float(std::uint16_t(p[64])|(std::uint16_t(p[65])<<8));
            for(unsigned group=0;group<2;++group)for(unsigned plane=0;plane<4;++plane)
                for(unsigned lane=0;lane<32;lane+=8){
                    auto w=tq2_decode8(p,group,plane,lane,d);
                    auto xv=_mm256_loadu_ps(x+b*256+group*128+plane*32+lane);
                    sums[lane/8]=_mm256_fmadd_ps(w,xv,sums[lane/8]);
                }
        }
        y[r]=hsum_ps256(_mm256_add_ps(_mm256_add_ps(sums[0],sums[1]),_mm256_add_ps(sums[2],sums[3])));
    }
    return true;
}

// Quantise one activation vector to Q8_0 form: 32 int8 per block plus a scale.
// Used only by the opt-in int8 path below.
static void quantise_acts_q8_0(const float* x, std::size_t columns,
                               std::vector<std::int8_t>& q, std::vector<float>& sc) {
    const std::size_t blocks = columns / 32;
    q.resize(blocks * 32);
    sc.resize(blocks);
    for (std::size_t b = 0; b < blocks; ++b) {
        const float* xb = x + b * 32;
        float amax = 0.0f;
        for (unsigned i = 0; i < 32; ++i) {
            const float a = std::fabs(xb[i]);
            amax = a > amax ? a : amax;
        }
        const float d = amax > 0.0f ? amax / 127.0f : 0.0f;
        const float inv = d > 0.0f ? 1.0f / d : 0.0f;
        sc[b] = d;
        std::int8_t* qb = q.data() + b * 32;
        for (unsigned i = 0; i < 32; ++i) {
            float v = xb[i] * inv;
            v = v > 127.0f ? 127.0f : (v < -127.0f ? -127.0f : v);
            qb[i] = static_cast<std::int8_t>(std::lround(v));
        }
    }
}

// int8 x int8 dot per block, 32 MACs in five instructions (two sign, maddubs,
// madd, convert). This is llama.cpp's non-VNNI AVX2 formulation; this CPU is Zen 3,
// so there is no _mm256_dpbusd_epi32 to use instead.
// Opt-in: quantising the activations changes the arithmetic, so the exact fp32
// path below stays the default and the parity tests keep covering it.
static bool gemv_q8_0_avx2_int8(const std::uint8_t* weights, const float* x, float* y,
                                std::size_t rows, std::size_t columns) {
    const std::size_t blocks = columns / 32, row_bytes = blocks * 34;
    thread_local std::vector<std::int8_t> xq;
    thread_local std::vector<float> xsc;
    quantise_acts_q8_0(x, columns, xq, xsc);

    for (std::size_t r = 0; r < rows; ++r) {
        const std::uint8_t* row = weights + r * row_bytes;
        __m256 acc = _mm256_setzero_ps();
        for (std::size_t b = 0; b < blocks; ++b) {
            const std::uint8_t* p = row + b * 34;
            const float d = q8_half_to_float(std::uint16_t(p[0]) | (std::uint16_t(p[1]) << 8)) * xsc[b];
            const __m256i qw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 2));
            const __m256i qa = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq.data() + b * 32));
            // |w| * (a * sign(w)) == w * a, in the unsigned-x-signed form maddubs
            // needs. |w| <= 128 and |a| <= 127, so a pair sum stays under 32767 and
            // the int16 intermediate cannot saturate.
            const __m256i aw = _mm256_sign_epi8(qw, qw);
            const __m256i sa = _mm256_sign_epi8(qa, qw);
            const __m256i p16 = _mm256_maddubs_epi16(aw, sa);
            const __m256i p32 = _mm256_madd_epi16(p16, _mm256_set1_epi16(1));
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(p32), acc);
        }
        y[r] = hsum_ps256(acc);
    }
    return true;
}

bool gemv_q8_0_avx2(const std::uint8_t* weights,const float* x,float* y,
                    std::size_t rows,std::size_t columns) {
    if(!weights||!x||!y||!rows||!columns||columns%32) return false;
    static const bool use_int8 = [] {
        const char* e = std::getenv("ZEN_CPU_Q8_ACT");
        return e && e[0] == '1';
    }();
    if (use_int8) return gemv_q8_0_avx2_int8(weights, x, y, rows, columns);

    // Exact fp32 activations (the parity path).
    //
    // Kept bit-comparable with dequantise-then-dot, but the reduction is now a
    // single horizontal sum at the end. The old version stored the vector to a
    // stack array and summed eight scalars for every 32-element block, so a
    // 2048-wide row paid 64 stores plus 512 scalar adds on top of the dot
    // products; here each block costs four vector adds and one FMA.
    const std::size_t blocks=columns/32, row_bytes=blocks*34;
    for(std::size_t r=0;r<rows;++r){
        const std::uint8_t* row=weights+r*row_bytes;
        __m256 acc=_mm256_setzero_ps();
        for(std::size_t b=0;b<blocks;++b){
            const std::uint8_t* p=row+b*34;
            const float d=q8_half_to_float(std::uint16_t(p[0])|(std::uint16_t(p[1])<<8));
            __m256 a0=_mm256_setzero_ps(),a1=_mm256_setzero_ps();
            __m256 a2=_mm256_setzero_ps(),a3=_mm256_setzero_ps();
            for(unsigned chunk=0;chunk<4;++chunk){
                std::uint64_t raw=0;std::memcpy(&raw,p+2+chunk*8,8);
                const __m128i bytes=_mm_cvtsi64_si128(static_cast<long long>(raw));
                const __m256i q32=_mm256_cvtepi8_epi32(bytes);
                const __m256 qf=_mm256_cvtepi32_ps(q32);
                const __m256 xv=_mm256_loadu_ps(x+b*32+chunk*8);
                if(chunk==0)a0=_mm256_fmadd_ps(qf,xv,a0);
                else if(chunk==1)a1=_mm256_fmadd_ps(qf,xv,a1);
                else if(chunk==2)a2=_mm256_fmadd_ps(qf,xv,a2);
                else a3=_mm256_fmadd_ps(qf,xv,a3);
            }
            const __m256 s=_mm256_add_ps(_mm256_add_ps(a0,a1),_mm256_add_ps(a2,a3));
            acc=_mm256_fmadd_ps(_mm256_set1_ps(d),s,acc);
        }
        y[r]=hsum_ps256(acc);
    }
    return true;
}
}

namespace zen {
namespace {
// exp(x) for x in [-88.4, 88.4]; ~1e-7 relative error (Cephes split + degree-6 poly).
inline __m256 exp256_ps(__m256 x) {
    x=_mm256_min_ps(x,_mm256_set1_ps(88.3762626647949f));
    x=_mm256_max_ps(x,_mm256_set1_ps(-88.3762626647949f));
    __m256 k=_mm256_floor_ps(_mm256_fmadd_ps(x,_mm256_set1_ps(1.44269504088896341f),_mm256_set1_ps(0.5f)));
    __m256 r=_mm256_fnmadd_ps(k,_mm256_set1_ps(0.693359375f),x);
    r=_mm256_fnmadd_ps(k,_mm256_set1_ps(-2.12194440e-4f),r);
    __m256 p=_mm256_set1_ps(1.9875691500e-4f);
    p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(1.3981999507e-3f));
    p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(8.3334519073e-3f));
    p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(4.1665795894e-2f));
    p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(1.6666665459e-1f));
    p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(5.0000001201e-1f));
    p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(1.0f));
    p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(1.0f));
    const __m256i e=_mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(k),_mm256_set1_epi32(127)),23);
    return _mm256_mul_ps(p,_mm256_castsi256_ps(e));
}
inline float half_to_float_local(std::uint16_t h) noexcept { return q8_half_to_float(h); }
} // namespace

// out = silu(gate) * up = (gate / (1 + exp(-gate))) * up
void silu_mul_avx2(const float* gate, const float* up, float* out, std::size_t n) {
    const __m256 one=_mm256_set1_ps(1.0f), zero=_mm256_setzero_ps();
    std::size_t i=0;
    for(;i+7<n;i+=8){
        const __m256 z=_mm256_loadu_ps(gate+i);
        const __m256 sig=_mm256_div_ps(one,_mm256_add_ps(one,exp256_ps(_mm256_sub_ps(zero,z))));
        _mm256_storeu_ps(out+i,_mm256_mul_ps(_mm256_mul_ps(z,sig),_mm256_loadu_ps(up+i)));
    }
    for(;i<n;++i){const float z=gate[i];out[i]=(z/(1.0f+std::exp(-z)))*up[i];}
}

float dot_f32_avx2(const float* a, const float* b, std::size_t n) noexcept {
    __m256 s0=_mm256_setzero_ps(),s1=_mm256_setzero_ps();
    std::size_t i=0;
    for(;i+15<n;i+=16){
        s0=_mm256_fmadd_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i),s0);
        s1=_mm256_fmadd_ps(_mm256_loadu_ps(a+i+8),_mm256_loadu_ps(b+i+8),s1);
    }
    __m256 s=_mm256_add_ps(s0,s1);
    for(;i+7<n;i+=8) s=_mm256_fmadd_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i),s);
    alignas(32) float lane[8];
    _mm256_store_ps(lane,s);
    float t=lane[0]+lane[1]+lane[2]+lane[3]+lane[4]+lane[5]+lane[6]+lane[7];
    for(;i<n;++i) t+=a[i]*b[i];
    return t;
}

void axpy_f32_avx2(float a, const float* x, float* y, std::size_t n) noexcept {
    const __m256 va=_mm256_set1_ps(a);
    std::size_t i=0;
    for(;i+7<n;i+=8)
        _mm256_storeu_ps(y+i,_mm256_fmadd_ps(va,_mm256_loadu_ps(x+i),_mm256_loadu_ps(y+i)));
    for(;i<n;++i) y[i]+=a*x[i];
}

// RMSNorm AVX2. Scalar reference (dense_transformer_runtime.cpp):
//   double ss=0; for i: ss+=double(x[i])*x[i];
//   float inv=1.0f/sqrt(float(ss/n)+eps); for i: y[i]=x[i]*inv*weight[i];
// The ss reduction here is NOT bit-identical to that scalar loop: the four
// __m256d accumulators accumulate disjoint 4-element chunks in parallel and
// are combined once at the end, so the double-precision partial sums are
// added in a different associativity order (scalar: strictly sequential).
// In double precision the reordering error is ~1e-13 relative on the ss for
// the hidden sizes used here (2048/8192), i.e. at most ~1 ulp of float on
// the final y after the float cast of ss/n -- well inside the core_tests
// tolerance for this kernel. inv is computed in float exactly as scalar
// (1.0f/sqrt(float(ss/n)+eps)); the apply loop is vectorised float
// x*inv*weight, one multiply chain identical to the scalar expression.
// Aliasing x==y is safe: each output element is read before it is written.
void rms_norm_avx2(const float* x, const float* weight, float* y, std::size_t n, float eps) {
    if(!x||!weight||!y||!n) return;
    __m256d s0=_mm256_setzero_pd(),s1=_mm256_setzero_pd();
    __m256d s2=_mm256_setzero_pd(),s3=_mm256_setzero_pd();
    std::size_t i=0;
    for(;i+15<n;i+=16){
        const __m256 v0=_mm256_loadu_ps(x+i);
        const __m256 v1=_mm256_loadu_ps(x+i+8);
        const __m256d d0=_mm256_cvtps_pd(_mm256_castps256_ps128(v0));
        const __m256d d1=_mm256_cvtps_pd(_mm256_extractf128_ps(v0,1));
        const __m256d d2=_mm256_cvtps_pd(_mm256_castps256_ps128(v1));
        const __m256d d3=_mm256_cvtps_pd(_mm256_extractf128_ps(v1,1));
        s0=_mm256_fmadd_pd(d0,d0,s0);
        s1=_mm256_fmadd_pd(d1,d1,s1);
        s2=_mm256_fmadd_pd(d2,d2,s2);
        s3=_mm256_fmadd_pd(d3,d3,s3);
    }
    if(i+7<n){
        const __m256 v=_mm256_loadu_ps(x+i);
        const __m256d d0=_mm256_cvtps_pd(_mm256_castps256_ps128(v));
        const __m256d d1=_mm256_cvtps_pd(_mm256_extractf128_ps(v,1));
        s0=_mm256_fmadd_pd(d0,d0,s0);
        s1=_mm256_fmadd_pd(d1,d1,s1);
        i+=8;
    }
    __m256d total_pd=_mm256_add_pd(_mm256_add_pd(s0,s1),_mm256_add_pd(s2,s3));
    alignas(32) double lanes[4];
    _mm256_store_pd(lanes,total_pd);
    double ss=lanes[0]+lanes[1]+lanes[2]+lanes[3];
    for(;i<n;++i) ss+=double(x[i])*x[i];
    const float inv=1.0f/std::sqrt(static_cast<float>(ss/double(n))+eps);
    const __m256 vinv=_mm256_set1_ps(inv);
    for(std::size_t j=0;j+7<n;j+=8)
        _mm256_storeu_ps(y+j,_mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x+j),vinv),
                                           _mm256_loadu_ps(weight+j)));
    for(std::size_t j=n&~(std::size_t)7;j<n;++j) y[j]=x[j]*inv*weight[j];
}

bool gemm_q8_0_avx2(const std::uint8_t* weights, const float* input, float* output,
                    std::size_t tokens, std::size_t rows, std::size_t columns,
                    std::size_t output_row_stride) {
    constexpr std::size_t kBlockElements=32, kBlockBytes=34;
    if(!weights||!input||!output||!tokens||!rows||!columns) return false;
    if(columns%kBlockElements) return false;
    const std::size_t row_bytes=(columns/kBlockElements)*kBlockBytes;
    std::vector<float> b0(columns),b1(columns);
    auto dequant_row=[&](const std::uint8_t* row,float* dst){
        for(std::size_t blk=0;blk<columns/kBlockElements;++blk){
            const std::uint8_t* src=row+blk*kBlockBytes;
            const float d=half_to_float_local(std::uint16_t(src[0])|(std::uint16_t(src[1])<<8));
            float* dst_row=dst+blk*kBlockElements;
            for(std::size_t j=0;j<kBlockElements;j+=8){
                const __m128i b8=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(src+2+j));
                const __m256i i32=_mm256_cvtepi8_epi32(b8);
                _mm256_storeu_ps(dst_row+j,_mm256_mul_ps(_mm256_cvtepi32_ps(i32),_mm256_set1_ps(d)));
            }
        }
    };
    std::size_t r=0;
    // Two weight rows x four tokens: eight accumulators fed by two weight loads
    // and four activation loads per column chunk. The 8x1 structure re-read the
    // activation matrix once per row; pairing rows halves that traffic (measured
    // 448 -> 649 GF/s on Llama-3.2-1B shapes, bit-identical results).
    for(;r+1<rows;r+=2){
        const std::uint8_t* wrow0=weights+r*row_bytes;
        const std::uint8_t* wrow1=weights+(r+1)*row_bytes;
        dequant_row(wrow0,b0.data());
        dequant_row(wrow1,b1.data());
        std::size_t t=0;
        for(;t+4<=tokens;t+=4){
            __m256 a00=_mm256_setzero_ps(),a01=_mm256_setzero_ps(),a02=_mm256_setzero_ps(),a03=_mm256_setzero_ps();
            __m256 a10=_mm256_setzero_ps(),a11=_mm256_setzero_ps(),a12=_mm256_setzero_ps(),a13=_mm256_setzero_ps();
            std::size_t c=0;
            for(;c+8<=columns;c+=8){
                const __m256 x0=_mm256_loadu_ps(input+(t+0)*columns+c),x1=_mm256_loadu_ps(input+(t+1)*columns+c);
                const __m256 x2=_mm256_loadu_ps(input+(t+2)*columns+c),x3=_mm256_loadu_ps(input+(t+3)*columns+c);
                const __m256 v0=_mm256_loadu_ps(b0.data()+c),v1=_mm256_loadu_ps(b1.data()+c);
                a00=_mm256_fmadd_ps(v0,x0,a00);a01=_mm256_fmadd_ps(v0,x1,a01);
                a02=_mm256_fmadd_ps(v0,x2,a02);a03=_mm256_fmadd_ps(v0,x3,a03);
                a10=_mm256_fmadd_ps(v1,x0,a10);a11=_mm256_fmadd_ps(v1,x1,a11);
                a12=_mm256_fmadd_ps(v1,x2,a12);a13=_mm256_fmadd_ps(v1,x3,a13);
            }
            __m256 acc[8]={a00,a01,a02,a03,a10,a11,a12,a13};
            alignas(32) float lane[8];
            for(int j=0;j<8;++j){
                const std::size_t ti=t+static_cast<std::size_t>(j&3);
                const std::size_t ri=r+static_cast<std::size_t>(j>>2);
                const float* wsrc=(j<4)?b0.data():b1.data();
                _mm256_store_ps(lane,acc[j]);
                float total=lane[0]+lane[1]+lane[2]+lane[3]+lane[4]+lane[5]+lane[6]+lane[7];
                for(std::size_t cc=columns&~(std::size_t)7;cc<columns;++cc)
                    total+=wsrc[cc]*input[ti*columns+cc];
                output[ti*output_row_stride+ri]=total;
            }
        }
        for(;t<tokens;++t){
            output[t*output_row_stride+r]  =dot_f32_avx2(b0.data(),input+t*columns,columns);
            output[t*output_row_stride+r+1]=dot_f32_avx2(b1.data(),input+t*columns,columns);
        }
    }
    if(r<rows){
        dequant_row(weights+r*row_bytes,b0.data());
        for(std::size_t t=0;t<tokens;++t)
            output[t*output_row_stride+r]=dot_f32_avx2(b0.data(),input+t*columns,columns);
    }
    return true;
}

// K-blocked batched Q8_0 GEMM (Priority 1). Loop order is K-panel outer,
// rows inner, tokens innermost per micro-tile, so the activation panel
// [tokens, Kc] is loaded from DRAM once per worker and then reused from L2
// across every row block of that worker. The legacy kernel above is
// rows-pair outer / full-K inner and re-streams the whole [tokens, columns]
// activation matrix once per row pair (~5.6 MB for T=171, K=8192), thrashing
// the 512 KB Zen 3 L2.
//
// Bit-identity: partials are carried as __m256 vectors in memory across
// panels and reduced with a single horizontal sum at the end, using the
// same lane order as gemm_q8_0_avx2 (lane0+...+lane7). The per-(row,token)
// FMA sequence is still c = 0,8,16,... in order, so the main 4-row block
// is bit-identical to the legacy kernel. No per-panel scalar reduction.
// Edge rows (<4 per worker slice) and the dot-product tails use
// dot_f32_avx2, matching the legacy tails.
//
// Tile: MR=4 rows x NR=2 tokens = 8 accumulators. With 6 operand loads per
// step (2 activations + 4 weights) the live set is ~14 ymm registers
// (8 acc + 6 temps) on AVX2's 16-register file, leaving headroom so the
// accumulators stay in registers (checked in disassembly; no stack spills
// of acc in the inner Kc loop). 3x4 (12 acc) was considered but 4x2 is the
// safer start. Rows are blocked by RB (ZEN_CPU_Q8_RB, default 64) so the
// vector partials [tokens, RB] plus the activation panel [tokens, Kc]
// stay L2-resident; panel loads/stores of partials amortise better at
// large Kc. Kc (ZEN_CPU_Q8_KC) and RB were swept on this machine
// (bench + single-thread isolation, see report): Kc=512/RB=64 wins the
// large-activation shapes (down T=171: ~10.1 ms vs legacy ~13.0 ms;
// single-thread ~65-67 ms vs ~71 ms) while Kc=256/RB=32 wins small
// shapes but loses large ones, so the defaults are Kc=512, RB=64.
// Loop nest per RB block is r0 { k0 { rl { t { k } } } }, i.e. rows are
// blocked outer and K panels inner, so partials for one RB stay in L2
// across panels instead of streaming a full-slice partial matrix.
// The weight-panel dequant reads inside k0 are strided across rows
// (consecutive rows are row_bytes apart in RAM); a software prefetch of
// the next row-tile's panel is issued while the current tile's token
// FMAs run (see the prefetch site in the rl loop).
bool gemm_q8_0_kblk_avx2(const std::uint8_t* weights, const float* input, float* output,
                     std::size_t tokens, std::size_t rows, std::size_t columns,
                     std::size_t output_row_stride) {
    constexpr std::size_t kBlockElements=32, kBlockBytes=34;
    constexpr std::size_t kMR=4, kNR=2;
    if(!weights||!input||!output||!tokens||!rows||!columns) return false;
    if(columns%kBlockElements) return false;
    if(columns&7) return false; // columns%32==0 already implies this; defensive
    static const std::size_t kKc=[](){
        const char* e=std::getenv("ZEN_CPU_Q8_KC");
        if(e){
            const long v=std::atol(e);
            if(v==64||v==128||v==256||v==512) return static_cast<std::size_t>(v);
        }
        return static_cast<std::size_t>(512);
    }();
    const std::size_t row_bytes=(columns/kBlockElements)*kBlockBytes;
    const std::size_t rows_main=(rows/kMR)*kMR;
    auto dequant_full_row=[&](const std::uint8_t* row,float* dst){
        for(std::size_t blk=0;blk<columns/kBlockElements;++blk){
            const std::uint8_t* src=row+blk*kBlockBytes;
            const float d=half_to_float_local(std::uint16_t(src[0])|(std::uint16_t(src[1])<<8));
            float* dst_row=dst+blk*kBlockElements;
            for(std::size_t j=0;j<kBlockElements;j+=8){
                const __m128i b8=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(src+2+j));
                const __m256i i32=_mm256_cvtepi8_epi32(b8);
                _mm256_storeu_ps(dst_row+j,_mm256_mul_ps(_mm256_cvtepi32_ps(i32),_mm256_set1_ps(d)));
            }
        }
    };
    static const std::size_t kRB=[](){
        const char* e=std::getenv("ZEN_CPU_Q8_RB");
        if(e){
            const long v=std::atol(e);
            if(v==32||v==64||v==128||v==256) return static_cast<std::size_t>(v);
        }
        return static_cast<std::size_t>(64);
    }();
    if(rows_main>0){
        std::vector<VectorAccumulator> partial(std::size_t(tokens)*kRB);
        std::vector<float> wpanel(kMR*kKc);
        alignas(32) float lane[8];
        for(std::size_t r0=0;r0<rows_main;r0+=kRB){
            const std::size_t RBc=(kRB<rows_main-r0)?kRB:(rows_main-r0);
            for(std::size_t k0=0;k0<columns;k0+=kKc){
                const std::size_t kc=std::size_t((kKc<columns-k0)?kKc:(columns-k0));
                const std::size_t panel_blk=k0/kBlockElements;
                const bool first_panel=(k0==0);
                for(std::size_t rl=0;rl<RBc;rl+=kMR){
                    const std::size_t r=r0+rl;
                // Dequantise the 4-row x kc weight panel once; it is reused
                // across every token block below (L1-resident: 4*512*4=8KB max).
                for(std::size_t dr=0;dr<kMR;++dr){
                    const std::uint8_t* src=weights+(r+dr)*row_bytes+panel_blk*kBlockBytes;
                    float* dst=wpanel.data()+dr*kKc;
                    const std::size_t nblk=kc/kBlockElements;
                    for(std::size_t blk=0;blk<nblk;++blk){
                        const std::uint8_t* p=src+blk*kBlockBytes;
                        const float d=half_to_float_local(std::uint16_t(p[0])|(std::uint16_t(p[1])<<8));
                        float* dst_row=dst+blk*kBlockElements;
                        for(std::size_t j=0;j<kBlockElements;j+=8){
                            const __m128i b8=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p+2+j));
                            const __m256i i32=_mm256_cvtepi8_epi32(b8);
                            _mm256_storeu_ps(dst_row+j,_mm256_mul_ps(_mm256_cvtepi32_ps(i32),_mm256_set1_ps(d)));
                        }
                    }
                }
                // Software prefetch (KBLK prefetch fix, one site): the weight
                // panel reads above are strided -- each row's kc panel sits
                // row_bytes (~8.7KB for K=8192) after the previous row's, so
                // the hardware prefetcher sees up to RB/kMR concurrent
                // strided streams and does not run ahead. Issue prefetches
                // for the NEXT row-tile's panel (same k0 panel, rows
                // r+kMR..r+2*kMR-1) now, while the token FMA loop below
                // (the long compute between this dequant and the next one)
                // runs, so those lines are in L2/L3 when the next dequant
                // reads them. Two lines per row (panel start and +256B,
                // the prefetch-distance point of a 544B Kc=512 panel)
                // keeps the instruction cost at 8 prefetches per tile.
                // Prefetch is read-only and non-faulting, so it cannot
                // change results (bit-identity is preserved).
                // Measured (see src/quant.cpp): this does NOT recover
                // the model-level KBLK regression -- the activation
                // panel evicts the prefetched lines before the next
                // dequant, so KBLK stays opt-in (default off).
                if(rl+kMR<RBc){
                    for(std::size_t dr=0;dr<kMR;++dr){
                        const std::uint8_t* nxt=weights+(r+kMR+dr)*row_bytes+panel_blk*kBlockBytes;
                        _mm_prefetch(reinterpret_cast<const char*>(nxt),_MM_HINT_T0);
                        _mm_prefetch(reinterpret_cast<const char*>(nxt+256),_MM_HINT_T0);
                    }
                }
                for(std::size_t t=0;t<tokens;t+=kNR){
                    const float* in0=input+(t+0)*columns+k0;
                    if(t+1<tokens){
                        const float* in1=input+(t+1)*columns+k0;
                        __m256 a00,a01,a10,a11,a20,a21,a30,a31;
                        VectorAccumulator* p;
                        if(first_panel){
                            a00=_mm256_setzero_ps();a01=_mm256_setzero_ps();
                            a10=_mm256_setzero_ps();a11=_mm256_setzero_ps();
                            a20=_mm256_setzero_ps();a21=_mm256_setzero_ps();
                            a30=_mm256_setzero_ps();a31=_mm256_setzero_ps();
                        }else{
                            p=&partial[t*RBc+rl];
                            a00=p[0*RBc].value;
                            a01=p[1*RBc].value;
                            a10=p[0*RBc+1].value;
                            a11=p[1*RBc+1].value;
                            a20=p[0*RBc+2].value;
                            a21=p[1*RBc+2].value;
                            a30=p[0*RBc+3].value;
                            a31=p[1*RBc+3].value;
                        }
                        const float* w=wpanel.data();
                        for(std::size_t s=0;s<kc;s+=8){
                            const __m256 x0=_mm256_loadu_ps(in0+s);
                            const __m256 x1=_mm256_loadu_ps(in1+s);
                            const __m256 v0=_mm256_loadu_ps(w+0*kKc+s);
                            const __m256 v1=_mm256_loadu_ps(w+1*kKc+s);
                            const __m256 v2=_mm256_loadu_ps(w+2*kKc+s);
                            const __m256 v3=_mm256_loadu_ps(w+3*kKc+s);
                            a00=_mm256_fmadd_ps(v0,x0,a00);a01=_mm256_fmadd_ps(v0,x1,a01);
                            a10=_mm256_fmadd_ps(v1,x0,a10);a11=_mm256_fmadd_ps(v1,x1,a11);
                            a20=_mm256_fmadd_ps(v2,x0,a20);a21=_mm256_fmadd_ps(v2,x1,a21);
                            a30=_mm256_fmadd_ps(v3,x0,a30);a31=_mm256_fmadd_ps(v3,x1,a31);
                        }
                        p=&partial[t*RBc+rl];
                        p[0*RBc].value=a00;
                        p[1*RBc].value=a01;
                        p[0*RBc+1].value=a10;
                        p[1*RBc+1].value=a11;
                        p[0*RBc+2].value=a20;
                        p[1*RBc+2].value=a21;
                        p[0*RBc+3].value=a30;
                        p[1*RBc+3].value=a31;
                    }else{
                        __m256 a0,a1,a2,a3;
                        VectorAccumulator* p;
                        if(first_panel){
                            a0=_mm256_setzero_ps();a1=_mm256_setzero_ps();
                            a2=_mm256_setzero_ps();a3=_mm256_setzero_ps();
                        }else{
                            p=&partial[t*RBc+rl];
                            a0=p[0].value;
                            a1=p[1].value;
                            a2=p[2].value;
                            a3=p[3].value;
                        }
                        const float* w=wpanel.data();
                        for(std::size_t s=0;s<kc;s+=8){
                            const __m256 x0=_mm256_loadu_ps(in0+s);
                            const __m256 v0=_mm256_loadu_ps(w+0*kKc+s);
                            const __m256 v1=_mm256_loadu_ps(w+1*kKc+s);
                            const __m256 v2=_mm256_loadu_ps(w+2*kKc+s);
                            const __m256 v3=_mm256_loadu_ps(w+3*kKc+s);
                            a0=_mm256_fmadd_ps(v0,x0,a0);
                            a1=_mm256_fmadd_ps(v1,x0,a1);
                            a2=_mm256_fmadd_ps(v2,x0,a2);
                            a3=_mm256_fmadd_ps(v3,x0,a3);
                        }
                        p=&partial[t*RBc+rl];
                        p[0].value=a0;
                        p[1].value=a1;
                        p[2].value=a2;
                        p[3].value=a3;
                    }
                }
            }
        }
        for(std::size_t t=0;t<tokens;++t){
            const VectorAccumulator* prow=&partial[t*RBc];
            for(std::size_t rl=0;rl<RBc;++rl){
                _mm256_store_ps(lane,prow[rl].value);
                output[t*output_row_stride+r0+rl]=lane[0]+lane[1]+lane[2]+lane[3]+lane[4]+lane[5]+lane[6]+lane[7];
            }
        }
        }
    }
    if(rows==rows_main) return true;
    std::vector<float> b0(columns);
    for(std::size_t r=rows_main;r<rows;++r){
        dequant_full_row(weights+r*row_bytes,b0.data());
        for(std::size_t t=0;t<tokens;++t)
            output[t*output_row_stride+r]=dot_f32_avx2(b0.data(),input+t*columns,columns);
    }
    return true;
}

// Repacked sequential variant of the kernel above. Same MR=4 x NR=2
// micro-tile, same __m256 partials and single hsum, same c=0,..., FMA order,
// so the main block is bit-identical to gemm_q8_0_kblk_avx2. The only
// difference is the weight-panel source: instead of weights+(r)*row_bytes+
// panel_blk*34 (row_bytes apart), each (global row, kpanel) panel is read
// from the panel-major buffer built by quant::repack_q8_0_kblk at
//   ((r/64)*n_kpanels + kpanel)*64*544 + (r%64)*544
// with Kc=512, RB=64 pinned (not env-driven) so the buffer built at load
// always matches the reader. The 4-row panel tile is one contiguous
// 2176 B chunk and successive row tiles within one (rblock,kpanel) tile are
// contiguous, so the hardware prefetcher streams the repack sequentially;
// no _mm_prefetch is issued (the activation-panel eviction that defeated
// the raw kernel's next-tile prefetch cannot happen when the next tile's
// lines are the next lines in RAM). row_offset is the global row of the
// first row in this slice: the offset is computed per row from the global
// index, so parallel slices that do not start on an RB boundary are
// correct (tiles that straddle an rblock boundary read both halves via
// their per-row offsets). Padded rows/blocks past the last rblock/kpanel
// are never read (kc and RBc are clamped to the slice).
bool gemm_q8_0_kblk_repack_avx2(const std::uint8_t* repacked, std::size_t row_offset,
                    const float* input, float* output,
                    std::size_t tokens, std::size_t slice_rows,
                    std::size_t columns, std::size_t output_row_stride) {
    constexpr std::size_t kBlockElements=32, kBlockBytes=34;
    constexpr std::size_t kMR=4, kNR=2;
    constexpr std::size_t kKc=512, kRB=64;
    constexpr std::size_t kBlocksPerPanel=kKc/kBlockElements; // 16
    constexpr std::size_t kPanelBytes=kBlocksPerPanel*kBlockBytes; // 544
    if(!repacked||!input||!output||!tokens||!slice_rows||!columns) return false;
    if(columns%kBlockElements) return false;
    if(columns&7) return false;
    const std::size_t blocks_per_row=columns/kBlockElements;
    const std::size_t n_kpanels=(blocks_per_row+kBlocksPerPanel-1)/kBlocksPerPanel;
    const std::size_t rows_main=(slice_rows/kMR)*kMR;
    auto panel_ptr=[&](std::size_t global_row, std::size_t kpanel){
        const std::size_t rblock=global_row/kRB;
        const std::size_t row_in=global_row%kRB;
        return repacked+((rblock*n_kpanels+kpanel)*kRB+row_in)*kPanelBytes;
    };
    auto dequant_full_row_repacked=[&](std::size_t global_row,float* dst){
        for(std::size_t kp=0;kp<n_kpanels;++kp){
            const std::uint8_t* src=panel_ptr(global_row,kp);
            const std::size_t base_blk=kp*kBlocksPerPanel;
            const std::size_t nblk=std::min(kBlocksPerPanel,blocks_per_row-base_blk);
            for(std::size_t blk=0;blk<nblk;++blk){
                const std::uint8_t* p=src+blk*kBlockBytes;
                const float d=half_to_float_local(std::uint16_t(p[0])|(std::uint16_t(p[1])<<8));
                float* dst_row=dst+(base_blk+blk)*kBlockElements;
                for(std::size_t j=0;j<kBlockElements;j+=8){
                    const __m128i b8=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p+2+j));
                    const __m256i i32=_mm256_cvtepi8_epi32(b8);
                    _mm256_storeu_ps(dst_row+j,_mm256_mul_ps(_mm256_cvtepi32_ps(i32),_mm256_set1_ps(d)));
                }
            }
        }
    };
    if(rows_main>0){
        std::vector<VectorAccumulator> partial(std::size_t(tokens)*kRB);
        std::vector<float> wpanel(kMR*kKc);
        alignas(32) float lane[8];
        for(std::size_t r0=0;r0<rows_main;r0+=kRB){
            const std::size_t RBc=(kRB<rows_main-r0)?kRB:(rows_main-r0);
            for(std::size_t k0=0;k0<columns;k0+=kKc){
                const std::size_t kc=std::size_t((kKc<columns-k0)?kKc:(columns-k0));
                const std::size_t kpanel=k0/kKc;
                const bool first_panel=(k0==0);
                for(std::size_t rl=0;rl<RBc;rl+=kMR){
                    const std::size_t r=r0+rl;
                    const std::size_t gr=row_offset+r;
                // Dequantise the 4-row x kc weight panel once (L1-resident:
                // 4*512*4=8KB max), reused across every token below.
                for(std::size_t dr=0;dr<kMR;++dr){
                    const std::uint8_t* src=panel_ptr(gr+dr,kpanel);
                    float* dst=wpanel.data()+dr*kKc;
                    const std::size_t nblk=kc/kBlockElements;
                    for(std::size_t blk=0;blk<nblk;++blk){
                        const std::uint8_t* p=src+blk*kBlockBytes;
                        const float d=half_to_float_local(std::uint16_t(p[0])|(std::uint16_t(p[1])<<8));
                        float* dst_row=dst+blk*kBlockElements;
                        for(std::size_t j=0;j<kBlockElements;j+=8){
                            const __m128i b8=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p+2+j));
                            const __m256i i32=_mm256_cvtepi8_epi32(b8);
                            _mm256_storeu_ps(dst_row+j,_mm256_mul_ps(_mm256_cvtepi32_ps(i32),_mm256_set1_ps(d)));
                        }
                    }
                }
                for(std::size_t t=0;t<tokens;t+=kNR){
                    const float* in0=input+(t+0)*columns+k0;
                    if(t+1<tokens){
                        const float* in1=input+(t+1)*columns+k0;
                        __m256 a00,a01,a10,a11,a20,a21,a30,a31;
                        VectorAccumulator* p;
                        if(first_panel){
                            a00=_mm256_setzero_ps();a01=_mm256_setzero_ps();
                            a10=_mm256_setzero_ps();a11=_mm256_setzero_ps();
                            a20=_mm256_setzero_ps();a21=_mm256_setzero_ps();
                            a30=_mm256_setzero_ps();a31=_mm256_setzero_ps();
                        }else{
                            p=&partial[t*RBc+rl];
                            a00=p[0*RBc].value;
                            a01=p[1*RBc].value;
                            a10=p[0*RBc+1].value;
                            a11=p[1*RBc+1].value;
                            a20=p[0*RBc+2].value;
                            a21=p[1*RBc+2].value;
                            a30=p[0*RBc+3].value;
                            a31=p[1*RBc+3].value;
                        }
                        const float* w=wpanel.data();
                        for(std::size_t s=0;s<kc;s+=8){
                            const __m256 x0=_mm256_loadu_ps(in0+s);
                            const __m256 x1=_mm256_loadu_ps(in1+s);
                            const __m256 v0=_mm256_loadu_ps(w+0*kKc+s);
                            const __m256 v1=_mm256_loadu_ps(w+1*kKc+s);
                            const __m256 v2=_mm256_loadu_ps(w+2*kKc+s);
                            const __m256 v3=_mm256_loadu_ps(w+3*kKc+s);
                            a00=_mm256_fmadd_ps(v0,x0,a00);a01=_mm256_fmadd_ps(v0,x1,a01);
                            a10=_mm256_fmadd_ps(v1,x0,a10);a11=_mm256_fmadd_ps(v1,x1,a11);
                            a20=_mm256_fmadd_ps(v2,x0,a20);a21=_mm256_fmadd_ps(v2,x1,a21);
                            a30=_mm256_fmadd_ps(v3,x0,a30);a31=_mm256_fmadd_ps(v3,x1,a31);
                        }
                        p=&partial[t*RBc+rl];
                        p[0*RBc].value=a00;
                        p[1*RBc].value=a01;
                        p[0*RBc+1].value=a10;
                        p[1*RBc+1].value=a11;
                        p[0*RBc+2].value=a20;
                        p[1*RBc+2].value=a21;
                        p[0*RBc+3].value=a30;
                        p[1*RBc+3].value=a31;
                    }else{
                        __m256 a0,a1,a2,a3;
                        VectorAccumulator* p;
                        if(first_panel){
                            a0=_mm256_setzero_ps();a1=_mm256_setzero_ps();
                            a2=_mm256_setzero_ps();a3=_mm256_setzero_ps();
                        }else{
                            p=&partial[t*RBc+rl];
                            a0=p[0].value;
                            a1=p[1].value;
                            a2=p[2].value;
                            a3=p[3].value;
                        }
                        const float* w=wpanel.data();
                        for(std::size_t s=0;s<kc;s+=8){
                            const __m256 x0=_mm256_loadu_ps(in0+s);
                            const __m256 v0=_mm256_loadu_ps(w+0*kKc+s);
                            const __m256 v1=_mm256_loadu_ps(w+1*kKc+s);
                            const __m256 v2=_mm256_loadu_ps(w+2*kKc+s);
                            const __m256 v3=_mm256_loadu_ps(w+3*kKc+s);
                            a0=_mm256_fmadd_ps(v0,x0,a0);
                            a1=_mm256_fmadd_ps(v1,x0,a1);
                            a2=_mm256_fmadd_ps(v2,x0,a2);
                            a3=_mm256_fmadd_ps(v3,x0,a3);
                        }
                        p=&partial[t*RBc+rl];
                        p[0].value=a0;
                        p[1].value=a1;
                        p[2].value=a2;
                        p[3].value=a3;
                    }
                }
            }
        }
        for(std::size_t t=0;t<tokens;++t){
            const VectorAccumulator* prow=&partial[t*RBc];
            for(std::size_t rl=0;rl<RBc;++rl){
                _mm256_store_ps(lane,prow[rl].value);
                output[t*output_row_stride+r0+rl]=lane[0]+lane[1]+lane[2]+lane[3]+lane[4]+lane[5]+lane[6]+lane[7];
            }
        }
        }
    }
    if(slice_rows==rows_main) return true;
    std::vector<float> b0(columns);
    for(std::size_t r=rows_main;r<slice_rows;++r){
        dequant_full_row_repacked(row_offset+r,b0.data());
        for(std::size_t t=0;t<tokens;++t)
            output[t*output_row_stride+r]=dot_f32_avx2(b0.data(),input+t*columns,columns);
    }
    return true;
}

// -----------------------------------------------------------------------------
// Non-Q8 K-blocked batched GEMM (F16, Q4_K). Loop order per 4-row block is
// K-panel outer, token-pairs inner, so the fp32 weight panel [4, Kc] is
// dequantised once per panel and reused across every token from L1
// (4*128*4 = 2 KB at the default Kc=128). Partials are carried as __m256 in
// memory across K-panels and reduced with a single horizontal sum at the
// end, same hsum lane order (lane[0]+...+lane[7]) as gemm_q8_0_kblk_avx2;
// per-element FMA order is c = 0,8,16,... The dequantised weights are
// bit-identical to the scalar dequantize() in quant.cpp (F16: exact integer
// half->float with a scalar patch for exp 0/31 lanes; Q4_K: mul+sub, not
// FMA), so only the dot-product accumulation grouping differs from the plain
// path (FP reassociation; the plain gemm path is fp32 lanes too, via
// gemv_avx2, not a double accumulator). Kc is taken from ZEN_CPU_F16_KC /
// ZEN_CPU_Q4K_KC (64/128/256, default 128). Rows that do not fill a 4-row
// tile use the plain tail (full-row dequant + dot_f32_avx2).
namespace {
inline std::uint16_t read16_local(const std::uint8_t* p) noexcept {
    return std::uint16_t(p[0]) | (std::uint16_t(p[1]) << 8);
}
// Copy of k_scale_min in src/quant.cpp (packed 6-bit scale/min pairs).
inline void k_scale_min_local(unsigned j, const std::uint8_t* q,
                              std::uint8_t& scale, std::uint8_t& min) noexcept {
    if (j < 4) {
        scale = std::uint8_t(q[j] & 63u);
        min = std::uint8_t(q[j + 4] & 63u);
    } else {
        scale = std::uint8_t((q[j + 4] & 0x0fu) | ((q[j - 4] >> 6) << 4));
        min = std::uint8_t((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

// Convert 8 fp16 values to 8 fp32 values. The vector fast path is the
// integer bit rewrite for normals (sign | (exp+112)<<23 | mant<<13); lanes
// with exp==0 (zero/subnormal) or exp==31 (inf/nan) are patched with the
// exact scalar converter, so the result is bit-identical to
// half_to_float_local for every input.
inline void f16x8_to_f32(const std::uint8_t* src, float* dst) {
    const __m128i hv = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
    const __m256i h = _mm256_cvtepu16_epi32(hv);
    const __m256i sign = _mm256_slli_epi32(_mm256_and_si256(h, _mm256_set1_epi32(0x8000)), 16);
    const __m256i expv = _mm256_and_si256(_mm256_srli_epi32(h, 10), _mm256_set1_epi32(0x1F));
    const __m256i mant = _mm256_and_si256(h, _mm256_set1_epi32(0x3FF));
    const __m256i bits = _mm256_or_si256(sign, _mm256_or_si256(
        _mm256_slli_epi32(_mm256_add_epi32(expv, _mm256_set1_epi32(112)), 23),
        _mm256_slli_epi32(mant, 13)));
    const __m256i is_special = _mm256_or_si256(
        _mm256_cmpeq_epi32(expv, _mm256_setzero_si256()),
        _mm256_cmpeq_epi32(expv, _mm256_set1_epi32(31)));
    if (_mm256_movemask_epi8(is_special) == 0) {
        _mm256_storeu_ps(dst, _mm256_castsi256_ps(bits));
        return;
    }
    alignas(32) float lanes[8];
    alignas(32) std::uint32_t hs[8];
    alignas(32) std::uint32_t ex[8];
    _mm256_store_ps(lanes, _mm256_castsi256_ps(bits));
    _mm256_store_si256(reinterpret_cast<__m256i*>(hs), h);
    _mm256_store_si256(reinterpret_cast<__m256i*>(ex), expv);
    for (unsigned i = 0; i < 8; ++i)
        if (ex[i] == 0 || ex[i] == 31)
            lanes[i] = half_to_float_local(static_cast<std::uint16_t>(hs[i]));
    _mm256_storeu_ps(dst, _mm256_load_ps(lanes));
}

inline void dequant_f16_panel(const std::uint8_t* wrow, std::size_t k0,
                              std::size_t kc, float* dst) {
    const std::uint8_t* srcp = wrow + k0 * 2;
    std::size_t i = 0;
    for (; i + 8 <= kc; i += 8) f16x8_to_f32(srcp + 2 * i, dst + i);
    for (; i < kc; ++i) dst[i] = half_to_float_local(read16_local(srcp + 2 * i));
}

} // namespace

bool gemv_f16_avx2(const std::uint8_t* weights, const float* x, float* y,
                   std::size_t rows, std::size_t columns) {
    if (!weights || !x || !y || !rows || !columns) return false;
    for (std::size_t r = 0; r < rows; ++r) {
        const auto* row = weights + r * columns * 2;
        __m256 sums[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(),
                          _mm256_setzero_ps(), _mm256_setzero_ps()};
        std::size_t c = 0;
        alignas(32) float values[8];
        for (; c + 8 <= columns; c += 8) {
            f16x8_to_f32(row + 2 * c, values);
            const auto w = _mm256_load_ps(values);
            sums[(c / 8) % 4] = _mm256_fmadd_ps(w, _mm256_loadu_ps(x + c), sums[(c / 8) % 4]);
        }
        float result = hsum_ps256(_mm256_add_ps(_mm256_add_ps(sums[0], sums[1]),
                                                 _mm256_add_ps(sums[2], sums[3])));
        for (; c < columns; ++c) result += half_to_float_local(read16_local(row + 2 * c)) * x[c];
        y[r] = result;
    }
    return true;
}

namespace {
// Dequantise one 64-element Q4_K sub-block (32 packed bytes) into 64 floats
// with two scale/min pairs. Low nibbles fill dst[0..31], high nibbles
// dst[32..63]. Uses mul+sub (not FMA) so each weight is bit-identical to
// the scalar dequantize() branch in quant.cpp.
inline void dequant_q4k_subblock(const std::uint8_t* q,
                                 float d1, float m1, float d2, float m2,
                                 float* dst) {
    const __m256 dv1 = _mm256_set1_ps(d1);
    const __m256 dv2 = _mm256_set1_ps(d2);
    const __m256 mv1 = _mm256_set1_ps(m1);
    const __m256 mv2 = _mm256_set1_ps(m2);
    const __m256i mask = _mm256_set1_epi32(0xF);
    for (unsigned chunk = 0; chunk < 4; ++chunk) {
        const __m128i b8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q + chunk * 8));
        const __m256i qu = _mm256_cvtepu8_epi32(b8);
        const __m256 qf = _mm256_cvtepi32_ps(_mm256_and_si256(qu, mask));
        const __m256 qh = _mm256_cvtepi32_ps(_mm256_srli_epi32(qu, 4));
        _mm256_storeu_ps(dst + chunk * 8, _mm256_sub_ps(_mm256_mul_ps(qf, dv1), mv1));
        _mm256_storeu_ps(dst + 32 + chunk * 8, _mm256_sub_ps(_mm256_mul_ps(qh, dv2), mv2));
    }
}

inline void dequant_q4k_panel(const std::uint8_t* wrow, std::size_t k0,
                              std::size_t kc, float* dst) {
    std::size_t done = 0;
    const std::size_t nsub = kc / 64;
    for (std::size_t s = 0; s < nsub; ++s) {
        const std::size_t col = k0 + s * 64;
        const std::uint8_t* block = wrow + (col / 256) * 144;
        const float d = half_to_float_local(read16_local(block));
        const float dmin = half_to_float_local(read16_local(block + 2));
        const std::uint8_t* scales = block + 4;
        const unsigned within = static_cast<unsigned>((col % 256) / 64);
        const unsigned is = within * 2;
        const std::uint8_t* q = block + 16 + within * 32;
        std::uint8_t sc{}, m{};
        k_scale_min_local(is + 0, scales, sc, m);
        const float d1 = d * float(sc), m1 = dmin * float(m);
        k_scale_min_local(is + 1, scales, sc, m);
        const float d2 = d * float(sc), m2 = dmin * float(m);
        dequant_q4k_subblock(q, d1, m1, d2, m2, dst + s * 64);
        done += 64;
    }
    // Defensive scalar tail (unreachable when columns%256==0 and Kc is a
    // divisor of 256, but kept exact for odd shapes).
    for (std::size_t i = done; i < kc; ++i) {
        const std::size_t col = k0 + i;
        const std::uint8_t* block = wrow + (col / 256) * 144;
        const float d = half_to_float_local(read16_local(block));
        const float dmin = half_to_float_local(read16_local(block + 2));
        const std::uint8_t* scales = block + 4;
        const unsigned within = static_cast<unsigned>((col % 256) / 64);
        const unsigned is = within * 2, e = static_cast<unsigned>(col % 64);
        std::uint8_t sc{}, m{};
        k_scale_min_local(is + (e >= 32 ? 1 : 0), scales, sc, m);
        const float ds = d * float(sc), ms = dmin * float(m);
        const std::uint8_t byte = block[16 + within * 32 + (e % 32)];
        const float qf = float(e < 32 ? (byte & 15u) : (byte >> 4));
        dst[i] = ds * qf - ms;
    }
}

// Q5_K uses the Q4_K nibbles plus a shared high-bit plane.
inline void dequant_q5k_panel(const std::uint8_t* wrow, std::size_t k0,
                              std::size_t kc, float* dst) {
    for (std::size_t i = 0; i < kc; i += 8) {
        const std::size_t col = k0 + i;
        const auto* p = wrow + (col / 256) * 176;
        const unsigned group = unsigned((col % 256) / 32);
        const unsigned lane = unsigned(col % 32);
        std::uint8_t sc{}, mn{};
        k_scale_min_local(group, p + 4, sc, mn);
        const float d = half_to_float_local(read16_local(p)) * float(sc);
        const float m = half_to_float_local(read16_local(p + 2)) * float(mn);
        std::uint64_t low, high;
        std::memcpy(&low, p + 48 + (group / 2) * 32 + lane, 8);
        std::memcpy(&high, p + 16 + lane, 8);
        __m256i q = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(static_cast<long long>(low)));
        if (group & 1) q = _mm256_srli_epi32(q, 4);
        q = _mm256_and_si256(q, _mm256_set1_epi32(15));
        __m256i h = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(static_cast<long long>(high)));
        h = _mm256_srlv_epi32(h, _mm256_set1_epi32(group));
        q = _mm256_or_si256(q, _mm256_slli_epi32(_mm256_and_si256(h, _mm256_set1_epi32(1)), 4));
        _mm256_storeu_ps(dst + i, _mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(d),
                           _mm256_cvtepi32_ps(q)), _mm256_set1_ps(m)));
    }
}

inline void dequant_q3k_panel(const std::uint8_t* wrow, std::size_t k0,
                               std::size_t kc, float* dst) {
    // Both the Kc=256 panel and tail are super-block aligned.
    for(std::size_t i=0;i<kc;i+=256){
        const auto* p=wrow+((k0+i)/256)*110;std::uint8_t scales[16];q3k_scales(p,scales);
        const float d=half_to_float_local(read16_local(p+108));
        for(unsigned h=0;h<2;++h)for(unsigned g=0;g<4;++g)for(unsigned l=0;l<32;l+=8)
            _mm256_storeu_ps(dst+i+h*128+g*32+l,q3k_decode8(p,scales,h,g,l,d));
    }
}

inline void dequant_tq2_panel(const std::uint8_t* wrow, std::size_t k0,
                               std::size_t kc, float* dst) {
    for(std::size_t i=0;i<kc;i+=8){
        const std::size_t col=k0+i;
        const auto* p=wrow+(col/256)*66;
        const float d=half_to_float_local(read16_local(p+64));
        const unsigned in=unsigned(col%256);
        _mm256_storeu_ps(dst+i,tq2_decode8(p,in/128,(in%128)/32,in%32,d));
    }
}

template<NewQuantKind Kind>
void dequant_new_quant_panel(const std::uint8_t* wrow,std::size_t k0,std::size_t kc,float* dst){
    const auto block=new_quant_elements(Kind), bytes=new_quant_bytes(Kind);
    for(std::size_t i=0;i<kc;i+=8){
        const auto col=k0+i;
        _mm256_storeu_ps(dst+i,new_quant_decode8(wrow+(col/block)*bytes,unsigned(col%block),Kind));
    }
}

using DequantPanelFn = void (*)(const std::uint8_t*, std::size_t, std::size_t, float*);

bool gemm_kblk_impl(const std::uint8_t* weights, const float* input, float* output,
                   std::size_t tokens, std::size_t rows, std::size_t columns,
                   std::size_t output_row_stride, std::size_t row_bytes,
                   std::size_t kKc, DequantPanelFn dequant_panel) {
    constexpr std::size_t kMR = 4, kNR = 2;
    if (!weights || !input || !output || !tokens || !rows || !columns) return false;
    const std::size_t rows_main = (rows / kMR) * kMR;
    std::vector<VectorAccumulator> partial(std::size_t(tokens) * kMR);
    std::vector<float> wpanel(std::size_t(kMR) * kKc);
    alignas(32) float lane[8];
    for (std::size_t r0 = 0; r0 < rows_main; r0 += kMR) {
        for (auto& p : partial) p.value = _mm256_setzero_ps();
        for (std::size_t k0 = 0; k0 < columns; k0 += kKc) {
            const std::size_t kc = std::size_t((kKc < columns - k0) ? kKc : (columns - k0));
            for (std::size_t dr = 0; dr < kMR; ++dr)
                dequant_panel(weights + (r0 + dr) * row_bytes, k0, kc, wpanel.data() + dr * kKc);
            for (std::size_t t = 0; t < tokens; t += kNR) {
                const float* in0 = input + t * columns + k0;
                VectorAccumulator* base = &partial[t * kMR];
                if (t + 1 < tokens) {
                    const float* in1 = input + (t + 1) * columns + k0;
                    VectorAccumulator* base1 = &partial[(t + 1) * kMR];
                    __m256 b0t = base[0].value, b1t = base[1].value, b2t = base[2].value, b3t = base[3].value;
                    __m256 b0u = base1[0].value, b1u = base1[1].value, b2u = base1[2].value, b3u = base1[3].value;
                    const float* w = wpanel.data();
                    for (std::size_t s = 0; s < kc; s += 8) {
                        const __m256 x0 = _mm256_loadu_ps(in0 + s);
                        const __m256 x1 = _mm256_loadu_ps(in1 + s);
                        const __m256 v0 = _mm256_loadu_ps(w + 0 * kKc + s);
                        const __m256 v1 = _mm256_loadu_ps(w + 1 * kKc + s);
                        const __m256 v2 = _mm256_loadu_ps(w + 2 * kKc + s);
                        const __m256 v3 = _mm256_loadu_ps(w + 3 * kKc + s);
                        b0t = _mm256_fmadd_ps(v0, x0, b0t); b0u = _mm256_fmadd_ps(v0, x1, b0u);
                        b1t = _mm256_fmadd_ps(v1, x0, b1t); b1u = _mm256_fmadd_ps(v1, x1, b1u);
                        b2t = _mm256_fmadd_ps(v2, x0, b2t); b2u = _mm256_fmadd_ps(v2, x1, b2u);
                        b3t = _mm256_fmadd_ps(v3, x0, b3t); b3u = _mm256_fmadd_ps(v3, x1, b3u);
                    }
                    base[0].value = b0t; base[1].value = b1t; base[2].value = b2t; base[3].value = b3t;
                    base1[0].value = b0u; base1[1].value = b1u; base1[2].value = b2u; base1[3].value = b3u;
                } else {
                    __m256 b0 = base[0].value, b1 = base[1].value, b2 = base[2].value, b3 = base[3].value;
                    const float* w = wpanel.data();
                    for (std::size_t s = 0; s < kc; s += 8) {
                        const __m256 x0 = _mm256_loadu_ps(in0 + s);
                        const __m256 v0 = _mm256_loadu_ps(w + 0 * kKc + s);
                        const __m256 v1 = _mm256_loadu_ps(w + 1 * kKc + s);
                        const __m256 v2 = _mm256_loadu_ps(w + 2 * kKc + s);
                        const __m256 v3 = _mm256_loadu_ps(w + 3 * kKc + s);
                        b0 = _mm256_fmadd_ps(v0, x0, b0);
                        b1 = _mm256_fmadd_ps(v1, x0, b1);
                        b2 = _mm256_fmadd_ps(v2, x0, b2);
                        b3 = _mm256_fmadd_ps(v3, x0, b3);
                    }
                    base[0].value = b0; base[1].value = b1; base[2].value = b2; base[3].value = b3;
                }
            }
        }
        for (std::size_t t = 0; t < tokens; ++t) {
            const VectorAccumulator* prow = &partial[t * kMR];
            for (std::size_t dr = 0; dr < kMR; ++dr) {
                _mm256_store_ps(lane, prow[dr].value);
                output[t * output_row_stride + r0 + dr] =
                    lane[0] + lane[1] + lane[2] + lane[3] + lane[4] + lane[5] + lane[6] + lane[7];
            }
        }
    }
    if (rows > rows_main) {
        std::vector<float> brow(columns);
        for (std::size_t r = rows_main; r < rows; ++r) {
            dequant_panel(weights + r * row_bytes, 0, columns, brow.data());
            for (std::size_t t = 0; t < tokens; ++t)
                output[t * output_row_stride + r] = dot_f32_avx2(brow.data(), input + t * columns, columns);
        }
    }
    return true;
}
}  // namespace

template<NewQuantKind Kind>
bool gemm_new_quant_kblk(const std::uint8_t* w,const float* x,float* y,std::size_t tokens,std::size_t rows,std::size_t cols,std::size_t stride){
    const auto block=new_quant_elements(Kind);
    if(!w||!x||!y||!tokens||!rows||!cols||cols%block||(Kind==NewQuantKind::f8_e4m3 && cols%8))return false;
    return gemm_kblk_impl(w,x,y,tokens,rows,cols,stride,(cols/block)*new_quant_bytes(Kind),256,&dequant_new_quant_panel<Kind>);
}
bool gemm_q1_0_kblk_avx2(const std::uint8_t* w,const float* x,float* y,std::size_t t,std::size_t r,std::size_t c,std::size_t s){return gemm_new_quant_kblk<NewQuantKind::q1_0>(w,x,y,t,r,c,s);}
bool gemm_q2_0_kblk_avx2(const std::uint8_t* w,const float* x,float* y,std::size_t t,std::size_t r,std::size_t c,std::size_t s){return gemm_new_quant_kblk<NewQuantKind::q2_0>(w,x,y,t,r,c,s);}
bool gemm_q8_1_kblk_avx2(const std::uint8_t* w,const float* x,float* y,std::size_t t,std::size_t r,std::size_t c,std::size_t s){return gemm_new_quant_kblk<NewQuantKind::q8_1>(w,x,y,t,r,c,s);}
bool gemm_q8_1_f32_kblk_avx2(const std::uint8_t* w,const float* x,float* y,std::size_t t,std::size_t r,std::size_t c,std::size_t s){return gemm_new_quant_kblk<NewQuantKind::q8_1_f32>(w,x,y,t,r,c,s);}
bool gemm_nvfp4_kblk_avx2(const std::uint8_t* w,const float* x,float* y,std::size_t t,std::size_t r,std::size_t c,std::size_t s){return gemm_new_quant_kblk<NewQuantKind::nvfp4>(w,x,y,t,r,c,s);}
bool gemm_f8_e4m3_kblk_avx2(const std::uint8_t* w,const float* x,float* y,std::size_t t,std::size_t r,std::size_t c,std::size_t s){return gemm_new_quant_kblk<NewQuantKind::f8_e4m3>(w,x,y,t,r,c,s);}

bool gemm_f16_kblk_avx2(const std::uint8_t* weights, const float* input, float* output,
                        std::size_t tokens, std::size_t rows, std::size_t columns,
                        std::size_t output_row_stride) {
    if (!weights || !input || !output || !tokens || !rows || !columns) return false;
    if (columns % 32) return false;
    static const std::size_t kKc = []() {
        const char* e = std::getenv("ZEN_CPU_F16_KC");
        if (e) {
            const long v = std::atol(e);
            if (v == 64 || v == 128 || v == 256) return static_cast<std::size_t>(v);
        }
        return static_cast<std::size_t>(128);
    }();
    return gemm_kblk_impl(weights, input, output, tokens, rows, columns,
                         output_row_stride, columns * 2, kKc, &dequant_f16_panel);
}

bool gemm_q3k_kblk_avx2(const std::uint8_t* weights,const float* input,float* output,
                        std::size_t tokens,std::size_t rows,std::size_t columns,
                        std::size_t output_row_stride){
    if(!weights||!input||!output||!tokens||!rows||!columns||columns%256)return false;
    return gemm_kblk_impl(weights,input,output,tokens,rows,columns,output_row_stride,
                          (columns/256)*110,256,&dequant_q3k_panel);
}

bool gemm_tq2_kblk_avx2(const std::uint8_t* weights, const float* input, float* output,
                        std::size_t tokens, std::size_t rows, std::size_t columns,
                        std::size_t output_row_stride) {
    if(!weights||!input||!output||!tokens||!rows||!columns||columns%256)return false;
    return gemm_kblk_impl(weights,input,output,tokens,rows,columns,output_row_stride,
                          (columns/256)*66,256,&dequant_tq2_panel);
}

bool gemm_q5k_kblk_avx2(const std::uint8_t* weights, const float* input, float* output,
                         std::size_t tokens, std::size_t rows, std::size_t columns,
                         std::size_t output_row_stride) {
    if (!weights || !input || !output || !tokens || !rows || !columns || columns % 256) return false;
    return gemm_kblk_impl(weights, input, output, tokens, rows, columns,
                          output_row_stride, (columns / 256) * 176, 256, &dequant_q5k_panel);
}

bool gemm_q4k_kblk_avx2(const std::uint8_t* weights, const float* input, float* output,
                        std::size_t tokens, std::size_t rows, std::size_t columns,
                        std::size_t output_row_stride) {
    if (!weights || !input || !output || !tokens || !rows || !columns) return false;
    if (columns % 256) return false;
    static const std::size_t kKc = []() {
        const char* e = std::getenv("ZEN_CPU_Q4K_KC");
        if (e) {
            const long v = std::atol(e);
            if (v == 64 || v == 128 || v == 256) return static_cast<std::size_t>(v);
        }
        return static_cast<std::size_t>(128);
    }();
    return gemm_kblk_impl(weights, input, output, tokens, rows, columns,
                         output_row_stride, (columns / 256) * 144, kKc, &dequant_q4k_panel);
}

// GDN (gated delta net) recurrence AVX2 helpers. State layout row-major S x S:
// sh[i*S + j]. All operate in-place where noted.
void gdn_decay_avx2(float* sh, float g, std::size_t n) {
    const __m256 gv = _mm256_set1_ps(g);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(sh + i);
        v = _mm256_mul_ps(v, gv);
        _mm256_storeu_ps(sh + i, v);
    }
    for (; i < n; ++i) sh[i] *= g;
}

void gdn_delta_avx2(const float* sh, const float* kh, const float* vh,
                    float* delta, float beta, std::size_t S) {
    for (std::size_t j = 0; j < S; ++j) {
        __m256 acc = _mm256_setzero_ps();
        std::size_t i = 0;
        for (; i + 8 <= S; i += 8) {
            __m256 v = _mm256_set_ps(
                sh[(i+7)*S + j], sh[(i+6)*S + j], sh[(i+5)*S + j], sh[(i+4)*S + j],
                sh[(i+3)*S + j], sh[(i+2)*S + j], sh[(i+1)*S + j], sh[(i+0)*S + j]);
            __m256 k = _mm256_loadu_ps(kh + i);
            acc = _mm256_fmadd_ps(v, k, acc);
        }
        float sum = hsum_ps256(acc);
        for (; i < S; ++i) sum += sh[i*S + j] * kh[i];
        delta[j] = (vh[j] - sum) * beta;
    }
}

void gdn_update_avx2(float* sh, const float* kh, const float* delta, std::size_t S) {
    for (std::size_t i = 0; i < S; ++i) {
        const __m256 kv = _mm256_set1_ps(kh[i]);
        std::size_t j = 0;
        for (; j + 8 <= S; j += 8) {
            __m256 v = _mm256_loadu_ps(sh + i*S + j);
            __m256 d = _mm256_loadu_ps(delta + j);
            v = _mm256_fmadd_ps(kv, d, v);
            _mm256_storeu_ps(sh + i*S + j, v);
        }
        for (; j < S; ++j) sh[i*S + j] += kh[i] * delta[j];
    }
}

void gdn_output_avx2(const float* sh, const float* qh, float* oh,
                     float scale, std::size_t S) {
    for (std::size_t j = 0; j < S; ++j) {
        __m256 acc = _mm256_setzero_ps();
        std::size_t i = 0;
        for (; i + 8 <= S; i += 8) {
            __m256 v = _mm256_set_ps(
                sh[(i+7)*S + j], sh[(i+6)*S + j], sh[(i+5)*S + j], sh[(i+4)*S + j],
                sh[(i+3)*S + j], sh[(i+2)*S + j], sh[(i+1)*S + j], sh[(i+0)*S + j]);
            __m256 q = _mm256_loadu_ps(qh + i);
            acc = _mm256_fmadd_ps(v, q, acc);
        }
        float sum = hsum_ps256(acc);
        for (; i < S; ++i) sum += sh[i*S + j] * qh[i];
        oh[j] = sum * scale;
    }
}



// BF16 dequant panel: bf16 (2 bytes) -> float. Conversion is just a left shift
// by 16 (bf16 is the upper 16 bits of float32). No scale, no block.
inline void dequant_bf16_panel(const std::uint8_t* wrow, std::size_t k0,
                               std::size_t kc, float* dst) {
    const std::uint16_t* srcp = reinterpret_cast<const std::uint16_t*>(wrow + k0 * 2);
    std::size_t i = 0;
    for (; i + 8 <= kc; i += 8) {
        __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + i));
        __m256i u = _mm256_cvtepu16_epi32(h);
        u = _mm256_slli_epi32(u, 16);
        __m256 f = _mm256_castsi256_ps(u);
        _mm256_storeu_ps(dst + i, f);
    }
    for (; i < kc; ++i) {
        std::uint32_t u = (std::uint32_t)srcp[i] << 16;
        std::memcpy(dst + i, &u, sizeof(float));
    }
}

bool gemm_bf16_kblk_avx2(const std::uint8_t* weights, const float* input, float* output,
                          std::size_t tokens, std::size_t rows, std::size_t columns,
                          std::size_t output_row_stride) {
    if (!weights || !input || !output || !tokens || !rows || !columns) return false;
    if (columns % 32) return false;
    static const std::size_t kKc = 128;
    return gemm_kblk_impl(weights, input, output, tokens, rows, columns,
                         output_row_stride, columns * 2, kKc, &dequant_bf16_panel);
}


// Q6_K dequant panel: 256-weight super-blocks. Each super-block has
// ql[128] (low 4 bits), qh[64] (high 2 bits), scales[16] (signed int8),
// d (fp16 overall scale). Dequant: q = (ql_nibble | (qh_bits<<4)) - 32,
// weight = d * scales[sub] * q. Panel dequantizes kc floats starting at k0.
inline void dequant_q6k_panel(const std::uint8_t* wrow, std::size_t k0,
                              std::size_t kc, float* dst) {
    std::size_t off = 0;
    while (off < kc) {
        const std::size_t sb = (k0 + off) / 256;
        const std::size_t in_sb = (k0 + off) % 256;
        const std::uint8_t* p = wrow + sb * 210;
        const auto* scales = reinterpret_cast<const std::int8_t*>(p + 192);
        const float d = half_to_float_local(read16_local(p + 208));
        const std::size_t count = std::min(kc - off, 256 - in_sb);
        std::size_t i = 0;
        for (; i + 8 <= count && (in_sb + i) % 8 == 0; i += 8) {
            const std::size_t idx = in_sb + i;
            const __m256 v = q6k_decode8(p, unsigned(idx / 128), unsigned((idx % 128) / 32),
                                          unsigned(idx % 32), d);
            _mm256_storeu_ps(dst + off + i, v);
        }
        for (; i < count; ++i) {
            const std::size_t idx = in_sb + i;
            const std::size_t half = idx / 128;
            const std::size_t group = (idx % 128) / 32;
            const std::size_t lane = idx % 32;
            const std::uint8_t low = p[half * 64 + (group % 2) * 32 + lane];
            const std::uint8_t high = p[128 + half * 32 + lane];
            const unsigned nibble = group < 2 ? low & 15u : low >> 4;
            const int q = int(nibble | (((high >> (group * 2)) & 3u) << 4)) - 32;
            const std::size_t scale_idx = half * 8 + group * 2 + lane / 16;
            dst[off + i] = d * scales[scale_idx] * q;
        }
        off += count;
    }
}

bool gemm_q6k_kblk_avx2(const std::uint8_t* weights, const float* input, float* output,
                        std::size_t tokens, std::size_t rows, std::size_t columns,
                        std::size_t output_row_stride) {
    if (!weights || !input || !output || !tokens || !rows || !columns) return false;
    if (columns % 256) return false;
    static const std::size_t kKc = 256;
    return gemm_kblk_impl(weights, input, output, tokens, rows, columns,
                         output_row_stride, (columns / 256) * 210, kKc, &dequant_q6k_panel);
}

}  // namespace zen
