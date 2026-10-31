#include "zen/kernels.h"
#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>

namespace zen {
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
bool gemv_q8_0_avx2(const std::uint8_t* weights,const float* x,float* y,
                    std::size_t rows,std::size_t columns) {
    if(!weights||!x||!y||!rows||!columns||columns%32) return false;
    const std::size_t blocks=columns/32, row_bytes=blocks*34;
    for(std::size_t r=0;r<rows;++r){
        const std::uint8_t* row=weights+r*row_bytes;
        double total=0.0;
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
            const __m256 sum=_mm256_add_ps(_mm256_add_ps(a0,a1),_mm256_add_ps(a2,a3));
            alignas(32) float lanes[8];_mm256_store_ps(lanes,sum);
            const float block_dot=(((lanes[0]+lanes[1])+lanes[2])+lanes[3])+(((lanes[4]+lanes[5])+lanes[6])+lanes[7]);
            total+=double(d)*block_dot;
        }
        y[r]=static_cast<float>(total);
    }
    return true;
}
}
