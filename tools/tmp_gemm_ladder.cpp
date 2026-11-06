// TEMP harness (not shipped): T-ladder for quant::gemm Q8_0 AVX2 vs double reference.
// Output layout per zen/quant.h: output[tokens, rows], output[t*rows+r].
#include "zen/quant.h"
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>
using namespace zen;
int main(){
    const auto* fmt = quant::find("Q8_0");
    if(!fmt){ std::printf("no Q8_0\n"); return 2; }
    struct Case{ size_t rows, cols; std::vector<size_t> Ts; };
    std::vector<Case> cases = {
        {100, 544, {1,5,7,8,9,12,16,17,171}},
        {101, 544, {8,12,171}},          // odd rows: exercises odd-row tail in kernel
        {64, 2048, {5,8,12,171}},        // evidence shape width
        {33,  96, {8,12,17}},            // small cols (3 blocks)
    };
    int failures=0;
    for(auto& cs: cases){
        size_t rows=cs.rows, cols=cs.cols;
        size_t rb = quant::row_bytes(*fmt, cols);
        std::vector<std::uint8_t> packed(rows*rb);
        // deterministic pseudo-random Q8_0: scale half = 1.0 (0x3c00 LE), int8 payload pattern
        uint32_t s=12345;
        auto rnd=[&](){ s=s*1664525u+1013904223u; return (s>>24)&0xff; };
        for(size_t r=0;r<rows;++r) for(size_t blk=0; blk<cols/32; ++blk){
            std::uint8_t* b=packed.data()+r*rb+blk*34; b[0]=0x00; b[1]=0x3c;
            for(size_t j=2;j<34;++j) b[j]=(std::uint8_t)rnd();
        }
        std::vector<float> ref(rows*cols);
        if(!quant::dequantize(*fmt, packed.data(), rows*cols, ref.data())){ std::printf("dequant fail\n"); return 2; }
        for(size_t T: cs.Ts){
            std::vector<float> xin(T*cols);
            for(size_t t=0;t<T;++t) for(size_t c=0;c<cols;++c)
                xin[t*cols+c]=float(int((t*7+c)%15)-7)*0.017f + float(int((t*13+c*3)%7)-3)*0.001f;
            for(unsigned workers : {1u,4u}){
                std::vector<float> out(T*rows, 0.0f);
                bool ok = quant::gemm(*fmt, packed.data(), xin.data(), out.data(), T, rows, cols, 1.0f, workers);
                double maxerr=0; size_t worst_t=0, worst_r=0;
                for(size_t t=0;t<T;++t) for(size_t r=0;r<rows;++r){
                    double expect=0; for(size_t c=0;c<cols;++c) expect+=double(ref[r*cols+c])*xin[t*cols+c];
                    double e=std::abs(double(out[t*rows+r])-expect);
                    if(e>maxerr){maxerr=e; worst_t=t; worst_r=r;}
                }
                bool pass = ok && maxerr < 5e-3; // ~4e-4 expected; 5e-3 generous for cols=2048 accumulation
                if(!pass) ++failures;
                std::printf("rows=%zu cols=%zu T=%zu workers=%u ok=%d maxerr=%.6g worst(t=%zu,r=%zu) %s\n",
                    rows, cols, T, workers, (int)ok, maxerr, worst_t, worst_r, pass?"PASS":"FAIL");
            }
        }
    }
    std::printf(failures?"RESULT: FAIL (%d)\n":"RESULT: PASS\n", failures);
    return failures?1:0;
}
