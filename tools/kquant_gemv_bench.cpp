// Synthetic single-thread decode microbenchmark; this is not model tok/s.
#include "zen/quant.h"
#include "zen/kernels.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
#ifndef ZEN_HAS_AVX2_KERNELS
    std::fprintf(stderr,"AVX2 kernels not built\n"); return 2;
#else
    if(!zen::cpu_has_avx2_fma()){std::fprintf(stderr,"AVX2/FMA unavailable\n");return 2;}
    constexpr std::size_t rows=1024, cols=4096;
    for(const char* name : {"Q4_K","Q5_K"}) {
        const auto& f=*zen::quant::find(name);
        const std::size_t rb=zen::quant::row_bytes(f,cols);
        std::vector<std::uint8_t> w(rows*rb);
        std::uint32_t seed=42;
        for(auto& v:w){seed=seed*1664525u+1013904223u;v=std::uint8_t(seed>>24);}
        for(std::size_t r=0;r<rows;++r)for(std::size_t b=0;b<cols/256;++b){
            auto* p=w.data()+r*rb+b*f.block_bytes;
            p[0]=0;p[1]=0x28;p[2]=0;p[3]=0x20;
        }
        std::vector<float> x(cols),old(rows),fast(rows),scratch(256);
        for(std::size_t i=0;i<cols;++i)x[i]=float(int(i%37)-18)*0.01f;
        auto baseline=[&]{
            for(std::size_t r=0;r<rows;++r){double sum=0;
                for(std::size_t b=0;b<cols/256;++b){
                    zen::quant::dequantize(f,w.data()+r*rb+b*f.block_bytes,256,scratch.data());
                    float part;zen::gemv_avx2(scratch.data(),x.data()+b*256,&part,1,256);sum+=part;
                }
                old[r]=float(sum);
            }
        };
        auto candidate=[&]{return zen::quant::gemv(f,w.data(),x.data(),fast.data(),rows,cols,1.0f,1);};
        baseline();if(!candidate())return 1;
        double max_error=0,max_ref=0;
        for(std::size_t r=0;r<rows;++r){max_error=std::max(max_error,std::abs(double(fast[r])-old[r]));max_ref=std::max(max_ref,std::abs(double(old[r])));}
        if(max_error>1e-4*std::max(1.0,max_ref)){std::fprintf(stderr,"%s parity failed\n",name);return 1;}
        auto measure=[](auto&& fn){std::vector<double> times;for(int i=0;i<7;++i){
            auto start=std::chrono::steady_clock::now();fn();auto end=std::chrono::steady_clock::now();
            times.push_back(std::chrono::duration<double,std::milli>(end-start).count());
        }std::sort(times.begin(),times.end());return times[times.size()/2];};
        const auto a=measure(baseline),b=measure(candidate);
        std::printf("%s rows=%zu cols=%zu workers=1 median_ms baseline=%.3f fused=%.3f speedup=%.2fx max_abs_error=%.6g\n",name,rows,cols,a,b,a/b,max_error);
    }
    return 0;
#endif
}
