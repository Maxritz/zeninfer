// Synthetic shape sweep: compare the same build and CPU; not real-model tok/s.
#include "zen/quant.h"
#include "zen/kernels.h"
#include "zen/diagnostics.h"
#include "../tests/support/quant_fixture.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>
#include <vector>
int main(int argc,char**argv){
    const std::size_t rows=argc>1?std::strtoull(argv[1],nullptr,10):256;
    const std::size_t cols=argc>2?std::strtoull(argv[2],nullptr,10):1024;
    const std::size_t tokens=argc>3?std::strtoull(argv[3],nullptr,10):8;
    const unsigned workers=argc>4?static_cast<unsigned>(std::strtoul(argv[4],nullptr,10)):1;
    const std::string filter=argc>5?argv[5]:"";
    if(!rows||!cols||!tokens||!workers){std::fprintf(stderr,"usage: [rows cols tokens workers [format]]\n");return 2;}
    std::fprintf(stderr,"rows=%zu cols=%zu tokens=%zu workers=%u\n",rows,cols,tokens,workers);
    std::vector<float>x(cols*tokens),y(rows*tokens);
    for(std::size_t i=0;i<x.size();++i)x[i]=float(int((i*37+11)%101)-50)/64.f;
    std::puts("format,gemv_us,gemm_us");
    int measured=0;
    for(const auto& f:zen::quant::registry())if(f.implemented && (filter.empty()||filter==f.name)){
        ++measured;
        if(!zen::quant::row_bytes(f,cols)){std::fprintf(stderr,"invalid columns for %s\n",f.name.c_str());return 2;}
        auto w=quant_fixture(f,rows,cols);
        auto measure=[&](bool batched){
            std::vector<double> ms;for(int i=0;i<6;++i){auto a=std::chrono::steady_clock::now();
                bool ok=batched?zen::quant::gemm(f,w.data(),x.data(),y.data(),tokens,rows,cols,1,workers):zen::quant::gemv(f,w.data(),x.data(),y.data(),rows,cols,1,workers);
                auto b=std::chrono::steady_clock::now();if(!ok)return -1.;if(i)ms.push_back(std::chrono::duration<double,std::micro>(b-a).count());
            }std::sort(ms.begin(),ms.end());return ms[ms.size()/2];};
        auto a=measure(false),b=measure(true);std::printf("%s,%.3f,%.3f\n",f.name.c_str(),a,b);if(a<0||b<0)return 1;
        const char* trace=std::getenv("ZEN_CPU_BENCH_TRACE");
        if(trace&&trace[0]=='1'){
            zen::diag::set_trace_enabled(true);zen::diag::reset_trace();
            if(!zen::quant::gemv(f,w.data(),x.data(),y.data(),rows,cols,1,workers))return 1;
            std::cerr<<f.name<<" GEMV Sherlock trace\n";zen::diag::print_trace(std::cerr);
            zen::diag::reset_trace();
            if(!zen::quant::gemm(f,w.data(),x.data(),y.data(),tokens,rows,cols,1,workers))return 1;
            std::cerr<<f.name<<" GEMM Sherlock trace\n";zen::diag::print_trace(std::cerr);
            zen::diag::set_trace_enabled(false);
        }
    }
    if(!measured){std::fprintf(stderr,"unknown format: %s\n",filter.c_str());return 2;}
}
