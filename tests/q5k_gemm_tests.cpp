#include "zen/kernels.h"
#include "zen/quant.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

// Independently construct GGML Q5_K blocks with known constant scale/min and
// independently retained quant values. Covers every high bit plane and nibble.
int main(){
    const auto& f=*zen::quant::find("Q5_K");
    std::uint32_t state=7;auto rnd=[&](){state=1664525u*state+1013904223u;return state;};
    int failures=0;
    for(std::size_t rows:{1u,3u,4u,5u,35u,67u})for(std::size_t cols:{256u,768u}){
        std::size_t rb=cols/256*176;std::vector<std::uint8_t>w(rows*rb);std::vector<float>known(rows*cols);
        for(std::size_t r=0;r<rows;++r)for(std::size_t b=0;b<cols/256;++b){
            auto* p=w.data()+r*rb+b*176;p[0]=0;p[1]=0x34;p[2]=0;p[3]=0x38;
            for(int i=0;i<8;++i)p[4+i]=1;for(int i=8;i<12;++i)p[4+i]=0x11;
            for(unsigned lane=0;lane<32;++lane){
                std::uint8_t high=0;
                for(unsigned group=0;group<8;++group){
                    unsigned q=rnd()>>27;known[r*cols+b*256+group*32+lane]=0.25f*q-0.5f;
                    high|=std::uint8_t((q>>4)<<group);
                    auto& low=p[48+(group/2)*32+lane];
                    if(group%2)low|=std::uint8_t((q&15)<<4);else low=std::uint8_t(q&15);
                }
                p[16+lane]=high;
            }
        }
        for(std::size_t tokens:{2u,3u,8u}){
            std::vector<float>x(tokens*cols);for(auto&v:x)v=float(int(rnd()>>24)-128)/128.f;
            std::vector<double>ref(tokens*rows),mag(tokens*rows);
            for(std::size_t t=0;t<tokens;++t)for(std::size_t r=0;r<rows;++r)for(std::size_t c=0;c<cols;++c){double v=double(known[r*cols+c])*x[t*cols+c];ref[t*rows+r]+=v;mag[t*rows+r]+=std::abs(v);}
            auto check=[&](const std::vector<float>& y,std::size_t stride){
                for(std::size_t t=0;t<tokens;++t)for(std::size_t r=0;r<rows;++r)if(!std::isfinite(y[t*stride+r])||std::abs(double(y[t*stride+r])-ref[t*rows+r])>2e-5*std::max(1.,mag[t*rows+r])){std::cerr<<"Q5_K mismatch rows="<<rows<<" cols="<<cols<<" tokens="<<tokens<<'\n';++failures;return;}
            };
            for(unsigned workers:{1u,4u}){std::vector<float>out(tokens*rows);if(!zen::quant::gemm(f,w.data(),x.data(),out.data(),tokens,rows,cols,1,workers))++failures;else check(out,rows);}
#ifdef ZEN_HAS_AVX2_KERNELS
            if(zen::cpu_has_avx2_fma()){
                auto stride=rows+3;std::vector<float>out(tokens*stride,12345.f);
                if(!zen::gemm_q5k_kblk_avx2(w.data(),x.data(),out.data(),tokens,rows,cols,stride))++failures;else check(out,stride);
                for(std::size_t t=0;t<tokens;++t)for(std::size_t i=rows;i<stride;++i)if(out[t*stride+i]!=12345.f)++failures;
            }
#endif
        }
    }
    std::cout<<"Q5_K GEMM failures="<<failures<<'\n';return failures?1:0;
}
