#include "zen/kernels.h"
#include "zen/quant.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>
int main(){
    auto& f=*zen::quant::find("TQ2_0");std::uint32_t state=71;
    auto rnd=[&](){state=1664525u*state+1013904223u;return state;};int failures=0;
    for(std::size_t rows:{1u,3u,4u,5u,35u,67u})for(std::size_t cols:{256u,768u}){
        auto rb=cols/256*66;std::vector<std::uint8_t>w(rows*rb);std::vector<float>known(rows*cols);
        for(std::size_t r=0;r<rows;++r)for(std::size_t b=0;b<cols/256;++b){
            auto*p=w.data()+r*rb+b*66;p[64]=0;p[65]=0x34;
            for(unsigned g=0;g<2;++g)for(unsigned lane=0;lane<32;++lane){
                std::uint8_t byte=0;for(unsigned plane=0;plane<4;++plane){unsigned q=rnd()>>30;
                    byte|=std::uint8_t(q<<(plane*2));known[r*cols+b*256+g*128+plane*32+lane]=(int(q)-1)*0.25f;}
                p[g*32+lane]=byte;
            }
        }
        for(std::size_t tokens:{1u,2u,3u,8u}){
            std::vector<float>x(tokens*cols);for(auto&v:x)v=float(int(rnd()>>24)-128)/128.f;
            std::vector<double>ref(tokens*rows),mag(tokens*rows);
            for(std::size_t t=0;t<tokens;++t)for(std::size_t r=0;r<rows;++r)for(std::size_t c=0;c<cols;++c){double v=double(known[r*cols+c])*x[t*cols+c];ref[t*rows+r]+=v;mag[t*rows+r]+=std::abs(v);}
            auto check=[&](const std::vector<float>& y,std::size_t stride){
                for(std::size_t t=0;t<tokens;++t)for(std::size_t r=0;r<rows;++r)if(!std::isfinite(y[t*stride+r])||std::abs(double(y[t*stride+r])-ref[t*rows+r])>2e-5*std::max(1.,mag[t*rows+r])){std::cerr<<"TQ2 mismatch rows="<<rows<<" cols="<<cols<<" tokens="<<tokens<<'\n';++failures;return;}
            };
            for(unsigned workers:{1u,4u}){std::vector<float>out(tokens*rows);
                bool ok=tokens==1?zen::quant::gemv(f,w.data(),x.data(),out.data(),rows,cols,1,workers):zen::quant::gemm(f,w.data(),x.data(),out.data(),tokens,rows,cols,1,workers);
                if(!ok)++failures;else check(out,rows);}
#ifdef ZEN_HAS_AVX2_KERNELS
            if(tokens>1&&zen::cpu_has_avx2_fma()){
                auto stride=rows+3;std::vector<float>out(tokens*stride,12345.f);
                if(!zen::gemm_tq2_kblk_avx2(w.data(),x.data(),out.data(),tokens,rows,cols,stride))++failures;else check(out,stride);
                for(std::size_t t=0;t<tokens;++t)for(std::size_t i=rows;i<stride;++i)if(out[t*stride+i]!=12345.f)++failures;
            }
#endif
        }
    }
    std::cout<<"TQ2_0 CPU failures="<<failures<<'\n';return failures?1:0;
}
