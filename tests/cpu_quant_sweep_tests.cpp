#include "zen/quant.h"
#include "zen/kernels.h"
#include "support/quant_fixture.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    int failures=0, formats=0;
    for(const auto& f:zen::quant::registry()) {
        if(!f.implemented) continue;
        ++formats;
        for(std::size_t cols : {std::size_t(256),std::size_t(768)})
        for(std::size_t rows : {std::size_t(1),std::size_t(5),std::size_t(67)}) {
            const auto rb=zen::quant::row_bytes(f,cols);
            auto w=quant_fixture(f,rows,cols);
            std::vector<float> decoded(rows*cols), x(8*cols), out(8*rows);
            for(std::size_t i=0;i<x.size();++i)x[i]=float(int((i*37+11)%101)-50)/64.f;
            for(float scale : {1.0f,0.25f}) {
            bool valid=true;
            for(std::size_t r=0;r<rows;++r) {
                if(!zen::quant::dequantize(f,w.data()+r*rb,cols,decoded.data()+r*cols,scale))valid=false;
                for(std::size_t c=0;c<cols;++c)if(!std::isfinite(decoded[r*cols+c]))valid=false;
            }
            if(!valid){std::cerr<<"invalid fixture "<<f.name<<'\n';return 1;}
            // For signed packed formats independently extract one bit at a time.
            if(f.encoding==zen::quant::Encoding::signed_uniform)for(std::size_t r=0;r<rows;++r)
                for(std::size_t c=0;c<cols;++c){
                    std::uint32_t q=0;for(unsigned b=0;b<f.bits;++b){std::size_t bit=c*f.bits+b;q|=std::uint32_t((w[r*rb+bit/8]>>(bit%8))&1u)<<b;}
                    const int v=(q&(1u<<(f.bits-1)))?int(q)-int(1u<<f.bits):int(q);
                    if(decoded[r*cols+c]!=float(v)*scale){std::cerr<<"signed decode mismatch "<<f.name<<'\n';return 1;}
                }
            for(std::size_t tokens:{std::size_t(1),std::size_t(2),std::size_t(3),std::size_t(8)})
            for(unsigned workers:{1u,4u}) {
                bool ok=tokens==1?zen::quant::gemv(f,w.data(),x.data(),out.data(),rows,cols,scale,workers):
                    zen::quant::gemm(f,w.data(),x.data(),out.data(),tokens,rows,cols,scale,workers);
                if(!ok){std::cerr<<"dispatch failed "<<f.name<<'\n';++failures;continue;}
                for(std::size_t t=0;t<tokens;++t)for(std::size_t r=0;r<rows;++r){
                    double expected=0,mag=0;for(std::size_t c=0;c<cols;++c){double v=double(decoded[r*cols+c])*x[t*cols+c];expected+=v;mag+=std::abs(v);}
                    if(!std::isfinite(out[t*rows+r])||std::abs(double(out[t*rows+r])-expected)>2e-5*std::max(1.0,mag)){
                        std::cerr<<"mismatch "<<f.name<<" rows="<<rows<<" cols="<<cols<<" tokens="<<tokens<<" workers="<<workers<<" row="<<r<<"\n";++failures;goto next_case;
                    }
                }
                next_case:;
            }
            }
        }
    }
    // Partial rows and byte boundaries in the signed formats.
    for(const auto& f:zen::quant::registry())if(f.encoding==zen::quant::Encoding::signed_uniform)
        for(std::size_t n=1;n<=65;++n){
            auto w=quant_fixture(f,1,n);std::vector<float> got(n),x(n,0.5f);float dot;
            if(!zen::quant::dequantize(f,w.data(),n,got.data(),0.25f)||!zen::quant::gemv(f,w.data(),x.data(),&dot,1,n,0.25f,1)){++failures;continue;}
            double ref=0;for(std::size_t c=0;c<n;++c){std::uint32_t q=0;for(unsigned b=0;b<f.bits;++b){std::size_t bit=c*f.bits+b;q|=std::uint32_t((w[bit/8]>>(bit%8))&1u)<<b;}
                const int v=(q&(1u<<(f.bits-1)))?int(q)-int(1u<<f.bits):int(q);
                if(got[c]!=float(v)*0.25f)++failures;ref+=double(v)*0.125;}
            if(std::abs(dot-ref)>1e-4*std::max(1.0,std::abs(ref)))++failures;
        }
    // F16 special values and tails across vector boundaries.
    for(std::size_t n:{1u,7u,8u,9u,31u,33u}){
        const std::uint16_t hs[]={0,0x8000,1,0x8001,0x3c00,0xb800,0x7bff,0x0400,0x3555};
        std::vector<std::uint8_t>w(n*2);std::vector<float>x(n),d(n);for(std::size_t i=0;i<n;++i){auto h=hs[i%9];w[2*i]=std::uint8_t(h);w[2*i+1]=std::uint8_t(h>>8);x[i]=float(i%5+1)/10;}
        auto& f=*zen::quant::find("F16");float y;zen::quant::dequantize(f,w.data(),n,d.data());double ref=0,mag=0;for(std::size_t i=0;i<n;++i){ref+=double(d[i])*x[i];mag+=std::abs(double(d[i])*x[i]);}
        if(!zen::quant::gemv(f,w.data(),x.data(),&y,1,n,1,1)||std::abs(y-ref)>2e-5*std::max(1.0,mag))++failures;
    }
    // Each possible half is converted through a full SIMD vector, including
    // zeros, subnormals, infinities and NaNs. Use x=1 to isolate conversion.
    {
        auto& f=*zen::quant::find("F16");float x[8]={1,1,1,1,1,1,1,1};
        for(unsigned h=0;h<65536;++h){
            std::uint8_t w[16]{};w[0]=std::uint8_t(h);w[1]=std::uint8_t(h>>8);
            float d[8],y;
            if(!zen::quant::dequantize(f,w,8,d)||!zen::quant::gemv(f,w,x,&y,1,8,1,1)||
               (std::isnan(d[0])?!std::isnan(y):(y!=d[0]))) {std::cerr<<"F16 special mismatch "<<h<<'\n';++failures;break;}
        }
    }
    std::cout<<formats<<" formats checked; failures="<<failures<<'\n';return failures?1:0;
}
