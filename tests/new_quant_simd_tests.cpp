#include "zen/quant.h"
#include "zen/kernels.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

int main(){
    int failures=0;
    for(const char* name:{"Q1_0","Q2_0","Q8_1","Q8_1_F32","NVFP4","F8_E4M3"}){
        const auto* f=zen::quant::find(name);if(!f)return 2;
        using E=zen::quant::Encoding;
        for(std::size_t cols:{std::size_t(f->block_elements),std::size_t(f->block_elements)*3,std::size_t(256),std::size_t(256)+f->block_elements})
        for(std::size_t rows:{1u,3u,5u,67u}){
            const auto rb=zen::quant::row_bytes(*f,cols);const auto nblocks=cols/f->block_elements;
            std::vector<std::uint8_t> storage(rows*rb+1), &w=storage; // deliberately unaligned
            std::vector<float> ref(rows*cols),x(8*cols),out(8*rows);
            for(std::size_t i=0;i<x.size();++i)x[i]=float(int(i%29)-14)/16.f;
            for(std::size_t r=0;r<rows;++r)for(std::size_t b=0;b<nblocks;++b){
                auto* p=w.data()+1+r*rb+b*f->block_bytes;
                const float d=(b+r)%3==0?0.5f:(b+r)%3==1?-0.25f:1.f;
                if(f->encoding==E::fp8_e4m3){}
                else if(f->encoding==E::nvfp4){for(unsigned g=0;g<4;++g)p[g]=std::uint8_t((r+b+g)*37u);}
                else if(f->encoding==E::q8_1_f32){std::uint32_t bits;std::memcpy(&bits,&d,4);for(int k=0;k<4;++k)p[k]=std::uint8_t(bits>>(8*k));}
                else {std::uint16_t h;zen::quant::encode_fp16(&d,&h,1);p[0]=std::uint8_t(h);p[1]=std::uint8_t(h>>8);}
                for(std::size_t j=0;j<f->block_elements;++j){
                    int q;
                    if(f->encoding==E::fp8_e4m3){
                        unsigned code=unsigned((r*31+b*7)%256);if((code&127u)==127)code=0;
                        p[j]=std::uint8_t(code);const unsigned a=code&127u,exp=(a>>3)&15u,m=a&7u;
                        const float v=exp==0?std::ldexp(float(m),-9):std::ldexp(1.f+float(m)/8.f,int(exp)-7);
                        ref[r*cols+b*f->block_elements+j]=(code&128u)?-v:v;continue;
                    }
                    if(f->encoding==E::nvfp4){
                        static constexpr int codebook[16]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
                        const unsigned group=unsigned(j/16),lane=unsigned(j%16),code=unsigned((j*7+r+b)%16);
                        p[4+group*8+lane%8]|=std::uint8_t(code<<(lane>=8?4:0));
                        const unsigned raw=p[group],exp=(raw>>3)&15u,m=raw&7u;
                        const float sc=(raw==0||raw==127)?0.f:(exp==0?std::ldexp(float(m),-9):std::ldexp(1.f+float(m)/8.f,int(exp)-7))*0.5f;
                        ref[r*cols+b*f->block_elements+j]=float(codebook[code])*sc;continue;
                    }
                    if(f->encoding==E::q1_0){q=((j+r+b)%3)?1:-1;if(q>0)p[2+j/8]|=std::uint8_t(1u<<(j%8));}
                    else if(f->encoding==E::q2_0){const unsigned v=(j+r+b)%4;q=int(v)-1;p[2+j/4]|=std::uint8_t(v<<(2*(j%4)));}
                    else {q=int((j*11+r*7+b*3)%255)-127;p[(f->encoding==E::q8_1?4:8)+j]=std::uint8_t(q);}
                    ref[r*cols+b*f->block_elements+j]=float(q)*d;
                }
            }
            for(std::size_t tokens:{1u,2u,3u,8u})for(unsigned workers:{1u,4u}){
                const bool ok=tokens==1?zen::quant::gemv(*f,w.data()+1,x.data(),out.data(),rows,cols,1,workers):
                    zen::quant::gemm(*f,w.data()+1,x.data(),out.data(),tokens,rows,cols,1,workers);
                if(!ok){std::cerr<<name<<" dispatch failed\n";++failures;continue;}
                for(std::size_t t=0;t<tokens;++t)for(std::size_t r=0;r<rows;++r){
                    double sum=0,mag=0;for(std::size_t c=0;c<cols;++c){const double v=double(ref[r*cols+c])*x[t*cols+c];sum+=v;mag+=std::abs(v);}
                    if(!std::isfinite(out[t*rows+r])||std::abs(double(out[t*rows+r])-sum)>2e-5*std::max(1.0,mag)){std::cerr<<name<<" mismatch cols="<<cols<<" rows="<<rows<<" t="<<t<<" r="<<r<<'\n';++failures;goto failed;}
                }
                failed:;
            }
#ifdef ZEN_HAS_AVX2_KERNELS
            if(zen::cpu_has_avx2_fma()){
                using Fn=bool(*)(const std::uint8_t*,const float*,float*,std::size_t,std::size_t,std::size_t,std::size_t);
                Fn fn=f->encoding==E::q1_0?zen::gemm_q1_0_kblk_avx2:f->encoding==E::q2_0?zen::gemm_q2_0_kblk_avx2:f->encoding==E::q8_1?zen::gemm_q8_1_kblk_avx2:f->encoding==E::q8_1_f32?zen::gemm_q8_1_f32_kblk_avx2:f->encoding==E::nvfp4?zen::gemm_nvfp4_kblk_avx2:zen::gemm_f8_e4m3_kblk_avx2;
                const std::size_t stride=rows+3;std::vector<float> padded(3*stride,-777.f);
                if(f->encoding==E::fp8_e4m3 && cols%8){
                    if(fn(w.data()+1,x.data(),padded.data(),3,rows,cols,stride)){std::cerr<<"FP8 direct kernel accepted non-vector-aligned columns\n";++failures;}
                }else if(!fn(w.data()+1,x.data(),padded.data(),3,rows,cols,stride)){++failures;}
                else for(std::size_t t=0;t<3;++t){
                    for(std::size_t r=0;r<rows;++r){double sum=0,mag=0;for(std::size_t c=0;c<cols;++c){double v=double(ref[r*cols+c])*x[t*cols+c];sum+=v;mag+=std::abs(v);}
                        if(!std::isfinite(padded[t*stride+r])||std::abs(double(padded[t*stride+r])-sum)>2e-5*std::max(1.0,mag)){std::cerr<<name<<" padded mismatch\n";++failures;}}
                    for(std::size_t r=rows;r<stride;++r)if(padded[t*stride+r]!=-777.f){std::cerr<<name<<" wrote padding\n";++failures;}
                }
            }
#endif
        }
    }
    // Exhaust all FP8 byte codes through a vector lane and through a scalar
    // tail. The expected FP8 values come from the separately checked decoder.
    {
        const auto& f=*zen::quant::find("F8_E4M3");float x[9]={1,1,1,1,1,1,1,1,1};
        for(unsigned code=0;code<256;++code)for(unsigned pos:{0u,8u}){
            std::uint8_t raw[9]{};raw[pos]=std::uint8_t(code);float decoded[9],out{};
            if(!zen::quant::dequantize(f,raw,9,decoded)||!zen::quant::gemv(f,raw,x,&out,1,9,1,1)||
               (std::isnan(decoded[pos])?!std::isnan(out):out!=decoded[pos])){
                std::cerr<<"FP8 code/tail mismatch code="<<code<<" pos="<<pos<<'\n';++failures;
            }
        }
#ifdef ZEN_HAS_AVX2_KERNELS
        if(zen::cpu_has_avx2_fma()){
            std::uint8_t raw[32]{};float input[32]{},output{};
            if(zen::gemv_new_quant_avx2(raw,input,&output,1,32,static_cast<zen::NewQuantKind>(255))){std::cerr<<"invalid kernel format accepted\n";++failures;}
        }
#endif
    }
    std::cout<<"Independent Q1_0/Q2_0/NVFP4/FP8/Q8_1 matrix tests: failures="<<failures<<'\n';return failures?1:0;
}
