// Direct GPU and quant-dispatch tests for raw GGML Q4_K/Q5_K GEMV/GEMM.
#include "zen/d3d12_backend.h"
#include "zen/quant.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

struct Shape { std::size_t rows, columns, tokens; };

bool check(const Shape& shape, bool q5) {
    const auto& f=*zen::quant::find(q5?"Q5_K":"Q4_K");
    const std::size_t rb=zen::quant::row_bytes(f,shape.columns);
    std::vector<std::uint8_t> w(shape.rows*rb);
    std::vector<float> x(shape.tokens*shape.columns), result(shape.tokens*shape.rows), decoded(shape.columns);
    std::uint32_t seed=42u;
    auto rnd=[&](){seed=seed*1664525u+1013904223u;return std::uint8_t(seed>>24);};
    for(auto& v:w)v=rnd();
    for(std::size_t r=0;r<shape.rows;++r)for(std::size_t b=0;b<shape.columns/256;++b) {
        auto* p=w.data()+r*rb+b*f.block_bytes;
        // Finite positive, negative, zero and subnormal halves. Random fields
        // exercise all scale/min and Q5 high-bit positions.
        const std::uint16_t d=(b%4==0?0x0001u:(r%2?0xa800u:0x2800u));
        const std::uint16_t dm=(b%3==0?0u:0x2000u);
        p[0]=std::uint8_t(d);p[1]=std::uint8_t(d>>8);
        p[2]=std::uint8_t(dm);p[3]=std::uint8_t(dm>>8);
    }
    for(auto& v:x)v=float(int(rnd())-128)*0.007f;
    // Do not let a recycled test allocation masquerade as immutable cached weights.
    zen::d3d12::release_kquant_weights(w.data());
    if(!zen::d3d12::gemm_kquant(w.data(),x.data(),result.data(),shape.tokens,shape.rows,shape.columns,q5)) {
        std::printf("GPU declined %s rows=%zu cols=%zu tokens=%zu\n",f.name.c_str(),shape.rows,shape.columns,shape.tokens); return false;
    }
    double worst=0;
    for(std::size_t r=0;r<shape.rows;++r){
        if(!zen::quant::dequantize(f,w.data()+r*rb,shape.columns,decoded.data()))return false;
        for(std::size_t t=0;t<shape.tokens;++t){
            double ref=0, magnitude=0;
            for(std::size_t c=0;c<shape.columns;++c){const double v=double(decoded[c])*x[t*shape.columns+c];ref+=v;magnitude+=std::abs(v);}
            const float actual=result[t*shape.rows+r];
            const double err=std::abs(double(actual)-ref);worst=std::max(worst,err);
            if(!std::isfinite(actual)||err>2e-5*std::max(1.0,magnitude)){
                std::printf("FAIL %s t=%zu row=%zu actual=%g expected=%g error=%g\n",f.name.c_str(),t,r,actual,ref,err);return false;
            }
        }
    }
    // Exercise the production quant dispatch too, and prove it really dispatched
    // instead of producing a misleading pass via CPU fallback.
    const auto before=zen::d3d12::stats().dispatches;
    std::vector<float> routed(result.size());
    const bool ok=shape.tokens==1
        ? zen::quant::gemv(f,w.data(),x.data(),routed.data(),shape.rows,shape.columns,1.0f,1)
        : zen::quant::gemm(f,w.data(),x.data(),routed.data(),shape.tokens,shape.rows,shape.columns,1.0f,1);
    if(!ok || zen::d3d12::stats().dispatches!=before+1){std::printf("FAIL %s dispatch fell back\n",f.name.c_str());return false;}
    for(std::size_t i=0;i<result.size();++i)if(!std::isfinite(routed[i]) || std::abs(routed[i]-result[i])>1e-5f*std::max(1.0f,std::abs(result[i])))return false;
    std::printf("PASS %s rows=%zu cols=%zu tokens=%zu max_abs=%g\n",f.name.c_str(),shape.rows,shape.columns,shape.tokens,worst);
    zen::d3d12::release_kquant_weights(w.data());
    return true;
}
bool check_q1q6(const Shape& shape, bool q6) {
    const auto& f=*zen::quant::find(q6?"Q6_K":"Q1_0");
    const std::size_t elem=q6?256u:128u, blk=q6?210u:18u;
    const std::size_t rb=zen::quant::row_bytes(f,shape.columns);
    std::vector<std::uint8_t> w(shape.rows*rb);
    std::vector<float> x(shape.tokens*shape.columns), result(shape.tokens*shape.rows), decoded(shape.columns);
    std::uint32_t seed=43u;
    auto rnd=[&](){seed=seed*1664525u+1013904223u;return std::uint8_t(seed>>24);};
    for(auto& v:w)v=rnd();
    for(std::size_t r=0;r<shape.rows;++r)for(std::size_t b=0;b<shape.columns/elem;++b){
        // Sane finite fp16 scales (incl. negative and small-magnitude cases).
        const std::uint16_t d=(b%4==0?0x0001u:(r%2?0xa800u:0x2c00u));
        auto* p=w.data()+r*rb+b*blk;
        const std::size_t doff=q6?208u:0u;
        p[doff]=std::uint8_t(d);p[doff+1]=std::uint8_t(d>>8);
    }
    for(auto& v:x)v=float(int(rnd())-128)*0.007f;
    zen::d3d12::release_kquant_weights(w.data());
    const bool direct=q6?zen::d3d12::gemm_q6_k(w.data(),x.data(),result.data(),shape.tokens,shape.rows,shape.columns)
                       :zen::d3d12::gemm_q1_0(w.data(),x.data(),result.data(),shape.tokens,shape.rows,shape.columns);
    if(!direct) {
        std::printf("GPU declined %s rows=%zu cols=%zu tokens=%zu\n",f.name.c_str(),shape.rows,shape.columns,shape.tokens); return false;
    }
    double worst=0;
    for(std::size_t r=0;r<shape.rows;++r){
        if(!zen::quant::dequantize(f,w.data()+r*rb,shape.columns,decoded.data()))return false;
        for(std::size_t t=0;t<shape.tokens;++t){
            double ref=0, magnitude=0;
            for(std::size_t c=0;c<shape.columns;++c){const double v=double(decoded[c])*x[t*shape.columns+c];ref+=v;magnitude+=std::abs(v);}
            const float actual=result[t*shape.rows+r];
            const double err=std::abs(double(actual)-ref);worst=std::max(worst,err);
            if(!std::isfinite(actual)||err>2e-5*std::max(1.0,magnitude)){
                std::printf("FAIL %s t=%zu row=%zu actual=%g expected=%g error=%g\n",f.name.c_str(),t,r,actual,ref,err);return false;
            }
        }
    }
    const auto before=zen::d3d12::stats().dispatches;
    std::vector<float> routed(result.size());
    const bool ok=shape.tokens==1
        ? zen::quant::gemv(f,w.data(),x.data(),routed.data(),shape.rows,shape.columns,1.0f,1)
        : zen::quant::gemm(f,w.data(),x.data(),routed.data(),shape.tokens,shape.rows,shape.columns,1.0f,1);
    if(!ok || zen::d3d12::stats().dispatches!=before+1){std::printf("FAIL %s dispatch fell back\n",f.name.c_str());return false;}
    for(std::size_t i=0;i<result.size();++i)if(!std::isfinite(routed[i]) || std::abs(routed[i]-result[i])>1e-5f*std::max(1.0f,std::abs(result[i])))return false;
    std::printf("PASS %s rows=%zu cols=%zu tokens=%zu max_abs=%g\n",f.name.c_str(),shape.rows,shape.columns,shape.tokens,worst);
    zen::d3d12::release_kquant_weights(w.data());
    return true;
}
int main(){
    if(!zen::d3d12::init()){std::printf("SKIP: D3D12 device unavailable\n");return 77;}
    const Shape shapes[]={{1,256,1},{7,512,3},{9,768,5},{33,1024,7},{65,4096,8}};
    for(bool q5:{false,true})for(const auto& shape:shapes)if(!check(shape,q5))return 1;
    const Shape shapes_q1[]={{1,128,1},{7,256,3},{9,512,5},{33,1024,7},{65,4096,8}};
    for(const auto& shape:shapes_q1)if(!check_q1q6(shape,false))return 1;
    const Shape shapes_q6[]={{1,256,1},{7,512,3},{9,768,5},{33,1024,7},{65,4096,8}};
    for(const auto& shape:shapes_q6)if(!check_q1q6(shape,true))return 1;
    // Reusing an address for a different format must replace its residency;
    // explicit release must also allow replacement with the same format/shape.
    std::vector<std::uint8_t> shared(176);
    std::vector<float> ones(256,1.0f),decoded(256);
    auto resident_check=[&](bool q5, std::uint8_t pattern){
        std::fill(shared.begin(),shared.end(),pattern);
        shared[0]=0;shared[1]=0x28;shared[2]=0;shared[3]=0x20;
        const auto& fmt=*zen::quant::find(q5?"Q5_K":"Q4_K");
        if(!zen::quant::dequantize(fmt,shared.data(),256,decoded.data()))return false;
        double expected=0;for(float v:decoded)expected+=v;
        float actual=0;
        return zen::d3d12::gemm_kquant(shared.data(),ones.data(),&actual,1,1,256,q5) &&
               std::isfinite(actual) && std::abs(actual-expected)<1e-4*std::max(1.0,std::abs(expected));
    };
    zen::d3d12::release_kquant_weights(shared.data());
    if(!resident_check(false,0x37) || !resident_check(true,0xd3))return 1;
    zen::d3d12::release_kquant_weights(shared.data());
    if(!resident_check(true,0x21))return 1;
    zen::d3d12::release_kquant_weights(shared.data());
    // Reject invalid and 32-bit-overflowing shapes before touching the pointers.
    std::uint8_t w=0;float x=0,y=0;
    if(zen::d3d12::gemm_kquant(&w,&x,&y,1,1,255,false) ||
       zen::d3d12::gemm_kquant(&w,&x,&y,65536u*4u,1,256,false) ||
       zen::d3d12::gemm_kquant(&w,&x,&y,1,65536u*8u,256,true) ||
       zen::d3d12::gemm_kquant(&w,&x,&y,2,1,std::size_t(1)<<30,true))return 1;
    zen::d3d12::shutdown();
    return 0;
}
