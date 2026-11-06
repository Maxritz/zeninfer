#pragma once
#include "zen/quant.h"
#include <cstdint>
#include <cstring>
#include <vector>

// Deterministic finite test data. This is not a model or quantizer.
inline std::vector<std::uint8_t> quant_fixture(const zen::quant::Format& f,
                                                std::size_t rows, std::size_t cols) {
    const std::size_t rb=zen::quant::row_bytes(f,cols);
    std::vector<std::uint8_t> out(rows*rb);
    std::uint32_t state=1729;
    auto rnd=[&](){state=state*1664525u+1013904223u;return std::uint8_t(state>>24);};
    for(auto& b:out)b=rnd();
    using E=zen::quant::Encoding;
    if(f.encoding==E::ternary) {
        for(auto& b:out){b=0;for(unsigned i=0;i<4;++i)b|=std::uint8_t((rnd()%3)<<(2*i));}
    } else if(f.encoding==E::fp16 || f.encoding==E::bf16 || f.encoding==E::fp32 ||
              f.encoding==E::fp64 || f.encoding==E::fp8_e4m3 || f.encoding==E::i8 ||
              f.encoding==E::i16 || f.encoding==E::i32 || f.encoding==E::i64) {
        for(std::size_t r=0;r<rows;++r)for(std::size_t c=0;c<cols;++c){
            float v=float(int(rnd())-128)/128.f;
            if(f.encoding==E::fp32) std::memcpy(out.data()+r*rb+4*c,&v,4);
            else if(f.encoding==E::fp64){double dv=v;std::memcpy(out.data()+r*rb+8*c,&dv,8);}
            else if(f.encoding==E::fp8_e4m3)out[r*rb+c]=std::uint8_t(0x20+(rnd()%32));
            else if(f.encoding==E::i8)out[r*rb+c]=rnd();
            else if(f.encoding==E::i16 || f.encoding==E::i32 || f.encoding==E::i64){
                std::int64_t iv=int(rnd())-128;const auto bytes=f.block_bytes;const auto bits=static_cast<std::uint64_t>(iv);
                for(unsigned i=0;i<bytes;++i)out[r*rb+bytes*c+i]=std::uint8_t(bits>>(8*i));
            }
            else if(f.encoding==E::fp16) {std::uint16_t h;zen::quant::encode_fp16(&v,&h,1);out[r*rb+2*c]=std::uint8_t(h);out[r*rb+2*c+1]=std::uint8_t(h>>8);}
            else {std::uint32_t bits;std::memcpy(&bits,&v,4);out[r*rb+2*c]=std::uint8_t(bits>>16);out[r*rb+2*c+1]=std::uint8_t(bits>>24);}
        }
    } else if(f.encoding!=E::signed_uniform) {
        for(std::size_t r=0;r<rows;++r)for(std::size_t b=0;b<cols/f.block_elements;++b){
            auto* p=out.data()+r*rb+b*f.block_bytes;
            auto half=[&](std::size_t i){p[i]=0;p[i+1]=0x30;};
            switch(f.encoding){
            case E::q2_k:half(80);half(82);break;
            case E::q3_k:half(108);break;
            case E::q6_k:half(208);break;
            case E::tq1_0:half(52);break;
            case E::tq2_0:half(64);break;
            case E::ptq1_0:half(26);break;
            case E::iq1_m: // assembled half from four high nibbles: 0x3000
                p[49]&=0x0f;p[51]&=0x0f;p[53]&=0x0f;p[55]=(p[55]&0x0f)|0x30;break;
            case E::q8_k: case E::q8_1_f32: {float d=0.125f;std::memcpy(p,&d,4);break;}
            case E::mxfp4:p[0]=127;break;
            case E::nvfp4:for(unsigned i=0;i<4;++i)p[i]=std::uint8_t(1+(rnd()%126));break;
            case E::rocmfp4:p[16]=48;p[17]=48;break;
            case E::rocmfp4_fast:p[16]=48;break;
            default:half(0);if(f.encoding==E::q4_1||f.encoding==E::q5_1||f.encoding==E::q4_k||f.encoding==E::q5_k)half(2);break;
            }
        }
    }
    return out;
}
