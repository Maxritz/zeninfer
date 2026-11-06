#include "zen/quant.h"
#include "zen/gguf.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {
std::uint32_t read32(std::istream& in){std::uint8_t b[4]{};in.read(reinterpret_cast<char*>(b),4);return std::uint32_t(b[0])|(std::uint32_t(b[1])<<8)|(std::uint32_t(b[2])<<16)|(std::uint32_t(b[3])<<24);}
void write32(std::ostream& o,std::uint32_t v){for(int i=0;i<4;++i)o.put(static_cast<char>(v>>(8*i)));}
void write64(std::ostream& o,std::uint64_t v){for(int i=0;i<8;++i)o.put(static_cast<char>(v>>(8*i)));}
void string(std::ostream& o,const std::string& s){write64(o,s.size());o.write(s.data(),s.size());}
bool same(float a,float b){return (std::isnan(a)&&std::isnan(b)) || (a==b && (a!=0 || std::signbit(a)==std::signbit(b)));}
}
int main(int argc,char** argv){
    if(argc!=2){std::cerr<<"expected golden file path\n";return 2;}
    std::ifstream in(argv[1],std::ios::binary);char magic[4]{};in.read(magic,4);
    if(!in||std::memcmp(magic,"ZQG1",4)){std::cerr<<"bad golden file\n";return 1;}
    const auto count=read32(in);int failures=0;
    const auto path=std::filesystem::temp_directory_path()/("zen-quant-golden-"+std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count())+".gguf");
    for(unsigned r=0;r<count;++r){
        const auto len=read32(in);if(len>100)return 1;std::string name(len,'\0');in.read(name.data(),len);
        const auto type=read32(in),bytes=read32(in),n=read32(in);
        std::vector<std::uint8_t> raw(bytes+1);in.read(reinterpret_cast<char*>(raw.data()+1),bytes); // unaligned
        std::vector<float> ref(n);for(auto& v:ref){auto bits=read32(in);std::memcpy(&v,&bits,4);}
        if(!in){std::cerr<<"truncated reference\n";return 1;}
        const auto* f=zen::quant::find(name);
        if(!f||zen::quant::row_bytes(*f,n)!=bytes){std::cerr<<name<<": wrong layout\n";++failures;continue;}
        std::vector<float> out(n);if(!zen::quant::dequantize(*f,raw.data()+1,n,out.data())){std::cerr<<name<<": dequant failed\n";++failures;continue;}
        bool good=true;for(std::size_t i=0;i<n;++i)if(!same(out[i],ref[i])){std::cerr<<name<<": decode mismatch at "<<i<<" got "<<out[i]<<" expected "<<ref[i]<<'\n';good=false;break;}
        if(!good)++failures;
        if(f->block_elements>1 && (zen::quant::row_bytes(*f,n-1)!=0 || zen::quant::dequantize(*f,raw.data()+1,n-1,out.data()))){std::cerr<<name<<": accepted partial block\n";++failures;}
        // Exercise directory sizing and the actual file-backed tensor decoder.
        {std::ofstream o(path,std::ios::binary);o.write("GGUF",4);write32(o,3);write64(o,1);write64(o,1);
         string(o,"general.architecture");write32(o,8);string(o,"llama");
         string(o,"test.weight");write32(o,1);write64(o,n);write32(o,type);write64(o,0);
         auto p=static_cast<std::uint64_t>(o.tellp());for(;p%32;++p)o.put('\0');o.write(reinterpret_cast<const char*>(raw.data()+1),bytes);}
        auto report=zen::gguf::inspect(path,{zen::gguf::Q8_1Layout::fp16_36});std::vector<float> loaded;std::string error;
        if(!report.ok||report.tensors.size()!=1||report.tensors[0].byte_size!=bytes||!zen::gguf::read_tensor_f32(path,report,"test.weight",loaded,&error)||loaded.size()!=n){
            std::cerr<<name<<": GGUF load failed "<<report.error<<" "<<error<<'\n';++failures;
        }else for(std::size_t i=0;i<n;++i)if(!same(loaded[i],ref[i])){std::cerr<<name<<": GGUF mismatch at "<<i<<'\n';++failures;break;}
        // A whole tensor can be block-aligned while individual rows are not.
        // GGML blocks cannot cross row boundaries, even for a 2D tensor.
        if(f->block_elements>1){
            const auto block=f->block_elements;
            {std::ofstream o(path,std::ios::binary);o.write("GGUF",4);write32(o,3);write64(o,1);write64(o,1);
             string(o,"general.architecture");write32(o,8);string(o,"llama");
             string(o,"test.weight");write32(o,2);write64(o,block/2);write64(o,2);write32(o,type);write64(o,0);
             auto p=static_cast<std::uint64_t>(o.tellp());for(;p%32;++p)o.put('\0');
             o.write(reinterpret_cast<const char*>(raw.data()+1),f->block_bytes);}
            if(zen::gguf::inspect(path,{zen::gguf::Q8_1Layout::fp16_36}).ok){std::cerr<<name<<": misaligned GGUF rows accepted\n";++failures;}
        }
        // Restore a valid one-dimensional file before testing truncation.
        {std::ofstream o(path,std::ios::binary);o.write("GGUF",4);write32(o,3);write64(o,1);write64(o,1);
         string(o,"general.architecture");write32(o,8);string(o,"llama");
         string(o,"test.weight");write32(o,1);write64(o,n);write32(o,type);write64(o,0);
         auto p=static_cast<std::uint64_t>(o.tellp());for(;p%32;++p)o.put('\0');o.write(reinterpret_cast<const char*>(raw.data()+1),bytes);}
        std::error_code ec;auto filebytes=std::filesystem::file_size(path,ec);std::filesystem::resize_file(path,filebytes-1,ec);
        if(ec||zen::gguf::inspect(path,{zen::gguf::Q8_1Layout::fp16_36}).ok){std::cerr<<name<<": truncated GGUF accepted\n";++failures;}
        if(std::all_of(ref.begin(),ref.end(),[](float v){return std::isfinite(v);})){
            std::vector<float> x(n*3), y(3);for(std::size_t i=0;i<x.size();++i)x[i]=float(int(i%19)-9)/16.0f;
            for(unsigned tokens:{1u,3u}){
                bool ok=tokens==1?zen::quant::gemv(*f,raw.data()+1,x.data(),y.data(),1,n):zen::quant::gemm(*f,raw.data()+1,x.data(),y.data(),tokens,1,n);
                if(!ok){std::cerr<<name<<": matmul failed\n";++failures;continue;}
                for(unsigned t=0;t<tokens;++t){double sum=0,mag=0;for(std::size_t i=0;i<n;++i){double v=double(ref[i])*x[t*n+i];sum+=v;mag+=std::abs(v);}
                    if(!std::isfinite(y[t])||std::abs(double(y[t])-sum)>2e-5*std::max(1.0,mag)){std::cerr<<name<<": matmul mismatch\n";++failures;break;}}
            }
        }
    }
    std::error_code ec;std::filesystem::remove(path,ec);
    std::cout<<count<<" independent golden records; failures="<<failures<<'\n';return failures?1:0;
}
