#include "zen/gguf.h"
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {
void u32(std::ofstream& o,std::uint32_t v){for(int i=0;i<4;++i)o.put(char(v>>(8*i)));}
void u64(std::ofstream& o,std::uint64_t v){for(int i=0;i<8;++i)o.put(char(v>>(8*i)));}
void str(std::ofstream& o,const std::string& s){u64(o,s.size());o.write(s.data(),s.size());}
void set_layout(const char* value){
#if defined(_WIN32)
    _putenv_s("ZEN_GGUF_Q8_1_LAYOUT",value?value:"");
#else
    if(value)setenv("ZEN_GGUF_Q8_1_LAYOUT",value,1);else unsetenv("ZEN_GGUF_Q8_1_LAYOUT");
#endif
}
void write_fixture(const std::filesystem::path& path,bool legacy) {
    std::ofstream o(path,std::ios::binary);
    o.write("GGUF",4);u32(o,3);u64(o,2);u64(o,1);
    str(o,"general.architecture");u32(o,8);str(o,"llama");
    str(o,"test.weight");u32(o,1);u64(o,64);u32(o,9);u64(o,0);
    str(o,"tail.weight");u32(o,1);u64(o,1);u32(o,0);u64(o,96);
    auto p=static_cast<std::uint64_t>(o.tellp());for(;p%32;++p)o.put('\0');
    for(int b=0;b<2;++b){
        if(legacy){u32(o,b?0x3f800000u:0x3f000000u);u32(o,0);}
        else {o.put(0);o.put(b?0x3c:0x38);o.put(0);o.put(0);}
        for(int j=0;j<32;++j)o.put(char((j+b*7)%31-15));
    }
    for(int i=legacy?80:72;i<96;++i)o.put('\0');
    u32(o,0x3f800000u);
}
}
int main(){
    set_layout(nullptr);
    int failures=0;
    const auto dir=std::filesystem::temp_directory_path();
    for(bool legacy:{false,true}){
        const auto path=dir/((legacy?"zen-q8-legacy-":"zen-q8-current-")+std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count())+".gguf");
        write_fixture(path,legacy);
        const auto unspecified=zen::gguf::inspect(path);
        if(unspecified.ok||unspecified.error.find("ambiguous")==std::string::npos){std::cerr<<"ambiguous file accepted\n";++failures;}
        std::ifstream stream(path,std::ios::binary);
        std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(stream)),{});stream.close();
        if(zen::gguf::inspect_bytes(bytes.data(),bytes.size()).ok){std::cerr<<"ambiguous buffer accepted\n";++failures;}
        const auto mode=legacy?zen::gguf::Q8_1Layout::fp32_40:zen::gguf::Q8_1Layout::fp16_36;
        set_layout("invalid");
        if(zen::gguf::inspect(path).ok){std::cerr<<"invalid environment layout accepted\n";++failures;}
        set_layout(legacy?"fp32":"fp16");
        if(!zen::gguf::inspect(path).ok){std::cerr<<"explicit environment layout rejected\n";++failures;}
        set_layout(legacy?"fp16":"fp32");
        const auto report=zen::gguf::inspect(path,{mode}); // explicit option overrides environment
        set_layout(nullptr);
        const auto buffer=zen::gguf::inspect_bytes(bytes.data(),bytes.size(),{mode});
        if(!report.ok||!buffer.ok||report.tensors.size()!=2||report.tensors[0].byte_size!=(legacy?80:72)){
            std::cerr<<"selected layout rejected: "<<report.error<<'\n';++failures;
        } else {
            std::vector<float> out;std::string error;
            if(!zen::gguf::read_tensor_f32(path,report,"test.weight",out,&error)||out.size()!=64){std::cerr<<"decode failed: "<<error<<'\n';++failures;}
            else for(int b=0;b<2;++b)for(int j=0;j<32;++j){
                const float expected=float((j+b*7)%31-15)*(b?1.f:0.5f);
                if(out[b*32+j]!=expected){std::cerr<<"wrong decoded layout\n";++failures;goto next_file;}
            }
        }
        next_file:
        std::error_code ec;std::filesystem::remove(path,ec);
    }
    std::cout<<"Q8_1 explicit modern/legacy layout tests: "<<(failures?"FAIL":"PASS")<<'\n';return failures?1:0;
}
