#include "zen/gguf.h"
#include <algorithm>
#include "zen/quant.h"
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static bool put(std::ofstream& f,const void* p,std::size_t n){f.write(static_cast<const char*>(p),static_cast<std::streamsize>(n));return bool(f);}
template<class T> static bool scalar(std::ofstream& f,const T& v){return put(f,&v,sizeof(v));}
int main(int argc,char** argv){
    if(argc!=3){std::cerr<<"usage: zen_parity_export model.gguf output.znp\n";return 2;}
    const auto report=zen::gguf::inspect(argv[1]);if(!report.ok){std::cerr<<report.error<<"\n";return 1;}
    const auto& c=report.config;
    const auto arch_it=c.metadata.find("general.architecture");
    const std::string arch=arch_it==c.metadata.end()?"llama":arch_it->second;
    const auto getf=[&](const char* key,float fallback){auto i=c.metadata.find(key);if(i==c.metadata.end())return fallback;try{return std::stof(i->second);}catch(...){return fallback;}};
    const auto getu=[&](const char* key,std::uint32_t fallback){auto i=c.metadata.find(key);if(i==c.metadata.end())return fallback;try{return static_cast<std::uint32_t>(std::stoul(i->second));}catch(...){return fallback;}};
    std::ofstream f(argv[2],std::ios::binary);if(!f){std::cerr<<"cannot create output\n";return 1;}
    const char magic[8]={'Z','N','P','A','R','0','1','\0'};if(!put(f,magic,8))return 1;
    const std::uint32_t vals[]={c.embedding_length,c.block_count,c.attention_head_count,c.kv_head_count,c.feed_forward_length,c.vocabulary_size,c.context_length,getu((arch+".rope.dimension_count").c_str(),c.embedding_length/std::max(1u,c.attention_head_count))};
    for(auto v:vals)if(!scalar(f,v))return 1;
    const float rope=getf((arch+".rope.freq_base").c_str(),10000.0f),eps=getf((arch+".attention.layer_norm_rms_epsilon").c_str(),1.0e-5f);
    if(!scalar(f,rope)||!scalar(f,eps))return 1;
    const std::uint32_t count=static_cast<std::uint32_t>(report.tensors.size());if(!scalar(f,count))return 1;
    for(const auto& t:report.tensors){
        std::vector<float> data;std::string error;
        if(!zen::gguf::read_tensor_f32(argv[1],report,t.name,data,&error)){std::cerr<<t.name<<": "<<error<<"\n";return 1;}
        const std::uint32_t nl=static_cast<std::uint32_t>(t.name.size()),nd=static_cast<std::uint32_t>(t.dimensions.size());
        const std::uint64_t n=data.size();
        if(!scalar(f,nl)||!put(f,t.name.data(),nl)||!scalar(f,nd))return 1;
        for(auto d:t.dimensions)if(!scalar(f,d))return 1;
        if(!scalar(f,n)||!put(f,data.data(),data.size()*sizeof(float)))return 1;
    }
    std::cout<<"exported "<<count<<" tensors to "<<argv[2]<<"\n";
}
