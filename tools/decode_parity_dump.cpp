// Decode logits parity dump: prefill + 1 decode forward, dump last_logits.
// Usage: zen_decode_parity_dump model.gguf prompt_ids logits.f32 [max_new]
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "zen/model_runtime.h"

static bool parse_ids(const std::string& s, std::vector<std::uint32_t>& v) {
    std::size_t p=0;
    while(p<s.size()){
        auto e=s.find(',',p); if(e==std::string::npos) e=s.size();
        if(e==p) return false;
        try{v.push_back(static_cast<std::uint32_t>(std::stoul(s.substr(p,e-p))));}catch(...){return false;}
        p=e+1;
    }
    return !v.empty();
}
int main(int argc,char**argv){
    if(argc<4){std::cerr<<"usage: zen_decode_parity_dump model.gguf prompt_ids logits.f32 [max_new]\n";return 2;}
    std::vector<std::uint32_t> prompt;
    if(!parse_ids(argv[2],prompt)){std::cerr<<"invalid ids\n";return 2;}
    std::uint32_t max_new=2;
    if(argc>4){try{max_new=static_cast<std::uint32_t>(std::stoul(argv[4]));}catch(...){return 2;}}
    std::string error;
    auto rt = zen::runtime::load_model_runtime(argv[1], nullptr, &error);
    if(!rt){std::cerr<<"load failed: "<<error<<"\n";return 1;}
    zen::runtime::Generation res;
    zen::runtime::GenerationOptions opts; // greedy
    if(!rt->generate(prompt, max_new, res, &error, {}, opts)){std::cerr<<"generate failed: "<<error<<"\n";return 1;}
    std::ofstream out(argv[3], std::ios::binary);
    if(!out){std::cerr<<"cannot create file\n";return 1;}
    out.write(reinterpret_cast<const char*>(res.last_logits.data()), static_cast<std::streamsize>(res.last_logits.size()*sizeof(float)));
    std::cout<<"tokens:"; for(auto t:res.token_ids) std::cout<<" "<<t;
    std::cout<<"\nlogits="<<res.last_logits.size()<<"\n";
    return 0;
}
