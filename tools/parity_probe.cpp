#include "zen/dense_transformer_runtime.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
static bool ids(const std::string& s,std::vector<std::uint32_t>& v){std::size_t p=0;while(p<s.size()){auto e=s.find(',',p);if(e==std::string::npos)e=s.size();try{v.push_back(static_cast<std::uint32_t>(std::stoul(s.substr(p,e-p))));}catch(...){return false;}p=e+1;}return !v.empty();}
int main(int argc,char**argv){
 if(argc<4){std::cerr<<"usage: zen_parity_probe model.gguf (--prompt-text text | --prompt-ids ids) logits.f32\n";return 2;}
 std::vector<std::uint32_t> prompt;std::string mode=argv[2];
 zen::runtime::DenseTransformerRuntime model;std::string error;if(!model.load(argv[1],&error)){std::cerr<<error<<"\n";return 1;}
 if(mode=="--prompt-text"){if(!model.tokenize(argv[3],prompt)){std::cerr<<"tokenization failed\n";return 1;}}
 else if(mode=="--prompt-ids"){if(!ids(argv[3],prompt)){std::cerr<<"invalid ids\n";return 2;}}
 else return 2;
 zen::runtime::Generation result;if(!model.generate(prompt,0,result,&error)){std::cerr<<error<<"\n";return 1;}
 std::ofstream out(argv[argc-1],std::ios::binary);if(!out){std::cerr<<"cannot create logits file\n";return 1;}
 out.write(reinterpret_cast<const char*>(result.last_logits.data()),static_cast<std::streamsize>(result.last_logits.size()*sizeof(float)));
 std::cout<<"prompt_ids=";for(std::size_t i=0;i<prompt.size();++i)std::cout<<(i?",":"")<<prompt[i];
 std::cout<<"\nlogits="<<result.last_logits.size()<<"\n";
}
