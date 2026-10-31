#include "zen/gguf.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace {
void json_string(std::ostream& out,const std::string& value){
    out<<'"';
    for(unsigned char c:value){
        switch(c){case '"':out<<"\\\"";break;case '\\':out<<"\\\\";break;case '\n':out<<"\\n";break;case '\r':out<<"\\r";break;case '\t':out<<"\\t";break;
        default:if(c<0x20)out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<static_cast<unsigned>(c)<<std::dec;else out<<static_cast<char>(c);}
    }
    out<<'"';
}
}
int main(int argc,char**argv){
    if(argc<2||argc>3){std::cerr<<"usage: zen_tokenizer_export model.gguf [output.json]\n";return 2;}
    const auto report=zen::gguf::inspect(argv[1]);
    if(!report.ok){std::cerr<<"GGUF inspection failed: "<<report.error<<'\n';return 1;}
    const auto& c=report.config;
    if(c.tokenizer_tokens.empty()||c.tokenizer_tokens.size()!=c.tokenizer_scores.size()||c.tokenizer_tokens.size()!=c.tokenizer_token_types.size()){
        std::cerr<<"GGUF tokenizer arrays are missing or inconsistent\n";return 1;
    }
    std::ofstream file;std::ostream* out=&std::cout;
    if(argc==3){file.open(argv[2],std::ios::binary);if(!file){std::cerr<<"cannot open output file\n";return 1;}out=&file;}
    const auto metadata=[&](const char* key,const std::string& fallback){auto it=c.metadata.find(key);return it==c.metadata.end()?fallback:it->second;};
    *out<<"{\"model\":";json_string(*out,c.architecture_name);
    *out<<",\"tokenizer_model\":";json_string(*out,metadata("tokenizer.ggml.model",""));
    *out<<",\"bos\":"<<c.bos_token_id<<",\"eos\":"<<c.eos_token_id<<",\"unk\":"<<c.unknown_token_id
        <<",\"add_bos\":"<<(c.add_bos_token?"true":"false")<<",\"add_eos\":"<<(c.add_eos_token?"true":"false")
        <<",\"add_space_prefix\":"<<(c.add_space_prefix?"true":"false")<<",\"tokens\":[";
    for(std::size_t i=0;i<c.tokenizer_tokens.size();++i){if(i)*out<<',';json_string(*out,c.tokenizer_tokens[i]);}
    *out<<"],\"scores\":[";for(std::size_t i=0;i<c.tokenizer_scores.size();++i){if(i)*out<<',';*out<<std::setprecision(9)<<c.tokenizer_scores[i];}
    *out<<"],\"types\":[";for(std::size_t i=0;i<c.tokenizer_token_types.size();++i){if(i)*out<<',';*out<<c.tokenizer_token_types[i];}
    *out<<"]}\n";
    return *out?0:1;
}
