#include "zen/gguf.h"
#include <iostream>
int main(int argc,char**argv){
    if(argc!=2){std::cerr<<"usage: zen_model_inspect model.gguf\n";return 2;}
    auto r=zen::gguf::inspect(argv[1]);
    if(!r.ok){std::cerr<<"GGUF inspection failed: "<<r.error<<"\n";return 1;}
    std::cout<<"GGUF v"<<r.version<<"\narchitecture: "<<r.config.architecture_name
             <<"\nmodel: "<<r.config.model_name<<"\ntensors: "<<r.tensors.size()
             <<"\ndata offset: "<<r.tensor_data_offset<<"\nembedding: "<<r.config.embedding_length
             <<"\nlayers: "<<r.config.block_count<<"\nheads: "<<r.config.attention_head_count
             <<"\nKV heads: "<<r.config.kv_head_count<<"\nexperts: "<<r.config.expert_count
             <<"\ntokenizer tokens: "<<r.config.tokenizer_tokens.size()
             <<"\ntokenizer scores: "<<r.config.tokenizer_scores.size()
             <<"\ntokenizer token types: "<<r.config.tokenizer_token_types.size()
             <<"\nBOS/EOS: "<<r.config.bos_token_id<<"/"<<r.config.eos_token_id<<"\n";
    std::size_t unsupported=0;for(const auto&t:r.tensors)if(!t.byte_size)++unsupported;
    std::cout<<"unresolved tensor sizes: "<<unsupported<<"\n";
    return 0;
}
