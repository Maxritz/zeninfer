#include "zen/model_runtime.h"
#include "zen/diagnostics.h"
#include <chrono>
#include <charconv>
#include <iostream>
#include <string>
#include <vector>

static bool parse_ids(const std::string& s,std::vector<std::uint32_t>& ids){
    std::size_t p=0;while(p<s.size()){
        auto e=s.find(',',p);if(e==std::string::npos)e=s.size();
        if(e==p)return false;std::uint32_t v{};auto r=std::from_chars(s.data()+p,s.data()+e,v);
        if(r.ec!=std::errc{}||r.ptr!=s.data()+e)return false;ids.push_back(v);p=e+1;
    }return !ids.empty();
}
int main(int argc,char**argv){
    if(argc<4){std::cerr<<"usage: zen_generate_ids model.gguf (--prompt-ids 1,2,3 | --prompt-text \"text\") [--max-new 8]\n";return 2;}
    std::vector<std::uint32_t> prompt;std::string prompt_text;bool has_text=false;std::uint32_t max_new=8;
    for(int i=2;i<argc;++i){std::string a=argv[i];
        if(a=="--prompt-ids"&&i+1<argc){if(!parse_ids(argv[++i],prompt)){std::cerr<<"invalid token ID list\n";return 2;}}
        else if(a=="--prompt-text"&&i+1<argc){prompt_text=argv[++i];has_text=true;}
        else if(a=="--max-new"&&i+1<argc){try{max_new=static_cast<std::uint32_t>(std::stoul(argv[++i]));}catch(...){return 2;}}
        else{std::cerr<<"unknown argument: "<<a<<"\n";return 2;}
    }
    if(prompt.empty()&&!has_text){std::cerr<<"--prompt-ids or --prompt-text is required\n";return 2;}
    std::string error;
    zen::diag::reset_trace();zen::diag::reset_breadcrumbs();
    zen::diag::set_trace_enabled(true);zen::diag::set_breadcrumbs_enabled(true);
    const auto load_start=std::chrono::steady_clock::now();
    auto model=zen::runtime::load_model_runtime(argv[1],nullptr,&error);
    const double load_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-load_start).count();
    if(!model){std::cerr<<"load failed: "<<error<<"\n";return 1;}
    zen::runtime::Generation out;std::string generated_text;
    if(has_text){
        if(!model->tokenize(prompt_text,prompt)){std::cerr<<"tokenisation failed or tokenizer unavailable\n";return 1;}
        if(!model->generate_text(prompt_text,max_new,out,generated_text,&error)){std::cerr<<"generation failed: "<<error<<"\n";return 1;}
    }else if(!model->generate(prompt,max_new,out,&error)){std::cerr<<"generation failed: "<<error<<"\n";return 1;}
    std::cout<<"architecture="<<model->architecture_name()<<" embedding="<<model->config().embedding<<" layers="<<model->config().layers
             <<" heads="<<model->config().heads<<" kv_heads="<<model->config().kv_heads<<" vocab="<<model->config().vocab<<"\n";
    std::cout<<"prompt_ids=";for(std::size_t i=0;i<prompt.size();++i)std::cout<<(i?",":"")<<prompt[i];
    std::cout<<"\ngenerated_ids=";for(std::size_t i=0;i<out.token_ids.size();++i)std::cout<<(i?",":"")<<out.token_ids[i];
    if(has_text)std::cout<<"\nprompt_text=\""<<prompt_text<<"\"";
    if(has_text)std::cout<<"\ngenerated_text=\""<<generated_text<<"\"";
    std::cout<<"\nload_ms="<<load_ms<<" prefill_ms="<<out.prefill_ms<<" decode_forward_ms="<<out.decode_ms;
    if(out.prefill_ms>0.0)std::cout<<" prefill_tokens_per_second="<<1000.0*double(prompt.size())/out.prefill_ms;
    if(out.decode_ms>0.0&&out.token_ids.size()>1)std::cout<<" tokengen_tokens_per_second="<<1000.0*double(out.token_ids.size()-1)/out.decode_ms;
    std::cout<<"\n";
    std::cout<<"Sherlock instrumentation floor: "<<zen::diag::measure_instrumentation_floor_us()<<" us/scope\n";
    zen::diag::print_trace(std::cout);zen::diag::dump_breadcrumbs(std::cout,32);
    return 0;
}
