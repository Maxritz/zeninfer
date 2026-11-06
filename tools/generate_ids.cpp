#include "zen/model_runtime.h"
#include "zen/diagnostics.h"
#ifdef ZEN_HAS_D3D12
#include "zen/d3d12_backend.h"
#endif
#ifdef ZEN_HAS_HIP
#include "zen/hip_backend.h"
#endif
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
    if(argc<4){std::cerr<<"usage: zen_generate_ids model.gguf (--prompt-ids 1,2,3 | --prompt-text \"text\") [--max-new 8] [--runs 3]\n";return 2;}
    std::vector<std::uint32_t> prompt;std::string prompt_text;bool has_text=false;std::uint32_t max_new=8;std::uint32_t runs=1;
    for(int i=2;i<argc;++i){std::string a=argv[i];
        if(a=="--prompt-ids"&&i+1<argc){if(!parse_ids(argv[++i],prompt)){std::cerr<<"invalid token ID list\n";return 2;}}
        else if(a=="--prompt-text"&&i+1<argc){prompt_text=argv[++i];has_text=true;}
        else if(a=="--max-new"&&i+1<argc){try{max_new=static_cast<std::uint32_t>(std::stoul(argv[++i]));}catch(...){return 2;}}
        else if(a=="--runs"&&i+1<argc){try{runs=static_cast<std::uint32_t>(std::stoul(argv[++i]));}catch(...){return 2;}}
        else{std::cerr<<"unknown argument: "<<a<<"\n";return 2;}
    }
    if(!runs)runs=1;
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
    }else{
        for(std::uint32_t run=0;run<runs;++run){
            // Distinct ids per run: an identical prompt would hit the prefix
            // cache and turn every run after the first into a re-append instead
            // of a real prefill. Offsetting keeps all runs full-prefill while
            // weights stay warm from the first run's residency upload.
            std::vector<std::uint32_t> run_prompt=prompt;
            const std::uint32_t vocab=model->config().vocab;
            if(run)for(std::size_t i=1;i<run_prompt.size();++i)run_prompt[i]=(run_prompt[i]+run*1103)%vocab;
#ifdef ZEN_HAS_D3D12
            const auto st0=zen::d3d12::stats();
#endif
            const auto gen_start=std::chrono::steady_clock::now();
            zen::diag::reset_trace();
            if(!model->generate(run_prompt,max_new,out,&error)){std::cerr<<"generation failed: "<<error<<"\n";return 1;}
            const double gen_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-gen_start).count();
            std::cout<<"run_"<<run<<" total_ms="<<gen_ms<<" prefill_ms="<<out.prefill_ms<<" decode_ms="<<out.decode_ms;
            if(out.prefill_ms>0.0)std::cout<<" prefill_tok_per_s="<<1000.0*double(prompt.size())/out.prefill_ms;
#ifdef ZEN_HAS_D3D12
            const auto st1=zen::d3d12::stats();
            std::cout<<" submits="<<(st1.submits-st0.submits)
                     <<" wait_ms="<<((st1.wait_us-st0.wait_us)/1000)
                     <<" readback_ms="<<((st1.readback_us-st0.readback_us)/1000)
                     <<" upload_ms="<<((st1.upload_us-st0.upload_us)/1000)
                     <<" alloc_ms="<<((st1.alloc_us-st0.alloc_us)/1000)
                     <<" repack_ms="<<((st1.repack_us-st0.repack_us)/1000)
                     <<" stage_ms="<<((st1.stage_us-st0.stage_us)/1000)
                     <<" dispatches="<<(st1.dispatches-st0.dispatches);
#endif
            std::cout<<"\n";
        }
    }    std::cout<<"architecture="<<model->architecture_name()<<" embedding="<<model->config().embedding<<" layers="<<model->config().layers
             <<" heads="<<model->config().heads<<" kv_heads="<<model->config().kv_heads<<" vocab="<<model->config().vocab<<"\n";
    std::cout<<"prompt_ids=";for(std::size_t i=0;i<prompt.size();++i)std::cout<<(i?",":"")<<prompt[i];
    std::cout<<"\ngenerated_ids=";for(std::size_t i=0;i<out.token_ids.size();++i)std::cout<<(i?",":"")<<out.token_ids[i];
    if(has_text)std::cout<<"\nprompt_text=\""<<prompt_text<<"\"";
    if(has_text)std::cout<<"\ngenerated_text=\""<<generated_text<<"\"";
    std::cout<<"\nload_ms="<<load_ms<<" prefill_ms="<<out.prefill_ms<<" decode_forward_ms="<<out.decode_ms;
    if(out.prefill_ms>0.0)std::cout<<" prefill_tokens_per_second="<<1000.0*double(prompt.size())/out.prefill_ms;
    if(out.decode_ms>0.0&&out.token_ids.size()>1)std::cout<<" tokengen_tokens_per_second="<<1000.0*double(out.token_ids.size()-1)/out.decode_ms;
    std::cout<<"\n";
#ifdef ZEN_HAS_D3D12
    if(const auto* gpu=zen::d3d12::info()){
        const auto& st=zen::d3d12::stats();
        std::cout<<"d3d12 adapter="<<gpu->name<<" fl="<<gpu->feature_level
                 <<" agility="<<gpu->agility_sdk_version
                 <<" gpu_upload_heap(rebar)="<<(gpu->gpu_upload_heap_supported?1:0)
                 <<" vram_mb="<<(gpu->dedicated_vram>>20)
                 <<" weights_mb="<<(st.weight_bytes>>20)
                 <<" upload_mb="<<(st.upload_bytes>>20)
                 <<" dispatches="<<st.dispatches<<" submits="<<st.submits<<" batches="<<st.batches<<" per_submit_us="<<(st.submits?((st.queue_us+st.wait_us)/st.submits):0)
                 <<" per_queue_us="<<(st.submits?(st.queue_us/st.submits):0)
                 <<" per_fence_us="<<(st.submits?(st.wait_us/st.submits):0)
                 <<" queue_ms="<<(st.queue_us/1000)
                 <<" qkv_wait_ms="<<(st.qkv_wait_us/1000)
                 <<" ffn_wait_ms="<<(st.ffn_wait_us/1000)
                 <<" qkv_gpu_ms="<<(st.qkv_gpu_us/1000)
                 <<" ffn_gpu_ms="<<(st.ffn_gpu_us/1000)
                 <<" qkv_copy_ms="<<(st.qkv_copy_us/1000)
                 <<" ffn_copy_ms="<<(st.ffn_copy_us/1000)
                 <<" upload_ms="<<(st.upload_us/1000)
                 <<" alloc_ms="<<(st.alloc_us/1000)
                 <<" stage_ms="<<(st.stage_us/1000)
                 <<" repack_ms="<<(st.repack_us/1000)
                 <<" transfer_ms="<<(st.transfer_us/1000)
                 <<" fallbacks="<<st.fallbacks<<std::endl;
        if(st.dispatches){
            const double calls=static_cast<double>(st.dispatches);
            std::cout<<"d3d12 phase_us_per_call prep="<<(st.prep_us/calls)
                     <<" reset="<<(st.reset_us/calls)
                     <<" setup="<<(st.setup_us/calls)
                     <<" record="<<(st.record_us/calls)
                     <<" exec_wait="<<(st.wait_us/calls)
                     <<" readback="<<(st.readback_us/calls)
                     <<" gpu="<<(st.gpu_us/calls)
                     <<" gpu_copy="<<(st.gpu_copy_us/calls)<<std::endl;
        }
        zen::d3d12::trace_report();
    }
#endif
#ifdef ZEN_HAS_HIP
    if(const auto* gpu=zen::hip::info()){
        const auto& st=zen::hip::stats();
        std::cout<<"hip adapter="<<gpu->name<<" arch="<<gpu->arch
                 <<" sm="<<gpu->sm_count<<" warp="<<gpu->warp_size
                 <<" vram_mb="<<(gpu->dedicated_vram>>20)
                 <<" weights_mb="<<(st.weight_bytes>>20)
                 <<" upload_mb="<<(st.upload_bytes>>20)
                 <<" dispatches="<<st.dispatches<<" submits="<<st.submits<<" batches="<<st.batches
                 <<" per_submit_us="<<(st.submits?((st.launch_us+st.wait_us)/st.submits):0)
                 <<" launch_ms="<<(st.launch_us/1000)
                 <<" wait_ms="<<(st.wait_us/1000)
                 <<" readback_ms="<<(st.readback_us/1000)
                 <<" upload_ms="<<(st.upload_us/1000)
                 <<" alloc_ms="<<(st.alloc_us/1000)
                 <<" repack_ms="<<(st.repack_us/1000)
                 <<" transfer_ms="<<(st.transfer_us/1000)
                 <<" gpu_ms="<<(st.gpu_us/1000)
                 <<" fallbacks="<<st.fallbacks<<std::endl;
    }
#endif
    std::cout<<"Sherlock instrumentation floor: "<<zen::diag::measure_instrumentation_floor_us()<<" us/scope\n";
    zen::diag::print_trace(std::cout);zen::diag::dump_breadcrumbs(std::cout,32);
    return 0;
}
