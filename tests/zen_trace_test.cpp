#include "zen/model_runtime.h"
#include "zen/d3d12_backend.h"
#include <atomic>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

int main(int argc,char**argv){
    _putenv("ZEN_D3D12_TRACE_LITE=1");
    if(argc<2||argc>3){std::cerr<<"usage: zen_runtime_tests model.gguf [draft.gguf]\n";return 2;}
    std::string error;auto runtime=zen::runtime::load_model_runtime(argv[1],nullptr,&error);
    if(!runtime){std::cerr<<"generic adapter load failed: "<<error<<"\n";return 1;}
    std::unique_ptr<zen::runtime::ModelRuntime> draft_runtime;const zen::runtime::ModelRuntime* draft=runtime.get();if(argc==3){draft_runtime=zen::runtime::load_model_runtime(argv[2],nullptr,&error);if(!draft_runtime){std::cerr<<"draft adapter load failed: "<<error<<"\n";return 1;}draft=draft_runtime.get();}
    if(!runtime->tokenizer_available()){std::cerr<<"tokenizer unavailable\n";return 1;}
    std::vector<std::uint32_t> ids;
    if(!runtime->tokenize("Once upon a time",ids)||ids.empty()){std::cerr<<"tokenization failed\n";return 1;}
    std::string roundtrip;if(!runtime->detokenize(ids,roundtrip)||roundtrip.empty()){std::cerr<<"tokenizer round-trip failed\n";return 1;}
    std::vector<std::uint32_t> perf_prompt;perf_prompt.reserve(128);
    for(std::size_t i=0;i<128;++i)perf_prompt.push_back(ids[i%ids.size()]);
    auto warm_prompt=perf_prompt;warm_prompt.back()=(warm_prompt.back()+97u)%runtime->config().vocab;
    zen::runtime::Generation warmup;
    if(!runtime->generate(warm_prompt,1,warmup,&error)){std::cerr<<"performance warmup failed: "<<error<<"\n";return 1;}
    constexpr std::size_t benchmark_runs=1;
    std::vector<double> prefill_rates,decode_rates,prefill_times,decode_times;
    std::size_t generated_count=0;
    for(std::size_t run=0;run<benchmark_runs;++run){
        auto run_prompt=perf_prompt;run_prompt.back()=static_cast<std::uint32_t>((run_prompt.back()+run+1)%runtime->config().vocab);
        zen::runtime::Generation perf;
        if(!runtime->generate(run_prompt,32,perf,&error)){std::cerr<<"performance generation failed: "<<error<<"\n";return 1;}
        if(perf.cache_hit){std::cerr<<"benchmark unexpectedly reused a cached prefix\n";return 1;}
        const std::size_t decode_steps=perf.token_ids.empty()?0:perf.token_ids.size()-1;
        if(perf.prefill_ms<=0.0||perf.decode_ms<=0.0||decode_steps==0){std::cerr<<"invalid benchmark timing or no decode steps\n";return 1;}
        prefill_times.push_back(perf.prefill_ms);decode_times.push_back(perf.decode_ms);
        prefill_rates.push_back(double(run_prompt.size())*1000.0/perf.prefill_ms);
        decode_rates.push_back(double(decode_steps)*1000.0/perf.decode_ms);generated_count+=perf.token_ids.size();
    }
    const auto median=[](std::vector<double> v){std::sort(v.begin(),v.end());return v[v.size()/2];};
    const double prefill_tps=median(prefill_rates),decode_tps=median(decode_rates);
    const double prefill_ms=median(prefill_times),decode_ms=median(decode_times);
    std::cout<<std::fixed<<std::setprecision(2)
             <<"PERF model="<<runtime->architecture_name()<<" prompt_tokens="<<perf_prompt.size()
             <<" generated_tokens_median="<<double(generated_count)/benchmark_runs<<" runs="<<benchmark_runs
             <<" prefill_ms_median="<<prefill_ms<<" prefill_tok_s_median="<<prefill_tps
             <<" decode_forward_ms_median="<<decode_ms<<" tokengen_tok_s_median="<<decode_tps
             <<" cache_hit=no warmup=one-distinct-prompt\n";
    
    // Print D3D12 stats
    const auto& stats = zen::d3d12::stats();
    std::cout << "D3D12_STATS: dispatches=" << stats.dispatches
              << " submits=" << stats.submits
              << " batches=" << stats.batches
              << " queue_us=" << stats.queue_us
              << " wait_us=" << stats.wait_us
              << " upload_us=" << stats.upload_us
              << " qkv_gpu_us=" << stats.qkv_gpu_us
              << " qkv_copy_us=" << stats.qkv_copy_us
              << " ffn_gpu_us=" << stats.ffn_gpu_us
              << " ffn_copy_us=" << stats.ffn_copy_us
              << " qkv_wait_us=" << stats.qkv_wait_us
              << " ffn_wait_us=" << stats.ffn_wait_us
              << " fallbacks=" << stats.fallbacks << "\n";
    
    zen::d3d12::trace_report();
    
    return 0;
}