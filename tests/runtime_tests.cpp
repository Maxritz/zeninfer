#include "zen/model_runtime.h"
#include <atomic>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

int main(int argc,char**argv){
    if(argc<2||argc>3){std::cerr<<"usage: zen_runtime_tests model.gguf [draft.gguf]\n";return 2;}
    std::string error;auto runtime=zen::runtime::load_model_runtime(argv[1],nullptr,&error);
    if(!runtime){std::cerr<<"generic adapter load failed: "<<error<<"\n";return 1;}
    std::unique_ptr<zen::runtime::ModelRuntime> draft_runtime;const zen::runtime::ModelRuntime* draft=runtime.get();if(argc==3){draft_runtime=zen::runtime::load_model_runtime(argv[2],nullptr,&error);if(!draft_runtime){std::cerr<<"draft adapter load failed: "<<error<<"\n";return 1;}draft=draft_runtime.get();}
    if(!runtime->tokenizer_available()){std::cerr<<"tokenizer unavailable\n";return 1;}
    std::vector<std::uint32_t> ids;
    if(!runtime->tokenize("Once upon a time",ids)||ids.empty()){std::cerr<<"tokenization failed\n";return 1;}
    // Golden IDs generated independently with Google's SentencePiece 0.2.1
    // from the tinyllama-15M-stories GGUF vocabulary/scores (nmt_nfkc normaliser).
    // Run only for that fixture; the other supplied GGUF has a different tokenizer.
    const auto fixture_name=std::filesystem::path(argv[1]).filename().string();
    if(fixture_name.find("tinyllama-15M-stories")!=std::string::npos){
        struct TokenCase { const char* text; std::vector<std::uint32_t> expected; };
        const std::vector<TokenCase> sentencepiece_cases{
            {"Once upon a time",{1,9038,2501,263,931}},
            {"Hello, world!",{1,15043,29892,3186,29991}},
            {"The quick brown fox jumps over the lazy dog.",{1,450,4996,17354,29871,8944,432,17204,975,278,17366,11203,29889}},
            {"ROCm RDNA4 Vulkan compute",{1,390,20166,29885,390,29928,3521,29946,478,352,11052,10272}},
            {"Once upon a time there was a little girl.",{1,9038,2501,263,931,727,471,263,2217,7826,29889}}
        };
        for(const auto& tc:sentencepiece_cases){std::vector<std::uint32_t> got;if(!runtime->tokenize(tc.text,got)||got!=tc.expected){
            std::cerr<<"independent SentencePiece token parity failed for: "<<tc.text<<"\nexpected=";
            for(auto id:tc.expected)std::cerr<<id<<",";std::cerr<<" got=";for(auto id:got)std::cerr<<id<<",";std::cerr<<"\n";return 1;}}
    }
    std::string roundtrip;if(!runtime->detokenize(ids,roundtrip)||roundtrip.empty()){std::cerr<<"tokenizer round-trip failed\n";return 1;}
    // Report real end-to-end model prefill and decode-forward throughput before
    // any repeated-prompt prefix-cache hit can affect the measurement. A 128-token
    // deterministic prompt avoids reporting a misleading 5-token prefill rate.
    std::vector<std::uint32_t> perf_prompt;perf_prompt.reserve(128);
    for(std::size_t i=0;i<128;++i)perf_prompt.push_back(ids[i%ids.size()]);
    auto warm_prompt=perf_prompt;warm_prompt.back()=(warm_prompt.back()+97u)%runtime->config().vocab;
    zen::runtime::Generation warmup;
    if(!runtime->generate(warm_prompt,1,warmup,&error)){std::cerr<<"performance warmup failed: "<<error<<"\n";return 1;}
    constexpr std::size_t benchmark_runs=5;
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
    zen::runtime::GenerationOptions options;options.temperature=0.85f;options.top_k=16;options.top_p=0.92f;options.seed=123456;
    zen::runtime::Generation a,b;std::string ta,tb;
    if(!runtime->generate_text("Once upon a time",4,a,ta,&error,{},options)){std::cerr<<"sample generation failed: "<<error<<"\n";return 1;}
    if(!runtime->generate_text("Once upon a time",4,b,tb,&error,{},options)){std::cerr<<"repeat generation failed: "<<error<<"\n";return 1;}
    if(a.token_ids!=b.token_ids){std::cerr<<"seeded sampling is not deterministic\n";return 1;}
    if(!b.cache_hit){std::cerr<<"identical prompt did not reuse the KV prefix cache\n";return 1;}
    for(float v:a.last_logits)if(!std::isfinite(v)){std::cerr<<"non-finite logits\n";return 1;}
    zen::runtime::GenerationOptions constrained;constrained.temperature=0.0f;constrained.allowed_token_ids={42};zen::runtime::Generation constrained_result;std::string constrained_text;
    if(!runtime->generate_text("Once upon a time",1,constrained_result,constrained_text,&error,{},constrained)||constrained_result.token_ids.size()!=1||constrained_result.token_ids[0]!=42){std::cerr<<"allowed-token constraint failed: "<<error<<"\n";return 1;}
    zen::runtime::GenerationOptions biased;biased.temperature=0.0f;biased.logit_bias[42]=100.0f;zen::runtime::Generation biased_result;std::string biased_text;
    if(!runtime->generate_text("Once upon a time",1,biased_result,biased_text,&error,{},biased)||biased_result.token_ids.size()!=1||biased_result.token_ids[0]!=42){std::cerr<<"logit-bias filter failed: "<<error<<"\n";return 1;}
    zen::runtime::GenerationOptions greedy;greedy.temperature=0.0f;
    zen::runtime::Generation ordinary, speculative;std::string spec_error;
    if(!runtime->generate(ids,4,ordinary,&spec_error,{},greedy)){std::cerr<<"ordinary greedy generation failed: "<<spec_error<<"\n";return 1;}
    if(!runtime->generate_speculative(ids,4,*draft,3,speculative,&spec_error,{},greedy)){std::cerr<<"speculative generation failed: "<<spec_error<<"\n";return 1;}
    if(ordinary.token_ids!=speculative.token_ids){std::cerr<<"speculative greedy output differs from target-only output\n";return 1;}
    if(speculative.speculative_proposed==0){std::cerr<<"speculative path silently fell back instead of verifying draft tokens\n";return 1;}
    // Stochastic speculative decoding must run the p/q acceptance and residual
    // correction path rather than silently falling back to ordinary decoding.
    zen::runtime::Generation stochastic_spec_a,stochastic_spec_b;
    zen::runtime::GenerationOptions stochastic_spec_options=options;stochastic_spec_options.seed=987654321;
    if(!runtime->generate_speculative(ids,8,*draft,3,stochastic_spec_a,&spec_error,{},stochastic_spec_options)){
        std::cerr<<"stochastic speculative generation failed: "<<spec_error<<"\n";return 1;
    }
    if(!runtime->generate_speculative(ids,8,*draft,3,stochastic_spec_b,&spec_error,{},stochastic_spec_options)){
        std::cerr<<"repeat stochastic speculative generation failed: "<<spec_error<<"\n";return 1;
    }
    if(stochastic_spec_a.token_ids!=stochastic_spec_b.token_ids||stochastic_spec_a.speculative_proposed==0){
        std::cerr<<"stochastic speculative generation is not deterministic or did not verify proposals\n";return 1;
    }
    for(auto id:stochastic_spec_a.token_ids)if(id>=runtime->config().vocab){std::cerr<<"stochastic speculative token outside vocabulary\n";return 1;}
    zen::runtime::Generation g;std::string gt;
    if(!runtime->generate_text("Once upon a time",2,g,gt,&error,{},greedy)){std::cerr<<"greedy generation failed: "<<error<<"\n";return 1;}
    zen::runtime::GenerationOptions stopped=options;if(!ta.empty())stopped.stop={ta};
    zen::runtime::Generation stop_result;std::string stop_text;
    if(!runtime->generate_text("Once upon a time",4,stop_result,stop_text,&error,{},stopped)){std::cerr<<"stop-sequence generation failed: "<<error<<"\n";return 1;}
    if(!ta.empty()&&!stop_text.empty()){std::cerr<<"stop sequence was not removed from output\n";return 1;}
    std::atomic<bool> cancelled{true};zen::runtime::Generation c;std::string ct;
    if(runtime->generate_text("Once upon a time",2,c,ct,&error,{},options,&cancelled)){std::cerr<<"pre-cancelled generation unexpectedly succeeded\n";return 1;}
    std::atomic<bool> mid_cancel{false};zen::runtime::Generation mid;std::string mid_text;
    const auto cancel_after_one=[&](const std::vector<std::uint32_t>& generated){if(generated.size()==1)mid_cancel.store(true);};
    if(runtime->generate(ids,4,mid,&error,cancel_after_one,options,&mid_cancel)){std::cerr<<"mid-generation cancellation was ignored\n";return 1;}
    std::cout<<"Runtime fixture PASS: "<<argv[1]<<" tokens="<<ids.size()<<" sample_tokens="<<a.token_ids.size()<<"\n";
    return 0;
}
