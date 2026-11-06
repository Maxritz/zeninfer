#include "zen/dense_transformer_runtime.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using zen::runtime::DenseTransformerRuntime;
struct Cache {
    std::vector<std::vector<float>> k,v,conv,state;
    explicit Cache(const zen::runtime::RuntimeConfig& c,std::size_t positions){
        const auto kv=positions*c.kv_heads*c.head_dim;
        const auto conv_dim=std::size_t(c.ssm_state)*(2*c.ssm_n_group+c.ssm_dt_rank);
        k.assign(c.layers,std::vector<float>(kv));v=k;
        conv.assign(c.layers,std::vector<float>((c.ssm_d_conv-1+positions)*conv_dim));
        state.assign(c.layers,std::vector<float>(std::size_t(c.ssm_dt_rank)*c.ssm_state*c.ssm_state));
    }
};
int main(int argc,char**argv){
    if(argc!=2){std::cerr<<"usage: q35_incremental_tests fixture.gguf\n";return 2;}
    DenseTransformerRuntime m;std::string err;if(!m.load(argv[1],&err)){std::cerr<<err<<'\n';return 1;}
    const auto& c=m.config();if(!c.qwen35_variant)return 1;
    const std::vector<std::uint32_t> tokens={11,51,30,25,17,42,99,7};
    float worst=0;
    for(std::size_t end=1;end<=tokens.size();++end){
        Cache full(c,tokens.size());std::vector<float> want;
        if(!m.forward_q35_batch({tokens.begin(),tokens.begin()+end},full.k,full.v,full.conv,full.state,want))return 1;
        for(std::size_t chunk:{1u,2u,3u,4u}){
            Cache split(c,tokens.size());std::vector<float> got;
            for(std::size_t start=0;start<end;){
                const auto n=std::min(chunk,end-start);
                if(!m.forward_q35_batch({tokens.begin()+start,tokens.begin()+start+n},split.k,split.v,split.conv,split.state,got,static_cast<std::uint32_t>(start))){std::cerr<<"forward failed\n";return 1;}
                start+=n;
            }
            if(got.size()!=want.size())return 1;
            float maxdiff=0,maxref=0;
            for(std::size_t i=0;i<want.size();++i){maxdiff=std::max(maxdiff,std::abs(got[i]-want[i]));maxref=std::max(maxref,std::abs(want[i]));}
            worst=std::max(worst,maxdiff);
            if(!std::isfinite(maxdiff)||maxdiff>0.002f+maxref*0.0002f){std::cerr<<"chunk parity failed end="<<end<<" chunk="<<chunk<<" max="<<maxdiff<<" refmax="<<maxref<<'\n';return 1;}
        }
    }
    // Position zero starts a new sequence even when buffers are reused.
    {Cache reused(c,tokens.size()), fresh(c,tokens.size());std::vector<float> a,b;
     if(!m.forward_q35_batch(tokens,reused.k,reused.v,reused.conv,reused.state,a) ||
        !m.forward_q35_batch({tokens[0]},reused.k,reused.v,reused.conv,reused.state,a) ||
        !m.forward_q35_batch({tokens[0]},fresh.k,fresh.v,fresh.conv,fresh.state,b) || a!=b){std::cerr<<"new sequence cache reset failed\n";return 1;}
     reused.state[0].clear();
     if(m.forward_q35_batch({tokens[0]},reused.k,reused.v,reused.conv,reused.state,a)){std::cerr<<"invalid recurrent state accepted\n";return 1;}}
    // A cache entry created for a short generation must grow for a longer
    // request with the same prompt; compare the result with a cold runtime.
    std::vector<std::uint32_t> prompt={11};zen::runtime::Generation warm,prime,cold;
    zen::runtime::GenerationOptions constrained;constrained.allowed_token_ids={11,17,25,30,51};
    if(!m.generate(prompt,0,prime,&err)||!m.generate(prompt,6,warm,&err,{},constrained)){std::cerr<<"warm generation failed: "<<err<<'\n';return 1;}
    DenseTransformerRuntime other;if(!other.load(argv[1],&err)||!other.generate(prompt,6,cold,&err,{},constrained)){std::cerr<<"cold generation failed: "<<err<<'\n';return 1;}
    if(!warm.cache_hit||warm.token_ids.size()!=6||warm.token_ids!=cold.token_ids||warm.last_logits.size()!=cold.last_logits.size()){
        std::cerr<<"prefix-cache warm/cold token mismatch\n";return 1;
    }
    for(std::size_t i=0;i<cold.last_logits.size();++i)if(std::abs(warm.last_logits[i]-cold.last_logits[i])>0.002f+std::abs(cold.last_logits[i])*0.0002f){std::cerr<<"prefix-cache logits mismatch\n";return 1;}
    // Validate generated token decisions against full prefix recomputation,
    // not merely the same cached path run twice.
    auto seq=prompt;
    for(std::size_t step=0;step<warm.token_ids.size();++step){
        Cache fresh(c,prompt.size()+warm.token_ids.size());std::vector<float> scores;
        if(!m.forward_q35_batch(seq,fresh.k,fresh.v,fresh.conv,fresh.state,scores))return 1;
        auto best=constrained.allowed_token_ids.front();
        for(auto id:constrained.allowed_token_ids)if(scores[id]>scores[best])best=id;
        if(best!=warm.token_ids[step]){std::cerr<<"greedy generation differs from full recomputation at step="<<step<<'\n';return 1;}
        seq.push_back(best);
    }
    std::cout<<"Qwen3.5 split/chunk and growing prefix-cache parity: PASS, max_abs="<<worst<<'\n';return 0;
}
