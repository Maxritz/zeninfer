#include "zen/model_backend.h"
#include "zen/chat_template.h"
#include <stdexcept>
#include <algorithm>
namespace zen::runtime {
bool ModelBackend::load(const std::filesystem::path& path,std::string id,std::string* error,const std::filesystem::path& draft_path){
    runtime_=load_model_runtime(path,nullptr,error);if(!runtime_)return false;
    if(!runtime_->tokenizer_available()){if(error)*error="Model has no supported tokenizer vocabulary";return false;}
    draft_runtime_.reset();if(!draft_path.empty()){draft_runtime_=load_model_runtime(draft_path,nullptr,error);if(!draft_runtime_)return false;if(!draft_runtime_->tokenizer_available()){if(error)*error="Draft model has no supported tokenizer vocabulary";return false;}}
    model_id_=id.empty()?path.stem().string():std::move(id);
    if(model_id_.empty())model_id_="zeninfer";
    return true;
}
http::GenerationResult ModelBackend::generate(const http::GenerationRequest& req,
        const std::function<void(const std::string&)>& on_token,const std::atomic<bool>& cancelled){
    std::string prompt=req.prompt;
    if(prompt.empty()){
        std::string template_error;
        if(!chat::format_legacy_chat(runtime_->chat_template(),runtime_->architecture_name(),req.messages,prompt,&template_error))
            throw std::runtime_error(template_error.empty()?"Chat messages could not be formatted":template_error);
    }
    if(prompt.empty())throw std::runtime_error("The request contains no text prompt");
    std::vector<std::uint32_t> prompt_ids;
    if(!runtime_->tokenize(prompt,prompt_ids))throw std::runtime_error("Model tokenizer is unavailable");
    runtime::Generation generated;std::string text,error;
    auto stream=[&](const std::string& piece){if(cancelled.load(std::memory_order_relaxed))throw std::runtime_error("Generation cancelled");if(on_token)on_token(piece);};
    GenerationOptions options;options.temperature=static_cast<float>(req.temperature);options.top_k=req.top_k;options.top_p=static_cast<float>(req.top_p);options.seed=req.seed;options.stop=req.stop;options.logit_bias=req.logit_bias;options.allowed_token_ids=req.allowed_token_ids;
    if(req.speculative_tokens){
        if(!draft_runtime_)throw std::runtime_error("speculative_tokens requires a draft model loaded with --draft-model");
        std::string streamed;auto token_stream=[&](const std::vector<std::uint32_t>& ids){if(!on_token)return;std::string current;if(!runtime_->detokenize(ids,current))return;std::size_t safe_end=current.size(),hold=0,stop_at=std::string::npos;for(const auto& stop:options.stop)if(!stop.empty()){hold=std::max(hold,stop.size());const auto at=current.find(stop);if(at!=std::string::npos)stop_at=std::min(stop_at,at);}if(stop_at!=std::string::npos)safe_end=stop_at;else if(hold>1)safe_end=current.size()>hold-1?current.size()-(hold-1):0;if(safe_end>streamed.size()){on_token(current.substr(streamed.size(),safe_end-streamed.size()));streamed=current.substr(0,safe_end);}};
        if(!runtime_->generate_speculative(prompt_ids,req.max_tokens,*draft_runtime_,req.speculative_tokens,generated,&error,on_token?token_stream:std::function<void(const std::vector<std::uint32_t>&)>{},options,&cancelled))throw std::runtime_error(error.empty()?"Speculative generation failed":error);
        if(!runtime_->detokenize(generated.token_ids,text))throw std::runtime_error("Generated token decoding failed");
        for(const auto& stop:options.stop)if(!stop.empty()){const auto at=text.find(stop);if(at!=std::string::npos)text.resize(at);}
        if(on_token&&text.size()>streamed.size())on_token(text.substr(streamed.size()));
    }else if(!runtime_->generate_text(prompt,req.max_tokens,generated,text,&error,on_token?stream:std::function<void(const std::string&)>{},options,&cancelled))
        throw std::runtime_error(error.empty()?"Model generation failed":error);
    http::GenerationResult result;result.text=std::move(text);
    result.prompt_tokens=static_cast<std::uint32_t>(prompt_ids.size());
    result.completion_tokens=static_cast<std::uint32_t>(generated.token_ids.size());
    result.finish_reason=(generated.stopped_on_eos||generated.stopped_on_stop)?"stop":"length";
    return result;
}
}
