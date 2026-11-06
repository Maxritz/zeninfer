#include "zen/chat_compaction.h"
#include "zen/model_backend.h"
#include <algorithm>
#include <atomic>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

class FakeModel final : public zen::runtime::ModelRuntime {
public:
    zen::runtime::RuntimeConfig cfg;
    std::string architecture{"qwen2"},tmpl{"<|im_start|>"};
    mutable unsigned summaries{}, answers{};mutable std::string last_summary_prompt,last_answer_prompt;
    bool fail_summary{},truncate{},empty{},whitespace{};
    FakeModel(){cfg.context=8192;}
    bool load(const std::filesystem::path&,std::string*) override{return true;}
    bool loaded()const noexcept override{return true;}
    bool tokenizer_available()const noexcept override{return true;}
    const zen::runtime::RuntimeConfig& config()const noexcept override{return cfg;}
    const std::string& architecture_name()const noexcept override{return architecture;}
    const std::string& chat_template()const noexcept override{return tmpl;}
    bool tokenize(const std::string& text,std::vector<std::uint32_t>& ids)const override{ids.assign(text.size(),1);return true;}
    bool detokenize(const std::vector<std::uint32_t>&,std::string&)const override{return true;}
    bool generate(const std::vector<std::uint32_t>&,std::uint32_t,zen::runtime::Generation&,std::string*,const std::function<void(const std::vector<std::uint32_t>&)>&,const zen::runtime::GenerationOptions&,const std::atomic<bool>*)const override{return false;}
    bool generate_text(const std::string& prompt,std::uint32_t max,zen::runtime::Generation& gen,std::string& text,std::string*,const std::function<void(const std::string&)>& on_token,const zen::runtime::GenerationOptions& options,const std::atomic<bool>*)const override{
        if(prompt.find("Conversation history to summarize:")==std::string::npos){
            ++answers;last_answer_prompt=prompt;text="answer";gen.token_ids.assign(2,1);gen.stopped_on_eos=true;if(on_token)on_token(text);return true;
        }
        ++summaries;last_summary_prompt=prompt;
        if(fail_summary||on_token||options.temperature!=0.0f||!options.stop.empty()||!options.allowed_token_ids.empty())return false;
        if(empty){text.clear();return true;}
        if(whitespace){text=" \n \t";return true;}
        text="The user wants blue shoes under $25.";gen.token_ids.assign(truncate?max:3,1);gen.stopped_on_eos=!truncate;return true;
    }
    bool generate_speculative(const std::vector<std::uint32_t>&,std::uint32_t,const zen::runtime::ModelRuntime&,std::uint32_t,zen::runtime::Generation&,std::string*,const std::function<void(const std::vector<std::uint32_t>&)>&,const zen::runtime::GenerationOptions&,const std::atomic<bool>*)const override{return false;}
};

int main(){
    int failures=0;auto check=[&](bool ok,const char* name){if(!ok){std::cerr<<"FAIL "<<name<<'\n';++failures;}};
    FakeModel model;std::atomic<bool> cancel{false};zen::http::GenerationRequest req;
    req.max_tokens=32;req.compaction.enabled=true;req.compaction.trigger_tokens=600;req.compaction.target_tokens=550;req.compaction.summary_max_tokens=64;req.compaction.keep_last_messages=2;
    req.messages={{"system","Always be helpful."},{"system","Preserve this rule."},{"user",std::string(700,'X')+" old user goal"},{"assistant","old decision"},{"user","recent user"},{"assistant","recent assistant"}};
    const auto original=req.messages;std::string prompt;std::vector<std::uint32_t> ids;zen::http::CompactionInfo info;
    try {
        zen::chat::prepare_compacted_chat(model,req,cancel,prompt,ids,info);
        check(info.requested&&info.applied&&info.compacted_messages==2,"compaction metadata");
        check(info.original_prompt_tokens>600&&info.final_prompt_tokens==ids.size()&&ids.size()<=550,"token accounting");
        check(info.summary_prompt_tokens>0&&info.summary_completion_tokens==3&&info.summary.find("blue shoes")!=std::string::npos,"summary accounting");
        check(prompt.find("Always be helpful.")!=std::string::npos&&prompt.find("Preserve this rule.")!=std::string::npos,"system preserved");
        check(prompt.find("recent user")!=std::string::npos&&prompt.find("recent assistant")!=std::string::npos,"recent turns preserved");
        check(prompt.find("old user goal")==std::string::npos&&prompt.find("old decision")==std::string::npos,"old turns replaced");
        check(prompt.find("generated, potentially incomplete summary")!=std::string::npos,"summary labelled");
        check(model.last_summary_prompt.find("old user goal")!=std::string::npos&&model.last_summary_prompt.find("recent user")==std::string::npos,"only older turns summarized");
        check(req.messages.size()==original.size()&&req.messages[2].content==original[2].content,"input unchanged");
    }catch(const std::exception& e){std::cerr<<"Unexpected "<<e.what()<<'\n';++failures;}
    auto expect_error=[&](const char* name){try{zen::chat::prepare_compacted_chat(model,req,cancel,prompt,ids,info);std::cerr<<"FAIL "<<name<<" accepted\n";++failures;}catch(const zen::chat::CompactionError&){}catch(...){std::cerr<<"FAIL "<<name<<" unexpected exception\n";++failures;}};
    req.compaction.trigger_tokens=4000;req.compaction.target_tokens=1000;const auto before=model.summaries;
    try{zen::chat::prepare_compacted_chat(model,req,cancel,prompt,ids,info);check(!info.applied&&info.requested&&model.summaries==before&&prompt.find("old user goal")!=std::string::npos,"below threshold no change");}catch(...){++failures;}
    req.compaction.trigger_tokens=600;req.compaction.target_tokens=550;
    model.cfg.context=500;expect_error("summary does not fit context");model.cfg.context=8192;
    req.compaction.target_tokens=20;expect_error("target too small");req.compaction.target_tokens=550;
    model.truncate=true;expect_error("truncated summary");model.truncate=false;
    model.empty=true;expect_error("empty summary");model.empty=false;
    model.whitespace=true;expect_error("whitespace-only summary");model.whitespace=false;
    req.max_tokens=8192;expect_error("final generation context budget");req.max_tokens=32;
    model.fail_summary=true;expect_error("failed summary");model.fail_summary=false;
    req.compaction.keep_last_messages=5;expect_error("nothing eligible");req.compaction.keep_last_messages=2;
    req.messages.insert(req.messages.begin()+3,{"system","late rule"});expect_error("interleaved system");req.messages=original;
    cancel=true;expect_error("cancellation");cancel=false;
    req.compaction.target_tokens=601;expect_error("invalid limits");
    // Exercise the real HTTP model adapter using the same fake runtime.
    {
        auto fake=std::make_unique<FakeModel>();auto* observed=fake.get();
        zen::runtime::ModelBackend backend(std::move(fake),"test");
        req.messages=original;req.compaction.target_tokens=550;std::string streamed;
        try {
            auto result=backend.generate(req,[&](const std::string& text){streamed+=text;},cancel);
            check(result.compaction.applied&&result.text=="answer"&&streamed=="answer","backend integration and no summary leak");
            check(observed->summaries==1&&observed->answers==1&&observed->last_answer_prompt.find("recent user")!=std::string::npos,"backend compaction flow");
            check(result.prompt_tokens==result.compaction.final_prompt_tokens&&result.completion_tokens==2,"backend usage");
            req.compaction.enabled=false;streamed.clear();
            auto plain=backend.generate(req,[&](const std::string& text){streamed+=text;},cancel);
            check(!plain.compaction.requested&&observed->summaries==1&&observed->answers==2&&observed->last_answer_prompt.find("old user goal")!=std::string::npos,"default backend unchanged");
        }catch(const std::exception&e){std::cerr<<"FAIL backend integration "<<e.what()<<'\n';++failures;}
    }
    std::cout<<"Compaction unit failures="<<failures<<'\n';return failures?1:0;
}
