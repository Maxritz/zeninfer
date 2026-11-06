#include "zen/model_runtime.h"
#include "zen/dense_transformer_runtime.h"
#include "zen/gguf.h"
#include <utility>

namespace zen::runtime {
namespace {
class DenseTransformerAdapter final : public ModelRuntime {
public:
    bool load(const std::filesystem::path& p, std::string* e) override { return runtime_.load(p,e); }
    bool loaded() const noexcept override { return runtime_.loaded(); }
    bool tokenizer_available() const noexcept override { return runtime_.tokenizer_available(); }
    const RuntimeConfig& config() const noexcept override { return runtime_.config(); }
    const std::string& architecture_name() const noexcept override { return runtime_.architecture_name(); }
    const std::string& chat_template() const noexcept override { return runtime_.chat_template(); }
    bool tokenize(const std::string& s,std::vector<std::uint32_t>& ids) const override { return runtime_.tokenize(s,ids); }
    bool detokenize(const std::vector<std::uint32_t>& ids,std::string& s) const override { return runtime_.detokenize(ids,s); }
    bool generate(const std::vector<std::uint32_t>& p,std::uint32_t n,Generation& g,std::string* e,
        const std::function<void(const std::vector<std::uint32_t>&)>& cb,const GenerationOptions& o,
        const std::atomic<bool>* c) const override { return runtime_.generate(p,n,g,e,cb,o,c); }
    bool generate_text(const std::string& p,std::uint32_t n,Generation& g,std::string& t,std::string* e,
        const std::function<void(const std::string&)>& cb,const GenerationOptions& o,const std::atomic<bool>* c) const override {
        return runtime_.generate_text(p,n,g,t,e,cb,o,c);
    }
    bool generate_speculative(const std::vector<std::uint32_t>& p,std::uint32_t n,const ModelRuntime& d,
        std::uint32_t k,Generation& g,std::string* e,const std::function<void(const std::vector<std::uint32_t>&)>& cb,
        const GenerationOptions& o,const std::atomic<bool>* c) const override {
        const auto* draft=dynamic_cast<const DenseTransformerAdapter*>(&d);
        if(!draft){if(e)*e="Draft and target models must use compatible executable adapters";return false;}
        return runtime_.generate_speculative(p,n,draft->runtime_,k,g,e,cb,o,c);
    }
private:
    DenseTransformerRuntime runtime_;
};
}

std::unique_ptr<ModelRuntime> create_model_runtime(const model::ModelConfig& config,std::string* error) {
    // Dispatch uses the full GGUF architecture string, not the coarse family enum.
    // This prevents Qwen3/MoE from accidentally entering the Qwen2 dense path.
    if (config.architecture==model::Architecture::llama ||
        (config.architecture==model::Architecture::qwen && (config.architecture_name=="qwen2" || config.architecture_name=="qwen3"))) {
        return std::make_unique<DenseTransformerAdapter>();
    }
    if (config.architecture==model::Architecture::bitnet) {
        return std::make_unique<DenseTransformerAdapter>();
    }
    if (config.architecture==model::Architecture::qwen_hybrid) {
        return std::make_unique<DenseTransformerAdapter>();
    }
    if(error) *error="No executable model adapter is registered for GGUF architecture '"+
        (config.architecture_name.empty()?std::string("unknown"):config.architecture_name)+"'";
    return {};
}

std::unique_ptr<ModelRuntime> load_model_runtime(const std::filesystem::path& path,
                                                  model::Architecture* architecture,std::string* error) {
    const auto report=gguf::inspect(path);
    if(!report.ok){if(error)*error=report.error;return {};}
    if(architecture)*architecture=report.config.architecture;
    auto runtime=create_model_runtime(report.config,error);
    if(!runtime)return {};
    if(!runtime->load(path,error))return {};
    return runtime;
}
}
