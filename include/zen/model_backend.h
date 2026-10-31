#pragma once
#include "zen/model_runtime.h"
#include "zen/openai_server.h"
#include <filesystem>
#include <memory>
#include <string>
namespace zen::runtime {
// Generic HTTP-facing backend. It owns an architecture-selected model adapter.
class ModelBackend final : public http::InferenceBackend {
public:
    bool load(const std::filesystem::path& path,std::string model_id={},std::string* error=nullptr,
              const std::filesystem::path& draft_path={});
    std::string model_id()const override{return model_id_;}
    http::GenerationResult generate(const http::GenerationRequest& request,
        const std::function<void(const std::string&)>& on_token,
        const std::atomic<bool>& cancelled)override;
private:
    std::unique_ptr<ModelRuntime> runtime_;
    std::unique_ptr<ModelRuntime> draft_runtime_;
    std::string model_id_;
};
}
