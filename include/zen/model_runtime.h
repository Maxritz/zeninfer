#pragma once
#include "zen/runtime_types.h"
#include "zen/model_ir.h"
#include <filesystem>
#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace zen::runtime {
// Architecture-neutral contract consumed by frontends and services.
// Implementations live in model adapters; the core never assumes a tensor layout.
class ModelRuntime {
public:
    virtual ~ModelRuntime() = default;
    virtual bool load(const std::filesystem::path&, std::string* error=nullptr) = 0;
    virtual bool loaded() const noexcept = 0;
    virtual bool tokenizer_available() const noexcept = 0;
    virtual const RuntimeConfig& config() const noexcept = 0;
    virtual const std::string& architecture_name() const noexcept = 0;
    virtual const std::string& chat_template() const noexcept = 0;
    virtual bool tokenize(const std::string&, std::vector<std::uint32_t>&) const = 0;
    virtual bool detokenize(const std::vector<std::uint32_t>&, std::string&) const = 0;
    virtual bool generate(const std::vector<std::uint32_t>&, std::uint32_t, Generation&,
        std::string* = nullptr, const std::function<void(const std::vector<std::uint32_t>&)>& = {},
        const GenerationOptions& = {}, const std::atomic<bool>* = nullptr) const = 0;
    virtual bool generate_text(const std::string&, std::uint32_t, Generation&, std::string&,
        std::string* = nullptr, const std::function<void(const std::string&)>& = {},
        const GenerationOptions& = {}, const std::atomic<bool>* = nullptr) const = 0;
    virtual bool generate_speculative(const std::vector<std::uint32_t>&, std::uint32_t,
        const ModelRuntime&, std::uint32_t, Generation&, std::string*,
        const std::function<void(const std::vector<std::uint32_t>&)>& = {},
        const GenerationOptions& = {}, const std::atomic<bool>* = nullptr) const = 0;
};

// The factory dispatches by GGUF architecture. A registered name is not enough:
// only adapters with an executable factory are returned.
std::unique_ptr<ModelRuntime> create_model_runtime(const model::ModelConfig& config,
                                                   std::string* error=nullptr);
std::unique_ptr<ModelRuntime> load_model_runtime(const std::filesystem::path& path,
                                                 model::Architecture* architecture=nullptr,
                                                 std::string* error=nullptr);
}
