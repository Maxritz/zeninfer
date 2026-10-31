#pragma once
#include "zen/gguf.h"
#include "zen/quant.h"
#include "zen/runtime_types.h"
#include <memory>
#include <atomic>
#include <mutex>
#include <unordered_set>
#include <unordered_map>
#include "zen/tokenizer.h"
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace zen::runtime {
class MappedFile;
class DenseTransformerRuntime {
public:
    bool load(const std::filesystem::path& path, std::string* error=nullptr);
    bool loaded() const noexcept { return loaded_; }
    bool tokenizer_available() const noexcept { return tokenizer_loaded_; }
    const RuntimeConfig& config() const noexcept { return config_; }
    const std::string& architecture_name() const noexcept { return architecture_name_; }
    const std::string& chat_template() const noexcept { return chat_template_; }
    bool generate(const std::vector<std::uint32_t>& prompt, std::uint32_t max_new_tokens,
                  Generation& result, std::string* error=nullptr,
                  const std::function<void(const std::vector<std::uint32_t>&)>& on_tokens={},
                  const GenerationOptions& options={},const std::atomic<bool>* cancelled=nullptr) const;
    bool generate_speculative(const std::vector<std::uint32_t>& prompt,std::uint32_t max_new,const DenseTransformerRuntime& draft,std::uint32_t draft_tokens,Generation& result,std::string* error=nullptr,const std::function<void(const std::vector<std::uint32_t>&)>& on_tokens={},const GenerationOptions& options={},const std::atomic<bool>* cancelled=nullptr) const;
    bool tokenize(const std::string& text,std::vector<std::uint32_t>& ids)const;
    bool detokenize(const std::vector<std::uint32_t>& ids,std::string& text)const;
    bool generate_text(const std::string& prompt,std::uint32_t max_new_tokens,
                       Generation& result,std::string& text,std::string* error=nullptr,
                       const std::function<void(const std::string&)>& on_text={},
                       const GenerationOptions& options={},const std::atomic<bool>* cancelled=nullptr)const;
public:
    struct Tensor { std::vector<std::uint64_t> dims; std::vector<float> values; const std::uint8_t* raw{}; std::size_t raw_bytes{}; const quant::Format* format{}; std::size_t elements{}; };
private:
    std::shared_ptr<MappedFile> mapping_;
    bool prefetch_enabled_{};
    struct PrefixCache { std::vector<std::uint32_t> tokens; std::vector<std::vector<float>> key,value; std::vector<float> logits;
                         // qwen35 hybrid: per-GDN-layer conv window rows and SSM states.
                         std::vector<std::vector<float>> conv, ssm; };
    mutable std::mutex prefix_cache_mutex_;
    mutable std::shared_ptr<const PrefixCache> prefix_cache_;
    // Hadamard-fold transform (PrismML): normalized block Sylvester-Walsh-
    // Hadamard plus explicit sign vectors. hadamard_rot_[i][j] over a
    // block_size block: parity(i&j) ? -1/s : +1/s with s=sqrt(block_size).
    struct Hadamard {
        std::uint32_t block_size{};
        std::vector<float> rot;              // [block_size*block_size]
        std::unordered_map<std::size_t, std::vector<float>> signs; // input width -> +/-1
        std::unordered_set<std::string> folded_names;  // weight names folded at quantisation
        bool tied_output{};
        // Applies signs then rotation along the last dimension (input transform).
        void transform(float* x, std::size_t width, std::vector<float>& scratch) const;
        // Rotation only, then signs (inverse-table restore order).
        void unrotate(float* x, std::size_t width, std::vector<float>& scratch) const;
    };
    Hadamard hadamard_;
    std::filesystem::path path_;
    RuntimeConfig config_;
    std::string architecture_name_,chat_template_;
    std::unordered_map<std::string,Tensor> tensors_;
    tokenizer::ModelTokenizer tokenizer_;
    bool tokenizer_loaded_{};
    bool loaded_{};
    // qwen35 helpers.
    bool layer_is_recurrent(std::uint32_t l) const noexcept;
    void hadamard_transform_input(float* x,std::size_t width,std::vector<float>& scratch) const;
    void hadamard_unrotate(float* x,std::size_t width,std::vector<float>& scratch) const;
    bool hadamard_folded(const char* suffix,std::size_t layer) const noexcept;
    const Tensor* tensor(const std::string& name) const noexcept;
    void prefetch_layer(std::uint32_t layer) const noexcept;
    bool forward_batch(const std::vector<std::uint32_t>& tokens,
                 std::vector<std::vector<float>>& key_cache,
                 std::vector<std::vector<float>>& value_cache,
                 std::vector<float>& logits,std::uint32_t start_pos=0,
                 std::size_t tail_count=1,std::vector<float>* tail_logits=nullptr) const;
    bool forward(std::uint32_t token, std::uint32_t position,
                 std::vector<std::vector<float>>& key_cache,
                 std::vector<std::vector<float>>& value_cache,
                 std::vector<float>& logits) const;
    bool forward_q35_batch(const std::vector<std::uint32_t>& tokens,
                 std::vector<std::vector<float>>& key_cache,
                 std::vector<std::vector<float>>& value_cache,
                 std::vector<std::vector<float>>& conv_cache,
                 std::vector<std::vector<float>>& ssm_cache,
                 std::vector<float>& logits,std::uint32_t start_pos=0) const;
    bool forward_q35_token(std::uint32_t token, std::uint32_t position,
                 std::vector<std::vector<float>>& key_cache,
                 std::vector<std::vector<float>>& value_cache,
                 std::vector<std::vector<float>>& conv_cache,
                 std::vector<std::vector<float>>& ssm_cache,
                 std::vector<float>& logits) const;
};
using LlamaRuntime = DenseTransformerRuntime; // compatibility alias for the original public API.
}
