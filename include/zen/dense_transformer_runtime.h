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
    ~DenseTransformerRuntime();
    bool load(const std::filesystem::path& path, std::string* error=nullptr);
    bool loaded() const noexcept { return loaded_; }
    bool tokenizer_available() const noexcept { return tokenizer_loaded_; }
    const RuntimeConfig& config() const noexcept { return config_; }
    const std::string& architecture_name() const noexcept { return architecture_name_; }
    const std::string& chat_template() const noexcept { return chat_template_; }
    bool generate(const std::vector<std::uint32_t>& prompt, std::uint32_t max_new_tokens,
                  Generation& result, std::string* error=nullptr,
                  const std::function<void(const std::vector<std::uint32_t>&)>& on_tokens={},
                  const GenerationOptions& options={}, const std::atomic<bool>* cancelled=nullptr) const;
    bool generate_speculative(const std::vector<std::uint32_t>& prompt, std::uint32_t max_new, const DenseTransformerRuntime& draft, std::uint32_t draft_tokens, Generation& result, std::string* error=nullptr, const std::function<void(const std::vector<std::uint32_t>&)>& on_tokens={}, const GenerationOptions& options={}, const std::atomic<bool>* cancelled=nullptr) const;
    bool tokenize(const std::string& text, std::vector<std::uint32_t>& ids) const;
    bool detokenize(const std::vector<std::uint32_t>& ids, std::string& text) const;
    bool generate_text(const std::string& prompt, std::uint32_t max_new_tokens,
                       Generation& result, std::string& text, std::string* error=nullptr,
                       const std::function<void(const std::string&)>& on_text={},
                       const GenerationOptions& options={}, const std::atomic<bool>* cancelled=nullptr) const;

    // Additional methods for pipeline support
    bool layer_is_recurrent(std::uint32_t layer) const noexcept;
    bool hadamard_folded(const char* suffix, std::size_t layer) const noexcept;
    bool forward_q35_batch(const std::vector<std::uint32_t>& tokens,
                           std::vector<std::vector<float>>& kc,
                           std::vector<std::vector<float>>& vc,
                           std::vector<std::vector<float>>& convc,
                           std::vector<std::vector<float>>& ssmc,
                           std::vector<float>& logits,
                           std::uint32_t start_pos = 0) const;
    bool forward_q35_token(std::uint32_t token, std::uint32_t position,
                           std::vector<std::vector<float>>& kc,
                           std::vector<std::vector<float>>& vc,
                           std::vector<std::vector<float>>& convc,
                           std::vector<std::vector<float>>& ssmc,
                           std::vector<float>& logits) const;
    bool forward_batch(const std::vector<std::uint32_t>& tokens,
                       std::vector<std::vector<float>>& kc,
                       std::vector<std::vector<float>>& vc,
                       std::vector<float>& logits,
                       std::uint32_t start_pos = 0,
                       std::size_t tail_count = 1,
                       std::vector<float>* tail_logits = nullptr,
                       std::vector<std::vector<float>>* nem_conv = nullptr,
                       std::vector<std::vector<float>>* nem_state = nullptr) const;
    bool forward(std::uint32_t token, std::uint32_t pos,
                 std::vector<std::vector<float>>& kc,
                 std::vector<std::vector<float>>& vc,
                 std::vector<float>& logits,
                 std::vector<std::vector<float>>* nem_conv = nullptr,
                 std::vector<std::vector<float>>* nem_state = nullptr) const;

public:
    struct Tensor { 
        std::vector<std::uint64_t> dims; 
        std::vector<float> values; 
        const std::uint8_t* raw{}; 
        std::size_t elements{}; 
        std::size_t raw_bytes{};  // Added for raw byte size
        std::string name; 
        const zen::quant::Format* format{}; 
    };
    
    struct Hadamard {
        std::uint32_t block_size{};
        bool tied_output{};
        bool gdn_v_grouped{};
        std::vector<float> rot;
        std::unordered_map<std::size_t, std::vector<float>> signs;
        std::unordered_set<std::string> folded_names;
        void transform(float* x, std::size_t width, std::vector<float>& scratch) const;
        void unrotate(float* x, std::size_t width, std::vector<float>& scratch) const;
    };

    struct PrefixCache {
        std::vector<std::uint32_t> tokens;
        std::vector<std::vector<float>> key;
        std::vector<std::vector<float>> value;
        std::vector<float> logits;
        std::vector<std::vector<float>> conv;
        std::vector<std::vector<float>> ssm;
    };

    // Member variables
    std::shared_ptr<MappedFile> mapping_;
    const std::uint8_t* data_{};
    std::size_t size_{};
    RuntimeConfig config_;
    bool loaded_{false};
    bool tokenizer_loaded_{false};
    std::string architecture_name_;
    std::string chat_template_;
    std::vector<std::vector<float>> kc, vc, attn;
    std::vector<std::vector<std::vector<float>>> k_cache, v_cache;
    std::vector<std::vector<float>> attn_norm_weights, ffn_norm_weights;
    std::vector<const std::uint8_t*> attn_q_weights, attn_k_weights, attn_v_weights, attn_output_weights;
    std::vector<const std::uint8_t*> ffn_gate_weights, ffn_up_weights, ffn_down_weights;
    std::vector<const float*> attn_q_weights_fp, attn_k_weights_fp, attn_v_weights_fp, attn_output_weights_fp;
    std::vector<const float*> attn_norm_weights_fp, ffn_norm_weights_fp;
    std::vector<float> cos_cache, sin_cache;
    std::unordered_map<std::string, Tensor> tensors_;
    Hadamard hadamard_;
    mutable std::mutex prefix_cache_mutex_;
    mutable std::shared_ptr<PrefixCache> prefix_cache_;
    std::filesystem::path path_;
    zen::tokenizer::ModelTokenizer tokenizer_;
    bool prefetch_enabled_{false};
    
    // Member functions
    const Tensor* tensor(const std::string& n) const noexcept;
    // Mamba-2 SSM mixer for nemotron_h recurrent layers
    // (llama.cpp build_mamba2_layer / ggml_ssm_scan semantics).
    bool mamba2_forward(const std::string& prefix, const float* norm, float* x,
                        std::size_t T, std::vector<float>& conv_state,
                        std::vector<float>& ssm_state) const;
    // Nemotron-H MoE FFN (llama.cpp nemotron-h build_ffn_layer):
    // sigmoid routing, optional latent projection, gated experts with
    // relu_sq activation, shared expert.
    bool nemotron_moe_forward(const std::string& prefix, const float* norm, float* x,
                              std::size_t T) const;
    // Gemma4-MoE routed FFN (llama.cpp gemma4): softmax router on the
    // post-attention hidden state (divided by sqrt(E), scaled by
    // ffn_gate_inp.scale); fused gate_up_exps (gelu), down_exps, down scale.
    bool gemma_moe_forward(const std::string& prefix, const float* attn_out,
                           float* x, std::size_t T) const;
    void prefetch_layer(std::uint32_t layer) const noexcept;
    void hadamard_transform_input(float* x, std::size_t width, std::vector<float>& scratch) const;
    void hadamard_unrotate(float* x, std::size_t width, std::vector<float>& scratch) const;
};

using LlamaRuntime = DenseTransformerRuntime; // compatibility alias for the original public API.
}