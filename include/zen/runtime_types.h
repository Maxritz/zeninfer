#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace zen::runtime {
struct RuntimeConfig {
    std::uint32_t embedding{}, layers{}, heads{}, kv_heads{}, ffn{}, context{}, vocab{}, rope_dim{};
    // Per-head key/value/query width. Modern Qwen3-family models decouple this
    // from embedding/heads (e.g. Qwen3-4B: embedding 2560, 32 heads, head_dim
    // 128, so attn_q is 2560x4096). 0 means embedding/heads.
    std::uint32_t head_dim{};
    float rope_base{10000.0f}, rms_epsilon{1.0e-5f}, rope_scaling_factor{1.0f}, rope_low_freq_factor{1.0f}, rope_high_freq_factor{4.0f}, rope_yarn_beta_fast{32.0f}, rope_yarn_beta_slow{1.0f};
    std::uint32_t sliding_window{}, rope_original_context{};
    bool use_alibi{};
    // BitNet (LLM_ARCH_BITNET) execution variant: SwiGLU FFN wrapped by
    // sub-layer RMS norms (attn_sub_norm after attention, ffn_sub_norm before
    // ffn_down), no per-head Q/K norms, and a tied lm_head via tok_embd.
    // Reference: ggml-org/llama.cpp src/models/bitnet.cpp.
    bool bitnet_variant{};
    // Qwen3.5 / Bonsai 2 hybrid variant (qwen35 GGUF): interleaved gated
    // delta-net (linear attention) and full-attention layers with sandwich
    // norms and Hadamard-folded weights. Reference: PrismML llama.cpp fork
    // src/models/qwen35.cpp.
    bool qwen35_variant{};
    std::uint32_t full_attention_interval{4};
    // GDN hyperparameters (qwen35 only).
    std::uint32_t ssm_state{};       // d_state / head_k_dim / head_v_dim
    std::uint32_t ssm_n_group{};     // key heads
    std::uint32_t ssm_dt_rank{};     // value heads
    std::uint32_t ssm_d_conv{};      // conv kernel width
    std::uint32_t rope_sections[4]{}; // M-RoPE section widths (text-only use)
    std::string rope_scaling_type;
    std::uint32_t bos_token{1}, eos_token{2};
    bool add_bos_token{}, add_eos_token{};
};
struct GenerationOptions {
    float temperature{0.0f}; std::uint32_t top_k{}; float top_p{1.0f}; std::uint64_t seed{};
    std::vector<std::string> stop; std::unordered_map<std::uint32_t,float> logit_bias;
    std::vector<std::uint32_t> allowed_token_ids;
};
struct Generation {
    std::vector<std::uint32_t> token_ids;
    std::vector<float> last_logits;
    double prefill_ms{}, decode_ms{};
    bool stopped_on_eos{}, stopped_on_stop{}, cache_hit{};
    std::uint32_t speculative_proposed{}, speculative_accepted{}, speculative_rejected{};
};
}
