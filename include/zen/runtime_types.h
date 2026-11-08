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
    // RoPE pairing convention. llama.cpp applies llama-family weights with its
    // "NORM" rope: consecutive pairs (x[2i], x[2i+1]) rotate with angle
    // pos*base^(-2i/head_dim) (src/llama-model.cpp llama_model_rope_type).
    // Qwen-family weights use the half-split NeoX layout: (x[i], x[i+pairs]).
    // The two mistakes cancel in per-op parity suites (both sides of such a
    // suite share the convention), so this flag needs unilateral agreement with
    // llama.cpp, not cross-testing inside ZenInfer.
    bool rope_pairs_adjacent{};
    // Gemma4 (LLM_ARCH_GEMMA4) execution variant: embedding scale sqrt(n_embd),
    // attention scale 1.0, GEGLU FFN, sandwich norms (post_attention_norm after
    // the attn projection before the residual, post_ffw_norm after the FFN),
    // per-layer output scale, optional final logit softcap (tanh). Reference:
    // ggml-org/llama.cpp src/models/gemma4.cpp.
    bool gemma_variant{};
    bool phi3_variant{false};
    float attn_scale{};   // 0 -> 1/sqrt(head_dim); Gemma4: 1.0
    float emb_scale{1.0f};
    float logit_softcap{};
    std::vector<std::uint32_t> head_dim_per_layer; // empty = uniform head_dim
    // Qwen3.5 / Bonsai 2 hybrid variant (qwen35 GGUF): interleaved gated
    // delta-net (linear attention) and full-attention layers with sandwich
    // norms and Hadamard-folded weights. Reference: PrismML llama.cpp fork
    // src/models/qwen35.cpp.
    bool qwen35_variant{};
    std::uint32_t full_attention_interval{4};
    // prism.hadamard.gdn_v_grouped: the ssm_out weight was folded with its value
    // heads in grouped order while the recurrence emits tiled order, so the
    // out-projection input is permuted first. Recorded independently of whether
    // the file also carries folded weights.
    bool hadamard_gdn_v_grouped{};
    // GDN hyperparameters (qwen35 only).
    std::uint32_t ssm_state{};       // d_state / head_k_dim / head_v_dim
    std::uint32_t ssm_n_group{};     // key heads
    std::uint32_t ssm_dt_rank{};     // value heads
    std::uint32_t ssm_d_conv{};      // conv kernel width
    // MoE hyperparameters (qwen3moe/laguna): router selects top-k of N experts
    // per token; each expert has its own ffn (moe_ffn) gate/up/down.
    std::uint32_t moe_expert_count{}, moe_used_count{}, moe_ffn{};
    bool moe_sigmoid{}; // laguna: sigmoid + score-bias (vs qwen3 softmax)
    bool moe_nemotron{}; // nemotron_h_moe: sigmoid post-bias, norm-weights * w_scale, relu_sq no-gate
    bool moe_gemma{};   // gemma4-MoE: softmax, normed weights, fused gate_up_exps, router scale divides sqrt(E)
    float moe_w_scale{1.0f};
    bool moe_norm_w{false};
    // DeepSeek2 MLA variant: compressed KV latent (kv_lora_rank + rope split
    // across attn_kv_a_mqa) cached per token; K/V decompressed per head via
    // attn_kv_b. Reference: llama.cpp src/models/deepseek2.cpp.
    bool deepseek_variant{};
    std::uint32_t kv_lora_rank{}; // MLA latent rank (deepseek only)
    // Starcoder2 variant: plain LayerNorm (+bias), sequential GELU FFN with
    // up/down biases, Q/K/V/O projection biases. Reference: llama.cpp
    // src/models/starcoder2.cpp.
    bool layernorm_variant{};
    bool gelu_variant{false};
    bool gelu_par_variant{false};
    // K2-Horizon variant: group-RMSNorm (attention.group_norm_groups slices),
    // attention output gate with log2 softplus, MoVA value-expert attention
    // (attn_v_gate router + attn_v_exps) on MoE layers. Reference: llama.cpp
    // src/models/k2-horizon.cpp.
    bool k2_variant{};
    std::uint32_t group_norm_groups{1};
    std::uint32_t value_expert_count{};   // MoVA experts (k2 only)
    std::uint32_t value_expert_used{};    // MoVA top-k (k2 only)
    // GPT-OSS variant: attention sinks per head, sliding window, post-attn
    // RMSNorm, biased SWIGLU MoE with softmax weights (no weight renorm).
    // Reference: llama.cpp src/models/openai-moe.cpp.
    bool gpt_oss_variant{};
    bool has_attn_sinks{};               // per-head attention-sink logits
    bool has_expert_bias{};              // per-expert gate/up/down biases
    // Spark-X2.5 variant: fused attn_qkv, sigmoid per-head attention gate,
    // GELU parallel FFN, SWA layers (3/4, pattern) with own rope base/dims.
    // Reference: llama.cpp src/models/spark2-5.cpp.
    bool spark2_5_variant{};
    float rope_base_swa{0.0f};           // rope.freq_base_swa
    std::uint32_t rope_dim_swa{};        // rope.dimension_count_swa (SWA layers)
    std::vector<std::uint8_t> swa_pattern; // 1 = SWA layer, 0 = full attention
    std::vector<std::uint8_t> rope_pattern;  // 1 = layer applies RoPE, 0 = none
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
