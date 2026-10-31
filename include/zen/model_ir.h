#pragma once
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <optional>

namespace zen::model {
enum class Architecture : std::uint8_t { unknown, llama, qwen, bitnet, bonsai, qwen_hybrid };
enum class TensorRole : std::uint8_t {
    unknown, token_embedding, output, output_norm, attention_norm,
    query, key, value, attention_output, ffn_norm, ffn_gate, ffn_up,
    ffn_down, router, expert_gate, expert_up, expert_down
};
struct TensorDesc {
    std::string name;
    TensorRole role{TensorRole::unknown};
    std::vector<std::uint64_t> dimensions;
    std::uint32_t ggml_type{};
    std::uint64_t offset{};
    std::uint64_t byte_size{};
};
struct ModelConfig {
    Architecture architecture{Architecture::unknown};
    std::string architecture_name;
    std::string model_name;
    std::uint32_t embedding_length{};
    std::uint32_t block_count{};
    std::uint32_t attention_head_count{};
    std::uint32_t kv_head_count{};
    std::uint32_t feed_forward_length{};
    std::uint32_t context_length{};
    std::uint32_t vocabulary_size{};
    std::uint32_t expert_count{};
    std::uint32_t expert_used_count{};
    bool has_rope{};
    bool has_sliding_window{};
    std::uint32_t bos_token_id{1}, eos_token_id{2}, unknown_token_id{};
    bool add_bos_token{true}, add_eos_token{}, add_space_prefix{true};
    std::vector<std::string> tokenizer_tokens;
    std::vector<float> tokenizer_scores;
    std::vector<std::uint32_t> tokenizer_token_types;
    // PrismML Hadamard-fold metadata (prism.hadamard.*). Weights listed in
    // hadamard_weight_names are stored as W_f = W*D*H; the runtime applies
    // signs D then the normalized block Sylvester-Walsh-Hadamard H to every
    // folded matmul input. Rows of hadamard_inverse_names (embedding tables)
    // are stored rotated and are restored with H then D.
    std::uint32_t hadamard_block_size{};
    bool hadamard_tied_output{};
    // prism.hadamard.* contract fields (PrismML-Eng/llama.cpp llama-model.cpp
    // load_hparams_hadamard). Version 1 is the shipped Bonsai 2 contract; the
    // strings are validated in the GGUF parser so a file carrying an
    // unsupported transform/axis/sign mode is rejected instead of silently run
    // in the wrong basis.
    std::uint32_t hadamard_version{};
    bool hadamard_gdn_v_grouped{};
    std::string hadamard_transform, hadamard_axis, hadamard_sign_mode;
    std::uint32_t rope_sections[4]{}; // *.rope.dimension_sections (M-RoPE)
    std::vector<std::string> hadamard_weight_names;
    std::vector<std::string> hadamard_inverse_names;
    std::vector<std::int32_t> hadamard_sign_widths;
    std::vector<std::int32_t> hadamard_sign_values;
    std::unordered_map<std::string, std::string> metadata;
};
struct ArchitectureDescriptor {
    Architecture architecture;
    std::string_view name;
    bool (*recognises)(const ModelConfig&) noexcept;
};
const std::vector<ArchitectureDescriptor>& architecture_descriptors();
Architecture identify_architecture(std::string_view name) noexcept;
std::string_view architecture_name(Architecture architecture) noexcept;
TensorRole infer_tensor_role(std::string_view tensor_name) noexcept;
}
