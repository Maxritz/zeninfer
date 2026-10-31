#include "zen/model_ir.h"
#include <algorithm>
namespace zen::model {
Architecture identify_architecture(std::string_view n) noexcept {
    if (n=="llama") return Architecture::llama;
    if (n=="qwen" || n=="qwen2" || n=="qwen2moe" || n=="qwen3" || n=="qwen3moe") return Architecture::qwen;
    if (n=="qwen35" || n=="qwen3next" || n=="qwen35moe") return Architecture::qwen_hybrid;
    if (n.find("bitnet")!=std::string_view::npos || n.find("b1")!=std::string_view::npos) return Architecture::bitnet;
    if (n.find("bonsai")!=std::string_view::npos) return Architecture::bonsai;
    return Architecture::unknown;
}
std::string_view architecture_name(Architecture a) noexcept {
    switch(a) {
        case Architecture::llama: return "llama";
        case Architecture::qwen: return "qwen";
        case Architecture::qwen_hybrid: return "qwen_hybrid";
        case Architecture::bitnet: return "bitnet";
        case Architecture::bonsai: return "bonsai";
        default: return "unknown";
    }
}
namespace {
bool llama(const ModelConfig& c) noexcept { return c.architecture==Architecture::llama; }
bool qwen(const ModelConfig& c) noexcept { return c.architecture==Architecture::qwen; }
bool qwen_hybrid(const ModelConfig& c) noexcept { return c.architecture==Architecture::qwen_hybrid; }
bool bitnet(const ModelConfig& c) noexcept { return c.architecture==Architecture::bitnet; }
bool bonsai(const ModelConfig& c) noexcept { return c.architecture==Architecture::bonsai; }
}
const std::vector<ArchitectureDescriptor>& architecture_descriptors() {
    static const std::vector<ArchitectureDescriptor> a{
        {Architecture::llama,"llama-family",llama},
        {Architecture::qwen,"qwen-family",qwen},
        {Architecture::qwen_hybrid,"qwen-hybrid-family",qwen_hybrid},
        {Architecture::bitnet,"bitnet-family",bitnet},
        {Architecture::bonsai,"bonsai-family",bonsai}
    };
    return a;
}
TensorRole infer_tensor_role(std::string_view n) noexcept {
    if(n.find("token_embd")!=n.npos || n.find("embed_tokens")!=n.npos) return TensorRole::token_embedding;
    if(n=="output.weight" || n.find("lm_head")!=n.npos) return TensorRole::output;
    if(n.find("output_norm")!=n.npos || n.find("final_layernorm")!=n.npos) return TensorRole::output_norm;
    if(n.find("attn_norm")!=n.npos || n.find("input_layernorm")!=n.npos) return TensorRole::attention_norm;
    // Qwen3.5 GDN projections are matrices and must raw-map like q/k/v.
    if(n.find("attn_qkv")!=n.npos) return TensorRole::query;
    if(n.find("attn_gate")!=n.npos) return TensorRole::value;
    if(n.find("ssm_out")!=n.npos) return TensorRole::attention_output;
    // Norm tensors must be classified before the projection checks: names like
    // attn_q_norm / attn_k_norm / attn_sub_norm / ffn_sub_norm contain the
    // projection substrings and would otherwise be raw-mapped as matrices
    // instead of decoded to FP32 values. The forward paths fetch norms by
    // exact name, so any non-matrix role is equivalent here.
    if(n.find("_norm.weight")!=n.npos || n.find("_norm.bias")!=n.npos) return TensorRole::attention_norm;
    if(n.find("attn_q")!=n.npos || n.find("q_proj")!=n.npos) return TensorRole::query;
    if(n.find("attn_k")!=n.npos || n.find("k_proj")!=n.npos) return TensorRole::key;
    if(n.find("attn_v")!=n.npos || n.find("v_proj")!=n.npos) return TensorRole::value;
    if(n.find("attn_output")!=n.npos || n.find("o_proj")!=n.npos) return TensorRole::attention_output;
    if(n.find("ffn_norm")!=n.npos || n.find("post_attention_layernorm")!=n.npos) return TensorRole::ffn_norm;
    // MoE names must be classified before dense FFN names: ffn_gate_inp and
    // ffn_gate_exps contain the same prefix as the dense ffn_gate projection.
    if(n.find("ffn_gate_inp")!=n.npos || n.find("router")!=n.npos) return TensorRole::router;
    if(n.find("experts")!=n.npos || n.find("_exps")!=n.npos) {
        if(n.find("gate")!=n.npos || n.find("w1")!=n.npos) return TensorRole::expert_gate;
        if(n.find("up")!=n.npos || n.find("w3")!=n.npos) return TensorRole::expert_up;
        if(n.find("down")!=n.npos || n.find("w2")!=n.npos) return TensorRole::expert_down;
    }
    if(n.find("ffn_gate")!=n.npos || n.find("gate_proj")!=n.npos) return TensorRole::ffn_gate;
    if(n.find("ffn_up")!=n.npos || n.find("up_proj")!=n.npos) return TensorRole::ffn_up;
    if(n.find("ffn_down")!=n.npos || n.find("down_proj")!=n.npos) return TensorRole::ffn_down;
    return TensorRole::unknown;
}
}
