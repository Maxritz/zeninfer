#pragma once
#include "zen/model_ir.h"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
namespace zen::tokenizer {
class ModelTokenizer {
public:
    bool load(const model::ModelConfig& config,std::string* error=nullptr);
    bool available()const noexcept{return !tokens_.empty();}
    bool encode(const std::string& text,std::vector<std::uint32_t>& ids,bool add_bos=true,bool add_eos=false)const;
    bool decode(const std::vector<std::uint32_t>& ids,std::string& text,bool skip_special=true)const;
    bool same_vocabulary(const ModelTokenizer& other) const noexcept;
    std::uint32_t bos_id()const noexcept{return bos_;}
    std::uint32_t eos_id()const noexcept{return eos_;}
private:
    std::vector<std::string> tokens_;
    std::vector<float> scores_;
    std::vector<std::uint32_t> types_;
    std::unordered_map<std::string,std::uint32_t> token_to_id_;
    std::vector<int> byte_to_id_;
    std::uint32_t bos_{1},eos_{2},unk_{};
    bool add_space_prefix_{};
    bool gpt2_bpe_{};
    std::unordered_map<std::uint32_t,std::uint8_t> gpt2_decoder_;
};
using LlamaSentencePiece = ModelTokenizer; // source-compatibility alias; implementation is model-neutral.
}
