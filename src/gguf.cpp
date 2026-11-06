#include "zen/gguf.h"
#include "zen/quant.h"
#include "zen/diagnostics.h"
#include <vector>
#include <string_view>
#include <fstream>
#include <limits>
#include <cstring>
#include <type_traits>
#include <bit>
#include <sstream>
namespace zen::gguf { namespace {
class Reader {
    std::ifstream f_;
public:
    explicit Reader(const std::filesystem::path&p):f_(p,std::ios::binary){}
    bool good()const{return f_.good();}
    template<class T> bool read(T&v){static_assert(std::is_trivially_copyable_v<T>);return bool(f_.read(reinterpret_cast<char*>(&v),sizeof(T)));}
    bool bytes(void*p,std::size_t n){return bool(f_.read(static_cast<char*>(p),static_cast<std::streamsize>(n)));}
    std::uint64_t position(){auto p=f_.tellg();return p<0?std::uint64_t(-1):static_cast<std::uint64_t>(p);}
    bool skip(std::uint64_t n){if(n>static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()))return false;f_.seekg(static_cast<std::streamoff>(n),std::ios::cur);return bool(f_);}
    bool string(std::string& s){std::uint64_t n{};if(!read(n)||n>(1ull<<24))return false;s.resize(static_cast<std::size_t>(n));return bytes(s.data(),s.size());}
};
enum Meta : std::uint32_t { U8=0,I8=1,U16=2,I16=3,U32=4,I32=5,F32=6,BOOL=7,STRING=8,ARRAY=9,U64=10,I64=11,F64=12 };
// Maximum metadata ARRAY nesting accepted. GGUF writers emit at most a few
// levels; a hostile file can nest hundreds of thousands of one-element arrays,
// and unbounded recursion there is a stack-overflow DoS (verified upstream: a
// 2 MB file with 180k levels crashed the parser with 0xC00000FD).
constexpr unsigned kMaxMetaDepth=64;
// The parse core is shared by the file path and the in-memory fuzz surface, so
// both readers must expose the same small interface (read/bytes/position/string).
template<class R>
bool skip_value(R&r,std::uint32_t t,std::string* str=nullptr,std::uint64_t* number=nullptr,unsigned depth=0){
    if(depth>kMaxMetaDepth)return false;
    switch(t){
    case U8:case I8:{std::uint8_t v;if(!r.read(v))return false;if(number)*number=v;return true;}
    case U16:case I16:{std::uint16_t v;if(!r.read(v))return false;if(number)*number=v;return true;}
    case U32:case I32:{std::uint32_t v;if(!r.read(v))return false;if(number)*number=v;return true;}
    case U64:case I64:{std::uint64_t v;if(!r.read(v))return false;if(number)*number=v;return true;}
    case F32:{std::uint32_t v;if(!r.read(v))return false;if(number)*number=v;return true;}
    case BOOL:{std::uint8_t v;if(!r.read(v)||v>1)return false;if(number)*number=v;return true;}
    case F64:{std::uint64_t v;if(!r.read(v))return false;if(number)*number=v;return true;}
    case STRING:{std::string v;if(!r.string(v))return false;if(str)*str=std::move(v);return true;}
    case ARRAY:{std::uint32_t subtype{};std::uint64_t n{};if(!r.read(subtype)||!r.read(n)||n>(1ull<<32))return false;if(number)*number=n;for(std::uint64_t i=0;i<n;++i)if(!skip_value(r,subtype,nullptr,nullptr,depth+1))return false;return true;}
    default:return false;
    }
}
std::uint64_t tensor_bytes(std::uint32_t type,std::uint64_t elements){
    // Canonical GGML type IDs. This table validates storage size only; it does
    // not imply that ZenInfer has a decoder for every listed quantisation.
    struct T{std::uint32_t id,block,bytes;};
    static constexpr T ts[]={
        {0,1,4},{1,1,2},{2,32,18},{3,32,20},
        {6,32,22},{7,32,24},{8,32,34},{9,32,40},
        {10,256,84},{11,256,110},{12,256,144},{13,256,176},{14,256,210},{15,256,292},
        {16,256,66},{17,256,74},{18,256,98},{19,256,50},{20,32,18},{21,256,110},
        {22,256,82},{23,256,136},{24,1,1},{25,1,2},{26,1,4},{27,1,8},{28,1,8},
        {29,256,56},{30,1,2},{34,256,54},{35,256,66},
        {39,32,17},{100,32,18},{101,32,17},{143,128,28}
    };
    for(auto t:ts)if(t.id==type){
        if(elements%t.block)return 0;
        const auto blocks=elements/t.block;
        if(blocks>std::numeric_limits<std::uint64_t>::max()/t.bytes)return 0;
        return blocks*t.bytes;
    }
    return 0;
}
}
/**
 * @brief Reader-agnostic GGUF parse core.
 *
 * Parses the header, metadata key/value pairs and tensor directory from any
 * reader that exposes the same interface as the streaming file Reader
 * (`good`, `read<T>`, `bytes`, `position`, `string`). `file_size` bounds every
 * payload-offset check, exactly as the on-disk file size does for `inspect()`.
 *
 * @param r         Reader positioned at the start of a GGUF image.
 * @param file_size Total size of the GGUF image the reader draws from.
 * @return LoadReport with ok=false and a human-readable error on any
 *         malformed, truncated or hostile input.
 */
template<class R>
LoadReport parse_gguf(R& r,std::uint64_t file_size){
    LoadReport out;
    char magic[4];if(!r.bytes(magic,4)||std::memcmp(magic,"GGUF",4)!=0){out.error="Invalid GGUF magic";return out;}
    std::uint64_t tensors{},metadata{};if(!r.read(out.version)||out.version<2||out.version>3||!r.read(tensors)||!r.read(metadata)){out.error="Unsupported or truncated GGUF header (supports v2/v3)";return out;}
    if(tensors>1000000||metadata>1000000){out.error="GGUF directory count exceeds safety limit";return out;}
    for(std::uint64_t i=0;i<metadata;++i){
        std::string key,value;std::uint32_t type{};if(!r.string(key)||!r.read(type)){out.error="Truncated metadata key";return out;}
        std::uint64_t number{};
        // PrismML Hadamard-fold metadata: string vectors and I32 vectors are
        // consumed into the config so the runtime can build the transforms.
        const bool hadamard_names = type==ARRAY&&key=="prism.hadamard.weight_names";
        const bool hadamard_inverse = type==ARRAY&&key=="prism.hadamard.inverse_weight_names";
        const bool hadamard_widths = type==ARRAY&&key=="prism.hadamard.sign_widths";
        const bool hadamard_values = type==ARRAY&&key=="prism.hadamard.sign_values";
        if((hadamard_names||hadamard_inverse)&&(key=="prism.hadamard.weight_names"||key=="prism.hadamard.inverse_weight_names")){
            std::uint32_t subtype{};std::uint64_t count{};
            if(!r.read(subtype)||!r.read(count)||count>100000){out.error="Invalid hadamard metadata array: "+key;return out;}
            if(subtype!=STRING){out.error="Hadamard name array is not a string array";return out;}
            auto& dst=hadamard_names?out.config.hadamard_weight_names:out.config.hadamard_inverse_names;
            dst.reserve(static_cast<std::size_t>(count));
            for(std::uint64_t j=0;j<count;++j){std::string token;if(!r.string(token)){out.error="Truncated hadamard name array";return out;}dst.push_back(std::move(token));}
            continue;
        }
        if(hadamard_widths||hadamard_values){
            std::uint32_t subtype{};std::uint64_t count{};
            if(!r.read(subtype)||!r.read(count)||count>10000000){out.error="Invalid hadamard metadata array: "+key;return out;}
            if(subtype!=I32&&subtype!=U32){out.error="Hadamard numeric array is not 32-bit integer";return out;}
            auto& dst=hadamard_widths?out.config.hadamard_sign_widths:out.config.hadamard_sign_values;
            dst.reserve(static_cast<std::size_t>(count));
            for(std::uint64_t j=0;j<count;++j){std::uint32_t v{};if(!r.read(v)){out.error="Truncated hadamard numeric array";return out;}dst.push_back(static_cast<std::int32_t>(v));}
            continue;
        }
        // *.rope.dimension_sections (M-RoPE section widths, I32[4]).
        if(type==ARRAY&&key.size()>23&&key.compare(key.size()-23,23,"rope.dimension_sections")==0){
            std::uint32_t subtype{};std::uint64_t count{};
            if(!r.read(subtype)||!r.read(count)||count>4){out.error="Invalid rope dimension_sections array";return out;}
            for(std::uint64_t j=0;j<4;++j){
                std::uint32_t v{};
                if(j<count){if(!r.read(v)){out.error="Truncated rope dimension_sections";return out;}}
                out.config.rope_sections[j]=v;
            }
            continue;
        }
        if(key=="prism.hadamard.block_size"&&number==0&&type!=ARRAY){/* fall through to generic handling */}
        if(type==ARRAY&&(key=="tokenizer.ggml.tokens"||key=="tokenizer.ggml.scores"||key=="tokenizer.ggml.token_type")){
            std::uint32_t subtype{};std::uint64_t count{};
            if(!r.read(subtype)||!r.read(count)||count>1000000){out.error="Invalid tokenizer metadata array: "+key;return out;}
            number=count;
            if(key=="tokenizer.ggml.tokens"){
                if(subtype!=STRING){out.error="Tokenizer tokens array is not a string array";return out;}
                out.config.tokenizer_tokens.reserve(static_cast<std::size_t>(count));
                for(std::uint64_t j=0;j<count;++j){std::string token;if(!r.string(token)){out.error="Truncated tokenizer token array";return out;}out.config.tokenizer_tokens.push_back(std::move(token));}
                out.config.vocabulary_size=static_cast<std::uint32_t>(count);
            }else if(key=="tokenizer.ggml.scores"){
                if(subtype!=F32){out.error="Tokenizer scores array is not FP32";return out;}
                out.config.tokenizer_scores.reserve(static_cast<std::size_t>(count));
                for(std::uint64_t j=0;j<count;++j){std::uint32_t bits{};if(!r.read(bits)){out.error="Truncated tokenizer scores array";return out;}out.config.tokenizer_scores.push_back(std::bit_cast<float>(bits));}
            }else{
                if(subtype!=I32&&subtype!=U32){out.error="Tokenizer token_type array is not 32-bit integer";return out;}
                out.config.tokenizer_token_types.reserve(static_cast<std::size_t>(count));
                for(std::uint64_t j=0;j<count;++j){std::uint32_t v{};if(!r.read(v)){out.error="Truncated tokenizer token_type array";return out;}out.config.tokenizer_token_types.push_back(v);}
            }
        }else if(!skip_value(r,type,&value,&number)){out.error="Unsupported or truncated metadata value: "+key;return out;}
        if(type==STRING)out.config.metadata[key]=value;
        else if(type==F32)out.config.metadata[key]=std::to_string(std::bit_cast<float>(static_cast<std::uint32_t>(number)));
        else if(type==F64)out.config.metadata[key]=std::to_string(std::bit_cast<double>(number));
        else if(type!=ARRAY)out.config.metadata[key]=std::to_string(number);
        // Only values that fit a uint32 are accepted: casting a hostile u64
        // (e.g. 2^32+4096) would silently truncate and corrupt the config.
        auto set=[&](const char*s,std::uint32_t&dst){if(key==s&&number<=std::numeric_limits<std::uint32_t>::max())dst=static_cast<std::uint32_t>(number);};
        set("llama.embedding_length",out.config.embedding_length);set("llama.block_count",out.config.block_count);
        set("llama.attention.head_count",out.config.attention_head_count);set("llama.attention.head_count_kv",out.config.kv_head_count);
        set("llama.feed_forward_length",out.config.feed_forward_length);set("llama.context_length",out.config.context_length);
        set("qwen2.embedding_length",out.config.embedding_length);set("qwen2.block_count",out.config.block_count);
        set("qwen2.attention.head_count",out.config.attention_head_count);set("qwen2.attention.head_count_kv",out.config.kv_head_count);
        set("qwen2.feed_forward_length",out.config.feed_forward_length);set("qwen2.context_length",out.config.context_length);
        set("qwen3.embedding_length",out.config.embedding_length);set("qwen3.block_count",out.config.block_count);
        set("qwen3.attention.head_count",out.config.attention_head_count);set("qwen3.attention.head_count_kv",out.config.kv_head_count);
        set("qwen3.feed_forward_length",out.config.feed_forward_length);set("qwen3.context_length",out.config.context_length);
        set("qwen2moe.embedding_length",out.config.embedding_length);set("qwen2moe.block_count",out.config.block_count);
        set("qwen2moe.attention.head_count",out.config.attention_head_count);set("qwen2moe.attention.head_count_kv",out.config.kv_head_count);
        set("qwen2moe.feed_forward_length",out.config.feed_forward_length);set("qwen2moe.context_length",out.config.context_length);
        set("qwen3moe.embedding_length",out.config.embedding_length);set("qwen3moe.block_count",out.config.block_count);
        set("qwen3moe.attention.head_count",out.config.attention_head_count);set("qwen3moe.attention.head_count_kv",out.config.kv_head_count);
        set("qwen3moe.feed_forward_length",out.config.feed_forward_length);set("qwen3moe.context_length",out.config.context_length);
        // BitNet b1.58 GGUFs use the `bitnet.` key prefix (arXiv 2504.12285).
        set("bitnet.embedding_length",out.config.embedding_length);set("bitnet.block_count",out.config.block_count);
        set("bitnet.attention.head_count",out.config.attention_head_count);set("bitnet.attention.head_count_kv",out.config.kv_head_count);
        set("bitnet.feed_forward_length",out.config.feed_forward_length);set("bitnet.context_length",out.config.context_length);
        // Qwen3.5 / Bonsai 2 hybrid (GDN + attention) GGUFs use `qwen35.` keys.
        set("qwen35.embedding_length",out.config.embedding_length);set("qwen35.block_count",out.config.block_count);
        set("qwen35.attention.head_count",out.config.attention_head_count);set("qwen35.attention.head_count_kv",out.config.kv_head_count);
        set("qwen35.feed_forward_length",out.config.feed_forward_length);set("qwen35.context_length",out.config.context_length);
        set("qwen2moe.expert_count",out.config.expert_count);set("qwen2moe.expert_used_count",out.config.expert_used_count);
        set("qwen3moe.expert_count",out.config.expert_count);set("qwen3moe.expert_used_count",out.config.expert_used_count);
        if(key=="prism.hadamard.block_size"&&type!=ARRAY)out.config.hadamard_block_size=static_cast<std::uint32_t>(number);
        if(key=="prism.hadamard.version"&&type!=ARRAY)out.config.hadamard_version=static_cast<std::uint32_t>(number);
        if(key=="prism.hadamard.tied_output")out.config.hadamard_tied_output=number!=0;
        if(key=="prism.hadamard.gdn_v_grouped")out.config.hadamard_gdn_v_grouped=number!=0;
        if((key=="prism.hadamard.transform"||key=="prism.hadamard.axis"||key=="prism.hadamard.sign_mode")&&type==STRING){
            auto& dst=key=="prism.hadamard.transform"?out.config.hadamard_transform
                    :key=="prism.hadamard.axis"?out.config.hadamard_axis:out.config.hadamard_sign_mode;
            dst=value;
        }
        if(key=="general.architecture"&&type==STRING){out.config.architecture_name=value;out.config.architecture=model::identify_architecture(value);}
        if(key=="general.name"&&type==STRING)out.config.model_name=value;
        set("tokenizer.ggml.bos_token_id",out.config.bos_token_id);
        set("tokenizer.ggml.eos_token_id",out.config.eos_token_id);
        set("tokenizer.ggml.unknown_token_id",out.config.unknown_token_id);
        if(key=="tokenizer.ggml.add_bos_token")out.config.add_bos_token=number!=0;
        if(key=="tokenizer.ggml.add_eos_token")out.config.add_eos_token=number!=0;
        if(key=="tokenizer.ggml.add_space_prefix")out.config.add_space_prefix=number!=0;
        if(key=="llama.expert_count"||key=="qwen2.expert_count"||key=="qwen3.expert_count"||key=="qwen2moe.expert_count"||key=="qwen3moe.expert_count")out.config.expert_count=static_cast<std::uint32_t>(number);
        if(key=="llama.expert_used_count"||key=="qwen2.expert_used_count"||key=="qwen3.expert_used_count"||key=="qwen2moe.expert_used_count"||key=="qwen3moe.expert_used_count")out.config.expert_used_count=static_cast<std::uint32_t>(number);
    }
    out.tensors.reserve(static_cast<std::size_t>(tensors));
    for(std::uint64_t i=0;i<tensors;++i){
        model::TensorDesc d;std::uint32_t nd{};if(!r.string(d.name)||!r.read(nd)||nd==0||nd>4){out.error="Invalid tensor descriptor";return out;}
        d.dimensions.resize(nd);std::uint64_t elements=1;
        for(auto&dim:d.dimensions){if(!r.read(dim)||dim==0||elements>std::numeric_limits<std::uint64_t>::max()/dim){out.error="Invalid tensor dimensions";return out;}elements*=dim;}
        if(!r.read(d.ggml_type)||!r.read(d.offset)){out.error="Truncated tensor descriptor";return out;}
        d.byte_size=tensor_bytes(d.ggml_type,elements);d.role=model::infer_tensor_role(d.name);out.tensors.push_back(std::move(d));
    }
    std::uint32_t alignment=32;auto it=out.config.metadata.find("general.alignment");
    if(it!=out.config.metadata.end())try{alignment=static_cast<std::uint32_t>(std::stoul(it->second));}catch(...){}
    if(alignment==0||(alignment&(alignment-1))!=0){out.error="Invalid GGUF tensor alignment";return out;}
    // Directory position is the beginning of the tensor data section, aligned as required by GGUF.
    auto pos=r.position();
    if(pos==std::uint64_t(-1)){out.error="Cannot determine GGUF data offset";return out;}
    out.tensor_data_offset=(pos+alignment-1)&~std::uint64_t(alignment-1);
    if(out.tensor_data_offset>file_size){out.error="GGUF tensor-data offset exceeds file size";return out;}
    for(auto& t:out.tensors){
        if(t.byte_size==0){out.error="Tensor uses unsupported or invalid GGML type "+std::to_string(t.ggml_type)+": "+t.name;return out;}
        if(t.offset>file_size-out.tensor_data_offset || t.byte_size>file_size-out.tensor_data_offset-t.offset){
            out.error="Tensor payload exceeds GGUF file bounds: "+t.name;return out;
        }
    }
    // A Prism ternary pack stores its weights in a rotated basis, so a runtime
    // that skips prism.hadamard.* reads the file "successfully" and produces
    // garbage with no error (upstream llama-quant now refuses to write one).
    // Refuse the combination here instead of guessing. Likewise refuse a
    // contract whose transform/axis/sign mode this runtime cannot apply.
    if(!out.config.hadamard_weight_names.empty()||!out.config.hadamard_inverse_names.empty()){
        if(out.config.hadamard_version!=0&&out.config.hadamard_version!=1&&out.config.hadamard_version!=2){
            out.error="Unsupported prism.hadamard.version "+std::to_string(out.config.hadamard_version);return out;
        }
        if(!out.config.hadamard_transform.empty()&&out.config.hadamard_transform!="normalized-sylvester-walsh-hadamard"){
            out.error="Unsupported prism.hadamard.transform: "+out.config.hadamard_transform;return out;
        }
        if(!out.config.hadamard_axis.empty()&&out.config.hadamard_axis!="input-last-dimension"){
            out.error="Unsupported prism.hadamard.axis: "+out.config.hadamard_axis;return out;
        }
        if(!out.config.hadamard_sign_mode.empty()&&out.config.hadamard_sign_mode!="explicit"&&out.config.hadamard_sign_mode!="identity"){
            out.error="Unsupported prism.hadamard.sign_mode: "+out.config.hadamard_sign_mode;return out;
        }
    }else{
        for(const auto& t:out.tensors){
            if(t.ggml_type==143||t.ggml_type==142){
                out.error="Tensor "+t.name+" uses Prism ternary type "+std::to_string(t.ggml_type)+
                          " but the file carries no prism.hadamard.* contract; its weights would be read in the wrong basis";
                return out;
            }
        }
    }
    out.ok=true;return out;
}
/** @brief In-memory reader with the same interface as the streaming file Reader. */
class BufferReader {
    const std::uint8_t* p_;
    std::size_t size_;
    std::size_t off_{0};
public:
    BufferReader(const std::uint8_t* p,std::size_t s):p_(p),size_(s){}
    bool good()const{return p_&&size_!=0;}
    template<class T> bool read(T&v){
        static_assert(std::is_trivially_copyable_v<T>);
        if(size_-off_<sizeof(T))return false;
        std::memcpy(&v,p_+off_,sizeof(T));off_+=sizeof(T);return true;
    }
    bool bytes(void* dst,std::size_t n){
        if(size_-off_<n)return false;
        std::memcpy(dst,p_+off_,n);off_+=n;return true;
    }
    std::uint64_t position()const{return static_cast<std::uint64_t>(off_);}
    bool string(std::string& s){std::uint64_t n{};if(!read(n)||n>(1ull<<24))return false;s.resize(static_cast<std::size_t>(n));return bytes(s.data(),s.size());}
};
LoadReport inspect(const std::filesystem::path&path){
    Reader r(path);
    if(!r.good()){LoadReport out;out.error="Cannot open GGUF file";return out;}
    std::error_code file_ec;const auto file_size=std::filesystem::file_size(path,file_ec);
    if(file_ec){LoadReport out;out.error="Cannot determine GGUF file size";return out;}
    return parse_gguf(r,file_size);
}
LoadReport inspect_bytes(const std::uint8_t* data,std::size_t size){
    BufferReader r(data,size);
    return parse_gguf(r,static_cast<std::uint64_t>(size));
}
bool read_tensor_f32(const std::filesystem::path& path,const LoadReport& report,
                     std::string_view tensor_name,std::vector<float>& output,std::string* error){
    zen::diag::Scope trace("gguf.tensor.read_decode",zen::diag::Phase::io);
    auto fail=[&](const char* message){if(error)*error=message;return false;};
    if(!report.ok)return fail("GGUF report is not valid");
    const model::TensorDesc* desc=nullptr;
    for(const auto& t:report.tensors)if(t.name==tensor_name){desc=&t;break;}
    if(!desc)return fail("Tensor name not found");
    std::uint64_t elements=1;
    for(auto d:desc->dimensions){if(!d||elements>std::numeric_limits<std::uint64_t>::max()/d)return fail("Tensor element count overflow");elements*=d;}
    if(elements>std::numeric_limits<std::size_t>::max()||desc->byte_size>std::numeric_limits<std::size_t>::max()||
       desc->byte_size>static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()))return fail("Tensor exceeds addressable memory");
    const char* format_name=nullptr;
    switch(desc->ggml_type){
        case 0:format_name="F32";break; case 1:format_name="F16";break;
        case 2:format_name="Q4_0";break; case 3:format_name="Q4_1";break;
        case 6:format_name="Q5_0";break; case 7:format_name="Q5_1";break;
        case 8:format_name="Q8_0";break; case 10:format_name="Q2_K";break;
        case 11:format_name="Q3_K";break; case 12:format_name="Q4_K";break;
        case 13:format_name="Q5_K";break; case 14:format_name="Q6_K";break;
        case 15:format_name="Q8_K";break; case 16:format_name="IQ2_XXS";break;
        case 17:format_name="IQ2_XS";break; case 18:format_name="IQ3_XXS";break;
        case 19:format_name="IQ1_S";break; case 20:format_name="IQ4_NL";break;
        case 21:format_name="IQ3_S";break; case 22:format_name="IQ2_S";break;
        case 23:format_name="IQ4_XS";break; case 29:format_name="IQ1_M";break;
        case 30:format_name="BF16";break; case 34:format_name="TQ1_0";break;
        case 35:format_name="TQ2_0";break;
        case 39:format_name="MXFP4";break;
        case 100:format_name="Q4_0_ROCMFP4";break;
        case 101:format_name="Q4_0_ROCMFP4_FAST";break;
        case 143:format_name="PTQ1_0";break;
        default:return fail("GGML tensor type has no implemented FP32 decoder");
    }
    const auto* format=quant::find(format_name); if(!format||!format->implemented)return fail("Tensor decoder is unavailable");
    const auto expected=quant::row_bytes(*format,static_cast<std::size_t>(elements));
    if(!expected||expected!=desc->byte_size)return fail("Tensor byte size does not match decoder block layout");
    if(desc->offset>std::numeric_limits<std::uint64_t>::max()-report.tensor_data_offset)return fail("Tensor file offset overflow");
    std::ifstream file(path,std::ios::binary);if(!file)return fail("Cannot reopen GGUF file");
    const auto absolute=report.tensor_data_offset+desc->offset;
    if(absolute>static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()))return fail("Tensor file offset is not seekable");
    file.seekg(static_cast<std::streamoff>(absolute));if(!file)return fail("Cannot seek to tensor payload");
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(desc->byte_size));
    if(!file.read(reinterpret_cast<char*>(bytes.data()),static_cast<std::streamsize>(bytes.size())))return fail("Truncated tensor payload");
    std::vector<float> decoded(static_cast<std::size_t>(elements));
    if(!quant::dequantize(*format,bytes.data(),static_cast<std::size_t>(elements),decoded.data()))return fail("Tensor dequantisation failed");
    output.swap(decoded); if(error)error->clear(); return true;
}

}
