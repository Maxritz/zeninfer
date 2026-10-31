#include "zen/dense_transformer_runtime.h"
#include "zen/diagnostics.h"
#include "zen/resource_governor.h"
#include "zen/kernels.h"
#include "zen/quant.h"
#include "zen/execution_planner.h"
#include "zen/attention.h"
#include <thread>
#include <memory>
#include <future>
#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <limits>
#include <numeric>
#include <random>

namespace zen::runtime {
class MappedFile {
public:
    static std::shared_ptr<MappedFile> open(const std::filesystem::path& path, std::string* error) {
        auto m=std::shared_ptr<MappedFile>(new MappedFile());
#if defined(_WIN32)
        m->file_=CreateFileW(path.c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
        if(m->file_==INVALID_HANDLE_VALUE){if(error)*error="CreateFileW failed for model mapping";return {};}
        LARGE_INTEGER sz{};if(!GetFileSizeEx(m->file_,&sz)||sz.QuadPart<=0){if(error)*error="Cannot determine mapped model size";return {};}
        m->size_=static_cast<std::size_t>(sz.QuadPart);m->mapping_=CreateFileMappingW(m->file_,nullptr,PAGE_READONLY,0,0,nullptr);
        if(!m->mapping_){if(error)*error="CreateFileMappingW failed";return {};}
        m->data_=static_cast<const std::uint8_t*>(MapViewOfFile(m->mapping_,FILE_MAP_READ,0,0,0));
#else
        m->fd_=::open(path.c_str(),O_RDONLY);if(m->fd_<0){if(error)*error="open() failed for model mapping";return {};}
        struct stat st{};if(fstat(m->fd_,&st)!=0||st.st_size<=0){if(error)*error="Cannot determine mapped model size";return {};}
        m->size_=static_cast<std::size_t>(st.st_size);void* p=mmap(nullptr,m->size_,PROT_READ,MAP_PRIVATE,m->fd_,0);
        if(p==MAP_FAILED){m->data_=nullptr;if(error)*error="mmap() failed for model";return {};}
        m->data_=static_cast<const std::uint8_t*>(p);
#endif
        if(!m->data_){if(error)*error="Mapping returned a null view";return {};}
        return m;
    }
    void prefetch(const std::uint8_t* ptr,std::size_t length) const noexcept {
        if(!ptr||!length)return;
#if defined(_WIN32)
        struct Range { PVOID VirtualAddress; SIZE_T NumberOfBytes; } range{const_cast<std::uint8_t*>(ptr),length};
        using PrefetchFn=BOOL(WINAPI*)(HANDLE,ULONG,Range*,ULONG);
        auto fn=reinterpret_cast<PrefetchFn>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"),"PrefetchVirtualMemory"));
        if(fn)fn(GetCurrentProcess(),1,&range,0);
#else
        const long page=sysconf(_SC_PAGESIZE);if(page<=0)return;const auto p=reinterpret_cast<std::uintptr_t>(ptr);const auto start=p-(p%static_cast<std::uintptr_t>(page));const auto span=length+(p-start);madvise(reinterpret_cast<void*>(start),span,MADV_WILLNEED);
#endif
    }
    ~MappedFile(){
#if defined(_WIN32)
        if(data_)UnmapViewOfFile(data_);if(mapping_)CloseHandle(mapping_);if(file_!=INVALID_HANDLE_VALUE)CloseHandle(file_);
#else
        if(data_)munmap(const_cast<std::uint8_t*>(data_),size_);
        if(fd_>=0)::close(fd_);
#endif
    }
    const std::uint8_t* data()const noexcept{return data_;}std::size_t size()const noexcept{return size_;}
private:
    MappedFile()=default;const std::uint8_t* data_{};std::size_t size_{};
#if defined(_WIN32)
    HANDLE file_{INVALID_HANDLE_VALUE},mapping_{};
#else
    int fd_{-1};
#endif
};
namespace {
float meta_float(const zen::model::ModelConfig& c,const char* key,float fallback){
    auto i=c.metadata.find(key); if(i==c.metadata.end())return fallback;
    try{return std::stof(i->second);}catch(...){return fallback;}
}
float meta_float_arch(const zen::model::ModelConfig& c,const std::string& arch,const char* suffix,float fallback){auto i=c.metadata.find(arch+"."+suffix);if(i==c.metadata.end())return meta_float(c,(std::string("llama.")+suffix).c_str(),fallback);try{return std::stof(i->second);}catch(...){return fallback;}}
std::uint32_t meta_uint(const zen::model::ModelConfig& c,const char* key,std::uint32_t fallback){
    auto i=c.metadata.find(key);if(i==c.metadata.end())return fallback;
    try{return static_cast<std::uint32_t>(std::stoul(i->second));}catch(...){return fallback;}
}
std::string meta_string_arch(const zen::model::ModelConfig& c,const std::string& arch,const char* suffix){auto i=c.metadata.find(arch+"."+suffix);if(i==c.metadata.end())i=c.metadata.find(std::string("llama.")+suffix);return i==c.metadata.end()?std::string{}:i->second;}
std::uint32_t meta_uint_arch(const zen::model::ModelConfig& c,const std::string& arch,const char* suffix,std::uint32_t fallback){auto i=c.metadata.find(arch+"."+suffix);if(i==c.metadata.end())return meta_uint(c,(std::string("llama.")+suffix).c_str(),fallback);try{return static_cast<std::uint32_t>(std::stoul(i->second));}catch(...){return fallback;}}
bool meta_bool_arch(const zen::model::ModelConfig& c,const std::string& arch,const char* suffix,bool fallback=false){
    auto i=c.metadata.find(arch+"."+suffix);if(i==c.metadata.end())i=c.metadata.find(std::string("llama.")+suffix);
    if(i==c.metadata.end())return fallback;
    return i->second=="1"||i->second=="true"||i->second=="TRUE";
}
void rms_norm(const float* x,const float* weight,float* y,std::size_t n,float eps){
    double ss=0.0;for(std::size_t i=0;i<n;++i)ss+=double(x[i])*x[i];
    const float inv=1.0f/std::sqrt(static_cast<float>(ss/n)+eps);
    for(std::size_t i=0;i<n;++i)y[i]=x[i]*inv*weight[i];
}
void qk_rms_norm(float* values,std::uint32_t heads,std::uint32_t head_dim,
                 const float* weight,float epsilon){
    for(std::uint32_t h=0;h<heads;++h)
        rms_norm(values+std::size_t(h)*head_dim,weight,values+std::size_t(h)*head_dim,head_dim,epsilon);
}
bool finite_values(const float* p,std::size_t n){for(std::size_t i=0;i<n;++i)if(!std::isfinite(p[i]))return false;return true;}
void matvec(const float* w,const float* x,float* y,std::size_t rows,std::size_t cols){
#ifdef ZEN_HAS_AVX2_KERNELS
    // Runtime dispatch is safe only after CPUID, OSXSAVE and XCR0 checks.
    // The scalar path remains the numerical reference and non-AVX2 fallback.
    static const bool avx2_available = zen::cpu_has_avx2_fma();
    if (avx2_available) {
        zen::gemv_avx2(w,x,y,rows,cols);
        return;
    }
#endif
    for(std::size_t r=0;r<rows;++r){double sum=0.0;const float* wr=w+r*cols;for(std::size_t c=0;c<cols;++c)sum+=double(wr[c])*x[c];y[r]=static_cast<float>(sum);}
}
// Normalized Sylvester-Walsh-Hadamard rotation of one block: out[j] =
// (1/sqrt(B)) * sum_i (-1)^(popcount(i&j)) * in[i]. Butterfly: O(B log B).
void swht_block(const float* in,float* out,std::uint32_t B){
    for(std::uint32_t i=0;i<B;++i){
        std::uint32_t r=0,b=i;
        for(std::uint32_t bit=1;bit<B;bit<<=1){r=(r<<1)|(b&1);b>>=1;}
        out[i]=in[r];
    }
    for(std::uint32_t len=2;len<=B;len<<=1){
        const std::uint32_t step=len>>1;
        for(std::uint32_t i=0;i<B;i+=len)
            for(std::uint32_t j=0;j<step;++j){
                const float a=out[i+j],b2=out[i+j+step];
                out[i+j]=a+b2;out[i+j+step]=a-b2;
            }
    }
    const float s=1.0f/std::sqrt(float(B));
    for(std::uint32_t i=0;i<B;++i)out[i]*=s;
}
void rope(float* x,std::uint32_t heads,std::uint32_t head_dim,std::uint32_t rotary_dim,
          std::uint32_t pos,float base,const std::string& scaling_type,float scaling_factor,std::uint32_t original_context,float low_freq_factor,float high_freq_factor,float yarn_beta_fast,float yarn_beta_slow){
    rotary_dim=std::min(rotary_dim,head_dim);rotary_dim-=rotary_dim%2;const std::uint32_t half=rotary_dim/2;
    float effective_base=base;
    if(scaling_type=="dynamic"||scaling_type=="dynamic_ntk"||scaling_type=="ntk")if(original_context>0&&pos+1>original_context&&rotary_dim>2){const float stretch=std::max(1.0f,scaling_factor*float(pos+1)/float(original_context)-(scaling_factor-1.0f));effective_base*=std::pow(stretch,float(rotary_dim)/float(rotary_dim-2));}
    float yarn_low=0.0f,yarn_high=float(half-1);if(scaling_type=="yarn"&&original_context>0){yarn_low=std::max(0.0f,float(rotary_dim)*std::log(float(original_context)/(yarn_beta_fast*2.0f*3.14159265f))/(2.0f*std::log(base)));yarn_high=std::min(float(half-1),float(rotary_dim)*std::log(float(original_context)/(yarn_beta_slow*2.0f*3.14159265f))/(2.0f*std::log(base)));if(yarn_high<=yarn_low)yarn_high=yarn_low+1.0f;}
    for(std::uint32_t h=0;h<heads;++h){float* v=x+std::size_t(h)*head_dim;
        for(std::uint32_t i=0;i<half;++i){float freq=std::pow(effective_base,-2.0f*float(i)/float(rotary_dim));float position=float(pos);
            if(scaling_type=="linear")position/=std::max(1.0f,scaling_factor);
            else if(scaling_type=="yarn"){const float ramp=std::clamp((float(i)-yarn_low)/(yarn_high-yarn_low),0.0f,1.0f);const float interpolated=freq/std::max(1.0f,scaling_factor);freq=interpolated*(1.0f-ramp)+freq*ramp;}
            else if(scaling_type=="llama3"&&original_context>0){const float wavelen=2.0f*3.14159265f/freq;const float low= float(original_context)/std::max(0.01f,low_freq_factor);const float high=float(original_context)/std::max(0.01f,high_freq_factor);if(wavelen>low)freq/=std::max(1.0f,scaling_factor);else if(wavelen>high){const float smooth=std::clamp((float(original_context)/wavelen-low_freq_factor)/(high_freq_factor-low_freq_factor),0.0f,1.0f);freq=(1.0f-smooth)*(freq/std::max(1.0f,scaling_factor))+smooth*freq;}}
            const float angle=position*freq,c=std::cos(angle),s=std::sin(angle);const float a=v[i],b=v[i+half];v[i]=a*c-b*s;v[i+half]=a*s+b*c;
        }
    }
}
// Return the exact filtered sampling distribution used by both ordinary and
// speculative decoding. Probabilities are indexed by token ID and sum to one.
std::vector<double> sampling_distribution(const std::vector<float>& logits,const GenerationOptions& o){
    std::vector<double> probabilities(logits.size(),0.0);
    if(logits.empty())return probabilities;
    std::vector<float> adjusted=logits;
    for(const auto&[id,bias]:o.logit_bias)if(id<adjusted.size())adjusted[id]+=bias;
    if(!o.allowed_token_ids.empty()){
        std::vector<bool> allowed(adjusted.size(),false);
        for(auto id:o.allowed_token_ids)if(id<allowed.size())allowed[id]=true;
        for(std::size_t i=0;i<adjusted.size();++i)if(!allowed[i])adjusted[i]=-std::numeric_limits<float>::infinity();
    }
    std::vector<std::size_t> order;order.reserve(adjusted.size());
    for(std::size_t i=0;i<adjusted.size();++i)if(std::isfinite(adjusted[i]))order.push_back(i);
    if(order.empty())return probabilities;
    const auto better=[&](std::size_t a,std::size_t b){return adjusted[a]>adjusted[b];};
    if(o.temperature<=0.0f||!std::isfinite(o.temperature)){
        std::size_t best=order.front();
        for(const auto id:order)if(adjusted[id]>adjusted[best])best=id;
        probabilities[best]=1.0;return probabilities;
    }
    if(o.top_k&&o.top_k<order.size()){
        std::partial_sort(order.begin(),order.begin()+o.top_k,order.end(),better);order.resize(o.top_k);
    }else std::sort(order.begin(),order.end(),better);
    const double maxv=adjusted[order.front()];double sum=0.0;
    for(auto id:order){const double p=std::exp((double(adjusted[id])-maxv)/o.temperature);probabilities[id]=p;sum+=p;}
    if(!(sum>0.0)||!std::isfinite(sum))return std::vector<double>(logits.size(),0.0);
    for(auto& p:probabilities)p/=sum;
    if(o.top_p>0.0f&&o.top_p<1.0f){
        double cumulative=0.0;std::size_t keep=0;
        for(;keep<order.size();++keep){cumulative+=probabilities[order[keep]];if(cumulative>=o.top_p){++keep;break;}}
        keep=std::max<std::size_t>(1,keep);
        std::vector<bool> retained(logits.size(),false);for(std::size_t i=0;i<keep;++i)retained[order[i]]=true;
        sum=0.0;for(std::size_t i=0;i<probabilities.size();++i){if(!retained[i])probabilities[i]=0.0;sum+=probabilities[i];}
        if(sum>0.0)for(auto& p:probabilities)p/=sum;
    }
    return probabilities;
}
std::uint32_t sample_distribution(const std::vector<double>& probabilities,std::mt19937_64& rng){
    if(probabilities.empty())return 0;
    std::discrete_distribution<std::size_t> dist(probabilities.begin(),probabilities.end());
    return static_cast<std::uint32_t>(dist(rng));
}
std::uint32_t sample_logits(const std::vector<float>& logits,const GenerationOptions& o,std::mt19937_64& rng){
    const auto probabilities=sampling_distribution(logits,o);
    return sample_distribution(probabilities,rng);
}
const quant::Format* format_for_ggml(std::uint32_t id){
    // Complete map of the runtime-decodable GGML storage types. A type missing
    // here silently degrades a raw-mapped weight to a full-file FP32 read (or
    // a load failure), even when quant.cpp has a working decoder.
    const char* name=nullptr;switch(id){
        case 0:name="F32";break; case 1:name="F16";break; case 30:name="BF16";break;
        case 2:name="Q4_0";break; case 3:name="Q4_1";break; case 6:name="Q5_0";break;
        case 7:name="Q5_1";break; case 8:name="Q8_0";break;
        case 10:name="Q2_K";break; case 11:name="Q3_K";break; case 12:name="Q4_K";break;
        case 13:name="Q5_K";break; case 14:name="Q6_K";break; case 15:name="Q8_K";break;
        case 16:name="IQ2_XXS";break; case 17:name="IQ2_XS";break; case 18:name="IQ3_XXS";break;
        case 19:name="IQ1_S";break; case 20:name="IQ4_NL";break; case 21:name="IQ3_S";break;
        case 22:name="IQ2_S";break; case 23:name="IQ4_XS";break; case 29:name="IQ1_M";break;
        case 34:name="TQ1_0";break; case 35:name="TQ2_0";break; case 39:name="MXFP4";break;
        case 100:name="Q4_0_ROCMFP4";break; case 101:name="Q4_0_ROCMFP4_FAST";break;
        case 143:name="PTQ1_0";break;
        default:break;}
    return name?quant::find(name):nullptr;
}
bool tensor_matvec(const DenseTransformerRuntime::Tensor& t,const float* x,float* y,std::size_t rows,std::size_t cols){
    const planner::ExecutionPlanner planner;const auto plan=planner.select(planner::Phase::decode,t.raw&&t.format,1,rows,cols);
    if(t.raw&&t.format){if(plan.kernel!=planner::Kernel::quantized_gemv||!quant::gemv(*t.format,t.raw,x,y,rows,cols,1.0f,plan.workers))return false;return finite_values(y,rows);}
    if(t.values.size()!=rows*cols)return false;
    matvec(t.values.data(),x,y,rows,cols);return finite_values(y,rows);
}
bool tensor_gemm(const DenseTransformerRuntime::Tensor& t,const float* input,float* output,std::size_t tokens,std::size_t rows,std::size_t cols){
    if(tokens==1)return tensor_matvec(t,input,output,rows,cols);
    const planner::ExecutionPlanner planner;const auto plan=planner.select(planner::Phase::prefill,t.raw&&t.format,tokens,rows,cols);
    if(t.raw&&t.format)return plan.kernel==planner::Kernel::quantized_gemm&&quant::gemm(*t.format,t.raw,input,output,tokens,rows,cols,1.0f,plan.workers);
    if(t.values.size()!=rows*cols)return false;
    for(std::size_t r=0;r<rows;++r){const float*w=t.values.data()+r*cols;for(std::size_t token=0;token<tokens;++token){
#ifdef ZEN_HAS_AVX2_KERNELS
        if(zen::cpu_has_avx2_fma()){zen::gemv_avx2(w,input+token*cols,output+token*rows+r,1,cols);continue;}
#endif
        double sum=0;for(std::size_t c=0;c<cols;++c)sum+=double(w[c])*input[token*cols+c];output[token*rows+r]=static_cast<float>(sum);
    }}return true;
}
}
const DenseTransformerRuntime::Tensor* DenseTransformerRuntime::tensor(const std::string& n)const noexcept{
    auto i=tensors_.find(n);return i==tensors_.end()?nullptr:&i->second;
}
void DenseTransformerRuntime::prefetch_layer(std::uint32_t layer)const noexcept{
    if(!mapping_)return;
    const std::string prefix="blk."+std::to_string(layer)+".";const std::uint8_t* first=nullptr;const std::uint8_t* last=nullptr;
    for(const auto&[name,t]:tensors_)if(name.rfind(prefix,0)==0&&t.raw&&t.raw_bytes){if(!first||t.raw<first)first=t.raw;const auto* end=t.raw+t.raw_bytes;if(!last||end>last)last=end;}
    if(first&&last>first)mapping_->prefetch(first,static_cast<std::size_t>(last-first));
}
bool DenseTransformerRuntime::layer_is_recurrent(std::uint32_t l)const noexcept{
    // (l+1) % full_attention_interval != 0 => GDN layer (0-based), per upstream.
    return (l+1)%config_.full_attention_interval!=0;
}
void DenseTransformerRuntime::hadamard_transform_input(float* x,std::size_t width,std::vector<float>& scratch)const{
    // Forward input transform for folded weights W_f = W*D*H: apply signs D,
    // then the normalized block Hadamard H (build_hadamard_input order).
    const auto& H=hadamard_;if(!H.block_size)return;
    const auto sit=H.signs.find(width);const float* signs=sit==H.signs.end()?nullptr:sit->second.data();
    if(signs)for(std::size_t i=0;i<width;++i)x[i]*=signs[i];
    scratch.resize(H.block_size);
    for(std::size_t off=0;off<width;off+=H.block_size){
        float* blk=x+off;
        std::copy_n(blk,H.block_size,scratch.begin());
        swht_block(scratch.data(),blk,H.block_size);
    }
}
void DenseTransformerRuntime::hadamard_unrotate(float* x,std::size_t width,std::vector<float>& scratch)const{
    // Inverse-table restore (rotated embedding rows): H first, then signs.
    const auto& H=hadamard_;if(!H.block_size)return;
    scratch.resize(H.block_size);
    for(std::size_t off=0;off<width;off+=H.block_size){
        float* blk=x+off;
        std::copy_n(blk,H.block_size,scratch.begin());
        swht_block(scratch.data(),blk,H.block_size);
    }
    const auto sit=H.signs.find(width);const float* signs=sit==H.signs.end()?nullptr:sit->second.data();
    if(signs)for(std::size_t i=0;i<width;++i)x[i]*=signs[i];
}
bool DenseTransformerRuntime::hadamard_folded(const char* suffix,std::size_t layer)const noexcept{
    if(!hadamard_.block_size)return false;
    std::string n="blk."+std::to_string(layer)+"."+suffix;
    return hadamard_.folded_names.count(n)!=0;
}
void DenseTransformerRuntime::Hadamard::transform(float* x,std::size_t width,std::vector<float>& scratch)const{
    (void)x;(void)width;(void)scratch;
}
void DenseTransformerRuntime::Hadamard::unrotate(float* x,std::size_t width,std::vector<float>& scratch)const{
    (void)x;(void)width;(void)scratch;
}
bool DenseTransformerRuntime::load(const std::filesystem::path& path,std::string* error){
    zen::diag::Scope trace("model.load",zen::diag::Phase::io);
    auto fail=[&](const std::string& e){if(error)*error=e;loaded_=false;tensors_.clear();mapping_.reset();{std::lock_guard<std::mutex> lock(prefix_cache_mutex_);prefix_cache_.reset();}return false;};
    auto report=gguf::inspect(path);if(!report.ok)return fail("GGUF: "+report.error);
    const auto adapter=std::find_if(model::architecture_descriptors().begin(),model::architecture_descriptors().end(),[&](const model::ArchitectureDescriptor& a){return a.recognises&&a.recognises(report.config);});
    if(adapter==model::architecture_descriptors().end())return fail("No architecture descriptor recognises this GGUF architecture");
    architecture_name_=report.config.architecture_name;chat_template_.clear();auto chat_it=report.config.metadata.find("tokenizer.chat_template");if(chat_it!=report.config.metadata.end())chat_template_=chat_it->second;
    const bool llama_arch=report.config.architecture==model::Architecture::llama;
    const bool qwen2_arch=report.config.architecture==model::Architecture::qwen&&architecture_name_=="qwen2";
    const bool qwen3_arch=report.config.architecture==model::Architecture::qwen&&architecture_name_=="qwen3";
    // BitNet b1.58 (arXiv 2504.12285): dense transformer with ternary BitLinear
    // projections and subln RMS norms; executed by this adapter with the
    // bitnet_variant branch. Reference: ggml-org/llama.cpp src/models/bitnet.cpp.
    const bool bitnet_arch=report.config.architecture==model::Architecture::bitnet;
    config_.bitnet_variant=bitnet_arch;
    // Qwen3.5 hybrid (Bonsai 2): gated delta-net + full-attention interleave.
    const bool qwen35_arch=report.config.architecture==model::Architecture::qwen_hybrid;
    config_.qwen35_variant=qwen35_arch;
    if(!llama_arch&&!qwen2_arch&&!qwen3_arch&&!bitnet_arch&&!qwen35_arch)return fail("Recognised architecture '"+architecture_name_+"' has no executable runtime adapter yet (dense Llama, Qwen2, Qwen3, BitNet and Qwen3.5-hybrid are supported)");
    if(report.config.expert_count>0||architecture_name_.find("moe")!=std::string::npos)return fail("MoE execution is not supported by this dense-model adapter");
    config_.embedding=report.config.embedding_length;config_.layers=report.config.block_count;
    config_.heads=report.config.attention_head_count;config_.kv_heads=report.config.kv_head_count;
    config_.ffn=report.config.feed_forward_length;config_.context=report.config.context_length;
    config_.vocab=report.config.vocabulary_size;
    // Head dimension: explicit attention.key_length metadata when present,
    // otherwise derived from the block attention-Q projection width (which
    // accommodates models whose q width is heads*head_dim != embedding), and
    // finally the classic embedding/heads.
    config_.head_dim=meta_uint_arch(report.config,architecture_name_,"attention.key_length",0);
    if(qwen35_arch){
        // qwen35: fused wq outputs Q+gate (2*QW). GGUF dims are ggml-order
        // {ne0=input, ne1=output}, so read the fused OUTPUT dim: dims[1].
        for(const auto& d:report.tensors)if(d.name.find(".attn_q.weight")!=std::string::npos&&d.dimensions.size()>=2&&d.dimensions[1]&&config_.heads){
            config_.head_dim=static_cast<std::uint32_t>(d.dimensions[1]/(2*config_.heads));break;}
    }
    if(!config_.head_dim){
        for(const auto& d:report.tensors)if(d.name=="blk.0.attn_q.weight"&&d.dimensions.size()>=1&&d.dimensions[0]&&config_.heads){
            config_.head_dim=static_cast<std::uint32_t>(d.dimensions[0]/config_.heads);break;}
    }
    if(!config_.head_dim)config_.head_dim=config_.embedding/config_.heads;
    config_.rope_dim=meta_uint_arch(report.config,architecture_name_,"rope.dimension_count",config_.embedding/config_.heads);
    config_.rope_base=meta_float_arch(report.config,architecture_name_,"rope.freq_base",10000.0f);
    config_.rms_epsilon=meta_float_arch(report.config,architecture_name_,"attention.layer_norm_rms_epsilon",1.0e-5f);
    config_.rope_scaling_type=meta_string_arch(report.config,architecture_name_,"rope.scaling.type");
    config_.rope_scaling_factor=meta_float_arch(report.config,architecture_name_,"rope.scaling.factor",meta_float_arch(report.config,architecture_name_,"rope.scale_linear",1.0f));
    config_.rope_low_freq_factor=meta_float_arch(report.config,architecture_name_,"rope.scaling.low_freq_factor",1.0f);
    config_.rope_high_freq_factor=meta_float_arch(report.config,architecture_name_,"rope.scaling.high_freq_factor",4.0f);
    config_.rope_yarn_beta_fast=meta_float_arch(report.config,architecture_name_,"rope.scaling.beta_fast",32.0f);
    config_.rope_yarn_beta_slow=meta_float_arch(report.config,architecture_name_,"rope.scaling.beta_slow",1.0f);
    config_.rope_original_context=meta_uint_arch(report.config,architecture_name_,"rope.scaling.original_context_length",report.config.context_length);
    config_.sliding_window=meta_uint_arch(report.config,architecture_name_,"attention.sliding_window",0);
    // Qwen3.5 GDN hyperparameters (meta key unit test: qwen35.ssm.*).
    config_.ssm_state=meta_uint_arch(report.config,architecture_name_,"ssm.state_size",0);
    config_.ssm_n_group=meta_uint_arch(report.config,architecture_name_,"ssm.group_count",0);
    config_.ssm_dt_rank=meta_uint_arch(report.config,architecture_name_,"ssm.time_step_rank",0);
    config_.ssm_d_conv=meta_uint_arch(report.config,architecture_name_,"ssm.conv_kernel",0);
    config_.full_attention_interval=meta_uint_arch(report.config,architecture_name_,"full_attention_interval",4);
    if(!config_.full_attention_interval)config_.full_attention_interval=4;
    config_.rope_sections[0]=report.config.rope_sections[0];
    config_.rope_sections[1]=report.config.rope_sections[1];
    config_.rope_sections[2]=report.config.rope_sections[2];
    config_.rope_sections[3]=report.config.rope_sections[3];
    config_.use_alibi=meta_bool_arch(report.config,architecture_name_,"attention.use_alibi",meta_bool_arch(report.config,architecture_name_,"attention.alibi",false));
    config_.bos_token=meta_uint(report.config,"tokenizer.ggml.bos_token_id",report.config.bos_token_id);
    config_.eos_token=meta_uint(report.config,"tokenizer.ggml.eos_token_id",report.config.eos_token_id);
    config_.add_bos_token=report.config.add_bos_token;
    config_.add_eos_token=report.config.add_eos_token;
    std::string tokenizer_error;tokenizer_loaded_=tokenizer_.load(report.config,&tokenizer_error);
    if(!config_.embedding||!config_.layers||!config_.heads||!config_.kv_heads||!config_.ffn||!config_.context||!config_.vocab||config_.embedding%config_.heads||config_.heads%config_.kv_heads)
        return fail("Incomplete or inconsistent dense transformer dimensions");
    const auto head_dim=config_.head_dim;
    if(config_.rope_dim==0||config_.rope_dim>head_dim||config_.rope_dim%2)return fail("Invalid rotary dimension");
    if(!std::isfinite(config_.rope_base)||config_.rope_base<=0.0f||!std::isfinite(config_.rms_epsilon)||config_.rms_epsilon<=0.0f||!std::isfinite(config_.rope_scaling_factor)||config_.rope_scaling_factor<1.0f||!std::isfinite(config_.rope_low_freq_factor)||config_.rope_low_freq_factor<=0.0f||!std::isfinite(config_.rope_high_freq_factor)||config_.rope_high_freq_factor<=config_.rope_low_freq_factor||!std::isfinite(config_.rope_yarn_beta_fast)||config_.rope_yarn_beta_fast<=0.0f||!std::isfinite(config_.rope_yarn_beta_slow)||config_.rope_yarn_beta_slow<=0.0f)return fail("Invalid RoPE base, scaling factor or RMS normalisation epsilon");
    if(!config_.rope_scaling_type.empty()&&config_.rope_scaling_type!="linear"&&config_.rope_scaling_type!="dynamic"&&config_.rope_scaling_type!="dynamic_ntk"&&config_.rope_scaling_type!="ntk"&&config_.rope_scaling_type!="yarn"&&config_.rope_scaling_type!="llama3")return fail("Unsupported RoPE scaling type: "+config_.rope_scaling_type);
    std::uint64_t estimated_weights=0;
    for(const auto& d:report.tensors){if(estimated_weights>std::numeric_limits<std::uint64_t>::max()-d.byte_size)return fail("Model file size overflow");estimated_weights+=d.byte_size;}
    zen::ResourceGovernor governor;const auto snapshot=governor.snapshot();const auto policy=governor.policy(snapshot);
    prefetch_enabled_=policy.prefetch_workers>0; // file-backed model pages are governed by the OS, not eagerly charged as resident FP32 weights
    std::string map_error;auto mapped=MappedFile::open(path,&map_error);if(!mapped)return fail(map_error);
    tensors_.clear();mapping_=mapped;path_=path;{std::lock_guard<std::mutex> lock(prefix_cache_mutex_);prefix_cache_.reset();}
    // Hadamard-fold transform setup (PrismML prism.hadamard.*). The normalized
    // Sylvester-Walsh-Hadamard block matrix and the explicit sign vectors are
    // rebuilt from metadata; folded weights consume signs-then-rotation of the
    // matmul input. Reference: PrismML llama.cpp llama-model.cpp hadamard setup.
    hadamard_=Hadamard{};
    if(!report.config.hadamard_weight_names.empty()){
        const std::uint32_t B=report.config.hadamard_block_size;
        if(!B||(B&(B-1))||B>4096)return fail("Invalid prism.hadamard.block_size");
        hadamard_.block_size=B;hadamard_.tied_output=report.config.hadamard_tied_output;
        // Upstream construction: rot[row*B+col] = (parity(row&col)?-1:+1)/sqrt(B).
        hadamard_.rot.resize(std::size_t(B)*B);
        const float hs=1.0f/std::sqrt(float(B));
        for(std::uint32_t row=0;row<B;++row)
            for(std::uint32_t col=0;col<B;++col){
                std::uint32_t parity=row&col;parity^=parity>>16;parity^=parity>>8;parity^=parity>>4;parity^=parity>>2;parity^=parity>>1;
                hadamard_.rot[std::size_t(row)*B+col]=(parity&1)?-hs:hs;
            }
        std::size_t off=0;
        for(std::int32_t w:report.config.hadamard_sign_widths){
            if(w<=0||off+std::size_t(w)>report.config.hadamard_sign_values.size())return fail("Invalid prism.hadamard sign widths");
            std::vector<float> sv;sv.reserve(std::size_t(w));
            for(std::int32_t j=0;j<w;++j){const std::int32_t v=report.config.hadamard_sign_values[off+std::size_t(j)];
                if(v!=1&&v!=-1)return fail("prism.hadamard sign values must be +/-1");
                sv.push_back(float(v));}
            hadamard_.signs.emplace(std::size_t(w),std::move(sv));off+=std::size_t(w);
        }
        if(off!=report.config.hadamard_sign_values.size())return fail("prism.hadamard sign_values length mismatch");
        for(const auto& n:report.config.hadamard_weight_names)hadamard_.folded_names.insert(n);
        for(const auto& n:report.config.hadamard_inverse_names)hadamard_.folded_names.insert("\x01"+n); // inverse marker
    }
    for(const auto& d:report.tensors){Tensor t;t.dims=d.dimensions;t.elements=1;for(auto dim:d.dimensions)t.elements*=static_cast<std::size_t>(dim);t.raw_bytes=static_cast<std::size_t>(d.byte_size);t.format=format_for_ggfl(d.ggml_type);
        const bool matrix=d.name.find(".bias")==std::string::npos&&(d.role==model::TensorRole::token_embedding||d.role==model::TensorRole::output||d.role==model::TensorRole::query||d.role==model::TensorRole::key||d.role==model::TensorRole::value||d.role==model::TensorRole::attention_output||d.role==model::TensorRole::ffn_gate||d.role==model::TensorRole::ffn_up||d.role==model::TensorRole::ffn_down);
        std::fprintf(stderr,"[map] %-38s type=%u role=%d matrix=%d fmt=%s bytes=%zu\n",d.name.c_str(),d.ggml_type,int(d.role),int(matrix),t.format?t.format->name.c_str():"null",t.raw_bytes);
        if(matrix&&t.format&&d.offset<=mapped->size()&&report.tensor_data_offset<=mapped->size()-d.offset&&t.raw_bytes<=mapped->size()-report.tensor_data_offset-d.offset){t.raw=mapped->data()+report.tensor_data_offset+d.offset;}
        else {std::string e;std::fprintf(stderr,"[map] fallback-decode %s\n",d.name.c_str());if(!gguf::read_tensor_f32(path,report,d.name,t.values,&e))return fail("Tensor "+d.name+": "+e);}
        tensors_.emplace(d.name,std::move(t));
    }
    auto require=[&](const std::string& n,std::size_t count){auto* t=tensor(n);return t&&t->elements==count&&(t->raw||t->values.size()==count);};
    if(!require("token_embd.weight",std::size_t(config_.vocab)*config_.embedding)||
       !require("output_norm.weight",config_.embedding))return fail("Missing or malformed embedding/output tensors");
    // GGUF files may omit output.weight when the vocabulary projection is tied to token embeddings.
    if(const auto* output=tensor("output.weight")){
        if(output->elements!=std::size_t(config_.vocab)*config_.embedding)return fail("Malformed output.weight tensor");
    }
    for(std::uint32_t l=0;l<config_.layers;++l){const std::string p="blk."+std::to_string(l)+".";
        const std::size_t QW=std::size_t(config_.heads)*head_dim,KW=std::size_t(config_.kv_heads)*head_dim;
        if(qwen35_arch){
            // Qwen3.5 hybrid: full-attention blocks every full_attention_interval
            // (0-based: (l+1)%interval!=0 is GDN), sandwich norms everywhere.
            // Reference: PrismML src/models/qwen35.cpp load_block_trunk.
            const std::uint32_t interval=config_.full_attention_interval;
            // (l+1) % interval != 0 => GDN; every 4th layer is full attention.
            const bool recurrent=(l+1)%interval!=0;
            auto req=[&](const char* n,std::size_t count){auto* t=tensor(p+n);return t&&t->elements==count&&(t->raw||t->values.size()==count);};
            std::string failed;
            if(recurrent){
                const std::size_t key_dim=std::size_t(config_.ssm_state)*config_.ssm_n_group;
                const std::size_t value_dim=std::size_t(config_.ssm_state)*config_.ssm_dt_rank;
                const std::size_t conv_dim=key_dim*2+value_dim;
                auto r=[&](const char* n,std::size_t c){if(!req(n,c)&&failed.empty())failed=n;};
                r("attn_norm.weight",config_.embedding);r("attn_qkv.weight",conv_dim*config_.embedding);
                r("attn_gate.weight",value_dim*config_.embedding);
                r("ssm_conv1d.weight",conv_dim*config_.ssm_d_conv);
                r("ssm_dt.bias",config_.ssm_dt_rank);r("ssm_a",config_.ssm_dt_rank);
                r("ssm_alpha.weight",config_.ssm_dt_rank*config_.embedding);
                r("ssm_beta.weight",config_.ssm_dt_rank*config_.embedding);
                r("ssm_norm.weight",config_.ssm_state);
                r("ssm_out.weight",config_.embedding*value_dim);
                r("post_attention_norm.weight",config_.embedding);
                r("ffn_gate.weight",config_.ffn*config_.embedding);
                r("ffn_up.weight",config_.ffn*config_.embedding);
                r("ffn_down.weight",config_.embedding*config_.ffn);
                if(!failed.empty())return fail("Missing or malformed Qwen3.5 GDN block tensors: "+p+" ["+failed+"]");
            }else{
                auto r=[&](const char* n,std::size_t c){if(!req(n,c)&&failed.empty())failed=n;};
                // Full-attention blocks fuse the output gate into the Q
                // projection: wq outputs n_embd_head_k*n_head*2 (Q|gate).
                r("attn_norm.weight",config_.embedding);r("attn_q.weight",2*QW*config_.embedding);
                r("attn_k.weight",KW*config_.embedding);
                r("attn_v.weight",KW*config_.embedding);
                r("attn_output.weight",QW*config_.embedding);
                r("attn_q_norm.weight",head_dim);r("attn_k_norm.weight",head_dim);
                r("post_attention_norm.weight",config_.embedding);
                r("ffn_gate.weight",config_.ffn*config_.embedding);
                r("ffn_up.weight",config_.ffn*config_.embedding);
                r("ffn_down.weight",config_.embedding*config_.ffn);
                if(!failed.empty())return fail("Missing or malformed Qwen3.5 attention block tensors: "+p+" ["+failed+"]");
            }
            continue;
        }
        if(bitnet_arch){
            // BitNet: ternary BitLinear projections, subln RMS norms, no
            // per-head Q/K norms, tied lm_head. Tensor set and execution order
            // per ggml-org/llama.cpp src/models/bitnet.cpp.
            if(!require(p+"attn_norm.weight",config_.embedding)||!require(p+"attn_q.weight",QW*config_.embedding)||
               !require(p+"attn_k.weight",KW*config_.embedding)||
               !require(p+"attn_v.weight",KW*config_.embedding)||
               !require(p+"attn_output.weight",QW*config_.embedding)||
               !require(p+"attn_sub_norm.weight",config_.embedding)||
               !require(p+"ffn_norm.weight",config_.embedding)||!require(p+"ffn_gate.weight",config_.ffn*config_.embedding)||
               !require(p+"ffn_up.weight",config_.ffn*config_.embedding)||
               !require(p+"ffn_sub_norm.weight",config_.ffn)||
               !require(p+"ffn_down.weight",config_.embedding*config_.ffn))return fail("Missing or malformed BitNet block tensors: "+p);
            continue;
        }
        if(!require(p+"attn_norm.weight",config_.embedding)||!require(p+"attn_q.weight",QW*config_.embedding)||
           !require(p+"attn_k.weight",KW*config_.embedding)||
           !require(p+"attn_v.weight",KW*config_.embedding)||
           !require(p+"attn_output.weight",QW*config_.embedding)||
           !require(p+"ffn_norm.weight",config_.embedding)||!require(p+"ffn_gate.weight",config_.ffn*config_.embedding)||
           !require(p+"ffn_up.weight",config_.ffn*config_.embedding)||!require(p+"ffn_down.weight",config_.embedding*config_.ffn))return fail("Missing or malformed dense transformer block tensors: "+p);
        const auto* qnorm=tensor(p+"attn_q_norm.weight");
        const auto* knorm=tensor(p+"attn_k_norm.weight");
        if((qnorm==nullptr)!=(knorm==nullptr))return fail("Q/K normalisation tensors must be supplied as a pair: "+p);
        if(qnorm&&(qnorm->elements!=head_dim||knorm->elements!=head_dim||qnorm->values.size()!=head_dim||knorm->values.size()!=head_dim))
            return fail("Malformed Q/K normalisation tensors: "+p+" (head_dim="+std::to_string(head_dim)+" q_elements="+std::to_string(qnorm->elements)+" q_values="+std::to_string(qnorm->values.size())+")");
        if(qwen3_arch&&(!qnorm||!knorm))return fail("Qwen3 requires per-head Q/K RMS normalisation tensors: "+p);
    }
    loaded_=true;if(error)error->clear();
    return true;
}
bool DenseTransformerRuntime::forward_q35_batch(const std::vector<std::uint32_t>& tokens,
        std::vector<std::vector<float>>& kc,std::vector<std::vector<float>>& vc,
        std::vector<std::vector<float>>& convc,std::vector<std::vector<float>>& ssmc,
        std::vector<float>& logits,std::uint32_t start_pos) const{
    const std::size_t T=tokens.size(),E=config_.embedding,H=config_.heads,KH=config_.kv_heads,D=config_.head_dim,F=config_.ffn;
    const std::size_t S=config_.ssm_state,NK=config_.ssm_n_group,NV=config_.ssm_dt_rank,CK=config_.ssm_d_conv;
    if(!T||!S||!NK||!NV||!CK)return false;
    const std::size_t QW=std::size_t(H)*D,KW=std::size_t(KH)*D;
    const std::size_t key_dim=std::size_t(S)*NK,value_dim=std::size_t(S)*NV,conv_dim=key_dim*2+value_dim;
    const std::size_t head_v=value_dim/NV; // == S
    const float eps=config_.rms_epsilon;
    std::vector<float> x(T*E),norm(T*E),tmp(T*E);
    std::vector<float> qkv(T*conv_dim),z(T*value_dim),conv_out(T*conv_dim);
    std::vector<float> q(T*QW),k(T*KW),v(T*KW),att(T*QW);
    std::vector<float> gate(T*F),up(T*F),ffn(T*F);
    std::vector<float> beta(T*NV),alpha(T*NV),a_gate(T*NV);
    std::vector<float> h_scratch;
    const auto* emb=tensor("token_embd.weight");if(!emb)return false;
    for(std::size_t t=0;t<T;++t){if(tokens[t]>=config_.vocab)return false;
        if(emb->raw&&emb->format){const auto rb=quant::row_bytes(*emb->format,E);if(!rb||!quant::dequantize(*emb->format,emb->raw+std::size_t(tokens[t])*rb,E,x.data()+t*E))return false;}
        else if(emb->values.size()==std::size_t(config_.vocab)*E)std::copy_n(emb->values.data()+std::size_t(tokens[t])*E,E,x.data()+t*E);else return false;
        // the embedding table is stored Hadamard-rotated: restore primal rows.
        if(hadamard_.block_size)hadamard_unrotate(x.data()+t*E,E,h_scratch);
    }
    // Per-layer folded matmul inputs get the forward Hadamard transform.
    auto fold=[&](const char* suffix,std::uint32_t l,float* input,std::size_t width){
        if(hadamard_folded(suffix,l))hadamard_transform_input(input,width,h_scratch);
    };
    for(std::uint32_t l=0;l<config_.layers;++l){
        const std::string p="blk."+std::to_string(l)+".";
        const bool recurrent=layer_is_recurrent(l);
        for(std::size_t t=0;t<T;++t)rms_norm(x.data()+t*E,tensor(p+"attn_norm.weight")->values.data(),norm.data()+t*E,E,eps);
        if(recurrent){
            // ---- gated delta net layer ----
            fold("attn_qkv.weight",l,norm.data(),E);fold("attn_gate.weight",l,norm.data(),E);
            fold("ssm_beta.weight",l,norm.data(),E);fold("ssm_alpha.weight",l,norm.data(),E);
            fold("ffn_gate.weight",l,norm.data(),E);fold("ffn_up.weight",l,norm.data(),E);
            if(!tensor_gemm(*tensor(p+"attn_qkv.weight"),norm.data(),qkv.data(),T,conv_dim,E)||
               !tensor_gemm(*tensor(p+"attn_gate.weight"),norm.data(),z.data(),T,value_dim,E))return false;
            if(!tensor_gemm(*tensor(p+"ssm_beta.weight"),norm.data(),beta.data(),T,NV,E)||
               !tensor_gemm(*tensor(p+"ssm_alpha.weight"),norm.data(),alpha.data(),T,NV,E))return false;
            std::vector<float> gdn_out(T*value_dim);
            // conv1d over [state_rows | qkv], per channel; state rows live in convc[l].
            for(std::size_t ch=0;ch<conv_dim;++ch){
                const auto* cw=tensor(p+"ssm_conv1d.weight")->values.data()+ch*CK;
                float* out=conv_out.data()+ch;
                for(std::size_t t=0;t<T;++t){
                    float sum=0.0f;
                    for(std::size_t w=0;w<CK;++w){
                        const std::int64_t idx=std::int64_t(t)+std::int64_t(w)-(std::int64_t)CK+1;
                        const float val=idx<0?convc[l][std::size_t(-idx-1)*conv_dim+ch]:qkv[std::size_t(idx)*conv_dim+ch];
                        sum+=cw[w]*val;
                    }
                    out[t*conv_dim]=sum;
                }
            }
            for(std::size_t i=0;i<T*conv_dim;++i)conv_out[i]=conv_out[i]/(1.0f+std::exp(-conv_out[i]));
            // q/k l2-norm (joint view), v as-is
            const std::size_t qk_span=key_dim*2;
            for(std::size_t t=0;t<T;++t){
                float* row=conv_out.data()+t*conv_dim;
                double ss=0.0;for(std::size_t i=0;i<qk_span;++i)ss+=double(row[i])*row[i];
                const float inv=1.0f/std::max(std::sqrt(float(ss)),eps);
                for(std::size_t i=0;i<qk_span;++i)row[i]*=inv;
                std::copy_n(row+qk_span,value_dim,v.data()+t*value_dim);
                std::copy_n(row,key_dim,q.data()+t*key_dim);
                std::copy_n(row+key_dim,key_dim,k.data()+t*key_dim);
            }
            // GDN recurrence per value head, per token (scalar-gate path).
            const std::uint32_t interval=config_.full_attention_interval;
            (void)interval;
            const std::size_t head_v_dim=S;
            std::vector<float> state(value_dim*value_dim); // per-layer single-sequence state [NV*S, NV*S] flattened per head below
            // state layout mirrors ggml: per head S x S, stored transposed rows.
            // ssm cache: ssmc[l] holds NV*S*S floats.
            std::copy_n(ssmc[l].begin(),ssmc[l].size(),state.begin());
            const float scale=1.0f/std::sqrt(float(S));
            // gates: beta = sigmoid(beta_proj); decay g = a * softplus(alpha + dt_bias); a = -exp(ssm_a)
            const auto* ssm_a=tensor(p+"ssm_a")->values.data();
            const auto* dt_bias=tensor(p+"ssm_dt.bias")->values.data();
            for(std::size_t t=0;t<T;++t){
                for(std::size_t h=0;h<NV;++h){
                    beta[t*NV+h]=1.0f/(1.0f+std::exp(-beta[t*NV+h]));
                    const float xx=alpha[t*NV+h]+dt_bias[h];
                    const float sp=xx>20.0f?xx:std::log(1.0f+std::exp(xx));
                    a_gate[h]=-std::exp(ssm_a[h])*sp;
                }
                // q/k repeat: NK != NV -> tile key heads to value heads
                const std::size_t rep=NV/NK;
                for(std::size_t t2=t;;){
                    (void)t2;break;
                }
                // per head update
                for(std::size_t h=0;h<NV;++h){
                    const std::size_t kh=(NV/NK==0)?0:h*key_dim/value_dim; // key head index (tiled)
                    const float* qh=q.data()+t*key_dim+kh*S;
                    const float* kh_=k.data()+t*key_dim+kh*S;
                    const float* vh=conv_out.data()+t*conv_dim+qk_span+h*S; // value from conv (grouped order)
                    float* sh=state.data()+h*S*S;
                    const float g=std::exp(a_gate[h]);
                    for(std::size_t i=0;i<S;++i)for(std::size_t j=0;j<S;++j)sh[i*S+j]*=g;
                    // delta[j] = (v[j] - sum_i S[i][j]*k[i]) * beta
                    std::vector<float> delta(S);
                    for(std::size_t j=0;j<S;++j){
                        float sum=0.0f;for(std::size_t i=0;i<S;++i)sum+=sh[i*S+j]*kh_[i];
                        delta[j]=(vh[j]-sum)*beta[t*NV+h];
                    }
                    for(std::size_t j=0;j<S;++j)for(std::size_t i=0;i<S;++i)sh[i*S+j]+=kh_[i]*delta[j];
                    // out[j] = sum_i S[i][j]*q[i] * scale
                    float* oh=gdn_out.data()+t*value_dim+h*S;
                    for(std::size_t j=0;j<S;++j){
                        float sum=0.0f;for(std::size_t i=0;i<S;++i)sum+=sh[i*S+j]*qh[i];
                        oh[j]=sum*scale;
                    }
                }
            }
            std::copy_n(state.begin(),ssmc[l].size(),ssmc[l].begin());
            // conv state update: last CK-1 inputs of [state|qkv]
            for(std::size_t ch=0;ch<conv_dim;++ch)
                for(std::size_t w=0;w<CK-1;++w){
                    const std::int64_t idx=std::int64_t(T)+std::int64_t(w)-(std::int64_t)(CK-1);
                    convc[l][w*conv_dim+ch]=idx<0?convc[l][std::size_t(-idx-1)*conv_dim+ch]:qkv[std::size_t(idx)*conv_dim+ch];
                }
            // gated norm: per-head RMS over S, then silu(z) * normalized.
            for(std::size_t t=0;t<T;++t){
                float* o=gdn_out.data()+t*value_dim;
                for(std::size_t h=0;h<NV;++h)rms_norm(o+h*S,tensor(p+"ssm_norm.weight")->values.data(),o+h*S,S,eps);
                for(std::size_t h=0;h<NV;++h)
                    for(std::size_t j=0;j<S;++j){
                        const float gate_z=z[t*value_dim+h*S+j];
                        o[h*S+j]*=gate_z/(1.0f+std::exp(-gate_z));
                    }
            }
            fold("ssm_out.weight",l,gdn_out.data(),value_dim);
            if(!tensor_gemm(*tensor(p+"ssm_out.weight"),gdn_out.data(),tmp.data(),T,E,value_dim))return false;
        }else{
            // ---- full attention layer: fused Q|gate projection (Qwen3.5) ----
            const std::size_t QGW=2*QW;
            q.resize(T*QGW);
            fold("attn_q.weight",l,norm.data(),E);fold("attn_k.weight",l,norm.data(),E);
            fold("attn_v.weight",l,norm.data(),E);
            fold("ffn_gate.weight",l,norm.data(),E);fold("ffn_up.weight",l,norm.data(),E);
            if(!tensor_gemm(*tensor(p+"attn_q.weight"),norm.data(),q.data(),T,QGW,E)||
               !tensor_gemm(*tensor(p+"attn_k.weight"),norm.data(),k.data(),T,KW,E)||
               !tensor_gemm(*tensor(p+"attn_v.weight"),norm.data(),v.data(),T,KW,E))return false;
            // split: first QW = Q, second QW = output gate
            std::vector<float> qgate(T*QW);
            for(std::size_t t=0;t<T;++t){
                std::copy_n(q.data()+t*QGW,QW,q.data()+t*QW);
                std::copy_n(q.data()+t*QGW+QW,QW,qgate.data()+t*QW);
            }
            for(std::size_t t=0;t<T;++t){
                const std::uint32_t pos=start_pos+static_cast<std::uint32_t>(t);
                qk_rms_norm(q.data()+t*QW,H,D,tensor(p+"attn_q_norm.weight")->values.data(),eps);
                qk_rms_norm(k.data()+t*KW,KH,D,tensor(p+"attn_k_norm.weight")->values.data(),eps);
                // Full-attention layers use M-RoPE sections [11,11,10,0]. For
                // text all four position streams equal the token position, so
                // ggml_mrope_cache_init degenerates to the standard RoPE cache
                // over the 64 rotary dims; rope() is exactly that. (Vision
                // inputs would need the section-split streams.)
                rope(q.data()+t*QW,H,D,config_.rope_dim,pos,config_.rope_base,config_.rope_scaling_type,config_.rope_scaling_factor,config_.rope_original_context,config_.rope_low_freq_factor,config_.rope_high_freq_factor,config_.rope_yarn_beta_fast,config_.rope_yarn_beta_slow);
                rope(k.data()+t*KW,KH,D,config_.rope_dim,pos,config_.rope_base,config_.rope_scaling_type,config_.rope_scaling_factor,config_.rope_original_context,config_.rope_low_freq_factor,config_.rope_high_freq_factor,config_.rope_yarn_beta_fast,config_.rope_yarn_beta_slow);
                std::copy_n(k.data()+t*KW,KW,kc[l].data()+std::size_t(pos)*KW);
                std::copy_n(v.data()+t*KW,KW,vc[l].data()+std::size_t(pos)*KW);
            }
            std::fill(att.begin(),att.end(),0.0f);
            std::vector<float> scores(config_.context);
            const std::uint32_t group=H/KH;const float inv_scale=1.0f/std::sqrt(float(D));
            for(std::size_t t=0;t<T;++t)for(std::uint32_t h=0;h<H;++h){
                const auto kh=h/group;const float*qh=q.data()+t*QW+std::size_t(h)*D;
                float max_score=-std::numeric_limits<float>::infinity();
                const std::uint32_t pos=start_pos+static_cast<std::uint32_t>(t);
                const std::size_t first=config_.sliding_window&&pos+1>config_.sliding_window?pos+1-config_.sliding_window:0;
                for(std::size_t j=first;j<=pos;++j){
                    const float*kt=kc[l].data()+j*KW+std::size_t(kh)*D;double dot=0;
                    for(std::size_t i=0;i<D;++i)dot+=double(qh[i])*kt[i];
                    scores[j]=static_cast<float>(dot)*inv_scale;
                    if(!std::isfinite(scores[j]))return false;max_score=std::max(max_score,scores[j]);
                }
                double denom=0;for(std::size_t j=first;j<=pos;++j){scores[j]=std::exp(scores[j]-max_score);denom+=scores[j];}
                if(!(denom>0.0)||!std::isfinite(denom))return false;
                float*out=att.data()+t*QW+std::size_t(h)*D;
                for(std::size_t j=first;j<=pos;++j){const float a=scores[j]/static_cast<float>(denom);
                    const float*vt=vc[l].data()+j*KW+std::size_t(kh)*D;for(std::size_t i=0;i<D;++i)out[i]+=a*vt[i];}
            }
            // sigmoid gate after attention (ggml: attn_pregate * sigmoid(gate))
            for(std::size_t t=0;t<T;++t)for(std::size_t i=0;i<QW;++i){const float g=qgate[t*QW+i];att[t*QW+i]*=g/(1.0f+std::exp(-g));}
            fold("attn_output.weight",l,att.data(),QW);
            if(!tensor_gemm(*tensor(p+"attn_output.weight"),att.data(),tmp.data(),T,E,QW))return false;
        }
        for(std::size_t i=0;i<T*E;++i)x[i]+=tmp[i];
        // post-attention norm then FFN (sandwich norm), residual from before it.
        std::vector<float> res_before(T*E);std::copy_n(x.data(),T*E,res_before.begin());
        for(std::size_t t=0;t<T;++t)rms_norm(x.data()+t*E,tensor(p+"post_attention_norm.weight")->values.data(),norm.data()+t*E,E,eps);
        fold("ffn_gate.weight",l,norm.data(),E);fold("ffn_up.weight",l,norm.data(),E);
        if(!tensor_gemm(*tensor(p+"ffn_gate.weight"),norm.data(),gate.data(),T,F,E)||
           !tensor_gemm(*tensor(p+"ffn_up.weight"),norm.data(),up.data(),T,F,E))return false;
        for(std::size_t i=0;i<T*F;++i){const float g=gate[i];ffn[i]=(g/(1.0f+std::exp(-g)))*up[i];}
        fold("ffn_down.weight",l,ffn.data(),F);
        if(!tensor_gemm(*tensor(p+"ffn_down.weight"),ffn.data(),tmp.data(),T,E,F))return false;
        for(std::size_t i=0;i<T*E;++i)x[i]+=res_before[i]+tmp[i];
        if(!finite_values(x.data(),x.size()))return false;
    }
    std::vector<float> norm_last(E);
    rms_norm(x.data()+(T-1)*E,tensor("output_norm.weight")->values.data(),norm_last.data(),E,eps);
    // embedding/LM head are Hadamard-folded: transform the input.
    if(hadamard_.block_size)hadamard_transform_input(norm_last.data(),E,h_scratch);
    std::vector<float> logits_full(config_.vocab);
    const auto* output=tensor("output.weight");
    if(!output)output=emb;
    if(!tensor_matvec(*output,norm_last.data(),logits_full.data(),config_.vocab,E))return false;
    logits=std::move(logits_full);
    return finite_values(logits.data(),logits.size());
}

bool DenseTransformerRuntime::forward_q35_token(std::uint32_t token, std::uint32_t position,
        std::vector<std::vector<float>>& kc,std::vector<std::vector<float>>& vc,
        std::vector<std::vector<float>>& convc,std::vector<std::vector<float>>& ssmc,
        std::vector<float>& logits) const{
    return forward_q35_batch(std::vector<std::uint32_t>{token},kc,vc,convc,ssmc,logits,position);
}

bool DenseTransformerRuntime::forward_batch(const std::vector<std::uint32_t>& tokens,
        std::vector<std::vector<float>>& kc,std::vector<std::vector<float>>& vc,
        std::vector<float>& logits,std::uint32_t start_pos,std::size_t tail_count,std::vector<float>* tail_logits)const{
    const std::size_t T=tokens.size(),E=config_.embedding,H=config_.heads,KH=config_.kv_heads,D=config_.head_dim,F=config_.ffn;
    if(!T||start_pos>config_.context||T>config_.context-start_pos||tail_count==0||tail_count>T)return false;
    const std::size_t QW=std::size_t(H)*D,KW=std::size_t(KH)*D;
    std::vector<float> x(T*E),norm(T*E),q(T*QW),k(T*KW),v(T*KW),att(T*QW),tmp(T*E),gate(T*F),up(T*F),ffn(T*F);
    const auto* emb=tensor("token_embd.weight");if(!emb)return false;
    for(std::size_t t=0;t<T;++t){if(tokens[t]>=config_.vocab)return false;
        if(emb->raw&&emb->format){const auto rb=quant::row_bytes(*emb->format,E);if(!rb||!quant::dequantize(*emb->format,emb->raw+std::size_t(tokens[t])*rb,E,x.data()+t*E))return false;}
        else if(emb->values.size()==std::size_t(config_.vocab)*E)std::copy_n(emb->values.data()+std::size_t(tokens[t])*E,E,x.data()+t*E);else return false;
    }
    std::future<void> prefetch;
    for(std::uint32_t l=0;l<config_.layers;++l){if(prefetch.valid())prefetch.get();if(l+1<config_.layers&&prefetch_enabled_)prefetch=std::async(std::launch::async,[this,l]{prefetch_layer(l+1);});const std::string p="blk."+std::to_string(l)+".";
        for(std::size_t t=0;t<T;++t)rms_norm(x.data()+t*E,tensor(p+"attn_norm.weight")->values.data(),norm.data()+t*E,E,config_.rms_epsilon);
        if(!finite_values(norm.data(),norm.size()))return false;
        if(!tensor_gemm(*tensor(p+"attn_q.weight"),norm.data(),q.data(),T,QW,E)||
           !tensor_gemm(*tensor(p+"attn_k.weight"),norm.data(),k.data(),T,KW,E)||
           !tensor_gemm(*tensor(p+"attn_v.weight"),norm.data(),v.data(),T,KW,E))return false;
        if(!finite_values(q.data(),q.size())||!finite_values(k.data(),k.size())||!finite_values(v.data(),v.size()))return false;
        for(const auto& item:std::array<std::pair<const char*,std::vector<float>*>,3>{{{"attn_q.bias",&q},{"attn_k.bias",&k},{"attn_v.bias",&v}}})if(const auto*bias=tensor(p+item.first)){const std::size_t width=item.first[5]=='q'?QW:KW;if(bias->values.size()!=width)return false;for(std::size_t t=0;t<T;++t)for(std::size_t i=0;i<width;++i)(*item.second)[t*width+i]+=bias->values[i];}
        const auto* qnorm=tensor(p+"attn_q_norm.weight");const auto* knorm=tensor(p+"attn_k_norm.weight");
        if((qnorm==nullptr)!=(knorm==nullptr))return false;
        if(qnorm&&(qnorm->values.size()!=D||knorm->values.size()!=D))return false;
        for(std::size_t t=0;t<T;++t){const std::uint32_t pos=start_pos+static_cast<std::uint32_t>(t);if(qnorm){qk_rms_norm(q.data()+t*QW,H,D,qnorm->values.data(),config_.rms_epsilon);qk_rms_norm(k.data()+t*KW,KH,D,knorm->values.data(),config_.rms_epsilon);}if(!config_.use_alibi){rope(q.data()+t*QW,H,D,config_.rope_dim,pos,config_.rope_base,config_.rope_scaling_type,config_.rope_scaling_factor,config_.rope_original_context,config_.rope_low_freq_factor,config_.rope_high_freq_factor,config_.rope_yarn_beta_fast,config_.rope_yarn_beta_slow);rope(k.data()+t*KW,KH,D,config_.rope_dim,pos,config_.rope_base,config_.rope_scaling_type,config_.rope_scaling_factor,config_.rope_original_context,config_.rope_low_freq_factor,config_.rope_high_freq_factor,config_.rope_yarn_beta_fast,config_.rope_yarn_beta_slow);}
            std::copy_n(k.data()+t*KW,KW,kc[l].data()+std::size_t(pos)*KW);std::copy_n(v.data()+t*KW,KW,vc[l].data()+std::size_t(pos)*KW);}
        std::fill(att.begin(),att.end(),0.0f);std::vector<float> scores(config_.context);const std::uint32_t group=H/KH;const float inv_scale=1.0f/std::sqrt(float(D));
        for(std::size_t t=0;t<T;++t)for(std::uint32_t h=0;h<H;++h){const auto kh=h/group;const float*qh=q.data()+t*QW+std::size_t(h)*D;float max_score=-std::numeric_limits<float>::infinity();const std::uint32_t pos=start_pos+static_cast<std::uint32_t>(t);const std::size_t first=config_.sliding_window&&pos+1>config_.sliding_window?pos+1-config_.sliding_window:0;
            for(std::size_t j=first;j<=pos;++j){const float*kt=kc[l].data()+j*KW+std::size_t(kh)*D;double dot=0;for(std::size_t i=0;i<D;++i)dot+=double(qh[i])*kt[i];scores[j]=static_cast<float>(dot)*inv_scale;if(config_.use_alibi)scores[j]+=zen::attention::alibi_bias(h,H,pos,static_cast<std::uint32_t>(j));if(!std::isfinite(scores[j]))return false;max_score=std::max(max_score,scores[j]);}
            double denom=0;for(std::size_t j=first;j<=pos;++j){scores[j]=std::exp(scores[j]-max_score);denom+=scores[j];}if(!(denom>0.0)||!std::isfinite(denom))return false;
            float*out=att.data()+t*QW+std::size_t(h)*D;for(std::size_t j=first;j<=pos;++j){const float a=scores[j]/static_cast<float>(denom);const float*vt=vc[l].data()+j*KW+std::size_t(kh)*D;for(std::size_t i=0;i<D;++i)out[i]+=a*vt[i];}
        }
        // BitNet subln: RMS-normalise each token's attention output before
        // the wo projection.
        if(config_.bitnet_variant)for(std::size_t t=0;t<T;++t)rms_norm(att.data()+t*QW,tensor(p+"attn_sub_norm.weight")->values.data(),att.data()+t*QW,QW,config_.rms_epsilon);
        if(!tensor_gemm(*tensor(p+"attn_output.weight"),att.data(),tmp.data(),T,E,QW))return false;
        for(std::size_t i=0;i<T*E;++i)x[i]+=tmp[i];
        for(std::size_t t=0;t<T;++t)rms_norm(x.data()+t*E,tensor(p+"ffn_norm.weight")->values.data(),norm.data()+t*E,E,config_.rms_epsilon);
        if(!finite_values(norm.data(),norm.size()))return false;
        if(!tensor_gemm(*tensor(p+"ffn_gate.weight"),norm.data(),gate.data(),T,F,E)||!tensor_gemm(*tensor(p+"ffn_up.weight"),norm.data(),up.data(),T,F,E))return false;
        for(std::size_t i=0;i<T*F;++i){const float z=gate[i];ffn[i]=(z/(1.0f+std::exp(-z)))*up[i];}
        // BitNet subln: RMS-normalise the FFN hidden activation before ffn_down.
        if(config_.bitnet_variant)for(std::size_t t=0;t<T;++t)rms_norm(ffn.data()+t*F,tensor(p+"ffn_sub_norm.weight")->values.data(),ffn.data()+t*F,F,config_.rms_epsilon);
        if(!tensor_gemm(*tensor(p+"ffn_down.weight"),ffn.data(),tmp.data(),T,E,F))return false;
        for(std::size_t i=0;i<T*E;++i)x[i]+=tmp[i];
        if(!finite_values(x.data(),x.size()))return false;
    }
    const auto* norm_weight=tensor("output_norm.weight");if(!norm_weight||norm_weight->values.size()!=E)return false;
    const std::size_t first_tail=T-tail_count;std::vector<float> norm_tail(tail_count*E),logits_tail(tail_count*config_.vocab);
    for(std::size_t t=0;t<tail_count;++t)rms_norm(x.data()+(first_tail+t)*E,norm_weight->values.data(),norm_tail.data()+t*E,E,config_.rms_epsilon);
    if(!finite_values(norm_tail.data(),norm_tail.size()))return false;
    const auto* output=tensor("output.weight");if(!output)output=emb;
    if(!tensor_gemm(*output,norm_tail.data(),logits_tail.data(),tail_count,config_.vocab,E))return false;
    if(!finite_values(logits_tail.data(),logits_tail.size()))return false;
    logits.assign(logits_tail.end()-config_.vocab,logits_tail.end());if(tail_logits)*tail_logits=std::move(logits_tail);
    return true;
}
bool DenseTransformerRuntime::forward(std::uint32_t token,std::uint32_t pos,
        std::vector<std::vector<float>>& kc,std::vector<std::vector<float>>& vc,
        std::vector<float>& logits)const{
    const auto E=config_.embedding,H=config_.heads,KH=config_.kv_heads,D=config_.head_dim,F=config_.ffn;
    if(token>=config_.vocab||pos>=config_.context)return false;
    const std::size_t QW=std::size_t(H)*D,KW=std::size_t(KH)*D;
    const auto* emb=tensor("token_embd.weight");
    std::vector<float> x(E),norm(E),q(QW),k(KW),v(KW),att(QW),tmp(E),gate(F),up(F),ffn(F);
    if(emb->raw&&emb->format){const auto rb=quant::row_bytes(*emb->format,E);if(!rb||!quant::dequantize(*emb->format,emb->raw+std::size_t(token)*rb,E,x.data()))return false;}
    else if(emb->values.size()==std::size_t(config_.vocab)*E)std::copy_n(emb->values.data()+std::size_t(token)*E,E,x.data());else return false;
    std::future<void> prefetch;
    for(std::uint32_t l=0;l<config_.layers;++l){if(prefetch.valid())prefetch.get();if(l+1<config_.layers&&prefetch_enabled_)prefetch=std::async(std::launch::async,[this,l]{prefetch_layer(l+1);});
        const std::string p="blk."+std::to_string(l)+".";
        rms_norm(x.data(),tensor(p+"attn_norm.weight")->values.data(),norm.data(),E,config_.rms_epsilon);
        if(!finite_values(norm.data(),E))return false;
        {
            zen::diag::Scope scope("transformer.attention.projection",zen::diag::Phase::compute);
            if(!tensor_matvec(*tensor(p+"attn_q.weight"),norm.data(),q.data(),QW,E))return false;
            if(!tensor_matvec(*tensor(p+"attn_k.weight"),norm.data(),k.data(),KW,E))return false;
            if(!tensor_matvec(*tensor(p+"attn_v.weight"),norm.data(),v.data(),KW,E))return false;
            for(const auto& item:std::array<std::pair<const char*,std::vector<float>*>,3>{{{"attn_q.bias",&q},{"attn_k.bias",&k},{"attn_v.bias",&v}}})if(const auto*bias=tensor(p+item.first)){if(bias->values.size()!=item.second->size())return false;for(std::size_t i=0;i<item.second->size();++i)(*item.second)[i]+=bias->values[i];}
            if(!finite_values(q.data(),q.size())||!finite_values(k.data(),k.size())||!finite_values(v.data(),v.size()))return false;
            const auto* qnorm=tensor(p+"attn_q_norm.weight");const auto* knorm=tensor(p+"attn_k_norm.weight");
            if((qnorm==nullptr)!=(knorm==nullptr))return false;
            if(qnorm){if(qnorm->values.size()!=D||knorm->values.size()!=D)return false;qk_rms_norm(q.data(),H,D,qnorm->values.data(),config_.rms_epsilon);qk_rms_norm(k.data(),KH,D,knorm->values.data(),config_.rms_epsilon);}
            if(!config_.use_alibi){
                rope(q.data(),H,D,config_.rope_dim,pos,config_.rope_base,config_.rope_scaling_type,config_.rope_scaling_factor,config_.rope_original_context,config_.rope_low_freq_factor,config_.rope_high_freq_factor,config_.rope_yarn_beta_fast,config_.rope_yarn_beta_slow);
                rope(k.data(),KH,D,config_.rope_dim,pos,config_.rope_base,config_.rope_scaling_type,config_.rope_scaling_factor,config_.rope_original_context,config_.rope_low_freq_factor,config_.rope_high_freq_factor,config_.rope_yarn_beta_fast,config_.rope_yarn_beta_slow);
            }
            std::copy(k.begin(),k.end(),kc[l].begin()+std::size_t(pos)*KW);
            std::copy(v.begin(),v.end(),vc[l].begin()+std::size_t(pos)*KW);
        }
        {
            zen::diag::Scope scope("transformer.attention.softmax",zen::diag::Phase::compute);
            std::fill(att.begin(),att.end(),0.0f);
            std::vector<float> scores(pos+1);
            const std::uint32_t group=H/KH;
            const float inv_scale=1.0f/std::sqrt(float(D));
            for(std::uint32_t h=0;h<H;++h){
                const std::uint32_t kh=h/group;
                const float* qh=q.data()+std::size_t(h)*D;
                float max_score=-std::numeric_limits<float>::infinity();const std::uint32_t first=config_.sliding_window&&pos+1>config_.sliding_window?pos+1-config_.sliding_window:0;
                for(std::uint32_t t=first;t<=pos;++t){
                    const float* kt=kc[l].data()+std::size_t(t)*KW+std::size_t(kh)*D;
                    double dot=0.0;
                    for(std::uint32_t i=0;i<D;++i)dot+=double(qh[i])*kt[i];
                    scores[t]=float(dot)*inv_scale;
                    if(config_.use_alibi)scores[t]+=zen::attention::alibi_bias(h,H,pos,t);
                    if(!std::isfinite(scores[t]))return false;
                    max_score=std::max(max_score,scores[t]);
                }
                double denom=0.0;
                for(std::uint32_t t=first;t<=pos;++t){scores[t]=std::exp(scores[t]-max_score);denom+=scores[t];}
                float* out=att.data()+std::size_t(h)*D;
                if(!(denom>0.0)||!std::isfinite(denom))return false;
                const float inv=1.0f/static_cast<float>(denom);
                for(std::uint32_t t=first;t<=pos;++t){
                    const float a=scores[t]*inv;
                    const float* vt=vc[l].data()+std::size_t(t)*KW+std::size_t(kh)*D;
                    for(std::uint32_t i=0;i<D;++i)out[i]+=a*vt[i];
                }
            }
            // BitNet subln: RMS-normalise the attention output before the
            // wo projection (src/models/bitnet.cpp: attn_sub_norm).
            if(config_.bitnet_variant)rms_norm(att.data(),tensor(p+"attn_sub_norm.weight")->values.data(),att.data(),QW,config_.rms_epsilon);
            if(!tensor_matvec(*tensor(p+"attn_output.weight"),att.data(),tmp.data(),E,QW))return false;
            for(std::uint32_t i=0;i<E;++i)x[i]+=tmp[i];
        }
        {
            zen::diag::Scope scope("transformer.ffn",zen::diag::Phase::compute);
            rms_norm(x.data(),tensor(p+"ffn_norm.weight")->values.data(),norm.data(),E,config_.rms_epsilon);
            if(!finite_values(norm.data(),E))return false;
            if(!tensor_matvec(*tensor(p+"ffn_gate.weight"),norm.data(),gate.data(),F,E))return false;
            if(!tensor_matvec(*tensor(p+"ffn_up.weight"),norm.data(),up.data(),F,E))return false;
            for(std::uint32_t i=0;i<F;++i){const float z=gate[i];ffn[i]=(z/(1.0f+std::exp(-z)))*up[i];}
            // BitNet subln: RMS-normalise the FFN hidden activation before
            // ffn_down (src/models/bitnet.cpp: ffn_sub_norm).
            if(config_.bitnet_variant)rms_norm(ffn.data(),tensor(p+"ffn_sub_norm.weight")->values.data(),ffn.data(),F,config_.rms_epsilon);
            if(!tensor_matvec(*tensor(p+"ffn_down.weight"),ffn.data(),tmp.data(),E,F))return false;
            for(std::uint32_t i=0;i<E;++i)x[i]+=tmp[i];
            if(!finite_values(x.data(),x.size()))return false;
        }
    }
    rms_norm(x.data(),tensor("output_norm.weight")->values.data(),norm.data(),E,config_.rms_epsilon);
    if(!finite_values(norm.data(),E))return false;
    logits.resize(config_.vocab);
    const auto* output=tensor("output.weight");
    if(!output)output=tensor("token_embd.weight");
    if(!tensor_matvec(*output,norm.data(),logits.data(),config_.vocab,E))return false;
    return finite_values(logits.data(),logits.size());
}
bool DenseTransformerRuntime::generate(const std::vector<std::uint32_t>& prompt,std::uint32_t max_new,
        Generation& result,std::string* error,
        const std::function<void(const std::vector<std::uint32_t>&)>& on_tokens,
        const GenerationOptions& options,const std::atomic<bool>* cancelled)const{
    auto fail=[&](const char* e){if(error)*error=e;return false;};
    if(!loaded_)return fail("No model is loaded");
    if(prompt.empty())return fail("Prompt token ID list is empty");
    if(!std::isfinite(options.temperature)||options.temperature<0.0f||options.temperature>5.0f)return fail("Temperature must be finite and between 0 and 5");
    if(!std::isfinite(options.top_p)||options.top_p<=0.0f||options.top_p>1.0f)return fail("top_p must be in (0, 1]");
    if(!options.allowed_token_ids.empty()){for(auto id:options.allowed_token_ids)if(id>=config_.vocab)return fail("Allowed token ID is outside vocabulary");}
    for(const auto&[id,bias]:options.logit_bias)if(id>=config_.vocab||!std::isfinite(bias))return fail("Logit bias contains an invalid token ID or non-finite value");
    if(prompt.size()>config_.context||max_new>config_.context-prompt.size())return fail("Prompt plus generation exceeds model context length");
    std::uint64_t kv_elements=config_.context;
    for(std::uint64_t factor:{std::uint64_t(config_.kv_heads),std::uint64_t(config_.embedding/config_.heads),std::uint64_t(config_.layers)}){
        if(factor&&kv_elements>std::numeric_limits<std::uint64_t>::max()/factor)return fail("KV cache size overflow");
        kv_elements*=factor;
    }
    if(kv_elements>std::numeric_limits<std::uint64_t>::max()/(2*sizeof(float)))return fail("KV cache size overflow");
    const auto kv_bytes=kv_elements*2*sizeof(float);
    zen::ResourceGovernor governor;const auto policy=governor.current_policy();
    if(kv_bytes>policy.transient_budget_bytes)return fail("KV cache exceeds ResourceGovernor transient-memory budget");
    if(kv_elements>std::numeric_limits<std::size_t>::max())return fail("KV cache exceeds addressable memory");
    const auto kvsize=static_cast<std::size_t>(std::uint64_t(config_.context)*config_.kv_heads*config_.head_dim);
    std::vector<std::vector<float>> kc,vc;std::vector<float> logits;bool cache_hit=false;
    // qwen35 hybrid: additional recurrent caches (conv windows + GDN states).
    const bool q35=config_.qwen35_variant;
    const std::size_t conv_dim=q35?(std::size_t(config_.ssm_state)*config_.ssm_n_group*2+std::size_t(config_.ssm_state)*config_.ssm_dt_rank):0;
    const std::size_t ssm_dim=q35?std::size_t(config_.ssm_dt_rank)*config_.ssm_state*config_.ssm_state:0;
    std::vector<std::vector<float>> convc,ssmc;
    using clock=std::chrono::steady_clock;auto t0=clock::now();
    {std::lock_guard<std::mutex> lock(prefix_cache_mutex_);if(prefix_cache_&&prefix_cache_->tokens==prompt){kc=prefix_cache_->key;vc=prefix_cache_->value;logits=prefix_cache_->logits;if(q35){convc=prefix_cache_->conv;ssmc=prefix_cache_->ssm;}cache_hit=true;}}
    if(!cache_hit){kc.assign(config_.layers,std::vector<float>(kvsize));vc.assign(config_.layers,std::vector<float>(kvsize));
        if(q35){convc.assign(config_.layers,std::vector<float>((config_.ssm_d_conv-1)*conv_dim));ssmc.assign(config_.layers,std::vector<float>(ssm_dim));}
        for(std::size_t i=0;i<prompt.size();++i){if(cancelled&&cancelled->load(std::memory_order_relaxed))return fail("Generation cancelled");if(prompt[i]>=config_.vocab)return fail("Prompt token ID is outside vocabulary");zen::diag::breadcrumb(401,static_cast<std::int64_t>(i),prompt[i]);}
        {zen::diag::Scope scope("transformer.prefill.batch",zen::diag::Phase::compute);
         if(q35){if(!forward_q35_batch(prompt,kc,vc,convc,ssmc,logits))return fail("Batched prefill forward pass failed");}
         else if(!forward_batch(prompt,kc,vc,logits))return fail("Batched prefill forward pass failed");}
        if(kv_bytes<=policy.cache_budget_bytes){auto saved=std::make_shared<PrefixCache>();saved->tokens=prompt;saved->key=kc;saved->value=vc;saved->logits=logits;if(q35){saved->conv=convc;saved->ssm=ssmc;}std::lock_guard<std::mutex> lock(prefix_cache_mutex_);prefix_cache_=std::move(saved);}
    }else zen::diag::breadcrumb(403,static_cast<std::int64_t>(prompt.size()),1);
    auto t1=clock::now();result.cache_hit=cache_hit;
    result.prefill_ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    result.decode_ms=0.0;result.stopped_on_eos=false;result.stopped_on_stop=false;result.speculative_proposed=0;result.speculative_accepted=0;result.speculative_rejected=0;
    result.token_ids.clear();result.last_logits=logits;
    std::mt19937_64 rng(options.seed?options.seed:std::random_device{}());
    for(std::uint32_t n=0;n<max_new;++n){
        if(cancelled&&cancelled->load(std::memory_order_relaxed))return fail("Generation cancelled");
        const auto next=sample_logits(logits,options,rng);result.token_ids.push_back(next);
        zen::diag::breadcrumb(402,static_cast<std::int64_t>(prompt.size()+n),next);
        if(on_tokens)on_tokens(result.token_ids);
        if(!options.stop.empty()&&tokenizer_loaded_){std::string partial;if(tokenizer_.decode(result.token_ids,partial,true)){bool stopped=false;for(const auto& stop:options.stop)if(!stop.empty()&&partial.find(stop)!=std::string::npos){stopped=true;break;}if(stopped){result.stopped_on_stop=true;break;}}}
        if(next==config_.eos_token){result.stopped_on_eos=true;break;}
        if(n+1==max_new)break;
        const auto pos=static_cast<std::uint32_t>(prompt.size()+n);
        if(pos>=config_.context)break;
        const auto d0=clock::now();
        {zen::diag::Scope scope("transformer.forward",zen::diag::Phase::compute);
         if(q35){if(!forward_q35_token(next,pos,kc,vc,convc,ssmc,logits))return fail("Decode forward pass failed");}
         else if(!forward(next,pos,kc,vc,logits))return fail("Decode forward pass failed");}
        result.decode_ms+=std::chrono::duration<double,std::milli>(clock::now()-d0).count();
        result.last_logits=logits;
    }
    if(error)error->clear();
    return true;
}
bool DenseTransformerRuntime::tokenize(const std::string& text,std::vector<std::uint32_t>& ids)const{
    return tokenizer_loaded_&&tokenizer_.encode(text,ids,config_.add_bos_token,config_.add_eos_token);
}
bool DenseTransformerRuntime::detokenize(const std::vector<std::uint32_t>& ids,std::string& text)const{
    return tokenizer_loaded_&&tokenizer_.decode(ids,text,true);
}
bool DenseTransformerRuntime::generate_speculative(const std::vector<std::uint32_t>& prompt,std::uint32_t max_new,
        const DenseTransformerRuntime& draft,std::uint32_t proposal_count,Generation& result,std::string* error,
        const std::function<void(const std::vector<std::uint32_t>&)>& on_tokens,
        const GenerationOptions& options,const std::atomic<bool>* cancelled)const{
    auto fail=[&](const std::string& e){if(error)*error=e;return false;};
    if(!loaded_||!draft.loaded_)return fail("Target and draft models must both be loaded");
    if(config_.qwen35_variant)return generate(prompt,max_new,result,error,on_tokens,options,cancelled); // hybrid speculative path not wired yet
    if(!tokenizer_.same_vocabulary(draft.tokenizer_)||config_.vocab!=draft.config_.vocab)return fail("Target and draft models must use the same token vocabulary");
    if(!proposal_count)return generate(prompt,max_new,result,error,on_tokens,options,cancelled);
    if(prompt.empty()||prompt.size()>config_.context||max_new>config_.context-prompt.size())return fail("Prompt plus generation exceeds target context length");
    auto cache_elements=[](const RuntimeConfig& c,std::uint64_t& out){out=c.context;for(std::uint64_t f:{std::uint64_t(c.kv_heads),std::uint64_t(c.embedding/c.heads),std::uint64_t(c.layers)}){if(f&&out>std::numeric_limits<std::uint64_t>::max()/f)return false;out*=f;}return out<=std::numeric_limits<std::uint64_t>::max()/(2*sizeof(float));};
    std::uint64_t target_elements{},draft_elements{};if(!cache_elements(config_,target_elements))return fail("Target KV cache size overflow");
    if(prompt.size()>draft.config_.context||!cache_elements(draft.config_,draft_elements))return generate(prompt,max_new,result,error,on_tokens,options,cancelled);
    zen::ResourceGovernor governor;const auto policy=governor.current_policy();
    if(target_elements>std::numeric_limits<std::uint64_t>::max()-draft_elements||target_elements+draft_elements>std::numeric_limits<std::uint64_t>::max()/(2*sizeof(float)))return generate(prompt,max_new,result,error,on_tokens,options,cancelled);
    const std::uint64_t combined_bytes=(target_elements+draft_elements)*2*sizeof(float);
    if(combined_bytes>policy.transient_budget_bytes)return generate(prompt,max_new,result,error,on_tokens,options,cancelled);
    const std::size_t target_kvsize=std::size_t(config_.context)*config_.kv_heads*config_.head_dim;
    const std::size_t draft_kvsize=std::size_t(draft.config_.context)*draft.config_.kv_heads*draft.config_.head_dim;
    std::vector<std::vector<float>> target_kc(config_.layers,std::vector<float>(target_kvsize)),target_vc(config_.layers,std::vector<float>(target_kvsize));
    std::vector<std::vector<float>> draft_kc(draft.config_.layers,std::vector<float>(draft_kvsize)),draft_vc(draft.config_.layers,std::vector<float>(draft_kvsize));
    std::vector<float> target_logits,draft_logits;
    using clock=std::chrono::steady_clock;const auto prefill_start=clock::now();
    if(!forward_batch(prompt,target_kc,target_vc,target_logits)||!draft.forward_batch(prompt,draft_kc,draft_vc,draft_logits))return fail("Target or draft prompt prefill failed");
    result={};result.last_logits=target_logits;result.prefill_ms=std::chrono::duration<double,std::milli>(clock::now()-prefill_start).count();
    std::vector<std::uint32_t> context=prompt;const auto decode_start=clock::now();std::mt19937_64 rng(options.seed?options.seed:std::random_device{}());
    auto stop_hit=[&](){if(options.stop.empty())return false;std::string text;if(!tokenizer_.decode(result.token_ids,text,true))return false;for(const auto& stop:options.stop)if(!stop.empty()&&text.find(stop)!=std::string::npos){result.stopped_on_stop=true;return true;}return false;};
    auto append=[&](std::uint32_t id){result.token_ids.push_back(id);context.push_back(id);if(on_tokens)on_tokens(result.token_ids);if(id==config_.eos_token){result.stopped_on_eos=true;return true;}return stop_hit();};
    while(result.token_ids.size()<max_new){
        if(cancelled&&cancelled->load(std::memory_order_relaxed))return fail("Generation cancelled");
        const auto remaining=max_new-result.token_ids.size();const auto draft_room=draft.config_.context>context.size()?draft.config_.context-context.size():0;
        const auto proposal_n=static_cast<std::uint32_t>(std::min<std::size_t>({proposal_count,remaining,draft_room}));
        if(!proposal_n){const auto next=sample_logits(target_logits,options,rng);if(append(next))break;if(result.token_ids.size()<max_new){if(!forward(next,static_cast<std::uint32_t>(context.size()-1),target_kc,target_vc,target_logits))return fail("Target-only fallback decode failed");}continue;}
        std::vector<std::uint32_t> proposals;proposals.reserve(proposal_n);
        std::vector<std::vector<double>> draft_distributions;draft_distributions.reserve(proposal_n);
        for(std::uint32_t i=0;i<proposal_n;++i){
            if(cancelled&&cancelled->load(std::memory_order_relaxed))return fail("Generation cancelled");
            auto q=sampling_distribution(draft_logits,options);
            if(q.empty()||std::none_of(q.begin(),q.end(),[](double p){return p>0.0;}))return fail("Draft sampling distribution has no valid token");
            const auto token=sample_distribution(q,rng);draft_distributions.push_back(std::move(q));proposals.push_back(token);
            if(token==draft.config_.eos_token)break;
            if(!draft.forward(token,static_cast<std::uint32_t>(context.size()+i),draft_kc,draft_vc,draft_logits))return fail("Draft proposal forward pass failed");
        }
        if(proposals.empty())return fail("Draft model produced no proposal tokens");
        result.speculative_proposed+=static_cast<std::uint32_t>(proposals.size());
        const auto start_pos=static_cast<std::uint32_t>(context.size());const auto target_before=target_logits;std::vector<float> verify_logits;
        if(!forward_batch(proposals,target_kc,target_vc,target_logits,start_pos,proposals.size(),&verify_logits))return fail("Target batched draft verification failed");
        std::vector<float> expected=target_before;bool rejected=false;
        for(std::size_t i=0;i<proposals.size()&&result.token_ids.size()<max_new;++i){
            if(cancelled&&cancelled->load(std::memory_order_relaxed))return fail("Generation cancelled");
            const auto p=sampling_distribution(expected,options);
            const auto& q=draft_distributions[i];
            const auto proposed=proposals[i];
            const double qx=proposed<q.size()?q[proposed]:0.0;
            const double px=proposed<p.size()?p[proposed]:0.0;
            const double accept_probability=qx>0.0?std::min(1.0,px/qx):0.0;
            std::uniform_real_distribution<double> uniform(0.0,1.0);
            if(uniform(rng)<accept_probability){
                ++result.speculative_accepted;
                if(append(proposed)){rejected=true;break;}
                expected.assign(verify_logits.begin()+i*config_.vocab,verify_logits.begin()+(i+1)*config_.vocab);
            }else{
                result.speculative_rejected+=static_cast<std::uint32_t>(proposals.size()-i);
                // Exact rejection correction: sample from normalised max(0, p-q).
                std::vector<double> residual(p.size(),0.0);double residual_sum=0.0;
                for(std::size_t j=0;j<p.size();++j){residual[j]=std::max(0.0,p[j]-(j<q.size()?q[j]:0.0));residual_sum+=residual[j];}
                std::uint32_t correction{};
                if(residual_sum>1.0e-15){for(auto& value:residual)value/=residual_sum;correction=sample_distribution(residual,rng);}
                else correction=sample_distribution(p,rng);
                if(append(correction)){rejected=true;break;}
                if(result.token_ids.size()<max_new){const auto pos=static_cast<std::uint32_t>(context.size()-1);if(!forward(correction,pos,target_kc,target_vc,target_logits))return fail("Target correction forward pass failed");if(pos<draft.config_.context&&!draft.forward(correction,pos,draft_kc,draft_vc,draft_logits))return fail("Draft correction forward pass failed");}
                else target_logits=expected;
                rejected=true;break;
            }
        }
        if(result.stopped_on_eos||result.stopped_on_stop)break;
        if(!rejected&&result.token_ids.size()<max_new){
            const auto bonus=sample_logits(expected,options,rng);
            if(append(bonus))break;
            if(result.token_ids.size()<max_new){const auto pos=static_cast<std::uint32_t>(context.size()-1);if(!forward(bonus,pos,target_kc,target_vc,target_logits))return fail("Target bonus forward pass failed");if(pos<draft.config_.context&&!draft.forward(bonus,pos,draft_kc,draft_vc,draft_logits))return fail("Draft bonus forward pass failed");}
            else target_logits=expected;
        }else if(!rejected&&result.token_ids.size()>=max_new)target_logits=expected;
    }
    result.decode_ms=std::chrono::duration<double,std::milli>(clock::now()-decode_start).count();result.last_logits=std::move(target_logits);
    if(error)error->clear();
    return true;
}

bool DenseTransformerRuntime::generate_text(const std::string& prompt,std::uint32_t max_new,
        Generation& result,std::string& text,std::string* error,
        const std::function<void(const std::string&)>& on_text,const GenerationOptions& options,const std::atomic<bool>* cancelled)const{
    std::vector<std::uint32_t> ids;
    if(!tokenize(prompt,ids)){if(error)*error="Tokenizer is unavailable or prompt tokenisation failed";return false;}
    std::string streamed;
    auto token_callback=[&](const std::vector<std::uint32_t>& generated){
        if(!on_text)return;
        std::string current;if(!detokenize(generated,current))return;
        std::size_t safe_end=current.size();std::size_t stop_at=std::string::npos;std::size_t hold=0;
        for(const auto& stop:options.stop)if(!stop.empty()){hold=std::max(hold,stop.size());const auto at=current.find(stop);if(at!=std::string::npos)stop_at=std::min(stop_at,at);}
        if(stop_at!=std::string::npos)safe_end=stop_at;else if(hold>1)safe_end=current.size()>hold-1?current.size()-(hold-1):0;
        if(safe_end>streamed.size()){on_text(current.substr(streamed.size(),safe_end-streamed.size()));streamed=current.substr(0,safe_end);}
    };
    if(!generate(ids,max_new,result,error,on_text?token_callback:std::function<void(const std::vector<std::uint32_t>&)>{},options,cancelled))return false;
    if(!detokenize(result.token_ids,text)){if(error)*error="Generated token decoding failed";return false;}
    for(const auto& stop:options.stop)if(!stop.empty()){const auto at=text.find(stop);if(at!=std::string::npos)text.resize(at);}
    if(on_text&&text.size()>streamed.size())on_text(text.substr(streamed.size()));
    return true;
}

}
