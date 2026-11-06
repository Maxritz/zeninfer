#include "zen/dense_transformer_runtime.h"
#include "zen/diagnostics.h"
#include "zen/resource_governor.h"
#include "zen/kernels.h"
#include "zen/quant.h"
#include "zen/execution_planner.h"
#include "zen/attention.h"
#include "zen/compute_pool.h"
#if defined(ZEN_HAS_D3D12)
#include "zen/d3d12_backend.h"
#endif
#include <thread>
#include <atomic>
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
#include <cstdio>
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
    try{return std::stof(i->second);}catch(...){std::fprintf(stderr,"[meta] unparseable numeric metadata '%s'='%s', using fallback %g\n",key,i->second.c_str(),static_cast<double>(fallback));return fallback;}
}
float meta_float_arch(const zen::model::ModelConfig& c,const std::string& arch,const char* suffix,float fallback){auto i=c.metadata.find(arch+"."+suffix);if(i==c.metadata.end())return meta_float(c,(std::string("llama.")+suffix).c_str(),fallback);try{return std::stof(i->second);}catch(...){std::fprintf(stderr,"[meta] unparseable numeric metadata '%s.%s'='%s', using fallback %g\n",arch.c_str(),suffix,i->second.c_str(),static_cast<double>(fallback));return fallback;}}
std::uint32_t meta_uint(const zen::model::ModelConfig& c,const char* key,std::uint32_t fallback){
    auto i=c.metadata.find(key);if(i==c.metadata.end())return fallback;
    try{return static_cast<std::uint32_t>(std::stoul(i->second));}catch(...){std::fprintf(stderr,"[meta] unparseable numeric metadata '%s'='%s', using fallback %u\n",key,i->second.c_str(),fallback);return fallback;}
}
std::string meta_string_arch(const zen::model::ModelConfig& c,const std::string& arch,const char* suffix){auto i=c.metadata.find(arch+"."+suffix);if(i==c.metadata.end())i=c.metadata.find(std::string("llama.")+suffix);return i==c.metadata.end()?std::string{}:i->second;}
std::uint32_t meta_uint_arch(const zen::model::ModelConfig& c,const std::string& arch,const char* suffix,std::uint32_t fallback){auto i=c.metadata.find(arch+"."+suffix);if(i==c.metadata.end())return meta_uint(c,(std::string("llama.")+suffix).c_str(),fallback);try{return static_cast<std::uint32_t>(std::stoul(i->second));}catch(...){std::fprintf(stderr,"[meta] unparseable numeric metadata '%s.%s'='%s', using fallback %u\n",arch.c_str(),suffix,i->second.c_str(),fallback);return fallback;}}
bool meta_bool_arch(const zen::model::ModelConfig& c,const std::string& arch,const char* suffix,bool fallback=false){
    auto i=c.metadata.find(arch+"."+suffix);if(i==c.metadata.end())i=c.metadata.find(std::string("llama.")+suffix);
    if(i==c.metadata.end())return fallback;
    return i->second=="1"||i->second=="true"||i->second=="TRUE";
}
// RMSNorm AVX2 gate: ZEN_CPU_RMS_AVX2=0 forces the scalar loop below
// (default on when AVX2/FMA is present). The AVX2 kernel is in
// kernels/avx2.cpp; its double-precision ss reduction reorders the sum
// vs this sequential scalar loop, so results can differ by ~1 ulp of
// float -- see the note on rms_norm_avx2.
inline bool rms_avx2_enabled() noexcept {
    static const bool enabled=[](){const char* e=std::getenv("ZEN_CPU_RMS_AVX2");return !(e&&e[0]=='0');}();
    return enabled;
}
// GDN AVX2 gate: ZEN_CPU_GDN_AVX2=0 forces scalar (default on when AVX2/FMA).
inline bool gdn_avx2_enabled() noexcept {
    static const bool enabled=[](){const char* e=std::getenv("ZEN_CPU_GDN_AVX2");return !(e&&e[0]=='0');}();
    return enabled;
}
void rms_norm(const float* x,const float* weight,float* y,std::size_t n,float eps){
#ifdef ZEN_HAS_AVX2_KERNELS
    static const bool avx2=zen::cpu_has_avx2_fma();
    if(avx2&&rms_avx2_enabled()){zen::rms_norm_avx2(x,weight,y,n,eps);return;}
#endif
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
// ZI-P4: elementwise helpers. The scalar path stays the numerical reference for
// non-AVX2 builds; the AVX2 implementations live in kernels/avx2.cpp.
static inline float dot_f32(const float* a,const float* b,std::size_t n){
#ifdef ZEN_HAS_AVX2_KERNELS
    return zen::dot_f32_avx2(a,b,n);
#else
    double s=0.0;for(std::size_t i=0;i<n;++i)s+=double(a[i])*b[i];return float(s);
#endif
}
static inline void axpy_f32(float a,const float* x,float* y,std::size_t n){
#ifdef ZEN_HAS_AVX2_KERNELS
    zen::axpy_f32_avx2(a,x,y,n);
#else
    for(std::size_t i=0;i<n;++i)y[i]+=a*x[i];
#endif
}
// SwiGLU hidden activation: out = silu(gate) * up.
inline bool silu_avx2_enabled() noexcept {
    static const bool enabled=[](){const char* e=std::getenv("ZEN_CPU_SILU_AVX2");return !(e&&e[0]=='0');}();
    return enabled;
}
void silu_mul(const float* gate,const float* up,float* out,std::size_t n){
#ifdef ZEN_HAS_AVX2_KERNELS
    static const bool avx2=zen::cpu_has_avx2_fma();
    if(avx2&&silu_avx2_enabled()){zen::silu_mul_avx2(gate,up,out,n);return;}
#endif
    for(std::size_t i=0;i<n;++i){const float z=gate[i];out[i]=(z/(1.0f+std::exp(-z)))*up[i];}
}
// In-place SiLU: x = silu(x) = x / (1 + exp(-x)).
// Routes through silu_mul with a ones buffer so it shares the AVX2 Cephes path
// and the ZEN_CPU_SILU_AVX2 kill switch. The ones buffer is thread-local so
// repeated calls do not pay a fresh allocation; scalar fallback is the same
// exp loop as silu_mul's scalar reference. Aliasing gate==out==x is safe:
// silu_mul reads gate[i] and up[i] before writing out[i].
void silu_inplace(float* x,std::size_t n){
#ifdef ZEN_HAS_AVX2_KERNELS
    static const bool avx2=zen::cpu_has_avx2_fma();
    if(avx2&&silu_avx2_enabled()){
        thread_local std::vector<float> ones;
        ones.assign(n,1.0f);
        zen::silu_mul_avx2(x,ones.data(),x,n);
        return;
    }
#endif
    for(std::size_t i=0;i<n;++i){const float z=x[i];x[i]=z/(1.0f+std::exp(-z));}
}
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
//
// NATURAL order: the output index IS the frequency index, out[j] pairs with
// in[i] through (-1)^popcount(i&j). There is deliberately no bit-reversal
// pre-pass. A sequency-ordered variant (bit-reversing the input first, or the
// standard "Walsh" ordering) is a *different matrix*: it agrees with Sylvester
// on rows 0 and 1 and diverges from row 2 on. Because every folded weight was
// folded against natural-order H, running a bit-reversed transform here leaves
// each folded matmul reading a basis it was never folded for -- the output is
// finite, plausible-looking and wrong, which is the worst failure mode.
//
// Cross-checked against PrismML-Eng/llama.cpp ggml/src/ggml-cuda/fwht.cu: the
// kernel scales by 1/sqrt(N), applies the signs, then runs plain
// (x, y) -> (x+y, x-y) butterflies in natural index order with no permutation,
// and the shfl_xor path (val + val2 / val2 - val) is the same transform.
void swht_block(const float* in,float* out,std::uint32_t B){
    std::copy_n(in,B,out);
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
// Builds the per-dimension cos/sin table that rope() uses, so the device RoPE kernel
// can be handed exactly the same numbers. The body below is the table half of rope(),
// copied verbatim rather than re-derived: the device kernel must not compute angles
// itself (powf/cosf would differ from these double-promoted std::pow/cos/sin in the last
// bits and could flip a sampled id), so the only safe contract is "same table".
void rope_table(std::uint32_t rotary_dim,std::uint32_t pos,float base,const std::string& scaling_type,float scaling_factor,std::uint32_t original_context,float low_freq_factor,float high_freq_factor,float yarn_beta_fast,float yarn_beta_slow,std::vector<float>& cos_out,std::vector<float>& sin_out){
    rotary_dim-=rotary_dim%2;const std::uint32_t half=rotary_dim/2;
    cos_out.assign(half,1.0f);sin_out.assign(half,0.0f);
    float effective_base=base;
    if(scaling_type=="dynamic"||scaling_type=="dynamic_ntk"||scaling_type=="ntk")if(original_context>0&&pos+1>original_context&&rotary_dim>2){const float stretch=std::max(1.0f,scaling_factor*float(pos+1)/float(original_context)-(scaling_factor-1.0f));effective_base*=std::pow(stretch,float(rotary_dim)/float(rotary_dim-2));}
    float yarn_low=0.0f,yarn_high=float(half-1);if(scaling_type=="yarn"&&original_context>0){yarn_low=std::max(0.0f,float(rotary_dim)*std::log(float(original_context)/(yarn_beta_fast*2.0f*3.14159265f))/(2.0f*std::log(base)));yarn_high=std::min(float(half-1),float(rotary_dim)*std::log(float(original_context)/(yarn_beta_slow*2.0f*3.14159265f))/(2.0f*std::log(base)));if(yarn_high<=yarn_low)yarn_high=yarn_low+1.0f;}
    for(std::uint32_t i=0;i<half;++i){float freq=std::pow(effective_base,-2.0f*float(i)/float(rotary_dim));float position=float(pos);
            if(scaling_type=="linear")position/=std::max(1.0f,scaling_factor);
            else if(scaling_type=="yarn"){const float ramp=std::clamp((float(i)-yarn_low)/(yarn_high-yarn_low),0.0f,1.0f);const float interpolated=freq/std::max(1.0f,scaling_factor);freq=interpolated*(1.0f-ramp)+freq*ramp;}
            else if(scaling_type=="llama3"&&original_context>0){const float wavelen=2.0f*3.14159265f/freq;const float low= float(original_context)/std::max(0.01f,low_freq_factor);const float high=float(original_context)/std::max(0.01f,high_freq_factor);if(wavelen>low)freq/=std::max(1.0f,scaling_factor);else if(wavelen>high){const float smooth=std::clamp((float(original_context)/wavelen-low_freq_factor)/(high_freq_factor-low_freq_factor),0.0f,1.0f);freq=(1.0f-smooth)*(freq/std::max(1.0f,scaling_factor))+smooth*freq;}}
            const float angle=position*freq;cos_out[i]=std::cos(angle);sin_out[i]=std::sin(angle);
    }
}
// Device KV cache capacity, in tokens. Allocated once, up front.
//
// This must be a constant. The previous code passed the *current* sequence
// length, so kv_cache_configure saw a new shape on almost every decode step,
// reallocated the cache, and discarded every key/value already appended. Decode
// then attended over a cache that had been silently emptied, which is why two
// identically seeded generations diverged. A request larger than this falls back
// to the CPU path instead of resizing.
constexpr std::size_t kDeviceKvTokens = 8192;

// True when decode can run its attention block on the device for this model. Set once
// by the runtime after the KV cache is configured; requires a D3D12 device, no ALiBi
// (the kernel has no bias term) and a cache that matches the model's shape.
bool device_attention_available(std::uint32_t layers,std::size_t needed,std::uint32_t kv_heads,std::uint32_t head_dim,float epsilon){
#if defined(ZEN_HAS_D3D12)
    (void)epsilon;
    if(!needed)return false;
    if(needed>kDeviceKvTokens)return false;
    if(!zen::d3d12::enabled())return false;
    return zen::d3d12::kv_cache_configure(layers,kDeviceKvTokens,kv_heads,head_dim);
#else
    (void)layers;(void)needed;(void)kv_heads;(void)head_dim;(void)epsilon;return false;
#endif
}

void rope(float* x,std::uint32_t heads,std::uint32_t head_dim,std::uint32_t rotary_dim,
          std::uint32_t pos,float base,const std::string& scaling_type,float scaling_factor,std::uint32_t original_context,float low_freq_factor,float high_freq_factor,float yarn_beta_fast,float yarn_beta_slow,
          bool adjacent_pairs){
    rotary_dim=std::min(rotary_dim,head_dim);rotary_dim-=rotary_dim%2;const std::uint32_t half=rotary_dim/2;
    float effective_base=base;
    if(scaling_type=="dynamic"||scaling_type=="dynamic_ntk"||scaling_type=="ntk")if(original_context>0&&pos+1>original_context&&rotary_dim>2){const float stretch=std::max(1.0f,scaling_factor*float(pos+1)/float(original_context)-(scaling_factor-1.0f));effective_base*=std::pow(stretch,float(rotary_dim)/float(rotary_dim-2));}
    float yarn_low=0.0f,yarn_high=float(half-1);if(scaling_type=="yarn"&&original_context>0){yarn_low=std::max(0.0f,float(rotary_dim)*std::log(float(original_context)/(yarn_beta_fast*2.0f*3.14159265f))/(2.0f*std::log(base)));yarn_high=std::min(float(half-1),float(rotary_dim)*std::log(float(original_context)/(yarn_beta_slow*2.0f*3.14159265f))/(2.0f*std::log(base)));if(yarn_high<=yarn_low)yarn_high=yarn_low+1.0f;}
    // ZI-P3: the rotation angle for dimension i depends only on (i, pos), never
    // on the head. Computing pow/cos/sin inside the head loop repeated the same
    // 32-40 trig evaluations once per head (heads=32 for Q, 8 for K) at ~247ns a
    // pair. Build the per-dimension table once, then apply it to every head.
    std::vector<float> rope_cos(half),rope_sin(half);
    for(std::uint32_t i=0;i<half;++i){float freq=std::pow(effective_base,-2.0f*float(i)/float(rotary_dim));float position=float(pos);
            if(scaling_type=="linear")position/=std::max(1.0f,scaling_factor);
            else if(scaling_type=="yarn"){const float ramp=std::clamp((float(i)-yarn_low)/(yarn_high-yarn_low),0.0f,1.0f);const float interpolated=freq/std::max(1.0f,scaling_factor);freq=interpolated*(1.0f-ramp)+freq*ramp;}
            else if(scaling_type=="llama3"&&original_context>0){const float wavelen=2.0f*3.14159265f/freq;const float low= float(original_context)/std::max(0.01f,low_freq_factor);const float high=float(original_context)/std::max(0.01f,high_freq_factor);if(wavelen>low)freq/=std::max(1.0f,scaling_factor);else if(wavelen>high){const float smooth=std::clamp((float(original_context)/wavelen-low_freq_factor)/(high_freq_factor-low_freq_factor),0.0f,1.0f);freq=(1.0f-smooth)*(freq/std::max(1.0f,scaling_factor))+smooth*freq;}}
            const float angle=position*freq;rope_cos[i]=std::cos(angle);rope_sin[i]=std::sin(angle);
    }
    // Pairing layout: llama.cpp rotates llama-family weights over CONSECUTIVE
    // head elements (x[2i], x[2i+1], LLAMA_ROPE_TYPE_NORM); qwen-family over
    // half-split (x[i], x[i+half], NEOX/IMROPE). layouts/route by caller.
    for(std::uint32_t h=0;h<heads;++h){float* v=x+std::size_t(h)*head_dim;
        for(std::uint32_t i=0;i<half;++i){const float c=rope_cos[i],s=rope_sin[i];
            const float a=adjacent_pairs?v[2*i]:v[i];
            const float b=adjacent_pairs?v[2*i+1]:v[i+half];
            if(adjacent_pairs){v[2*i]=a*c-b*s;v[2*i+1]=a*s+b*c;}
            else{v[i]=a*c-b*s;v[i+half]=a*s+b*c;}
        }
    }
}
// Prefill rope rotation with a PREBUILT table slice (one token's cos/sin).
// Bit-identical to rope() above: rope_table is the verbatim table half of rope(),
// and this loop is rope()'s rotation half verbatim (same a*c-b*s / a*s+b*c order,
// dims past 2*pairs untouched). Prefill builds [tokens * pairs] tables once per
// forward_batch (rope() rebuilt them per (layer, token, Q/K) call: 2*16*128 = 4096
// pow/cos/sin table builds at T=128) and this is the CPU fallback that shares them
// with the GPU prefill rope path.
void rope_apply_table(float* x,std::uint32_t heads,std::uint32_t head_dim,std::uint32_t pairs,
                      const float* cos_in,const float* sin_in,bool adjacent_pairs){
    for(std::uint32_t h=0;h<heads;++h){float* v=x+std::size_t(h)*head_dim;
        for(std::uint32_t i=0;i<pairs;++i){const float c=cos_in[i],s=sin_in[i];
            const float a=adjacent_pairs?v[2*i]:v[i];
            const float b=adjacent_pairs?v[2*i+1]:v[i+pairs];
            if(adjacent_pairs){v[2*i]=a*c-b*s;v[2*i+1]=a*s+b*c;}
            else{v[i]=a*c-b*s;v[i+pairs]=a*s+b*c;}
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
    // ZI-P8: greedy fast path. sampling_distribution materialises ~2.5MB per token
    // (probability vector, adjusted copy, finite-order list, top-p mask) and
    // isfinite-scans every logit only to select one argmax at temperature<=0,
    // where the distribution is one-hot. A single pass returns the identical
    // token. Tie-breaking replicates the reference greedy branch exactly: its
    // `order` holds the FINITE logits in ascending id order, `best` starts at the
    // first of them and is replaced only on a strictly greater value, so the
    // first (lowest-index) maximum wins — the loop below does the same. The
    // degenerate no-finite-candidate case falls through to the reference path so
    // even its all-zero-distribution behaviour is preserved verbatim. The
    // reference consumed RNG draws inside sample_distribution although a one-hot
    // outcome ignores them; every other draw in a temperature==0 generation is
    // likewise RNG-independent, so token outputs are unchanged. Greedy with
    // logit_bias or allowed_token_ids keeps the reference path below (identical
    // results, unchanged cost).
    if((o.temperature<=0.0f||!std::isfinite(o.temperature))&&o.logit_bias.empty()&&o.allowed_token_ids.empty()){
        std::size_t best=std::numeric_limits<std::size_t>::max();float best_v=0.0f;
        for(std::size_t i=0;i<logits.size();++i){
            const float v=logits[i];
            if(!std::isfinite(v))continue; // reference: only isfinite ids enter `order`
            if(best==std::numeric_limits<std::size_t>::max()||v>best_v){best=i;best_v=v;}
        }
        if(best!=std::numeric_limits<std::size_t>::max())return static_cast<std::uint32_t>(best);
    }
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
// Fused D3D12 layer/FFN kernels interpret every matrix as raw Q8_0.
// A mapped tensor alone does not establish its quantization format.
#if defined(ZEN_HAS_D3D12)
bool tensor_is_q8(const DenseTransformerRuntime::Tensor* t) {
    return t && t->raw && t->format && t->format->encoding==quant::Encoding::q8_0;
}
#endif
// `validate_output=false` skips the output finite scan. Only the per-token decode
// LM head passes false (ZEN_VALIDATE_LOGITS gate in forward()): it is the one call
// whose rows == vocab (128256), so the scan is a second full pass over the logit
// row every token. Every other caller keeps the default true and is unchanged.
bool tensor_matvec(const DenseTransformerRuntime::Tensor& t,const float* x,float* y,std::size_t rows,std::size_t cols,bool validate_output=true){
    const planner::ExecutionPlanner planner;const auto plan=planner.select(planner::Phase::decode,t.raw&&t.format,1,rows,cols);
    if(t.raw&&t.format){if(plan.kernel!=planner::Kernel::quantized_gemv||!quant::gemv(*t.format,t.raw,x,y,rows,cols,1.0f,plan.workers))return false;return !validate_output||finite_values(y,rows);}
    if(t.values.size()!=rows*cols)return false;
    matvec(t.values.data(),x,y,rows,cols);return !validate_output||finite_values(y,rows);
}
bool tensor_matvec_batch(const DenseTransformerRuntime::Tensor* const* ts,const float* x,
                         float* const* ys,const std::size_t* rows,std::size_t count,std::size_t cols){
    // Only worth grouping when every task is a quantised tensor the GPU path can take.
    bool all_quant=true;
    for(std::size_t i=0;i<count;++i)if(!ts[i]||!ts[i]->raw||!ts[i]->format){all_quant=false;break;}
    if(all_quant&&count<=8){
        const planner::ExecutionPlanner planner;
        const auto plan=planner.select(planner::Phase::decode,true,1,rows[0],cols);
        if(plan.kernel==planner::Kernel::quantized_gemv){
            bool same_format=true;
            const std::uint8_t* w[8];float* o[8];std::size_t r[8];
            for(std::size_t i=0;i<count;++i){
                w[i]=ts[i]->raw;o[i]=ys[i];r[i]=rows[i];
                if(ts[i]->format->encoding!=ts[0]->format->encoding)same_format=false;
            }
            if(same_format&&quant::gemv_batch(*ts[0]->format,w,x,o,r,cols,count,1.0f,plan.workers)){
                for(std::size_t i=0;i<count;++i)if(!finite_values(ys[i],rows[i]))return false;
                return true;
            }
        }
    }
    for(std::size_t i=0;i<count;++i)if(!tensor_matvec(*ts[i],x,ys[i],rows[i],cols))return false;
    return true;
}
bool tensor_ffn(const DenseTransformerRuntime::Tensor& gate,const DenseTransformerRuntime::Tensor& up,
                const DenseTransformerRuntime::Tensor& down,const float* x,float* out,
                std::size_t ffn_rows,std::size_t cols,std::size_t rows){
    if(!gate.raw||!up.raw||!down.raw||!gate.format||!up.format||!down.format)return false;
    if(!quant::ffn_gemv(*gate.format,gate.raw,*up.format,up.raw,*down.format,down.raw,x,out,
                        ffn_rows,cols,rows))return false;
    return finite_values(out,rows);
}
// The FFN half of a layer in one GPU submission: attn_output matvec, residual add,
// FFN norm, gate|up, SiLU, down, residual add. `x` is both the residual base and the
// destination; the backend stages the base into VRAM before it writes anything back,
// so the in-place update is safe.
bool tensor_ffn_residual_fused(const DenseTransformerRuntime::Tensor& attn_output,
                               const DenseTransformerRuntime::Tensor& gate,
                               const DenseTransformerRuntime::Tensor& up,
                               const DenseTransformerRuntime::Tensor& down,
                               const float* norm_weight,const float* x,const float* attn_out,
                               float* out,std::size_t hidden,std::size_t attn_width,
                               std::size_t ffn_rows,float epsilon){
    if(!attn_output.raw||!gate.raw||!up.raw||!down.raw||
       !attn_output.format||!gate.format||!up.format||!down.format)return false;
    if(!quant::ffn_residual_fused(*attn_output.format,attn_output.raw,
                                  *gate.format,gate.raw,*up.format,up.raw,*down.format,down.raw,
                                  norm_weight,x,attn_out,out,hidden,attn_width,ffn_rows,
                                  epsilon))return false;
    return finite_values(out,hidden);
}

bool tensor_gemm(const DenseTransformerRuntime::Tensor& t,const float* input,float* output,std::size_t tokens,std::size_t rows,std::size_t cols){
    if(tokens==1)return tensor_matvec(t,input,output,rows,cols);
    const planner::ExecutionPlanner planner;const auto plan=planner.select(planner::Phase::prefill,t.raw&&t.format,tokens,rows,cols);
    if(t.raw&&t.format)return plan.kernel==planner::Kernel::quantized_gemm&&quant::gemm(*t.format,t.raw,input,output,tokens,rows,cols,1.0f,plan.workers);
    if(t.values.size()!=rows*cols)return false;
    // Priority 2: FP32 prefill GEMM honours plan.workers like the Q8 path.
    // Row-parallel: each row r computes every token with the same gemv_avx2
    // kernel the serial path used, so the K-order per output element is
    // unchanged and the result is bit-identical (only the r partition moved).
    // gemm_avx2 (4x8) was not used: it expects a transposed/column-major B
    // (contiguous n) while input is [tokens x cols] row-major, which would
    // require a transpose copy, and its single-accumulator FMA order differs
    // from gemv_avx2's 4-way split, so it would not be bit-identical.
    // ZEN_CPU_GEMM_PAR=0 restores the serial path below (default enabled).
    static const bool gemm_par_enabled=[](){const char* e=std::getenv("ZEN_CPU_GEMM_PAR");return !(e&&e[0]=='0');}();
    const unsigned workers=gemm_par_enabled?plan.workers:1u;
#ifdef ZEN_HAS_AVX2_KERNELS
    const bool has_avx2=zen::cpu_has_avx2_fma();
#else
    const bool has_avx2=false;
#endif
    if(workers>1&&rows>=32&&cols>=128){
        const bool use_avx2=has_avx2;
        const float* wbase=t.values.data();
        zen::ComputePool::instance().parallel_for(rows,workers,[&,use_avx2,wbase](std::size_t first,std::size_t last){
            for(std::size_t r=first;r<last;++r){
                const float* w=wbase+r*cols;
                for(std::size_t token=0;token<tokens;++token){
                    float* y=output+token*rows+r;
                    const float* x=input+token*cols;
#ifdef ZEN_HAS_AVX2_KERNELS
                    if(use_avx2){zen::gemv_avx2(w,x,y,1,cols);continue;}
#endif
                    double sum=0;for(std::size_t c=0;c<cols;++c)sum+=double(w[c])*x[c];*y=static_cast<float>(sum);
                }
            }
        });
        return true;
    }
    for(std::size_t r=0;r<rows;++r){const float*w=t.values.data()+r*cols;for(std::size_t token=0;token<tokens;++token){
#ifdef ZEN_HAS_AVX2_KERNELS
        if(has_avx2){zen::gemv_avx2(w,input+token*cols,output+token*rows+r,1,cols);continue;}
#endif
        double sum=0;for(std::size_t c=0;c<cols;++c)sum+=double(w[c])*input[token*cols+c];output[token*rows+r]=static_cast<float>(sum);
    }}return true;
}
// Batched prefill GEMM wrapper for the projections that share one input:
// the Q|K|V projections and the gate|up projections of a transformer
// prefill. Each group has its own tensor count and row widths but the same
// input matrix and column count, so they can share one activation upload and
// one fence wait. Single-submit path is gated by ZEN_D3D12_BATCH=1 inside
// quant::gemm_batch (default off); with the gate unset this wrapper loops
// over tensor_gemm(), which is exactly the old per-projection path, so
// default behaviour is unchanged.
bool tensor_gemm_batch(const DenseTransformerRuntime::Tensor* const* ts,const float* input,
                       float* const* outputs,const std::size_t* rows,std::size_t count,
                       std::size_t tokens,std::size_t cols){
    if(!ts||!input||!outputs||!rows||!count||!tokens||!cols)return false;
    if(tokens==1)return tensor_matvec_batch(ts,input,outputs,rows,count,cols);
    bool all_quant=true;
    for(std::size_t i=0;i<count;++i)if(!ts[i]||!ts[i]->raw||!ts[i]->format){all_quant=false;break;}
    if(all_quant&&count<=8){
        const planner::ExecutionPlanner planner;
        const auto plan=planner.select(planner::Phase::prefill,true,tokens,rows[0],cols);
        if(plan.kernel==planner::Kernel::quantized_gemm){
            bool same_format=true;
            const std::uint8_t* w[8];float* o[8];std::size_t r[8];
            for(std::size_t i=0;i<count;++i){
                w[i]=ts[i]->raw;o[i]=outputs[i];r[i]=rows[i];
                if(ts[i]->format->encoding!=ts[0]->format->encoding)same_format=false;
            }
            if(same_format&&quant::gemm_batch(*ts[0]->format,w,input,o,r,tokens,cols,count,1.0f,plan.workers))return true;
        }
    }
    for(std::size_t i=0;i<count;++i)if(!tensor_gemm(*ts[i],input,outputs[i],tokens,rows[i],cols))return false;
    return true;
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
    // Forward input transform for a folded weight. Reference order:
    //   permute (handled by the caller) -> signs -> rotate
    // llm_graph_context::build_hadamard_input is
    //   if (t.perm_rep > 1) { ...permute... }
    //   if (t.signs)        x = ggml_mul(ctx0, x, t.signs);
    //   x = llama_mul_mat_hadamard(ctx0, x, t.rot);
    // i.e. x' = H * S * P * x, matching the documented y = W'*(H*(s*P*x)).
    // Signs and the rotation do NOT commute, so a folded weight only reads the
    // basis it was folded against when this order is exact: W_f = W*S*H must
    // multiply H*S*x. Applying the rotation first silently folds the sign
    // vector into the wrong basis and every Hadamard-folded weight is wrong.
    const auto& H=hadamard_;if(!H.block_size)return;
    if(width%H.block_size)return;           // never rotate a partial trailing block
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
    // Inverse-table restore, and deliberately the OPPOSITE order to the forward
    // transform above: rotate first, then signs. The table stores z = S*H*e, so
    // the exact inverse is (S*H)^-1 = H*S -- rotate then signs. This mirrors
    // llm_graph_context::build_embd_rows:
    //   cur = llama_mul_mat_hadamard(ctx0, cur, it->second.rot);
    //   if (it->second.signs) cur = ggml_mul(ctx0, cur, it->second.signs);
    // The two paths are not symmetric; making them identical is a bug either way.
    const auto& H=hadamard_;if(!H.block_size)return;
    if(width%H.block_size)return;
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
DenseTransformerRuntime::~DenseTransformerRuntime() {
#if defined(ZEN_HAS_D3D12)
    for (const auto& entry : tensors_) {
        const auto& t=entry.second;
        if (t.raw && t.format && (t.format->encoding==quant::Encoding::q4_k || t.format->encoding==quant::Encoding::q5_k))
            zen::d3d12::release_kquant_weights(t.raw);
    }
#endif
}
bool DenseTransformerRuntime::load(const std::filesystem::path& path,std::string* error){
#if defined(ZEN_HAS_D3D12)
    // The old mapping may be unmapped on either a successful reload or an
    // early failure. Drop its cached K-quant entries before addresses can recur.
    for (const auto& entry : tensors_) {
        const auto& t=entry.second;
        if (t.raw && t.format && (t.format->encoding==quant::Encoding::q4_k || t.format->encoding==quant::Encoding::q5_k))
            zen::d3d12::release_kquant_weights(t.raw);
    }
#endif
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
    // Llama-family RoPE rotates consecutive pairs (llama.cpp LLAMA_ROPE_TYPE_NORM);
    // qwen-family (qwen2/3, qwen35) rotate half-split pairs (NEOX/IMROPE).
    config_.rope_pairs_adjacent=llama_arch;
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
        // attn_q is stored ne0 = input width, ne1 = output width. The per-head
        // width is output/heads; using ne0/heads instead only happens to be
        // right when head_dim == embedding/heads (it is not for Qwen3-family
        // models whose q width is heads*head_dim != embedding).
        for(const auto& d:report.tensors)if(d.name=="blk.0.attn_q.weight"&&d.dimensions.size()>=2&&config_.heads){
            const auto width=(d.dimensions[0]==config_.embedding)?d.dimensions[1]:d.dimensions[0];
            if(width)config_.head_dim=static_cast<std::uint32_t>(width/config_.heads);
            break;}
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
    config_.hadamard_gdn_v_grouped=report.config.hadamard_gdn_v_grouped;
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
    // F-5: a present-but-unparseable study-critical numeric override (rope.freq_base,
    // attention.layer_norm_rms_epsilon, rope.scaling.* factors/conditions) is a load
    // failure, not a silent fallback. Default-when-absent is unchanged: only keys
    // whose value failed to parse above (each already logged to stderr) trip this.
    {
        const auto* md=&report.config.metadata;
        auto bad=[&](const char* key)->bool{
            auto it=md->find(key);if(it==md->end())return false;
            auto is_num=[&](const std::string& s)->bool{
                try{size_t n=0;std::stof(s,&n);return n==s.size()&&!s.empty();}
                catch(...){return false;}
            };
            return !is_num(it->second);
        };
        auto bad_arch=[&](const char* suffix)->bool{
            const std::string a=architecture_name_+"."+suffix;
            const std::string b=std::string("llama.")+suffix;
            return bad(a.c_str())||bad(b.c_str());
        };
        if(bad_arch("rope.freq_base")||bad_arch("attention.layer_norm_rms_epsilon")||
           bad_arch("rope.scaling.factor")||bad_arch("rope.scale_linear")||
           bad_arch("rope.scaling.low_freq_factor")||bad_arch("rope.scaling.high_freq_factor")||
           bad_arch("rope.scaling.beta_fast")||bad_arch("rope.scaling.beta_slow"))
            return fail("Unparseable numeric GGUF metadata (see [meta] fallback log)");
    }
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
    // file-backed model pages are governed by the OS, not eagerly charged as resident FP32 weights.
    // ZI-P7: ZEN_PREFETCH=0 disables the per-layer prefetch thread. Each layer
    // spawns one std::async thread per decoded token, which costs more than the
    // page-touch when the mapping is already resident in the OS page cache.
    const char* prefetch_env=std::getenv("ZEN_PREFETCH");
    const bool prefetch_permitted=!(prefetch_env&&prefetch_env[0]=='0'&&prefetch_env[1]=='\0');
    prefetch_enabled_=prefetch_permitted&&policy.prefetch_workers>0;
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
        hadamard_.gdn_v_grouped=report.config.hadamard_gdn_v_grouped;
        hadamard_.tied_output=report.config.hadamard_tied_output;
        const auto foldable_w=[](const std::string& n){
            static const char* kinds[]={"attn_q","attn_k","attn_v","attn_qkv","attn_gate","attn_output",
                "ffn_gate","ffn_up","ffn_down","ssm_out","nextn.draft_head"};
            if(n=="output.weight")return true;
            if(n.compare(0,4,"blk.")!=0)return false;
            std::size_t pos=4;while(pos<n.size()&&std::isdigit(static_cast<unsigned char>(n[pos])))++pos;
            if(pos==4||pos>=n.size()||n[pos]!='.')return false;
            ++pos;
            for(const char* kind:kinds)if(n.compare(pos,std::string::npos,std::string(kind)+".weight")==0)return true;
            return false;
        };
        for(const auto& n:report.config.hadamard_weight_names){
            // The reference refuses a folded weight that is not routed through a
            // Hadamard-aware matmul; accepting it here would run that weight in the
            // wrong basis with no error. Every builder in this file is Hadamard-aware
            // for exactly these kinds, so the same list is the right gate.
            if(!foldable_w(n))return fail("prism.hadamard: weight '"+n+"' is not on a Hadamard-aware matmul path");
            if(!hadamard_.folded_names.insert(n).second)return fail("Duplicate prism.hadamard weight: "+n);
        }
        for(const auto& n:report.config.hadamard_inverse_names){
            if(n!="token_embd.weight")return fail("prism.hadamard: inverse weight '"+n+"' is not an inverse-after-lookup table");
            if(hadamard_.folded_names.count(n))return fail("prism.hadamard: weight is both folded and inverse: "+n);
            hadamard_.folded_names.insert("\x01"+n); // inverse marker
        }
        // version 2 ties the LM head to the latent embedding: the head then reads the
        // folded token table, which is only valid because the inverse is undone first.
        if(hadamard_.tied_output&&!report.config.hadamard_tied_output)return fail("Invalid prism.hadamard.tied_output");
    }
    for(const auto& d:report.tensors){Tensor t;t.dims=d.dimensions;t.elements=1;for(auto dim:d.dimensions)t.elements*=static_cast<std::size_t>(dim);t.raw_bytes=static_cast<std::size_t>(d.byte_size);t.format=format_for_ggml(d.ggml_type);
        const bool matrix=d.name.find(".bias")==std::string::npos&&(d.role==model::TensorRole::token_embedding||d.role==model::TensorRole::output||d.role==model::TensorRole::query||d.role==model::TensorRole::key||d.role==model::TensorRole::value||d.role==model::TensorRole::attention_output||d.role==model::TensorRole::ffn_gate||d.role==model::TensorRole::ffn_up||d.role==model::TensorRole::ffn_down);
        // Per-tensor map trace. Flushed per line: the point of this trace is that it
        // survives a crash, and an abort on the next tensor discards a buffered
        // stderr stream (which is exactly why the probe appeared to "die silently").
        // ZEN_MAP_TRACE=0 silences it.
        static const bool map_trace=[](){const char* v=std::getenv("ZEN_MAP_TRACE");return !(v&&v[0]=='0'&&v[1]=='\0');}();
        if(map_trace)std::fprintf(stderr,"[map] %-38s type=%u role=%d matrix=%d fmt=%s bytes=%zu\n",d.name.c_str(),d.ggml_type,int(d.role),int(matrix),t.format?t.format->name.c_str():"null",t.raw_bytes);
        if(matrix&&t.format&&d.offset<=mapped->size()&&report.tensor_data_offset<=mapped->size()-d.offset&&t.raw_bytes<=mapped->size()-report.tensor_data_offset-d.offset){t.raw=mapped->data()+report.tensor_data_offset+d.offset;}
        else {std::string e;if(map_trace)std::fprintf(stderr,"[map] fallback-decode %s\n",d.name.c_str());if(!gguf::read_tensor_f32(path,report,d.name,t.values,&e)){std::fflush(stderr);return fail("Tensor "+d.name+": "+e);}}
        if(map_trace)std::fflush(stderr);
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
    // Sequential K-block repack at load (candidate (a)): when
    // ZEN_CPU_Q8_REPACK_KBLK=1, transpose every Q8 GEMM weight that the
    // K-blocked kernel would stream into the panel-major layout so the
    // K-panel reads are sequential. Qwen3.5 fused tensors have a different
    // row count per name, so they are not registered (they keep falling
    // back to the raw K-blocked path when KBLK=1). The output head uses
    // GEMV in decode and a single tail token in prefill, so its repack
    // would pay hundreds of MB for one small GEMM: skipped.
    // Repack without KBLK would never be consumed (the repacked kernel is
    // the K-blocked kernel), so both gates must be on to build the buffers.
    if(!qwen35_arch&&quant::q8_repack_enabled()&&quant::q8_kblk_enabled()){
        auto register_repack=[&](const std::string& name,std::size_t rows,std::size_t cols){
            const auto* t=tensor(name);
            if(!t||!t->raw||!t->format||t->format->encoding!=zen::quant::Encoding::q8_0)return;
            if(!rows||!cols||cols%32)return;
            if(t->elements!=rows*cols){std::fprintf(stderr,"[repack] skip %s: elements=%zu rows*cols=%zu\n",name.c_str(),t->elements,rows*cols);return;}
            auto packed=quant::repack_q8_0_kblk(t->raw,rows,cols);
            if(packed.empty()){std::fprintf(stderr,"[repack] skip %s: repack failed\n",name.c_str());return;}
            std::fprintf(stderr,"[repack] %-38s rows=%zu cols=%zu repack_bytes=%zu raw_bytes=%zu\n",name.c_str(),rows,cols,packed.size(),t->raw_bytes);
            quant::register_q8_repack(t->raw,rows,cols,std::move(packed));
        };
        const std::size_t QW=std::size_t(config_.heads)*head_dim, KW=std::size_t(config_.kv_heads)*head_dim;
        for(std::uint32_t l=0;l<config_.layers;++l){
            const std::string p="blk."+std::to_string(l)+".";
            register_repack(p+"attn_q.weight",QW,config_.embedding);
            register_repack(p+"attn_k.weight",KW,config_.embedding);
            register_repack(p+"attn_v.weight",KW,config_.embedding);
            register_repack(p+"attn_output.weight",QW,config_.embedding);
            register_repack(p+"ffn_gate.weight",config_.ffn,config_.embedding);
            register_repack(p+"ffn_up.weight",config_.ffn,config_.embedding);
            register_repack(p+"ffn_down.weight",config_.embedding,config_.ffn);
        }
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
    const float eps=config_.rms_epsilon;
    std::vector<float> x(T*E),norm(T*E),tmp(T*E);
    std::vector<float> qkv(T*conv_dim),z(T*value_dim),conv_out(T*conv_dim);
    // q/k/v are shared by both layer kinds: the attention path needs T*QW / T*KW,
    // the GDN path needs T*key_dim (q, k) and T*value_dim (v). With S*NV larger
    // than kv_heads*head_dim the GDN copy below ran past the end of `v`, which is
    // the 0xC0000005 / 0xC0000374 pair seen in the field.
    const std::size_t q_cap=std::max<std::size_t>(QW,key_dim);
    const std::size_t v_cap=std::max<std::size_t>(KW,value_dim);
    std::vector<float> q(T*q_cap),k(T*q_cap),v(T*v_cap),att(T*QW);
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
    // Per-layer folded matmul inputs get the forward Hadamard transform exactly
    // ONCE per activation. Upstream memoises the transform per (activation,
    // rotation) and several weights share one input, so re-applying it per weight
    // (the previous shape of this lambda) silently re-transformed the buffer.
    // `input` holds T consecutive rows of `width` floats. The transform is
    // PER ROW: every token's activation is its own vector, and the reference
    // transforms the whole [T, width] activation matrix. Calling it once on the
    // base pointer only ever transformed row 0, so in prefill every token after
    // the first multiplied its folded weights by an unrotated vector -- correct
    // for a single-token decode, silently wrong for anything longer.
    auto fold_for=[&](std::uint32_t l,float* input,std::size_t width,std::initializer_list<const char*> suffixes){
        for(const char* suffix:suffixes)if(hadamard_folded(suffix,l)){
            for(std::size_t t=0;t<T;++t)hadamard_transform_input(input+t*width,width,h_scratch);
            return;
        }
    };
    for(std::uint32_t l=0;l<config_.layers;++l){
        const std::string p="blk."+std::to_string(l)+".";
        const bool recurrent=layer_is_recurrent(l);
        for(std::size_t t=0;t<T;++t)rms_norm(x.data()+t*E,tensor(p+"attn_norm.weight")->values.data(),norm.data()+t*E,E,eps);
        if(recurrent){
            // ---- gated delta net layer ----
            fold_for(l,norm.data(),E,{"attn_qkv.weight","attn_gate.weight","ssm_beta.weight","ssm_alpha.weight"});
            if(!tensor_gemm(*tensor(p+"attn_qkv.weight"),norm.data(),qkv.data(),T,conv_dim,E)||
               !tensor_gemm(*tensor(p+"attn_gate.weight"),norm.data(),z.data(),T,value_dim,E))return false;
            if(!tensor_gemm(*tensor(p+"ssm_beta.weight"),norm.data(),beta.data(),T,NV,E)||
               !tensor_gemm(*tensor(p+"ssm_alpha.weight"),norm.data(),alpha.data(),T,NV,E))return false;
            std::vector<float> gdn_out(T*value_dim);
            // conv state ring: [CK-1, conv_dim] at offset 0, plus one scratch
            // window per position this batch can visit (prefill or decode).
            if(convc[l].size()<std::size_t(CK-1+start_pos+T)*conv_dim)return false;
            // The window has to be published BEFORE the convolution reads it;
            // window_row() below reads these same rows for idx >= 0.
            const float* state_ring=convc[l].data();
            float* window_base=convc[l].data()+std::size_t(CK-1)*conv_dim;
            std::copy_n(qkv.data(),T*conv_dim,window_base);
            const auto window_row=[&](std::int64_t idx)->const float*{
                if(idx<0)return state_ring+std::size_t(-idx-1)*conv_dim;
                return window_base+std::size_t(idx)*conv_dim;
            };
            // conv1d over [state_rows | qkv window], per channel.
            for(std::size_t ch=0;ch<conv_dim;++ch){
                const auto* cw=tensor(p+"ssm_conv1d.weight")->values.data()+ch*CK;
                float* out=conv_out.data()+ch;
                for(std::size_t t=0;t<T;++t){
                    float sum=0.0f;
                    for(std::size_t w=0;w<CK;++w){
                        const std::int64_t idx=std::int64_t(t)+std::int64_t(w)-(std::int64_t)CK+1;
                        sum+=cw[w]*window_row(idx)[ch];
                    }
                    out[t*conv_dim]=sum;
                }
            }
            silu_inplace(conv_out.data(),T*conv_dim);
            // q/k l2-norm. Upstream normalises the joint [S, 2*NK] view of the
            // convolved q|k block, i.e. ONE norm per key head over its S
            // consecutive dims of q followed by the same S of k (qwen35.cpp
            // build_layer_attn_linear: qk_conv = ggml_l2_norm(view(head_k_dim,
            // 2*num_k_heads, ...))). Normalising the whole 2*key_dim span at
            // once, as the earlier scalar loop did, is a different operation.
            const std::size_t qk_span=key_dim*2;
            for(std::size_t t=0;t<T;++t){
                float* row=conv_out.data()+t*conv_dim;
                for(std::size_t h=0;h<NK;++h){
                    float* qh=row+h*S; double ss=0.0;
                    for(std::size_t i=0;i<S;++i)ss+=double(qh[i])*qh[i];
                    const float inv=1.0f/std::max(std::sqrt(float(ss)),eps);
                    for(std::size_t i=0;i<S;++i)qh[i]*=inv;
                    float* kh2=row+key_dim+h*S; ss=0.0;
                    for(std::size_t i=0;i<S;++i)ss+=double(kh2[i])*kh2[i];
                    const float inv2=1.0f/std::max(std::sqrt(float(ss)),eps);
                    for(std::size_t i=0;i<S;++i)kh2[i]*=inv2;
                }
                std::copy_n(row+qk_span,value_dim,v.data()+t*v_cap);
                std::copy_n(row,key_dim,q.data()+t*q_cap);
                std::copy_n(row+key_dim,key_dim,k.data()+t*q_cap);
            }
            // GDN recurrence per value head, per token (scalar-gate path of
            // ggml_compute_forward_gated_delta_net_one_chunk, ggml-cpu/ops.cpp).
            const std::uint32_t interval=config_.full_attention_interval;
            (void)interval;
            // Per-layer single-sequence state: NV heads of S x S. Sizing this
            // by value_dim^2 (NV^2 times larger) cost a 300 MB allocation per
            // recurrent layer at 27B scale for a 4.4 MB working set.
            std::vector<float> state(std::size_t(NV)*S*S);
            // state layout mirrors ggml: per head S x S, stored transposed rows.
            // ssm cache: ssmc[l] holds NV*S*S floats.
            std::copy_n(ssmc[l].begin(),ssmc[l].size(),state.begin());
            const float scale=1.0f/std::sqrt(float(S));
            // gates: beta = sigmoid(beta_proj); decay g = a * softplus(alpha + dt_bias); a = -exp(ssm_a)
            const auto* ssm_a=tensor(p+"ssm_a")->values.data();
            const auto* dt_bias=tensor(p+"ssm_dt.bias")->values.data();
            std::vector<float> delta(S);
            for(std::size_t t=0;t<T;++t){
                for(std::size_t h=0;h<NV;++h){
                    beta[t*NV+h]=1.0f/(1.0f+std::exp(-beta[t*NV+h]));
                    const float xx=alpha[t*NV+h]+dt_bias[h];
                    const float sp=xx>20.0f?xx:std::log(1.0f+std::exp(xx));
                    // ssm_a already holds -A_log.exp() as stored in the checkpoint
                    // (qwen35.cpp: `gate = alpha_softplus * ssm_a  // -A_log.exp() *
                    // softplus`, and ops.cpp applies raw_a[iv1] directly). Applying
                    // exp() and a negation again here reset the decay to a
                    // completely different curve.
                    a_gate[h]=ssm_a[h]*sp;
                }
                // q/k are repeated over value heads in TILED order: value head h
                // reads key head h % NK (ops.cpp: iq1 = iv1 % neq1). The previous
                // h*key_dim/value_dim mapped [0,1,2,3] to [0,0,0,0] for NK=... .
                for(std::size_t h=0;h<NV;++h){
                    const std::size_t kh=h%NK;
                    const float* qh=q.data()+t*q_cap+kh*S;
                    const float* kh_=k.data()+t*q_cap+kh*S;
                    const float* vh=conv_out.data()+t*conv_dim+qk_span+h*S;
                    float* sh=state.data()+h*S*S;
                    const float g=std::exp(a_gate[h]);
#ifdef ZEN_HAS_AVX2_KERNELS
                    if (gdn_avx2_enabled()) {
                        zen::gdn_decay_avx2(sh, g, S*S);
                        zen::gdn_delta_avx2(sh, kh_, vh, delta.data(), beta[t*NV+h], S);
                        zen::gdn_update_avx2(sh, kh_, delta.data(), S);
                        float* oh=gdn_out.data()+t*value_dim+h*S;
                        zen::gdn_output_avx2(sh, qh, oh, scale, S);
                    } else
#endif
                    {
                    for(std::size_t p=0;p<S*S;++p)sh[p]*=g;
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
            }
            std::copy_n(state.begin(),ssmc[l].size(),ssmc[l].begin());
            // conv state update. The ring is indexed newest-first along dim 0
            // ([d_conv-1, channels], qwen35 build_conv_state), and the window fed
            // to the conv above reads it as convc[-idx-1]; so row 0 gets the LAST
            // input token and row w token T-1-w. Writing it forward (row 0 <- token
            // T-CK+1) reversed the window and corrupted every later step.
            // publish this batch's conv inputs at scratch rows [0, T), shift the
            // ring by T, then take the newest CK-1 rows of [ring | scratch].
            for(std::size_t ch=0;ch<conv_dim;++ch){
                // a fresh sequence starts with an empty ring
                if(start_pos==0)for(std::size_t w=0;w<CK-1;++w)convc[l][w*conv_dim+ch]=0.0f;
                for(std::size_t w=0;w<CK-1;++w){
                    const std::int64_t idx=std::int64_t(T)-1-std::int64_t(w);
                    convc[l][w*conv_dim+ch]=window_row(idx)[ch];
                }
            }
            // Gated norm: qwen35 build_norm_gated calls
            //   ggml_swiglu_split(ctx0, gate=z, normalized)
            // and, with two distinct sources, ggml_compute_forward_swiglu_f32 does
            //   y = silu(src0) * src1 = silu(z) * normalized
            // (the swapped/cat form only applies when src1 is absent). So the gate
            // z is what gets the SiLU, and it multiplies the normalised value.
            for(std::size_t t=0;t<T;++t){
                float* o=gdn_out.data()+t*value_dim;
                for(std::size_t h=0;h<NV;++h)rms_norm(o+h*S,tensor(p+"ssm_norm.weight")->values.data(),o+h*S,S,eps);
                // o *= silu(z) elementwise == silu_mul(z, o, o): same formula via
                // the AVX2 Cephes path (ZEN_CPU_SILU_AVX2=0 -> scalar). Aliasing
                // up==out is safe (reads before writes per element/block).
                silu_mul(z.data()+t*value_dim,o,o,value_dim);
            }
            // prism.hadamard.gdn_v_grouped: the weight was folded with its value
            // heads in GROUPED order [hd, rep, nk] while the recurrence emits
            // TILED [hd, nk, rep]; permute to grouped before the input transform
            // (llama_hadamard_transform::perm_* in the reference).
            // Driven by the GGUF key itself: a checkpoint can declare the
            // permutation without (or before) any folded weight, and the earlier
            // condition silently tied it to hadamard_.block_size being set.
            if(config_.hadamard_gdn_v_grouped&&NV%NK==0&&NV/NK>1){
                const std::size_t rep=NV/NK;
                std::vector<float> tiled(gdn_out);
                for(std::size_t t=0;t<T;++t){
                    const float* src=tiled.data()+t*value_dim;
                    float* dst=gdn_out.data()+t*value_dim;
                    for(std::size_t h=0;h<NV;++h){
                        const std::size_t nk=h%NK, rp=h/NK;
                        // ggml_permute(0,2,1,3) over the (hd, nk, rep) view puts
                        // the rep axis first, so grouped head = rp + rep*nk.
                        // (rp*NK+nk collapses to h and made this whole block a
                        // no-op: the permutation was never actually applied.)
                        std::copy_n(src+std::size_t(h)*S,S,dst+(rp+rep*nk)*S);
                    }
                }
            }
            fold_for(l,gdn_out.data(),value_dim,{"ssm_out.weight"});
            if(!tensor_gemm(*tensor(p+"ssm_out.weight"),gdn_out.data(),tmp.data(),T,E,value_dim))return false;
        }else{
            // ---- full attention layer: fused Q|gate projection (Qwen3.5) ----
            const std::size_t QGW=2*QW;
            q.resize(T*QGW);
            fold_for(l,norm.data(),E,{"attn_q.weight","attn_k.weight","attn_v.weight"});
            if(!tensor_gemm(*tensor(p+"attn_q.weight"),norm.data(),q.data(),T,QGW,E)||
               !tensor_gemm(*tensor(p+"attn_k.weight"),norm.data(),k.data(),T,KW,E)||
               !tensor_gemm(*tensor(p+"attn_v.weight"),norm.data(),v.data(),T,KW,E))return false;
            // Split the fused Q|gate projection. The reference does NOT store it
            // as [all Q | all gate]: qwen35.cpp build_layer_attn takes two 3-D
            // views of the SAME buffer with element strides of 2*head_dim,
            //   Qcur  = view_3d(Qcur_full, D, H, T, stride 2D, 2D*H, offset 0)
            //   gate  = view_3d(Qcur_full, D, H, T, stride 2D, 2D*H, offset D)
            // so the layout is INTERLEAVED per head: [q0 g0 q1 g1 ...].
            // Splitting contiguously ([all q | all gate]) reads q0,g1,q2,... as
            // the query and q1,g0,q3,... as the gate: every attention output is
            // gated with a different head's values and the queries mix head
            // halves. GDN layers never touch this path, so it only showed up
            // once the fixture grew past the first full-attention layer.
            std::vector<float> qgate(T*QW);
            for(std::size_t t=0;t<T;++t){
                const float* src=q.data()+t*QGW;
                float* qdst=q.data()+t*QW;
                float* gdst=qgate.data()+t*QW;
                for(std::size_t h=0;h<H;++h){
                    std::copy_n(src+std::size_t(h)*2*D,D,qdst+std::size_t(h)*D);
                    std::copy_n(src+std::size_t(h)*2*D+D,D,gdst+std::size_t(h)*D);
                }
            }
            for(std::size_t t=0;t<T;++t){
                const std::uint32_t pos=start_pos+static_cast<std::uint32_t>(t);
                qk_rms_norm(q.data()+t*QW,H,D,tensor(p+"attn_q_norm.weight")->values.data(),eps);
                qk_rms_norm(k.data()+t*KW,KH,D,tensor(p+"attn_k_norm.weight")->values.data(),eps);
                // Full-attention layers use M-RoPE sections [11,11,10,0]. For
                // text all four position streams equal the token position, so
                // ggml_mrope_cache_init degenerates to the standard RoPE cache
                // over the 64 rotary dims; rope() is exactly that with the
                // half-split layout this arch uses. (Vision inputs would need
                // the section-split streams.)
                rope(q.data()+t*QW,H,D,config_.rope_dim,pos,config_.rope_base,config_.rope_scaling_type,config_.rope_scaling_factor,config_.rope_original_context,config_.rope_low_freq_factor,config_.rope_high_freq_factor,config_.rope_yarn_beta_fast,config_.rope_yarn_beta_slow,false);
                rope(k.data()+t*KW,KH,D,config_.rope_dim,pos,config_.rope_base,config_.rope_scaling_type,config_.rope_scaling_factor,config_.rope_original_context,config_.rope_low_freq_factor,config_.rope_high_freq_factor,config_.rope_yarn_beta_fast,config_.rope_yarn_beta_slow,false);
                std::copy_n(k.data()+t*KW,KW,kc[l].data()+std::size_t(pos)*KW);
                std::copy_n(v.data()+t*KW,KW,vc[l].data()+std::size_t(pos)*KW);
            }
            std::fill(att.begin(),att.end(),0.0f);
            // positions are absolute: decode re-enters this function with
            // start_pos != 0, so the write/read offset must use pos, not the
            // batch-local index t.
            if(T>0&&start_pos+T>kc[l].size()/KW)return false;
            std::vector<float> scores(std::size_t(start_pos)+T);
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
                    if(!std::isfinite(scores[j]))return false;
                    max_score=std::max(max_score,scores[j]);
                }
                double denom=0;for(std::size_t j=first;j<=pos;++j){scores[j]=std::exp(scores[j]-max_score);denom+=scores[j];}
                if(!(denom>0.0)||!std::isfinite(denom))return false;
                float*out=att.data()+t*QW+std::size_t(h)*D;
                for(std::size_t j=first;j<=pos;++j){const float a=scores[j]/static_cast<float>(denom);
                    const float*vt=vc[l].data()+j*KW+std::size_t(kh)*D;for(std::size_t i=0;i<D;++i)out[i]+=a*vt[i];}
            }
            // sigmoid gate after attention. Reference (qwen35.cpp build_layer_attn):
            //   gate_sigmoid = ggml_sigmoid(ctx0, gate);  cur = ggml_mul(ctx0, cur, gate_sigmoid);
            // i.e. attn_pregate * sigmoid(gate). This used to compute
            // g/(1+exp(-g)), which is SiLU, not sigmoid -- a different function
            // that agrees only at g=0 and biases every attention output.
            for(std::size_t t=0;t<T;++t)for(std::size_t i=0;i<QW;++i){const float g=qgate[t*QW+i];att[t*QW+i]/=(1.0f+std::exp(-g));}
            fold_for(l,att.data(),QW,{"attn_output.weight"});
            if(!tensor_gemm(*tensor(p+"attn_output.weight"),att.data(),tmp.data(),T,E,QW))return false;
        }
        for(std::size_t i=0;i<T*E;++i)x[i]+=tmp[i];
        // post-attention norm then FFN (sandwich norm), residual from before it.
        std::vector<float> res_before(T*E);std::copy_n(x.data(),T*E,res_before.begin());
        for(std::size_t t=0;t<T;++t)rms_norm(x.data()+t*E,tensor(p+"post_attention_norm.weight")->values.data(),norm.data()+t*E,E,eps);
        fold_for(l,norm.data(),E,{"ffn_gate.weight","ffn_up.weight"});
        if(!tensor_gemm(*tensor(p+"ffn_gate.weight"),norm.data(),gate.data(),T,F,E)||
           !tensor_gemm(*tensor(p+"ffn_up.weight"),norm.data(),up.data(),T,F,E))return false;
        silu_mul(gate.data(),up.data(),ffn.data(),T*F);
        fold_for(l,ffn.data(),F,{"ffn_down.weight"});
        if(!tensor_gemm(*tensor(p+"ffn_down.weight"),ffn.data(),tmp.data(),T,E,F))return false;
        // x here is `res + attn`; the FFN wraps that same residual, so the result is
        // res_before + ffn_out. Adding res_before to the existing x double-counted it.
        for(std::size_t i=0;i<T*E;++i)x[i]=res_before[i]+tmp[i];
        if(!finite_values(x.data(),x.size()))return false;
    }
    std::vector<float> norm_last(E);
    rms_norm(x.data()+(T-1)*E,tensor("output_norm.weight")->values.data(),norm_last.data(),E,eps);
    // The LM head reads a folded input only when output.weight is a folded weight
    // (a folded tied table is handled by the version-2 tied_output path).
    if(hadamard_.block_size&&(hadamard_.folded_names.count("output.weight")||hadamard_.tied_output))
        hadamard_transform_input(norm_last.data(),E,h_scratch);
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
    
    // Initialize D3D12 backend if available
    #if defined(ZEN_HAS_D3D12)
    if (!zen::d3d12::init()) {
        if (const char* trace = std::getenv("ZEN_D3D12_TRACE"); trace && trace[0] == '1') {
            std::printf("[d3d12] forward_batch: zen::d3d12::init() failed\n");
        }
    }
    #endif
    
    const std::size_t QW=std::size_t(H)*D,KW=std::size_t(KH)*D;
    std::vector<float> x(T*E),norm(T*E),q(T*QW),k(T*KW),v(T*KW),att(T*QW),tmp(T*E),gate(T*F),up(T*F),ffn(T*F);
    const auto* emb=tensor("token_embd.weight");if(!emb)return false;
    for(std::size_t t=0;t<T;++t){if(tokens[t]>=config_.vocab)return false;
        if(emb->raw&&emb->format){const auto rb=quant::row_bytes(*emb->format,E);if(!rb||!quant::dequantize(*emb->format,emb->raw+std::size_t(tokens[t])*rb,E,x.data()+t*E))return false;}
        else if(emb->values.size()==std::size_t(config_.vocab)*E)std::copy_n(emb->values.data()+std::size_t(tokens[t])*E,E,x.data()+t*E);else return false;
    }
    // Prefill RoPE tables: [tokens * pairs], token-major, built ONCE per
    // forward_batch (positions start_pos..start_pos+T-1, layer-independent). The
    // old path rebuilt them inside rope() for every (layer, token, Q/K) call --
    // 2*16*128 = 4096 pow/cos/sin table builds at T=128, ~29 ms of the ~165 ms
    // prefill (zen_sherlock transformer.prefill.rope, 16 ops, 1839 us/layer) --
    // even though rope() already hoisted within a call (ZI-P3). The tables below
    // are shared by the hoisted CPU fallback (rope_apply_table, bit-identical:
    // rope_table is the verbatim table half of rope()) and the optional GPU
    // prefill rope (zen::d3d12::rope_apply_prefill), so both paths consume the
    // same numbers. Built lazily inside layer 0's rope scope so the build cost
    // stays inside the same "transformer.prefill.rope" accounting.
    // Switches (default OFF / safe):
    //   ZEN_D3D12_ROPE_PREFILL=1   opt in to GPU prefill rope (decode unchanged)
    //   ZEN_D3D12_ROPE_PREFILL=0   (or unset) forces the CPU rope path
    //   ZEN_CPU_ROPE=0             strict: CPU rope fallback disabled; GPU must
    //                              succeed or the forward fails. Requires
    //                              ZEN_D3D12_ROPE_PREFILL=1, and bypasses the
    //                              MIN_T gate below (GPU for any T>=1).
    //   ZEN_D3D12_ROPE_PREFILL_MIN_T=N  minimum T for GPU (default 4; the
    //                              crossover bench sets the shippable value).
    // ALiBi models skip RoPE entirely (no tables, neither path runs).
    static const bool rope_prefill_gpu_on = []{const char* e=std::getenv("ZEN_D3D12_ROPE_PREFILL");return e&&e[0]=='1';}();
    static const bool rope_cpu_disallowed = []{const char* e=std::getenv("ZEN_CPU_ROPE");return e&&e[0]=='0';}();
    static const std::size_t rope_prefill_min_t = []{const char* e=std::getenv("ZEN_D3D12_ROPE_PREFILL_MIN_T");if(e){int v=std::atoi(e);if(v>=1)return static_cast<std::size_t>(v);}return std::size_t(4);}();
    // Prefill elementwise on the GPU (RMSNorm, SiLU, residuals, device-resident
    // FFN). Default OFF: the CPU path below is unchanged when unset.
    //   ZEN_D3D12_PREFILL_GPU=1   opt in: attn/output RMSNorm via
    //                             zen::d3d12::rms_norm, the whole FFN half
    //                             (attn_output GEMM, both residuals, FFN norm,
    //                             gate|up, SiLU, down) via the device-resident
    //                             zen::d3d12::ffn_prefill_fused in ONE submit
    //                             per layer instead of 4 GEMM submits plus a
    //                             CPU SiLU and two CPU adds.
    // T<16 stays on the host path: speculative verify batches (T=4..8) and
    // short prompts keep the bit-identical single-GEMM route, and the fused
    // working buffers only amortise over multi-token prefills.
    // Any GPU failure (backend absent, non-Q8 weights, BitNet subln, shape
    // guards) falls back to the host path below per layer, never fails.
    static const bool prefill_gpu_on = []{const char* e=std::getenv("ZEN_D3D12_PREFILL_GPU");return e&&e[0]=='1';}();
    const bool use_prefill_gpu = prefill_gpu_on&&T>=16;
    std::vector<float> prefill_cos,prefill_sin;
    std::size_t prefill_pairs=0;
    bool prefill_tables_built=false;
    std::future<void> prefetch;
    for(std::uint32_t l=0;l<config_.layers;++l){if(prefetch.valid())prefetch.get();if(l+1<config_.layers&&prefetch_enabled_)prefetch=std::async(std::launch::async,[this,l]{prefetch_layer(l+1);});const std::string p="blk."+std::to_string(l)+".";
        bool attn_norm_gpu=false;
#if defined(ZEN_HAS_D3D12)
        if(use_prefill_gpu)attn_norm_gpu=zen::d3d12::rms_norm(x.data(),tensor(p+"attn_norm.weight")->values.data(),norm.data(),T,E,config_.rms_epsilon);
#endif
        if(!attn_norm_gpu)for(std::size_t t=0;t<T;++t)rms_norm(x.data()+t*E,tensor(p+"attn_norm.weight")->values.data(),norm.data()+t*E,E,config_.rms_epsilon);
        if(!finite_values(norm.data(),norm.size()))return false;
        if(T==1){
            // Decode: Q, K and V share the normed input, so submitting them as one
            // batch replaces two extra queue round trips (~70us each) per layer.
            const DenseTransformerRuntime::Tensor* const qkv_t[3]={tensor(p+"attn_q.weight"),tensor(p+"attn_k.weight"),tensor(p+"attn_v.weight")};
            float* const qkv_y[3]={q.data(),k.data(),v.data()};
            const std::size_t qkv_r[3]={QW,KW,KW};
            if(!tensor_matvec_batch(qkv_t,norm.data(),qkv_y,qkv_r,3,E))return false;
        }else{
            // Prefill Q|K|V share the normed input: submit as one batched GEMM
            // when ZEN_D3D12_BATCH=1, otherwise tensor_gemm_batch loops the
            // single path. Attn O and FFN down stay separate (attention / SiLU
            // in between).
            const DenseTransformerRuntime::Tensor* const qkv_t[3]={tensor(p+"attn_q.weight"),tensor(p+"attn_k.weight"),tensor(p+"attn_v.weight")};
            float* const qkv_y[3]={q.data(),k.data(),v.data()};
            const std::size_t qkv_r[3]={QW,KW,KW};
            if(!tensor_gemm_batch(qkv_t,norm.data(),qkv_y,qkv_r,3,T,E))return false;
        }
        if(!finite_values(q.data(),q.size())||!finite_values(k.data(),k.size())||!finite_values(v.data(),v.size()))return false;
        for(const auto& item:std::array<std::pair<const char*,std::vector<float>*>,3>{{{"attn_q.bias",&q},{"attn_k.bias",&k},{"attn_v.bias",&v}}})if(const auto*bias=tensor(p+item.first)){const std::size_t width=item.first[5]=='q'?QW:KW;if(bias->values.size()!=width)return false;for(std::size_t t=0;t<T;++t)for(std::size_t i=0;i<width;++i)(*item.second)[t*width+i]+=bias->values[i];}
        const auto* qnorm=tensor(p+"attn_q_norm.weight");const auto* knorm=tensor(p+"attn_k_norm.weight");
        if((qnorm==nullptr)!=(knorm==nullptr))return false;
        if(qnorm&&(qnorm->values.size()!=D||knorm->values.size()!=D))return false;
        {zen::diag::Scope scope_rope("transformer.prefill.rope",zen::diag::Phase::compute);
        // Collect all K/V pointers for batched KV cache append
        std::vector<std::size_t> kv_layers;
        std::vector<std::size_t> kv_positions;
        std::vector<const float*> kv_ks;
        std::vector<const float*> kv_vs;
        kv_layers.reserve(T);
        kv_positions.reserve(T);
        kv_ks.reserve(T);
        kv_vs.reserve(T);

        // Build the per-token tables once (layer 0's scope); reuse for every layer
        // and for BOTH the CPU fallback and the GPU prefill rope. rope_dim is
        // already validated even and <= D (forward load check), but apply the same
        // clamp rope() applies so signed-off shapes cannot drift apart.
        if(!prefill_tables_built){
            prefill_tables_built=true;
            if(!config_.use_alibi){
                std::uint32_t rd=std::min(config_.rope_dim,static_cast<std::uint32_t>(D));
                rd-=rd%2;
                prefill_pairs=rd/2;
                if(prefill_pairs){
                    prefill_cos.assign(T*std::size_t(prefill_pairs),0.0f);
                    prefill_sin.assign(T*std::size_t(prefill_pairs),0.0f);
                    for(std::size_t t=0;t<T;++t){
                        const std::uint32_t pos=start_pos+static_cast<std::uint32_t>(t);
                        std::vector<float> tc,ts;
                        rope_table(rd,pos,config_.rope_base,config_.rope_scaling_type,config_.rope_scaling_factor,config_.rope_original_context,config_.rope_low_freq_factor,config_.rope_high_freq_factor,config_.rope_yarn_beta_fast,config_.rope_yarn_beta_slow,tc,ts);
                        std::copy_n(tc.begin(),std::min<std::size_t>(tc.size(),std::size_t(prefill_pairs)),prefill_cos.data()+t*std::size_t(prefill_pairs));
                        std::copy_n(ts.begin(),std::min<std::size_t>(ts.size(),std::size_t(prefill_pairs)),prefill_sin.data()+t*std::size_t(prefill_pairs));
                    }
                }
            }
        }
        // qk RMS-norm for every token BEFORE rope (unchanged order:
        // GEMM -> bias -> qk_norm -> rope). Hoisted out of the per-token loop so
        // the batched GPU rope sees fully-normalised Q/K in one dispatch; the
        // CPU fallback below applies rope to the same normalised rows.
        if(qnorm)for(std::size_t t=0;t<T;++t){
            qk_rms_norm(q.data()+t*QW,H,D,qnorm->values.data(),config_.rms_epsilon);
            qk_rms_norm(k.data()+t*KW,KH,D,knorm->values.data(),config_.rms_epsilon);
        }
        if(!config_.use_alibi&&prefill_pairs){
            bool q_done=false,k_done=false;
#if defined(ZEN_HAS_D3D12)
            // GPU prefill rope: ONE batched dispatch per tensor covering all T
            // tokens. In-place on host (staging precedes the dispatch, the readback
            // follows the fence), so no extra host copy is needed. Q and K are
            // attempted independently: a single failure falls back only that tensor
            // (the other keeps its validated GPU result) instead of re-rotating.
            // KV append is deliberately NOT fused here: routed K must land in
            // BOTH the host cache (CPU attention fallback) and the device cache
            // (decode/flash), and the host copy is ~256 KB/layer at T=128 (see
            // kv_cache_append_batched below). Fusing would save ~4 MB total
            // across all layers while risking host/device divergence; the rope
            // win is the trig + rotation work, not the cache copy.
            // rope_pairs_adjacent (config.rope_pairs_adjacent) is forwarded as
            // `interleaved_pairs` (shader pad0) so llama NORM vs NeoX matches the
            // CPU path exactly.
            const bool try_gpu_rope=rope_prefill_gpu_on&&prefill_pairs*2<=std::size_t(D)
                &&(rope_cpu_disallowed||T>=rope_prefill_min_t);
            if(try_gpu_rope){
                if(zen::d3d12::rope_apply_prefill(q.data(),prefill_cos.data(),prefill_sin.data(),q.data(),T,H,D,std::size_t(prefill_pairs),config_.rope_pairs_adjacent))q_done=true;
                if(zen::d3d12::rope_apply_prefill(k.data(),prefill_cos.data(),prefill_sin.data(),k.data(),T,KH,D,std::size_t(prefill_pairs),config_.rope_pairs_adjacent))k_done=true;
            }
#endif
            if(!q_done||!k_done){
                // Strict mode: CPU rope is disabled, so any tensor not rotated by
                // the GPU is a hard failure rather than a silent fallback to the
                // disabled path. This also fires when GPU was never eligible
                // (ZEN_D3D12_ROPE_PREFILL unset): ZEN_CPU_ROPE=0 alone has no
                // rope implementation and must fail, not silently use CPU.
                if(rope_cpu_disallowed)return false;
                if(!q_done)for(std::size_t t=0;t<T;++t)rope_apply_table(q.data()+t*QW,H,D,static_cast<std::uint32_t>(prefill_pairs),prefill_cos.data()+t*std::size_t(prefill_pairs),prefill_sin.data()+t*std::size_t(prefill_pairs),config_.rope_pairs_adjacent);
                if(!k_done)for(std::size_t t=0;t<T;++t)rope_apply_table(k.data()+t*KW,KH,D,static_cast<std::uint32_t>(prefill_pairs),prefill_cos.data()+t*std::size_t(prefill_pairs),prefill_sin.data()+t*std::size_t(prefill_pairs),config_.rope_pairs_adjacent);
            }
        }
        for(std::size_t t=0;t<T;++t){
            const std::uint32_t pos = start_pos + static_cast<std::uint32_t>(t);
            std::copy_n(k.data()+t*KW,KW,kc[l].data()+std::size_t(pos)*KW);
            std::copy_n(v.data()+t*KW,KW,vc[l].data()+std::size_t(pos)*KW);
            // Keep the device cache in step with the host one: decode reads the device
            // cache, so without this the prompt positions are missing from it and decode
            // attends over the wrong window. Not fatal if it fails - decode then falls
            // back to the host path, which uses kc/vc.
#if defined(ZEN_HAS_D3D12)
            if(!config_.use_alibi && device_attention_available(config_.layers,kc[l].size()/KW,KH,D,config_.rms_epsilon)){
                kv_layers.push_back(l);
                kv_positions.push_back(pos);
                kv_ks.push_back(k.data()+t*KW);
                kv_vs.push_back(v.data()+t*KW);
            }
#endif
        }
        // Batched KV cache append for this layer
#if defined(ZEN_HAS_D3D12)
        if(!kv_layers.empty() && !config_.use_alibi && device_attention_available(config_.layers,kc[l].size()/KW,KH,D,config_.rms_epsilon)){
            if (const char* trace = std::getenv("ZEN_D3D12_TRACE"); trace && trace[0] == '1') {
                std::printf("[d3d12] batched kv append (runtime): layer=%zu num_tokens=%zu\n", l, kv_layers.size());
            }
            zen::d3d12::kv_cache_append_batched(kv_layers.data(), kv_layers.size(), kv_positions.data(), kv_ks.data(), kv_vs.data(), KH, D);
        }
#endif
        }
        // FlashAttention for prefill: try GPU first, fallback to CPU
        const float inv_scale = 1.0f / std::sqrt(float(D));
        bool flash_ok = false;
        #if defined(ZEN_HAS_D3D12)
        // ZEN_D3D12_FLASH=0 forces the CPU attention path. The GPU kernel is
        // numerically verified against the CPU reference by zen_d3d12_flash_parity,
        // so this exists to A/B a suspected interaction, not as a normal switch.
        static const bool flash_enabled = [] {
            const char* e = std::getenv("ZEN_D3D12_FLASH");
            return !(e && e[0] == '0');
        }();
        // The shader has no sliding-window mask (first_pos is always 0), so only
        // take the GPU path when the model attends to the whole prefix.
        if(flash_enabled && !config_.use_alibi && config_.sliding_window==0 &&
           device_attention_available(config_.layers,kc[l].size()/KW,KH,D,config_.rms_epsilon)){
            // Pass the context the KV cache was actually configured with, not the
            // model's maximum context: the layer stride is derived from it, and a
            // mismatch makes the shader read megabytes past the cache.
            const std::size_t kv_ctx = kc[l].size()/KW;

            if (const char* trace = std::getenv("ZEN_D3D12_TRACE"); trace && trace[0] == '1') {
                std::printf("[d3d12] flash_attention: layer=%u T=%zu H=%u KH=%u D=%u start_pos=%u kv_ctx=%zu\n",
                           l, T, H, KH, D, start_pos, kv_ctx);
            }

            flash_ok = zen::d3d12::flash_attention_prefill(
                q.data(), k.data(), v.data(), att.data(),
                T, H, KH, D, inv_scale,
                l, start_pos, kv_ctx, KH, D
            );
            
            if (flash_ok) {
                if (!finite_values(att.data(), att.size())) {
                    std::fprintf(stderr, "[fa] non-finite flash_attention output layer=%u\n", l);
                    return false;
                }
            } else {
                if (const char* trace = std::getenv("ZEN_D3D12_TRACE"); trace && trace[0] == '1') {
                    std::printf("[d3d12] flash_attention fallback to CPU\n");
                }
            }
        }
        #endif
        if (!flash_ok) {
            {zen::diag::Scope scope_attn("transformer.prefill.attention",zen::diag::Phase::compute);
        std::fill(att.begin(),att.end(),0.0f);const std::uint32_t group=H/KH;const float inv_scale=1.0f/std::sqrt(float(D));
        std::atomic<bool> attn_ok{true};
        zen::ResourceGovernor attn_gov;
        const unsigned attn_workers=attn_gov.current_policy().worker_threads;
        // Tiled CPU attention (llama path only; Q35 stays scalar). Env gate
        // ZEN_CPU_ATTN_TILED=1, default off. Query-block tiling with online
        // softmax: (block,head) tasks stream KV tiles once across the query
        // block (K/V row loaded once per (tile,j), reused for QB queries from
        // L1/L2), instead of the baseline which rescans all KV per query and
        // buffers a scores[context] row in three passes. KV layout unchanged
        // (kc[l]+j*KW). Causal + sliding-window + ALiBi mask identically:
        // for each j the valid query range is [j-start-q0, +sw-1] clamped,
        // scores for valid pairs only, same finite checks, double denom,
        // float exp/scores/intermediate as baseline (online rescale differs
        // only in summation order, well within tolerance).
        // QB/KB default 32/32 (env ZEN_CPU_ATTN_QB/KB override, 1..128).
        static const bool attn_tiled=[](){const char*e=std::getenv("ZEN_CPU_ATTN_TILED");return e&&e[0]=='1';}();
        static const std::size_t attn_qb=[](){const char*e=std::getenv("ZEN_CPU_ATTN_QB");if(e){int v=std::atoi(e);if(v>=1&&v<=128)return static_cast<std::size_t>(v);}return std::size_t(32);}();
        static const std::size_t attn_kb=[](){const char*e=std::getenv("ZEN_CPU_ATTN_KB");if(e){int v=std::atoi(e);if(v>=1&&v<=128)return static_cast<std::size_t>(v);}return std::size_t(32);}();
        const std::size_t QB=attn_qb,KB=attn_kb;
        if(attn_tiled){
            const std::size_t num_blocks=(T+QB-1)/QB;
            const std::size_t tiled_tasks=num_blocks*static_cast<std::size_t>(H);
            auto tiled_block=[&](std::size_t block,std::uint32_t h,std::vector<float>& m,std::vector<double>& denom,std::vector<float>& acc,std::vector<std::size_t>& firsts){
                const std::size_t q0=block*QB,qn=std::min(QB,T-q0);
                const std::uint32_t kh=h/group;
                const std::uint32_t pos0=start_pos+static_cast<std::uint32_t>(q0);
                const std::size_t first_min=config_.sliding_window&&pos0+1>config_.sliding_window?pos0+1-config_.sliding_window:0;
                const std::size_t pos_max=static_cast<std::size_t>(start_pos+static_cast<std::uint32_t>(q0+qn-1));
                for(std::size_t qi=0;qi<qn;++qi){const std::uint32_t pos=start_pos+static_cast<std::uint32_t>(q0+qi);firsts[qi]=config_.sliding_window&&pos+1>config_.sliding_window?pos+1-config_.sliding_window:0;m[qi]=-std::numeric_limits<float>::infinity();denom[qi]=0.0;}
                std::fill(acc.begin(),acc.begin()+qn*D,0.0f);
                for(std::size_t j0=first_min;j0<=pos_max&&attn_ok.load(std::memory_order_relaxed);){
                    const std::size_t j1=std::min(j0+KB,pos_max+1);
                    for(std::size_t j=j0;j<j1;++j){
                        const std::int64_t low=(std::int64_t)j-(std::int64_t)start_pos-(std::int64_t)q0;
                        std::size_t qi_lo=low<0?0:(std::size_t)low;if(qi_lo>=qn)continue;
                        std::size_t qi_hi=qn-1;
                        if(config_.sliding_window){const std::int64_t high=(std::int64_t)j+(std::int64_t)config_.sliding_window-1-(std::int64_t)start_pos-(std::int64_t)q0;if(high<(std::int64_t)qi_lo)continue;if(high<(std::int64_t)qi_hi)qi_hi=(std::size_t)high;}
                        if(qi_lo>qi_hi)continue;
                        const float*kt=kc[l].data()+j*KW+std::size_t(kh)*D;
                        const float*vt=vc[l].data()+j*KW+std::size_t(kh)*D;
                        for(std::size_t qi=qi_lo;qi<=qi_hi;++qi){
                            const std::uint32_t pos=start_pos+static_cast<std::uint32_t>(q0+qi);
                            if(j>pos||j<firsts[qi])continue;
                            const float*qh=q.data()+(q0+qi)*QW+std::size_t(h)*D;
                            float s=dot_f32(qh,kt,D)*inv_scale;
                            if(config_.use_alibi)s+=zen::attention::alibi_bias(h,H,pos,static_cast<std::uint32_t>(j));
                            if(!std::isfinite(s)){attn_ok.store(false,std::memory_order_relaxed);return;}
                            float& mq=m[qi];double& lq=denom[qi];float* aq=acc.data()+qi*D;
                            if(s>mq){
                                if(mq!=-std::numeric_limits<float>::infinity()){
                                    const float scale=std::exp(mq-s);
                                    for(std::size_t i=0;i<D;++i)aq[i]*=scale;
                                    lq*=(double)scale;
                                }
                                mq=s;lq+=1.0;
                                axpy_f32(1.0f,vt,aq,D);
                            }else{
                                const float w=std::exp(s-mq);
                                lq+=(double)w;
                                axpy_f32(w,vt,aq,D);
                            }
                        }
                    }
                    j0+=KB;
                }
                for(std::size_t qi=0;qi<qn;++qi){
                    const double lq=denom[qi];
                    if(!(lq>0.0)||!std::isfinite(lq)){attn_ok.store(false,std::memory_order_relaxed);return;}
                    const float inv=1.0f/static_cast<float>(lq);
                    float*out=att.data()+(q0+qi)*QW+std::size_t(h)*D;
                    const float* aq=acc.data()+qi*D;
                    for(std::size_t i=0;i<D;++i)out[i]=aq[i]*inv;
                }
            };
            if(attn_workers>1&&tiled_tasks>1){
                zen::ComputePool::instance().parallel_for(tiled_tasks,attn_workers,[&](std::size_t begin,std::size_t end){
                    std::vector<float> m(QB),acc(QB*D);std::vector<double> denom(QB);std::vector<std::size_t> firsts(QB);
                    for(std::size_t idx=begin;idx<end&&attn_ok.load(std::memory_order_relaxed);++idx)
                        tiled_block(idx/static_cast<std::size_t>(H),static_cast<std::uint32_t>(idx%static_cast<std::size_t>(H)),m,denom,acc,firsts);
                });
            }else{
                std::vector<float> m(QB),acc(QB*D);std::vector<double> denom(QB);std::vector<std::size_t> firsts(QB);
                for(std::size_t block=0;block<num_blocks&&attn_ok.load(std::memory_order_relaxed);++block)
                    for(std::uint32_t h=0;h<H;++h)tiled_block(block,h,m,denom,acc,firsts);
            }
        }else{
        // ZI-P6: every (token, head) score row is independent, so the causal
        // attention of a prefill batch runs across the pool instead of on one
        // thread. This was the last serial O(T^2) block in the batch path.
        const std::size_t attn_tasks=T*static_cast<std::size_t>(H);
        auto attn_row=[&](std::size_t t,std::uint32_t h,std::vector<float>& scores){
            const std::uint32_t kh=h/group;const float*qh=q.data()+t*QW+std::size_t(h)*D;
            float max_score=-std::numeric_limits<float>::infinity();
            const std::uint32_t pos=start_pos+static_cast<std::uint32_t>(t);
            const std::size_t first=config_.sliding_window&&pos+1>config_.sliding_window?pos+1-config_.sliding_window:0;
            for(std::size_t j=first;j<=pos;++j){
                const float*kt=kc[l].data()+j*KW+std::size_t(kh)*D;
                scores[j]=dot_f32(qh,kt,D)*inv_scale;
                if(config_.use_alibi)scores[j]+=zen::attention::alibi_bias(h,H,pos,static_cast<std::uint32_t>(j));
                if(!std::isfinite(scores[j])){attn_ok.store(false,std::memory_order_relaxed);return;}
                max_score=std::max(max_score,scores[j]);
            }
            double denom=0;for(std::size_t j=first;j<=pos;++j){scores[j]=std::exp(scores[j]-max_score);denom+=scores[j];}
            if(!(denom>0.0)||!std::isfinite(denom)){attn_ok.store(false,std::memory_order_relaxed);return;}
            const float inv=1.0f/static_cast<float>(denom);
            float*out=att.data()+t*QW+std::size_t(h)*D;
            for(std::size_t j=first;j<=pos;++j){
                const float a=scores[j]*inv;
                const float*vt=vc[l].data()+j*KW+std::size_t(kh)*D;
                axpy_f32(a,vt,out,D);
            }
        };
        if(attn_workers>1&&attn_tasks>1){
            zen::ComputePool::instance().parallel_for(attn_tasks,attn_workers,[&](std::size_t begin,std::size_t end){
                std::vector<float> scores(config_.context);
                for(std::size_t idx=begin;idx<end&&attn_ok.load(std::memory_order_relaxed);++idx)
                    attn_row(idx/H,static_cast<std::uint32_t>(idx%H),scores);
            });
        }else{
            std::vector<float> scores(config_.context);
            for(std::size_t t=0;t<T&&attn_ok.load(std::memory_order_relaxed);++t)
                for(std::uint32_t h=0;h<H;++h)attn_row(t,h,scores);
        }
        }
        if(!attn_ok.load())return false;
        }
        }
        // FFN half: the whole block (attn_output GEMM, residual, FFN norm,
        // gate|up GEMMs, SiLU, down GEMM, residual) in one device-resident
        // submission when the prefill GPU gate is on; otherwise the separate
        // host ops below. BitNet subln skips the fused path (its norms are
        // CPU), as do non-quantised or differently-shaped weights.
        bool ffn_gpu=false;
#if defined(ZEN_HAS_D3D12)
        if(use_prefill_gpu&&!config_.bitnet_variant){
            const auto* lo=tensor(p+"attn_output.weight");const auto* lg=tensor(p+"ffn_gate.weight");
            const auto* lu=tensor(p+"ffn_up.weight");const auto* ldw=tensor(p+"ffn_down.weight");
            const auto* lfn=tensor(p+"ffn_norm.weight");
            if(tensor_is_q8(lo)&&tensor_is_q8(lg)&&tensor_is_q8(lu)&&tensor_is_q8(ldw)&&lfn&&
               tensor(p+"attn_q.bias")==nullptr&&tensor(p+"attn_k.bias")==nullptr&&tensor(p+"attn_v.bias")==nullptr){
                ffn_gpu=zen::d3d12::ffn_prefill_fused(lo->raw,lg->raw,lu->raw,ldw->raw,
                    lfn->values.data(),x.data(),att.data(),x.data(),T,E,QW,F,config_.rms_epsilon);
            }
        }
#endif
        if(ffn_gpu){
            if(!finite_values(x.data(),x.size()))return false;
        }else{
        // BitNet subln: RMS-normalise each token's attention output before
        // the wo projection.
        if(config_.bitnet_variant)for(std::size_t t=0;t<T;++t)rms_norm(att.data()+t*QW,tensor(p+"attn_sub_norm.weight")->values.data(),att.data()+t*QW,QW,config_.rms_epsilon);
        if(!tensor_gemm(*tensor(p+"attn_output.weight"),att.data(),tmp.data(),T,E,QW))return false;
        for(std::size_t i=0;i<T*E;++i)x[i]+=tmp[i];
        for(std::size_t t=0;t<T;++t)rms_norm(x.data()+t*E,tensor(p+"ffn_norm.weight")->values.data(),norm.data()+t*E,E,config_.rms_epsilon);
        if(!finite_values(norm.data(),norm.size()))return false;
        if(T==1){
            const DenseTransformerRuntime::Tensor* const fu_t[2]={tensor(p+"ffn_gate.weight"),tensor(p+"ffn_up.weight")};
            float* const fu_y[2]={gate.data(),up.data()};
            const std::size_t fu_r[2]={F,F};
            if(!tensor_matvec_batch(fu_t,norm.data(),fu_y,fu_r,2,E))return false;
        }else{
            // Prefill gate|up share the normed input: one batched GEMM when
            // ZEN_D3D12_BATCH=1, otherwise the single path. FFN down stays
            // separate (SiLU in between).
            const DenseTransformerRuntime::Tensor* const fu_t[2]={tensor(p+"ffn_gate.weight"),tensor(p+"ffn_up.weight")};
            float* const fu_y[2]={gate.data(),up.data()};
            const std::size_t fu_r[2]={F,F};
            if(!tensor_gemm_batch(fu_t,norm.data(),fu_y,fu_r,2,T,E))return false;
        }
        {zen::diag::Scope scope_silu("transformer.prefill.silu",zen::diag::Phase::compute);
        silu_mul(gate.data(),up.data(),ffn.data(),T*F);}
        // BitNet subln: RMS-normalise the FFN hidden activation before ffn_down.
        if(config_.bitnet_variant)for(std::size_t t=0;t<T;++t)rms_norm(ffn.data()+t*F,tensor(p+"ffn_sub_norm.weight")->values.data(),ffn.data()+t*F,F,config_.rms_epsilon);
        if(!tensor_gemm(*tensor(p+"ffn_down.weight"),ffn.data(),tmp.data(),T,E,F))return false;
        for(std::size_t i=0;i<T*E;++i)x[i]+=tmp[i];
        if(!finite_values(x.data(),x.size()))return false;
        }
    }
    const auto* norm_weight=tensor("output_norm.weight");if(!norm_weight||norm_weight->values.size()!=E)return false;
    const std::size_t first_tail=T-tail_count;std::vector<float> norm_tail(tail_count*E),logits_tail(tail_count*config_.vocab);
    // Output-norm tail on the GPU when the prefill gate is on (one dispatch,
    // fallback to the CPU loop when the call declines).
    bool out_norm_gpu=false;
#if defined(ZEN_HAS_D3D12)
    if(use_prefill_gpu)out_norm_gpu=zen::d3d12::rms_norm(x.data()+(first_tail)*E,norm_weight->values.data(),norm_tail.data(),tail_count,E,config_.rms_epsilon);
#endif
    if(!out_norm_gpu)for(std::size_t t=0;t<tail_count;++t)rms_norm(x.data()+(first_tail+t)*E,norm_weight->values.data(),norm_tail.data()+t*E,E,config_.rms_epsilon);
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
    // Stage timing, opt-in via ZEN_D3D12_STAGE=1. Printed as a cumulative
    // average so the cost of each phase is attributable instead of inferred
    // from the gap between wall clock and the GPU fence.
    const bool stage_trace=[](){const char* e=std::getenv("ZEN_D3D12_STAGE");return e&&e[0]=='1';}();
    struct StageTimer {
        bool on; std::chrono::steady_clock::time_point t0;
        explicit StageTimer(bool o):on(o),t0(o?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{}){}
        double us()const{return on?std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t0).count():0.0;}
    };
    static double acc_setup=0,acc_pipeline=0,acc_head=0; static std::size_t acc_n=0;
    StageTimer tm_setup(stage_trace);
    const std::size_t QW=std::size_t(H)*D,KW=std::size_t(KH)*D;
    const auto* emb=tensor("token_embd.weight");
    std::vector<float> x(E),norm(E),q(QW),k(KW),v(KW),att(QW),tmp(E),gate(F),up(F),ffn(F);
    // ZI-P4/P6: one scratch score row per head per call instead of a heap vector
    // allocated inside the per-layer attention scope, so heads can run in parallel.
    std::vector<float> attn_scratch(static_cast<std::size_t>(H)*(pos+1));
    if(emb->raw&&emb->format){const auto rb=quant::row_bytes(*emb->format,E);if(!rb||!quant::dequantize(*emb->format,emb->raw+std::size_t(token)*rb,E,x.data()))return false;}
    else if(emb->values.size()==std::size_t(config_.vocab)*E)std::copy_n(emb->values.data()+std::size_t(token)*E,E,x.data());else return false;
    // ZI-P7: no prefetch thread in the per-token path. Spawning one std::async
    // per layer cost ~7ms of a 61ms decode step (measured 16.3 -> 18.7 tok/s with
    // it removed) because the weights are already resident in the page cache.

    // ZI-P8: rope_table depends only on (rope_dim, pos, base, scaling parameters) —
    // never on the layer — yet the device paths below rebuilt identical tables up to
    // three times per layer per token (pipeline qualification, layer_decode,
    // attention_block): three pow/trig table builds plus two heap allocations per
    // layer, every token. Build one shared pair per token on first use and hand
    // every site the same table. Bit-identical: it is the same deterministic
    // function with the same arguments the per-layer calls used, and every
    // consumer takes it read-only (const float*).
    std::vector<float> rope_cos,rope_sin;
    auto device_rope_table=[&](){
        if(rope_cos.empty())
            rope_table(config_.rope_dim,pos,config_.rope_base,config_.rope_scaling_type,config_.rope_scaling_factor,config_.rope_original_context,config_.rope_low_freq_factor,config_.rope_high_freq_factor,config_.rope_yarn_beta_fast,config_.rope_yarn_beta_slow,rope_cos,rope_sin);
    };
    bool pipeline_completed=false;
    bool lm_head_fused=false;
    double setup_us = 0.0, pipeline_us = 0.0;
#if defined(ZEN_HAS_D3D12)
    // Try to run all layers through the GPU pipeline (single submit+wait for all layers)
    // First check if all layers qualify for the pipeline path
    bool all_layers_qualify = true;
    struct LayerParams {
        const std::uint8_t* q_w, *k_w, *v_w, *o_w, *gate_w, *up_w, *down_w;
        const float* attn_norm_w; const float* ffn_norm_w;
        const float* cos; const float* sin; // shared per-token table (rope_cos/rope_sin above)
        std::size_t pairs, layer, pos, first_pos;
    };
    std::vector<LayerParams> pipeline_layers;
    pipeline_layers.reserve(config_.layers);

    for(std::uint32_t l=0;l<config_.layers;++l){
        const std::string p="blk."+std::to_string(l)+".";
        const auto* lq=tensor(p+"attn_q.weight");const auto* lk=tensor(p+"attn_k.weight");
        const auto* lv=tensor(p+"attn_v.weight");const auto* lo=tensor(p+"attn_output.weight");
        const auto* lg=tensor(p+"ffn_gate.weight");const auto* lu=tensor(p+"ffn_up.weight");
        const auto* ldw=tensor(p+"ffn_down.weight");
        const auto* ln=tensor(p+"attn_norm.weight");const auto* lfn=tensor(p+"ffn_norm.weight");
        if(!config_.use_alibi&&!config_.bitnet_variant&&lq&&lk&&lv&&lo&&lg&&lu&&ldw&&ln&&lfn
           &&tensor_is_q8(lq)&&tensor_is_q8(lk)&&tensor_is_q8(lv)&&tensor_is_q8(lo)
           &&tensor_is_q8(lg)&&tensor_is_q8(lu)&&tensor_is_q8(ldw)
           &&!tensor(p+"attn_q.bias")&&!tensor(p+"attn_k.bias")&&!tensor(p+"attn_v.bias")
           &&!tensor(p+"attn_q_norm.weight")
           &&device_attention_available(config_.layers,kc[l].size()/KW,KH,D,config_.rms_epsilon)){
            // Qualifies for pipeline
            LayerParams lp;
            lp.q_w = lq->raw; lp.k_w = lk->raw; lp.v_w = lv->raw; lp.o_w = lo->raw;
            lp.gate_w = lg->raw; lp.up_w = lu->raw; lp.down_w = ldw->raw;
            lp.attn_norm_w = ln->values.data(); lp.ffn_norm_w = lfn->values.data();
            lp.layer = l; lp.pos = pos;
            lp.first_pos = config_.sliding_window&&pos+1>config_.sliding_window?pos+1-config_.sliding_window:0;
            device_rope_table();
            lp.cos = rope_cos.data(); lp.sin = rope_sin.data();
            lp.pairs = rope_cos.size();
            pipeline_layers.push_back(std::move(lp));
        } else {
            all_layers_qualify = false;
        }
    }

    // If all layers qualify, run the entire model through the GPU pipeline
    // (single submit+wait for all layers). Default ON: the backend records every
    // layer into the OWNING list (s.list, the one submitted) with a per-layer
    // hin staging buffer, and every arena is returned to its entry state
    // (ybuf/fbuf/hbuf UAV, datt COMMON, caches UAV, hin NSR) so the next layer
    // and the next token start from known states. Seeded determinism verified
    // (same seed twice -> identical ids + logits max_abs < 2e-6, rel < 1e-6) and
    // pipeline-vs-per-layer parity holds on the oracle prompt. Kill switch:
    // ZEN_D3D12_PIPELINE=0 restores the per-layer path.
    static const bool pipeline_enabled = [] {
        const char* e = std::getenv("ZEN_D3D12_PIPELINE");
        return !(e && e[0] == '0');
    }();
    // Fused LM head plumbing: the pipeline may record final-rmsnorm + vocab GEMV +
    // logits readback into its single submission; fused==true means logits[] is
    // already valid and the host skips both its final rms and the vocab matvec.
    // The backend rejects (fused=false) on kill-switch/VRAM-pressure/shape limits.
    zen::d3d12::LmHeadFuse lm_fuse{};
    if(pipeline_enabled && all_layers_qualify && pipeline_layers.size() == config_.layers && pipeline_layers.size() <= 64) {
        setup_us = tm_setup.us();
        StageTimer tm_pipe(stage_trace);
        zen::d3d12::LayerDecodeParams d3d_layers[64];
        for(std::size_t i=0;i<pipeline_layers.size();++i){
            auto& lp = pipeline_layers[i];
            d3d_layers[i].q_w = lp.q_w; d3d_layers[i].k_w = lp.k_w; d3d_layers[i].v_w = lp.v_w; d3d_layers[i].o_w = lp.o_w;
            d3d_layers[i].gate_w = lp.gate_w; d3d_layers[i].up_w = lp.up_w; d3d_layers[i].down_w = lp.down_w;
            d3d_layers[i].attn_norm_w = lp.attn_norm_w; d3d_layers[i].ffn_norm_w = lp.ffn_norm_w;
            d3d_layers[i].cos = lp.cos; d3d_layers[i].sin = lp.sin;
            d3d_layers[i].pairs = lp.pairs; d3d_layers[i].layer = lp.layer; d3d_layers[i].pos = lp.pos; d3d_layers[i].first_pos = lp.first_pos;
            d3d_layers[i].rope_pairs_adjacent = config_.rope_pairs_adjacent ? 1u : 0u;
        }
        // Fused LM head: the pipeline records final-rmsnorm + vocab GEMV + logits
        // readback into the same submission, so the host skips both its final rms
        // and the 128256-row matvec below.
        logits.resize(config_.vocab);
        const auto* lm_out=tensor("output.weight");
        if(!lm_out)lm_out=tensor("token_embd.weight");
        if(lm_out){
            lm_fuse.output_norm_w=tensor("output_norm.weight")->values.data();
            lm_fuse.lm_head_w=tensor_is_q8(lm_out)?lm_out->raw:nullptr;
            lm_fuse.vocab_rows=config_.vocab;
            lm_fuse.epsilon=config_.rms_epsilon;
            lm_fuse.logits=logits.data();
        }
        bool pipeline_ok = zen::d3d12::layer_decode_pipeline(d3d_layers, pipeline_layers.size(), x.data(), x.data(), E, H, KH, D, F, config_.rms_epsilon, lm_fuse.lm_head_w?&lm_fuse:nullptr);
        pipeline_us = tm_pipe.us();
        if (pipeline_ok) {
            pipeline_completed=true;
            lm_head_fused=lm_fuse.fused;
            // Pipeline succeeded - skip per-layer path
            if (const char* trace = std::getenv("ZEN_D3D12_TRACE"); trace && trace[0] == '1') {
                std::printf("[d3d12] pipeline: SUCCESS for token pos=%u\n", pos);
            }
        } else {
            // Pipeline failed - fall through to per-layer path
            all_layers_qualify = false;
            if (const char* trace = std::getenv("ZEN_D3D12_TRACE"); trace && trace[0] == '1') {
                std::printf("[d3d12] pipeline: FALLBACK for token pos=%u (pipeline_decode returned false)\n", pos);
            }
        }
        
        // Also log when not all layers qualify
        if (!all_layers_qualify && pipeline_layers.size() < config_.layers) {
            if (const char* trace = std::getenv("ZEN_D3D12_TRACE"); trace && trace[0] == '1') {
                std::printf("[d3d12] pipeline: NOT ALL QUALIFY pipeline_layers=%zu vs config_.layers=%u\n", pipeline_layers.size(), config_.layers);
            }
        }
    }

#endif
    if(!pipeline_completed) {
        for(std::uint32_t l=0;l<config_.layers;++l){
        const std::string p="blk."+std::to_string(l)+".";
        // ZI-P8: one map lookup per weight per layer per token. The loop body used
        // to re-run tensor(p+"...") — a string concatenation plus hash find each —
        // up to three times for the same weight (the fused-submission probe, then
        // again in the attention/FFN per-op fallbacks). tensors_ is immutable while
        // forward() runs (only load() writes it), so these hoisted pointers are
        // exactly what the inline lookups returned.
        const auto* lq=tensor(p+"attn_q.weight");const auto* lk=tensor(p+"attn_k.weight");
        const auto* lv=tensor(p+"attn_v.weight");const auto* lo=tensor(p+"attn_output.weight");
        const auto* lg=tensor(p+"ffn_gate.weight");const auto* lu=tensor(p+"ffn_up.weight");
        const auto* ldw=tensor(p+"ffn_down.weight");
        const auto* ln=tensor(p+"attn_norm.weight");const auto* lfn=tensor(p+"ffn_norm.weight");
        const auto* qb=tensor(p+"attn_q.bias");const auto* kb=tensor(p+"attn_k.bias");const auto* vb=tensor(p+"attn_v.bias");
        const auto* qnorm=tensor(p+"attn_q_norm.weight");const auto* knorm=tensor(p+"attn_k_norm.weight");
        // Whole layer in one device submission (task 7), when this block needs nothing
        // from the host: no ALiBi, no biases, no qk-norm and not BitNet. Anything else
        // falls through to the per-op path below, which handles those cases.
#if defined(ZEN_HAS_D3D12)
        {
            if(!config_.use_alibi&&!config_.bitnet_variant&&lq&&lk&&lv&&lo&&lg&&lu&&ldw&&ln&&lfn
               &&tensor_is_q8(lq)&&tensor_is_q8(lk)&&tensor_is_q8(lv)&&tensor_is_q8(lo)
               &&tensor_is_q8(lg)&&tensor_is_q8(lu)&&tensor_is_q8(ldw)
               &&!qb&&!kb&&!vb
               &&!qnorm
               &&device_attention_available(config_.layers,kc[l].size()/KW,KH,D,config_.rms_epsilon)){
                device_rope_table();
                const std::uint32_t lf_pos=config_.sliding_window&&pos+1>config_.sliding_window?pos+1-config_.sliding_window:0;
                if(zen::d3d12::layer_decode(lq->raw,lk->raw,lv->raw,lo->raw,lg->raw,lu->raw,ldw->raw,ln->values.data(),lfn->values.data(),rope_cos.data(),rope_sin.data(),rope_cos.size(),x.data(),x.data(),l,pos,lf_pos,E,H,KH,D,F,config_.rms_epsilon,config_.rope_pairs_adjacent))continue;
            }
        }
#endif
        rms_norm(x.data(),ln->values.data(),norm.data(),E,config_.rms_epsilon);
        if(!finite_values(norm.data(),E))return false;
        bool attn_device_ok=false;
        {
            zen::diag::Scope scope("transformer.attention.projection",zen::diag::Phase::compute);
            // One submission for Q|K|V: they share the normed input, and a lone
            // dispatch pays ~70us of fixed queue latency that dwarfs its arithmetic.
            const DenseTransformerRuntime::Tensor* const qkv_t[3]={lq,lk,lv};
            float* const qkv_y[3]={q.data(),k.data(),v.data()};
            const std::size_t qkv_r[3]={QW,KW,KW};
            if(!tensor_matvec_batch(qkv_t,norm.data(),qkv_y,qkv_r,3,E))return false;
            for(const auto& item:std::array<std::pair<const DenseTransformerRuntime::Tensor*,std::vector<float>*>,3>{{{qb,&q},{kb,&k},{vb,&v}}})if(const auto*bias=item.first){if(bias->values.size()!=item.second->size())return false;for(std::size_t i=0;i<item.second->size();++i)(*item.second)[i]+=bias->values[i];}
            if(!finite_values(q.data(),q.size())||!finite_values(k.data(),k.size())||!finite_values(v.data(),v.size()))return false;
            if((qnorm==nullptr)!=(knorm==nullptr))return false;
            if(qnorm){if(qnorm->values.size()!=D||knorm->values.size()!=D)return false;qk_rms_norm(q.data(),H,D,qnorm->values.data(),config_.rms_epsilon);qk_rms_norm(k.data(),KH,D,knorm->values.data(),config_.rms_epsilon);}
            // Device attention: rope(q), rope(k), the cache append and GQA attention in
            // one submission. q/k are still pre-rope here, which is what the kernel
            // expects. On failure the host path below runs unchanged.
#if defined(ZEN_HAS_D3D12)
            if(!config_.use_alibi&&device_attention_available(config_.layers,kc[l].size()/KW,KH,D,config_.rms_epsilon)){
                device_rope_table();
                const std::uint32_t dev_first=config_.sliding_window&&pos+1>config_.sliding_window?pos+1-config_.sliding_window:0;
                attn_device_ok=zen::d3d12::attention_block(q.data(),k.data(),v.data(),rope_cos.data(),rope_sin.data(),att.data(),H,KH,D,rope_cos.size(),l,pos,dev_first,config_.rope_pairs_adjacent);
            }
#endif
            if(!config_.use_alibi){
                rope(q.data(),H,D,config_.rope_dim,pos,config_.rope_base,config_.rope_scaling_type,config_.rope_scaling_factor,config_.rope_original_context,config_.rope_low_freq_factor,config_.rope_high_freq_factor,config_.rope_yarn_beta_fast,config_.rope_yarn_beta_slow,config_.rope_pairs_adjacent);
                rope(k.data(),KH,D,config_.rope_dim,pos,config_.rope_base,config_.rope_scaling_type,config_.rope_scaling_factor,config_.rope_original_context,config_.rope_low_freq_factor,config_.rope_high_freq_factor,config_.rope_yarn_beta_fast,config_.rope_yarn_beta_slow,config_.rope_pairs_adjacent);
            }
            std::copy(k.begin(),k.end(),kc[l].begin()+std::size_t(pos)*KW);
            std::copy(v.begin(),v.end(),vc[l].begin()+std::size_t(pos)*KW);
        }
        // Set when the FFN half of this layer ran as one GPU submission, which also
        // covers the attention output matvec and its residual.
        bool half_fused=false;
        {
            zen::diag::Scope scope("transformer.attention.softmax",zen::diag::Phase::compute);
            // The device block already filled `att`; only the host fallback recomputes
            // it. Everything after this guard (BitNet subln, attn_output, FFN) still runs.
            if(!attn_device_ok){
            std::fill(att.begin(),att.end(),0.0f);
            const std::uint32_t group=H/KH;
            const float inv_scale=1.0f/std::sqrt(float(D));
            const std::uint32_t first_pos=config_.sliding_window&&pos+1>config_.sliding_window?pos+1-config_.sliding_window:0;
            std::atomic<bool> attn_ok{true};
            // ZI-P6: one scratch row per head so the heads of a decode step run in
            // parallel. Each head row is independent; only `att`/`scores` per head
            // are touched, so the split is race-free.
            auto head_row=[&](std::uint32_t h,float* scores){
                const std::uint32_t kh=h/group;
                const float* qh=q.data()+std::size_t(h)*D;
                float max_score=-std::numeric_limits<float>::infinity();
                for(std::uint32_t t=first_pos;t<=pos;++t){
                    const float* kt=kc[l].data()+std::size_t(t)*KW+std::size_t(kh)*D;
                    scores[t]=dot_f32(qh,kt,D)*inv_scale;
                    if(config_.use_alibi)scores[t]+=zen::attention::alibi_bias(h,H,pos,t);
                    if(!std::isfinite(scores[t])){attn_ok.store(false,std::memory_order_relaxed);return;}
                    max_score=std::max(max_score,scores[t]);
                }
                double denom=0.0;
                for(std::uint32_t t=first_pos;t<=pos;++t){scores[t]=std::exp(scores[t]-max_score);denom+=scores[t];}
                if(!(denom>0.0)||!std::isfinite(denom)){attn_ok.store(false,std::memory_order_relaxed);return;}
                float* out=att.data()+std::size_t(h)*D;
                const float inv=1.0f/static_cast<float>(denom);
                for(std::uint32_t t=first_pos;t<=pos;++t){
                    const float a=scores[t]*inv;
                    const float* vt=vc[l].data()+std::size_t(t)*KW+std::size_t(kh)*D;
                    axpy_f32(a,vt,out,D);
                }
            };
            static const unsigned attn_head_budget=[](){zen::ResourceGovernor g;return std::max(1u,g.current_policy().worker_threads);}();
            const unsigned attn_heads=std::min<unsigned>(H,attn_head_budget);
            if(attn_heads>1){
                const std::size_t row=static_cast<std::size_t>(pos)+1;
                zen::ComputePool::instance().parallel_for(H,attn_heads,[&](std::size_t begin,std::size_t end){
                    for(std::size_t h=begin;h<end&&attn_ok.load(std::memory_order_relaxed);++h)
                        head_row(static_cast<std::uint32_t>(h),attn_scratch.data()+h*row);
                });
            }else{
                for(std::uint32_t h=0;h<H&&attn_ok.load(std::memory_order_relaxed);++h)
                    head_row(h,attn_scratch.data()+std::size_t(h)*(pos+1));
            }
            if(!attn_ok.load())return false;
            }  // !attn_device_ok
            // BitNet subln: RMS-normalise the attention output before the
            // wo projection (src/models/bitnet.cpp: attn_sub_norm).
            if(config_.bitnet_variant)rms_norm(att.data(),tensor(p+"attn_sub_norm.weight")->values.data(),att.data(),QW,config_.rms_epsilon);
            if(!config_.bitnet_variant){
                if(lo&&lfn&&lg&&lu&&ldw)
                    half_fused=tensor_ffn_residual_fused(*lo,*lg,*lu,*ldw,lfn->values.data(),
                                                         x.data(),att.data(),x.data(),E,QW,F,
                                                         config_.rms_epsilon);
            }
            if(!half_fused){
                if(!tensor_matvec(*lo,att.data(),tmp.data(),E,QW))return false;
                for(std::uint32_t i=0;i<E;++i)x[i]+=tmp[i];
            }
        }
        {
            zen::diag::Scope scope("transformer.ffn",zen::diag::Phase::compute);
            // The fused path already normed into VRAM, so this only runs on the
            // separate-op fallback.
            if(!half_fused)rms_norm(x.data(),lfn->values.data(),norm.data(),E,config_.rms_epsilon);
            if(!finite_values(norm.data(),E))return false;
            // Seeded from half_fused so a fused layer skips the FFN batch entirely
            // instead of recomputing what the single submission already produced.
            bool ffn_fused=half_fused;
            if(!half_fused&&!config_.bitnet_variant&&lg&&lu&&ldw){
                // gate|up|SiLU|down in one submission: the activation never leaves VRAM,
                // which removes a full queue round trip from every layer.
                ffn_fused=tensor_ffn(*lg,*lu,*ldw,norm.data(),tmp.data(),F,E,E);
                if(ffn_fused)for(std::uint32_t i=0;i<E;++i)x[i]+=tmp[i];
            }
            if(!ffn_fused){
                const DenseTransformerRuntime::Tensor* const fu_t[2]={lg,lu};
                float* const fu_y[2]={gate.data(),up.data()};
                const std::size_t fu_r[2]={F,F};
                if(!tensor_matvec_batch(fu_t,norm.data(),fu_y,fu_r,2,E))return false;
                silu_mul(gate.data(),up.data(),ffn.data(),F);
                // BitNet subln: RMS-normalise the FFN hidden activation before
                // ffn_down (src/models/bitnet.cpp: ffn_sub_norm).
                if(config_.bitnet_variant)rms_norm(ffn.data(),tensor(p+"ffn_sub_norm.weight")->values.data(),ffn.data(),F,config_.rms_epsilon);
                if(!tensor_matvec(*ldw,ffn.data(),tmp.data(),E,F))return false;
                for(std::uint32_t i=0;i<E;++i)x[i]+=tmp[i];
            }
            if(!finite_values(x.data(),x.size()))return false;
        }
    }
    }
    if(!lm_head_fused){
        rms_norm(x.data(),tensor("output_norm.weight")->values.data(),norm.data(),E,config_.rms_epsilon);
        if(!finite_values(norm.data(),E))return false;
    }
    // ZI-P8: the exhaustive finite scan over the vocab-sized logit row (128256
    // floats, ~512KB) ran twice per decode token — once inside tensor_matvec below,
    // once in the return statement — and is part of the measured ~2.4ms/token CPU
    // fat between the LM head and the next sample. ZEN_VALIDATE_LOGITS=1 restores
    // both scans. By default non-finite logits are still caught before any token is
    // returned: generate() checks the selected token's logit, greedy/normal
    // sampling only ever select finite entries, and prefill keeps its full scans
    // (forward_batch's logits_tail check and every prefill matvec validation).
    static const bool validate_logits=[](){const char* e=std::getenv("ZEN_VALIDATE_LOGITS");return e&&e[0]=='1';}();
    logits.resize(config_.vocab);
    const auto* output=tensor("output.weight");
    if(!output)output=tensor("token_embd.weight");
    StageTimer tm_head(stage_trace);
    // lm_fuse.fused: the pipeline submission already ran final-rmsnorm + vocab GEMV
    // on the device and read the logits back into logits[] — nothing to do here.
    if(!lm_head_fused&&!tensor_matvec(*output,norm.data(),logits.data(),config_.vocab,E,validate_logits))return false;
    if(stage_trace){
        acc_setup+=setup_us;acc_pipeline+=pipeline_us;acc_head+=tm_head.us();++acc_n;
        if(acc_n%2==0){
            std::fprintf(stderr,"[stage] n=%zu avg_us setup=%.0f pipeline=%.0f lm_head=%.0f total=%.0f\n",
                acc_n,acc_setup/acc_n,acc_pipeline/acc_n,acc_head/acc_n,
                (acc_setup+acc_pipeline+acc_head)/acc_n);
        }
    }
    if(validate_logits)return finite_values(logits.data(),logits.size());
    return true;
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
    zen::ResourceGovernor governor;const auto policy=governor.current_policy();
    if(kv_elements>std::numeric_limits<std::size_t>::max())return fail("KV cache exceeds addressable memory");
    // The KV cache only has to cover the positions this call actually visits:
    // the prompt plus one slot per generated token, never the model's full
    // advertised context. Sizing it by config_.context reserved the whole
    // 262144-token window up front (gigabytes, or a hard allocation failure)
    // before a single token ran.
    constexpr std::size_t kMaxCachePositions=1u<<17;
    std::size_t cache_positions=static_cast<std::size_t>(prompt.size())+static_cast<std::size_t>(max_new);
    const bool full_context_cache=cache_positions<=kMaxCachePositions;
    if(!full_context_cache)return fail("Prompt plus requested generation exceeds the KV cache position limit");
    std::uint64_t kv_bytes=std::uint64_t(cache_positions)*config_.layers*config_.kv_heads*config_.head_dim*2*sizeof(float);
    if(kv_bytes>policy.transient_budget_bytes)return fail("KV cache exceeds ResourceGovernor transient-memory budget");
    const auto kvsize=static_cast<std::size_t>(std::uint64_t(cache_positions)*config_.kv_heads*config_.head_dim);
    std::vector<std::vector<float>> kc,vc;std::vector<float> logits;bool cache_hit=false;
    // qwen35 hybrid: additional recurrent caches (conv windows + GDN states).
    const bool q35=config_.qwen35_variant;
    const std::size_t conv_dim=q35?(std::size_t(config_.ssm_state)*config_.ssm_n_group*2+std::size_t(config_.ssm_state)*config_.ssm_dt_rank):0;
    const std::size_t ssm_dim=q35?std::size_t(config_.ssm_dt_rank)*config_.ssm_state*config_.ssm_state:0;
    std::vector<std::vector<float>> convc,ssmc;
    using clock=std::chrono::steady_clock;auto t0=clock::now();
    {std::lock_guard<std::mutex> lock(prefix_cache_mutex_);if(prefix_cache_&&prefix_cache_->tokens==prompt){kc=prefix_cache_->key;vc=prefix_cache_->value;logits=prefix_cache_->logits;if(q35){convc=prefix_cache_->conv;ssmc=prefix_cache_->ssm;}cache_hit=true;}}
#if defined(ZEN_HAS_D3D12)
    // The device KV cache is not part of the host prefix cache. A hit restores kc/vc
    // on the host but leaves the device cache holding whatever the previous
    // generation left there, so decode would attend over keys that no longer match
    // the restored values. Re-append the prefix before decoding.
    if(cache_hit&&!q35&&!config_.use_alibi&&
       device_attention_available(config_.layers,prompt.size(),config_.kv_heads,config_.head_dim,config_.rms_epsilon)){
        // ZI-P8: on a cache hit the device cache often still holds exactly these
        // prefix positions — the previous generation appended them and nothing has
        // appended over them, reset or reconfigured the cache since. One call to
        // the backend's residency check answers that; only when the prefix is NOT
        // resident do we rebuild the four layers*|prompt|-entry vectors and issue
        // the append submission. Skipping is value-safe either way: the host kc/vc
        // this hit restored hold the same key/value rows these positions were
        // appended with, and identical (token, position) inputs produce identical
        // keys, so a resident prefix already matches what the re-append would write.
        if(!zen::d3d12::kv_cache_prefix_resident(0,prompt.size())){
            const std::size_t rKW=std::size_t(config_.kv_heads)*config_.head_dim;
            std::vector<std::size_t> rl,rp;std::vector<const float*> rk,rv;
            rl.reserve(std::size_t(config_.layers)*prompt.size());
            for(std::uint32_t l=0;l<config_.layers;++l)
                for(std::size_t t=0;t<prompt.size();++t){
                    rl.push_back(l);rp.push_back(t);
                    rk.push_back(kc[l].data()+t*rKW);rv.push_back(vc[l].data()+t*rKW);
                }
            zen::d3d12::kv_cache_append_batched(rl.data(),rl.size(),rp.data(),rk.data(),rv.data(),config_.kv_heads,config_.head_dim);
        }
    }
#endif
    if(!cache_hit){kc.assign(config_.layers,std::vector<float>(kvsize));vc.assign(config_.layers,std::vector<float>(kvsize));
        if(q35){
            // [CK-1, conv_dim] state ring + one scratch window per reachable position
            convc.assign(config_.layers,std::vector<float>(std::size_t(config_.ssm_d_conv-1+cache_positions)*conv_dim));
            ssmc.assign(config_.layers,std::vector<float>(ssm_dim));
        }
        for(std::size_t i=0;i<prompt.size();++i){if(cancelled&&cancelled->load(std::memory_order_relaxed))return fail("Generation cancelled");if(prompt[i]>=config_.vocab)return fail("Prompt token ID is outside vocabulary");zen::diag::breadcrumb(401,static_cast<std::int64_t>(i),prompt[i]);}
        {zen::diag::Scope scope("transformer.prefill.batch",zen::diag::Phase::compute);
         if(q35){if(!forward_q35_batch(prompt,kc,vc,convc,ssmc,logits))return fail("Batched prefill forward pass failed");}
         else if(!forward_batch(prompt,kc,vc,logits))return fail("Batched prefill forward pass failed");}
        if(kv_bytes<=policy.cache_budget_bytes){auto saved=std::make_shared<PrefixCache>();saved->tokens=prompt;saved->key=kc;saved->value=vc;saved->logits=logits;if(q35){saved->conv=convc;saved->ssm=ssmc;}std::lock_guard<std::mutex> lock(prefix_cache_mutex_);prefix_cache_=std::move(saved);}
    }else zen::diag::breadcrumb(403,static_cast<std::int64_t>(prompt.size()),1);
    auto t1=clock::now();result.cache_hit=cache_hit;
    result.prefill_ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    result.decode_ms=0.0;result.stopped_on_eos=false;result.stopped_on_stop=false;result.speculative_proposed=0;result.speculative_accepted=0;result.speculative_rejected=0;
    result.token_ids.clear();
    // ZI-P8: last_logits owns the current logit row from here on — the loop below
    // samples from it and swaps it against the decode buffer after every forward,
    // so the fresh logits never have to be copied into the result (was a ~513KB
    // copy per token). Content is unchanged on every path: on early breaks/failures
    // it holds exactly what the old per-iteration assignment left, and forward()
    // fully rewrites the stale buffer it receives back from the swap before anyone
    // reads it.
    result.last_logits=std::move(logits);
    std::mt19937_64 rng(options.seed?options.seed:std::random_device{}());
    for(std::uint32_t n=0;n<max_new;++n){
        if(cancelled&&cancelled->load(std::memory_order_relaxed))return fail("Generation cancelled");
        const auto next=sample_logits(result.last_logits,options,rng);result.token_ids.push_back(next);
        // ZI-P8 selected-token validation: with ZEN_VALIDATE_LOGITS unset the decode
        // forward no longer finite-scans the whole vocab row, so this is the check
        // that still stops a non-finite logit from being returned. Greedy and the
        // reference distribution only ever select finite entries, so on healthy
        // models this never fires; prefill-side scans are unchanged.
        if(!std::isfinite(result.last_logits[next]))return fail("Non-finite logit for the selected token");
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
        // Swap, not copy: result.last_logits takes the fresh row, and the stale
        // buffer handed back to `logits` is fully rewritten by the next forward
        // before any read (sampling reads result.last_logits).
        std::swap(result.last_logits,logits);
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
    // KV sizing: cover only the positions this call visits (prompt + max_new),
    // as generate() does. Sizing to the full advertised context (e.g. 131k)
    // allocated gigabytes and pushed kc[l].size()/KW past kDeviceKvTokens,
    // which disables the device attention/flash paths and forces verify onto
    // the CPU attention fallback. That made speculative verify ~14x a single
    // decode token. With visit-sized caches the device paths stay enabled and
    // the allocation matches the work.
    const std::size_t cache_positions=prompt.size()+static_cast<std::size_t>(max_new);
    auto cache_elements=[](const RuntimeConfig& c,std::uint64_t& out){
        out=c.context;for(std::uint64_t f:{std::uint64_t(c.kv_heads),std::uint64_t(c.embedding/c.heads),std::uint64_t(c.layers)}){if(f&&out>std::numeric_limits<std::uint64_t>::max()/f)return false;out*=f;}return out<=std::numeric_limits<std::uint64_t>::max()/(2*sizeof(float));};
    std::uint64_t target_elements{},draft_elements{};if(!cache_elements(config_,target_elements))return fail("Target KV cache size overflow");
    if(prompt.size()>draft.config_.context||!cache_elements(draft.config_,draft_elements))return generate(prompt,max_new,result,error,on_tokens,options,cancelled);
    zen::ResourceGovernor governor;const auto policy=governor.current_policy();
    // Budget is on the visited positions, not the full context, to match generate().
    const std::uint64_t visit_target=static_cast<std::uint64_t>(cache_positions)*config_.kv_heads*config_.head_dim*config_.layers;
    const std::uint64_t visit_draft=static_cast<std::uint64_t>(cache_positions)*draft.config_.kv_heads*draft.config_.head_dim*draft.config_.layers;
    if(visit_target>std::numeric_limits<std::uint64_t>::max()-visit_draft||visit_target+visit_draft>std::numeric_limits<std::uint64_t>::max()/(2*sizeof(float)))return generate(prompt,max_new,result,error,on_tokens,options,cancelled);
    const std::uint64_t combined_bytes=(visit_target+visit_draft)*2*sizeof(float);
    if(combined_bytes>policy.transient_budget_bytes)return generate(prompt,max_new,result,error,on_tokens,options,cancelled);
    const std::size_t target_kvsize=cache_positions*config_.kv_heads*config_.head_dim;
    const std::size_t draft_kvsize=cache_positions*draft.config_.kv_heads*draft.config_.head_dim;
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

} // namespace zen::runtime
