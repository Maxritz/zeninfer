#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Definition {
    std::string name, architecture, fixture_kind, base_architecture, weight_format;
    std::size_t hidden=0, vocab=0, layers=0, experts=0, top_k=0, group_size=0;
    std::uint32_t seed=1, prefill_tokens=32, decode_tokens=32;
    double expected_checksum=0.0; bool has_expected_checksum=false;
    bool hadamard=false;
};
Definition load_definition(const std::string& path) {
    std::ifstream in(path); if(!in) throw std::runtime_error("cannot open model definition: "+path);
    std::string line; if(!std::getline(in,line)||line!="ZENMODEL 1") throw std::runtime_error("invalid ZENMODEL header: "+path);
    std::map<std::string,std::string> kv;
    while(std::getline(in,line)){ if(line.empty()||line[0]=='#') continue; auto p=line.find('='); if(p==std::string::npos) throw std::runtime_error("invalid definition line: "+line); kv[line.substr(0,p)]=line.substr(p+1); }
    auto s=[&](const char*k,const char*d=""){auto i=kv.find(k);return i==kv.end()?std::string(d):i->second;};
    auto u=[&](const char*k,std::uint32_t d=0){auto v=s(k);return v.empty()?d:static_cast<std::uint32_t>(std::stoul(v));};
    Definition d; d.name=s("name");d.architecture=s("architecture");d.fixture_kind=s("fixture_kind");d.base_architecture=s("base_architecture");d.weight_format=s("weight_format");
    d.hidden=u("hidden");d.vocab=u("vocab");d.layers=u("layers");d.experts=u("experts");d.top_k=u("top_k");d.group_size=u("group_size");d.seed=u("seed",1);d.prefill_tokens=u("prefill_tokens",32);d.decode_tokens=u("decode_tokens",32);d.hadamard=s("hadamard_transform")=="true";if(!s("expected_checksum").empty()){d.expected_checksum=std::stod(s("expected_checksum"));d.has_expected_checksum=true;}
    if(d.fixture_kind!="architecture_kernel_micro_model") throw std::runtime_error("refusing non-micro model definition");
    if(!d.hidden||!d.vocab||!d.layers||!d.group_size) throw std::runtime_error("model dimensions must be nonzero");
    if(d.architecture=="qwen3_moe" && (!d.experts||!d.top_k||d.top_k>d.experts)) throw std::runtime_error("invalid Qwen3 MoE expert configuration");
    if(d.architecture!="qwen2_dense"&&d.architecture!="qwen3_dense"&&d.architecture!="qwen3_moe"&&d.architecture!="bitnet_ternary"&&d.architecture!="bonsai_ternary_qwen3") throw std::runtime_error("unsupported micro architecture: "+d.architecture);
    if(d.hidden>4096||d.vocab>65536||d.layers>64||d.experts>256) throw std::runtime_error("micro model dimensions exceed safety limits");
    return d;
}
struct Rng { std::uint64_t s; explicit Rng(std::uint64_t seed):s(seed?seed:1){} std::uint32_t next(){s^=s<<13;s^=s>>7;s^=s<<17;return static_cast<std::uint32_t>(s>>16);} float f(){return (float(next()%20001)-10000.0f)/10000.0f*0.12f;} int trit(){return int(next()%3)-1;} };
struct Matrix { std::size_t rows=0,cols=0,group=128;bool tq2_layout=false,allow_plus2=false;std::vector<float> dense,scales;std::vector<std::uint8_t> packed2; };
Matrix dense_matrix(Rng&r,std::size_t rows,std::size_t cols){Matrix m;m.rows=rows;m.cols=cols;m.dense.resize(rows*cols);for(auto&v:m.dense)v=r.f();return m;}
Matrix ternary_matrix(Rng&r,std::size_t rows,std::size_t cols,std::size_t group,bool tq2_layout=false,bool allow_plus2=false){Matrix m;m.rows=rows;m.cols=cols;m.group=group;m.tq2_layout=tq2_layout;m.allow_plus2=allow_plus2;const auto count=rows*cols;m.packed2.assign((count+3)/4,0);const auto n=(count+group-1)/group;m.scales.resize(n);for(auto&s:m.scales)s=0.035f+float(r.next()%1000)/100000.0f;for(std::size_t i=0;i<count;++i){const int t=r.trit();const unsigned code=t<0?0u:(t==0?1u:2u);std::size_t byte=0,shift=0;if(tq2_layout){const auto block=i/256,local=i%256;byte=block*64+(local/128)*32+(local%32);shift=((local%128)/32)*2;}else{byte=i/4;shift=(i%4)*2;}m.packed2[byte]|=static_cast<std::uint8_t>(code<<shift);}return m;}
void matvec(const Matrix&m,const std::vector<float>&x,std::vector<float>&y){if(x.size()!=m.cols)throw std::runtime_error("matvec shape mismatch");y.assign(m.rows,0.0f);for(std::size_t r=0;r<m.rows;++r){double a=0;for(std::size_t c=0;c<m.cols;++c)a+=double(m.dense[r*m.cols+c])*x[c];y[r]=float(a);}}
void ternary_matvec(const Matrix&m,const std::vector<float>&x,std::vector<float>&y){if(x.size()!=m.cols)throw std::runtime_error("ternary matvec shape mismatch");y.assign(m.rows,0.0f);for(std::size_t r=0;r<m.rows;++r){double a=0;for(std::size_t c=0;c<m.cols;++c){const auto i=r*m.cols+c;std::size_t byte=0,shift=0;if(m.tq2_layout){const auto block=i/256,local=i%256;byte=block*64+(local/128)*32+(local%32);shift=((local%128)/32)*2;}else{byte=i/4;shift=(i%4)*2;}const unsigned code=(m.packed2[byte]>>shift)&3u;if(code==3u&&!m.allow_plus2)throw std::runtime_error("reserved ternary Q2_0 code encountered");const int q=code==0u?-1:(code==1u?0:(code==2u?1:2));a+=double(q)*m.scales[i/m.group]*x[c];}y[r]=float(a);}}
void hadamard(std::vector<float>&x){for(std::size_t stride=1;stride<x.size();stride*=2)for(std::size_t i=0;i<x.size();i+=2*stride)for(std::size_t j=0;j<stride;++j){const float a=x[i+j],b=x[i+j+stride];x[i+j]=a+b;x[i+j+stride]=a-b;}const float s=1.0f/std::sqrt(float(x.size()));for(auto&v:x)v*=s;}
struct Expert { Matrix gate,up,down; };
class MicroModel {
public:
 explicit MicroModel(Definition def):d(std::move(def)),rng(d.seed){
   embedding.resize(d.vocab*d.hidden);for(auto&v:embedding)v=rng.f();
   output=(d.architecture=="qwen3_moe"||d.architecture=="qwen2_dense"||d.architecture=="qwen3_dense")?dense_matrix(rng,d.vocab,d.hidden):ternary_matrix(rng,d.vocab,d.hidden,d.group_size,d.weight_format=="tq2_0",d.weight_format=="tq2_0");
   if(d.architecture=="qwen3_moe"){
     router=dense_matrix(rng,d.experts,d.hidden);experts.resize(d.experts);expert_usage.assign(d.experts,0);
     for(auto&e:experts){e.gate=dense_matrix(rng,d.hidden,d.hidden);e.up=dense_matrix(rng,d.hidden,d.hidden);e.down=dense_matrix(rng,d.hidden,d.hidden);}
   } else if(d.architecture=="qwen2_dense"||d.architecture=="qwen3_dense") {
     for(std::size_t i=0;i<d.layers;++i) layers.push_back(dense_matrix(rng,d.hidden,d.hidden));
   } else {
     for(std::size_t i=0;i<d.layers;++i) layers.push_back(ternary_matrix(rng,d.hidden,d.hidden,d.group_size,d.weight_format=="tq2_0",d.weight_format=="tq2_0"));
   }
 }
 std::vector<float> step(std::uint32_t token){
   token%=static_cast<std::uint32_t>(d.vocab);std::vector<float>x(d.hidden);for(std::size_t i=0;i<d.hidden;++i)x[i]=embedding[token*d.hidden+i]+state[i]*0.08f;
   if(d.architecture=="qwen3_moe"){
     std::vector<float> scores;matvec(router,x,scores);std::vector<std::size_t> order(d.experts);std::iota(order.begin(),order.end(),0);std::partial_sort(order.begin(),order.begin()+d.top_k,order.end(),[&](auto a,auto b){return scores[a]>scores[b];});
     float maxv=scores[order[0]],den=0;for(std::size_t j=0;j<d.top_k;++j)den+=std::exp(scores[order[j]]-maxv);std::vector<float>combined(d.hidden,0.0f);
     for(std::size_t j=0;j<d.top_k;++j){const auto id=order[j];++expert_usage[id];std::vector<float>g,u,h,z;matvec(experts[id].gate,x,g);matvec(experts[id].up,x,u);for(std::size_t k=0;k<d.hidden;++k)g[k]=(g[k]/(1.0f+std::exp(-g[k])))*u[k];matvec(experts[id].down,g,h);const float w=std::exp(scores[id]-maxv)/den;for(std::size_t k=0;k<d.hidden;++k)combined[k]+=w*h[k];}
     for(std::size_t k=0;k<d.hidden;++k)x[k]=std::tanh(x[k]+combined[k]);
   } else if(d.architecture=="qwen2_dense"||d.architecture=="qwen3_dense") {
     for(const auto&m:layers){std::vector<float>y;matvec(m,x,y);for(std::size_t k=0;k<d.hidden;++k)x[k]=std::tanh(x[k]+y[k]);}
   } else {
     for(const auto&m:layers){std::vector<float>input=x,y;if(d.hadamard)hadamard(input);ternary_matvec(m,input,y);for(std::size_t k=0;k<d.hidden;++k)x[k]=std::tanh(x[k]+y[k]);}
   }
   state=x;std::vector<float>logits;if(d.architecture=="qwen3_moe"||d.architecture=="qwen2_dense"||d.architecture=="qwen3_dense")matvec(output,x,logits);else ternary_matvec(output,x,logits);return logits;
 }
 std::uint32_t next(const std::vector<float>&v)const{return static_cast<std::uint32_t>(std::max_element(v.begin(),v.end())-v.begin());}
 const Definition& def()const{return d;}
 std::uint64_t expert_selections()const{return std::accumulate(expert_usage.begin(),expert_usage.end(),std::uint64_t(0));}
 std::size_t active_experts()const{return static_cast<std::size_t>(std::count_if(expert_usage.begin(),expert_usage.end(),[](auto n){return n>0;}));}
private: Definition d;Rng rng;std::vector<float>embedding,state=std::vector<float>(d.hidden,0.0f);Matrix output,router;std::vector<Expert>experts;std::vector<Matrix>layers;std::vector<std::uint64_t>expert_usage;
};
double rate(std::size_t n,double ms){return ms>0.0?double(n)*1000.0/ms:0.0;}
}
int main(int argc,char**argv){try{if(argc<2){std::cerr<<"usage: zen_arch_microbench model.zenmodel [model.zenmodel ...]\n";return 2;}std::cout<<std::fixed<<std::setprecision(2);
 for(int a=1;a<argc;++a){std::cout<<std::setprecision(2);auto d=load_definition(argv[a]);MicroModel model(d);std::vector<std::uint32_t>prompt(d.prefill_tokens);for(std::size_t i=0;i<prompt.size();++i)prompt[i]=static_cast<std::uint32_t>((i*17+3)%d.vocab);
   for(int i=0;i<2;++i){for(auto t:prompt)(void)model.step(t);}
   const auto p0=std::chrono::steady_clock::now();std::vector<float>logits;for(auto t:prompt)logits=model.step(t);const auto p1=std::chrono::steady_clock::now();const double prefill_ms=std::chrono::duration<double,std::milli>(p1-p0).count();
   std::uint32_t token=model.next(logits);const auto g0=std::chrono::steady_clock::now();for(std::uint32_t i=0;i<d.decode_tokens;++i){logits=model.step(token);token=model.next(logits);}const auto g1=std::chrono::steady_clock::now();const double decode_ms=std::chrono::duration<double,std::milli>(g1-g0).count();
   if(logits.empty()||!std::all_of(logits.begin(),logits.end(),[](float v){return std::isfinite(v);}))throw std::runtime_error("non-finite or empty logits");
   std::cout<<d.name<<" architecture="<<d.architecture<<" hidden="<<d.hidden<<" layers="<<d.layers<<" experts="<<d.experts<<" top_k="<<d.top_k<<" group="<<d.group_size<<"\n";
   std::cout<<"  prefill: "<<d.prefill_tokens<<" tokens / "<<prefill_ms<<" ms = "<<rate(d.prefill_tokens,prefill_ms)<<" tok/s\n";
   std::cout<<"  tokengen: "<<d.decode_tokens<<" tokens / "<<decode_ms<<" ms = "<<rate(d.decode_tokens,decode_ms)<<" tok/s\n";
   double checksum=0;for(std::size_t i=0;i<logits.size();++i)checksum+=double(i+1)*logits[i];const bool golden_ok=!d.has_expected_checksum||std::abs(checksum-d.expected_checksum)<=1.0e-4*std::max(1.0,std::abs(d.expected_checksum));
   if(!golden_ok)throw std::runtime_error("golden checksum mismatch for "+d.name);
   std::cout<<"  logits_checksum="<<std::setprecision(8)<<checksum;
   if(d.architecture=="qwen3_moe")std::cout<<" expert_selections="<<model.expert_selections()<<" active_experts="<<model.active_experts();
   std::cout<<" validation=PASS fixture=synthetic-kernel-only\n";
 }
 return 0;}catch(const std::exception&e){std::cerr<<"architecture microbench failed: "<<e.what()<<"\n";return 1;}}
