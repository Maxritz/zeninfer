#include "zen/openai_server.h"
#include "zen/diagnostics.h"
#include "zen/resource_governor.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <variant>
#include <cctype>
#if defined(_WIN32)
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
using sock_t=SOCKET; static constexpr sock_t bad=INVALID_SOCKET; static void close_sock(sock_t s){closesocket(s);}
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
using sock_t=int; static constexpr sock_t bad=-1; static void close_sock(sock_t s){close(s);}
#endif
namespace zen::http { namespace {
struct J{using A=std::vector<J>;using O=std::map<std::string,J>;std::variant<std::nullptr_t,bool,double,std::string,A,O>v{nullptr};const J*get(const std::string&k)const{auto*o=std::get_if<O>(&v);if(!o)return nullptr;auto i=o->find(k);return i==o->end()?nullptr:&i->second;}std::string str(std::string d={})const{auto*p=std::get_if<std::string>(&v);return p?*p:d;}double num(double d)const{auto*p=std::get_if<double>(&v);return p?*p:d;}bool boolean()const{auto*p=std::get_if<bool>(&v);return p&&*p;}};
class Parser {
    const std::string& s;
    std::size_t p{};
    void ws(){while(p<s.size()&&std::isspace(static_cast<unsigned char>(s[p])))++p;}
    static void utf8(std::string& out,std::uint32_t cp){
        if(cp<=0x7f)out+=char(cp);
        else if(cp<=0x7ff){out+=char(0xc0|(cp>>6));out+=char(0x80|(cp&0x3f));}
        else if(cp<=0xffff){out+=char(0xe0|(cp>>12));out+=char(0x80|((cp>>6)&0x3f));out+=char(0x80|(cp&0x3f));}
        else{out+=char(0xf0|(cp>>18));out+=char(0x80|((cp>>12)&0x3f));out+=char(0x80|((cp>>6)&0x3f));out+=char(0x80|(cp&0x3f));}
    }
    bool hex4(std::uint32_t& cp){
        if(p+4>s.size()) return false;
        cp=0;
        for(int i=0;i<4;++i){unsigned char h=s[p++];cp<<=4;if(h>='0'&&h<='9')cp|=h-'0';else if(h>='a'&&h<='f')cp|=h-'a'+10;else if(h>='A'&&h<='F')cp|=h-'A'+10;else return false;}
        return true;
    }
    bool str(std::string& out){
        if(p>=s.size()||s[p++]!='"')return false;
        while(p<s.size()){
            unsigned char c=s[p++];if(c=='"')return true;
            if(c=='\\'){
                if(p>=s.size()) return false;
                char e=s[p++];
                switch(e){
                    case '"':case '\\':case '/':out+=e;break;
                    case 'b':out+='\b';break;case 'f':out+='\f';break;case 'n':out+='\n';break;
                    case 'r':out+='\r';break;case 't':out+='\t';break;
                    case 'u':{
                        std::uint32_t cp;if(!hex4(cp))return false;
                        if(cp>=0xd800&&cp<=0xdbff){
                            if(p+2>s.size()||s[p++]!='\\'||s[p++]!='u')return false;
                            std::uint32_t low;if(!hex4(low)||low<0xdc00||low>0xdfff)return false;
                            cp=0x10000+((cp-0xd800)<<10)+(low-0xdc00);
                        }else if(cp>=0xdc00&&cp<=0xdfff)return false;
                        utf8(out,cp);break;
                    }
                    default:return false;
                }
            }else{if(c<0x20)return false;out+=char(c);}
        }
        return false;
    }
    bool val(J& x,unsigned depth=0){
        // Nesting bound: each object/array level costs a stack frame, and the
        // request body is untrusted. A few hundred kB of '[' once exhausted the
        // thread stack and killed the server process (verified 0xC00000FD), so
        // recursion is capped well below any legitimate request's needs.
        if(depth>128)return false;
        ws();if(p>=s.size())return false;
        if(s[p]=='"'){std::string z;if(!str(z))return false;x.v=std::move(z);return true;}
        if(s[p]=='{'){
            ++p;J::O o;ws();if(p<s.size()&&s[p]=='}'){++p;x.v=std::move(o);return true;}
            while(p<s.size()){ws();std::string k;if(!str(k))return false;ws();if(p>=s.size()||s[p++]!=':')return false;J v;if(!val(v,depth+1))return false;o.emplace(k,std::move(v));ws();if(p<s.size()&&s[p]=='}'){++p;x.v=std::move(o);return true;}if(p>=s.size()||s[p++]!=',')return false;}
            return false;
        }
        if(s[p]=='['){
            ++p;J::A a;ws();if(p<s.size()&&s[p]==']'){++p;x.v=std::move(a);return true;}
            while(p<s.size()){J v;if(!val(v,depth+1))return false;a.push_back(std::move(v));ws();if(p<s.size()&&s[p]==']'){++p;x.v=std::move(a);return true;}if(p>=s.size()||s[p++]!=',')return false;}
            return false;
        }
        if(s.compare(p,4,"true")==0){p+=4;x.v=true;return true;}
        if(s.compare(p,5,"false")==0){p+=5;x.v=false;return true;}
        if(s.compare(p,4,"null")==0){p+=4;return true;}
        const auto begin=p;if(s[p]=='-')++p;while(p<s.size()&&(std::isdigit(static_cast<unsigned char>(s[p]))||s[p]=='.'||s[p]=='e'||s[p]=='E'||s[p]=='+'||s[p]=='-'))++p;
        if(begin==p) return false;
        try {
            std::size_t used=0;
            double d=std::stod(s.substr(begin,p-begin),&used);
            if(used!=p-begin) return false;
            x.v=d;return true;
        } catch(...) { return false; }
    }
public:
    explicit Parser(const std::string& x):s(x){}
    bool parse(J& x){if(!val(x))return false;ws();return p==s.size();}
};
std::string esc(const std::string&s){static const char*h="0123456789abcdef";std::string o;for(unsigned char c:s){switch(c){case'"':o+="\\\"";break;case'\\':o+="\\\\";break;case'\b':o+="\\b";break;case'\f':o+="\\f";break;case'\n':o+="\\n";break;case'\r':o+="\\r";break;case'\t':o+="\\t";break;default:if(c<0x20){o+="\\u00";o+=h[c>>4];o+=h[c&15];}else o+=char(c);}}return o;}
std::string err(const std::string&m,const std::string&code="invalid_request"){return "{\"error\":{\"message\":\""+esc(m)+"\",\"type\":\"invalid_request_error\",\"param\":null,\"code\":\""+code+"\"}}";}
bool sendall(sock_t s,const std::string&x){size_t p=0;while(p<x.size()){
#if defined(_WIN32)
int n=send(s,x.data()+p,int(x.size()-p),0);
#else
auto n=send(s,x.data()+p,x.size()-p,MSG_NOSIGNAL);
#endif
if(n<=0) return false;
p+=size_t(n);
}return true;}
struct Req{std::string method,path,auth,body;};
bool receive(sock_t s,Req&r,size_t max){std::string d;char b[8192];size_t end=std::string::npos;while((end=d.find("\r\n\r\n"))==std::string::npos){
#if defined(_WIN32)
int n=recv(s,b,sizeof(b),0);
#else
auto n=recv(s,b,sizeof(b),0);
#endif
if(n<=0) return false;
d.append(b,size_t(n));
if(d.size()>65536) return false;
}std::istringstream h(d.substr(0,end));h>>r.method>>r.path;std::string line;std::getline(h,line);size_t len=0;while(std::getline(h,line)){auto c=line.find(':');if(c==std::string::npos)continue;std::string k=line.substr(0,c),v=line.substr(c+1);for(char&x:k)x=char(std::tolower((unsigned char)x));while(!v.empty()&&std::isspace((unsigned char)v.front()))v.erase(v.begin());while(!v.empty()&&std::isspace((unsigned char)v.back()))v.pop_back();if(k=="content-length")try{len=std::stoull(v);}catch(...){return false;}if(k=="authorization")r.auth=v;}if(len>max)return false;r.body=d.substr(end+4);while(r.body.size()<len){
#if defined(_WIN32)
int n=recv(s,b,sizeof(b),0);
#else
auto n=recv(s,b,sizeof(b),0);
#endif
if(n<=0) return false;
r.body.append(b,size_t(n));
if(r.body.size()>max) return false;
}r.body.resize(len);return true;}
void response(sock_t s,int code,const std::string&body,const char*type="application/json"){const char*reason=code==200?"OK":code==400?"Bad Request":code==401?"Unauthorized":code==404?"Not Found":code==413?"Payload Too Large":code==503?"Service Unavailable":"Internal Server Error";std::ostringstream h;h<<"HTTP/1.1 "<<code<<' '<<reason<<"\r\nContent-Type: "<<type<<"\r\nContent-Length: "<<body.size()<<"\r\nConnection: close\r\n\r\n";sendall(s,h.str());sendall(s,body);}
void serve(sock_t s,const ServerConfig&cfg,InferenceBackend*backend){zen::diag::Scope scope("http.request",zen::diag::Phase::host);zen::diag::breadcrumb(100,1);Req r;if(!receive(s,r,cfg.max_request_bytes)){response(s,413,err("Malformed request or request body too large."));return;}if(!cfg.api_key.empty()&&r.auth!="Bearer "+cfg.api_key){response(s,401,err("Invalid bearer token.","invalid_api_key"));return;}if(r.method=="GET"&&r.path=="/health"){response(s,200,std::string("{\"status\":\"ok\",\"engine_loaded\":")+(backend?"true":"false")+"}");return;}if(r.method=="GET"&&r.path=="/v1/models"){std::string id=backend?backend->model_id():"zeninfer-unloaded";response(s,200,"{\"object\":\"list\",\"data\":[{\"id\":\""+esc(id)+"\",\"object\":\"model\",\"created\":0,\"owned_by\":\"zeninfer\"}]}");return;}if(r.method=="GET"&&r.path=="/metrics"){ResourceGovernor rg;auto snap=rg.snapshot();auto pol=rg.policy(snap);std::ostringstream m;m<<"zeninfer_logical_cpus "<<snap.logical_cpus<<"\nzeninfer_physical_cpus "<<snap.physical_cpus<<"\nzeninfer_ram_available_bytes "<<snap.available_ram_bytes<<"\nzeninfer_process_memory_limit_bytes "<<snap.process_memory_limit_bytes<<"\nzeninfer_policy_worker_threads "<<pol.worker_threads<<"\nzeninfer_policy_cache_budget_bytes "<<pol.cache_budget_bytes<<"\nzeninfer_policy_transient_budget_bytes "<<pol.transient_budget_bytes<<"\nzeninfer_policy_prefetch_workers "<<pol.prefetch_workers<<"\n";response(s,200,m.str(),"text/plain; version=0.0.4");return;}if(r.method=="GET"&&r.path=="/debug/sherlock"){std::ostringstream o;zen::diag::print_trace(o);response(s,200,o.str(),"text/plain; charset=utf-8");return;}if(r.method=="GET"&&r.path=="/debug/breadcrumbs"){std::ostringstream o;zen::diag::dump_breadcrumbs(o,512);response(s,200,o.str(),"text/plain; charset=utf-8");return;}bool chat=r.path=="/v1/chat/completions",completion=r.path=="/v1/completions";if(r.method!="POST"||(!chat&&!completion)){response(s,404,err("Unknown endpoint.","not_found"));return;}J root;{zen::diag::Scope ps("http.json_parse",zen::diag::Phase::host);Parser p(r.body);if(!p.parse(root)){response(s,400,err("Invalid JSON body."));return;}}GenerationRequest g;auto*m=root.get("model");if(!m||!std::holds_alternative<std::string>(m->v)){response(s,400,err("model must be a string."));return;}g.model=m->str();auto*st=root.get("stream");if(st&&!std::holds_alternative<bool>(st->v)){response(s,400,err("stream must be a boolean."));return;}bool streaming=st&&st->boolean();auto*mt=root.get("max_tokens");if(!mt)mt=root.get("max_completion_tokens");if(mt){if(!std::holds_alternative<double>(mt->v)||!std::isfinite(mt->num(0))||mt->num(0)<1||mt->num(0)>65536||std::floor(mt->num(0))!=mt->num(0)){response(s,400,err("max_tokens must be an integer from 1 to 65536."));return;}g.max_tokens=static_cast<std::uint32_t>(mt->num(0));}auto*t=root.get("temperature");if(t){if(!std::holds_alternative<double>(t->v)||!std::isfinite(t->num(0))||t->num(0)<0||t->num(0)>5){response(s,400,err("temperature must be a finite number from 0 to 5."));return;}g.temperature=t->num(0);}auto*tk=root.get("top_k");if(tk){if(!std::holds_alternative<double>(tk->v)||tk->num(0)<0||tk->num(0)>32000||std::floor(tk->num(0))!=tk->num(0)){response(s,400,err("top_k must be an integer from 0 to 32000."));return;}g.top_k=static_cast<std::uint32_t>(tk->num(0));}auto*tp=root.get("top_p");if(tp){if(!std::holds_alternative<double>(tp->v)||!std::isfinite(tp->num(0))||tp->num(0)<=0||tp->num(0)>1){response(s,400,err("top_p must be in (0, 1]."));return;}g.top_p=tp->num(0);}auto*spec=root.get("speculative_tokens");if(spec){if(!std::holds_alternative<double>(spec->v)||spec->num(0)<0||spec->num(0)>16||std::floor(spec->num(0))!=spec->num(0)){response(s,400,err("speculative_tokens must be an integer from 0 to 16."));return;}g.speculative_tokens=static_cast<std::uint32_t>(spec->num(0));}auto*seed=root.get("seed");if(seed){if(!std::holds_alternative<double>(seed->v)||seed->num(0)<0||seed->num(0)>9007199254740991.0||std::floor(seed->num(0))!=seed->num(0)){response(s,400,err("seed must be a non-negative safe integer."));return;}g.seed=static_cast<std::uint64_t>(seed->num(0));}auto*stop=root.get("stop");if(stop){if(auto*ss=std::get_if<std::string>(&stop->v)){if(ss->empty()){response(s,400,err("stop strings must be non-empty."));return;}g.stop.push_back(*ss);}else if(auto*sa=std::get_if<J::A>(&stop->v)){if(sa->size()>4){response(s,400,err("stop accepts at most four strings."));return;}for(const auto&v:*sa){if(!std::holds_alternative<std::string>(v.v)||v.str().empty()){response(s,400,err("stop must contain non-empty strings."));return;}g.stop.push_back(v.str());}}else{response(s,400,err("stop must be a string or an array of strings."));return;}}auto*lb=root.get("logit_bias");if(lb){auto*obj=std::get_if<J::O>(&lb->v);if(!obj||obj->size()>256){response(s,400,err("logit_bias must be an object with at most 256 token entries."));return;}for(const auto&[key,value]:*obj){std::uint32_t id{};try{std::size_t used=0;const auto parsed=std::stoul(key,&used);if(used!=key.size()||parsed>UINT32_MAX)throw std::runtime_error("id");id=static_cast<std::uint32_t>(parsed);}catch(...){response(s,400,err("logit_bias keys must be token IDs."));return;}if(!std::holds_alternative<double>(value.v)||!std::isfinite(value.num(0))||std::abs(value.num(0))>100){response(s,400,err("logit_bias values must be finite numbers in [-100,100]."));return;}g.logit_bias[id]=static_cast<float>(value.num(0));}}auto*allowed=root.get("allowed_token_ids");if(allowed){auto*arr=std::get_if<J::A>(&allowed->v);if(!arr||arr->empty()||arr->size()>4096){response(s,400,err("allowed_token_ids must be a non-empty array of at most 4096 IDs."));return;}for(const auto&v:*arr){if(!std::holds_alternative<double>(v.v)||v.num(0)<0||v.num(0)>4294967295.0||std::floor(v.num(0))!=v.num(0)){response(s,400,err("allowed_token_ids must contain non-negative integer IDs."));return;}g.allowed_token_ids.push_back(static_cast<std::uint32_t>(v.num(0)));}}if(g.model.empty()){response(s,400,err("A model field is required."));return;}if(chat){auto*ms=root.get("messages");auto*a=ms?std::get_if<J::A>(&ms->v):nullptr;if(!a||a->empty()||a->size()>1024){response(s,400,err("messages must be a non-empty array with at most 1024 entries."));return;}for(auto&x:*a){auto*role=x.get("role");auto*content=x.get("content");if(!role||!content||!std::holds_alternative<std::string>(role->v)||!std::holds_alternative<std::string>(content->v)){response(s,400,err("Each message requires string role and content fields."));return;}const auto role_name=role->str();if(role_name!="system"&&role_name!="user"&&role_name!="assistant"){response(s,400,err("Message role must be system, user or assistant."));return;}if(content->str().size()>1024*1024){response(s,400,err("Message content exceeds 1 MiB."));return;}g.messages.push_back({role_name,content->str()});}}else{auto*p=root.get("prompt");if(!p||!std::holds_alternative<std::string>(p->v)||p->str().empty()){response(s,400,err("prompt must be a non-empty string."));return;}if(p->str().size()>2*1024*1024){response(s,400,err("prompt exceeds 2 MiB."));return;}g.prompt=p->str();}if(!backend){response(s,503,"{\"error\":{\"message\":\"No GGUF inference backend is loaded.\",\"type\":\"server_error\",\"param\":null,\"code\":\"model_not_loaded\"}}");return;}if(g.model!="*"&&g.model!=backend->model_id()){response(s,404,err("Model not found.","model_not_found"));return;}
static std::atomic<unsigned long long> seq{1};std::string id="chatcmpl-zen-"+std::to_string(seq++);auto now=std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());std::atomic<bool>cancel{false};std::string accumulated;
if(streaming){sendall(s,"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n");std::string model=esc(backend->model_id());auto chunk=[&](const std::string&piece,const char*finish){const std::string obj=chat?"chat.completion.chunk":"text_completion";const std::string field=chat?"\"delta\":"+piece:"\"text\":\""+piece+"\"";return "{\"id\":\""+id+"\",\"object\":\""+obj+"\",\"created\":"+std::to_string(now)+",\"model\":\""+model+"\",\"choices\":[{\"index\":0,"+field+",\"finish_reason\":"+(finish?std::string("\"")+finish+"\"":"null")+"}]}";};if(chat)sendall(s,"data: "+chunk("{\"role\":\"assistant\"}",nullptr)+"\n\n");try{GenerationResult out;{zen::diag::Scope gs("inference.generate",zen::diag::Phase::compute);out=backend->generate(g,[&](const std::string&tok){accumulated+=tok;if(!sendall(s,"data: "+chunk(chat?"{\"content\":\""+esc(tok)+"\"}":esc(tok),nullptr)+"\n\n"))cancel=true;},cancel);}sendall(s,"data: "+chunk("",out.finish_reason.c_str())+"\n\ndata: [DONE]\n\n");}catch(...){sendall(s,"data: "+err("Generation failed.","generation_error")+"\n\n");}}
else{try{GenerationResult out;{zen::diag::Scope gs("inference.generate",zen::diag::Phase::compute);out=backend->generate(g,[&](const std::string&tok){accumulated+=tok;},cancel);}if(out.text.empty())out.text=accumulated;std::string usage="{\"prompt_tokens\":"+std::to_string(out.prompt_tokens)+",\"completion_tokens\":"+std::to_string(out.completion_tokens)+",\"total_tokens\":"+std::to_string(out.prompt_tokens+out.completion_tokens)+"}";std::string body;if(chat)body="{\"id\":\""+id+"\",\"object\":\"chat.completion\",\"created\":"+std::to_string(now)+",\"model\":\""+esc(backend->model_id())+"\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\""+esc(out.text)+"\"},\"finish_reason\":\""+esc(out.finish_reason)+"\"}],\"usage\":"+usage+"}";else body="{\"id\":\""+id+"\",\"object\":\"text_completion\",\"created\":"+std::to_string(now)+",\"model\":\""+esc(backend->model_id())+"\",\"choices\":[{\"index\":0,\"text\":\""+esc(out.text)+"\",\"finish_reason\":\""+esc(out.finish_reason)+"\"}],\"usage\":"+usage+"}";response(s,200,body);}catch(...){response(s,500,err("Generation failed.","generation_error"));}}zen::diag::breadcrumb(100,2,int64_t(accumulated.size()));}
}
OpenAIServer::OpenAIServer(ServerConfig c,InferenceBackend*b):config_(std::move(c)),backend_(b){}OpenAIServer::~OpenAIServer(){stop();}
void OpenAIServer::stop()noexcept{stopping_=true;std::intptr_t x=listen_socket_.load();if(x>=0){auto s=sock_t(x);
#if defined(_WIN32)
shutdown(s,SD_BOTH);
#else
shutdown(s,SHUT_RDWR);
#endif
}}
bool OpenAIServer::run(){
#if defined(_WIN32)
WSADATA w{};if(WSAStartup(MAKEWORD(2,2),&w))return false;
#endif
stopping_=false;sock_t srv=socket(AF_INET,SOCK_STREAM,IPPROTO_TCP);if(srv==bad)return false;int yes=1;
#if defined(_WIN32)
setsockopt(srv,SOL_SOCKET,SO_REUSEADDR,(const char*)&yes,sizeof(yes));
#else
setsockopt(srv,SOL_SOCKET,SO_REUSEADDR,&yes,sizeof(yes));
#endif
sockaddr_in a{};a.sin_family=AF_INET;a.sin_port=htons(config_.port);if(inet_pton(AF_INET,config_.host.c_str(),&a.sin_addr)!=1||bind(srv,(sockaddr*)&a,sizeof(a))<0||listen(srv,64)<0){close_sock(srv);return false;}listen_socket_=static_cast<std::intptr_t>(srv);ResourceGovernor rg;std::mutex m;std::condition_variable cv;unsigned active=0;
struct Worker { std::thread thread; std::shared_ptr<std::atomic<bool>> done; };
std::vector<Worker> workers;
while(!stopping_){sockaddr_in peer{};
#if defined(_WIN32)
int len=sizeof(peer);
#else
socklen_t len=sizeof(peer);
#endif
sock_t c=accept(srv,(sockaddr*)&peer,&len);if(c==bad){if(stopping_)break;continue;}
 const unsigned cap=config_.max_connections?config_.max_connections:std::max(1u,rg.current_policy().worker_threads);
 {std::unique_lock l(m);cv.wait(l,[&]{return active<cap||stopping_;});if(stopping_){close_sock(c);break;}++active;}
 for(auto it=workers.begin();it!=workers.end();){if(it->done->load(std::memory_order_acquire)){if(it->thread.joinable())it->thread.join();it=workers.erase(it);}else ++it;}
 auto done=std::make_shared<std::atomic<bool>>(false);
 workers.push_back(Worker{std::thread([&,c,done]{serve(c,config_,backend_);close_sock(c);{std::lock_guard l(m);--active;cv.notify_one();}done->store(true,std::memory_order_release);}),done});}
 {std::unique_lock l(m);cv.wait(l,[&]{return active==0;});}
 for(auto&w:workers)if(w.thread.joinable())w.thread.join();
 auto owned=listen_socket_.exchange(-1);if(owned>=0)close_sock(sock_t(owned));
#if defined(_WIN32)
WSACleanup();
#endif
return true;
}
}
