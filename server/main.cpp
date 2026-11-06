#include "zen/openai_server.h"
#include "zen/model_backend.h"
#include "zen/resource_governor.h"
#include "zen/diagnostics.h"
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
int main(int argc,char**argv){
    zen::http::ServerConfig config;
    if(const char* h=std::getenv("ZEN_HOST"))config.host=h;
    if(const char* p=std::getenv("ZEN_PORT"))config.port=std::uint16_t(std::atoi(p));
    if(const char* k=std::getenv("ZEN_API_KEY"))config.api_key=k;
    zen::diag::set_breadcrumbs_enabled(true);
    std::string model_path,model_id,draft_model_path;
    for(int i=1;i<argc;++i){
        const std::string arg=argv[i];
        if(arg=="--model"&&i+1<argc)model_path=argv[++i];
        else if(arg=="--model-id"&&i+1<argc)model_id=argv[++i];
        else if(arg=="--draft-model"&&i+1<argc)draft_model_path=argv[++i];
        else if(arg=="--host"&&i+1<argc)config.host=argv[++i];
        else if(arg=="--port"&&i+1<argc)config.port=std::uint16_t(std::atoi(argv[++i]));
        else if(arg=="--api-key"&&i+1<argc)config.api_key=argv[++i];
        else if(arg=="--max-connections"&&i+1<argc)config.max_connections=unsigned(std::atoi(argv[++i]));
        else if(arg=="--trace")zen::diag::set_trace_enabled(true);
        else if(arg=="--breadcrumbs")zen::diag::set_breadcrumbs_enabled(true);
        else if(arg=="--help"){
            std::cout<<"zen_server [--model model.gguf] [--draft-model draft.gguf] [--model-id ID] [--host IP] [--port N] [--api-key KEY] [--max-connections N] [--trace] [--breadcrumbs]\n";
            return 0;
        }else{std::cerr<<"Unknown or incomplete argument: "<<arg<<"\n";return 2;}
    }
    std::unique_ptr<zen::runtime::ModelBackend> backend;
    if(!model_path.empty()){
        backend=std::make_unique<zen::runtime::ModelBackend>();std::string error;
        if(!backend->load(model_path,model_id,&error,draft_model_path)){std::cerr<<"Model load failed: "<<error<<"\n";return 1;}
        std::cout<<"Loaded model: "<<backend->model_id()<<"\n";if(!draft_model_path.empty())std::cout<<"Loaded speculative draft model: "<<draft_model_path<<"\n";
    }else std::cout<<"No model loaded; generation requests return model_not_loaded.\n";
    zen::ResourceGovernor governor;const auto snapshot=governor.snapshot();const auto policy=governor.current_policy();
    std::cout<<"ZenInfer server | logical CPUs "<<snapshot.logical_cpus<<" | physical cores "<<snapshot.physical_cpus
             <<" | available RAM "<<(snapshot.available_ram_bytes>>20)<<" MiB | workers "<<policy.worker_threads
             <<" | cache budget "<<(policy.cache_budget_bytes>>20)<<" MiB | "<<policy.memory_tier<<"\n";
    zen::http::OpenAIServer server(config,backend.get());
    std::cout<<"Listening at http://"<<config.host<<":"<<config.port<<"\n";
    if(!server.run()){std::cerr<<"HTTP server startup failed\n";return 1;}
    return 0;
}
