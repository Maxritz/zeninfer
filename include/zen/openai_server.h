#pragma once
#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>
#include <unordered_map>
namespace zen::http {
struct ChatMessage { std::string role, content; };
struct CompactionOptions {
    bool enabled{false};
    std::uint32_t trigger_tokens{4096}, target_tokens{2048}, summary_max_tokens{256}, keep_last_messages{4};
};
struct CompactionInfo {
    bool requested{false}, applied{false};
    std::uint32_t original_prompt_tokens{}, final_prompt_tokens{}, summary_prompt_tokens{}, summary_completion_tokens{}, compacted_messages{};
    std::string summary;
};
struct GenerationRequest { std::string model, prompt; std::vector<ChatMessage> messages; std::uint32_t max_tokens{128}, top_k{}, speculative_tokens{}; double temperature{0.8}, top_p{1.0}; std::uint64_t seed{}; std::vector<std::string> stop; std::unordered_map<std::uint32_t,float> logit_bias; std::vector<std::uint32_t> allowed_token_ids; CompactionOptions compaction; };
struct GenerationResult { std::string text; std::uint32_t prompt_tokens{}, completion_tokens{}; std::string finish_reason{"stop"}; CompactionInfo compaction; };
class InferenceBackend { public: virtual ~InferenceBackend()=default; virtual std::string model_id()const=0; virtual bool supports_context_compaction()const noexcept{return false;} virtual GenerationResult generate(const GenerationRequest&,const std::function<void(const std::string&)>&,const std::atomic<bool>&)=0; };
struct ServerConfig { std::string host{"127.0.0.1"}, api_key; std::uint16_t port{8000}; std::size_t max_request_bytes{4u*1024u*1024u}; unsigned max_connections{}; };
class OpenAIServer { public: explicit OpenAIServer(ServerConfig,InferenceBackend* =nullptr); ~OpenAIServer(); OpenAIServer(const OpenAIServer&)=delete; OpenAIServer& operator=(const OpenAIServer&)=delete; bool run(); void stop()noexcept; private: ServerConfig config_; InferenceBackend* backend_{}; std::atomic<bool> stopping_{false}; std::atomic<std::intptr_t> listen_socket_{-1}; };
}
