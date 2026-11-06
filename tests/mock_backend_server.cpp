#include "zen/openai_server.h"
#include "zen/diagnostics.h"
#include "zen/chat_compaction.h"
#include <cstdlib>
#include <string>

class MockBackend final : public zen::http::InferenceBackend {
public:
    std::string model_id() const override { return "zeninfer-mock-test"; }
    bool supports_context_compaction() const noexcept override { return std::getenv("ZEN_MOCK_NO_COMPACTION") == nullptr; }
    zen::http::GenerationResult generate(
        const zen::http::GenerationRequest& request,
        const std::function<void(const std::string&)>& emit,
        const std::atomic<bool>& cancelled) override {
        if(request.compaction.enabled && !request.messages.empty() && request.messages.back().content=="FAIL_COMPACTION")
            throw zen::chat::CompactionError("Mock compaction failed");
        std::string answer = "Mock backend received ";
        answer += request.messages.empty() ? request.prompt : request.messages.back().content;
        if (answer.size() > request.max_tokens * 8u) answer.resize(request.max_tokens * 8u);
        for (std::size_t i = 0; i < answer.size(); i += 8) {
            if (cancelled.load()) break;
            emit(answer.substr(i, 8));
        }
        zen::http::GenerationResult result{answer, 8u, static_cast<std::uint32_t>((answer.size() + 3) / 4), "stop", {}};
        if(request.compaction.enabled){
            result.compaction.requested=true; result.compaction.applied=true;
            result.compaction.original_prompt_tokens=request.compaction.trigger_tokens;
            result.compaction.final_prompt_tokens=request.compaction.target_tokens;
            result.compaction.summary_prompt_tokens=10;result.compaction.summary_completion_tokens=3;
            result.compaction.compacted_messages=request.compaction.keep_last_messages;
            result.compaction.summary="Mock \"summary\"";
        }
        return result;
    }
};

int main(int argc, char** argv) {
    zen::http::ServerConfig config;
    if (argc > 1) config.port = static_cast<std::uint16_t>(std::atoi(argv[1]));
    zen::diag::set_breadcrumbs_enabled(true);
    zen::diag::set_trace_enabled(true);
    MockBackend backend;
    zen::http::OpenAIServer server(config, &backend);
    return server.run() ? 0 : 1;
}
