#include "zen/openai_server.h"
#include "zen/diagnostics.h"
#include <cstdlib>
#include <string>

class MockBackend final : public zen::http::InferenceBackend {
public:
    std::string model_id() const override { return "zeninfer-mock-test"; }
    zen::http::GenerationResult generate(
        const zen::http::GenerationRequest& request,
        const std::function<void(const std::string&)>& emit,
        const std::atomic<bool>& cancelled) override {
        std::string answer = "Mock backend received ";
        answer += request.messages.empty() ? request.prompt : request.messages.back().content;
        if (answer.size() > request.max_tokens * 8u) answer.resize(request.max_tokens * 8u);
        for (std::size_t i = 0; i < answer.size(); i += 8) {
            if (cancelled.load()) break;
            emit(answer.substr(i, 8));
        }
        return {answer, 8u, static_cast<std::uint32_t>((answer.size() + 3) / 4), "stop"};
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
