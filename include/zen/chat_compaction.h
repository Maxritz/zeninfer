#pragma once
#include "zen/model_runtime.h"
#include "zen/openai_server.h"
#include <atomic>
#include <stdexcept>
#include <string>
#include <vector>

namespace zen::chat {
// Controlled, user-visible failures; never silently drop history or proceed
// with an oversized context when an explicitly requested compaction fails.
class CompactionError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Renders and tokenizes the original chat, optionally summarizes its older
// non-system messages, and renders/tokenizes the resulting request. No client
// history is changed or stored by this function.
void prepare_compacted_chat(const runtime::ModelRuntime& model,
                            const http::GenerationRequest& request,
                            const std::atomic<bool>& cancelled,
                            std::string& prompt,
                            std::vector<std::uint32_t>& prompt_ids,
                            http::CompactionInfo& info);
}
