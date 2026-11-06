#pragma once
#include "zen/openai_server.h"
#include <string>
namespace zen::chat {
// Formats a chat transcript using the currently supported legacy formats.
// This is intentionally not advertised as a Jinja2 interpreter.
bool format_legacy_chat(const std::string& stored_template,const std::string& architecture,
                        const std::vector<http::ChatMessage>& messages,std::string& prompt,
                        std::string* error=nullptr);
}
