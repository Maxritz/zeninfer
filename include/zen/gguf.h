#pragma once
#include "zen/model_ir.h"
#include <filesystem>
#include <string>
namespace zen::gguf {
struct LoadReport {
    bool ok{};
    std::string error;
    std::uint32_t version{};
    std::uint64_t tensor_data_offset{};
    model::ModelConfig config;
    std::vector<model::TensorDesc> tensors;
};
LoadReport inspect(const std::filesystem::path& path);
// Parses an in-memory GGUF image through the same reader-agnostic parse core
// as the file path. Every malformed, truncated or hostile input returns
// ok=false with an error string; no parse attempt may crash or hang.
LoadReport inspect_bytes(const std::uint8_t* data, std::size_t size);
// Reads one named tensor from the GGUF payload and converts it to row-major FP32.
bool read_tensor_f32(const std::filesystem::path& path, const LoadReport& report,
                     std::string_view tensor_name, std::vector<float>& output,
                     std::string* error=nullptr);
}
