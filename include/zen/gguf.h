#pragma once
#include "zen/model_ir.h"
#include <filesystem>
#include <string>
namespace zen::gguf {
// GGML type 9 has two indistinguishable historical layouts. Do not infer it
// from padding or tensor offsets: callers must explicitly select a layout.
enum class Q8_1Layout : std::uint8_t { unspecified, fp16_36, fp32_40 };
struct InspectOptions { Q8_1Layout q8_1_layout{Q8_1Layout::unspecified}; };
struct LoadReport {
    bool ok{};
    std::string error;
    std::uint32_t version{};
    std::uint64_t tensor_data_offset{};
    Q8_1Layout q8_1_layout{Q8_1Layout::unspecified};
    model::ModelConfig config;
    std::vector<model::TensorDesc> tensors;
};
// If options leave Q8_1 unspecified, ZEN_GGUF_Q8_1_LAYOUT=fp16 or fp32
// is an explicit process-wide override; otherwise type 9 fails closed.
LoadReport inspect(const std::filesystem::path& path, InspectOptions options={});
// Parses an in-memory GGUF image through the same reader-agnostic parse core
// as the file path. Every malformed, truncated or hostile input returns
// ok=false with an error string; no parse attempt may crash or hang.
LoadReport inspect_bytes(const std::uint8_t* data, std::size_t size, InspectOptions options={});
// Reads one named tensor from the GGUF payload and converts it to row-major FP32.
bool read_tensor_f32(const std::filesystem::path& path, const LoadReport& report,
                     std::string_view tensor_name, std::vector<float>& output,
                     std::string* error=nullptr);
}
