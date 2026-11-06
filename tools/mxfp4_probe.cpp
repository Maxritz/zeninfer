// Generic real-model dequant probe: decodes the first rows of a named tensor
// from a real GGUF through quant::dequantize and writes raw f32 for external
// cross-checking (scripts/mxfp4_check.py, scripts/rocmfp4_check.py,
// scripts/dequant_row_check.py).
// Usage: zen_mxfp4_probe model.gguf tensor_name [n_rows] [out.f32]
#include "zen/gguf.h"
#include "zen/quant.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {
const char* format_name_for(std::uint32_t t) {
    switch (t) {
        case 0: return "F32"; case 1: return "F16"; case 30: return "BF16";
        case 2: return "Q4_0"; case 3: return "Q4_1"; case 6: return "Q5_0";
        case 7: return "Q5_1"; case 8: return "Q8_0"; case 10: return "Q2_K";
        case 11: return "Q3_K"; case 12: return "Q4_K"; case 13: return "Q5_K";
        case 14: return "Q6_K"; case 15: return "Q8_K"; case 16: return "IQ2_XXS";
        case 17: return "IQ2_XS"; case 18: return "IQ3_XXS"; case 19: return "IQ1_S";
        case 20: return "IQ4_NL"; case 21: return "IQ3_S"; case 22: return "IQ2_S";
        case 23: return "IQ4_XS"; case 29: return "IQ1_M"; case 34: return "TQ1_0";
        case 35: return "TQ2_0"; case 39: return "MXFP4";
        case 100: return "Q4_0_ROCMFP4"; case 101: return "Q4_0_ROCMFP4_FAST";
        case 143: return "PTQ1_0";
        default: return nullptr;
    }
}
}

int main(int argc, char** argv) {
    if (argc < 3) { std::printf("usage: zen_mxfp4_probe model.gguf tensor_name [n_rows] [out.f32]\n"); return 2; }
    const std::filesystem::path model = argv[1];
    const std::string tensor_name = argv[2];
    const std::size_t n_rows = argc > 3 ? std::strtoul(argv[3], nullptr, 10) : 16;
    const char* out_path = argc > 4 ? argv[4] : "mxfp4_probe.f32";

    auto report = zen::gguf::inspect(model);
    if (!report.ok) { std::printf("inspect failed: %s\n", report.error.c_str()); return 1; }
    const zen::model::TensorDesc* desc = nullptr;
    for (const auto& t : report.tensors) if (t.name == tensor_name) { desc = &t; break; }
    if (!desc) { std::printf("tensor not found: %s\n", tensor_name.c_str()); return 1; }
    std::printf("tensor=%s ggml_type=%u byte_size=%llu dims=%zu",
                tensor_name.c_str(), desc->ggml_type,
                static_cast<unsigned long long>(desc->byte_size), desc->dimensions.size());
    for (auto d : desc->dimensions) std::printf(" %lld", static_cast<long long>(d));
    std::printf("\n");
    const char* fname = format_name_for(desc->ggml_type);
    if (!fname) { std::printf("unsupported probe type %u\n", desc->ggml_type); return 1; }
    const auto* fmt = zen::quant::find(fname);
    if (!fmt || !fmt->implemented) { std::printf("decoder not implemented: %s\n", fname); return 1; }
    if (desc->dimensions.empty() || desc->dimensions[0] % fmt->block_elements) {
        std::printf("row width not block-aligned\n"); return 1;
    }
    const std::size_t row_values = static_cast<std::size_t>(desc->dimensions[0]);
    const std::size_t row_bytes = zen::quant::row_bytes(*fmt, row_values);
    if (!row_bytes) { std::printf("row_bytes failed\n"); return 1; }
    const std::size_t total_rows = static_cast<std::size_t>(desc->byte_size) / row_bytes;
    const std::size_t rows = std::min(total_rows, n_rows);

    std::ifstream f(model, std::ios::binary);
    if (!f) { std::printf("cannot reopen model\n"); return 1; }
    const auto absolute = report.tensor_data_offset + desc->offset;
    f.seekg(static_cast<std::streamoff>(absolute));
    if (!f) { std::printf("seek failed\n"); return 1; }

    std::vector<std::uint8_t> packed(rows * row_bytes);
    f.read(reinterpret_cast<char*>(packed.data()), static_cast<std::streamsize>(packed.size()));
    if (!f) { std::printf("short read (tensor smaller than probe?)\n"); return 1; }

    std::vector<float> out(rows * row_values);
    if (!zen::quant::dequantize(*fmt, packed.data(), out.size(), out.data())) {
        std::printf("dequantize failed\n"); return 1;
    }

    std::ofstream o(out_path, std::ios::binary);
    o.write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size() * 4));
    std::printf("decoded %zu rows x %zu values (%zu B/row) -> %s\n",
                rows, row_values, row_bytes, out_path);
    double sum = 0; float amax = 0; int nan = 0;
    for (float v : out) {
        if (std::isnan(v)) ++nan;
        else { sum += v; amax = std::max(amax, std::abs(v)); }
    }
    std::printf("summary: mean=%.6g amax=%.6g nan=%d\n", sum / static_cast<double>(out.size()), amax, nan);
    return nan ? 1 : 0;
}
