#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
namespace zen::quant {
enum class Encoding : std::uint8_t { fp32, fp16, bf16, q4_0, q4_1, q5_0, q5_1, q8_0, q3_k, q6_k, iq4_nl, signed_uniform, ternary, tq1_0, tq2_0,
                                     q2_k, q4_k, q5_k, q8_k, iq1_s, iq1_m, iq2_xxs, iq2_xs, iq2_s, iq3_xxs, iq3_s, iq4_xs, mxfp4, rocmfp4, rocmfp4_fast, ptq1_0 };
struct Format {
    std::string name;
    Encoding encoding{};
    std::uint8_t bits{};
    std::uint16_t block_elements{};
    std::uint16_t block_bytes{};
    bool implemented{};
    std::string note;
};
const std::vector<Format>& registry();
const Format* find(std::string_view name) noexcept;
bool decode_fp16(const std::uint16_t* src, float* dst, std::size_t count) noexcept;
bool encode_fp16(const float* src, std::uint16_t* dst, std::size_t count) noexcept;
bool dequantize(const Format& format, const std::uint8_t* src,
                std::size_t elements, float* dst, float scale=1.0f) noexcept;
// Row-major quantised matrix-vector product. Each row is independently block-aligned.
bool gemv(const Format& format, const std::uint8_t* weights, const float* input,
          float* output, std::size_t rows, std::size_t columns,
          float scale=1.0f,unsigned workers=0);
// Computes input[tokens, columns] x weights[rows, columns]^T into output[tokens, rows].
bool gemm(const Format& format,const std::uint8_t* weights,const float* input,float* output,
          std::size_t tokens,std::size_t rows,std::size_t columns,float scale=1.0f,unsigned workers=0);
std::size_t row_bytes(const Format& format, std::size_t columns) noexcept;
}
