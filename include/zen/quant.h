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
// Several matvecs that share one input vector (the Q|K|V and gate|up pattern in a
// transformer block). Submitting them together avoids paying the fixed queue round
// trip once per matrix; falls back to one gemv per task when no GPU path applies.
bool gemv_batch(const Format& format,const std::uint8_t* const* weights,const float* input,
                float* const* outputs,const std::size_t* rows,std::size_t columns,
                std::size_t count,float scale=1.0f,unsigned workers=0);
// gate|up|SiLU|down in a single GPU submission, with the FFN activation kept on the
// device. Returns false when the path is unavailable, so callers can run the separate
// matvecs instead.
bool ffn_gemv(const Format& gate_format,const std::uint8_t* gate_weights,
              const Format& up_format,const std::uint8_t* up_weights,
              const Format& down_format,const std::uint8_t* down_weights,
              const float* input,float* output,std::size_t ffn_rows,
              std::size_t columns,std::size_t output_rows);
// The whole FFN half of a layer in one GPU submission: the attention output matvec,
// the residual add, the FFN norm, gate|up, SiLU, down and the second residual add.
// Returns false when the path is unavailable so the caller can run the separate ops.
bool ffn_residual_fused(const Format& attn_output_format,const std::uint8_t* attn_output_weights,
                        const Format& gate_format,const std::uint8_t* gate_weights,
                        const Format& up_format,const std::uint8_t* up_weights,
                        const Format& down_format,const std::uint8_t* down_weights,
                        const float* norm_weight,const float* x,const float* attn_out,
                        float* output,std::size_t hidden,std::size_t attn_width,
                        std::size_t ffn_rows,float epsilon);
std::size_t row_bytes(const Format& format, std::size_t columns) noexcept;
}
