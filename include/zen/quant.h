#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
namespace zen::quant {
enum class Encoding : std::uint8_t { fp32, fp16, bf16, q4_0, q4_1, q5_0, q5_1, q8_0, q3_k, q6_k, iq4_nl, signed_uniform, ternary, tq1_0, tq2_0,
                                     pq2_0, q2_k, q4_k, q5_k, q8_k, iq1_s, iq1_m, iq2_xxs, iq2_xs, iq2_s, iq3_xxs, iq3_s, iq4_xs, mxfp4, rocmfp4, rocmfp4_fast, ptq1_0,
                                     fp64, fp8_e4m3, i8, i16, i32, i64, q1_0, q2_0, q8_1, q8_1_f32, nvfp4 };
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
// Sequential K-block repack (Q8_0 only). Raw GGUF rows are row-major: row r
// holds (cols/32) blocks of 34 B (fp16 scale + 32 int8 quants). The
// panel-major repack re-tiles those blocks into
//   [rblock RB=64][kpanel Kc=512][row-in-rblock][block-in-panel]
// where one kpanel is 16 blocks = 544 B and one (rblock,kpanel) tile is
// 64*544 = 34816 B. The last rblock/kpanel are padded (zero-filled) up to
// the full tile pitch, so the byte offset of (row, col_block) is
//   ((r/64)*n_kpanels + col_block/16)*64*544 + (r%64)*544 + (col_block%16)*34
// with n_kpanels = ceil((cols/32)/16). Inside the K-blocked loop this turns
// up to 8.7 KiB strided row-panel jumps (K=8192) into 544 B steps: the 4-row
// panel tile becomes one contiguous 2176 B chunk and the whole rblock sweep
// streams sequentially, letting the hardware prefetcher run ahead. The
// repack is a pure byte transpose (no requantisation), so the dequantised
// weights -- and hence every FP result -- are identical to the raw path.
// Built once at load (dense runtime) when ZEN_CPU_Q8_REPACK_KBLK=1 and
// consumed only while ZEN_CPU_Q8_KBLK=1; otherwise the raw row-major buffer
// is used. Follows the ZEN_CPU_Q8_KBLK precedent: any value other than
// "1" (including unset) is off.
bool q8_repack_enabled() noexcept;
bool q8_kblk_enabled() noexcept;
bool q8_use_kblk(std::size_t tokens, std::size_t cols) noexcept;
std::vector<std::uint8_t> repack_q8_0_kblk(const std::uint8_t* weights,
                                            std::size_t rows, std::size_t columns);
std::size_t repacked_q8_0_kblk_bytes(std::size_t rows, std::size_t columns) noexcept;
// Row-parallel repacked GEMM over a buffer built by repack_q8_0_kblk.
// row_offset is ignored (always 0) because the buffer is global; rows/cols
// must match the repack's shape. Used directly by the bench/tests; the
// model path goes through quant::gemm via register_q8_repack.
bool gemm_q8_0_kblk_repack_parallel(const std::uint8_t* repacked, const float* input,
                                    float* output, std::size_t tokens, std::size_t rows,
                                    std::size_t cols, unsigned workers);
// Registry keyed by the raw (mmap) weight pointer. The model runtime builds
// each eligible Q8 tensor's repack at load and registers it here; quant::gemm
// (after the GPU checks, before the CPU K-block dispatch) looks the raw
// pointer up and, when both env gates are on and the (tokens,cols)
// threshold is met, runs the repacked parallel kernel instead of the raw
// one. Short-lived caller buffers (bench/tests) must use
// gemm_q8_0_kblk_repack_parallel directly and never register, so a recycled
// address can never resolve to stale bytes.
void register_q8_repack(const std::uint8_t* raw, std::size_t rows, std::size_t cols,
                        std::vector<std::uint8_t> packed);
// Several matvecs that share one input vector (the Q|K|V and gate|up pattern in a
// transformer block). Submitting them together avoids paying the fixed queue round
// trip once per matrix; falls back to one gemv per task when no GPU path applies.
bool gemv_batch(const Format& format,const std::uint8_t* const* weights,const float* input,
                float* const* outputs,const std::size_t* rows,std::size_t columns,
                std::size_t count,float scale=1.0f,unsigned workers=0);
// Several GEMMs that share one input matrix (the Q|K|V and gate|up pattern
// in a transformer prefill). Submitting them together avoids paying the fixed
// queue round trip once per matrix; falls back to one gemm per task when no
// GPU path applies. The GPU path is gated by ZEN_D3D12_BATCH=1 (default off);
// with the gate unset the loop below runs the ordinary single-gemm path, so
// default behaviour is unchanged.
bool gemm_batch(const Format& format,const std::uint8_t* const* weights,const float* input,
                float* const* outputs,const std::size_t* rows,std::size_t tokens,
                std::size_t columns,std::size_t count,float scale=1.0f,unsigned workers=0);
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
