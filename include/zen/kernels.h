#pragma once
#include <cstddef>
#include <cstdint>

namespace zen {
using GemvFn = void (*)(const float*, const float*, float*, std::size_t, std::size_t);
using GemmFn = void (*)(const float*, const float*, float*, std::size_t, std::size_t, std::size_t);

struct KernelSet {
    const char* name;
    GemvFn gemv;
    GemmFn gemm;
};

void gemv_scalar(const float* a, const float* x, float* y,
                 std::size_t m, std::size_t k);
void gemm_scalar(const float* a, const float* b, float* c,
                 std::size_t m, std::size_t n, std::size_t k);

#ifdef ZEN_HAS_AVX2_KERNELS
void gemv_avx2(const float* a, const float* x, float* y,
               std::size_t m, std::size_t k);
void gemm_avx2(const float* a, const float* b, float* c,
               std::size_t m, std::size_t n, std::size_t k);
// Fused GGML Q8_0 matrix-vector kernel. Rows must be block-aligned (K % 32 == 0).
bool gemv_q8_0_avx2(const std::uint8_t* weights, const float* x, float* y,
                    std::size_t rows, std::size_t columns);
// ZI-P5 batched Q8_0 kernel: dequantise one row, then feed each weight vector to
// up to 8 token accumulators. output points at the first row of this range and
// is advanced by output_row_stride per token, matching the row-major
// [tokens, rows] GEMM result layout.
bool gemm_q8_0_avx2(const std::uint8_t* weights, const float* input, float* output,
                    std::size_t tokens, std::size_t rows, std::size_t columns,
                    std::size_t output_row_stride);
// K-blocked variant: the activation panel [tokens, Kc] stays in L2 while the
// kernel sweeps all rows for that panel, instead of re-streaming the whole
// activation matrix per row pair. Partials are carried as __m256 vectors
// through the K loop and reduced once at the end, so the per-(row,token)
// FMA order matches gemm_q8_0_avx2 and results are bit-identical on the
// main block (see kernels/avx2.cpp). Kc is taken from ZEN_CPU_Q8_KC
// (64/128/256/512, default 512) and the row block from ZEN_CPU_Q8_RB
// (32/64/128/256, default 64). Same layout/stride contract as above.
bool gemm_q8_0_kblk_avx2(const std::uint8_t* weights, const float* input, float* output,
                    std::size_t tokens, std::size_t rows, std::size_t columns,
                    std::size_t output_row_stride);
// Repacked sequential variant: weights were rebuilt at load into the
// panel-major layout [rblock][kpanel][row][block] described in
// quant::repack_q8_0_kblk (Kc=512, RB=64), so the K-panel reads inside the
// K-blocked loop touch consecutive memory (panel_bytes=544 B apart) instead
// of row_bytes (up to 8.7 KiB for K=8192) apart. Row-panel traffic matches
// the hardware prefetcher and the 4-row panel tile is one contiguous
// 2176 B chunk. The activation tiling, __m256 partials and single-hsum
// reduction are unchanged from gemm_q8_0_kblk_avx2, so results are
// bit-identical on the main block. row_offset is the global (un-sliced)
// row index of the first row in this slice: parallel slices that do not
// start on an RB boundary still compute their repacked offset per row.
// Returns false on non-Q8 shapes (cols%32, missing pointers), mirroring the
// raw kernels; slice_rows is the number of output rows to write.
bool gemm_q8_0_kblk_repack_avx2(const std::uint8_t* repacked, std::size_t row_offset,
                    const float* input, float* output,
                    std::size_t tokens, std::size_t slice_rows, std::size_t columns,
                    std::size_t output_row_stride);
// ZI-P4 elementwise helpers. The scalar reference for each lives in the caller.
void silu_mul_avx2(const float* gate, const float* up, float* out, std::size_t n);
// RMSNorm: y[i] = x[i] * inv * weight[i], inv = 1/sqrt(mean(x^2)+eps).
// The sum of squares is accumulated in double via __m256d accumulators
// (see kernels/avx2.cpp for the reduction-order note vs the scalar loop).
void rms_norm_avx2(const float* x, const float* weight, float* y, std::size_t n, float eps);
float dot_f32_avx2(const float* a, const float* b, std::size_t n) noexcept;
void axpy_f32_avx2(float a, const float* x, float* y, std::size_t n) noexcept;
// Non-Q8 K-blocked batched GEMM (F16, Q4_K). Opt-in via ZEN_CPU_FAST_F16 /
// ZEN_CPU_FAST_Q4K in quant::gemm (default off); Q6_K/IQ stay on the generic
// path. 4-row x 2-token micro-tile, K-panel Kc from ZEN_CPU_F16_KC /
// ZEN_CPU_Q4K_KC (64/128/256, default 128). The fp32 weight panel [4, Kc]
// is dequantised once per panel and reused across every token from L1;
// partials stay __m256 until a single hsum at the end (lane order 0..7), so
// per-element FMA order is c=0,8,16,... like gemm_q8_0_kblk_avx2. The
// dequantised weights are bit-identical to quant::dequantize (F16: exact
// integer half->float, scalar patch for exp 0/31 lanes; Q4_K: mul+sub, not
// FMA); only the dot-product accumulation grouping differs from the plain
// path, within FP reassociation tolerance. Same layout/stride contract as
// gemm_q8_0_kblk_avx2.
bool gemm_f16_kblk_avx2(const std::uint8_t* weights, const float* input, float* output,
                        std::size_t tokens, std::size_t rows, std::size_t columns,
                        std::size_t output_row_stride);
bool gemm_q4k_kblk_avx2(const std::uint8_t* weights, const float* input, float* output,
                         std::size_t tokens, std::size_t rows, std::size_t columns,
                         std::size_t output_row_stride);
#endif

bool cpu_has_avx2_fma() noexcept;
KernelSet select_kernels(bool force_scalar = false) noexcept;
}
