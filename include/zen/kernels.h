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
// ZI-P4 elementwise helpers. The scalar reference for each lives in the caller.
void silu_mul_avx2(const float* gate, const float* up, float* out, std::size_t n);
float dot_f32_avx2(const float* a, const float* b, std::size_t n) noexcept;
void axpy_f32_avx2(float a, const float* x, float* y, std::size_t n) noexcept;
#endif

bool cpu_has_avx2_fma() noexcept;
KernelSet select_kernels(bool force_scalar = false) noexcept;
}
