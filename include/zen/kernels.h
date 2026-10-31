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
#endif

bool cpu_has_avx2_fma() noexcept;
KernelSet select_kernels(bool force_scalar = false) noexcept;
}
