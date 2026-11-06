#include "zen/quant.h"
#include "zen/kernels.h"
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
int main() {
    const auto* fmt = zen::quant::find("Q8_0");
    if (!fmt) return 2;
    if (!zen::cpu_has_avx2_fma()) return 2;
    struct Case { std::size_t rows, cols; };
    std::vector<Case> cases = {{2048, 2048}, {512, 2048}, {2048, 8192}, {8192, 2048}, {32768, 2048}};
    for (auto c : cases) {
        const std::size_t rows = c.rows, cols = c.cols;
        const std::size_t rb = zen::quant::row_bytes(*fmt, cols);
        std::vector<std::uint8_t> packed(rows * rb);
        std::uint32_t s = 999;
        auto rndb = [&]() { s = s * 1664525u + 1013904223u; return static_cast<std::uint8_t>((s >> 24) & 0xffu); };
        for (std::size_t r = 0; r < rows; ++r)
            for (std::size_t blk = 0; blk < cols / 32; ++blk) {
                std::uint8_t* b = packed.data() + r * rb + blk * 34;
                b[0] = 0x00; b[1] = 0x3c;
                for (std::size_t j = 2; j < 34; ++j) b[j] = rndb();
            }
        for (std::size_t T : {std::size_t(32), std::size_t(128)}) {
            std::vector<float> xin(T * cols), out(T * rows, 0.0f);
            for (std::size_t i = 0; i < xin.size(); ++i) xin[i] = float(int(i % 15) - 7) * 0.017f;
            auto bench = [&](auto&& body) {
                body();
                double best = 1e18;
                for (int i = 0; i < 5; ++i) {
                    auto t0 = std::chrono::steady_clock::now();
                    body();
                    double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
                    if (us < best) best = us;
                }
                return best / 1000.0;
            };
            double ms = bench([&]() { zen::quant::gemm(*fmt, packed.data(), xin.data(), out.data(), T, rows, cols, 1.0f, 0); });
            double ms1 = bench([&]() { zen::quant::gemm(*fmt, packed.data(), xin.data(), out.data(), T, rows, cols, 1.0f, 1); });
            const char* kblk = std::getenv("ZEN_CPU_Q8_KBLK");
            std::printf("rows=%zu cols=%zu T=%zu ms=%.3f ms1=%.3f KBLK=%s\n", rows, cols, T, ms, ms1, kblk ? kblk : "(1)");
        }
    }
    return 0;
}
