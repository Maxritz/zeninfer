// TEMP harness (not shipped, not in CMake): quant::gemm worker sweep on gate shape.
// Gate shape rows=8192 cols=2048 (Llama-1B FFN up-proj). Q8_0 synthetic weights,
// CPU path only (ZEN_BACKEND=cpu). Reports GF/s per (T, workers).
#include "zen/quant.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

using namespace zen;
static double bench_gemm(const quant::Format* fmt, const uint8_t* w, const float* x, float* o,
                         size_t T, size_t rows, size_t cols, unsigned workers, int reps) {
    // warmup
    quant::gemm(*fmt, w, x, o, T, rows, cols, 1.0f, workers);
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) quant::gemm(*fmt, w, x, o, T, rows, cols, 1.0f, workers);
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count() / reps;
}

int main() {
    const auto* fmt = quant::find("Q8_0");
    if (!fmt) { printf("no Q8_0\n"); return 2; }
    const size_t rows = 8192, cols = 2048;
    size_t rb = quant::row_bytes(*fmt, cols);
    printf("gate rows=%zu cols=%zu row_bytes=%zu weights=%.2f MB\n", rows, cols, rb, rows*rb/1048576.0);
    std::vector<uint8_t> packed(rows*rb);
    uint32_t s = 12345;
    auto rnd = [&](){ s = s*1664525u + 1013904223u; return (s>>24)&0xff; };
    for (size_t r = 0; r < rows; ++r) for (size_t blk = 0; blk < cols/32; ++blk) {
        uint8_t* b = packed.data() + r*rb + blk*34; b[0]=0x00; b[1]=0x3c; // fp16 1.0
        for (size_t j = 2; j < 34; ++j) b[j] = (uint8_t)rnd();
    }
    printf("T\tworkers\tms\tGF/s\tGF/s/core\tmax|out|\n");
    for (size_t T : {32u, 128u, 171u}) {
        std::vector<float> xin(T*cols);
        for (size_t t = 0; t < T; ++t) for (size_t c = 0; c < cols; ++c)
            xin[t*cols+c] = float(int((t*7+c)%15)-7)*0.017f + float(int((t*13+c*3)%7)-3)*0.001f;
        std::vector<float> out(T*rows);
        int reps = (T==171)?5:(T==128)?5:10;
        for (unsigned w : {1u, 2u, 4u, 8u, 16u, 32u}) {
            double sec = bench_gemm(fmt, packed.data(), xin.data(), out.data(), T, rows, cols, w, reps);
            double gflops = 2.0*double(T)*double(rows)*double(cols)/1e9;
            double gfs = gflops/sec;
            double mx = 0; for (float v : out) mx = std::max(mx, (double)std::fabs(v));
            printf("%zu\t%u\t%.3f\t%.1f\t%.2f\t%.3f\n", T, w, sec*1000.0, gfs, gfs/w, mx);
            fflush(stdout);
        }
    }
    return 0;
}
