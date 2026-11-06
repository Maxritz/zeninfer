// Isolates the LM head: one Q8_0 matvec of [vocab x hidden] through the D3D12
// path, timed separately from the transformer layers.
//
// Why this exists: end-to-end decode was 7.9 ms/token while the 16-layer GPU
// pipeline measured 2.8 ms, leaving ~5 ms unaccounted. The vocabulary
// projection is the only other large weight read per token, so it either runs
// on the GPU (sub-millisecond at VRAM bandwidth) or falls back to the CPU
// (~7 ms at DDR4 bandwidth). This prints which, and what it costs, instead of
// inferring it.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "zen/d3d12_backend.h"

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

}  // namespace

int main(int argc, char** argv) {
    // Llama-3.2-1B defaults; override with vocab and hidden on the command line.
    const std::size_t vocab = argc > 1 ? static_cast<std::size_t>(std::atoi(argv[1])) : 128256;
    const std::size_t hidden = argc > 2 ? static_cast<std::size_t>(std::atoi(argv[2])) : 2048;
    const int iters = argc > 3 ? std::atoi(argv[3]) : 10;

    if (!zen::d3d12::enabled()) {
        std::printf("SKIP: D3D12 backend not enabled\n");
        return 77;
    }

    // Q8_0 layout: 32-element blocks of (fp16 scale, 32 int8).
    const std::size_t blocks_per_row = hidden / 32;
    const std::size_t row_bytes = blocks_per_row * 34;
    std::vector<std::uint8_t> weights(vocab * row_bytes);

    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    for (std::uint8_t& b : weights) {
        b = static_cast<std::uint8_t>(byte_dist(rng));
    }
    // Give each block a small positive fp16 scale so the values stay finite.
    for (std::size_t r = 0; r < vocab; ++r) {
        for (std::size_t b = 0; b < blocks_per_row; ++b) {
            std::uint8_t* blk = weights.data() + r * row_bytes + b * 34;
            blk[0] = 0x00;  // fp16 ~0.0001
            blk[1] = 0x2E;
        }
    }

    std::vector<float> x(hidden, 0.01f);
    std::vector<float> y(vocab, 0.0f);

    const double weight_mb = double(weights.size()) / (1024.0 * 1024.0);
    std::printf("lm_head vocab=%zu hidden=%zu weights=%.1f MB iters=%d\n", vocab, hidden, weight_mb,
                iters);

    // First call pays the one-off repack + upload into VRAM.
    const auto cold = Clock::now();
    const bool cold_ok = zen::d3d12::gemv_q8_0(weights.data(), x.data(), y.data(), vocab, hidden);
    const double cold_ms = ms_since(cold);
    std::printf("cold (upload+dispatch) = %.2f ms  ok=%d\n", cold_ms, int(cold_ok));
    if (!cold_ok) {
        std::printf("RESULT: GPU path REJECTED -> this matvec runs on the CPU\n");
        return 1;
    }

    double best = 1e9;
    double sum = 0.0;
    for (int i = 0; i < iters; ++i) {
        const auto t = Clock::now();
        const bool ok = zen::d3d12::gemv_q8_0(weights.data(), x.data(), y.data(), vocab, hidden);
        const double ms = ms_since(t);
        if (!ok) {
            std::printf("RESULT: GPU path failed on iteration %d\n", i);
            return 1;
        }
        best = ms < best ? ms : best;
        sum += ms;
    }
    const double avg = sum / iters;
    std::printf("warm avg=%.3f ms best=%.3f ms  -> %.1f GB/s\n", avg, best,
                (double(weights.size()) / (avg / 1000.0)) / 1e9);

    // What the CPU would cost at DDR4 bandwidth, for comparison.
    const double ddr_gbs = 40.0;
    const double cpu_ms = (double(weights.size()) / (ddr_gbs * 1e9)) * 1000.0;
    std::printf("cpu-equivalent at %.0f GB/s = %.2f ms\n", ddr_gbs, cpu_ms);
    std::printf("RESULT: LM head is %s (%.2fx faster than CPU-bound)\n",
                avg < cpu_ms ? "ON GPU" : "effectively CPU-bound", cpu_ms / avg);
    return 0;
}
