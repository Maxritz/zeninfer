// RMSNorm micro-benchmark: scalar reference loop (as in
// dense_transformer_runtime.cpp rms_norm) vs zen::rms_norm_avx2.
// Reports best-of-reps ms for a [T, hidden] slab, maxerr vs the scalar
// reference, and the speedup. Run once: it times both kernels directly,
// so ZEN_CPU_RMS_AVX2 is not consulted here.
#include "zen/kernels.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {
using clock_t = std::chrono::steady_clock;

void rms_scalar(const float* x, const float* w, float* y, std::size_t n, float eps) {
    double ss = 0.0;
    for (std::size_t i = 0; i < n; ++i) ss += double(x[i]) * x[i];
    const float inv = 1.0f / std::sqrt(static_cast<float>(ss / double(n)) + eps);
    for (std::size_t i = 0; i < n; ++i) y[i] = x[i] * inv * w[i];
}

double bench_best_us(auto&& body, int warmup, int reps) {
    body();
    for (int i = 0; i < warmup; ++i) body();
    double best = 1e18;
    for (int i = 0; i < reps; ++i) {
        const auto t0 = clock_t::now();
        body();
        const double us = std::chrono::duration<double, std::micro>(clock_t::now() - t0).count();
        if (us < best) best = us;
    }
    return best;
}

void run_shape(std::size_t T, std::size_t hidden, int reps) {
    std::vector<float> x(T * hidden), w(hidden), out_s(T * hidden), out_v(T * hidden);
    // Pseudo-random activations (not a short-period pattern): a patterned
    // input can make the ss sum exactly representable in both orders and
    // hide the lane-reordering difference this bench is meant to measure.
    std::mt19937 rng(1234567u + std::uint32_t(T * 131u + hidden));
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = dist(rng);
    for (std::size_t i = 0; i < hidden; ++i) w[i] = 1.0f + float(int(i % 5)) * 0.01f;
    const float eps = 1e-5f;
    for (std::size_t t = 0; t < T; ++t)
        rms_scalar(x.data() + t * hidden, w.data(), out_s.data() + t * hidden, hidden, eps);
    const double scalar_us = bench_best_us(
        [&]() { for (std::size_t t = 0; t < T; ++t) rms_scalar(x.data() + t * hidden, w.data(), out_s.data() + t * hidden, hidden, eps); }, 2, reps);
#ifdef ZEN_HAS_AVX2_KERNELS
    for (std::size_t t = 0; t < T; ++t)
        zen::rms_norm_avx2(x.data() + t * hidden, w.data(), out_v.data() + t * hidden, hidden, eps);
    double maxerr = 0;
    for (std::size_t i = 0; i < out_s.size(); ++i) {
        const double e = std::abs(double(out_v[i]) - double(out_s[i]));
        if (e > maxerr) maxerr = e;
    }
    const double avx2_us = bench_best_us(
        [&]() { for (std::size_t t = 0; t < T; ++t) zen::rms_norm_avx2(x.data() + t * hidden, w.data(), out_v.data() + t * hidden, hidden, eps); }, 2, reps);
    const double elems = double(T) * double(hidden);
    std::printf("T=%zu hidden=%zu scalar=%.3f ms (%.2f GB/s) avx2=%.3f ms (%.2f GB/s) speedup=%.2fx maxerr=%.4g %s\n",
                T, hidden, scalar_us / 1000.0, elems * 12.0 / (scalar_us / 1e6) / 1e9,
                avx2_us / 1000.0, elems * 12.0 / (avx2_us / 1e6) / 1e9,
                scalar_us / avx2_us, maxerr, (maxerr <= 1e-5 ? "PASS" : "FAIL"));
#else
    std::printf("T=%zu hidden=%zu scalar=%.3f ms (no AVX2 kernels in this build)\n", T, hidden, scalar_us / 1000.0);
#endif
}
} // namespace

int main(int argc, char** argv) {
    int reps = 20;
    if (argc > 1) reps = std::max(1, std::atoi(argv[1]));
#ifdef ZEN_HAS_AVX2_KERNELS
    if (!zen::cpu_has_avx2_fma()) { std::printf("no AVX2+FMA\n"); return 2; }
#endif
    // Llama-3.2-1B: hidden=2048. Also head_dim=64 (qk norm) and a large slab.
    for (const std::size_t T : {std::size_t(128), std::size_t(171), std::size_t(512), std::size_t(1024)})
        run_shape(T, 2048, reps);
    run_shape(512, 64, reps);
    run_shape(128, 8192, reps);
    return 0;
}
