// Q8_0 GEMM micro-benchmark (Priority 1, K-blocked vs legacy, plus the
// sequential repack candidate (a)). Times quant::gemm (dispatch honours
// ZEN_CPU_Q8_KBLK / ZEN_CPU_Q8_KC per process) at real prefill shapes plus
// direct-kernel comparisons. Repack timings are measured directly through
// quant::gemm_q8_0_kblk_repack_parallel, so they do not depend on the
// process env gates (which only steer the model load/registry path).
// Run twice per shape: ZEN_CPU_Q8_KBLK=0 (legacy) vs =1 (K-blocked).
// Prints: ms, tokens/ms, GFLOP/s, max err vs double reference.
#include "zen/quant.h"
#include "zen/kernels.h"
#include "zen/resource_governor.h"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
using clock_t = std::chrono::steady_clock;

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

struct Shape { std::size_t rows, cols; };

void run_shape(const zen::quant::Format* fmt, Shape shape, std::size_t T, unsigned workers, int reps) {
    const std::size_t rows = shape.rows, cols = shape.cols;
    const std::size_t rb = zen::quant::row_bytes(*fmt, cols);
    std::vector<std::uint8_t> packed(rows * rb);
    std::uint32_t s = 12345u + std::uint32_t(rows * 1000003u + cols);
    auto rndb = [&]() { s = s * 1664525u + 1013904223u; return static_cast<std::uint8_t>((s >> 24) & 0xffu); };
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t blk = 0; blk < cols / 32; ++blk) {
            std::uint8_t* b = packed.data() + r * rb + blk * 34;
            b[0] = 0x00; b[1] = 0x3c;
            for (std::size_t j = 2; j < 34; ++j) b[j] = rndb();
        }
    std::vector<float> xin(T * cols), out(T * rows, 0.0f);
    for (std::size_t t = 0; t < T; ++t)
        for (std::size_t c = 0; c < cols; ++c)
            xin[t * cols + c] = float(int((t * 7 + c) % 15) - 7) * 0.017f + float(int((t * 13 + c * 3) % 7) - 3) * 0.001f;

    // Correctness vs double naive (dequantised reference).
    std::vector<float> ref(rows * cols);
    bool ok_ref = zen::quant::dequantize(*fmt, packed.data(), rows * cols, ref.data());
    double maxerr = 0;
    if (ok_ref) {
        bool ok = zen::quant::gemm(*fmt, packed.data(), xin.data(), out.data(), T, rows, cols, 1.0f, workers);
        if (!ok) { std::printf("rows=%zu cols=%zu T=%zu workers=%u gemm FAILED\n", rows, cols, T, workers); return; }
        for (std::size_t t = 0; t < T; ++t)
            for (std::size_t r = 0; r < rows; ++r) {
                double sum = 0;
                for (std::size_t c = 0; c < cols; ++c) sum += double(ref[r * cols + c]) * xin[t * cols + c];
                const double e = std::abs(double(out[t * rows + r]) - sum);
                if (e > maxerr) maxerr = e;
            }
    }

    const double best_us = bench_best_us([&]() { zen::quant::gemm(*fmt, packed.data(), xin.data(), out.data(), T, rows, cols, 1.0f, workers); }, 2, reps);
    const double ms = best_us / 1000.0;
    const double tok_per_ms = ms > 0 ? double(T) / ms : 0;
    const double gflops = ms > 0 ? (2.0 * double(T) * double(rows) * double(cols)) / (ms / 1000.0) / 1e9 : 0;
    const char* kblk = std::getenv("ZEN_CPU_Q8_KBLK");
    const char* kc = std::getenv("ZEN_CPU_Q8_KC");
    std::printf("rows=%zu cols=%zu T=%zu workers=%u best=%.3f ms tok_per_ms=%.3f GFLOP/s=%.1f maxerr=%.4g KBLK=%s KC=%s\n",
                rows, cols, T, workers, ms, tok_per_ms, gflops, maxerr, kblk ? kblk : "(unset=legacy)", kc ? kc : "(unset=512)");
}
} // namespace

int main(int argc, char** argv) {
    int reps = 5;
    if (argc > 1) reps = std::max(1, std::atoi(argv[1]));
    const auto* fmt = zen::quant::find("Q8_0");
    if (!fmt) { std::printf("no Q8_0\n"); return 2; }
    if (!zen::cpu_has_avx2_fma()) { std::printf("no AVX2+FMA\n"); return 2; }
    zen::ResourceGovernor governor;
    std::printf("workers policy=%u\n", governor.current_policy().worker_threads);
    // FFN down (2048x8192, activation-thrash case) + gate/up (8192x2048).
    for (const std::size_t T : {std::size_t(8), std::size_t(32), std::size_t(128), std::size_t(171)}) {
        run_shape(fmt, {2048, 8192}, T, 0, reps);
        run_shape(fmt, {8192, 2048}, T, 0, reps);
    }
    // Single-thread isolation (no pool variance) for the down shape.
    for (const std::size_t T : {std::size_t(32), std::size_t(171)}) run_shape(fmt, {2048, 8192}, T, 1, reps);
    // Hyperthread check: the model planner uses every logical CPU (32) for
    // prefill; the governor default (workers=0 -> 16) is one per physical
    // core. K-blocked L2 residency assumes one thread per core, so measure
    // both at 32 workers to see the HT cost.
    for (const std::size_t T : {std::size_t(32), std::size_t(128), std::size_t(171)}) {
        run_shape(fmt, {2048, 8192}, T, 32, reps);
        run_shape(fmt, {8192, 2048}, T, 32, reps);
    }
    // Direct-kernel comparison, T=32 down shape (bit-identity spot check).
    {
#ifdef ZEN_HAS_AVX2_KERNELS
        const std::size_t rows = 2048, cols = 8192, T = 32;
        const std::size_t rb = zen::quant::row_bytes(*fmt, cols);
        std::vector<std::uint8_t> packed(rows * rb);
        std::uint32_t s = 777;
        auto rndb = [&]() { s = s * 1664525u + 1013904223u; return static_cast<std::uint8_t>((s >> 24) & 0xffu); };
        for (std::size_t r = 0; r < rows; ++r)
            for (std::size_t blk = 0; blk < cols / 32; ++blk) {
                std::uint8_t* b = packed.data() + r * rb + blk * 34;
                b[0] = 0x00; b[1] = 0x3c;
                for (std::size_t j = 2; j < 34; ++j) b[j] = rndb();
            }
        std::vector<float> xin(T * cols), o_legacy(T * rows, 0.0f), o_kblk(T * rows, 0.0f);
        for (std::size_t i = 0; i < xin.size(); ++i) xin[i] = float(int(i % 15) - 7) * 0.017f;
        zen::gemm_q8_0_avx2(packed.data(), xin.data(), o_legacy.data(), T, rows, cols, rows);
        zen::gemm_q8_0_kblk_avx2(packed.data(), xin.data(), o_kblk.data(), T, rows, cols, rows);
        std::size_t n_diff = 0; float maxdiff = 0;
        for (std::size_t i = 0; i < o_legacy.size(); ++i) {
            const float d = std::abs(o_legacy[i] - o_kblk[i]);
            if (d != 0) ++n_diff;
            if (d > maxdiff) maxdiff = d;
        }
        std::printf("direct rows=2048 cols=8192 T=32 legacy-vs-kblk n_diff=%zu/%zu maxdiff=%.6g %s\n",
                    n_diff, o_legacy.size(), maxdiff, (n_diff == 0 ? "BIT-IDENTICAL" : "DIFFERS"));
        // Repacked sequential variant: same packed matrix, repacked once,
        // timed through the repacked parallel wrapper vs the double
        // reference and the raw kblk/legacy outputs above.
        {
            auto repacked = zen::quant::repack_q8_0_kblk(packed.data(), rows, cols);
            std::printf("repack rows=%zu cols=%zu bytes=%zu raw_bytes=%zu repack_bytes=%zu\n",
                        rows, cols, repacked.size(), rows*rb, zen::quant::repacked_q8_0_kblk_bytes(rows, cols));
            std::vector<float> o_repack(T * rows, 0.0f);
            const bool ok = zen::quant::gemm_q8_0_kblk_repack_parallel(repacked.data(), xin.data(), o_repack.data(), T, rows, cols, 0);
            std::size_t n_diff_r = 0; float maxdiff_r = 0;
            for (std::size_t i = 0; i < o_legacy.size(); ++i) {
                const float d = std::abs(o_legacy[i] - o_repack[i]);
                if (d != 0) ++n_diff_r;
                if (d > maxdiff_r) maxdiff_r = d;
            }
            std::printf("direct rows=2048 cols=8192 T=32 repack ok=%d legacy-vs-repack n_diff=%zu/%zu maxdiff=%.6g %s\n",
                        ok ? 1 : 0, n_diff_r, o_legacy.size(), maxdiff_r, (ok && n_diff_r == 0 ? "BIT-IDENTICAL" : "DIFFERS"));
            const double rep_us = bench_best_us([&]() { zen::quant::gemm_q8_0_kblk_repack_parallel(repacked.data(), xin.data(), o_repack.data(), T, rows, cols, 0); }, 2, reps);
            std::printf("repack rows=%zu cols=%zu T=%zu workers=0 best=%.3f ms\n", rows, cols, T, rep_us / 1000.0);
        }
        // End-to-end shapes: repacked vs raw dispatch timings at the
        // prefill lengths that exercise the K-block threshold.
        for (const std::size_t TT : {std::size_t(128), std::size_t(171)}) {
            for (const Shape sh : {Shape{2048, 8192}, Shape{8192, 2048}}) {
                const std::size_t rr = sh.rows, cc = sh.cols;
                const std::size_t rb2 = zen::quant::row_bytes(*fmt, cc);
                std::vector<std::uint8_t> pk(rr * rb2);
                std::uint32_t ss = 4242u + std::uint32_t(rr + cc);
                auto rd = [&]() { ss = ss * 1664525u + 1013904223u; return static_cast<std::uint8_t>((ss >> 24) & 0xffu); };
                for (std::size_t r = 0; r < rr; ++r)
                    for (std::size_t blk = 0; blk < cc / 32; ++blk) {
                        std::uint8_t* b = pk.data() + r * rb2 + blk * 34;
                        b[0] = 0x00; b[1] = 0x3c;
                        for (std::size_t j = 2; j < 34; ++j) b[j] = rd();
                    }
                auto rp = zen::quant::repack_q8_0_kblk(pk.data(), rr, cc);
                std::vector<float> xx(TT * cc);
                for (std::size_t i = 0; i < xx.size(); ++i) xx[i] = float(int(i % 15) - 7) * 0.017f;
                std::vector<float> oo(TT * rr, 0.0f);
                const double t_repack = bench_best_us([&]() { zen::quant::gemm_q8_0_kblk_repack_parallel(rp.data(), xx.data(), oo.data(), TT, rr, cc, 0); }, 2, reps) / 1000.0;
                const double t_raw = bench_best_us([&]() { zen::quant::gemm(*fmt, pk.data(), xx.data(), oo.data(), TT, rr, cc, 1.0f, 0); }, 2, reps) / 1000.0;
                std::printf("shape rows=%zu cols=%zu T=%zu repack=%.3f ms raw_dispatch=%.3f ms (KBLK=%s)\n",
                            rr, cc, TT, t_repack, t_raw, std::getenv("ZEN_CPU_Q8_KBLK") ? std::getenv("ZEN_CPU_Q8_KBLK") : "unset");
            }
        }
#else
        (void)fmt;
#endif
    }
    return 0;
}
