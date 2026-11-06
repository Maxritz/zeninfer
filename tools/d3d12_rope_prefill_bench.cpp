// CPU-vs-GPU crossover bench for batched prefill RoPE.
//
// What it measures: for each T in {1,2,4,8,16,32,128}, the time to rope Q and K
// at Llama-3.2-1B shapes (H=32, KH=8, D=64, pairs=32) with (a) the OLD CPU rope
// (rope() rebuilds its table per (token, Q/K) call, as pre-change forward_batch
// did), (b) the hoisted CPU rope (rope_apply_table, tables prebuilt once), and
// (c) the GPU prefill rope (zen::d3d12::rope_apply_prefill, one dispatch per
// tensor). The wall includes the submit + fence wait, so this is the honest
// end-to-end saving the runtime sees over 16 layers, not just kernel time.
//
// Expected: GPU loses at tiny T (fixed ~70 us submit/queue per dispatch vs
// ~1-2 us of CPU rotation per token). Whether GPU beats hoisted CPU at large T
// decides the shippable gate ZEN_D3D12_ROPE_PREFILL_MIN_T. Tables are built once
// per T with the same rope_table inputs the runtime uses, so the comparison is
// arithmetic-only.
//
// Usage: zen_d3d12_rope_prefill_bench
//   ZEN_BACKEND=d3d12  (required; exit 77 without a device)

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "zen/d3d12_backend.h"

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

// Hoisted CPU rope for one token (same arithmetic as the runtime's helper).
void cpu_rope_token(float* x, std::size_t heads, std::size_t head_dim,
                    std::size_t pairs, const float* cos, const float* sin,
                    bool adjacent) {
    for (std::size_t h = 0; h < heads; ++h) {
        float* v = x + h * head_dim;
        for (std::size_t i = 0; i < pairs; ++i) {
            const float c = cos[i], s = sin[i];
            const float a = adjacent ? v[2 * i] : v[i];
            const float b = adjacent ? v[2 * i + 1] : v[i + pairs];
            if (adjacent) {
                v[2 * i] = a * c - b * s;
                v[2 * i + 1] = a * s + b * c;
            } else {
                v[i] = a * c - b * s;
                v[i + pairs] = a * s + b * c;
            }
        }
    }
}

// OLD CPU rope: rebuild the per-dimension table (pow/cos/sin) on every call,
// exactly as pre-change rope() did. One call covers one tensor (Q or K) for one
// token. This is the correct baseline for "CPU rope -> GPU" savings claims:
// the hoisted table is a separate optimisation.
void cpu_rope_old(float* x, std::size_t heads, std::size_t head_dim,
                  std::size_t pairs, std::uint32_t pos, bool adjacent) {
    const float base = 500000.0f;  // Llama-3.2 rope.freq_base
    const std::size_t half = pairs;
    std::vector<float> cos(half), sin(half);
    for (std::size_t i = 0; i < half; ++i) {
        const float freq = std::pow(base, -2.0f * float(i) / float(head_dim));
        const float angle = float(pos) * freq;
        cos[i] = std::cos(angle);
        sin[i] = std::sin(angle);
    }
    cpu_rope_token(x, heads, head_dim, pairs, cos.data(), sin.data(), adjacent);
}

}  // namespace

int main() {
    if (!zen::d3d12::enabled()) {
        std::printf("SKIP: no D3D12.1 backend\n");
        return 77;
    }

    constexpr std::size_t kH = 32, kKH = 8, kD = 64, kPairs = 32;
    constexpr int kReps = 5;
    const std::size_t tokens_list[] = {1, 2, 4, 8, 16, 32, 128};

    std::mt19937 rng(12345u);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::printf("T    %-14s %-14s %-14s %-10s %-10s %s\n", "CPU_old_ms", "CPU_hoist_ms",
                "GPU_ms", "old->GPU", "hoist->GPU", "note");
    std::printf("---- ------------- ------------- ------------- ----------- ----------- %s\n",
                "----");

    for (std::size_t T : tokens_list) {
        const std::size_t qw = T * kH * kD;
        const std::size_t kw = T * kKH * kD;
        std::vector<float> q0(qw), k0(kw), cos(T * kPairs), sin(T * kPairs);
        for (float& v : q0) v = dist(rng);
        for (float& v : k0) v = dist(rng);
        for (std::size_t t = 0; t < T; ++t) {
            for (std::size_t i = 0; i < kPairs; ++i) {
                const float ang = dist(rng) * 3.14159265f;
                cos[t * kPairs + i] = std::cos(ang);
                sin[t * kPairs + i] = std::sin(ang);
            }
        }

        std::vector<float> q = q0, k = k0;
        double cpu_old_best = 1e30, cpu_best = 1e30, gpu_best = 1e30;

        // OLD: rebuild tables per (token, Q/K), as before the change.
        for (int r = 0; r < kReps; ++r) {
            q = q0;
            k = k0;
            const auto t0 = Clock::now();
            for (std::size_t t = 0; t < T; ++t) {
                cpu_rope_old(q.data() + t * kH * kD, kH, kD, kPairs,
                             static_cast<std::uint32_t>(t), true);
                cpu_rope_old(k.data() + t * kKH * kD, kKH, kD, kPairs,
                             static_cast<std::uint32_t>(t), true);
            }
            cpu_old_best = std::min(cpu_old_best, ms_since(t0));
        }

        // Hoisted CPU: copy inputs per rep so rotation starts from the same state.
        for (int r = 0; r < kReps; ++r) {
            q = q0;
            k = k0;
            const auto t0 = Clock::now();
            for (std::size_t t = 0; t < T; ++t) {
                cpu_rope_token(q.data() + t * kH * kD, kH, kD, kPairs,
                               cos.data() + t * kPairs, sin.data() + t * kPairs, true);
                cpu_rope_token(k.data() + t * kKH * kD, kKH, kD, kPairs,
                               cos.data() + t * kPairs, sin.data() + t * kPairs, true);
            }
            cpu_best = std::min(cpu_best, ms_since(t0));
        }

        // GPU: one rope_apply_prefill per tensor (the runtime's shape: two submits
        // per layer). Outputs are in place; re-copy inputs per rep.
        bool gpu_ok = true;
        for (int r = 0; r < kReps; ++r) {
            q = q0;
            k = k0;
            const auto t0 = Clock::now();
            const bool okq =
                zen::d3d12::rope_apply_prefill(q.data(), cos.data(), sin.data(), q.data(),
                                               T, kH, kD, kPairs, true);
            const bool okk =
                zen::d3d12::rope_apply_prefill(k.data(), cos.data(), sin.data(), k.data(),
                                               T, kKH, kD, kPairs, true);
            gpu_ok = gpu_ok && okq && okk;
            gpu_best = std::min(gpu_best, ms_since(t0));
        }

        const double sp_old = gpu_ok && gpu_best > 0 ? cpu_old_best / gpu_best : 0.0;
        const double sp_hoist = gpu_ok && gpu_best > 0 ? cpu_best / gpu_best : 0.0;
        std::printf("%-4zu %-14.4f %-14.4f %-14.4f %-10.2fx %-10.2fx %s\n", T, cpu_old_best,
                    cpu_best, gpu_best, sp_old, sp_hoist,
                    gpu_ok ? (sp_hoist >= 1.0 ? "GPU beats hoisted" : "CPU wins")
                           : "GPU unavailable");
        if (!gpu_ok) {
            std::printf("FAIL: rope_apply_prefill failed at T=%zu\n", T);
            return 1;
        }
        // Sanity: last-rep outputs must still match the CPU rotation.
        // (Full parity is gated by zen_d3d12_rope_parity; this checks the bench
        // did not leave stale outputs.)
        std::vector<float> qr = q0;
        for (std::size_t t = 0; t < T; ++t) {
            cpu_rope_token(qr.data() + t * kH * kD, kH, kD, kPairs,
                           cos.data() + t * kPairs, sin.data() + t * kPairs, true);
        }
        double max_abs = 0.0;
        for (std::size_t i = 0; i < qr.size(); ++i) {
            max_abs = std::max(max_abs, double(std::fabs(q[i] - qr[i])));
        }
        if (max_abs > 1.0e-5) {
            std::printf("FAIL: bench sanity max_abs=%.3e at T=%zu\n", max_abs, T);
            return 1;
        }
    }

    std::printf("PASS\n");
    return 0;
}
