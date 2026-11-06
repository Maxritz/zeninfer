// Parity harness for the D3D12 device attention block (tasks 4 and 6 of
// docs/d3d12_full_backend_todo.md): KV cache append, RoPE, GQA attention.
//
// Calls zen::d3d12::attention_block once per position in ascending order, so the
// device-to-device cache append is exercised at every step, and compares each output
// against a CPU reference that keeps its own cache. Step t can only be right if every
// earlier step's keys and values were written to the right cache slot.
//
// Tolerance, not exactness: dxc contracts the rope multiply-adds into fma, and the
// device sums the score/softmax reductions in a different order than this reference.
// 2e-5 is far below any real indexing or binding error.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

#include "zen/d3d12_backend.h"

namespace {

constexpr double kTolerance = 2.0e-5;

void cpu_rope(std::vector<float>& x, const std::vector<float>& cos,
              const std::vector<float>& sin, std::size_t heads, std::size_t head_dim,
              std::size_t pairs, bool interleaved_pairs = false) {
    for (std::size_t h = 0; h < heads; ++h) {
        const std::size_t base = h * head_dim;
        for (std::size_t i = 0; i < pairs; ++i) {
            const float a = x[base + (interleaved_pairs ? 2 * i : i)];
            const float b = x[base + (interleaved_pairs ? 2 * i + 1 : i + pairs)];
            x[base + (interleaved_pairs ? 2 * i : i)] = a * cos[i] - b * sin[i];
            x[base + (interleaved_pairs ? 2 * i + 1 : i + pairs)] = a * sin[i] + b * cos[i];
        }
    }
}

bool run(std::size_t heads, std::size_t kv_heads, std::size_t head_dim, std::size_t pairs,
         std::size_t context, std::size_t steps, std::uint32_t seed,
         bool interleaved_pairs = false) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    if (!zen::d3d12::kv_cache_configure(1, context, kv_heads, head_dim)) {
        std::printf("FAIL: kv_cache_configure rejected layers=1 context=%zu kv_heads=%zu "
                    "head_dim=%zu\n",
                    context, kv_heads, head_dim);
        return false;
    }

    // One table per position, exactly as the CPU path builds it (index -> cos/sin).
    std::vector<std::vector<float>> cos_tab(steps, std::vector<float>(pairs));
    std::vector<std::vector<float>> sin_tab(steps, std::vector<float>(pairs));
    for (std::size_t p = 0; p < steps; ++p) {
        for (std::size_t i = 0; i < pairs; ++i) {
            const float freq = std::pow(10000.0f, -2.0f * float(i) / float(pairs * 2));
            const float angle = float(p) * freq;
            cos_tab[p][i] = std::cos(angle);
            sin_tab[p][i] = std::sin(angle);
        }
    }

    const std::size_t q_width = heads * head_dim;
    const std::size_t kv_width = kv_heads * head_dim;
    std::vector<float> kcache(context * kv_width, 0.0f);
    std::vector<float> vcache(context * kv_width, 0.0f);

    double max_abs = 0.0;
    for (std::size_t p = 0; p < steps; ++p) {
        std::vector<float> q(q_width), k(kv_width), v(kv_width);
        for (float& e : q) {
            e = dist(rng);
        }
        for (float& e : k) {
            e = dist(rng);
        }
        for (float& e : v) {
            e = dist(rng);
        }

        // CPU reference: rope, cache append, then GQA over every visible position.
        std::vector<float> qr = q, kr = k;
        cpu_rope(qr, cos_tab[p], sin_tab[p], heads, head_dim, pairs, interleaved_pairs);
        cpu_rope(kr, cos_tab[p], sin_tab[p], kv_heads, head_dim, pairs, interleaved_pairs);
        std::copy(kr.begin(), kr.end(), kcache.begin() + p * kv_width);
        std::copy(v.begin(), v.end(), vcache.begin() + p * kv_width);

        std::vector<float> ref(q_width, 0.0f);
        const std::size_t group = heads / kv_heads;
        const float inv_scale = 1.0f / std::sqrt(float(head_dim));
        std::vector<double> scores(p + 1);
        for (std::size_t h = 0; h < heads; ++h) {
            const std::size_t kh = h / group;
            double mx = -std::numeric_limits<double>::infinity();
            for (std::size_t t = 0; t <= p; ++t) {
                double acc = 0.0;
                for (std::size_t d = 0; d < head_dim; ++d) {
                    acc += double(qr[h * head_dim + d]) *
                           double(kcache[t * kv_width + kh * head_dim + d]);
                }
                scores[t] = acc * double(inv_scale);
                mx = std::max(mx, scores[t]);
            }
            double denom = 0.0;
            for (std::size_t t = 0; t <= p; ++t) {
                scores[t] = std::exp(scores[t] - mx);
                denom += scores[t];
            }
            for (std::size_t d = 0; d < head_dim; ++d) {
                double acc = 0.0;
                for (std::size_t t = 0; t <= p; ++t) {
                    acc += (scores[t] / denom) * double(vcache[t * kv_width + kh * head_dim + d]);
                }
                ref[h * head_dim + d] = float(acc);
            }
        }

        std::vector<float> got(q_width, 0.0f);
        if (!zen::d3d12::attention_block(q.data(), k.data(), v.data(), cos_tab[p].data(),
                                         sin_tab[p].data(), got.data(), heads, kv_heads, head_dim,
                                         pairs, 0, p, 0, interleaved_pairs)) {
            std::printf("FAIL: attention_block returned false at pos=%zu\n", p);
            return false;
        }

        for (std::size_t i = 0; i < q_width; ++i) {
            max_abs = std::max(max_abs, std::fabs(double(got[i]) - double(ref[i])));
        }
    }

    const bool pass = max_abs <= kTolerance;
    std::printf("attn %s heads=%-2zu kv=%-2zu d=%-3zu pairs=%-2zu ctx=%-3zu steps=%-3zu "
                "max_abs_err=%.3e %s\n",
                interleaved_pairs ? "adjacent-pairs" : "             ", heads, kv_heads,
                head_dim, pairs, context, steps, max_abs, pass ? "ok" : "MISMATCH");
    return pass;
}

}  // namespace

int main() {
    if (!zen::d3d12::enabled()) {
        std::printf("SKIP: no D3D12.1 backend\n");
        return 77;
    }

    bool ok = true;
    // Decode from the very first token, so the append at pos 0 is covered.
    ok = run(4, 2, 8, 4, 16, 6, 20261110u) && ok;
    // Llama-3.2-1B real shape, short window.
    ok = run(32, 8, 64, 32, 32, 8, 20261111u) && ok;
    // A single KV head shared by every query head (MQA).
    ok = run(8, 1, 16, 8, 12, 5, 20261112u) && ok;
    // Adjacent-pairs layout (llama.cpp NORM rope, RopeParams.pad0=1). Regression
    // guard for the RoPE pairing-convention fix (RUN-004).
    ok = run(4, 2, 8, 4, 16, 6, 20261113u, true) && ok;
    ok = run(32, 8, 64, 32, 32, 8, 20261114u, true) && ok;

    if (!ok) {
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
