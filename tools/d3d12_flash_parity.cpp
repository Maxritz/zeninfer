// Parity harness for FlashAttention prefill (zen::d3d12::flash_attention_prefill).
//
// The kernel reads K/V out of the device KV cache, so this test mirrors the real
// call sequence: configure the cache, append every token's K/V through the
// public append path, then run the whole prefill batch in one dispatch and
// compare each (token, head) row against a CPU reference.
//
// This is the test that would have caught the original TDR: the old kernel
// reported SUCCESS on layers 0-2 while returning garbage, because nothing
// compared its output to a reference. Speed is not evidence of correctness.
//
// Reference is the same online-softmax-free form the CPU runtime uses: full
// max, then exp, then normalize. Tolerance covers the different summation order
// and the fma contraction dxc performs, not an indexing error.

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

bool run(std::size_t tokens, std::size_t heads, std::size_t kv_heads, std::size_t head_dim,
         std::size_t context, std::uint32_t seed, const char* label) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const std::size_t q_width = heads * head_dim;
    const std::size_t kv_width = kv_heads * head_dim;
    const float inv_scale = 1.0f / std::sqrt(float(head_dim));

    if (!zen::d3d12::kv_cache_configure(1, context, kv_heads, head_dim)) {
        std::printf("FAIL[%s]: kv_cache_configure layers=1 context=%zu kv_heads=%zu head_dim=%zu\n",
                    label, context, kv_heads, head_dim);
        return false;
    }

    std::vector<float> q(tokens * q_width);
    for (float& e : q) {
        e = dist(rng);
    }

    // Host-side K/V cache, written with the same append path the device uses.
    std::vector<float> host_k(context * kv_width, 0.0f);
    std::vector<float> host_v(context * kv_width, 0.0f);

    std::vector<std::size_t> layers(tokens);
    std::vector<std::size_t> positions(tokens);
    std::vector<const float*> ks(tokens);
    std::vector<const float*> vs(tokens);
    std::vector<std::vector<float>> k_storage(tokens, std::vector<float>(kv_width));
    std::vector<std::vector<float>> v_storage(tokens, std::vector<float>(kv_width));

    for (std::size_t t = 0; t < tokens; ++t) {
        for (float& e : k_storage[t]) {
            e = dist(rng);
        }
        for (float& e : v_storage[t]) {
            e = dist(rng);
        }
        std::copy(k_storage[t].begin(), k_storage[t].end(), host_k.begin() + t * kv_width);
        std::copy(v_storage[t].begin(), v_storage[t].end(), host_v.begin() + t * kv_width);
        layers[t] = 0;
        positions[t] = t;
        ks[t] = k_storage[t].data();
        vs[t] = v_storage[t].data();
    }

    if (!zen::d3d12::kv_cache_append_batched(layers.data(), tokens, positions.data(), ks.data(),
                                             vs.data(), kv_heads, head_dim)) {
        std::printf("FAIL[%s]: kv_cache_append_batched\n", label);
        return false;
    }

    // CPU reference: causal GQA, full max then normalize.
    std::vector<float> ref(tokens * q_width, 0.0f);
    const std::size_t group = heads / kv_heads;
    std::vector<double> scores(context);
    for (std::size_t t = 0; t < tokens; ++t) {
        for (std::size_t h = 0; h < heads; ++h) {
            const std::size_t kh = h / group;
            double mx = -std::numeric_limits<double>::infinity();
            for (std::size_t j = 0; j <= t; ++j) {
                double acc = 0.0;
                for (std::size_t d = 0; d < head_dim; ++d) {
                    acc += double(q[t * q_width + h * head_dim + d]) *
                           double(host_k[j * kv_width + kh * head_dim + d]);
                }
                scores[j] = acc * double(inv_scale);
                mx = std::max(mx, scores[j]);
            }
            double denom = 0.0;
            for (std::size_t j = 0; j <= t; ++j) {
                scores[j] = std::exp(scores[j] - mx);
                denom += scores[j];
            }
            for (std::size_t d = 0; d < head_dim; ++d) {
                double acc = 0.0;
                for (std::size_t j = 0; j <= t; ++j) {
                    acc += (scores[j] / denom) * double(host_v[j * kv_width + kh * head_dim + d]);
                }
                ref[t * q_width + h * head_dim + d] = float(acc);
            }
        }
    }

    std::vector<float> got(tokens * q_width, 0.0f);
    if (!zen::d3d12::flash_attention_prefill(q.data(), nullptr, nullptr, got.data(), tokens, heads,
                                             kv_heads, head_dim, inv_scale, 0, 0, context,
                                             kv_heads, head_dim)) {
        std::printf("FAIL[%s]: flash_attention_prefill returned false\n", label);
        return false;
    }

    double max_abs = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i])) {
            std::printf("FAIL[%s]: non-finite output at %zu\n", label, i);
            return false;
        }
        max_abs = std::max(max_abs, std::fabs(double(got[i]) - double(ref[i])));
    }

    const bool pass = max_abs <= kTolerance;
    std::printf("flash tokens=%-4zu heads=%-2zu kv=%-2zu d=%-3zu ctx=%-4zu max_abs_err=%.3e %s\n",
                tokens, heads, kv_heads, head_dim, context, max_abs, pass ? "ok" : "MISMATCH");
    return pass;
}

}  // namespace

int main() {
    if (!zen::d3d12::enabled()) {
        std::printf("SKIP: no D3D12.1 backend\n");
        return 77;
    }

    bool ok = true;
    // Llama-3.2-1B real shape, single prefill batch.
    ok = run(128, 32, 8, 64, 256, 20261120u, "llama-prefill") && ok;
    // Qwen3-4B real shape: head_dim == 128 and 4 query heads per KV head. That pair is
    // what the keys-across-lanes kernel requires, so without this case it is covered
    // only by the model-level token-id oracles.
    ok = run(128, 32, 8, 128, 256, 20261124u, "qwen3-prefill") && ok;
    // Same shape, one token (degenerate: softmax over a single key).
    ok = run(1, 32, 8, 64, 256, 20261121u, "single-token") && ok;
    // MQA: every query head shares one KV head.
    ok = run(32, 8, 1, 16, 128, 20261122u, "mqa") && ok;
    // Non-multiple-of-32 token count, to catch block-edge masking.
    ok = run(37, 4, 2, 32, 128, 20261123u, "ragged") && ok;

    if (!ok) {
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
