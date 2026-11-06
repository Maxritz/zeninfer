// Parity harness for the D3D12 RoPE paths (task 5 of docs/d3d12_full_backend_todo.md).
//
// Compares zen::d3d12::rope_apply (decode, single token) AND
// zen::d3d12::rope_apply_prefill (prefill, batched T tokens with a per-token
// table) against the CPU reference rotation the runtime uses:
// within a head, element i pairs with i + pairs (NeoX) or 2i/2i+1 (adjacent),
// and the pair rotates as
//   (a*cos - b*sin, a*sin + b*cos)
// with the cos/sin table indexed by dimension (and token, for prefill).
//
// Tolerance, not exactness: dxc contracts a*c - b*s into an fma at -O3, which rounds
// differently from the two separate float multiplies here. The CPU path tolerates that
// (it is the same class of difference the matvec kernels already produce), but an
// exact-match assertion would fail for a reason that is not a bug. Indexing or binding
// errors move values by far more than this bound. Prefill results sit in the same
// ~1e-7 band as decode (max_abs_err 1.2e-07 in prior runs), i.e. rel <=1e-7 for
// unit-magnitude rope values.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <random>
#include <vector>

#include "zen/d3d12_backend.h"

namespace {

// 1e-6 absolute is well below the ~1e-7 fma/separate-multiply difference and far above
// any measurement noise, so only real bugs trip it.
constexpr double kTolerance = 1.0e-5;

void cpu_rope(const std::vector<float>& x, const std::vector<float>& cos,
              const std::vector<float>& sin, std::vector<float>& out, std::size_t heads,
              std::size_t head_dim, std::size_t pairs, bool interleaved_pairs = false) {
    for (std::size_t h = 0; h < heads; ++h) {
        const std::size_t base = h * head_dim;
        for (std::size_t i = 0; i < pairs; ++i) {
            const float a = x[base + (interleaved_pairs ? 2 * i : i)];
            const float b = x[base + (interleaved_pairs ? 2 * i + 1 : i + pairs)];
            out[base + (interleaved_pairs ? 2 * i : i)] = a * cos[i] - b * sin[i];
            out[base + (interleaved_pairs ? 2 * i + 1 : i + pairs)] = a * sin[i] + b * cos[i];
        }
        // Dimensions past rotary_dim are passed through untouched; leaving them zero
        // here would make the harness expect a rotation the kernel must not perform.
        for (std::size_t i = pairs * 2; i < head_dim; ++i) {
            out[base + i] = x[base + i];
        }
    }
}

bool check(std::size_t heads, std::size_t head_dim, std::size_t pairs, std::uint32_t seed,
           bool interleaved_pairs = false) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> x(heads * head_dim), cos(pairs), sin(pairs);
    for (float& v : x) {
        v = dist(rng);
    }
    // A real table is cos/sin of an angle, so keep each pair on the unit circle.
    for (std::size_t i = 0; i < pairs; ++i) {
        const float angle = dist(rng) * 3.14159265f;
        cos[i] = std::cos(angle);
        sin[i] = std::sin(angle);
    }

    std::vector<float> out(heads * head_dim, 0.0f), ref(heads * head_dim, 0.0f);
    cpu_rope(x, cos, sin, ref, heads, head_dim, pairs, interleaved_pairs);

    if (!zen::d3d12::rope_apply(x.data(), cos.data(), sin.data(), out.data(), heads, head_dim,
                                pairs, interleaved_pairs)) {
        std::printf("FAIL: rope_apply returned false (heads=%zu head_dim=%zu pairs=%zu interleaved_pairs=%d)\n", heads,
                    head_dim, pairs, interleaved_pairs ? 1 : 0);
        return false;
    }

    double max_abs = 0.0;
    std::size_t worst = 0;
    for (std::size_t i = 0; i < out.size(); ++i) {
        const double e = std::fabs(double(out[i]) - double(ref[i]));
        if (e > max_abs) {
            max_abs = e;
            worst = i;
        }
    }
    const bool pass = max_abs <= kTolerance;
    std::printf("rope %s heads=%-3zu head_dim=%-4zu pairs=%-3zu max_abs_err=%.3e %s\n",
                interleaved_pairs ? "adjacent-pairs" : "              ", heads,
                head_dim, pairs, max_abs, pass ? "ok" : "MISMATCH");
    if (!pass) {
        std::printf("    worst i=%zu gpu=%.9g ref=%.9g\n", worst, double(out[worst]),
                    double(ref[worst]));
    }
    return pass;
}

// Batched prefill check: x is [tokens, heads, head_dim] token-major, cos/sin are
// [tokens * pairs] token-major. Each token's slice must rotate with ITS OWN table
// exactly as the CPU loop does per token (rope_apply_table semantics). The Q shape
// (32 heads) and K shape (8 heads) are exercised, plus partial rotary and an odd
// group shape, for both pairings.
bool check_prefill(std::size_t tokens, std::size_t heads, std::size_t head_dim,
                   std::size_t pairs, std::uint32_t seed, bool interleaved_pairs = false) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const std::size_t per_token = heads * head_dim;
    std::vector<float> x(tokens * per_token), cos(tokens * pairs), sin(tokens * pairs);
    for (float& v : x) {
        v = dist(rng);
    }
    for (std::size_t t = 0; t < tokens; ++t) {
        for (std::size_t i = 0; i < pairs; ++i) {
            const float angle = dist(rng) * 3.14159265f;
            cos[t * pairs + i] = std::cos(angle);
            sin[t * pairs + i] = std::sin(angle);
        }
    }

    std::vector<float> out(tokens * per_token, 0.0f), ref(tokens * per_token, 0.0f);
    for (std::size_t t = 0; t < tokens; ++t) {
        // CPU reference for this token uses only slice t of the tables.
        std::vector<float> xt(x.begin() + t * per_token, x.begin() + (t + 1) * per_token);
        std::vector<float> co(cos.begin() + t * pairs, cos.begin() + (t + 1) * pairs);
        std::vector<float> si(sin.begin() + t * pairs, sin.begin() + (t + 1) * pairs);
        std::vector<float> rt(per_token, 0.0f);
        cpu_rope(xt, co, si, rt, heads, head_dim, pairs, interleaved_pairs);
        std::copy_n(rt.begin(), per_token, ref.begin() + t * per_token);
    }

    if (!zen::d3d12::rope_apply_prefill(x.data(), cos.data(), sin.data(), out.data(),
                                        tokens, heads, head_dim, pairs,
                                        interleaved_pairs)) {
        std::printf(
            "FAIL: rope_apply_prefill returned false "
            "(tokens=%zu heads=%zu head_dim=%zu pairs=%zu interleaved_pairs=%d)\n",
            tokens, heads, head_dim, pairs, interleaved_pairs ? 1 : 0);
        return false;
    }

    double max_abs = 0.0, max_ref = 0.0;
    std::size_t worst = 0;
    for (std::size_t i = 0; i < out.size(); ++i) {
        const double e = std::fabs(double(out[i]) - double(ref[i]));
        if (e > max_abs) {
            max_abs = e;
            worst = i;
        }
        max_ref = std::max(max_ref, std::fabs(double(ref[i])));
    }
    // Relative error against the largest reference magnitude in the batch: for
    // unit-range rope values this is ~ max_abs itself, so it tracks the 1e-7 band.
    const double rel = max_ref > 0.0 ? max_abs / max_ref : max_abs;
    const bool pass = max_abs <= kTolerance && rel <= 1.0e-6;
    std::printf("rope-prefill %s tokens=%-3zu heads=%-3zu head_dim=%-4zu pairs=%-3zu "
                "max_abs_err=%.3e rel=%.3e %s\n",
                interleaved_pairs ? "adjacent-pairs " : "               ", tokens, heads,
                head_dim, pairs, max_abs, rel, pass ? "ok" : "MISMATCH");
    if (!pass) {
        std::printf("    worst i=%zu gpu=%.9g ref=%.9g\n", worst, double(out[worst]),
                    double(ref[worst]));
    }
    return pass;
}

}  // namespace

int main() {
    if (!zen::d3d12::enabled()) {
        std::printf("SKIP: no D3D12.1 backend\n");
        return 77;
    }

    bool ok = true;
    // Llama-3.2-1B shape: 32 heads, 64-wide heads, full rotary.
    ok = check(32, 64, 32, 20261106u) && ok;
    // 8 KV heads, the GQA shape the k tensor uses.
    ok = check(8, 64, 32, 20261107u) && ok;
    // Partial rotary: dimensions past rotary_dim must pass through unchanged.
    ok = check(32, 128, 32, 20261108u) && ok;
    // Total pair count not a multiple of the 256-thread group, exercising the guard.
    ok = check(3, 64, 17, 20261109u) && ok;
    // Adjacent-pairs layout (llama.cpp NORM rope, RopeParams.pad0=1): same shapes,
    // consecutive-element pairing, same table. Regression guard for the RoPE
    // pairing-convention fix (RUN-004).
    ok = check(32, 64, 32, 20261110u, true) && ok;
    ok = check(8, 64, 32, 20261111u, true) && ok;
    ok = check(3, 64, 17, 20261112u, true) && ok;

    // Batched prefill rope: T x heads x head_dim with per-token tables. T=1 must
    // agree with the decode kernel; larger T proves the token stride/offset math.
    // Both pairings, both head counts, partial rotary, and a 128-token prefill.
    ok = check_prefill(1, 32, 64, 32, 20261113u) && ok;
    ok = check_prefill(4, 32, 64, 32, 20261114u) && ok;
    ok = check_prefill(8, 32, 64, 32, 20261115u) && ok;
    ok = check_prefill(16, 32, 64, 32, 20261116u) && ok;
    ok = check_prefill(32, 32, 64, 32, 20261117u) && ok;
    ok = check_prefill(128, 32, 64, 32, 20261118u) && ok;
    ok = check_prefill(128, 8, 64, 32, 20261119u) && ok;
    ok = check_prefill(37, 8, 64, 32, 20261120u) && ok;  // non-power-of-two T
    ok = check_prefill(16, 32, 128, 32, 20261121u) && ok;  // partial rotary
    ok = check_prefill(1, 32, 64, 32, 20261122u, true) && ok;
    ok = check_prefill(4, 32, 64, 32, 20261123u, true) && ok;
    ok = check_prefill(8, 32, 64, 32, 20261124u, true) && ok;
    ok = check_prefill(16, 32, 64, 32, 20261125u, true) && ok;
    ok = check_prefill(32, 32, 64, 32, 20261126u, true) && ok;
    ok = check_prefill(128, 32, 64, 32, 20261127u, true) && ok;
    ok = check_prefill(128, 8, 64, 32, 20261128u, true) && ok;
    ok = check_prefill(37, 8, 64, 32, 20261129u, true) && ok;
    ok = check_prefill(16, 32, 128, 32, 20261130u, true) && ok;

    if (!ok) {
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
