// Parity harness for the D3D12 prefill elementwise GPU path: RMSNorm and SiLU.
// Both kernels are elementwise/reduction fp32 work that must match the CPU
// reference within the core_tests tolerance:
//   RMS : CPU rms_norm (kernels/avx2.cpp, double ss with 4-acc reorder) vs
//         the scalar double reference; gate is maxerr <= 1e-5 * max(1, |ref|).
//         The GPU kernel sums in fp32, so a few ulps of float are expected;
//         the gate is the same one core_tests uses for AVX2 vs scalar.
//   SiLU: CPU silu_mul (CEPHES poly in AVX2, scalar exp otherwise) vs the
//         scalar exp reference; measured maxabs ~4.77e-07, gate uses the
//         same 1e-5 relative form so a backend-wide tolerance is visible.
// The RMS check also covers the prefill shapes (rows = T, cols = hidden)
// and the in-place alias (xq norms normalise in place on the host).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <random>
#include <vector>

#include "zen/d3d12_backend.h"

namespace {

void rms_reference(const float* x, const float* w, float* y, std::size_t n, float eps) {
    double ss = 0.0;
    for (std::size_t i = 0; i < n; ++i) ss += double(x[i]) * x[i];
    const float inv = 1.0f / std::sqrt(static_cast<float>(ss / double(n)) + eps);
    for (std::size_t i = 0; i < n; ++i) y[i] = x[i] * inv * w[i];
}

void silu_reference(const float* gate, const float* up, float* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        const float z = gate[i];
        out[i] = (z / (1.0f + std::exp(-z))) * up[i];
    }
}

bool check_rms(std::size_t rows, std::size_t cols, std::uint32_t seed, float eps) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<float> x(rows * cols), w(cols), out(rows * cols, 0.0f), ref(rows * cols);
    for (auto& v : x) v = dist(rng);
    for (std::size_t i = 0; i < cols; ++i) w[i] = 1.0f + float(i % 5) * 0.01f;
    for (std::size_t t = 0; t < rows; ++t)
        rms_reference(x.data() + t * cols, w.data(), ref.data() + t * cols, cols, eps);

    if (!zen::d3d12::rms_norm(x.data(), w.data(), out.data(), rows, cols, eps)) {
        std::printf("FAIL: rms_norm(rows=%zu, cols=%zu) returned false\n", rows, cols);
        return false;
    }
    double maxabs = 0.0, maxref = 1.0;
    std::size_t worst = 0;
    for (std::size_t i = 0; i < out.size(); ++i) {
        const double e = std::fabs(double(out[i]) - double(ref[i]));
        if (e > maxabs) {
            maxabs = e;
            worst = i;
        }
        maxref = std::max(maxref, std::fabs(double(ref[i])));
    }
    const double tol = 1e-5 * maxref;
    const bool pass = maxabs <= tol;
    std::printf("rms rows=%-4zu cols=%-5zu maxabs=%.3e tol=%.3e %s\n", rows, cols, maxabs,
                tol, pass ? "ok" : "MISMATCH");
    if (!pass)
        std::printf("    worst i=%zu gpu=%.9g ref=%.9g\n", worst, double(out[worst]),
                    double(ref[worst]));
    // In-place alias (qk-norm writes through the same pointer).
    std::vector<float> inplace = x;
    if (!zen::d3d12::rms_norm(inplace.data(), w.data(), inplace.data(), rows, cols, eps)) {
        std::printf("FAIL: rms_norm in-place returned false (rows=%zu cols=%zu)\n", rows,
                    cols);
        return false;
    }
    double ip_abs = 0.0;
    for (std::size_t i = 0; i < inplace.size(); ++i)
        ip_abs = std::max(ip_abs, std::fabs(double(inplace[i]) - double(ref[i])));
    const bool ip_pass = ip_abs <= tol;
    std::printf("rms in-place rows=%-4zu cols=%-5zu maxabs=%.3e %s\n", rows, cols, ip_abs,
                ip_pass ? "ok" : "MISMATCH");
    return pass && ip_pass;
}

bool check_silu(std::size_t count, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
    std::vector<float> gate(count), up(count), out(count, 0.0f), ref(count);
    for (auto& v : gate) v = dist(rng);
    for (auto& v : up) v = dist(rng);
    silu_reference(gate.data(), up.data(), ref.data(), count);

    if (!zen::d3d12::silu_mul(gate.data(), up.data(), out.data(), count)) {
        std::printf("FAIL: silu_mul(count=%zu) returned false\n", count);
        return false;
    }
    double maxabs = 0.0, maxref = 1.0;
    std::size_t worst = 0;
    for (std::size_t i = 0; i < out.size(); ++i) {
        const double e = std::fabs(double(out[i]) - double(ref[i]));
        if (e > maxabs) {
            maxabs = e;
            worst = i;
        }
        maxref = std::max(maxref, std::fabs(double(ref[i])));
    }
    const double tol = 1e-5 * maxref;
    const bool pass = maxabs <= tol;
    std::printf("silu count=%-7zu maxabs=%.3e tol=%.3e %s\n", count, maxabs, tol,
                pass ? "ok" : "MISMATCH");
    if (!pass)
        std::printf("    worst i=%zu gpu=%.9g ref=%.9g gate=%.9g up=%.9g\n", worst,
                    double(out[worst]), double(ref[worst]), double(gate[worst]),
                    double(up[worst]));
    return pass;
}

}  // namespace

int main() {
    if (!zen::d3d12::enabled()) {
        std::printf("SKIP: no D3D12.1 backend\n");
        // 77 is ctest's SKIP_RETURN_CODE so an untested machine reports a skip
        // rather than a pass.
        return 77;
    }

    const float eps = 1e-5f;
    bool ok = true;
    // Prefill rows (T) x model widths, plus a head-dim (qk) shape and tails.
    ok = check_rms(1, 2048, 20261110u, eps) && ok;
    ok = check_rms(128, 2048, 20261111u, eps) && ok;
    ok = check_rms(171, 2048, 20261112u, eps) && ok;
    ok = check_rms(512, 2048, 20261113u, eps) && ok;
    ok = check_rms(64, 8192, 20261114u, eps) && ok;
    ok = check_rms(32, 64, 20261115u, eps) && ok;

    // FFN hidden sizes at prefill token counts, plus a tail-guard shape.
    ok = check_silu(128 * 8192, 20261120u) && ok;
    ok = check_silu(171 * 8192, 20261121u) && ok;
    ok = check_silu(512 * 8192, 20261122u) && ok;
    ok = check_silu(4097, 20261123u) && ok;

    if (!ok) {
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
