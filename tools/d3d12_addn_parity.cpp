// Parity harness for the D3D12 residual-add path: out[i] = a[i] + b[i].
//
// This replaces an earlier stand-in harness that called rms_norm and compared against
// an (a + b) - row_mean reference, which could never pass and never exercised the add
// kernel at all. The elementwise add has no reduction and no shared memory, so an
// exact fp32 match is expected; any deviation is a real indexing or binding bug.

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <random>
#include <vector>

#include "zen/d3d12_backend.h"

namespace {

bool check(std::size_t count, std::uint32_t seed) {
    std::mt19937 rng(seed);
    // Dyadic values keep every sum exactly representable in fp32, so this compares
    // arithmetic, not rounding.
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> a(count), b(count), out(count, 0.0f), ref(count);
    for (std::size_t i = 0; i < count; ++i) {
        a[i] = std::round(dist(rng) * 256.0f) / 256.0f;
        b[i] = std::round(dist(rng) * 256.0f) / 256.0f;
        ref[i] = a[i] + b[i];
    }

    if (!zen::d3d12::residual_add(a.data(), b.data(), out.data(), count)) {
        std::printf("FAIL: residual_add returned false for count=%zu\n", count);
        return false;
    }

    double max_abs = 0.0;
    std::size_t worst = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const double e = std::fabs(double(out[i]) - double(ref[i]));
        if (e > max_abs) {
            max_abs = e;
            worst = i;
        }
    }
    const bool pass = max_abs == 0.0;
    std::printf("addn count=%-6zu max_abs_err=%.3e %s\n", count, max_abs,
                pass ? "ok" : "MISMATCH");
    if (!pass) {
        std::printf("    worst i=%zu gpu=%.9g ref=%.9g a=%.9g b=%.9g\n", worst, double(out[worst]),
                    double(ref[worst]), double(a[worst]), double(b[worst]));
    }
    return pass;
}

}  // namespace

int main() {
    if (!zen::d3d12::enabled()) {
        std::printf("SKIP: no D3D12.1 backend\n");
        // 77 is ctest's SKIP_RETURN_CODE so an untested machine reports a skip rather
        // than a pass.
        return 77;
    }

    bool ok = true;
    // Non-multiples of the 256-thread group exercise the tail guard.
    ok = check(1024, 20261102u) && ok;
    ok = check(65536, 20261103u) && ok;
    ok = check(1023, 20261104u) && ok;
    ok = check(4097, 20261105u) && ok;

    if (!ok) {
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
