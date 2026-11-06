// CPU-vs-GPU parity check for the HIP Q8_0 matrix-vector kernel.
//
// The end-to-end token comparison only proves the pipeline agrees; this pins the
// kernel against a double-precision reference so a kernel or repack regression is
// caught even when sampling happens to be robust to small errors.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "zen/hip_backend.h"

namespace {

// Exact for the scales used below (m * 2^-8 for small integer m).
std::uint16_t float_to_half(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    int exponent = static_cast<int>((bits >> 23) & 0xFFu) - 127 + 15;
    std::uint32_t mantissa = bits & 0x7FFFFFu;
    if (exponent <= 0) {
        return static_cast<std::uint16_t>(sign);
    }
    if (exponent >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7C00u);
    }
    double scaled = static_cast<double>(mantissa) / static_cast<double>(1u << 23);
    std::uint32_t half_mantissa = static_cast<std::uint32_t>(scaled * 1024.0 + 0.5) & 0x3FFu;
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exponent) << 10) |
                                      half_mantissa);
}

float half_to_float(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    const std::uint32_t exp = (h >> 10) & 0x1Fu;
    const std::uint32_t mant = h & 0x3FFu;
    std::uint32_t bits;
    if (exp == 0) {
        bits = sign;
        if (mant) {
            std::uint32_t m = mant;
            int e = -1;
            do {
                m <<= 1;
                ++e;
            } while ((m & 0x400u) == 0);
            bits = sign | (static_cast<std::uint32_t>(127 - 15 - e) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

struct Fixture {
    std::size_t rows{};
    std::size_t cols{};
    std::size_t blocks_per_row{};
    std::size_t row_bytes{};
    std::vector<std::uint8_t> weights;
    std::vector<float> x;
};

Fixture make_fixture(std::size_t rows, std::size_t cols, std::uint32_t seed) {
    Fixture f;
    f.rows = rows;
    f.cols = cols;
    f.blocks_per_row = cols / 32;
    f.row_bytes = f.blocks_per_row * 34;
    f.weights.assign(rows * f.row_bytes, 0);
    f.x.resize(cols);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> quant_dist(-127, 127);
    std::uniform_real_distribution<float> x_dist(-1.0f, 1.0f);

    for (float& v : f.x) {
        v = x_dist(rng);
    }
    for (std::size_t r = 0; r < rows; ++r) {
        std::uint8_t* row = f.weights.data() + r * f.row_bytes;
        for (std::size_t b = 0; b < f.blocks_per_row; ++b) {
            std::uint8_t* block = row + b * 34;
            // (1 + (b % 8)) * 2^-8 is exact in fp16 for every branch.
            const float scale = static_cast<float>(1 + (b % 8)) / 256.0f;
            const std::uint16_t half = float_to_half(scale);
            block[0] = static_cast<std::uint8_t>(half & 0xFFu);
            block[1] = static_cast<std::uint8_t>(half >> 8);
            for (std::size_t k = 0; k < 32; ++k) {
                block[2 + k] =
                    static_cast<std::uint8_t>(static_cast<std::int8_t>(quant_dist(rng)));
            }
        }
    }
    return f;
}

void reference_gemv(const Fixture& f, std::vector<float>& out) {
    out.assign(f.rows, 0.0f);
    for (std::size_t r = 0; r < f.rows; ++r) {
        const std::uint8_t* row = f.weights.data() + r * f.row_bytes;
        double acc = 0.0;
        for (std::size_t b = 0; b < f.blocks_per_row; ++b) {
            const std::uint8_t* block = row + b * 34;
            const float scale =
                half_to_float(static_cast<std::uint16_t>(block[0] | (block[1] << 8)));
            double block_dot = 0.0;
            for (std::size_t k = 0; k < 32; ++k) {
                const auto q = static_cast<std::int8_t>(block[2 + k]);
                block_dot += static_cast<double>(q) * static_cast<double>(f.x[b * 32 + k]);
            }
            acc += static_cast<double>(scale) * block_dot;
        }
        out[r] = static_cast<float>(acc);
    }
}

double max_relative_error(const std::vector<float>& got, const std::vector<float>& want) {
    double max_abs = 0.0;
    double max_ref = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        max_abs = std::fmax(max_abs, std::fabs(double(got[i]) - double(want[i])));
        max_ref = std::fmax(max_ref, std::fabs(double(want[i])));
    }
    return max_ref > 0.0 ? max_abs / max_ref : max_abs;
}

}  // namespace

int main() {
    if (!zen::hip::enabled()) {
        std::printf("SKIP: no HIP backend available\n");
        // 77 is ctest's SKIP_RETURN_CODE: exiting 0 here would make an untested
        // machine report this suite as a green PASS.
        return 77;
    }

    const auto* adapter = zen::hip::info();
    if (adapter) {
        std::printf("adapter=%s arch=%s sm=%d warp=%d\n", adapter->name, adapter->arch,
                    adapter->sm_count, adapter->warp_size);
    }

    // fp32 accumulation of int8 products is exact to well within this bound; a
    // repack or sign-extension bug moves the error orders of magnitude higher.
    constexpr double tolerance = 1e-5;
    bool ok = true;

    struct Shape {
        std::size_t rows;
        std::size_t cols;
    };
    // 2048-wide rows are the Llama-3.2 hidden size; 5632 is its FFN width; 512 a
    // row count that is not a multiple of the 8-row workgroup.
    const Shape shapes[] = {{512, 1024}, {2048, 2048}, {5632, 2048}, {512, 4096}};

    // Fixtures stay alive for the whole run: residency is keyed by (host pointer,
    // rows, columns), so freeing one and reusing its address would return stale
    // weights for a same-shaped successor.
    std::vector<Fixture> fixtures;
    fixtures.reserve(sizeof(shapes) / sizeof(shapes[0]));
    for (std::size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); ++i) {
        fixtures.push_back(make_fixture(shapes[i].rows, shapes[i].cols,
                                        static_cast<std::uint32_t>(1000 + i)));
    }

    for (std::size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); ++i) {
        const Fixture& f = fixtures[i];
        std::vector<float> reference;
        reference_gemv(f, reference);
        std::vector<float> gpu(f.rows, 0.0f);
        if (!zen::hip::gemv_q8_0(f.weights.data(), f.x.data(), gpu.data(), f.rows, f.cols)) {
            std::printf("FAIL: gemv_q8_0 returned false for rows=%zu cols=%zu\n", f.rows, f.cols);
            return 1;
        }
        const double rel = max_relative_error(gpu, reference);
        std::printf("gemv_q8_0 parity rows=%-5zu cols=%-5zu relative_err=%.3e\n", f.rows, f.cols, rel);
        if (!(rel <= tolerance)) {
            std::printf("FAIL: relative error %.3e exceeds %.1e\n", rel, tolerance);
            ok = false;
        }
    }

    // Batched path: the same tensors submitted as one group must match the
    // single-tensor results, since Q|K|V and gate|up use it in the hot loop.
    {
        Fixture a = make_fixture(512, 1024, 2001);
        Fixture b = make_fixture(256, 1024, 2002);
        std::vector<float> ref_a, ref_b;
        reference_gemv(a, ref_a);
        reference_gemv(b, ref_b);
        std::vector<float> got_a(a.rows, 0.0f), got_b(b.rows, 0.0f);
        // Both tasks must share the same input vector, so use the first fixture's.
        zen::hip::GemvTask tasks[2]{};
        tasks[0].weights = a.weights.data();
        tasks[0].y = got_a.data();
        tasks[0].rows = a.rows;
        tasks[0].columns = a.cols;
        tasks[1].weights = b.weights.data();
        tasks[1].y = got_b.data();
        tasks[1].rows = b.rows;
        tasks[1].columns = a.cols;
        if (!zen::hip::gemv_q8_0_batch(a.x.data(), tasks, 2)) {
            std::printf("FAIL: gemv_q8_0_batch returned false\n");
            return 1;
        }
        // b was generated with its own activation, so recompute its reference with a.x.
        Fixture b_same = b;
        b_same.x = a.x;
        std::vector<float> ref_b_same;
        reference_gemv(b_same, ref_b_same);
        const double rel_a = max_relative_error(got_a, ref_a);
        const double rel_b = max_relative_error(got_b, ref_b_same);
        std::printf("gemv_batch parity    first=%.3e second=%.3e\n", rel_a, rel_b);
        if (!(rel_a <= tolerance && rel_b <= tolerance)) {
            std::printf("FAIL: batched relative error exceeds %.1e\n", tolerance);
            ok = false;
        }
    }

    if (!ok) {
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
