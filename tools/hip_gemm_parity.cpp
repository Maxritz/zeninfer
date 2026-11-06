// CPU-vs-GPU parity check for the HIP batched Q8_0 product and rms_norm.
//
// The batched path is what prefill uses, and its token-major output layout
// (y[token * rows + row]) is easy to get subtly wrong in a way end-to-end ids
// can mask, so it is pinned here against a double-precision reference.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "zen/hip_backend.h"

namespace {

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
    std::vector<float> input;  // [tokens][cols]
};

Fixture make_fixture(std::size_t rows, std::size_t cols, std::size_t tokens, std::uint32_t seed) {
    Fixture f;
    f.rows = rows;
    f.cols = cols;
    f.blocks_per_row = cols / 32;
    f.row_bytes = f.blocks_per_row * 34;
    f.weights.assign(rows * f.row_bytes, 0);
    f.input.resize(tokens * cols);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> quant_dist(-127, 127);
    std::uniform_real_distribution<float> x_dist(-1.0f, 1.0f);

    for (float& v : f.input) {
        v = x_dist(rng);
    }
    for (std::size_t r = 0; r < rows; ++r) {
        std::uint8_t* row = f.weights.data() + r * f.row_bytes;
        for (std::size_t b = 0; b < f.blocks_per_row; ++b) {
            std::uint8_t* block = row + b * 34;
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

void reference_gemm(const Fixture& f, std::size_t tokens, std::vector<float>& out) {
    out.assign(tokens * f.rows, 0.0f);
    for (std::size_t t = 0; t < tokens; ++t) {
        for (std::size_t r = 0; r < f.rows; ++r) {
            const std::uint8_t* row = f.weights.data() + r * f.row_bytes;
            const float* xrow = f.input.data() + t * f.cols;
            double acc = 0.0;
            for (std::size_t b = 0; b < f.blocks_per_row; ++b) {
                const std::uint8_t* block = row + b * 34;
                const float scale =
                    half_to_float(static_cast<std::uint16_t>(block[0] | (block[1] << 8)));
                double block_dot = 0.0;
                for (std::size_t k = 0; k < 32; ++k) {
                    const auto q = static_cast<std::int8_t>(block[2 + k]);
                    block_dot += static_cast<double>(q) * static_cast<double>(xrow[b * 32 + k]);
                }
                acc += static_cast<double>(scale) * block_dot;
            }
            out[t * f.rows + r] = static_cast<float>(acc);
        }
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
        return 77;
    }

    constexpr double tolerance = 1e-5;
    bool ok = true;

    struct Case {
        std::size_t tokens;
        std::size_t rows;
        std::size_t cols;
    };
    // 1 token exercises the tile boundary, 8 exactly fills one tile, 17 straddles
    // tiles without filling a row block.
    const Case cases[] = {{1, 512, 1024}, {8, 512, 1024}, {17, 256, 2048}, {64, 2048, 2048}};

    // Build every fixture up front and keep them alive for the whole run. The
    // backend caches device residency by (host pointer, rows, columns), which holds
    // in production because each tensor owns its own allocation; letting a fixture
    // die here would hand the next case the same address with the same shape and
    // silently return the previous tensor's weights.
    std::vector<Fixture> fixtures;
    fixtures.reserve(sizeof(cases) / sizeof(cases[0]));
    for (std::size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        fixtures.push_back(make_fixture(cases[i].rows, cases[i].cols, cases[i].tokens,
                                       static_cast<std::uint32_t>(4000 + i)));
    }

    for (std::size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        const Fixture& f = fixtures[i];
        std::vector<float> reference;
        reference_gemm(f, cases[i].tokens, reference);
        std::vector<float> gpu(cases[i].tokens * cases[i].rows, 0.0f);
        if (!zen::hip::gemm_q8_0(f.weights.data(), f.input.data(), gpu.data(), cases[i].tokens,
                                 cases[i].rows, cases[i].cols)) {
            std::printf("FAIL: gemm_q8_0 returned false for tokens=%zu rows=%zu cols=%zu\n",
                        cases[i].tokens, cases[i].rows, cases[i].cols);
            return 1;
        }
        const double rel = max_relative_error(gpu, reference);
        std::printf("gemm_q8_0 parity tokens=%-3zu rows=%-5zu cols=%-5zu relative_err=%.3e\n",
                    cases[i].tokens, cases[i].rows, cases[i].cols, rel);
        if (!(rel <= tolerance)) {
            std::printf("FAIL: relative error %.3e exceeds %.1e\n", rel, tolerance);
            ok = false;
        }
    }

    // rms_norm: reference is the plain mean-of-squares form the kernel documents.
    {
        constexpr std::size_t rows = 4;
        constexpr std::size_t cols = 2048;
        constexpr float epsilon = 1e-5f;
        std::mt19937 rng(5555);
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
        std::vector<float> x(rows * cols), w(cols), got(rows * cols, 0.0f), want(rows * cols, 0.0f);
        for (float& v : x) {
            v = dist(rng);
        }
        for (float& v : w) {
            v = dist(rng);
        }
        for (std::size_t r = 0; r < rows; ++r) {
            double ss = 0.0;
            for (std::size_t i = 0; i < cols; ++i) {
                const double v = x[r * cols + i];
                ss += v * v;
            }
            const float scale = static_cast<float>(1.0 / std::sqrt(ss / double(cols) + double(epsilon)));
            for (std::size_t i = 0; i < cols; ++i) {
                want[r * cols + i] = x[r * cols + i] * scale * w[i];
            }
        }
        if (!zen::hip::rms_norm(x.data(), w.data(), got.data(), rows, cols, epsilon)) {
            std::printf("FAIL: rms_norm returned false\n");
            return 1;
        }
        const double rel = max_relative_error(got, want);
        std::printf("rms_norm parity rows=%zu cols=%zu relative_err=%.3e\n", rows, cols, rel);
        if (!(rel <= tolerance)) {
            std::printf("FAIL: rms_norm relative error %.3e exceeds %.1e\n", rel, tolerance);
            ok = false;
        }
    }

    if (!ok) {
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
