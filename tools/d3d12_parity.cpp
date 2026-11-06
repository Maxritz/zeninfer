// CPU-vs-GPU parity check for the Q8_0 matrix-vector kernel.
//
// The end-to-end token comparison only proves the pipeline agrees; this pins the
// kernel against a double-precision reference so a shader or repack regression is
// caught even when the sampling happens to be robust to small errors.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "zen/d3d12_backend.h"

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

}  // namespace

int main() {
    if (!zen::d3d12::enabled()) {
        std::printf("SKIP: no Direct3D 12.1 backend available\n");
        // 77 is ctest's SKIP_RETURN_CODE: exiting 0 here made an untested machine
        // report this suite as a green PASS.
        return 77;
    }

    constexpr std::size_t rows = 512;
    constexpr std::size_t cols = 1024;
    constexpr std::size_t blocks_per_row = cols / 32;
    constexpr std::size_t raw_block_bytes = 34;
    constexpr std::size_t row_bytes = blocks_per_row * raw_block_bytes;

    std::mt19937 rng(20261102);
    std::uniform_int_distribution<int> quant_dist(-127, 127);
    std::uniform_real_distribution<float> x_dist(-1.0f, 1.0f);

    std::vector<std::uint8_t> weights(rows * row_bytes);
    std::vector<float> x(cols);
    std::vector<float> reference(rows, 0.0f);
    std::vector<float> gpu(rows, 0.0f);

    for (float& v : x) {
        v = x_dist(rng);
    }

    for (std::size_t r = 0; r < rows; ++r) {
        std::uint8_t* row = weights.data() + r * row_bytes;
        for (std::size_t b = 0; b < blocks_per_row; ++b) {
            std::uint8_t* block = row + b * raw_block_bytes;
            // (1 + (b % 8)) * 2^-8 is exact in fp16 for every branch.
            const float scale = static_cast<float>(1 + (b % 8)) / 256.0f;
            const std::uint16_t half = float_to_half(scale);
            block[0] = static_cast<std::uint8_t>(half & 0xFFu);
            block[1] = static_cast<std::uint8_t>(half >> 8);
            for (std::size_t k = 0; k < 32; ++k) {
                block[2 + k] = static_cast<std::uint8_t>(
                    static_cast<std::int8_t>(quant_dist(rng)));
            }
        }
    }

    for (std::size_t r = 0; r < rows; ++r) {
        const std::uint8_t* row = weights.data() + r * row_bytes;
        double acc = 0.0;
        for (std::size_t b = 0; b < blocks_per_row; ++b) {
            const std::uint8_t* block = row + b * raw_block_bytes;
            const float scale =
                half_to_float(static_cast<std::uint16_t>(block[0] | (block[1] << 8)));
            double block_dot = 0.0;
            for (std::size_t k = 0; k < 32; ++k) {
                const auto q = static_cast<std::int8_t>(block[2 + k]);
                block_dot += static_cast<double>(q) * static_cast<double>(x[b * 32 + k]);
            }
            acc += static_cast<double>(scale) * block_dot;
        }
        reference[r] = static_cast<float>(acc);
    }

    if (!zen::d3d12::gemv_q8_0(weights.data(), x.data(), gpu.data(), rows, cols)) {
        std::printf("FAIL: gemv_q8_0 returned false\n");
        return 1;
    }

    double max_abs = 0.0;
    double max_ref = 0.0;
    for (std::size_t r = 0; r < rows; ++r) {
        max_abs = std::fmax(max_abs, std::fabs(double(gpu[r]) - double(reference[r])));
        max_ref = std::fmax(max_ref, std::fabs(double(reference[r])));
    }
    const double relative = max_ref > 0.0 ? max_abs / max_ref : max_abs;
    std::printf("gemv_q8_0 parity: rows=%zu max_abs_err=%.3e relative_err=%.3e\n", rows, max_abs,
                relative);

    // fp32 accumulation of int8 products is exact to well within this bound; a
    // repack or sign-extension bug moves the error orders of magnitude higher.
    constexpr double tolerance = 1e-5;
    if (!(relative <= tolerance)) {
        std::printf("FAIL: relative error %.3e exceeds %.1e\n", relative, tolerance);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
