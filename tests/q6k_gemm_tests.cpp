#include "zen/kernels.h"
#include "zen/quant.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {
int failures = 0;
std::uint32_t state = 0x7136a95bu;
std::uint32_t random_u32() { state = state * 1664525u + 1013904223u; return state; }

// Build valid Q6_K blocks from known signed quant values and per-16 scales.
// The expected weights come from those inputs, not from ZenInfer's decoder.
void make_weights(std::size_t rows, std::size_t cols,
                  std::vector<std::uint8_t>& packed, std::vector<float>& expected) {
    const std::size_t blocks = cols / 256, rb = blocks * 210;
    packed.assign(rows * rb, 0);
    expected.resize(rows * cols);
    constexpr std::uint16_t halves[] = {0x3400, 0xb800, 0x0001, 0x3c00};
    constexpr float factors[] = {0.25f, -0.5f, 0x1p-24f, 1.0f};
    for (std::size_t r = 0; r < rows; ++r) for (std::size_t b = 0; b < blocks; ++b) {
        std::uint8_t* p = packed.data() + r * rb + b * 210;
        const std::size_t di = (r + b) % 4;
        p[208] = std::uint8_t(halves[di]); p[209] = std::uint8_t(halves[di] >> 8);
        int scales[16];
        for (int i = 0; i < 16; ++i) {
            scales[i] = int(random_u32() >> 24) % 31 - 15;
            p[192 + i] = std::uint8_t(static_cast<std::int8_t>(scales[i]));
        }
        for (std::size_t half = 0; half < 2; ++half) for (std::size_t l = 0; l < 32; ++l) {
            unsigned q[4];
            for (std::size_t group = 0; group < 4; ++group) {
                q[group] = random_u32() >> 26;
                const std::size_t col = b * 256 + half * 128 + group * 32 + l;
                expected[r * cols + col] = factors[di] * float(scales[half * 8 + group * 2 + l / 16]) * float(int(q[group]) - 32);
            }
            p[half * 64 + l] = std::uint8_t((q[0] & 15u) | ((q[2] & 15u) << 4));
            p[half * 64 + 32 + l] = std::uint8_t((q[1] & 15u) | ((q[3] & 15u) << 4));
            p[128 + half * 32 + l] = std::uint8_t((q[0] >> 4) | ((q[1] >> 4) << 2) |
                                                   ((q[2] >> 4) << 4) | ((q[3] >> 4) << 6));
        }
    }
}

void check_output(const char* route, const std::vector<float>& out, std::size_t stride,
                  const std::vector<double>& reference, const std::vector<double>& magnitude,
                  std::size_t tokens, std::size_t rows, std::size_t cols, unsigned workers) {
    for (std::size_t t = 0; t < tokens; ++t) for (std::size_t r = 0; r < rows; ++r) {
        const std::size_t i = t * rows + r;
        const double actual = out[t * stride + r];
        if (!std::isfinite(actual) || std::abs(actual - reference[i]) > 2e-5 * std::max(1.0, magnitude[i])) {
            std::cerr << "FAIL " << route << " rows=" << rows << " cols=" << cols << " tokens=" << tokens
                      << " workers=" << workers << " t=" << t << " r=" << r << " got=" << actual
                      << " expected=" << reference[i] << '\n';
            ++failures;
            return;
        }
    }
}
}

int main() {
    const auto* format = zen::quant::find("Q6_K");
    if (!format) return 1;
    for (std::size_t cols : {256u, 512u, 768u, 1280u})
    for (std::size_t rows : {1u, 3u, 4u, 5u, 35u, 67u}) {
        std::vector<std::uint8_t> packed;
        std::vector<float> expected;
        make_weights(rows, cols, packed, expected);
        std::vector<float> decoded(cols);
        for (std::size_t r = 0; r < rows; ++r) {
            if (!zen::quant::dequantize(*format, packed.data() + r * (cols / 256) * 210, cols, decoded.data()) ||
                !std::equal(decoded.begin(), decoded.end(), expected.begin() + r * cols)) {
                std::cerr << "FAIL Q6_K dequantization disagrees with constructed values\n";
                return 1;
            }
        }
        for (std::size_t tokens : {1u, 2u, 3u, 5u}) {
            std::vector<float> input(tokens * cols);
            for (float& x : input) x = float(int(random_u32() >> 24) - 128) / 128.0f;
            std::vector<double> ref(tokens * rows), magnitude(tokens * rows);
            for (std::size_t t = 0; t < tokens; ++t) for (std::size_t r = 0; r < rows; ++r)
                for (std::size_t c = 0; c < cols; ++c) {
                    const double term = double(expected[r * cols + c]) * input[t * cols + c];
                    ref[t * rows + r] += term; magnitude[t * rows + r] += std::abs(term);
                }
            for (unsigned workers : {1u, 4u}) {
                if (tokens == 1) {
                    std::vector<float> vout(rows);
                    if (!zen::quant::gemv(*format, packed.data(), input.data(), vout.data(), rows, cols, 1.0f, workers)) {
                        std::cerr << "FAIL production Q6_K GEMV declined a valid shape\n"; ++failures;
                    } else check_output("GEMV", vout, rows, ref, magnitude, 1, rows, cols, workers);
                }
                std::vector<float> output(tokens * rows);
                if (!zen::quant::gemm(*format, packed.data(), input.data(), output.data(), tokens, rows, cols, 1.0f, workers)) {
                    std::cerr << "FAIL production Q6_K GEMM declined a valid shape\n"; ++failures;
                } else check_output("production", output, rows, ref, magnitude, tokens, rows, cols, workers);
            }
#ifdef ZEN_HAS_AVX2_KERNELS
            if (zen::cpu_has_avx2_fma()) {
                const std::size_t stride = rows + 3;
                std::vector<float> output(tokens * stride, 123456.0f);
                if (!zen::gemm_q6k_kblk_avx2(packed.data(), input.data(), output.data(), tokens, rows, cols, stride)) {
                    std::cerr << "FAIL direct Q6_K GEMM declined a valid shape\n"; ++failures;
                } else check_output("direct AVX2", output, stride, ref, magnitude, tokens, rows, cols, 1);
                for (std::size_t t = 0; t < tokens; ++t) for (std::size_t i = rows; i < stride; ++i)
                    if (output[t * stride + i] != 123456.0f) { std::cerr << "FAIL output stride overwritten\n"; ++failures; }
            }
#endif
        }
    }
    if (failures) std::cerr << failures << " Q6_K GEMM/GEMV failures\n";
    else std::cout << "Q6_K GEMM/GEMV parity passed\n";
    return failures ? 1 : 0;
}
