// CPU-vs-GPU parity check for the D3D12 batched matrix product (prefill path).
//
// A per-element double-precision reference isolates where a shader or repack
// regression moves the error, including shapes that exercise the row-block and
// token-tile boundaries.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "zen/d3d12_backend.h"

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

double max_error(const std::vector<float>& a, const std::vector<float>& b) {
    double best = 0.0;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
        best = std::max(best, std::fabs(double(a[i]) - double(b[i])));
    }
    return best;
}

struct Shape {
    std::size_t rows;
    std::size_t cols;
    std::size_t tokens;
};

bool check_gemm(const Shape& shape, bool verbose) {
    const std::size_t blocks_per_row = shape.cols / 32;
    constexpr std::size_t raw_block_bytes = 34;
    const std::size_t row_bytes = blocks_per_row * raw_block_bytes;

    const std::uint32_t seed =
        static_cast<std::uint32_t>(20261102u + shape.rows * 7u + shape.cols);
    // F-17: weights and activations need separate generators. The backend keys VRAM
    // residency on (pointer, rows, cols), so two shapes that share rows/cols must
    // present identical bytes at the heap address they both reuse. Deriving the
    // weights from the same stream as the token*cols input values meant the second of
    // such a pair regenerated different weights while the GPU kept serving the first
    // one's cached copy - a harness defect, not a kernel one.
    std::mt19937 w_rng(seed);
    std::mt19937 x_rng(seed ^ 0x9E3779B9u);
    std::uniform_int_distribution<int> quant_dist(-127, 127);
    std::uniform_real_distribution<float> x_dist(-1.0f, 1.0f);

    std::vector<std::uint8_t> weights(shape.rows * row_bytes);
    std::vector<float> input(shape.tokens * shape.cols);
    std::vector<float> reference(shape.tokens * shape.rows, 0.0f);
    std::vector<float> gpu(shape.tokens * shape.rows, 0.0f);

    for (float& v : input) {
        v = x_dist(x_rng);
    }
    for (std::size_t r = 0; r < shape.rows; ++r) {
        std::uint8_t* row = weights.data() + r * row_bytes;
        for (std::size_t b = 0; b < blocks_per_row; ++b) {
            std::uint8_t* block = row + b * raw_block_bytes;
            const float scale = static_cast<float>(1 + (b % 8)) / 256.0f;
            const std::uint16_t half = float_to_half(scale);
            block[0] = static_cast<std::uint8_t>(half & 0xFFu);
            block[1] = static_cast<std::uint8_t>(half >> 8);
            for (std::size_t k = 0; k < 32; ++k) {
                block[2 + k] = static_cast<std::uint8_t>(static_cast<std::int8_t>(quant_dist(w_rng)));
            }
        }
    }

    for (std::size_t t = 0; t < shape.tokens; ++t) {
        const float* x = input.data() + t * shape.cols;
        for (std::size_t r = 0; r < shape.rows; ++r) {
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
            reference[t * shape.rows + r] = static_cast<float>(acc);
        }
    }

    if (!zen::d3d12::gemm_q8_0(weights.data(), input.data(), gpu.data(), shape.tokens,
                               shape.rows, shape.cols)) {
        // main() already established the backend is up, so a decline here is the
        // opt-in flag or the size threshold refusing the call. The old message blamed
        // the size threshold for every cause and counted the skip as a pass (F-13).
        std::printf(
            "gemm rows=%zu cols=%zu tokens=%zu: ERROR (backend declined - set "
            "ZEN_D3D12_GEMM_EXPERIMENTAL=1 and check kMinElements)\n",
            shape.rows, shape.cols, shape.tokens);
        return false;
    }

    double max_abs = 0.0;
    double max_ref = 0.0;
    std::size_t worst_i = 0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        const double err = std::fabs(double(gpu[i]) - double(reference[i]));
        if (err > max_abs) {
            max_abs = err;
            worst_i = i;
        }
        max_ref = std::max(max_ref, std::fabs(double(reference[i])));
    }
    const double relative = max_ref > 0.0 ? max_abs / max_ref : max_abs;
    const bool ok = relative <= 1e-5;

    std::printf(
        "gemm rows=%zu cols=%zu tokens=%zu: relative_err=%.3e worst=(t=%zu,r=%zu) %s\n",
        shape.rows, shape.cols, shape.tokens, relative, worst_i / shape.rows,
        (worst_i % shape.rows), ok ? "ok" : "MISMATCH");

    if (verbose) {
        for (std::size_t t = 0; t < shape.tokens; ++t) {
            double per_token = 0.0;
            for (std::size_t r = 0; r < shape.rows; ++r) {
                per_token = std::max(
                    per_token,
                    std::fabs(double(gpu[t * shape.rows + r]) -
                              double(reference[t * shape.rows + r])));
            }
            std::printf("    token %zu: max_abs_err=%.3e\n", t, per_token);
        }
        for (std::size_t t = 0; t < shape.tokens; ++t) {
            std::printf("    t=%zu r=0: gpu=%.6f ref=%.6f\n", t,
                        double(gpu[t * shape.rows]), double(reference[t * shape.rows]));
        }
        if (shape.tokens > 1) {
            std::vector<float> shifted(reference.size(), 0.0f);
            for (std::size_t t = 1; t < shape.tokens; ++t) {
                std::memcpy(&shifted[t * shape.rows], &reference[(t - 1) * shape.rows],
                            shape.rows * sizeof(float));
            }
            std::printf("    shifted (gpu[t] vs ref[t-1]): max_err=%.3e\n",
                        max_error(gpu, shifted));
        }
    }
    return ok;
}


bool check_gemm_batch() {
    // Batched prefill: Q|K|V shapes (2048, 512, 512 rows, 2048 cols) and
    // gate|up shapes (8192, 8192 rows, 2048 cols) share one [tokens x cols]
    // input. The batch must be bit-identical to three/two single submissions
    // (same PSO/params, only the wrapper differs) and within 1e-5 of the
    // double-precision reference.
    struct TaskShape { std::size_t rows; std::size_t cols; };
    const std::size_t tokens = 17;
    const std::size_t cols = 2048;
    constexpr std::size_t raw_block_bytes = 34;
    const std::size_t blocks_per_row = cols / 32;
    const std::size_t row_bytes = blocks_per_row * raw_block_bytes;

    auto run_group = [&](const std::vector<TaskShape>& group, const char* label, std::uint32_t base_seed) -> bool {
        std::mt19937 x_rng(base_seed ^ 0x9E3779B9u);
        std::uniform_real_distribution<float> x_dist(-1.0f, 1.0f);
        std::vector<float> input(tokens * cols);
        for (float& v : input) v = x_dist(x_rng);

        struct TaskData {
            std::vector<std::uint8_t> weights;
            std::vector<float> reference;
            std::vector<float> single;
            std::vector<float> batched;
        };
        std::vector<TaskData> tasks(group.size());
        for (std::size_t ti = 0; ti < group.size(); ++ti) {
            const std::size_t rows = group[ti].rows;
            std::mt19937 w_rng(base_seed + static_cast<std::uint32_t>(ti * 1315423911u));
            std::uniform_int_distribution<int> quant_dist(-127, 127);
            TaskData& td = tasks[ti];
            td.weights.assign(rows * row_bytes, 0);
            for (std::size_t r = 0; r < rows; ++r) {
                std::uint8_t* row = td.weights.data() + r * row_bytes;
                for (std::size_t b = 0; b < blocks_per_row; ++b) {
                    std::uint8_t* block = row + b * raw_block_bytes;
                    const float scale = static_cast<float>(1 + (b % 8)) / 256.0f;
                    const std::uint16_t half = float_to_half(scale);
                    block[0] = static_cast<std::uint8_t>(half & 0xFFu);
                    block[1] = static_cast<std::uint8_t>(half >> 8);
                    for (std::size_t k = 0; k < 32; ++k)
                        block[2 + k] = static_cast<std::uint8_t>(static_cast<std::int8_t>(quant_dist(w_rng)));
                }
            }
            td.reference.assign(tokens * rows, 0.0f);
            for (std::size_t t = 0; t < tokens; ++t) {
                const float* x = input.data() + t * cols;
                for (std::size_t r = 0; r < rows; ++r) {
                    const std::uint8_t* row = td.weights.data() + r * row_bytes;
                    double acc = 0.0;
                    for (std::size_t b = 0; b < blocks_per_row; ++b) {
                        const std::uint8_t* block = row + b * raw_block_bytes;
                        const float scale = half_to_float(static_cast<std::uint16_t>(block[0] | (block[1] << 8)));
                        double block_dot = 0.0;
                        for (std::size_t k = 0; k < 32; ++k) {
                            const auto q = static_cast<std::int8_t>(block[2 + k]);
                            block_dot += static_cast<double>(q) * static_cast<double>(x[b * 32 + k]);
                        }
                        acc += static_cast<double>(scale) * block_dot;
                    }
                    td.reference[t * rows + r] = static_cast<float>(acc);
                }
            }
            td.single.assign(tokens * rows, 0.0f);
            td.batched.assign(tokens * rows, 0.0f);
            if (!zen::d3d12::gemm_q8_0(td.weights.data(), input.data(), td.single.data(), tokens, rows, cols)) {
                std::printf("gemm_batch %s task %zu: ERROR (single declined)\n", label, ti);
                return false;
            }
        }
        zen::d3d12::GemmTask batch_tasks[8];
        for (std::size_t ti = 0; ti < group.size(); ++ti)
            batch_tasks[ti] = {tasks[ti].weights.data(), tasks[ti].batched.data(), group[ti].rows};
        if (!zen::d3d12::gemm_q8_0_batch(input.data(), batch_tasks, group.size(), tokens, cols)) {
            std::printf("gemm_batch %s: ERROR (batch declined)\n", label);
            return false;
        }
        bool ok = true;
        for (std::size_t ti = 0; ti < group.size(); ++ti) {
            const TaskData& td = tasks[ti];
            const double batch_vs_single = max_error(td.batched, td.single);
            double max_abs = 0.0; double max_ref = 0.0;
            for (std::size_t i = 0; i < td.reference.size(); ++i) {
                max_abs = std::max(max_abs, std::fabs(double(td.batched[i]) - double(td.reference[i])));
                max_ref = std::max(max_ref, std::fabs(double(td.reference[i])));
            }
            const double rel = max_ref > 0.0 ? max_abs / max_ref : max_abs;
            const bool task_ok = (batch_vs_single == 0.0) && (rel <= 1e-5);
            ok = task_ok && ok;
            std::printf("gemm_batch %s task %zu rows=%zu cols=%zu tokens=%zu: batch_vs_single=%.3e relative_err=%.3e %s\n",
                        label, ti, group[ti].rows, cols, tokens, batch_vs_single, rel, task_ok ? "ok" : "MISMATCH");
        }
        return ok;
    };

    bool ok = true;
    ok = run_group({{2048, cols}, {512, cols}, {512, cols}}, "QKV", 20261106u) && ok;
    ok = run_group({{8192, cols}, {8192, cols}}, "gate_up", 20261107u) && ok;
    return ok;
}

bool check_rms_norm() {
    const std::size_t rows = 7;
    const std::size_t cols = 1024;
    std::vector<float> x(rows * cols), w(cols), out(rows * cols), ref(rows * cols);
    std::mt19937 rng(4242);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& v : x) {
        v = dist(rng);
    }
    for (float& v : w) {
        v = dist(rng);
    }

    const float eps = 1e-5f;
    for (std::size_t r = 0; r < rows; ++r) {
        double ss = 0.0;
        for (std::size_t c = 0; c < cols; ++c) {
            ss += double(x[r * cols + c]) * double(x[r * cols + c]);
        }
        const double scale = 1.0 / std::sqrt(ss / double(cols) + double(eps));
        for (std::size_t c = 0; c < cols; ++c) {
            ref[r * cols + c] = float(double(x[r * cols + c]) * scale * double(w[c]));
        }
    }

    if (!zen::d3d12::rms_norm(x.data(), w.data(), out.data(), rows, cols, eps)) {
        std::printf("rms_norm: ERROR (backend declined)\n");
        return false;
    }

    double err = 0.0;
    double mag = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        err = std::max(err, std::fabs(double(out[i]) - double(ref[i])));
        mag = std::max(mag, std::fabs(double(ref[i])));
    }
    const double rel = mag > 0.0 ? err / mag : err;
    const bool ok = rel <= 1e-5;
    std::printf("rms_norm rows=%zu cols=%zu: relative_err=%.3e %s\n", rows, cols, rel,
                ok ? "ok" : "MISMATCH");
    return ok;
}

}  // namespace

int main() {
    if (!zen::d3d12::enabled()) {
        std::printf("SKIP: no Direct3D 12.1 backend available\n");
        // 77 is ctest's SKIP_RETURN_CODE; exiting 0 made an untested machine report
        // this suite as a green PASS.
        return 77;
    }

    const Shape shapes[] = {
        {512, 256, 3},    // smallest exercised token count
        {512, 256, 2},    // same rows/cols as the shape above: exercises residency
        {512, 2048, 8},   // reuse across differing token counts (F-17)
        {2048, 2048, 17}, // full-size matrix, more than one token tile
    };

    bool ok = true;
    bool verbose = true;
    for (std::size_t i = 0; i < std::size(shapes); ++i) {
        ok = check_gemm(shapes[i], verbose) && ok;
        verbose = false;
        if (!ok) {
            break;
        }
    }
    ok = check_gemm_batch() && ok;
    ok = check_rms_norm() && ok;

    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
