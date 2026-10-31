// Cross-check the GGML Q8_K / TQ / IQ decoders and their generic GEMV path
// against the independent Python reference produced by iq_reference.py.
#include "zen/quant.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {
struct Record {
    std::string name;
    std::vector<std::uint8_t> block;
    std::vector<float> expected;
};

float activation(std::size_t i) { return float(int((i * 37 + 11) % 101) - 50) * 0.0137f; }
} // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: iq_check file.bin\n"); return 2; }
    std::ifstream in(argv[1], std::ios::binary);
    if (!in) { std::printf("cannot open %s\n", argv[1]); return 2; }
    std::uint32_t count = 0;
    in.read(reinterpret_cast<char*>(&count), 4);
    std::vector<Record> records;
    for (std::uint32_t i = 0; i < count; ++i) {
        char name[8]{};
        std::uint32_t size = 0;
        in.read(name, 8);
        in.read(reinterpret_cast<char*>(&size), 4);
        Record rec;
        rec.name = name;
        rec.block.resize(size);
        in.read(reinterpret_cast<char*>(rec.block.data()), size);
        rec.expected.resize(256);
        in.read(reinterpret_cast<char*>(rec.expected.data()), static_cast<std::streamsize>(256) * 4);
        records.push_back(std::move(rec));
    }

    int failures = 0, checks = 0;
    double worst_rel = 0.0;

    // 1) Decoder output against the independent reference.
    for (const auto& rec : records) {
        const auto* fmt = zen::quant::find(rec.name);
        if (!fmt || !fmt->implemented) { std::printf("missing format %s\n", rec.name.c_str()); ++failures; continue; }
        std::vector<float> actual(256);
        if (!zen::quant::dequantize(*fmt, rec.block.data(), 256, actual.data())) {
            std::printf("%s: decode failed\n", rec.name.c_str()); ++failures; continue;
        }
        float scale = 0.0f;
        for (float v : rec.expected) scale = std::max(scale, std::abs(v));
        if (scale <= 0.0f) scale = 1.0f;
        ++checks;
        for (std::size_t k = 0; k < 256; ++k) {
            const double rel = std::abs(double(actual[k]) - double(rec.expected[k])) / scale;
            if (rel > worst_rel) worst_rel = rel;
            if (rel > 1e-5) {
                std::printf("%s elem %zu: actual=%g expected=%g rel=%g\n",
                            rec.name.c_str(), k, actual[k], rec.expected[k], rel);
                ++failures;
                break;
            }
        }
    }

    // 2) Generic GEMV against a dot product of the reference values, serial
    //    and parallel dispatch.
    for (const auto& fmt_name : std::vector<std::string>{"Q8_K", "TQ1_0", "TQ2_0", "IQ4_XS",
                                                         "IQ2_XXS", "IQ2_XS", "IQ2_S",
                                                         "IQ3_XXS", "IQ3_S", "IQ1_S", "IQ1_M"}) {
        std::vector<const Record*> pool;
        for (const auto& rec : records) if (rec.name == fmt_name) pool.push_back(&rec);
        if (pool.empty()) { std::printf("%s: no reference records\n", fmt_name.c_str()); ++failures; continue; }
        const auto* fmt = zen::quant::find(fmt_name);
        const std::size_t rows = 8, columns = 256, blocks_per_row = columns / fmt->block_elements;
        const std::size_t row_bytes = zen::quant::row_bytes(*fmt, columns);
        std::vector<std::uint8_t> matrix(rows * row_bytes);
        std::vector<float> input(columns), output(rows), expected(rows);
        for (std::size_t i = 0; i < columns; ++i) input[i] = activation(i);
        for (std::size_t r = 0; r < rows; ++r) {
            const Record& rec = *pool[r % pool.size()];
            for (std::size_t b = 0; b < blocks_per_row; ++b)
                std::memcpy(matrix.data() + r * row_bytes + b * rec.block.size(), rec.block.data(), rec.block.size());
            double sum = 0.0;
            for (std::size_t k = 0; k < columns; ++k) sum += double(rec.expected[k]) * input[k];
            expected[r] = float(sum);
        }
        for (unsigned workers : {1u, 4u}) {
            if (!zen::quant::gemv(*fmt, matrix.data(), input.data(), output.data(), rows, columns, 1.0f, workers)) {
                std::printf("%s: gemv failed (workers=%u)\n", fmt_name.c_str(), workers); ++failures; continue;
            }
            ++checks;
            for (std::size_t r = 0; r < rows; ++r) {
                // Tolerance scaled by the accumulation magnitude, not the
                // (possibly cancelling) result: 256 float32 products summed
                // in float32 carry ~1e-7 relative error per term.
                double magnitude = 0.0;
                for (std::size_t k = 0; k < columns; ++k)
                    magnitude += std::abs(double(pool[r % pool.size()]->expected[k]) * input[k]);
                const double tol = 1e-5 * std::max(1.0, magnitude);
                const double diff = std::abs(double(output[r]) - double(expected[r]));
                if (diff > tol) {
                    std::printf("%s row %zu workers=%u: actual=%g expected=%g diff=%g tol=%g\n",
                                fmt_name.c_str(), r, workers, output[r], expected[r], diff, tol);
                    ++failures;
                }
            }
        }
    }

    std::printf("iq_check: %d checks, %d failures, worst relative decoder diff %.3g\n",
                checks, failures, worst_rel);
    return failures ? 1 : 0;
}
