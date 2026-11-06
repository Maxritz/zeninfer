// Bench + parity harness for the non-Q8 K-blocked GEMM kernels
// (gemm_f16_kblk_avx2, gemm_q4k_kblk_avx2; opt-in via ZEN_CPU_FAST_F16=1 /
// ZEN_CPU_FAST_Q4K=1, K-panel via ZEN_CPU_F16_KC / ZEN_CPU_Q4K_KC).
//
// Usage:
//   zen_nonq8_fast_bench <gguf> <tensor> <format> [dump <out.bin>]
//     Times quant::gemm at T=8/32/128 for workers 1 and 16 (same pattern as
//     the scout's tools/tmp_nonq8_gemm.cpp, current env routing) and, with
//     `dump`, writes the T=32 / w1 output as raw little-endian floats for
//     parity comparison (run the same command with and without the env
//     gates, then use `cmp`).
//   zen_nonq8_fast_bench cmp <a.bin> <b.bin>
//     Prints max abs / max rel error between two dump files and exits 0
//     when the gate passes (maxabs <= 1e-2 and maxrel <= 2e-3), else 2.
//
// Examples:
//   zen_nonq8_fast_bench Spark-Q4_K_M.gguf blk.0.ffn_gate.weight Q4_K dump plain.bin
//   ZEN_CPU_FAST_Q4K=1 zen_nonq8_fast_bench ... dump fast.bin
//   zen_nonq8_fast_bench cmp plain.bin fast.bin
//   ZEN_CPU_FAST_F16=1 ZEN_CPU_F16_KC=256 zen_nonq8_fast_bench VibeThinker.f16.gguf blk.0.ffn_gate.weight F16
#include "zen/gguf.h"
#include "zen/quant.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

using namespace zen;

static void print_env(bool is_f16) {
    const char* fast = std::getenv(is_f16 ? "ZEN_CPU_FAST_F16" : "ZEN_CPU_FAST_Q4K");
    const char* kc = is_f16 ? std::getenv("ZEN_CPU_F16_KC") : std::getenv("ZEN_CPU_Q4K_KC");
    std::printf("env %s=%s %s=%s\n", is_f16 ? "ZEN_CPU_FAST_F16" : "ZEN_CPU_FAST_Q4K", fast ? fast : "(unset)",
                is_f16 ? "ZEN_CPU_F16_KC" : "ZEN_CPU_Q4K_KC", kc ? kc : "(unset)");
    std::fflush(stdout);
}

static int bench(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: zen_nonq8_fast_bench <gguf> <tensor> <format> [dump <out.bin>]\n"); return 2; }
    const char* path = argv[1];
    const char* tname = argv[2];
    const char* fname = argv[3];
    const char* dump = nullptr;
    if (argc >= 6 && std::strcmp(argv[4], "dump") == 0) dump = argv[5];
    const auto* fmt = quant::find(fname);
    if (!fmt) { std::printf("no format %s\n", fname); return 2; }
    auto rep = gguf::inspect(path);
    if (!rep.ok) { std::printf("inspect failed: %s\n", rep.error.c_str()); return 1; }
    const model::TensorDesc* td = nullptr;
    for (auto& t : rep.tensors) if (t.name == tname) { td = &t; break; }
    if (!td) { std::printf("tensor %s not found\n", tname); return 1; }
    if (td->dimensions.size() != 2) { std::printf("tensor not 2D dims=%zu\n", td->dimensions.size()); return 1; }
    std::size_t cols = static_cast<std::size_t>(td->dimensions[0]);
    std::size_t rows = static_cast<std::size_t>(td->dimensions[1]);
    std::printf("tensor=%s dims=[%llu,%llu] -> rows=%zu cols=%zu ggml_type=%u byte_size=%llu data_off=%llu\n",
                tname, (unsigned long long)td->dimensions[0], (unsigned long long)td->dimensions[1],
                rows, cols, td->ggml_type, (unsigned long long)td->byte_size, (unsigned long long)rep.tensor_data_offset);
    std::size_t rb = quant::row_bytes(*fmt, cols);
    std::printf("format=%s block_elem=%u block_bytes=%u row_bytes=%zu expect_total=%zu\n",
                fname, fmt->block_elements, fmt->block_bytes, rb, rb * rows);
    if (rb * rows != td->byte_size) std::printf("WARN byte_size mismatch fmt rb*rows=%zu vs gguf %llu\n", rb * rows, (unsigned long long)td->byte_size);
    std::vector<std::uint8_t> w(rb * rows);
    {
        std::ifstream f(path, std::ios::binary);
        if (!f) { std::printf("open failed\n"); return 1; }
        f.seekg(static_cast<std::streamoff>(rep.tensor_data_offset + td->offset));
        f.read(reinterpret_cast<char*>(w.data()), static_cast<std::streamsize>(w.size()));
        if (static_cast<std::size_t>(f.gcount()) != w.size()) { std::printf("short read\n"); return 1; }
    }
    std::printf("weights loaded %.2f MB\n", w.size() / 1048576.0);
    {
        std::vector<float> ref(rows * cols);
        auto t0 = std::chrono::steady_clock::now();
        bool ok = quant::dequantize(*fmt, w.data(), rows * cols, ref.data());
        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        double elems = double(rows) * double(cols);
        std::printf("dequant-only ok=%d sec=%.4f Gelems/s=%.3f MB/s=%.1f\n", (int)ok, sec, elems / 1e9 / sec, double(w.size()) / sec / 1e6);
        std::printf("  ref[0..3]=%.4f %.4f %.4f %.4f\n", ref[0], ref[1], ref[2], ref[3]);
    }
    if (std::strcmp(fname, "F16") == 0) print_env(true);
    else if (std::strcmp(fname, "Q4_K") == 0) print_env(false);
    std::printf("T\tworkers\tms\tGF/s\tGF/s/core\tmax|out|\n");
    for (std::size_t T : {8u, 32u, 128u}) {
        std::vector<float> xin(T * cols);
        for (std::size_t t = 0; t < T; ++t) for (std::size_t c = 0; c < cols; ++c)
            xin[t * cols + c] = float(int((t * 7 + c) % 15) - 7) * 0.017f + float(int((t * 13 + c * 3) % 7) - 3) * 0.001f;
        std::vector<float> out(T * rows);
        for (unsigned wk : {1u, 16u}) {
            int reps = (T == 8) ? 5 : (T == 32) ? 3 : 2;
            quant::gemm(*fmt, w.data(), xin.data(), out.data(), T, rows, cols, 1.0f, wk);
            auto t0 = std::chrono::steady_clock::now();
            bool ok = false;
            for (int i = 0; i < reps; ++i) ok = quant::gemm(*fmt, w.data(), xin.data(), out.data(), T, rows, cols, 1.0f, wk);
            auto t1 = std::chrono::steady_clock::now();
            double sec = std::chrono::duration<double>(t1 - t0).count() / reps;
            double gfs = 2.0 * double(T) * double(rows) * double(cols) / 1e9 / sec;
            double mx = 0; for (float v : out) mx = std::max(mx, (double)std::fabs(v));
            std::printf("%zu\t%u\t%.3f\t%.1f\t%.2f\t%.3f ok=%d\n", T, wk, sec * 1000.0, gfs, gfs / wk, mx, (int)ok);
            std::fflush(stdout);
            if (dump && T == 32 && wk == 1) {
                std::ofstream o(dump, std::ios::binary);
                if (!o) { std::printf("dump open failed\n"); return 1; }
                o.write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size() * sizeof(float)));
                if (!o) { std::printf("dump write failed\n"); return 1; }
                std::printf("dumped T=32/w1 output (%zu floats) -> %s\n", out.size(), dump);
            }
        }
    }
    return 0;
}

static int cmp(int argc, char** argv) {
    if (argc != 4) { std::fprintf(stderr, "usage: zen_nonq8_fast_bench cmp <a.bin> <b.bin>\n"); return 2; }
    auto load = [](const char* p) {
        std::ifstream f(p, std::ios::binary);
        if (!f) return std::vector<float>{};
        f.seekg(0, std::ios::end);
        std::size_t n = static_cast<std::size_t>(f.tellg()) / sizeof(float);
        f.seekg(0, std::ios::beg);
        std::vector<float> v(n);
        f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(float)));
        return v;
    };
    auto a = load(argv[2]);
    auto b = load(argv[3]);
    if (a.empty() || b.empty()) { std::printf("cmp: open failed\n"); return 1; }
    if (a.size() != b.size()) { std::printf("cmp: size mismatch %zu vs %zu\n", a.size(), b.size()); return 2; }
    double maxabs = 0, maxrel = 0;
    std::size_t wi = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double e = std::fabs(double(a[i]) - double(b[i]));
        double rel = e / std::max(1.0, std::fabs(double(a[i])));
        if (e > maxabs) { maxabs = e; wi = i; }
        if (rel > maxrel) maxrel = rel;
    }
    std::printf("cmp n=%zu maxabs=%.6g maxrel=%.6g worst_i=%zu a=%.6g b=%.6g\n",
                a.size(), maxabs, maxrel, wi, double(a[wi]), double(b[wi]));
    const bool pass = maxabs <= 1e-2 && maxrel <= 2e-3;
    std::printf("cmp gate: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 2;
}

int main(int argc, char** argv) {
    if (argc >= 2 && std::strcmp(argv[1], "cmp") == 0) return cmp(argc, argv);
    return bench(argc, argv);
}
