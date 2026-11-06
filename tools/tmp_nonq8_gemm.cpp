// TEMP harness (not shipped, not in CMake): real-tensor quant::gemm bench for non-Q8 scout.
// Usage: tmp_nonq8_gemm <gguf> <tensor_name> <format_name>
//   e.g. tmp_nonq8_gemm Spark-Q4_K_M.gguf blk.0.ffn_gate.weight Q4_K
// Loads raw quantized weights for the named tensor via zen::gguf::inspect,
// then times quant::gemm at T=8/32/128 for workers 1 and 16 + dequant-only rate.
#include "zen/gguf.h"
#include "zen/quant.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <vector>

using namespace zen;
int main(int argc, char** argv) {
    if (argc != 4) { fprintf(stderr, "usage: tmp_nonq8_gemm <gguf> <tensor> <format>\n"); return 2; }
    const char* path = argv[1];
    const char* tname = argv[2];
    const char* fname = argv[3];
    const auto* fmt = quant::find(fname);
    if (!fmt) { printf("no format %s\n", fname); return 2; }
    auto rep = gguf::inspect(path);
    if (!rep.ok) { printf("inspect failed: %s\n", rep.error.c_str()); return 1; }
    const model::TensorDesc* td = nullptr;
    for (auto& t : rep.tensors) if (t.name == tname) { td = &t; break; }
    if (!td) { printf("tensor %s not found\n", tname); return 1; }
    if (td->dimensions.size() != 2) { printf("tensor not 2D dims=%zu\n", td->dimensions.size()); return 1; }
    size_t cols = (size_t)td->dimensions[0];
    size_t rows = (size_t)td->dimensions[1];
    printf("tensor=%s dims=[%llu,%llu] -> rows=%zu cols=%zu ggml_type=%u byte_size=%llu data_off=%llu\n",
           tname, (unsigned long long)td->dimensions[0], (unsigned long long)td->dimensions[1],
           rows, cols, td->ggml_type, (unsigned long long)td->byte_size, (unsigned long long)rep.tensor_data_offset);
    size_t rb = quant::row_bytes(*fmt, cols);
    printf("format=%s block_elem=%u block_bytes=%u row_bytes=%zu expect_total=%zu\n",
           fname, fmt->block_elements, fmt->block_bytes, rb, rb*rows);
    if (rb*rows != td->byte_size) printf("WARN byte_size mismatch fmt rb*rows=%zu vs gguf %llu\n", rb*rows, (unsigned long long)td->byte_size);
    std::vector<uint8_t> w(rb*rows);
    {
        std::ifstream f(path, std::ios::binary);
        if (!f) { printf("open failed\n"); return 1; }
        f.seekg((std::streamoff)(rep.tensor_data_offset + td->offset));
        f.read((char*)w.data(), (std::streamsize)w.size());
        if ((size_t)f.gcount() != w.size()) { printf("short read\n"); return 1; }
    }
    printf("weights loaded %.2f MB\n", w.size()/1048576.0);
    // dequant-only throughput (full tensor once)
    {
        std::vector<float> ref(rows*cols);
        auto t0 = std::chrono::steady_clock::now();
        bool ok = quant::dequantize(*fmt, w.data(), rows*cols, ref.data());
        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t1-t0).count();
        double elems = double(rows)*double(cols);
        printf("dequant-only ok=%d sec=%.4f Gelems/s=%.3f MB/s=%.1f\n", (int)ok, sec, elems/1e9/sec, double(w.size())/sec/1e6);
        // first values sanity
        printf("  ref[0..3]=%.4f %.4f %.4f %.4f\n", ref[0], ref[1], ref[2], ref[3]);
    }
    printf("T\tworkers\tms\tGF/s\tGF/s/core\tmax|out|\n");
    for (size_t T : {8u, 32u, 128u}) {
        std::vector<float> xin(T*cols);
        for (size_t t = 0; t < T; ++t) for (size_t c = 0; c < cols; ++c)
            xin[t*cols+c] = float(int((t*7+c)%15)-7)*0.017f + float(int((t*13+c*3)%7)-3)*0.001f;
        std::vector<float> out(T*rows);
        for (unsigned wk : {1u, 16u}) {
            int reps = (T==8)?5:(T==32)?3:2;
            quant::gemm(*fmt, w.data(), xin.data(), out.data(), T, rows, cols, 1.0f, wk); // warmup
            auto t0 = std::chrono::steady_clock::now();
            bool ok = false;
            for (int i = 0; i < reps; ++i) ok = quant::gemm(*fmt, w.data(), xin.data(), out.data(), T, rows, cols, 1.0f, wk);
            auto t1 = std::chrono::steady_clock::now();
            double sec = std::chrono::duration<double>(t1-t0).count()/reps;
            double gfs = 2.0*double(T)*double(rows)*double(cols)/1e9/sec;
            double mx = 0; for (float v : out) mx = std::max(mx, (double)std::fabs(v));
            printf("%zu\t%u\t%.3f\t%.1f\t%.2f\t%.3f ok=%d\n", T, wk, sec*1000.0, gfs, gfs/wk, mx, (int)ok);
            fflush(stdout);
        }
    }
    return 0;
}
