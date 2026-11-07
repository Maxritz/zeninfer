// Isolated Q8_0 GEMM benchmark at real prefill shapes.
//
// Prefill wall time conflates the kernel with layer orchestration, so tuning the
// kernel against it is guesswork. This calls gemm_q8_0 directly at the shapes
// Llama-3.2-1B uses and reports achieved TFLOP/s, which is the only number that
// says whether the inner loop or the memory system is the limit.
//
// Reference points for this model:
//   ffn_gate / ffn_up   8192 x 2048
//   ffn_down            2048 x 8192
//   attn_q              2048 x 2048
//   attn_output         2048 x 2048
//
// 128 tokens of prefill is 2 * 960M * 128 = 246 GFLOP, so 8000 tok/s means
// 16 ms, i.e. 15.4 TFLOP/s.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "zen/d3d12_backend.h"

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

// Q8_0: 32-element blocks of (fp16 scale, 32 int8).
void fill_q8(std::vector<std::uint8_t>& w, std::size_t rows, std::size_t cols, std::mt19937& rng) {
    const std::size_t bpr = cols / 32;
    w.assign(rows * bpr * 34, 0);
    std::uniform_int_distribution<int> d(0, 255);
    for (std::uint8_t& b : w) b = static_cast<std::uint8_t>(d(rng));
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t b = 0; b < bpr; ++b) {
            std::uint8_t* blk = w.data() + (r * bpr + b) * 34;
            blk[0] = 0x00;  // small positive fp16 scale
            blk[1] = 0x2E;
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t tokens = argc > 1 ? static_cast<std::size_t>(std::atoi(argv[1])) : 128;
    const int iters = argc > 2 ? std::atoi(argv[2]) : 5;

    if (!zen::d3d12::enabled()) {
        std::printf("SKIP: D3D12 not enabled\n");
        return 77;
    }

    struct Shape { std::size_t rows, cols; const char* name; };
    const Shape shapes[] = {
        {8192, 2048, "ffn_gate"},
        {8192, 2048, "ffn_up"},
        {2048, 8192, "ffn_down"},
        {2048, 2048, "attn_q"},
    };

    std::mt19937 rng(7);
    double total_flops = 0.0, total_ms = 0.0, total_kernel_ms = 0.0;

    for (const auto& sh : shapes) {
        std::vector<std::uint8_t> w;
        fill_q8(w, sh.rows, sh.cols, rng);
        std::vector<float> x(tokens * sh.cols);
        std::uniform_real_distribution<float> d(-1.0f, 1.0f);
        for (float& v : x) v = d(rng);
        std::vector<float> y(tokens * sh.rows, 0.0f);

        // Cold call pays the repack + VRAM upload.
        const auto cold = Clock::now();
        const bool ok1 = zen::d3d12::gemm_q8_0(w.data(), x.data(), y.data(), tokens, sh.rows, sh.cols);
        const double cold_ms = ms_since(cold);
        if (!ok1) {
            std::printf("%-9s rows=%-5zu cols=%-5zu GPU REJECTED\n", sh.name, sh.rows, sh.cols);
            continue;
        }

        // Wall clock per call includes a synchronous fence wait (~1 ms) and the host
        // memcpy of x and y, which dwarf the kernel at these sizes: stripping 96% of the
        // kernel's arithmetic in an ablation moved wall time by 4%. So report the
        // stamped dispatch time separately and divide that for the real TFLOP/s.
        const zen::d3d12::Stats s0 = zen::d3d12::stats();
        double best = 1e9;
        for (int i = 0; i < iters; ++i) {
            const auto t = Clock::now();
            if (!zen::d3d12::gemm_q8_0(w.data(), x.data(), y.data(), tokens, sh.rows, sh.cols)) {
                std::printf("%-9s failed on iter %d\n", sh.name, i);
                return 1;
            }
            const double ms = ms_since(t);
            best = ms < best ? ms : best;
        }

        const zen::d3d12::Stats s1 = zen::d3d12::stats();
        const double kernel_ms = double(s1.gpu_us - s0.gpu_us) / 1000.0 / iters;
        const double fence_ms = double(s1.wait_us - s0.wait_us) / 1000.0 / iters;
        const double copy_ms = double(s1.gpu_copy_us - s0.gpu_copy_us) / 1000.0 / iters;
        const double flops = 2.0 * double(tokens) * double(sh.rows) * double(sh.cols);
        const double tflops = flops / (best / 1000.0) / 1e12;
        const double ktflops = kernel_ms > 0.0 ? flops / (kernel_ms / 1000.0) / 1e12 : 0.0;
        total_flops += flops;
        total_ms += best;
        total_kernel_ms += kernel_ms;
        std::printf("%-9s %5zux%-5zu t=%-4zu best=%7.2f ms (%.2f) | kernel=%6.2f ms "
                    "(%6.2f TFLOP/s) fence=%5.2f copy=%5.2f ms\n",
                    sh.name, sh.rows, sh.cols, tokens, best, tflops, kernel_ms, ktflops,
                    fence_ms, copy_ms);
    }

    if (total_ms > 0) {
        std::printf("TOTAL wall %.2f ms -> %.2f TFLOP/s | kernel %.2f ms -> %.2f TFLOP/s "
                    "(one layer's 4 big matmuls; 8000 tok/s needs ~15.4)\n",
                    total_ms, total_flops / (total_ms / 1000.0) / 1e12,
                    total_kernel_ms,
                    total_kernel_ms > 0.0 ? total_flops / (total_kernel_ms / 1000.0) / 1e12 : 0.0);
    }
    return 0;
}
