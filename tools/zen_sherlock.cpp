// Component-level tracer for the Sherlock workflow.
//
// Prefill and decode are separate measurements because they have different
// bottlenecks, and one end-to-end PERF line cannot tell them apart. This runs each
// phase in isolation, resets the diagnostic counters between them, and prints the
// per-component table plus the backend's own counters.
//
// It also measures an instrumentation floor: the same number of empty scopes
// pushed through the same begin/end path. Any component whose measured cost sits
// below that floor is noise, not signal, and reporting it as a bottleneck would be
// wrong.
//
// Usage: zen_sherlock model.gguf [prefill_tokens] [decode_tokens]
//        ZEN_D3D12=1 selects the GPU backend; unset selects CPU.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include "zen/diagnostics.h"
#include "zen/model_runtime.h"
#if defined(ZEN_HAS_D3D12)
#include "zen/d3d12_backend.h"
#endif

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

// Push N empty scopes through the same instrumentation path so the measurement
// overhead itself is known.
double instrumentation_floor_us(std::size_t n) {
    const auto t0 = Clock::now();
    for (std::size_t i = 0; i < n; ++i) {
        zen::diag::Scope s("__floor__", zen::diag::Phase::compute);
    }
    const double total = std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
    return n ? total / double(n) : 0.0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: zen_sherlock model.gguf [prefill_tokens] [decode_tokens]\n");
        return 2;
    }
    const std::size_t prefill_tokens =
        argc > 2 ? static_cast<std::size_t>(std::atoi(argv[2])) : 128;
    const std::size_t decode_tokens =
        argc > 3 ? static_cast<std::size_t>(std::atoi(argv[3])) : 32;

    std::string error;
    auto runtime = zen::runtime::load_model_runtime(argv[1], nullptr, &error);
    if (!runtime) {
        std::fprintf(stderr, "load failed: %s\n", error.c_str());
        return 1;
    }

    std::vector<std::uint32_t> ids;
    if (!runtime->tokenize("Once upon a time", ids) || ids.empty()) {
        std::fprintf(stderr, "tokenize failed\n");
        return 1;
    }
    std::vector<std::uint32_t> prompt;
    for (std::size_t i = 0; i < prefill_tokens; ++i) {
        prompt.push_back(ids[i % ids.size()]);
    }

#if defined(ZEN_HAS_D3D12)
    const bool gpu = zen::d3d12::enabled();
#else
    const bool gpu = false;
#endif

    // Warm-up: forces the one-off weight repack/upload and fills the page cache,
    // so neither phase is charged for cold-start residency.
    {
        zen::runtime::Generation warm;
        auto p = prompt;
        p.back() = (p.back() + 97u) % runtime->config().vocab;
        if (!runtime->generate(p, 2, warm, &error)) {
            std::fprintf(stderr, "warmup failed: %s\n", error.c_str());
            return 1;
        }
    }

    zen::diag::set_trace_enabled(true);

    // ---- Prefill: one call, one token out, so decode does not contaminate it.
    zen::diag::reset_trace();
    zen::runtime::Generation pre;
    const auto t_pre = Clock::now();
    {
        auto p = prompt;
        p.back() = (p.back() + 11u) % runtime->config().vocab;  // avoid prefix cache
        if (!runtime->generate(p, 1, pre, &error)) {
            std::fprintf(stderr, "prefill failed: %s\n", error.c_str());
            return 1;
        }
    }
    const double prefill_wall = ms_since(t_pre);

    // ---- Decode: second call on the identical prompt hits the prefix cache, so
    // only the decode loop is measured.
    zen::diag::reset_trace();
    zen::runtime::Generation dec;
    const auto t_dec = Clock::now();
    if (!runtime->generate(prompt, decode_tokens, dec, &error)) {
        std::fprintf(stderr, "decode failed: %s\n", error.c_str());
        return 1;
    }
    const double decode_wall = ms_since(t_dec);
    const std::size_t decode_steps = dec.token_ids.empty() ? 0 : dec.token_ids.size() - 1;

    std::printf("\n================ BASELINE (%s) ================\n", gpu ? "D3D12 GPU" : "CPU");
    std::printf("prefill  tokens=%-4zu  wall=%8.2f ms  %8.1f tok/s\n", prefill_tokens,
                prefill_wall, double(prefill_tokens) * 1000.0 / prefill_wall);
    std::printf("decode   tokens=%-4zu  wall=%8.2f ms  %8.1f tok/s  (%.2f ms/token)\n",
                decode_steps, decode_wall,
                decode_steps ? double(decode_steps) * 1000.0 / decode_wall : 0.0,
                decode_steps ? decode_wall / double(decode_steps) : 0.0);
    std::printf("floor: %.3f us per empty scope (any row below this is noise)\n",
                instrumentation_floor_us(2000));

    std::printf("\n---------------- PREFILL components ----------------\n");
    {
        std::ostringstream o;
        zen::diag::print_trace(o);
        std::printf("%s", o.str().c_str());
    }

    // Re-run the decode trace on its own so the table is decode-only.
    zen::diag::reset_trace();
    {
        zen::runtime::Generation d2;
        runtime->generate(prompt, decode_tokens, d2, &error);
    }
    std::printf("\n---------------- DECODE components -----------------\n");
    {
        std::ostringstream o;
        zen::diag::print_trace(o);
        std::printf("%s", o.str().c_str());
    }

#if defined(ZEN_HAS_D3D12)
    if (gpu) {
        const zen::d3d12::Stats& st = zen::d3d12::stats();
        std::printf("\n---------------- D3D12 counters -------------------\n");
        std::printf("weights_resident=%llu MB submits=%llu dispatches=%llu "
                    "queue=%llu us wait=%llu us gpu=%llu us fallbacks=%llu\n",
                    static_cast<unsigned long long>(st.weight_bytes / (1024ull * 1024ull)),
                    static_cast<unsigned long long>(st.submits),
                    static_cast<unsigned long long>(st.dispatches),
                    static_cast<unsigned long long>(st.queue_us),
                    static_cast<unsigned long long>(st.wait_us),
                    static_cast<unsigned long long>(st.gpu_us),
                    static_cast<unsigned long long>(st.fallbacks));
        zen::d3d12::trace_report();
    }
#endif
    return 0;
}
