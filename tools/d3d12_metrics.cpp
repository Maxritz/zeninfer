// D3D12 decode/prefill probe.
//
// The PERF line reports only wall-clock time. This tool reports the backend's own
// counters, which is what separates "the GPU is slow" from "the host is feeding
// it badly":
//
//   weight_bytes  bytes of repacked weights resident in VRAM right now. If this
//                 is near zero after warmup, every token is re-uploading the
//                 model from system RAM and no amount of kernel tuning helps.
//   submits       ExecuteCommandLists calls. Decode should be one per token.
//   queue_us      CPU time inside ExecuteCommandLists (driver acceptance).
//   wait_us       fence wait, i.e. GPU retire latency.
//
// queue_us and wait_us together are the whole per-token cost, so if they do not
// add up to the measured decode time, the host is doing work outside the backend.

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "zen/d3d12_backend.h"
#include "zen/model_runtime.h"

namespace {

void dump(const char* tag) {
    const zen::d3d12::Stats& st = zen::d3d12::stats();
    std::printf("[probe %-9s] weights_resident=%llu MB submits=%llu dispatches=%llu "
                "queue=%llu us wait=%llu us gpu=%llu us fallbacks=%llu\n",
                tag,
                static_cast<unsigned long long>(st.weight_bytes / (1024ull * 1024ull)),
                static_cast<unsigned long long>(st.submits),
                static_cast<unsigned long long>(st.dispatches),
                static_cast<unsigned long long>(st.queue_us),
                static_cast<unsigned long long>(st.wait_us),
                static_cast<unsigned long long>(st.gpu_us),
                static_cast<unsigned long long>(st.fallbacks));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: zen_d3d12_probe model.gguf [decode_steps]\n");
        return 2;
    }
    const std::size_t steps = argc > 2 ? static_cast<std::size_t>(std::atoi(argv[2])) : 8;

    if (!zen::d3d12::enabled()) {
        std::printf("SKIP: D3D12 backend not enabled (set ZEN_BACKEND=d3d12)\n");
        return 77;
    }

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

    // Prefill enough tokens to force every weight through the residency cache.
    std::vector<std::uint32_t> prompt;
    for (std::size_t i = 0; i < 64; ++i) {
        prompt.push_back(ids[i % ids.size()]);
    }

    dump("start");
    zen::runtime::Generation gen;
    if (!runtime->generate(prompt, steps, gen, &error)) {
        std::fprintf(stderr, "generate failed: %s\n", error.c_str());
        return 1;
    }
    dump("after");
    std::printf("[probe] prefill_ms=%.2f decode_ms=%.2f steps=%zu\n", gen.prefill_ms, gen.decode_ms,
                gen.token_ids.empty() ? 0 : gen.token_ids.size() - 1);

    // A second identical run must not re-upload: weights_resident stays flat and
    // submits only grow by the per-token count.
    dump("run2-before");
    zen::runtime::Generation gen2;
    if (!runtime->generate(prompt, steps, gen2, &error)) {
        std::fprintf(stderr, "generate 2 failed: %s\n", error.c_str());
        return 1;
    }
    dump("run2-after");

    zen::d3d12::trace_report();
    return 0;
}
