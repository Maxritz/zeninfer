// Diagnoses the "seeded sampling is not deterministic" failure by running the
// same seeded generation twice and printing where the token sequences diverge.
//
// Divergence at token 0 means the two runs disagree about the *prompt's* logits
// (prefill / prefix cache). Divergence at token 1+ means they agree on the
// prompt but disagree during decode, which points at the KV cache being read
// beyond the causal boundary or not being rewritten when the prefix is cached.

#include <cstdio>
#include <string>
#include <vector>

#include "zen/model_runtime.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: zen_d3d12_determinism model.gguf\n");
        return 2;
    }
    std::string error;
    auto runtime = zen::runtime::load_model_runtime(argv[1], nullptr, &error);
    if (!runtime) {
        std::fprintf(stderr, "load failed: %s\n", error.c_str());
        return 1;
    }

    zen::runtime::GenerationOptions opts;
    opts.temperature = 0.85f;
    opts.top_k = 16;
    opts.top_p = 0.92f;
    opts.seed = 123456;

    zen::runtime::Generation a, b;
    std::string ta, tb;
    if (!runtime->generate_text("Once upon a time", 4, a, ta, &error, {}, opts)) {
        std::fprintf(stderr, "run a failed: %s\n", error.c_str());
        return 1;
    }
    if (!runtime->generate_text("Once upon a time", 4, b, tb, &error, {}, opts)) {
        std::fprintf(stderr, "run b failed: %s\n", error.c_str());
        return 1;
    }

    std::printf("a.cache_hit=%d b.cache_hit=%d\n", int(a.cache_hit), int(b.cache_hit));
    std::printf("a tokens:");
    for (auto t : a.token_ids) std::printf(" %u", t);
    std::printf("\nb tokens:");
    for (auto t : b.token_ids) std::printf(" %u", t);
    std::printf("\n");

    const std::size_t n = a.token_ids.size() < b.token_ids.size() ? a.token_ids.size()
                                                                  : b.token_ids.size();
    std::size_t first_diff = n;
    for (std::size_t i = 0; i < n; ++i) {
        if (a.token_ids[i] != b.token_ids[i]) {
            first_diff = i;
            break;
        }
    }
    if (first_diff == n && a.token_ids.size() == b.token_ids.size()) {
        std::printf("RESULT: identical\n");
        return 0;
    }
    std::printf("RESULT: diverge at index %zu\n", first_diff);

    // Compare the logits each run started from.
    double max_abs = 0.0;
    const std::size_t m = a.last_logits.size() < b.last_logits.size() ? a.last_logits.size()
                                                                     : b.last_logits.size();
    for (std::size_t i = 0; i < m; ++i) {
        const double d = double(a.last_logits[i]) - double(b.last_logits[i]);
        max_abs = d < 0 ? -d : (d > max_abs ? d : max_abs);
    }
    std::printf("last_logits max_abs_diff=%.6e (size a=%zu b=%zu)\n", max_abs,
                a.last_logits.size(), b.last_logits.size());
    return 0;
}
