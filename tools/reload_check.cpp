// Reload check: load one model twice in a single process and require identical tokens.
//
// Why this exists
// ---------------
// The D3D12 backend keeps weight tensors resident in VRAM keyed on their host address
// (lookup_weights in d3d12_backend.cpp). A pointer is not a lifetime: when a model is
// reloaded, the old mapping is released and the new one routinely lands on the addresses
// it just gave back, so the cached device copies have to be dropped or the reload is
// served the previous model's bytes. DenseTransformerRuntime::load() does that at the top
// (and ~DenseTransformerRuntime() does it for every tensor on unload), which is the path
// this exercises:
//
//   run 1  load the model, generate
//   run 2  load() again on the *same* instance, generate  (release-at-reload)
//   run 3  destroy the instance, load a fresh one, generate  (release-at-unload)
//
// All three must produce the same tokens. Each run must also be a real forward pass:
// a prefix-cache hit would return the previous run's result and prove nothing, so
// cache_hit is required to be false.
//
// What this does not prove: loading the same file twice cannot by itself detect a stale
// residency entry, because the bytes are identical either way. zen_d3d12_gemm_parity's
// weight_reuse check is the deterministic detector for that (same address, different
// bytes); this check covers the runtime path around it - no state corruption, no reads
// from a released mapping, identical output.
//
// Usage: zen_reload_check model.gguf [--prompt-ids 1,2,3,4] [--max-new 4]
// Exit: 0 identical, 1 mismatch or failure, 77 model missing (ctest SKIP).

#include "zen/diagnostics.h"
#include "zen/model_runtime.h"
#ifdef ZEN_HAS_D3D12
#include "zen/d3d12_backend.h"
#endif

#include <charconv>
#include <cstdio>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool parse_ids(const std::string& s, std::vector<std::uint32_t>& ids) {
    std::size_t p = 0;
    while (p < s.size()) {
        auto e = s.find(',', p);
        if (e == std::string::npos) e = s.size();
        if (e == p) return false;
        std::uint32_t v{};
        auto r = std::from_chars(s.data() + p, s.data() + e, v);
        if (r.ec != std::errc{} || r.ptr != s.data() + e) return false;
        ids.push_back(v);
        p = e + 1;
    }
    return !ids.empty();
}

std::string join(const std::vector<std::uint32_t>& ids) {
    std::string out;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i) out += ',';
        out += std::to_string(ids[i]);
    }
    return out;
}

struct Result {
    std::vector<std::uint32_t> ids;
    bool cache_hit{};
    double ms{};
    std::uint64_t weight_bytes{};
};

bool run_once(zen::runtime::ModelRuntime& rt, const std::vector<std::uint32_t>& prompt,
              std::uint32_t max_new, Result& result, std::string& error) {
    zen::runtime::Generation out;
    const auto start = std::chrono::steady_clock::now();
    if (!rt.generate(prompt, max_new, out, &error)) return false;
    result.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                    .count();
    result.ids = out.token_ids;
    result.cache_hit = out.cache_hit;
#ifdef ZEN_HAS_D3D12
    result.weight_bytes = zen::d3d12::stats().weight_bytes;
#endif
    return !result.ids.empty();
}

void report(const char* label, const Result& r) {
    std::cout << "  " << label << " ms=" << r.ms << " cache_hit=" << (r.cache_hit ? 1 : 0)
              << " weight_bytes=" << r.weight_bytes << " ids=" << join(r.ids) << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: zen_reload_check model.gguf [--prompt-ids 1,2,3,4] [--max-new 4]\n";
        return 2;
    }
    const std::filesystem::path path = argv[1];
    std::vector<std::uint32_t> prompt{1, 2, 3, 4};
    std::uint32_t max_new = 4;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--prompt-ids" && i + 1 < argc) {
            prompt.clear();
            if (!parse_ids(argv[++i], prompt)) {
                std::cerr << "invalid token ID list\n";
                return 2;
            }
        } else if (a == "--max-new" && i + 1 < argc) {
            try {
                max_new = static_cast<std::uint32_t>(std::stoul(argv[++i]));
            } catch (...) {
                return 2;
            }
        } else {
            std::cerr << "unknown argument: " << a << "\n";
            return 2;
        }
    }
    if (!max_new) {
        std::cerr << "--max-new must be at least 1\n";
        return 2;
    }
    // ctest turns 77 into SKIP, so a machine without the fixture is not reported green.
    if (!std::filesystem::exists(path)) {
        std::cout << "SKIP: model not found: " << path << "\n";
        return 77;
    }

    // Unbuffered: this check exists to locate failures, and the backend reports device
    // removals on stderr and traces with printf, so every stream has to interleave in
    // real order or a hang cannot be attributed to an operation.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::cout << std::unitbuf;
    zen::diag::set_trace_enabled(false);
    std::cout << "reload_check model=" << path.string() << " prompt=" << join(prompt)
              << " max_new=" << max_new << "\n";

    std::string error;
    auto model = zen::runtime::load_model_runtime(path, nullptr, &error);
    if (!model) {
        std::cerr << "load failed: " << error << "\n";
        return 1;
    }
    Result first;
    if (!run_once(*model, prompt, max_new, first, error)) {
        std::cerr << "generation failed (first load): " << error << "\n";
        return 1;
    }
    report("load       ", first);

    // Same instance, loaded again: this is the release-at-start-of-load path.
    if (!model->load(path, &error)) {
        std::cerr << "reload failed: " << error << "\n";
        return 1;
    }
    Result reloaded;
    if (!run_once(*model, prompt, max_new, reloaded, error)) {
        std::cerr << "generation failed (reload): " << error << "\n";
        return 1;
    }
    report("reload     ", reloaded);

    // Unload (destructor releases every tensor), then a fresh instance.
    model.reset();
#ifdef ZEN_HAS_D3D12
    // Must be 0: everything the destructor released was resident. A huge value here means
    // the release accounting undershot and underflowed the unsigned total, which makes
    // every later upload think the VRAM budget is exhausted.
    std::cout << "  after unload weight_bytes=" << zen::d3d12::stats().weight_bytes << "\n";
#endif
    auto again = zen::runtime::load_model_runtime(path, nullptr, &error);
    if (!again) {
        std::cerr << "second load failed: " << error << "\n";
        return 1;
    }
    Result fresh;
    if (!run_once(*again, prompt, max_new, fresh, error)) {
        std::cerr << "generation failed (unload+load): " << error << "\n";
        return 1;
    }
    report("unload+load", fresh);

    const bool same = first.ids == reloaded.ids && first.ids == fresh.ids;
    // Direct evidence that the reload dropped the previous mapping's device copies: a
    // live reload must re-upload the same total, not keep the old entries alive next to
    // the new ones. Only meaningful when the backend actually keeps weights resident
    // (an F32 or CPU-only run reports 0 for every run).
    const bool resident = first.weight_bytes > 0;
    auto not_cumulative = [&](const Result& r) {
        if (!resident) return true;
        return r.weight_bytes <= first.weight_bytes + first.weight_bytes / 20 + 65536;
    };
    // A cache hit would have returned the previous run's tokens without a forward pass,
    // which would make the comparison above meaningless.
    const bool recomputed = !reloaded.cache_hit && !fresh.cache_hit;
    if (!same) {
        std::cout << "reload_check: MISMATCH - a reload changed the tokens\n";
        return 1;
    }
    if (!recomputed) {
        std::cout << "reload_check: MISMATCH - a reload was served from the prefix cache\n";
        return 1;
    }
    if (!not_cumulative(reloaded) || !not_cumulative(fresh)) {
        std::cout << "reload_check: MISMATCH - resident weights accumulated across a reload"
                  << " (first=" << first.weight_bytes << " reload=" << reloaded.weight_bytes
                  << " unload+load=" << fresh.weight_bytes
                  << "): the old mapping's entries were not released\n";
        return 1;
    }
    std::cout << "reload_check: resident=" << (resident ? "yes" : "no")
              << " (0 means this model does not use the D3D12 weight cache)\n";
    std::cout << "reload_check: PASS\n";
    return 0;
}
