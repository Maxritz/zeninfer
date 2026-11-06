// Speculative timing harness: ordinary vs speculative (draft = same GGUF when
// only one file given, e.g. to measure the verify cost against a 100%
// acceptance draft; with a real small draft the acceptance will be lower
// and per-token draft cost smaller).
//
// Usage:
//   zen_speculative_timing target.gguf [draft.gguf] [--prompt-ids 12805,5304,264,892] [--max-new 32] [--proposals 4]
//
// Prints: prefill_ms, decode_ms, speculative counters, effective tok/s
// (generated tokens / decode_ms), and the same for the ordinary baseline
// in the same process so the two are directly comparable. Greedy is used
// (temperature 0) so the speculative stream is deterministic and equals the
// ordinary stream; acceptance is 100% for the self-draft case (p == q) and
// <100% with a genuinely smaller draft.

#include "zen/model_runtime.h"
#include <chrono>
#include <charconv>
#include <iostream>
#include <string>
#include <vector>

static bool parse_ids(const std::string& s, std::vector<std::uint32_t>& ids) {
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

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: zen_speculative_timing target.gguf [draft.gguf] [--prompt-ids 1,2,3] [--max-new 32] [--proposals 4]\n";
        return 2;
    }
    std::string target_path = argv[1];
    std::string draft_path;
    std::vector<std::uint32_t> prompt{12805, 5304, 264, 892};
    std::uint32_t max_new = 32, proposals = 4;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--prompt-ids" && i + 1 < argc) {
            if (!parse_ids(argv[++i], prompt)) { std::cerr << "invalid --prompt-ids\n"; return 2; }
        } else if (a == "--max-new" && i + 1 < argc) {
            max_new = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        } else if (a == "--proposals" && i + 1 < argc) {
            proposals = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        } else if (!a.empty() && a[0] != '-') {
            draft_path = a;
        } else {
            std::cerr << "unknown argument: " << a << "\n"; return 2;
        }
    }
    if (draft_path.empty()) {
        draft_path = target_path;
        std::cout << "[spec] no draft given, using self-draft (same GGUF): acceptance will be 100%, draft cost == target cost\n";
    }

    std::string error;
    auto target = zen::runtime::load_model_runtime(target_path, nullptr, &error);
    if (!target) { std::cerr << "target load failed: " << error << "\n"; return 1; }
    auto draft = zen::runtime::load_model_runtime(draft_path, nullptr, &error);
    if (!draft) { std::cerr << "draft load failed: " << error << "\n"; return 1; }

    zen::runtime::GenerationOptions greedy;
    greedy.temperature = 0.0f;

    auto run_spec = [&](std::uint32_t P, zen::runtime::Generation& out, double& total_ms) {
        const auto t0 = std::chrono::steady_clock::now();
        if (!target->generate_speculative(prompt, max_new, *draft, P, out, &error, {}, greedy)) {
            std::cerr << "speculative generation failed: " << error << "\n"; return false;
        }
        total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return true;
    };
    auto run_ordinary = [&](zen::runtime::Generation& out, double& total_ms) {
        const auto t0 = std::chrono::steady_clock::now();
        if (!target->generate(prompt, max_new, out, &error, {}, greedy)) {
            std::cerr << "ordinary generation failed: " << error << "\n"; return false;
        }
        total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return true;
    };

    // Warm the ordinary path first so VRAM residency is paid there, not in spec.
    zen::runtime::Generation warm;
    double warm_ms = 0;
    if (!run_ordinary(warm, warm_ms)) return 1;

    zen::runtime::Generation spec_out;
    double spec_ms = 0;
    if (!run_spec(proposals, spec_out, spec_ms)) return 1;
    zen::runtime::Generation ord_out;
    double ord_ms = 0;
    if (!run_ordinary(ord_out, ord_ms)) return 1;

    const double spec_eff = spec_out.token_ids.empty() ? 0.0 : 1000.0 * double(spec_out.token_ids.size()) / spec_ms;
    const double spec_decode_tps = (spec_out.decode_ms > 0 && spec_out.token_ids.size() > 1)
        ? 1000.0 * double(spec_out.token_ids.size() - 1) / spec_out.decode_ms : 0.0;
    const double ord_decode_tps = (ord_out.decode_ms > 0 && ord_out.token_ids.size() > 1)
        ? 1000.0 * double(ord_out.token_ids.size() - 1) / ord_out.decode_ms : 0.0;

    std::cout << "[spec] proposals=" << proposals << " target=" << target_path << " draft=" << draft_path << "\n";
    std::cout << "spec   tokens=" << spec_out.token_ids.size() << " total_ms=" << spec_ms << " decode_ms=" << spec_out.decode_ms
              << " prefill_ms=" << spec_out.prefill_ms << " eff_tok_s_total=" << spec_eff
              << " decode_tok_s=" << spec_decode_tps << "\n";
    std::cout << "spec   proposed=" << spec_out.speculative_proposed << " accepted=" << spec_out.speculative_accepted
              << " rejected=" << spec_out.speculative_rejected
              << " accept_rate=" << (spec_out.speculative_proposed ? double(spec_out.speculative_accepted) / spec_out.speculative_proposed : 0.0)
              << " tokens_per_verify=" << (spec_out.speculative_proposed
                  ? double(spec_out.speculative_accepted + 0) / double((spec_out.speculative_proposed + proposals - 1) / proposals) : 0.0) << "\n";
    // tokens_per_verify above is rough; the exact mean is (total tokens - correction/bonus count)/ verifies.
    // Print the raw ids for oracle comparison.
    std::cout << "spec_ids=";
    for (std::size_t i = 0; i < spec_out.token_ids.size(); ++i) std::cout << (i ? "," : "") << spec_out.token_ids[i];
    std::cout << "\n";
    std::cout << "ord    tokens=" << ord_out.token_ids.size() << " total_ms=" << ord_ms << " decode_ms=" << ord_out.decode_ms
              << " decode_tok_s=" << ord_decode_tps << "\n";
    std::cout << "ord_ids=";
    for (std::size_t i = 0; i < ord_out.token_ids.size(); ++i) std::cout << (i ? "," : "") << ord_out.token_ids[i];
    std::cout << "\n";
    std::cout << "spec_matches_ord=" << (spec_out.token_ids == ord_out.token_ids ? "1" : "0") << "\n";
    const double speedup = (ord_decode_tps > 0) ? (spec_decode_tps / ord_decode_tps) : 0.0;
    std::cout << "speedup_decode=" << speedup << " (spec effective / ordinary decode)\n";
    return 0;
}
