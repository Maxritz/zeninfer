# ZenInfer 0.4.2-alpha

## Changes

- Replaced greedy-only speculative token comparison with stochastic speculative decoding.
- Draft proposals are sampled from the draft distribution `q` after applying the same temperature, top-k, top-p, logit-bias and allowed-token filters used by target sampling.
- Target verification samples acceptance with `min(1, p(x)/q(x))`.
- Rejected proposals use the normalised residual distribution `max(0, p-q)`; this preserves the target distribution rather than substituting a target argmax/sample.
- Removed the HTTP validation rule that rejected speculative requests whenever temperature was non-zero.
- Added seeded stochastic speculative repeatability and proposal-execution tests.
- Updated HTTP API smoke coverage and release documentation.

## Verification

| Configuration | Build | CTest |
|---|---|---:|
| Linux GCC 14.2, Release, AVX2 enabled | PASS | 5/5 PASS |
| Linux GCC 14.2, Release, scalar-only | PASS | 5/5 PASS |

Both runs used the two supplied Llama GGUF fixtures. This does not establish cross-architecture parity or Windows/MSVC compatibility.

## Known limitations

- Qwen3 dense/MoE, BitNet and Bonsai execution remain unimplemented.
- Qwen2 remains experimental and lacks independent reference parity with a real Qwen2 fixture.
- Chat templates remain legacy-family formatters, not a general Jinja interpreter.
- Grammar/JSON-schema constrained decoding remains unimplemented; allow-lists are supported.
- Native CPU FP16 arithmetic remains unimplemented; F16 storage is decoded/consumed through FP32 accumulation.
- Application-managed NVMe eviction and explicit residency control remain unimplemented; current mapping relies on OS demand paging and prefetch hints.
- Windows/MSVC runtime verification remains pending.
