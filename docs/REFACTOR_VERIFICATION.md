# Model-neutral refactor verification

Date: 2026-10-31

## Clean Linux build

Environment: GCC 14.2.0, CMake 3.31-series environment, Linux x86-64, Release, AVX2 kernels enabled.

Configure command:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DZEN_BUILD_TESTS=ON -DZEN_BUILD_SERVER=ON -DZEN_BUILD_AVX2=ON \
  -DZEN_GGUF_FIXTURE_1=/path/to/Tiny-LLM.Q4_K_M.gguf \
  -DZEN_GGUF_FIXTURE_2=/path/to/tinyllama-15M-stories-Q2_K.gguf
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

Result: clean configure and full build succeeded. CTest: 5/5 passed.

| Test | Result |
|---|---|
| `zen_core_tests` | PASS |
| `zen_bench_parity` | PASS |
| `zen_runtime_fixture_1` | PASS |
| `zen_runtime_fixture_2` | PASS |
| `zen_http_api_smoke` | PASS |

Both GGUF runtime fixtures are now loaded through `load_model_runtime()` and the `ModelRuntime` interface. Core tests verify executable factory dispatch for Llama and Qwen2 and fail-closed dispatch for Qwen3, BitNet and Bonsai.

## HTTP integration check

Started the newly built `zen_server` with `tinyllama-15M-stories-Q2_K.gguf` on localhost. `/health` returned `engine_loaded: true`. A real `/v1/completions` request with prompt `Once upon a time`, two output tokens and greedy sampling returned `, there` with a two-token completion.

## Scope and limitations

- This verifies Linux/GCC, not Windows/MSVC. The repository has a Windows CI matrix, but no Windows runner was executed during this work.
- The two supplied fixtures are Llama fixtures. No Qwen2 reference fixture was supplied, so Qwen2 execution and independent parity are not verified here.
- Qwen3, MoE, BitNet and Bonsai remain unsupported and are rejected by architecture dispatch.
- Chat formatting is separated from the HTTP backend, but the formatter still implements legacy format heuristics. It is not a general Jinja interpreter.
- FP16 storage/decode is not native CPU FP16 arithmetic.
- Existing mmap/prefetch is not application-managed NVMe eviction/streaming.
- Existing speculative decoding remains greedy-only, not stochastic speculative decoding.
- General JSON Schema/grammar compilation and constrained decoding remain outstanding.

## Scalar-only build

A separate clean build with `ZEN_BUILD_AVX2=OFF` also configured and compiled successfully. CTest: 3/3 passed (`zen_core_tests`, `zen_bench_parity`, `zen_http_api_smoke`). The scalar-only run did not include real GGUF fixtures.

A separate clean scalar-only build after the adapter-library split also succeeded. CTest: 3/3 passed (`zen_core_tests`, `zen_bench_parity`, `zen_http_api_smoke`).
