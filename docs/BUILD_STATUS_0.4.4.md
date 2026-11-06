# Build and test status: 0.4.4-alpha

Environment: Linux x86-64, GCC 14.2, CMake Release, AMD EPYC 9V74 virtual CPU
allocation with 5 vCPUs. The build uses AVX2/FMA3 only; AVX-512 was not enabled
as a project execution path.

## AVX2 build

Configuration:

```bash
cmake -S . -B build-nextbench -DCMAKE_BUILD_TYPE=Release \
  -DZEN_BUILD_AVX2=ON -DZEN_BUILD_TESTS=ON -DZEN_BUILD_SERVER=ON \
  -DZEN_GGUF_FIXTURE_1=/path/to/Tiny-LLM.Q4_K_M.gguf \
  -DZEN_GGUF_FIXTURE_2=/path/to/tinyllama-15M-stories-Q2_K.gguf
cmake --build build-nextbench --parallel 4
ctest --test-dir build-nextbench --output-on-failure
```

Result: **6/6 PASS**.

- `zen_core_tests`
- `zen_arch_micro_models`
- `zen_bench_parity`
- `zen_runtime_fixture_1`
- `zen_runtime_fixture_2`
- `zen_http_api_smoke`

## Scalar-only build

Configuration: same fixtures and tests with `-DZEN_BUILD_AVX2=OFF`.

Result: **6/6 PASS**.

## Scope of the architecture micro-model tests

The `.zenmodel` fixtures exercise architecture kernels and fixed golden
checksums. They do not demonstrate full model execution or reference parity.
Production Qwen3 MoE, BitNet and Bonsai GGUF adapters remain incomplete.

## Windows

Windows/MSVC has not been run in this environment. The existing workflow is a
CI definition only; it is not evidence of a successful Windows build.
