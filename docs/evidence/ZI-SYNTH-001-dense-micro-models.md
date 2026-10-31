# ZI-SYNTH-001: dense architecture micro-model definitions

Status: PARTIAL, synthetic fixture gate only
Date: 2026-10-31

## Scope

Add deterministic model definitions for Qwen2 dense and Qwen3 dense where the
synthetic architecture suite previously had no dense definitions. Extend the
architecture micro-model runner to execute their dense residual-projection path
and include both fixtures in the architecture CTest.

## Changes

- Added `models/synthetic/qwen2-dense-tiny.zenmodel`.
- Added `models/synthetic/qwen3-dense-tiny.zenmodel`.
- Extended `tools/architecture_microbench.cpp` to recognise both dense fixture
  architecture IDs, initialise deterministic FP32 projection matrices, execute
  residual dense projections and produce logits.
- Added both definitions to `zen_arch_micro_models` in `CMakeLists.txt`.
- Updated `models/synthetic/README.md` to describe their scope and limitations.

## RED/GREEN record

- RED: not captured before implementation. This is an additive fixture/runner
  extension, not evidence of a production adapter gap being fixed.
- GREEN: both new definitions execute in the architecture microbench and are
  included in the CTest fixture command.

## Results

AVX2-configured build:

```text
cmake -S . -B build-avx2 -DZEN_BUILD_AVX2=ON -DZEN_BUILD_TESTS=ON -DZEN_BUILD_SERVER=ON -DZEN_BUILD_BENCH=ON
cmake --build build-avx2 -j2
ctest --test-dir build-avx2 --output-on-failure
100% tests passed, 0 tests failed out of 8
```

Scalar build:

```text
cmake -S . -B build-scalar -DZEN_BUILD_AVX2=OFF -DZEN_BUILD_TESTS=ON -DZEN_BUILD_SERVER=ON -DZEN_BUILD_BENCH=ON
cmake --build build-scalar -j2
ctest --test-dir build-scalar --output-on-failure
100% tests passed, 0 tests failed out of 8
```

Direct fixture run executed all five `.zenmodel` files. Both new dense fixtures
reported finite logits and `validation=PASS fixture=synthetic-kernel-only`.

## Interpretation

This proves that the micro-model definition loader and synthetic dense matrix
execution path can execute the Qwen2/Qwen3 dense fixture definitions. It does
not prove Qwen2/Qwen3 production GGUF tensor mapping, attention, RoPE, GQA,
RMSNorm, tokenizer/template behaviour, trained-checkpoint compatibility or
reference parity. The existing Qwen2 GGUF fixture remains a relabelled-weight
layout test, not a genuine Qwen2 checkpoint.

## Review notes

- Dense fixture path uses deterministic FP32 matrices and residual tanh
  projections. It is intentionally a small architecture smoke path, not a
  faithful transformer implementation.
- No external model weights were downloaded or represented as trained weights.
- Windows/MSVC was not run in this environment.
