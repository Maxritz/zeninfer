# ZenInfer 0.3.3-alpha build and verification record

## Build environment

- Compiler: GCC 14.2, C++20, CMake Release.
- Host: virtualised AMD EPYC 9V74 environment; runtime policy reported four available CPUs.
- AVX2/FMA3: runtime detected and selected.
- Physical Ryzen and Windows/MSVC: not tested.

## Build and tests

| Configuration | Result |
|---|---|
| Release, AVX2 enabled | Build passed; CTest 1/1 passed |
| Release, AVX2 disabled | Build passed; CTest 1/1 passed |
| ASan + UBSan, AVX2 enabled, core tools | Build passed; CTest 1/1 passed with leak detection |

## Real GGUF fixtures

| Fixture | Tensors decoded | Elements | Non-finite values |
|---|---:|---:|---:|
| Tiny LLM | 12/12 | 12,988,992 | 0 |
| TinyLlama Stories | 56/56 | 15,191,712 | 0 |

## Numerical parity

A separately implemented NumPy reference recomputed final prompt logits from the same dequantised tensors. Four prompts were tested against each fixture, eight comparisons total.

- Tiny LLM max absolute error: 4.7684e-6.
- TinyLlama Stories max absolute error: 6.6757e-6.
- All logits finite.
- Greedy argmax agreement: 8/8.
- Top-five token ordering agreement: 8/8.

This is independent transformer-math parity, not third-party GGUF decoder or full llama.cpp parity. Details and limitations are in `PARITY_AND_KERNELS_0.3.3.md`.

## CPU performance

Five-run decode benchmark, TinyLlama Stories, prompt `Once upon a time`, up to 100 generated tokens:

- AVX2/FMA3 median: 336.42 tokens/s, 294.28 ms decode.
- Scalar median: 70.70 tokens/s, 1,400.31 ms decode.
- Median throughput ratio: 4.76x.
- Tested 64-token generated ID sequences were identical between scalar and AVX2 builds.

Raw runs are preserved in `AVX2_RUNTIME_RAW_RUNS.txt`. These are virtualised container measurements, not target Ryzen results.

## Quantised GEMV

- Fused Q8_0 AVX2 GEMV implemented and regression-tested against scalar dequantise-then-dot.
- Scalar-only fallback build passes the same test.
- The Llama runtime still stores expanded FP32 weights. Raw quantised tensor retention and wiring quantised projections into the transformer are not complete.
- Fused Q5_0, Q3_K, Q6_K and IQ4_NL kernels are not implemented.

## Outstanding

1. Third-party GGUF decoder comparison and llama.cpp activation/logit comparison.
2. Raw quantised tensor residency and format-dispatched projection kernels.
3. Fused Q5_0, Q3_K, Q6_K and IQ4_NL GEMV; quantised prefill GEMM.
4. Per-layer activation parity and tokenizer reference parity.
5. Windows/MSVC and physical Ryzen validation.
