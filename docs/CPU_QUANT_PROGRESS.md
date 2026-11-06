# CPU quant performance: incremental progress

This is **not** completion of all-quant CPU optimization. All 48 registered
formats have been exercised by the new sweep on small synthetic matrices;
many still use generic decoders and have not received specialized kernels.
Target Windows, the 16-core/32-thread CPU, real models, longer contexts and
worker scaling have not been measured here.

## Changes

- Fused AVX2/FMA decode GEMV for Q3_K, Q6_K and TQ2_0; vectorized prefill
  panel decoding for Q3_K, Q5_K, Q6_K and TQ2_0. F16 decode GEMV converts in
  vectors, including scalar fixup of exceptional half-precision values.
- The generic signed bit-packed decoder now extracts bytes rather than
  branching per bit, for all `Q1_SIGNED` through `Q18_SIGNED`. These are
  generic encodings, not necessarily GGUF model formats.
- The generic CPU GEMV/GEMM fallback decodes one whole row per call, removing
  per-block dispatch and repeated GEMV reductions. This adds a row-sized float
  buffer per participating worker (GEMM already used one), and changes the
  floating-point summation order. Not all shapes are guaranteed faster.
- Existing Q4_K/Q5_K GEMV and Q8_0 kernels remain. The IQ, PTQ, most other
  GGML/ternary encodings and non-AVX2 CPUs continue to use generic paths.

## Synthetic measurements

GCC 12 Release, Intel Xeon 2.60 GHz Linux VM, 2 logical CPUs (1 physical
core), warm buffers, 1 worker. Each result below is the range of **two
separate five-sample medians**, for the same randomized 512x4096 matrix;
GEMM uses 8 input tokens. GEMV and GEMM are microbenchmarks, not model tok/s;
the VM was noisy. Before is baseline revision `0f4bda2` and after is this
working-tree change. Different formats encode different random matrices, so
only compare before/after within a format.

| Format | GEMV before ms | GEMV after ms | GEMM before ms | GEMM after ms |
|---|---:|---:|---:|---:|
| Q3_K | 2.66 | 0.54–0.80 | 3.90–3.92 | 1.13–1.15 |
| Q5_K | 0.57–0.58 | 0.57–0.59 | 4.35–4.62 | 2.59 |
| Q6_K | 2.98–3.00 | 0.47 | 7.18–7.50 | 1.63–1.67 |
| TQ2_0 | 2.41–2.43 | 0.24–0.31 | 3.61–3.66 | 1.57–1.68 |
| F16 | 3.49–3.50 | 0.83–0.84 | 1.47–1.49 | 1.46–1.53 |
| Q8_SIGNED | 92.2–93.1 | 12.0–12.1 | 92.1–92.8 | 13.4–13.5 |

A 48-format snapshot at 256x1024, T=8 is in
[`benchmarks/cpu-quant-before.csv`](benchmarks/cpu-quant-before.csv) and
[`benchmarks/cpu-quant-after.csv`](benchmarks/cpu-quant-after.csv). Those are
single benchmark invocations (each number is a 5-sample median) and are not
reliable evidence of small differences on the shared VM. Results at T=2 and
T=32 were also measured for the first five formats; they do not establish an
optimal route across all shapes.

## Correctness and tracing

`zen_cpu_quant_sweep_tests` compares GEMV/GEMM against the repository's
full-row decoder plus a double-precision dot for all 48 formats, with 1 and 4
workers, 256/768 columns, 1/5/67 rows, token counts 1/2/3/8 and two scale
arguments. This shared decoder is not an independent proof of every format.
For all signed bit widths, a separate bit-by-bit reference checks whole and
partial rows. F16 checks all 65,536 bit patterns in a vector, plus tails.
Q3_K, Q5_K, Q6_K and TQ2_0 have independently constructed-weight tests; Q5_K,
Q6_K, Q3_K and TQ2_0 include direct padded-output-stride tests. The Q3_K
constructed test uses constant scale fields; randomized scale packing is
checked against the repository decoder. Other existing independent IQ
references remain available.

On Linux, Release AVX2 and scalar targeted tests passed; ASan/UBSan passed
all five new/extended quant test executables. A full AVX2 CTest with server
enabled passed 12/15 (including HTTP smoke); three fail because the checkout lacks
`models/synthetic/qwen2-dense-tiny.gguf` and
`models/synthetic/qwen3-moe-tiny.zenmodel`. Windows compilation and actual
models remain untested.

With `ZEN_CPU_BENCH_TRACE=1` the benchmark separately resets Sherlock tracing
for a single GEMV and GEMM after timing; `ZEN_QGEMM_TRACE=1` additionally
prints process-wide route counters at exit. [Per-format isolated traces for all
48 formats](benchmarks/cpu-quant-sherlock.txt) use the 256x1024 shape. On the
512x4096 shape above, Q3_K, Q5_K, Q6_K, TQ2_0 and F16 each showed 7 GEMV
and 7 GEMM calls on the AVX2 route
(the six benchmark calls plus the traced call). The isolated trace only
contains the quant operation; it does **not** establish end-to-end model
bottlenecks.

## Reproduce and A/B

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DZEN_BUILD_AVX2=ON
cmake --build build --config Release
# All formats; optional args: rows columns tokens workers [format]
./build/zen_cpu_quant_sweep_bench 256 1024 8 1
ZEN_CPU_BENCH_TRACE=1 ZEN_QGEMM_TRACE=1 ./build/zen_cpu_quant_sweep_bench 512 4096 8 1 Q6_K
ctest --test-dir build --output-on-failure -L quant
```

In a fresh process, `ZEN_CPU_KQUANT_GEMV=0` disables Q3_K/Q4_K/Q5_K/Q6_K
fused GEMV; `ZEN_CPU_FAST_F16_GEMV=0` disables the F16 decode route;
`ZEN_CPU_FAST_Q3K=0`, `ZEN_CPU_FAST_Q5K=0` disable their prefill routes;
`ZEN_CPU_FAST_TQ2=0` disables both TQ2 routes. Q6_K prefill remains ungated.
`ZEN_QGEMM_TRACE=1` appends counters to stdout, so do not treat that output as
pure CSV. Environment gates are cached at first use; set them before launching.

Next: optimize and independently validate remaining formats; benchmark actual
GGUF models and per-format tensor mixes on the target CPU with worker scaling,
separate uncached prefill/decode, stable process settings, and model-level
correctness. GPU work is deferred. The previously observed Qwen3.5 synthetic
incremental mismatch was subsequently fixed and regression-tested; see
[CPU runtime validation](CPU_RUNTIME_VALIDATION.md). This does not establish
real-model hybrid correctness or performance.

## CPU format-coverage follow-up

The earlier 48-format sweep in this document predates the additional upstream GGML types.
The current registry contains 58 formats, including scalar types. Q8_1,
NVFP4, Q1_0, Q2_0 and F8_E4M3 have GGUF decoding and generic CPU GEMV/GEMM;
I8/I16/I32/I64/F64 are also converted to FP32. These are functional fallback
paths, not new fused SIMD implementations. Independent upstream-GGML golden
blocks and a synthetic mapped-weight runtime test were added. For the precise
format inventory, Q8_1 legacy-layout warning and validation limits, see
[FORMAT_SUPPORT.md](FORMAT_SUPPORT.md). No real-model or target-CPU throughput
measurement was made for the new types.
