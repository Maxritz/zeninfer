# Prefill/decode performance investigation (ZI-P4 .. ZI-P7)

Date: 2026-11-02
Host: AMD Ryzen 9 5900XT (16C/32T, Zen3, AVX2), Windows x64
Build: `build-win` (MinGW GCC 13.2, `-O3 -DNDEBUG`, `ZEN_BUILD_AVX2=ON`)
Model: `Llama-3.2-1B-Instruct-Q8_0.gguf` (1.321 GB)

## Workload (identical before and after)

171-token text prompt ("The history of computing begins ... every pocket."),
`--max-new 32`, `zen_generate_ids`.

## Headline result

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Prefill | 43.4 tok/s (3941 ms) | 240.1 tok/s (659 ms) | **5.5x** |
| Decode | 13.8 tok/s (73.4 ms/tok) | 20.8 tok/s (48.2 ms/tok) | **1.50x** |

Qwen3-4B Q8_0, same prompt: prefill 77.4 tok/s, decode 6.8 tok/s (decode was 5.6 tok/s).

Generated text stays coherent (`"Paris, the city of which is not only known ..."`)
and logits parity is unchanged: `zen_runtime_qwen2_synthetic` (2e-3 max abs vs NumPy)
still passes.

## Prefill decomposition (171 tokens), before -> after

| Component | Before | After | Root cause / fix |
|---|---:|---:|---|
| `quant.gemm` | 1462 ms | 515 ms | generic per-token dot; replaced by 2-row x 4-token register-blocked AVX2 Q8_0 kernel (ZI-P5) + logical-CPU fan-out |
| `prefill.silu` | 1089 ms | 14 ms | scalar `std::exp` at 48 ns/element; AVX2 Cephes exp (ZI-P4) |
| `prefill.rope` | 864 ms | 31 ms | `pow`/`cos`/`sin` recomputed per head at 247 ns/pair; hoisted to a per-dimension table (ZI-P3) |
| `prefill.attention` | 643 ms | 71 ms | serial scalar double dots; AVX2 dot/axpy + parallel over (token, head) (ZI-P6) |
| other | ~58 ms | ~28 ms | — |
| **total** | **4116 ms** | **659 ms** | |

## Decode decomposition (per token), before -> after

| Component | Before | After | Notes |
|---|---:|---:|---|
| `quant.gemv` | 45.4 ms | 45.3 ms | DRAM-bound: 1.321 GB/token at ~29 GB/s |
| `attention.softmax` | 14.2 ms | 2.1 ms | scalar double dots + AVX2/parallel heads (ZI-P4/P6) |
| `ffn` exclusive | 8.1 ms | 0.5 ms | scalar sigmoid (ZI-P4) |
| prefetch thread spawn | ~5-7 ms | 0 ms | one `std::async` per layer per token removed (ZI-P7) |

## Hardware limits measured on this host

* Weight stream 1.321 GB/token; practical dual-channel DDR4-3200 ~40 GB/s
  => decode floor ~33 ms/token (~30 tok/s). llama.cpp reports 29.5 tok/s on this box.
* FP32 AVX2 peak = 16 cores x 32 FLOP/cycle (2x 256-bit FMA):
  1.69 TFLOPS at the 3.3 GHz base clock, 2.46 TFLOPS at the 4.8 GHz max boost.
  Realistic sustained all-core under AVX2 load is ~4.2-4.4 GHz (~2.15-2.25 TFLOPS).
* 171-token prefill is 424 GFLOP. Current 240 tok/s = 659 ms = 643 GF/s, i.e.
  ~26% of the 2.46 TFLOPS boost peak -- there is roughly 2-2.5x of GEMM headroom left.
* Ceiling for this workload: 144 ms of non-GEMM work (attention 71, rope 31, silu 14,
  misc 28) plus GEMM. At an unrealisable 100% of the 2.46 TFLOPS peak the GEMM alone
  costs 172 ms, so total 316 ms = 541 tok/s is the absolute FP32 ceiling at T=171.
  800 tok/s (214 ms) would need ~2.5x peak in FP32; it is only reachable with an
  int8xint8 kernel plus near-zero elementwise overhead. 1800 tok/s is out of reach.
* Realistic FP32 ladder: 643 GF/s now -> ~1.0 TF/s (~320 tok/s) with cache-blocked
  tiling -> ~1.5-2.1 TF/s (~400-500 tok/s) at 60-85% of sustained peak.

## Verification

* `ctest` in `build-win`: **9/9 pass** (core, iq, fuzz, parity prerequisites,
  governance, arch micro models, bench parity, qwen2 synthetic, http smoke).
* Found and fixed a pre-existing syntax error in `tests/core_tests.cpp` that had
  prevented the ZI-P1 GEMM amortisation gate from ever compiling.
* GEMM kernel equivalence checked against the previous implementation:
  max abs difference 0.0 (bit-identical) on 8192x2048xT=171.

## Not done / limitations

* No int8xint8 dot-product kernel (would need Q8 activation quantization); that is
  the remaining lever for prefill above ~350-400 tok/s.
* Decode remains DRAM-bound; only a smaller quantisation or more memory channels
  moves it materially.
