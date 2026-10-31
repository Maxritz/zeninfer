# ZenInfer 0.3.3-alpha: numerical parity and CPU kernels

## Scope and interpretation

This release adds a separate NumPy implementation of the dense Llama forward-pass mathematics and compares its complete vocabulary logits against the C++ runtime. The NumPy implementation independently computes RMSNorm, RoPE, grouped-query attention, softmax, residuals, SwiGLU and the output projection.

The comparison exporter obtains FP32 tensors through ZenInfer's GGUF loader. Therefore this validates the transformer math and C++ runtime wiring against a separately implemented numerical path, but it does **not** independently validate the GGUF quantisation decoders against llama.cpp or another third-party GGUF implementation. No third-party inference engine was available in this build environment. This distinction is intentional.

## Fixture parity results

Prompt set (four prompts per model):

1. `Once upon a time`
2. `The quick brown fox jumps over the lazy dog.`
3. `Hello, world!`
4. `Once upon a time\nthere was a little girl.`

| Model | Layers | Comparisons | Maximum absolute logit error | Maximum mean absolute error across runs | Argmax agreement |
|---|---:|---:|---:|---:|---|
| Tiny LLM | 1 | 4 | 4.7684e-6 | 1.3368e-6 | 4/4 |
| TinyLlama Stories | 6 | 4 | 6.6757e-6 | 1.5058e-6 | 4/4 |

All compared logits were finite. The top-five token ordering matched in all eight comparisons. Prompt IDs are captured in each JSON report produced by `scripts/run_parity_fixtures.sh`.

These errors are consistent with normal FP32 operation-order differences between the NumPy and AVX2/FMA3 paths. The parity gate is `max_abs_error <= 2e-3` and exact greedy argmax agreement. Observed errors are over two orders of magnitude below that gate.

## Runtime AVX2 GEMV

The Llama runtime now dispatches FP32 projection GEMV to AVX2/FMA3 when runtime CPU and OS state checks confirm support. The scalar double-accumulation path remains available as a fallback. The AVX2 implementation uses four independent accumulators over 32-element chunks to reduce dependency chains.

Five runs, each generating up to 100 tokens from `Once upon a time` on `tinyllama-15M-stories-Q2_K.gguf`:

| Path | Median decode throughput | Median decode time |
|---|---:|---:|
| Scalar fallback | 70.70 tokens/s | 1,400.31 ms |
| AVX2/FMA3 | 336.42 tokens/s | 294.28 ms |

Observed median throughput ratio: **4.76x**. A 64-token generated-ID comparison between scalar and AVX2 builds produced identical token sequences.

Host: virtualised AMD EPYC 9V74 environment, with four CPUs available to the runtime policy. These are container measurements, not measurements on the user's Ryzen 7 PRO 4750U or a physical Zen 3 system. The benchmark includes the runtime's normal diagnostic scopes, and the same instrumentation was enabled for both paths.

## Fused Q8_0 quantised GEMV

Added `gemv_q8_0_avx2`, a fused block decoder/dot-product kernel. It reads the FP16 block scale and signed int8 values directly from GGML Q8_0 blocks, converts eight signed values at a time, and accumulates against the corresponding FP32 input vector. It does not materialise a full FP32 weight row.

`quant::gemv` selects this path when AVX2/FMA3 is available and scale is the native Q8_0 scale. Other formats retain the dequantise-then-dot fallback; that fallback uses AVX2 for the dot product when available.

A two-row, two-block Q8_0 regression fixture compares the selected path against an independently dequantised scalar dot product. The same test also passes in a build with `ZEN_BUILD_AVX2=OFF`, verifying the fallback.

**Important integration boundary:** the current Llama model loader still expands model tensors to FP32 and the transformer projections therefore use FP32 GEMV. The fused Q8_0 path is available through the quantised GEMV API but is not yet used by the transformer runtime. Keeping raw quantised tensors and dispatching projections through format-specific kernels is the next memory/performance milestone.

## Verification

- Release build with AVX2 enabled: passed.
- Release build with AVX2 disabled: passed.
- CTest AVX2 build: 1/1 passed.
- CTest scalar-only build: 1/1 passed.
- ASan + UBSan core build: 1/1 passed, leak detection enabled.
- Real GGUF tensor conversion: Tiny LLM 12/12 tensors, 12,988,992 elements; TinyLlama Stories 56/56 tensors, 15,191,712 elements. All values finite.
- NumPy parity: 8/8 prompts passed.
- Scalar versus AVX2 generated IDs: identical for the tested 64-token run.

## Reproduction

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DZEN_BUILD_AVX2=ON
cmake --build build -j2
ctest --test-dir build --output-on-failure

scripts/run_parity_fixtures.sh build \
  /path/to/Tiny-LLM.Q4_K_M.gguf \
  /path/to/tinyllama-15M-stories-Q2_K.gguf
```

The parity script requires Python 3 and NumPy. It writes tensor exports, logits and JSON reports under `parity-results/` (or `$ZEN_PARITY_OUT`).

## Remaining parity work

- Compare dequantised tensors against a separate GGUF implementation.
- Compare per-layer activations and logits against llama.cpp on the same fixtures.
- Verify tokenizer output against the model's original tokenizer implementation, including Unicode and special-token edge cases.
- Add deterministic sampling parity once non-greedy samplers are implemented.
- Validate on Windows/MSVC and the target Ryzen hardware.
