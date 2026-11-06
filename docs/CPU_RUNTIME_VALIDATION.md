# CPU runtime validation: 2026-11-06

## Fixes and regression coverage

* **Qwen3.5 convolution history:** on a batch shorter than the convolution
  history, the newest-first ring was overwritten in forward order. Earlier
  rows were read after being replaced, duplicating token history and changing
  incremental logits. On the F32 synthetic fixture the pre-fix two-token
  one-at-a-time run differed from full prefill by up to 17.1857. Copying the
  oldest destination first preserves the old source rows. A position-zero
  forward resets the recurrent state and validates cache sizes before use.
* **Exact-prefix cache:** the saved buffers had the capacity required by the
  request that created them. A later hit with a longer output limit could write
  beyond them or fail. Request-local KV and hybrid convolution buffers now
  resize to the current visit length while retaining the saved prompt state.
* **Malformed dimensions:** dense/hybrid loader dimensions are validated before
  deriving head widths, so a zero head count fails instead of dividing by zero.
  KV overflow checks use the configured head dimension rather than an assumed
  embedding/head ratio.

`zen_q35_incremental` generates four-layer synthetic Qwen3.5 fixtures with F32
and PTQ1_0 weights and compares full prefill with chunk sizes 1, 2, 3 and 4
across prefixes of 1–8 tokens. It checks reset/reuse, a growing prefix cache,
and greedy generated tokens against full-prefix recomputation. Its F32 case
also checks each full-prefix row against an independent NumPy implementation:
worst absolute logit error in the AVX2 run was **6.49e-5**. The full vs chunked
comparisons use the same ZenInfer forward implementation, so they are not by
themselves an independent model oracle. PTQ1_0's check is internal parity, not
an independent end-to-end quantized-model oracle.

`zen_runtime_qwen2_synthetic` tests greedy recomputation and a cache hit after a
shorter request; `zen_qwen2_numpy_parity` independently checks eight prompt
prefixes. `zen_http_model_smoke` runs the real server and synthetic GGUF through
health, model listing, normal and streamed completions, chat, cache growth and
12 requests with four concurrent clients. `zen_invalid_model` exercises a
zero-head malformed model. The original missing Qwen2 GGUF has been replaced
by a small deterministic, generated Qwen2-layout fixture; five missing
`.zenmodel` micro-model definitions are now supplied. None is a trained model.

## Local results

Linux, GCC 12, small shared VM; Release with server and benchmark enabled:

* AVX2: **21/21 CTest passed**.
* Scalar-only: **21/21 CTest passed**.
* Debug scalar ASan/UBSan: **7/7 targeted tests passed** (quant sweep/new
  decoders, Qwen3.5, Qwen2 runtime/NumPy, malformed model, real-server smoke;
  leak detection disabled). This was not a complete sanitizer suite.

Reproduce with Python 3 and NumPy installed:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DZEN_BUILD_TESTS=ON -DZEN_BUILD_SERVER=ON
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

The Qwen2 synthetic GGUF is generated in the build directory; it is not
committed as a binary. Optional trained-model tests can still be configured via
`ZEN_GGUF_FIXTURE_1` and `ZEN_GGUF_FIXTURE_2`.

## Remaining limits

No trained checkpoint was available for this run. These tests do not establish
production readiness, every architecture/quant combination, tokenizer parity,
long-context behavior, load testing, or model-quality parity. Qwen3 MoE and
other unsupported architecture paths remain separate work. The previously
reported real-model CPU prefill discrepancy is not resolved by a passing small
synthetic Qwen2 fixture. Windows/MSVC, the target 16-core CPU, AMD 9070 XT, and
GPU execution were not tested here. No OpenBLAS dependency or GPU changes were
added in this validation pass.
