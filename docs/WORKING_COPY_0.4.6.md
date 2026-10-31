# ZenInfer 0.4.6 working copy

This is a work-in-progress implementation snapshot, not a release. It does not
claim that the repeated 0.4.5 outstanding list is complete.

## Implemented in this working copy

### Qwen2 dense-layout adapter verification

- Added a reproducible generator for a synthetic Qwen2-layout GGUF fixture.
- The fixture is produced from the supplied Tiny-LLM GGUF by rewriting
  `general.architecture` and `llama.*` metadata keys to `qwen2.*`. All 12 tensor
  payloads are preserved byte-for-byte.
- The weights remain Tiny-LLM/Llama-trained weights. The fixture verifies the
  Qwen2 dense adapter's GGUF metadata dispatch, tensor binding, prefill and
  decode path. It is not a Qwen2-trained model and cannot establish production
  Qwen2 checkpoint compatibility.
- `zen_parity_export` now reads architecture-prefixed RoPE and RMSNorm metadata
  rather than hard-coding the `llama.*` namespace.
- An independent NumPy FP32 forward implementation compares the runtime's
  prompt logits for a fixed token-ID prompt.

### Numerical parity result

| Metric | Result |
|---|---:|
| Prompt token IDs | `1,9038,2501,263,931` |
| Vocabulary | 32,000 |
| Layers | 1 |
| Maximum absolute logit error | 5.7220458984375e-6 |
| Mean absolute logit error | 7.013059548626188e-7 |
| RMS logit error | 9.407100268923686e-7 |
| Argmax | 29,892 on both paths |
| Top-5 | Exact match |

The reference receives tensors dequantised by the GGUF tensor reader, then
computes transformer math independently in NumPy. This validates forward math
and adapter dispatch, not the GGUF quantisation decoder independently.

### Throughput (container only)

Five-run median, 128 prompt tokens, 32 generated tokens, one distinct prompt per
run. Prefill and decode-forward are reported separately. These are not results
for the user's Zen 3 or Meteor Lake hardware.

| Build | Prefill tok/s | Tokengen tok/s |
|---|---:|---:|
| AVX2 | 8,406.20 | 587.45 |
| Scalar | 2,756.33 | 208.11 |

The fixture is tiny and throughput is not a language-quality measure.

## Verification

- AVX2: clean CMake configure/build; 8/8 CTest passed.
- Scalar: clean CMake configure/build; 8/8 CTest passed.
- The 8 tests include core, architecture micro-models, benchmark parity, both
  supplied real GGUF runtime fixtures, synthetic Qwen2 runtime and NumPy parity,
  and HTTP API smoke tests.
- Windows/MSVC was not run in this Linux environment.

## Remaining work

The other repeated blockers remain open: Qwen3 MoE production execution,
BitNet BitLinear adapter, Bonsai production GGUF adapter and custom type/layout,
full Jinja interpretation, hardware-native FP16 on supported hardware,
application-managed NVMe eviction, grammar-constrained JSON Schema decoding,
GPT-2/Hugging Face tokenizer parity, and genuine Qwen2 checkpoint parity.
See `docs/REMAINING_WORK.md` for the acceptance criteria and limitations.
