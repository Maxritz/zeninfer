# ZenInfer 0.4.4-alpha performance report

## Measurement definitions

- **Prefill tok/s** = prompt tokens / measured batched-prefill wall time.
- **Tokengen tok/s** = decode-forward steps / measured decode-forward wall time.
  The first output token is sampled from prefill logits, so it is not counted as
  a decode-forward step. Sampling and HTTP overhead are not included.
- The runtime benchmark uses a deterministic 128-token prompt, one distinct
  prompt for warm-up, and 32 requested output tokens. Prefix-cache hits are
  rejected from the timed run (`cache_hit=no`).
- Synthetic architecture fixtures run their own forward kernels. Their rates
  are **not comparable to a full transformer model** and are not marketed as
  production model throughput.

## Real GGUF fixture results

Measured in this container on an AMD EPYC 9V74 virtual CPU allocation (5 vCPUs),
Linux x86-64, GCC 14.2, Release build, AVX2 backend. These are container results,
not results from the user's local hardware.

| Fixture | Prompt tokens | Generated tokens | Prefill | Decode forward |
|---|---:|---:|---:|---:|
| `Tiny-LLM.Q4_K_M(2).gguf` | 128 | 32 | 8,855 tok/s | 616 tok/s |
| `tinyllama-15M-stories-Q2_K.gguf` | 128 | 32 | 997 tok/s | 120 tok/s |

Single-run indicative measurements. Repeat on the target machine before using
these as a release performance baseline.

## Synthetic architecture micro-model results

Each definition has a 64-token prefill and 64 generated tokens. Weights are
seed-generated and golden-logit checksums are fixed in the model definitions.

| Synthetic model | Shape | Prefill | Tokengen | Additional check |
|---|---|---:|---:|---|
| `zen-qwen3-moe-tiny` | hidden 64, 2 layers, 4 experts, top-2 | 39,906 tok/s | 38,473 tok/s | 512 expert selections, all 4 experts selected |
| `zen-bitnet-tiny` | hidden 64, 3 layers, TQ2_0, group 256 | 20,317 tok/s | 17,704 tok/s | golden logits checksum |
| `zen-bonsai-ternary-tiny` | hidden 128, 2 layers, custom ternary Q2_0-g128 | 8,855 tok/s | 9,076 tok/s | golden logits checksum |

Synthetic rates are kernel-prototype throughput only. The fixture does not
implement full attention, production tokenisation, chat templates, full GGUF
model loading, KV cache semantics or reference-logit parity.

## BitNet quantisation work

The GGUF tensor decoder now implements GGML `TQ1_0` (256 values / 54 bytes) and
`TQ2_0` (256 values / 66 bytes), including the TQ2_0 4-group x 32-element
interleaved bit layout. `gguf::read_tensor_f32` has an end-to-end TQ2_0 fixture.
This is quant-format support, not yet the complete BitNet transformer adapter
or BitLinear activation quantisation path.

## Test results

AVX2 Release build: 6/6 CTests passed, including core quantisation tests,
synthetic architecture micro-models, benchmark parity, two real GGUF runtime
fixtures and HTTP API smoke test.

Windows/MSVC has not been run in this environment.
