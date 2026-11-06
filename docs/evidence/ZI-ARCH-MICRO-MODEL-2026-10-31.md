# Synthetic architecture model verification

Date: 2026-10-31
Source: ZenInfer 0.4.6 ECC-governed working copy

## Scope

This evidence covers only the deterministic `architecture_kernel_micro_model` fixtures. It does not establish production GGUF adapters, full transformer attention/KV execution, checkpoint compatibility, or language-generation quality.

## Fixtures

| Fixture | Architecture path | Seed | Expected checksum |
|---|---|---:|---:|
| `qwen3-moe-tiny.zenmodel` | Router scoring, top-2 expert selection, expert SwiGLU, weighted combination | 31031 | -41.04994414 |
| `bitnet-tiny.zenmodel` | TQ2_0 ternary block layout and group scaling | 31032 | 4.36009153 |
| `bonsai-ternary-tiny.zenmodel` | Group-scaled ternary Q2_0-style residual MLP | 31033 | 21.74052589 |

## Direct AVX2 micro-model run

Command:

```sh
./build-avx2/zen_arch_microbench \
  models/synthetic/qwen3-moe-tiny.zenmodel \
  models/synthetic/bitnet-tiny.zenmodel \
  models/synthetic/bonsai-ternary-tiny.zenmodel
```

Observed results:

- Qwen3 MoE: checksum `-41.04994414`, 512 expert selections, all 4 experts active, validation PASS.
- BitNet ternary: checksum `4.36009153`, validation PASS.
- Bonsai ternary: checksum `21.74052589`, validation PASS.
- All three reported finite logits.

Throughput numbers from this microbench are not model inference benchmarks and must not be compared with GGUF runtime tok/s.

## Full CTest runs

AVX2:

```sh
ctest --test-dir build-avx2 --output-on-failure -j2
```

Result: 8/8 passed. This includes the three architecture micro-model fixture test, Qwen2 synthetic GGUF runtime and NumPy parity, supplied GGUF runtime fixtures, benchmark parity and HTTP smoke test.

Scalar:

```sh
ctest --test-dir build-scalar --output-on-failure -j2
```

Result: 8/8 passed.

## Reproduction helper

Run:

```sh
scripts/verify_architecture_fixtures.sh build-avx2
scripts/verify_architecture_fixtures.sh build-scalar
```

## Limitations and open acceptance gates

- `create_model_runtime()` currently registers only Llama, Qwen2 and Qwen3 dense adapters.
- Qwen3 MoE, BitNet and Bonsai are not production executable adapters.
- The Qwen3 MoE fixture is not a full Qwen3 MoE transformer and has no attention/KV path.
- The BitNet fixture does not prove a complete BitLinear transformer or production BitNet GGUF support.
- The Bonsai fixture represents the documented ternary Qwen3-derived micro-path only. It does not implement Bonsai2 hybrid/recurrent attention or Hadamard transforms.
- Genuine Qwen2 checkpoint parity remains open; the current Qwen2 fixture preserves Llama-trained tensor payloads.
- All other ZI-001 through ZI-018 acceptance gaps remain as listed in `docs/REMAINING_WORK.md`.
