# Synthetic architecture micro-models

These small `.zenmodel` files are deterministic test definitions for ZenInfer's
architecture-kernel prototypes. Weights are generated from the declared seed at
load time. They are not GGUF checkpoints, trained language models, or reference
implementations of full Qwen3/BitNet/Bonsai inference.

- `qwen3-moe-tiny.zenmodel`: exercises router scoring, top-k expert selection,
  expert SwiGLU projections and weighted expert combination.
- `qwen2-dense-tiny.zenmodel` and `qwen3-dense-tiny.zenmodel`: deterministic
  dense residual-projection micro-models for exercising the fixture loader and
  dense matrix path. They do not implement full Qwen attention, RoPE, GQA,
  RMSNorm, tokenizer rules or production tensor mappings.
- `bitnet-tiny.zenmodel`: exercises GGML TQ2_0 block layout (256 weights, transposed 2-bit packing),
  group scaling and residual layers.
- `bonsai-ternary-tiny.zenmodel`: exercises a Qwen3-dense-shaped residual MLP
  with group-scaled ternary Q2_0-style codes. This fixture models the published
  Ternary Bonsai 4B family at micro scale, not Bonsai2's Hadamard/hybrid path.

The model definition format is intentionally tiny and versioned. Dense fixtures use deterministic FP32 matrices; ternary fixtures use deterministic packed ternary matrices. The architecture field selects only the micro-model execution path, not a complete production adapter. `fixture_kind`
prevents these definitions from being confused with production model files.

## Architecture references and limits

The Qwen3 MoE micro-model follows the high-level router/top-k/expert-combine
pattern. It is not a full Qwen3 MoE tensor-map or attention implementation.

The BitNet micro-model uses the GGML TQ2_0 block organisation. The production
TQ1_0/TQ2_0 decoder is implemented separately in `src/quant.cpp` and tested
against GGML block layouts.

The Bonsai fixture is based on the public Ternary Bonsai 4B description: a
Qwen3-4B-derived dense GQA/SwiGLU/RoPE/RMSNorm transformer with ternary Q2_0
group-scaled weights. The synthetic definition reduces dimensions and uses a
local g128 packed test layout. It does not claim the production model's exact
GGUF converter, tensor map or runtime. Bonsai2 is a separate Qwen3.5 hybrid
attention/recurrent model with signed Hadamard transforms and must be treated
as a separate future adapter, not as this fixture.

References:
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c
- https://huggingface.co/prism-ml/Ternary-Bonsai-4B-gguf
- https://github.com/zhongkaifu/TensorSharp/blob/main/docs/models/bonsai2.md

## Generated Qwen2 dense-layout GGUF fixture

`qwen2-dense-tiny.gguf` is a separate 12 MB GGUF fixture for exercising the
existing dense Qwen2 adapter through GGUF parsing, tensor binding, prefill and
decode. It was generated from the supplied Tiny-LLM GGUF by changing the
architecture metadata from `llama` to `qwen2` and renaming the corresponding
`llama.*` architecture metadata keys to `qwen2.*`. All 12 tensor payloads are
preserved byte-for-byte. The tensors are Llama-trained Tiny-LLM weights, not a
Qwen2-trained checkpoint. This is therefore a synthetic Qwen2 *layout/dispatch*
fixture, not production Qwen2 model validation.

Recreate it with:

```sh
python scripts/make_qwen2_layout_fixture.py \
  /path/to/Tiny-LLM.Q4_K_M.gguf \
  models/synthetic/qwen2-dense-tiny.gguf
```

The CTest `zen_runtime_qwen2_synthetic` runs the generated fixture through the
full runtime. `zen_qwen2_numpy_parity` independently compares prompt logits
against the NumPy FP32 reference implementation. Production Qwen2 checkpoint
parity remains open until a genuine Qwen2 GGUF is tested.
