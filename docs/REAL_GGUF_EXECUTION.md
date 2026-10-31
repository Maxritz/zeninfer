# Real GGUF execution audit (2026-10-30)

## Fixtures

| File | Model metadata | Tensor inventory | Tokenizer |
|---|---|---|---|
| `Tiny-LLM.Q4_K_M(2).gguf` | Llama, 1 layer, embedding 192, 2 query heads, 1 KV head, FFN 1024 | Q8_0 x2, F32 x3, Q5_0 x6, Q6_K x1 | 32,000 tokens, scores and token types |
| `tinyllama-15M-stories-Q2_K.gguf` | Llama, 6 layers, embedding 288, 6 query heads, 6 KV heads, FFN 768 | F32 x13, Q8_0 x1, IQ4_NL x36, Q3_K x6 | 32,000 tokens, scores and token types |

The filenames are not tensor-format inventories. The first file contains no Q4_K tensors; the second contains no Q2_K tensors.

## Implemented and exercised

1. GGUF v3 metadata, tokenizer arrays and tensor directory parsing.
2. Exact storage-size validation and tensor payload bounds.
3. Scalar dequantisation of all tensor types present in both files.
4. FP32 loading of every named tensor.
5. Llama SentencePiece-style byte-level merge tokenisation and token decoding, including byte fallback.
6. Llama forward pass and greedy text generation on both models.
7. OpenAI-compatible HTTP backend for text completions, chat completions and SSE streaming.

`zen_tensor_check` reported 12/12 and 56/56 tensors converted. Total decoded elements were 12,988,992 and 15,191,712 respectively.

Example command:

```bash
zen_generate_ids model.gguf --prompt-text "Once upon a time" --max-new 8
```

Observed output in this build:

- Tiny LLM: `, and\nthe other of the time`
- TinyLlama Stories: `, there was a big bear who was`

These are smoke-test outputs, not proof of numerical parity or model quality. A separate trusted-runtime logit comparison is outstanding.

## Execution implementation

- Tokenizer token, score and type arrays are read from GGUF metadata.
- SPM-style UTF-8 character pair merges use token scores, with byte fallback.
- BOS/EOS and metaspace whitespace handling are supported for the tested tokenizer layout.
- Embedding lookup from `token_embd.weight`.
- RMSNorm using `llama.attention.layer_norm_rms_epsilon`.
- Q/K/V projection with GGML matrix layout (`[input, output]` dimensions).
- Split-half RoPE using `llama.rope.dimension_count` and `llama.rope.freq_base`.
- GQA head-to-KV-head mapping.
- Causal attention with a per-layer KV cache.
- Attention output projection and residual addition.
- SwiGLU feed-forward block and residual addition.
- Final RMSNorm and vocabulary projection.
- Tied output embeddings when `output.weight` is absent.

## HTTP validation

With `zen_server --model tinyllama-15M-stories-Q2_K.gguf`:

- `/health` reported `engine_loaded: true`.
- `/v1/models` returned the loaded model ID.
- `/v1/completions` returned `, there was a big` for prompt `Once upon a time` with `max_tokens=5`.
- `/v1/chat/completions` emitted role, incremental content chunks and `[DONE]` over SSE.

The Chat Completions adapter currently concatenates message contents as plain text. It does not interpret a model-specific chat template. Sampling is greedy; the API's temperature field is not yet applied.

## Smoke timing

One run in the virtualised Intel build container, with instrumentation enabled:

| Model | Prompt | Generated tokens | Load | Prefill | Decode | Decode rate |
|---|---|---:|---:|---:|---:|---:|
| Tiny LLM | 5 tokens | 8 | 91 ms | 49 ms | 71 ms | 98 tokens/s |
| TinyLlama Stories | 5 tokens | 8 | 175 ms | 138 ms | 143 ms | 49 tokens/s |

These are single-run smoke measurements, not stable benchmark results. The decode rate excludes the first token selection and reflects the subsequent decode forward passes. They are not Ryzen/Zen3 measurements.

## Outstanding validation

- Compare logits and token IDs against a trusted independent Llama runtime using the same model files and token IDs.
- Verify tokenizer parity on a representative multilingual and byte-fallback corpus.
- Confirm RoPE variant and any architecture-specific metadata for both checkpoints.
- Add non-greedy samplers and model-specific chat-template interpretation.
- Benchmark warm decode separately from model load/dequantisation on Ryzen 7 PRO 4750U and Zen3 hardware.
- Validate Windows/MSVC builds and run ASan/UBSan on fixture loads, tokenisation and generation.
