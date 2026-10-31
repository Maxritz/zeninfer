# Real GGUF fixture audit

This document records the initial 0.3.1 parser failures and their resolution. For current decoder, tokenizer and execution evidence, see [`REAL_GGUF_EXECUTION.md`](REAL_GGUF_EXECUTION.md).

## Initial defects found

1. GGUF metadata Boolean values are one byte. The original reader consumed four bytes, desynchronising subsequent metadata.
2. GGML type 20 (`IQ4_NL`, 32 values / 18 bytes) was missing from the tensor-size table.
3. The initial 0.3.1 release recognised Q5_0, Q6_K, Q3_K and IQ4_NL without decoding them.

## Resolved in 0.3.2-alpha

- Boolean metadata is read at its declared one-byte width.
- Tokenizer token, score and type arrays are retained in the model configuration.
- IQ4_NL, Q5_0, Q6_K and Q3_K scalar dequantisers are implemented.
- The GGUF tensor reader converts all tensor payloads present in the two fixtures to FP32.
- The Llama runtime and tokenizer execute both fixtures end to end.

## Fixture inventory

| Fixture | Tensor count | Tensor types |
|---|---:|---|
| Tiny LLM | 12 | Q8_0, F32, Q5_0, Q6_K |
| TinyLlama Stories | 56 | F32, Q8_0, IQ4_NL, Q3_K |

The filenames `Q4_K_M` and `Q2_K` are not accurate descriptions of the per-tensor quant types in these files. Neither file contains Q4_K or Q2_K tensors.
