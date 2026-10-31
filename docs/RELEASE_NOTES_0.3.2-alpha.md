# ZenInfer 0.3.2-alpha

## Added
- Exact scalar GGML Q5_0, Q3_K, Q6_K and IQ4_NL decoders.
- GGUF tensor payload reader converting supported tensor types to FP32.
- `zen_tensor_check` for whole-file payload conversion and finite-value checks.
- Standard dense Llama CPU runtime with RMSNorm, QKV projections, split-half RoPE, GQA, KV cache, SwiGLU, residuals and output projection.
- Tied output embeddings when `output.weight` is omitted.
- GGUF tokenizer array parsing and SPM-style byte-level merge tokeniser.
- Text and token-ID generation, with token-to-text decoding.
- `LlamaBackend` connected to the OpenAI-compatible HTTP server, including SSE content chunks.
- Sherlock timing scopes around tensor loading, attention projections, attention/softmax and FFN.
- Real GGUF fixture and execution audit.

## Verified
- GCC 14.2 Release build.
- Core CTest suite passed, including non-zero synthetic decoder cases.
- All tensors in both uploaded fixtures decoded to finite FP32 values.
- Both models completed an 8-token greedy text-generation smoke test.
- `/health`, `/v1/models`, `/v1/completions` and streamed `/v1/chat/completions` were exercised with a loaded model.

## Not claimed
- No independent logit parity test yet; tokenizer parity is only smoke-tested.
- Chat-template interpretation and non-greedy samplers are not implemented.
- No physical Ryzen or Windows/MSVC performance result.
- No quantised GEMV, prefill GEMM, MoE or NVMe weight streaming. Full OpenAI API feature parity is not claimed.
