# ZenInfer execution architecture

ZenInfer separates model metadata, execution planning, quantisation and CPU kernels. The current release is a dense Llama-family CPU engine with an experimental dense Qwen2 adapter. BitNet, Bonsai and MoE families are recognised by the GGUF metadata layer but are not yet executable.

```text
GGUF file
  |
GGUF v2/v3 parser + canonical GGML storage-size table
  |
Model-family recognition and runtime adapter selection
  |-- Llama dense
  |-- Qwen2 dense (experimental)
  |
File-backed model mapping
  |-- quantised matrix weights remain compressed
  |-- non-matrix/norm tensors converted to FP32
  |-- OS demand paging + asynchronous page prefetch
  |
Execution planner
  |-- prompt prefill: quantised/FP32 GEMM path
  |-- token decode: quantised/FP32 GEMV path
  |-- ResourceGovernor worker selection
  |
Transformer execution
  |-- embeddings / RMSNorm / QKV / RoPE
  |-- causal or sliding-window GQA attention
  |-- KV cache + identical-prompt prefix reuse
  |-- SwiGLU FFN / residual / output projection
  |
Sampling
  |-- greedy / temperature / top-k / top-p
  |-- greedy speculative draft/target verification
  |-- seed / stop strings / logit bias / allowed-token constraints
  |
Tokenizer + chat prompt formatter
  |
CLI / OpenAI-compatible HTTP API
```

## Implemented invariants

- GGML storage-size recognition is not treated as a decoder promise.
- Unsupported quantisation decoders fail closed.
- Quantised model matrices are not eagerly expanded into full FP32 tensors.
- Prefill and decode use distinct planner phases and GEMM/GEMV entry points.
- Worker count and prefetch timing do not change the mathematical operations.
- Intermediate logits and attention values are checked for non-finite values.
- HTTP request fields are type/range validated before inference.
- Sherlock traces and Breadcrumb events expose prefill, decode, quantised GEMV/GEMM and cache activity.

## Current limitations

- Dense Llama is the validated runtime path. Qwen2 dense support is experimental and needs a real Qwen2 GGUF parity fixture.
- Qwen3, Qwen MoE, Mixtral, BitNet and Bonsai execution are not implemented.
- RoPE scaling implemented: standard, linear, dynamic NTK, YaRN and Llama-3 frequency scaling. LongRoPE/MRoPE/vision RoPE are not implemented.
- Chat prompt formatting recognises common Llama 2, Llama 3 and ChatML families. It is not a general Jinja interpreter for arbitrary GGUF chat templates.
- Mapped files rely on OS demand paging. ZenInfer does not yet implement its own explicit cold-layer eviction, NVMe scheduler or remote storage protocol.
- KV prefix reuse currently caches one exact prompt per loaded model, not a general radix/prefix cache.
- Allowed-token constraints and token bias are available; a general grammar engine and stochastic speculative sampling are not yet implemented.
