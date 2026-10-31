# ZenInfer 0.4.0-alpha

## Runtime and memory
- Replaced eager FP32 expansion of supported projection matrices with file-backed GGUF mapping.
- Added block-streaming quantised GEMV and quantised prefill GEMM, with ResourceGovernor-selected row parallelism.
- Added batched prompt prefill, asynchronous OS page prefetch and one-entry exact-prompt KV prefix reuse.
- Added experimental dense Qwen2 adapter and byte-level GPT-2 tokenizer path.

## Correctness and API
- Corrected the GGML storage-size map using canonical GGML type IDs; removed legacy IDs remain rejected.
- Added standard/linear/dynamic-NTK/YaRN/Llama-3 RoPE scaling and sliding-window attention.
- Added temperature, top-k, top-p, seeded sampling, stop strings, logit bias, allowed-token constraints and in-loop cancellation.
- Added greedy speculative decoding with a separate draft GGUF, batched target verification and exact vocabulary compatibility checks.
- Added common Llama 2/Llama 3/ChatML chat prompt formatting and `/v1/completions` SSE.
- Added HTTP schema/range checks and non-finite intermediate/output guards.

## Verification
- Added optional real-GGUF CTest fixtures, runtime generation tests, HTTP API smoke tests and benchmark parity/performance gates.
- Added Windows/Linux CI and Linux coverage workflow.
- See `BUILD_STATUS.md` for verified results and explicit remaining limitations.
