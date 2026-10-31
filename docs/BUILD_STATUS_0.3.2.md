# ZenInfer 0.3.2-alpha build and verification record

## Build environment

- Compiler: GCC 14.2
- Configuration: CMake Release, C++20
- Host: virtualised Intel container
- Target Ryzen hardware: not available
- Windows/MSVC: not tested

## Build and tests

- Clean Release build completed for `zen_core`, `zen_avx2`, `zen_http`, `zen_bench`, `zen_model_inspect`, `zen_tensor_check`, `zen_generate_ids`, `zen_server`, `zen_mock_server` and `zen_core_tests`.
- CTest: 1/1 passed.
- AddressSanitizer + UndefinedBehaviorSanitizer build of core/runtime tools: CTest passed; both GGUF fixtures passed complete tensor conversion and text-generation smoke tests with leak detection enabled.
- The sanitizer build of the full HTTP server was not completed within the build time limit. The normal Release server was exercised separately.

## Real model fixtures

| Fixture | Tensor payloads | Tokenizer | Text generation |
|---|---:|---|---|
| Tiny LLM | 12/12 | 32,000 tokens/scores/types | PASS |
| TinyLlama Stories | 56/56 | 32,000 tokens/scores/types | PASS |

- `zen_tensor_check` converted all tensor payloads to finite FP32 values.
- `zen_generate_ids --prompt-text "Once upon a time" --max-new 8` completed on both models.
- Tiny LLM output: `, and\nthe other of the time`.
- TinyLlama Stories output: `, there was a big bear who was`.
- No independent logit parity against a separate engine has been run.
- `zen_bench 3` passed scalar-versus-selected AVX2 checks. On this virtualised 4-vCPU/4-GiB host, FP32 GEMV speedups ranged from 3.24x to 4.17x and the simple 4x8 GEMM kernel from 20.29x to 29.61x. These are host-specific smoke measurements, not Ryzen targets.

## HTTP validation

With `zen_server --model tinyllama-15M-stories-Q2_K.gguf`:
- `/health`: 200, `engine_loaded=true`.
- `/v1/models`: returned the loaded model ID.
- `/v1/completions`: returned `, there was a big` for `Once upon a time`, `max_tokens=5`.
- `/v1/chat/completions`: emitted role, incremental content chunks and a final `length` finish reason over SSE.

## Limitations

- Only standard dense Llama is executable. Qwen, BitNet and Bonsai remain recognisers.
- The runtime expands all supported tensors to FP32 in RAM.
- Greedy sampling only; temperature is not applied.
- Chat messages are concatenated as plain text; no chat-template interpreter.
- Quantised GEMV, prefill GEMM, MoE execution, expert cache, NVMe streaming and full Sherlock trap suite remain unimplemented.
- Resource Governor checks model and per-request KV budgets, but does not reserve aggregate memory across concurrent sessions.
- Physical Ryzen performance, Windows/MSVC and independent logit parity remain unverified.
