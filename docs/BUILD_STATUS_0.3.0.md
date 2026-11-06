# ZenInfer 0.3.0-alpha build and test status

## Build environment

- Compiler: GCC 14.2
- Configuration: CMake Release
- Host: virtualised Intel Xeon environment
- Target Ryzen hardware: not available in this build environment
- Windows/MSVC: not tested

## Verified

- `zen_core`, `zen_http`, `zen_bench`, `zen_server`, `zen_model_inspect`, `zen_core_tests` compile.
- CTest core suite passes (1/1).
- Architecture-family identification for Llama, Qwen3, BitNet and Bonsai names.
- Tensor-role mapping for representative Llama/Qwen naming patterns.
- Generic signed Q1-Q18 reference bit unpacking.
- Ternary2 decode and invalid-code rejection.
- IEEE binary16 decode.
- GGML Q4_0 and Q8_0 scalar block decode.
- Q8_0 reference GEMV against an expected dot product.
- GGUF invalid-magic rejection and valid minimal v3 metadata/alignment inspection.
- Minimal valid GGUF v3 header/metadata inspection.

## Not verified / not implemented

- No end-to-end token generation.
- No tokenizer or chat-template implementation.
- No transformer forward pass.
- No model adapter is yet executable.
- No Q2_K/Q3_K/Q4_K/Q5_K/Q6_K, IQ, TQ1_0/TQ2_0 or BF16 decoder.
- No complete GGUF tensor payload mapping or memory-mapped tensor store.
- No prefill GEMM implementation for quantised model weights.
- No attention, KV cache, sampling or MoE execution.
- No prefill/decode tokens-per-second result.
- No physical Ryzen or Windows/MSVC test.

This alpha adds tested low-level foundations. It must not be described as a functioning LLM inference runtime.
