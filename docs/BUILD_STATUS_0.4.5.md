# Build and test status: 0.4.5-alpha

## Verified in this container

- GCC 14.2, Linux x86-64, Release AVX2: clean configure/build, CTest 6/6 PASS.
- GCC 14.2, Linux x86-64, Release scalar-only: clean configure/build, CTest 6/6 PASS.
- Real GGUF runtime fixtures: `Tiny-LLM.Q4_K_M(2).gguf` and `tinyllama-15M-stories-Q2_K.gguf`.
- Independent SentencePiece 0.2.1 parity: 8/8 cases PASS for `tinyllama-15M-stories-Q2_K.gguf`, AVX2 and scalar builds.
- Five-run median throughput measurement: prefill and decode-forward reported separately; exact-prompt prefix-cache hits excluded.
- 200-token Tinyllama 15M generation completed. Output quality was poor/repetitive and is not treated as a correctness pass.
- Windows smoke script added.

## Not verified

- Windows/MSVC build and runtime. This environment has no MSVC STL or Windows SDK. A `clang-cl` probe failed at the standard library include stage, so it is not a valid Windows build result.
- Production Qwen3 MoE, BitNet BitLinear, Bonsai GGUF adapter/custom type, full Jinja, JSON Schema grammar, app-managed NVMe eviction, and native FP16 arithmetic.
- Qwen2 independent model parity and GPT-2/Hugging Face tokenizer parity.

See `REMAINING_WORK.md` for the acceptance criteria and outstanding implementation scope.
