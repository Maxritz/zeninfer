# ZenInfer 0.4.5-alpha

ZenInfer is an independent C++20 CPU inference engine. It reads GGUF files directly and executes dense transformer models without wrapping llama.cpp. The runtime is organised around a model-neutral frontend contract and architecture-selected adapters. The executable dense adapter supports Llama and Qwen2. A Qwen3 dense execution path has now been added, including optional per-head Q/K RMS normalisation, but it remains unverified against a real Qwen3 model and independent reference logits. Qwen3 MoE, BitNet and Bonsai production adapters are not yet executable. BitNet TQ1_0/TQ2_0 tensor decoding and synthetic architecture micro-models are included as intermediate validation.

## Implemented

- GGUF v2/v3 parsing with canonical GGML tensor storage-size validation.
- File-backed model mapping. Supported projection weights remain F32/F16 or quantised in mapped storage instead of being expanded into full FP32 matrices.
- Runtime decoders for F32, F16, Q4_0, Q5_0, Q8_0, Q3_K, Q6_K, IQ4_NL, TQ1_0 and TQ2_0.
- Block-streaming quantised GEMV and quantised prefill GEMM. Q8_0 has a fused AVX2 GEMV path; AVX2/FMA3 is selected at runtime with scalar fallback.
- Batched prompt prefill, token-by-token decode, GQA attention, RMSNorm, RoPE, SwiGLU, residual connections and KV cache.
- One-entry exact-prompt KV prefix reuse and OS-backed demand paging with asynchronous page prefetch.
- Dense Llama execution and experimental dense Qwen2 adapter with byte-level GPT-2 BPE tokenisation. Llama SentencePiece now uses unigram Viterbi segmentation and byte fallback.
- Standard, linear, dynamic NTK, YaRN and Llama-3 RoPE scaling; sliding-window attention; ALiBi bias support for metadata-declared ALiBi models.
- Greedy, temperature, top-k and top-p sampling; deterministic seed; stop strings; token bias; allowed-token constraints; cancellation.
- Speculative decoding with an optional second GGUF draft model (`--draft-model`), target verification in batched blocks, stochastic p/q acceptance and residual correction, and token-vocabulary compatibility checks.
- Common Llama 2, Llama 3 and ChatML chat prompt formatting.
- OpenAI-compatible `/v1/completions` and `/v1/chat/completions`, including SSE streaming for both endpoints.
- Persistent CPU compute pool for quantised GEMV/GEMM, Sherlock diagnostics, Breadcrumbs, Resource Governor, finite-value guards and request validation.

A GGML type being size-known by the parser does not imply a decoder exists. See `docs/FORMAT_SUPPORT.md` for the exact distinction.

## Runtime architecture

The HTTP frontend depends on `ModelRuntime`, not on a Llama implementation. GGUF architecture metadata is dispatched through `load_model_runtime`; concrete adapters are compiled into `zen_model_adapters`, separately from the generic `zen_core` library. The current dense transformer adapter contains Llama, experimental Qwen2 and an unverified Qwen3 dense execution path. `DenseTransformerRuntime` is an implementation detail of that adapter. Tokenisation is exposed as `ModelTokenizer`, with the old `LlamaSentencePiece` name retained only as a source-compatibility alias.

The architecture boundary is explicit, but the adapter portfolio is incomplete: production Qwen3 MoE, BitNet and Bonsai execution still require full architecture adapters and real-model parity. The generic factory fails closed rather than treating metadata recognition as proof of support. General Jinja interpretation, schema-constrained decoding, and application-managed NVMe eviction are not claimed by this release.

## Build

Linux:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Windows x64 / Visual Studio 2022:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

Scalar-only build:

```bash
cmake -S . -B build-scalar -DCMAKE_BUILD_TYPE=Release -DZEN_BUILD_AVX2=OFF
cmake --build build-scalar -j
ctest --test-dir build-scalar --output-on-failure
```

To include real GGUF runtime tests, configure with fixture paths:

```bash
cmake -S . -B build-fixtures -DCMAKE_BUILD_TYPE=Release \
  -DZEN_GGUF_FIXTURE_1=/path/to/model-one.gguf \
  -DZEN_GGUF_FIXTURE_2=/path/to/model-two.gguf
cmake --build build-fixtures -j
ctest --test-dir build-fixtures --output-on-failure
```

## Inspect and run a model

```bash
./build/zen_model_inspect model.gguf
./build/zen_tensor_check model.gguf
./build/zen_generate_ids model.gguf --prompt-text "Once upon a time" --max-new 32
```

The generation tool reports `prefill_tokens_per_second` and
`tokengen_tokens_per_second` separately, in addition to load/prefill/decode
latencies. Decode throughput counts decode-forward steps after the first token
selected from prefill logits.

`zen_tensor_check` attempts to decode every tensor and therefore fails closed when a model contains a storage type for which ZenInfer has no decoder. The runtime CLI reports prompt, prefill and decode timings plus Sherlock diagnostics.

## HTTP API

```bash
./build/zen_server --model model.gguf --host 127.0.0.1 --port 8000
```

Supported request controls include `max_tokens`, `temperature`, `top_k`, `top_p`, `seed`, `stop`, `logit_bias`, `allowed_token_ids` and `speculative_tokens`. Set `--draft-model draft.gguf` on the server to enable speculative decoding. Stochastic speculative decoding uses the target/draft probability ratio and residual correction; it is not a greedy-only comparison. Chat messages accept `system`, `user` and `assistant` roles. Common Llama 2, Llama 3 and ChatML prompt layouts are selected from the GGUF chat-template markers. Arbitrary Jinja templates are not interpreted.

## Architecture micro-models and throughput

Three deterministic `.zenmodel` fixtures exercise Qwen3 MoE router/top-k/expert-combine kernels, BitNet GGML TQ2_0 ternary blocks, and a Bonsai-style ternary Q2_0-g128 matrix path. They include fixed seeds and golden-logit checksums. These are kernel micro-models, not trained checkpoints or full transformer adapters.

Real GGUF runtime tests report prefill tokens/s and decode-forward tokens/s separately. Synthetic micro-model rates are printed separately and must not be compared to full-model throughput. See `docs/PERFORMANCE_0.4.5-alpha.md` and `models/synthetic/README.md`.

## Verification summary

The 0.4.5-alpha source was rebuilt with GCC 14.2 in AVX2 and scalar-only configurations. Both configurations passed all six CTest checks, including architecture micro-models, two real GGUF runtime fixtures and HTTP smoke test. Independent SentencePiece parity passed 8/8 cases on the Tinyllama 15M Stories fixture. TQ1_0/TQ2_0 decoder tests include direct GGUF tensor-read coverage. Qwen3 MoE router/expert logic is tested only in the synthetic micro-model; production Qwen3 MoE model generation and independent parity are not verified.

See `docs/BUILD_STATUS.md` for executed checks and unverified items. Windows/Linux CI and Linux coverage workflows are included. Windows/MSVC is still unverified in this environment. `scripts/windows-smoke.ps1` runs a real GGUF generation and prints prefill and tokengen rates on a Windows build.

## Explicit limitations

- Qwen2 dense support is experimental and has not yet been tested against a real Qwen2 fixture. Qwen3 dense code is present but lacks a real Qwen3 fixture and independent parity. Qwen3 MoE, BitNet and Bonsai production adapters remain incomplete; synthetic micro-models exercise selected kernels only.
- The GGUF chat-template string is used to select common template families; this is not a full Jinja interpreter. ALiBi support is implemented but still needs model-fixture parity validation.
- File-backed mmap uses OS demand paging and prefetch. ZenInfer does not yet have its own cold-layer eviction, NVMe scheduler, remote weight streaming or multi-model cache manager.
- KV prefix reuse caches one exact prompt, not a general radix/prefix tree.
- Allowed-token constraints are supported, but there is no general grammar/JSON-schema constraint engine.
- Speculative decoding supports greedy and stochastic sampling with exact draft/target vocabulary matching. It still requires compatible executable target and draft adapters; current executable support includes dense Llama, experimental Qwen2 and unverified Qwen3 dense code.
- SentencePiece parity now passes an independent 8-case corpus on the supplied Tinyllama 15M Stories fixture. GPT-2/Qwen external tokenizer parity and broader multilingual normaliser parity remain outstanding. Run `scripts/sentencepiece_reference_parity.py` to reproduce the SentencePiece check.
- Windows/MSVC runtime verification is delegated to the included CI workflow and has not been executed in this environment.
- AVX-512 and ARM/NEON are intentionally out of scope for this release.
