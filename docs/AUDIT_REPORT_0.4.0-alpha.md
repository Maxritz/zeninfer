# ZenInfer 0.4.0-alpha: Goli-mat-de implementation audit

**Baseline:** `ZenInfer-0.3.3-alpha.zip`  
**Audit method:** Goli-mat-de, evidence-led implementation audit  
**Scope:** source implementation, build wiring, runtime path, tests, documentation and release packaging  
**Local toolchain:** GCC 14.2, Linux x86-64  
**Disposition:** core release blockers fixed; several architecture-level items remain explicitly incomplete.

## Findings disposition

| Original finding | Disposition | Evidence / implementation |
|---|---|---|
| Quantised GEMV not wired into runtime | **Fixed** | Supported matrix tensors remain mapped; `tensor_matvec` dispatches through the execution planner into `quant::gemv`. Real GGUF runtime fixtures exercise the path. |
| No fused GEMV for Q5_0, Q3_K, Q6_K, IQ4_NL | **Substantially fixed** | These formats now use block-streaming quantised GEMV: each quant block is decoded into bounded scratch and dotted before moving to the next block. AVX2 accelerates the dot. Q8_0 retains a dedicated fused AVX2 kernel. Dedicated SIMD unpack/dequant kernels for Q5_0/Q3_K/Q6_K/IQ4_NL remain performance work. |
| `tensor_bytes()` missing IQ type IDs | **Fixed** | Canonical upstream GGML storage-size entries added for IQ/TQ and scalar types. Removed/deprecated IDs fail closed instead of using guessed sizes. Unit tests construct GGUFs for the affected IDs. |
| No multi-threading; worker count unused | **Fixed for projection kernels** | Execution planner obtains ResourceGovernor worker policy; quantised GEMV/GEMM partition output rows across worker threads. Tests compare one-worker and four-worker outputs. Attention loops and HTTP concurrency are separate. There is not yet a persistent compute thread pool. |
| No memory mapping; FP32 expansion | **Fixed for supported matrix weights** | GGUF is file-mapped. Supported F32/F16/quantised projection weights use mapped bytes. Norm and small auxiliary tensors remain FP32 vectors. OS handles demand paging. |
| Temperature/top-k/top-p ignored | **Fixed** | Runtime sampling supports greedy, temperature, top-k, top-p and seeded sampling. Runtime fixture tests cover seeded repeatability and greedy mode. |
| No model adapter dispatch | **Partially fixed** | Runtime now selects an executable dense Llama or experimental dense Qwen2 path. GPT-2 byte-level BPE support was added for Qwen2. Qwen2 has no real-model parity fixture yet. Qwen3, BitNet, Bonsai and MoE remain unsupported. |
| No execution planner | **Fixed for current dense execution** | `ExecutionPlanner` selects prefill GEMM versus decode GEMV and chooses worker counts. Prefill and decode now have separate entry paths. |
| KV cache allocated per request, never reused | **Partially fixed** | One exact-prompt KV prefix cache is retained when it fits the cache budget. It is not a general radix/prefix cache and still copies the cached KV arrays into a request-local decode buffer. |
| Sequential prefill GEMV | **Fixed** | Prompt prefill batches Q/K/V, attention output and FFN projections through quantised/FP32 GEMM. Causal attention remains token-position aware. |
| No quantised prefill GEMM | **Fixed** | `quant::gemm` consumes mapped quantised weights block-by-block and writes token-major output. Real fixture logits were compared against the prior sequential baseline. |
| No F16 compute path | **Partially fixed** | F16 matrix weights can remain mapped and are consumed through a row conversion path. Native FP16 arithmetic/SIMD is not implemented; accumulation remains FP32. |
| No MoE execution | **Not fixed** | Router/top-k expert dispatch, expert tensor layout, expert caching and combination are not implemented. MoE metadata is rejected explicitly rather than silently run as dense. |
| No sliding-window attention | **Fixed** | Sliding-window metadata is read and both batched prefill and decode attention restrict their causal range accordingly. |
| No RoPE variants | **Partially fixed** | Standard, linear, dynamic NTK, YaRN and Llama-3 frequency scaling are implemented, including partial rotary dimensions. LongRoPE/MRoPE/vision RoPE are not implemented and variant parity has not been established across model families. |
| No chat templates | **Partially fixed** | Common Llama 2, Llama 3 and ChatML layouts are selected from GGUF template markers. This is not a general Jinja interpreter. |
| No stop sequences | **Fixed** | Runtime checks stop strings and removes them from returned text. Streaming holds a suffix buffer so stop text is not emitted prematurely. |
| No `/v1/completions` SSE | **Fixed** | Completion and chat-completion SSE are covered by HTTP integration tests. |
| No NVMe/disk offload | **Partially fixed** | File-backed mapping uses OS demand paging; Linux `madvise(MADV_WILLNEED)` and Windows `PrefetchVirtualMemory` hints are used. ZenInfer does not yet implement explicit cold-layer eviction, an NVMe scheduler, remote streaming or user-managed residency. |
| No prefetch pipeline | **Fixed for mapped layer pages** | A background prefetch task requests the next layer's mapped page range while the current layer executes, subject to ResourceGovernor policy. |
| No NaN/Inf propagation checks | **Fixed for core compute outputs** | Norm, projection, attention score/denominator, residual and logits paths reject non-finite values. This is not a formal proof against every possible numerical fault. |
| No JSON request validation | **Fixed for exposed request fields** | Prompt/message types and sizes, roles, token counts, temperature/top-k/top-p, stop strings, seed, logit bias, allowed IDs and speculative controls are checked before dispatch. |
| No tokenizer parity tests | **Partially fixed** | Added synthetic GPT-2 byte-BPE merge/space-boundary tests and real-fixture encode/decode smoke checks. Independent parity against upstream SentencePiece and Hugging Face tokenizers remains outstanding. |
| No end-to-end CTest generation | **Fixed** | Optional real-GGUF runtime CTests load, tokenize, generate, sample, stop, cancel and exercise speculative decoding. |
| No sampling tests | **Fixed** | Seeded sampling, greedy output, top-k/top-p path, logit bias and allowed-token constraints are covered. |
| No Windows runtime verification | **CI added; execution pending** | Windows/MSVC and Linux GitHub Actions jobs build and run tests. No Windows runner result was available during this local audit. |
| No speculative decoding | **Partially fixed** | Greedy draft/target speculative decoding is implemented with incremental draft KV, batched target verification, accepted/rejected proposal counters and a draft vocabulary compatibility check. Stochastic speculative sampling is not implemented. |
| No token bias/logit filters | **Fixed in API/runtime** | `logit_bias` and `allowed_token_ids` are accepted and applied before selection. |
| No grammar/constrained sampling | **Partially fixed** | Token allow-list constraints exist. General grammar, JSON-schema and incremental parser constraints do not. |
| No code coverage CI | **Fixed in workflow** | Linux coverage job builds with gcov instrumentation and uploads an XML report. Workflow execution is pending CI. |
| No performance regression test | **Fixed with a conservative gate** | CTest runs quick numerical benchmarks and enforces a 1.05x AVX2-over-scalar minimum when AVX2 is selected. Hosts without AVX2 skip the speed threshold. This is a coarse kernel regression gate, not a hardware-independent throughput guarantee. |
| Stale `BUILD_STATUS.md` | **Fixed** | `docs/BUILD_STATUS.md` is now the canonical 0.4.0-alpha verification record. |
| Placeholder `docs/samples/README.md` | **Fixed** | Replaced with usage and interpretation guidance for diagnostic samples. |

## Verification results

| Build / test | Result |
|---|---|
| GCC 14.2 Release, AVX2 enabled | Build PASS; 5/5 CTest PASS |
| GCC 14.2 Release, scalar-only | Build PASS; 5/5 CTest PASS |
| ASan + UBSan, scalar-only, leak detection enabled | Build PASS; 5/5 CTest PASS |
| Real GGUF fixture A: `Tiny-LLM.Q4_K_M(2).gguf` | Runtime PASS; speculative draft fixture B PASS |
| Real GGUF fixture B: `tinyllama-15M-stories-Q2_K.gguf` | Runtime PASS; speculative draft fixture A PASS |
| Real-fixture inspection, full tensor decode and generation script | PASS on both fixtures |
| NumPy reference parity suite | 8/8 prompts; all finite; argmax and top-5 match; max absolute error `5.7220459e-6` |
| HTTP mock API: completions SSE, chat SSE, malformed sampling request | PASS |
| Manual actual-model HTTP completion with target + draft GGUF | PASS |
| AVX2 logits vs 0.3.3 baseline, fixture A | 32,000 logits; max absolute error `6.6757202e-6`; argmax identical (`29892`) |
| AVX2 logits vs 0.3.3 baseline, fixture B | 32,000 logits; max absolute error `7.3909760e-6`; argmax identical (`29892`) |
| Windows/MSVC build and runtime | NOT RUN LOCALLY; CI workflow added |
| Real Qwen2 GGUF parity | NOT RUN |

The two GGUF fixture files are not included in this source ZIP. Supply their paths through `ZEN_GGUF_FIXTURE_1` and `ZEN_GGUF_FIXTURE_2` to reproduce the fixture tests.

## Remaining release risks

1. MoE execution is still absent. Qwen2-MoE/Mixtral must not be described as supported.
2. Qwen2 dense support is experimental until tested with a real Qwen2 GGUF and an independent reference.
3. General Jinja templates, grammar/JSON-schema constraints, stochastic speculative decoding and a general KV prefix tree remain unimplemented.
4. OS demand paging is not a custom NVMe/offload scheduler. Very large models may still fail under address-space, I/O or process memory constraints.
5. Dedicated SIMD unpack/dequant kernels for Q5_0/Q3_K/Q6_K/IQ4_NL and native FP16 arithmetic remain optimisation work.
6. Windows/MSVC and CI coverage results must be checked after the workflow runs.
7. AVX-512 and ARM/NEON were explicitly excluded from this release scope.

## Release judgement

The original 0.3.3-alpha critical runtime wiring and test-integrity gaps are materially reduced. The 0.4.0-alpha package is suitable for further controlled testing on the two validated Llama fixtures. It is **not** a complete model-neutral or MoE-capable inference engine, and the remaining risks above must stay visible in release documentation.
