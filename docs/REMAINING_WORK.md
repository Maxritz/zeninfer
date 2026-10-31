# ZenInfer remaining implementation and verification work

This is the release gate after 0.4.3-alpha. A feature is not complete merely because its architecture name, metadata or interface exists.

## P0: executable architecture coverage

| ID | Work | Required implementation | Acceptance evidence |
|---|---|---|---|
| ZI-001 | Qwen3 dense adapter | **PARTIAL:** factory dispatch and optional per-head Q/K RMSNorm path added. **Real Qwen3-4B Q8_0 now loads and passes the full runtime fixture** (head_dim decoupled from embedding/heads via attention.key_length; attn_q/output validation fixed; norm-tensor role classification fixed). Independent NumPy logits parity on the real checkpoint: worst rel diff 6.4e-06, top-8 exact. Remaining: architecture-specific fixture wired into CTest | Real Qwen3 GGUF generation plus independent reference logits/token parity |
| ZI-002 | Qwen3 MoE | **PROTOTYPE ONLY:** deterministic micro-model now exercises router scoring, top-2 selection, expert SwiGLU projections and weighted combine. Production GGUF tensor mapping, full attention/KV path and adapter integration remain | Real Qwen3 MoE fixture; router/expert unit tests and independent reference parity |
| ZI-003 | BitNet adapter | **PARTIAL:** **Real BitNet b1.58-2B TQ1_0 GGUF now loads and passes the full runtime fixture** (bitnet.* metadata mapping, subln RMS norms after attention and inside FFN, tied lm_head — graph per ggml-org/llama.cpp src/models/bitnet.cpp and arXiv 2504.12285). Independent NumPy logits parity on the real checkpoint: worst rel diff 3.9e-06, top-8 exact. Remaining: BitLinear activation quantisation (A8) variant, fused ternary kernels | Real BitNet GGUF fixture and independent output parity |
| ZI-004 | Bonsai adapter | **PARTIAL:** the qwen35-hybrid (Ternary-Bonsai-2) path now loads `prism.hadamard.*` + PTQ1_0 packs and executes without the crash class reported in the field: five OOB/oversized allocations fixed, GDN gate/conv/norm/Hadamard-order defects corrected against PrismML-Eng/llama.cpp. New synthetic qwen35 fixture + independent NumPy reference wired into CTest as a smoke guard (ASan clean). **Numerical parity not achieved** — `--numeric` still fails on first-token logits. See `docs/evidence/ZI-004-qwen35-hybrid-loader-and-gdn.md` | Real Bonsai-2 GGUF fixture and independent output parity |

## P1: generic services and memory

| ID | Work | Required implementation | Acceptance evidence |
|---|---|---|---|
| ZI-005 | General Jinja templates | Real Jinja-compatible interpreter or a clearly specified compatible subset, including control flow, filters, loops, tool calls and generation suffix | Reference-template corpus compared with Hugging Face tokenizer/template output |
| ZI-006 | FP16 arithmetic/conversion | **PARTIAL:** software IEEE binary16 float32 encode/decode added, including ties-to-even, subnormal, overflow, infinity and NaN handling. Exhaustive 65,536-pattern half round-trip and boundary tests pass in scalar and AVX2 builds. This is not native FP16 arithmetic; Zen 3/Meteor Lake do not expose general packed FP16 arithmetic and AVX-512-FP16 remains excluded. | Independent float32-to-half differential corpus; Windows/MSVC run; hardware arithmetic only on a supported non-excluded ISA |
| ZI-007 | Application-managed NVMe residency | Explicit page/tensor residency map, bounded cache, eviction policy, async read/prefetch, pinning and back-pressure | Small-RAM test that proves bounded resident bytes and repeatable cold/warm paging |
| ZI-008 | Grammar and JSON Schema | Schema validation/compilation to grammar, incremental parser state and constrained token selection | JSON schema corpus, invalid-prefix rejection, generated-output validation |
| ZI-009 | Independent tokenizer parity | **PARTIAL:** Llama SentencePiece unigram Viterbi path now matches Google SentencePiece 0.2.1 on 8 cases for the supplied Tinyllama 15M Stories GGUF. GPT-2/HF BPE and broader multilingual normaliser parity remain | Exact token ID parity over multilingual and special-token corpus for each tokenizer family |
| ZI-010 | Qwen2 independent parity | **PARTIAL:** synthetic fixture parity plus **genuine Qwen2.5-Coder-3B checkpoint now loads, passes the full runtime fixture, and shows independent NumPy logits parity (worst rel diff 5.5e-06, top-8 exact) including the qwen2 attention-bias path**. Remaining: generated-token parity on the genuine checkpoint | Genuine Qwen2 GGUF, independent reference logits and generated-token parity |
| ZI-011 | Persistent compute pool | **PARTIAL:** process-wide pool integrated into quantised GEMV/GEMM; deterministic partitions, single-worker path, four simultaneous callers with isolated 1,000-item jobs, and worker exception propagation are tested. Cancellation, shutdown/lifecycle, queue saturation, race-detector and broad scaling stress remain | Cancellation semantics, ThreadSanitizer/race run, shutdown and queue saturation tests, scaling across worker counts |

## P2: correctness, coverage and performance

| ID | Work | Required implementation | Acceptance evidence |
|---|---|---|---|
| ZI-012 | Quant format completeness | **MOSTLY COMPLETE:** all GGML storage types in the canonical table now have runtime decoders — Q2_K/Q4_K/Q5_K/Q8_K, IQ1_S/IQ1_M, IQ2_XXS/IQ2_XS/IQ2_S, IQ3_XXS/IQ3_S, IQ4_XS, MXFP4, plus the community ROCmFP4 pair (types 100/101, `ciru-ai/ROCmFPX` Codebook10 + UE4M3). Differential harness `zen_iq_check` runs 88 checks against an independent Python reference with worst relative diff 0 in scalar and AVX2. Real-model validation: MXFP4 bit-exact on `qwable-v1-mxfp4_moe.gguf` (262,144 values) and ROCmFP4 bit-exact on granite/Spark models (393,216 values across both variants). Remaining: fused ternary kernels, real-model token-parity runs on K-quant weights. See `docs/evidence/ZI-012-kquant-iq-mxfp4-decoders.md` | Block and full-tensor fixtures for each required format |
| ZI-013 | Attention variants | **PARTIAL:** ALiBi slope/bias and prefill/decode integration added; metadata-driven model fixture and independent attention comparison remain. Other declared variants remain open | Model fixtures and reference attention comparisons |
| ZI-014 | KV cache manager | Bounded cache policy, reusable prefix/radix structure, lifecycle and concurrent request isolation | Cache hit/miss, eviction and concurrency tests |
| ZI-015 | HTTP and scheduler | Continuous batching/request scheduling and complete streaming/error semantics | Multi-client load and cancellation tests |
| ZI-016 | Security and robustness | **PARTIAL:** GGUF metadata recursion capped at 64 levels, u64-width gate on config fields, JSON parser depth capped at 128, reader-agnostic `parse_gguf<R>` core with `inspect_bytes` in-memory surface, structured GGUF fuzzer (4096 LCG cases + 3 regressions, no crash/hang) and hostile-input regression tests in the default suite. Remaining: sanitizer builds, coverage-guided libFuzzer run, resource-exhaustion fault injection. See `docs/evidence/ZI-016-gguf-json-hardening.md` | Sanitizer, fuzz and fault-injection results |
| ZI-017 | Windows/MSVC | Windows CI workflow and `scripts/windows-smoke.ps1` added. Local MSVC execution is unavailable in this Linux container; user-side Windows validation remains required | Actual Windows runner logs and real GGUF generation, not merely a workflow file |
| ZI-018 | Benchmark quality | **PARTIAL:** runtime tests report five-run median 128-token prefill tok/s and decode-forward tok/s. Synthetic architecture rates remain separate. Named user hardware baseline and regression thresholds remain | Reproducible benchmark report on named hardware |

## Partial-work summary (all items with remaining effort)

**P0 — architecture adapters (highest priority, all blocked on real-fixture work):**
- ZI-001 Qwen3 dense adapter — PARTIAL
- ZI-002 Qwen3 MoE — PROTOTYPE ONLY
- ZI-003 BitNet adapter — PARTIAL
- ZI-004 Bonsai adapter — PARTIAL (loads and runs; numerical parity open)

**P1 — generic services:**
- ZI-005 General Jinja templates — OPEN (no compatible interpreter yet)
- ZI-006 FP16 conversion — PARTIAL (software path done; Windows/MSVC run outstanding)
- ZI-007 App-managed NVMe residency — OPEN
- ZI-008 Grammar / JSON Schema constrained decoding — OPEN
- ZI-009 Tokenizer parity — PARTIAL (Llama SP done; GPT-2/HF BPE parity open)
- ZI-010 Qwen2 checkpoint parity — PARTIAL (synthetic fixture done; genuine checkpoint open)
- ZI-011 Compute pool — PARTIAL (correctness done; cancellation/shutdown/TSan/scaling open)

**P2 — correctness/coverage/performance:**
- ZI-012 Quant formats — MOSTLY COMPLETE (decoders + real-model validation done; fused ternary kernels and K-quant token-parity open)
- ZI-013 Attention variants — PARTIAL (ALiBi done; other variants + reference comparison open)
- ZI-014 KV cache manager — OPEN
- ZI-015 HTTP/scheduler (continuous batching) — OPEN
- ZI-016 Security — PARTIAL (hardening + fuzz landed; sanitizers/coverage-guided fuzzing open)
- ZI-017 Windows/MSVC — workflow present, local validation outstanding
- ZI-018 Benchmark quality — PARTIAL (runtime metrics exist; named-hardware baseline open)

## Already present by 0.4.3-alpha, subject to current scope

- Quantised GEMV and quantised batched prefill GEMM for the supported decoder formats.
- AVX2 Q8_0 fused GEMV and scalar fallback.
- Persistent compute pool for quantised GEMV/GEMM row partitions (0.4.3); projection-level worker scheduling still needs broader integration.
- File-backed mapping and OS demand paging/prefetch; not app-managed eviction.
- Batched prefill and token decode separation.
- Sampling, top-k/top-p, stop strings, token bias, allow-list filtering and cancellation.
- Standard/linear/dynamic NTK/YaRN/Llama-3 RoPE scaling, sliding-window attention, and ALiBi (0.4.3).
- Finite-value checks and request validation in the current exposed API.
- Greedy and stochastic speculative decoding for compatible dense adapters.

## Platform exclusions

AVX-512 and ARM/NEON are excluded by the user's instruction. They are not release blockers. Windows x64/MSVC remains in scope and must be verified.

## Work log (reverse chronological)

## 2026-11-01 ZI-001/ZI-003/ZI-010 real-checkpoint execution and independent parity

- **Qwen3 decoupled head_dim** (`RuntimeConfig::head_dim`): real Qwen3-4B has head_dim 128 with embedding 2560 and 32 heads (attn_q 2560x4096, attn_output 4096x2560). Loader reads `attention.key_length`, falls back to block attn_q width, then embedding/heads; both forward paths, KV sizing and speculative sizing use the decoupled width.
- **Norm-tensor role classification fixed**: `attn_q_norm`/`attn_k_norm` matched the `attn_q` projection substring and were raw-mapped with empty FP32 values. Generic `*_norm.weight` is now classified before projection names.
- **BitNet execution variant** (`bitnet_variant`, per ggml-org/llama.cpp `src/models/bitnet.cpp`, arXiv 2504.12285): `bitnet.*` metadata mapping, factory dispatch, subln RMS norms after attention (before wo) and after SwiGLU (before ffn_down), tied lm_head. The released artifact runs a SiLU-gated FFN (upstream graph: LLM_FFN_SILU with gate+up tensors) — verified by coherent generation; the paper's ReLU2 describes the training recipe.
- **Independent NumPy forward reference** (`scripts/model_numpy_reference.py`): own GGUF parser and dequantizers (TQ1_0/Q8_0/F16/F32), full graph in NumPy, GGUF ne[0]-contiguous orientation handled; compares against `zen_parity_probe` dumps.
- Parity on real checkpoints (last-position logits, full vocabulary):
  - BitNet b1.58-2B: worst rel 3.9e-06, argmax + top-8 exact.
  - Qwen3-4B: worst rel 6.4e-06, argmax + top-8 exact.
  - Qwen2.5-Coder-3B: worst rel 5.5e-06, argmax + top-8 exact (qwen2 attention-bias path).
- Runtime fixtures PASS on all three real models (prefill/decode benchmark, seeded determinism, KV prefix reuse, constrained decoding, speculative decoding, cancellation).
- Regression: AVX2 7/7, scalar 7/7 CTest.

## 2026-11-01 ROCmFP4 (GGUF types 100/101) support

- Traced the unknown types in `granite-4.1-3b-Q4_0_ROCMFP4_COHERENT.gguf` and `Spark-X2.5-4B-Q4_0_ROCMFP4_STRIX_LEAN.gguf` to the `ciru-ai/ROCmFPX` llama.cpp fork (Hugging Face model-card research after GitHub code search proved auth-walled; format confirmed by cloning the fork and reading `ggml/rocmfp4/rocmfp4.{h,c}`).
- Implemented both variants: type 100 `Q4_0_ROCMFP4` (18 B, dual UE4M3 half-block scales, 4.5 bpw) and type 101 `Q4_0_ROCMFP4_FAST` (17 B, single scale, 4.25 bpw); Codebook10 `{0,±1,±2,±3,±4,±6,±8,±10}`; finite-unsigned-E4M3 scale semantics (byte 0x7F -> 0).
- Real-model validation: 393,216 weight values across three tensors in two files, both variants, bit-exact vs the independent `scripts/rocmfp4_check.py` (worst rel diff 0).
- Two implementation bugs fixed: fp32 subnormal-bit construction for the UE4M3 E==0 branch; a wrong test-vector nibble.
- Regression: AVX2 7/7, scalar 7/7.

## 2026-11-01 ZI-012 real-model MXFP4 validation

- Scanned the model library (`scripts/scan_gguf_types.py`): `qwable-v1-mxfp4_moe.gguf` carries 123 genuine MXFP4 tensors (17.5 GB). The `*ROCMFP4*` files use custom types 100/101 (handled separately above).
- `tools/mxfp4_probe.cpp` + `scripts/mxfp4_check.py`: 262,144 real-model weight values (blk.0.ffn_down_exps, blk.5.ffn_gate_exps), bit-exact, 0 NaN.
- Reference-draft note: the first Python draft used adjacent nibble interleaving; the spec (split-half, byte j -> outputs j and j+16) is what the decoder implements; corrected reference agrees exactly.
- Regression: AVX2 7/7, scalar 7/7.

## 2026-10-31 ZI-012 + ZI-016 decoder completion and security hardening

- Ported all 13 remaining GGML storage decoders (Q2_K/Q4_K/Q5_K/Q8_K, IQ1_S/IQ1_M, IQ2_XXS/IQ2_XS/IQ2_S, IQ3_XXS/IQ3_S, IQ4_XS, MXFP4) from the hardened 0.4.1-alpha tree, with grid codebooks and the corrected TQ1_0/TQ2_0 delta-at-end layout (verified against upstream ggml-common.h).
- `zen_iq_check` + `zen_iq_ref` wired into CTest: 88 checks vs the independent Python reference, worst relative diff 0 (numpy dependency removed from the generator).
- ZI-016 hardening: GGUF metadata recursion cap (64), u64-width gate, JSON depth cap (128), reader-agnostic `parse_gguf<R>` core, `inspect_bytes` in-memory surface, structured GGUF fuzzer (4096 LCG cases + 3 regressions, no crash/hang), hostile-input regression tests in the default suite.
- RED observed mid-increment (905 CHECK failures while layouts were reconciled); GREEN: scalar 7/7, AVX2 7/7.
- Evidence: `docs/evidence/ZI-012-kquant-iq-mxfp4-decoders.md`, `docs/evidence/ZI-016-gguf-json-hardening.md`.

## 2026-10-31 ZI-011 compute-pool concurrency update

- Concurrent-caller regression (four threads x four-way 1,000-item jobs, no lost/duplicated work) and worker exception propagation. Scalar and AVX2 suites passed at the time of the increment. ZI-011 remains PARTIAL (cancellation/lifecycle/shutdown, queue saturation, ThreadSanitizer, scaling).

## 2026-10-31 Milestone 1 numeric reference verification

- ZI-006: deterministic 2,048-case float32-to-binary16 differential corpus (Python `struct.pack('<e')`), exact half-bit match in both build configurations.
- ZI-013: independent ALiBi slopes (8 heads) and 128 head/distance bias comparisons.

## 2026-10-31 ZI-009 Unicode GPT-2 pre-tokenisation increment

- UTF-8 codepoint classification with a generated static Unicode L/N/whitespace table (Unicode 15.1.0); GPT-2-style pre-token spans; cross-boundary merge regression.
