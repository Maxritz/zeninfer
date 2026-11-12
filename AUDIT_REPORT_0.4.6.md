# ZenInfer evidence-led audit — 0.4.6 working copy

**Method:** `goli-mat-de.md` (evidence-led implementation/verification audit) + `audit-matrix.md` (audit categories + evidence-coverage ledger).
**Branch:** `arena/7e2860d0-zeninfer`, single commit `27e1ade Add A2 and A5 design notes` (branched from `main` @ `27e1adefa11de0fc94e42c237162c2ee18d24ef6`). Working tree clean.
**Audited revision:** `27e1ade` (the only commit; the whole tree was added in one commit).
**Audit environment:** Linux x86-64, g++ 12.2.0, no CMake, no NumPy, no Windows/MSVC, no D3D12 or HIP hardware, no GPU.
**Conventions:** Each finding separates FACT (directly observed) from HYPOTHESIS (not reproduced). Statuses per audit-matrix: PASS / FAIL / BLOCKED / NOT RUN / UNVALIDATED. No dynamic values are invented; no check is labelled PASS unless actually executed here.

---

## 1. Scope and coverage

Inventoried: all source under `include/`, `src/`, `kernels/`, `shaders/`, `tools/`, `tests/`, `server/`, `bench/`, `models/`, `scripts/`, `docs/` (41 markdown docs + evidence/), `CMakeLists.txt`, CI workflows, governance files.

Buildability: CMake is not installed in this environment, so a full configure/build/ctest could NOT be executed. Instead every Linux-compilable translation unit was syntax-checked with `g++ -fsyntax-only -std=c++20 -I include` (AVX2 units with `-mavx2 -mfma -DZEN_HAS_AVX2_KERNELS=1`). D3D12 and HIP backends are Windows/MSVC-only in `CMakeLists.txt` and cannot be compiled or executed here. All D3D12 performance/correctness claims below are therefore **NOT RUN / UNVALIDATED in this environment** — they are reporter claims from a Windows/RX 9070 XT machine and are treated as unverified, not as findings.

Git history: one commit only. No prior commits to diff against; no "regression since last commit" analysis possible.

---

## 2. Evidence-coverage ledger

| # | Unit | Expected contract | Call / wiring path | Failure / cleanup | Verification | Status | Evidence |
|---|------|-------------------|--------------------|-------------------|--------------|--------|----------|
| L1 | `src/adapters/dense_transformer_runtime.cpp` (4933 lines) | Dense transformer forward/load/generate for Llama, Qwen2, Qwen3, BitNet, hybrids, MoE | `DenseTransformerAdapter` → `DenseTransformerRuntime`, factory in `dense_transformer_adapter.cpp` | `fail()` returns false + error string on unrecognised/unsupported architecture | Syntax-checked clean with g++ (-Wall -Wextra -Wpedantic), no errors | **PASS** | `g++ -fsyntax-only` exit 0; 3 `layer_decode_pipeline` overloads in `d3d12_backend.h` match the 11-arg runtime call |
| L2 | `src/quant.cpp` (1779 lines) | Quantised GEMV/GEMM, format registry `quant::find`, dequantizers | Called by runtime + CPU kernels; AVX2 path macro-guarded | Falls back to scalar `gemv()` | Syntax-checked clean WITH `-DZEN_HAS_AVX2_KERNELS=1`; fails only when that macro is omitted (expected — macro is a CMake define, not a source bug) | **PASS** | `g++ -fsyntax-only -DZEN_HAS_AVX2_KERNELS=1` exit 0 |
| L3 | `kernels/avx2.cpp` (1772 lines) | AVX2/FMA3 quant kernels | Linked as `zen_avx2`, defines `ZEN_HAS_AVX2_KERNELS` | Scalar fallback in `kernels/scalar.cpp` | Syntax-checked clean with `-mavx2 -mfma`; one `-Wpsabi` warning only (expected without `-mavx2` at call site) | **PASS** | `g++ -fsyntax-only -mavx2 -mfma` exit 0 |
| L4 | `tests/core_tests.cpp` (851 lines) | Architecture ID, tensor roles, quant-format executability, Q4_1/Q5_1/BF16 dequant, row bytes | CTest `zen_core_tests` | `CHECK` failure counter | NOT executed (no CMake); source reviewed — asserts qwen3/qwen3moe/bitnet non-null, bonsai null | **UNVALIDATED** | Source review only; build tooling absent |
| L5 | `tests/*.cpp` + `tests/*.py` (runtime, quant, fuzz, chat, http) | Runtime fixtures, quant differential, GGUF fuzz (4096 LCG), HTTP smoke | CTest targets | Test-specific | NOT executed (no CMake, no NumPy, no GGUF fixtures) | **NOT RUN** | `zen_parity_prerequisites` guard would FAIL without NumPy |
| L6 | `src/backends/d3d12/d3d12_backend.cpp` (9015 lines) + 28 HLSL shaders | Q8_0 GEMV/GEMM, attention, RoPE, RMSNorm, FlashAttention, DP4A, Work Graphs | Windows-only, `ZEN_BUILD_D3D12=ON`, `ZEN_HAS_D3D12=1` | CPU fallback on any failure | NOT executable here (Windows/DXC/GPU) | **BLOCKED** | CMakeLists guards; `SKIP_RETURN_CODE 77` on no device |
| L7 | `src/backends/hip/hip_backend.hip` + tools | HIP/ROCm backend | `ZEN_BUILD_HIP=ON`, `hipcc` | — | NOT executable here (Windows/hipcc) | **BLOCKED** | CMakeLists guards |
| L8 | `src/model_backend.cpp`, `src/openai_server.cpp`, `src/chat_compaction.cpp` | HTTP `/v1/completions`, `/v1/chat/completions`, SSE streaming, compaction | `zen_server`, `ModelBackend` | Throws `std::runtime_error` on failure | NOT executed | **NOT RUN** | Source review only |
| L9 | Governance: `ECC_BUILD_GOVERNANCE.md`, `BUILD_PLAN.md`, `docs/REMAINING_WORK.md`, `.zeninfer-governance-check.py` | Task discipline (RED/GREEN/Review/Verify/Record), backlog source of truth | Guard test registered as CTest `zen_governance_check` | Exits 1 on failed check | NOT executed (no Python-run CTest) | **UNVALIDATED** | Script reviewed; its self-checks would pass |

Uninspected / inaccessible: Windows/MSVC build, D3D12 runtime on RX 9070 XT, HIP/ROCm, real GGUF checkpoints, NumPy-dependent parity tests, sanitizer/ASan/UBSan/TSan runs, CI workflows (`.github/workflows/ci.yml`, `windows-msvc.yml`) — not executed.

---

## 3. Prioritised findings

Per goli-mat-de: severity and confidence are tracked separately; suspicion alone is not promoted to P0/P1.

### F-01 (P2, confidence CONFIRMED) — STATUS_REPORT.md describes defects that do not exist in current code

**FACT:** `STATUS_REPORT.md` ("ZenInfer RDNA4 D3D12 Backend - Status Report") lists under "⚠️ Runtime Adapter Issues":
- "Brace matching errors when any modifications are made"
- "`static_cast<std::uint32_t}(t)` syntax typo in original code (missing `(`)"
- "Duplicate `inv_scale` variable declarations in attention block"
- "`layer_decode_pipeline` signature mismatch (10 args vs expected)"

**Observation:** `g++ -fsyntax-only -std=c++20 -Wall -Wextra -Wpedantic` on `src/adapters/dense_transformer_runtime.cpp` exits 0 with no errors. `static_cast<std::uint32_t}(t)` does not occur anywhere (grep: no matches). `inv_scale` is declared in separate functions/scopes (lines 1897, 3082, 3148, 4283), never duplicated in one block. `layer_decode_pipeline` has 3 overloads in `include/zen/d3d12_backend.h`; the runtime's 11-arg call matches overload 3 exactly. All D3D12 calls are `#if defined(ZEN_HAS_D3D12)`-guarded.

**Conclusion:** These four reported defects are stale — they do not reproduce against revision `27e1ade`. `STATUS_REPORT.md` is inaccurate about the current code and misleads anyone triaging the runtime.

### F-02 (P2, confidence CONFIRMED) — FIXES.md references fixes not present in current code

**FACT:** `FIXES.md` ("ZenInfer fix log (0.4.6-fix1)") claims item 3: "`src/adapters/dense_transformer_runtime.cpp` (`forward_q35_batch`) carried `-Wunused-variable` (`head_v`, `head_v_dim`) and `-Wmisleading-indentation`".

**Observation:** `head_v` / `head_v_dim` do not occur anywhere in the runtime (grep: no matches). The runtime compiles warning-clean under the flags used by CMake (`-Wall -Wextra -Wpedantic`).

**Conclusion:** Either already fixed before this revision or never present. `FIXES.md` should not list them as current defects.

### F-03 (P2, confidence CONFIRMED) — REFACTOR_VERIFICATION.md dispatch claim contradicts current factory + tests

**FACT:** `docs/REFACTOR_VERIFICATION.md` states: "Qwen3, MoE, BitNet and Bonsai remain unsupported and are rejected by architecture dispatch."

**Observation:** `src/adapters/dense_transformer_adapter.cpp` `create_model_runtime()` returns an executable `DenseTransformerAdapter` for `qwen3`, `qwen3moe`, `bitnet` (and many others). `tests/core_tests.cpp` asserts `create_model_runtime(...qwen3...) != nullptr`, `...qwen3moe... != nullptr`, `...bitnet... != nullptr`, and `...bonsai... == nullptr`. So qwen3/qwen3moe/bitnet ARE dispatched to an executable adapter; only bonsai is correctly rejected.

**Conclusion:** `REFACTOR_VERIFICATION.md` describes a pre-adapter-library state and is stale relative to revision `27e1ade`. Note this does NOT mean Qwen3/MoE/BitNet are *verified correct* — only that the factory dispatches them. Real-model parity remains open (see REMAINING_WORK.md ZI-001/ZI-002/ZI-003).

### F-04 (P3, confidence CONFIRMED) — Dead/disabled code: `hybrid_stmt` hardcoded to `false`

**FACT:** `src/adapters/dense_transformer_runtime.cpp:733`:
```cpp
const bool hybrid_stmt=(architecture_name_=="nemotron_h_moe"&&false);
```
and line 735 `if(hybrid_stmt)return fail(...)` is therefore unreachable. `hybrid_stmt` is read nowhere else.

**Hypothesis:** Either (a) a deliberately disabled Nemotron-H MoE hybrid path awaiting `mamba2`/`shortconv` kernels, or (b) leftover dead code. `mamba2_forward`, `nemotron_moe_forward`, `gemma_moe_forward` all have real (non-stub) bodies in the runtime, so MoE/SSM machinery exists.

**Conclusion:** Low-risk cleanup candidate: either remove `hybrid_stmt` and its unreachable `if`, or replace `&&false` with a real feature gate. Not a defect, but harms readability.

### F-05 (P2, confidence UNVALIDATED) — Factory over-recognition of architectures

**FACT:** `create_model_runtime()` returns a `DenseTransformerAdapter` for a long list including `gpt-oss`, `flint`, `k2-horizon`, `deepseek2`, `gemma4`, `starcoder2`, `spark2_5`, `qwen3vl`, `mellum`, `muse-glimmer`, `phi3`, `llama`+experts, etc. `DenseTransformerRuntime::load()` then re-checks and may fail with "Recognised architecture ... has no executable runtime adapter yet".

**Hypothesis:** Some of these architectures are routed to the adapter but rejected at load(); the two-layer gate is redundant and could return a confusing "executable adapter created, then load fails" instead of a clean factory rejection.

**Conclusion:** Verify each routed architecture against `DenseTransformerRuntime::load()` acceptance; align factory and load() so the factory fails closed cleanly. Needs real GGUF fixtures — NOT RUN here.

### F-06 (P1, confidence BLOCKED) — All D3D12 performance & correctness claims unverifiable in this environment

**FACT:** `FINAL_STATUS.md`, `STATUS_REPORT.md`, `IMPLEMENTATION_STATUS.md`, `docs/d3d12_full_backend_todo.md`, `docs/DX12_OPTIMIZATION_*.md`, `.opencode/sherlock-analysis.md`, `BENCHMARKS.md` report D3D12 results (e.g. decode 117–146 tok/s, prefill 744–8000 tok/s targets, TDR fixes, RoPE llama.cpp NORM fix, flash attention, parity ≤1.9e-07), all measured on an RX 9070 XT / Windows machine (paths like `H:\LLAMA-bins\llama.cpp`).

**Conclusion:** These claims are reporter evidence from a Windows/GPU environment that is not present here. They are **UNVALIDATED**, not confirmed and not refuted. No D3D12 binary can be built or run on this Linux box. Mark all D3D12 status as BLOCKED pending a Windows runner with an RX 9070 XT (or compatible D3D12.1 device) + DXC + Agility SDK.

### F-07 (P1, confidence BLOCKED) — Windows/MSVC and real-GGUF parity unverifiable

**FACT:** `REMAINING_WORK.md` ZI-017 (Windows/MSVC — local execution unavailable, user-side validation required), ZI-001/ZI-003/ZI-010 (real Qwen3/BitNet/Qwen2 GGUF parity), ZI-004 (real Bonsai-2 checkpoint), ZI-009 (GPT-2/HF tokenizer parity), ZI-011 (TSan/shutdown/saturation) all require hardware/tooling/fixtures not present.

**Conclusion:** NOT RUN / BLOCKED in this environment. Do not treat the extensive written claims as verified.

---

## 4. Documentation accuracy assessment

| Document | Verdict against revision `27e1ade` | Action |
|----------|-----------------------------------|--------|
| `README.md` | General overview; no falsifiable structural claims found in spot check | Keep |
| `STATUS_REPORT.md` | INACCURATE: 4 runtime defects claimed do not exist (F-01) | Rectify |
| `FIXES.md` | INACCURATE: `head_v`/`head_v_dim`/`-Wmisleading-indentation` not present (F-02) | Rectify |
| `IMPLEMENTATION_STATUS.md` | D3D12-only claims; UNVALIDATED here | Annotate as reporter-claimed |
| `FINAL_STATUS.md` | D3D12/performance claims; UNVALIDATED here | Annotate as reporter-claimed |
| `REFACTOR_VERIFICATION.md` | INACCURATE dispatch claim (F-03) | Rectify |
| `REMAINING_WORK.md` | Consistent with code state (PARTIAL/DONE per architecture); the source-of-truth backlog | Keep as backlog |
| `WORKING_COPY_0.4.6.md` | Describes synthetic Qwen2 fixture; consistent | Keep |
| `BUILD_STATUS.md` | Points to 0.4.5-alpha verification; consistent | Keep |
| `BENCHMARKS.md`, `DX12_*.md`, `FINAL_STATUS.md`, `sherlock-analysis.md` | All GPU/Windows-measured; UNVALIDATED here | Annotate |
| `ECC_BUILD_GOVERNANCE.md`, `BUILD_PLAN.md`, `.zeninfer-governance-check.py` | Present and self-consistent; guard test registered | Keep |

**Documentation status we are on:** The project's source-of-truth backlog is `docs/REMAINING_WORK.md` (per `BUILD_PLAN.md`). Governance is `ECC_BUILD_GOVERNANCE.md`. Multiple status documents (`STATUS_REPORT.md`, `FIXES.md`, `REFACTOR_VERIFICATION.md`) drift from revision `27e1ade` and overstate defects that have already been resolved — they need rectification before they can be trusted for release decisions.

---

## 5. Completeness status (honest, revision `27e1ade`)

| Area | Complete? | Basis |
|------|-----------|-------|
| CPU build (AVX2 + scalar), Linux/GCC | Source compiles cleanly (syntax-checked); full build NOT executed (no CMake) | UNVALIDATED for full build |
| Quant format decoders (Q2_K…Q8_K, IQ*, MXFP4, ROCmFP4, Q4_1, Q5_1, BF16) | Declared implemented; `zen_iq_check` 88 checks vs Python reference, worst rel diff 0 — NOT run here | UNVALIDATED |
| Qwen3 dense / BitNet / Qwen2.5-Coder real-checkpoint parity | Claimed in REMAINING_WORK + FINAL_STATUS; requires real GGUF + NumPy | BLOCKED here |
| Qwen3 MoE / Bonsai production adapters | Prototype / synthetic only | OPEN (ZI-002, ZI-004) |
| D3D12 GPU backend | Implemented (shaders + C++), Windows-only | BLOCKED here |
| HIP/ROCm backend | Implemented, Windows-only | BLOCKED here |
| HTTP OpenAI-compatible server | Implemented; smoke tests not run here | NOT RUN |
| Jinja templates / grammar / JSON Schema | OPEN (ZI-005, ZI-008) | OPEN |
| App-managed NVMe residency | OPEN (ZI-007) | OPEN |
| Sanitizer / fuzz / fault injection | PARTIAL (fuzz harness present, coverage-guided fuzzing open — ZI-016) | PARTIAL |
| Windows/MSVC | CI workflow present; not executed | BLOCKED (ZI-017) |

**Bottom line:** The CPU source tree is internally consistent and compiles; the extensive real-model parity and performance claims are reporter-evidence from a Windows/GPU environment and cannot be confirmed or denied here. Nothing is "production-ready" on the strength of this audit alone — every D3D12 and real-GGUF claim remains UNVALIDATED until run on the target hardware.

---

## 6. Proposed tasks (require acknowledgement before any issue/task record is created)

Per the goli-mat-de Acknowledgement gate, no issues/tickets/task records are created below. On acknowledgement, I will implement in this order:

| ID | Task | Dependencies | Effort | Acceptance criteria |
|----|------|--------------|--------|---------------------|
| T-001 | Rectify `STATUS_REPORT.md` — remove the 4 non-existent runtime defects; mark D3D12 claims as reporter-claimed/UNVALIDATED on Linux | — | Small | Document matches `g++` syntax result; no false defect claims |
| T-002 | Rectify `FIXES.md` — drop `head_v`/`head_v_dim`/misleading-indentation claims | — | Small | No references to symbols absent from code |
| T-003 | Rectify `REFACTOR_VERIFICATION.md` — correct the dispatch claim; note qwen3/qwen3moe/bitnet are dispatched, bonsai rejected | — | Small | Consistent with `dense_transformer_adapter.cpp` + `core_tests.cpp` |
| T-004 | Remove/clarify dead `hybrid_stmt = (...)&&false` + unreachable `if` in runtime adapter | — | Small | `hybrid_stmt` gone; no behavior change (unreachable) |
| T-005 | Add a `docs/AUDIT_REPORT_0.4.6.md`-style coverage note and a per-doc "verified on this revision" annotation | T-001..T-003 | Medium | Every status doc carries an accuracy verdict |
| T-006 | Attempt full Linux CMake build + ctest if CMake becomes available; record exact commands/exit codes | CMake install | Medium | Green AVX2 + scalar ctest, or honest NOT RUN |
| T-007 | Align `create_model_runtime()` factory with `DenseTransformerRuntime::load()` architecture acceptance (F-05) | Real fixtures | Medium | Factory fails closed cleanly; no redundant two-layer gate |
| T-008 | Windows/MSVC + real-GGUF + D3D12 validation on target hardware | Windows runner, GPU, NumPy, GGUF fixtures | Large | Actual runner logs; not a workflow file |

**Items I can do without further tooling:** T-001, T-002, T-003, T-004, T-005 (doc/code rectification on this Linux box).
**Items requiring acknowledgement + external tooling:** T-006, T-007, T-008.

Please acknowledge which tasks you authorise, and whether you want any GitHub issues created for them (the goli-mat-de gate requires explicit approval of that destination).

---

## 7. Residual risks

- No git history beyond one commit: cannot establish what changed or when; cannot diff-based bisect.
- All performance numbers (484 tok/s hardware ceiling, 76% measured, 8.75→6.82 ms/token, etc.) are single-machine reporter claims; unverified here.
- `zen_parity_prerequisites` will FAIL ctest without NumPy (intentional per FIXES.md) — a "green" ctest on this box would require `-DZEN_SKIP_PARITY_GUARD=ON`, which suppresses evidence.
- One commit means the working tree IS the baseline; there is no separate "main" state to compare the branch against.

End of audit.
