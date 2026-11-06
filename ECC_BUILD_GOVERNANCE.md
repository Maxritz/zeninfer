# ZenInfer Build Governance (ECC-adapted)

**Status:** Active project workflow for the 0.4.6 working copy.  
**Source:** Adapted from [affaan-m/ECC](https://github.com/affaan-m/ECC), principally its root `AGENTS.md`, `skills/tdd-workflow/SKILL.md`, and `docs/MANUAL-ADAPTATION-GUIDE.md`.  
**Boundary:** This is a manual adaptation. It does not install or execute ECC hooks, skills, agents, or automatic routing in this environment.

## 1. Purpose and operating rule

Keep ZenInfer focused on closing measurable runtime capability gaps, not accumulating design documents, metadata labels, placeholder adapters, or another alpha version with the same gaps.

A task is complete only when its production path is implemented, a relevant test demonstrates the behavior, the test is run, independent review is recorded, and all known limitations are stated. A structure, type, parser stub, synthetic micro-kernel, README entry, or CI workflow alone is not completion evidence.

## 2. ECC workflow adapted for ZenInfer

Every non-trivial change follows this sequence:

1. **Plan:** identify one milestone, its caller-to-runtime execution path, dependencies, risks and explicit non-goals. Do not plan multiple unrelated features in one implementation batch.
2. **Specify:** translate each acceptance criterion into executable tests. Use existing user journeys and fixtures where available.
3. **RED:** add tests first, build the relevant test target and run it. Record the exact failure and prove it fails for the missing behaviour, not a broken harness. Do not edit production code before this gate.
4. **GREEN:** implement the smallest complete production change. Re-run the focused test, then the full applicable suite.
5. **Review:** inspect the diff for correctness, API contracts, numerical stability, concurrency, memory ownership, portability and silent failure. For C/C++, use the ECC C++ reviewer checklist below.
6. **Verify:** run clean AVX2 and scalar builds/tests where applicable, sanitizers for memory-sensitive paths, and real GGUF tests when the fixture is relevant. Report exact commands and outcomes.
7. **Record:** update `docs/REMAINING_WORK.md` and this task's evidence record. Mark partial only when the acceptance criteria are only partially satisfied.

Git checkpoints are recommended for a real repository: a test/RED checkpoint, implementation/GREEN checkpoint, and optional refactor checkpoint. This workspace is not currently a Git checkout, so do not fabricate commit evidence. Preserve equivalent command logs and diffs in `docs/evidence/` until version control is initialised.

## 3. Scope control

- One active P0/P1 implementation milestone at a time.
- No version bump, release notes or ZIP packaging until a meaningful acceptance gate is closed.
- No opportunistic refactors unrelated to the selected milestone.
- No generated fixtures that masquerade as production model checkpoints.
- Synthetic fixtures are allowed for deterministic unit/integration tests, but must be labelled synthetic and cannot establish checkpoint compatibility.
- No dependency added without checking licence, maintenance, portability, binary size and Windows/MSVC viability.
- AVX-512 and ARM/NEON remain excluded. AVX2 and scalar paths remain mandatory where applicable. Windows x64/MSVC remains in scope.
- Prefill throughput, decode tok/s, numerical parity and generated-text quality are separate measurements. Never use one as a proxy for another.

## 4. Current milestone queue

The source-of-truth queue remains `docs/REMAINING_WORK.md`. The first focused sequence is:

| Order | Work | Gate to close |
|---:|---|---|
| 1 | Genuine Qwen2 parity | Test at least one genuine Qwen2 GGUF checkpoint against an independent reference for prompt logits and generated token IDs. Existing relabelled-weight fixture is layout-only. |
| 2 | GPT-2/Hugging Face tokenizer parity | Exact token ID parity against a trusted independent tokenizer on ASCII, Unicode, whitespace, punctuation and special-token cases. |
| 3 | Grammar/JSON Schema | Compile supported schema constructs into incremental constraints; prove invalid prefixes are masked and generated JSON validates. Publish supported/unsupported schema dialect/features. |
| 4 | Managed NVMe residency | App-owned bounded residency map, pin/unpin, eviction, prefetch, back-pressure and cold/warm tests proving resident-byte bound. |
| 5 | General Jinja templates | Define compatibility target, implement interpreter or selected maintained C++ implementation, and differential-test representative HF templates. Do not call a subset full Jinja. |
| 6 | Qwen3 MoE / BitNet / Bonsai production adapters | Separate milestones per architecture. Each must cover GGUF parsing, tensor map, complete forward path, tests and independent reference. |
| 7 | FP16 semantics | Software IEEE binary16 conversion/rounding correctness for AVX2/scalar. Do not claim native half arithmetic on Zen 3/Meteor Lake absent supported hardware ISA. |

Reorder only when a concrete blocker or user priority justifies it. Record the reason in the work log.

## 5. Definition of Done

A feature can be marked **DONE** only if all applicable checks pass:

- [ ] Production call path is traced from public API/adapter dispatch to the actual compute operation.
- [ ] Acceptance tests were written before implementation and a meaningful RED result was observed.
- [ ] Unit and integration tests cover normal, boundary, malformed-input and failure cases.
- [ ] Independent reference/parity evidence exists for numerical/tokenization/template claims.
- [ ] Full applicable CTest suite passes on AVX2 and scalar builds.
- [ ] Sanitizer/fuzz/concurrency checks run when the changed code handles untrusted files, parsing, memory or concurrency.
- [ ] Benchmark reports prefill and decode separately; output quality is separately inspected for generation changes.
- [ ] Windows/MSVC CI is green before claiming cross-platform support; otherwise state unverified.
- [ ] C++ review completed and critical/high findings fixed or explicitly accepted.
- [ ] Documentation and `REMAINING_WORK.md` reflect exactly what was tested and what remains open.

If any item is not applicable, state why. If a gate was not run, state **NOT RUN**, never PASS.

## 6. C++ reviewer checklist (ECC C++ reviewer adaptation)

- Ownership and lifetime: no dangling spans/views, invalidated mappings, use-after-free or implicit shared mutable state.
- Bounds and arithmetic: validate GGUF offsets, tensor dimensions, products, byte counts, alignment and allocation overflow.
- Concurrency: prove worker lifetime, cancellation, exception propagation, deterministic partitioning and no data races.
- Numerics: document accumulation type, rounding, quant block layout, NaN/Inf handling and tolerance.
- SIMD: runtime dispatch is safe; scalar fallback matches semantics; no AVX-512/NEON leakage.
- Portability: Windows file APIs, path handling, WinSock, compiler flags and MSVC C++ standard/library support.
- Errors: propagate failures with context; do not silently substitute zeros, fallback formats or pretend success.
- API: preserve ownership contracts, cancellation, deterministic output and backwards compatibility where promised.
- Tests: ensure test actually reaches changed production path; avoid testing duplicated test-only implementations.

## 7. Evidence format

For each milestone create `docs/evidence/ZI-NNN-<slug>.md` containing:

- `FACT`: directly observed source/runtime facts.
- `HYPOTHESIS`: expected behaviour not yet measured.
- `RESULT`: exact command, environment, exit code, test count and relevant output.
- `CONCLUSION`: what the result proves and what it does not prove.
- RED evidence, GREEN evidence, independent reference and review findings.
- Open defects and next smallest step.

Use precise labels: `NOT STARTED`, `IN PROGRESS`, `PARTIAL`, `BLOCKED`, `DONE`. Do not let documentation status override code/test evidence.

## 8. Tool and agent discipline

ECC recommends specialised agents and parallel execution. This environment does not expose ECC's native agent runtime, so do not claim that agents were spawned or that hooks enforced this workflow. Simulate the roles explicitly in the task record:

- Planner: narrow scope and dependency map.
- TDD guide: derive tests and require RED/GREEN evidence.
- C++ reviewer: inspect diff and execution path.
- Performance reviewer: benchmark hot path and memory behaviour.
- Verification reviewer: independently rerun tests and challenge claims.

Parallelise only independent read-only audits or test suites. Do not parallelise edits to the same files or dependent runtime changes.

## 9. Current governance baseline

- Working tree: `ZenInfer-work-0.4.6`.
- Existing 0.4.6 evidence: synthetic Qwen2-layout fixture and NumPy parity; it is not genuine Qwen2 checkpoint parity.
- Existing AVX2/scalar CTest baseline: previously reported 8/8 each; rerun after governance changes before relying on it.
- No native ECC install, hooks, automatic agent dispatch, or Git checkpoint commits are active in this workspace.
