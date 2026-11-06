# ZenInfer build plan

## Active rule
Use `ECC_BUILD_GOVERNANCE.md` for every implementation task. `docs/REMAINING_WORK.md` is the implementation backlog and acceptance-criteria source of truth.

## Current focus
Do not start another architecture prototype. The next milestone must close an end-to-end contract, not only add metadata or a micro-kernel. Start with genuine Qwen2 parity if a suitable checkpoint can be obtained and lawfully used; otherwise close GPT-2/Hugging Face tokenizer parity using a deterministic independent corpus and reference implementation.

## Required task handoff
Every task begins with:
1. Target ID from `docs/REMAINING_WORK.md`.
2. Files/call path inspected.
3. Acceptance tests to add.
4. Expected RED failure.
5. Minimal production files expected to change.
6. Commands for AVX2, scalar, and any independent reference.
7. Explicit non-goals.

Every task ends with a milestone evidence file and an updated status in the backlog.

## Workstream split (agent-ready hand-off contracts)

The environment used for this working session exposes no independent coding-agent
runtime. These are scoped hand-off contracts, not claims that agents were spawned.
Do not run concurrent edits against shared runtime files. A real agent runner may
assign one owner per workstream and return diffs plus evidence for integration.

| Workstream | IDs | Ownership boundary | Required output |
|---|---|---|---|
| A: model adapters | ZI-001, ZI-002, ZI-003, ZI-004, ZI-010 | `src/adapters/`, architecture-specific fixture generators and parity harnesses | Genuine or explicitly synthetic model definitions, tensor maps, end-to-end runtime and reference parity per architecture |
| B: tokenizer and templates | ZI-005, ZI-009 | `src/tokenizer.cpp`, `src/chat_template.cpp`, tokenizer/template tests and reference scripts | Differential token-ID and rendered-template parity corpus; supported-feature matrix |
| C: numeric kernels | ZI-006, ZI-012, ZI-013 | `src/quant.cpp`, `src/attention.cpp`, kernel headers and numeric tests | Format-by-format decoder parity, binary16 semantics and attention reference comparisons |
| D: memory and scheduling | ZI-007, ZI-011, ZI-014, ZI-015 | `src/compute_pool.cpp`, resource governor, runtime/cache/server scheduling | Bounded residency, cancellation/lifecycle, multi-request isolation and load tests |
| E: quality and release verification | ZI-016, ZI-017, ZI-018 | CI, scripts, fuzz harnesses, benchmarks and evidence docs | Sanitizer/fuzz reports, actual Windows runner evidence and reproducible hardware baselines |

Integration rule: each workstream must submit a narrow diff, RED/GREEN commands,
full relevant test output, independent review notes and a status update. Workstream
A depends on C for quant/attention contracts; D depends on stable runtime ownership
contracts from A; E validates integrated results. The current synthetic dense
fixture addition is recorded separately in `docs/evidence/ZI-SYNTH-001-dense-micro-models.md`.

## 2026-10-31 continuation update

ZI-009 received a Unicode-aware GPT-2-style pre-tokenisation increment in the completion workspace. Static Unicode L/N/whitespace tables avoid runtime ICU/Python dependencies. Scalar and AVX2 clean builds each pass 6/6 CTest targets. This does not establish independent Hugging Face/tiktoken token-ID parity; ZI-009 remains partial and Milestone 1 remains open.
