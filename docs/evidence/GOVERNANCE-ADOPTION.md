# ECC governance adoption evidence

## Source review
- Retrieved public repository metadata for `affaan-m/ECC` (default branch `main`).
- Read root `AGENTS.md`, `skills/tdd-workflow/SKILL.md`, `docs/MANUAL-ADAPTATION-GUIDE.md`, and README installation/platform guidance.
- ECC root instructions call for agent-first orchestration, test-first development, security-first review, planning, specialised code review, and verification.
- ECC TDD skill requires executed RED evidence before production edits, minimal GREEN implementation, review/refactor, and explicit evidence mapping.
- ECC manual adaptation guide says select only relevant skills, preserve focused context and hook intent, and be explicit that simulated instructions are not true runtime hooks.

## Adaptation decisions
- Adopted planning, RED/GREEN, C++ review, verification, evidence ledger, one-milestone scope and explicit agent-runtime limitation.
- Did not import ECC installers, hooks, skills tree or agent definitions because they target coding-agent harnesses, while this working directory is a standalone C++ project and this session lacks an ECC native runtime.
- Applied ZenInfer-specific constraints from the project: AVX2/scalar, Windows/MSVC, no AVX-512/NEON, synthetic fixture labelling, prefill/decode separation, generation-quality separation.

## Validation
- Governance documents are plain Markdown and do not alter runtime code.
- `python3 .zeninfer-governance-check.py`: PASS (5/5 policy assertions).
- AVX2 configure/build: PASS. CTest: PASS, 8/8 tests.
- Scalar configure/build: PASS. CTest: PASS, 8/8 tests. The first combined invocation hit the command time limit during `zen_bench_parity`; the standalone rerun completed all eight tests successfully.
- CMake warned that `ZEN_ENABLE_AVX2` was unused. Therefore the AVX2/scalar labels must be interpreted from the project's actual CMake option logic/build targets, not that manually specified variable. Verify the actual configured compiler definitions before treating the builds as distinct ISA paths.
- No runtime C++ source changed as part of governance adoption.
