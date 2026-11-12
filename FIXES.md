# ZenInfer fix log (0.4.6-fix1)

This patch set was produced against `Maxritz/zeninfer` @ `5999cfd`
("Merge pull request #3", 0.4.6-WIP). All ten CTest checks pass in AVX2,
scalar-only and ASan/UBSan configurations after the changes.

## What was broken / wrong

1. **Independent-parity tests vanished silently.** `zen_qwen2_numpy_parity` and
   `zen_bonsai2_hybrid_parity` were only registered when NumPy happened to be
   importable at configure time. On a machine without NumPy the suite reported
   a green 7/7, so "parity verified" claims could rest on runs where the parity
   evidence was never produced. Fixed by registering a `zen_parity_prerequisites`
   guard test that fails loudly with remediation instructions when NumPy is
   missing (opt out with `-DZEN_SKIP_PARITY_GUARD=ON`), plus a configure-time
   CMake warning. CI workflows now install NumPy so the guard stays green.

2. **The governance check was dead code.** `.zeninfer-governance-check.py`
   ships at the repository root but was executed by no test and no workflow, so
   stale-status drift would never have been caught. It is now a CTest test
   (`zen_governance_check`) whenever Python 3 is available.

3. **Compiler warnings in the newest forward path.**
   Audit against revision `27e1ade`: `head_v` / `head_v_dim` do not occur in
   `src/adapters/dense_transformer_runtime.cpp`, and the file compiles cleanly
   under `-Wall -Wextra -Wpedantic` (exit 0). No `-Wunused-variable` or
   `-Wmisleading-indentation` finding reproduces; this entry is treated as
   already-resolved / stale and is not a current defect.

4. **Stale verification claims.** `docs/WORKING_COPY_0.4.6.md` reported "8/8"
   for a suite that no longer matches that count in either direction. The
   verification section now describes the real suite composition.

## Verified

- Release AVX2 build: clean; 10/10 CTest pass.
- Scalar (`-DZEN_BUILD_AVX2=OFF`): clean; 10/10 CTest pass.
- ASan/UBSan scalar: 10/10 CTest pass (incl. `zen_bonsai2_hybrid_parity`).
- `tests/check_bonsai2_big.py` (full-size ~50 MB PTQ1_0 checkpoint): 9/9
  stages pass, peak RSS bounded, logits finite and non-zero.
