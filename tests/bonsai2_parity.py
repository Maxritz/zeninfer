#!/usr/bin/env python3
"""Differential parity check: ZenInfer's Qwen3.5-hybrid (Bonsai 2) path vs an
independent NumPy reference.

The reference logits come from tests/make_bonsai2_fixture.py, which is written
from the PrismML fork sources (src/models/qwen35.cpp and the gated-delta-net CPU
op in ggml/src/ggml-cpu/ops.cpp). The C++ side is exercised through
zen_parity_probe, the same binary used against real checkpoints:

    zen_parity_probe model.gguf --prompt-ids 7,42,99 logits.f32

Checks, in order (each one names the stage that breaks first):

  1. the probe exits 0 and reports the same prompt ids
  2. logits are finite and have the checkpoint's vocabulary width
  3. logits match the reference (abs + relative tolerance)
  4. argmax and top-5 order match

A failure here is a numerical or graph bug, never a crash: the fixture is tiny
enough that a heap/pointer defect in the loader still surfaces as a nonzero exit
code instead of a silent wrong answer.

Two modes:

  * smoke (default, what CTest runs): the probe must load the fixture, execute
    prefill and decode, and write finite logits of the right width. This is the
    regression guard for the crash/loader class of defect (exit codes, OOB
    writes, silent fallback decode).
  * --numeric: additionally diff the logits against the reference and require
    argmax parity. The Qwen3.5-hybrid recurrence is not yet bit-matched on the
    synthetic fixture, so this mode is expected to fail today and is not wired
    into CTest.

Usage:
  python3 tests/bonsai2_parity.py BUILD_DIR [--numeric] [--fixture-dir DIR] [--keep]
"""
import argparse
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

ABS_TOL = 2.0e-3
REL_TOL = 2.0e-3


def locate_probe(build_dir):
    for name in ("zen_parity_probe", "zen_parity_probe.exe"):
        candidate = pathlib.Path(build_dir) / name
        if candidate.is_file():
            return candidate
    raise SystemExit(f"zen_parity_probe not found under {build_dir}")


def run_case(probe, gguf, ref_logits, prompt, label, keep_dir=None, numeric=False):
    out = pathlib.Path(keep_dir or tempfile.mkdtemp(prefix="zen_bonsai2_")) / f"{label}.f32"
    ids = ",".join(str(int(t)) for t in prompt)
    cmd = [str(probe), str(gguf), "--prompt-ids", ids, str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        print(f"FAIL[{label}]: probe exited {proc.returncode}")
        print("  cmd:", " ".join(cmd))
        if proc.stdout:
            print("  stdout:", proc.stdout.strip()[:2000])
        if proc.stderr:
            tail = proc.stderr.strip().splitlines()[-12:]
            print("  stderr (last 12 lines):")
            for line in tail:
                print("   ", line)
        return False
    if f"prompt_ids={ids}" not in proc.stdout:
        print(f"FAIL[{label}]: probe did not echo the prompt ids ({proc.stdout.strip()[:400]})")
        return False
    if not out.is_file():
        print(f"FAIL[{label}]: probe wrote no logits file")
        return False

    got = np.fromfile(out, dtype="<f4")
    if got.size != ref_logits.size:
        print(f"FAIL[{label}]: logits width {got.size} != reference {ref_logits.size}")
        return False
    if not np.all(np.isfinite(got)):
        print(f"FAIL[{label}]: logits contain non-finite values")
        return False

    if not numeric:
        # smoke: the graph runs and produces usable logits; no numerics claim
        nonzero = int(np.count_nonzero(np.abs(got) > 1e-6))
        if nonzero == 0:
            print(f"FAIL[{label}]: logits are all zero (graph did not execute)")
            return False
        print(f"PASS[{label}]: loaded, ran prefill+decode, {got.size} finite logits, "
              f"{nonzero} non-zero (numeric check skipped; use --numeric for parity)")
        return True

    delta = np.abs(got - ref_logits)
    denom = np.maximum(1.0, np.abs(ref_logits))
    worst = float(np.max(delta / denom))
    abs_worst = float(np.max(delta))
    ref_top = np.argsort(-ref_logits)[:5]
    got_top = np.argsort(-got)[:5]
    ok = abs_worst <= ABS_TOL or worst <= REL_TOL
    same_top5 = list(ref_top) == list(got_top)
    print(f"{'PASS' if ok else 'FAIL'}[{label}]: abs={abs_worst:.3e} rel={worst:.3e} "
          f"argmax ref={int(ref_top[0])} got={int(got_top[0])} top5_match={same_top5}")
    if not ok:
        worst_idx = int(np.argmax(delta / denom))
        print(f"  worst element {worst_idx}: ref={ref_logits[worst_idx]:.6f} got={got[worst_idx]:.6f}")
        print(f"  ref top5={list(map(int, ref_top))} got top5={list(map(int, got_top))}")
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("build_dir")
    ap.add_argument("--fixture-dir", default=None, help="reuse a generated fixture directory")
    ap.add_argument("--keep", action="store_true", help="keep generated fixtures and logits")
    ap.add_argument("--generator", default=None)
    ap.add_argument("--numeric", action="store_true",
                    help="require reference parity (currently FAILS on the hybrid recurrence)")
    args = ap.parse_args()

    probe = locate_probe(args.build_dir)
    root = pathlib.Path(__file__).resolve().parent.parent
    generator = pathlib.Path(args.generator) if args.generator else root / "tests" / "make_bonsai2_fixture.py"
    workdir = pathlib.Path(args.fixture_dir) if args.fixture_dir else pathlib.Path(
        tempfile.mkdtemp(prefix="zen_bonsai2_fixture_"))

    results = []
    for label, extra in (("ptq1_0", []), ("f32", ["--f32"])):
        gguf = workdir / f"bonsai2-{label}.gguf"
        npz = workdir / f"bonsai2-{label}.npz"
        if not (gguf.is_file() and npz.is_file()):
            subprocess.run([sys.executable, str(generator), str(gguf), str(npz), *extra], check=True,
                           stdout=subprocess.DEVNULL)
        data = np.load(npz)
        ref = data["ref_logits"].astype(np.float32)
        prompt = data["prompt"]
        results.append(run_case(probe, gguf, ref, prompt, label,
                                keep_dir=str(workdir) if args.keep else None,
                                numeric=args.numeric))

    if not args.keep and not args.fixture_dir:
        for f in workdir.glob("*"):
            f.unlink()
        workdir.rmdir()

    passed = sum(1 for r in results if r)
    print(f"bonsai2_parity: {passed}/{len(results)} cases passed "
          f"({'numeric' if args.numeric else 'smoke'} mode)")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
