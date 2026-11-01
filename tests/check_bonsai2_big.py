#!/usr/bin/env python3
"""Exercise the full-size synthetic Bonsai 2 model end to end.

`make_bonsai2_big.py` builds a ~50 MB PTQ1_0 checkpoint that is large enough to
put real pressure on the loader, the KV/conv/SSM caches and the quantised
matvec path, while staying synthetic (no upstream weights involved). This
driver generates the model, then walks it through every stage that has bitten
us at least once:

  * load + configure (dimension guards, tensor roles, Hadamard fold wiring)
  * batched prefill at several prompt lengths
  * incremental decode, which is where the GDN recurrence actually shows up
  * resource ceilings -- the 51 MB file must not fan out into a f32 copy

Any stage returning non-zero, or any non-finite / all-zero logits, fails.

Usage:
    python3 tests/check_bonsai2_big.py [--build /tmp/bA] [--target-mb 50]

The generated .gguf is written outside the repository by default and is
gitignored in any case: it is a build product, not a fixture.
"""

from __future__ import annotations

import argparse
import os
import resource
import subprocess
import sys
import tempfile
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(label, argv, expect_ok=True, logits_file=None, vocab=None):
    """Run one stage, report wall time and peak RSS, return (ok, note)."""
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    t0 = time.time()
    proc = subprocess.run(argv, capture_output=True, text=True)
    elapsed = time.time() - t0
    peak_mb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0
    combined = proc.stdout + proc.stderr

    ok = (proc.returncode == 0)
    note = ""
    if ok and logits_file:
        v = np.fromfile(logits_file, dtype="<f4")
        if v.size != vocab:
            ok, note = False, f"expected {vocab} logits, got {v.size}"
        elif not np.isfinite(v).all():
            ok, note = False, "logits contain NaN/Inf"
        elif np.count_nonzero(v) == 0:
            ok, note = False, "logits are all zero"
        else:
            note = f"min {v.min():.3f} max {v.max():.3f} std {v.std():.3f}"
    if not ok and not note:
        for line in combined.splitlines():
            if "failed" in line or "error" in line.lower():
                note = line.strip()[:100]
                break
        note = note or combined.strip().splitlines()[-1][:100] if combined.strip() else note

    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label:<34s} {elapsed:6.2f}s  rss {peak_mb:6.1f} MB  {note}")
    return ok, peak_mb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default=os.environ.get("ZEN_BUILD_DIR", "build"))
    ap.add_argument("--target-mb", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=20240601)
    ap.add_argument("--keep", action="store_true", help="keep the generated .gguf")
    a = ap.parse_args()

    probe = os.path.join(a.build, "zen_parity_probe")
    gen = os.path.join(a.build, "zen_generate_ids")
    for exe in (probe, gen):
        if not os.path.exists(exe):
            print(f"error: {exe} not found -- build first, or pass --build", file=sys.stderr)
            return 2

    tmp = tempfile.mkdtemp(prefix="zen_big_")
    model = os.path.join(tmp, "big.gguf")
    out = os.path.join(tmp, "logits.f32")

    print(f"generating a ~{a.target_mb:.0f} MB model ...")
    subprocess.run(
        [sys.executable, os.path.join(REPO, "tests", "make_bonsai2_big.py"),
         model, "--target-mb", str(a.target_mb), "--seed", str(a.seed)],
        check=True, capture_output=True,
    )
    size_mb = os.path.getsize(model) / (1024 * 1024)
    rss_budget = max(256.0, size_mb * 4.0)   # would catch an accidental f32 materialisation
    print(f"model: {model} ({size_mb:.2f} MB)\n")

    env_vocab = 8192
    prompts = [1, 2, 3, 8, 32, 64]
    results = []

    for n in prompts:
        ids = ",".join(str((i * 37 + 11) % env_vocab) for i in range(n))
        ok, rss = run(f"prefill {n:2d} token(s)", [probe, model, "--prompt-ids", ids, out],
                      logits_file=out, vocab=env_vocab)
        results.append(ok and rss <= rss_budget)

    for n, steps in ((1, 4), (3, 16), (32, 8)):
        ids = ",".join(str((i * 37 + 11) % env_vocab) for i in range(n))
        ok, rss = run(f"decode {steps:2d} from {n:2d}-token prompt",
                      [gen, model, "--prompt-ids", ids, "--max-new", str(steps)])
        results.append(ok and rss <= rss_budget)

    if not a.keep:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    passed = sum(results)
    print(f"\n{passed}/{len(results)} stages passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
