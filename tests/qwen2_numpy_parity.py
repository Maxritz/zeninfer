#!/usr/bin/env python3
"""Independent NumPy parity for the generated Qwen2-layout GGUF fixture."""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def run(*args):
    subprocess.run([str(x) for x in args], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("build_dir", type=Path)
    ap.add_argument("fixture", type=Path)
    ap.add_argument("reference_script", type=Path)
    ap.add_argument("--prompt-ids", default="1,9038,2501,263,931",
                    help="token IDs valid for the selected fixture")
    ap.add_argument("--length-ladder", action="store_true", help="check every prefix of the supplied IDs")
    args = ap.parse_args()
    b = args.build_dir.resolve(); fixture = args.fixture.resolve()
    def executable(name):
        candidate = b / name
        return candidate if candidate.is_file() else b / (name + ".exe")
    with tempfile.TemporaryDirectory(prefix="zeninfer-qwen2-parity-") as td:
        td = Path(td); export = td / "model.znp"; logits = td / "actual.f32"
        report = td / "parity.json"
        run(executable("zen_parity_export"), fixture, export)
        # IDs are specified explicitly, not retokenised through the runtime.
        ids = args.prompt_ids.split(',')
        lengths = range(1, len(ids)+1) if args.length_ladder else [len(ids)]
        for length in lengths:
            prompt_ids = ','.join(ids[:length])
            run(executable("zen_parity_probe"), fixture, "--prompt-ids", prompt_ids, logits)
            run(sys.executable, args.reference_script.resolve(), export, logits,
                "--prompt-ids", prompt_ids, "--json-out", report)
        print(f"Qwen2 independent NumPy parity: {len(list(lengths))} prefix(es) passed")

if __name__ == "__main__": main()
