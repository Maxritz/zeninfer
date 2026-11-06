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
    args = ap.parse_args()
    b = args.build_dir.resolve(); fixture = args.fixture.resolve()
    with tempfile.TemporaryDirectory(prefix="zeninfer-qwen2-parity-") as td:
        td = Path(td); export = td / "model.znp"; logits = td / "actual.f32"
        report = td / "parity.json"
        run(b / "zen_parity_export", fixture, export)
        # Stable IDs from the Tiny-LLM SentencePiece vocabulary, not retokenised
        # through the implementation under test.
        prompt_ids = "1,9038,2501,263,931"
        run(b / "zen_parity_probe", fixture, "--prompt-ids", prompt_ids, logits)
        run(sys.executable, args.reference_script.resolve(), export, logits,
            "--prompt-ids", prompt_ids, "--json-out", report)
        print(report.read_text())

if __name__ == "__main__": main()
