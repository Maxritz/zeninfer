#!/usr/bin/env python3
"""Malformed config should fail cleanly rather than divide by zero at load."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path
with tempfile.TemporaryDirectory(prefix='zen-invalid-gguf-') as tmp:
    path=Path(tmp)/'bad.gguf'
    subprocess.run([sys.executable,str(Path(__file__).with_name('make_qwen2_synthetic.py')),str(path),'--invalid-heads'],check=True)
    p=subprocess.run([sys.argv[1],str(path),'--prompt-ids','1',str(Path(tmp)/'out')],
                     env={**os.environ,'ZEN_MAP_TRACE':'0'},capture_output=True,text=True)
    if p.returncode!=1 or 'Incomplete or inconsistent dense transformer dimensions' not in p.stderr:
        raise SystemExit(f'invalid model did not fail cleanly: returncode={p.returncode} stderr={p.stderr}')
print('zero-head GGUF rejected cleanly')
