#!/usr/bin/env python3
"""Generate synthetic qwen35 fixtures and verify split-prefill/decode parity."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import make_bonsai2_fixture as gen
import numpy as np

def main():
    with tempfile.TemporaryDirectory(prefix='zen-q35-incremental-') as tmp:
        for label, quantised in [('f32',False), ('ptq1_0',True)]:
            model=Path(tmp)/(label+'.gguf')
            gen.make(model, Path(tmp)/(label+'.npz'), quantised=quantised, layers=4, prompt=[11,51,30])
            result=subprocess.run([sys.argv[1],str(model)], env={**os.environ,'ZEN_MAP_TRACE':'0'}, text=True)
            if result.returncode: raise SystemExit(f'{label}: incremental validation failed')
            if not quantised:
                # The batch/chunk check alone shares runtime code. Independently
                # compare each full-prefix row to the NumPy recurrence oracle.
                with np.load(Path(tmp)/(label+'.npz')) as data:
                    params={k[3:]: (data[k].item() if data[k].ndim==0 else data[k])
                            for k in data.files if k.startswith('p__')}
                    weights={k[3:]:data[k] for k in data.files if k.startswith('w__')}
                tokens=[11,51,30,25,17,42,99,7]
                worst=0.0
                for end in range(1,len(tokens)+1):
                    ref=gen.reference_logits(params,weights,tokens[:end])
                    out=Path(tmp)/'logits.f32'
                    run=subprocess.run([sys.argv[2],str(model),'--prompt-ids',
                        ','.join(str(t) for t in tokens[:end]),str(out)],
                        env={**os.environ,'ZEN_MAP_TRACE':'0'},capture_output=True,text=True)
                    if run.returncode: raise RuntimeError(run.stderr)
                    got=np.fromfile(out,dtype='<f4')
                    worst=max(worst,float(np.max(np.abs(got-ref))))
                    if not np.isfinite(got).all() or not np.allclose(got,ref,rtol=2e-3,atol=2e-3):
                        raise AssertionError(f'NumPy prefix {end} mismatch: max_abs={np.max(np.abs(got-ref))}')
                print(f'Qwen3.5 F32 eight prefixes: independent NumPy logits PASS max_abs={worst:.6g}')
    print('Qwen3.5 F32 and PTQ1_0 split-prefill/decode: PASS')
if __name__=='__main__': main()
