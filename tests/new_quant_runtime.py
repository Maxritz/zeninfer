#!/usr/bin/env python3
"""Compare mapped GGUF projection weights with upstream-decoded F32 twins.

The synthetic Qwen3.5 model is a layout fixture, not a trained checkpoint or
architecture-wide parity test. Golden blocks are from new_quant_ggml.bin.
"""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import numpy as np
import make_bonsai2_fixture as generator


def records(path):
    data = path.read_bytes(); pos = 4
    def u32():
        nonlocal pos
        v = struct.unpack_from('<I', data, pos)[0]; pos += 4; return v
    count = u32(); result = {}
    for _ in range(count):
        length = u32(); name = data[pos:pos+length].decode(); pos += length
        typ, size, n = u32(), u32(), u32()
        raw = data[pos:pos+size]; pos += size
        ref = np.frombuffer(data, '<f4', n, pos).copy(); pos += n*4
        result.setdefault(name, (typ, raw, ref))
    return result


def main():
    probe = Path(sys.argv[1]); golden = records(Path(__file__).parent / 'data/new_quant_ggml.bin')
    captured = []
    orig_write = generator.GgufWriter.write
    def capture(writer, path):
        captured.append(writer)
        return orig_write(writer, path)
    generator.GgufWriter.write = capture
    with tempfile.TemporaryDirectory(prefix='zen-new-quant-runtime-') as tmp:
        root = Path(tmp)
        generator.make(root/'base.gguf', root/'base.npz', quantised=False, hadamard=False, layers=1, prompt=[7,42,99])
        writer = captured[0]
        ix = next(i for i, t in enumerate(writer.tensors) if t[0] == 'blk.0.ffn_up.weight')
        name, dims, _, _ = writer.tensors[ix]
        cols, rows = dims
        assert cols == 128 and rows == 256
        env = dict(os.environ, ZEN_MAP_TRACE='1', ZEN_GGUF_Q8_1_LAYOUT='fp16')
        for fmt, block, blockbytes, index in [('Q1_0',128,18,6),('Q2_0',64,18,6),
                                               ('Q8_1',32,36,6),('Q8_1_F32',32,36,6),('NVFP4',64,36,32),
                                               ('F8_E4M3',1,1,0x20), ('I8',1,1,1), ('I16',1,2,1),
                                               ('I32',1,4,1), ('I64',1,8,1), ('F64',1,8,0)]:
            typ, raw, ref = golden['Q8_1' if fmt == 'Q8_1_F32' else fmt]
            chunk = bytearray(raw[index*blockbytes:(index+1)*blockbytes])
            decoded = ref[index*block:(index+1)*block].copy()
            if fmt in ('Q1_0','Q2_0','Q8_1'):
                # Reduce half scale by 2^8 to keep the fixture well-conditioned.
                d = struct.unpack_from('<H', chunk)[0]; struct.pack_into('<H', chunk, 0, d-0x2000)
                decoded /= 256
            if fmt == 'Q8_1_F32':
                # Legacy layout: independent fp32 scale/sum plus signed bytes.
                d = 1.0 / 1024.0
                decoded = np.frombuffer(chunk[4:], dtype=np.int8).astype(np.float32) * d
                chunk = bytearray(struct.pack('<ff', d, 0.0) + chunk[4:])
            row_payload = chunk * (cols//block)
            row_float = np.tile(decoded, cols//block).astype('<f4').tobytes()
            qfile, ffile = root/(fmt+'.gguf'), root/(fmt+'-f32.gguf')
            writer.tensors[ix] = (name, dims, typ, row_payload * rows); orig_write(writer, qfile)
            writer.tensors[ix] = (name, dims, 0, row_float * rows); orig_write(writer, ffile)
            case_env = dict(env, ZEN_GGUF_Q8_1_LAYOUT='fp32' if fmt == 'Q8_1_F32' else 'fp16')
            if fmt in ('Q8_1', 'Q8_1_F32'):
                unspecified_env = dict(env); unspecified_env.pop('ZEN_GGUF_Q8_1_LAYOUT')
                failed = subprocess.run([str(probe), str(qfile), '--prompt-ids', '7', str(root/'logits.bin')],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=unspecified_env, text=True)
                if failed.returncode == 0 or 'ambiguous' not in failed.stderr:
                    raise AssertionError(f'{fmt}: unspecified Q8_1 layout was not rejected: {failed.stderr}')
            for prompt in ('7','7,42,99'):
                outputs = []
                for path in (qfile, ffile):
                    outfile = root/'logits.bin'
                    proc = subprocess.run([str(probe), str(path), '--prompt-ids', prompt, str(outfile)],
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=case_env, text=True)
                    if proc.returncode:
                        raise RuntimeError(f'{fmt} {prompt}: {proc.stderr}')
                    if path == qfile:
                        marker = next((line for line in proc.stderr.splitlines() if name in line), '')
                        if f'fmt={fmt} ' not in marker or 'matrix=1' not in marker:
                            raise AssertionError(f'{fmt}: mapped route missing: {marker}')
                    outputs.append(np.fromfile(outfile, '<f4'))
                if not np.isfinite(outputs[0]).all() or not np.allclose(outputs[0], outputs[1], rtol=1e-4, atol=5e-4):
                    raise AssertionError(f'{fmt} prompt={prompt}: max error {np.max(np.abs(outputs[0]-outputs[1]))}')
            print(f'{fmt}: mapped projection prefill T=1,3 matches F32 twin')
    return 0

if __name__ == '__main__':
    sys.exit(main())
