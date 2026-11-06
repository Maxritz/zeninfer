#!/usr/bin/env python3
"""Regenerate tests/data/new_quant_ggml.bin using the pinned upstream ggml-base.

Build llama.cpp commit 240d68fcfe0c2a1f298c8e18ca181368b1914259 with
BUILD_SHARED_LIBS=ON, then pass the path to libggml-base as argv[1]. Q8_1
and scalar types use Python's struct/arithmetic: upstream Q8_1 has no
public to-float function. Fixture is intentionally checked in for offline CI.
"""
import ctypes
import math
import random
import struct
import sys
from pathlib import Path

lib = ctypes.CDLL(sys.argv[1])
rng = random.Random(20261106)
records = []
def upstream(name, tid, payload, n):
    f = getattr(lib, 'dequantize_row_' + name.lower())
    f.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
    f.restype = None
    src = ctypes.create_string_buffer(bytes(payload))
    out = (ctypes.c_float * n)()
    f(src, out, n)
    records.append((name, tid, payload, list(out)))

# FP16 block scales include negative, zero, subnormal and larger magnitudes.
scales = [0x0000, 0x8000, 0x0001, 0x03ff, 0x3c00, 0xbc00, 0x3400, 0x4800]
for name, tid, block in [('Q1_0', 41, 128), ('Q2_0', 42, 64)]:
    payload = bytearray()
    for i in range(32):
        payload += struct.pack('<H', scales[i % len(scales)])
        payload += bytes(rng.randrange(256) for _ in range(16))
    upstream(name, tid, payload, 32 * block)
# Covers every possible scale byte, every nibble, all subblocks and block strides.
payload = bytearray()
for scale in range(256):
    payload += bytes([scale] * 4)
    payload += bytes((j * 17 + scale) % 256 for j in range(32))
upstream('NVFP4', 40, payload, 256 * 64)
upstream('F8_E4M3', 43, bytearray(range(256)), 256)
upstream('F8_E4M3', 43, bytearray(i for i in range(256) if i not in (127, 255)), 254)
# Q8_1: FP16 d, FP16 sum s (ignored in decode), followed by signed bytes.
payload, expected = bytearray(), []
for i in range(32):
    d = struct.unpack('<e', struct.pack('<H', scales[i % len(scales)]))[0]
    payload += struct.pack('<HH', scales[i % len(scales)], 0x7e00) # s deliberately NaN
    q = [rng.randrange(-128, 128) for _ in range(32)]
    payload += struct.pack('<32b', *q)
    expected.extend(d * v for v in q)
records.append(('Q8_1', 9, payload, expected))
for name, tid, fmt, values in [
    ('I8',24,'b',[-128,-1,0,1,127]),
    ('I16',25,'h',[-32768,-1,0,1,32767]),
    ('I32',26,'i',[-2147483648,-1,0,1,2147483647]),
    ('I64',27,'q',[-9223372036854775808,-1,0,1,9223372036854775807]),
    ('F64',28,'d',[-1.5,-0.0,0.0,1.0,3.25])]:
    records.append((name,tid,bytearray(struct.pack('<'+str(len(values))+fmt,*values)),values))

out = bytearray(b'ZQG1') + struct.pack('<I',len(records))
for name, tid, data, expected in records:
    name = name.encode()
    out += struct.pack('<I',len(name)) + name
    out += struct.pack('<III',tid,len(data),len(expected)) + data
    out += struct.pack('<'+str(len(expected))+'f', *expected)
Path('tests/data/new_quant_ggml.bin').write_bytes(out)
print(len(out),'bytes',len(records),'records')
