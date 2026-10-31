"""Validate the computed payload base offset using a known-format tensor.

For an F32 tensor, bytes at base+offset must decode as sane small floats.
For Q8_0, the fp16 scale at offset 0 of each 34-byte block must be sane.
Tries several alignments and reports which produces sane data.

Usage: python verify_base.py model.gguf
"""
import struct
import sys
import math

META_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def walk(path):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF"
    version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))

    def rs():
        (n,) = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", "replace")

    for _ in range(n_kv):
        rs()
        (t,) = struct.unpack("<I", f.read(4))
        if t == 8:
            rs()
        elif t == 9:
            (sub,) = struct.unpack("<I", f.read(4))
            (count,) = struct.unpack("<Q", f.read(8))
            for _ in range(count):
                if sub == 8:
                    rs()
                else:
                    f.read(META_SIZES[sub])
        else:
            f.read(META_SIZES[t])
    tensors = []
    for _ in range(n_tensors):
        name = rs()
        (nd,) = struct.unpack("<I", f.read(4))
        dims = struct.unpack("<" + "Q" * nd, f.read(8 * nd))
        (t,) = struct.unpack("<I", f.read(4))
        (off,) = struct.unpack("<Q", f.read(8))
        tensors.append((name, t, dims, off))
    dir_end = f.tell()
    import os
    return tensors, dir_end, os.path.getsize(path)


def main():
    path = sys.argv[1]
    tensors, dir_end, fsize = walk(path)
    print(f"dir_end={dir_end} file_size={fsize}")
    # candidate bases: directory end aligned to 32/64/128/256, plus padded sizes
    bases = sorted({(dir_end + a - 1) & ~(a - 1) for a in (8, 16, 32, 64, 128, 256, 512, 4096)})
    # find a F32 tensor and a Q8_0 tensor of decent size
    f32 = next((t for t in tensors if t[1] == 0 and t[2][0] >= 256), None)
    q8 = next((t for t in tensors if t[1] == 8 and t[2][0] >= 256), None)
    print("F32 probe tensor:", f32 and f32[0], " Q8_0 probe:", q8 and q8[0])
    import os
    for base in bases:
        verdict = []
        if f32:
            with open(path, "rb") as fh:
                fh.seek(base + f32[3])
                vals = struct.unpack("<16f", fh.read(64))
            sane = sum(1 for v in vals if v == v and abs(v) != math.inf and abs(v) < 1e4)
            verdict.append(f"F32 sane {sane}/16 (max {max(abs(v) for v in vals):.4g})")
        if q8:
            with open(path, "rb") as fh:
                fh.seek(base + q8[3])
                blob = fh.read(34 * 16)
            scales = [struct.unpack("<e", blob[i * 34:i * 34 + 2])[0] for i in range(16)]
            sane = sum(1 for s in scales if s == s and abs(s) != math.inf and 1e-6 < abs(s) < 1e2)
            verdict.append(f"Q8_0 scales sane {sane}/16")
        print(f"  base={base} (pad {base-dir_end}): " + " | ".join(verdict))


if __name__ == "__main__":
    main()
