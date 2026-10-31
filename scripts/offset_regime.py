"""Offset-regime detection: are tensor offsets base-relative or absolute?

For each file: report min/max tensor offsets, dir_end, file_size.
- If min(offset) ~ 0 and max(offset)+size ~ file_size - base: relative (standard).
- If min(offset) ~ dir_end (a few MB): offsets are absolute positions.
Also derives base from the assumption that the last tensor ends at file_size.

Usage: python offset_regime.py model.gguf
"""
import struct
import sys
import os

META_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
BLOCK = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24),
         8: (32, 34), 10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176),
         14: (256, 210), 15: (256, 292), 16: (256, 66), 17: (256, 74), 18: (256, 98),
         19: (256, 50), 20: (32, 18), 21: (256, 110), 22: (256, 82), 23: (256, 136),
         30: (1, 2), 34: (256, 54), 35: (256, 66), 39: (32, 17)}


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
        elements = 1
        for d in dims:
            elements *= d
        tensors.append((name, t, elements, off))
    return tensors, f.tell()


def main():
    path = sys.argv[1]
    tensors, dir_end = walk(path)
    fsize = os.path.getsize(path)
    offs = sorted(t[3] for t in tensors)
    print(f"dir_end={dir_end} file_size={fsize}")
    print(f"offsets: min={offs[0]} max={offs[-1]}")
    gap_to_dir = offs[0] - dir_end
    print(f"min_offset - dir_end = {gap_to_dir}")
    if 0 <= gap_to_dir < 65536:
        print("REGIME: base-relative (min offset sits just past directory)")
    elif offs[0] >= 1_000_000:
        print("REGIME: ABSOLUTE file offsets!")
    # per-type byte footprint check with absolute hypothesis:
    # sum of known-type sizes should fit within [min_off, file_size]
    known_bytes = 0
    unknown_bytes = 0
    for _, t, el, _o in tensors:
        spec = BLOCK.get(t)
        if spec and el % spec[0] == 0:
            known_bytes += (el // spec[0]) * spec[1]
        else:
            unknown_bytes += el  # placeholder
    print(f"known-type bytes={known_bytes/1e9:.3f} GB, unknown-type elements={unknown_bytes/1e6:.1f}M")


if __name__ == "__main__":
    main()
