"""Scan all (stride, in-block offset) positions for plausible fp16 scale bytes.

A real quantized format stores block scales as fp16 or E8M0 at a fixed
position in a fixed-size block. Scanning every position of every candidate
stride over thousands of blocks, a real scale position shows a spike of
plausible values; random data does not.

Usage: python stride_scan.py model.gguf <type_id> <bytes_per_block_guess...>
"""
import struct
import sys
import math

META_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def payload_start(path):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF"
    version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))

    def rs():
        (n,) = struct.unpack("<Q", f.read(8))
        return f.read(n)

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
    for _ in range(n_tensors):
        rs()
        (nd,) = struct.unpack("<I", f.read(4))
        f.read(8 * nd)
        f.read(4)
        f.read(8)
    return f, (f.tell() + 31) & ~31


def plausible_f16(x):
    return x == x and abs(x) != math.inf and 1e-5 < abs(x) < 1e3


def main():
    path, type_id = sys.argv[1], int(sys.argv[2])
    strides = [int(s) for s in sys.argv[3:]] or [17, 18, 68, 72, 136, 144]
    f, base = payload_start(path)
    f.seek(base)
    blob = f.read(2_000_000)
    print(f"payload base={base}")
    for stride in strides:
        nblocks = len(blob) // stride
        best = []
        for pos in range(stride):
            # sample up to 1500 blocks for speed
            idxs = range(0, nblocks, max(1, nblocks // 1500))
            vals = []
            for i in idxs:
                off = i * stride + pos
                if off + 2 > len(blob):
                    break
                (x,) = struct.unpack("<e", blob[off:off + 2])
                vals.append(x)
            frac = sum(1 for x in vals if plausible_f16(x)) / max(1, len(vals))
            best.append((frac, pos, len(vals)))
        best.sort(reverse=True)
        top = best[:3]
        print(f"  stride {stride:4d}: " + "  ".join(
            f"pos={p} plausible={fr:.2%} (n={n})" for fr, p, n in top))


if __name__ == "__main__":
    main()
