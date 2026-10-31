"""Pin the payload base using a Q6_K tensor (fp16 delta at pos 208, stride 210).

For candidate base B in [dir_end, dir_end+window), the delta for block k sits
at B + off + k*210 + 208. For each k this is a contiguous window, so we read
one window per block and score every base by fp16 sanity + smoothness.

Usage: python pin_base_q6k.py model.gguf [window=16384]
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
    for _ in range(n_tensors):
        name = rs()
        (nd,) = struct.unpack("<I", f.read(4))
        f.read(8 * nd)
        (t,) = struct.unpack("<I", f.read(4))
        (off,) = struct.unpack("<Q", f.read(8))
        if t == 14:  # Q6_K
            return off
    return None


def main():
    path = sys.argv[1]
    window = int(sys.argv[2]) if len(sys.argv) > 2 else 16384
    off = walk(path)
    if off is None:
        raise SystemExit("no Q6_K tensor in file")
    f = open(path, "rb")
    dir_end = None
    # recompute dir_end
    f.seek(0)
    f.read(4)
    struct.unpack("<IQQ", f.read(20))

    def rs():
        (n,) = struct.unpack("<Q", f.read(8))
        f.read(n)

    # replay to end of directory (same walk logic)
    version, n_tensors, n_kv = struct.unpack("<IQQ", open(path, "rb").read(0) or b"") if False else (None, None, None)
    f2 = open(path, "rb")
    f2.read(4)
    version, n_tensors, n_kv = struct.unpack("<IQQ", f2.read(20))
    for _ in range(n_kv):
        rs2 = rs
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
        (nd,) = struct.unpack("<I", f2.read(4)) if False else (None,)
        break
    # dir_end via separate clean walk
    f3 = open(path, "rb")
    f3.read(4)
    struct.unpack("<IQQ", f3.read(20))
    for _ in range(n_kv):
        (n,) = struct.unpack("<Q", f3.read(8)); f3.read(n)
        (t,) = struct.unpack("<I", f3.read(4))
        if t == 8:
            (n,) = struct.unpack("<Q", f3.read(8)); f3.read(n)
        elif t == 9:
            (sub,) = struct.unpack("<I", f3.read(4))
            (count,) = struct.unpack("<Q", f3.read(8))
            for _ in range(count):
                if sub == 8:
                    (n,) = struct.unpack("<Q", f3.read(8)); f3.read(n)
                else:
                    f3.read(META_SIZES[sub])
        else:
            f3.read(META_SIZES[t])
    for _ in range(n_tensors):
        (n,) = struct.unpack("<Q", f3.read(8)); f3.read(n)
        (nd,) = struct.unpack("<I", f3.read(4))
        f3.read(8 * nd)
        f3.read(4)
        f3.read(8)
    dir_end = f3.tell()

    nblocks = 384
    windows = []
    f.seek(0)
    for k in range(nblocks):
        start = dir_end + off + k * 210 + 208  # lowest candidate base
        f.seek(start)
        windows.append(f.read(window + 2))
    best = []
    for w in range(window):
        sane = 0
        prev = None
        smooth = 0.0
        for k in range(nblocks):
            chunk = windows[k]
            if w + 2 > len(chunk):
                break
            (x,) = struct.unpack("<e", chunk[w:w + 2])
            if x == x and abs(x) != math.inf and 1e-6 < x < 1e3:
                sane += 1
                if prev:
                    smooth += 1.0 / (1.0 + abs(math.log2(x / prev)))
                prev = x
        best.append((sane / nblocks, smooth / nblocks, dir_end + w))
    best.sort(reverse=True)
    for frac, sm, base in best[:5]:
        print(f"  base={base} (dir_end+{base-dir_end}): sane fp16={frac:6.1%} smoothness={sm:.3f}")


if __name__ == "__main__":
    main()
