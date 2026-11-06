"""Scan single-byte E8M0 scale positions: value in plausible exponent range
AND smooth across consecutive blocks (block amax varies slowly in real nets).

Usage: python e8m0_scan.py model.gguf <type_id> <stride...>
"""
import struct
import sys

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


def main():
    path, type_id = sys.argv[1], int(sys.argv[2])
    strides = [int(s) for s in sys.argv[3:]] or [17, 136]
    f, base = payload_start(path)
    f.seek(base)
    blob = f.read(2_000_000)
    for stride in strides:
        nb = len(blob) // stride
        idxs = list(range(0, nb, max(1, nb // 1200)))
        results = []
        for pos in range(stride):
            vals = [blob[i * stride + pos] for i in idxs if i * stride + pos < len(blob)]
            inrange = sum(1 for v in vals if 100 <= v <= 175) / len(vals)
            # smoothness: mean |diff| between consecutive exponents
            diffs = [abs(vals[i + 1] - vals[i]) for i in range(len(vals) - 1)]
            smooth = 1.0 / (1.0 + sum(diffs) / len(diffs))
            results.append((inrange * smooth, inrange, sum(diffs) / len(diffs), pos))
        results.sort(reverse=True)
        top = results[:4]
        print(f"  stride {stride:4d}:")
        for score, ir, md, pos in top:
            print(f"    pos={pos:4d} in-range={ir:6.1%} mean|diff|={md:6.2f} score={score:.3f}")


if __name__ == "__main__":
    main()
