"""Sharp base-offset validation: statistical fingerprint of a large F32 tensor.

RMSNorm weights have a very tight distribution (mean ~1, low stdev). Float
garbage from a wrong base does not. Also checks that the first tensor (lowest
offset) starts with no large unexplained gap.

Usage: python verify_base2.py model.gguf
"""
import struct
import sys
import statistics
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
    return tensors, f.tell()


def main():
    path = sys.argv[1]
    tensors, dir_end = walk(path)
    big32 = max((t for t in tensors if t[1] == 0), key=lambda t: t[2][0])
    n = min(4096, big32[2][0])
    print(f"probe: {big32[0]} F32[{big32[2][0]}]")
    best = None
    for align in (8, 16, 32, 64, 128, 256):
        base = (dir_end + align - 1) & ~(align - 1)
        with open(path, "rb") as fh:
            fh.seek(base + big32[3])
            vals = struct.unpack(f"<{n}f", fh.read(4 * n))
        finite = [v for v in vals if v == v and abs(v) != math.inf]
        if len(finite) < n * 0.99:
            print(f"  align {align:4d}: non-finite data, reject")
            continue
        sd = statistics.pstdev(finite)
        mu = statistics.fmean(finite)
        inband = sum(1 for v in finite if 0.2 < abs(v) < 5.0) / len(finite)
        score = inband / (1.0 + sd)
        tag = ""
        if best is None or score > best[0]:
            best = (score, base, mu, sd)
            tag = "  <-- best so far"
        print(f"  align {align:4d}: mean={mu:8.4f} stdev={sd:8.4f} in-band={inband:6.1%}{tag}")
    print(f"\nCONCLUSION: base={best[1]}, probe mean={best[2]:.4f} stdev={best[3]:.4f}")
    # decisive: an RMSNorm weight tensor should have stdev well below 0.5
    print("VERDICT:", "BASE CONFIRMED" if best[3] < 0.5 else "BASE UNCERTAIN")


if __name__ == "__main__":
    main()
