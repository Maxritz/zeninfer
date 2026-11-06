"""Independent PTQ1_0 (GGUF type 143) dequant cross-check.

Reference implementation written from the PrismML fork's dequantize_row_ptq1_0
(ggml-quants.c): TQ1_0 base-3 trit packing at group size 128, block
{ qs[24]; qh[2]; f16 d; } = 28 B. Digit decode: q = byte*3^n truncated to 8
bits, xi = (q*3)>>8, value = (xi-1)*d.

Usage: python ptq1_0_check.py model.gguf tensor_name probe.f32 [n_blocks]
"""
import math
import struct
import sys

META_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def f16(bits):
    sign = -1.0 if bits & 0x8000 else 1.0
    exp = (bits >> 10) & 0x1F
    mant = bits & 0x03FF
    if exp == 0:
        return sign * mant * (2.0 ** -24)
    if exp == 0x1F:
        return float("inf") if mant == 0 else float("nan")
    return sign * (1.0 + mant / 1024.0) * (2.0 ** (exp - 15))


def dequant_ptq1_0(block):
    d = f16(struct.unpack_from("<H", block, 26)[0])
    out = []
    qs, qh = block[0:24], block[24:26]
    for n in range(5):
        for byte in qs:
            q = (byte * (3 ** n)) & 0xFF
            out.append((((q * 3) >> 8) - 1) * d)
    for n in range(4):
        for byte in qh:
            q = (byte * (3 ** n)) & 0xFF
            out.append((((q * 3) >> 8) - 1) * d)
    assert len(out) == 128, len(out)
    return out


def locate(path, name):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF"
    version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
    found = None

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
        tname = rs()
        (nd,) = struct.unpack("<I", f.read(4))
        dims = struct.unpack("<" + "Q" * nd, f.read(8 * nd))
        (t,) = struct.unpack("<I", f.read(4))
        (off,) = struct.unpack("<Q", f.read(8))
        if tname == name:
            found = (t, dims, off)
    if not found:
        raise SystemExit(f"tensor not found: {name}")
    return found[0], found[1], found[2], f.tell()


def main():
    model, tname, dump = sys.argv[1], sys.argv[2], sys.argv[3]
    n_blocks = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    ggml_type, dims, off, dir_end = locate(model, tname)
    if ggml_type != 143:
        raise SystemExit(f"not PTQ1_0 (type {ggml_type})")
    row = int(dims[0])
    assert row % 128 == 0, row
    blocks_per_row = row // 128
    base = (dir_end + 31) & ~31
    with open(model, "rb") as f:
        f.seek(base + off)
        raw = f.read(n_blocks * 28)

    reference = []
    for b in range(n_blocks):
        reference.extend(dequant_ptq1_0(raw[b * 28:(b + 1) * 28]))

    with open(dump, "rb") as f:
        actual = struct.unpack("<%df" % len(reference), f.read(len(reference) * 4))

    mismatches = 0
    worst = 0.0
    for i, (ref, act) in enumerate(zip(reference, actual)):
        if math.isnan(ref) != math.isnan(act):
            mismatches += 1
            continue
        if math.isnan(ref):
            continue
        denom = max(1.0, abs(ref))
        worst = max(worst, abs(ref - act) / denom)
        if ref != act and abs(ref - act) / denom > 1e-6:
            mismatches += 1
            if mismatches <= 5:
                print(f"  mismatch at {i}: python={ref!r} cpp={act!r}")
    print(f"ptq1_0_check: {len(reference)} values, {mismatches} mismatches, worst relative diff {worst:g}")
    if mismatches:
        raise SystemExit(1)
    print("PTQ1_0 real-model cross-check: PASS (bit-exact)")


if __name__ == "__main__":
    main()
