"""Independent MXFP4 cross-check for real GGUF tensors.

Parses the GGUF tensor directory, extracts the same raw block bytes the C++
probe (zen_mxfp4_probe) decoded, dequantizes them with an independent
implementation of the MXFP4 spec (E8M0 -> GGML_E8M0_TO_FP32_HALF semantics,
doubled-E2M1 kvalues_fp4 table from ggml-common.h), and compares against the
probe's binary dump.

Usage: python mxfp4_check.py model.gguf tensor_name mxfp4_probe.f32 [n_blocks]
"""
import math
import struct
import sys

# kvalues_fp4 verbatim from ggml-common.h (kvalues_mxfp4 aliases this table).
KVALUES_FP4 = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12]

META_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def e8m0_to_fp32_half(x):
    """GGML_E8M0_TO_FP32_HALF: 0.5 * 2^(x-127), with subnormals for x < 2."""
    if x < 2:
        bits = 0x00200000 << x
    else:
        bits = (x - 1) << 23
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def find_tensor(f, name):
    f.seek(0)
    assert f.read(4) == b"GGUF"
    version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))

    def read_str():
        (n,) = struct.unpack("<Q", f.read(8))
        return f.read(n)

    for _ in range(n_kv):
        read_str()
        (t,) = struct.unpack("<I", f.read(4))
        if t == 8:
            read_str()
        elif t == 9:
            (sub,) = struct.unpack("<I", f.read(4))
            (count,) = struct.unpack("<Q", f.read(8))
            for _ in range(count):
                if sub == 8:
                    read_str()
                else:
                    f.read(META_SIZES[sub])
        else:
            f.read(META_SIZES[t])

    for _ in range(n_tensors):
        tname = read_str().decode("utf-8", "replace")
        (nd,) = struct.unpack("<I", f.read(4))
        struct.unpack("<" + "Q" * nd, f.read(8 * nd))
        (ggml_type,) = struct.unpack("<I", f.read(4))
        (offset,) = struct.unpack("<Q", f.read(8))
        if tname == name:
            return ggml_type, offset, version, n_tensors, n_kv
    return None


def data_offset(f, version, n_tensors, n_kv):
    """Re-walk the file to the end of the tensor directory and align."""
    f.seek(0)
    f.read(4)
    struct.unpack("<IQQ", f.read(20))

    def read_str():
        (n,) = struct.unpack("<Q", f.read(8))
        f.read(n)

    for _ in range(n_kv):
        read_str()
        (t,) = struct.unpack("<I", f.read(4))
        if t == 8:
            read_str()
        elif t == 9:
            (sub,) = struct.unpack("<I", f.read(4))
            (count,) = struct.unpack("<Q", f.read(8))
            for _ in range(count):
                if sub == 8:
                    read_str()
                else:
                    f.read(META_SIZES[sub])
        else:
            f.read(META_SIZES[t])
    for _ in range(n_tensors):
        read_str()
        (nd,) = struct.unpack("<I", f.read(4))
        f.read(8 * nd)
        f.read(4)
        f.read(8)
    pos = f.tell()
    return (pos + 31) & ~31  # default alignment 32


def dequant_mxfp4(block):
    """Independent MXFP4 decode of one 17-byte block -> 32 floats.

    Per dequantize_row_mxfp4 in ggml-quants.c: for j in 0..15, the LOW nibble
    of qs[j] decodes output j and the HIGH nibble decodes output j+16
    (split-half layout, not adjacent interleaving).
    """
    e = block[0]
    d = e8m0_to_fp32_half(e)
    out = [0.0] * 32
    for j, byte in enumerate(block[1:]):
        out[j] = d * float(KVALUES_FP4[byte & 15])
        out[j + 16] = d * float(KVALUES_FP4[byte >> 4])
    return out


def main():
    model, tensor_name, dump = sys.argv[1], sys.argv[2], sys.argv[3]
    n_blocks = int(sys.argv[4]) if len(sys.argv) > 4 else 16

    with open(model, "rb") as f:
        found = find_tensor(f, tensor_name)
        if not found:
            raise SystemExit(f"tensor not found: {tensor_name}")
        ggml_type, t_offset, version, n_tensors, n_kv = found
        if ggml_type != 39:
            raise SystemExit(f"not MXFP4 (type {ggml_type})")
        base = data_offset(f, version, n_tensors, n_kv)
        f.seek(base + t_offset)
        raw = f.read(n_blocks * 17)

    reference = []
    for b in range(n_blocks):
        reference.extend(dequant_mxfp4(raw[b * 17:(b + 1) * 17]))

    with open(dump, "rb") as f:
        actual = struct.unpack("<%df" % (n_blocks * 32), f.read(n_blocks * 32 * 4))

    worst = 0.0
    mismatches = 0
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
    print(f"mxfp4_check: {n_blocks*32} values, {mismatches} mismatches, "
          f"worst relative diff {worst:g}")
    if mismatches:
        raise SystemExit(1)
    print("MXFP4 real-model cross-check: PASS (bit-exact)")


if __name__ == "__main__":
    main()
