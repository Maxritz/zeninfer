"""Independent ROCmFP4 cross-check for real GGUF tensors (types 100/101).

Reference implementation of the ciru-ai/ROCmFPX Codebook10 + UE4M3 format,
written from the fork's rocmfp4.c semantics (independent of the C++ decoder):
  - type 100: { qs[16]; e[2]; }  18 B/block, dual UE4M3 half-block scales
  - type 101: { qs[16]; e;   }  17 B/block, one UE4M3 scale
  - codebook: 0,1,2,3,4,6,8,10, 0,-1,-2,-3,-4,-6,-8,-10
  - UE4M3 (bias 7): byte<=0x7e -> (8+M)*2^(E-10), E==0 subnormal M*2^-10,
    byte 0x7f -> 0
  - nibble order: byte j -> outputs j (low) and j+16 (high)

Usage: python rocmfp4_check.py model.gguf tensor_name probe.f32 [n_blocks]
"""
import math
import struct
import sys

CODEBOOK = [0, 1, 2, 3, 4, 6, 8, 10, 0, -1, -2, -3, -4, -6, -8, -10]
META_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def ue4m3_to_fp32_half(e):
    if e >= 0x7F:
        return 0.0
    M, E = e & 7, e >> 3
    return M * 2.0 ** -10 if E == 0 else (8 + M) * 2.0 ** (E - 10)


def dequant(type_id, block):
    out = [0.0] * 32
    if type_id == 100:
        d0 = ue4m3_to_fp32_half(block[16])
        d1 = ue4m3_to_fp32_half(block[17])
        for j in range(16):
            out[j] = d0 * CODEBOOK[block[j] & 15]
            out[j + 16] = d1 * CODEBOOK[block[j] >> 4]
    else:
        d = ue4m3_to_fp32_half(block[16])
        for j in range(16):
            out[j] = d * CODEBOOK[block[j] & 15]
            out[j + 16] = d * CODEBOOK[block[j] >> 4]
    return out


def locate(f, name):
    f.seek(0)
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
        tname = rs().decode("utf-8", "replace")
        (nd,) = struct.unpack("<I", f.read(4))
        struct.unpack("<" + "Q" * nd, f.read(8 * nd))
        (ggml_type,) = struct.unpack("<I", f.read(4))
        (offset,) = struct.unpack("<Q", f.read(8))
        if tname == name:
            return ggml_type, offset, version, n_tensors, n_kv
    return None


def data_offset(f, version, n_tensors, n_kv):
    f.seek(0)
    f.read(4)
    struct.unpack("<IQQ", f.read(20))

    def rs():
        (n,) = struct.unpack("<Q", f.read(8))
        f.read(n)

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
    return (f.tell() + 31) & ~31


def main():
    model, tensor_name, dump = sys.argv[1], sys.argv[2], sys.argv[3]
    n_blocks = int(sys.argv[4]) if len(sys.argv) > 4 else 16

    with open(model, "rb") as f:
        found = locate(f, tensor_name)
        if not found:
            raise SystemExit(f"tensor not found: {tensor_name}")
        ggml_type, t_offset, version, n_tensors, n_kv = found
        if ggml_type not in (100, 101):
            raise SystemExit(f"not a ROCmFP4 tensor (type {ggml_type})")
        base = data_offset(f, version, n_tensors, n_kv)
        f.seek(base + t_offset)
        raw = f.read(n_blocks * 18)

    block_bytes = 18 if ggml_type == 100 else 17
    reference = []
    for b in range(n_blocks):
        reference.extend(dequant(ggml_type, raw[b * block_bytes:(b + 1) * block_bytes]))

    with open(dump, "rb") as f:
        actual = struct.unpack("<%df" % (n_blocks * 32), f.read(n_blocks * 32 * 4))

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
    print(f"rocmfp4_check: {n_blocks*32} values, {mismatches} mismatches, "
          f"worst relative diff {worst:g}")
    if mismatches:
        raise SystemExit(1)
    print(f"ROCmFP4 real-model cross-check (type {ggml_type}): PASS (bit-exact)")


if __name__ == "__main__":
    main()
