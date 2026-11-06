"""Test GPU-repack hypotheses for ROCMFP4 type 100/101.

Hypotheses for 4.5 bpw (type 100):
  H1: Q4_0 plain          - fp16 scale at stride-18 pos-0
  H2: Q4_0_4x4 repack     - 4 Q4_0 blocks interleaved qs-first, scales grouped:
                            llama.cpp legacy block_q4_0x4 = { ggml_half2 d[4];
                            qs[4*16] } = 8 + 64 = 72 bytes / 128 elements.
                            fp16 scales at stride-72, pos 0,2,4,6.
  H3: Q4_0_8x8 repack     - 8 blocks: d[8] (16 B) + qs[128] = 144 B / 256 el.
                            fp16 scales at stride-144, pos 0..14 step 2.
Hypotheses for 4.25 bpw (type 101):
  H4: MXFP4               - E8M0 at stride-17 pos-0  (already rejected)
  H5: IQ4_XS plain        - fp16 at 136 pos-0        (already rejected)
  H6: repacked IQ4_XS tile- interleaved in groups.

Score = fraction of sampled scales that are finite AND in [1e-6, 1e3] AND
consecutive-log2-smooth. Real weights score near 100%; noise ~50-80%.

Usage: python repack_probe.py model.gguf <type_id>
"""
import struct
import sys
import math

META_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def payload_base(path):
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


def score_fp16_series(vals):
    ok = smooth = 0
    prev = None
    for x in vals:
        if x == x and abs(x) != math.inf and 1e-6 < abs(x) < 1e3:
            ok += 1
            if prev is not None:
                smooth += 1.0 / (1.0 + abs(math.log2(abs(x) / abs(prev))))
            prev = x
    n = max(1, len(vals))
    return ok / n, smooth / n


def main():
    path, type_id = sys.argv[1], int(sys.argv[2])
    f, base = payload_base(path)
    f.seek(base)
    blob = f.read(1_000_000)
    N = 1200

    print(f"== {path} type {type_id} ==")
    # H2/H3 grouped-scales: scales packed at the START of each super-block
    for name, stride, positions in [
        ("Q4_0x4 grouped (72B/128el)", 72, [0, 2, 4, 6]),
        ("Q4_0x8 grouped (144B/256el)", 144, [0, 2, 4, 6, 8, 10, 12, 14]),
        ("Q4_0 plain (18B)", 18, [0]),
        ("Q4_K (144B, scales at 4..15)", 144, [4, 6, 8, 10, 12, 14]),
    ]:
        nb = len(blob) // stride
        idxs = list(range(0, nb, max(1, nb // N)))
        allscores = []
        for pos in positions:
            vals = []
            for i in idxs:
                o = i * stride + pos
                if o + 2 <= len(blob):
                    (x,) = struct.unpack("<e", blob[o:o + 2])
                    vals.append(x)
            okr, sm = score_fp16_series(vals)
            allscores.append((okr, sm))
        avg_ok = sum(a for a, _ in allscores) / len(allscores)
        avg_sm = sum(b for _, b in allscores) / len(allscores)
        print(f"  {name:32s} sanity={avg_ok:6.1%} smoothness={avg_sm:.3f}")

    # H6-style: for 4.25 bpw, test interleaved MXFP4 groups (17*k strides) and
    # IQ4_XS variants with grouped scales at start
    if type_id == 101:
        for name, stride, positions in [
            ("IQ4_XS grouped? (136B)", 136, [0, 2]),
            ("MXFP4 x8 grouped (136B/256el)", 136, [0, 1, 2, 3, 4, 5, 6, 7]),
            ("MXFP4 plain (17B)", 17, [0]),
        ]:
            nb = len(blob) // stride
            idxs = list(range(0, nb, max(1, nb // N)))
            allscores = []
            for pos in positions:
                vals = []
                for i in idxs:
                    o = i * stride + pos
                    if o < len(blob):
                        vals.append(blob[o])  # E8M0 single byte
                okr = sum(1 for v in vals if 100 <= v <= 175) / max(1, len(vals))
                diffs = [abs(vals[i + 1] - vals[i]) for i in range(len(vals) - 1)]
                sm = 1.0 / (1.0 + (sum(diffs) / len(diffs) if diffs else 255))
                allscores.append((okr, sm))
            avg_ok = sum(a for a, _ in allscores) / len(allscores)
            avg_sm = sum(b for _, b in allscores) / len(allscores)
            print(f"  {name:32s} e8m0-range={avg_ok:6.1%} smoothness={avg_sm:.3f}")


if __name__ == "__main__":
    main()
