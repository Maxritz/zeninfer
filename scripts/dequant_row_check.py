"""Row-level dequant differential: NumPy reference vs zen_mxfp4_probe dump.

Independent re-implementation of Q4_K / Q6_K dequant for the first rows of a
tensor, compared bit-tight against the C++ probe output.

Usage: python dequant_row_check.py model.gguf tensor_name probe.f32 [n_rows]
"""
import struct
import sys
import numpy as np

META_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


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
        raise SystemExit("tensor not found")
    return found[0], found[1], found[2], f.tell()


def deq_q4_k(raw):
    nb = raw.shape[0]
    d = raw[:, 0:2].copy().view("<f2").astype(np.float64)
    dmin = raw[:, 2:4].copy().view("<f2").astype(np.float64)
    sc = raw[:, 4:16].astype(np.int32)
    qs = raw[:, 16:144]
    out = np.empty((nb, 256), dtype=np.float64)

    def gsmk(j):
        if j < 4:
            return sc[:, j] & 63, sc[:, j + 4] & 63
        return (sc[:, j + 4] & 0xF) | ((sc[:, j - 4] >> 6) << 4), \
               (sc[:, j + 4] >> 4) | ((sc[:, j] >> 6) << 4)

    for g in range(4):
        s0, m0 = gsmk(2 * g)
        s1, m1 = gsmk(2 * g + 1)
        q = qs[:, g * 32:(g + 1) * 32]
        out[:, g * 64:g * 64 + 32] = (d[:, 0] * s0)[:, None] * (q & 15) - (dmin[:, 0] * m0)[:, None]
        out[:, g * 64 + 32:(g + 1) * 64] = (d[:, 0] * s1)[:, None] * (q >> 4) - (dmin[:, 0] * m1)[:, None]
    return out


def deq_q6_k(raw):
    nb = raw.shape[0]
    ql, qh = raw[:, 0:128], raw[:, 128:192]
    sc = raw[:, 192:208].copy().view(np.int8).astype(np.float64)
    d = raw[:, 208:210].copy().view("<f2").astype(np.float64)[:, 0]
    out = np.empty((nb, 256), dtype=np.float64)
    l = np.arange(32)
    isx = l // 16
    for n128 in range(2):
        qlh = ql[:, n128 * 64:(n128 + 1) * 64]
        qhh = qh[:, n128 * 32:(n128 + 1) * 32]
        sch = sc[:, n128 * 8:(n128 + 1) * 8]
        q1 = (qlh[:, l] & 15) | (((qhh[:, l] >> 0) & 3) << 4)
        q2 = (qlh[:, l + 32] & 15) | (((qhh[:, l] >> 2) & 3) << 4)
        q3 = (qlh[:, l] >> 4) | (((qhh[:, l] >> 4) & 3) << 4)
        q4 = (qlh[:, l + 32] >> 4) | (((qhh[:, l] >> 6) & 3) << 4)
        base = n128 * 128 + l
        for k, q in ((0, q1), (32, q2), (64, q3), (96, q4)):
            out[:, base + k] = d[:, None] * sch[:, isx + (k // 32) * 2] * (q.astype(np.int32) - 32)
    return out


def main():
    model, tname, dump = sys.argv[1], sys.argv[2], sys.argv[3]
    n_rows = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    ggml_type, dims, off, dir_end = locate(model, tname)
    row = int(dims[0])
    if ggml_type == 12:
        rb, per_row = 144, 256
    elif ggml_type == 14:
        rb, per_row = 210, 256
    else:
        raise SystemExit(f"unsupported type {ggml_type}")
    assert row % per_row == 0
    nb_per_row = row // per_row
    base = (dir_end + 31) & ~31
    with open(model, "rb") as f:
        f.seek(base + off)
        raw = np.frombuffer(f.read(n_rows * rb * nb_per_row), dtype=np.uint8).reshape(n_rows * nb_per_row, rb)
    if ggml_type == 12:
        ref = deq_q4_k(raw)
    else:
        ref = deq_q6_k(raw)
    ref = ref.reshape(n_rows, row)
    with open(dump, "rb") as f:
        actual = np.frombuffer(f.read(n_rows * row * 4), dtype="<f4").astype(np.float64).reshape(n_rows, row)
    diff = np.abs(ref - actual)
    denom = np.maximum(1.0, np.abs(ref))
    worst = float((diff / denom).max())
    print(f"{n_rows} rows x {row} values: max_abs={diff.max():.6g} worst_rel={worst:.3g}")
    print("ROW DIFF:", "PASS" if worst < 1e-6 else "FAIL")
    raise SystemExit(0 if worst < 1e-6 else 1)


if __name__ == "__main__":
    main()
