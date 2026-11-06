"""Independent NumPy forward-pass reference for ZenInfer dense/BitNet models.

Implements the transformer graph from scratch (own GGUF parser, own
dequantizers, NumPy math) and compares last-position logits against the
zen_parity_probe dump. Supported: qwen3 (per-head Q/K norms, decoupled
head_dim), bitnet (subln RMS norms, tied lm_head, TQ1_0 weights).

Usage: python model_numpy_reference.py model.gguf arch logits.f32
  arch: qwen3 | bitnet
"""
import math
import struct
import sys
import numpy as np

META_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def parse(path):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF"
    version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))

    def rs():
        (n,) = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", "replace")

    meta = {}
    for _ in range(n_kv):
        k = rs()
        (t,) = struct.unpack("<I", f.read(4))
        if t == 8:
            meta[k] = rs()
        elif t == 9:
            (sub,) = struct.unpack("<I", f.read(4))
            (count,) = struct.unpack("<Q", f.read(8))
            for _ in range(count):
                if sub == 8:
                    rs()
                else:
                    f.read(META_SIZES[sub])
        else:
            raw = f.read(META_SIZES[t])
            if t == 6:
                meta[k] = str(struct.unpack("<f", raw)[0])
            elif t == 12:
                meta[k] = str(struct.unpack("<d", raw)[0])
            else:
                meta[k] = str(int.from_bytes(raw, "little", signed=t in (1, 3, 5)))
    tensors = []
    for _ in range(n_tensors):
        name = rs()
        (nd,) = struct.unpack("<I", f.read(4))
        dims = struct.unpack("<" + "Q" * nd, f.read(8 * nd))
        (t,) = struct.unpack("<I", f.read(4))
        (off,) = struct.unpack("<Q", f.read(8))
        tensors.append((name, t, dims, off))
    base = (f.tell() + 31) & ~31
    return f, base, meta, {t[0]: t for t in tensors}


def dequant(f, base, entry):
    name, t, dims, off = entry
    f.seek(base + off)
    elements = 1
    for d in dims:
        elements *= d
    # GGUF stores ne[0] as the contiguous (input) dimension. NumPy needs the
    # transposed orientation: a ggml {a, b} weight becomes a (b, a) matrix so
    # that y = x @ W.T matches llama's row-major matmul convention.
    np_shape = (int(dims[1]), int(dims[0])) if len(dims) == 2 else tuple(int(d) for d in dims)
    if t == 0:
        return np.frombuffer(f.read(4 * elements), dtype="<f4").reshape(np_shape).astype(np.float64)
    if t == 1:
        return np.frombuffer(f.read(2 * elements), dtype="<f2").reshape(np_shape).astype(np.float64)
    if t == 8:  # Q8_0: 34B block { f16 d; int8 qs[32]; }
        nb = elements // 32
        raw = np.frombuffer(f.read(34 * nb), dtype=np.uint8).reshape(nb, 34)
        d = raw[:, 0:2].copy().view("<f2").astype(np.float64)
        qs = raw[:, 2:34].copy().view(np.int8).astype(np.float64)
        return (qs * d).reshape(np_shape)
    if t == 12:  # Q4_K: 144B { f16 d; f16 dmin; scales[12]; qs[128]; }
        nb = elements // 256
        raw = np.frombuffer(f.read(144 * nb), dtype=np.uint8).reshape(nb, 144)
        d = raw[:, 0:2].copy().view("<f2").astype(np.float64)
        dmin = raw[:, 2:4].copy().view("<f2").astype(np.float64)
        sc = raw[:, 4:16].astype(np.int32)
        qs = raw[:, 16:144]
        out = np.empty((nb, 256), dtype=np.float64)

        def gsmk(j):
            # get_scale_min_k4, vectorised over blocks
            if j < 4:
                s = sc[:, j] & 63
                m = sc[:, j + 4] & 63
            else:
                s = (sc[:, j + 4] & 0xF) | ((sc[:, j - 4] >> 6) << 4)
                m = (sc[:, j + 4] >> 4) | ((sc[:, j] >> 6) << 4)
            return s, m
        # Scale PAIRS per 64 outputs: (is,is+1) = (2g, 2g+1) for each 64-group.
        for g in range(4):
            s0, m0 = gsmk(2 * g)
            s1, m1 = gsmk(2 * g + 1)
            q = qs[:, g * 32:(g + 1) * 32]
            d1 = (d[:, 0] * s0)[:, None]
            m1v = (dmin[:, 0] * m0)[:, None]
            d2 = (d[:, 0] * s1)[:, None]
            m2v = (dmin[:, 0] * m1)[:, None]
            out[:, g * 64:g * 64 + 32] = d1 * (q & 15).astype(np.float64) - m1v
            out[:, g * 64 + 32:(g + 1) * 64] = d2 * (q >> 4).astype(np.float64) - m2v
        return out.reshape(np_shape)
    if t == 14:  # Q6_K: 210B { ql[128]; qh[64]; i8 scales[16]; f16 d; }
        nb = elements // 256
        raw = np.frombuffer(f.read(210 * nb), dtype=np.uint8).reshape(nb, 210)
        ql, qh = raw[:, 0:128], raw[:, 128:192]
        sc = raw[:, 192:208].copy().view(np.int8).astype(np.float64)
        d = raw[:, 208:210].copy().view("<f2").astype(np.float64)[:, 0]
        out = np.empty((nb, 256), dtype=np.float64)
        for n128 in range(2):
            qlh = ql[:, n128 * 64:(n128 + 1) * 64]
            qhh = qh[:, n128 * 32:(n128 + 1) * 32]
            sch = sc[:, n128 * 8:(n128 + 1) * 8]
            l = np.arange(32)
            isx = l // 16
            q1 = (qlh[:, l] & 15) | (((qhh[:, l] >> 0) & 3) << 4)
            q2 = (qlh[:, l + 32] & 15) | (((qhh[:, l] >> 2) & 3) << 4)
            q3 = (qlh[:, l] >> 4) | (((qhh[:, l] >> 4) & 3) << 4)
            q4 = (qlh[:, l + 32] >> 4) | (((qhh[:, l] >> 6) & 3) << 4)
            base = n128 * 128 + l
            for k, q in ((0, q1), (32, q2), (64, q3), (96, q4)):
                out[:, base + k] = d[:, None] * sch[:, isx + (k // 32) * 2] * (q.astype(np.int32) - 32)
        return out.reshape(np_shape)
    if t == 34:  # TQ1_0: 54B { qs[48]; qh[4]; f16 d; } base-3 trits
        nb = elements // 256
        raw = np.frombuffer(f.read(54 * nb), dtype=np.uint8).reshape(nb, 54)
        qs, qh = raw[:, 0:48], raw[:, 48:52]
        d = raw[:, 52:54].copy().view("<f2").astype(np.float64)
        out = np.empty((nb, 256), dtype=np.float64)

        def dec(b, n):
            q = (b.astype(np.uint16) * np.uint16(3 ** n)) & np.uint16(0xFF)
            xi = ((q * np.uint16(3)) >> np.uint16(8)).astype(np.int16) - 1
            return xi.astype(np.float64)
        q32, q16 = qs[:, :32], qs[:, 32:48]
        for n in range(5):
            out[:, n * 32:(n + 1) * 32] = dec(q32, n)
            out[:, 160 + n * 16:160 + (n + 1) * 16] = dec(q16, n)
        for n in range(4):
            out[:, 240 + n * 4:240 + (n + 1) * 4] = dec(qh, n)
        return (out * d).reshape(np_shape)
    raise SystemExit(f"unsupported type {t} for {name}")


def rms_norm(x, w, eps):
    ss = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(ss + eps) * w


def rope(x, head_dim, rotary, base, positions):
    # x: (T, heads, head_dim); rotate pairs (i, i+half) like the C++ path.
    half = rotary // 2
    inv = base ** (-np.arange(0, rotary, 2, dtype=np.float64) / rotary)
    ang = positions[:, None].astype(np.float64) * inv[None, :]  # (T, half)
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    out = x.copy()
    a, b = x[..., :half], x[..., half:rotary]
    out[..., :half] = a * cos - b * sin
    out[..., half:rotary] = a * sin + b * cos
    return out


def forward(path, arch, prompt):
    f, base, meta, tensors = parse(path)
    arch_p = meta.get("general.architecture", arch)
    E = int(meta[f"{arch_p}.embedding_length"])
    L = int(meta[f"{arch_p}.block_count"])
    H = int(meta[f"{arch_p}.attention.head_count"])
    KH = int(meta[f"{arch_p}.attention.head_count_kv"])
    F = int(meta[f"{arch_p}.feed_forward_length"])
    D = int(meta.get(f"{arch_p}.attention.key_length", E // H))
    base_rope = float(meta.get(f"{arch_p}.rope.freq_base", 10000.0))
    eps = float(meta.get(f"{arch_p}.attention.layer_norm_rms_epsilon", 1e-5))
    rotary = int(meta.get(f"{arch_p}.rope.dimension_count", E // H))
    rotary = min(rotary, D)
    print(f"arch={arch_p} E={E} L={L} H={H} KH={KH} D={D} F={F} rope_base={base_rope} rotary={rotary}")

    def T(name):
        return dequant(f, base, tensors[name])

    emb = T("token_embd.weight")
    x = emb[prompt].copy()  # (T, E)
    Tt = len(prompt)
    positions = np.arange(Tt, dtype=np.int64)
    group = H // KH
    for l in range(L):
        p = f"blk.{l}."
        h = rms_norm(x, T(p + "attn_norm.weight"), eps)
        q = (h @ T(p + "attn_q.weight").T).reshape(Tt, H, D)
        k = (h @ T(p + "attn_k.weight").T).reshape(Tt, KH, D)
        v = (h @ T(p + "attn_v.weight").T).reshape(Tt, KH, D)
        # Qwen2-family models keep attention output biases (qwen3 removed them).
        for nm, arr in (("attn_q.bias", q), ("attn_k.bias", k), ("attn_v.bias", v)):
            if p + nm in tensors:
                b = T(p + nm).reshape(-1)
                heads_n = H if nm[5] == "q" else KH
                arr += b.reshape(heads_n, D)
        if arch == "qwen3":
            q = rms_norm(q, T(p + "attn_q_norm.weight"), eps)
            k = rms_norm(k, T(p + "attn_k_norm.weight"), eps)
        q = rope(q, D, rotary, base_rope, positions)
        k = rope(k, D, rotary, base_rope, positions)
        att = np.zeros((Tt, H, D))
        for t_ in range(Tt):
            for h_ in range(H):
                kh_ = h_ // group
                scores = k[:t_ + 1, kh_] @ q[t_, h_]
                scores = scores / math.sqrt(D)
                scores = np.exp(scores - scores.max())
                scores /= scores.sum()
                att[t_, h_] = scores @ v[:t_ + 1, kh_]
        att = att.reshape(Tt, H * D)
        if arch == "bitnet":
            att = rms_norm(att, T(p + "attn_sub_norm.weight"), eps)
        att = att @ T(p + "attn_output.weight").T
        x = x + att
        h = rms_norm(x, T(p + "ffn_norm.weight"), eps)
        gate = h @ T(p + "ffn_gate.weight").T
        up = h @ T(p + "ffn_up.weight").T
        ffn = gate * (1.0 / (1.0 + np.exp(-gate))) * up
        if arch == "bitnet":
            ffn = rms_norm(ffn, T(p + "ffn_sub_norm.weight"), eps)
        x = x + ffn @ T(p + "ffn_down.weight").T
    x = rms_norm(x, T("output_norm.weight"), eps)
    out_w = T("output.weight") if "output.weight" in tensors else emb
    logits = x[-1] @ out_w.T
    return logits


def main():
    path, arch, dump = sys.argv[1], sys.argv[2], sys.argv[3]
    prompt = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [128000, 791, 6864, 315, 9822, 374]
    print(f"prompt={prompt}")
    ref = forward(path, arch, prompt)
    with open(dump, "rb") as fh:
        actual = np.frombuffer(fh.read(), dtype="<f4").astype(np.float64)
    n = min(len(ref), len(actual))
    ref, actual = ref[:n], actual[:n]
    diff = np.abs(ref - actual)
    scale = np.maximum(1.0, np.abs(ref))
    worst = float((diff / scale).max())
    top_ref = np.argsort(-ref)[:8]
    top_act = np.argsort(-actual)[:8]
    print(f"values={n} max_abs_diff={diff.max():.6g} worst_rel={worst:.3g}")
    print(f"argmax ref={int(top_ref[0])} actual={int(np.argmax(actual))}")
    print(f"top8 overlap={len(set(top_ref) & set(top_act))}/8")
    ok = worst < 1e-3 and int(top_ref[0]) == int(np.argmax(actual))
    print("PARITY:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
