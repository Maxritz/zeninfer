#!/usr/bin/env python3
"""Generate a synthetic Ternary-Bonsai-2-shaped GGUF and its independent reference logits.

This is a *layout and numerics* fixture, not a checkpoint. It reproduces the
contract of the PrismML Ternary-Bonsai 2 packs closely enough to execute the
Qwen3.5-hybrid (gated delta net + full attention) path end to end:

  * general.architecture = qwen35, with the qwen35.* hyperparameter keys
  * PTQ1_0 (ggml type 143) weights: block_ptq1_0 = { qs[24]; qh[2]; f16 d; },
    block-128 base-3 trit packing, one f16 scale per block
  * the prism.hadamard.* contract: version, block_size, transform, axis,
    explicit signs, folded weight names and one inverse-lookup table
  * weights *actually stored in the folded basis* W_f = W*D*H with H*H = I and
    D*D = I, so the runtime must apply H then D (or D then H -- the model can
    tell, because signs and the rotation do not commute) to the matmul input
  * ssm_out folded with prism.hadamard.gdn_v_grouped=true, i.e. its value-head
    feature axis in grouped [hd, rep, nk] order while the recurrence emits
    tiled [hd, nk, rep]

Two files are written: the GGUF, and a .npz holding the independently computed
reference logits for a fixed prompt plus the fixture hyperparameters. The
reference (reference_logits()) is written from the PrismML fork sources
(src/models/qwen35.cpp build_layer_attn_linear/build_layer_attn, and
ggml_compute_forward_gated_delta_net_one_chunk in ggml/src/ggml-cpu/ops.cpp),
not from the C++ runtime.

Usage:
  python3 tests/make_bonsai2_fixture.py OUT.gguf OUT.npz [--seed N]
"""
import argparse
import struct

import numpy as np

# ---------------------------------------------------------------- GGUF writer

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32

# GGUF metadata value types
U32, I32, F32, BOOL, STRING, ARRAY = 4, 5, 6, 7, 8, 9

# GGML tensor types
F32_T, PTQ1_0_T = 0, 143


class GgufWriter:
    def __init__(self):
        self.kv = []       # (key, type, value)
        self.tensors = []  # (name, dims, type, payload)

    def add(self, key, value_type, value):
        self.kv.append((key, value_type, value))

    def tensor(self, name, array, ggml_type=F32_T):
        a = np.ascontiguousarray(array, dtype=np.float32)
        if ggml_type == F32_T:
            payload = a.astype("<f4").tobytes()
        elif ggml_type == PTQ1_0_T:
            payload = encode_ptq1_0(a)
        else:
            raise ValueError(f"unsupported type {ggml_type}")
        # GGUF stores dims in ggml order: ne[0] (the input width) first, so a
        # row-major (rows, cols) array is written as (cols, rows).
        self.tensors.append((name, [a.shape[1], a.shape[0]] if a.ndim == 2 else list(a.shape),
                             ggml_type, payload))

    @staticmethod
    def _put_string(out, s):
        raw = s.encode("utf-8")
        out += struct.pack("<Q", len(raw)) + raw

    def _put_value(self, out, value_type, value):
        if value_type == STRING:
            self._put_string(out, value)
        elif value_type == ARRAY:
            elem_type, items = value
            out += struct.pack("<IQ", elem_type, len(items))
            for item in items:
                self._put_value(out, elem_type, item)
        elif value_type == U32:
            out += struct.pack("<I", value)
        elif value_type == I32:
            out += struct.pack("<i", value)
        elif value_type == F32:
            out += struct.pack("<f", value)
        elif value_type == BOOL:
            out += struct.pack("<B", 1 if value else 0)
        else:
            raise ValueError(value_type)

    def write(self, path):
        # tensor data section: offsets are relative to the aligned data start, and
        # each tensor is aligned to 32 bytes so a single alignment covers all.
        offsets, cursor = [], 0
        for _, _, _, payload in self.tensors:
            offsets.append(cursor)
            cursor += len(payload)
            cursor = (cursor + GGUF_DEFAULT_ALIGNMENT - 1) & ~(GGUF_DEFAULT_ALIGNMENT - 1)

        head = bytearray(GGUF_MAGIC)
        head += struct.pack("<IQQ", GGUF_VERSION, len(self.tensors), len(self.kv))
        for key, value_type, value in self.kv:
            self._put_string(head, key)
            head += struct.pack("<I", value_type)
            self._put_value(head, value_type, value)
        for (name, dims, ggml_type, payload), offset in zip(self.tensors, offsets):
            self._put_string(head, name)
            head += struct.pack("<I", len(dims))
            for dim in dims:
                head += struct.pack("<Q", dim)
            head += struct.pack("<IQ", ggml_type, offset)
        head += b"\0" * (((len(head) + GGUF_DEFAULT_ALIGNMENT - 1) & ~(GGUF_DEFAULT_ALIGNMENT - 1)) - len(head))

        out = bytearray(head)
        for (_, _, _, payload), offset in zip(self.tensors, offsets):
            if len(out) - len(head) < offset:
                out += b"\0" * (offset - (len(out) - len(head)))
            out += payload
        with open(path, "wb") as f:
            f.write(bytes(out))
        return len(head)


# --------------------------------------------------------------- PTQ1_0 codec
# PrismML block_ptq1_0: 24 bytes of 5 trits each (120 values) + 2 bytes of 4
# trits each (8 values) + one fp16 scale. digit q = byte * 3**n truncated to 8
# bits, value = (((q * 3) >> 8) - 1) * d.

def _ptq1_0_digit(byte, n):
    q = (int(byte) * (3 ** n)) & 0xFF
    return ((q * 3) >> 8) - 1          # -1, 0 or +1


def decode_ptq1_0(block):
    d = np.frombuffer(block[26:28], dtype="<f2")[0].astype(np.float32)
    qs, qh = block[0:24], block[24:26]
    out = np.empty(128, dtype=np.float32)
    i = 0
    for n in range(5):
        for byte in qs:
            out[i] = _ptq1_0_digit(byte, n)
            i += 1
    for n in range(4):
        for byte in qh:
            out[i] = _ptq1_0_digit(byte, n)
            i += 1
    return out * d


def encode_ptq1_0(a):
    """Quantise a float array to PTQ1_0 blocks (exact for ternary group-scaled input)."""
    flat = np.ascontiguousarray(a, dtype=np.float32).ravel()
    if flat.size % 128:
        raise ValueError("PTQ1_0 rows must be a multiple of 128 elements")
    out = bytearray()
    for b in range(flat.size // 128):
        blk = flat[b * 128:(b + 1) * 128]
        scale = float(np.max(np.abs(blk)))
        d = np.float16(scale if scale else 1.0)
        levels = np.zeros(128, dtype=np.int8)
        if scale:
            levels = np.rint(blk / float(d)).astype(np.int8)
            if np.max(np.abs(levels)) > 1:
                raise ValueError("value outside PTQ1_0's ternary range for its block scale")
        # invert q = byte*3**n: the encoder picks byte = value * 3**(5-n) for the
        # five trit lanes of a byte and keeps the lane where the product fits a byte.
        qs = bytearray(24)
        for lane in range(5):
            for m in range(24):
                idx = lane * 24 + m
                v = int(levels[idx])
                qs[m] |= _encode_trit_lane(v, lane)
        qh = bytearray(2)
        for lane in range(4):
            for j in range(2):
                idx = 120 + lane * 2 + j
                v = int(levels[idx])
                qh[j] |= _encode_trit_lane(v, lane)
        out += bytes(qs) + bytes(qh) + np.float16(d).tobytes()
    return bytes(out)


_LANE_BYTE = {}


def _encode_trit_lane(value, lane):
    """Inverse of the digit decode for one trit lane: find a byte whose base-3
    digit `lane` is value. The decode is a truncating ladder over 8 bits, so a
    trit 0/1/2 in lane n is produced by the byte constant below."""
    if not _LANE_BYTE:
        for byte in range(256):
            for n in range(5):
                digit = _ptq1_0_digit(byte, n) + 1
                _LANE_BYTE.setdefault((n, digit), byte)
    # a trit is encoded in one lane only: 0/1/2 -> the smallest byte carrying it
    return _LANE_BYTE[(lane, value + 1)]


# ------------------------------------------------------------------ reference

def swht_matrix(n):
    """Normalized Sylvester-Hadamard matrix exactly as ggml_gen_hadamard builds it."""
    h = np.zeros((n, n), dtype=np.float64)
    h[0, 0] = 1.0 / np.sqrt(n)
    s = 1
    while s < n:
        for i in range(s):
            for j in range(s):
                val = h[i, j]
                h[(i + s), j] = val
                h[i, (j + s)] = val
                h[(i + s), (j + s)] = -val
        s *= 2
    return h.astype(np.float32)


def reference_logits(p, w, prompt, trace=None, trace_layer=None):
    """Forward pass written from the PrismML fork's qwen35 graph and GDN CPU op.

    `trace` (a dict) receives the GDN stage tensors of `trace_layer` so a parity
    failure can be localised to conv / qk-norm / gates / recurrence / out-proj."""
    if trace is None:
        trace = {}

    def record(name, value):
        # the C++ dump appends once per layer call, so for a T-token batch the
        # file holds T copies of the same buffer; keep the final one here.
        trace[name] = np.asarray(value, dtype=np.float32).reshape(-1).copy()
    E, H, KH, D, F = p["E"], p["H"], p["KH"], p["D"], p["F"]
    S, NK, NV, CK = p["S"], p["NK"], p["NV"], p["CK"]
    key_dim, value_dim = S * NK, S * NV
    conv_dim, qk_span = 2 * key_dim + value_dim, 2 * key_dim
    eps = p["eps"]
    Hm, signs = p["hadamard_matrix"], p["sign_vector"]
    block = Hm.shape[0]
    use_hadamard = bool(p.get("hadamard", True))


    def rms(x, wgt):
        return x / np.sqrt((x * x).mean() + eps) * wgt

    def fold(x, width):
        if not use_hadamard:
            return np.asarray(x, dtype=np.float32)
        return apply_fold(x, width, block, Hm, signs)

    state = {l: np.zeros((NV, S, S), dtype=np.float32) for l in range(p["L"])}
    conv = {l: np.zeros((CK - 1, conv_dim), dtype=np.float32) for l in range(p["L"])}
    kc = {l: np.zeros(((p["L"], p["ctx"]))[1], dtype=np.float32) for l in range(p["L"])}
    vc = {l: np.zeros(((p["L"], p["ctx"]))[1], dtype=np.float32) for l in range(p["L"])}
    kv_shape = {l: (p["ctx"], KH * D) for l in range(p["L"])}
    kc = {l: np.zeros(kv_shape[l], dtype=np.float32) for l in range(p["L"])}
    vc = {l: np.zeros(kv_shape[l], dtype=np.float32) for l in range(p["L"])}

    logits = None
    for pos, tok in enumerate(prompt):
        x = w["token_embd.weight"][tok].copy()
        if use_hadamard and p["token_embd_inverse"]:
            x = (Hm_block(x, block, Hm) * signs[:len(x)]).astype(np.float32)
        for l in range(p["L"]):
            rec = (l + 1) % p["interval"] != 0
            n = rms(x, w[f"blk.{l}.attn_norm.weight"])
            if rec:
                qkv = fold(n, E) @ w[f"blk.{l}.attn_qkv.weight"].T
                z = fold(n, E) @ w[f"blk.{l}.attn_gate.weight"].T
                beta = fold(n, E) @ w[f"blk.{l}.ssm_beta.weight"].T
                alpha = fold(n, E) @ w[f"blk.{l}.ssm_alpha.weight"].T
                # conv over [state rows | qkv] in channel order (ggml_ssm_conv)
                window = np.concatenate([conv[l], qkv[None, :]], axis=0)  # [CK, conv_dim]
                cout = np.zeros(conv_dim, dtype=np.float32)
                for ch in range(conv_dim):
                    cout[ch] = float(np.dot(window[:, ch], w[f"blk.{l}.ssm_conv1d.weight"][ch]))
                cout = cout / (1.0 + np.exp(-cout))                     # silu
                if trace_layer == l:
                    record("conv_silu", cout)
                q = cout[0:key_dim].reshape(NK, S)
                k = cout[key_dim:qk_span].reshape(NK, S)
                v = cout[qk_span:].reshape(NV, S)
                # joint [S, 2*NK] l2 norm: one norm per key head over q then k
                for h in range(NK):
                    for vec in (q[h], k[h]):
                        vec /= max(np.sqrt(float(np.dot(vec, vec))), eps)
                if trace_layer == l:
                    record("qkv", qkv); record("z", z)
                    record("beta_raw", beta); record("alpha_raw", alpha)
                    record("conv_raw", np.array([np.dot(window[:, ch], w[f"blk.{l}.ssm_conv1d.weight"][ch]) for ch in range(conv_dim)]))
                    record("qk_normed", np.concatenate([q.ravel(), k.ravel()]))
                conv[l] = window[-(CK - 1):]                             # newest last
                # gates
                dt = w[f"blk.{l}.ssm_dt.bias"]
                a_log = w[f"blk.{l}.ssm_a"]
                beta = 1.0 / (1.0 + np.exp(-beta))
                g = (a_log * np.log1p(np.exp(alpha + dt))).astype(np.float32)
                # recurrence, per value head, scale 1/sqrt(S_v)
                out = np.zeros((NV, S), dtype=np.float32)
                scale = 1.0 / np.sqrt(S)
                for h in range(NV):
                    kh = h % NK
                    st = state[l][h]
                    st *= np.float32(np.exp(g[h]))
                    delta = (v[h] - st.T @ k[kh]) * np.float32(beta[h])
                    st += np.outer(k[kh], delta)
                    out[h] = (st.T @ q[kh]) * scale
                if trace_layer == l:
                    record("gdn_state_after", np.stack([state[l][h] for h in range(NV)]))
                    record("gdn_out_raw", out.ravel())
                    record("beta_act", beta.ravel()); record("gate_act", g.ravel())
                # gated norm: rms per head, then silu(normalised) * z
                for h in range(NV):
                    out[h] = rms(out[h], w[f"blk.{l}.ssm_norm.weight"])
                if trace_layer == l:
                    record("gdn_out_normed", out.ravel())
                gl = (out * (z.reshape(NV, S) / (1.0 + np.exp(-z.reshape(NV, S)))))
                flat = gl.reshape(value_dim)
                if p["gdn_v_grouped"]:
                    # tiled [hd, nk, rep] -> grouped [hd, rep, nk]
                    tiled = gl.reshape(NV, S)
                    grouped = np.zeros_like(tiled)
                    for h in range(NV):
                        grouped[(h % NK) * (NV // NK) + h // NK] = tiled[h]
                    flat = grouped.reshape(value_dim)
                if trace_layer == l:
                    record("ssm_out_input", fold(flat, value_dim))
                attn = fold(flat, value_dim) @ w[f"blk.{l}.ssm_out.weight"].T
                if trace_layer == l:
                    record("attn_out", attn)
            else:
                qfull = fold(n, E) @ w[f"blk.{l}.attn_q.weight"].T
                qg = qfull.reshape(H, 2, D)
                q, gate = qg[:, 0, :].copy(), qg[:, 1, :].copy()
                kk = (fold(n, E) @ w[f"blk.{l}.attn_k.weight"].T).reshape(KH, D)
                vv = (fold(n, E) @ w[f"blk.{l}.attn_v.weight"].T).reshape(KH, D)
                for h in range(H):
                    q[h] = rms(q[h], w[f"blk.{l}.attn_q_norm.weight"])
                for h in range(KH):
                    kk[h] = rms(kk[h], w[f"blk.{l}.attn_k_norm.weight"])
                q = rope(q, pos, p)
                kk = rope(kk, pos, p)
                kc[l][pos] = kk.reshape(KH * D)
                vc[l][pos] = vv.reshape(KH * D)
                gsel = H // KH
                att = np.zeros((H, D), dtype=np.float32)
                first = max(0, pos + 1 - p["sliding"]) if p["sliding"] else 0
                for h in range(H):
                    kh = h // gsel
                    keys = kc[l][first:pos + 1].reshape(-1, KH, D)[:, kh, :]
                    vals = vc[l][first:pos + 1].reshape(-1, KH, D)[:, kh, :]
                    sc = (keys @ q[h]) / np.sqrt(D)
                    pr = np.exp(sc - sc.max())
                    pr /= pr.sum()
                    att[h] = pr @ vals
                gsig = 1.0 / (1.0 + np.exp(-gate))
                att = (att * gsig).reshape(H * D)
                attn = fold(att, H * D) @ w[f"blk.{l}.attn_output.weight"].T
            x = x + attn
            res = x.copy()
            n2 = rms(x, w[f"blk.{l}.post_attention_norm.weight"])
            gate_v = fold(n2, E) @ w[f"blk.{l}.ffn_gate.weight"].T
            up_v = fold(n2, E) @ w[f"blk.{l}.ffn_up.weight"].T
            hid = (gate_v / (1.0 + np.exp(-gate_v))) * up_v
            x = res + fold(hid, F) @ w[f"blk.{l}.ffn_down.weight"].T
            if trace_layer == l:
                record("hidden", x)
        norm = rms(x, w["output_norm.weight"])
        logits = fold(norm, E) @ w["output.weight"].T
    return logits.astype(np.float32)


def apply_fold(x, width, block, Hm, signs):
    """H then signs: build_hadamard_input(rot, then ggml_mul(signs))."""
    assert width % block == 0, (width, block)
    y = x.reshape(-1, block) @ Hm.T
    return (y.reshape(width) * signs[:width]).astype(np.float32)


def Hm_block(x, block, Hm):
    return (x.reshape(-1, block) @ Hm.T).reshape(x.shape)


def rope(x, pos, p):
    half = p["rope_dim"] // 2
    i = np.arange(half, dtype=np.float32)
    freq = np.power(np.float32(p["rope_base"]), -2.0 * i / np.float32(p["rope_dim"]))
    ang = pos * freq
    c, s = np.cos(ang), np.sin(ang)
    y = x.copy()
    a, b = y[:, :half].copy(), y[:, half:half * 2].copy()
    y[:, :half] = a * c - b * s
    y[:, half:half * 2] = a * s + b * c
    return y


# ------------------------------------------------------------------- fixture

def make(path_gguf, path_npz, seed=1234, quantised=True, hadamard=True, identity_signs=False,
         layers=None, prompt=None, zero=frozenset()):
    rng = np.random.default_rng(seed)
    p = dict(E=128, H=4, KH=2, D=32, F=256, L=layers or 8, ctx=256, vocab=256,
             S=32, NK=2, NV=4, CK=4, interval=4, eps=1.0e-5,
             rope_dim=32, rope_base=10000.0, sliding=0,
             block=64, gdn_v_grouped=True, token_embd_inverse=True)
    p["n_layer_recurrent"] = sum(1 for l in range(p["L"]) if (l + 1) % p["interval"] != 0)

    key_dim, value_dim = p["S"] * p["NK"], p["S"] * p["NV"]
    conv_dim = 2 * key_dim + value_dim
    E, F = p["E"], p["F"]

    # random signs per input width; identity_signs keeps the rotation but drops D,
    # which separates an unapplied rotation from a misordered one
    widths = [E, conv_dim, value_dim, F]
    if identity_signs:
        signs = np.ones(sum(widths), dtype=np.float32)
    else:
        signs = np.concatenate([rng.choice([-1.0, 1.0], size=w).astype(np.float32) for w in widths])
    Hm = swht_matrix(p["block"])
    assert np.allclose(Hm @ Hm, np.eye(p["block"]), atol=1e-5)   # involutory

    def ternary(rows, cols):
        """group-scaled ternary, i.e. a value PTQ1_0 represents exactly when
        cols % 128 == 0 (its block-128 scale is shared by the whole block)."""
        if cols % 128:
            return (rng.standard_normal((rows, cols)) * 0.2).astype(np.float32)
        blocks = cols // 128
        scale = rng.uniform(0.5, 1.5, size=(rows, blocks)).astype(np.float32)
        signs_ = np.sign(rng.standard_normal((rows, cols))).astype(np.float32)
        return (signs_ * np.repeat(scale, 128, axis=1)).astype(np.float32)

    def folded(w, cols):
        # W_f = W * D * H, per block of the input axis. With hadamard=False the
        # weights stay in the primal basis (an ablation, not a valid checkpoint).
        if not hadamard:
            return np.asarray(w, dtype=np.float32)
        b = w.reshape(w.shape[0], -1, p["block"])
        b = b @ Hm.T
        return (b.reshape(w.shape) * signs[:cols]).astype(np.float32)

    w, roles, folded_names = {}, {}, []
    folded_names.append("output.weight")
    # the table stores latent rows: rotated by H and scaled by the sign vector
    # (inverse_weight_names), restored by the runtime after the row lookup
    emb = ternary(p["vocab"], E)
    w["token_embd.weight"] = (Hm_block(emb.reshape(-1, p["block"]), p["block"], Hm)
                              .reshape(p["vocab"], E) * signs[:E]).astype(np.float32) \
        if hadamard else emb
    w["output.weight"] = folded(ternary(p["vocab"], E), E)
    w["output_norm.weight"] = rng.uniform(0.5, 1.5, E).astype(np.float32)
    # explicit per-tensor spec: (name, rows, cols or 0 for a 1-D vector, folding)
    common = [("attn_norm.weight", 0, E, False), ("post_attention_norm.weight", 0, E, False),
              ("ffn_gate.weight", F, E, True), ("ffn_up.weight", F, E, True),
              ("ffn_down.weight", E, F, True)]
    rec_only = [("attn_qkv.weight", conv_dim, E, True), ("attn_gate.weight", value_dim, E, True),
                ("ssm_beta.weight", p["NV"], E, True), ("ssm_alpha.weight", p["NV"], E, True),
                ("ssm_conv1d.weight", conv_dim, p["CK"], False),
                ("ssm_dt.bias", 0, p["NV"], False), ("ssm_a", 0, p["NV"], False),
                ("ssm_norm.weight", 0, p["S"], False), ("ssm_out.weight", E, value_dim, True)]
    att_only = [("attn_q.weight", 2 * p["H"] * p["D"], E, True), ("attn_k.weight", p["KH"] * p["D"], E, True),
                ("attn_v.weight", p["KH"] * p["D"], E, True), ("attn_output.weight", p["H"] * p["D"], E, True),
                ("attn_q_norm.weight", 0, p["D"], False), ("attn_k_norm.weight", 0, p["D"], False)]
    for l in range(p["L"]):
        rec = (l + 1) % p["interval"] != 0
        for name, rows, cols, is_folded in common + (rec_only if rec else att_only):
            full = f"blk.{l}.{name}"
            if rows == 0:
                # norms, ss m_dt.bias and ssm_a are plain vectors in a stride of "near one"
                lo, hi = (0.5, 1.5) if name.endswith("_norm.weight") else (-1.0, 1.0)
                w[full] = rng.uniform(lo, hi, cols).astype(np.float32)
            elif name == "ssm_conv1d.weight":
                w[full] = (rng.standard_normal((rows, cols)) * 0.3).astype(np.float32)
            elif name in ("ssm_beta.weight", "ssm_alpha.weight"):
                w[full] = (rng.standard_normal((rows, cols)) * 0.05).astype(np.float32)
            else:
                group = ("ffn" if name.startswith("ffn_") else
                         "gdn" if (name.startswith("ssm_") or name.startswith("attn_qkv") or name.startswith("attn_gate")) else
                         "attn")
                mat = np.zeros((rows, cols), dtype=np.float32) if group in zero else ternary(rows, cols)
                w[full] = folded(mat, cols) if is_folded else mat
                if is_folded:
                    folded_names.append(full)

    prompt = list(prompt) if prompt is not None else [7, 42, 99]
    p["hadamard"] = bool(hadamard)
    p["hadamard_matrix"], p["sign_vector"] = Hm, signs
    trace = {}
    trace_layer = next(l for l in range(p["L"]) if (l + 1) % p["interval"] != 0)
    ref = reference_logits(p, w, prompt, trace=trace, trace_layer=trace_layer)

    # ---- write GGUF
    t = GgufWriter()
    t.add("general.architecture", STRING, "qwen35")
    t.add("general.name", STRING, "bonsai2-hybrid-synthetic-fixture")
    t.add("qwen35.embedding_length", U32, E)
    t.add("qwen35.block_count", U32, p["L"])
    t.add("qwen35.attention.head_count", U32, p["H"])
    t.add("qwen35.attention.head_count_kv", U32, p["KH"])
    t.add("qwen35.feed_forward_length", U32, F)
    t.add("qwen35.context_length", U32, p["ctx"])
    t.add("qwen35.attention.key_length", U32, p["D"])
    t.add("qwen35.attention.layer_norm_rms_epsilon", F32, p["eps"])
    t.add("qwen35.rope.dimension_count", U32, p["rope_dim"])
    t.add("qwen35.rope.freq_base", F32, p["rope_base"])
    t.add("qwen35.ssm.conv_kernel", U32, p["CK"])
    t.add("qwen35.ssm.inner_size", U32, p["S"] * p["NV"])
    t.add("qwen35.ssm.state_size", U32, p["S"])
    t.add("qwen35.ssm.time_step_rank", U32, p["NV"])
    t.add("qwen35.ssm.group_count", U32, p["NK"])
    t.add("qwen35.full_attention_interval", U32, p["interval"])
    t.add("tokenizer.ggml.bos_token_id", U32, 1)
    t.add("tokenizer.ggml.eos_token_id", U32, 2)
    t.add("tokenizer.ggml.tokens", ARRAY, (STRING, [f"t{i}" for i in range(p["vocab"])]))
    # prism.hadamard contract (omitted in the no-fold ablation)
    if hadamard:
        t.add("prism.hadamard.version", U32, 1)
        t.add("prism.hadamard.block_size", U32, p["block"])
        t.add("prism.hadamard.transform", STRING, "normalized-sylvester-walsh-hadamard")
        t.add("prism.hadamard.axis", STRING, "input-last-dimension")
        t.add("prism.hadamard.sign_mode", STRING, "explicit")
        t.add("prism.hadamard.gdn_v_grouped", BOOL, p["gdn_v_grouped"])
        t.add("prism.hadamard.sign_widths", ARRAY, (I32, widths))
        t.add("prism.hadamard.sign_values", ARRAY, (I32, [int(v) for v in signs]))
        t.add("prism.hadamard.weight_names", ARRAY, (STRING, sorted(folded_names)))
        t.add("prism.hadamard.inverse_weight_names", ARRAY, (STRING, ["token_embd.weight"]))

    for name, arr in w.items():
        # everything that is a matmul weight and 128-aligned on its input axis
        # is stored as PTQ1_0, matching the real packs' dominant type
        is_matrix = arr.ndim == 2 and arr.shape[1] % 128 == 0 and "_norm" not in name \
            and "ssm_dt" not in name and "ssm_a" not in name and "ssm_conv1d" not in name
        t.tensor(name, arr, PTQ1_0_T if (quantised and is_matrix) else F32_T)
    data_offset = t.write(path_gguf)

    np.savez_compressed(
        path_npz,
        ref_logits=ref, prompt=np.array(prompt, dtype=np.int64),
        **{f"w__{k}": v for k, v in w.items()},
        **{f"p__{k}": np.array(v) for k, v in p.items() if k not in ("Hm", "signs", "role")},
        signs=signs, hadamard=Hm, data_offset=np.array(data_offset),
        trace_layer=np.array(trace_layer),
        **{f"trace__{k}": v for k, v in trace.items()},
    )
    return dict(gguf=path_gguf, npz=path_npz, tensors=len(w), folded=len(folded_names),
                ptq1_0=t.tensors and sum(1 for x in t.tensors if x[2] == PTQ1_0_T),
                vocab=p["vocab"], layers=p["L"], ref_argmax=int(np.argmax(ref)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("npz")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--f32", action="store_true", help="store weights as F32 instead of PTQ1_0")
    ap.add_argument("--no-hadamard", action="store_true", help="ablation: store weights in the primal basis")
    ap.add_argument("--identity-signs", action="store_true", help="ablation: keep the rotation, drop the signs")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--zero", type=str, default="", help="comma list of attn,gdn,ffn to zero out")
    a = ap.parse_args()
    print(make(a.gguf, a.npz, a.seed, quantised=not a.f32, hadamard=not a.no_hadamard,
               identity_signs=a.identity_signs, layers=a.layers,
               prompt=[int(x) for x in a.prompt.split(",")] if a.prompt else None,
               zero=frozenset(x for x in a.zero.split(",") if x)))
