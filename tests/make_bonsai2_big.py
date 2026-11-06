#!/usr/bin/env python3
"""Build a FULL-SIZE synthetic qwen35-hybrid (Ternary-Bonsai-2 shaped) GGUF.

tests/make_bonsai2_fixture.py is deliberately tiny so it can also produce a
NumPy reference in a test budget. This script exists for the other job: produce
a real-sized checkpoint (~50 MB of PTQ1_0 weights, ~230M parameters) to
exercise the loader, the block decoder, cache sizing and the forward pass at a
scale where allocation bugs actually show up.

It streams: no tensor is ever held at full f32 width, and the GGUF data section
is written strictly forward, so peak RSS stays in the tens of MB regardless of
how many parameters the file has.

    python3 tests/make_bonsai2_big.py /tmp/big.gguf

The output is NOT a usable model (random ternary weights) and is NOT a parity
fixture: there is no reference to compare against. It answers "does a
checkpoint this size load and run", nothing more.

Two ranges are not arbitrary -- both were load-time or (worse) first-decode
failures before being pinned down:

  * ssm_a is -exp(A_log) and must stay NEGATIVE. A positive entry makes the
    GDN gate exp(a * softplus(...)) grow, so prefill of a single token passes
    and the first decode step overflows to inf, where tensor_matvec reports
    it as a generic "Decode forward pass failed".
  * ssm_beta/ssm_alpha/ssm_conv1d are NOT Hadamard-folded. prism.hadamard
    only accepts weights sitting on a Hadamard-aware matmul, and the loader
    rejects the whole file otherwise.

See tests/check_bonsai2_big.py for the end-to-end exercise of the result.
"""
import argparse
import struct
import sys

import numpy as np

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32
F32_T = 0
PTQ1_0_T = 143

(U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64) = (
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)


def put_string(out, s):
    raw = s.encode("utf-8")
    out += struct.pack("<Q", len(raw)) + raw


def put_value(out, value_type, value):
    if value_type == STRING:
        put_string(out, value)
    elif value_type == ARRAY:
        elem_type, items = value
        out += struct.pack("<IQ", elem_type, len(items))
        for item in items:
            put_value(out, elem_type, item)
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


def swht_matrix(n):
    """Normalized Sylvester-Hadamard, matching swht_block in the runtime."""
    h = np.zeros((n, n), dtype=np.float32)
    h[0, 0] = 1.0 / np.sqrt(n)
    k = 1
    while k < n:
        for i in range(k):
            for j in range(k):
                h[i + k, j] = h[i, j]
                h[i, j + k] = h[i, j]
                h[i + k, j + k] = -h[i, j]
        k *= 2
    return h


def block_ternary(rng, rows, cols, block=128):
    """Group-scaled ternary: exactly what PTQ1_0 can carry."""
    nblk = cols // block
    scale = rng.uniform(0.5, 1.5, size=(rows, nblk)).astype(np.float16).astype(np.float32)
    vals = rng.integers(-1, 2, size=(rows, cols), dtype=np.int8).astype(np.float32)
    return (vals * np.repeat(scale, block, axis=1)).astype(np.float32)


def digit(byte, n):
    q = (int(byte) * (3 ** n)) & 0xFF
    return ((q * 3) >> 8) - 1


def build_trit_table():
    table = {}
    for byte in range(256):
        key = tuple(digit(byte, n) for n in range(5))
        table.setdefault(key, byte)
    return table


def pack(trits, table):
    return table[tuple(int(t) for t in trits) + (0,) * (5 - len(trits))]


def make_codes():
    """index -> byte lookup, where index = sum_k (trit_k + 1) * 3**k over k=0..4."""
    codes = np.zeros(243, dtype=np.uint8)
    seen = np.zeros(243, dtype=bool)
    for byte in range(256):
        pat = [digit(byte, n) + 1 for n in range(5)]
        idx = sum(pat[k] * (3 ** k) for k in range(5))
        if not seen[idx]:
            seen[idx] = True
            codes[idx] = byte
    if not seen.all():
        raise AssertionError("PTQ1_0 trit patterns are not fully representable")
    return codes


CODES = make_codes()
WEIGHTS = (3 ** np.arange(5, dtype=np.int64))


def encode_rows(wd, table=None):
    """Vectorised PTQ1_0 encoding of an [n, cols] f32 matrix (cols % 128 == 0).

    Layout per 28-byte block: 24 qs bytes carry 120 values (byte m encodes
    levels m, 24+m, 48+m, 72+m, 96+m), 2 qh bytes carry the last 8
    (byte j encodes levels 120+j, 122+j, 124+j, 126+j), then one f16 scale.
    """
    wd = np.asarray(wd, dtype=np.float32)
    n, cols = wd.shape
    if cols % 128:
        raise ValueError("PTQ1_0 rows must be a multiple of 128")
    nb = cols // 128
    g = wd.reshape(n, nb, 128)
    scale = np.abs(g).max(axis=2)                              # [n, nb]
    d = scale.astype(np.float16)
    d[d == 0] = np.float16(1.0)
    level = np.rint(g / d.astype(np.float32)[:, :, None])
    level = np.clip(level, -1, 1).astype(np.int8).astype(np.int64) + 1   # 0,1,2

    out = np.empty((n, nb, 28), dtype=np.uint8)
    qs = level[:, :, :120].reshape(n, nb, 5, 24)               # [n, nb, k, m]
    out[:, :, :24] = CODES[np.tensordot(qs, WEIGHTS, axes=([2], [0]))].transpose(0, 1, 2)
    qh = level[:, :, 120:128].reshape(n, nb, 4, 2)             # [n, nb, k, j]
    out[:, :, 24:26] = CODES[np.tensordot(qh, WEIGHTS[:4], axes=([2], [0]))]
    # each f16 scale is 2 little-endian bytes, broadcast over the last axis
    out[:, :, 26:28] = d.view(np.uint16).astype("<u2").view(np.uint8).reshape(n, nb, 2)
    return out.tobytes()


def self_check():
    """The vectorised encoder must be a bit-exact inverse of the reference decode."""
    rng = np.random.default_rng(0)
    w = block_ternary(rng, 7, 256)
    enc = encode_rows(w)
    lines = []
    for b in range(len(enc) // 28):
        blk = enc[b * 28:(b + 1) * 28]
        dv = np.frombuffer(blk[26:28], dtype="<f2")[0].astype(np.float32)
        vals = []
        for n2 in range(5):
            for byte in blk[0:24]:
                vals.append(digit(byte, n2))
        for n2 in range(4):
            for byte in blk[24:26]:
                vals.append(digit(byte, n2))
        lines.append(np.array(vals, dtype=np.float32) * dv)
    back = np.concatenate(lines)
    assert np.array_equal(back, w.ravel()), "encoder is not a bit-exact inverse"
    print("encoder self-check: bit-exact", file=sys.stderr)


def sample_vector(name, n, rng):
    """Vectors in the ranges the real checkpoint uses, not N(0,1).

    ssm_a is -exp(A_log): a *positive* value here turns the GDN gate into
    exp(+x) and the recurrence diverges after one or two decode steps, so it
    must stay strictly negative. dt_bias is a small log-spaced bias, and the
    norm weights sit near one.
    """
    if name.endswith("ssm_a"):
        return rng.uniform(-3.0, -0.25, n)
    if name.endswith("ssm_dt.bias"):
        return rng.uniform(-1.0, 1.0, n)
    if name.endswith("_norm.weight"):
        return rng.uniform(0.5, 1.5, n)
    return rng.uniform(-1.0, 1.0, n)


def is_ptq1_0(rows, cols, vec):
    """PTQ1_0 needs whole 128-element groups and is meant for real matrices.

    Tiny projections (ssm_alpha, ssm_beta at NV wide) and the conv kernel have
    fewer than 128 columns, so they stay F32 -- same rule the small fixture uses
    for any non-matrix tensor.
    """
    return (not vec) and cols % 128 == 0 and rows * cols >= 128


def payload_bytes(rows, cols, vec):
    if vec:
        return vec * 4
    return rows * cols // 128 * 28 if is_ptq1_0(rows, cols, vec) else rows * cols * 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--target-mb", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--E", type=int, default=1024)
    ap.add_argument("--F", type=int, default=2816)
    ap.add_argument("--layers", type=int, default=20)
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--S", type=int, default=64)
    ap.add_argument("--NK", type=int, default=4)
    ap.add_argument("--NV", type=int, default=8)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--KH", type=int, default=4)
    ap.add_argument("--D", type=int, default=64)
    a = ap.parse_args()

    E, F, L, vocab = a.E, a.F, a.layers, a.vocab
    S, NK, NV, H, KH, D = a.S, a.NK, a.NV, a.H, a.KH, a.D
    block, interval, CK = 128, 4, 4
    key_dim, value_dim = S * NK, S * NV
    conv_dim = 2 * key_dim + value_dim
    QW, KW = H * D, KH * D

    self_check()
    rng = np.random.default_rng(a.seed)
    Hm = swht_matrix(block)
    widths = list(dict.fromkeys([E, conv_dim, value_dim, F]))
    signs_by_width = {w: rng.choice([-1.0, 1.0], size=w).astype(np.float32) for w in widths}
    all_signs = np.concatenate([signs_by_width[w] for w in widths])

    # ---- tensor specs: (name, rows, cols, folded, vec_len)
    specs = [("token_embd.weight", vocab, E, False, 0),
             ("output.weight", vocab, E, True, 0),
             ("output_norm.weight", 0, E, False, E)]
    for l in range(L):
        rec = (l + 1) % interval != 0
        base = [("attn_norm.weight", 0, E, False, E),
                ("post_attention_norm.weight", 0, E, False, E),
                ("ffn_gate.weight", F, E, True, 0),
                ("ffn_up.weight", F, E, True, 0),
                ("ffn_down.weight", E, F, True, 0)]
        if rec:
            base += [("attn_qkv.weight", conv_dim, E, True, 0),
                     ("attn_gate.weight", value_dim, E, True, 0),
                     # NOT folded: prism.hadamard only accepts weights that sit on a
                     # Hadamard-aware matmul, and the loader rejects these (correctly).
                     ("ssm_beta.weight", NV, E, False, 0),
                     ("ssm_alpha.weight", NV, E, False, 0),
                     ("ssm_conv1d.weight", conv_dim, CK, False, 0),
                     ("ssm_dt.bias", 0, NV, False, NV),
                     ("ssm_a", 0, NV, False, NV),
                     ("ssm_norm.weight", 0, S, False, S),
                     ("ssm_out.weight", E, value_dim, True, 0)]
        else:
            base += [("attn_q.weight", 2 * QW, E, True, 0),
                     ("attn_k.weight", KW, E, True, 0),
                     ("attn_v.weight", KW, E, True, 0),
                     ("attn_output.weight", QW, E, True, 0),
                     ("attn_q_norm.weight", 0, D, False, D),
                     ("attn_k_norm.weight", 0, D, False, D)]
        for name, rows, cols, folded, vec in base:
            specs.append((f"blk.{l}.{name}", rows, cols, folded, vec))

    total = sum(payload_bytes(r, c, v) for _, r, c, _, v in specs)
    nparams = sum(r * c if not v else v for _, r, c, _, v in specs)
    print(f"weights: {nparams/1e6:.1f}M  PTQ1_0 data: {total/1e6:.2f} MB", file=sys.stderr)
    if abs(total / 1e6 - a.target_mb) > 0.02 * a.target_mb:
        print(f"note: target {a.target_mb} MB, built {total/1e6:.2f} MB", file=sys.stderr)

    # ---- metadata
    meta = [
        ("general.architecture", STRING, "qwen35"),
        ("qwen35.embedding_length", U32, E),
        ("qwen35.feed_forward_length", U32, F),
        ("qwen35.block_count", U32, L),
        ("qwen35.attention.head_count", U32, H),
        ("qwen35.attention.head_count_kv", U32, KH),
        ("qwen35.attention.key_length", U32, D),
        ("qwen35.attention.value_length", U32, D),
        ("qwen35.attention.layer_norm_rms_epsilon", F32, 1.0e-5),
        ("qwen35.rope.dimension_count", U32, D),
        ("qwen35.rope.freq_base", F32, 10000.0),
        ("qwen35.ssm.conv_kernel", U32, CK),
        ("qwen35.ssm.inner_size", U32, value_dim),
        ("qwen35.ssm.state_size", U32, S),
        ("qwen35.ssm.group_count", U32, NK),
        ("qwen35.ssm.time_step_rank", U32, NV),
        ("qwen35.ssm.v_heads", U32, NV),
        ("qwen35.full_attention_interval", U32, interval),
        ("qwen35.context_length", U32, 512),
        ("tokenizer.ggml.model", STRING, "gpt2"),
        ("general.name", STRING, "bonsai2-synthetic-big"),
        ("tokenizer.ggml.tokens", ARRAY,
         (STRING, [f"t{i}" for i in range(vocab)])),
        ("tokenizer.ggml.scores", ARRAY, (F32, [0.0] * vocab)),
        ("tokenizer.ggml.token_type", ARRAY, (I32, [1] * vocab)),
        ("tokenizer.ggml.bos_token_id", U32, 1),
        ("tokenizer.ggml.eos_token_id", U32, 2),
        ("prism.hadamard.version", U32, 1),
        ("prism.hadamard.block_size", U32, block),
        ("prism.hadamard.transform", STRING, "normalized-sylvester-walsh-hadamard"),
        ("prism.hadamard.axis", STRING, "input-last-dimension"),
        ("prism.hadamard.sign_mode", STRING, "explicit"),
        ("prism.hadamard.sign_widths", ARRAY, (I32, [int(w) for w in widths])),
        ("prism.hadamard.sign_values", ARRAY, (I32, [int(v) for v in all_signs])),
        ("prism.hadamard.gdn_v_grouped", BOOL, 1),
        ("prism.hadamard.weight_names", ARRAY,
         (STRING, [s[0] for s in specs if s[3]])),
        ("prism.hadamard.inverse_weight_names", ARRAY, (STRING, ["token_embd.weight"])),
    ]
    head = bytearray(GGUF_MAGIC)
    head += struct.pack("<IQQ", GGUF_VERSION, len(specs), len(meta))
    for key, vt, val in meta:
        put_string(head, key)
        head += struct.pack("<I", vt)
        put_value(head, vt, val)

    offsets, cursor = [], 0
    for name, rows, cols, folded, vec in specs:
        offsets.append(cursor)
        cursor += payload_bytes(rows, cols, vec)
        cursor = (cursor + GGUF_DEFAULT_ALIGNMENT - 1) & ~(GGUF_DEFAULT_ALIGNMENT - 1)
    for (name, rows, cols, folded, vec), off in zip(specs, offsets):
        put_string(head, name)
        dims = [vec] if vec else [cols, rows]
        head += struct.pack("<I", len(dims))
        for dim in dims:
            head += struct.pack("<Q", dim)
        head += struct.pack("<IQ", PTQ1_0_T if is_ptq1_0(rows, cols, vec) else F32_T, off)
    pad = ((len(head) + GGUF_DEFAULT_ALIGNMENT - 1) & ~(GGUF_DEFAULT_ALIGNMENT - 1)) - len(head)
    head += b"\0" * pad

    written = 0
    with open(a.gguf, "wb") as f:
        f.write(head)
        for idx, (name, rows, cols, folded, vec) in enumerate(specs):
            if vec:
                data = sample_vector(name, vec, rng).astype("<f4").tobytes()
            elif not is_ptq1_0(rows, cols, vec):
                data = (rng.standard_normal((rows, cols)) * 0.3).astype("<f4").tobytes()
            else:
                chunks = []
                for r0 in range(0, rows, 64):
                    n = min(64, rows - r0)
                    wsub = block_ternary(rng, n, cols)
                    if folded:
                        # W_f = (W * S) * H, then re-encode to trits: the fold is
                        # dense, so this is a lossy-but-honest on-disk weight.
                        s = signs_by_width[cols]
                        wd = np.ascontiguousarray(wsub * s, dtype=np.float32)
                        wd = (wd.reshape(n, -1, block) @ Hm.T).reshape(n, cols)
                    else:
                        # latent embedding: store z = H*S*e, the inverse of the
                        # runtime's S*H restore
                        wd = np.ascontiguousarray(wsub * signs_by_width[cols], dtype=np.float32)
                        wd = (wd.reshape(n, -1, block) @ Hm.T).reshape(n, cols)
                    chunks.append(encode_rows(np.asarray(wd, dtype=np.float32)))
                data = b"".join(chunks)
            f.write(data)
            written += len(data)
            padn = (GGUF_DEFAULT_ALIGNMENT - len(data) % GGUF_DEFAULT_ALIGNMENT) % GGUF_DEFAULT_ALIGNMENT
            f.write(b"\0" * padn)
            if idx % 25 == 0:
                print(f"  {idx}/{len(specs)} {name:34s} {len(data)/1e6:6.2f} MB",
                      file=sys.stderr)
    print(f"wrote {a.gguf}: {written/1e6:.2f} MB of weights, {nparams/1e6:.1f}M params",
          file=sys.stderr)


if __name__ == "__main__":
    main()
