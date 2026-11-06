"""Empirically derive the block layout of unknown GGML type IDs from real files.

Tensor offsets in the GGUF directory are cumulative. For consecutive tensors
(sorted by offset) the byte size of tensor i is offset[i+1] - offset[i].
Combined with the known element count, we can solve for (block, bytes) pairs
and check consistency across every tensor of that type in the file.

Usage: python derive_unknown_types.py model.gguf [model2.gguf ...]
"""
import struct
import sys
import math
from collections import defaultdict

META_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def read_dir(path):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF"
    version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))

    def read_str():
        (n,) = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", "replace")

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

    import os
    file_size = os.path.getsize(path)
    tensors = []
    for _ in range(n_tensors):
        name = read_str()
        (nd,) = struct.unpack("<I", f.read(4))
        dims = struct.unpack("<" + "Q" * nd, f.read(8 * nd))
        (ggml_type,) = struct.unpack("<I", f.read(4))
        (offset,) = struct.unpack("<Q", f.read(8))
        elements = 1
        for d in dims:
            elements *= d
        tensors.append((name, ggml_type, elements, offset))
    tensors.sort(key=lambda t: t[3])
    # derive byte size of each tensor from the next offset (or file tail)
    derived = []
    for i, (name, t, el, off) in enumerate(tensors):
        end = tensors[i + 1][3] if i + 1 < len(tensors) else file_size
        derived.append((name, t, el, off, end - off))
    return derived


def fit(tensors, type_id):
    """Find all (block, bytes) with block<=256 dividing elements consistently."""
    rows = [t for t in tensors if t[1] == type_id]
    if not rows:
        return []
    candidates = []
    for block in range(1, 257):
        if any(el % block for _, _, el, _, _ in rows):
            continue
        ratios = set()
        ok = True
        for _, _, el, _, nbytes in rows:
            blocks = el // block
            if nbytes % blocks:
                ok = False
                break
            ratios.add(nbytes // blocks)
            if len(ratios) > 1:
                ok = False
                break
        if ok:
            candidates.append((block, ratios.pop()))
    return candidates, len(rows), sum(r[4] for r in rows)


def main(paths):
    all_t = []
    for p in paths:
        all_t.extend(read_dir(p))
    unknown = sorted({t[1] for t in all_t if t[1] > 43 or t[1] in (39, 40)})
    for t in unknown:
        rows = [x for x in all_t if x[1] == t]
        print(f"\n=== type {t}: {len(rows)} tensors, {sum(r[4] for r in rows)/1e9:.3f} GB ===")
        for name, _, el, _, nbytes in rows[:4]:
            bpe = nbytes / el
            print(f"  {name}: elements={el} bytes={nbytes} bytes/element={bpe:.6f}")
        result = fit(all_t, t)
        if isinstance(result, tuple):
            cands, n, total = result
            print(f"  consistent (block,bytes) candidates: {cands}")
            print(f"  -> bits per weight: " + ", ".join(
                f"{8*b/el:.3f}" for b, el in cands for el in [b]) if cands else "")
            for block, byt in cands:
                print(f"     block={block} bytes={byt} -> {8.0*byt/block:.4f} bpw")
        names = defaultdict(int)
        for name, _, _, _, _ in rows:
            parts = name.split(".")
            names[".".join(parts[1:])] += 1
        print("  tensor roles:", dict(sorted(names.items(), key=lambda kv: -kv[1])[:6]))


if __name__ == "__main__":
    main(sys.argv[1:])
