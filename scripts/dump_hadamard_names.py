"""Dump the prism.hadamard.weight_names grouping from a Bonsai GGUF."""
import struct
import sys
from collections import Counter

META_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def main(path):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF"
    version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))

    def rs():
        (n,) = struct.unpack("<Q", f.read(8))
        return f.read(n)

    def rs_str():
        return rs().decode("utf-8", "replace")

    names = None
    for _ in range(n_kv):
        key = rs_str()
        (t,) = struct.unpack("<I", f.read(4))
        if key == "prism.hadamard.weight_names" and t == 9:
            (sub,) = struct.unpack("<I", f.read(4))
            (count,) = struct.unpack("<Q", f.read(8))
            assert sub == 8
            names = [rs_str() for _ in range(count)]
        elif t == 8:
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
    if names is None:
        raise SystemExit("no hadamard names")
    c = Counter(n.split(".", 2)[2] if n.startswith("blk.") else n for n in names)
    print(dict(c))
    print("attn_output folded on layers:", sorted({n.split(".")[1] for n in names if ".attn_output." in n}))
    print("attn_q folded on layers:", sorted({n.split(".")[1] for n in names if n.endswith(".attn_q.weight")}))
    print("total", len(names))


if __name__ == "__main__":
    main(sys.argv[1])
