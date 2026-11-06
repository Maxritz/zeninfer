"""Scan a GGUF file's tensor directory and count tensors per GGML type.

Usage: python scan_gguf_types.py model.gguf
"""
import struct
import sys
import collections

TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
    20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
    26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0",
    35: "TQ2_0", 39: "MXFP4", 40: "NVFP4", 41: "Q1_0", 42: "Q2_0", 43: "F8_E4M3",
}

TYPE_SIZES = {  # (block_size, bytes_per_block); None = rowwise/other
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24),
    8: (32, 34), 9: (32, 40), 10: (256, 84), 11: (256, 110), 12: (256, 144),
    13: (256, 176), 14: (256, 210), 15: (256, 292), 16: (256, 66), 17: (256, 74),
    18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110), 22: (256, 82),
    23: (256, 136), 24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8),
    29: (256, 56), 30: (1, 2), 34: (256, 54), 35: (256, 66), 39: (32, 17),
    40: (64, 36),
}

META_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def main(path):
    f = open(path, "rb")
    magic = f.read(4)
    assert magic == b"GGUF", "not a GGUF file"
    version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
    print(f"version={version} tensors={n_tensors} kv={n_kv}")

    def read_str():
        (n,) = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", "replace")

    # skip metadata
    for _ in range(n_kv):
        read_str()
        (t,) = struct.unpack("<I", f.read(4))
        if t == 8:  # string
            read_str()
        elif t == 9:  # array
            (sub,) = struct.unpack("<I", f.read(4))
            (count,) = struct.unpack("<Q", f.read(8))
            for _ in range(count):
                if sub == 8:
                    read_str()
                else:
                    f.read(META_SIZES[sub])
        else:
            f.read(META_SIZES[t])

    counts = collections.Counter()
    bytes_by_type = collections.Counter()
    names_by_type = {}
    for _ in range(n_tensors):
        name = read_str()
        (nd,) = struct.unpack("<I", f.read(4))
        dims = struct.unpack("<" + "Q" * nd, f.read(8 * nd))
        (ggml_type,) = struct.unpack("<I", f.read(4))
        (offset,) = struct.unpack("<Q", f.read(8))
        elements = 1
        for d in dims:
            elements *= d
        counts[ggml_type] += 1
        spec = TYPE_SIZES.get(ggml_type)
        nbytes = (elements // spec[0]) * spec[1] if spec and elements % spec[0] == 0 else None
        bytes_by_type[ggml_type] += nbytes or 0
        names_by_type.setdefault(ggml_type, []).append(name)

    total = sum(bytes_by_type.values())
    for t in sorted(counts, key=lambda x: -bytes_by_type[x]):
        label = TYPE_NAMES.get(t, f"TYPE_{t}")
        gb = bytes_by_type[t] / 1e9
        print(f"  {label:<10} x{counts[t]:<5} {gb:8.3f} GB")
        if t == 39 or t > 43:  # MXFP4 and unknown types: sample tensor names
            for n in names_by_type[t][:5]:
                print(f"      e.g. {n}")
    print(f"  total accounted: {total/1e9:.3f} GB")


if __name__ == "__main__":
    main(sys.argv[1])
