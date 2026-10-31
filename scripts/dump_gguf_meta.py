"""Dump all GGUF metadata key/values from a model file.

Usage: python dump_gguf_meta.py model.gguf [max_array_items]
"""
import struct
import sys

TYPE_NAMES = {0: "U8", 1: "I8", 2: "U16", 3: "I16", 4: "U32", 5: "I32",
              6: "F32", 7: "BOOL", 8: "STRING", 9: "ARRAY", 10: "U64",
              11: "I64", 12: "F64"}
META_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def main(path, max_items=8):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF"
    version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
    print(f"version={version} tensors={n_tensors} kv={n_kv}")

    def read_str():
        (n,) = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", "replace")

    def read_val(t):
        if t == 8:
            return read_str()
        if t == 9:
            (sub,) = struct.unpack("<I", f.read(4))
            (count,) = struct.unpack("<Q", f.read(8))
            items = []
            for i in range(count):
                if i >= max_items:
                    # fast-skip remaining scalars/strings
                    for _ in range(count - i):
                        if sub == 8:
                            read_str()
                        else:
                            f.read(META_SIZES[sub])
                    break
                items.append(read_val(sub))
            head = ", ".join(repr(x) if isinstance(x, str) else str(x) for x in items)
            return f"[{sub}:{count}] {{{head}{', ...' if count > max_items else ''}}}"
        raw = f.read(META_SIZES[t])
        if t == 6:
            return struct.unpack("<f", raw)[0]
        if t == 12:
            return struct.unpack("<d", raw)[0]
        if t in (10, 11, 12):
            return struct.unpack("<q" if t == 11 else "<Q", raw)[0]
        if t in (2, 3):
            return struct.unpack("<h" if t == 3 else "<H", raw)[0]
        return int.from_bytes(raw, "little", signed=t in (1, 3, 5))

    for _ in range(n_kv):
        key = read_str()
        (t,) = struct.unpack("<I", f.read(4))
        print(f"  {key} ({TYPE_NAMES[t]}) = {read_val(t)}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8)
