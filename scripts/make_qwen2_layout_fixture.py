#!/usr/bin/env python3
"""Create a deterministic Qwen2-layout GGUF adapter fixture from a dense Tiny-LLM GGUF.

This rewrites only GGUF metadata keys and preserves all tensor bytes. It is an
adapter/layout fixture, NOT a Qwen2-trained checkpoint or production parity claim.
"""
import argparse
import hashlib
import struct
from pathlib import Path

# GGUF metadata scalar type -> struct format.
FMT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
       7: "?", 10: "Q", 11: "q", 12: "d"}

class Reader:
    def __init__(self, data): self.data, self.pos = data, 0
    def take(self, n):
        if n < 0 or self.pos + n > len(self.data): raise ValueError("truncated GGUF")
        b = self.data[self.pos:self.pos+n]; self.pos += n; return b
    def unpack(self, fmt): return struct.unpack("<"+fmt, self.take(struct.calcsize("<"+fmt)))[0]
    def string(self): return self.take(self.unpack("Q")).decode("utf-8")
    def value(self, typ):
        if typ == 8: return self.string()
        if typ == 9:
            elem = self.unpack("I"); count = self.unpack("Q")
            return (elem, [self.value(elem) for _ in range(count)])
        return self.unpack(FMT[typ])

class Writer:
    def __init__(self): self.b = bytearray()
    def put(self, fmt, value): self.b += struct.pack("<"+fmt, value)
    def string(self, value):
        raw = value.encode("utf-8"); self.put("Q", len(raw)); self.b += raw
    def value(self, typ, value):
        if typ == 8: self.string(value)
        elif typ == 9:
            elem, values = value; self.put("I", elem); self.put("Q", len(values))
            for item in values: self.value(elem, item)
        else: self.put(FMT[typ], value)

def convert(source: Path, dest: Path):
    raw = source.read_bytes(); r = Reader(raw)
    if r.take(4) != b"GGUF": raise ValueError("input is not GGUF")
    version, n_tensors, n_meta = r.unpack("I"), r.unpack("Q"), r.unpack("Q")
    if version != 3: raise ValueError(f"expected GGUF v3, got v{version}")
    metadata = []
    source_arch = None
    for _ in range(n_meta):
        key = r.string(); typ = r.unpack("I"); value = r.value(typ)
        if key == "general.architecture": source_arch = value
        if key.startswith("llama."): key = "qwen2." + key[len("llama."):]
        if key == "general.architecture": value = "qwen2"
        metadata.append((key, typ, value))
    if source_arch != "llama": raise ValueError(f"expected a Llama dense source fixture, got {source_arch!r}")
    tensors = []
    for _ in range(n_tensors):
        name = r.string(); nd = r.unpack("I"); dims = [r.unpack("Q") for _ in range(nd)]
        typ, offset = r.unpack("I"), r.unpack("Q")
        tensors.append((name, dims, typ, offset))
    old_data_start = (r.pos + 31) & ~31
    payload = raw[old_data_start:]
    w = Writer(); w.b += b"GGUF"; w.put("I", version); w.put("Q", n_tensors); w.put("Q", len(metadata))
    for key, typ, value in metadata:
        w.string(key); w.put("I", typ); w.value(typ, value)
    for name, dims, typ, offset in tensors:
        w.string(name); w.put("I", len(dims))
        for dim in dims: w.put("Q", dim)
        w.put("I", typ); w.put("Q", offset)
    w.b += b"\0" * (((len(w.b)+31)&~31)-len(w.b))
    new_data_start = len(w.b)
    if new_data_start % 32: raise AssertionError("output tensor data is not 32-byte aligned")
    w.b += payload
    dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(w.b)
    return {"source_bytes": len(raw), "fixture_bytes": len(w.b), "tensors": len(tensors),
            "tensor_data_bytes": len(payload), "metadata_entries": len(metadata),
            "architecture": "qwen2", "tensor_payload_preserved": payload == raw[old_data_start:],
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "fixture_sha256": hashlib.sha256(w.b).hexdigest(),
            "source_data_offset": old_data_start, "fixture_data_offset": new_data_start}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("source", type=Path); ap.add_argument("output", type=Path)
    args = ap.parse_args(); print(convert(args.source, args.output))
