"""Hex + statistical analysis of raw blocks for unknown GGML types.

Usage: python block_forensics.py model.gguf <type_id> [n_blocks]
"""
import struct
import sys
import statistics

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
    for _ in range(n_tensors):
        read_str()
        (nd,) = struct.unpack("<I", f.read(4))
        f.read(8 * nd)
        (t,) = struct.unpack("<I", f.read(4))
        (off,) = struct.unpack("<Q", f.read(8))
        if t == int(sys.argv[2]):
            return f, off
    raise SystemExit(f"no type {sys.argv[2]} tensor found")


def main():
    f, off = read_dir(sys.argv[1])
    # locate end of directory for alignment
    pos = f.tell()
    base = (pos + 31) & ~31
    f.seek(base + off)
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    raw = f.read(n * 32 * 18)  # generous read
    print(f"tensor payload at file offset {base + off}, first {len(raw)} bytes:\n")

    # candidate geometries
    print("--- as (32,18) blocks, first 6 ---")
    for b in range(6):
        blk = raw[b * 18:(b + 1) * 18]
        d = struct.unpack("<e", blk[0:2])[0]
        print(f"  blk{b}: scale_f16={d:.6g} (0x{struct.unpack('<H',blk[0:2])[0]:04x}) nibbles={blk[2:10].hex()}")
    print("\n--- as (32,17) MXFP4 blocks, first 6 ---")
    for b in range(6):
        blk = raw[b * 17:(b + 1) * 17]
        print(f"  blk{b}: e8m0={blk[0]} nibbles={blk[1:9].hex()}")
    print("\n--- as (256,136) IQ4_XS blocks, first 3 ---")
    for b in range(3):
        blk = raw[b * 136:(b + 1) * 136]
        d = struct.unpack("<e", blk[0:2])[0]
        sh = struct.unpack("<H", blk[2:4])[0]
        print(f"  blk{b}: d_f16={d:.6g} scales_h=0x{sh:04x} scales_l={blk[4:8].hex()} qs={blk[8:16].hex()}")

    # statistical discriminator over many blocks
    print("\n--- scale-byte statistics over 2000 blocks ---")
    # Q4_0 hypothesis: bytes [0:2] of every 18 = fp16 scale
    f16s = [struct.unpack("<e", raw[i * 18:i * 18 + 2])[0] for i in range(2000)]
    e8s = [raw[i * 17] for i in range(2000)]
    print(f"  f16@18: min={min(f16s):.4g} max={max(f16s):.4g} "
          f"finite={sum(1 if x==x and abs(x)!=float('inf') else 0 for x in f16s)}/2000")
    print(f"  e8m0@17: min={min(e8s)} max={max(e8s)} mean={statistics.mean(e8s):.1f}")


if __name__ == "__main__":
    main()
