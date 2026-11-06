"""Independent reference for the GGML Q8_K / TQ / IQ decoders.

Parses the grid tables out of src/iq_tables.h (single source of truth), decodes
randomly generated blocks with an independent Python implementation of the
format spec, and writes iq_ref.bin for iq_check.cpp:
    u32 count, then per record: 8-byte name, u32 size, block bytes, 256 f32.
"""
import re
import random
import struct
import sys

# Stdlib RNG keeps this script dependency-free (no numpy), so the differential
# reference can run on any Python. A fixed seed makes the corpus reproducible.
rng = random.Random(20261101)

header = open(sys.argv[1] if len(sys.argv) > 1 else
              r'H:\ZenInfer\ZenInfer-0.4.1-alpha-model-neutral\src\iq_tables.h', encoding='utf-8').read()


def table(name):
    m = re.search(r'constexpr [\w:]+\s+' + name + r'\[(\d+)\]=\{(.*?)\};', header, re.S)
    if not m:
        raise SystemExit(f'table {name} not found in iq_tables.h')
    return [int(v, 16) if v.startswith('0x') else int(v)
            for v in re.findall(r'0x[0-9a-fA-F]+|\d+', m.group(2))]


ksigns = table('ksigns_iq2xs')
kmask = [1, 2, 4, 8, 16, 32, 64, 128]
iq2xxs = table('iq2xxs_grid')
iq2xs = table('iq2xs_grid')
iq2s = table('iq2s_grid')
iq3xxs = table('iq3xxs_grid')
iq3s = table('iq3s_grid')
iq1s = table('iq1s_grid')
IQLUT = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]
DELTA = 0.125


def f16(bits):
    # IEEE binary16 -> float, pure stdlib (numpy-free).
    sign = -1.0 if bits & 0x8000 else 1.0
    exp = (bits >> 10) & 0x1f
    mant = bits & 0x03ff
    if exp == 0:                     # subnormal or zero
        return sign * mant * (2.0 ** -24)
    if exp == 0x1f:                  # inf / nan
        return float('inf') if mant == 0 else float('nan')
    return sign * (1.0 + mant / 1024.0) * (2.0 ** (exp - 15))


def grid_u64(entry):
    return [(entry >> (8 * j)) & 0xff for j in range(8)]


def grid_u32(entry):
    return [(entry >> (8 * j)) & 0xff for j in range(4)]


def q8_k(block):
    d = struct.unpack_from('<f', block, 0)[0]
    return [d * (b - 256 if b > 127 else b) for b in block[4:260]]


def tq2_0(block):
    d = f16(struct.unpack_from('<H', block, 64)[0])
    out = []
    for j in (0, 32):
        for l in range(4):
            for m in range(32):
                out.append((((block[j + m] >> (2 * l)) & 3) - 1) * d)
    return out


def tq1_0(block):
    d = f16(struct.unpack_from('<H', block, 52)[0])
    out = []
    for section, chunk in ((0, block[0:32]), (0, block[32:48]), (1, block[48:52])):
        digits = 5 if section == 0 else 4
        for n in range(digits):
            for byte in chunk:
                q = (byte * (3 ** n)) & 255
                out.append((((q * 3) >> 8) - 1) * d)
    return out


def iq4_xs(block):
    d = f16(struct.unpack_from('<H', block, 0)[0])
    scales_h = struct.unpack_from('<H', block, 2)[0]
    scales_l = block[4:8]
    qs = block[8:136]
    out = [0.0] * 256
    for ib in range(8):
        ls = ((scales_l[ib // 2] >> (4 * (ib % 2))) & 0xf) | (((scales_h >> (2 * ib)) & 3) << 4)
        dl = d * (ls - 32)
        for j in range(16):
            out[ib * 32 + j] = dl * IQLUT[qs[j] & 15]
            out[ib * 32 + 16 + j] = dl * IQLUT[qs[j] >> 4]
        qs = qs[16:]
    return out


def iq2_xxs(block):
    d = f16(struct.unpack_from('<H', block, 0)[0])
    qs = block[2:66]
    out = []
    for ib in range(8):
        aux = struct.unpack_from('<II', qs, 4 * ib)
        aux8 = list(struct.pack('<II', *aux))
        db = d * (0.5 + (aux[1] >> 28)) * 0.25
        for l in range(4):
            grid = grid_u64(iq2xxs[aux8[l]])
            signs = ksigns[(aux[1] >> (7 * l)) & 127]
            for j in range(8):
                out.append(db * grid[j] * (-1.0 if signs & kmask[j] else 1.0))
    return out


def iq2_xs(block):
    d = f16(struct.unpack_from('<H', block, 0)[0])
    qs = struct.unpack_from('<32H', block, 2)
    scales = block[66:74]
    out = []
    for ib in range(8):
        db = [d * (0.5 + (scales[ib] & 0xf)) * 0.25, d * (0.5 + (scales[ib] >> 4)) * 0.25]
        for l in range(4):
            idx = qs[4 * ib + l]
            grid = grid_u64(iq2xs[idx & 511])
            signs = ksigns[idx >> 9]
            for j in range(8):
                out.append(db[l // 2] * grid[j] * (-1.0 if signs & kmask[j] else 1.0))
    return out


def iq2_s(block):
    d = f16(struct.unpack_from('<H', block, 0)[0])
    qs = block[2:66]
    qh = block[66:74]
    scales = block[74:82]
    signs = block[34:66]
    out = []
    for ib in range(8):
        db = [d * (0.5 + (scales[ib] & 0xf)) * 0.25, d * (0.5 + (scales[ib] >> 4)) * 0.25]
        for l in range(4):
            grid = grid_u64(iq2s[qs[l] | ((qh[ib] << (8 - 2 * l)) & 0x300)])
            sg = signs[4 * ib + l]
            for j in range(8):
                out.append(db[l // 2] * grid[j] * (-1.0 if sg & kmask[j] else 1.0))
        qs = qs[4:]
    return out


def iq3_xxs(block):
    d = f16(struct.unpack_from('<H', block, 0)[0])
    qs = block[2:66]
    ssh = block[66:98]
    out = [0.0] * 256
    for ib in range(8):
        aux = struct.unpack_from('<I', ssh, 4 * ib)[0]
        db = d * (0.5 + (aux >> 28)) * 0.5
        for l in range(4):
            signs = ksigns[(aux >> (7 * l)) & 127]
            g1 = grid_u32(iq3xxs[qs[2 * l]])
            g2 = grid_u32(iq3xxs[qs[2 * l + 1]])
            base = ib * 32 + l * 8
            for j in range(4):
                out[base + j] = db * g1[j] * (-1.0 if signs & kmask[j] else 1.0)
                out[base + 4 + j] = db * g2[j] * (-1.0 if signs & kmask[j + 4] else 1.0)
        qs = qs[8:]
    return out


def iq3_s(block):
    d = f16(struct.unpack_from('<H', block, 0)[0])
    qs = block[2:66]
    qh = block[66:74]
    signs = block[74:106]
    scales = block[106:110]
    out = [0.0] * 256
    for ib in range(0, 8, 2):
        db1 = d * (1 + 2 * (scales[ib // 2] & 0xf))
        db2 = d * (1 + 2 * (scales[ib // 2] >> 4))
        for group, (db, qhp) in enumerate(((db1, qh[0]), (db2, qh[1]))):
            for l in range(4):
                g1 = grid_u32(iq3s[qs[2 * l] | ((qhp << (8 - 2 * l)) & 256)])
                g2 = grid_u32(iq3s[qs[2 * l + 1] | ((qhp << (7 - 2 * l)) & 256)])
                sg = signs[l]
                base = ib * 32 + group * 32 + l * 8
                for j in range(4):
                    out[base + j] = db * g1[j] * (-1.0 if sg & kmask[j] else 1.0)
                    out[base + 4 + j] = db * g2[j] * (-1.0 if sg & kmask[j + 4] else 1.0)
            qs = qs[8:]
            signs = signs[4:]
        qh = qh[2:]
    return out


def iq1_s(block):
    d = f16(struct.unpack_from('<H', block, 0)[0])
    qs = block[2:34]
    qh = struct.unpack_from('<8H', block, 34)
    out = []
    for ib in range(8):
        h = qh[ib]
        dl = d * (2 * ((h >> 12) & 7) + 1)
        delta = -DELTA if h & 0x8000 else DELTA
        for l in range(4):
            grid = [(v if v < 128 else v - 256) for v in grid_u64(iq1s[qs[l] | (((h >> (3 * l)) & 7) << 8)])]
            for j in range(8):
                out.append(dl * (grid[j] + delta))
        qs = qs[4:]
    return out


def iq1_m(block):
    qs = list(block[0:32])
    qh = list(block[32:48])          # 16 bytes, consumed two per 32 values
    sc = struct.unpack_from('<4H', block, 48)
    scale = (sc[0] >> 12) | ((sc[1] >> 8) & 0xf0) | ((sc[2] >> 4) & 0xf00) | (sc[3] & 0xf000)
    d = f16(scale)
    out = []
    for ib in range(8):
        s16 = sc[ib // 2]
        dl1 = d * (2 * ((s16 >> (6 * (ib % 2))) & 7) + 1)
        dl2 = d * (2 * ((s16 >> (6 * (ib % 2) + 3)) & 7) + 1)
        h0, h1 = qh[2 * ib], qh[2 * ib + 1]
        idx = [qs[0] | ((h0 << 8) & 0x700), qs[1] | ((h0 << 4) & 0x700),
               qs[2] | ((h1 << 8) & 0x700), qs[3] | ((h1 << 4) & 0x700)]
        dd = [-DELTA if h0 & 8 else DELTA, -DELTA if h0 & 0x80 else DELTA,
              -DELTA if h1 & 8 else DELTA, -DELTA if h1 & 0x80 else DELTA]
        for l in range(2):
            grid = [(v if v < 128 else v - 256) for v in grid_u64(iq1s[idx[l]])]
            for j in range(8):
                out.append(dl1 * (grid[j] + dd[l]))
        for l in range(2, 4):
            grid = [(v if v < 128 else v - 256) for v in grid_u64(iq1s[idx[l]])]
            for j in range(8):
                out.append(dl2 * (grid[j] + dd[l]))
        qs = qs[4:]
    return out


FORMATS = {
    'Q8_K': (292, q8_k), 'TQ1_0': (54, tq1_0), 'TQ2_0': (66, tq2_0),
    'IQ4_XS': (136, iq4_xs), 'IQ2_XXS': (66, iq2_xxs), 'IQ2_XS': (74, iq2_xs),
    'IQ2_S': (82, iq2_s), 'IQ3_XXS': (98, iq3_xxs), 'IQ3_S': (110, iq3_s),
    'IQ1_S': (50, iq1_s), 'IQ1_M': (56, iq1_m),
}


def rand_half():
    bits = rng.randrange(0x0800, 0x3c00)
    return bits | (0x8000 if rng.random() < 0.5 else 0)


def make_block(name):
    size = FORMATS[name][0]
    block = bytearray(rng.randrange(256) for _ in range(size))
    if name == 'Q8_K':
        block[0:4] = struct.pack('<f', rng.uniform(-2, 2))
    else:
        d_off = {'TQ1_0': 52, 'TQ2_0': 64}.get(name, 0)
        block[d_off:d_off + 2] = struct.pack('<H', rand_half())
        if name == 'IQ1_M':  # the assembled super-scale must be a sane f16
            block[48:56] = b''.join(struct.pack('<H', rand_half()) for _ in range(4))
    return bytes(block)


def write(path, per_format=6):
    rows = []
    for name, (size, decode) in FORMATS.items():
        for _ in range(per_format):
            block = make_block(name)
            values = [float(v) for v in decode(block)]
            assert len(values) == 256, (name, len(values))
            rows.append((name, block, values))
    with open(path, 'wb') as fh:
        fh.write(struct.pack('<I', len(rows)))
        for name, block, values in rows:
            fh.write(name.encode().ljust(8, b'\0'))
            fh.write(struct.pack('<I', len(block)))
            fh.write(block)
            fh.write(struct.pack('<%df' % len(values), *values))
    print(f'wrote {len(rows)} records to {path}')


if __name__ == '__main__':
    write(sys.argv[2] if len(sys.argv) > 2 else 'iq_ref.bin')
