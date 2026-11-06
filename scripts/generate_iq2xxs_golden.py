"""Generate IQ2_XXS records with the actual upstream decoder and grid tables.

Inputs: PrismML-Eng/llama.cpp revision 62c418ccd7aefee37997e94c2d9fac1eed7fc89b
files ggml/src/ggml-quants.c and ggml/src/ggml-common.h; then output path.
GCC/Clang with _Float16 is only needed for regeneration.
"""
import pathlib
import random
import re
import struct
import subprocess
import sys
import tempfile

source = pathlib.Path(sys.argv[1]).read_text(encoding='utf-8')
common = pathlib.Path(sys.argv[2]).read_text(encoding='utf-8')
start = source.index('void dequantize_row_iq2_xxs(')
end = source.index('\n// ====================== 2.3125', start)
function = source[start:end]
tables = ''
for kind, name, count in [('uint8_t', 'kmask_iq2xs', 8), ('uint8_t', 'ksigns_iq2xs', 128), ('uint64_t', 'iq2xxs_grid', 256)]:
    pattern = r'GGML_TABLE_BEGIN\(' + kind + r',\s*' + name + r',\s*' + str(count) + r'\)(.*?)GGML_TABLE_END\(\)'
    match = re.search(pattern, common, re.S)
    if not match:
        raise SystemExit(f'missing upstream table {name}')
    tables += f'static const {kind} {name}[{count}] = {{{match.group(1)}}};\n'
shim = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define QK_K 256
#define GGML_RESTRICT restrict
typedef struct { uint16_t d; uint16_t qs[32]; } block_iq2_xxs;
static float cvt(uint16_t bits) { _Float16 h; memcpy(&h, &bits, 2); return (float) h; }
#define GGML_FP16_TO_FP32(x) cvt(x)
'''
main = r'''
int main(void) {
    block_iq2_xxs input; float output[256];
    while (fread(&input, 1, sizeof(input), stdin) == sizeof(input)) {
        dequantize_row_iq2_xxs(&input, output, 256);
        if (fwrite(output, sizeof(float), 256, stdout) != 256) return 1;
    }
    return ferror(stdin) ? 1 : 0;
}
'''
rng = random.Random(6102027)
blocks = []
for _ in range(32):
    raw = bytearray(rng.randrange(256) for _ in range(66))
    raw[:2] = struct.pack('<H', rng.choice([0x0001, 0x3000, 0x3400, 0x3800, 0x3c00, 0xb400, 0xbc00]))
    blocks.append(bytes(raw))
with tempfile.TemporaryDirectory() as temp:
    cfile, exe = pathlib.Path(temp) / 'reference.c', pathlib.Path(temp) / 'reference'
    cfile.write_text(shim + tables + function + main, encoding='utf-8')
    subprocess.run(['cc', '-std=c11', '-O2', str(cfile), '-o', str(exe)], check=True)
    result = subprocess.run([str(exe)], input=b''.join(blocks), stdout=subprocess.PIPE, check=True).stdout
    assert len(result) == len(blocks) * 256 * 4
out = bytearray(struct.pack('<I', len(blocks)))
for index, block in enumerate(blocks):
    out.extend(b'IQ2_XXS\0')
    out.extend(struct.pack('<I', len(block)))
    out.extend(block)
    out.extend(result[index*1024:(index+1)*1024])
pathlib.Path(sys.argv[3]).write_bytes(out)
print(f'wrote {len(blocks)} upstream IQ2_XXS golden records to {sys.argv[3]}')
