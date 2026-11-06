"""Generate PTQ1_0 golden records using the *actual* PrismML C decoder.

Reference revision: PrismML-Eng/llama.cpp 62c418ccd7aefee37997e94c2d9fac1eed7fc89b,
ggml/src/ggml-quants.c. Supply that file as argv[1], and the output path as
argv[2]. The unmodified upstream decoder function is compiled with its exact
block layout, stage sizes, and a binary16 conversion shim. GCC/Clang with
_Float16 support is needed only to regenerate the fixture, not to run tests.
"""
import pathlib
import random
import struct
import subprocess
import sys
import tempfile

source = pathlib.Path(sys.argv[1]).read_text(encoding='utf-8')
start = source.index('void dequantize_row_ptq1_0(')
end = source.index('\nsize_t quantize_ptq1_0(', start)
function = source[start:end]
shim = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define QK_PTQ1_0 128
#define GGML_RESTRICT restrict
static const size_t ptq1_0_stages[3] = {32, 16, 8};
typedef struct { uint8_t qs[24]; uint8_t qh[2]; uint16_t d; } block_ptq1_0;
static float cvt(uint16_t bits) { _Float16 h; memcpy(&h, &bits, 2); return (float) h; }
#define GGML_FP16_TO_FP32(x) cvt(x)
'''
main = r'''
int main(void) {
    block_ptq1_0 input[2]; float output[256];
    while (fread(input, 1, sizeof(input), stdin) == sizeof(input)) {
        dequantize_row_ptq1_0(input, output, 256);
        if (fwrite(output, sizeof(float), 256, stdout) != 256) return 1;
    }
    return ferror(stdin) ? 1 : 0;
}
'''
rng = random.Random(6102026)
blocks = []
for record in range(32):
    raw = bytearray(rng.randrange(256) for _ in range(56))
    for pos in (26, 54):
        raw[pos:pos+2] = struct.pack('<H', rng.choice([0x0001, 0x3000, 0x3400, 0x3800, 0x3c00, 0xb400, 0xbc00]))
    blocks.append(bytes(raw))
with tempfile.TemporaryDirectory() as temp:
    cfile, exe = pathlib.Path(temp) / 'reference.c', pathlib.Path(temp) / 'reference'
    cfile.write_text(shim + function + main, encoding='utf-8')
    subprocess.run(['cc', '-std=c11', '-O2', str(cfile), '-o', str(exe)], check=True)
    result = subprocess.run([str(exe)], input=b''.join(blocks), stdout=subprocess.PIPE, check=True).stdout
    assert len(result) == len(blocks) * 256 * 4
out = bytearray(struct.pack('<I', len(blocks)))
for index, block in enumerate(blocks):
    out.extend(b'PTQ1_0\0\0')
    out.extend(struct.pack('<I', len(block)))
    out.extend(block)
    out.extend(result[index*1024:(index+1)*1024])
pathlib.Path(sys.argv[2]).write_bytes(out)
print(f'wrote {len(blocks)} upstream Prism PTQ1_0 golden records to {sys.argv[2]}')
