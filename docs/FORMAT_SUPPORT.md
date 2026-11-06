# ZenInfer CPU GGML/GGUF format support

This inventory is against upstream llama.cpp GGML commit
[`240d68fcfe0c2a1f298c8e18ca181368b1914259`](https://github.com/ggml-org/llama.cpp/blob/240d68fcfe0c2a1f298c8e18ca181368b1914259/ggml/include/ggml.h).
All its currently defined tensor types below have CPU FP32 decoding, GGUF
size validation and dense mapped-weight GEMV/GEMM dispatch. Q8_1 GGUF tensors
require explicit layout selection as described below. This is **format
coverage**, not a claim that every architecture or checkpoint using those types
is supported, or that every path has a specialized SIMD kernel.

| GGML IDs | Types | Elements / bytes per block |
|---|---|---|
| 0, 1 | F32, F16 | 1/4, 1/2 |
| 2, 3 | Q4_0, Q4_1 | 32/18, 32/20 |
| 6, 7, 8, 9 | Q5_0, Q5_1, Q8_0, Q8_1 | 32/22, 32/24, 32/34, 32/36 |
| 10–15 | Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K | 256/84, 256/110, 256/144, 256/176, 256/210, 256/292 |
| 16–20 | IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ4_NL | 256/66, 256/74, 256/98, 256/50, 32/18 |
| 21–23 | IQ3_S, IQ2_S, IQ4_XS | 256/110, 256/82, 256/136 |
| 24–28 | I8, I16, I32, I64, F64 | 1/1, 1/2, 1/4, 1/8, 1/8 |
| 29, 30 | IQ1_M, BF16 | 256/56, 1/2 |
| 34, 35 | TQ1_0, TQ2_0 | 256/54, 256/66 |
| 39–43 | MXFP4, NVFP4, Q1_0, Q2_0, F8_E4M3 | 32/17, 64/36, 128/18, 64/18, 1/1 |

Removed/deprecated IDs 4, 5, 31–33 and 36–38, unknown future IDs and
unrecognized fork-specific formats are rejected rather than guessed. GGML
blocks must fit within each row (dimension 0), not just the whole tensor.

### Important compatibility details

* **Q8_1 is ambiguous:** the current block has FP16 scale and sum (36 bytes),
  while a historical block has FP32 scale and sum (40 bytes), both with type
  ID 9. Padding or following tensors can make either interpretation fit.
  ZenInfer now rejects Q8_1 GGUF tensors by default instead of guessing. After
  verifying the source file's layout, set `ZEN_GGUF_Q8_1_LAYOUT=fp16` or
  `ZEN_GGUF_Q8_1_LAYOUT=fp32` before starting the CLI/server. C++ callers can
  instead pass `gguf::InspectOptions` with `Q8_1Layout::fp16_36` or
  `fp32_40`; the selected layout is retained in the report and used for both
  tensor reads and mapped inference. Both decoders ignore the precomputed sum.
  The environment setting is process-wide, so it cannot safely select different
  layouts for different models at once. **Selecting the wrong layout can still
  give incorrect values:** the file alone cannot reliably prove the choice.
  For example, on Windows PowerShell set `$env:ZEN_GGUF_Q8_1_LAYOUT = "fp32"`
  before launching ZenInfer; on Linux use `export ZEN_GGUF_Q8_1_LAYOUT=fp32`.
  Use `fp16` instead only for a verified 36-byte layout.
* I32/I64/F64 tensors are converted to FP32 for inference and can lose
  precision; F64 can overflow to infinity. F8_E4M3 preserves signed zero and
  decodes both NaN codes as NaN. Non-finite model weights may fail runtime
  finite-value checks. NVFP4 follows the pinned upstream UE4M3 decoding even
  for noncanonical scale bytes with the top bit set.
* CPU execution uses existing specialized kernels for some formats and a
  generic row-decode/dot or GEMM path for others. In particular, the five
  Q1_0, Q2_0, current and legacy Q8_1, NVFP4 and F8_E4M3 now have AVX2 GEMV
  and batched GEMM paths; F8 GEMM falls back for columns not divisible by 8.
  Many other formats still use generic paths. There is no claim of uniform
  performance or full optimization.
  Supported matrices are mapped; generic paths use bounded scratch rather than
  expanding the entire matrix. Vector/non-matrix tensors can be decoded to RAM.
* This work adds **no GPU support**; GPU format support is a separate question.

## Explicit fork and ZenInfer formats

The parser also recognizes fork-specific ROCmFP4 IDs 100/101 (32/18 and
32/17) and PrismML PTQ1_0 ID 143 (128/28). These ID assignments are not
universal across forks. `Q1_SIGNED` through `Q18_SIGNED` and `TERNARY2` are
ZenInfer reference encodings, not GGUF quantization IDs. Altogether the CPU
registry has 59 encodings, including scalar types, the explicit legacy Q8_1
layout and these reference formats.

## Verification scope

`zen_new_quant_tests` checks new formats against checked-in vectors generated
from the pinned upstream GGML reference (Q1_0, Q2_0, NVFP4 and F8_E4M3), and
independent arithmetic for Q8_1 and scalar types. It tests all 256 FP8 codes,
all 256 NVFP4 scale bytes, unaligned multi-block decoding, truncated GGUFs,
GGUF tensor reads and CPU matmuls. `zen_new_quant_runtime` compares mapped
projection execution against independently decoded F32 twin weights in a small
synthetic Qwen3.5 fixture, including both Q8_1 layouts. The Q8_1 layout tests
cover default rejection, padded ambiguous files, explicit selections and both
file and buffer readers. `zen_new_quant_simd_tests` compares new kernels against
independent element formulas over tail shapes, unaligned weights, worker counts,
token counts and padded output strides. The broader CPU sweep covers all registry formats,
but mostly compares matmul with ZenInfer's own decoder. See
`docs/CPU_QUANT_PROGRESS.md` for earlier independent checks and benchmark
limits. Real checkpoints for these new types and Windows/target-CPU performance
have not been validated by these tests.
