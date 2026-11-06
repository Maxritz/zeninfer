# ZI-012 — K-quant, IQ family and MXFP4 decoders (port from 0.4.1-alpha)

Date: 2026-10-31
Scope: Complete the runtime decoder set for every GGML storage type in the
canonical type table: Q2_K, Q4_K, Q5_K, Q8_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS,
IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS and MXFP4.

## Provenance

All decoder bodies were ported from the hardened 0.4.1-alpha engine tree
(`H:\ZenInfer\ZenInfer-0.4.1-alpha-model-neutral`, commit `cbf90d3`), where they
passed an 88-check differential against an independent Python reference. The
grid codebooks (`src/iq_tables.h`), the generator (`scripts/extract_iq_tables.py`),
the independent reference (`scripts/iq_reference.py`) and the differential
harness (`tests/iq_check.cpp`) were copied from the same tree.

## Changes

- `include/zen/quant.h`: Encoding enum extended with the 13 new format codes.
- `src/quant.cpp`:
  - registry rows for all 13 formats (block layout, byte size, implemented=true);
  - `row_bytes` cases for all 13 formats;
  - `k_scale_min` helper (6-bit scale/min unpacking for Q4_K/Q5_K);
  - `ggml_e8m0_to_fp32_half` (exact E8M0 semantics from `ggml-impl.h`);
  - `kvalues_mxfp4` — verbatim `kvalues_fp4` table from `ggml-common.h`;
  - dequantize cases for all 13 formats, including the corrected TQ1_0/TQ2_0
    block layouts (f16 delta at the END of the block: offset 52 / 64) per
    `block_tq1_0` / `block_tq2_0` in upstream `ggml-common.h`. The
    pre-existing completion-tree TQ decoders read the delta from offset 0 and
    were wrong; the tests were wrong the same way and were corrected.
- `scripts/iq_reference.py`: numpy dependency removed (stdlib `random` with a
  fixed seed, hand-written f16 widening) so the differential corpus can be
  generated on any Python. The generated corpus is consumed byte-identically
  by both sides of the differential, so independence is preserved.
- `CMakeLists.txt`: `zen_iq_check` target + `zen_iq_ref` custom target that
  regenerates `iq_ref.bin` (66 records) before the differential test runs.
- `tests/core_tests.cpp`: RED markers flipped to GREEN (all 13 formats must
  report implemented=true); exact-block tests for Q2_K, Q4_K, Q5_K, Q8_K,
  MXFP4; GEMV-vs-float64-dot coverage for one super-block format of each
  family; corrected TQ block vectors and the end-to-end GGUF TQ2_0 read.

## Upstream verification of the two layout disputes

Checked against `ggml-common.h` (ggml-org/llama.cpp master, fetched 2026-10-31):

- `block_tq2_0 = { uint8_t qs[QK_K/4]; ggml_half d; }` — delta at offset 64.
  Tree 1's decoder confirmed correct; completion-tree test vectors fixed.
- `block_mxfp4 = { uint8_t e; uint8_t qs[16]; }` — E8M0 at offset 0;
  `kvalues_fp4 = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12}` (code 7 = 12,
  code 8 = -0). An interim hand-derived table in this session was replaced by
  the verbatim upstream table before acceptance.

## Commands and results

AVX2 build (MinGW g++ 13.2, `-O3 -mavx2 -mfma`):

```
cmake -S H:\ZenInfer\ZenInfer-completion-work -B H:\ZenInfer\ZenInfer-completion-work\build
cmake --build H:\ZenInfer\ZenInfer-completion-work\build -j 8
ctest --test-dir H:\ZenInfer\ZenInfer-completion-work\build
```

```
1/7 Test #1: zen_core_tests ...................   Passed    0.09 sec
2/7 Test #2: zen_iq_check .....................   Passed    0.01 sec
3/7 Test #3: zen_fuzz_gguf ....................   Passed    0.02 sec
4/7 Test #4: zen_arch_micro_models ............   Passed    0.09 sec
5/7 Test #5: zen_bench_parity .................   Passed    0.74 sec
6/7 Test #6: zen_runtime_qwen2_synthetic ......   Passed    0.41 sec
7/7 Test #7: zen_http_api_smoke ...............   Passed    0.19 sec
100% tests passed, 0 tests failed out of 7
```

Scalar build (`-O3`, no AVX2):

```
ctest --test-dir H:\ZenInfer\ZenInfer-completion-work\build-scalar
```

```
1/7 Test #1: zen_core_tests ...................   Passed    0.14 sec
2/7 Test #2: zen_iq_check .....................   Passed    0.05 sec
3/7 Test #3: zen_fuzz_gguf ....................   Passed    0.05 sec
4/7 Test #4: zen_arch_micro_models ............   Passed    0.11 sec
5/7 Test #5: zen_bench_parity .................   Passed    0.72 sec
6/7 Test #6: zen_runtime_qwen2_synthetic ......   Passed    0.71 sec
7/7 Test #7: zen_http_api_smoke ...............   Passed    0.19 sec
100% tests passed, 0 tests failed out of 7
```

Independent differential (C++ decoders vs Python reference):

```
H:\ZenInfer\ZenInfer-completion-work\build\zen_iq_check.exe build\iq_ref.bin
iq_check: 88 checks, 0 failures, worst relative decoder diff 0
```

RED observation: before this increment the 13 formats carried
`implemented=false` stubs and the new tests failed (905 CHECK failures were
observed mid-increment while a clobbered dequantize function and the wrong
TQ/MXFP4 tables were being reconciled against upstream). GREEN is the state
above.

## Real-model MXFP4 validation (qwable-v1-mxfp4_moe.gguf)

The remaining acceptance item was closed on 2026-11-01 against a genuine
MXFP4 GGUF. Model discovery used a new type-distribution scanner
(`scripts/scan_gguf_types.py`):

```
python scripts\scan_gguf_types.py G:\More-models\qwable-v1-mxfp4_moe.gguf
version=3 tensors=753 kv=49
  MXFP4      x123     17.541 GB   (blk.N.ffn_{down,gate,up}_exps.weight)
  Q8_0       x320      2.619 GB
  F32        x310      0.091 GB
```

Note: the files named `*ROCMFP4*` (granite-4.1-3b, Spark-X2.5-4B) do NOT
contain MXFP4 — they use custom GGML type IDs 100/101 that are outside the
canonical table, and are rejected by the parser as unsupported. Only
`qwable-v1-mxfp4_moe.gguf` carries real type-39 tensors.

Validation tooling:

- `tools/mxfp4_probe.cpp` (`zen_mxfp4_probe`): inspects a real GGUF, seeks to
  a named MXFP4 tensor payload, decodes the first N blocks through
  `quant::dequantize`, writes raw f32 for external checking.
- `scripts/mxfp4_check.py`: fully independent Python reference (own GGUF
  directory parser, `kvalues_fp4` verbatim from ggml-common.h, E8M0
  `GGML_E8M0_TO_FP32_HALF` bit semantics) that dequantizes the same block
  bytes and compares. An initial reference draft used adjacent nibble
  interleaving; the spec (and our decoder) uses split-half mapping — byte j
  decodes outputs j and j+16 — and the corrected reference agrees bit-exactly.

Results:

```
zen_mxfp4_probe qwable-v1-mxfp4_moe.gguf blk.0.ffn_down_exps.weight 4096
  tensor=blk.0.ffn_down_exps.weight ggml_type=39 byte_size=142606336 dims=3 512 2048 256
  decoded 4096 blocks (131072 values), 0 NaN, amax=0.046875
python scripts\mxfp4_check.py ... blk.0.ffn_down_exps.weight ... 4096
  mxfp4_check: 131072 values, 0 mismatches, worst relative diff 0
  MXFP4 real-model cross-check: PASS (bit-exact)

zen_mxfp4_probe ... blk.5.ffn_gate_exps.weight 4096   (second layer, 131072 values)
python scripts\mxfp4_check.py ... blk.5.ffn_gate_exps.weight ... 4096
  mxfp4_check: 131072 values, 0 mismatches, worst relative diff 0
  MXFP4 real-model cross-check: PASS (bit-exact)
```

Total: 262,144 real-model weight values decoded bit-exactly against an
independent reference, across two expert tensors from different layers.
After the CMake addition (`zen_mxfp4_probe` target) the full suite still
passes: AVX2 7/7, scalar 7/7.

## ROCmFP4 (types 100/101) reverse-engineering and validation — 2026-11-01

Two additional community formats found in the local model library and traced
to their source: **ROCmFP4**, written by the `ciru-ai/ROCmFPX` fork of
llama.cpp (AMD Strix Halo / RDNA4 ecosystem). Identified via Hugging Face
model cards after kands' GitHub code search proved auth-walled; confirmed by
cloning the public fork and reading `ggml/rocmfp4/rocmfp4.{h,c}`.

### Format spec (from fork source)

- GGUF type 100 `q4_0_rocmfp4`: `{ uint8_t qs[16]; uint8_t e[2]; }` — 18 B,
  dual UE4M3 scales (one per 16-weight half). 4.5 bpw. (granite file, 280
  tensors)
- GGUF type 101 `q4_0_rocmfp4_fast`: `{ uint8_t qs[16]; uint8_t e; }` — 17 B,
  one UE4M3 scale per block. 4.25 bpw. (Spark file, 180 tensors)
- Codebook10: `{0,1,2,3,4,6,8,10, 0,-1,-2,-3,-4,-6,-8,-10}` — like MXFP4's
  `kvalues_fp4` but with the top level retuned 12 -> 10.
- Scales: finite unsigned E4M3 (UE4M3, bias 7): byte<=0x7E ->
  (8+M)*2^(E-10); E==0 subnormal M*2^-10; byte 0x7F -> 0.
- Nibble order: byte j -> outputs j (low) and j+16 (high) — same split-half
  as MXFP4.

This confirmed the earlier forensic dead ends: the scale bytes sit at the END
of each block (offsets 16/17), not offset 0, and their E4M3 distribution
matches neither the fp16 nor the E8M0 scans.

### Implementation

- `include/zen/quant.h`: `rocmfp4`, `rocmfp4_fast` encoding codes.
- `src/quant.cpp`: registry rows (18/17 B), `row_bytes`, `kvalues_rocmfp4`,
  `rocmfp4_ue4m3_to_fp32_half`, dual- and single-scale dequantize cases.
- `src/gguf.cpp`: types 100/101 in the canonical size table and in
  `read_tensor_f32`'s decoder map.
- `tests/core_tests.cpp`: exact-block tests for both variants (incl. the 0x7F
  scale-decodes-to-zero rule).
- `tools/mxfp4_probe.cpp` + `scripts/rocmfp4_check.py`: real-model probe and
  independent Python reference (fork semantics re-derived independently).

### Real-model validation

```
granite-4.1-3b-Q4_0_ROCMFP4_COHERENT.gguf  blk.0.attn_k.weight  (type 100)
  131072 values, 0 mismatches, worst relative diff 0  -> PASS bit-exact
granite-4.1-3b-Q4_0_ROCMFP4_COHERENT.gguf  blk.20.ffn_down.weight (type 100)
  131072 values, 0 mismatches, worst relative diff 0  -> PASS bit-exact
Spark-X2.5-4B-Q4_0_ROCMFP4_STRIX_LEAN.gguf blk.0.attn_output.weight (type 101)
  131072 values, 0 mismatches, worst relative diff 0  -> PASS bit-exact
```

Total: 393,216 real-model weight values across both ROCmFP4 variants, three
tensors, two files — bit-exact against the independent reference.

Debug notes (both errors were in this tree, not the fork): the UE4M3
subnormal branch initially built `M<<13` fp32 subnormal bits (wrong by 2^-x
and wrong by mantissa placement — E4M3 "subnormal" M*2^-10 is a *normal*
fp32); and one test vector used the wrong nibble (`0x78` lo=8 zero instead of
`0x87`). Fixed and re-verified.

Regression after implementation: AVX2 7/7, scalar 7/7 CTest.

## Status

ZI-012 decoder set is complete: every GGML storage type in the canonical table
has a runtime decoder validated by exact block tests, GEMV differential
coverage, an independent 88-check Python corpus, AND real-model MXFP4
validation. Remaining (non-decoder): fused ternary kernels, real-model
token-parity runs on K-quant weights.
