# Milestone 1 numeric reference verification

Date: 2026-10-31

## ZI-006: binary16 conversion

Added `tests/data/fp16_reference_corpus.h`, a deterministic 2,048-case corpus generated independently using Python's IEEE-754 binary16 `struct.pack('<e')` implementation. The C++ test compares the exact encoded 16-bit result for every corpus entry. The existing exhaustive 65,536-pattern half decode/encode round-trip and rounding-boundary tests remain.

This strengthens software conversion verification. It does not implement native FP16 arithmetic, and does not establish Windows/MSVC parity.

## ZI-013: ALiBi reference verification

Added independent expected slopes for the eight-head power-of-two ALiBi construction and compared all eight slopes. Added 128 head/distance bias comparisons, plus invalid head-count and out-of-range-head checks.

This verifies the ALiBi slope and additive-bias primitives. It is not a full attention-output reference test and does not close other declared attention variants, RoPE/GQA/MQA edge cases, sliding-window fixtures, or metadata-driven model integration.

## Verification

Commands:

```sh

cmake --build build-scalar -j2
(cd build-scalar && ctest --output-on-failure)
cmake --build build-avx2 -j2
(cd build-avx2 && ctest --output-on-failure)
```

Results:
- Scalar: 8/8 CTest passed.
- AVX2: 8/8 CTest passed.
- `zen_core_tests` includes 2,048 independent binary16 conversion comparisons and ALiBi reference checks.

## Remaining in Milestone 1

- ZI-009: GPT-2/Hugging Face tokenizer parity is still open. Current GPT-2 pre-tokenisation is explicitly approximate and requires exact regex/Unicode pre-tokenisation and an independent token-ID corpus.
- ZI-012: only Q4_1, Q5_1 and BF16 were added in the latest decoder increment. K-quants, IQ families, MXFP4, complete GGUF type-size coverage and independent decoder corpus remain.
- ZI-013: only ALiBi primitive reference checks are now strengthened. Full attention output/reference coverage and other declared variants remain.
- ZI-006: independent float32-to-binary16 differential corpus now passes; Windows/MSVC verification remains.
