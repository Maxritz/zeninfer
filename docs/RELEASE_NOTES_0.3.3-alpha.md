# ZenInfer 0.3.3-alpha

## Added

- Independent NumPy forward-pass parity harness (`zen_parity_export`, `zen_parity_probe`, `scripts/numpy_reference.py`).
- Repeatable two-fixture parity runner covering four text prompts per model.
- Runtime AVX2/FMA3 FP32 GEMV dispatch with scalar reference fallback.
- Fused Q8_0 AVX2 GEMV kernel and scalar fallback.
- Q8_0 fused-kernel regression test against dequantise-then-dot reference.

## Verified

- Eight NumPy-versus-C++ final-logit comparisons passed. Maximum absolute error was 6.6757e-6; greedy argmax agreed in all cases.
- Five-run TinyLlama Stories decode median: 336.42 tokens/s AVX2 versus 70.70 tokens/s scalar on the virtualised AMD EPYC 9V74 host.
- Scalar and AVX2 generated token IDs matched in a 64-token test.
- AVX2, scalar-only and ASan/UBSan test builds passed.

## Limitations

- NumPy parity uses GGUF tensors dequantised by ZenInfer. It validates transformer math, not third-party quant decoder parity.
- The Llama runtime still expands weights to FP32. The fused Q8_0 kernel is implemented and tested through the quantised GEMV API but not yet wired into the model's projection path.
- Q5_0, Q3_K, Q6_K and IQ4_NL still lack fused quantised GEMV kernels.
- No Windows/MSVC or physical Ryzen validation in this release.
