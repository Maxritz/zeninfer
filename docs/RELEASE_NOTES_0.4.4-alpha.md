# ZenInfer 0.4.4-alpha

## Implemented

- Added a deterministic `.zenmodel` micro-fixture runner and three checked-in definitions:
  - Qwen3 MoE: router scoring, top-2 expert selection, SwiGLU expert projections and weighted expert combine.
  - BitNet: GGML TQ2_0 256-element block layout and group scale path.
  - Bonsai-style ternary Q2_0-g128: Qwen3-dense-shaped residual MLP micro-model.
- Added golden-logit checksums and expert-selection accounting to the micro-model tests.
- Implemented GGML TQ1_0 and TQ2_0 dequantisers in the quantisation registry.
- Connected GGUF tensor read/decode for GGML type IDs 34 (TQ1_0) and 35 (TQ2_0).
- Added TQ1_0 and TQ2_0 decode tests plus an end-to-end GGUF TQ2_0 tensor fixture.
- Added real-model prefill tok/s and decode-forward tok/s reporting to runtime fixture tests. Timed prefill uses a 128-token prompt and a distinct warm-up prompt to avoid exact-prefix cache reuse.
- Added `docs/PERFORMANCE_0.4.4-alpha.md` with real GGUF and synthetic micro-model measurements and clear measurement definitions.

## Verification

- Linux x86-64, GCC 14.2, Release AVX2: 6/6 CTests PASS.
- Linux x86-64, GCC 14.2, Release scalar-only: 6/6 CTests PASS.
- Real GGUF runtime fixture tests: PASS for both supplied Tiny model fixtures.
- Qwen3 MoE, BitNet and Bonsai: synthetic kernel micro-models pass golden checks; no production checkpoint parity is claimed.
- Windows/MSVC: not run in this environment.

## Explicit scope boundaries

The synthetic model format is for deterministic architecture-kernel tests only. It is not a substitute for GGUF model loading, production tokenisation, attention/KV execution, Jinja templates or independent reference parity. TQ1_0/TQ2_0 decoding is a real quant-format implementation, but it does not by itself constitute a BitNet transformer adapter. Bonsai's public low-bit variants differ: the Ternary Bonsai Q2_0-g128 profile is distinct from Bonsai2's Hadamard-rotated hybrid Qwen3.5 architecture. They must not be merged into one assumed format.
