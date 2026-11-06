# Synthetic test fixtures

These tiny seeded `.zenmodel` definitions exercise the standalone
`zen_arch_microbench` kernel/model simulator. They are **not trained models,
GGUFs, full transformer adapters, or end-to-end language quality tests**.
The Qwen3 MoE, BitNet and Bonsai definitions reproduce the seeds, dimensions
and checksums documented in `docs/evidence/ZI-ARCH-MICRO-MODEL-2026-10-31.md`.
The dense definitions use new deterministic seeds and locally recorded
checksums; those checksums are regression guards, not independent oracles.

`tests/make_qwen2_synthetic.py` produces a small, deterministic F32 GGUF in the
build directory at build time, using only Python's standard library. The Qwen2
runtime and independent NumPy parity tests use that file. It is a synthetic
layout/behavior fixture, not the older TinyLlama-derived Qwen2 fixture and not
a trained Qwen2 checkpoint. Optional real-model tests remain separately
configurable with `ZEN_GGUF_FIXTURE_1` / `ZEN_GGUF_FIXTURE_2`.
