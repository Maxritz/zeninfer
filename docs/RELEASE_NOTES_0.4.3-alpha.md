# ZenInfer 0.4.3-alpha

## Implemented in this iteration

- Added a process-wide persistent CPU compute pool. Quantised GEMV/GEMM row partitions now reuse worker threads instead of creating and joining threads for every operation. The calling thread participates in each job; worker count is capped at 32.
- Added ALiBi slope and causal-bias primitives and integrated ALiBi into dense attention prefill and decode. When ALiBi is enabled by GGUF metadata, RoPE is bypassed.
- Enabled the Qwen3 dense adapter path in the architecture factory and added optional per-head Q/K RMS normalisation before positional processing.
- Corrected decode attention's sliding-window softmax and value accumulation to operate only over the active window. Previously, the softmax could include uninitialised scores outside the window.
- Added tests for persistent-pool partition coverage, deterministic single-worker execution, ALiBi slope ordering and causal bias.

## Verification

- Linux x86-64, GCC 14.2, AVX2 build: PASS.
- CTest: 5/5 PASS, including two existing real GGUF fixtures and HTTP smoke test.
- Qwen3 dense: factory dispatch and implementation path are present, but no Qwen3 GGUF fixture or independent reference logits were available in this workspace. Do not treat Qwen3 parity as verified.

## Not implemented / not verified

Qwen3 MoE, BitNet, Bonsai, general Jinja execution, hardware-native FP16 arithmetic, application-managed NVMe eviction, JSON Schema/grammar decoding, independent tokenizer parity, independent Qwen2 parity, full KV-cache lifecycle, continuous batching, additional attention families beyond ALiBi, Windows/MSVC execution and full benchmark regression thresholds remain open.

FP16 note: Zen 3 and Meteor Lake-class AVX2 CPUs provide FP16 storage/conversion support through F16C where present, but do not provide general packed FP16 arithmetic. AVX-512-FP16 is excluded by project scope. ZenInfer must not label FP32 accumulation or software half emulation as native FP16 arithmetic.
