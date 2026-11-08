# A5 — Gate device attention behind sink-aware variants (silent divergence class)

## Context
`has_attn_sinks` models (gpt-oss, flint): attention is softmax over
valid KV + one virtual sink entry with score `sinks[h]` and zero value.

`zen::d3d12::attention` (and its prefill front) has no sink entry — it
diverges tokens by silently ignoring the sink (flint: CPU 260 vs GPU 269,
logits max-diff 25.3). We shipped a per-OOD gate:

```cpp
if (!config_.use_alibi && !config_.has_attn_sinks && device_attention_available(...))
```

in `dense_transformer_runtime.cpp:4249` after the flint DX mismatch.

## Proposal
1. Add sink-aware variants of the device kernels when a model needs them:
   - Decode: `attention_sinks_decode.hlsl` (one extra softmax denominator
     counter + sink score, value path unchanged).
   - Prefill: `attn_prefill_front` gets a `sinks` parameter when
     `has_attn_sinks` and the model quant matches (currently Q8-only).
2. Route: `attention_block` → sinks variant when `has_attn_sinks` and
   available, else keep the existing host fallback (current behaviour).
3. Validation: keep `find` parity tests flipping through `ZEN_GPTOSS_SINKS`
   (already present) and add a flint DX parity check to ctest so it can't
   regress silently.

## Why
At the moment every sink model is stuck on host attention — fine for
correctness, but it's wasted GPU headroom and the class of bug is silent.

## Acceptance
- Flint DX greedy tokens match CPU exactly (they do today via fallback).
  After the kernel lands, they must match *with the device path taken*.
- New parity test: `flint-q4_1.gguf` `--prompt-ids 1..8 --max-new 8` →
  `332,260,332,269,7437,332,269,280` on CPU and DX (default flags).
- Fence-wait table (A2) shows attn dispatch remains one submit per layer.

## Effort
M. One HLSL kernel (sinks are a denominator note), one routing gate, one test.
