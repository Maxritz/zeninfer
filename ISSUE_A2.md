# A2 — Add device-execution vs fence-wait timing split (ZEN_D3D12_TRACE=1)

## Context
`zen_sherlock` currently prints the wall clock (116 ms prefill T=32) and
`[d3d12] pipeline: gpu 9 ms` stubs, but nothing separates "the queue had
nothing to do" from "the kernel was working". From the audit at 45c813f:
- `[d3d12] trace_report` already reports `weights_resident`, `submits`,
  `dispatches`, `queue`, `wait`, `gpu`, `fallbacks` for the whole session,
  but the per-component fence-wait table is only populated under a
  development flag, and `gpu_us` is often 0 even when submits happen.

## Proposal
Extend `ZEN_D3D12_TRACE=1` to emit a sortable per-submit table:

```
op                 submits   %submits   dev_us   wait_us   idle_us
layer_decode            36           14        0      3500     3500
ffn_prefill_fused       36           14     4200      8300     4100
attn_prefill_front      36           14     2000      6100     4100
```

Requirements:
1. Attribute every `ID3D12Fence` wait to the op that owned the submission —
   thread-legally, per op name (pipeline / gemv_batch / gemm_batch /
   ffn_prefill_fused / attn_prefill_front / layer_decode / attention_block).
2. Device timestamp the dispatch (`Dispatch` → `EndQuery`) so `dev_us` is
   real GPU time, not wall time.
3. Keep `ZEN_D3D12_TRACE` atoi-1-gated (default off); the table prints to
   stderr once via `trace_report()`.

## Why
Today you can't tell a 50 ms pipeline sync tax from a 50 ms kernel inside
one number — the exact ambiguity that made the earlier "10x slow" claim
hard to adjudicate. Every perf PR after this can cite the table.

## Acceptance
- `ZEN_D3D12_TRACE=1 zen_sherlock qwen3-4b-q8_0.gguf 512 2` prints the table
  and `sum(dev_us) <= wall` for the prefill window.
- Uniform instrumentation free — i.e. trace-on overhead ≤ 2% on a 512-token
  run (measure against trace-off).
- Cross-checked against `budget_bar` in `d3d12_gemm_bench` if present.

## Effort
S. Backend-only; no kernel or runtime changes.
