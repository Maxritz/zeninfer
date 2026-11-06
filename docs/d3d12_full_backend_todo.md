# Completing the D3D12 backend

Goal: every dense-model op runs on the GPU, and one transformer layer costs one
submission instead of three. Verification is per-kernel parity against a CPU
reference plus end-to-end identical token ids.

## Done

- [x] Q8_0 repack + VRAM residency, keyed by weight pointer
- [x] `gemv_q8_0` matvec (decode), parity 1.26e-07
- [x] `gemv_q8_0_batch` — Q|K|V in one submission
- [x] `gemv_q8_0_ffn` — gate|up + device SiLU + down in one submission
- [x] `residual_add` (elementwise `a + b`), parity **exact** over four sizes
- [x] `ffn_residual_fused` — the FFN half of a layer in one command list
  (attn_output + residual + norm + gate|up + SiLU + down + residual), ids identical,
  decode 79.6 -> 91.9 tok/s
- [x] `rms_norm` now has a production caller (the fused FFN path)
- [x] `gemm_q8_0` prefill matmul, row-blocked workgroups — parity **9.02e-08 /
  1.02e-07 / 1.64e-07 / 1.87e-07** across the four harness shapes (2026-11-03).
  Three defects were fixed to get there: `gemm_locked` never issued its `Dispatch`,
  the shader indexed the activation at column 0 for every lane instead of the lane's
  own column, and the chunk offset was counted twice in the activation load.
  `ZEN_BACKEND=d3d12 ZEN_D3D12_GEMM_EXPERIMENTAL=1` end-to-end ids are identical to
  CPU. (still opt-in: cold upload makes a first run slower than AVX2)

## Task list

1. **Device-side RMSNorm kernel** (`rmsnorm.hlsl`) — **done 2026-11-03**: wired as
   the FFN norm inside `ffn_residual_fused`, so `zen::d3d12::rms_norm` now has a
   production caller and the layer's norm runs from VRAM. Kernel parity 1.39e-07.
   The layer-input norm (`attn_norm`) still runs on the host, because that input is
   produced by the previous layer's readback.

2. **Device-side residual add** — **done 2026-11-03**, as `addn.hlsl` +
   `zen::d3d12::residual_add`. The old `addn.hlsl` was not a residual add at all: it
   assigned instead of accumulating into the reduction partial and subtracted a row
   mean, and it had no entry point. It is now a true elementwise `out = a + b` with
   no reduction, exported through `quant::` and verified by
   `zen_d3d12_addn_parity` (exact, 0.000e+00, four sizes including tails).

3. **Fuse the attention output half of a layer into one submission** — **done
   2026-11-03** as `zen::d3d12::ffn_residual_fused`, which records attn_output
   matvec + residual add + FFN norm + gate|up + SiLU + down + residual add in a
   single command list (8 dispatches, one readback). Two device buffers alternate
   SRV/UAV roles so no dispatch both reads and writes the same resource.
   End-to-end ids identical to CPU (`2af03108899071e1`, 24 tokens); decode 79.6 ->
   **91.9 tok/s** (128 tokens, same prompt). `ZEN_D3D12_FUSE_FFN=0` restores the
   separate ops for A/B and as a per-model escape hatch.

4. **VRAM-resident KV cache** — **done 2026-11-03**: `kv_cache_configure` /
   `kv_cache_reset` own `[layers][context][kv_heads][head_dim]` in VRAM, k and v in
   separate DEFAULT buffers. The block appends a token with one contiguous
   device-to-device `CopyBufferRegion`, so k/v never reach the host.

5. **RoPE kernel** (`rope.hlsl`) — **done 2026-11-03**: wired as
   `zen::d3d12::rope_apply` and verified by `zen_d3d12_rope_parity` (4 shapes:
   32x64 full rotary, 8x64 GQA keys, 32x128 partial rotary, and a 3x64x17 case whose
   pair count is not a multiple of the 256-thread group). Two defects fixed on the way:
   (a) the CSO was never included, so `kRopeCso` did not exist and the kernel could
   not have been called; (b) the kernel launched heads*pairs threads and wrote only
   the rotated pairs, leaving dimensions past rotary_dim undefined — it now covers
   the whole head and copies the tail through, which is what a partial-rotary model
   such as Qwen needs. Comparison is tolerance-based (1e-5), not exact, because dxc
   contracts `a*c - b*s` into an fma at -O3; stating that is more honest than an
   exactness claim the compiler makes impossible. The cos/sin table stays host-built:
   powf/cosf on the device would differ in the last bits from the double-promoted
   reference and could flip a sampled id.

6. **GQA attention kernel** (`attention.hlsl`) — **done 2026-11-03** (decode only):
   `zen::d3d12::attention_block` runs rope(q), rope(k), the cache append and GQA
   attention as ONE submission, then reads back just the attention output.
   `zen_d3d12_attn_parity` walks positions in ascending order so the append path is
   covered too: 4 heads/2 kv, 32 heads/8 kv at the Llama-3.2-1B shape, and MQA 8/1.
   Max abs error 1.2e-07. Three real bugs found on the way:
   (a) `attn_device` created UAV buffers without `ALLOW_UNORDERED_ACCESS`, an invalid
       call that removed the device and made every later allocation fail — the symptom
       looked like "KV cache allocation is broken", not like a UAV flag;
   (b) the two rope dispatches shared one cbuffer slot, and the GPU reads it at
       execution time, so q rotated with the k head count;
   (c) the shader indexed `[kv_head][context][dim]` while the append writes
       `[context][kv_head][dim]`. The two coincide only when kv_heads == 1, so the MQA
       case passed while every real GQA case was wrong. Shader now matches the write
       layout, which is also what makes the append a single copy.
   **Wired 2026-11-03**: prefill keeps host attention but now also appends every prompt
   token to the device cache (`kv_cache_append`, a pure copy), so decode finds the whole
   window there; decode runs the block and skips the host loop on success. Verified
   end-to-end: ids still 621ed5b6dcd1ae3f, `fallbacks=0`, and the submit/dispatch counts
   confirm the path is live (9 -> 12 dispatches, 2 -> 3 submits per layer per token;
   1576/530 for 16 layers x 8 tokens). ALiBi models stay on the host path because the
   kernel has no bias term. Still open: attentive prefill (a causal tile) is NOT
   implemented, so prefill is not device-complete.

7. **`layer_decode()`** — whole layer in one command list: norm, QKV, RoPE,
   attention, attn_output, residual, norm, gate|up, SiLU, down, residual.
   One submission per layer, one readback per token.
   **Blocker found 2026-11-03**: task 6 cannot be switched on in the runtime yet.
   Prefill and decode share one host KV cache, and decode reads positions written by
   prefill. Wiring the device attention block while prefill still fills only the host
   cache would leave the device cache empty for every prompt token and silently change
   generated ids. So task 7 must also make prefill populate the device cache (or run
   prefill through the same device path, which needs the causal-tile branch of
   attention.hlsl). The block is parity-clean and self-contained but stays unwired
   **Done 2026-11-03**: `zen::d3d12::layer_decode` runs a whole layer as one submission -
   13 dispatches, one readback - and decode uses it whenever the block needs nothing from
   the host (no ALiBi, no biases, no qk-norm, not BitNet; anything else falls through to
   the per-op path). q/k/v never leave VRAM.
   Three activation arenas are required, not two: the residual base h1 has to stay
   readable across the whole FFN half while both other arenas are written, so with only
   A/B the final residual would need one buffer in two states at once. Layout is
   A: h,qr,kr,t0,h2,f,fin  B: q,k,v,g,u,d  C: h1.
   Verified: ids 621ed5b6dcd1ae3f identical to CPU, `fallbacks=0`, submits 3 -> 1 per
   layer (48 -> 16 per token, 306 for 10 tokens incl. prefill), dispatches 1464.
   Decode 100.8 -> **122.1 tok/s** mean of 4 runs (121.15 / 121.42 / 122.00 / 123.00),
   which does not overlap the previous 106-111 range. ctest 14/14.

8. **BitNet / Qwen3.5 variants** — either support their extra norms on the device or
   **done 2026-11-03 as the explicit fallback**: decode already drops to the per-op host
   path when `config_.bitnet_variant` is set (fused FFN skipped, `attn_sub_norm` on the
   host), and the Qwen3.5 hybrid has its own `forward_q35_token`. This is the documented
   fallback option, not device support — BitNet norms are still host-side.

9. **Register GEMM parity in ctest** — **done** 2026-11-03: `zen_d3d12_gemm_parity`
   is registered alongside `zen_d3d12_gemv_parity`, both with `ZEN_BACKEND` (and
   `ZEN_D3D12_GEMM_EXPERIMENTAL` for GEMM) set, and both exit 77 on a machine with no
   D3D12.1 device so ctest reports *Skipped* instead of a green PASS.

Measure after each step; do not enable a fused path until its ids match exactly.

## Targets

- prefill > 8000 tok/s at 64 tokens: needs <= 4 weight passes, i.e. a 16-token tile,
  plus near-peak bandwidth (1.325 GB passes at 640 GB/s = 2.07 ms each).
- decode: hardware wall at Q8_0 is 640 GB/s / 1.321 GB = **484 tok/s**. 600 tok/s is
  only reachable with smaller weights (Q6_K ~1.05 GB, Q5_K ~0.90 GB). At Q8_0 the
  work is to close the gap from ~110 to ~400 tok/s, which is entirely the
  per-submission cost (3 submits/layer x ~150 us).

## Agility SDK + ReBAR (verified 2026-11-03)

Agility SDK 1.722 is active, and ReBAR is live on this machine:
`d3d12 adapter=AMD Radeon RX 9070 XT fl=122 agility=722 gpu_upload_heap(rebar)=1`

GPU upload heaps measured *neutral*, so they are opt-in (`ZEN_D3D12_GPU_UPLOAD=1`):
104.78 tok/s on system-memory upload heaps vs 104.18 tok/s through the ReBAR
aperture (mean of 3, 256 tokens). Activations are a few KB and already L2-resident;
weights already live in HEAP_TYPE_DEFAULT (VRAM), so nothing was crossing PCIe.

## Decode bottleneck (measured)

`submits=8273 dispatches=45136 per_fence_us=185` for 256 tokens = 32 submits/token
= 2 per layer, and 185 us of that is a genuine GPU retirement (kSpinWaitUs=2000, so
it is not the OS wait path). ~145 us per submission is fixed submission overhead.

decode_forward_ms=2395.8 for 256 tokens = 9.36 ms/token, of which ~5.9 ms is
submission overhead and ~2.07 ms is the unavoidable 1.3 GB of weight traffic at
644 GB/s. The next step is therefore layer_decode: device KV cache + GPU rope +
GPU attention so all 16 layers record into one command list per token, i.e. 1-2
fences per token instead of 32.

## q8_0_gemv rewrite: one warp per row (verified 2026-11-03)

Root cause: 64 threads/row meant exactly one 32-byte block per thread on a 2048-wide
row, so every thread issued three loads and stalled. Latency-bound, not bandwidth-bound.
Rewrote to 8 rows/group with 32 lanes per row (kRowsPerGroup 4 -> 8), giving 2 blocks
in flight per thread on 2048-wide rows and up to 5 on 5632-wide ones.

Trap hit on the way: WaveActiveSum is wrong here. This GPU runs wave64, so a wave spans
two rows and folded their accumulators together (parity: relative_err 1.000e+00). Fixed
with a fixed 32-lane groupshared window, correct for either wave width.

GPU device time, ZEN_D3D12_TRACE=1, 256 tokens:

| component  | before dev us | after dev us | change |
|------------|---------------|--------------|--------|
| projections| 201157        | 85556        | -57.5% |
| ffn-fused  | 718983        | 548614       | -23.7% |
| TOTAL      | 920140        | 634170       | -31.1% |

Per op: projections 49.3 -> 21.0 us, ffn-fused 176.2 -> 134.5 us.
Effective bandwidth: projections 137 -> 320 GB/s, ffn-fused 233 -> 305 GB/s (ceiling 644).

End-to-end decode, 5 runs, non-overlapping with baseline:
before 96.1 / 98.7 / 100.0 / 104.8 / 104.2  (mean 100.8)
after  106.5 / 106.9 / 106.9 / 110.5 / 110.8 (mean 108.3)

ctest 12/12, ids 621ed5b6dcd1ae3f unchanged.
Remaining: idle 866310 us of 1500480 us fence time; kernels still ~50% of ceiling.

## Attention kernel tuning + layer trace (2026-11-03)

Rewrote attention.hlsl from one 128-thread workgroup per head with a full block
reduction per cached position (7 barriers x context per head) to one 32-thread group
per head with an explicit 32-lane shuffle reduction. Two traps:

- `__shfl_down_sync` is CUDA; HLSL has WaveReadLaneAt. On this wave64 GPU any
  wave-wide sum folds two adjacent logical warps - two different heads - so the lane
  index is rebased onto the caller's 32-lane group.
- Packing 4 heads into a 128-thread group cut the grid from 32 groups to 8, i.e. a
  quarter of the CUs. Measured 120.1 tok/s against 122.1 before. One group per head
  restored it: 121.4 tok/s.

So the rewrite is perf-NEUTRAL here (121.4 vs 122.1, ranges overlap) and parity-clean
(1.2e-07). The honest reading is that attention is NOT the decode bottleneck at
context 256; the 13 matvec/norm/silu dispatches are. The barrier-free version still
wins on paper at long context and is kept on that basis, not on a measured gain.

Layer trace (ZEN_D3D12_TRACE=1, 256 tokens, 4161 layer submissions):
  dev 867110 us = 3.40 ms/token of real GPU work
  wait 1834340 us = 7.19 ms/token of fence time, idle share 52.7% -> 3.79 ms idle
The 2.07 ms Q8_0 memory floor means GPU work is still ~1.64x off, and idle is now the
larger of the two defects. NOTE: the FfnFused half of the layer did not register its
own row in this trace (only Projections appeared), so the intra-layer split is not yet
trustworthy - do not quote a norm/FFN breakdown from it.
