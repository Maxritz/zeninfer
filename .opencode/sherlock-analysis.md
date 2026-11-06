# Sherlock Analysis Log

## [RUN-003] 2026-11-05T15:30:00Z — MODE: FULL — TARGET: ZenInfer D3D12 + CPU

### BASELINE

- Hardware: RX 9070 XT (RDNA4, Navi 48, 64 MB Infinity Cache, 256-bit GDDR6 @ 20 Gbps = 640 GB/s
  theoretical); Ryzen 9 5900XT (Zen 3, 16C/32T, DDR4-3000 dual-channel = 48 GB/s theoretical)
- Workload: Llama-3.2-1B-Instruct Q8_0 (1.24 G params, 1.32 GB), prefill 128 tokens, decode 32
- Instrumentation floor: **0.062 us per empty scope** (2000 empty scopes through the same path)

| Path | Prefill | Decode |
|---|---|---|
| D3D12 GPU | 169.72 ms / **754 tok/s** | **8.2 ms/token** |
| CPU only | 465.58 ms / **275 tok/s** | **66.6 ms/token** |

### DECOMPOSITION

Tool: `zen_sherlock` (new). Separates prefill from decode, prints per-component scope
time and the backend counters.

**CPU prefill (465 ms)**
| Component | Ops | Cost | Share |
|---|---|---|---|
| quant.gemm | 112 | 355 ms | 76% |
| quant.gemv | (parallel sum) | 1446 ms* | — |
| attention.softmax | 496 | 51 ms | 11% |
| prefill.batch (attention) | 1 | 31 ms | 7% |
| prefill.attention | 16 | 27 ms | 6% |
| prefill.rope | 16 | 18 ms | 4% |
| prefill.silu | 16 | 13 ms | 3% |

\* summed across worker threads, not wall time — not directly comparable.

**GPU prefill (170 ms)**: `quant.gemm` 112 ops = 95 ms (56%), `transformer.forward`
exclusive 243 ms summed, `prefill.rope` 28.6 ms, `prefill.batch` 29 ms.

**GPU decode (8.2 ms/token)**: 16 layers x 1 submission = **16 submissions per token**.

### FINDINGS

| Rank | Component | Cost | Evidence | Status |
|---|---|---|---|---|
| 1 | GPU decode: submission latency | 93% of fence wait | wait 99.1 ms vs dev 7.1 ms over 15 tokens; 16 submits/token | CONFIRMED |
| 2 | GPU prefill: quant.gemm | 56% | 848 us x 112 = 95 ms of 170 ms; 2.65 TFLOP/s vs 15.4 needed | CONFIRMED |
| 3 | CPU prefill: quant.gemm | 76% | 3172 us x 112 = 355 ms of 465 ms; 651 GFLOP/s of 1.69 TFLOP/s peak | CONFIRMED |
| 4 | CPU decode: quant.gemv | 94% | 412 us/call; 1.32 GB at ~20 GB/s of ~40 GB/s | CONFIRMED |
| 5 | CPU prefill: rope | 18 ms | 1144 us/layer for 128x2048 elements | SUSPECTED |

### ROOT CAUSES

**GPU decode — submission latency, not bandwidth.** This overturns the earlier
"76% of peak bandwidth" reading. Device execution is only 0.47 ms/token; the fence
wait is 6.6 ms/token across 16 submissions, i.e. ~380 us of pure launch latency per
submit. The GPU is idle 93% of the wall clock. `layer_decode_pipeline` collapses this
to 1 submission and measured 2.8 ms of GPU time, but it is currently disabled
because its resource barriers were recorded into a closed command list.

**GPU prefill — GEMM at 2.65 TFLOP/s of 15.4 required.** Inner loop dequantises per
weight element. Reducing weight traffic 4x changed prefill by 6%, proving it is
instruction-bound.

**CPU prefill — GEMM at 39% of AVX2 peak.** The AVX2 kernel dequantises a weight
row once and reuses it across 4 tokens, which is why it reaches 651 GFLOP/s while
still converting. Not the same defect as the GPU path.

**CPU decode — memory-bound and near the practical limit for this kernel.**
412 us per 8192x2048 matvec = 40 GB/s per call, which is the bandwidth of this
CPU's DDR4. The 66 ms/token total is ~2x the 33 ms bandwidth floor, so the
remaining loss is per-call overhead across ~113 sequential matvec calls.

### HYPOTHESES

| ID | Claim | For | Against | Test | Cost | Status |
|---|---|---|---|---|---|---|
| H1 | GPU decode is submission-latency bound | dev 7.1 ms vs wait 99.1 ms | — | collapse to 1 submit | med | CONFIRMED |
| H2 | GPU prefill limited by per-element dequant | 4x less weight traffic = +6% only | — | DP4A kernel | med | TESTING (0.94 TFLOP/s, slower — unexplained) |
| H3 | CPU prefill limited by GEMM not being blocked for tokens | 39% of peak, 4-token blocking only | — | 8-token blocking | low | PENDING |
| H4 | CPU decode limited by sequential per-call overhead | 113 calls/token, 2x floor | — | batch more matvecs | med | PENDING |

### ACTIONS

- [ ] Re-enable `layer_decode_pipeline` with per-buffer resource-state tracking (H1)
- [ ] Explain DP4A running 3x slower than fp32 before writing more variants (H2)
- [ ] Widen CPU GEMM token blocking from 4 to 8 (H3)
- [ ] Batch the CPU decode matvecs that still run individually (H4)

## [RUN-004] 2026-11-05T17:00:00Z — MODE: FULL — TARGET: correctness vs llama.cpp oracle + per-op GPU ground truth

### BASELINE (post RUN-003 fixes)

Same hardware/model. All 7 suites + determinism + runtime fixture PASS after this run's
changes (numbers in FINAL_STATUS.md). Pipeline decode is enabled and device-removal-free.

### METHOD — per-op in-stream timestamps

`ZEN_D3D12_OPSTAMPS=1` stamps each op into one GPU query heap resolved after the
submission (stamp slots: 0 global, 1+2li/2+2li layer pairs, 33 end, 41..57 layer-0 ops,
34/35 probe, 58/59/60 copies/transitions; resolve width 64, map 512 B). Unlike the RTT
micro-benches, these measure GPU execution inside a real submission.

**Methodology caveat (confirmed):** RTT-style benches measure ~70-90 µs of fence/queue
round-trip latency per submission, not GPU work. In-stream timestamps are the only
ground truth: 1024 empty dispatches = 63 µs GPU ≈ 62 ns/dispatch; a dependent
write→read chain link costs ~4.9 µs (16 vs 8 links).

### MEASURED — layer-0 op table (decode, per-op stamps)

| Op | GPU time |
|---|---|
| kcopy / vcopy / state transitions | 11 / 1.6 / 0.3 µs |
| probe dispatch | 5 µs |
| q-gemv | 13.5-15.9 µs |
| gate / up / silu | 28-39 µs each |
| **ops-sum per layer** | **≈330 µs, constant across positions** |
| attn-half | ~226 µs (vs ~51 GB/s streaming need → ~150 µs/layer unexplained) |

BENCH_BARRIER variants v0-v9: copies and barriers are ≈ free — they are not the stall.

### RESULT — stage-1 fusion null

`ZEN_D3D12_FUSE=1` (qkv concat + gate/up concat, `ensure_fused_mat`): bit-identical
outputs, zero speedup. The "dispatch boundary count" theory is dead; fusion only pays
when it removes a measured cost.

### RESULT — correctness: RoPE pairing convention (outputs now match llama.cpp)

Oracle bracketing on greedy logits (llama-server 0.5.0-dev 7fe450e19, CPU-only, raw ids,
`add_special:false`; our `generate()` never prepends BOS): 1 token matched, 2 tokens
margins drifted 30%, 3 tokens collapsed (our argmax p=0.0034 vs llama " time" 0.099).
Root cause: llama.cpp llama-family uses **NORM rope — consecutive pairs (x[2i], x[2i+1])**;
ZenInfer used half-split NeoX (x[i], x[i+32]). Frequencies/base/eps were already right.
Internal parity could never catch this: both sides of every suite shared the convention.

Fix: `RuntimeConfig::rope_pairs_adjacent` (llama → true, default false for qwen),
threaded through host `rope()`, all call sites, the rope shader (`RopeParams.pad0` layout
selector), and decode/pipeline params. Also fixed: pipeline per-layer cos/sin tables were
sized/bound for one table (layer N read layer 0 / OOB; bit-identical outputs by luck of
identical tables, but allocation-illegal) — now one 256B-aligned slot per layer.

Validation: after [Once,upon,a] our top-6 = llama's top-6, same order, ≤6% margin error;
prompt [12805,5304,264,892] → all three paths (CPU / GPU per-op / GPU pipeline) emit
`11,220,16,13,20,3610,1274,304`, byte-identical to llama. 24-token stream: identical to
llama except one knife-edge flip at token 20 (p('202')=0.345 vs p('203')=0.322) that
re-synchronizes immediately — accumulation noise, not a defect. GPU per-op vs pipeline
bitwise equal; CPU vs GPU max_abs 1.0e-5.

### OPEN → RESOLVED (same day)

- CPU manual prefill attention diverges from the oracle at T=8 on [1..8] — **root-caused
  by the runtime-fix subagent and fixed**: NOT the KV append, NOT the flash gate, NOT
  parallel attention (all ruled out by experiment: `ZEN_D3D12_FLASH=0` unchanged,
  bit-identical repeats, GPU==oracle at T=8). The defect is in `gemm_q8_0_avx2`
  (kernels/avx2.cpp:299): the 4-token micro-block loaded `input+0..3*columns` instead of
  `input+(t+k)*columns` — every block after the first recomputed tokens 0-3's inputs
  while writing t-relative outputs. Explains the ladder exactly: T≤7 correct (one block
  + scalar tail), T=8 = first second-full-block length (max_abs 14.4), T=16 12/16 wrong.
  One-line fix; post-fix CPU top-8 at T=8 == oracle order, CPU==GPU 1.3e-05, all three
  paths byte-identical on the [1..8]x16 stream; remaining llama fork flips are 7%
  near-ties on a degenerate prompt (top-4 within 0.043-0.057) — noise, not defect.
  `ZEN_CPU_Q8_ACT` confirmed present (avx2.cpp:163, decode int8-act gate; +5.4% CPU
  prefill measured) — correcting the audit's "missing flag" note.

### PERF FINDINGS (perf-research subagent, same day)

**Attn-half stall root-caused: it is the KV-cache append copies, not attention.**
Layer-0 op stamps (pos=158): ops=368 µs ≈ device 362 µs/layer ✓; attn itself stamps
29.4 µs (~8%); the unexplained window is **cache-copy = 189.7 µs/layer** — two 2 KB
`CopyBufferRegion`s into the 256 MB default-heap kcache/vcache + 4 transitions
(d3d12_backend.cpp:4618-4649, same pattern in `attn_block_locked`). Dispositioned:
root-sig rebinds (<1 µs/layer, once-per-layer + 62 ns empty-dispatch floor), barriers
(0.2-0.3 µs stamps), UAV aliasing, descriptor heaps — all not the cause. Corroboration:
`kv_cache_append_batched` re-appends cost 46/52/97/170 ms one-time at 16/32/64/128-token
prompts (~30-45 µs per 2 KB copy + ~40 ms floor) — this also silently depresses
cache-hit decode benchmarks (70 tok/s headline vs ~115 honest). Stamp skew noted:
per-op attribution inside a layer is unreliable (down-gemv stamps 3.2 µs for 18.9 MB);
trust ops-sum / attn-half / ffn-half / window boundaries only.

Re-measured baselines: prefill 740-750 tok/s; decode pipeline 70.1 headline / 8.7 ms
wall per token (16.1 ms on the per-layer path, ~32 submits); GEMM t=128 3.04 TFLOP/s;
DP4A 1.02 TFLOP/s (still 3x slower — dead); tracing overhead ≈ 0% (5.81→5.80 ms).
CPU: prefill 270 tok/s (285 with ZEN_CPU_Q8_ACT=1, +5.4% — flag confirmed present in
the quant kernels, correcting the earlier audit note); decode 114 ms/token (2x worse
than the 02-11 61 ms/token record — power/thermal re-check warranted, thread count is
not the lever; governor already picks min(logical,physical)=16).

**Tuning plan (mechanism → expected decode tok/s):**
1. Write K/V to caches directly from compute (rope-k → kcache UAV, v likewise); delete
   both copies + 4 transitions per layer → removes ~190 µs/layer → ~175 tok/s. MEASURED cost.
2. Skip the device re-append when the prefix is already resident (generation counter,
   invalidate on reset/configure) → honest 115 becomes the headline; −170 ms per
   repeated-prompt server request. MEASURED cost.
3. Fuse LM head into the per-token submission + trim CPU fat (rope_table hoist out of
   the 16x/token loop, drop the 128 k-logit finite scan and the 513 KB logits copy,
   single-pass argmax) → ~240-260 tok/s. Costs measured, gain estimated.
4. GEMV lane-stride widening + 2 heads/wave64 attention → ~280-300 (estimated).
5. 450 tok/s additionally needs an overlapped double-buffered decode loop (record
   token N+1 while sampling N) — an orchestration project, not a kernel one;
   `generate_speculative` is the cheaper route to effective >400.
6. Prefill: move rope+append out of the CPU loop (−30 ms), SM 6.8 WaveMatrix f16 GEMM
   spike (FL 12.2 confirmed) or fixed DP4A (RGP capture first), flash-attn multi-token
   j-iterations → 740 → ~2500-4000 tok/s estimated; 5000-8000 needs item 6b fully.
7. CPU blocking 4→8/16: ≤6% upside — deprioritized. Micro: cache the
   `device_attention_available` answer (~6.3k calls/process on the prefill path).

Sequence 1→2→3 are independent and low-risk. Evidence: `tmp_perfA_trace2.txt`.

### ACTIONS

- [x] Thread RoPE pairing flag end-to-end; regenerate rope.cso (done this run)
- [x] Land the T=8 CPU prefill fix (avx2 gemm block indexing; verified vs oracle)
- [x] Metadata fallback now logs + fails load on unparseable study-critical numerics
- [x] Extend `zen_d3d12_rope_parity`/`attn_parity` with adjacent-pairs phases (3+2 shapes,
      all ≤1.79e-07) + runtime length-ladder determinism fixture — all PASS
- [x] Perf root cause + tuning plan appended (KV-append copies; plan items 1-7 above)
- [x] core_tests GEMM T-ladder guard (T∈{5,7,8,9,16} vs dequant-then-naive-dot, both
      worker counts) — closes the coverage gap that hid the avx2 block bug. Note: with
      ZEN_D3D12=1 the GEMMs route to the device, so CPU-kernel bugs are invisible in
      device-enabled runs; the isolating A/B was pure-CPU vs ZEN_D3D12=1+FLASH=0.
- [x] Implement tuning plan items 1-3 (KV direct-write, re-append skip, LM-head fuse)
      — LANDED with three integration defects found and fixed by the parent:
      (a) build_root_signature wrote params[6]/params[7] into a params[6] array — stack
      overrun, AV inside D3D12Core!ConvertRootSignature (cdb-traced); now params[8].
      (b) the backend-side resident-prefix SKIP in kv_cache_append_batched silently
      dropped legitimate overwrites (prefill re-appends different K/V; flash_parity
      tokens=1 and the fixture's stop-sequence test failed) — skip removed; the
      runtime's kv_cache_prefix_resident query is the only skip path. (c) the pipeline's
      shared attention cbuffer slot (kAttnParamOffset) carried per-layer kv_base/v_base;
      the params buffer is read at execution time so every layer attended with layer
      15's slice offsets (garbage decode in all kill-switch modes) — per-layer slots at
      8192+li*256, buffer 8192→24576 bytes; fused-mat use made all-or-nothing for the
      same class of bug. Result: 115 → 146.6 tok/s pipeline decode (8.75 → 6.82 ms/token),
      all suites + oracle byte-exactness green. LMFUSE wired runtime-side by the parent
      (11-arg layer_decode_pipeline overload + host rms/matvec skip when fused).
- [ ] Item 4 (GEMV lane widening + 2 heads/wave64) and item 5 (overlapped decode loop)
      — the remaining path toward 300+ / 450 tok/s.

## [RUN-005] 2026-11-06 — MODE: HIGH — TARGET: D3D12 DP4A decode + prefill GEMM

### BASELINE
- Hardware: RX 9070 XT; model Llama-3.2-1B-Q8_0.
- fp32 decode 263 tok/s; DP4A 251 tok/s (after staging hill); prefill ~744 tok/s; GEMM 3.0 TFLOP/s.

### FINDINGS
- DP4A 205 → 251 tok/s via groupshared int8 staging + UAV-only `qact` (bit-identical).
- Isolated GEMV ceiling 359 GB/s (lmhead bench); decode is kernel-bandwidth-bound.
- fp32 staging neutral (264 vs 263 tok/s).
- GEMM v2 (256-thread/16-tile) −25% — falsified.
- DML FP16 GEMM dead end: DEVICE_REMOVED at recorder, 0x887a0005 (`tools/d3d12_dml_probe.cpp`).
- dxc 1.9.2609 max SM 6.9 (no 6.10 LinAlg HLSL); T=8 GEMM 0.68 TFLOP/s (32-tile waste).

### HYPOTHESES
- H-scalar-GEMM-tiling: FALSIFIED (GEMM v2 −25%).
- H-DML-WMMA: DEAD-END (driver, DEVICE_REMOVED at recorder).
- H-speculative: PENDING — only credible route to effective 450+ tok/s.

### ACTIONS
- [x] Mark GEMM v2 as evaluated default-off (`ZEN_D3D12_GEMM_V2=1`).
- [x] Keep DML probe as diagnostic only (`tools/d3d12_dml_probe.cpp`).
- [ ] Next: producer-fused quantize, CPU-rope GPU move, speculative decode.

## [RUN-006] 2026-11-06 — MODE: FULL — TARGET: speculative verify bottleneck (450+ credibility)

### BASELINE
- Hardware: RX 9070 XT (0x7550, 15.7 GB VRAM, FL 12.2, ReBAR=1, Agility 722); model Llama-3.2-1B-Q8_0 (16 layers, emb 2048, 32H/8KV, head_dim 64, vocab 128256, 1.32 GB). Only GGUF in `H:\llamadx\models\` and repo is the target; no Qwen-0.5B / TinyLlama draft anywhere.
- Ordinary (per-layer path, pipeline off, prompt 4, greedy 32): decode 155-166 ms / 31 forwards = ~5.0-5.4 ms/token, 186-199 tok/s this run (RUN-005 reported fp32 warm 263 tok/s; run-to-run clock variance). Runtime fixture (128-token prompt): prefill ~160-166 ms (~770-800 tok/s), decode ~240 ms /31 (~124-129 tok/s).
- Instrumentation floor: 0.10 us/scope this run (`zen_generate_ids`: 0.102 us; cf. RUN-003 0.062 us). Stateless, negligible vs ms-scale GEMM.

### TRACE (current speculative flow)
- `generate_speculative` (dense_transformer_runtime.cpp:1763): requires both models loaded + same vocab; prefill both via `forward_batch(prompt)`; per step: `P-1` draft `forward()` + one target `forward_batch(proposals, start_pos, P, &verify_logits)` (T=P, not P+1; `target_before` holds the P-th prior) + 1 target `forward(correction|bonus)` + 1 draft `forward(correction|bonus)` when continuing.
- `forward_batch` (T>1) is the verify path: per layer 7× `tensor_gemm` (q,k,v,attn_output,gate,up,down) = 112 single submits, each → `quant::gemm` → `d3d12::gemm_q8_0` (or CPU AVX2 if GPU declines). Decode (`forward()`, T=1) uses `layer_decode` (1 submit/layer). With full-context KV sizing the verify also missed `flash_attention`/device attention (see fix below).
- Measured with self-draft (same GGUF as draft, greedy, P=4, 32 tokens): `proposed=26 accepted=26 rejected=0 accept_rate=1.0`, `spec decode 507-683 ms` (before/after KV fix) vs ordinary `155-166 ms`, `speedup 0.24-0.36×`. Correct (`spec_ids == ord_ids`), but slower: draft cost = target cost (5 ms) so the verify + correction dominate. Per verify ≈ 97.6 ms before KV fix, 72.4 ms after (draft 5×5 ms + verify ~47 ms). This is the verify bottleneck quantified end-to-end, not just isolated GEMM.
- Effective formula (greedy): `tok/step = accepted+1`, `step_time = P·t_draft + t_verify(T=P) + t_target + t_draft(follow)`, `speedup = E[tok/step]·t_target / step_time`, `effective = ordinary · speedup`. With self-draft `t_draft = t_target = 5 ms`, `t_verify ≈ 47 ms (T=4, V1)`: speedup <1 always. A tiny draft with `t_draft ≈ 2 ms` and `α=0.6, P=4 (E[a]=1.31, tok/step=2.31)` gives step ≈ 8+47+7 = 62 ms → 2.3 tokens/62 ms = 37 tok/s — still slower. Even at the prefill-ceiling 3 TFLOP/s, `t_verify(T=4) ≈ 4.3 ms` (12.8 GFLOP all matmuls) leaves step ≈ 8+4.3+7 = 19.3 ms → 2.3/19.3 = 119 tok/s — only reaches 450+ if draft is ~0.5 ms and acceptance ≥0.7 and verify is batched.

### FINDINGS
- **Verify is 9-14× one decode token.** Isolated GEMM (`zen_d3d12_gemm_bench`, 4 big matmuls/layer, best-of-5): T=4 V1 0.42, T=8 0.80, T=16 1.32, T=32 2.37, T=128 2.99 TFLOP/s. T=8 was 0.77-0.80, matching RUN-005's 0.68 (32-tile waste). At T=8 each GEMM is 0.15-0.33 ms; ~70 us fixed submit is ~47% of the call.
- **V3 (BLOCK_T=8) helps 1.6-1.7× at verify T but does not hit 2 TFLOP/s.** Same K order as V1 (chunks outer → blocks inner, same scale fold, warp reduction), 1024 threads, 32 rows, CHUNK 8 (8 KB groupshared). V3: T=4 0.70, T=8 1.38, T=16 1.89, T=32 2.29, T=128 2.75 TFLOP/s. attn_q (2048 rows) stays 0.3-0.5 TFLOP/s at T=4-8 — split-K starvation, not tile, is the remaining ceiling. V2 at T=8 was 0.91 (still falsified vs V3 1.38).
- **KV sizing bug disabled the device verify path.** Speculative sized caches to the full advertised context (`context*…`, GBs) so `device_attention_available` (`needed > 8192`) rejected the device path and verify fell back to CPU attention + no flash. Fixed to `prompt+max_new` (parity with `generate()`): spec decode 683→507 ms. Still slower than ordinary (draft = target), but the GPU path is now live for verify.
- **Draft asset gap blocks the shipped path.** No draft GGUF exists. Path is wired (`--draft-model`, `speculative_tokens`, counters `proposed/accepted/rejected`) and verified greedy-exact with self-draft, but effective 450+ cannot be measured without a real tiny draft. SD/Medusa self-spec: `forward()` has no layer-subset/early-exit API and the D3D12 KV cache is a singleton (layers/shape clash if draft ≠ target); CPU-only early-exit is feasible with a `layer_limit` param, GPU needs dual KV. DP4A verify is not a lever (quantize already once per batch; DP4A prefill 0.94 TFLOP/s is slower and changes math).

### ACTIONS
- [x] Land V3 verify GEMM: `shaders/q8_0_gemm_v3.hlsl` (BLOCK_T=8, CHUNK 8, ROWS 32, 1024 threads, K-order identical to V1) behind `ZEN_D3D12_GEMM_V3=1` (default off); `pso_gemm_v3` + single/batched dispatch with `tile=8` (DP4A > V3 > V2 > V1), `CMakeLists` + `shutdown()` cleanup. Backup `backup/spec-2026-11-06/`.
- [x] Land speculative KV sizing fix (visit-sized caches) — restores device attention/flash for verify.
- [x] Add `zen_speculative_timing` harness (ordinary vs self-draft counters + effective tok/s); greedy stream bit-exact in both modes.
- [x] Validate: `zen_d3d12_gemm_parity` V1==V3 (9.0e-08 / 1.02e-07 / 1.64e-07 / 1.87e-07, identical printed values ⇒ Δ≤2e-7), `zen_core_tests` PASS, `zen_runtime_tests` PASS, oracle 32-token stream `11,220,16,13,…19,13` identical V1/V3.
- [ ] Next for 450+: (1) batched verify submits (fewer than 112 — QKV/gate-up batches already exist but `forward_batch` still calls per-tensor `tensor_gemm`), (2) split-K for 2048-row matmuls, (3) real tiny draft GGUF (same vocab) to measure `α` + `t_draft` for the speedup formula above. Do not claim 450+ until measured.

## [RUN-007] 2026-11-06 — MODE: FULL — TARGET: 2-heads-per-wave64 decode attention (attention2.hlsl)

### CONTEXT
- Decode attention was 1 head per 32-thread group (WARPS_PER_GROUP=1, grid=heads=32, rebased shuffle). 4-heads/group measured worse. Long-context decode unmeasured at pos>~158 until now.

### TRACE (measured, pipeline path, Llama-3.2-1B 32H/8KV/64d, op/window GPU stamps + wall)
- attn-half window (total over 16 layers): pos~4: 777us; pos~40: 1316us; pos 512: 10906us; pos 1024: ~19580us; pos 2048: ~35900us.
- Slope ~1.1us/pos/layer (0.034us/pos/head); attn-half share of device: 25% @40 → 82% @512 → 90% @1024 → 94% @2048. Wall decode: 248 tok/s @40 → 60 @512 → 39 @1024 → 24 @2048 (pipeline / per-layer consistent).
- CONCLUSION: attention dominates long-context decode by exhaustion — it is the only O(pos) work in the attn-half window (QKV/rope/append/o-gemv are pos-independent) and ffn-half stays ~2.3ms total across all contexts.
- Op-stamp caveat (extends RUN-004): at long context the per-op "cache-copy" stamp absorbs ~2.3ms/layer while the attn kernel stamp is only 35-45us/layer. KVDIRECT=0/1 wall is identical (97.7 vs 99.7ms @512), so it is not copies; attribution inside the attn-half window is unreliable despite being the right window. The growth is real (device total matches wall); the per-op split is not.
- The attn kernel itself: 13.2us @~5, 31-35us @512, 43.8us @1024, ~40us @2048 (layer0 op stamp).

### DESIGN (implemented)
- New shaders/attention2.hlsl (Attention2, numthreads 64, grid=ceil(heads/2)): lanes 0-31 head 2*gid, lanes 32-63 head 2*gid+1; same K/V UAV layout, kv_base/v_base, same softmax.
- NOT WaveActiveSum: on wave64 it folds both heads into one sum (same bug class as the q8_0_gemv wave64 bug). Kept the identical 32-lane rebased shuffle tree per half -> bit-identical by construction.
- Backend: pso_attn2 + use_attn2() gate wired into the 3 non-fuseq attention dispatch sites (legacy attn_block_locked, layer_decode_pipeline, layer_decode). ZEN_D3D12_ATTN2=1 (default off); ZEN_D3D12_ATTN2_MINPOS=N optional context gate (default 0). DP4A fuseq attention (attention_fuseq, own 1-head grid) untouched. Flash untouched (already 1 wave per (token,head); 2-head prefill would need a new token/head mapping — nontrivial, and prefill is GEMM-bound at 3 TFLOP/s, not attention-bound).
- Kill switch: ZEN_D3D12_ATTN2 unset/0 = old path byte-identical. Backup: backup/attn2-2026-11-06/.

### FINDINGS — attention2 is a wash (falsified as speedup)
- attn kernel stamp, attention2 vs attention: short 13.2 vs 13.1us; 512: ~33 vs ~31-35us; 2048: ~44 vs ~40us.
- Device total: attn2 ~0% @short, +2.3% @512, +2.4% @2048 (512: 13.1 vs 12.8ms; 2048: 39.2 vs 38.3ms). Wall 2048: 25.44 vs 25.91 tok/s.
- Mechanism: same #threads, same serial softmax per head, same UAV traffic, same barrier stall window -> wave packing removes nothing. No crossover exists (attn2 never measurably better), so NO context gate was added; keeping pos>128 would only expose the flat-negative kernel at long context.

### VALIDATION
- zen_d3d12_attn_parity: PASS both gates, identical max_abs_err (5.96e-08 / 1.19e-07 / 1.19e-07 / 1.79e-07).
- Oracle stream [12805,5304,264,892] -> 11,220,16,13,20,3610,1274,304 byte-identical: attn2 on/off x pipeline/per-layer (4 runs).
- ctest: 7/7 pass (all 6 d3d12 + core) both gates. (zen_runtime_qwen2_synthetic fails identically with gate on/off: missing GGUF fixture, pre-existing.)
- GEMV/DP4A/speculative/rope untouched.

## [RUN-008] 2026-11-06 — MODE: FULL — TARGET: producer-fused quantize (DP4A decode)
### TRACE
- Per-token dispatches (decode, DP4A, default unfused): 19/layer (15 base + 4 quantize) x16 + 2 LM = 306 in 1 submit (pipeline) or 16 submits (legacy, default ship path). Quantize = 64/token, GEMV = 7/layer x16 = 112, rest = 130. UAV barriers 6/layer x16 = 96 (4 quantize-ordering + 2 KV/attn). 32-token legacy counters: unfused 9568 vs fused 7584 dispatches (delta 1984 = 64 x 31 decode forwards), submits 753 both.
- OPSTAMPS: quantize cost is inside the gemv buckets (stamp_op layout), per-op attribution unreliable (RUN-004 caveat). Reasoned floor: empty dispatch 62ns, dependent write->read link ~4.9us (RUN-004); 64 tiny 1-group kernels (64-256 blocks) are launch/link-bound, not bandwidth-bound. Barrier retention is required (UAV producer write must order before GEMV read).
### DESIGN (implemented, default off)
- New shaders/rmsnorm_fuseq.hlsl (RmsNormFused), silu_fuseq.hlsl (SiluMulFused), attention_fuseq.hlsl (AttentionFused): each writes fp32 (u0, unchanged) + qact int8 (u3) + scales (u4), bit-identical to quantize_acts.hlsl (amax/127, clamp, half-away lround). RMS reads back written fp32 after DeviceMemoryBarrier; SiLU is block-thread with register values; Attention is lane-0 per head (needs head_dim%32==0, else O site falls back). Root sig 8->10 params (u3/u4); non-fuseq shaders ignore. Host ZEN_D3D12_DP4A_FUSEQ=1 (needs DP4A=1) in pipeline + legacy; barrier kept, dispatch removed; LM head stays fp32. GEMV tiling, attention math, prefill untouched.
### FINDINGS — correctness yes, speedup no (keep fuse off)
- zen_d3d12_dp4a_parity: fp32 rel 3.98e-07 both gates; DP4A rel 1.4503e-02 IDENTICAL fused vs unfused (PASS <=2e-2).
- Real model [12805,5304,264,892] x8/16/32: fused vs unfused DP4A streams token-identical all 32 (11,358,574,... first token 11 matches fp32 oracle; later drift is DP4A approximation, same both gates). 32-token: fused 173.6 tok/s vs unfused 138.5 cold pair, but warm 16-token A/B removes it: unfused 174-175 vs fused 167-173 (noise), pipeline device ~3.20 vs ~3.29 ms/token. Verdict: dispatch -64/token, tok/s delta ~0 (fused slightly slower, within contention noise).
- Why: decode is GEMV weight-bandwidth bound (359 GB/s ceiling, RUN-005); orchestration ~4%. Fused producers add work (RMS readback, SiLU 32->1 group, attn lane-0 serial) that offsets the dispatch saving. FUSE=1 was already null (RUN-004 stage-1 fusion null) for the same reason.
### VALIDATION
- All 6 d3d12 suites + core/iq/fuzz/governance/bench PASS default-off and FUSEQ-on (synthetic dp4a). zen_runtime_qwen2_synthetic fails pre-existing (missing GGUF fixture, identical gate on/off).
- Kill switch: FUSEQ unset/0 = unfused DP4A (default); DP4A unset/0 = fp32 parity default. Backup: backup/producer-2026-11-06/.
- Note: `tools/d3d12_rope_prefill_bench.cpp` placeholder landed to satisfy a concurrent CMake target whose source had not landed yet; replace when real bench lands.
### ACTIONS
- [ ] Do not enable FUSEQ by default; next lever is GEMV lane-spec (item 4) / speculative draft asset, not more fusion.

## [RUN-009] 2026-11-06 — MODE: HIGH — TARGET: single-submit decode pipeline default-on

### BASELINE (per-layer, ZEN_D3D12_PIPELINE=0)
- Hardware: RX 9070 XT, ReBAR=1, Agility 722; model Llama-3.2-1B-Q8_0 (16L, 2048H, 32H/8KV, 64d, vocab 128256)
- Workload: prompt "Once upon a time" greedy 8, and oracle [12805,5304,264,892] greedy 8
- Per-layer: 369 submits / 7 forwards (~17/token), per_submit 168us (queue 2us, fence 165us), decode 43.4ms/7 = 6.2ms/token, 161 tok/s
- Pipeline (ZEN_D3D12_PIPELINE=1): 264 submits, per_submit 190us (queue 1us, fence 189us, single 3.3ms GPU submit), decode 33.5ms/7 = 4.8ms/token, 212 tok/s
- Instrumentation floor: negligible vs ms-scale

### FINDINGS
- Per-layer submission latency dominates fence wait; pipeline collapses 16 submits to 1
- LM fuse on/off is neutral (212 vs 217 tok/s); orchestration is not the bandwidth ceiling

### ROOT CAUSE
- Historical per-layer pipe_lists design recorded transitions into the wrong (closed) s.list; GPU ran with stale states and seeded runs diverged. Current backend line 4977-4982 already records every layer into the OWNING list (list = s.list.Get()) with hin staging and full UAV/NSR/COPY cycling; every arena returns to entry state per layer. Caller comment was stale. Parity gate green (below), default flipped on.

### VALIDATION
- Determinism: same seed twice identical ids both modes (11 304 264 2678)
- Decode logits pipeline vs per-layer: max_abs 1.9e-06 rel 2.1e-07 (gate <=1e-6 PASS), top5 identical
- Oracle [12805,5304,264,892] -> 11,220,16,13,20,3610,1274,304 identical pipeline/per-layer
- ctest 7/7 d3d12+core green; no DRED DEVICE_REMOVED
- Tok/s: 161 (per-layer) -> 209 (pipeline default) this run, +30%
- Default: ON (ZEN_D3D12_PIPELINE unset/1); kill switch ZEN_D3D12_PIPELINE=0 restores per-layer. Backup backup/pipe-2026-11-06/
### ACTIONS
- [x] Flip ZEN_D3D12_PIPELINE default ON
- [ ] Next: 480 tok/s needs GEMV weight-bandwidth/kernel work, not orchestration


## [RUN-010] 2026-11-06 — MODE: HIGH — TARGET: GPU prefill batched GEMM (QKV + gate|up)

### BASELINE (before batch, ZEN_D3D12_BATCH unset/0)
- Hardware: RX 9070 XT (RDNA4, ReBAR=1, Agility 722); model Llama-3.2-1B-Q8_0 (16L, 2048H, 32H/8KV/64d)
- Workload: zen_generate_ids, warm runs (run_1..2), prompt length T exact
- T=128: prefill 162-183 ms (765-787 tok/s), submits=144/prefill, dispatches=113 (112 GEMM + 1 LM GEMV), wait=92-96 ms, batches=0, quant.gemm 112 ops = 94-99 ms
- T=171: prefill 228-238 ms (718-751 tok/s), same 144 submits, wait=122-128 ms, quant.gemm 112 ops = 128-135 ms
- Decomposition per prefill: 112 GEMM submits (7/layer x16) + 16 flash_attention + 16 kv_cache_append_batched = 144; dispatches 113
- Instrumentation floor: 0.11 us/scope (negligible)

### TRACE (per-submit overhead)
- Total fence wait / submits: T128 92 ms/144 = 639 us; T171 126/144 = 875 us. Most fence time is GPU kernel work itself (gemm_bench T128: ffn_gate 1.50 ms, attn_q 0.47 ms, TOTAL 4.72 ms/14 GFLOP = 2.96 TFLOP/s).
- Marginal submit overhead is the gap between kernels, not the full fence. It is ~70-200 us/submit at T128 (measured saving /48 removed submits), not 0.7-1.1 ms. At small T (verify T=4) it is larger fractionally but absolute wall saving is only ~1-3 ms.
- Expected batch saving ceiling: 48 removed submits x ~0.2 ms = ~10 ms at T128/171, ~2-5 ms at T=4-32.

### IMPLEMENTATION
- quant::gemm_batch() (include/zen/quant.h, src/quant.cpp): tries d3d12::gemm_q8_0_batch when ZEN_D3D12_BATCH=1, q8_0, scale==1, count<=8, backend enabled, DP4A off; else loops quant::gemm() per task (GPU single + AVX2/CPU). Route counters add count per batch attempt.
- tensor_gemm_batch() (dense_transformer_runtime.cpp anonymous namespace): T==1 -> tensor_matvec_batch; otherwise planner prefill + same-encoding check -> quant::gemm_batch, else per-tensor tensor_gemm loop (FP32 fallback preserved).
- forward_batch wiring only: Q/K/V 3->1, gate/up 2->1; O/down stay separate (attention/SiLU dependency). T==1 decode unchanged. Default gate off preserves old path.
- Backend guards (d3d12_backend.cpp gemm_q8_0_batch): mirror single-gemm gates - gemm_opted_in, kMinElements per task, DP4A decline (batch is fp32 only). Prevents bypassing GEMM_EXPERIMENTAL=0.
- Parity harness (tools/d3d12_gemm_parity.cpp): added check_gemm_batch (QKV 2048/512/512 + gate/up 8192/8192, T=17) comparing batch vs single bitwise and vs double reference.

### RESULTS (ZEN_D3D12_BATCH=1, warm runs)
- Submits/prefill: 144 -> 96 (GEMM 112 -> 64, -48). Batches counter: 0 -> 32 per prefill (96 across 3 runs). Dispatches unchanged 113.
- T=128: 162-183 ms -> 155-166 ms (-7 to -19 ms, -5 to -10%), 765-787 -> 802-821 tok/s (+5-7%). wait 92-96 -> 83-85 ms (-9 to -11 ms).
- T=171: 228-238 ms -> 215-229 ms (-10 to -13 ms, -5%), 718-751 -> 785-792 tok/s (+5%). wait 122-128 -> 114-120 ms.
- Small T: T=4 50.8->50.1 ms (-0.7), T=8 53.8->50.1 (-3.7), T=32 58.8->54.7 (-4.1) ms.
- quant.gemm ops: 112 (94 ms) -> gemm_batch 32 (56 ms) + gemm 32 (29 ms)=85 ms at T128; 135 -> 77+42=120 ms at T171.
- DP4A A/B: B0 vs B1 identical stream under DP4A=1 (11,358,574,...) because batch skips DP4A and calls single path.

### VALIDATION
- gemm_parity: single rel 9.0e-08 / 1.02e-07 / 1.64e-07 / 1.87e-07; batch_vs_single 0.000e+00 for all 5 tasks, batch rel 1.4e-07 / 1.65e-07 / 1.77e-07 / 1.64e-07 / 1.74e-07. PASS.
- Greedy stream [12805,5304,264,892] x32: GPU B0 == GPU B1 == CPU B0 == CPU B1 == 11,220,16,13,20,3610,1274,304,279,...,19,13.
- zen_core_tests PASS; ctest gemm_parity + core_tests PASS. zen_runtime_tests with real GGUF PASS (prefill 182ms median this run).
- Default gate off: ZEN_D3D12_BATCH unset/0 = old loop, bit-identical. Backup: backup/gpubatch-2026-11-06/

### ACTIONS
- [x] Land batch wiring default-off; keep gate off until perf accepted (this run shows +5-10% prefill, 48 fewer submits, no numerics change)
- [ ] Next levers for prefill (rope 27-36 ms and silu 9-13 ms are now 17% and 6%; GEMM still 57% - kernel TFLOP/s, split-K for 2048-row, and KV/flash submits remain 32/prefill) - not more orchestration.


## [RUN-011] 2026-11-06 — RMS AVX2 default-ON correctness vs scalar reference

### QUESTION
User review: "RMS AVX2 default-ON changes the accumulation from double to float. Confirm greedy output against the scalar reference on at least two models, or document the drift."

### CODE FACT (no code change)
`kernels/avx2.cpp:rms_norm_avx2` accumulates `ss` in `__m256d` (double) lanes (`s0..s3`), combines once, then tails in double; `inv = 1.0f/sqrt(float(ss/n)+eps)` exactly as scalar. So accumulation is NOT float — it stays double. The only difference vs `dense_transformer_runtime.cpp:rms_norm` scalar loop is associativity of the double partial sums (4 parallel 4-wide chunks + 4-lane horizontal vs strictly sequential). The predicted drift is <= ~1 ulp of float on `y` (documented in the kernel comment), not an accumulation-precision change.

### METHOD
- CPU path only (ZEN_BACKEND unset; ZEN_D3D12 unset). ZEN_CPU_RMS_AVX2=0 (scalar) vs =1 (AVX2 default), same model/prompt/greedy.
- Tool: `zen_generate_ids` (greedy generated_ids, 32 new tokens) and `zen_decode_parity_dump` (last_logits max_abs over full vocab).
- Prompt: "The quick brown fox" (text) for generated_ids; ids [12805,5304,264] for logits dump (max_new=8).

### EVIDENCE
| Model | Arch | Prompt | max_new | generated_ids RMS=0 vs 1 | last_logits max_abs_diff | Match |
|---|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct-Q8_0 (H:/OLLAMA-Models/GGUF) | llama | "The quick brown fox" | 32 | identical (32 ids) | 0.0 (vocab 128256) | yes |
| Qwen3-4B-Q8_0 (G:/More-models) | qwen3 | "The quick brown fox" | 32 | identical (32 ids) | 0.0 (vocab 151936) | yes |
| Qwen2.5-Coder-7B-Instruct-Q5_K_M (G:/More-models) | qwen2 | "The quick brown fox" | 32 | identical (32 ids) | 0.0 (vocab 152064) | yes |

Additional: `tools/rms_bench` (pseudo-random rows, hidden 64/2048/8192, T up to 1024) reports maxerr 0.0 PASS at 1e-5 tolerance; speedup 2.5-4.8x.

### FINDINGS
- Greedy generated_ids are byte-identical across 32 tokens on 3 models spanning llama/qwen2/qwen3, at T up to 32 (8..32 covered).
- last_logits max_abs_diff = 0.0 on all three (full vocab), i.e. no measurable drift in end-to-end logits despite the associativity difference.
- The predicted <=1-ulp float drift does not materialise as a distinct float value in these runs; rounding lands on the same float.

### RECOMMENDATION
Keep ZEN_CPU_RMS_AVX2 default ON. No tolerance gate or kill-switch change needed. The user's premise that the AVX2 path "changes accumulation from double to float" is incorrect: it accumulates in double (`__m256d`), and the only difference is lane-order of the double sum, which measured 0.0 max_abs on real models. If a future model/activation distribution ever shows nonzero drift, it is bounded to ~1 ulp float on `y` and is already inside `core_tests` tolerance; gate by `ZEN_CPU_RMS_AVX2=0` only if bit-exact reproducibility is required.


## [RUN-011] 2026-11-06 — RMS AVX2 default-ON correctness vs scalar reference

### QUESTION
User review: "RMS AVX2 default-ON changes the accumulation from double to float. Confirm greedy output against the scalar reference on at least two models, or document the drift."

### CODE FACT (no code change)
`kernels/avx2.cpp:rms_norm_avx2` accumulates `ss` in `__m256d` (double) lanes (`s0..s3`), combines once, then tails in double; `inv = 1.0f/sqrt(float(ss/n)+eps)` exactly as scalar. So accumulation is NOT float — it stays double. The only difference vs `dense_transformer_runtime.cpp:rms_norm` scalar loop is associativity of the double partial sums (4 parallel 4-wide chunks + 4-lane horizontal vs strictly sequential). The predicted drift is <= ~1 ulp of float on `y` (documented in the kernel comment), not an accumulation-precision change.

### METHOD
- CPU path only (ZEN_BACKEND unset; ZEN_D3D12 unset). ZEN_CPU_RMS_AVX2=0 (scalar) vs =1 (AVX2 default), same model/prompt/greedy.
- Tool: `zen_generate_ids` (greedy generated_ids, 32 new tokens) and `zen_decode_parity_dump` (last_logits max_abs over full vocab).
- Prompt: "The quick brown fox" (text) for generated_ids; ids [12805,5304,264] for logits dump (max_new=8).

### EVIDENCE
| Model | Arch | Prompt | max_new | generated_ids RMS=0 vs 1 | last_logits max_abs_diff | Match |
|---|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct-Q8_0 (H:/OLLAMA-Models/GGUF) | llama | "The quick brown fox" | 32 | identical (32 ids) | 0.0 (vocab 128256) | yes |
| Qwen3-4B-Q8_0 (G:/More-models) | qwen3 | "The quick brown fox" | 32 | identical (32 ids) | 0.0 (vocab 151936) | yes |
| Qwen2.5-Coder-7B-Instruct-Q5_K_M (G:/More-models) | qwen2 | "The quick brown fox" | 32 | identical (32 ids) | 0.0 (vocab 152064) | yes |

Additional: `tools/rms_bench` (pseudo-random rows, hidden 64/2048/8192, T up to 1024) reports maxerr 0.0 PASS at 1e-5 tolerance; speedup 2.5-4.8x.

### FINDINGS
- Greedy generated_ids are byte-identical across 32 tokens on 3 models spanning llama/qwen2/qwen3, at T up to 32 (8..32 covered).
- last_logits max_abs_diff = 0.0 on all three (full vocab), i.e. no measurable drift in end-to-end logits despite the associativity difference.
- The predicted <=1-ulp float drift does not materialise as a distinct float value in these runs; rounding lands on the same float.

### RECOMMENDATION
Keep ZEN_CPU_RMS_AVX2 default ON. No tolerance gate or kill-switch change needed. The user's premise that the AVX2 path "changes accumulation from double to float" is incorrect: it accumulates in double (`__m256d`), and the only difference is lane-order of the double sum, which measured 0.0 max_abs on real models. If a future model/activation distribution ever shows nonzero drift, it is bounded to ~1 ulp float on `y` and is already inside `core_tests` tolerance; gate by `ZEN_CPU_RMS_AVX2=0` only if bit-exact reproducibility is required.


## [RUN-011] 2026-11-06 — RMS AVX2 default-ON correctness vs scalar reference

### QUESTION
User review: "RMS AVX2 default-ON changes the accumulation from double to float. Confirm greedy output against the scalar reference on at least two models, or document the drift."

### CODE FACT (no code change)
`kernels/avx2.cpp:rms_norm_avx2` accumulates `ss` in `__m256d` (double) lanes (`s0..s3`), combines once, then tails in double; `inv = 1.0f/sqrt(float(ss/n)+eps)` exactly as scalar. So accumulation is NOT float — it stays double. The only difference vs `dense_transformer_runtime.cpp:rms_norm` scalar loop is associativity of the double partial sums (4 parallel 4-wide chunks + 4-lane horizontal vs strictly sequential). The predicted drift is <= ~1 ulp of float on `y` (documented in the kernel comment), not an accumulation-precision change.

### METHOD
- CPU path only (ZEN_BACKEND unset; ZEN_D3D12 unset). ZEN_CPU_RMS_AVX2=0 (scalar) vs =1 (AVX2 default), same model/prompt/greedy.
- Tool: `zen_generate_ids` (greedy generated_ids, 32 new tokens) and `zen_decode_parity_dump` (last_logits max_abs over full vocab).
- Prompt: "The quick brown fox" (text) for generated_ids; ids [12805,5304,264] for logits dump (max_new=8).

### EVIDENCE
| Model | Arch | Prompt | max_new | generated_ids RMS=0 vs 1 | last_logits max_abs_diff | Match |
|---|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct-Q8_0 (H:/OLLAMA-Models/GGUF) | llama | "The quick brown fox" | 32 | identical (32 ids) | 0.0 (vocab 128256) | yes |
| Qwen3-4B-Q8_0 (G:/More-models) | qwen3 | "The quick brown fox" | 32 | identical (32 ids) | 0.0 (vocab 151936) | yes |
| Qwen2.5-Coder-7B-Instruct-Q5_K_M (G:/More-models) | qwen2 | "The quick brown fox" | 32 | identical (32 ids) | 0.0 (vocab 152064) | yes |

Additional: `tools/rms_bench` (pseudo-random rows, hidden 64/2048/8192, T up to 1024) reports maxerr 0.0 PASS at 1e-5 tolerance; speedup 2.5-4.8x.

### FINDINGS
- Greedy generated_ids are byte-identical across 32 tokens on 3 models spanning llama/qwen2/qwen3, at T up to 32 (8..32 covered).
- last_logits max_abs_diff = 0.0 on all three (full vocab), i.e. no measurable drift in end-to-end logits despite the associativity difference.
- The predicted <=1-ulp float drift does not materialise as a distinct float value in these runs; rounding lands on the same float.

### RECOMMENDATION
Keep ZEN_CPU_RMS_AVX2 default ON. No tolerance gate or kill-switch change needed. The user's premise that the AVX2 path "changes accumulation from double to float" is incorrect: it accumulates in double (`__m256d`), and the only difference is lane-order of the double sum, which measured 0.0 max_abs on real models. If a future model/activation distribution ever shows nonzero drift, it is bounded to ~1 ulp float on `y` and is already inside `core_tests` tolerance; gate by `ZEN_CPU_RMS_AVX2=0` only if bit-exact reproducibility is required.
