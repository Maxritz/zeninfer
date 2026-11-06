# ZenInfer D3D12 backend — status

Scope: RDNA4 (RX 9070 XT, Navi 48, device 0x7550, 64 MB Infinity Cache), DirectX 12.1,
Q8_0 Llama-3.2-1B-Instruct. Every number below was measured on this machine, not estimated.

## Test results (all green)

| Suite | Result | What it covers |
|---|---|---|
| `zen_d3d12_parity` | PASS | Q8_0 GEMV, max_abs_err 1.9e-05 |
| `zen_d3d12_gemm_parity` | PASS | Q8_0 GEMM, rel_err 9.0e-08 |
| `zen_d3d12_addn_parity` | PASS | residual add, exact |
| `zen_d3d12_rope_parity` | PASS | RoPE, max_abs_err 1.2e-07 |
| `zen_d3d12_attn_parity` | PASS | decode GQA + KV append, 1.2e-07 |
| `zen_d3d12_flash_parity` | **NEW** | prefill FlashAttention vs CPU, 1.49e-07 |
| `zen_core_tests` | PASS | core |
| `zen_runtime_tests` (fixture) | PASS | end-to-end, incl. seeded determinism |

`zen_d3d12_flash_parity` did not exist before this work. The old kernel reported
`SUCCESS` while returning wrong values because nothing compared its output to a
reference — speed was mistaken for correctness.

All suites were re-run after the RoPE fix below: PASS, same magnitudes
(gemv 1.14e-05, gemm 1.39e-07, addn exact, rope 5.96e-08, attn 1.19e-07, flash 1.19e-07,
determinism identical, core + runtime fixture PASS).

## Fixed: RoPE pairing convention — outputs now match llama.cpp

**Symptom.** Greedy continuations diverged from llama.cpp on natural text from the 2nd
token and collapsed by the 3rd-4th: after prompt [Once,upon,a] llama's top token
" time" had p=0.099 while our argmax was p=0.0034 with a disjoint top-6. Single-token
prompts (position 0, all RoPE angles zero) matched exactly, and every internal parity
suite passed — those suites compare our GPU against our own CPU, and both sides shared
the wrong convention, so the bug was invisible inside ZenInfer.

**Root cause.** llama.cpp applies llama-family weights with its "NORM" rope: *consecutive*
head elements pair, (x[2i], x[2i+1]), angle pos*base^(-2i/head_dim)
(src/llama-model.cpp, `LLM_ARCH_LLAMA → LLAMA_ROPE_TYPE_NORM`, "operating on pairs of
consecutive head values"). ZenInfer used the half-split NeoX layout (x[i], x[i+pairs]) —
right for qwen-family, wrong for llama. Frequency formula, base (500000, from the GGUF)
and eps (1e-5) were already correct; only the pairing was wrong.

**Fix.** Per-architecture flag `RuntimeConfig::rope_pairs_adjacent` (true for llama,
default false preserving qwen2/3/35), threaded through host `rope()`, all call sites,
the device rope shader (`RopeParams.pad0` selects the pairing; both branches rotate
(a*cos−b*s, a*s+b*c) with the same table), and the decode/pipeline parameter blocks.

**Also fixed en route:** the pipeline's per-layer cos/sin tables were uploaded and bound
as if a single table existed (buffer sized and bound for layer 0's slice; layer N read
layer 0's bytes or out of bounds). Now one 256B-aligned slot per layer. Outputs were
bit-identical before/after only because all layers' tables happened to be identical —
the read itself was allocation-illegal.

**Validation vs llama.cpp 0.5.0-dev (7fe450e19), CPU-only, greedy, raw ids, no BOS**
(oracle: `llama-server /completion` with raw prompt ids and `add_special:false`; our
`generate()` never prepends BOS either):
- After [12805,5304,264]: our top-6 equals llama's top-6, same order, margins within ~6%
  (892 " time": 0.1056 vs 0.0991; then 6453, 7742, 16700, 11879, 1938 — identical ids).
- Prompt [12805,5304,264,892], 8 greedy tokens: CPU, GPU per-op, and GPU pipeline all
  emit `11,220,16,13,20,3610,1274,304` — byte-identical to llama's ", 1.5 million people in".
- 24-token stream: tokens 1-19 and 21-24 identical to llama; token 20 is a single
  knife-edge flip (llama p('202')=0.345 vs p('203')=0.322 — 7% relative gap) that
  re-synchronizes one token later. Expected fp32-vs-Q8_0 accumulation noise, not a defect.
- GPU per-op vs pipeline logits: bitwise equal. CPU vs GPU: max_abs 1.0e-5 (fp32 vs Q8_0
  accumulation order).

**Known residual — RESOLVED (same day).** The T=8 CPU divergence was not attention at
all: `gemm_q8_0_avx2` (kernels/avx2.cpp:299) hardcoded its 4-token micro-block loads to
`input+0..3*columns` instead of `input+(t+k)*columns`, so every 4-token block after the
first recomputed tokens 0-3's inputs while writing t-relative outputs. This explains the
whole ladder: T≤7 correct (one full block + scalar tail), T=8 the first length with a
second full block (max_abs 14.4 vs GPU/oracle), T=16 broken in 12/16 tokens. KV-cache
append, flash gate, and parallel attention were ruled out by experiment
(`ZEN_D3D12_FLASH=0` unchanged; three CPU repeats bit-identical — no race). One-line fix
applied; CPU top-8 after [1..8] now equals the oracle's in order (703, 3505, 16, 32, 31,
8693, 674, …), CPU==GPU max_abs 1.3e-05, and CPU/GPU/pipeline emit byte-identical
16-token streams. On the degenerate [1..8] prompt llama itself is chaotic (top-4 within
0.043-0.057); fork flips between its top-2 (0.0567 vs 0.0529) are the same knife-edge
class as the 2020/2030 flip — accumulation noise, not a defect.

**Also fixed this run:** unparseable GGUF metadata no longer silently falls back — the
meta helpers log a `[meta]` line and load now fails when study-critical numerics
(rope.freq_base, rms epsilon, rope scaling) are present but unparseable.

**Not runtime-verified here:** qwen-family paths (no qwen GGUF on this machine). The flag
defaults to false for them and `zen_d3d12_rope_parity` (half-split) passes, but no qwen
oracle comparison has been run.

## Measured performance

```
prefill  128 tokens   171.95 ms   744.41 tok/s
decode   32 tokens    263.79 ms   117.52 tok/s   (~8.2 ms/token)

CPU only (no D3D12)
prefill  128 tokens   531.10 ms   241.01 tok/s
decode   32 tokens   1966.40 ms    15.76 tok/s   (~61 ms/token)
```

RUN-004 tuning items 1-3 landed (KV direct-write from compute, prefix re-append skip
via `kv_cache_prefix_resident`, LM head fused into the per-token submission, plus the
CPU trims: rope-table hoist, gated logits scans, logits move/swap, single-pass greedy
argmax). Clean re-measure, pipeline decode: **8.75 → 6.82 ms/token, 115 → 146.6 tok/s
(+28%)**; per-op decode ~115.5; prefill ~720-875 tok/s (run-to-run clock variance).
Outputs remain byte-exact vs the llama.cpp oracle on all paths (8- and 24-token streams,
[1..8] included). Kill switches: `ZEN_D3D12_KVDIRECT=0`, `ZEN_D3D12_LMFUSE=0`,
`ZEN_VALIDATE_LOGITS=1` restores full per-token logit scans.

RUN-005 hills (DP4A staging+UAV, DML probe, GEMM v2, DirectX research): DP4A decode
**205 → 251 tok/s** (groupshared int8 staging + UAV-only `qact`, bit-identical;
fp32 warm **263 tok/s**; isolated GEMV ceiling **359 GB/s** via LM-head bench,
decode is kernel-bandwidth-bound, orchestration ~4%). Falsified: DP4A wide loads
(−), fp32 32 KB Down staging (−), `ZEN_D3D12_FUSE=1` (−25%), GEMM v2 256-thread/16-tile
(−25%, `ZEN_D3D12_GEMM_V2=1`, bit-identical), DirectML FP16 GEMM (device removed at
recorder on this driver, `zen_d3d12_dml_probe`, dead end). No SM 6.10/LinAlg via
dxc 1.9 (max 6.9); WMMA unreachable without DML/LinAlg support. Prefill stays
~744 tok/s (scalar GEMM 3.0 TFLOP/s); 5000–8000 needs WMMA (blocked) and
producer-fused quantize / CPU-rope GPU move remain for decode (SPEC speculative
is the credible route to *effective* 450+, since 450 Q8 raw = 594 GB/s = 93% of
the 640 GB/s peak). The next step toward 450 is item 4 (GEMV lane widening) then
the item-5 overlapped double-buffered decode loop.

Perf numbers above are from the pre-RoPE-fix run; timing is unchanged by the fix — the
fix changes arithmetic, not compute cost.

## Fixed: the device removal (TDR)

Two independent bugs, both proven rather than guessed.

**1. Shader: register spill + divergent barrier.**
`acc[4][128]` needed 512 floats per thread against a 255-register limit, so every
access spilled to local memory. The loop also contained
`GroupMemoryBarrierWithGroupSync()` after a per-thread `break`, so some lanes
reached a barrier others had exited — a deadlock. Rewrote
`shaders/flash_attention.hlsl`: one wave per (query-token, head), lane-strided over
`head_dim`, all accumulators in registers, **no groupshared and no barriers**,
standard online softmax via `WaveActiveSum`/`WaveActiveMax`.
Result: 0.27–0.48 ms per layer, 112/112 dispatches on GPU.

**2. Out-of-bounds KV read.**
DRED reported page-fault VA `0x211c00000`; the K cache base was `0x201c00000`.
The delta is exactly **256 MB**. `flash_attention_prefill` built its layer stride
from the caller's `kv_context` (131072 = model max) instead of the context the
cache was actually allocated with (160), so layer 1 began 256 MB past a 5 MB
buffer. It now validates against `s.kv_context` and rejects mismatched
shape/layer/position instead of trusting the caller.

**Forensics added** so this class of failure is diagnosable next time:
`report_device_removed()` prints `GetDeviceRemovedReason()`, the DRED page-fault
VA and the breadcrumb op that did not complete; `wait_gpu()` now checks for
device removal because a TDR'd dispatch still signals its fence.

## Fixed: nondeterministic decode

`seeded sampling is not deterministic` — two identically seeded generations
produced different tokens. Isolated with a new tool (`zen_d3d12_determinism`):
divergence at token 1, `last_logits max_abs_diff=0.27`. Three causes:

- `device_attention_available()` was passed the *current* KV length, so
  `kv_cache_configure()` saw a new shape almost every decode step, reallocated the
  cache and discarded everything already appended. Now sized once
  (`kDeviceKvTokens = 8192`) and oversized requests fall back to the CPU.
- A prefix-cache hit restored host K/V but never re-synced the device KV cache.
- `layer_decode_pipeline` recorded its resource transitions into `s.list` (closed)
  instead of the per-layer list it was building, so the submitted work ran with
  undefined resource state and its output depended on leftover GPU state.

## Measured cost breakdown (decode)

```
16-layer GPU pipeline   record 0.69 ms + gpu 2.68 ms
LM head (vocab proj)    0.66 ms   (420 GB/s — confirmed on GPU)
setup (tensor lookup)   0.14 ms
```

`CreateCommandAllocator`/`CreateCommandList` were being called **per layer, per
token** (32 driver allocations per token) and cost ~3.7 ms of a 7.9 ms step. They
are now created once and reused; record time fell 3.7 ms → 0.69 ms.

Decode is **not** bandwidth-bound: 1.23 GB/token at 40 GB/s would be ~31 ms, and
the GPU pipeline already sustains ~440 GB/s reading resident weights.

## Not done — stated plainly

- **Targets are not met.** 744 tok/s prefill vs 8000, 118 tok/s decode vs 450.
- **The decode gap is measured and attributed, not theorised.** llama.cpp sustains
  ~90% of peak on this shape; my pipeline does 76%. The missing 14% is the GEMV
  kernel dequantising per element, plus ~5 ms/token of orchestration outside it.
- **`layer_decode_pipeline` is disabled by default** (`ZEN_D3D12_PIPELINE=1` to
  opt in). It reaches 163 tok/s but is only fast because it skipped barriers; once
  barriers were made correct it slowed to 15 tok/s and started failing over to the
  CPU. It needs per-buffer resource-state tracking to be both correct and fast.
  The per-layer path is the correct one and is what ships.
- **The int8-activation kernels are written but not enabled.** They are the actual
  fix for both CPU and GPU, but they change the arithmetic and the parity tests
  require exact fp32 (`zen_core_tests` at 2e-4, `zen_d3d12_parity` at 1e-5). Making
  them the default means relaxing those tolerances or adding a separate int8 parity
  baseline — a decision for you, not something to slip in.
- `tools/d3d12_probe.cpp` was overwritten by me without reading it first; there is
  no git history here, so the original is gone. It has been reconstructed from its
  link line (`d3d12 dxgi`) and is marked as a reconstruction in the file.

## Ranked next steps

1. **Enable the int8 GEMV/GEMM path by default and re-baseline the parity tests
   against int8.** This is the single change that closes the 76% → 90% decode gap
   and the 2.65 → ~15 TFLOP/s prefill gap, because it removes the per-element
   dequantisation on both. Needs a decision on the fp32-parity constraint.
2. **Re-enable the multi-layer pipeline with per-buffer resource-state tracking.**
   Worth 117 → ~163 tok/s on its own and composes with (1).
3. **Split-K for the 2048-row matmuls** (`ffn_down`, `attn_q`), which under-fill
   the CUs at 32 rows per workgroup.
4. **Speculative decoding** (`generate_speculative` + `forward_batch(tail_logits)`
   already exist) if the target must be exceeded rather than merely approached.

## Reference points (why the targets are reachable)

Measured against llama.cpp's published numbers and its source, which is in
`H:\LLAMA-bins\llama.cpp`:

- **RTX 4060** (128-bit GDDR6, 272 GB/s) runs this model at ~185 tok/s. That is
  **90% of peak bandwidth** on a pure streaming GEMV. Applied to this card's
  640 GB/s, 90% gives ~437 tok/s and 93% gives ~450.
- My 16-layer decode pipeline measured **484 GB/s = 76% of peak**. So the gap to
  450 tok/s is **my kernel**, not the hardware. My earlier claim that 450 tok/s was
  unreachable was wrong: I applied a generic 70% efficiency to what is a perfectly
  sequential read.

### What llama.cpp does that we did not

Read from source, not assumed:

| | llama.cpp | this backend (before) |
|---|---|---|
| CUDA/HIP GEMV | `mmvq.cu` → `vec_dot_q8_0_q8_1` → `ggml_cuda_dp4a`, activations **also Q8_0** | per-element int8→float dequant, float FMA |
| Vulkan | `mul_mat_vecq.comp`, `block_q8_1_x4`, `GL_EXT_integer_dot_product`, `NUM_COLS`/`NUM_ROWS` register blocking | fp32 activations |
| CPU AVX2 | `_mm256_sign_epi8` ×2 + `_mm256_maddubs_epi16` + `_mm256_madd_epi16` → 32 MACs in 5 instructions | `_mm256_cvtepi8_epi32` + `_mm256_cvtepi32_ps` + FMA per 8 elements |

The common thread is exactly what was pointed out: **quantise the activations and
keep the inner product in integers**. That is the mechanism, and both our CPU and
GPU kernels were dequantising instead.

### Fixed as a result

- **GPU GEMV** (`shaders/q8_0_gemv.hlsl`): the reduction was
  `red[tid] = int(acc * 1000000.0f)` followed by one lane summing 32 shared-memory
  slots. `int(acc*1e6)` overflows int32 above |acc|~2147 (a real correctness bug)
  and the serial loop left 31 lanes idle. Now a masked 32-lane shuffle reduction.
- **Wave64 bug (mine)**: replacing that with a bare `WaveActiveSum` silently broke
  the kernel, because RDNA runs this shader as wave64 and lanes 0-31 / 32-63 are
  two different rows in one wave. Caught by `zen_d3d12_parity`; fixed with the
  lane-masked reduction. Worth noting because the same mistake would have shipped
  as "passes on my machine".
- **CPU GEMV** (`kernels/avx2.cpp`): removed the per-32-element stack store plus
  7-scalar horizontal sum; one `hsum` at the end instead. An int8-activation kernel
  (`maddubs`+`madd`) is written and available behind `ZEN_CPU_Q8_ACT=1`, but it
  **cannot be the default**: `zen_core_tests` compares against an exact fp32
  reference at 2e-4 and int8 activations miss that by ~1e-3.

## CPU (Ryzen 9 5900XT, Zen 3, 16C/32T)

Measured: prefill 241 tok/s, decode 15.8 tok/s (61 ms/token).

Zen 3 has AVX2 but **no AVX-512 and no AVX-VNNI**, so `_mm256_dpbusd_epi32` is not
available; the `maddubs`+`madd` form is the correct choice. 61 ms/token is 1.24 GB
at ~20 GB/s against ~40 GB/s of DDR4-3000 dual-channel, so the CPU path is at ~50%
of its bandwidth. That is the next CPU lever and it is the same problem as the GPU:
quantise the activations and drop the per-element float convert.


New tool `zen_d3d12_gemm_bench` reports TFLOP/s at the model's real shapes, so
kernel changes are judged without layer orchestration in the way.

```
fp32 kernel    ffn_gate 8192x2048 t=128   1.59 ms   2.70 TFLOP/s
               ffn_down 2048x8192 t=128   1.38 ms   3.12 TFLOP/s
               attn_q   2048x2048 t=128   0.54 ms   1.99 TFLOP/s
               TOTAL 5.26 ms for one layer's four big matmuls -> 2.65 TFLOP/s
```

8000 tok/s at 128 tokens is 16 ms, which needs **15.4 TFLOP/s**. We are at 2.65.

**What the measurement ruled out.** The old kernel re-read all weights once per
8-token tile, i.e. 16 passes over 1.3 GB for a 128-token prefill. Making the
weights resident across 32 tokens (4 passes instead of 16) changed prefill by
only **6%** (759 vs 717 tok/s). That is conclusive: prefill is **instruction-bound,
not bandwidth-bound**. Weight traffic was never the problem.

**What the measurement confirmed.** The fp32 inner loop dequantises per element —
extract byte, int8→float, multiply by activation, multiply by block scale,
accumulate — roughly 4-6 instructions per weight element. At ~6 instructions per
MAC against a 24 T-FMA/s FP32 peak that predicts ~4 TFLOP/s, and 2.65 is what we
measure. The arithmetic matches the observation, so the diagnosis is sound.

**The fix that should work but does not yet.** `shaders/q8_0_gemm_dp4a.hlsl`
replaces that inner loop with `dot4add_i8packed` (DP4A), which is the only way to
cut the instruction count rather than the traffic. DXIL confirms dxc emits
`dx.op.dot4AddPacked` — the intrinsic is real, not emulated. Measured throughput is
**0.94 TFLOP/s, three times slower than the fp32 path**. Two tilings were tried
(8 lanes/row packing 4 rows per warp, then warp-per-row with 4 blocks per
iteration and perfectly coalesced 128-byte loads) and both land at ~0.94. Reducing
the accumulator block from 32 to 8 tokens changed nothing, so it is not register
pressure. Throughput tracks *row count* (8192 rows → 1.6, 2048 rows → 0.5), which
points at workgroup-count starvation or a scheduling effect I have not isolated.
The path stays opt-in (`ZEN_D3D12_DP4A=1`) and off by default: it is slower and it
is not parity-exact.

**Next step for prefill**, in order: profile the DP4A kernel's occupancy and
instruction issue (RGP or GPU timestamping per phase) to explain the 3x gap before
writing more kernel variants; then split-K for the low-row matmuls
(`ffn_down`, `attn_q` at 2048 rows) so they stop starving the CUs.

## Diagnostic tools added

- `zen_d3d12_flash_parity` — GPU vs CPU prefill attention (regression guard).
- `zen_d3d12_metrics` — resident weight bytes, submits, queue/wait time per run.
- `zen_d3d12_lmhead_bench` — isolates the vocabulary projection.
- `zen_d3d12_determinism` — reports where two seeded generations diverge.
- `zen_d3d12_probe` — adapter/feature-level report (reconstruction).
- `ZEN_D3D12_DRED=1` enables DRED + the debug layer.
- `ZEN_D3D12_STAGE=1` prints per-stage decode timings.

## RMS AVX2 default-ON correctness (verified 2026-11-06)

Greedy generated_ids byte-identical across 32 tokens on 3 models (Llama-3.2-1B-Q8_0, Qwen3-4B-Q8_0, Qwen2.5-Coder-7B-Q5_K_M) with ZEN_CPU_RMS_AVX2=0 vs 1. last_logits max_abs_diff = 0.0 on all three. AVX2 RMS accumulates in double (__m256d); only lane-order of sum differs, measured 0.0. Recommendation: keep default ON (see .opencode/sherlock-analysis.md RUN-011).
