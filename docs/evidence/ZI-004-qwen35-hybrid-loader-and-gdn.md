# ZI-004 evidence: Ternary-Bonsai-2 / qwen35-hybrid loader and forward-path repair

Date: 2026-11-01. Branch: `arena/3134eb69-zeninfer`. Linux x86-64, GCC 12.2.
**Windows/MSVC: NOT RUN** (no MSVC in this environment; see ZI-017).

## What was wrong, and what the evidence is

The reported symptom was a probe that died with no map output and two different
NTSTATUS codes on consecutive identical commands. Working that backwards:

### 1. The exit codes in the report were mapped wrongly

`-1073740940` is `0xC0000374`, **STATUS_HEAP_CORRUPTION**, not `0xC0000094`
(integer divide by zero, which would be `-1073741676`). So the pair was
`0xC0000005` + `0xC0000374` — i.e. heap corruption, not an alternating
AV/DIV0. That, plus the trace never appearing, pointed at an out-of-bounds
write rather than a divide.

```
0xC0000005 ACCESS_VIOLATION    -> exit -1073741819
0xC0000094 INT_DIVIDE_BY_ZERO  -> exit -1073741676
0xC0000374 HEAP_CORRUPTION     -> exit -1073740940   <-- the second code
```

### 2. Two defects made the whole class of load failures invisible

- `dense_transformer_runtime.cpp` called `format_for_ggfl(...)`; the function is
  named `format_for_ggml`. **`zen_parity_probe` and `zen_runtime_tests` did not
  compile at all on this commit.** Any "run" against the previous tree was a
  stale binary, which is consistent with "I never saw its output".
- Every `[map]` line, and any `return fail(...)` message printed after the map
  began, went to an unflushed `stderr`. A crash on the next tensor discarded the
  buffer, so the trace died exactly when it mattered.

Fixed: the call site; `fflush(stderr)` per tensor; `ZEN_MAP_TRACE=0` to silence.

### 3. Confirmed out-of-bounds writes on the qwen35 path (ASan)

Reproduced with a new synthetic Bonsai-2-shaped fixture, `-fsanitize=address`:

| site | finding |
|---|---|
| `v` sized `T*KW` but written `T*value_dim` | heap-buffer-overflow **WRITE** (the 0xC0000005 / 0xC0000374 pair) |
| conv ring write, one row per token | heap-buffer-overflow **READ/WRITE** beyond the `(CK-1)*conv_dim` ring |
| decode (`start_pos != 0`) | KV/conv caches indexed by batch-local `t`, not absolute `pos` → OOB read on the first generated token |
| `state` sized `value_dim*value_dim` | not OOB, but `NV²` larger than the `NV*S*S` buffer actually indexed (~300 MB/layer at 27B) |
| KV cache sized by `config_.context` | 262144-position allocation (GB-scale, hard `bad_alloc` on Windows) before any token ran |

All five reproduce as a nonzero exit on the fixture and are clean under ASan
after the fix.

### 4. GDN gate math against the reference (ops.cpp one_chunk / qwen35.cpp)

Reference: `PrismML-Eng/llama.cpp` `src/models/qwen35.cpp`
(`build_layer_attn_linear`, `build_layer_attn`) and
`ggml/src/ggml-cpu/ops.cpp` (`ggml_compute_forward_gated_delta_net_one_chunk`).

| item | before | reference |
|---|---|---|
| `ssm_a` | `-exp(ssm_a) * softplus(alpha+dt_bias)` | `ssm_a` is already `-A_log.exp()`: `ssm_a * softplus(...)` |
| q/k l2-norm | one norm over the whole `2*key_dim` span | one norm **per key head** over its joint q,k view |
| conv ring update | row 0 ← oldest token (window reversed) | row 0 ← newest token (newest-first ring) |
| carried input token | `t + w - CK + 1` (touches token `t`) | `t + w - CK` (strictly causal) |
| Hadamard input order | signs **then** rotate | rotate **then** signs (`build_hadamard_input`) |
| folds per activation | re-applied per weight sharing the buffer | one transform per activation (`get_hdmd_input` memoisation) |
| FFN residual | `x = (res+attn) + res + ffn` | `x = res + ffn` (residual counted twice) |
| LM head input | always rotated | rotated only when `output.weight` is a folded weight |
| `ssm_out` grouped permutation | tied to `block_size != 0` | driven by `prism.hadamard.gdn_v_grouped` |

Re-derived from the reference and **not** changed under time pressure:
`beta = sigmoid(beta_proj)`, `g = ssm_a * softplus(alpha+dt_bias + dt_bias[h])`,
`1/sqrt(S_v)` output scale, the `i*S+j` transposed state layout, the tiled
`kh = h % NK` head repeat, the conv weight layout `[CK, conv_dim]`, and the
gated norm `ssm_norm` then `silu(z) * normalised` (verified against
`ggml_vec_swiglu_f32`: `y = silu(src0) * src1` with `src0 = gate = z`).

Also added: the `prism.hadamard.*` contract is now validated the way the
reference validates it (version/transform/axis/sign-mode, folded weights that
must sit on a Hadamard-aware matmul, only `token_embd.weight` as an
inverse-after-lookup table), and a Prism ternary file with **no** contract is
refused at parse time instead of being decoded in the wrong basis.

## New fixture and harness

- `tests/make_bonsai2_fixture.py` — writes a synthetic `qwen35` GGUF
  (PTQ1_0/type-143 blocks, `prism.hadamard.*` with weights genuinely stored as
  `W*D*H`, `gdn_v_grouped`, 4 GDN + 4 attention layers) plus an independent
  NumPy reference written from the fork's graph and CPU op.
- `tests/bonsai2_parity.py` — drives `zen_parity_probe`; smoke mode asserts the
  graph loads, runs prefill + decode and produces finite, non-degenerate logits
  (this is the regression guard for everything in §3). `--numeric` diffs
  logits against the reference.

## Status

CTest `zen_bonsai2_hybrid_parity` (smoke) **PASS**, ASan build **clean**.
AVX2 suite 7/7 and scalar suite 8/8 still pass.

**Numerical parity is NOT achieved.** `--numeric` still fails: first-token
logits differ, so the qwen35 recurrence is not yet bit-matched end to end.
Stage diffing shows qkv / conv / qk-norm / GDN core / state agreeing to
`1e-7…1e-6` while later stages diverge, which leaves the reference's
`gdn_v_grouped` + gated-norm stage and the multi-layer residual path as the
open items. ZI-004 stays PARTIAL: it is no longer a crash, but it is not parity.
