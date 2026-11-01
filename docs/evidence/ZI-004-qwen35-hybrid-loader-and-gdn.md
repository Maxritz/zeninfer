# ZI-004 evidence: Ternary-Bonsai-2 / qwen35-hybrid loader and forward-path repair

Date: 2026-11-01. Branch: `arena/20e50e23-zeninfer`. Linux x86-64, GCC 12.2.
**Windows/MSVC: NOT RUN locally**; covered by CI (see BUILD_VERIFIED).

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

## Status: parity achieved on the synthetic fixture

CTest `zen_bonsai2_hybrid_parity` now runs `--numeric` and **PASSES**.

| depth | max abs error | argmax | top-5 |
|---|---|---|---|
| L=1 | 2.29e-05 | match | match |
| L=2 | 2.29e-05 | match | match |
| L=3 | 3.43e-05 | match | match |
| L=4 | 3.62e-05 | match | match |
| L=5 | 4.20e-05 | match | match |
| L=8 | 4.20e-05 | match | match |

f32 logits rank-correlate with the reference at Spearman **1.000000**. ASan/UBSan
clean on 1-, 2-, 3- and 5-token prompts across f32 / PTQ1_0 / identity-signs.
AVX2, scalar and ASan builds: 9/9 CTest each.

### The defects that were still outstanding after the first pass

1. **`swht_block` was a sequency-ordered transform.** It bit-reversed the input
   before the butterfly, so it computed a different matrix from the
   natural-order Sylvester `H` every folded weight was folded against. Rows 0
   and 1 agree; everything from row 2 diverges. Cross-checked against
   `ggml/src/ggml-cuda/fwht.cu`, which is plain natural-order butterflies with
   `1/sqrt(N)` scaling and no permutation.
2. **Signs/sign_widths slicing.** `sign_values` is a concatenation over
   `sign_widths`, so a width's slice starts after every earlier declared width.
   The reference read from offset 0 for all widths; the runtime was right.
3. **Duplicate sign widths.** The fixture emitted `[128, 256, 128, 256]` (E ==
   value_dim, conv_dim == F). `load_hparams_hadamard` rejects duplicates.
4. **Sign/rotate order.** `build_hadamard_input` applies signs *then* rotation;
   the embedding restore `build_embd_rows` does the opposite. The two paths are
   deliberately asymmetric, and the embedding table must store the inverse of
   the restore.
5. **`fold_for` transformed only row 0.** The lambda received a `T*width`
   activation buffer but called a per-vector transform once, so in prefill every
   token after the first multiplied folded weights by an unrotated vector.
   Invisible in single-token decode; wrong for any longer prompt.
6. **Attention output gate used SiLU.** `g/(1+exp(-g))` instead of
   `1/(1+exp(-g))` (qwen35.cpp `gate_sigmoid` + `ggml_mul`).
7. **Fused Q|gate split was contiguous.** The reference takes two views of the
   same buffer with a `2*head_dim` element stride, i.e. **interleaved per head**
   `[q0 g0 q1 g1 ...]`, not `[all q | all gate]`.
8. **GDN `ssm_out` permutation was an identity no-op**: `(h/NK)*NK + h%NK == h`.
   `ggml_permute(0,2,1,3)` puts the rep axis first, so the grouped head is
   `rp + rep*nk`.

### Fixture-side defects (these were masking the runtime ones)

The harness was also wrong in ways that made correct code look broken:

- **PTQ1_0 encoder corrupted every byte.** Each byte carries five trits and the
  decoder reads all five, but the encoder wrote one lane per byte and OR-ed the
  five lane constants together. `3^5 == 243 <= 256`, so the mapping is a clean
  bijection; it now packs properly and asserts a lossless round-trip.
- **Sign slicing from offset 0** (defect 2 above, reference side).
- **The embedding table stored the restore instead of its inverse.**
- **The numeric reference consumed pre-quantisation weights.** A folded weight
  `W*S*H` is dense and PTQ1_0 can only carry `{-d,0,+d}`, so the reference was
  comparing against numbers the file does not contain. The npz now stores
  exactly the decoded bytes (verified bit-exact against the GGUF payload).

### PTQ1_0 is rank-graded, on purpose

A latent folded embedding row is dense; storing it at 1.58 bits loses ~4.8 per
element against embeddings of magnitude ~1. Quantisation is therefore the floor,
not the forward pass. The ptq1_0 case is graded on Spearman rank correlation
(>= 0.95; measured 0.988-0.994) plus argmax, while **f32 is the actual parity
gate** at 5.7e-05 with exact top-5. The harness prints "rank-graded: quantization
loss, not a parity claim" so this cannot be mistaken for a full-precision result.

## What is still NOT validated

- **No real Ternary-Bonsai-2 checkpoint has been run.** Everything here is a
  synthetic fixture built to the fork's contract. The fixture exercises the code
  paths; it does not prove the real file's metadata matches.
- Windows/MSVC covers the build via CI, but the crash repro was never reproduced
  on the reporter's machine with the real model.
