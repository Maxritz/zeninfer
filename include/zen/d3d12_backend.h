#pragma once
#include <cstddef>
#include <cstdint>

// Direct3D 12.1 compute backend for the Q8_0 matrix kernels.
// Only compiled into the build when ZEN_HAS_D3D12 is defined.
namespace zen::d3d12 {

struct AdapterInfo {
    char name[128]{};
    std::uint64_t dedicated_vram{};
    std::uint64_t shared_memory{};
    int feature_level{};  // 121 or 122; 0 when unknown
    // True when D3D12_OPTIONS16 reports GPU upload heaps, which on a discrete GPU
    // means the entire VRAM is CPU-visible. There is no direct ReBAR query in
    // D3D12 and ReBAR is a firmware setting code cannot change, so this flag is the
    // only reliable in-process evidence that it is live.
    bool gpu_upload_heap_supported{};
    unsigned agility_sdk_version{};  // from the exported D3D12SDKVersion, 0 if inbox
};

struct Stats {
    std::uint64_t dispatches{};
    std::uint64_t submits{};
    std::uint64_t batches{};
    std::uint64_t upload_bytes{};
    std::uint64_t weight_bytes{};
    std::uint64_t queue_us{};   // CPU time inside ExecuteCommandLists (driver acceptance)
    std::uint64_t wait_us{};    // fence wait only, GPU retire latency
    std::uint64_t upload_us{};
    std::uint64_t alloc_us{};   // committed-resource creation + staging growth
    std::uint64_t stage_us{};   // staging buffer allocation/growth only
    std::uint64_t repack_us{};
    std::uint64_t transfer_us{};
    // Per-call decomposition, only collected when ZEN_D3D12_TRACE=1.
    std::uint64_t prep_us{};
    std::uint64_t reset_us{};
    std::uint64_t setup_us{};
    std::uint64_t record_us{};
    std::uint64_t readback_us{};
    std::uint64_t gpu_us{};      // stamped Dispatch execution
    std::uint64_t gpu_copy_us{};  // stamped output copy to the readback heap
    // Fence time split by submission shape, so achieved GB/s can be computed per path
    // and a fixed per-submit cost separated from real memory work.
    std::uint64_t qkv_wait_us{};   // gemv_batch (attn_q|k|v)
    std::uint64_t ffn_wait_us{};   // gemv_ffn (gate|up|SiLU|down)
    // Device-timestamped execution for the same two paths, so fence wait can be split
    // into real GPU work versus host round-trip latency.
    std::uint64_t qkv_gpu_us{};
    std::uint64_t ffn_gpu_us{};
    std::uint64_t qkv_copy_us{};  // dispatches -> readback copy, same two paths
    std::uint64_t ffn_copy_us{};
    std::uint64_t fallbacks{};
};

// True when the process asked for this backend (ZEN_BACKEND=d3d12|dx12|gpu or
// ZEN_D3D12=1) *and* a 12.1 device exists. Initialisation is lazy, cached and
// thread-safe; a failed init is remembered so later calls stay on the CPU.
bool enabled() noexcept;
bool init() noexcept;
void shutdown() noexcept;
const AdapterInfo* info() noexcept;
const Stats& stats() noexcept;

// Prints the per-component trace table (ops, GPU time, fence wait, idle time) for the
// current run. Only populated when ZEN_D3D12_TRACE=1; otherwise it says so.
void trace_report();

// Weights stay resident in VRAM, keyed by the host pointer the loader handed out.
// Q8_0 rows are repacked on upload because a raw GGML block is 34 bytes, which
// leaves every odd block index misaligned for a ByteAddressBuffer load.
bool gemv_q8_0(const std::uint8_t* weights, const float* x, float* y,
               std::size_t rows, std::size_t columns) noexcept;
bool gemm_q8_0(const std::uint8_t* weights, const float* input, float* output,
               std::size_t tokens, std::size_t rows, std::size_t columns) noexcept;

// One matvec target for a batched submission. All tasks in a batch share the input
// vector, which is exactly the Q|K|V and gate|up pattern in a transformer block.
struct GemvTask {
    const std::uint8_t* weights;
    float* y;
    std::size_t rows;
    std::size_t columns;
};
constexpr std::size_t kMaxBatch = 8;

// Submits every task in a single command list. A lone dispatch costs ~70us of fixed
// queue latency on this stack against ~1-2us marginal per extra dispatch, so batching
// the matvecs that share an input is worth far more than tuning the kernel.
bool gemv_q8_0_batch(const float* x, const GemvTask* tasks, std::size_t count) noexcept;

// One GEMM target for a batched prefill submission. All tasks share the same
// [tokens x columns] input matrix and differ only in weight pointer and row
// count - the split Q|K|V and gate|up projections of a transformer prefill.
// Rows may differ per task; tokens and columns must match across the batch.
struct GemmTask {
    const std::uint8_t* weights;
    float* y;
    std::size_t rows;
};

// Submits every task in one command list: one x upload, one fence wait, one
// packed output readback. Kernels and parameters are the same as gemm_q8_0,
// so batched results are bit-identical to the unbatched path; the win is the
// removed queue round trips (~0.7-1.1 ms per submit measured on RDNA4).
bool gemm_q8_0_batch(const float* x, const GemmTask* tasks, std::size_t count,
                     std::size_t tokens, std::size_t columns) noexcept;

// gate|up|SiLU|down as a single submission. The FFN activation stays in VRAM between
// the three matvecs, so one queue round trip replaces two.
// Root-mean-square normalisation on the device, so a layer's activation can stay in
// VRAM between matvecs instead of round-tripping through the host.
bool rms_norm(const float* x, const float* weight, float* y, std::size_t rows,
              std::size_t columns, float epsilon) noexcept;

// Element-wise device residual add, out[i] = a[i] + b[i] over `count` floats.
// Keeping it on the device is what lets the residual connection stop costing a host
// round trip between the attention output matvec and the FFN norm.
bool residual_add(const float* a, const float* b, float* out, std::size_t count) noexcept;

// Rotary position embedding over q or k, applied on the device.
// `x` is [heads][head_dim]; `cos`/`sin` are the host-evaluated [pairs] table, where
// pairs = rotary_dim / 2. `interleaved_pairs` selects the layout: false (NeoX,
// qwen-style) pairs element i with i + pairs; true (llama.cpp NORM rope, llama
// family) pairs the consecutive elements 2i and 2i+1. The pair rotates as
// (a*cos - b*sin, a*sin + b*cos), matching the CPU path exactly. The table is built
// on the host on purpose: recomputing it with powf/cosf on the device would differ
// in the last bits from the double-promoted reference and could flip a sampled id.
bool rope_apply(const float* x, const float* cos, const float* sin, float* out,
                std::size_t heads, std::size_t head_dim, std::size_t pairs,
                bool interleaved_pairs = false) noexcept;

// Batched prefill RoPE over [tokens, heads, head_dim] (T > 1). `cos`/`sin` are
// host-evaluated [tokens * pairs] tables, token-major: token t's table is
// cos[t*pairs .. t*pairs+pairs) (the rope_table for pos=start_pos+t), matching
// exactly what a per-token CPU rope() would build. Rotation, pairing (`pad0`)
// and copy-through rules are the same as rope_apply; only the grid widens to
// tokens*heads*head_dim and the table index adds `token * pairs`.
// `x` and `out` may alias (in-place): the host staging copy precedes the dispatch
// and the readback memcpy follows the fence, so no device hazard exists.
// Returns false (caller stays on the CPU rope path) when the backend is absent,
// the PSO failed to build, the shape is invalid, or scratch cannot be reserved.
// Decode rope_apply is unaffected.
bool rope_apply_prefill(const float* x, const float* cos, const float* sin, float* out,
                        std::size_t tokens, std::size_t heads, std::size_t head_dim,
                        std::size_t pairs, bool interleaved_pairs = false) noexcept;

// --- Device-resident KV cache and attention (tasks 4 and 6) -------------------
//
// The cache is [layers][kv_heads][context][head_dim] floats, with k and v in separate
// VRAM buffers. It is created once per model shape and appended to in place, so keys
// and values never travel to the host during decode.
bool kv_cache_configure(std::size_t layers, std::size_t context, std::size_t kv_heads,
                        std::size_t head_dim) noexcept;
void kv_cache_reset() noexcept;

// True when the device KV cache already holds, for EVERY layer, a contiguous resident
// prefix covering every position in [first_pos, first_pos + count). A caller about to
// re-append exactly that range (restoring a cached prompt prefix before decode) can
// then skip the copies entirely - the bytes are already in VRAM. The record is built
// from every successful append and every successful decode-layer/pipeline K/V write,
// and is discarded by kv_cache_reset and by any kv_cache_configure that (re)creates
// the cache buffers. false when the backend is absent, not initialised, or reset.
bool kv_cache_prefix_resident(std::size_t first_pos, std::size_t count) noexcept;   // false when backend absent/reset

// Appends one token's k and v to the device cache, used by prefill so decode finds the
// prompt positions there. Pure copy: two CopyBufferRegion calls, no dispatch.
bool kv_cache_append(std::size_t layer, std::size_t pos, const float* k, const float* v,
                     std::size_t kv_heads, std::size_t head_dim) noexcept;

// Batched KV cache append for prefill: appends multiple layers' K/V in one submission.
bool kv_cache_append_batched(const std::size_t* layers, std::size_t num_layers,
                             const std::size_t* positions,
                             const float** ks, const float** vs,
                             std::size_t kv_heads, std::size_t head_dim) noexcept;

// The whole attention block of one decode step in a single submission:
//   qr = rope(q);  kr = rope(k);  cache_k[layer][pos] = kr;  cache_v[layer][pos] = v;
//   att = softmax(qr . cache_k[layer][..pos] / sqrt(head_dim)) . cache_v[layer][..pos]
// `cos`/`sin` are the host-evaluated table with `pairs` entries, indexed by dimension,
// as in rope_apply. `first_pos` is the sliding-window start. `out` receives
// [heads][head_dim] floats. Returns false (and the caller stays on the CPU path) when
// the cache was not configured for this layer or shape.
bool attention_block(const float* q, const float* k, const float* v, const float* cos,
                     const float* sin, float* out, std::size_t heads, std::size_t kv_heads,
                     std::size_t head_dim, std::size_t pairs, std::size_t layer,
                     std::size_t pos, std::size_t first_pos,
                     bool interleaved_pairs = false) noexcept;

// The whole transformer layer in ONE submission (task 7): rmsnorm, Q|K|V, rope(q),
// rope(k), KV append, GQA attention, attn_output matvec, residual, FFN norm,
// gate|up, SiLU, down, residual, one readback. q/k/v never leave VRAM.
// `cos`/`sin` are the host table with `pairs` entries. `layer`, `pos` and `first_pos`
// address the device KV cache, which must already be configured for this shape.
// Returns false (caller keeps its own path) when the cache or weights are unavailable.
bool layer_decode(const std::uint8_t* q_w, const std::uint8_t* k_w, const std::uint8_t* v_w,
                  const std::uint8_t* o_w, const std::uint8_t* gate_w, const std::uint8_t* up_w,
                  const std::uint8_t* down_w, const float* attn_norm_w, const float* ffn_norm_w,
                  const float* cos, const float* sin, std::size_t pairs, const float* x,
                  float* out, std::size_t layer, std::size_t pos, std::size_t first_pos,
                  std::size_t hidden, std::size_t heads, std::size_t kv_heads,
                  std::size_t head_dim, std::size_t ffn_rows, float epsilon,
                  bool interleaved_pairs = false) noexcept;

// FlashAttention for prefill with KV cache: computes multi-head attention for all tokens in parallel
// on GPU. Input Q is [tokens, heads, head_dim], K/V come from device KV cache.
// Output is [tokens, heads, head_dim].
// Uses tiled FlashAttention algorithm with causal masking and KV cache.
// Returns false if not supported (falls back to CPU attention).
bool flash_attention_prefill(const float* q, const float* k, const float* v, float* out,
                             std::size_t tokens, std::size_t heads, std::size_t kv_heads,
                             std::size_t head_dim, float inv_scale,
                             std::size_t layer, std::size_t start_pos,
                             std::size_t kv_context, std::size_t k_cache_heads, std::size_t k_cache_head_dim) noexcept;

// Pipelined multi-layer decode: records all layers into command lists, submits once,
// waits once. Keeps intermediate activations in VRAM (GPU-to-GPU). Only final output
// is read back to host. `num_layers` must be >= 1. Each layer's parameters are
// provided as arrays of length `num_layers`. Returns false on any failure.
// cos/sin tables are shared across all layers (single RoPE table per position).
struct LayerDecodeParams {
    const std::uint8_t* q_w;
    const std::uint8_t* k_w;
    const std::uint8_t* v_w;
    const std::uint8_t* o_w;
    const std::uint8_t* gate_w;
    const std::uint8_t* up_w;
    const std::uint8_t* down_w;
    const float* attn_norm_w;
    const float* ffn_norm_w;
    std::size_t pairs;
    // RoPE layout the host tables were built for: 0 = NeoX half-split
    // (element i with i + pairs), 1 = llama.cpp NORM rope (2i with 2i+1).
    std::uint32_t rope_pairs_adjacent{};
    std::size_t layer;
    std::size_t pos;
    std::size_t first_pos;
    const float* cos;
    const float* sin;
};

bool layer_decode_pipeline(const LayerDecodeParams* layers, std::size_t num_layers,
                           const float* input, float* output,
                           const float* cos, const float* sin, std::size_t pairs,
                           std::size_t hidden, std::size_t heads, std::size_t kv_heads,
                           std::size_t head_dim, std::size_t ffn_rows, float epsilon) noexcept;

// Internal overload used by runtime (cos/sin/pairs embedded in LayerDecodeParams)
bool layer_decode_pipeline(const LayerDecodeParams* layers, std::size_t num_layers,
                           const float* input, float* output,
                           std::size_t hidden, std::size_t heads, std::size_t kv_heads,
                           std::size_t head_dim, std::size_t ffn_rows, float epsilon) noexcept;

// Optional fused LM head (one submit+wait per token instead of two). When `lm` is
// non-null and ZEN_D3D12_LMFUSE is not "0", the pipeline records the final rmsnorm
// (over `output_norm_w`) and the vocab GEMV (`lm_head_w` = output.weight, or the tied
// token_embd; Q8_0, vocab_rows x hidden) plus a logits readback into the SAME command
// list as the layers. Numerics are identical to calling rms_norm + gemv_q8_0
// separately: same rmsnorm and GEMV kernels, same repacked weight table, epsilon from
// `lm->epsilon`. On return `lm->fused` is true only when `lm->logits` holds the
// vocab_rows device-computed logits from that submission; false means the caller must
// compute the head itself (kill switch set, weights not resident or over the VRAM
// budget, or the row count exceeds the dispatch limit). `output` always receives the
// final hidden state, exactly as the two-argument-less overload does.
struct LmHeadFuse {
    const float* output_norm_w;      // final rmsnorm table, `hidden` floats
    const std::uint8_t* lm_head_w;   // Q8_0 rows in raw GGML layout, vocab_rows x hidden
    std::size_t vocab_rows;          // e.g. 128256
    float epsilon;                   // final rmsnorm epsilon
    float* logits;                   // OUT: vocab_rows floats when fused == true
    bool fused;                      // OUT: set by the backend
};

bool layer_decode_pipeline(const LayerDecodeParams* layers, std::size_t num_layers,
                           const float* input, float* output,
                           std::size_t hidden, std::size_t heads, std::size_t kv_heads,
                           std::size_t head_dim, std::size_t ffn_rows, float epsilon,
                           const LmHeadFuse* lm) noexcept;

// The whole FFN half of a layer in one submission:
//   t0 = attn_output @ att;  h1 = t0 + x;  h2 = rmsnorm(h1);
//   g  = gate @ h2;  u = up @ h2;  f = silu(g)*u;  d = down @ f;
//   out = h1 + d
// The activation never leaves VRAM between those steps, so the matvec, both residual
// adds and the norm stop costing one queue round trip each.
bool ffn_residual_fused(const std::uint8_t* attn_output, const std::uint8_t* gate,
                        const std::uint8_t* up, const std::uint8_t* down,
                        const float* norm_weight, const float* x, const float* attn_out,
                        float* out, std::size_t hidden, std::size_t attn_width,
                        std::size_t ffn_rows, float epsilon) noexcept;

// SiLU(gate)*up on the device, out[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i]
// over `count` floats. Mirrors rms_norm/residual_add (host in, host out, one
// submit): used by the parity harness and by the fallback paths. The prefill
// hot path fuses this into ffn_prefill_fused instead of paying a submit for it.
bool silu_mul(const float* gate, const float* up, float* out, std::size_t count) noexcept;

// The whole FFN half of a PREFILL layer (T > 1) in one submission:
//   t0 = attn_output @ att;  h1 = t0 + x;  h2 = rmsnorm(h1);
//   g  = gate @ h2;  u = up @ h2;  f = silu(g)*u;  d = down @ f;
//   out = h1 + d
// The GEMM (not GEMV) variant of ffn_residual_fused: same eight-dispatch
// shape, but the four projections dispatch the prefill GEMM shader over
// `tokens` rows, so gate/up/the SiLU result stay in VRAM between the gate/up
// GEMMs and the down GEMM - no readback between up and down. `x` ([tokens x
// hidden] residual base) and `attn_out` ([tokens x attn_width]) are staged to
// the upload heaps on entry, so `out` may alias `x`. Returns false (caller
// keeps the separate GEMM + CPU SiLU/add path) when the backend is absent,
// the shape is not Q8_0 GEMM-eligible, DP4A is on (the fused path is fp32
// only), or the working buffers exceed the device-resident cap.
bool ffn_prefill_fused(const std::uint8_t* attn_output, const std::uint8_t* gate,
                       const std::uint8_t* up, const std::uint8_t* down,
                       const float* norm_weight, const float* x, const float* attn_out,
                       float* out, std::size_t tokens, std::size_t hidden,
                       std::size_t attn_width, std::size_t ffn_rows,
                       float epsilon) noexcept;

bool gemv_q8_0_ffn(const std::uint8_t* gate, const std::uint8_t* up, const std::uint8_t* down,
                   const float* x, float* y, std::size_t ffn_rows, std::size_t columns,
                   std::size_t out_rows) noexcept;

}  // namespace zen::d3d12
