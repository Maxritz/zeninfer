// ZenInfer Direct3D 12.1 compute backend.
//
// Scope: the Q8_0 matrix-vector (decode) and matrix-matrix (prefill) products that
// dominate both phases. Everything else stays on the CPU, so the transformer logic
// is untouched and the backend can always fall back.
//
// Why this is worth doing: decode re-reads the entire weight file once per token
// (~1.32 GB for Llama-3.2-1B-Q8_0) from DDR4. The same stream served from GDDR6 is
// an order of magnitude faster, provided the weights are repacked into VRAM once.

#ifdef ZEN_HAS_D3D12

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <intrin.h>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "q8_0_gemm_cso.h"
#include "q8_0_gemm_sh_cso.h"
#include "q8_0_gemm_v2_cso.h"
#include "q8_0_gemm_v3_cso.h"
#include "q8_0_gemm_fused_cso.h"
#include "q8_0_gemm_rt_cso.h"
#include "q8_0_gemm_fused_cso.h"
#include "q8_0_gemm_dp4a_cso.h"
#include "q8_0_gemv_dp4a_cso.h"
#include "quantize_acts_cso.h"
#include "q8_0_gemv_fp32_cso.h"
#include "silu_mul_cso.h"
#include "addn_cso.h"
#include "rmsnorm_cso.h"
#include "rope_cso.h"
#include "rope_prefill_cso.h"
#include "attention_cso.h"
#include "attention2_cso.h"
#include "rmsnorm_fuseq_cso.h"
#include "silu_fuseq_cso.h"
#include "attention_fuseq_cso.h"
#include "flash_attention_cso.h"
#include "repack.h"
#include "zen/d3d12_backend.h"

using Microsoft::WRL::ComPtr;

namespace zen::d3d12 {

// STRACE-style breadcrumb tracing for D3D12 backend
static inline void d3d12_trace(const char* comp, const char* evt, const char* fmt, ...) {
    const char* trace = std::getenv("ZEN_D3D12_TRACE");
    if (!trace || trace[0] != '1') return;
    va_list args;
    va_start(args, fmt);
    std::fprintf(stderr, "STRACE component=%s event=%s ", comp, evt);
    std::vfprintf(stderr, fmt, args);
    std::fprintf(stderr, "\n");
    va_end(args);
}

static inline void d3d12_trace_scope(const char* comp, const char* evt, const char* fmt, ...) {
    const char* trace = std::getenv("ZEN_D3D12_TRACE");
    if (!trace || trace[0] != '1') return;
    va_list args;
    va_start(args, fmt);
    std::fprintf(stderr, "STRACE component=%s event=%s:enter ", comp, evt);
    std::vfprintf(stderr, fmt, args);
    std::fprintf(stderr, "\n");
    va_end(args);
}

static inline void d3d12_trace_scope_end(const char* comp, const char* evt) {
    const char* trace = std::getenv("ZEN_D3D12_TRACE");
    if (!trace || trace[0] != '1') return;
    std::fprintf(stderr, "STRACE component=%s event=%s:exit\n", comp, evt);
}

// Auto RAII scope timer
struct TraceScope {
    const char* comp;
    const char* evt;
    TraceScope(const char* c, const char* e, const char* fmt, ...) : comp(c), evt(e) {
        va_list args;
        va_start(args, fmt);
        d3d12_trace_scope(comp, evt, fmt, args);
        va_end(args);
    }
    ~TraceScope() { d3d12_trace_scope_end(comp, evt); }
};

#define STRACE(comp, evt, fmt, ...) d3d12_trace(comp, evt, fmt, __VA_ARGS__)
#define STRACE_SCOPE(comp, evt, fmt, ...) TraceScope __ts_##__LINE__(comp, evt, fmt, __VA_ARGS__)

// Defined in agility_export.cpp. Declared outside the anonymous namespace so the
// reference below binds to the exported object rather than a new internal symbol.
unsigned agility_version() noexcept;

// ============================================================================
// D3D12 Feature Support Detection (Agility SDK 1.722+)
// ============================================================================

struct D3D12FeatureSupport {
    // OPTIONS12
    bool enhanced_barriers = false;
    bool relaxed_format_casting = false;
    
    // OPTIONS13
    bool unrestricted_buffer_texture_copy = false;
    bool unrestricted_vertex_alignment = false;
    
    // OPTIONS14
    bool advanced_texture_ops = false;
    bool writeable_msaa = false;
    
    // OPTIONS15
    bool triangle_fan = false;
    bool dynamic_index_buffer_strip = false;
    
    // OPTIONS16
    bool dynamic_depth_bias = false;
    bool gpu_upload_heap = false;
    
    // OPTIONS17
    bool non_normalized_samplers = false;
    bool manual_write_tracking = false;
    
    // OPTIONS18
    bool render_passes_valid = false;
    
    // OPTIONS19
    bool mismatching_output_dims = false;
    bool compute_only_custom_heap = false;
    
    // OPTIONS20
    bool compute_only_write_watch = false;
    
    // OPTIONS21
    int work_graphs_tier = 0;  // 0=unsupported, 10=1.0, 11=1.1
    int execute_indirect_tier = 0;
    bool sample_cmp_gradient_bias = false;
    bool extended_command_info = false;
    
    // OPTIONS22 (may not be in all SDKs)
    bool shader_execution_reordering = false;
    bool create_byte_offset_views = false;
    uint32_t max_1d_dispatch = 0;
    uint32_t max_1d_dispatch_mesh = 0;
    
    // Architecture
    bool uma = false;
    uint32_t shader_model = 0;
    int resource_binding_tier = 0;
};

D3D12FeatureSupport g_features;

// Forward declaration
struct State;

// Function declarations (defined after State)
void query_d3d12_features(State& s);
void enhanced_barrier(State& s, ID3D12Resource* resource,
                      D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after);
void enhanced_barrier_begin(State& s, ID3D12Resource* resource,
                            D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after,
                            D3D12_RESOURCE_BARRIER* out_barrier);
void enhanced_barrier_end(State& s, D3D12_RESOURCE_BARRIER* barrier);

// ============================================================================
// Constants
// ============================================================================

constexpr std::uint32_t kElemsPerBlock = 32;
constexpr std::uint32_t kRawBlockBytes = 34;
// One warp per row in q8_0_gemv.hlsl, so 8 rows per 256-thread group. The shader
// derives local_row as tid/32, which only lines up if this stays 8.
constexpr std::uint32_t kRowsPerGroup = 8;
// The GEMM shader blocks 8 output rows per workgroup (ROWS_PER_GROUP in the HLSL),
// which is independent of the GEMV shader's row-block size above.
// Must match ROWS_PER_GROUP in q8_0_gemm.hlsl: the shader takes rows-per-workgroup
// as a compile-time constant, so a mismatch here silently under- or over-dispatches.
// One warp owns one output row, so a thread block covers 32 rows.
constexpr std::uint32_t kGemmRowsPerGroup = 32;
constexpr std::uint32_t kTokensPerTile = 32;
// GEMM v2 (q8_0_gemm_v2.hlsl, ZEN_D3D12_GEMM_V2=1): 256-thread groups of
// 8 rows, 16 tokens per tile. Bit-identical math; different tiling only.
constexpr std::uint32_t kGemmRowsPerGroupV2 = 8;
constexpr std::uint32_t kTokensPerTileV2 = 16;
// GEMM v3 (q8_0_gemm_v3.hlsl, ZEN_D3D12_GEMM_V3=1): 1024-thread groups of
// 32 rows, 8 tokens per tile. Same K order as v1 so bit-identical; the
// smaller tile removes the 32-token waste at verify T=4..8. Groups stay
// 32-wide at 1024 threads; only BLOCK_T (and groupshared, 32 KB -> 8 KB) moves.
constexpr std::uint32_t kGemmRowsPerGroupV3 = 32;
constexpr std::uint32_t kTokensPerTileV3 = 8;
// Below this many elements a dispatch costs more than the arithmetic it saves.
constexpr std::size_t kMinElements = 256u * 1024u;
constexpr std::uint64_t kVRAMBudgetPercent = 80;
constexpr double kSpinWaitUs = 2000.0;

struct GemvParams {
    std::uint32_t rows;
    std::uint32_t blocks_per_row;
    std::uint32_t rows_per_group;
    std::uint32_t reserved;
};

// Matches quantize_acts.hlsl QuantParams: {uint blocks; uint columns;}.
struct QuantParams {
    std::uint32_t blocks;
    std::uint32_t columns;
};

struct GemmParams {
    std::uint32_t rows;
    std::uint32_t blocks_per_row;
    std::uint32_t token_count;
    std::uint32_t tokens_per_tile;
};

struct WeightEntry {
    // One resource holds [quants | 256B pad | scales]. Two committed resources per
    // tensor cost ~0.85ms each (every CreateCommittedResource creates its own implicit
    // heap), measured at 191ms of a 476ms staging pass across 112 tensors. scales_va
    // points into this same buffer, so holding `quants` keeps both alive - which is
    // also what preserves the F-09 lifetime guarantee.
    ComPtr<ID3D12Resource> quants;  // 32 bytes per block, 16-byte aligned
    std::uint64_t scales_va{};      // quants GPU VA + scale_offset
    std::size_t rows{};
    std::size_t columns{};
};

// Concatenated weight table for the pipeline's fused dispatches: one buffer holds
// the part tensors' quant bytes back to back, then all their scale blocks, so a
// single multi-row GEMV covers q|k|v or gate|up. Built by ensure_fused_mat.
struct FusedMat {
    ComPtr<ID3D12Resource> quants{};
    std::uint64_t scales_va{};
    std::size_t rows{};
    std::size_t cols{};
};

// Component tags for the trace table. Each maps to the code path that owns one
// command list (and therefore one fence wait), which is the unit that actually costs.
enum class Comp : int {
    Projections,  // gemv_batch_locked: qkv and attn-out matvec lists
    FfnUpDown,    // gemv_ffn_locked: gate|up + SiLU + down in one list
    FfnFused,     // ffn_residual_fused_locked: 8 dispatches in one list
    Norms,        // rms_norm_locked
    Residual,     // addn_locked
    Count
};

struct State {
    std::mutex mutex;
    bool ready{};
    bool failed{};
    AdapterInfo adapter{};
    Stats stats{};

    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    HANDLE fence_event{};
    std::uint64_t fence_value{};
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> pso_gemv;
    // DP4A decode path (ZEN_D3D12_DP4A=1): int8-activation GEMV + the activation
    // quantizer that produces its input. Null when the CSO failed to build.
    ComPtr<ID3D12PipelineState> pso_gemv_dp4a;
    ComPtr<ID3D12PipelineState> pso_quantize_acts;
    ComPtr<ID3D12PipelineState> pso_rope;
    // Batched prefill RoPE (ZEN_D3D12_ROPE_PREFILL=1, T >= MIN_T on the host).
    // Optional: init does not fail when this PSO is null; rope_apply_prefill then
    // returns false and the caller stays on the CPU rope path. Decode rope is
    // pso_rope above, untouched.
    ComPtr<ID3D12PipelineState> pso_rope_prefill;
    ComPtr<ID3D12PipelineState> pso_attn;
    // attention2 (2 heads per 64-thread wave64), behind ZEN_D3D12_ATTN2=1.
    // Optional: null when its CSO failed to build, in which case attention.hlsl
    // stays selected regardless of the env gate.
    ComPtr<ID3D12PipelineState> pso_attn2;

    // Device-resident KV cache, [layers][kv_heads][context][head_dim] floats, with k and
    // v in separate buffers. Configured once per model by kv_cache_configure; the
    // attention block appends at `pos` with a device-to-device copy, so k and v never
    // travel to the host.
    // Both caches carry D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS and live in
    // UNORDERED_ACCESS permanently: decode writes the current token's k/v rows straight
    // from compute (rope-k and the v matvec bound at the UAV slots) and attention reads
    // them as UAVs, so no per-layer state transition ever touches these 256MB buffers.
    ComPtr<ID3D12Resource> kcache;
    ComPtr<ID3D12Resource> vcache;
    std::size_t kv_layers{};
    std::size_t kv_context{};
    std::size_t kv_heads{};
    std::size_t kv_head_dim{};

    // Resident-prefix record for kv_cache_prefix_resident: per layer, the contiguous
    // position range [first, first + count) known to be resident in that layer's cache
    // slices. Grown by every successful K/V write this backend performs (appends and
    // decode paths), discarded whenever the cache buffers are (re)created or reset.
    // Under-reports on gappy append patterns by design: claiming a position resident
    // that is not would hand decode a wrong attention window.
    struct KvResidentSpan {
        std::size_t first{};
        std::size_t count{};
        bool valid{};
    };
    std::vector<KvResidentSpan> kv_resident;

    // Small CPU-visible operands for the attention block. Upload heaps mean the host
    // writes them directly, so the block needs no staging copy for q/k/v or the table.
    ComPtr<ID3D12Resource> dq, dk, dv, dqr, dkr, dcos, dsin, datt;
    // Third activation arena for the whole-layer path: the first residual base has to
    // stay readable while another buffer is being written, and two arenas cannot hold
    // it alongside everything else.
    ComPtr<ID3D12Resource> hbuf;
    std::size_t h_cap{};
    // Pipeline path per-layer input staging, on the default heap: every layer
    // copies its input here once (xbuf for layer 0, the previous layer's output
    // otherwise) so rms/addn bindings are uniform across layers and no
    // upload-heap resource is ever used as a copy destination.
    ComPtr<ID3D12Resource> hin;
    std::size_t hin_cap{};
    // Pipeline path norm table: one row per (layer, position-in-layer) at 2*layer
    // and 2*layer+1 offsets. A single shared slot overwritten per layer made
    // every layer execute with the last layer's norm weights.
    ComPtr<ID3D12Resource> nrm;
    void* nrm_cpu{};
    std::size_t nrm_cap{};
    void* dq_cpu{};
    void* dk_cpu{};
    void* dv_cpu{};
    void* dqr_cpu{};
    void* dkr_cpu{};
    void* dcos_cpu{};
    void* dsin_cpu{};
    // One capacity per buffer: sharing a single cap made a smaller buffer look
    // large enough as soon as a bigger sibling had been sized first.
    std::size_t dq_cap{}, dk_cap{}, dv_cap{}, dqr_cap{}, dkr_cap{};
    std::size_t dcos_cap{}, dsin_cap{}, datt_cap{};
    ComPtr<ID3D12PipelineState> pso_gemm;
    ComPtr<ID3D12PipelineState> pso_gemm_sh;
    ComPtr<ID3D12PipelineState> pso_gemm_v2;
    ComPtr<ID3D12PipelineState> pso_gemm_v3;
    ComPtr<ID3D12PipelineState> pso_gemm_rt;
    ComPtr<ID3D12PipelineState> pso_gemm_fused;
    ComPtr<ID3D12PipelineState> pso_gemm_dp4a;

    // Per-token activation scales for the DP4A GEMM, one float each.
    ComPtr<ID3D12Resource> gemm_xscale;
    void* gemm_xscale_cpu{};
    std::size_t gemm_xscale_cap{};
    ComPtr<ID3D12PipelineState> pso_silu;
    ComPtr<ID3D12PipelineState> pso_rms;
    ComPtr<ID3D12PipelineState> pso_addn;
    ComPtr<ID3D12PipelineState> pso_flash_attn;
    // Producer-fused quantize variants (ZEN_D3D12_DP4A_FUSEQ=1): rms/silu/
    // attention dispatches that also write the qact int8/scale streams (u3/u4).
    // Null when the CSO failed to build; the unfused path is then used.
    ComPtr<ID3D12PipelineState> pso_rms_fuse;
    ComPtr<ID3D12PipelineState> pso_silu_fuse;
    ComPtr<ID3D12PipelineState> pso_attn_fuse;

    // FlashAttention prefill scratch, reused across layers. Reallocating these per
    // call meant four CreateCommittedResource round trips and a fresh 2 MB
    // upload/readback pair for every layer of every prefill, which cost more than
    // the kernel itself. Grown on demand, never shrunk.
    ComPtr<ID3D12Resource> fa_q;      // device Q [tokens, heads, head_dim]
    ComPtr<ID3D12Resource> fa_out;    // device O (UAV)
    ComPtr<ID3D12Resource> fa_qup;    // upload staging for Q
    ComPtr<ID3D12Resource> fa_rd;     // readback staging for O
    void* fa_qup_cpu{};
    std::size_t fa_cap{};

    // Persistent command allocators/lists for the multi-layer decode pipeline.
    // Creating these inside the per-layer loop cost ~3.7 ms of a 7.9 ms token:
    // CreateCommandAllocator + CreateCommandList are expensive driver calls and
    // must be hoisted out of any hot path and Reset() instead.
    std::vector<ComPtr<ID3D12CommandAllocator>> pipe_allocs;
    std::vector<ComPtr<ID3D12GraphicsCommandList4>> pipe_lists;

    // Async copy queue for KV cache operations (overlaps with compute)
    ComPtr<ID3D12CommandQueue> copy_queue;
    ComPtr<ID3D12CommandAllocator> copy_allocator;
    ComPtr<ID3D12GraphicsCommandList> copy_list;
    ComPtr<ID3D12Fence> copy_fence;
    HANDLE copy_fence_event{};
    std::uint64_t copy_fence_value{};

    ComPtr<ID3D12Resource> params;
    void* params_cpu{};
    ComPtr<ID3D12Resource> xbuf;
    void* x_cpu{};
    std::size_t x_cap{};
    ComPtr<ID3D12Resource> ybuf;
    ComPtr<ID3D12Resource> yread;
    ComPtr<ID3D12Resource> fbuf;
    std::size_t f_cap{};
    // DP4A decode activation-quant staging: [int8 acts (1B/elem) | fp32 scales
    // per 32-elem block], dedicated DEFAULT-heap buffer. It MUST live in a
    // separate resource from fbuf: the DP4A matvec reads the tail as SRV (needs
    // NSR) and writes arena B as UAV in the SAME dispatch - one resource cannot
    // be in two states at once, and SRV reads of UAV-state resources do not work
    // on this runtime. So each site does: quantize (UAV write) -> UAV barrier ->
    // transition UAV -> NSR -> gemv_dp4a (NSR read) -> transition NSR -> UAV
    // (restores the write state for the next site/layer).
    ComPtr<ID3D12Resource> qact;
    std::size_t qact_cap{};
    ComPtr<ID3D12Resource> wbuf;
    void* w_cpu{};
    std::size_t w_cap{};
    // Third upload operand, needed by the fused FFN path: it feeds an activation, a
    // residual base and a norm weight to one command list at the same time.
    ComPtr<ID3D12Resource> nbuf;
    void* n_cpu{};
    std::size_t n_cap{};
    std::size_t y_cap{};
    ComPtr<ID3D12Resource> stage;
    void* stage_cpu{};
    std::size_t stage_cap{};

    bool trace{};
    bool no_readback{};
    ComPtr<ID3D12QueryHeap> ts_heap;
    ComPtr<ID3D12Resource> ts_read;
    std::uint64_t ts_freq{};

    // Per-component trace accumulators (ZEN_D3D12_TRACE=1).
    struct CompStat {
        std::uint64_t ops{};
        std::uint64_t dev_us{};
        std::uint64_t wait_us{};
    };
    CompStat comp_stats[static_cast<int>(Comp::Count)]{};
    Comp current_comp{Comp::Projections};

    std::unordered_map<const void*, WeightEntry> weights;
    std::uint64_t weight_bytes{};
};

// ============================================================================
// D3D12 Feature Support Detection Implementation
// ============================================================================

// Maps legacy D3D12_RESOURCE_STATES to new BARRIER_LAYOUT
D3D12_BARRIER_LAYOUT state_to_layout(D3D12_RESOURCE_STATES state) {
    // Simplified mapping for compute-only resources
    if (state & D3D12_RESOURCE_STATE_UNORDERED_ACCESS) return D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS;
    if (state & D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE) return D3D12_BARRIER_LAYOUT_SHADER_RESOURCE;
    if (state & D3D12_RESOURCE_STATE_COPY_DEST) return D3D12_BARRIER_LAYOUT_COPY_DEST;
    if (state & D3D12_RESOURCE_STATE_COPY_SOURCE) return D3D12_BARRIER_LAYOUT_COPY_SOURCE;
    if (state & D3D12_RESOURCE_STATE_COMMON) return D3D12_BARRIER_LAYOUT_COMMON;
    if (state & D3D12_RESOURCE_STATE_GENERIC_READ) return D3D12_BARRIER_LAYOUT_GENERIC_READ;
    return D3D12_BARRIER_LAYOUT_COMMON;
}

D3D12_BARRIER_SYNC state_to_sync(D3D12_RESOURCE_STATES state) {
    if (state & D3D12_RESOURCE_STATE_UNORDERED_ACCESS) return D3D12_BARRIER_SYNC_COMPUTE_SHADING;
    if (state & D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE) return D3D12_BARRIER_SYNC_COMPUTE_SHADING;
    if (state & D3D12_RESOURCE_STATE_COPY_DEST) return D3D12_BARRIER_SYNC_COPY;
    if (state & D3D12_RESOURCE_STATE_COPY_SOURCE) return D3D12_BARRIER_SYNC_COPY;
    return D3D12_BARRIER_SYNC_NONE;
}

void query_d3d12_features(State& s) {
    // OPTIONS12
    D3D12_FEATURE_DATA_D3D12_OPTIONS12 opts12{};
    if (SUCCEEDED(s.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS12, &opts12, sizeof(opts12)))) {
        g_features.enhanced_barriers = opts12.EnhancedBarriersSupported;
        g_features.relaxed_format_casting = opts12.RelaxedFormatCastingSupported;
    }
    
    // OPTIONS13
    D3D12_FEATURE_DATA_D3D12_OPTIONS13 opts13{};
    if (SUCCEEDED(s.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS13, &opts13, sizeof(opts13)))) {
        g_features.unrestricted_buffer_texture_copy = opts13.UnrestrictedBufferTextureCopyPitchSupported;
        g_features.unrestricted_vertex_alignment = opts13.UnrestrictedVertexElementAlignmentSupported;
    }
    
    // OPTIONS14
    D3D12_FEATURE_DATA_D3D12_OPTIONS14 opts14{};
    if (SUCCEEDED(s.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS14, &opts14, sizeof(opts14)))) {
        g_features.advanced_texture_ops = opts14.AdvancedTextureOpsSupported;
        g_features.writeable_msaa = opts14.WriteableMSAATexturesSupported;
    }
    
    // OPTIONS15
    D3D12_FEATURE_DATA_D3D12_OPTIONS15 opts15{};
    if (SUCCEEDED(s.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS15, &opts15, sizeof(opts15)))) {
        g_features.triangle_fan = opts15.TriangleFanSupported;
        g_features.dynamic_index_buffer_strip = opts15.DynamicIndexBufferStripCutSupported;
    }
    
    // OPTIONS16
    D3D12_FEATURE_DATA_D3D12_OPTIONS16 opts16{};
    if (SUCCEEDED(s.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS16, &opts16, sizeof(opts16)))) {
        g_features.dynamic_depth_bias = opts16.DynamicDepthBiasSupported;
        g_features.gpu_upload_heap = opts16.GPUUploadHeapSupported;
    }
    
    // OPTIONS17
    D3D12_FEATURE_DATA_D3D12_OPTIONS17 opts17{};
    if (SUCCEEDED(s.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS17, &opts17, sizeof(opts17)))) {
        g_features.non_normalized_samplers = opts17.NonNormalizedCoordinateSamplersSupported;
        g_features.manual_write_tracking = opts17.ManualWriteTrackingResourceSupported;
    }
    
    // OPTIONS18
    D3D12_FEATURE_DATA_D3D12_OPTIONS18 opts18{};
    if (SUCCEEDED(s.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS18, &opts18, sizeof(opts18)))) {
        g_features.render_passes_valid = opts18.RenderPassesValid;
    }
    
    // OPTIONS19
    D3D12_FEATURE_DATA_D3D12_OPTIONS19 opts19{};
    if (SUCCEEDED(s.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS19, &opts19, sizeof(opts19)))) {
        g_features.mismatching_output_dims = opts19.MismatchingOutputDimensionsSupported;
        g_features.compute_only_custom_heap = opts19.ComputeOnlyCustomHeapSupported;
    }
    
    // OPTIONS21
    D3D12_FEATURE_DATA_D3D12_OPTIONS21 opts21{};
    if (SUCCEEDED(s.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS21, &opts21, sizeof(opts21)))) {
        g_features.execute_indirect_tier = static_cast<int>(opts21.ExecuteIndirectTier);
        g_features.sample_cmp_gradient_bias = opts21.SampleCmpGradientAndBiasSupported;
        g_features.extended_command_info = opts21.ExtendedCommandInfoSupported;
    }
    
    // OPTIONS22 (may not be in all SDKs - guard with macro check)
#if defined(D3D12_FEATURE_D3D12_OPTIONS22) && defined(D3D12_FEATURE_DATA_D3D12_OPTIONS22)
    D3D12_FEATURE_DATA_D3D12_OPTIONS22 opts22{};
    if (SUCCEEDED(s.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS22, &opts22, sizeof(opts22)))) {
        g_features.shader_execution_reordering = opts22.ShaderExecutionReorderingActuallyReorders;
        g_features.create_byte_offset_views = opts22.CreateByteOffsetViewsSupported;
        g_features.max_1d_dispatch = opts22.Max1DDispatchSize;
        g_features.max_1d_dispatch_mesh = opts22.Max1DDispatchMeshSize;
    }
#endif
    
    // Architecture
    D3D12_FEATURE_DATA_ARCHITECTURE arch{};
    if (SUCCEEDED(s.device->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE, &arch, sizeof(arch)))) {
        g_features.uma = arch.UMA;
    }
    
    D3D12_FEATURE_DATA_SHADER_MODEL sm{};
    sm.HighestShaderModel = D3D_SHADER_MODEL_6_8;
    if (SUCCEEDED(s.device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm, sizeof(sm)))) {
        g_features.shader_model = static_cast<uint32_t>(sm.HighestShaderModel & 0xF);
    }
    
    D3D12_FEATURE_DATA_D3D12_OPTIONS opts{};
    if (SUCCEEDED(s.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &opts, sizeof(opts)))) {
        g_features.resource_binding_tier = static_cast<int>(opts.ResourceBindingTier);
    }
    
    // Update adapter info with new features
    s.adapter.gpu_upload_heap_supported = g_features.gpu_upload_heap;
    
    // Log features
    if (const char* trace = std::getenv("ZEN_D3D12_TRACE"); trace && trace[0] == '1') {
        std::printf("[d3d12] EnhancedBarriers=%d GPUUploadHeap=%d "
                    "ShaderReordering=%d ExecuteIndirectTier=%d ComputeOnlyHeap=%d\n",
                    g_features.enhanced_barriers, g_features.gpu_upload_heap,
                    g_features.shader_execution_reordering,
                    g_features.execute_indirect_tier, g_features.compute_only_custom_heap);
    }
}

// ============================================================================
// Enhanced Barriers Emulation using Legacy D3D12_RESOURCE_BARRIER
// True Enhanced Barriers API (D3D12_BARRIER, D3D12_BUFFER_BARRIER, ID3D12Device10::Barrier)
// requires Agility SDK 1.8+ and Windows 11, which is not available in SDK 1.722.
// This implementation uses legacy D3D12_RESOURCE_BARRIER with BEGIN_ONLY/END_ONLY flags
// to emulate split barriers for overlapping transitions.
// ============================================================================

// Split barrier: begin transition early, end late (legacy D3D12_RESOURCE_BARRIER with BEGIN_ONLY/END_ONLY flags)
void enhanced_barrier_begin(State& s, ID3D12Resource* resource,
                            D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after,
                            D3D12_RESOURCE_BARRIER* out_barrier) {
    out_barrier->Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    out_barrier->Flags = D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY;
    out_barrier->Transition.pResource = resource;
    out_barrier->Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    out_barrier->Transition.StateBefore = before;
    out_barrier->Transition.StateAfter = after;
}

void enhanced_barrier_end(State& s, D3D12_RESOURCE_BARRIER* barrier) {
    if (!barrier->Transition.pResource) return;
    
    barrier->Flags = D3D12_RESOURCE_BARRIER_FLAG_END_ONLY;
    s.list->ResourceBarrier(1, barrier);
    
    barrier->Transition.pResource = nullptr;  // Mark as consumed
}

// Records one transition barrier into a specific command list. The copy-queue
// paths reset s.copy_list, so they must not reuse the s.list-only wrapper:
// barriers recorded into a closed list are silently dropped by the runtime,
// which leaves the copy queue writing a shader-resource and is exactly the kind
// of illegal state that ends in a removed device.
void transition_on(ID3D12GraphicsCommandList* list, ID3D12Resource* resource,
                   D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    list->ResourceBarrier(1, &barrier);
}

// Legacy wrapper for transition() function
void transition(State& s, ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
                D3D12_RESOURCE_STATES after) {
    transition_on(s.list.Get(), resource, before, after);
}

// UAV barrier: every unordered-access command recorded before it on `resource`
// (nullptr = every UAV) completes before any UAV access recorded after it. This is
// what orders the compute-written KV cache rows (rope-k / v matvec) with the attention
// dispatch that reads them, without paying a full state transition on the 256MB cache
// buffers. Unlike a transition, a UAV barrier moves no resource state.
void uav_barrier_on(ID3D12GraphicsCommandList* list, ID3D12Resource* resource) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = resource;
    list->ResourceBarrier(1, &barrier);
}

void uav_barrier(State& s, ID3D12Resource* resource) {
    uav_barrier_on(s.list.Get(), resource);
}

// KV direct-write switch (default ON): rope-k and the v matvec write the current
// token's cache rows straight from compute and attention reads the caches as UAVs.
// ZEN_D3D12_KVDIRECT=0 restores the CopyBufferRegion append path.
bool kv_direct_enabled() {
    const char* e = std::getenv("ZEN_D3D12_KVDIRECT");
    return !(e && e[0] == '0');
}

// Fused LM-head switch (default ON): layer_decode_pipeline also records the final
// rmsnorm, the vocab GEMV and the logits readback into its single command list.
// ZEN_D3D12_LMFUSE=0 makes the backend ignore a passed LmHeadFuse so the caller
// computes the head itself as before.
bool lm_fuse_enabled() {
    const char* e = std::getenv("ZEN_D3D12_LMFUSE");
    return !(e && e[0] == '0');
}

// DP4A decode switch: ZEN_D3D12_DP4A=1 quantises decode GEMV activations to
// int8-activation DP4A. Default OFF: the fp32 path is the parity default.
bool dp4a_enabled() {
    const char* e = std::getenv("ZEN_D3D12_DP4A");
    return e && e[0] == '1';
}

// Scale-hoist GEMM switch: ZEN_D3D12_GEMM_SCALEHOIST=1 selects the scale-hoisted
// Q8_0 GEMM (scale multiplied once per block, mad() accumulation). Default OFF.
bool gemm_sh_enabled() {
    const char* e = std::getenv("ZEN_D3D12_GEMM_SCALEHOIST");
    return e && e[0] == '1';
}

// Producer-fused quantize switch: ZEN_D3D12_DP4A_FUSEQ=1 makes the DP4A
// producer dispatches (rms1, attention, rms2, silu) write the qact int8/scale
// streams themselves, skipping the 4 standalone quantize_acts dispatches per
// layer (64/token). Requires ZEN_D3D12_DP4A=1; without it the flag is ignored
// and the fp32 parity default is unchanged. Default OFF: unfused DP4A only.
bool dp4a_fuseq_enabled() {
    const char* e = std::getenv("ZEN_D3D12_DP4A_FUSEQ");
    return e && e[0] == '1';
}

// GEMM v2 switch (prefill): ZEN_D3D12_GEMM_V2=1 selects the 256-thread-group
// tiling (q8_0_gemm_v2). Bit-identical math; v1 stays default until validated.
bool gemm_v2_enabled() {
    const char* e = std::getenv("ZEN_D3D12_GEMM_V2");
    return e && e[0] == '1';
}

// GEMM v3 switch (speculative verify): ZEN_D3D12_GEMM_V3=1 selects the
// 8-token-tile kernel (q8_0_gemm_v3). Bit-identical math; v1 stays default
// until validated. Takes precedence over v2 (it also changes tiling, but
// v3's tile is the one verify T=4..8 wants); DP4A still outranks both
// because it changes the arithmetic.
bool gemm_v3_enabled() {
    const char* e = std::getenv("ZEN_D3D12_GEMM_V3");
    return e && e[0] == '1';
}

bool gemm_rt_enabled() {
    const char* e = std::getenv("ZEN_D3D12_GEMM_RT");
    return e && e[0] == '1';
}
bool gemm_fused_enabled() {
    const char* e = std::getenv("ZEN_D3D12_GEMM_FUSED");
    return e && e[0] == '1';
}

// attention2 switch (2 heads per 64-thread wave64): ZEN_D3D12_ATTN2=1 selects
// shaders/attention2.hlsl; 0 or unset is the default (attention.hlsl). Bit-
// identical math, so the gate is a pure perf switch, never a numeric one.
bool attn2_enabled() {
    const char* e = std::getenv("ZEN_D3D12_ATTN2");
    return e && e[0] == '1';
}

// Optional context gate for attention2: ZEN_D3D12_ATTN2_MINPOS=N engages the
// 2-head kernel only at pos >= N (0 = any position). Exists so a short-context
// regression, if any, can be excluded without a rebuild.
std::size_t attn2_min_pos() {
    const char* e = std::getenv("ZEN_D3D12_ATTN2_MINPOS");
    if (!e || e[0] == '\0') return 0;
    try { return static_cast<std::size_t>(std::stoul(e)); } catch (...) { return 0; }
}

// True when attention2 should run for this decode position: env gate on, PSO
// built, and the context gate met. Single place for the kill switch.
bool use_attn2(State& s, std::size_t pos) {
    return attn2_enabled() && s.pso_attn2 != nullptr && pos >= attn2_min_pos();
}

// Forward declaration
bool wait_gpu(State& s);
// Forward declaration
bool wait_gpu(State& s);

// ============================================================================
// Work Graph Implementation
// ============================================================================

bool build_work_graph(State& s) {
    if (g_features.work_graphs_tier < 10) {
        return false;
    }
    return false;
}

State& state() {
    static State s;
    return s;
}

// The GEMM shader re-reads the activation tile from global memory once per output
// row instead of staging it through groupshared. Parity tests pass at <=1.9e-07
// relative error across all shapes. Enabled by default for prefill.
bool gemm_opted_in() {
    static const bool value = [] {
        const char* flag = std::getenv("ZEN_D3D12_GEMM_EXPERIMENTAL");
        // Default to enabled; allow opt-out with ZEN_D3D12_GEMM_EXPERIMENTAL=0
        if (flag) return flag[0] != '0';
        return true;
    }();
    return value;
}

bool opted_in() {
    static const bool value = [] {
        const char* backend = std::getenv("ZEN_BACKEND");
        if (backend) {
            std::string name(backend);
            for (char& c : name) {
                c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            }
            if (name == "d3d12" || name == "dx12" || name == "gpu") {
                return true;
            }
        }
        const char* flag = std::getenv("ZEN_D3D12");
        return flag && flag[0] == '1';
    }();
    return value;
}

// When the adapter reports CPU-visible VRAM (i.e. Resizable BAR is live), small
// buffers that the CPU rewrites every dispatch can live in VRAM instead of system
// memory, so the GPU reads them locally instead of crossing PCIe.
//
// A GPU upload heap resource must be created in COMMON and is permanently readable,
// so it never takes a transition barrier; it is still CPU-mappable, so Map() and the
// existing write paths are unchanged. The only new requirement is that a CPU write
// must not race the GPU still reading the previous contents, which holds here because
// every submit path fences with wait_gpu() before the next CPU write.
bool g_gpu_upload_heap = false;

D3D12_HEAP_TYPE upload_heap_type() {
    return g_gpu_upload_heap ? D3D12_HEAP_TYPE_GPU_UPLOAD : D3D12_HEAP_TYPE_UPLOAD;
}

D3D12_RESOURCE_STATES upload_heap_state() {
    return g_gpu_upload_heap ? D3D12_RESOURCE_STATE_COMMON : D3D12_RESOURCE_STATE_GENERIC_READ;
}

// Names the DRED breadcrumb operations, so the faulting dispatch is identified
// instead of inferred. Only the ops this backend records are spelled out; the
// rest fall back to the numeric enum so nothing is silently mislabelled.
const char* breadcrumb_op_name(D3D12_AUTO_BREADCRUMB_OP op) {
    switch (op) {
        case D3D12_AUTO_BREADCRUMB_OP_DISPATCH: return "Dispatch";
        case D3D12_AUTO_BREADCRUMB_OP_COPYBUFFERREGION: return "CopyBufferRegion";
        case D3D12_AUTO_BREADCRUMB_OP_COPYRESOURCE: return "CopyResource";
        case D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER: return "ResourceBarrier";
        case D3D12_AUTO_BREADCRUMB_OP_EXECUTEBUNDLE: return "ExecuteBundle";
        case D3D12_AUTO_BREADCRUMB_OP_EXECUTEINDIRECT: return "ExecuteIndirect";
        case D3D12_AUTO_BREADCRUMB_OP_CLEARUNORDEREDACCESSVIEW: return "ClearUnorderedAccessView";
        case D3D12_AUTO_BREADCRUMB_OP_RESOLVEQUERYDATA: return "ResolveQueryData";
        case D3D12_AUTO_BREADCRUMB_OP_SETMARKER: return "SetMarker";
        case D3D12_AUTO_BREADCRUMB_OP_BEGINEVENT: return "BeginEvent";
        case D3D12_AUTO_BREADCRUMB_OP_ENDEVENT: return "EndEvent";
        default: return "Other";
    }
}

// Device-removal forensics.
//
// Once the device is removed every later CreateCommittedResource fails with
// DXGI_ERROR_DEVICE_REMOVED, so the call that reports the error is never the one
// that caused it. GetDeviceRemovedReason() distinguishes a hang (TDR: the GPU ran
// past the 2 s watchdog) from a fault (bad address / illegal state), and DRED adds
// the faulting VA plus the last completed command-list operation. Printing this at
// the first observation point is what turns "allocation failed" into a root cause.
void report_device_removed(ID3D12Device* device, const char* where) {
    if (!device) {
        return;
    }
    const HRESULT reason = device->GetDeviceRemovedReason();
    if (SUCCEEDED(reason)) {
        return;
    }
    // Every later API call fails with the same reason, so report only the first
    // observation; otherwise one removed device produces tens of thousands of
    // identical lines and buries the real signal.
    static bool reported = false;
    if (reported) {
        return;
    }
    reported = true;
    std::fprintf(stderr, "[d3d12] DEVICE REMOVED (first observed at %s) reason=0x%08x\n",
                 where, static_cast<unsigned>(reason));
    switch (reason) {
        case DXGI_ERROR_DEVICE_HUNG:
            std::fprintf(stderr, "[d3d12]   -> DEVICE_HUNG: a dispatch exceeded the TDR watchdog\n");
            break;
        case DXGI_ERROR_DEVICE_RESET:
            std::fprintf(stderr, "[d3d12]   -> DEVICE_RESET\n");
            break;
        case DXGI_ERROR_DRIVER_INTERNAL_ERROR:
            std::fprintf(stderr, "[d3d12]   -> DRIVER_INTERNAL_ERROR\n");
            break;
        case DXGI_ERROR_INVALID_CALL:
            std::fprintf(stderr, "[d3d12]   -> INVALID_CALL (bad API usage, not a GPU fault)\n");
            break;
        default:
            std::fprintf(stderr,
                         "[d3d12]   -> generic DEVICE_REMOVED; set ZEN_D3D12_DRED=1 for the faulting VA\n");
            break;
    }
    ComPtr<ID3D12DeviceRemovedExtendedData> dred;
    if (SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&dred)))) {
        D3D12_DRED_PAGE_FAULT_OUTPUT pf{};
        if (SUCCEEDED(dred->GetPageFaultAllocationOutput(&pf)) && pf.PageFaultVA != 0) {
            std::fprintf(stderr, "[d3d12]   DRED page-fault VA=0x%llx\n",
                         static_cast<unsigned long long>(pf.PageFaultVA));
        }
        D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT bc{};
        if (SUCCEEDED(dred->GetAutoBreadcrumbsOutput(&bc))) {
            const D3D12_AUTO_BREADCRUMB_NODE* node = bc.pHeadAutoBreadcrumbNode;
            for (int i = 0; node && i < 4; ++i, node = node->pNext) {
                const UINT32 done = node->pLastBreadcrumbValue ? *node->pLastBreadcrumbValue : 0u;
                std::fprintf(stderr,
                             "[d3d12]   DRED cmdlist ops=%u last_completed=%u\n",
                             node->BreadcrumbCount, done);
                // Name every op in the list plus the one that did not complete.
                if (node->pCommandHistory && node->BreadcrumbCount) {
                    for (UINT32 k = 0; k < node->BreadcrumbCount; ++k) {
                        std::fprintf(stderr, "[d3d12]     op[%u]=%s%s\n", k,
                                     breadcrumb_op_name(node->pCommandHistory[k]),
                                     k == done ? "   <== NOT COMPLETED" : "");
                    }
                }
            }
        }
    }
}

ComPtr<ID3D12Resource> create_buffer(ID3D12Device* device, std::uint64_t bytes,
                                     D3D12_HEAP_TYPE heap, D3D12_RESOURCE_STATES state,
                                     D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
    D3D12_HEAP_PROPERTIES props{};
    props.Type = heap;
    props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    props.CreationNodeMask = 1;
    props.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes ? bytes : 4;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;

    ComPtr<ID3D12Resource> res;
    HRESULT hr = device->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                               IID_PPV_ARGS(&res));
    const char* trace_env = std::getenv("ZEN_D3D12_TRACE");
    if (FAILED(hr)) {
        // Report the first removal once; afterwards every allocation fails and the
        // log would be nothing but repeats of the same reason code.
        report_device_removed(device, "create_buffer");
        if (trace_env && trace_env[0] == '1') {
            static bool printed = false;
            if (!printed) {
                printed = true;
                std::printf("[d3d12] create_buffer FAILED: bytes=%llu heap=%d state=0x%x hr=0x%08x\n",
                            (unsigned long long)bytes, heap, state, hr);
            }
        }
        return nullptr;
    }
    return res;
}

// Adds device time to a component. The op count and fence wait are attributed
// centrally by execute_and_wait(), so this only fills in the GPU-time column.
void note_dev(State& s, Comp c, std::uint64_t dev_us) {
    if (!s.trace) {
        return;
    }
    s.comp_stats[static_cast<int>(c)].dev_us += dev_us;
}

// Prints the per-component decomposition. dev_us comes from GPU timestamp queries,
// wait_us from the fence; the difference is time the queue spent idle waiting for the
// GPU, which is where fixed per-submission latency shows up.
void print_trace_report() {
    State& s = state();
    if (!s.trace) {
        std::cout << "[trace] per-op tracing disabled (set ZEN_D3D12_TRACE=1)" << std::endl;
        return;
    }
    std::cout << "\n     component        ops  submits     dev us    wait us    idle us"
              << std::endl;
    std::uint64_t tot_dev = 0, tot_wait = 0, tot_ops = 0;
    auto row = [](const char* name, const State::CompStat& e) {
        const std::uint64_t idle = e.wait_us > e.dev_us ? e.wait_us - e.dev_us : 0;
        std::printf("     %-15s %5llu %8llu %10llu %10llu %10llu\n", name,
                    static_cast<unsigned long long>(e.ops),
                    static_cast<unsigned long long>(e.ops),
                    static_cast<unsigned long long>(e.dev_us),
                    static_cast<unsigned long long>(e.wait_us),
                    static_cast<unsigned long long>(idle));
    };
    for (int i = 0; i < static_cast<int>(Comp::Count); ++i) {
        const auto& e = s.comp_stats[i];
        if (!e.ops) {
            continue;
        }
        const char* name = i == static_cast<int>(Comp::Projections) ? "projections"
                           : i == static_cast<int>(Comp::FfnUpDown) ? "ffn"
                           : i == static_cast<int>(Comp::FfnFused)   ? "ffn-fused"
                           : i == static_cast<int>(Comp::Norms)      ? "norms"
                                                                     : "residual";
        row(name, e);
        tot_ops += e.ops;
        tot_dev += e.dev_us;
        tot_wait += e.wait_us;
    }
    const std::uint64_t tot_idle = tot_wait > tot_dev ? tot_wait - tot_dev : 0;
    std::printf("     %-15s %5llu %8llu %10llu %10llu %10llu\n", "TOTAL",
                static_cast<unsigned long long>(tot_ops),
                static_cast<unsigned long long>(tot_ops),
                static_cast<unsigned long long>(tot_dev),
                static_cast<unsigned long long>(tot_wait),
                static_cast<unsigned long long>(tot_idle));
    if (tot_wait) {
        std::printf("     idle share of fence time: %.1f%%\n",
                    100.0 * static_cast<double>(tot_idle) / static_cast<double>(tot_wait));
    }
    std::cout.flush();
}

bool wait_gpu(State& s);

bool wait_gpu(State& s) {
    const std::uint64_t value = ++s.fence_value;
    if (FAILED(s.queue->Signal(s.fence.Get(), value))) {
        return false;
    }
    if (s.fence->GetCompletedValue() < value) {
        // A decode-sized dispatch retires in tens of microseconds, while the
        // SetEventOnCompletion/WaitForSingleObject round trip alone costs more
        // than that. Spin for a bounded window, then block for anything longer.
        LARGE_INTEGER frequency{}, started{}, now{};
        QueryPerformanceFrequency(&frequency);
        QueryPerformanceCounter(&started);
        for (;;) {
            if (s.fence->GetCompletedValue() >= value) {
                return true;
            }
            _mm_pause();
            QueryPerformanceCounter(&now);
            const double elapsed_us = static_cast<double>(now.QuadPart - started.QuadPart) *
                                      1000000.0 / static_cast<double>(frequency.QuadPart);
            if (elapsed_us > kSpinWaitUs) {
                break;
            }
        }
        if (s.fence->GetCompletedValue() < value) {
            if (FAILED(s.fence->SetEventOnCompletion(value, s.fence_event))) {
                return false;
            }
            WaitForSingleObject(s.fence_event, INFINITE);
        }
    }
    // A TDR'd dispatch still signals the fence, so "the wait returned" is not proof
    // the work succeeded. Ask the device directly before the caller reuses anything.
    if (FAILED(s.device->GetDeviceRemovedReason())) {
        report_device_removed(s.device.Get(), "wait_gpu");
        return false;
    }
    return true;
}

// Submits the closed command list and blocks to completion, recording the CPU-side
// queue cost separately from the fence wait. The split decides the fix: a large
// queue_us means the driver is slow to accept many small command lists (fewer, bigger
// lists), a large wait_us means the GPU is slow to retire them (fewer submissions).
bool execute_and_wait(State& s) {
    const auto queue_start = std::chrono::steady_clock::now();
    ID3D12CommandList* lists[] = {s.list.Get()};
    ++s.stats.submits;
    s.queue->ExecuteCommandLists(1, lists);
    s.stats.queue_us += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - queue_start)
            .count());

    const auto fence_start = std::chrono::steady_clock::now();
    if (!wait_gpu(s)) {
        return false;
    }
    const std::uint64_t waited = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - fence_start)
            .count());
    s.stats.wait_us += waited;
    if (s.trace) {
        auto& e = s.comp_stats[static_cast<int>(s.current_comp)];
        ++e.ops;
        e.wait_us += waited;
    }
    return true;
}

// Adds the GPU time between stamps[lo] and stamps[hi] as resolved by the retired
// command list. No-op without the trace flag, and never trusts a non-increasing pair.
void accumulate_gpu_us(State& s, std::uint64_t& out, unsigned lo, unsigned hi) {
    if (!s.trace) {
        return;
    }
    std::uint64_t* stamps = nullptr;
    const D3D12_RANGE range{0, 3 * sizeof(std::uint64_t)};
    if (FAILED(s.ts_read->Map(0, &range, reinterpret_cast<void**>(&stamps)))) {
        return;
    }
    if (stamps[hi] > stamps[lo]) {
        out += static_cast<std::uint64_t>(
            static_cast<double>(stamps[hi] - stamps[lo]) * 1000000.0 /
            static_cast<double>(s.ts_freq));
    }
    const D3D12_RANGE none{0, 0};
    s.ts_read->Unmap(0, &none);
}

bool build_root_signature(State& s) {
    // Slot 5 is the per-token activation scale the GEMM needs now that it runs on
    // DP4A (int8 activations). Unused by every other shader.
    D3D12_ROOT_PARAMETER params[10]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    params[0].Descriptor.ShaderRegister = 0;
    params[0].Descriptor.RegisterSpace = 0;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    for (std::uint32_t i = 0; i < 4; ++i) {
        params[1 + i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        params[1 + i].Descriptor.ShaderRegister = i;
        params[1 + i].Descriptor.RegisterSpace = 0;
        params[1 + i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }
    params[5].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[5].Descriptor.ShaderRegister = 0;
    params[5].Descriptor.RegisterSpace = 0;
    params[5].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // Slots 6/7 are the KV cache UAVs (u1/u2). The attention kernel reads the caches as
    // RWByteAddressBuffer so decode can keep them in UNORDERED_ACCESS permanently and
    // write the current token's rows straight from compute - no per-layer transitions
    // and no CopyBufferRegion on the 256MB cache buffers. Shaders that do not declare
    // u1/u2 simply ignore the two extra root arguments.
    params[6].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[6].Descriptor.ShaderRegister = 1;
    params[6].Descriptor.RegisterSpace = 0;
    params[6].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[7].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[7].Descriptor.ShaderRegister = 2;
    params[7].Descriptor.RegisterSpace = 0;
    params[7].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // Slots 8/9 are the producer-fused quantize UAVs (u3/u4): the qact int8
    // acts + scale table the *_fuseq producer kernels (rms/silu/attention)
    // write alongside their fp32 output. Unbound (null descriptor) for every
    // other shader, which does not declare u3/u4.
    params[8].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[8].Descriptor.ShaderRegister = 3;
    params[8].Descriptor.RegisterSpace = 0;
    params[8].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[9].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[9].Descriptor.ShaderRegister = 4;
    params[9].Descriptor.RegisterSpace = 0;
    params[9].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC desc{};
    desc.NumParameters = 10;
    desc.pParameters = params;
    desc.NumStaticSamplers = 0;
    desc.pStaticSamplers = nullptr;
    desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ComPtr<ID3DBlob> blob;
    ComPtr<ID3DBlob> error;
    if (FAILED(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1,
                                           blob.GetAddressOf(), error.GetAddressOf()))) {
        return false;
    }
    return SUCCEEDED(s.device->CreateRootSignature(0, blob->GetBufferPointer(),
                                                   blob->GetBufferSize(),
                                                   IID_PPV_ARGS(&s.root)));
}

ComPtr<ID3D12PipelineState> build_pso(State& s, const void* code, std::size_t size) {
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = s.root.Get();
    desc.CS.pShaderBytecode = code;
    desc.CS.BytecodeLength = size;
    desc.NodeMask = 0;
    desc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;
    
    // Enable Shader Execution Reordering (SER) when supported (OPTIONS22)
    if (g_features.shader_execution_reordering) {
        // Check if the device supports the flag
        #if defined(D3D12_PIPELINE_STATE_FLAG_ALLOW_SHADER_EXECUTION_REORDERING)
        desc.Flags |= D3D12_PIPELINE_STATE_FLAG_ALLOW_SHADER_EXECUTION_REORDERING;
        #endif
    }
    
    ComPtr<ID3D12PipelineState> pso;
    if (FAILED(s.device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso)))) {
        // Retry without SER flag if it fails
        desc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;
        if (FAILED(s.device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso)))) {
            return nullptr;
        }
    }
    return pso;
}

bool reserve_scratch(State& s, std::size_t x_bytes, std::size_t y_bytes);

bool init_locked(State& s) {
    if (s.ready) {
        return true;
    }
    if (s.failed) {
        return false;
    }

    // Set trace flag early so failure paths can report
    const char* trace_env = std::getenv("ZEN_D3D12_TRACE");
    s.trace = trace_env && trace_env[0] == '1';

    auto fail = [&s] {
        s.failed = true;
        return false;
    };

    // Require a genuine 12.1 device: the shaders are SM 6.0 but the feature level
    // is what the caller asked for, and refusing early keeps the failure obvious.
    // Opt-in validation: ZEN_D3D12_DRED=1 turns on DRED (fault VA + breadcrumbs) and
    // the debug layer, which is the only way to see an illegal resource state or an
    // out-of-bounds descriptor before the device disappears.
    {
        const char* dred_env = std::getenv("ZEN_D3D12_DRED");
        if (dred_env && dred_env[0] == '1') {
            ComPtr<ID3D12DeviceRemovedExtendedDataSettings> dred_settings;
            if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dred_settings)))) {
                dred_settings->SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                dred_settings->SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            }
            ComPtr<ID3D12Debug> debug;
            if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) {
                debug->EnableDebugLayer();
            }
        }
    }
    if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(&s.device)))) {
        return fail();
    }

    D3D_FEATURE_LEVEL requested[] = {D3D_FEATURE_LEVEL_12_2, D3D_FEATURE_LEVEL_12_1};
    D3D12_FEATURE_DATA_FEATURE_LEVELS levels{};
    levels.pFeatureLevelsRequested = requested;
    levels.NumFeatureLevels = 2;
    if (SUCCEEDED(s.device->CheckFeatureSupport(D3D12_FEATURE_FEATURE_LEVELS, &levels,
                                                sizeof(levels)))) {
        s.adapter.feature_level =
            (levels.MaxSupportedFeatureLevel == D3D_FEATURE_LEVEL_12_2) ? 122 : 121;
    }

    // Agility SDK activation and ReBAR evidence.
    s.adapter.agility_sdk_version = agility_version();
    // Query ALL D3D12 options (12-22) for modern features
    query_d3d12_features(s);
    
    // GPU upload heaps are the only D3D12 feature that consumes Resizable BAR, and
    // on a discrete GPU the runtime only reports them when the whole VRAM is
    // CPU-visible. So this single flag is the only in-process answer to "is ReBAR
    // actually live here"; there is no dedicated ReBAR query.
    // Enable by default when ReBAR is live (GPU upload heap supported on discrete GPU)
    // since it eliminates PCIe crossings for per-token activation/param buffers.
    if (s.adapter.gpu_upload_heap_supported && !g_features.uma) {
        g_gpu_upload_heap = true;
    }
    // Allow explicit override
    if (const char* e = std::getenv("ZEN_D3D12_GPU_UPLOAD"); e) {
        g_gpu_upload_heap = (e[0] == '1') && s.adapter.gpu_upload_heap_supported;
    }

    if (ComPtr<IDXGIFactory4> factory;
        SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
        ComPtr<IDXGIAdapter1> adapter;
        const LUID luid = s.device->GetAdapterLuid();
        if (SUCCEEDED(factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter)))) {
            DXGI_ADAPTER_DESC desc{};
            if (SUCCEEDED(adapter->GetDesc(&desc))) {
                WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, s.adapter.name,
                                    static_cast<int>(sizeof(s.adapter.name)), nullptr, nullptr);
                s.adapter.dedicated_vram = desc.DedicatedVideoMemory;
                s.adapter.shared_memory = desc.SharedSystemMemory;
            }
        }
    }

    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    if (FAILED(s.device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&s.queue)))) {
        return fail();
    }
    if (FAILED(s.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                                IID_PPV_ARGS(&s.allocator)))) {
        return fail();
    }
    if (FAILED(s.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, s.allocator.Get(),
                                           nullptr, IID_PPV_ARGS(&s.list)))) {
        return fail();
    }
    s.list->Close();
    if (FAILED(s.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&s.fence)))) {
        return fail();
    }
    s.fence_event = CreateEventExW(nullptr, nullptr, 0, EVENT_ALL_ACCESS);
    if (!s.fence_event) {
        return fail();
    }

    // Create async copy queue for KV cache operations
    D3D12_COMMAND_QUEUE_DESC copy_queue_desc{};
    copy_queue_desc.Type = D3D12_COMMAND_LIST_TYPE_COPY;
    copy_queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    if (FAILED(s.device->CreateCommandQueue(&copy_queue_desc, IID_PPV_ARGS(&s.copy_queue)))) {
        return fail();
    }
    if (FAILED(s.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY,
                                                IID_PPV_ARGS(&s.copy_allocator)))) {
        return fail();
    }
    if (FAILED(s.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, s.copy_allocator.Get(),
                                           nullptr, IID_PPV_ARGS(&s.copy_list)))) {
        return fail();
    }
    s.copy_list->Close();
    if (FAILED(s.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&s.copy_fence)))) {
        return fail();
    }
    s.copy_fence_event = CreateEventExW(nullptr, nullptr, 0, EVENT_ALL_ACCESS);
    if (!s.copy_fence_event) {
        return fail();
    }

    if (!build_root_signature(s)) {
        return fail();
    }
    s.pso_gemv = build_pso(s, zen_dx12_shaders::kQ8_0GemvCso, zen_dx12_shaders::kQ8_0GemvCsoSize);
    // DP4A decode path: built but never required (fall back to fp32 gemv when the
    // CSO is missing or the env gate is off).
    s.pso_gemv_dp4a = build_pso(s, zen_dx12_shaders::kQ8_0GemvDp4aCso, zen_dx12_shaders::kQ8_0GemvDp4aCsoSize);
    s.pso_quantize_acts = build_pso(s, zen_dx12_shaders::kQuantizeActsCso, zen_dx12_shaders::kQuantizeActsCsoSize);
    s.pso_rope = build_pso(s, zen_dx12_shaders::kRopeCso, zen_dx12_shaders::kRopeCsoSize);
    // Batched prefill RoPE: optional (default-off path). Null only disables the
    // GPU prefill rope; init must still succeed so the CPU fallback runs.
    s.pso_rope_prefill = build_pso(s, zen_dx12_shaders::kRopePrefillCso,
                                   zen_dx12_shaders::kRopePrefillCsoSize);
    s.pso_attn = build_pso(s, zen_dx12_shaders::kAttnCso, zen_dx12_shaders::kAttnCsoSize);
    // attention2 is optional: a failed build leaves pso_attn2 null and the
    // ZEN_D3D12_ATTN2=1 selection falls back to attention.hlsl, never fails init.
    s.pso_attn2 = build_pso(s, zen_dx12_shaders::kAttn2Cso, zen_dx12_shaders::kAttn2CsoSize);
    s.pso_rms = build_pso(s, zen_dx12_shaders::kRmsNormCso, zen_dx12_shaders::kRmsNormCsoSize);
    s.pso_silu = build_pso(s, zen_dx12_shaders::kSiluMulCso, zen_dx12_shaders::kSiluMulCsoSize);
    s.pso_gemm = build_pso(s, zen_dx12_shaders::kQ8_0GemmCso, zen_dx12_shaders::kQ8_0GemmCsoSize);
    s.pso_gemm_sh = build_pso(s, zen_dx12_shaders::kQ8_0GemmShCso, zen_dx12_shaders::kQ8_0GemmShCsoSize);
    // GEMM v2 (256-thread groups, halved accumulators, bit-identical): optional,
    // selected by ZEN_D3D12_GEMM_V2=1; v1 stays the default until validated.
    s.pso_gemm_v2 = build_pso(s, zen_dx12_shaders::kQ8_0GemmV2Cso, zen_dx12_shaders::kQ8_0GemmV2CsoSize);
    // GEMM v3 (verify tile BLOCK_T=8, bit-identical): optional,
    // selected by ZEN_D3D12_GEMM_V3=1; v1 stays the default until validated.
    s.pso_gemm_v3 = build_pso(s, zen_dx12_shaders::kQ8_0GemmV3Cso, zen_dx12_shaders::kQ8_0GemmV3CsoSize);
    s.pso_gemm_rt = build_pso(s, zen_dx12_shaders::kQ8_0GemmRtCso, zen_dx12_shaders::kQ8_0GemmRtCsoSize);
    s.pso_gemm_fused = build_pso(s, zen_dx12_shaders::kQ8_0GemmFusedCso, zen_dx12_shaders::kQ8_0GemmFusedCsoSize);
    s.pso_gemm_dp4a = build_pso(s, zen_dx12_shaders::kQ8_0GemmDp4aCso,
                                zen_dx12_shaders::kQ8_0GemmDp4aCsoSize);
    s.pso_addn = build_pso(s, zen_dx12_shaders::kAddnCso, zen_dx12_shaders::kAddnCsoSize);
    s.pso_flash_attn = build_pso(s, zen_dx12_shaders::kFlashAttnCso, zen_dx12_shaders::kFlashAttnCsoSize);
    // Producer-fused quantize variants (ZEN_D3D12_DP4A_FUSEQ=1): optional,
    // like the DP4A pair above - a missing CSO just disables the fused path.
    s.pso_rms_fuse = build_pso(s, zen_dx12_shaders::kRmsNormFusedCso,
                               zen_dx12_shaders::kRmsNormFusedCsoSize);
    s.pso_silu_fuse = build_pso(s, zen_dx12_shaders::kSiluFusedCso,
                                zen_dx12_shaders::kSiluFusedCsoSize);
    s.pso_attn_fuse = build_pso(s, zen_dx12_shaders::kAttnFusedCso,
                                zen_dx12_shaders::kAttnFusedCsoSize);
    // The fp32 and DP4A gemv CSOs are mandatory (decode needs one of them).
    if (!s.pso_gemv || !s.pso_gemm || !s.pso_silu || !s.pso_rms || !s.pso_addn ||
        !s.pso_rope || !s.pso_attn || !s.pso_flash_attn) {
        return fail();
    }

    // Two slots past the batch range are reserved for the fused FFN path's two
    // residual adds, which must not alias a batch slot that a matvec group uses.
    // 24576 bytes: the legacy slots below 8192 are unchanged - lone matvec (0),
    // gemm/rms (256), addn (768), the batch slots (512+256*i), rope (1024/1536),
    // attention (1280), the whole-layer path's seven gemv slots, two norms, two
    // residual adds and a SiLU (4096..6928) and the fused LM head (7168/7424).
    // Above 8192 live the pipeline's per-layer attention parameters, one 256B slot
    // per layer (root CBV addresses must be 256B aligned, hence the stride): the
    // attention cbuffer is the only per-layer-varying parameter set (kv_base/
    // v_base), and a shared slot would make every layer execute with the last
    // layer's cache-slice offsets because the GPU reads upload-heap CBVs at
    // execution time, after all record-time writes are done. 64 layers max
    // (layer_decode_pipeline cap) -> 8192 + 64*256 = 24576.
    s.params = create_buffer(s.device.Get(), 8192 + 64 * 256, upload_heap_type(),
                             upload_heap_state());
    if (!s.params) {
        return fail();
    }
    if (FAILED(s.params->Map(0, nullptr, &s.params_cpu))) {
        return fail();
    }

    if (s.trace) {
        D3D12_QUERY_HEAP_DESC query_desc{};
        query_desc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        // Pipeline needs 16 layers * 2 queries = 32. Prefill needs 4. Use max.
        query_desc.Count = 64;
        query_desc.NodeMask = 0;
        if (FAILED(s.device->CreateQueryHeap(&query_desc, IID_PPV_ARGS(&s.ts_heap)))) {
            return fail();
        }
        s.ts_read = create_buffer(s.device.Get(), 64 * sizeof(std::uint64_t),
                                  D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        if (!s.ts_read) {
            return fail();
        }
        UINT64 frequency = 0;
        if (FAILED(s.queue->GetTimestampFrequency(&frequency)) || !frequency) {
            return fail();
        }
        s.ts_freq = frequency;
    }

    // Diagnostic floor probe: what does an empty submit + fence round trip cost?
    // Whatever remains above the real work in gemv is this, not arithmetic.
    if (const char* probe = std::getenv("ZEN_D3D12_BENCH_SUBMIT"); probe && probe[0] == '1') {
        double total = 0.0;
        constexpr int kRounds = 200;
        for (int i = 0; i < kRounds; ++i) {
            s.allocator->Reset();
            s.list->Reset(s.allocator.Get(), nullptr);
            s.list->Close();
            const auto start = std::chrono::steady_clock::now();
            ID3D12CommandList* lists[] = {s.list.Get()};
            s.queue->ExecuteCommandLists(1, lists);
            wait_gpu(s);
            total += std::chrono::duration<double, std::micro>(
                         std::chrono::steady_clock::now() - start)
                         .count();
        }
        std::cerr << "[d3d12] empty submit round trip: " << (total / kRounds) << " us" << std::endl;

        // How does the round trip scale with the number of dispatches in one
        // submission? A flat fixed cost means batching calls is the only lever.
        if (!reserve_scratch(s, 64 * 1024, 8 * 1024)) {
            return fail();
        }
        auto* probe_params = static_cast<GemvParams*>(s.params_cpu);
        probe_params->rows = 64;
        probe_params->blocks_per_row = 1;
        probe_params->rows_per_group = kRowsPerGroup;
        probe_params->reserved = 0;
        for (UINT count : {1u, 2u, 4u, 8u, 16u}) {
            double batch_total = 0.0;
            for (int i = 0; i < 50; ++i) {
                s.allocator->Reset();
                s.list->Reset(s.allocator.Get(), s.pso_gemv.Get());
                s.list->SetComputeRootSignature(s.root.Get());
                s.list->SetComputeRootConstantBufferView(0, s.params->GetGPUVirtualAddress());
                s.list->SetComputeRootShaderResourceView(1, s.xbuf->GetGPUVirtualAddress());
                s.list->SetComputeRootShaderResourceView(2, s.xbuf->GetGPUVirtualAddress());
                s.list->SetComputeRootShaderResourceView(3, s.xbuf->GetGPUVirtualAddress());
                s.list->SetComputeRootUnorderedAccessView(5, s.ybuf->GetGPUVirtualAddress());
                for (UINT d = 0; d < count; ++d) {
                    s.list->Dispatch(16, 1, 1);
                }
                s.list->Close();
                const auto start = std::chrono::steady_clock::now();
                ID3D12CommandList* lists[] = {s.list.Get()};
                s.queue->ExecuteCommandLists(1, lists);
                wait_gpu(s);
                batch_total += std::chrono::duration<double, std::micro>(
                                   std::chrono::steady_clock::now() - start)
                                   .count();
            }
            std::cerr << "[d3d12] submit round trip with " << count << " dispatches: "
                      << (batch_total / 50.0) << " us" << std::endl;
        }
    }

    // ZEN_D3D12_BENCH_BARRIER: what do the per-layer transition patterns cost in
    // isolation? The pipeline op stamps place ~163us between rope-k and the k/v
    // cache copies (a copy of 4 KB total), so each pattern gets its own fenced
    // round trip here:
    //   v0: plain copy (COMMON -> COMMON, no transitions)
    //   v1: v0 + 16 tiny dispatches writing a UAV buffer (ybuf stays UAV)
    //   v2: v1 + the pipeline's UAV->COPY_SOURCE flush on that written buffer
    //   v3: the NSR->COPY_DEST->NSR cycling used for the per-layer input staging
    // Each number is a 200-round mean of submit+wait; the empty-submit round trip
    // is re-measured so the ceiling matches this run exactly.
    if (const char* bb = std::getenv("ZEN_D3D12_BENCH_BARRIER"); bb && bb[0] == '1') {
        if (!reserve_scratch(s, 64 * 1024, 8 * 1024)) {
            return fail();
        }
        ComPtr<ID3D12Resource> csrc = create_buffer(s.device.Get(), 4096,
                                                   D3D12_HEAP_TYPE_DEFAULT,
                                                   D3D12_RESOURCE_STATE_COMMON);
        ComPtr<ID3D12Resource> cdst = create_buffer(s.device.Get(), 4096,
                                                   D3D12_HEAP_TYPE_DEFAULT,
                                                   D3D12_RESOURCE_STATE_COMMON);
        ComPtr<ID3D12Resource> ccycle = create_buffer(s.device.Get(), 4096,
                                                     D3D12_HEAP_TYPE_DEFAULT,
                                                     D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        if (!csrc || !cdst || !ccycle) {
            return fail();
        }
        auto* par = static_cast<GemvParams*>(s.params_cpu);
        par->rows = 64;
        par->blocks_per_row = 1;
        par->rows_per_group = kRowsPerGroup;
        par->reserved = 0;

        double empty_total = 0.0;
        for (int i = 0; i < 200; ++i) {
            s.allocator->Reset();
            s.list->Reset(s.allocator.Get(), nullptr);
            s.list->Close();
            const auto st = std::chrono::steady_clock::now();
            ID3D12CommandList* lists[] = {s.list.Get()};
            s.queue->ExecuteCommandLists(1, lists);
            wait_gpu(s);
            empty_total += std::chrono::duration<double, std::micro>(
                               std::chrono::steady_clock::now() - st).count();
        }
        const double empty_us = empty_total / 200.0;

        auto rtt = [&](const auto& record) {
            double total = 0.0;
            for (int i = 0; i < 200; ++i) {
                s.allocator->Reset();
                s.list->Reset(s.allocator.Get(), s.pso_gemv.Get());
                s.list->SetComputeRootSignature(s.root.Get());
                record();
                if (FAILED(s.list->Close())) break;
                const auto st = std::chrono::steady_clock::now();
                ID3D12CommandList* lists[] = {s.list.Get()};
                s.queue->ExecuteCommandLists(1, lists);
                wait_gpu(s);
                total += std::chrono::duration<double, std::micro>(
                             std::chrono::steady_clock::now() - st).count();
            }
            return total / 200.0;
        };
        auto dummy16 = [&] {
            s.list->SetPipelineState(s.pso_gemv.Get());
            s.list->SetComputeRootConstantBufferView(0, s.params->GetGPUVirtualAddress());
            s.list->SetComputeRootShaderResourceView(1, s.xbuf->GetGPUVirtualAddress());
            s.list->SetComputeRootShaderResourceView(2, s.xbuf->GetGPUVirtualAddress());
            s.list->SetComputeRootShaderResourceView(3, s.xbuf->GetGPUVirtualAddress());
            s.list->SetComputeRootUnorderedAccessView(5, s.ybuf->GetGPUVirtualAddress());
            for (UINT d = 0; d < 16; ++d) {
                s.list->Dispatch(16, 1, 1);
            }
        };

        const double v0 = rtt([&] {
            s.list->CopyBufferRegion(cdst.Get(), 0, csrc.Get(), 0, 4096);
        });
        const double v1 = rtt([&] {
            dummy16();
            s.list->CopyBufferRegion(cdst.Get(), 0, csrc.Get(), 0, 4096);
        });
        const double v2 = rtt([&] {
            dummy16();
            transition_on(s.list.Get(), s.ybuf.Get(),
                          D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                          D3D12_RESOURCE_STATE_COPY_SOURCE);
            s.list->CopyBufferRegion(cdst.Get(), 0, s.ybuf.Get(), 0, 4096);
            transition_on(s.list.Get(), s.ybuf.Get(),
                          D3D12_RESOURCE_STATE_COPY_SOURCE,
                          D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        });
        const double v3 = rtt([&] {
            transition_on(s.list.Get(), ccycle.Get(),
                          D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                          D3D12_RESOURCE_STATE_COPY_DEST);
            s.list->CopyBufferRegion(ccycle.Get(), 0, csrc.Get(), 0, 4096);
            transition_on(s.list.Get(), ccycle.Get(),
                          D3D12_RESOURCE_STATE_COPY_DEST,
                          D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        });
        // Large-destination variants: the pipeline's k/v cache append copies
        // into 256 MB cache buffers, while the per-layer input copies into an
        // 8 KB buffer show no bubble at all. Size of the destination is the
        // remaining discriminator; v4/v5/v6/v9 separate barrier-on-big vs
        // copy-into-big vs per-copy marginal cost.
        const std::uint64_t big_bytes = 256ull * 1024 * 1024;
        ComPtr<ID3D12Resource> big_nsr = create_buffer(s.device.Get(), big_bytes,
                                                      D3D12_HEAP_TYPE_DEFAULT,
                                                      D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        ComPtr<ID3D12Resource> big_cmn = create_buffer(s.device.Get(), big_bytes,
                                                      D3D12_HEAP_TYPE_DEFAULT,
                                                      D3D12_RESOURCE_STATE_COMMON);
        if (!big_nsr || !big_cmn) {
            return fail();
        }
        const double v4 = rtt([&] { // cache-append pattern: cycle big NSR + 2 KB copy
            transition_on(s.list.Get(), big_nsr.Get(),
                          D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                          D3D12_RESOURCE_STATE_COPY_DEST);
            s.list->CopyBufferRegion(big_nsr.Get(), 0, csrc.Get(), 0, 2048);
            transition_on(s.list.Get(), big_nsr.Get(),
                          D3D12_RESOURCE_STATE_COPY_DEST,
                          D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        });
        const double v5 = rtt([&] { // copy into a big COMMON buffer, no transitions
            s.list->CopyBufferRegion(big_cmn.Get(), 0, csrc.Get(), 0, 2048);
        });
        const double v6 = rtt([&] { // barrier-only on the big NSR buffer
            transition_on(s.list.Get(), big_nsr.Get(),
                          D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                          D3D12_RESOURCE_STATE_COPY_DEST);
            transition_on(s.list.Get(), big_nsr.Get(),
                          D3D12_RESOURCE_STATE_COPY_DEST,
                          D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        });
        const double v9 = rtt([&] { // 16 copies: is the cost per copy or per submit?
            for (UINT d = 0; d < 16; ++d) {
                s.list->CopyBufferRegion(cdst.Get(), (d % 8) * 512, csrc.Get(), 0, 512);
            }
        });
        std::cerr << "[d3d12] barrier bench: empty=" << empty_us
                  << " v0_copy=" << v0
                  << " v1_copy+16disp=" << v1
                  << " v2_copy+16disp+UA_flush=" << v2
                  << " v3_nsr_cycle=" << v3
                  << " v4_big_cycle+copy=" << v4
                  << " v5_big_copy=" << v5
                  << " v6_big_cycle=" << v6
                  << " v9_16copies=" << v9 << std::endl;

        // Dependent-kernel-chain test: the pipeline stamps show a ~150us
        // per-layer stall that floats between adjacent op buckets, which is the
        // signature of a fixed cost per dependent write->read kernel boundary.
        // This variant chains 16 dependent dispatches with the exact production
        // pattern (dispatch writes ybuf, UA->NPSR->UA barriers between them).
        // Per-boundary cost = (result - empty - work) / boundaries.
        const double dep = rtt([&] {
            s.list->SetPipelineState(s.pso_gemv.Get());
            s.list->SetComputeRootConstantBufferView(0, s.params->GetGPUVirtualAddress());
            s.list->SetComputeRootShaderResourceView(1, s.xbuf->GetGPUVirtualAddress());
            s.list->SetComputeRootShaderResourceView(2, s.xbuf->GetGPUVirtualAddress());
            s.list->SetComputeRootShaderResourceView(3, s.xbuf->GetGPUVirtualAddress());
            s.list->SetComputeRootUnorderedAccessView(5, s.ybuf->GetGPUVirtualAddress());
            for (UINT d = 0; d < 16; ++d) {
                s.list->Dispatch(4, 1, 1);
                transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            }
        });
        const double dep8 = rtt([&] {
            s.list->SetPipelineState(s.pso_gemv.Get());
            s.list->SetComputeRootConstantBufferView(0, s.params->GetGPUVirtualAddress());
            s.list->SetComputeRootShaderResourceView(1, s.xbuf->GetGPUVirtualAddress());
            s.list->SetComputeRootShaderResourceView(2, s.xbuf->GetGPUVirtualAddress());
            s.list->SetComputeRootShaderResourceView(3, s.xbuf->GetGPUVirtualAddress());
            s.list->SetComputeRootUnorderedAccessView(5, s.ybuf->GetGPUVirtualAddress());
            for (UINT d = 0; d < 8; ++d) {
                s.list->Dispatch(4, 1, 1);
                transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            }
        });
        std::cerr << "[d3d12] dependent chain: 16-link=" << dep
                  << " 8-link=" << dep8
                  << " (~" << (dep - dep8) << " us per extra 8 links)" << std::endl;
    }

    // ZEN_D3D12_BENCH_GPUTS: are the GPU timestamps trustworthy? Production brackets
    // only the Dispatch, and on prefill it reports ~5us per call while the CPU blocks
    // 11.35ms on the same fence - four orders of magnitude apart (F-19). Dispatch N
    // times inside one list, timestamp them exactly as production does, and compare the
    // reported GPU time against the fence round trip. If the stamp does not scale with N
    // it is unusable, and so is every conclusion drawn from gpu_us.
    if (const char* gputs = std::getenv("ZEN_D3D12_BENCH_GPUTS"); gputs && gputs[0] == '1') {
        if (s.ts_heap && s.ts_read && reserve_scratch(s, 64 * 1024, 8 * 1024)) {
            auto* probe_params = static_cast<GemvParams*>(s.params_cpu);
            probe_params->rows = 64;
            probe_params->blocks_per_row = 1;
            probe_params->rows_per_group = kRowsPerGroup;
            probe_params->reserved = 0;
            for (UINT count : {1u, 32u, 256u, 1024u}) {
                double fence_total = 0.0;
                double stamp_total = 0.0;
                int stamp_samples = 0;
                constexpr int kRounds = 16;
                for (int r = 0; r < kRounds; ++r) {
                    s.allocator->Reset();
                    s.list->Reset(s.allocator.Get(), s.pso_gemv.Get());
                    s.list->SetComputeRootSignature(s.root.Get());
                    s.list->SetComputeRootConstantBufferView(0, s.params->GetGPUVirtualAddress());
                    s.list->SetComputeRootShaderResourceView(1, s.xbuf->GetGPUVirtualAddress());
                    s.list->SetComputeRootShaderResourceView(2, s.xbuf->GetGPUVirtualAddress());
                    s.list->SetComputeRootShaderResourceView(3, s.xbuf->GetGPUVirtualAddress());
                    s.list->SetComputeRootUnorderedAccessView(5, s.ybuf->GetGPUVirtualAddress());
                    s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
                    for (UINT d = 0; d < count; ++d) {
                        s.list->Dispatch(16, 1, 1);
                    }
                    s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
                    s.list->ResolveQueryData(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2,
                                             s.ts_read.Get(), 0);
                    s.list->Close();

                    const auto start = std::chrono::steady_clock::now();
                    ID3D12CommandList* lists[] = {s.list.Get()};
                    s.queue->ExecuteCommandLists(1, lists);
                    wait_gpu(s);
                    fence_total += std::chrono::duration<double, std::micro>(
                                       std::chrono::steady_clock::now() - start)
                                       .count();

                    void* mapped = nullptr;
                    const D3D12_RANGE range{0, 2 * sizeof(std::uint64_t)};
                    if (SUCCEEDED(s.ts_read->Map(0, &range, &mapped))) {
                        const auto* stamps = static_cast<const std::uint64_t*>(mapped);
                        if (stamps[1] > stamps[0]) {
                            stamp_total += static_cast<double>(stamps[1] - stamps[0]) *
                                           1000000.0 / static_cast<double>(s.ts_freq);
                            ++stamp_samples;
                        }
                        const D3D12_RANGE none{0, 0};
                        s.ts_read->Unmap(0, &none);
                    }
                }
                std::cerr << "[d3d12] " << count << " dispatches: gpu_stamp="
                          << (stamp_samples ? stamp_total / stamp_samples : -1.0)
                          << " us, fence=" << (fence_total / kRounds) << " us, samples="
                          << stamp_samples << "/" << kRounds << std::endl;
            }
        }
    }

    // Does an idle gap between submissions make the next one expensive? The real path
    // pays 265us of fence wait per list while an isolated one costs 82us; the CPU does
    // attention between submissions, so the GPU gets to fall asleep in between. If
    // retire latency tracks the gap, this is clock/power wake latency, and no amount
    // of command-list batching removes it - only keeping the GPU busy does.
    if (const char* idle = std::getenv("ZEN_D3D12_BENCH_IDLE"); idle && idle[0] == '1') {
        if (reserve_scratch(s, 64 * 1024, 8 * 1024)) {
            auto* probe_params = static_cast<GemvParams*>(s.params_cpu);
            probe_params->rows = 64;
            probe_params->blocks_per_row = 1;
            probe_params->rows_per_group = kRowsPerGroup;
            probe_params->reserved = 0;
            LARGE_INTEGER perf_freq{};
            QueryPerformanceFrequency(&perf_freq);
            auto spin_us = [&](double us) {
                LARGE_INTEGER t0{}, t1{};
                QueryPerformanceCounter(&t0);
                for (;;) {
                    QueryPerformanceCounter(&t1);
                    const double elapsed = static_cast<double>(t1.QuadPart - t0.QuadPart) *
                                           1000000.0 / static_cast<double>(perf_freq.QuadPart);
                    if (elapsed >= us) {
                        break;
                    }
                    _mm_pause();
                }
            };
            for (double gap : {0.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0}) {
                double total = 0.0;
                constexpr int kRounds = 60;
                for (int i = 0; i < kRounds; ++i) {
                    s.allocator->Reset();
                    s.list->Reset(s.allocator.Get(), s.pso_gemv.Get());
                    s.list->SetComputeRootSignature(s.root.Get());
                    s.list->SetComputeRootConstantBufferView(0, s.params->GetGPUVirtualAddress());
                    s.list->SetComputeRootShaderResourceView(1, s.xbuf->GetGPUVirtualAddress());
                    s.list->SetComputeRootShaderResourceView(2, s.xbuf->GetGPUVirtualAddress());
                    s.list->SetComputeRootShaderResourceView(3, s.xbuf->GetGPUVirtualAddress());
                    s.list->SetComputeRootUnorderedAccessView(5, s.ybuf->GetGPUVirtualAddress());
                    for (UINT d = 0; d < 4; ++d) {
                        s.list->Dispatch(16, 1, 1);
                    }
                    s.list->Close();
                    if (gap > 0.0) {
                        spin_us(gap);  // let the GPU fall idle between submissions
                    }
                    const auto start = std::chrono::steady_clock::now();
                    ID3D12CommandList* lists[] = {s.list.Get()};
                    s.queue->ExecuteCommandLists(1, lists);
                    wait_gpu(s);
                    total += std::chrono::duration<double, std::micro>(
                                 std::chrono::steady_clock::now() - start)
                                 .count();
                }
                std::cerr << "[d3d12] retire latency after " << gap << " us idle gap: "
                          << (total / kRounds) << " us" << std::endl;
            }
        }
    }

    // Is the per-submission cost a drain/empty-queue effect? Submit many lists
    // back-to-back without waiting, then wait once at the end.
    if (const char* pipe = std::getenv("ZEN_D3D12_BENCH_PIPE"); pipe && pipe[0] == '1') {
        if (reserve_scratch(s, 64 * 1024, 8 * 1024)) {
            auto* probe_params = static_cast<GemvParams*>(s.params_cpu);
            probe_params->rows = 64;
            probe_params->blocks_per_row = 1;
            probe_params->rows_per_group = kRowsPerGroup;
            probe_params->reserved = 0;
            constexpr int kLists = 64;
            s.allocator->Reset();
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < kLists; ++i) {
                s.list->Reset(s.allocator.Get(), s.pso_gemv.Get());
                s.list->SetComputeRootSignature(s.root.Get());
                s.list->SetComputeRootConstantBufferView(0, s.params->GetGPUVirtualAddress());
                s.list->SetComputeRootShaderResourceView(1, s.xbuf->GetGPUVirtualAddress());
                s.list->SetComputeRootShaderResourceView(2, s.xbuf->GetGPUVirtualAddress());
                s.list->SetComputeRootShaderResourceView(3, s.xbuf->GetGPUVirtualAddress());
                s.list->SetComputeRootUnorderedAccessView(5, s.ybuf->GetGPUVirtualAddress());
                s.list->Dispatch(16, 1, 1);
                s.list->Close();
                ID3D12CommandList* lists[] = {s.list.Get()};
                s.queue->ExecuteCommandLists(1, lists);
            }
            wait_gpu(s);
            const double per_list = std::chrono::duration<double, std::micro>(
                                        std::chrono::steady_clock::now() - start)
                                        .count() / kLists;
            std::cerr << "[d3d12] pipelined submit, no wait between: " << per_list << " us/list"
                      << std::endl;
        }
    }

    s.ready = true;
    return true;
}

bool reserve_scratch(State& s, std::size_t x_bytes, std::size_t y_bytes) {
    if (x_bytes > s.x_cap) {
        s.xbuf.Reset();
        s.xbuf = create_buffer(s.device.Get(), x_bytes, upload_heap_type(),
                               upload_heap_state());
        if (!s.xbuf || FAILED(s.xbuf->Map(0, nullptr, &s.x_cpu))) {
            s.x_cap = 0;
            return false;
        }
        s.x_cap = x_bytes;
    }
    if (y_bytes > s.y_cap) {
        s.ybuf.Reset();
        s.yread.Reset();
        s.ybuf = create_buffer(s.device.Get(), y_bytes, D3D12_HEAP_TYPE_DEFAULT,
                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                               D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        s.yread = create_buffer(s.device.Get(), y_bytes, D3D12_HEAP_TYPE_READBACK,
                                D3D12_RESOURCE_STATE_COPY_DEST);
        if (!s.ybuf || !s.yread) {
            s.y_cap = 0;
            return false;
        }
        s.y_cap = y_bytes;
    }
    return true;
}

// Upload a Q8_0 matrix, repacking every 34-byte block into an aligned quant stream
// plus a separate fp32 scale stream.
bool upload_weights(State& s, const std::uint8_t* weights, std::size_t rows,
                    std::size_t columns, WeightEntry& entry) {
    const auto upload_start = std::chrono::steady_clock::now();
    const std::size_t blocks_per_row = (columns + kElemsPerBlock - 1) / kElemsPerBlock;
    const std::size_t row_bytes = blocks_per_row * kRawBlockBytes;
    const std::uint64_t quant_bytes =
        static_cast<std::uint64_t>(rows) * blocks_per_row * kElemsPerBlock;
    const std::uint64_t scale_bytes = static_cast<std::uint64_t>(rows) * blocks_per_row * sizeof(float);

    if (s.adapter.dedicated_vram) {
        const std::uint64_t budget = s.adapter.dedicated_vram / 100 * kVRAMBudgetPercent;
        if (s.weight_bytes + quant_bytes + scale_bytes > budget) {
            s.weights.clear();
            s.weight_bytes = 0;
        }
        if (quant_bytes + scale_bytes > budget) {
            return false;
        }
    }

    // Layout of the single buffer this tensor needs: [quants | 256B pad | scales].
    const std::uint64_t scale_offset = (quant_bytes + 255u) & ~static_cast<std::uint64_t>(255u);
    const std::uint64_t needed = scale_offset + scale_bytes;

    entry.quants = create_buffer(s.device.Get(), needed, D3D12_HEAP_TYPE_DEFAULT,
                                 D3D12_RESOURCE_STATE_COPY_DEST);
    if (!entry.quants) {
        return false;
    }
    entry.scales_va = entry.quants->GetGPUVirtualAddress() + scale_offset;

    // One staging buffer, reused for every tensor and kept for the life of the
    // backend. Allocating fresh system pages per tensor cost more than the
    // transfers themselves: ~1.3 GB of first-touch faults across 112 uploads.
    if (needed > s.stage_cap) {
        const auto stage_start = std::chrono::steady_clock::now();
        s.stage.Reset();
        s.stage_cpu = nullptr;
        s.stage = create_buffer(s.device.Get(), needed, upload_heap_type(),
                                upload_heap_state());
        if (!s.stage || FAILED(s.stage->Map(0, nullptr, &s.stage_cpu))) {
            s.stage_cap = 0;
            return false;
        }
        s.stage_cap = needed;
        // Only the growth events count here: the first one includes the initial map.
        s.stats.stage_us += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - stage_start)
                .count());
    }

    auto* q_dst = static_cast<std::uint8_t*>(s.stage_cpu);
    auto* s_dst =
        reinterpret_cast<float*>(static_cast<std::uint8_t*>(s.stage_cpu) + scale_offset);

    // Everything from entry to here is committed-resource creation and staging-buffer
    // growth. upload_ms spans all of it, so without this split that cost hides behind
    // the repack and transfer timers and looks like slow PCIe.
    s.stats.alloc_us += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - upload_start)
            .count());

    const auto repack_start = std::chrono::steady_clock::now();
    detail::repack_q8_0(weights, rows, blocks_per_row, q_dst, s_dst);
    s.stats.repack_us += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() -
                                                              repack_start)
            .count());
    const auto transfer_start = std::chrono::steady_clock::now();
    (void)row_bytes;

    s.allocator->Reset();
    s.list->Reset(s.allocator.Get(), nullptr);

    // One copy: the staging buffer already carries [quants | pad | scales], so both
    // regions move in a single transfer. The resource is still in COPY_DEST here and
    // only afterwards may it take the shader-readable state the root SRVs require.
    s.list->CopyBufferRegion(entry.quants.Get(), 0, s.stage.Get(), 0, needed);

    D3D12_RESOURCE_BARRIER back{};
    back.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    back.Transition.pResource = entry.quants.Get();
    back.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    back.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    back.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    s.list->ResourceBarrier(1, &back);

    s.list->Close();
    ID3D12CommandList* lists[] = {s.list.Get()};
    ++s.stats.submits;
    s.queue->ExecuteCommandLists(1, lists);
    if (!wait_gpu(s)) {
        return false;
    }

    s.stats.transfer_us += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() -
                                                              transfer_start)
            .count());
    s.weight_bytes += needed;
    s.stats.upload_bytes += needed;
    s.stats.upload_us += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() -
                                                              upload_start)
            .count());
    s.stats.weight_bytes = s.weight_bytes;
    entry.rows = rows;
    entry.columns = columns;
    return true;
}

WeightEntry* lookup_weights(State& s, const std::uint8_t* weights, std::size_t rows,
                            std::size_t columns) {
    auto it = s.weights.find(weights);
    if (it != s.weights.end() && it->second.rows == rows && it->second.columns == columns) {
        return &it->second;
    }
    // A different shape at the same address means the loader reused the storage;
    // drop the stale residency before uploading the new layout.
    if (it != s.weights.end()) {
        // One buffer now holds both regions, so the removal must subtract its full
        // width - exactly what upload_weights adds below - or the residency accounting
        // drifts by the 256B pad on every stale-shape replacement.
        s.weight_bytes -= it->second.quants->GetDesc().Width;
        s.weights.erase(it);
    }
    WeightEntry entry;
    if (!upload_weights(s, weights, rows, columns, entry)) {
        return nullptr;
    }
    auto& slot = s.weights[weights];
    slot = std::move(entry);
    return &slot;
}

FusedMat* ensure_fused_mat(
    State& s, const std::uint8_t* key, std::size_t columns,
    const std::vector<std::pair<const std::uint8_t*, std::size_t>>& parts) {
    static std::map<const std::uint8_t*, FusedMat> cache;
    auto cached = cache.find(key);
    if (cached != cache.end()) return &cached->second;

    if (columns % kElemsPerBlock) return nullptr;
    const std::size_t blocks_per_row = columns / kElemsPerBlock;
    std::uint64_t qbytes = 0;
    std::size_t rows = 0;
    for (const auto& [w, r] : parts) {
        qbytes += static_cast<std::uint64_t>(r) * blocks_per_row * 32;
        rows += r;
        for (const auto& [w2, r2] : parts) {
            if (w2 == w && r2 != r) return nullptr; // address reused with new shape
        }
    }
    const std::uint64_t scale_bytes =
        static_cast<std::uint64_t>(rows) * blocks_per_row * sizeof(float);
    const std::uint64_t scale_offset = (qbytes + 255u) & ~static_cast<std::uint64_t>(255u);
    const std::uint64_t total = scale_offset + scale_bytes;

    FusedMat mk;
    mk.rows = rows;
    mk.cols = columns;
    mk.quants = create_buffer(s.device.Get(), total, D3D12_HEAP_TYPE_DEFAULT,
                              D3D12_RESOURCE_STATE_COPY_DEST);
    if (!mk.quants) return nullptr;
    mk.scales_va = mk.quants->GetGPUVirtualAddress() + scale_offset;

    // One build list for every part: source bytes sit in the registry buffers the
    // per-op path also reads, so each takes a COPY_SOURCE cycle, one quant copy
    // and one scale copy into its concatenated offset. No CPU repack, no second
    // PCIe pass, and the per-op path never observes the state changes.
    if (FAILED(s.allocator->Reset())) return nullptr;
    if (FAILED(s.list->Reset(s.allocator.Get(), nullptr))) return nullptr;
    std::uint64_t row_off = 0;
    for (const auto& [w, r] : parts) {
        WeightEntry* e = lookup_weights(s, w, r, columns);
        if (!e || !e->quants) return nullptr;
        const std::uint64_t part_q =
            static_cast<std::uint64_t>(r) * blocks_per_row * 32;
        const std::uint64_t part_s =
            static_cast<std::uint64_t>(r) * blocks_per_row * sizeof(float);
        transition_on(s.list.Get(), e->quants.Get(),
                      D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                      D3D12_RESOURCE_STATE_COPY_SOURCE);
        s.list->CopyBufferRegion(mk.quants.Get(),
                                 row_off * blocks_per_row * 32,
                                 e->quants.Get(), 0, part_q);
        s.list->CopyBufferRegion(mk.quants.Get(),
                                 scale_offset + row_off * blocks_per_row * sizeof(float),
                                 e->quants.Get(),
                                 e->scales_va - e->quants->GetGPUVirtualAddress(),
                                 part_s);
        transition_on(s.list.Get(), e->quants.Get(),
                      D3D12_RESOURCE_STATE_COPY_SOURCE,
                      D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        row_off += r;
    }
    transition_on(s.list.Get(), mk.quants.Get(),
                  D3D12_RESOURCE_STATE_COPY_DEST,
                  D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (FAILED(s.list->Close())) return nullptr;
    ID3D12CommandList* build_lists[] = {s.list.Get()};
    s.stats.submits++;
    s.queue->ExecuteCommandLists(1, build_lists);
    if (!wait_gpu(s)) return nullptr;

    s.weight_bytes += total;
    s.stats.weight_bytes = s.weight_bytes;
    auto& slot = cache[key];
    slot = std::move(mk);
    return &slot;
}

bool gemv_batch_locked(State& s, const float* x, const GemvTask* tasks, std::size_t count) {
    s.current_comp = Comp::Projections;
    if (count == 0 || count > kMaxBatch) {
        return false;
    }
    const std::size_t cols = tasks[0].columns;
    if (cols % kElemsPerBlock != 0) {
        return false;
    }
    std::size_t total_rows = 0;
    for (std::size_t i = 0; i < count; ++i) {
        if (tasks[i].columns != cols) {
            return false;
        }
        total_rows += tasks[i].rows;
    }

    // Refcounted copies, not WeightEntry*: a later lookup_weights() inside this same
    // batch can hit the VRAM budget and clear the residency cache, which would free an
    // entry we have not bound yet (F-09).
    ComPtr<ID3D12Resource> quants[kMaxBatch];
    std::uint64_t scales[kMaxBatch];
    for (std::size_t i = 0; i < count; ++i) {
        WeightEntry* entry = lookup_weights(s, tasks[i].weights, tasks[i].rows, tasks[i].columns);
        if (!entry) {
            return false;
        }
        quants[i] = entry->quants;
        scales[i] = entry->scales_va;
    }

    // Outputs are packed back-to-back in the shared UAV buffer at 256-byte offsets,
    // which is what a root UAV descriptor requires.
    std::uint64_t offsets[kMaxBatch];
    std::uint64_t span = 0;
    for (std::size_t i = 0; i < count; ++i) {
        offsets[i] = span;
        span = (span + tasks[i].rows * sizeof(float) + 255u) & ~static_cast<std::uint64_t>(255u);
    }

    if (!reserve_scratch(s, cols * sizeof(float), span)) {
        return false;
    }
    std::memcpy(s.x_cpu, x, cols * sizeof(float));

    auto* params_base = static_cast<std::uint8_t*>(s.params_cpu);
    for (std::size_t i = 0; i < count; ++i) {
        auto* params = reinterpret_cast<GemvParams*>(params_base + 512 + 256 * i);
        params->rows = static_cast<std::uint32_t>(tasks[i].rows);
        params->blocks_per_row = static_cast<std::uint32_t>(cols / kElemsPerBlock);
        params->rows_per_group = kRowsPerGroup;
        params->reserved = 0;
    }

    s.allocator->Reset();
    s.list->Reset(s.allocator.Get(), s.pso_gemv.Get());
    s.list->SetComputeRootSignature(s.root.Get());
    if (s.trace) {
        s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
    }
    const bool use_rt_b = gemm_rt_enabled() && s.pso_gemm_rt != nullptr;
    for (std::size_t i = 0; i < count; ++i) {
        s.list->SetComputeRootConstantBufferView(
            0, s.params->GetGPUVirtualAddress() + 512 + 256 * i);
        s.list->SetComputeRootShaderResourceView(1, quants[i]->GetGPUVirtualAddress());
        s.list->SetComputeRootShaderResourceView(2, scales[i]);
        s.list->SetComputeRootShaderResourceView(3, s.xbuf->GetGPUVirtualAddress());
        s.list->SetComputeRootUnorderedAccessView(5, s.ybuf->GetGPUVirtualAddress() + offsets[i]);
        s.list->Dispatch(static_cast<UINT>((tasks[i].rows + kRowsPerGroup - 1) / kRowsPerGroup), 1,
                         1);
    }
    if (s.trace) {
        s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
    }

    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = s.ybuf.Get();
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    s.list->ResourceBarrier(1, &barrier);
    s.list->CopyBufferRegion(s.yread.Get(), 0, s.ybuf.Get(), 0, span);
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    s.list->ResourceBarrier(1, &barrier);
    if (s.trace) {
        s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 2);
        s.list->ResolveQueryData(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 3,
                                 s.ts_read.Get(), 0);
    }
    s.list->Close();

    const std::uint64_t qkv_mark = s.stats.wait_us;
    if (!execute_and_wait(s)) {
        return false;
    }
    s.stats.qkv_wait_us += s.stats.wait_us - qkv_mark;
    accumulate_gpu_us(s, s.stats.qkv_gpu_us, 0, 1);
    accumulate_gpu_us(s, s.stats.qkv_copy_us, 1, 2);
    {
        std::uint64_t dev = 0;
        accumulate_gpu_us(s, dev, 0, 1);
        note_dev(s, Comp::Projections, dev);
    }

    void* mapped = nullptr;
    const D3D12_RANGE read_range{0, span};
    if (FAILED(s.yread->Map(0, &read_range, &mapped))) {
        return false;
    }
    const auto* base = static_cast<const std::uint8_t*>(mapped);
    for (std::size_t i = 0; i < count; ++i) {
        std::memcpy(tasks[i].y, base + offsets[i], tasks[i].rows * sizeof(float));
    }
    const D3D12_RANGE write_range{0, 0};
    s.yread->Unmap(0, &write_range);
    s.stats.dispatches += count;
    ++s.stats.batches;
    return true;
}

bool gemv_locked(State& s, const std::uint8_t* weights, const float* x, float* y,
                 std::size_t rows, std::size_t columns) {
    using clock = std::chrono::steady_clock;
    auto us_since = [](clock::time_point start) {
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - start).count());
    };

    if (columns % kElemsPerBlock != 0) {
        return false;
    }
    WeightEntry* entry = lookup_weights(s, weights, rows, columns);
    if (!entry) {
        return false;
    }

    auto mark = clock::now();
    const std::size_t blocks_per_row = columns / kElemsPerBlock;
    if (!reserve_scratch(s, columns * sizeof(float), rows * sizeof(float))) {
        return false;
    }
    std::memcpy(s.x_cpu, x, columns * sizeof(float));

    auto* params = static_cast<GemvParams*>(s.params_cpu);
    params->rows = static_cast<std::uint32_t>(rows);
    params->blocks_per_row = static_cast<std::uint32_t>(blocks_per_row);
    params->rows_per_group = kRowsPerGroup;
    params->reserved = 0;
    s.stats.prep_us += us_since(mark);

    mark = clock::now();
    s.allocator->Reset();
    s.list->Reset(s.allocator.Get(), s.pso_gemv.Get());
    s.stats.reset_us += us_since(mark);

    mark = clock::now();
    s.list->SetComputeRootSignature(s.root.Get());
    s.list->SetComputeRootConstantBufferView(0, s.params->GetGPUVirtualAddress());
    s.list->SetComputeRootShaderResourceView(1, entry->quants->GetGPUVirtualAddress());
    s.list->SetComputeRootShaderResourceView(2, entry->scales_va);
    s.list->SetComputeRootShaderResourceView(3, s.xbuf->GetGPUVirtualAddress());
    s.list->SetComputeRootUnorderedAccessView(5, s.ybuf->GetGPUVirtualAddress());

    mark = clock::now();
    if (s.trace) {
        s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
    }
    s.list->Dispatch(static_cast<UINT>((rows + kRowsPerGroup - 1) / kRowsPerGroup), 1, 1);
    if (s.trace) {
        s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
    }

    if (!s.no_readback) {
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = s.ybuf.Get();
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        s.list->ResourceBarrier(1, &barrier);
        s.list->CopyBufferRegion(s.yread.Get(), 0, s.ybuf.Get(), 0, rows * sizeof(float));
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        s.list->ResourceBarrier(1, &barrier);
    }
    if (s.trace) {
        s.list->ResolveQueryData(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2,
                                 s.ts_read.Get(), 0);
    }
    s.list->Close();
    s.stats.record_us += us_since(mark);

    mark = clock::now();
    ID3D12CommandList* lists[] = {s.list.Get()};
    s.queue->ExecuteCommandLists(1, lists);
    if (!wait_gpu(s)) {
        return false;
    }
    s.stats.wait_us += us_since(mark);

    mark = clock::now();
    if (!s.no_readback) {
        void* mapped = nullptr;
        const D3D12_RANGE read_range{0, rows * sizeof(float)};
        if (FAILED(s.yread->Map(0, &read_range, &mapped))) {
            return false;
        }
        std::memcpy(y, mapped, rows * sizeof(float));
        const D3D12_RANGE write_range{0, 0};
        s.yread->Unmap(0, &write_range);
    }

    if (s.trace) {
        std::uint64_t* stamps = nullptr;
        const D3D12_RANGE stamp_range{0, 2 * sizeof(std::uint64_t)};
        if (SUCCEEDED(s.ts_read->Map(0, &stamp_range, reinterpret_cast<void**>(&stamps)))) {
            if (stamps[1] > stamps[0]) {
                s.stats.gpu_us += static_cast<std::uint64_t>(
                    static_cast<double>(stamps[1] - stamps[0]) * 1000000.0 /
                    static_cast<double>(s.ts_freq));
            }
            const D3D12_RANGE none{0, 0};
            s.ts_read->Unmap(0, &none);
        }
    }
    s.stats.readback_us += us_since(mark);
    ++s.stats.dispatches;
    return true;
}

bool gemm_locked(State& s, const std::uint8_t* weights, const float* input, float* output,
                 std::size_t tokens, std::size_t rows, std::size_t columns) {
    if (columns % kElemsPerBlock != 0) {
        return false;
    }
    WeightEntry* entry = lookup_weights(s, weights, rows, columns);
    if (!entry) {
        return false;
    }

    const std::size_t blocks_per_row = columns / kElemsPerBlock;
    // ZEN_D3D12_DP4A=1 selects the int8-activation kernel: roughly 4x fewer inner
    // loop instructions, but it changes the arithmetic, so the fp32 path stays the
    // default and the parity tests cover it at full precision.
    static const bool use_dp4a = [] {
        const char* e = std::getenv("ZEN_D3D12_DP4A");
        return e && e[0] == '1';
    }();
    const bool dp4a = use_dp4a && s.pso_gemm_dp4a != nullptr;
    // GEMM v3 (verify tile, opt-in, bit-identical): 1024-thread groups, 8
    // tokens per tile. It takes precedence over v2 when both gates are on
    // (verify T wants the 8-tile); DP4A still outranks both.
    const bool use_v3 = !dp4a && gemm_v3_enabled() && s.pso_gemm_v3 != nullptr;
    const bool use_v2 = !dp4a && !use_v3 && gemm_v2_enabled() && s.pso_gemm_v2 != nullptr;
    const bool use_fused = !dp4a && !use_v3 && !use_v2 && gemm_fused_enabled() && s.pso_gemm_fused != nullptr;
    const bool use_sh = !dp4a && !use_v3 && !use_v2 && !use_fused && gemm_sh_enabled() && s.pso_gemm_sh != nullptr;
    const std::uint32_t tile_v2 = use_v3 ? kTokensPerTileV3
                                    : (use_v2 ? kTokensPerTileV2 : kTokensPerTile);
    const std::uint32_t rpg_v2 = use_v3 ? kGemmRowsPerGroupV3
                                   : (use_v2 ? kGemmRowsPerGroupV2 : kGemmRowsPerGroup);

    const std::size_t xscale_bytes = tokens * sizeof(float);
    const std::size_t x_bytes = dp4a ? tokens * columns
        : (use_fused ? tokens * columns * sizeof(float)
        : (((tokens + tile_v2 - 1) / tile_v2 * tile_v2) * columns * sizeof(float)));
    const std::size_t y_bytes = tokens * rows * sizeof(float);
    if (!reserve_scratch(s, x_bytes, y_bytes + (dp4a ? xscale_bytes : 0))) {
        return false;
    }
    if (dp4a) {
        // One scale per token: max|x|/127, finer than the per-32-element scale the
        // weights already carry. The scale table is kept in its own vector: the
        // staging region handed back by reserve_scratch is exactly x_bytes, so
        // writing past it (as an earlier version did) overflows the heap.
        auto* qdst = static_cast<std::int8_t*>(s.x_cpu);
        std::vector<float> scales(tokens);
        for (std::size_t t = 0; t < tokens; ++t) {
            const float* src = input + t * columns;
            float amax = 0.0f;
            for (std::size_t c = 0; c < columns; ++c) {
                const float a = src[c] < 0.0f ? -src[c] : src[c];
                amax = a > amax ? a : amax;
            }
            const float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
            scales[t] = (amax > 0.0f) ? (amax / 127.0f) : 0.0f;
            std::int8_t* dst = qdst + t * columns;
            for (std::size_t c = 0; c < columns; ++c) {
                float v = src[c] * inv;
                v = v > 127.0f ? 127.0f : (v < -127.0f ? -127.0f : v);
                dst[c] = static_cast<std::int8_t>(std::lround(v));
            }
        }
        if (!s.gemm_xscale || xscale_bytes > s.gemm_xscale_cap) {
            s.gemm_xscale.Reset();
            s.gemm_xscale = create_buffer(s.device.Get(), xscale_bytes, D3D12_HEAP_TYPE_UPLOAD,
                                          D3D12_RESOURCE_STATE_GENERIC_READ);
            if (!s.gemm_xscale) {
                s.gemm_xscale_cap = 0;
                return false;
            }
            if (FAILED(s.gemm_xscale->Map(0, nullptr, &s.gemm_xscale_cpu))) {
                s.gemm_xscale_cap = 0;
                return false;
            }
            s.gemm_xscale_cap = xscale_bytes;
        }
        std::memcpy(s.gemm_xscale_cpu, scales.data(), xscale_bytes);
    } else {
        const std::size_t padded_tokens = x_bytes / (columns * sizeof(float));
        std::memcpy(s.x_cpu, input, tokens * columns * sizeof(float));
        if (padded_tokens > tokens) {
            std::memset(static_cast<std::uint8_t*>(s.x_cpu) + tokens * columns * sizeof(float), 0,
                        (padded_tokens - tokens) * columns * sizeof(float));
        }
    }

    auto* params = reinterpret_cast<GemmParams*>(static_cast<std::uint8_t*>(s.params_cpu) + 256);
    params->rows = static_cast<std::uint32_t>(rows);
    params->blocks_per_row = static_cast<std::uint32_t>(blocks_per_row);
    params->token_count = static_cast<std::uint32_t>(tokens);
    params->tokens_per_tile = tile_v2;

    s.allocator->Reset();
    s.list->Reset(s.allocator.Get(),
                  dp4a ? s.pso_gemm_dp4a.Get()
                        : (use_fused ? s.pso_gemm_fused.Get()
                        : (use_v3 ? s.pso_gemm_v3.Get()
                        : (use_v2 ? s.pso_gemm_v2.Get()
                        : (use_sh ? s.pso_gemm_sh.Get() : s.pso_gemm.Get())))));
    s.list->SetComputeRootSignature(s.root.Get());
    s.list->SetComputeRootConstantBufferView(0, s.params->GetGPUVirtualAddress() + 256);
    s.list->SetComputeRootShaderResourceView(1, entry->quants->GetGPUVirtualAddress());
    s.list->SetComputeRootShaderResourceView(2, entry->scales_va);
    s.list->SetComputeRootShaderResourceView(3, s.xbuf->GetGPUVirtualAddress());
    s.list->SetComputeRootShaderResourceView(
        4, dp4a ? s.gemm_xscale->GetGPUVirtualAddress() : s.xbuf->GetGPUVirtualAddress());
    s.list->SetComputeRootUnorderedAccessView(5, s.ybuf->GetGPUVirtualAddress());

    if (s.trace) {
        s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
    }
    // The DP4A kernel uses the same 32-row tiling as the fp32 path; only the inner
    // loop differs. Keeping them equal avoids a second dispatch geometry.
    // (v2 uses its own 8-row tiling; dp4a forces v1 tiling.)
    const std::uint32_t rows_per_group = dp4a ? kGemmRowsPerGroup : rpg_v2;
    if (use_fused) {
        s.list->Dispatch(static_cast<UINT>((rows + 3u) / 4u), static_cast<UINT>((tokens + 3u) / 4u), 1);
    } else {
        s.list->Dispatch(static_cast<UINT>((rows + rows_per_group - 1) / rows_per_group), 1, 1);
    }
    if (s.trace) {
        s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
    }

    if (!s.no_readback) {
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = s.ybuf.Get();
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        s.list->ResourceBarrier(1, &barrier);
        s.list->CopyBufferRegion(s.yread.Get(), 0, s.ybuf.Get(), 0, y_bytes);
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        s.list->ResourceBarrier(1, &barrier);
    }
    // Stamp 2 closes the copy region: 0..1 is the dispatch, 1..2 the readback copy.
    // Without it prefill had no GPU timing at all, and NO_READBACK was silently
    // ignored here, so an earlier "readback ruled out" test proved nothing.
    if (s.trace) {
        s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 2);
        s.list->ResolveQueryData(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 3,
                                 s.ts_read.Get(), 0);
    }
    s.list->Close();

    const auto start = std::chrono::steady_clock::now();
    ID3D12CommandList* lists[] = {s.list.Get()};
    ++s.stats.submits;
    s.queue->ExecuteCommandLists(1, lists);
    if (!wait_gpu(s)) {
        return false;
    }
    s.stats.wait_us += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start)
            .count());

    if (!s.no_readback) {
        void* mapped = nullptr;
        const D3D12_RANGE read_range{0, y_bytes};
        if (FAILED(s.yread->Map(0, &read_range, &mapped))) {
            return false;
        }
        std::memcpy(output, mapped, y_bytes);
        const D3D12_RANGE write_range{0, y_bytes};
        s.yread->Unmap(0, &write_range);
    }

    if (s.trace) {
        std::uint64_t* stamps = nullptr;
        const D3D12_RANGE stamp_range{0, 3 * sizeof(std::uint64_t)};
        if (SUCCEEDED(s.ts_read->Map(0, &stamp_range, reinterpret_cast<void**>(&stamps)))) {
            if (stamps[1] > stamps[0]) {
                s.stats.gpu_us += static_cast<std::uint64_t>(
                    static_cast<double>(stamps[1] - stamps[0]) * 1000000.0 /
                    static_cast<double>(s.ts_freq));
            }
            if (stamps[2] > stamps[1]) {
                s.stats.gpu_copy_us += static_cast<std::uint64_t>(
                    static_cast<double>(stamps[2] - stamps[1]) * 1000000.0 /
                    static_cast<double>(s.ts_freq));
            }
            const D3D12_RANGE none{0, 0};
            s.ts_read->Unmap(0, &none);
        }
    }
    ++s.stats.dispatches;
    return true;
}

// Batched prefill GEMM: N tasks share one input payload, one command list, one
// fence wait, one packed readback. Measured on RDNA4 (warm): a lone submit costs
// ~1.1 ms round trip after ~0.4 ms of actual kernel, so collapsing 3 Q|K|V or
// 2 gate|up submissions into one each cuts ~0.7-0.8 ms of queue latency per
// removed submit. Results are bit-identical to the unbatched gemm_locked path:
// same PSO, same parameter layout, only the execution wrapper differs.
bool gemm_batch_locked(State& s, const float* x, const GemmTask* tasks,
                       std::size_t count, std::size_t tokens,
                       std::size_t columns) {
    if (count == 0 || count > kMaxBatch) {
        return false;
    }
    if (columns % kElemsPerBlock != 0) {
        return false;
    }
    for (std::size_t i = 0; i < count; ++i) {
        if (!tasks[i].weights || !tasks[i].y || !tasks[i].rows) {
            return false;
        }
    }

    // Refcount the weight buffers before recording: a lookup_weights() call in
    // this loop may trip the VRAM budget and clear residency, which would free
    // an entry already captured but not yet bound (F-09, same as gemv_batch).
    ComPtr<ID3D12Resource> quants[kMaxBatch];
    std::uint64_t scales[kMaxBatch];
    for (std::size_t i = 0; i < count; ++i) {
        WeightEntry* entry = lookup_weights(s, tasks[i].weights, tasks[i].rows, columns);
        if (!entry) {
            return false;
        }
        quants[i] = entry->quants;
        scales[i] = entry->scales_va;
    }

    const std::size_t blocks_per_row = columns / kElemsPerBlock;
    const bool use_v3_b = gemm_v3_enabled() && s.pso_gemm_v3 != nullptr;
    const bool use_v2_b = !use_v3_b && gemm_v2_enabled() && s.pso_gemm_v2 != nullptr;
    const bool use_fused_b = !use_v3_b && !use_v2_b && gemm_fused_enabled() && s.pso_gemm_fused != nullptr;
    const bool use_sh_b = !use_v3_b && !use_v2_b && !use_fused_b && gemm_sh_enabled() && s.pso_gemm_sh != nullptr;
    const bool use_rt_b = !use_v3_b && !use_v2_b && !use_fused_b && !use_sh_b && gemm_rt_enabled() && s.pso_gemm_rt != nullptr;
    const std::uint32_t tile_b = use_v3_b ? kTokensPerTileV3
                              : (use_v2_b ? kTokensPerTileV2 : kTokensPerTile);
    const std::uint32_t rpg_b = use_v3_b ? kGemmRowsPerGroupV3
                            : (use_v2_b ? kGemmRowsPerGroupV2 : kGemmRowsPerGroup);
    const std::size_t padded_tokens =
        (tokens + tile_b - 1) / tile_b * tile_b;
    const std::size_t x_bytes = padded_tokens * columns * sizeof(float);
    // Outputs packed back-to-back in the shared UAV buffer at 256-byte offsets,
    // mirroring gemv_batch_locked - EXCEPT the region size: the gemm shader
    // writes y[t*rows + r] for every token, so each task's region is
    // tokens*rows floats (the gemv vector case would be rows floats only).
    // Getting this wrong made later dispatches overwrite earlier outputs.
    std::uint64_t offsets[kMaxBatch];
    std::uint64_t span = 0;
    for (std::size_t i = 0; i < count; ++i) {
        offsets[i] = span;
        const std::uint64_t y_bytes =
            static_cast<std::uint64_t>(tokens) * tasks[i].rows * sizeof(float);
        span = (span + y_bytes + 255u) & ~static_cast<std::uint64_t>(255u);
    }
    if (!reserve_scratch(s, x_bytes, span)) {
        return false;
    }
    std::memcpy(s.x_cpu, x, tokens * columns * sizeof(float));
    if (padded_tokens > tokens) {
        std::memset(static_cast<std::uint8_t*>(s.x_cpu) + tokens * columns * sizeof(float), 0,
                    (padded_tokens - tokens) * columns * sizeof(float));
    }
    for (std::size_t i = 0; i < count; ++i) {
        auto* params = reinterpret_cast<GemmParams*>(
            static_cast<std::uint8_t*>(s.params_cpu) + 1024 + 256 * i);
        params->rows = static_cast<std::uint32_t>(tasks[i].rows);
        params->blocks_per_row = static_cast<std::uint32_t>(blocks_per_row);
        params->token_count = static_cast<std::uint32_t>(tokens);
        params->tokens_per_tile = tile_b;
    }

    s.allocator->Reset();
    s.list->Reset(s.allocator.Get(),
                  use_fused_b ? s.pso_gemm_fused.Get()
                  : (use_v3_b ? s.pso_gemm_v3.Get()
                  : (use_v2_b ? s.pso_gemm_v2.Get() : s.pso_gemm.Get())));
    s.list->SetComputeRootSignature(s.root.Get());
    if (s.trace) {
        s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
    }
    // Each dispatch writes a disjoint y region, so no UAV barrier is needed
    // between them (same reasoning as gemv_batch_locked).
    for (std::size_t i = 0; i < count; ++i) {
        s.list->SetComputeRootConstantBufferView(
            0, s.params->GetGPUVirtualAddress() + 1024 + 256 * i);
        s.list->SetComputeRootShaderResourceView(1, quants[i]->GetGPUVirtualAddress());
        s.list->SetComputeRootShaderResourceView(2, scales[i]);
        s.list->SetComputeRootShaderResourceView(3, s.xbuf->GetGPUVirtualAddress());
        s.list->SetComputeRootUnorderedAccessView(5, s.ybuf->GetGPUVirtualAddress() + offsets[i]);
        if (use_fused_b) {
            s.list->Dispatch(
                static_cast<UINT>((tasks[i].rows + 3u) / 4u),
                static_cast<UINT>((tokens + 3u) / 4u), 1);
        } else {
            s.list->Dispatch(
                static_cast<UINT>((tasks[i].rows + rpg_b - 1) / rpg_b), 1, 1);
        }
    }
    if (s.trace) {
        s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
    }

    if (!s.no_readback) {
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = s.ybuf.Get();
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        s.list->ResourceBarrier(1, &barrier);
        s.list->CopyBufferRegion(s.yread.Get(), 0, s.ybuf.Get(), 0, span);
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        s.list->ResourceBarrier(1, &barrier);
    }
    if (s.trace) {
        s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 2);
        s.list->ResolveQueryData(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 3,
                                 s.ts_read.Get(), 0);
    }
    s.list->Close();

    if (!execute_and_wait(s)) {
        return false;
    }

    if (!s.no_readback) {
        void* mapped = nullptr;
        const D3D12_RANGE read_range{0, span};
        if (FAILED(s.yread->Map(0, &read_range, &mapped))) {
            return false;
        }
        const auto* base = static_cast<const std::uint8_t*>(mapped);
        for (std::size_t i = 0; i < count; ++i) {
            std::memcpy(tasks[i].y, base + offsets[i],
                        tokens * tasks[i].rows * sizeof(float));
        }
        const D3D12_RANGE write_range{0, 0};
        s.yread->Unmap(0, &write_range);
    }
    s.stats.dispatches += count;
    ++s.stats.batches;
    return true;
}

// Register-tiled RT variant (fp32 only). Mirrors gemm_locked but uses 64x64 tile.
bool gemm_rt_locked(State& s, const std::uint8_t* weights, const float* input, float* output,
                    std::size_t tokens, std::size_t rows, std::size_t columns) {
    if (columns % kElemsPerBlock != 0) {
        return false;
    }
    WeightEntry* entry = lookup_weights(s, weights, rows, columns);
    if (!entry) {
        return false;
    }
    const std::size_t blocks_per_row = columns / kElemsPerBlock;
    const std::size_t x_bytes = tokens * columns * sizeof(float);
    const std::size_t y_bytes = tokens * rows * sizeof(float);
    if (!reserve_scratch(s, x_bytes, y_bytes)) {
        return false;
    }
    std::memcpy(s.x_cpu, input, tokens * columns * sizeof(float));

    auto* params = reinterpret_cast<GemmParams*>(static_cast<std::uint8_t*>(s.params_cpu) + 256);
    params->rows = static_cast<std::uint32_t>(rows);
    params->blocks_per_row = static_cast<std::uint32_t>(blocks_per_row);
    params->token_count = static_cast<std::uint32_t>(tokens);
    params->tokens_per_tile = 64;

    s.allocator->Reset();
    s.list->Reset(s.allocator.Get(), s.pso_gemm_rt.Get());
    s.list->SetComputeRootSignature(s.root.Get());
    s.list->SetComputeRootConstantBufferView(0, s.params->GetGPUVirtualAddress() + 256);
    s.list->SetComputeRootShaderResourceView(1, entry->quants->GetGPUVirtualAddress());
    s.list->SetComputeRootShaderResourceView(2, entry->scales_va);
    s.list->SetComputeRootShaderResourceView(3, s.xbuf->GetGPUVirtualAddress());
    s.list->SetComputeRootShaderResourceView(4, s.xbuf->GetGPUVirtualAddress());
    s.list->SetComputeRootUnorderedAccessView(5, s.ybuf->GetGPUVirtualAddress());

    if (s.trace) {
        s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
    }
    const std::uint32_t gx = static_cast<std::uint32_t>((rows + 3u) / 4u);
    const std::uint32_t gy = static_cast<std::uint32_t>((tokens + 7u) / 8u);
    s.list->Dispatch(gx, gy, 1);
    if (s.trace) {
        s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
    }

    if (!s.no_readback) {
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = s.ybuf.Get();
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        s.list->ResourceBarrier(1, &barrier);
        s.list->CopyBufferRegion(s.yread.Get(), 0, s.ybuf.Get(), 0, y_bytes);
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        s.list->ResourceBarrier(1, &barrier);
    }
    if (s.trace) {
        s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 2);
        s.list->ResolveQueryData(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 3,
                                 s.ts_read.Get(), 0);
    }
    s.list->Close();

    const auto start = std::chrono::steady_clock::now();
    ID3D12CommandList* lists[] = {s.list.Get()};
    ++s.stats.submits;
    s.queue->ExecuteCommandLists(1, lists);
    if (!wait_gpu(s)) {
        return false;
    }
    s.stats.wait_us += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start)
            .count());

    if (!s.no_readback) {
        void* mapped = nullptr;
        const D3D12_RANGE read_range{0, y_bytes};
        if (FAILED(s.yread->Map(0, &read_range, &mapped))) {
            return false;
        }
        std::memcpy(output, mapped, y_bytes);
        const D3D12_RANGE write_range{0, 0};
        s.yread->Unmap(0, &write_range);
    }
    ++s.stats.dispatches;
    return true;
}

bool gemm_rt_batch(State& s, const float* x, const GemmTask* tasks,
                   std::size_t count, std::size_t tokens, std::size_t columns) {
    if (count == 0 || count > kMaxBatch) {
        return false;
    }
    if (columns % kElemsPerBlock != 0) {
        return false;
    }
    for (std::size_t i = 0; i < count; ++i) {
        if (!tasks[i].weights || !tasks[i].y || !tasks[i].rows) {
            return false;
        }
    }
    ComPtr<ID3D12Resource> quants[kMaxBatch];
    std::uint64_t scales[kMaxBatch];
    for (std::size_t i = 0; i < count; ++i) {
        WeightEntry* entry = lookup_weights(s, tasks[i].weights, tasks[i].rows, columns);
        if (!entry) {
            return false;
        }
        quants[i] = entry->quants;
        scales[i] = entry->scales_va;
    }
    const std::size_t blocks_per_row = columns / kElemsPerBlock;
    const std::size_t x_bytes = tokens * columns * sizeof(float);
    std::uint64_t offsets[kMaxBatch];
    std::uint64_t span = 0;
    for (std::size_t i = 0; i < count; ++i) {
        offsets[i] = span;
        const std::uint64_t y_bytes =
            static_cast<std::uint64_t>(tokens) * tasks[i].rows * sizeof(float);
        span = (span + y_bytes + 255u) & ~static_cast<std::uint64_t>(255u);
    }
    if (!reserve_scratch(s, x_bytes, span)) {
        return false;
    }
    std::memcpy(s.x_cpu, x, tokens * columns * sizeof(float));

    for (std::size_t i = 0; i < count; ++i) {
        auto* params = reinterpret_cast<GemmParams*>(
            static_cast<std::uint8_t*>(s.params_cpu) + 1024 + 256 * i);
        params->rows = static_cast<std::uint32_t>(tasks[i].rows);
        params->blocks_per_row = static_cast<std::uint32_t>(blocks_per_row);
        params->token_count = static_cast<std::uint32_t>(tokens);
        params->tokens_per_tile = 64;
    }

    s.allocator->Reset();
    s.list->Reset(s.allocator.Get(), s.pso_gemm_rt.Get());
    s.list->SetComputeRootSignature(s.root.Get());
    if (s.trace) {
        s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
    }
    for (std::size_t i = 0; i < count; ++i) {
        s.list->SetComputeRootConstantBufferView(
            0, s.params->GetGPUVirtualAddress() + 1024 + 256 * i);
        s.list->SetComputeRootShaderResourceView(1, quants[i]->GetGPUVirtualAddress());
        s.list->SetComputeRootShaderResourceView(2, scales[i]);
        s.list->SetComputeRootShaderResourceView(3, s.xbuf->GetGPUVirtualAddress());
        s.list->SetComputeRootShaderResourceView(4, s.xbuf->GetGPUVirtualAddress());
        s.list->SetComputeRootUnorderedAccessView(5, s.ybuf->GetGPUVirtualAddress() + offsets[i]);
        std::uint32_t gx = static_cast<std::uint32_t>((tasks[i].rows + 3u) / 4u);
        std::uint32_t gy = static_cast<std::uint32_t>((tokens + 7u) / 8u);
        s.list->Dispatch(gx, gy, 1);
    }
    if (s.trace) {
        s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
    }

    if (!s.no_readback) {
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = s.ybuf.Get();
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        s.list->ResourceBarrier(1, &barrier);
        s.list->CopyBufferRegion(s.yread.Get(), 0, s.ybuf.Get(), 0, span);
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        s.list->ResourceBarrier(1, &barrier);
    }
    if (s.trace) {
        s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 2);
        s.list->ResolveQueryData(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 3,
                                 s.ts_read.Get(), 0);
    }
    s.list->Close();

    if (!execute_and_wait(s)) {
        return false;
    }

    if (!s.no_readback) {
        void* mapped = nullptr;
        const D3D12_RANGE read_range{0, span};
        if (FAILED(s.yread->Map(0, &read_range, &mapped))) {
            return false;
        }
        const auto* base = static_cast<const std::uint8_t*>(mapped);
        for (std::size_t i = 0; i < count; ++i) {
            std::memcpy(tasks[i].y, base + offsets[i],
                        tokens * tasks[i].rows * sizeof(float));
        }
        const D3D12_RANGE write_range{0, 0};
        s.yread->Unmap(0, &write_range);
    }
    s.stats.dispatches += count;
    ++s.stats.batches;
    return true;
}

// Public entry for batched prefill GEMM.
bool gemm_q8_0_batch(const float* x, const GemmTask* tasks, std::size_t count,
                     std::size_t tokens, std::size_t columns) noexcept {
    if (count == 0 || count > kMaxBatch) {
        return false;
    }
    if (columns % kElemsPerBlock != 0) {
        return false;
    }
    // Mirror the single-gemm entry gates so the batch path cannot bypass a
    // size, experimental, or DP4A decision the single path honours. The batch
    // kernel is the fp32 variant only: with DP4A on, the single path would run
    // int8-activation math and the batch would silently differ.
    if (!gemm_opted_in()) {
        return false;
    }
    if (dp4a_enabled()) {
        return false;
    }
    if (tokens == 0) {
        return false;
    }
    for (std::size_t i = 0; i < count; ++i) {
        if (!tasks[i].weights || !tasks[i].y || !tasks[i].rows) {
            return false;
        }
        if (tokens * tasks[i].rows * columns < kMinElements) {
            return false;
        }
    }
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!init_locked(s)) {
        return false;
    }
    if (gemm_rt_enabled() && s.pso_gemm_rt != nullptr) {
        if (gemm_rt_batch(s, x, tasks, count, tokens, columns)) {
            return true;
        }
    }
    return gemm_batch_locked(s, x, tasks, count, tokens, columns);
}

// FlashAttention for prefill with KV cache: computes multi-head attention for all tokens in parallel
// on GPU. Input Q is [tokens, heads, head_dim], K/V come from device KV cache.
// Output is [tokens, heads, head_dim].
// Uses tiled FlashAttention algorithm with causal masking and KV cache.
// Returns false if not supported (falls back to CPU attention).
bool flash_attention_prefill(const float* q, const float* k, const float* v, float* out,
                             std::size_t tokens, std::size_t heads, std::size_t kv_heads,
                             std::size_t head_dim, float inv_scale,
                             std::size_t layer, std::size_t start_pos,
                             std::size_t kv_context, std::size_t k_cache_heads, std::size_t k_cache_head_dim) noexcept {
    (void)k; (void)v; // K/V from host not used; read from device KV cache

    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!init_locked(s)) {
        if (s.trace) std::printf("[d3d12] flash_attention: FAIL - init_locked\n");
        return false;
    }
    if (!q || !out || tokens == 0 || heads == 0 || kv_heads == 0 || head_dim == 0) {
        if (s.trace) std::printf("[d3d12] flash_attention: FAIL - null ptr or zero dims\n");
        return false;
    }
    // Only support head_dim up to 128 (shared memory limit)
    if (head_dim > 128) {
        if (s.trace) std::printf("[d3d12] flash_attention: FAIL - head_dim > 128\n");
        return false;
    }
    // Only support up to 16 kv_heads (shared memory limit)
    if (kv_heads > 16) {
        if (s.trace) std::printf("[d3d12] flash_attention: FAIL - kv_heads > 16\n");
        return false;
    }
    // Only support up to 64 heads
    if (heads > 64) {
        if (s.trace) std::printf("[d3d12] flash_attention: FAIL - heads > 64\n");
        return false;
    }
    // Only support tokens that are reasonable for BLOCK_Q=32
    if (tokens > 2048) {
        if (s.trace) std::printf("[d3d12] flash_attention: FAIL - tokens > 2048\n");
        return false;
    }
    if (!s.pso_flash_attn) {
        if (s.trace) std::printf("[d3d12] flash_attention: FAIL - pso_flash_attn null\n");
        return false;
    }
    if (!s.kcache || !s.vcache) {
        if (s.trace) std::printf("[d3d12] flash_attention: FAIL - kcache/vcache null\n");
        return false; // KV cache not configured
    }
    // The layer stride must come from the context the cache was actually
    // allocated with, never from the caller. A caller that passes the model's
    // maximum context here (131072) while the cache was configured for 160
    // produces a stride 256 MB too large per layer, and the shader page-faults
    // on layer 1 - which is exactly how this backend used to lose the device.
    if (kv_heads != s.kv_heads || head_dim != s.kv_head_dim) {
        if (s.trace) std::printf("[d3d12] flash_attention: FAIL - kv shape %zu/%zu != cache %zu/%zu\n",
                                 kv_heads, head_dim, s.kv_heads, s.kv_head_dim);
        return false;
    }
    if (layer >= s.kv_layers) {
        if (s.trace) std::printf("[d3d12] flash_attention: FAIL - layer %zu >= kv_layers %zu\n",
                                 layer, s.kv_layers);
        return false;
    }
    if (start_pos + tokens > s.kv_context) {
        if (s.trace) std::printf("[d3d12] flash_attention: FAIL - start_pos %zu + tokens %zu > kv_context %zu\n",
                                 start_pos, tokens, s.kv_context);
        return false;
    }
    const std::size_t kv_ctx = s.kv_context;

    if (s.trace) std::printf("[d3d12] flash_attention: START tokens=%zu heads=%zu kv_heads=%zu head_dim=%zu layer=%zu start_pos=%zu\n",
                         tokens, heads, kv_heads, head_dim, layer, start_pos);

    // Upload Q to device (K/V are already in KV cache)
    const std::size_t q_bytes = tokens * heads * head_dim * sizeof(float);
    const std::size_t out_bytes = tokens * heads * head_dim * sizeof(float);

    // Grow the reusable scratch only when the batch got bigger. Both device
    // buffers are created in the state they hold between calls, so the round trip
    // below is identical on the first and every later use.
    if (!s.fa_q || !s.fa_out || !s.fa_qup || !s.fa_rd || s.fa_cap < q_bytes) {
        s.fa_q.Reset();
        s.fa_out.Reset();
        s.fa_qup.Reset();
        s.fa_rd.Reset();
        s.fa_qup_cpu = nullptr;
        const std::size_t cap = q_bytes > out_bytes ? q_bytes : out_bytes;
        s.fa_q = create_buffer(s.device.Get(), cap, D3D12_HEAP_TYPE_DEFAULT,
                               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        s.fa_out = create_buffer(s.device.Get(), cap, D3D12_HEAP_TYPE_DEFAULT,
                                 D3D12_RESOURCE_STATE_COPY_SOURCE,
                                 D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        s.fa_qup = create_buffer(s.device.Get(), cap, D3D12_HEAP_TYPE_UPLOAD,
                                 D3D12_RESOURCE_STATE_GENERIC_READ);
        s.fa_rd = create_buffer(s.device.Get(), cap, D3D12_HEAP_TYPE_READBACK,
                                D3D12_RESOURCE_STATE_COPY_DEST);
        if (!s.fa_q || !s.fa_out || !s.fa_qup || !s.fa_rd) {
            if (s.trace) std::printf("[d3d12] flash_attention: FAIL - scratch create\n");
            return false;
        }
        if (FAILED(s.fa_qup->Map(0, nullptr, &s.fa_qup_cpu))) {
            if (s.trace) std::printf("[d3d12] flash_attention: FAIL - scratch map\n");
            return false;
        }
        s.fa_cap = cap;
    }
    ID3D12Resource* qbuf = s.fa_q.Get();
    ID3D12Resource* outbuf = s.fa_out.Get();

    std::memcpy(s.fa_qup_cpu, q, q_bytes);

    s.allocator->Reset();
    s.list->Reset(s.allocator.Get(), nullptr);
    transition(s, qbuf, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               D3D12_RESOURCE_STATE_COPY_DEST);
    s.list->CopyBufferRegion(qbuf, 0, s.fa_qup.Get(), 0, q_bytes);
    transition(s, qbuf, D3D12_RESOURCE_STATE_COPY_DEST,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    // The KV caches live in UNORDERED_ACCESS (decode writes them from compute and its
    // attention reads them as UAVs). This shader reads them as SRVs, so they take one
    // NSR cycle here - once per prefill submission, never per decode token.
    transition(s, s.kcache.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(s, s.vcache.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    // Set up constant buffer for FlashAttentionWithCache
    struct FlashAttnParamsCB {
        uint32_t tokens;
        uint32_t heads;
        uint32_t kv_heads;
        uint32_t head_dim;
        float inv_scale;
        uint32_t kv_cache_base;
        uint32_t kv_cache_stride;
        uint32_t valid_kv_tokens;
        uint32_t start_pos;
        uint32_t first_pos;
        uint32_t pad0;
        uint32_t pad1;
    };
    FlashAttnParamsCB cb{};
    cb.tokens = static_cast<uint32_t>(tokens);
    cb.heads = static_cast<uint32_t>(heads);
    cb.kv_heads = static_cast<uint32_t>(kv_heads);
    cb.head_dim = static_cast<uint32_t>(head_dim);
    cb.inv_scale = inv_scale;
    cb.kv_cache_base = static_cast<uint32_t>(layer * kv_ctx * kv_heads * head_dim);
    cb.kv_cache_stride = static_cast<uint32_t>(kv_heads * head_dim);
    cb.valid_kv_tokens = static_cast<uint32_t>(start_pos + tokens);
    cb.start_pos = static_cast<uint32_t>(start_pos);
    cb.first_pos = 0;
    cb.pad0 = 0;
    cb.pad1 = 0;
    
    // Use params buffer for constant data
    std::memcpy(s.params_cpu, &cb, sizeof(cb));
    
    // Set up root signature and pipeline
    s.list->SetComputeRootSignature(s.root.Get());
    s.list->SetPipelineState(s.pso_flash_attn.Get());
    s.list->SetComputeRootConstantBufferView(0, s.params->GetGPUVirtualAddress());
    s.list->SetComputeRootShaderResourceView(1, qbuf->GetGPUVirtualAddress());
    s.list->SetComputeRootShaderResourceView(2, s.kcache->GetGPUVirtualAddress());
    s.list->SetComputeRootShaderResourceView(3, s.vcache->GetGPUVirtualAddress());
    // Transition output to copy source, after first bringing it out of the state
    // it was left in by the previous call.
    transition(s, outbuf, D3D12_RESOURCE_STATE_COPY_SOURCE,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    s.list->SetComputeRootUnorderedAccessView(5, outbuf->GetGPUVirtualAddress());
    s.list->Dispatch(static_cast<uint32_t>(heads), static_cast<uint32_t>(tokens), 1);
    // Hand the caches back to the decode path in the UAV state it assumes.
    transition(s, s.kcache.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    transition(s, s.vcache.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    transition(s, outbuf, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_COPY_SOURCE);
    s.list->CopyBufferRegion(s.fa_rd.Get(), 0, outbuf, 0, out_bytes);
    s.list->Close();

    const auto dispatch_start = std::chrono::steady_clock::now();
    if (!execute_and_wait(s)) {
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - dispatch_start)
                              .count();
        std::fprintf(stderr,
                     "[d3d12] flash_attention: FAIL - execute_and_wait after %.2f ms "
                     "(layer=%zu tokens=%zu)\n",
                     ms, layer, tokens);
        report_device_removed(s.device.Get(), "flash_attention_prefill");
        return false;
    }
    if (s.trace) {
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - dispatch_start)
                              .count();
        std::printf("[d3d12] flash_attention: gpu %.2f ms layer=%zu\n", ms, layer);
    }

    // Map and copy output
    void* mapped = nullptr;
    const D3D12_RANGE read_range{0, out_bytes};
    if (FAILED(s.fa_rd->Map(0, &read_range, &mapped))) {
        if (s.trace) std::printf("[d3d12] flash_attention: FAIL - readback Map\n");
        return false;
    }
    std::memcpy(out, mapped, out_bytes);
    const D3D12_RANGE write_range{0, 0};
    s.fa_rd->Unmap(0, &write_range);

    if (s.trace) std::printf("[d3d12] flash_attention: SUCCESS\n");
    return true;
}

// Public API functions in zen::d3d12 namespace
bool enabled() noexcept {
    const char* trace_env = std::getenv("ZEN_D3D12_TRACE");
    bool trace = trace_env && trace_env[0] == '1';
    bool opt = opted_in();
    if (!opt) {
        if (trace) std::printf("[d3d12] enabled: FAIL - opted_in=%d\n", opt);
        return false;
    }
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (trace) std::printf("[d3d12] enabled: ready=%d opted_in=%d\n", s.ready, opt);
    // Return true if backend is available (opted in), even if not yet initialized.
    // Callers that need initialization should call init() explicitly.
    return true;
}

const AdapterInfo* info() noexcept {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!s.ready) {
        return nullptr;
    }
    return &s.adapter;
}

const Stats& stats() noexcept { return state().stats; }

void trace_report() {
    print_trace_report();
}

// Public wrapper for pipelined multi-layer decode
bool layer_decode_pipeline(const LayerDecodeParams* layers, std::size_t num_layers,
                           const float* input, float* output,
                           const float* cos, const float* sin, std::size_t pairs,
                           std::size_t hidden, std::size_t heads, std::size_t kv_heads,
                           std::size_t head_dim, std::size_t ffn_rows, float epsilon) noexcept {
    if (!layers || !input || !output || num_layers == 0 || num_layers > 64) {
        return false;
    }
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!init_locked(s)) {
        return false;
    }
    return layer_decode_pipeline(layers, num_layers, input, output,
                                 cos, sin, pairs, hidden, heads, kv_heads,
                                 head_dim, ffn_rows, epsilon);
}

bool init() noexcept {
    if (!opted_in()) {
        return false;
    }
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    const char* trace = std::getenv("ZEN_D3D12_TRACE");
    s.trace = trace && trace[0] == '1';
    // The timestamp calibration bench needs the query heap that only the trace flag
    // creates. Implying trace here keeps it from silently reporting zero samples.
    if (const char* gputs = std::getenv("ZEN_D3D12_BENCH_GPUTS"); gputs && gputs[0] == '1') {
        s.trace = true;
    }
    const char* no_readback = std::getenv("ZEN_D3D12_NO_READBACK");
    s.no_readback = no_readback && no_readback[0] == '1';
    return init_locked(s);
}

void shutdown() noexcept {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.ready) {
        wait_gpu(s);
    }
    s.weights.clear();
    s.weight_bytes = 0;
    s.ybuf.Reset();
    s.yread.Reset();
    s.stage.Reset();
    s.stage_cpu = nullptr;
    s.stage_cap = 0;
    s.xbuf.Reset();
    s.x_cap = 0;
    s.y_cap = 0;
    s.params.Reset();
    s.params_cpu = nullptr;
    s.nbuf.Reset();
    s.n_cpu = nullptr;
    s.n_cap = 0;
    // Every device-created object must go, and every cached size must go with it:
    // leaving f_cap/w_cap set while the buffer is released would let a later init
    // reuse a scratch buffer that belongs to the destroyed device.
    s.pso_gemv.Reset();
    s.pso_gemm.Reset();
    s.pso_gemm_v2.Reset();
    s.pso_gemm_v3.Reset();
    s.pso_gemm_rt.Reset();
    s.pso_gemm_fused.Reset();
    s.pso_gemm_dp4a.Reset();
    s.pso_gemv_dp4a.Reset();
    s.pso_quantize_acts.Reset();
    s.pso_rope.Reset();
    s.pso_rope_prefill.Reset();
    s.pso_attn.Reset();
    s.pso_attn2.Reset();
    s.pso_silu.Reset();
    s.pso_rms.Reset();
    s.pso_rms_fuse.Reset();
    s.pso_silu_fuse.Reset();
    s.pso_attn_fuse.Reset();
    s.pso_addn.Reset();
    s.pso_flash_attn.Reset();
    s.fbuf.Reset();
    s.f_cap = 0;
    s.wbuf.Reset();
    s.w_cap = 0;
    s.w_cpu = nullptr;
    s.ts_heap.Reset();
    s.ts_read.Reset();
    s.ts_freq = 0;
    s.root.Reset();
    if (s.copy_fence_event) {
        CloseHandle(s.copy_fence_event);
        s.copy_fence_event = nullptr;
    }
    s.copy_fence.Reset();
    s.copy_list.Reset();
    s.copy_allocator.Reset();
    s.copy_queue.Reset();
    if (s.fence_event) {
        CloseHandle(s.fence_event);
        s.fence_event = nullptr;
    }
    s.fence.Reset();
    s.list.Reset();
    s.allocator.Reset();
    s.queue.Reset();
    s.device.Reset();
    s.ready = false;
    // A failed init latches this; clearing it here is what makes shutdown() a real
    // reset instead of a one-way trip into permanent failure.
    s.failed = false;
    s.trace = false;
    s.no_readback = false;
}

struct SiluParams {
    std::uint32_t count;
    std::uint32_t pad0;
    std::uint32_t pad1;
    std::uint32_t pad2;
};

// gate -> up -> SiLU -> down, all in one command list. The activation never leaves
// the device, so this replaces two submissions (each carrying ~70us of fixed cost)
// with one.
bool gemv_ffn_locked(State& s, const std::uint8_t* gate_w, const std::uint8_t* up_w,
                     const std::uint8_t* down_w, const float* x, float* y,
                     std::size_t ffn_rows, std::size_t columns, std::size_t out_rows) {
    if (ffn_rows == 0 || out_rows == 0 || columns % kElemsPerBlock != 0) {
        return false;
    }
    // Same F-09 reasoning as the batched path: each lookup may evict what the previous
    // one just inserted, so keep refcounted handles instead of raw entry pointers.
    ComPtr<ID3D12Resource> gate_q, up_q, down_q;
    std::uint64_t gate_s{}, up_s{}, down_s{};
    if (WeightEntry* e = lookup_weights(s, gate_w, ffn_rows, columns)) {
        gate_q = e->quants;
        gate_s = e->scales_va;
    } else {
        return false;
    }
    if (WeightEntry* e = lookup_weights(s, up_w, ffn_rows, columns)) {
        up_q = e->quants;
        up_s = e->scales_va;
    } else {
        return false;
    }
    if (WeightEntry* e = lookup_weights(s, down_w, out_rows, ffn_rows)) {
        down_q = e->quants;
        down_s = e->scales_va;
    } else {
        return false;
    }

    const std::uint64_t align = 256u;
    const std::uint64_t row_span = (ffn_rows * sizeof(float) + align - 1) & ~(align - 1);
    const std::uint64_t out_span = (out_rows * sizeof(float) + align - 1) & ~(align - 1);
    const std::uint64_t off_up = row_span;
    const std::uint64_t off_down = row_span + row_span;
    if (!reserve_scratch(s, columns * sizeof(float), off_down + out_span)) {
        return false;
    }

    const std::size_t f_bytes = ffn_rows * sizeof(float);
    if (f_bytes > s.f_cap) {
        s.fbuf.Reset();
        s.fbuf = create_buffer(s.device.Get(), f_bytes, D3D12_HEAP_TYPE_DEFAULT,
                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                               D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        if (!s.fbuf) {
            s.f_cap = 0;
            return false;
        }
        s.f_cap = f_bytes;
    }

    std::memcpy(s.x_cpu, x, columns * sizeof(float));

    auto* base = static_cast<std::uint8_t*>(s.params_cpu);
    auto write_gemv = [&](std::size_t slot, std::size_t rows, std::size_t cols) {
        auto* params = reinterpret_cast<GemvParams*>(base + 512 + 256 * slot);
        params->rows = static_cast<std::uint32_t>(rows);
        params->blocks_per_row = static_cast<std::uint32_t>(cols / kElemsPerBlock);
        params->rows_per_group = kRowsPerGroup;
        params->reserved = 0;
    };
    write_gemv(0, ffn_rows, columns);
    write_gemv(1, ffn_rows, columns);
    write_gemv(2, out_rows, ffn_rows);
    auto* silu = reinterpret_cast<SiluParams*>(base + 512 + 256 * 3);
    silu->count = static_cast<std::uint32_t>(ffn_rows);
    silu->pad0 = silu->pad1 = silu->pad2 = 0;

    const UINT ffn_groups = static_cast<UINT>((ffn_rows + kRowsPerGroup - 1) / kRowsPerGroup);
    const UINT out_groups = static_cast<UINT>((out_rows + kRowsPerGroup - 1) / kRowsPerGroup);
    const UINT silu_groups = static_cast<UINT>((ffn_rows + 255) / 256);
    const std::uint64_t params_base = s.params->GetGPUVirtualAddress();
    const std::uint64_t y_base = s.ybuf->GetGPUVirtualAddress();

    s.allocator->Reset();
    s.list->Reset(s.allocator.Get(), s.pso_gemv.Get());
    s.list->SetComputeRootSignature(s.root.Get());
    if (s.trace) {
        s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
    }

    s.list->SetComputeRootConstantBufferView(0, params_base + 512);
    s.list->SetComputeRootShaderResourceView(1, gate_q->GetGPUVirtualAddress());
    s.list->SetComputeRootShaderResourceView(2, gate_s);
    s.list->SetComputeRootShaderResourceView(3, s.xbuf->GetGPUVirtualAddress());
    s.list->SetComputeRootUnorderedAccessView(5, y_base);
    s.list->Dispatch(ffn_groups, 1, 1);

    s.list->SetComputeRootConstantBufferView(0, params_base + 512 + 256);
    s.list->SetComputeRootShaderResourceView(1, up_q->GetGPUVirtualAddress());
    s.current_comp = Comp::FfnUpDown;
    s.list->SetComputeRootShaderResourceView(2, up_s);
    s.list->SetComputeRootUnorderedAccessView(5, y_base + off_up);
    s.list->Dispatch(ffn_groups, 1, 1);

    // The SiLU kernel reads the two matvec results as shader resources.
    transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    s.list->SetPipelineState(s.pso_silu.Get());
    s.list->SetComputeRootConstantBufferView(0, params_base + 512 + 256 * 3);
    s.list->SetComputeRootShaderResourceView(1, y_base);
    s.list->SetComputeRootShaderResourceView(2, y_base + off_up);
    s.list->SetComputeRootUnorderedAccessView(5, s.fbuf->GetGPUVirtualAddress());
    s.list->Dispatch(silu_groups, 1, 1);
    transition(s, s.fbuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    s.list->SetPipelineState(s.pso_gemv.Get());
    s.list->SetComputeRootConstantBufferView(0, params_base + 512 + 256 * 2);
    s.list->SetComputeRootShaderResourceView(1, down_q->GetGPUVirtualAddress());
    s.list->SetComputeRootShaderResourceView(2, down_s);
    s.list->SetComputeRootShaderResourceView(3, s.fbuf->GetGPUVirtualAddress());
    s.list->SetComputeRootUnorderedAccessView(5, y_base + off_down);
    s.list->Dispatch(out_groups, 1, 1);
    if (s.trace) {
        s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
    }

    transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_COPY_SOURCE);
    s.list->CopyBufferRegion(s.yread.Get(), 0, s.ybuf.Get(), off_down, out_rows * sizeof(float));
    transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (s.trace) {
        s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 2);
        s.list->ResolveQueryData(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 3,
                                 s.ts_read.Get(), 0);
    }
    s.list->Close();

    const std::uint64_t ffn_mark = s.stats.wait_us;
    if (!execute_and_wait(s)) {
        return false;
    }
    s.stats.ffn_wait_us += s.stats.wait_us - ffn_mark;
    accumulate_gpu_us(s, s.stats.ffn_gpu_us, 0, 1);
    accumulate_gpu_us(s, s.stats.ffn_copy_us, 1, 2);
    {
        std::uint64_t dev = 0;
        accumulate_gpu_us(s, dev, 0, 1);
        note_dev(s, Comp::FfnUpDown, dev);
    }

    void* mapped = nullptr;
    const D3D12_RANGE read_range{0, out_rows * sizeof(float)};
    if (FAILED(s.yread->Map(0, &read_range, &mapped))) {
        return false;
    }
    std::memcpy(y, mapped, out_rows * sizeof(float));
    const D3D12_RANGE written{0, 0};
    s.yread->Unmap(0, &written);
    s.stats.dispatches += 4;
    ++s.stats.batches;
    return true;
}

bool gemv_q8_0_batch(const float* x, const GemvTask* tasks, std::size_t count) noexcept {
    if (count < 2) {
        return false;
    }
    std::size_t work = 0;
    for (std::size_t i = 0; i < count; ++i) {
        work += tasks[i].rows * tasks[i].columns;
    }
    if (work < kMinElements) {
        return false;
    }
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!init_locked(s)) {
        return false;
    }
    if (!gemv_batch_locked(s, x, tasks, count)) {
        ++s.stats.fallbacks;
        return false;
    }
    return true;
}

struct AddnParams {
    std::uint32_t count;
    std::uint32_t pad0;
    std::uint32_t pad1;
    std::uint32_t pad2;
};

// Element-wise out = a + b. The two operands arrive through the two upload buffers
// already used for activations and small vectors, so no new residency is needed.
bool addn_locked(State& s, const float* a, const float* b, float* out, std::size_t count) {
    if (!count) {
        return false;
    }
    const std::size_t bytes = count * sizeof(float);
    if (!reserve_scratch(s, bytes, bytes)) {
        return false;
    }
    std::memcpy(s.x_cpu, a, bytes);

    if (bytes > s.w_cap) {
        s.wbuf.Reset();
        s.wbuf = create_buffer(s.device.Get(), bytes, upload_heap_type(),
                               upload_heap_state());
        if (!s.wbuf || FAILED(s.wbuf->Map(0, nullptr, &s.w_cpu))) {
            s.w_cap = 0;
            return false;
        }
        s.w_cap = bytes;
    }
    std::memcpy(s.w_cpu, b, bytes);

    // Offset 768 is outside both the lone-matvec slot (0) and the gemm/rms slot (256),
    // so a residual add never aliases parameters of another in-flight shape.
    s.current_comp = Comp::Residual;
    constexpr std::uint64_t kAddnParamOffset = 768;
    auto* params =
        reinterpret_cast<AddnParams*>(static_cast<std::uint8_t*>(s.params_cpu) + kAddnParamOffset);
    params->count = static_cast<std::uint32_t>(count);
    params->pad0 = params->pad1 = params->pad2 = 0;

    const UINT groups = static_cast<UINT>((count + 255) / 256);

    s.allocator->Reset();
    s.list->Reset(s.allocator.Get(), s.pso_addn.Get());
    s.list->SetComputeRootSignature(s.root.Get());
    s.list->SetComputeRootConstantBufferView(0, s.params->GetGPUVirtualAddress() + kAddnParamOffset);
    s.list->SetComputeRootShaderResourceView(1, s.xbuf->GetGPUVirtualAddress());
    s.list->SetComputeRootShaderResourceView(2, s.wbuf->GetGPUVirtualAddress());
    s.list->SetComputeRootUnorderedAccessView(5, s.ybuf->GetGPUVirtualAddress());
    s.list->Dispatch(groups, 1, 1);

    transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_COPY_SOURCE);
    s.list->CopyBufferRegion(s.yread.Get(), 0, s.ybuf.Get(), 0, bytes);
    transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    s.list->Close();

    if (!execute_and_wait(s)) {
        return false;
    }

    void* mapped = nullptr;
    const D3D12_RANGE read_range{0, bytes};
    if (FAILED(s.yread->Map(0, &read_range, &mapped))) {
        return false;
    }
    std::memcpy(out, mapped, bytes);
    const D3D12_RANGE written{0, 0};
    s.yread->Unmap(0, &written);
    ++s.stats.dispatches;
    return true;
}

struct RmsParams {
    std::uint32_t count;
    std::uint32_t rows;
    float epsilon;
    std::uint32_t pad;
};

// Matches rope.hlsl's cbuffer. `pairs` is rotary_dim/2; the field is not called
// `half` because that is a reserved HLSL type and shadowing it breaks the shader.
struct RopeParams {
    std::uint32_t heads;
    std::uint32_t head_dim;
    std::uint32_t pairs;
    std::uint32_t pad0;
};

namespace {
// Parameter slots. 1024/1280 sit clear of the lone-matvec slot (0), the gemm/rms slot
// (256), the addn slot (768) and both fused-FFN slots (>= 2560), so no two in-flight
// shapes ever alias the same bytes of the parameter buffer.
constexpr std::uint64_t kRopeParamOffset = 1024;
constexpr std::uint64_t kRopeParamOffset2 = 1536;  // second rope dispatch (k)
// DP4A decode activation-quant params: four 256B slots, one per matvec site group
// (QKV input, o input, gate/up input, down input). Upload-heap CBVs are read at
// execution time, not record time, so each distinct (rows, cols) group writes its
// OWN slot at record time: a single shared slot would make every dispatch use the
// LAST written values. These sit clear of the pipeline whole-layer slots (4096+),
// rope (1024/1536), attention (1280) and the pipeline per-layer attn slots
// (8192+). The only other users of 1792/2048/2304/2560 are the batch paths
// (512+256*i, i=5..7) and the fused-FFN addn pair (2560/2816), which are separate
// submissions serialized under the same mutex after a full fence wait, so
// sequential reuse across paths is safe.
constexpr std::uint64_t kQActQkv = 1792;
constexpr std::uint64_t kQActO = 2048;
constexpr std::uint64_t kQActGu = 2304;
constexpr std::uint64_t kQActDown = 2560;
}  // namespace

// Rotary position embedding applied on the device to q or k.
//
// The cos/sin table is evaluated on the host and uploaded. The CPU path builds it with
// std::pow/cos/sin in double-promoted arithmetic; recomputing it with powf/cosf here
// would differ in the last bits and could flip a sampled id, so the kernel only
// rotates. Both tables share one upload buffer, with the sin table pushed to a
// 256-byte boundary because that is what a root SRV wants for its base address.
bool rope_locked(State& s, const float* x, const float* cos, const float* sin, float* out,
                 std::size_t heads, std::size_t head_dim, std::size_t pairs,
                 bool interleaved_pairs) {
    if (!heads || !head_dim || !pairs || pairs * 2 > head_dim) {
        return false;
    }
    const std::size_t bytes = heads * head_dim * sizeof(float);
    if (!reserve_scratch(s, bytes, bytes)) {
        return false;
    }
    std::memcpy(s.x_cpu, x, bytes);

    const std::size_t tab_bytes = pairs * sizeof(float);
    const std::size_t sin_off = (tab_bytes + 255u) & ~std::size_t{255u};
    const std::size_t w_bytes = sin_off + tab_bytes;
    if (w_bytes > s.w_cap) {
        s.wbuf.Reset();
        s.wbuf = create_buffer(s.device.Get(), w_bytes, upload_heap_type(),
                               upload_heap_state());
        if (!s.wbuf || FAILED(s.wbuf->Map(0, nullptr, &s.w_cpu))) {
            s.w_cap = 0;
            return false;
        }
        s.w_cap = w_bytes;
    }
    std::memcpy(s.w_cpu, cos, tab_bytes);
    std::memcpy(static_cast<std::uint8_t*>(s.w_cpu) + sin_off, sin, tab_bytes);

    auto* params =
        reinterpret_cast<RopeParams*>(static_cast<std::uint8_t*>(s.params_cpu) + kRopeParamOffset);
    params->heads = static_cast<std::uint32_t>(heads);
    params->head_dim = static_cast<std::uint32_t>(head_dim);
    params->pairs = static_cast<std::uint32_t>(pairs);
    // pad0 carries the pairing layout the shader pairs elements with:
    // 0 = NeoX half-split (i with i + pairs), 1 = llama NORM (2i with 2i+1).
    params->pad0 = interleaved_pairs ? 1u : 0u;

    // One thread per output element: the kernel covers the whole head and copies the
    // dimensions past rotary_dim through, so the grid is heads * head_dim.
    const UINT groups = static_cast<UINT>((heads * head_dim + 255) / 256);

    s.allocator->Reset();
    s.list->Reset(s.allocator.Get(), s.pso_rope.Get());
    s.list->SetComputeRootSignature(s.root.Get());
    s.list->SetComputeRootConstantBufferView(0, s.params->GetGPUVirtualAddress() +
                                                    kRopeParamOffset);
    s.list->SetComputeRootShaderResourceView(1, s.xbuf->GetGPUVirtualAddress());
    s.list->SetComputeRootShaderResourceView(2, s.wbuf->GetGPUVirtualAddress());
    s.list->SetComputeRootShaderResourceView(3, s.wbuf->GetGPUVirtualAddress() + sin_off);
    s.list->SetComputeRootUnorderedAccessView(5, s.ybuf->GetGPUVirtualAddress());

    s.current_comp = Comp::Norms;
    s.list->Dispatch(groups, 1, 1);

    transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_COPY_SOURCE);
    s.list->CopyBufferRegion(s.yread.Get(), 0, s.ybuf.Get(), 0, bytes);
    transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    s.list->Close();

    if (!execute_and_wait(s)) {
        return false;
    }

    void* mapped = nullptr;
    const D3D12_RANGE read_range{0, bytes};
    if (FAILED(s.yread->Map(0, &read_range, &mapped))) {
        return false;
    }
    std::memcpy(out, mapped, bytes);
    const D3D12_RANGE written{0, 0};
    s.yread->Unmap(0, &written);
    ++s.stats.dispatches;
    return true;
}

bool rope_apply(const float* x, const float* cos, const float* sin, float* out,
                std::size_t heads, std::size_t head_dim, std::size_t pairs,
                bool interleaved_pairs) noexcept {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!init_locked(s)) {
        return false;
    }
    if (!rope_locked(s, x, cos, sin, out, heads, head_dim, pairs, interleaved_pairs)) {
        ++s.stats.fallbacks;
        return false;
    }
    return true;
}

// Matches rope_prefill.hlsl's cbuffer exactly. Eight dwords (32 bytes) so the
// struct stays whole inside one 256B param slot; tokens is the grid's T axis.
struct RopePrefillParams {
    std::uint32_t heads;
    std::uint32_t head_dim;
    std::uint32_t pairs;
    std::uint32_t pad0;
    std::uint32_t tokens;
    std::uint32_t pad1;
    std::uint32_t pad2;
    std::uint32_t pad3;
};

// Batched prefill clone of rope_locked: same staging (xbuf -> dispatch -> yread),
// same root signature/slots, same table upload through wbuf, but the grid and the
// table index carry a token axis. Kept SEPARATE from rope_locked on purpose: decode
// rope (rope.hlsl, [heads,head_dim] grid, single table) is bit-identical and
// untouched; the only new risk surface is this function and its shader.
bool rope_prefill_locked(State& s, const float* x, const float* cos, const float* sin,
                         float* out, std::size_t tokens, std::size_t heads,
                         std::size_t head_dim, std::size_t pairs,
                         bool interleaved_pairs) {
    if (!tokens || !heads || !head_dim || !pairs || pairs * 2 > head_dim) {
        return false;
    }
    if (!s.pso_rope_prefill) {
        return false;
    }
    // Guard the 32-bit dispatch math and the persistent scratch: a prefill of
    // tokens*heads*head_dim floats above 64 MB would force a 64 MB xbuf/ybuf
    // growth for a rotation, so fall back to the CPU loop instead. Real Llama-3.2-1B
    // prefill is 1 MB (Q) / 256 KB (K) at T=128; the cap is generous yet bounded.
    const std::uint64_t total = static_cast<std::uint64_t>(tokens) * heads * head_dim;
    if (total > (64ull * 1024 * 1024) / sizeof(float)) {
        return false;
    }
    const std::size_t bytes = static_cast<std::size_t>(total) * sizeof(float);
    if (!reserve_scratch(s, bytes, bytes)) {
        return false;
    }
    std::memcpy(s.x_cpu, x, bytes);

    // Tables are token-major [tokens * pairs]; both must share one upload buffer
    // with the sin half 256-byte aligned (root-SRV base-address requirement).
    const std::size_t tab_floats = tokens * pairs;
    const std::size_t tab_bytes = tab_floats * sizeof(float);
    const std::size_t sin_off = (tab_bytes + 255u) & ~std::size_t{255u};
    const std::size_t w_bytes = sin_off + tab_bytes;
    if (w_bytes > s.w_cap) {
        s.wbuf.Reset();
        s.wbuf = create_buffer(s.device.Get(), w_bytes, upload_heap_type(),
                               upload_heap_state());
        if (!s.wbuf || FAILED(s.wbuf->Map(0, nullptr, &s.w_cpu))) {
            s.w_cap = 0;
            return false;
        }
        s.w_cap = w_bytes;
    }
    std::memcpy(s.w_cpu, cos, tab_bytes);
    std::memcpy(static_cast<std::uint8_t*>(s.w_cpu) + sin_off, sin, tab_bytes);

    auto* params =
        reinterpret_cast<RopePrefillParams*>(static_cast<std::uint8_t*>(s.params_cpu) + kRopeParamOffset);
    params->heads = static_cast<std::uint32_t>(heads);
    params->head_dim = static_cast<std::uint32_t>(head_dim);
    params->pairs = static_cast<std::uint32_t>(pairs);
    params->pad0 = interleaved_pairs ? 1u : 0u;
    params->tokens = static_cast<std::uint32_t>(tokens);
    params->pad1 = params->pad2 = params->pad3 = 0;

    // One thread per output element across the whole [tokens, heads, head_dim]
    // tensor; the kernel decodes (token, head, dim) and copies dims past
    // rotary_dim through, exactly like rope.hlsl.
    const UINT groups = static_cast<UINT>((total + 255) / 256);

    s.allocator->Reset();
    s.list->Reset(s.allocator.Get(), s.pso_rope_prefill.Get());
    s.list->SetComputeRootSignature(s.root.Get());
    s.list->SetComputeRootConstantBufferView(0, s.params->GetGPUVirtualAddress() +
                                                    kRopeParamOffset);
    s.list->SetComputeRootShaderResourceView(1, s.xbuf->GetGPUVirtualAddress());
    s.list->SetComputeRootShaderResourceView(2, s.wbuf->GetGPUVirtualAddress());
    s.list->SetComputeRootShaderResourceView(3, s.wbuf->GetGPUVirtualAddress() + sin_off);
    s.list->SetComputeRootUnorderedAccessView(5, s.ybuf->GetGPUVirtualAddress());

    s.current_comp = Comp::Norms;
    s.list->Dispatch(groups, 1, 1);

    transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_COPY_SOURCE);
    s.list->CopyBufferRegion(s.yread.Get(), 0, s.ybuf.Get(), 0, bytes);
    transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    s.list->Close();

    if (!execute_and_wait(s)) {
        return false;
    }

    void* mapped = nullptr;
    const D3D12_RANGE read_range{0, bytes};
    if (FAILED(s.yread->Map(0, &read_range, &mapped))) {
        return false;
    }
    std::memcpy(out, mapped, bytes);
    const D3D12_RANGE written{0, 0};
    s.yread->Unmap(0, &written);
    ++s.stats.dispatches;
    return true;
}

bool rope_apply_prefill(const float* x, const float* cos, const float* sin, float* out,
                        std::size_t tokens, std::size_t heads, std::size_t head_dim,
                        std::size_t pairs, bool interleaved_pairs) noexcept {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!init_locked(s)) {
        return false;
    }
    if (!rope_prefill_locked(s, x, cos, sin, out, tokens, heads, head_dim, pairs,
                             interleaved_pairs)) {
        ++s.stats.fallbacks;
        return false;
    }
    return true;
}

// Matches attention.hlsl's cbuffer exactly: 8 uints then floats, then the two
// layer-slice element offsets the UAV reads are relative to - 14 dwords in total.
struct AttnParams {
    std::uint32_t heads;
    std::uint32_t kv_heads;
    std::uint32_t head_dim;
    std::uint32_t context;
    std::uint32_t first_pos;
    std::uint32_t pos;
    std::uint32_t scores_off;
    std::uint32_t out_off;
    float inv_scale;
    float pad0, pad1, pad2;
    std::uint32_t kv_base;
    std::uint32_t v_base;
};

namespace {
constexpr std::uint64_t kAttnParamOffset = 1280;
}

// (Re)creates a CPU-visible buffer and maps it. Upload-heap resources are used for
// q/k/v and the cos/sin table because those are only ever read as SRVs on the GPU, and
// staying CPU-visible avoids a staging copy for a few kilobytes per layer.
bool attn_upload(State& s, ComPtr<ID3D12Resource>& buf, void*& cpu, std::size_t& cap,
                 std::size_t bytes) {
    if (bytes <= cap && buf) {
        return true;
    }
    buf.Reset();
    cpu = nullptr;
    cap = 0;
    buf = create_buffer(s.device.Get(), bytes, upload_heap_type(), upload_heap_state());
    if (!buf || FAILED(buf->Map(0, nullptr, &cpu))) {
        buf.Reset();
        cpu = nullptr;
        return false;
    }
    cap = bytes;
    return true;
}

// Creates a VRAM buffer whose state the caller manages explicitly. COMMON as the
// initial state keeps the first barrier honest whatever heap the caller picked.
bool attn_device(State& s, ComPtr<ID3D12Resource>& buf, std::size_t& cap, std::size_t bytes) {
    if (bytes <= cap && buf) {
        return true;
    }
    buf.Reset();
    cap = 0;
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    // Without this flag every use as a UAV is an invalid call that removes the device,
    // which is what made the first attention_block submission poison every allocation
    // after it. reserve_scratch sets it for ybuf for exactly this reason.
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    if (FAILED(s.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                 D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                 IID_PPV_ARGS(&buf)))) {
        buf.Reset();
        return false;
    }
    cap = bytes;
    return true;
}

bool kv_cache_configure(std::size_t layers, std::size_t context, std::size_t kv_heads,
                        std::size_t head_dim) noexcept {
    const char* trace_env = std::getenv("ZEN_D3D12_TRACE");
    bool trace = trace_env && trace_env[0] == '1';
    if (trace) std::printf("[d3d12] kv_cache_configure: layers=%zu context=%zu kv_heads=%zu head_dim=%zu\n", layers, context, kv_heads, head_dim);

    if (!layers || !context || !kv_heads || !head_dim) {
        if (trace) std::printf("[d3d12] kv_cache_configure: FAIL - zero params\n");
        return false;
    }
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!init_locked(s)) {
        if (trace) std::printf("[d3d12] kv_cache_configure: FAIL - init_locked\n");
        return false;
    }
    if (s.kcache && s.kv_layers == layers && s.kv_context == context &&
        s.kv_heads == kv_heads && s.kv_head_dim == head_dim) {
        if (trace) std::printf("[d3d12] kv_cache_configure: already configured, returning true\n");
        return true;
    }
    const std::size_t bytes = layers * kv_heads * context * head_dim * sizeof(float);
    if (trace) std::printf("[d3d12] kv_cache_configure: allocating %zu bytes\n", bytes);
    s.kcache.Reset();
    s.vcache.Reset();
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    // The caches are written by compute (rope-k / v matvec UAV stores when
    // ZEN_D3D12_KVDIRECT is on) and read back by the attention kernel as UAVs, so both
    // buffers carry the UAV flag and are created directly in UNORDERED_ACCESS. They
    // stay in that state for their whole life: no per-layer transition ever runs on
    // these 256MB buffers, only COPY_DEST cycles around the prefill appends and a
    // NSR cycle inside the flash-attention prefill submission.
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    if (FAILED(s.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                 IID_PPV_ARGS(&s.kcache)))) {
        std::fprintf(stderr, "kv: kcache create %zu bytes failed\n", bytes);
        if (trace) std::printf("[d3d12] kv_cache_configure: FAIL - kcache create\n");
        return false;
    }
    if (FAILED(s.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                 IID_PPV_ARGS(&s.vcache)))) {
        std::fprintf(stderr, "kv: vcache create %zu bytes failed\n", bytes);
        s.kcache.Reset();
        if (trace) std::printf("[d3d12] kv_cache_configure: FAIL - vcache create\n");
        return false;
    }
    s.kv_layers = layers;
    s.kv_context = context;
    s.kv_heads = kv_heads;
    s.kv_head_dim = head_dim;
    // Fresh buffers hold nothing: no position can be claimed resident. (The
    // same-shape early return above deliberately KEEPS the existing record - the
    // runtime re-checks the cache on every decode step through
    // device_attention_available, and invalidating there would make the resident
    // skip unreachable.)
    s.kv_resident.assign(layers, {});
    if (trace) std::printf("[d3d12] kv_cache_configure: SUCCESS\n");
    return true;
}

void kv_cache_reset() noexcept {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!s.kcache) {
        return;
    }
    s.kcache.Reset();
    s.vcache.Reset();
    s.kv_layers = s.kv_context = s.kv_heads = s.kv_head_dim = 0;
    // The buffers are gone; nothing is resident any more.
    s.kv_resident.clear();
}

namespace {

// Merge [pos, pos + n) into layer `layer`'s resident span. The span always denotes a
// contiguous range this backend has written; disjoint segments are dropped rather than
// merged, so the record can only ever under-report, never over-report.
void kv_resident_merge(State& s, std::size_t layer, std::size_t pos, std::size_t n) {
    if (layer >= s.kv_resident.size() || n == 0) {
        return;
    }
    State::KvResidentSpan& span = s.kv_resident[layer];
    if (!span.valid) {
        span.first = pos;
        span.count = n;
        span.valid = true;
        return;
    }
    const std::size_t lo = span.first;
    const std::size_t hi = span.first + span.count;  // exclusive
    const std::size_t p_hi = pos + n;                // exclusive
    if (p_hi < lo || pos > hi) {
        return;  // disjoint with a gap: keep the existing (under-reporting) span
    }
    const std::size_t new_lo = pos < lo ? pos : lo;
    const std::size_t new_hi = p_hi > hi ? p_hi : hi;
    span.first = new_lo;
    span.count = new_hi - new_lo;
}

// True when layer `layer`'s span covers [first, first + count).
bool kv_resident_covers(State& s, std::size_t layer, std::size_t first, std::size_t count) {
    if (layer >= s.kv_resident.size()) {
        return false;
    }
    const State::KvResidentSpan& span = s.kv_resident[layer];
    if (!span.valid || first < span.first || first - span.first > span.count) {
        return false;
    }
    return count <= span.count - (first - span.first);
}

}  // namespace

bool kv_cache_prefix_resident(std::size_t first_pos, std::size_t count) noexcept {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!s.ready || s.failed || !s.kcache || s.kv_layers == 0) {
        return false;
    }
    if (s.kv_resident.size() != s.kv_layers) {
        return false;
    }
    for (std::size_t layer = 0; layer < s.kv_layers; ++layer) {
        if (!kv_resident_covers(s, layer, first_pos, count)) {
            return false;
        }
    }
    return true;
}

// Whole attention block of one decode step on the device:
//   qr = rope(q);  kr = rope(k);  cache[..][pos] = kr / v;
//   att = softmax(qr . cache_k / sqrt(d)) . cache_v
// One submission, one readback of `att`. The cache never leaves VRAM, which is the
// point of tasks 4 and 6 in docs/d3d12_full_backend_todo.md.
bool attn_block_locked(State& s, const float* q, const float* k, const float* v,
                       const float* cos, const float* sin, std::size_t heads,
                       std::size_t kv_heads, std::size_t head_dim, std::size_t pairs,
                       std::size_t layer, std::size_t pos, std::size_t first_pos,
                       float* out, bool interleaved_pairs) {
    if (!heads || !kv_heads || !head_dim || !pairs || pairs * 2 > head_dim) {
        return false;
    }
    if (!s.kcache || layer >= s.kv_layers || pos >= s.kv_context) {
        return false;
    }
    if (s.kv_heads != kv_heads || s.kv_head_dim != head_dim) {
        return false;
    }

    const bool kv_direct = kv_direct_enabled();
    const std::size_t stride = s.kv_heads * s.kv_context * s.kv_head_dim * sizeof(float);
    const std::size_t dst = layer * stride + pos * kv_heads * head_dim * sizeof(float);
    // Direct k append: rope-k writes its output straight into the cache row through the
    // u0 UAV binding. Root UAV addresses must be 256B aligned, so a row that lands on an
    // odd offset (never true for real models) keeps the copy path.
    const bool k_direct = kv_direct && (dst % 256 == 0);

    const std::size_t q_bytes = heads * head_dim * sizeof(float);
    const std::size_t kv_bytes = kv_heads * head_dim * sizeof(float);
    const std::size_t tab_bytes = pairs * sizeof(float);
    // Scores and the block output are disjoint regions of one UAV, because the root
    // signature exposes a single UAV slot. See attention.hlsl.
    const std::size_t scores_floats = heads * s.kv_context;
    const std::size_t att_floats = heads * head_dim;
    const std::size_t datt_bytes = (scores_floats + att_floats) * sizeof(float);

    if (!attn_upload(s, s.dq, s.dq_cpu, s.dq_cap, q_bytes)) {
        std::fprintf(stderr, "attn: dq upload %zu bytes failed\n", q_bytes);
        return false;
    }
    if (!attn_upload(s, s.dk, s.dk_cpu, s.dk_cap, kv_bytes)) {
        std::fprintf(stderr, "attn: dk upload %zu bytes failed\n", kv_bytes);
        return false;
    }
    if (!attn_upload(s, s.dv, s.dv_cpu, s.dv_cap, kv_bytes)) {
        std::fprintf(stderr, "attn: dv upload %zu bytes failed\n", kv_bytes);
        return false;
    }
    if (!attn_upload(s, s.dcos, s.dcos_cpu, s.dcos_cap, tab_bytes)) {
        std::fprintf(stderr, "attn: dcos upload %zu bytes failed\n", tab_bytes);
        return false;
    }
    if (!attn_upload(s, s.dsin, s.dsin_cpu, s.dsin_cap, tab_bytes)) {
        std::fprintf(stderr, "attn: dsin upload %zu bytes failed\n", tab_bytes);
        return false;
    }
    if (!attn_device(s, s.dqr, s.dqr_cap, q_bytes)) {
        std::fprintf(stderr, "attn: dqr device %zu bytes failed\n", q_bytes);
        return false;
    }
    // dkr is only a staging stop for the copy path; with the direct append rope-k never
    // writes it, so the buffer is not even created.
    if (!k_direct && !attn_device(s, s.dkr, s.dkr_cap, kv_bytes)) {
        std::fprintf(stderr, "attn: dkr device %zu bytes failed\n", kv_bytes);
        return false;
    }
    if (!attn_device(s, s.datt, s.datt_cap, datt_bytes)) {
        std::fprintf(stderr, "attn: datt device %zu bytes failed\n", datt_bytes);
        return false;
    }
    if (!reserve_scratch(s, q_bytes, q_bytes)) {
        std::fprintf(stderr, "attn: reserve_scratch %zu bytes failed\n", q_bytes);
        return false;
    }

    std::memcpy(s.dq_cpu, q, q_bytes);
    std::memcpy(s.dk_cpu, k, kv_bytes);
    std::memcpy(s.dv_cpu, v, kv_bytes);
    std::memcpy(s.dcos_cpu, cos, tab_bytes);
    std::memcpy(s.dsin_cpu, sin, tab_bytes);

    auto* rp = reinterpret_cast<RopeParams*>(static_cast<std::uint8_t*>(s.params_cpu) +
                                             kRopeParamOffset);
    auto* ap = reinterpret_cast<AttnParams*>(static_cast<std::uint8_t*>(s.params_cpu) +
                                             kAttnParamOffset);

    s.allocator->Reset();
    s.list->Reset(s.allocator.Get(), s.pso_rope.Get());
    s.list->SetComputeRootSignature(s.root.Get());
    s.current_comp = Comp::Norms;

    // 1/2. RoPE on q and k. q lands in VRAM scratch; k lands either straight in the
    // cache row (direct) or in dkr for the copy below.
    transition(s, s.dqr.Get(), D3D12_RESOURCE_STATE_COMMON,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (!k_direct) {
        transition(s, s.dkr.Get(), D3D12_RESOURCE_STATE_COMMON,
                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }

    const UINT rope_groups = static_cast<UINT>((heads * head_dim + 255) / 256);
    rp->heads = static_cast<std::uint32_t>(heads);
    rp->head_dim = static_cast<std::uint32_t>(head_dim);
    rp->pairs = static_cast<std::uint32_t>(pairs);
    rp->pad0 = interleaved_pairs ? 1u : 0u;
    s.list->SetComputeRootConstantBufferView(0, s.params->GetGPUVirtualAddress() +
                                                    kRopeParamOffset);
    s.list->SetComputeRootShaderResourceView(1, s.dq->GetGPUVirtualAddress());
    s.list->SetComputeRootShaderResourceView(2, s.dcos->GetGPUVirtualAddress());
    s.list->SetComputeRootShaderResourceView(3, s.dsin->GetGPUVirtualAddress());
    s.list->SetComputeRootUnorderedAccessView(5, s.dqr->GetGPUVirtualAddress());
    s.list->Dispatch(rope_groups, 1, 1);

    // Separate parameter slot: both rope dispatches used to share one cbuffer, and the
    // GPU reads the slot at execution time, so the q dispatch would have run with the
    // k dispatch's head count.
    const UINT krope_groups = static_cast<UINT>((kv_heads * head_dim + 255) / 256);
    auto* krp = reinterpret_cast<RopeParams*>(static_cast<std::uint8_t*>(s.params_cpu) +
                                              kRopeParamOffset2);
    krp->heads = static_cast<std::uint32_t>(kv_heads);
    krp->head_dim = static_cast<std::uint32_t>(head_dim);
    krp->pairs = static_cast<std::uint32_t>(pairs);
    krp->pad0 = interleaved_pairs ? 1u : 0u;
    s.list->SetComputeRootConstantBufferView(0, s.params->GetGPUVirtualAddress() +
                                                    kRopeParamOffset2);
    s.list->SetComputeRootShaderResourceView(1, s.dk->GetGPUVirtualAddress());
    s.list->SetComputeRootUnorderedAccessView(
        5, k_direct ? (s.kcache->GetGPUVirtualAddress() + dst) : s.dkr->GetGPUVirtualAddress());
    s.list->Dispatch(krope_groups, 1, 1);

    // 3. Append k and v at this position. Direct mode: k is already in the cache (the
    // rope store above) and only the UAV barrier below orders it; v still comes from
    // the host, so its row is the one copy left in this path. Copy mode: both rows are
    // device-to-device CopyBufferRegions, cycling the caches through COPY_DEST from
    // their permanent UNORDERED_ACCESS state.
    transition(s, s.dqr.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (!k_direct) {
        transition(s, s.dkr.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                   D3D12_RESOURCE_STATE_COPY_SOURCE);
    }
    transition(s, s.vcache.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_COPY_DEST);
    if (!k_direct) {
        transition(s, s.kcache.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                   D3D12_RESOURCE_STATE_COPY_DEST);
        s.list->CopyBufferRegion(s.kcache.Get(), dst, s.dkr.Get(), 0, kv_bytes);
        transition(s, s.kcache.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    s.list->CopyBufferRegion(s.vcache.Get(), dst, s.dv.Get(), 0, kv_bytes);
    transition(s, s.vcache.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (k_direct) {
        // Order rope-k's cache write (and the v copy) before attention reads them.
        uav_barrier(s, nullptr);
    }

    // 4. Attention over the whole visible window. The caches are read as UAVs (u1/u2,
    // whole buffers, always 256B-aligned); the layer slice offset rides in the cbuffer.
    transition(s, s.datt.Get(), D3D12_RESOURCE_STATE_COMMON,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    const bool a2 = use_attn2(s, pos);
    s.list->SetPipelineState(a2 ? s.pso_attn2.Get() : s.pso_attn.Get());
    ap->heads = static_cast<std::uint32_t>(heads);
    ap->kv_heads = static_cast<std::uint32_t>(kv_heads);
    ap->head_dim = static_cast<std::uint32_t>(head_dim);
    ap->context = static_cast<std::uint32_t>(s.kv_context);
    ap->first_pos = static_cast<std::uint32_t>(first_pos);
    ap->pos = static_cast<std::uint32_t>(pos);
    ap->scores_off = 0;
    ap->out_off = static_cast<std::uint32_t>(scores_floats);
    ap->inv_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    ap->pad0 = ap->pad1 = ap->pad2 = 0.0f;
    const std::uint32_t kv_base =
        static_cast<std::uint32_t>(layer * kv_heads * s.kv_context * head_dim);
    ap->kv_base = kv_base;
    ap->v_base = kv_base;
    s.list->SetComputeRootConstantBufferView(0, s.params->GetGPUVirtualAddress() +
                                                    kAttnParamOffset);
    s.list->SetComputeRootShaderResourceView(1, s.dqr->GetGPUVirtualAddress());
    s.list->SetComputeRootUnorderedAccessView(6, s.kcache->GetGPUVirtualAddress());
    s.list->SetComputeRootUnorderedAccessView(7, s.vcache->GetGPUVirtualAddress());
    s.list->SetComputeRootUnorderedAccessView(5, s.datt->GetGPUVirtualAddress());
    s.current_comp = Comp::Projections;
    // attention.hlsl: one 32-thread group per head, grid = heads.
    // attention2 (a2): two heads per 64-thread group, grid = ceil(heads/2).
    s.list->Dispatch(a2 ? static_cast<UINT>((heads + 1) / 2) : static_cast<UINT>(heads), 1, 1);
    if (kv_direct) {
        // Order this layer's cache reads before the next call's cache writes (WAR).
        uav_barrier(s, nullptr);
    }

    // 5. Read back only the attention output region.
    transition(s, s.datt.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_COPY_SOURCE);
    s.list->CopyBufferRegion(s.yread.Get(), 0, s.datt.Get(), scores_floats * sizeof(float),
                             q_bytes);
    // Return every scratch buffer to COMMON so the next call can assume that state.
    // Leaving them in UAV/SRV made the second position transition from a stale state,
    // which is an invalid barrier and removes the device.
    transition(s, s.datt.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
               D3D12_RESOURCE_STATE_COMMON);
    transition(s, s.dqr.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               D3D12_RESOURCE_STATE_COMMON);
    if (!k_direct) {
        transition(s, s.dkr.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                   D3D12_RESOURCE_STATE_COMMON);
    }
    s.list->Close();

    if (!execute_and_wait(s)) {
        std::fprintf(stderr, "attn: execute_and_wait failed (pos=%zu)\n", pos);
        return false;
    }

    void* mapped = nullptr;
    const D3D12_RANGE read_range{0, q_bytes};
    if (FAILED(s.yread->Map(0, &read_range, &mapped))) {
        std::fprintf(stderr, "attn: readback Map failed pos=%zu removed_reason=0x%08lx\n", pos,
                     static_cast<unsigned long>(s.device->GetDeviceRemovedReason()));
        return false;
    }
    std::memcpy(out, mapped, q_bytes);
    const D3D12_RANGE written{0, 0};
    s.yread->Unmap(0, &written);
    s.stats.dispatches += 3;
    kv_resident_merge(s, layer, pos, 1);
    return true;
}

// Appends one token's k and v to the device cache during prefill.
//
// Prefill computes its attention on the host, but decode reads the device cache, so the
// prompt positions have to land there too. Without this the device cache holds nothing
// but decoded positions and decode attends over a window starting at the prompt end.
// This is a pure copy - no dispatch - so it costs two CopyBufferRegion calls.
bool kv_cache_append_locked(State& s, std::size_t layer, std::size_t pos, const float* k,
                            const float* v, std::size_t kv_heads, std::size_t head_dim) {
    if (!s.kcache || layer >= s.kv_layers || pos >= s.kv_context) {
        return false;
    }
    if (s.kv_heads != kv_heads || s.kv_head_dim != head_dim) {
        return false;
    }
    const std::size_t kv_bytes = kv_heads * head_dim * sizeof(float);
    if (!reserve_scratch(s, kv_bytes * 2, kv_bytes * 2)) {
        return false;
    }
    std::memcpy(s.x_cpu, k, kv_bytes);
    std::memcpy(static_cast<std::uint8_t*>(s.x_cpu) + kv_bytes, v, kv_bytes);

    const std::size_t stride = s.kv_heads * s.kv_context * s.kv_head_dim * sizeof(float);
    const std::size_t dst = layer * stride + pos * kv_bytes;

    // Same-queue append. The KV cache is read by the attention dispatches on the
    // compute queue, so writing it from the separate copy queue would race a
    // resource that the compute queue still had bound as an SRV - the barrier
    // tracking here is per-queue, so that hazard is invisible to the runtime and
    // ends in a removed device. One queue, one ordering, no cross-queue sync.
    // The caches live in UNORDERED_ACCESS, so the COPY_DEST cycles run from there.
    s.allocator->Reset();
    s.list->Reset(s.allocator.Get(), nullptr);
    transition(s, s.kcache.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_COPY_DEST);
    transition(s, s.vcache.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_COPY_DEST);
    s.list->CopyBufferRegion(s.kcache.Get(), dst, s.xbuf.Get(), 0, kv_bytes);
    s.list->CopyBufferRegion(s.vcache.Get(), dst, s.xbuf.Get(), kv_bytes, kv_bytes);
    transition(s, s.kcache.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    transition(s, s.vcache.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    s.list->Close();

    if (!execute_and_wait(s)) {
        return false;
    }
    kv_resident_merge(s, layer, pos, 1);
    return true;
}

bool kv_cache_append(std::size_t layer, std::size_t pos, const float* k, const float* v,
                     std::size_t kv_heads, std::size_t head_dim) noexcept {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!init_locked(s)) {
        return false;
    }
    if (!kv_cache_append_locked(s, layer, pos, k, v, kv_heads, head_dim)) {
        ++s.stats.fallbacks;
        return false;
    }
    return true;
}

// Batched KV cache append for prefill: appends multiple layers' K/V in one submission.
// layers/pos/ks/vs arrays must have the same length (num_layers).
//
// Resident-prefix short circuit: when every (layer, position) the batch would write is
// already recorded resident - the requested range [first_pos, first_pos + count) for
// every layer present in the batch - the bytes are already in VRAM and the whole
// submission is skipped. This is what makes restoring a cached prompt prefix before
// decode free once it has landed in the device cache.
bool kv_cache_append_batched_locked(State& s,
                                    const std::size_t* layers, std::size_t num_layers,
                                    const std::size_t* positions,
                                    const float** ks, const float** vs,
                                    std::size_t kv_heads, std::size_t head_dim) {
    if (const char* trace = std::getenv("ZEN_D3D12_TRACE"); trace && trace[0] == '1') {
        std::printf("[d3d12] batched kv append: num_layers=%zu\n", num_layers);
    }
    if (!s.kcache || num_layers == 0) return false;
    if (s.kv_heads != kv_heads || s.kv_head_dim != head_dim) return false;
    if (!positions) return false;

    // NOTE: deliberately NO resident-prefix skip here. The resident record exists
    // only to answer kv_cache_prefix_resident for the runtime's cache-hit fast path
    // (which skips the whole re-append call). An append submitted to this function
    // is an unconditional overwrite request � prefill re-writes positions with
    // different K/V every run, and silently dropping those writes corrupts the
    // cache (stale-key attention, wrong generations). Under-reports on purpose.
    for (std::size_t i = 0; i < num_layers; ++i) {
        if (layers[i] >= s.kv_layers) return false;
        if (positions[i] >= s.kv_context) return false;
    }

    const std::size_t kv_bytes = kv_heads * head_dim * sizeof(float);
    const std::size_t total_bytes = num_layers * kv_bytes * 2;
    if (!reserve_scratch(s, total_bytes, total_bytes)) return false;

    // Pack all K/V into staging buffer
    std::uint8_t* dst = static_cast<std::uint8_t*>(s.x_cpu);
    for (std::size_t i = 0; i < num_layers; ++i) {
        std::memcpy(dst, ks[i], kv_bytes);
        dst += kv_bytes;
    }
    for (std::size_t i = 0; i < num_layers; ++i) {
        std::memcpy(dst, vs[i], kv_bytes);
        dst += kv_bytes;
    }

    const std::size_t stride = s.kv_heads * s.kv_context * s.kv_head_dim * sizeof(float);

    // Same-queue append: see kv_cache_append_locked for why the copy queue is not
    // usable here. The caches live in UNORDERED_ACCESS, so the COPY_DEST cycles run
    // from there.
    s.allocator->Reset();
    s.list->Reset(s.allocator.Get(), nullptr);
    transition(s, s.kcache.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_COPY_DEST);
    transition(s, s.vcache.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_COPY_DEST);

    std::uint8_t* src_k = static_cast<std::uint8_t*>(s.x_cpu);
    std::uint8_t* src_v = src_k + num_layers * kv_bytes;
    for (std::size_t i = 0; i < num_layers; ++i) {
        const std::size_t layer = layers[i];
        const std::size_t pos = positions[i];
        const std::size_t dst_offset = layer * s.kv_heads * s.kv_context * s.kv_head_dim * sizeof(float) + positions[i] * kv_bytes;
        s.list->CopyBufferRegion(s.kcache.Get(), dst_offset, s.xbuf.Get(), src_k - static_cast<std::uint8_t*>(s.x_cpu), kv_bytes);
        s.list->CopyBufferRegion(s.vcache.Get(), dst_offset, s.xbuf.Get(), src_v - static_cast<std::uint8_t*>(s.x_cpu), kv_bytes);
        src_k += kv_bytes;
        src_v += kv_bytes;
    }

    transition(s, s.kcache.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    transition(s, s.vcache.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    s.list->Close();

    if (!execute_and_wait(s)) {
        return false;
    }
    for (std::size_t i = 0; i < num_layers; ++i) {
        kv_resident_merge(s, layers[i], positions[i], 1);
    }
    return true;
}

bool kv_cache_append_batched(const std::size_t* layers, std::size_t num_layers,
                             const std::size_t* positions,
                             const float** ks, const float** vs,
                             std::size_t kv_heads, std::size_t head_dim) noexcept {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!init_locked(s)) return false;
    if (!kv_cache_append_batched_locked(s, layers, num_layers, positions, ks, vs, kv_heads, head_dim)) {
        ++s.stats.fallbacks;
        return false;
    }
    return true;
}

bool attention_block(const float* q, const float* k, const float* v, const float* cos,
                     const float* sin, float* out, std::size_t heads, std::size_t kv_heads,
                     std::size_t head_dim, std::size_t pairs, std::size_t layer,
                     std::size_t pos, std::size_t first_pos,
                     bool interleaved_pairs) noexcept {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!init_locked(s)) {
        return false;
    }
    if (!attn_block_locked(s, q, k, v, cos, sin, heads, kv_heads, head_dim, pairs, layer, pos,
                           first_pos, out, interleaved_pairs)) {
        ++s.stats.fallbacks;
        return false;
    }
    return true;
}

bool rms_norm_locked(State& s, const float* x, const float* weight, float* y, std::size_t rows,
                     std::size_t columns, float epsilon) {
    if (rows == 0 || columns == 0) {
        return false;
    }
    const std::size_t bytes = rows * columns * sizeof(float);
    if (!reserve_scratch(s, bytes, bytes)) {
        return false;
    }
    std::memcpy(s.x_cpu, x, bytes);

    const std::size_t weight_bytes = columns * sizeof(float);
    if (weight_bytes > s.w_cap) {
        s.wbuf.Reset();
        s.wbuf = create_buffer(s.device.Get(), weight_bytes, upload_heap_type(),
                               upload_heap_state());
        if (!s.wbuf || FAILED(s.wbuf->Map(0, nullptr, &s.w_cpu))) {
            s.w_cap = 0;
            return false;
        }
        s.w_cap = weight_bytes;
    }
    std::memcpy(s.w_cpu, weight, weight_bytes);

    auto* params = reinterpret_cast<RmsParams*>(static_cast<std::uint8_t*>(s.params_cpu) + 256);
    params->count = static_cast<std::uint32_t>(columns);
    params->rows = static_cast<std::uint32_t>(rows);
    params->epsilon = epsilon;
    params->pad = 0;

    s.allocator->Reset();
    s.list->Reset(s.allocator.Get(), s.pso_rms.Get());
    s.list->SetComputeRootSignature(s.root.Get());
    s.list->SetComputeRootConstantBufferView(0, s.params->GetGPUVirtualAddress() + 256);
    s.list->SetComputeRootShaderResourceView(1, s.xbuf->GetGPUVirtualAddress());
    s.list->SetComputeRootShaderResourceView(2, s.wbuf->GetGPUVirtualAddress());
    s.list->SetComputeRootUnorderedAccessView(5, s.ybuf->GetGPUVirtualAddress());

    // One workgroup per row, exactly as rmsnorm.hlsl assumes. Without this call the
    // kernel never ran and the readback returned the zero-initialised buffer.
    s.current_comp = Comp::Norms;
    s.list->Dispatch(static_cast<UINT>(rows), 1, 1);

    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = s.ybuf.Get();
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    s.list->ResourceBarrier(1, &barrier);
    s.list->CopyBufferRegion(s.yread.Get(), 0, s.ybuf.Get(), 0, bytes);
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    s.list->ResourceBarrier(1, &barrier);
    s.list->Close();

    if (!execute_and_wait(s)) {
        return false;
    }

    void* mapped = nullptr;
    const D3D12_RANGE read_range{0, bytes};
    if (FAILED(s.yread->Map(0, &read_range, &mapped))) {
        return false;
    }
    std::memcpy(y, mapped, bytes);
    const D3D12_RANGE written{0, 0};
    s.yread->Unmap(0, &written);
    ++s.stats.dispatches;
    return true;
}

bool rms_norm(const float* x, const float* weight, float* y, std::size_t rows,
              std::size_t columns, float epsilon) noexcept {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!init_locked(s)) {
        return false;
    }
    if (!rms_norm_locked(s, x, weight, y, rows, columns, epsilon)) {
        ++s.stats.fallbacks;
        return false;
    }
    return true;
}

bool gemv_q8_0_ffn(const std::uint8_t* gate, const std::uint8_t* up, const std::uint8_t* down,
                   const float* x, float* y, std::size_t ffn_rows, std::size_t columns,
                   std::size_t out_rows) noexcept {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!init_locked(s)) {
        return false;
    }
    if (!gemv_ffn_locked(s, gate, up, down, x, y, ffn_rows, columns, out_rows)) {
        ++s.stats.fallbacks;
        return false;
    }
    return true;
}

bool gemv_q8_0(const std::uint8_t* weights, const float* x, float* y, std::size_t rows,
               std::size_t columns) noexcept {
    if (rows * columns < kMinElements) {
        return false;
    }
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!init_locked(s)) {
        return false;
    }
    if (!gemv_locked(s, weights, x, y, rows, columns)) {
        ++s.stats.fallbacks;
        return false;
    }
    return true;
}

bool gemm_q8_0(const std::uint8_t* weights, const float* input, float* output, std::size_t tokens,
               std::size_t rows, std::size_t columns) noexcept {
    if (tokens * rows * columns < kMinElements) {
        return false;
    }
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!init_locked(s)) {
        return false;
    }
    if (!gemm_opted_in()) {
        // Prefill still runs on the CPU, but it is the right place to pay the
        // one-off residency cost: otherwise the first decoded token absorbs it.
        lookup_weights(s, weights, rows, columns);
        return false;
    }
    bool use_rt = gemm_rt_enabled() && s.pso_gemm_rt != nullptr;
    bool ok = false;
    if (use_rt) {
        ok = gemm_rt_locked(s, weights, input, output, tokens, rows, columns);
        if (!ok) {
            ok = gemm_locked(s, weights, input, output, tokens, rows, columns);
        }
    } else {
        ok = gemm_locked(s, weights, input, output, tokens, rows, columns);
    }
    if (!ok) {
        ++s.stats.fallbacks;
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Fused FFN half of a transformer layer.
//
// One command list performs: attn_output matvec, residual add, FFN RMSNorm,
// gate|up matvecs, SiLU multiply, down matvec, residual add, then one readback.
// That replaces a matvec submission, a host-side add, a host-side norm, an FFN
// submission and another host-side add.
//
// Two device buffers alternate roles so that no dispatch both reads and writes the
// same resource: a dispatch that consumes a buffer needs it in
// NON_PIXEL_SHADER_RESOURCE, while the one it writes to stays UNORDERED_ACCESS.
// A and B swap roles at every step, and each swap is an explicit full-resource
// transition between dispatches.
namespace {

bool ensure_upload(ComPtr<ID3D12Resource>& buf, void*& cpu, std::size_t& cap,
                   ID3D12Device* device, std::size_t bytes) {
    if (buf && bytes <= cap) {
        return true;
    }
    buf.Reset();
    buf = create_buffer(device, bytes, upload_heap_type(), upload_heap_state());
    if (!buf || FAILED(buf->Map(0, nullptr, &cpu))) {
        cap = 0;
        return false;
    }
    cap = bytes;
    return true;
}

}  // namespace

bool ffn_residual_fused_locked(State& s, const std::uint8_t* attn_out_w, const std::uint8_t* gate_w,
                               const std::uint8_t* up_w, const std::uint8_t* down_w,
                               const float* norm_weight, const float* x, const float* attn_out,
                               float* out, std::size_t hidden, std::size_t attn_width,
                               std::size_t ffn_rows, float epsilon) {
    if (!hidden || !attn_width || !ffn_rows) {
        return false;
    }
    if (attn_width % kElemsPerBlock || hidden % kElemsPerBlock || ffn_rows % kElemsPerBlock) {
        return false;
    }

    // Resolve every tensor first so a residency eviction cannot invalidate a buffer
    // that an already-recorded dispatch is still reading.
    ComPtr<ID3D12Resource> ao_q, g_q, u_q, d_q;
    std::uint64_t ao_s = 0, g_s = 0, u_s = 0, d_s = 0;
    if (WeightEntry* e = lookup_weights(s, attn_out_w, hidden, attn_width)) {
        ao_q = e->quants;
        ao_s = e->scales_va;
    } else {
        return false;
    }
    if (WeightEntry* e = lookup_weights(s, gate_w, ffn_rows, hidden)) {
        g_q = e->quants;
        g_s = e->scales_va;
    } else {
        return false;
    }
    if (WeightEntry* e = lookup_weights(s, up_w, ffn_rows, hidden)) {
        u_q = e->quants;
        u_s = e->scales_va;
    } else {
        return false;
    }
    if (WeightEntry* e = lookup_weights(s, down_w, hidden, ffn_rows)) {
        d_q = e->quants;
        d_s = e->scales_va;
    } else {
        return false;
    }

    constexpr std::uint64_t kAlign = 256u;
    auto span = [](std::uint64_t bytes) { return (bytes + kAlign - 1) & ~(kAlign - 1); };
    const std::uint64_t hidden_span = span(hidden * sizeof(float));
    const std::uint64_t ffn_span = span(ffn_rows * sizeof(float));

    // A (ybuf): attn projection, normed activation, SiLU result, final output.
    const std::uint64_t a_t0 = 0;
    const std::uint64_t a_h2 = hidden_span;
    const std::uint64_t a_f = a_h2 + hidden_span;
    const std::uint64_t a_fin = a_f + ffn_span;
    const std::uint64_t a_total = a_fin + hidden_span;

    // B (fbuf): residual base, gate, up, down projection.
    const std::uint64_t b_h1 = 0;
    const std::uint64_t b_g = hidden_span;
    const std::uint64_t b_u = b_g + ffn_span;
    const std::uint64_t b_d = b_u + ffn_span;
    const std::uint64_t b_total = b_d + hidden_span;

    if (!reserve_scratch(s, attn_width * sizeof(float), a_total)) {
        return false;
    }
    if (!s.fbuf || b_total > s.f_cap) {
        s.fbuf.Reset();
        s.fbuf = create_buffer(s.device.Get(), b_total, D3D12_HEAP_TYPE_DEFAULT,
                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                               D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        if (!s.fbuf) {
            s.f_cap = 0;
            return false;
        }
        s.f_cap = static_cast<std::size_t>(b_total);
    }

    const std::size_t hidden_bytes = hidden * sizeof(float);
    if (!ensure_upload(s.wbuf, s.w_cpu, s.w_cap, s.device.Get(), hidden_bytes) ||
        !ensure_upload(s.nbuf, s.n_cpu, s.n_cap, s.device.Get(), hidden_bytes)) {
        return false;
    }
    std::memcpy(s.x_cpu, attn_out, attn_width * sizeof(float));
    std::memcpy(s.w_cpu, x, hidden_bytes);
    std::memcpy(s.n_cpu, norm_weight, hidden_bytes);

    constexpr std::uint64_t kParamOffset = 512;
    const std::uint64_t addn0 = 512 + 256 * kMaxBatch;
    const std::uint64_t addn1 = addn0 + 256;
    auto* base = static_cast<std::uint8_t*>(s.params_cpu);
    auto write_gemv = [&](std::uint64_t slot, std::size_t rows, std::size_t cols) {
        auto* params = reinterpret_cast<GemvParams*>(base + slot);
        params->rows = static_cast<std::uint32_t>(rows);
        params->blocks_per_row = static_cast<std::uint32_t>(cols / kElemsPerBlock);
        params->rows_per_group = kRowsPerGroup;
        params->reserved = 0;
    };
    auto write_addn = [&](std::uint64_t slot, std::size_t count) {
        auto* params = reinterpret_cast<AddnParams*>(base + slot);
        params->count = static_cast<std::uint32_t>(count);
        params->pad0 = params->pad1 = params->pad2 = 0;
    };

    write_gemv(0, hidden, attn_width);
    write_gemv(kParamOffset, ffn_rows, hidden);
    write_gemv(kParamOffset + 256, ffn_rows, hidden);
    write_gemv(kParamOffset + 768, hidden, ffn_rows);
    write_addn(addn0, hidden);
    write_addn(addn1, hidden);
    auto* rms = reinterpret_cast<RmsParams*>(base + 256);
    rms->count = static_cast<std::uint32_t>(hidden);
    rms->rows = 1;
    rms->epsilon = epsilon;
    rms->pad = 0;
    auto* silu = reinterpret_cast<SiluParams*>(base + kParamOffset + 512);
    silu->count = static_cast<std::uint32_t>(ffn_rows);
    silu->pad0 = silu->pad1 = silu->pad2 = 0;

    const std::uint64_t params_base = s.params->GetGPUVirtualAddress();
    const std::uint64_t a_base = s.ybuf->GetGPUVirtualAddress();
    const std::uint64_t b_base = s.fbuf->GetGPUVirtualAddress();

    const UINT attn_groups = static_cast<UINT>((hidden + kRowsPerGroup - 1) / kRowsPerGroup);
    const UINT ffn_groups = static_cast<UINT>((ffn_rows + kRowsPerGroup - 1) / kRowsPerGroup);
    const UINT down_groups = static_cast<UINT>((hidden + kRowsPerGroup - 1) / kRowsPerGroup);
    const UINT silu_groups = static_cast<UINT>((ffn_rows + 255) / 256);

    s.allocator->Reset();
    s.list->Reset(s.allocator.Get(), s.pso_gemv.Get());
    s.list->SetComputeRootSignature(s.root.Get());
    // Stamp 0 brackets the whole 8-dispatch run so the trace can tell real device
    // work from fence latency; stamp 1 closes before the readback copy.
    if (s.trace) {
        s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
    }

    // 1. attn_output matvec: A.t0 = W @ att
    s.list->SetComputeRootConstantBufferView(0, params_base);
    s.list->SetComputeRootShaderResourceView(1, ao_q->GetGPUVirtualAddress());
    s.list->SetComputeRootShaderResourceView(2, ao_s);
    s.list->SetComputeRootShaderResourceView(3, s.xbuf->GetGPUVirtualAddress());
    s.list->SetComputeRootUnorderedAccessView(5, a_base + a_t0);
    s.list->Dispatch(attn_groups, 1, 1);

    // 2. residual: B.h1 = A.t0 + x  (addn reads A.t0, so A becomes an SRV)
    transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    s.list->SetPipelineState(s.pso_addn.Get());
    s.list->SetComputeRootConstantBufferView(0, params_base + addn0);
    s.list->SetComputeRootShaderResourceView(1, a_base + a_t0);
    s.list->SetComputeRootShaderResourceView(2, s.wbuf->GetGPUVirtualAddress());
    s.list->SetComputeRootUnorderedAccessView(5, b_base + b_h1);
    s.list->Dispatch(static_cast<UINT>((hidden + 255) / 256), 1, 1);

    // 3. FFN norm: A.h2 = rmsnorm(B.h1) * w  (B becomes the SRV, A the writer)
    transition(s, s.fbuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    s.list->SetPipelineState(s.pso_rms.Get());
    s.list->SetComputeRootConstantBufferView(0, params_base + 256);
    s.list->SetComputeRootShaderResourceView(1, b_base + b_h1);
    s.list->SetComputeRootShaderResourceView(2, s.nbuf->GetGPUVirtualAddress());
    s.list->SetComputeRootUnorderedAccessView(5, a_base + a_h2);
    s.list->Dispatch(1, 1, 1);

    // 4/5. gate and up matvecs from the normed activation into B.
    transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(s, s.fbuf.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    s.list->SetPipelineState(s.pso_gemv.Get());
    s.list->SetComputeRootConstantBufferView(0, params_base + kParamOffset);
    s.list->SetComputeRootShaderResourceView(1, g_q->GetGPUVirtualAddress());
    s.list->SetComputeRootShaderResourceView(2, g_s);
    s.list->SetComputeRootShaderResourceView(3, a_base + a_h2);
    s.list->SetComputeRootUnorderedAccessView(5, b_base + b_g);
    s.list->Dispatch(ffn_groups, 1, 1);

    s.list->SetComputeRootConstantBufferView(0, params_base + kParamOffset + 256);
    s.list->SetComputeRootShaderResourceView(1, u_q->GetGPUVirtualAddress());
    s.list->SetComputeRootShaderResourceView(2, u_s);
    s.list->SetComputeRootUnorderedAccessView(5, b_base + b_u);
    s.list->Dispatch(ffn_groups, 1, 1);

    // 6. SiLU(gate)*up into A.f  (B becomes the SRV again)
    transition(s, s.fbuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    s.list->SetPipelineState(s.pso_silu.Get());
    s.list->SetComputeRootConstantBufferView(0, params_base + kParamOffset + 512);
    s.list->SetComputeRootShaderResourceView(1, b_base + b_g);
    s.list->SetComputeRootShaderResourceView(2, b_base + b_u);
    s.list->SetComputeRootUnorderedAccessView(5, a_base + a_f);
    s.list->Dispatch(silu_groups, 1, 1);

    // 7. down matvec: B.d = W_down @ A.f
    transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(s, s.fbuf.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    s.list->SetPipelineState(s.pso_gemv.Get());
    s.list->SetComputeRootConstantBufferView(0, params_base + kParamOffset + 768);
    s.list->SetComputeRootShaderResourceView(1, d_q->GetGPUVirtualAddress());
    s.list->SetComputeRootShaderResourceView(2, d_s);
    s.list->SetComputeRootShaderResourceView(3, a_base + a_f);
    s.list->SetComputeRootUnorderedAccessView(5, b_base + b_d);
    s.list->Dispatch(down_groups, 1, 1);

    // 8. final residual: A.fin = B.h1 + B.d
    transition(s, s.fbuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    s.list->SetPipelineState(s.pso_addn.Get());
    s.list->SetComputeRootConstantBufferView(0, params_base + addn1);
    s.current_comp = Comp::FfnFused;
    s.list->SetComputeRootShaderResourceView(1, b_base + b_d);
    s.list->SetComputeRootShaderResourceView(2, b_base + b_h1);
    s.list->SetComputeRootUnorderedAccessView(5, a_base + a_fin);
    s.list->Dispatch(static_cast<UINT>((hidden + 255) / 256), 1, 1);

    if (s.trace) {
        s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
    }
    transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_COPY_SOURCE);
    s.list->CopyBufferRegion(s.yread.Get(), 0, s.ybuf.Get(), a_fin, hidden_bytes);
    transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (s.trace) {
        s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 2);
        s.list->ResolveQueryData(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 3,
                                 s.ts_read.Get(), 0);
    }
    s.list->Close();

    if (!execute_and_wait(s)) {
        return false;
    }

    void* mapped = nullptr;
    const D3D12_RANGE read_range{0, hidden_bytes};
    if (FAILED(s.yread->Map(0, &read_range, &mapped))) {
        return false;
    }
    {
        std::uint64_t dev = 0;
        accumulate_gpu_us(s, dev, 0, 1);
        note_dev(s, Comp::FfnFused, dev);
    }
    std::memcpy(out, mapped, hidden_bytes);
    const D3D12_RANGE written{0, 0};
    s.yread->Unmap(0, &written);
    s.stats.dispatches += 8;
    ++s.stats.batches;
    return true;
}

// ---------------------------------------------------------------------------
// Pipelined multi-layer decode: records all layers into command lists, submits
// them as a batch, then waits once. Keeps intermediate activations in VRAM.
// GPU-to-GPU for intermediate layers, only final output read back to host.
// Returns false on any failure (caller falls back to per-layer path).
bool layer_decode_pipeline(const LayerDecodeParams* layers, std::size_t num_layers,
                           const float* input, float* output,
                           std::size_t hidden, std::size_t heads, std::size_t kv_heads,
                           std::size_t head_dim, std::size_t ffn_rows, float epsilon) noexcept {
    return layer_decode_pipeline(layers, num_layers, input, output, hidden, heads, kv_heads,
                                 head_dim, ffn_rows, epsilon, nullptr);
}

bool layer_decode_pipeline(const LayerDecodeParams* layers, std::size_t num_layers,
                           const float* input, float* output,
                           std::size_t hidden, std::size_t heads, std::size_t kv_heads,
                           std::size_t head_dim, std::size_t ffn_rows, float epsilon,
                           const LmHeadFuse* lm_head) noexcept {
    if (num_layers == 0 || num_layers > 64) return false;
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!init_locked(s)) return false;

    const std::size_t q_width = heads * head_dim;
    const std::size_t kv_width = kv_heads * head_dim;
    if (!hidden || !q_width || !kv_width || !ffn_rows) return false;
    if (hidden % kElemsPerBlock || ffn_rows % kElemsPerBlock || q_width % kElemsPerBlock ||
        kv_width % kElemsPerBlock) return false;
    if (!s.kcache || layers[0].layer >= s.kv_layers || layers[0].pos >= s.kv_context ||
        s.kv_heads != kv_heads || s.kv_head_dim != head_dim) {
        if (const char* trace = std::getenv("ZEN_D3D12_TRACE"); trace && trace[0] == '1') {
            std::printf("[d3d12] pipeline: early fail - kcache=%p layer0.layer=%zu kv_layers=%zu pos=%zu kv_ctx=%zu kv_heads=%zu/%zu head_dim=%zu/%zu\n",
                (void*)s.kcache.Get(), layers[0].layer, s.kv_layers, layers[0].pos, s.kv_context,
                s.kv_heads, kv_heads, s.kv_head_dim, head_dim);
        }
        return false;
    }

    // Verify all layers have consistent parameters
    for (std::size_t i = 1; i < num_layers; ++i) {
        if (layers[i].layer != layers[0].layer + i ||
            layers[i].pos != layers[0].pos ||
            layers[i].first_pos != layers[0].first_pos) {
            if (const char* trace = std::getenv("ZEN_D3D12_TRACE"); trace && trace[0] == '1') {
                std::printf("[d3d12] pipeline: param mismatch at layer %zu: layer=%zu pos=%zu first_pos=%zu vs layer0 layer=%zu pos=%zu first_pos=%zu\n",
                    i, layers[i].layer, layers[i].pos, layers[i].first_pos,
                    layers[0].layer, layers[0].pos, layers[0].first_pos);
            }
            return false;
        }
    }

    if (const char* trace = std::getenv("ZEN_D3D12_TRACE"); trace && trace[0] == '1') {
        std::printf("[d3d12] pipeline: START num_layers=%zu hidden=%zu heads=%zu kv_heads=%zu head_dim=%zu ffn_rows=%zu\n",
            num_layers, hidden, heads, kv_heads, head_dim, ffn_rows);
    }

    // Resolve all weights upfront (prevents residency eviction during recording)
    const auto bind_start = std::chrono::steady_clock::now();
    struct LayerWeights {
        ComPtr<ID3D12Resource> qq, kq, vq, oq, gq, uq, dq_;
        std::uint64_t qs=0, ks=0, vs=0, os_=0, gs=0, us=0, ds=0;
    };
    std::vector<LayerWeights> lw(num_layers);
    auto bind = [&](const std::uint8_t* w, std::size_t rows, std::size_t cols,
                    ComPtr<ID3D12Resource>& q, std::uint64_t& sc) {
        WeightEntry* e = lookup_weights(s, w, rows, cols);
        if (!e) return false;
        q = e->quants;
        sc = e->scales_va;
        return true;
    };
    for (std::size_t li = 0; li < num_layers; ++li) {
        const auto& p = layers[li];
        if (!bind(p.q_w, q_width, hidden, lw[li].qq, lw[li].qs) ||
            !bind(p.k_w, kv_width, hidden, lw[li].kq, lw[li].ks) ||
            !bind(p.v_w, kv_width, hidden, lw[li].vq, lw[li].vs) ||
            !bind(p.o_w, hidden, q_width, lw[li].oq, lw[li].os_) ||
            !bind(p.gate_w, ffn_rows, hidden, lw[li].gq, lw[li].gs) ||
            !bind(p.up_w, ffn_rows, hidden, lw[li].uq, lw[li].us) ||
            !bind(p.down_w, hidden, ffn_rows, lw[li].dq_, lw[li].ds)) {
            if (const char* trace = std::getenv("ZEN_D3D12_TRACE"); trace && trace[0] == '1') {
                std::printf("[d3d12] pipeline: bind failed at layer %zu\n", li);
            }
            return false;
        }
    }

    constexpr std::uint64_t kAlign = 256u;
    auto span = [](std::uint64_t bytes) { return (bytes + kAlign - 1) & ~(kAlign - 1); };
    const std::uint64_t hidden_span = span(hidden * sizeof(float));
    const std::uint64_t ffn_span = span(ffn_rows * sizeof(float));
    const std::uint64_t q_span = span(q_width * sizeof(float));
    const std::uint64_t kv_span = span(kv_width * sizeof(float));
    const std::size_t hidden_bytes = hidden * sizeof(float);
    const std::size_t kv_bytes = kv_width * sizeof(float);
    const std::size_t tab_bytes = layers[0].pairs * sizeof(float);
    // Each layer gets its own table slice. The stride is 256B-aligned because a
    // root SRV's buffer address must be, and because odd multiples of the raw
    // stride (= pairs*4, only 128B here) would misalign every second layer.
    const std::size_t tab_stride = (tab_bytes + 255u) & ~static_cast<std::size_t>(255u);
    const std::size_t tables_bytes = num_layers * tab_stride;
    const auto bind_done = std::chrono::steady_clock::now();
    const std::size_t scores_floats = heads * s.kv_context;
    const std::size_t att_floats = heads * head_dim;
    const std::size_t datt_bytes = (scores_floats + att_floats) * sizeof(float);

    // --- Fused LM head (ZEN_D3D12_LMFUSE=0 restores the split head) -------------
    // When active, the final rmsnorm, the vocab GEMV and the logits readback are
    // recorded into this same command list after the last layer's addn2, so the token
    // costs one submit+wait instead of two. lm->fused reports what actually happened;
    // every failure below degrades to the caller's own head, never to a wrong answer.
    LmHeadFuse* lm = const_cast<LmHeadFuse*>(lm_head);
    bool lm_active = lm != nullptr && lm_fuse_enabled() && lm->output_norm_w != nullptr &&
                     lm->lm_head_w != nullptr && lm->logits != nullptr && lm->vocab_rows > 0;
    WeightEntry* lm_entry = nullptr;
    std::uint64_t vocab_span = 0;
    std::uint64_t logits_off = 0;
    if (lm_active) {
        // The GEMV dispatches one group per kRowsPerGroup rows; D3D12 caps the group
        // count at 65535. 128256 vocab rows -> 16032 groups, comfortably inside.
        if (lm->vocab_rows / kRowsPerGroup + 1 > 65535) {
            lm_active = false;
        }
    }
    if (lm_active) {
        // Resolve (and upload on first use) the head weights before anything is
        // recorded, exactly like the layer weights above - a residency eviction during
        // recording must not be able to free a bound buffer.
        lm_entry = lookup_weights(s, lm->lm_head_w, lm->vocab_rows, hidden);
        if (!lm_entry) {
            lm_active = false;
        }
    }
    if (lm_active) {
        vocab_span = span(lm->vocab_rows * sizeof(float));
    }
    if (lm) {
        lm->fused = false;
    }

    // Arena A: h, qr, kr, t0, h2, f, fin
    const std::uint64_t a_h = 0;
    const std::uint64_t a_qr = a_h + hidden_span;
    const std::uint64_t a_kr = a_qr + q_span;
    const std::uint64_t a_t0 = a_kr + kv_span;
    const std::uint64_t a_h2 = a_t0 + hidden_span;
    const std::uint64_t a_f = a_h2 + hidden_span;
    const std::uint64_t a_fin = a_f + ffn_span;
    const std::uint64_t a_total = a_fin + hidden_span;

    // Arena B: q, k, v, g, u, d. The DP4A staging buffer (s.qact) is a separate
    // resource, created below.
    const std::uint64_t b_q = 0;
    const std::uint64_t b_k = b_q + q_span;
    const std::uint64_t b_v = b_k + kv_span;
    const std::uint64_t b_g = b_v + kv_span;
    const std::uint64_t b_u = b_g + ffn_span;
    const std::uint64_t b_d = b_u + ffn_span;
    const std::uint64_t b_total = b_d + hidden_span;
    // DP4A staging layout inside s.qact: [int8 acts (1B/elem) | fp32 scales].
    const std::uint64_t qact_max_cols =
        std::max<std::uint64_t>({hidden, ffn_rows, q_width, kv_width});
    const std::uint64_t qact_acts_span = span(qact_max_cols);
    const std::uint64_t qact_scales_span =
        span((qact_max_cols / kElemsPerBlock) * sizeof(float));
    const std::uint64_t qact_total = qact_acts_span + qact_scales_span;

    // Logits from the fused LM head land right behind arena A in ybuf (and at the same
    // offset in the readback heap).
    if (lm_active) {
        logits_off = a_total;
    }

    // Arena C: h1 (residual base)
    if (!s.hbuf || hidden_span > s.h_cap) {
        s.hbuf.Reset();
        s.hbuf = create_buffer(s.device.Get(), hidden_span, D3D12_HEAP_TYPE_DEFAULT,
                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                               D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        if (!s.hbuf) { s.h_cap = 0; return false; }
        s.h_cap = static_cast<std::size_t>(hidden_span);
    }
    if (!s.fbuf || b_total > s.f_cap) {
        s.fbuf.Reset();
        s.fbuf = create_buffer(s.device.Get(), b_total, D3D12_HEAP_TYPE_DEFAULT,
                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                               D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        if (!s.fbuf) { s.f_cap = 0; return false; }
        s.f_cap = static_cast<std::size_t>(b_total);
    }
    if (!s.qact || qact_total > s.qact_cap) {
        s.qact.Reset();
        s.qact = create_buffer(s.device.Get(), qact_total, D3D12_HEAP_TYPE_DEFAULT,
                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                               D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        if (!s.qact) { s.qact_cap = 0; return false; }
        s.qact_cap = static_cast<std::size_t>(qact_total);
    }
    if (!reserve_scratch(s, hidden_bytes, a_total + vocab_span)) return false;
    // One table slice per layer: the record loop writes and per-layer binds
    // tables at li * tab_stride, so the buffers must cover all of them.
    if (!attn_upload(s, s.dcos, s.dcos_cpu, s.dcos_cap, tables_bytes) ||
        !attn_upload(s, s.dsin, s.dsin_cpu, s.dsin_cap, tables_bytes) ||
        !attn_device(s, s.datt, s.datt_cap, datt_bytes)) {
        return false;
    }
    if (!ensure_upload(s.wbuf, s.w_cpu, s.w_cap, s.device.Get(), hidden_bytes)) {
        return false;
    }
    // All 2*num_layers norm rows in one upload table, each at its own offset. The
    // single shared slot (nbuf/wbuf) was memcpy'd once per layer at record time,
    // so by execution time it held the last layer's rows for every layer.
    // Distinct per-layer offsets are safe because no write aliases another.
    // The fused LM head appends the final output_norm row after them.
    const std::size_t nrm_bytes = (2 * num_layers + (lm_active ? 1u : 0u)) * hidden_bytes;
    if (!ensure_upload(s.nrm, s.nrm_cpu, s.nrm_cap, s.device.Get(), nrm_bytes)) {
        return false;
    }
    for (std::size_t li = 0; li < num_layers; ++li) {
        std::memcpy(static_cast<std::uint8_t*>(s.nrm_cpu) + (2 * li) * hidden_bytes,
                    layers[li].attn_norm_w, hidden_bytes);
        std::memcpy(static_cast<std::uint8_t*>(s.nrm_cpu) + (2 * li + 1) * hidden_bytes,
                    layers[li].ffn_norm_w, hidden_bytes);
    }
    if (lm_active) {
        std::memcpy(static_cast<std::uint8_t*>(s.nrm_cpu) + (2 * num_layers) * hidden_bytes,
                    lm->output_norm_w, hidden_bytes);
    }
    // Default-heap staging for the per-layer input copy. The previous version
    // copied each layer's output into s.xbuf, an upload-heap resource, in COPY_DEST
    // state: invalid API usage (upload heaps are never copy destinations) that
    // removed the device with INVALID_CALL on the first pipeline submission. Every
    // layer now reads its input and its residual from s.hin, so the bindings are
    // uniform and layering needs no per-layer state juggling of s.xbuf. Created in
    // NON_PIXEL_SHADER_RESOURCE so each layer's NSR -> COPY_DEST -> NSR cycle is
    // uniform and the call ends in that same state.
    if (!s.hin || hidden_span > s.hin_cap) {
        s.hin.Reset();
        s.hin = create_buffer(s.device.Get(), hidden_span, D3D12_HEAP_TYPE_DEFAULT,
                              D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                              D3D12_RESOURCE_FLAG_NONE);
        if (!s.hin) { s.hin_cap = 0; return false; }
        s.hin_cap = static_cast<std::size_t>(hidden_span);
    }

    // Upload first layer's input
    std::memcpy(s.x_cpu, input, hidden_bytes);

    // Upload the per-layer cos/sin tables (each at its 256B-aligned slot).
    for (std::size_t li = 0; li < num_layers; ++li) {
        std::memcpy(static_cast<std::uint8_t*>(s.dcos_cpu) + li * tab_stride,
                    layers[li].cos, tab_bytes);
        std::memcpy(static_cast<std::uint8_t*>(s.dsin_cpu) + li * tab_stride,
                    layers[li].sin, tab_bytes);
    }
    // We'll upload norm weights per-layer during recording

    // Stage-1 fused dispatches (ZEN_D3D12_FUSE=1): concatenate q|k|v and gate|up
    // into single multi-row tables, cutting two dispatches per layer and enriching
    // each GEMV's row span. The arena already lays q/k/v and g/u contiguous with
    // no padding, so the fused output writes need no layout change either.
    const bool fuse = [] {
        const char* f = std::getenv("ZEN_D3D12_FUSE");
        return f && f[0] == '1';
    }();
    // KV direct-write (default ON): the rope-k dispatch and the v matvec write the
    // current token's cache rows straight from compute, and attention reads the caches
    // as UAVs, so the per-layer copies and all four cache transitions disappear.
    // ZEN_D3D12_KVDIRECT=0 restores the CopyBufferRegion append path.
    const bool kv_direct = kv_direct_enabled();
    std::vector<FusedMat*> fqkv(num_layers, nullptr), fgu(num_layers, nullptr);
    if (fuse) {
        for (std::size_t li = 0; li < num_layers; ++li) {
            fqkv[li] = ensure_fused_mat(s, layers[li].q_w, hidden,
                                        {{layers[li].q_w, q_width},
                                         {layers[li].k_w, kv_width},
                                         {layers[li].v_w, kv_width}});
            fgu[li] = ensure_fused_mat(s, layers[li].gate_w, hidden,
                                       {{layers[li].gate_w, ffn_rows},
                                        {layers[li].up_w, ffn_rows}});
        }
        if (const char* t = std::getenv("ZEN_D3D12_TRACE"); t && t[0] == '1') {
            std::printf("[d3d12] pipeline: fused qkv=%s gu=%s (%zu layers)\n",
                        fqkv[0] ? "ready" : "FAIL", fgu[0] ? "ready" : "FAIL", num_layers);
        }
    }
    // The fused and unfused shapes write DIFFERENT `rows` values into the same shared
    // gemv parameter slots, and the GPU reads those slots at execution time - after
    // every layer's record-time write. A mixed fused/unfused batch would therefore run
    // every dispatch with the last layer's row count, so fusion is all-or-nothing for
    // the whole pipeline call: one layer failing to build its fused table drops the
    // whole batch to the separate per-tensor matvecs.
    const bool fuse_qkv_all =
        std::all_of(fqkv.begin(), fqkv.end(), [](const FusedMat* m) { return m != nullptr; });
    const bool fuse_gu_all =
        std::all_of(fgu.begin(), fgu.end(), [](const FusedMat* m) { return m != nullptr; });

    // Parameter slots for each layer
    auto write_gemv = [&](std::uint64_t slot, std::size_t rows, std::size_t cols) {
        auto* p = reinterpret_cast<GemvParams*>(static_cast<std::uint8_t*>(s.params_cpu) + slot);
        p->rows = static_cast<std::uint32_t>(rows);
        p->blocks_per_row = static_cast<std::uint32_t>(cols / kElemsPerBlock);
        p->rows_per_group = kRowsPerGroup;
        p->reserved = 0;
    };
    auto write_rms = [&](std::uint64_t slot) {
        auto* p = reinterpret_cast<RmsParams*>(static_cast<std::uint8_t*>(s.params_cpu) + slot);
        p->count = static_cast<std::uint32_t>(hidden);
        p->rows = 1;
        p->epsilon = epsilon;
        p->pad = 0;
    };
    auto write_addn = [&](std::uint64_t slot) {
        auto* p = reinterpret_cast<AddnParams*>(static_cast<std::uint8_t*>(s.params_cpu) + slot);
        p->count = static_cast<std::uint32_t>(hidden);
        p->pad0 = p->pad1 = p->pad2 = 0;
    };
    auto write_silu = [&](std::uint64_t slot) {
        auto* p = reinterpret_cast<SiluParams*>(static_cast<std::uint8_t*>(s.params_cpu) + slot);
        p->count = static_cast<std::uint32_t>(ffn_rows);
        p->pad0 = p->pad1 = p->pad2 = 0;
    };
    auto write_rope = [&](std::uint64_t slot, std::size_t h, std::size_t hd, std::size_t pr,
                          bool adjacent) {
        auto* p = reinterpret_cast<RopeParams*>(static_cast<std::uint8_t*>(s.params_cpu) + slot);
        p->heads = static_cast<std::uint32_t>(h);
        p->head_dim = static_cast<std::uint32_t>(hd);
        p->pairs = static_cast<std::uint32_t>(pr);
        // pad0 = pairing layout: 0 NeoX half-split, 1 llama.cpp NORM adjacent.
        p->pad0 = adjacent ? 1u : 0u;
    };
    auto write_attn = [&](std::uint64_t slot, std::size_t h, std::size_t kh, std::size_t hd,
                          std::size_t ctx, std::size_t fp, std::size_t pos, float inv_sc,
                          std::size_t sc_off, std::size_t out_off, std::size_t kv_base) {
        auto* ap = reinterpret_cast<AttnParams*>(static_cast<std::uint8_t*>(s.params_cpu) + slot);
        ap->heads = static_cast<std::uint32_t>(h);
        ap->kv_heads = static_cast<std::uint32_t>(kh);
        ap->head_dim = static_cast<std::uint32_t>(hd);
        ap->context = static_cast<std::uint32_t>(ctx);
        ap->first_pos = static_cast<std::uint32_t>(fp);
        ap->pos = static_cast<std::uint32_t>(pos);
        ap->scores_off = static_cast<std::uint32_t>(sc_off);
        ap->out_off = static_cast<std::uint32_t>(out_off);
        ap->inv_scale = inv_sc;
        ap->pad0 = ap->pad1 = ap->pad2 = 0.0f;
        ap->kv_base = static_cast<std::uint32_t>(kv_base);
        ap->v_base = static_cast<std::uint32_t>(kv_base);
    };

    // Record every layer into one command list, then submit once.
    const auto record_start = std::chrono::steady_clock::now();
    if (FAILED(s.allocator->Reset())) return false;
    if (FAILED(s.list->Reset(s.allocator.Get(), nullptr))) return false;

    // TRACE-mode device + per-layer brackets. Layout: 0 global start; layers
    // l=0..15 stamp 1+2l (start) and 2+2l (post-attention); 33 global end
    // (after the last readback). 34 of 64 slots. Production skips this: the
    // per-token resolve/map costs a small extra sync the hot path should not pay.
    const bool trace_pipe = [] {
        const char* t = std::getenv("ZEN_D3D12_TRACE");
        return t && t[0] == '1';
    }();
    const bool layer_stamps = [] {
        const char* t = std::getenv("ZEN_D3D12_LAYERSTAMPS");
        return t && t[0] == '1';
    }();
    // TRACE-mode per-op stamps for layer 0: one EndQuery before each of the 17
    // op boundaries, so the attn-half cost can be attributed to individual ops
    // instead of the whole-half buckets.
    const bool op_stamps = [] {
        const char* t = std::getenv("ZEN_D3D12_OPSTAMPS");
        return t && t[0] == '1';
    }();
    if (trace_pipe && s.ts_heap && s.ts_read && num_layers <= 24) {
        s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
    }
    const UINT op_stamp_first = 41; // layer-0 op stamps live at 41..57

    for (std::size_t li = 0; li < num_layers; ++li) {
        const auto& p = layers[li];
        const auto& w = lw[li];
        const std::size_t layer_idx = p.layer;
        const std::size_t pos = p.pos;
        const std::size_t first_pos = p.first_pos;
        const std::size_t pairs = p.pairs;
        const std::size_t stride = s.kv_heads * s.kv_context * s.kv_head_dim * sizeof(float);
        const std::size_t dst = layer_idx * stride + pos * kv_bytes;
        // Direct append binding offset: root UAV addresses must be 256B aligned. Real
        // model shapes always are (kv row = kv_heads*head_dim*4 >= 256B); tiny
        // test shapes may not be, and those keep the copy path.
        const bool row_aligned = (dst % 256 == 0);
        const bool k_direct = kv_direct && row_aligned;
        const std::uint32_t kv_base =
            static_cast<std::uint32_t>(layer_idx * kv_heads * s.kv_context * head_dim);
        const float inv_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        // All layers share one command list. Recording them into separate lists
        // put the resource transitions on a different (already closed) list, so
        // the submitted work ran with undefined resource states and its output
        // depended on leftover GPU state. One list keeps every barrier with the
        // work it orders, and still costs a single submission.
        ID3D12GraphicsCommandList* list = s.list.Get();

        // Per-layer parameter offsets (use same base, different slots)
        const std::uint64_t kGemvBase = 4096;
        const std::uint64_t kRms1 = 5888;
        const std::uint64_t kRms2 = 6144;
        const std::uint64_t kAdd1 = 6400;
        const std::uint64_t kAdd2 = 6656;
        const std::uint64_t kSilu = 6912;
        const std::uint64_t kRopeParamOffset = 1024;
        const std::uint64_t kRopeParamOffset2 = 1536;
        // Fused LM-head param slots, past every per-layer slot in the param buffer.
        // Used by the per-layer write loop below and the LM-head recording after it.
        const std::uint64_t kGemvLm = 7168;
        const std::uint64_t kRmsLm = 7424;
        // The attention cbuffer is the ONLY per-layer-varying parameter set in this
        // loop (kv_base/v_base). Every other slot carries layer-invariant values, so
        // the record-time CPU overwrites are idempotent; a shared attention slot is
        // not, because the GPU reads upload-heap CBVs at execution time - every layer
        // would attend to the LAST layer's cache slice and decode emits garbage from
        // the first token. Each layer therefore records and binds its own 256B-aligned
        // slot past the legacy region (the file-scope kAttnParamOffset at 1280 stays
        // for the one-layer-per-submission paths, which cannot alias).
        const std::uint64_t kAttnParamSlot = 8192 + li * 256;

        // The fused q|k|v table writes one contiguous output, so it cannot target the
        // vcache row directly; KVDIRECT takes the three separate matvecs (two extra
        // tiny dispatches against a ~190us/layer copy-visibility win).
        const bool use_fqkv = fuse_qkv_all && !kv_direct;
        if (use_fqkv) {
            write_gemv(kGemvBase + 0, q_width + 2 * kv_width, hidden);
        } else {
            write_gemv(kGemvBase + 0, q_width, hidden);
            write_gemv(kGemvBase + 256, kv_width, hidden);
            write_gemv(kGemvBase + 512, kv_width, hidden);
        }
        write_gemv(kGemvBase + 768, hidden, q_width);
        if (fuse_gu_all) {
            write_gemv(kGemvBase + 1024, 2 * ffn_rows, hidden);
        } else {
            write_gemv(kGemvBase + 1024, ffn_rows, hidden);
            write_gemv(kGemvBase + 1280, ffn_rows, hidden);
        }
        write_gemv(kGemvBase + 1536, hidden, ffn_rows);
        write_rms(kRms1);
        write_rms(kRms2);
        write_addn(kAdd1);
        write_addn(kAdd2);
        write_silu(kSilu);
        // The rope cbuffers (1024/1536) are shared across layers and are only safe
        // because heads/head_dim/pairs/pad0 are model-constant - identical for every
        // layer, so the record-time overwrites are idempotent. If a model ever varies
        // the rotary config per layer, these slots need the same per-layer treatment
        // as the attention cbuffer below.
        write_rope(kRopeParamOffset, heads, head_dim, pairs, layers[li].rope_pairs_adjacent != 0);
        write_rope(kRopeParamOffset2, kv_heads, head_dim, pairs, layers[li].rope_pairs_adjacent != 0);
        write_attn(kAttnParamSlot, heads, kv_heads, head_dim, s.kv_context, first_pos, pos, inv_scale, 0, scores_floats, kv_base);
        if (lm_active) {
            write_gemv(kGemvLm, lm->vocab_rows, hidden);
            auto* lmrp = reinterpret_cast<RmsParams*>(static_cast<std::uint8_t*>(s.params_cpu) + kRmsLm);
            lmrp->count = static_cast<std::uint32_t>(hidden);
            lmrp->rows = 1;
            lmrp->epsilon = lm->epsilon;
            lmrp->pad = 0;
        }

        // Copy this layer's input (host-provided x for layer 0, the previous
        // layer's residual output otherwise) into the default-heap staging. For
        // layer 0 the source is an upload-heap buffer: buffers in COMMON/GENERIC_READ
        // may be read by copy commands without transitions. From layer 1 on the
        // source is s.ybuf's a_fin region, which sits in UAV after the previous
        // layer's residual add; a pair of transitions makes it a COPY_SOURCE.
        if (trace_pipe && s.ts_heap && s.ts_read && num_layers <= 24) {
            s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1 + 2 * li);
        }
        transition_on(list, s.hin.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                      D3D12_RESOURCE_STATE_COPY_DEST);
        if (li == 0) {
            list->CopyBufferRegion(s.hin.Get(), 0, s.xbuf.Get(), 0, hidden_bytes);
        } else {
            transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                       D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->CopyBufferRegion(s.hin.Get(), 0, s.ybuf.Get(), a_fin, hidden_bytes);
            transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        transition_on(list, s.hin.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                      D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        const std::uint64_t pb = s.params->GetGPUVirtualAddress();
        const std::uint64_t a_base = s.ybuf->GetGPUVirtualAddress();
        const std::uint64_t b_base = s.fbuf->GetGPUVirtualAddress();
        const std::uint64_t c_base = s.hbuf->GetGPUVirtualAddress();
        const std::uint64_t hin_gva = s.hin->GetGPUVirtualAddress();
        const std::uint64_t nrm_gva = s.nrm->GetGPUVirtualAddress() + (2 * li) * hidden_bytes;
        const std::uint64_t nrm2_gva = s.nrm->GetGPUVirtualAddress() + (2 * li + 1) * hidden_bytes;

        auto groups = [](std::size_t rows) { return static_cast<UINT>((rows + kRowsPerGroup - 1) / kRowsPerGroup); };

        // Layer-0 op boundary stamping: the previous stamp is each op's start,
        // the next stamp its end, so consecutive deltas are per-op GPU times.
        auto stamp_op = [&](UINT delta) {
            if (op_stamps && li == 0 && s.ts_heap && s.ts_read) {
                s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
                                 op_stamp_first + delta);
            }
        };

        const bool use_dp4a = dp4a_enabled() && s.pso_gemv_dp4a != nullptr &&
                              s.pso_quantize_acts != nullptr;
        // Producer-fused quantize (ZEN_D3D12_DP4A_FUSEQ=1, needs use_dp4a):
        // the producer dispatches themselves write the qact int8/scale
        // streams, so the 4 standalone quantize dispatches per layer (QKV,
        // O, GU, Down) are skipped. The UAV barrier ordering producer
        // writes before the DP4A GEMV reads is kept at every site. The
        // attention site additionally needs head_dim % 32 == 0 so its
        // 32-blocks align with heads; otherwise only that site falls back
        // to the separate quantize dispatch.
        const bool fuseq_pso = dp4a_fuseq_enabled() && s.pso_rms_fuse != nullptr &&
                               s.pso_silu_fuse != nullptr && s.pso_attn_fuse != nullptr;
        const bool use_fuseq_rms = use_dp4a && fuseq_pso;
        const bool use_fuseq_attn = use_fuseq_rms && (head_dim % kElemsPerBlock) == 0;
        const bool use_fuseq_silu = use_fuseq_rms;
        // DP4A decode: quantise the fp32 input vector into the int8 staging
        // buffer + scale table first (per 32-element block, mirroring
        // quantise_acts_q8_0), then the int8 GEMV consumes both. Inputs live
        // in a_base (a_h / a_h2 / a_f / datt), so in_gva is the source.
        // Each site group writes its own QuantParams slot (kQAct* above): the
        // GPU reads upload-heap CBVs at execution time, so a shared slot would
        // hand every dispatch the last-written cols/blocks. All four slots hold
        // model-constant values, so the per-layer record-time overwrites are
        // idempotent (same discipline as the GemvParams writes above).
        auto write_quant = [&](std::uint64_t slot, std::size_t cols) {
            auto* q = reinterpret_cast<QuantParams*>(
                static_cast<std::uint8_t*>(s.params_cpu) + slot);
            q->blocks = static_cast<std::uint32_t>(cols / kElemsPerBlock);
            q->columns = static_cast<std::uint32_t>(cols);
        };
        // DP4A staging buffer (s.qact, DEFAULT heap, created UAV in the prolog).
        // It lives in its own resource and stays in UNORDERED_ACCESS for the whole
        // submission: both quantize outputs and gemv_dp4a inputs are UAVs (u0/u1
        // and u1/u2 respectively), so a single UAV barrier orders the staging
        // writes before the reads. No state transitions (the old SRV-read dance
        // cost two transitions per site).
        const std::uint64_t qact_base = s.qact->GetGPUVirtualAddress();
        auto qact_to_nsr = [&]() {
            // Kept as a no-op so the call sites below read unchanged: qact is
            // UAV-only now (see above).
            (void)list;
        };
        auto qact_to_uav = [&]() { (void)list; };
        auto quantize = [&](std::uint64_t site_slot, std::uint64_t in_gva,
                            std::size_t cols) {
            write_quant(site_slot, cols);
            list->SetPipelineState(s.pso_quantize_acts.Get());
            // QuantParams: {blocks, columns}
            list->SetComputeRootConstantBufferView(0, pb + site_slot);
            list->SetComputeRootShaderResourceView(1, in_gva);
            list->SetComputeRootUnorderedAccessView(5, qact_base);
            // The scale table is a RWStructuredBuffer (u1), so it goes through
            // the u1 UAV root slot (6), not a t-register SRV slot.
            list->SetComputeRootUnorderedAccessView(6, qact_base + qact_acts_span);
            list->Dispatch(static_cast<UINT>((cols / 32 + 255) / 256), 1, 1);
        };
        auto gemv = [&](std::uint64_t slot, ID3D12Resource* wq, std::uint64_t ws,
                        std::uint64_t in_gva, std::uint64_t out_gva, std::size_t rows) {
            list->SetPipelineState(s.pso_gemv.Get());
            list->SetComputeRootConstantBufferView(0, pb + slot);
            list->SetComputeRootShaderResourceView(1, wq->GetGPUVirtualAddress());
            list->SetComputeRootShaderResourceView(2, ws);
            list->SetComputeRootShaderResourceView(3, in_gva);
            list->SetComputeRootUnorderedAccessView(5, out_gva);
            list->Dispatch(groups(rows), 1, 1);
        };
        auto gemv_dp4a = [&](std::uint64_t slot, ID3D12Resource* wq, std::uint64_t ws,
                             std::size_t cols, std::uint64_t out_gva, std::size_t rows) {
            (void)cols;  // row span comes from the GemvParams at `slot`
            list->SetPipelineState(s.pso_gemv_dp4a.Get());
            list->SetComputeRootConstantBufferView(0, pb + slot);
            list->SetComputeRootShaderResourceView(1, wq->GetGPUVirtualAddress());
            list->SetComputeRootShaderResourceView(2, ws);
            // Staging inputs are UAVs (u1/u2) so qact never leaves
            // UNORDERED_ACCESS; the quantizer's writes are ordered by the UAV
            // barrier at each site.
            list->SetComputeRootUnorderedAccessView(6, qact_base);
            list->SetComputeRootUnorderedAccessView(7, qact_base + qact_acts_span);
            list->SetComputeRootUnorderedAccessView(5, out_gva);
            list->Dispatch(groups(rows), 1, 1);
        };
        // Same call site, either arithmetic: DP4A reads the quantized staging
        // tail (written by the site's hoisted quantize), fp32 reads the raw vector.
        auto gemv_auto = [&](std::uint64_t slot, ID3D12Resource* wq, std::uint64_t ws,
                             std::uint64_t in_gva, std::uint64_t out_gva, std::size_t rows,
                             std::size_t cols) {
            if (use_dp4a) {
                gemv_dp4a(slot, wq, ws, cols, out_gva, rows);
            } else {
                gemv(slot, wq, ws, in_gva, out_gva, rows);
            }
        };

        list->SetComputeRootSignature(s.root.Get());

        stamp_op(0); // rms1 = [41, 42)
        if (use_fuseq_rms) {
            list->SetPipelineState(s.pso_rms_fuse.Get());
            list->SetComputeRootConstantBufferView(0, pb + kRms1);
            list->SetComputeRootShaderResourceView(1, hin_gva);
            list->SetComputeRootShaderResourceView(2, nrm_gva);
            list->SetComputeRootUnorderedAccessView(5, a_base + a_h);
            list->SetComputeRootUnorderedAccessView(8, qact_base);
            list->SetComputeRootUnorderedAccessView(9, qact_base + qact_acts_span);
            list->Dispatch(1, 1, 1);
        } else {
            list->SetPipelineState(s.pso_rms.Get());
            list->SetComputeRootConstantBufferView(0, pb + kRms1);
            list->SetComputeRootShaderResourceView(1, hin_gva);
            list->SetComputeRootShaderResourceView(2, nrm_gva);
            list->SetComputeRootUnorderedAccessView(5, a_base + a_h);
            list->Dispatch(1, 1, 1);
        }

        // 2-4. Q|K|V matvecs (fused into one dispatch over the concatenated
        // table when available; identical per-row math, so parity holds).
        // KVDIRECT: the v matvec writes its row straight into the vcache slice - the
        // GEMV stores exactly kv_width floats at the bound UAV, which is the cache row
        // layout, so no kernel change is needed and no copy ever runs for v.
        // The Q/K/V matvecs read A.h as an SRV (ybuf goes UAV -> NSR) and write
        // arena B; fbuf itself stays UAV (creation state), so the DP4A quantize
        // writes the staging buffer and the DP4A matvecs read it back without any
        // transition - a UAV barrier is what orders the staging writes, since
        // the staging buffer (s.qact) is disjoint from arena B (qact_* prolog).
        transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        stamp_op(1); // q = [42, 43)
        if (use_dp4a) {
            // One quantize for the shared Q/K/V input (fused and unfused read the
            // same vector); the matvecs below consume the staging buffer. The
            // staging buffer lives in s.qact (its own UAV-only resource), read
            // back as UAVs (u1/u2) - only the barrier orders the writes. When
            // the producer fused (use_fuseq_rms), rms1 already wrote both
            // streams, so only the ordering barrier remains.
            if (!use_fuseq_rms) {
                quantize(kQActQkv, a_base + a_h, hidden);
            }
            uav_barrier_on(list, s.qact.Get());
            qact_to_nsr();
        }
        if (use_fqkv) {
            gemv_auto(kGemvBase + 0, fqkv[li]->quants.Get(), fqkv[li]->scales_va,
                      a_base + a_h, b_base + b_q, q_width + 2 * kv_width, hidden);
        } else {
            gemv_auto(kGemvBase + 0, w.qq.Get(), w.qs, a_base + a_h, b_base + b_q, q_width, hidden);
            stamp_op(2); // k = [43, 44)
            gemv_auto(kGemvBase + 256, w.kq.Get(), w.ks, a_base + a_h, b_base + b_k, kv_width, hidden);
            stamp_op(3); // v = [44, 45)
            gemv_auto(kGemvBase + 512, w.vq.Get(), w.vs, a_base + a_h,
                      k_direct ? (s.vcache->GetGPUVirtualAddress() + dst) : (b_base + b_v), kv_width,
                      hidden);
        }
        if (use_dp4a) {
            qact_to_uav();  // restore the staging write state for the next site
        }

        // 5/6. RoPE on q and k. Rope-k writes its output straight into the kcache row
        // when the append is direct: the kernel stores heads*head_dim (= kv_width)
        // floats at the bound UAV, which is exactly the [context][kv_heads][head_dim]
        // row layout, so the same instruction stream that used to feed the
        // CopyBufferRegion now populates the cache with no intermediate stop.
        transition(s, s.fbuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        stamp_op(4); // rope-q = [45, 46)
        list->SetPipelineState(s.pso_rope.Get());
        list->SetComputeRootConstantBufferView(0, pb + kRopeParamOffset);
        list->SetComputeRootShaderResourceView(1, b_base + b_q);
        list->SetComputeRootShaderResourceView(2, s.dcos->GetGPUVirtualAddress() + li * tab_stride);
        list->SetComputeRootShaderResourceView(3, s.dsin->GetGPUVirtualAddress() + li * tab_stride);
        list->SetComputeRootUnorderedAccessView(5, a_base + a_qr);
        list->Dispatch(static_cast<UINT>((q_width + 255) / 256), 1, 1);

        list->SetComputeRootConstantBufferView(0, pb + kRopeParamOffset2);
        list->SetComputeRootShaderResourceView(1, b_base + b_k);
        list->SetComputeRootUnorderedAccessView(
            5, k_direct ? (s.kcache->GetGPUVirtualAddress() + dst) : (a_base + a_kr));
        stamp_op(5); // rope-k = [46, 47)
        list->Dispatch(static_cast<UINT>((kv_width + 255) / 256), 1, 1);

        // 7. Append k and v.
        // KVDIRECT: k and v were written by the dispatches above, so there is nothing
        // to copy and the caches never leave UNORDERED_ACCESS. One UAV barrier orders
        // those writes with the attention reads below (and the next layer's writes).
        // Legacy path: two CopyBufferRegions cycling the caches through COPY_DEST, plus
        // the copy-source round trips on the arenas they read from.
        D3D12_RESOURCE_STATES fbuf_after_append = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        stamp_op(6); // cache = [47, 48)
        if (k_direct) {
            // In-stream probes kept at their historic slot positions so trace tooling
            // stays comparable; probe0 (xbuf->yread, zero transitions) remains the
            // no-transition copy baseline.
            if (op_stamps && li == 0 && s.ts_heap && s.ts_read) {
                s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 34);
                s.list->CopyBufferRegion(s.yread.Get(), 0, s.xbuf.Get(), 0, 2048);
                s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 35);
            }
            uav_barrier(s, nullptr);
            if (op_stamps && li == 0 && s.ts_heap && s.ts_read) {
                s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 58);
                s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 59);
                s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 60);
            }
        } else {
            fbuf_after_append = D3D12_RESOURCE_STATE_COPY_SOURCE;
            transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
            transition(s, s.fbuf.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE);
            transition_on(list, s.kcache.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
            transition_on(list, s.vcache.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
            // In-stream probes: probe0 (xbuf->yread, zero transitions) is a
            // no-transition copy baseline; the two per-copy stamps split the two
            // cache appends so the big-destination cost can be attributed per copy.
            if (op_stamps && li == 0 && s.ts_heap && s.ts_read) {
                s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 34);
                s.list->CopyBufferRegion(s.yread.Get(), 0, s.xbuf.Get(), 0, 2048);
                s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 35);
            }
            list->CopyBufferRegion(s.kcache.Get(), dst, s.ybuf.Get(), a_kr, kv_bytes);
            if (op_stamps && li == 0 && s.ts_heap && s.ts_read) {
                s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 58);
            }
            list->CopyBufferRegion(s.vcache.Get(), dst, s.fbuf.Get(), b_v, kv_bytes);
            if (op_stamps && li == 0 && s.ts_heap && s.ts_read) {
                s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 59);
            }
            transition_on(list, s.kcache.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            transition_on(list, s.vcache.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            if (op_stamps && li == 0 && s.ts_heap && s.ts_read) {
                s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 60);
            }
        }

        // 8. Attention over the whole visible window. The caches are bound as UAVs at
        // the buffer base (always 256B aligned); the layer slice offset rides in the
        // attention cbuffer (kv_base/v_base).
        if (k_direct) {
            // rope-q wrote a_qr through the u0 slot, so ybuf is in UAV; the attention
            // kernel reads it as an SRV.
            transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        transition(s, s.datt.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        stamp_op(7); // attn = [48, 49)
        if (use_fuseq_attn) {
            list->SetPipelineState(s.pso_attn_fuse.Get());
            list->SetComputeRootConstantBufferView(0, pb + kAttnParamSlot);
            list->SetComputeRootShaderResourceView(1, a_base + a_qr);
            list->SetComputeRootUnorderedAccessView(6, s.kcache->GetGPUVirtualAddress());
            list->SetComputeRootUnorderedAccessView(7, s.vcache->GetGPUVirtualAddress());
            list->SetComputeRootUnorderedAccessView(5, s.datt->GetGPUVirtualAddress());
            list->SetComputeRootUnorderedAccessView(8, qact_base);
            list->SetComputeRootUnorderedAccessView(9, qact_base + qact_acts_span);
            list->Dispatch(static_cast<UINT>(heads), 1, 1);
        } else {
            // attention2 (a2): two heads per 64-thread group, grid = ceil(heads/2).
            // The fuseq variant above has its own 1-head grid and is untouched:
            // it only runs under ZEN_D3D12_DP4A=1, out of scope for this gate.
            const bool a2 = use_attn2(s, pos);
            list->SetPipelineState(a2 ? s.pso_attn2.Get() : s.pso_attn.Get());
            list->SetComputeRootConstantBufferView(0, pb + kAttnParamSlot);
            list->SetComputeRootShaderResourceView(1, a_base + a_qr);
            list->SetComputeRootUnorderedAccessView(6, s.kcache->GetGPUVirtualAddress());
            list->SetComputeRootUnorderedAccessView(7, s.vcache->GetGPUVirtualAddress());
            list->SetComputeRootUnorderedAccessView(5, s.datt->GetGPUVirtualAddress());
            list->Dispatch(a2 ? static_cast<UINT>((heads + 1) / 2) : static_cast<UINT>(heads), 1, 1);
        }
        if (kv_direct) {
            // Order this layer's cache reads before the next layer's cache writes
            // (WAR hazard on the shared cache buffers).
            uav_barrier(s, nullptr);
        }

        // 9. attn_output matvec
        transition(s, s.datt.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        stamp_op(8); // o-gemv = [49, 50)
        if (use_dp4a) {
            // The staging buffer is a separate resource, so the o site needs no
            // fbuf transition dance at all: quantize (UAV write) -> UAV barrier
            // -> UAV->NSR -> DP4A matvec (SRV read of the tail, UAV write of
            // arena A) -> NSR->UAV, and the append state of fbuf is untouched.
            // When attention fused the quantize (use_fuseq_attn), only the
            // ordering barrier before the DP4A matvec remains.
            if (!use_fuseq_attn) {
                quantize(kQActO, s.datt->GetGPUVirtualAddress() + scores_floats * sizeof(float), q_width);
            }
            uav_barrier_on(list, s.qact.Get());
            qact_to_nsr();
            gemv_dp4a(kGemvBase + 768, w.oq.Get(), w.os_, q_width, a_base + a_t0, hidden);
            qact_to_uav();
        } else {
            gemv(kGemvBase + 768, w.oq.Get(), w.os_, s.datt->GetGPUVirtualAddress() + scores_floats * sizeof(float), a_base + a_t0, hidden);
        }
        transition(s, s.datt.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);

        // 10. C.h1 = A.t0 + x
        if (trace_pipe && s.ts_heap && s.ts_read && num_layers <= 24) {
            s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 2 + 2 * li);
        }
        transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        stamp_op(9); // addn1 = [50, 51)
        list->SetPipelineState(s.pso_addn.Get());
        list->SetComputeRootConstantBufferView(0, pb + kAdd1);
        list->SetComputeRootShaderResourceView(1, a_base + a_t0);
        list->SetComputeRootShaderResourceView(2, hin_gva);
        list->SetComputeRootUnorderedAccessView(5, c_base);
        list->Dispatch(static_cast<UINT>((hidden + 255) / 256), 1, 1);

        // 11. A.h2 = rmsnorm(C.h1, ffn_norm)
        transition(s, s.hbuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        stamp_op(10); // rms2 = [51, 52)
        if (use_fuseq_rms) {
            list->SetPipelineState(s.pso_rms_fuse.Get());
            list->SetComputeRootConstantBufferView(0, pb + kRms2);
            list->SetComputeRootShaderResourceView(1, c_base);
            list->SetComputeRootShaderResourceView(2, nrm2_gva);
            list->SetComputeRootUnorderedAccessView(5, a_base + a_h2);
            list->SetComputeRootUnorderedAccessView(8, qact_base);
            list->SetComputeRootUnorderedAccessView(9, qact_base + qact_acts_span);
            list->Dispatch(1, 1, 1);
        } else {
            list->SetPipelineState(s.pso_rms.Get());
            list->SetComputeRootConstantBufferView(0, pb + kRms2);
            list->SetComputeRootShaderResourceView(1, c_base);
            list->SetComputeRootShaderResourceView(2, nrm2_gva);
            list->SetComputeRootUnorderedAccessView(5, a_base + a_h2);
            list->Dispatch(1, 1, 1);
        }

        // 12/13. gate|up from A.h2 into B
        transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transition(s, s.fbuf.Get(), fbuf_after_append, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        stamp_op(11); // gate = [52, 53)
        if (use_dp4a) {
            // Both gate and up read A.h2 (fused or not), so one quantize serves
            // both matvecs; the staging buffer (s.qact) is UAV on entry (the
            // previous site's closing transition restored it). When rms2 fused
            // the quantize (use_fuseq_rms), only the ordering barrier remains.
            if (!use_fuseq_rms) {
                quantize(kQActGu, a_base + a_h2, hidden);
            }
            uav_barrier_on(list, s.qact.Get());
            qact_to_nsr();
        }
        if (fuse_gu_all) {
            gemv_auto(kGemvBase + 1024, fgu[li]->quants.Get(), fgu[li]->scales_va,
                      a_base + a_h2, b_base + b_g, 2 * ffn_rows, hidden);
        } else {
            gemv_auto(kGemvBase + 1024, w.gq.Get(), w.gs, a_base + a_h2, b_base + b_g, ffn_rows, hidden);
            stamp_op(12); // up = [53, 54)
            gemv_auto(kGemvBase + 1280, w.uq.Get(), w.us, a_base + a_h2, b_base + b_u, ffn_rows, hidden);
        }
        if (use_dp4a) {
            qact_to_uav();  // restore the staging write state for the next site
        }

        // 14. A.f = silu(B.g) * B.u
        transition(s, s.fbuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        stamp_op(13); // silu = [54, 55)
        if (use_fuseq_silu) {
            list->SetPipelineState(s.pso_silu_fuse.Get());
            list->SetComputeRootConstantBufferView(0, pb + kSilu);
            list->SetComputeRootShaderResourceView(1, b_base + b_g);
            list->SetComputeRootShaderResourceView(2, b_base + b_u);
            list->SetComputeRootUnorderedAccessView(5, a_base + a_f);
            list->SetComputeRootUnorderedAccessView(8, qact_base);
            list->SetComputeRootUnorderedAccessView(9, qact_base + qact_acts_span);
            list->Dispatch(static_cast<UINT>((ffn_rows / 32 + 255) / 256), 1, 1);
        } else {
            list->SetPipelineState(s.pso_silu.Get());
            list->SetComputeRootConstantBufferView(0, pb + kSilu);
            list->SetComputeRootShaderResourceView(1, b_base + b_g);
            list->SetComputeRootShaderResourceView(2, b_base + b_u);
            list->SetComputeRootUnorderedAccessView(5, a_base + a_f);
            list->Dispatch(static_cast<UINT>((ffn_rows + 255) / 256), 1, 1);
        }

        // 15. B.d = down @ A.f
        transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transition(s, s.fbuf.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        stamp_op(14); // down = [55, 56)
        if (use_dp4a) {
            if (!use_fuseq_silu) {
                quantize(kQActDown, a_base + a_f, ffn_rows);
            }
            uav_barrier_on(list, s.qact.Get());
            qact_to_nsr();
        }
        gemv_auto(kGemvBase + 1536, w.dq_.Get(), w.ds, a_base + a_f, b_base + b_d, hidden, ffn_rows);
        if (use_dp4a) {
            qact_to_uav();  // restore the staging write state for the next layer
        }

        // 16. A.fin = C.h1 + B.d
        transition(s, s.fbuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        stamp_op(15); // addn2 = [56, 57)
        list->SetPipelineState(s.pso_addn.Get());
        list->SetComputeRootConstantBufferView(0, pb + kAdd2);
        list->SetComputeRootShaderResourceView(1, c_base);
        list->SetComputeRootShaderResourceView(2, b_base + b_d);
        list->SetComputeRootUnorderedAccessView(5, a_base + a_fin);
        list->Dispatch(static_cast<UINT>((hidden + 255) / 256), 1, 1);
        stamp_op(16); // end of layer 0 = [57]

        // Final layer: read back to host. Other layers: nothing - the next
        // layer's input copy (to s.hin) reads the result directly out of s.ybuf,
        // so no per-layer output copy exists any more.
        if (li == num_layers - 1) {
            transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->CopyBufferRegion(s.yread.Get(), 0, s.ybuf.Get(), a_fin, hidden_bytes);
            transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }

        // Restore buffers to expected states for next layer
        transition(s, s.fbuf.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        transition(s, s.hbuf.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    }

    // Fused LM head (same command list): final rmsnorm over the last residual, vocab
    // GEMV, logits readback. Numerics match rms_norm + gemv_q8_0 exactly: same
    // rmsnorm/GEMV kernels and parameter shapes, the norm table comes from the same
    // upload, the head weights from the same repacked residency, epsilon from the
    // caller. ybuf ends the layer loop in UNORDERED_ACCESS (a_fin written by addn2);
    // datt was returned to COMMON after the last o-gemv.
    // Parameter slots sit past every per-layer slot (7168/7424 < 8192); the per-layer
    // attention slots start at 8192.
    constexpr std::uint64_t kGemvLm = 7168;
    constexpr std::uint64_t kRmsLm = 7424;
    if (lm_active) {
        const std::uint64_t pb_lm = s.params->GetGPUVirtualAddress();
        const std::uint64_t a_base_lm = s.ybuf->GetGPUVirtualAddress();
        const std::uint64_t lm_norm_off =
            static_cast<std::uint64_t>(2 * num_layers) * hidden_bytes;
        transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transition(s, s.datt.Get(), D3D12_RESOURCE_STATE_COMMON,
                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        s.list->SetPipelineState(s.pso_rms.Get());
        s.list->SetComputeRootConstantBufferView(0, pb_lm + kRmsLm);
        s.list->SetComputeRootShaderResourceView(1, a_base_lm + a_fin);
        s.list->SetComputeRootShaderResourceView(2, s.nrm->GetGPUVirtualAddress() + lm_norm_off);
        s.list->SetComputeRootUnorderedAccessView(5, s.datt->GetGPUVirtualAddress());
        s.list->Dispatch(1, 1, 1);

        // The GEMV reads the normed activation as an SRV: buffers promote from COMMON
        // on first use, so datt only needs to leave UAV.
        transition(s, s.datt.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                   D3D12_RESOURCE_STATE_COMMON);
        transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        s.list->SetPipelineState(s.pso_gemv.Get());
        s.list->SetComputeRootConstantBufferView(0, pb_lm + kGemvLm);
        s.list->SetComputeRootShaderResourceView(1, lm_entry->quants->GetGPUVirtualAddress());
        s.list->SetComputeRootShaderResourceView(2, lm_entry->scales_va);
        s.list->SetComputeRootShaderResourceView(3, s.datt->GetGPUVirtualAddress());
        s.list->SetComputeRootUnorderedAccessView(5, a_base_lm + logits_off);
        s.list->Dispatch(
            static_cast<UINT>((lm->vocab_rows + kRowsPerGroup - 1) / kRowsPerGroup), 1, 1);

        // Both readbacks ride the same list: hidden state at yread[0..hidden), logits
        // at yread[logits_off..).
        transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                   D3D12_RESOURCE_STATE_COPY_SOURCE);
        s.list->CopyBufferRegion(s.yread.Get(), logits_off, s.ybuf.Get(), logits_off,
                                 lm->vocab_rows * sizeof(float));
        transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }

    // Submit the single command list, then wait once for all layers.
    if (trace_pipe && s.ts_heap && s.ts_read && num_layers <= 24) {
        s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 2 * num_layers + 1);
        // Resolve the whole heap (64 slots): layer-0 op stamps sit at 41..57, so
        // a count large enough to cover them removes count bookkeeping. Slots
        // that were never stamped resolve as stale values and go unused.
        s.list->ResolveQueryData(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0,
                                 64, s.ts_read.Get(), 0);
    }
    if (FAILED(s.list->Close())) return false;
    {
        const auto pipe_start = std::chrono::steady_clock::now();
        ID3D12CommandList* lists[] = {s.list.Get()};
        s.queue->ExecuteCommandLists(1, lists);
        ++s.stats.submits;
        s.stats.batches += static_cast<std::uint64_t>(num_layers);
        {
            // Exact per-layer dispatch count: 15 base (rms1, q/k/v, 2x rope,
            // attn, o, addn1, rms2, gate/up, silu, down, addn2), minus 2 for a
            // fused QKV and 1 for a fused gate|up, plus the standalone
            // quantize dispatches DP4A adds that FUSEQ has not removed.
            const bool sub_dp4a = dp4a_enabled() && s.pso_gemv_dp4a != nullptr &&
                                  s.pso_quantize_acts != nullptr;
            const bool sub_rms_silu = sub_dp4a && dp4a_fuseq_enabled() &&
                                      s.pso_rms_fuse != nullptr &&
                                      s.pso_silu_fuse != nullptr &&
                                      s.pso_attn_fuse != nullptr;
            const bool sub_attn = sub_rms_silu && (head_dim % kElemsPerBlock) == 0;
            const std::size_t n_quant = sub_dp4a ? (sub_attn ? 0u : (sub_rms_silu ? 1u : 4u)) : 0u;
            const bool sub_fqkv = fuse_qkv_all && !kv_direct;
            std::size_t per_layer = 15u - (sub_fqkv ? 2u : 0u) - (fuse_gu_all ? 1u : 0u) + n_quant;
            s.stats.dispatches += static_cast<std::uint64_t>(num_layers * per_layer);
        }
        if (lm_active) {
            s.stats.dispatches += 2;  // final rmsnorm + vocab GEMV
        }

        if (!wait_gpu(s)) return false;
        // This path waits directly instead of through execute_and_wait(), so
        // without this the whole multi-layer submission is missing from the
        // wait/fence accounting and the trace report understates decode.
        const std::uint64_t pipe_us = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - pipe_start).count());
        s.stats.wait_us += pipe_us;
        s.stats.ffn_wait_us += pipe_us;
        if (const char* trace = std::getenv("ZEN_D3D12_TRACE"); trace && trace[0] == '1') {
            std::printf("[d3d12] pipeline: gpu %.3f ms num_layers=%zu\n", pipe_us / 1000.0, num_layers);
            if (trace_pipe && s.ts_heap && s.ts_read) {
                std::uint64_t* stamps = nullptr;
                const D3D12_RANGE ts_range{0, sizeof(std::uint64_t) * 64};
                if (SUCCEEDED(s.ts_read->Map(0, &ts_range, reinterpret_cast<void**>(&stamps)))) {
                    const auto stamp_us = [freq = s.ts_freq](std::uint64_t a, std::uint64_t b) {
                        return a <= b ? static_cast<double>(b - a) * 1e6 /
                                        static_cast<double>(freq) : 0.0;
                    };
                    const double device_ms =
                        stamp_us(stamps[0], stamps[2 * num_layers + 1]) / 1000.0;
                    if (device_ms >= 0.0) {
                        s.stats.gpu_us += static_cast<std::uint64_t>(device_ms * 1000.0);
                        std::printf("[d3d12] pipeline: device %.3f ms wall %.3f ms overhead %.3f ms\n",
                                    device_ms, pipe_us / 1000.0,
                                    std::max(0.0, pipe_us / 1000.0 - device_ms));
                        double attn_us = 0.0, ffn_us = 0.0;
                        for (std::size_t li = 0; li < num_layers; ++li) {
                            const double a = stamp_us(stamps[1 + 2 * li], stamps[2 + 2 * li]);
                            const double f = li + 1 < num_layers
                                ? stamp_us(stamps[2 + 2 * li], stamps[1 + 2 * (li + 1)])
                                : stamp_us(stamps[2 + 2 * li], stamps[2 * num_layers + 1]);
                            attn_us += a;
                            ffn_us += f;
                            if (layer_stamps) {
                                std::printf("[d3d12] layer %2zu: attn-half %.1f us ffn-half+copy %.1f us\n",
                                            li, a, f);
                            }
                        }
                        std::printf("[d3d12] pipeline: attn-half=%.0f us ffn-half+copy=%.0f us total=%.0f us (walls include inter-layer gaps)\n",
                                    attn_us, ffn_us, attn_us + ffn_us);
                    }
                    if (op_stamps && device_ms >= 0.0) {
                        // Layer-0 per-op decomposition: op i spans stamps
                        // (41+i) -> (42+i); copy-in spans layer start (1) -> 41.
                        static const char* const op_names[] = {
                            "copy-in", "rms1", "q-gemv", "k-gemv", "v-gemv",
                            "rope-q", "rope-k", "cache-copy", "attn", "o-gemv",
                            "addn1", "rms2", "gate-gemv", "up-gemv", "silu",
                            "down-gemv", "addn2"};
                        double op_total = 0.0;
                        std::printf("[d3d12] layer0 ops:");
                        for (std::size_t oi = 0; oi < 17; ++oi) {
                            const std::size_t beg = oi == 0 ? 1 : op_stamp_first + oi - 1;
                            const std::size_t end = op_stamp_first + oi;
                            const double t = stamp_us(stamps[beg], stamps[end]);
                            op_total += t;
                            std::printf(" %s=%.1f", op_names[oi], t);
                        }
                        const double probe0 = stamp_us(stamps[34], stamps[35]);
                        const double kcopy = stamp_us(stamps[op_stamp_first + 6], stamps[58]);
                        const double vcopy = stamp_us(stamps[58], stamps[59]);
                        const double cacheback = stamp_us(stamps[59], stamps[60]);
                        std::printf(" | probe0=%.1f kcopy=%.1f vcopy=%.1f cache-transitions=%.1f us\n",
                                    probe0, kcopy, vcopy, cacheback);
                        std::printf(" | ops=%.0f us\n", op_total);
                    }
                    s.ts_read->Unmap(0, &ts_range);
                } else {
                    std::printf("[d3d12] pipeline: ts map failed\n");
                }
            }
        }
        if (const char* st = std::getenv("ZEN_D3D12_STAGE"); st && st[0] == '1') {
            const auto submit_done = std::chrono::steady_clock::now();
            const double bind_us = std::chrono::duration<double, std::micro>(bind_done - bind_start).count();
            const double prep_us = std::chrono::duration<double, std::micro>(record_start - bind_done).count();
            const double rec_us = std::chrono::duration<double, std::micro>(pipe_start - record_start).count();
            std::printf("[pipe] bind=%.0f us prep=%.0f us record=%.0f us gpu=%.0f us\n",
                        bind_us, prep_us, rec_us, double(pipe_us));
        }
    }

    // Read back final output - and the fused logits when the LM head ran in the same
    // submission (hidden state at yread[0..hidden), logits at yread[logits_off..)).
    const std::size_t read_bytes =
        lm_active ? static_cast<std::size_t>(logits_off) + lm->vocab_rows * sizeof(float)
                  : hidden_bytes;
    void* mapped = nullptr;
    const D3D12_RANGE read_range{0, read_bytes};
    if (FAILED(s.yread->Map(0, &read_range, &mapped))) return false;
    std::memcpy(output, mapped, hidden_bytes);
    if (lm_active) {
        std::memcpy(lm->logits, static_cast<const std::uint8_t*>(mapped) + logits_off,
                    lm->vocab_rows * sizeof(float));
    }
    const D3D12_RANGE written{0, 0};
    s.yread->Unmap(0, &written);
    if (lm_active) {
        lm->fused = true;
    }

    // Every layer wrote this token's k/v rows for position `pos` in the submission that
    // just completed, so the whole [pos, pos + 1) range is now resident.
    for (std::size_t li = 0; li < num_layers; ++li) {
        kv_resident_merge(s, layers[li].layer, layers[li].pos, 1);
    }

    if (const char* trace = std::getenv("ZEN_D3D12_TRACE"); trace && trace[0] == '1') {
        std::printf("[d3d12] pipeline: SUCCESS num_layers=%zu%s\n", num_layers,
                    lm_active ? " lm=fused" : "");
    }

    return true;
}
//
// norm -> QKV matvecs -> rope(q) -> rope(k) -> KV append -> GQA attention ->
// attn_output matvec -> residual -> FFN norm -> gate|up -> SiLU -> down -> residual,
// then one readback. q, k and v never leave VRAM, which is what removes the two
// extra submissions (and their ~130us each of fixed fence latency) that the split
// path pays per layer.
//
// Three activation arenas are needed, not two. A dispatch cannot read and write the
// same resource, and the residual base h1 has to stay readable through the whole
// FFN half while both other arenas are being written; with only A and B the final
// residual would need B.h1 and B.d in two different states at once.
//
//   A: h, qr, kr, t0, h2, f, fin      B: q, k, v, g, u, d      C: h1
//
// Every role swap is an explicit full-resource transition, and every buffer is
// returned to the state its owner expects so the next call can assume it.
bool layer_decode(const std::uint8_t* q_w, const std::uint8_t* k_w, const std::uint8_t* v_w,
                  const std::uint8_t* o_w, const std::uint8_t* gate_w, const std::uint8_t* up_w,
                  const std::uint8_t* down_w, const float* attn_norm_w, const float* ffn_norm_w,
                  const float* cos, const float* sin, std::size_t pairs, const float* x,
                  float* out, std::size_t layer, std::size_t pos, std::size_t first_pos,
                  std::size_t hidden, std::size_t heads, std::size_t kv_heads,
                  std::size_t head_dim, std::size_t ffn_rows, float epsilon,
                  bool interleaved_pairs) noexcept {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!init_locked(s)) {
        return false;
    }

    

    const std::size_t q_width = heads * head_dim;
    const std::size_t kv_width = kv_heads * head_dim;
    if (!hidden || !q_width || !kv_width || !ffn_rows || !pairs) {
        ++s.stats.fallbacks;
        return false;
    }
    if (hidden % kElemsPerBlock || ffn_rows % kElemsPerBlock || q_width % kElemsPerBlock ||
        kv_width % kElemsPerBlock) {
        ++s.stats.fallbacks;
        return false;
    }
    if (!s.kcache || layer >= s.kv_layers || pos >= s.kv_context || s.kv_heads != kv_heads ||
        s.kv_head_dim != head_dim) {
        ++s.stats.fallbacks;
        return false;
    }

    // Resolve all seven weights before recording anything, so a residency eviction
    // cannot invalidate a buffer an already-recorded dispatch is reading.
    ComPtr<ID3D12Resource> qq, kq, vq, oq, gq, uq, dq_;
    std::uint64_t qs = 0, ks = 0, vs = 0, os_ = 0, gs = 0, us = 0, ds = 0;
    auto bind = [&](const std::uint8_t* w, std::size_t rows, std::size_t cols,
                    ComPtr<ID3D12Resource>& q, std::uint64_t& sc) {
        WeightEntry* e = lookup_weights(s, w, rows, cols);
        if (!e) {
            return false;
        }
        q = e->quants;
        sc = e->scales_va;
        return true;
    };
    if (!bind(q_w, q_width, hidden, qq, qs) || !bind(k_w, kv_width, hidden, kq, ks) ||
        !bind(v_w, kv_width, hidden, vq, vs) || !bind(o_w, hidden, q_width, oq, os_) ||
        !bind(gate_w, ffn_rows, hidden, gq, gs) || !bind(up_w, ffn_rows, hidden, uq, us) ||
        !bind(down_w, hidden, ffn_rows, dq_, ds)) {
        ++s.stats.fallbacks;
        return false;
    }

    constexpr std::uint64_t kAlign = 256u;
    auto span = [](std::uint64_t bytes) { return (bytes + kAlign - 1) & ~(kAlign - 1); };
    const std::uint64_t hidden_span = span(hidden * sizeof(float));
    const std::uint64_t ffn_span = span(ffn_rows * sizeof(float));
    const std::uint64_t q_span = span(q_width * sizeof(float));
    const std::uint64_t kv_span = span(kv_width * sizeof(float));

    const std::uint64_t a_h = 0;
    const std::uint64_t a_qr = a_h + hidden_span;
    const std::uint64_t a_kr = a_qr + q_span;
    const std::uint64_t a_t0 = a_kr + kv_span;
    const std::uint64_t a_h2 = a_t0 + hidden_span;
    const std::uint64_t a_f = a_h2 + hidden_span;
    const std::uint64_t a_fin = a_f + ffn_span;
    const std::uint64_t a_total = a_fin + hidden_span;

    const std::uint64_t b_q = 0;
    const std::uint64_t b_k = b_q + q_span;
    const std::uint64_t b_v = b_k + kv_span;
    const std::uint64_t b_g = b_v + kv_span;
    const std::uint64_t b_u = b_g + ffn_span;
    const std::uint64_t b_d = b_u + ffn_span;
    const std::uint64_t b_total = b_d + hidden_span;

    // DP4A activation-quant staging layout inside s.qact (its own resource,
    // separate from fbuf - see the State field comment). Offsets match the
    // pipeline's bit for bit, and the buffer only ever grows, so interleaving
    // pipeline/legacy calls under the same mutex never reallocates it away.
    const std::uint64_t qact_max_cols =
        std::max<std::uint64_t>({hidden, ffn_rows, q_width, kv_width});
    const std::uint64_t qact_acts_span = span(qact_max_cols);
    const std::uint64_t qact_scales_span =
        span((qact_max_cols / kElemsPerBlock) * sizeof(float));
    const std::uint64_t qact_total = qact_acts_span + qact_scales_span;

    const std::size_t hidden_bytes = hidden * sizeof(float);
    const std::size_t kv_bytes = kv_width * sizeof(float);
    const std::size_t tab_bytes = pairs * sizeof(float);
    const std::size_t scores_floats = heads * s.kv_context;
    const std::size_t att_floats = heads * head_dim;
    const std::size_t datt_bytes = (scores_floats + att_floats) * sizeof(float);

    if (!reserve_scratch(s, hidden_bytes, a_total)) {
        ++s.stats.fallbacks;
        return false;
    }
    if (!s.fbuf || b_total > s.f_cap) {
        s.fbuf.Reset();
        s.fbuf = create_buffer(s.device.Get(), b_total, D3D12_HEAP_TYPE_DEFAULT,
                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                               D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        if (!s.fbuf) {
            s.f_cap = 0;
            ++s.stats.fallbacks;
            return false;
        }
        s.f_cap = static_cast<std::size_t>(b_total);
    }
    if (!s.qact || qact_total > s.qact_cap) {
        s.qact.Reset();
        s.qact = create_buffer(s.device.Get(), qact_total, D3D12_HEAP_TYPE_DEFAULT,
                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                               D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        if (!s.qact) {
            s.qact_cap = 0;
            ++s.stats.fallbacks;
            return false;
        }
        s.qact_cap = static_cast<std::size_t>(qact_total);
    }
    if (!s.hbuf || hidden_span > s.h_cap) {
        s.hbuf.Reset();
        s.hbuf = create_buffer(s.device.Get(), hidden_span, D3D12_HEAP_TYPE_DEFAULT,
                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                               D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        if (!s.hbuf) {
            s.h_cap = 0;
            ++s.stats.fallbacks;
            return false;
        }
        s.h_cap = static_cast<std::size_t>(hidden_span);
    }
    if (!ensure_upload(s.wbuf, s.w_cpu, s.w_cap, s.device.Get(), hidden_bytes) ||
        !ensure_upload(s.nbuf, s.n_cpu, s.n_cap, s.device.Get(), hidden_bytes)) {
        ++s.stats.fallbacks;
        return false;
    }
    if (!attn_upload(s, s.dcos, s.dcos_cpu, s.dcos_cap, tab_bytes) ||
        !attn_upload(s, s.dsin, s.dsin_cpu, s.dsin_cap, tab_bytes) ||
        !attn_device(s, s.datt, s.datt_cap, datt_bytes)) {
        ++s.stats.fallbacks;
        return false;
    }

    std::memcpy(s.x_cpu, x, hidden_bytes);      // residual base and norm input
    std::memcpy(s.w_cpu, ffn_norm_w, hidden_bytes);
    std::memcpy(s.n_cpu, attn_norm_w, hidden_bytes);
    std::memcpy(s.dcos_cpu, cos, tab_bytes);
    std::memcpy(s.dsin_cpu, sin, tab_bytes);

    auto* base = static_cast<std::uint8_t*>(s.params_cpu);
    constexpr std::uint64_t kGemvBase = 4096;
    constexpr std::uint64_t kRms1 = 5888;
    constexpr std::uint64_t kRms2 = 6144;
    constexpr std::uint64_t kAdd1 = 6400;
    constexpr std::uint64_t kAdd2 = 6656;
    constexpr std::uint64_t kSilu = 6912;
    auto write_gemv = [&](std::uint64_t slot, std::size_t rows, std::size_t cols) {
        auto* p = reinterpret_cast<GemvParams*>(base + slot);
        p->rows = static_cast<std::uint32_t>(rows);
        p->blocks_per_row = static_cast<std::uint32_t>(cols / kElemsPerBlock);
        p->rows_per_group = kRowsPerGroup;
        p->reserved = 0;
    };
    auto write_rms = [&](std::uint64_t slot) {
        auto* p = reinterpret_cast<RmsParams*>(base + slot);
        p->count = static_cast<std::uint32_t>(hidden);
        p->rows = 1;
        p->epsilon = epsilon;
        p->pad = 0;
    };
    auto write_addn = [&](std::uint64_t slot) {
        auto* p = reinterpret_cast<AddnParams*>(base + slot);
        p->count = static_cast<std::uint32_t>(hidden);
        p->pad0 = p->pad1 = p->pad2 = 0;
    };
    write_gemv(kGemvBase + 0, q_width, hidden);
    write_gemv(kGemvBase + 256, kv_width, hidden);
    write_gemv(kGemvBase + 512, kv_width, hidden);
    write_gemv(kGemvBase + 768, hidden, q_width);
    write_gemv(kGemvBase + 1024, ffn_rows, hidden);
    write_gemv(kGemvBase + 1280, ffn_rows, hidden);
    write_gemv(kGemvBase + 1536, hidden, ffn_rows);
    write_rms(kRms1);
    write_rms(kRms2);
    write_addn(kAdd1);
    write_addn(kAdd2);
    auto* silu = reinterpret_cast<SiluParams*>(base + kSilu);
    silu->count = static_cast<std::uint32_t>(ffn_rows);
    silu->pad0 = silu->pad1 = silu->pad2 = 0;
    auto* rp = reinterpret_cast<RopeParams*>(base + kRopeParamOffset);
    rp->heads = static_cast<std::uint32_t>(heads);
    rp->head_dim = static_cast<std::uint32_t>(head_dim);
    rp->pairs = static_cast<std::uint32_t>(pairs);
    rp->pad0 = interleaved_pairs ? 1u : 0u;
    auto* krp = reinterpret_cast<RopeParams*>(base + kRopeParamOffset2);
    krp->heads = static_cast<std::uint32_t>(kv_heads);
    krp->head_dim = static_cast<std::uint32_t>(head_dim);
    krp->pairs = static_cast<std::uint32_t>(pairs);
    krp->pad0 = interleaved_pairs ? 1u : 0u;
    auto* ap = reinterpret_cast<AttnParams*>(base + kAttnParamOffset);
    ap->heads = static_cast<std::uint32_t>(heads);
    ap->kv_heads = static_cast<std::uint32_t>(kv_heads);
    ap->head_dim = static_cast<std::uint32_t>(head_dim);
    ap->context = static_cast<std::uint32_t>(s.kv_context);
    ap->first_pos = static_cast<std::uint32_t>(first_pos);
    ap->pos = static_cast<std::uint32_t>(pos);
    ap->scores_off = 0;
    ap->out_off = static_cast<std::uint32_t>(scores_floats);
    ap->inv_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    ap->pad0 = ap->pad1 = ap->pad2 = 0.0f;
    const std::size_t stride = s.kv_heads * s.kv_context * s.kv_head_dim * sizeof(float);
    const std::size_t dst = layer * stride + pos * kv_bytes;
    // Direct append binding offset must be 256B aligned (root UAV rule); see the
    // pipeline path for the full rationale.
    const bool k_direct = kv_direct_enabled() && (dst % 256 == 0);
    const std::uint32_t kv_base =
        static_cast<std::uint32_t>(layer * kv_heads * s.kv_context * head_dim);
    ap->kv_base = kv_base;
    ap->v_base = kv_base;

    const std::uint64_t pb = s.params->GetGPUVirtualAddress();
    const std::uint64_t a_base = s.ybuf->GetGPUVirtualAddress();
    const std::uint64_t b_base = s.fbuf->GetGPUVirtualAddress();
    const std::uint64_t c_base = s.hbuf->GetGPUVirtualAddress();
    const std::uint64_t x_gva = s.xbuf->GetGPUVirtualAddress();

    auto groups = [](std::size_t rows) {
        return static_cast<UINT>((rows + kRowsPerGroup - 1) / kRowsPerGroup);
    };
    // out_gva is absolute: the QKV and FFN matvecs write into arena B, the
    // attention-output matvec into arena A, so a single fixed base would be wrong.
    auto gemv = [&](std::uint64_t slot, ID3D12Resource* wq, std::uint64_t ws,
                    std::uint64_t in_gva, std::uint64_t out_gva, std::size_t rows) {
        s.list->SetPipelineState(s.pso_gemv.Get());
        s.list->SetComputeRootConstantBufferView(0, pb + slot);
        s.list->SetComputeRootShaderResourceView(1, wq->GetGPUVirtualAddress());
        s.list->SetComputeRootShaderResourceView(2, ws);
        s.list->SetComputeRootShaderResourceView(3, in_gva);
        s.list->SetComputeRootUnorderedAccessView(5, out_gva);
        s.list->Dispatch(groups(rows), 1, 1);
    };

    const bool use_dp4a = dp4a_enabled() && s.pso_gemv_dp4a != nullptr &&
                          s.pso_quantize_acts != nullptr;
    // Producer-fused quantize (ZEN_D3D12_DP4A_FUSEQ=1, needs use_dp4a): the
    // producer dispatches write the qact int8/scale streams themselves and
    // the 4 standalone quantize dispatches are skipped. The UAV barrier
    // ordering producer writes before the DP4A GEMV reads is kept. The O
    // site also needs head_dim % 32 == 0 (block/head alignment), else it
    // alone falls back to the separate quantize.
    const bool fuseq_pso = dp4a_fuseq_enabled() && s.pso_rms_fuse != nullptr &&
                           s.pso_silu_fuse != nullptr && s.pso_attn_fuse != nullptr;
    const bool use_fuseq_rms = use_dp4a && fuseq_pso;
    const bool use_fuseq_attn = use_fuseq_rms && (head_dim % kElemsPerBlock) == 0;
    const bool use_fuseq_silu = use_fuseq_rms;
    // DP4A decode gateway (mirrors the pipeline one): each site group writes its
    // own 256B QuantParams slot (kQAct* file-scope) because the GPU reads
    // upload-heap CBVs at execution time. The staging lives in s.qact (its own
    // UAV-only resource), read back as UAVs (u1/u2) - only the barrier orders
    // the writes. One layer per call here, so no per-layer overwrite reasoning
    // is needed.
    auto write_quant = [&](std::uint64_t slot, std::size_t cols) {
        auto* q = reinterpret_cast<QuantParams*>(
            static_cast<std::uint8_t*>(s.params_cpu) + slot);
        q->blocks = static_cast<std::uint32_t>(cols / kElemsPerBlock);
        q->columns = static_cast<std::uint32_t>(cols);
    };
    const std::uint64_t qact_base = s.qact->GetGPUVirtualAddress();
    auto qact_to_nsr = [&]() {
        // Kept as a no-op for the call sites: qact is UAV-only now.
        (void)s;
    };
    auto qact_to_uav = [&]() { (void)s; };
    auto quantize = [&](std::uint64_t site_slot, std::uint64_t in_gva, std::size_t cols) {
        write_quant(site_slot, cols);
        s.list->SetPipelineState(s.pso_quantize_acts.Get());
        s.list->SetComputeRootConstantBufferView(0, pb + site_slot);
        s.list->SetComputeRootShaderResourceView(1, in_gva);
        s.list->SetComputeRootUnorderedAccessView(5, qact_base);
        // The scale table is a RWStructuredBuffer (u1), so it goes through the
        // u1 UAV root slot (6), not a t-register SRV slot.
        s.list->SetComputeRootUnorderedAccessView(6, qact_base + qact_acts_span);
        s.list->Dispatch(static_cast<UINT>((cols / 32 + 255) / 256), 1, 1);
    };
    auto gemv_dp4a = [&](std::uint64_t slot, ID3D12Resource* wq, std::uint64_t ws,
                         std::size_t cols, std::uint64_t out_gva, std::size_t rows) {
        (void)cols;  // row span comes from the GemvParams at `slot`
        s.list->SetPipelineState(s.pso_gemv_dp4a.Get());
        s.list->SetComputeRootConstantBufferView(0, pb + slot);
        s.list->SetComputeRootShaderResourceView(1, wq->GetGPUVirtualAddress());
        s.list->SetComputeRootShaderResourceView(2, ws);
        // UAV-only staging inputs (u1/u2) - see the gateway comment.
        s.list->SetComputeRootUnorderedAccessView(6, qact_base);
        s.list->SetComputeRootUnorderedAccessView(7, qact_base + qact_acts_span);
        s.list->SetComputeRootUnorderedAccessView(5, out_gva);
        s.list->Dispatch(groups(rows), 1, 1);
    };
    auto gemv_auto = [&](std::uint64_t slot, ID3D12Resource* wq, std::uint64_t ws,
                         std::uint64_t in_gva, std::uint64_t out_gva, std::size_t rows,
                         std::size_t cols) {
        if (use_dp4a) {
            gemv_dp4a(slot, wq, ws, cols, out_gva, rows);
        } else {
            gemv(slot, wq, ws, in_gva, out_gva, rows);
        }
    };

    s.allocator->Reset();
    s.list->Reset(s.allocator.Get(), use_fuseq_rms ? s.pso_rms_fuse.Get() : s.pso_rms.Get());
    s.list->SetComputeRootSignature(s.root.Get());
    s.current_comp = Comp::Projections;
    // Stamp 0 opens the layer, 1 closes the attention half (norm/QKV/rope/attention/
    // attn_output/residual), 2 closes the FFN half, 3 closes the readback copy, so the
    // trace can separate the two halves and the copy instead of billing the layer as one
    // block.
    if (s.trace) {
        s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
    }

    // 1. A.h = rmsnorm(x, attn_norm). A starts UNORDERED_ACCESS.
    s.list->SetComputeRootConstantBufferView(0, pb + kRms1);
    s.list->SetComputeRootShaderResourceView(1, x_gva);
    s.list->SetComputeRootShaderResourceView(2, s.nbuf->GetGPUVirtualAddress());
    s.list->SetComputeRootUnorderedAccessView(5, a_base + a_h);
    if (use_fuseq_rms) {
        s.list->SetComputeRootUnorderedAccessView(8, qact_base);
        s.list->SetComputeRootUnorderedAccessView(9, qact_base + qact_acts_span);
    }
    s.list->Dispatch(1, 1, 1);

    // 2-4. Q|K|V matvecs read A.h, write B. A becomes SRV, B stays UAV.
    // KVDIRECT: the v matvec targets the vcache row directly (exactly kv_width floats
    // at the bound UAV), so v never round-trips through arena B.
    transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (use_dp4a) {
        // One quantize for the shared Q/K/V input; the staging buffer (s.qact,
        // its own UAV-only resource) keeps the DP4A matvec UAV reads legal while
        // arena B continues to be written as UAV. When rms1 fused (use_fuseq_rms)
        // it already wrote both streams, so only the barrier remains.
        if (!use_fuseq_rms) {
            quantize(kQActQkv, a_base + a_h, hidden);
        }
        uav_barrier(s, s.qact.Get());
        qact_to_nsr();
    }
    gemv_auto(kGemvBase + 0, qq.Get(), qs, a_base + a_h, b_base + b_q, q_width, hidden);
    gemv_auto(kGemvBase + 256, kq.Get(), ks, a_base + a_h, b_base + b_k, kv_width, hidden);
    gemv_auto(kGemvBase + 512, vq.Get(), vs, a_base + a_h,
              k_direct ? (s.vcache->GetGPUVirtualAddress() + dst) : (b_base + b_v), kv_width,
              hidden);
    if (use_dp4a) {
        qact_to_uav();  // restore the staging write state for the next site
    }

    // 5/6. RoPE on q and k: read B, write A.
    transition(s, s.fbuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    s.list->SetPipelineState(s.pso_rope.Get());
    s.list->SetComputeRootConstantBufferView(0, pb + kRopeParamOffset);
    s.list->SetComputeRootShaderResourceView(1, b_base + b_q);
    s.list->SetComputeRootShaderResourceView(2, s.dcos->GetGPUVirtualAddress());
    s.list->SetComputeRootShaderResourceView(3, s.dsin->GetGPUVirtualAddress());
    s.list->SetComputeRootUnorderedAccessView(5, a_base + a_qr);
    s.list->Dispatch(static_cast<UINT>((q_width + 255) / 256), 1, 1);

    s.list->SetComputeRootConstantBufferView(0, pb + kRopeParamOffset2);
    s.list->SetComputeRootShaderResourceView(1, b_base + b_k);
    s.list->SetComputeRootUnorderedAccessView(
        5, k_direct ? (s.kcache->GetGPUVirtualAddress() + dst) : (a_base + a_kr));
    s.list->Dispatch(static_cast<UINT>((kv_width + 255) / 256), 1, 1);

    // 7. Append k and v at this position. KVDIRECT: both rows were written by the
    // dispatches above (k by rope-k, v by the v matvec), so only a UAV barrier is
    // needed; the legacy path keeps both CopyBufferRegions, now cycling the caches
    // through COPY_DEST from their permanent UNORDERED_ACCESS state.
    D3D12_RESOURCE_STATES fbuf_after_append = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    if (k_direct) {
        uav_barrier(s, nullptr);
    } else {
        fbuf_after_append = D3D12_RESOURCE_STATE_COPY_SOURCE;
        transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                   D3D12_RESOURCE_STATE_COPY_SOURCE);
        transition(s, s.fbuf.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                   D3D12_RESOURCE_STATE_COPY_SOURCE);
        transition(s, s.kcache.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                   D3D12_RESOURCE_STATE_COPY_DEST);
        transition(s, s.vcache.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                   D3D12_RESOURCE_STATE_COPY_DEST);
        s.list->CopyBufferRegion(s.kcache.Get(), dst, s.ybuf.Get(), a_kr, kv_bytes);
        s.list->CopyBufferRegion(s.vcache.Get(), dst, s.fbuf.Get(), b_v, kv_bytes);
        transition(s, s.kcache.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        transition(s, s.vcache.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }

    // 8. Attention: read A.qr and the caches (as UAVs; slice offset in the cbuffer),
    // write the scratch UAV.
    if (k_direct) {
        // rope-q wrote a_qr through u0, so ybuf is in UAV; the kernel reads it as SRV.
        transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
    transition(s, s.datt.Get(), D3D12_RESOURCE_STATE_COMMON,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (use_fuseq_attn) {
        s.list->SetPipelineState(s.pso_attn_fuse.Get());
        s.list->SetComputeRootConstantBufferView(0, pb + kAttnParamOffset);
        s.list->SetComputeRootShaderResourceView(1, a_base + a_qr);
        s.list->SetComputeRootUnorderedAccessView(6, s.kcache->GetGPUVirtualAddress());
        s.list->SetComputeRootUnorderedAccessView(7, s.vcache->GetGPUVirtualAddress());
        s.list->SetComputeRootUnorderedAccessView(5, s.datt->GetGPUVirtualAddress());
        s.list->SetComputeRootUnorderedAccessView(8, qact_base);
        s.list->SetComputeRootUnorderedAccessView(9, qact_base + qact_acts_span);
        // One 32-thread group per head, so the grid width is exactly `heads`.
        s.list->Dispatch(static_cast<UINT>(heads), 1, 1);
    } else {
        // attention2 (a2): two heads per 64-thread group, grid = ceil(heads/2).
        // The fuseq variant above has its own 1-head grid and is untouched:
        // it only runs under ZEN_D3D12_DP4A=1, out of scope for this gate.
        const bool a2 = use_attn2(s, pos);
        s.list->SetPipelineState(a2 ? s.pso_attn2.Get() : s.pso_attn.Get());
        s.list->SetComputeRootConstantBufferView(0, pb + kAttnParamOffset);
        s.list->SetComputeRootShaderResourceView(1, a_base + a_qr);
        s.list->SetComputeRootUnorderedAccessView(6, s.kcache->GetGPUVirtualAddress());
        s.list->SetComputeRootUnorderedAccessView(7, s.vcache->GetGPUVirtualAddress());
        s.list->SetComputeRootUnorderedAccessView(5, s.datt->GetGPUVirtualAddress());
        s.list->Dispatch(a2 ? static_cast<UINT>((heads + 1) / 2) : static_cast<UINT>(heads), 1, 1);
    }
    if (kv_direct_enabled()) {
        // Order this layer's cache reads before any later cache write (WAR hazard).
        uav_barrier(s, nullptr);
    }

    // 9. attn_output matvec: read the attention result, write A.t0.
    transition(s, s.datt.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (use_dp4a) {
        // The staging buffer is a separate resource, so the o site needs no
        // fbuf transition dance at all (same reasoning as the pipeline o site):
        // quantize (UAV write) -> UAV barrier -> UAV->NSR -> DP4A matvec
        // (SRV read of the tail, UAV write of arena A) -> NSR->UAV, and the
        // append state of fbuf is untouched. When attention fused
        // (use_fuseq_attn), only the ordering barrier remains.
        if (!use_fuseq_attn) {
            quantize(kQActO, s.datt->GetGPUVirtualAddress() + scores_floats * sizeof(float), q_width);
        }
        uav_barrier(s, s.qact.Get());
        qact_to_nsr();
        gemv_dp4a(kGemvBase + 768, oq.Get(), os_, q_width, a_base + a_t0, hidden);
        qact_to_uav();
    } else {
        gemv(kGemvBase + 768, oq.Get(), os_, s.datt->GetGPUVirtualAddress() +
                                                  scores_floats * sizeof(float),
             a_base + a_t0, hidden);
    }
    transition(s, s.datt.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               D3D12_RESOURCE_STATE_COMMON);

    // 10. C.h1 = A.t0 + x.
    transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    s.list->SetPipelineState(s.pso_addn.Get());
    s.list->SetComputeRootConstantBufferView(0, pb + kAdd1);
    s.list->SetComputeRootShaderResourceView(1, a_base + a_t0);
    s.list->SetComputeRootShaderResourceView(2, x_gva);
    s.list->SetComputeRootUnorderedAccessView(5, c_base);
    s.list->Dispatch(static_cast<UINT>((hidden + 255) / 256), 1, 1);

    if (s.trace) {
        s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
    }

    // 11. A.h2 = rmsnorm(C.h1, ffn_norm).
    transition(s, s.hbuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (use_fuseq_rms) {
        s.list->SetPipelineState(s.pso_rms_fuse.Get());
        s.list->SetComputeRootConstantBufferView(0, pb + kRms2);
        s.list->SetComputeRootShaderResourceView(1, c_base);
        s.list->SetComputeRootShaderResourceView(2, s.wbuf->GetGPUVirtualAddress());
        s.list->SetComputeRootUnorderedAccessView(5, a_base + a_h2);
        s.list->SetComputeRootUnorderedAccessView(8, qact_base);
        s.list->SetComputeRootUnorderedAccessView(9, qact_base + qact_acts_span);
        s.list->Dispatch(1, 1, 1);
    } else {
        s.list->SetPipelineState(s.pso_rms.Get());
        s.list->SetComputeRootConstantBufferView(0, pb + kRms2);
        s.list->SetComputeRootShaderResourceView(1, c_base);
        s.list->SetComputeRootShaderResourceView(2, s.wbuf->GetGPUVirtualAddress());
        s.list->SetComputeRootUnorderedAccessView(5, a_base + a_h2);
        s.list->Dispatch(1, 1, 1);
    }

    // 12/13. gate|up from A.h2 into B.
    transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(s, s.fbuf.Get(), fbuf_after_append, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (use_dp4a) {
        // Both gate and up read A.h2, so one quantize serves both matvecs; the
        // staging buffer (s.qact) is UAV on entry (the previous site's closing
        // transition restored it). When rms2 fused, only the barrier remains.
        if (!use_fuseq_rms) {
            quantize(kQActGu, a_base + a_h2, hidden);
        }
        uav_barrier(s, s.qact.Get());
        qact_to_nsr();
    }
    gemv_auto(kGemvBase + 1024, gq.Get(), gs, a_base + a_h2, b_base + b_g, ffn_rows, hidden);
    gemv_auto(kGemvBase + 1280, uq.Get(), us, a_base + a_h2, b_base + b_u, ffn_rows, hidden);
    if (use_dp4a) {
        qact_to_uav();  // restore the staging write state for the next site
    }

    // 14. A.f = silu(B.g) * B.u
    transition(s, s.fbuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (use_fuseq_silu) {
        s.list->SetPipelineState(s.pso_silu_fuse.Get());
        s.list->SetComputeRootConstantBufferView(0, pb + kSilu);
        s.list->SetComputeRootShaderResourceView(1, b_base + b_g);
        s.list->SetComputeRootShaderResourceView(2, b_base + b_u);
        s.list->SetComputeRootUnorderedAccessView(5, a_base + a_f);
        s.list->SetComputeRootUnorderedAccessView(8, qact_base);
        s.list->SetComputeRootUnorderedAccessView(9, qact_base + qact_acts_span);
        s.list->Dispatch(static_cast<UINT>((ffn_rows / 32 + 255) / 256), 1, 1);
    } else {
        s.list->SetPipelineState(s.pso_silu.Get());
        s.list->SetComputeRootConstantBufferView(0, pb + kSilu);
        s.list->SetComputeRootShaderResourceView(1, b_base + b_g);
        s.list->SetComputeRootShaderResourceView(2, b_base + b_u);
        s.list->SetComputeRootUnorderedAccessView(5, a_base + a_f);
        s.list->Dispatch(static_cast<UINT>((ffn_rows + 255) / 256), 1, 1);
    }

    // 15. B.d = down @ A.f
    transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(s, s.fbuf.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (use_dp4a) {
        if (!use_fuseq_silu) {
            quantize(kQActDown, a_base + a_f, ffn_rows);
        }
        uav_barrier(s, s.qact.Get());
        qact_to_nsr();
    }
    gemv_auto(kGemvBase + 1536, dq_.Get(), ds, a_base + a_f, b_base + b_d, hidden, ffn_rows);
    if (use_dp4a) {
        qact_to_uav();  // restore the staging write state for the next call
    }

    // 16. A.fin = C.h1 + B.d
    transition(s, s.fbuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    s.list->SetPipelineState(s.pso_addn.Get());
    s.list->SetComputeRootConstantBufferView(0, pb + kAdd2);
    s.list->SetComputeRootShaderResourceView(1, c_base);
    s.list->SetComputeRootShaderResourceView(2, b_base + b_d);
    s.list->SetComputeRootUnorderedAccessView(5, a_base + a_fin);
    s.list->Dispatch(static_cast<UINT>((hidden + 255) / 256), 1, 1);

    // 17. Read back only the hidden state. Restore every buffer to the state its next
    // user assumes: A and B UNORDERED_ACCESS (their creation state), C likewise, and
    // the attention scratch to COMMON, which is where attn_block_locked expects it.
    if (s.trace) {
        s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 2);
    }
    transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_COPY_SOURCE);
    s.list->CopyBufferRegion(s.yread.Get(), 0, s.ybuf.Get(), a_fin, hidden_bytes);
    transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    transition(s, s.fbuf.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    transition(s, s.hbuf.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (s.trace) {
        s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 3);
        s.list->ResolveQueryData(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 4,
                                 s.ts_read.Get(), 0);
    }
    s.list->Close();

    if (!execute_and_wait(s)) {
        ++s.stats.fallbacks;
        return false;
    }
    // This call wrote position `pos` of `layer` in the submission that just completed.
    kv_resident_merge(s, layer, pos, 1);

    void* mapped = nullptr;
    const D3D12_RANGE read_range{0, hidden_bytes};
    if (FAILED(s.yread->Map(0, &read_range, &mapped))) {
        ++s.stats.fallbacks;
        return false;
    }
    if (s.trace) {
        std::uint64_t dev = 0;
        accumulate_gpu_us(s, dev, 0, 1);
        note_dev(s, Comp::Projections, dev);
        dev = 0;
        accumulate_gpu_us(s, dev, 1, 2);
        note_dev(s, Comp::FfnFused, dev);
    }
    std::memcpy(out, mapped, hidden_bytes);
    const D3D12_RANGE written{0, 0};
    s.yread->Unmap(0, &written);
    // Exact: 15 base + up to 4 standalone quantizes DP4A adds (0 when FUSEQ
    // fused them all, 1 when only the O site fell back).
    s.stats.dispatches += static_cast<std::uint64_t>(
        15u + (use_dp4a ? (use_fuseq_attn ? 0u : (use_fuseq_rms ? 1u : 4u)) : 0u));
    ++s.stats.batches;
    return true;
}

bool ffn_residual_fused(const std::uint8_t* attn_out_w, const std::uint8_t* gate_w,
                        const std::uint8_t* up_w, const std::uint8_t* down_w,
                        const float* norm_weight, const float* x, const float* attn_out,
                        float* out, std::size_t hidden, std::size_t attn_width,
                        std::size_t ffn_rows, float epsilon) noexcept {
    if (!attn_out_w || !gate_w || !up_w || !down_w || !norm_weight || !x || !attn_out || !out) {
        return false;
    }
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!init_locked(s)) {
        return false;
    }
    if (!ffn_residual_fused_locked(s, attn_out_w, gate_w, up_w, down_w, norm_weight, x, attn_out,
                                   out, hidden, attn_width, ffn_rows, epsilon)) {
        ++s.stats.fallbacks;
        return false;
    }
    return true;
}

bool residual_add(const float* a, const float* b, float* out, std::size_t count) noexcept {
    if (!a || !b || !out || !count) {
        return false;
    }
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!init_locked(s)) {
        return false;
    }
    if (!addn_locked(s, a, b, out, count)) {
        ++s.stats.fallbacks;
        return false;
    }
    return true;
}

// SiLU(gate)*up for prefill, host in / host out (mirrors rms_norm_locked /
// addn_locked). One submit. The prefill hot path fuses this into
// ffn_prefill_fused instead of paying a submit for it; this entry exists so
// the SiLU kernel has a standalone parity surface and a fallback when the
// fused path declines.
bool silu_mul_locked(State& s, const float* gate, const float* up, float* out,
                     std::size_t count) {
    if (!count || count > 0xFFFFFFFFu) {
        return false;
    }
    const std::size_t bytes = count * sizeof(float);
    if (!reserve_scratch(s, bytes, bytes)) {
        return false;
    }
    std::memcpy(s.x_cpu, gate, bytes);

    if (!ensure_upload(s.wbuf, s.w_cpu, s.w_cap, s.device.Get(), bytes)) {
        return false;
    }
    std::memcpy(s.w_cpu, up, bytes);

    // Slot 3072 is clear of the batch range (512..2560), the addn slots
    // (768, 2560/2816), rope (1024/1536), attention (1280) and the pipeline
    // range (4096+), so no in-flight shape aliases this submission's params.
    constexpr std::uint64_t kSiluParamOffset = 3072;
    auto* params =
        reinterpret_cast<SiluParams*>(static_cast<std::uint8_t*>(s.params_cpu) + kSiluParamOffset);
    params->count = static_cast<std::uint32_t>(count);
    params->pad0 = params->pad1 = params->pad2 = 0;

    const UINT groups = static_cast<UINT>((count + 255) / 256);

    s.current_comp = Comp::Residual;
    s.allocator->Reset();
    s.list->Reset(s.allocator.Get(), s.pso_silu.Get());
    s.list->SetComputeRootSignature(s.root.Get());
    s.list->SetComputeRootConstantBufferView(0, s.params->GetGPUVirtualAddress() + kSiluParamOffset);
    s.list->SetComputeRootShaderResourceView(1, s.xbuf->GetGPUVirtualAddress());
    s.list->SetComputeRootShaderResourceView(2, s.wbuf->GetGPUVirtualAddress());
    s.list->SetComputeRootUnorderedAccessView(5, s.ybuf->GetGPUVirtualAddress());
    s.list->Dispatch(groups, 1, 1);

    transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_COPY_SOURCE);
    s.list->CopyBufferRegion(s.yread.Get(), 0, s.ybuf.Get(), 0, bytes);
    transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    s.list->Close();

    if (!execute_and_wait(s)) {
        return false;
    }

    void* mapped = nullptr;
    const D3D12_RANGE read_range{0, bytes};
    if (FAILED(s.yread->Map(0, &read_range, &mapped))) {
        return false;
    }
    std::memcpy(out, mapped, bytes);
    const D3D12_RANGE written{0, 0};
    s.yread->Unmap(0, &written);
    ++s.stats.dispatches;
    return true;
}

bool silu_mul(const float* gate, const float* up, float* out, std::size_t count) noexcept {
    if (!gate || !up || !out || !count) {
        return false;
    }
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!init_locked(s)) {
        return false;
    }
    if (!silu_mul_locked(s, gate, up, out, count)) {
        ++s.stats.fallbacks;
        return false;
    }
    return true;
}

// Prefill GEMM variant of ffn_residual_fused_locked: eight dispatches
// (attn_out GEMM, residual add, RMS, gate GEMM, up GEMM, SiLU, down GEMM,
// residual add) in one command list, so gate/up/the SiLU result stay in VRAM
// between the gate/up GEMMs and the down GEMM. Two buffers (ybuf = A,
// fbuf = B) alternate reader/writer roles so no dispatch both reads and
// writes the same resource; every swap is an explicit full-resource
// transition, same pattern as the decode fused path.
bool ffn_prefill_fused_locked(State& s, const std::uint8_t* attn_out_w, const std::uint8_t* gate_w,
                               const std::uint8_t* up_w, const std::uint8_t* down_w,
                               const float* norm_weight, const float* x, const float* attn_out,
                               float* out, std::size_t tokens, std::size_t hidden,
                               std::size_t attn_width, std::size_t ffn_rows, float epsilon) {
    // The fused path is fp32-only: with DP4A on the single-GEMM path runs
    // int8-activation math while the fused input never passes through the
    // host quantizer. Mirror the single-gemm gate (gemm_opted_in) so
    // ZEN_D3D12_GEMM_EXPERIMENTAL=0 still disables every GPU GEMM.
    if (dp4a_enabled() || !gemm_opted_in()) {
        return false;
    }
    if (!tokens || !hidden || !attn_width || !ffn_rows) {
        return false;
    }
    if (hidden % kElemsPerBlock || attn_width % kElemsPerBlock || ffn_rows % kElemsPerBlock) {
        return false;
    }
    // 32-bit count fields (RMS rows/count, addn count, SiLU count) and the
    // persistent scratch bound the token range; larger prefills stay on the
    // host path instead of growing hundreds of MB of scratch.
    if (tokens > 8192 || hidden > 0xFFFFFFFFu || attn_width > 0xFFFFFFFFu ||
        ffn_rows > 0xFFFFFFFFu) {
        return false;
    }
    const std::uint64_t th = static_cast<std::uint64_t>(tokens) * hidden;
    const std::uint64_t tf = static_cast<std::uint64_t>(tokens) * ffn_rows;
    if (th > 0xFFFFFFFFu || tf > 0xFFFFFFFFu) {
        return false;
    }
    // Same minimum-work gate the single GEMM honours, applied per projection.
    if (tokens * hidden * attn_width < kMinElements ||
        tokens * ffn_rows * hidden < kMinElements ||
        tokens * hidden * ffn_rows < kMinElements) {
        return false;
    }

    // Resolve every tensor first so a residency eviction cannot invalidate a
    // buffer that an already-recorded dispatch is still reading.
    ComPtr<ID3D12Resource> ao_q, g_q, u_q, d_q;
    std::uint64_t ao_s = 0, g_s = 0, u_s = 0, d_s = 0;
    if (WeightEntry* e = lookup_weights(s, attn_out_w, hidden, attn_width)) {
        ao_q = e->quants;
        ao_s = e->scales_va;
    } else {
        return false;
    }
    if (WeightEntry* e = lookup_weights(s, gate_w, ffn_rows, hidden)) {
        g_q = e->quants;
        g_s = e->scales_va;
    } else {
        return false;
    }
    if (WeightEntry* e = lookup_weights(s, up_w, ffn_rows, hidden)) {
        u_q = e->quants;
        u_s = e->scales_va;
    } else {
        return false;
    }
    if (WeightEntry* e = lookup_weights(s, down_w, hidden, ffn_rows)) {
        d_q = e->quants;
        d_s = e->scales_va;
    } else {
        return false;
    }

    constexpr std::uint64_t kAlign = 256u;
    auto span = [](std::uint64_t bytes) { return (bytes + kAlign - 1) & ~(kAlign - 1); };
    const std::uint64_t hidden_span = span(th * sizeof(float));
    const std::uint64_t ffn_span = span(tf * sizeof(float));

    // A (ybuf): attn projection, normed activation, SiLU result, final output.
    const std::uint64_t a_t0 = 0;
    const std::uint64_t a_h2 = hidden_span;
    const std::uint64_t a_f = a_h2 + hidden_span;
    const std::uint64_t a_fin = a_f + ffn_span;
    const std::uint64_t a_total = a_fin + hidden_span;

    // B (fbuf): residual base, gate, up, down projection.
    const std::uint64_t b_h1 = 0;
    const std::uint64_t b_g = hidden_span;
    const std::uint64_t b_u = b_g + ffn_span;
    const std::uint64_t b_d = b_u + ffn_span;
    const std::uint64_t b_total = b_d + hidden_span;

    constexpr std::uint64_t kCap = 256ull * 1024 * 1024;
    if (a_total > kCap || b_total > kCap) {
        return false;
    }

    if (!reserve_scratch(s, tokens * attn_width * sizeof(float),
                         static_cast<std::size_t>(a_total))) {
        return false;
    }
    if (!s.fbuf || b_total > s.f_cap) {
        s.fbuf.Reset();
        s.fbuf = create_buffer(s.device.Get(), static_cast<std::size_t>(b_total),
                               D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                               D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        if (!s.fbuf) {
            s.f_cap = 0;
            return false;
        }
        s.f_cap = static_cast<std::size_t>(b_total);
    }

    const std::size_t x_bytes = tokens * hidden * sizeof(float);
    if (!ensure_upload(s.wbuf, s.w_cpu, s.w_cap, s.device.Get(), x_bytes) ||
        !ensure_upload(s.nbuf, s.n_cpu, s.n_cap, s.device.Get(), hidden * sizeof(float))) {
        return false;
    }
    std::memcpy(s.x_cpu, attn_out, tokens * attn_width * sizeof(float));
    std::memcpy(s.w_cpu, x, x_bytes);
    std::memcpy(s.n_cpu, norm_weight, hidden * sizeof(float));

    // Distinct 256B param slots per dispatch (upload-heap CBVs are read at
    // execution, so a shared slot would make every dispatch use the last
    // written values): GEMMs 1024+256*i, RMS 256 (single-rms slot), SiLU
    // 3072, adds 2560/2816. No rope/attention dispatch shares this list, so
    // the gemm-batch slot convention is safe here.
    constexpr std::uint64_t kGemmBase = 1024;
    constexpr std::uint64_t kRmsSlot = 256;
    constexpr std::uint64_t kSiluSlot = 3072;
    constexpr std::uint64_t kAddn0 = 2560;
    constexpr std::uint64_t kAddn1 = 2816;
    auto* base = static_cast<std::uint8_t*>(s.params_cpu);
    // Tiling mirrors gemm_batch_locked (v3 > v2 > v1, all bit-identical fp32).
    const bool use_v3 = gemm_v3_enabled() && s.pso_gemm_v3 != nullptr;
    const bool use_v2 = !use_v3 && gemm_v2_enabled() && s.pso_gemm_v2 != nullptr;
    const std::uint32_t tile = use_v3 ? kTokensPerTileV3
                                      : (use_v2 ? kTokensPerTileV2 : kTokensPerTile);
    const std::uint32_t rpg = use_v3 ? kGemmRowsPerGroupV3
                                     : (use_v2 ? kGemmRowsPerGroupV2 : kGemmRowsPerGroup);
    ID3D12PipelineState* gemm_pso = use_v3 ? s.pso_gemm_v3.Get()
                                           : (use_v2 ? s.pso_gemm_v2.Get() : s.pso_gemm.Get());
    auto fill_gemm = [&](std::uint64_t slot, std::size_t rows, std::size_t cols) {
        auto* params = reinterpret_cast<GemmParams*>(base + slot);
        params->rows = static_cast<std::uint32_t>(rows);
        params->blocks_per_row = static_cast<std::uint32_t>(cols / kElemsPerBlock);
        params->token_count = static_cast<std::uint32_t>(tokens);
        params->tokens_per_tile = tile;
    };
    fill_gemm(kGemmBase, hidden, attn_width);
    fill_gemm(kGemmBase + 256, ffn_rows, hidden);
    fill_gemm(kGemmBase + 512, ffn_rows, hidden);
    fill_gemm(kGemmBase + 768, hidden, ffn_rows);
    auto write_addn = [&](std::uint64_t slot, std::uint64_t count) {
        auto* params = reinterpret_cast<AddnParams*>(base + slot);
        params->count = static_cast<std::uint32_t>(count);
        params->pad0 = params->pad1 = params->pad2 = 0;
    };
    write_addn(kAddn0, th);
    write_addn(kAddn1, th);
    auto* rms = reinterpret_cast<RmsParams*>(base + kRmsSlot);
    rms->count = static_cast<std::uint32_t>(hidden);
    rms->rows = static_cast<std::uint32_t>(tokens);
    rms->epsilon = epsilon;
    rms->pad = 0;
    auto* silu = reinterpret_cast<SiluParams*>(base + kSiluSlot);
    silu->count = static_cast<std::uint32_t>(tf);
    silu->pad0 = silu->pad1 = silu->pad2 = 0;

    const std::uint64_t params_base = s.params->GetGPUVirtualAddress();
    const std::uint64_t a_base = s.ybuf->GetGPUVirtualAddress();
    const std::uint64_t b_base = s.fbuf->GetGPUVirtualAddress();

    const UINT attn_groups = static_cast<UINT>((hidden + rpg - 1) / rpg);
    const UINT ffn_groups = static_cast<UINT>((ffn_rows + rpg - 1) / rpg);
    const UINT down_groups = static_cast<UINT>((hidden + rpg - 1) / rpg);
    const UINT addn_groups = static_cast<UINT>((th + 255u) / 256u);
    const UINT silu_groups = static_cast<UINT>((tf + 255u) / 256u);

    s.allocator->Reset();
    s.list->Reset(s.allocator.Get(), gemm_pso);
    s.list->SetComputeRootSignature(s.root.Get());
    if (s.trace) {
        s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
    }

    // 1. attn_output GEMM: A.t0 = W_attn @ att  (input staged on xbuf).
    s.list->SetComputeRootConstantBufferView(0, params_base + kGemmBase);
    s.list->SetComputeRootShaderResourceView(1, ao_q->GetGPUVirtualAddress());
    s.list->SetComputeRootShaderResourceView(2, ao_s);
    s.list->SetComputeRootShaderResourceView(3, s.xbuf->GetGPUVirtualAddress());
    s.list->SetComputeRootShaderResourceView(4, s.xbuf->GetGPUVirtualAddress());
    s.list->SetComputeRootUnorderedAccessView(5, a_base + a_t0);
    s.list->Dispatch(attn_groups, 1, 1);

    // 2. residual: B.h1 = A.t0 + x  (addn reads A.t0, so A becomes an SRV).
    transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    s.list->SetPipelineState(s.pso_addn.Get());
    s.list->SetComputeRootConstantBufferView(0, params_base + kAddn0);
    s.list->SetComputeRootShaderResourceView(1, a_base + a_t0);
    s.list->SetComputeRootShaderResourceView(2, s.wbuf->GetGPUVirtualAddress());
    s.list->SetComputeRootUnorderedAccessView(5, b_base + b_h1);
    s.list->Dispatch(addn_groups, 1, 1);

    // 3. FFN norm over T rows: A.h2 = rmsnorm(B.h1)  (B becomes the SRV).
    transition(s, s.fbuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    s.list->SetPipelineState(s.pso_rms.Get());
    s.list->SetComputeRootConstantBufferView(0, params_base + kRmsSlot);
    s.list->SetComputeRootShaderResourceView(1, b_base + b_h1);
    s.list->SetComputeRootShaderResourceView(2, s.nbuf->GetGPUVirtualAddress());
    s.list->SetComputeRootUnorderedAccessView(5, a_base + a_h2);
    s.list->Dispatch(static_cast<UINT>(tokens), 1, 1);

    // 4/5. gate and up GEMMs from the normed rows into B (no state change
    // between them: both read the same A.h2 and write the same B resource).
    transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(s, s.fbuf.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    s.list->SetPipelineState(gemm_pso);
    s.list->SetComputeRootConstantBufferView(0, params_base + kGemmBase + 256);
    s.list->SetComputeRootShaderResourceView(1, g_q->GetGPUVirtualAddress());
    s.list->SetComputeRootShaderResourceView(2, g_s);
    s.list->SetComputeRootShaderResourceView(3, a_base + a_h2);
    s.list->SetComputeRootShaderResourceView(4, a_base + a_h2);
    s.list->SetComputeRootUnorderedAccessView(5, b_base + b_g);
    s.list->Dispatch(ffn_groups, 1, 1);

    s.list->SetComputeRootConstantBufferView(0, params_base + kGemmBase + 512);
    s.list->SetComputeRootShaderResourceView(1, u_q->GetGPUVirtualAddress());
    s.list->SetComputeRootShaderResourceView(2, u_s);
    s.list->SetComputeRootShaderResourceView(3, a_base + a_h2);
    s.list->SetComputeRootShaderResourceView(4, a_base + a_h2);
    s.list->SetComputeRootUnorderedAccessView(5, b_base + b_u);
    s.list->Dispatch(ffn_groups, 1, 1);

    // 6. SiLU(gate)*up into A.f  (B becomes the SRV again).
    transition(s, s.fbuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    s.list->SetPipelineState(s.pso_silu.Get());
    s.list->SetComputeRootConstantBufferView(0, params_base + kSiluSlot);
    s.list->SetComputeRootShaderResourceView(1, b_base + b_g);
    s.list->SetComputeRootShaderResourceView(2, b_base + b_u);
    s.list->SetComputeRootUnorderedAccessView(5, a_base + a_f);
    s.list->Dispatch(silu_groups, 1, 1);

    // 7. down GEMM: B.d = W_down @ A.f
    transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(s, s.fbuf.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    s.list->SetPipelineState(gemm_pso);
    s.list->SetComputeRootConstantBufferView(0, params_base + kGemmBase + 768);
    s.list->SetComputeRootShaderResourceView(1, d_q->GetGPUVirtualAddress());
    s.list->SetComputeRootShaderResourceView(2, d_s);
    s.list->SetComputeRootShaderResourceView(3, a_base + a_f);
    s.list->SetComputeRootShaderResourceView(4, a_base + a_f);
    s.list->SetComputeRootUnorderedAccessView(5, b_base + b_d);
    s.list->Dispatch(down_groups, 1, 1);

    // 8. final residual: A.fin = B.h1 + B.d
    transition(s, s.fbuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    s.list->SetPipelineState(s.pso_addn.Get());
    s.list->SetComputeRootConstantBufferView(0, params_base + kAddn1);
    s.current_comp = Comp::FfnFused;
    s.list->SetComputeRootShaderResourceView(1, b_base + b_d);
    s.list->SetComputeRootShaderResourceView(2, b_base + b_h1);
    s.list->SetComputeRootUnorderedAccessView(5, a_base + a_fin);
    s.list->Dispatch(addn_groups, 1, 1);

    if (s.trace) {
        s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
    }
    transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_COPY_SOURCE);
    s.list->CopyBufferRegion(s.yread.Get(), 0, s.ybuf.Get(), a_fin, x_bytes);
    transition(s, s.ybuf.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (s.trace) {
        s.list->EndQuery(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 2);
        s.list->ResolveQueryData(s.ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 3,
                                 s.ts_read.Get(), 0);
    }
    s.list->Close();

    if (!execute_and_wait(s)) {
        return false;
    }

    void* mapped = nullptr;
    const D3D12_RANGE read_range{0, x_bytes};
    if (FAILED(s.yread->Map(0, &read_range, &mapped))) {
        return false;
    }
    {
        std::uint64_t dev = 0;
        accumulate_gpu_us(s, dev, 0, 1);
        note_dev(s, Comp::FfnFused, dev);
    }
    std::memcpy(out, mapped, x_bytes);
    const D3D12_RANGE written{0, 0};
    s.yread->Unmap(0, &written);
    s.stats.dispatches += 8;
    ++s.stats.batches;
    return true;
}

bool ffn_prefill_fused(const std::uint8_t* attn_out_w, const std::uint8_t* gate_w,
                       const std::uint8_t* up_w, const std::uint8_t* down_w,
                       const float* norm_weight, const float* x, const float* attn_out,
                       float* out, std::size_t tokens, std::size_t hidden,
                       std::size_t attn_width, std::size_t ffn_rows, float epsilon) noexcept {
    if (!attn_out_w || !gate_w || !up_w || !down_w || !norm_weight || !x || !attn_out || !out) {
        return false;
    }
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!init_locked(s)) {
        return false;
    }
    if (!ffn_prefill_fused_locked(s, attn_out_w, gate_w, up_w, down_w, norm_weight, x, attn_out,
                                  out, tokens, hidden, attn_width, ffn_rows, epsilon)) {
        ++s.stats.fallbacks;
        return false;
    }
    return true;
}

}  // namespace zen::d3d12

#endif  // ZEN_HAS_D3D12
