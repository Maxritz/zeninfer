// ZenInfer HIP (ROCm) backend: Q8_0 compute kernels and host runtime.
//
// This single translation unit holds both the device kernels and the host-side
// residency/launch logic, and is compiled by hipcc. MSVC never includes a HIP
// header: CMake's Windows-Clang guard refuses to mix Clang (which HIP requires)
// with MSVC anywhere else in the same configuration, so the language boundary is
// drawn at the object-file level and this file exports only the plain declarations
// in <zen/hip_backend.h>.
//
// Deviations from the D3D12 backend, all deliberate:
//   * One wave (32 lanes) owns one weight row. RDNA's wavefront is 32, so the row
//     reduction is a shuffle and needs neither shared memory nor a barrier. The
//     D3D12 shader carries a shared-memory fallback tree purely for wave64 parts.
//   * Buffers are read directly as a byte per lane: lane `l` owns element `l` of a
//     32-element Q8_0 block, so a warp's weight load is one contiguous 32-byte
//     transaction and the repacked quant stream needs no 16-byte alignment at all.
//   * Work is queued on one stream and synchronised once per API call instead of
//     one D3D12 fence round trip per command list.

#include <zen/hip_backend.h>

#include <zen/compute_pool.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace zen::hip {
namespace {

using clock_type = std::chrono::steady_clock;

constexpr std::size_t kElemsPerBlock = 32;   // Q8_0 block width
constexpr std::size_t kRawBlockBytes = 34;   // fp16 scale + 32 signed bytes
constexpr std::size_t kLanes = 32;           // one wavefront per weight row
constexpr unsigned kRowsPerBlock = 8;        // 8 warps per 256-thread workgroup
constexpr unsigned kThreadsPerBlock = 256;
constexpr unsigned kTokenTile = 8;
constexpr std::size_t kMinElements = 256u * 1024u;
constexpr std::uint64_t kVRAMBudgetPercent = 80;
// ROCm's __shfl_*_sync requires a mask *type* of 64 bits, but the value must only
// name lanes that exist: on a 32-wide RDNA wave, setting the upper 32 bits leaves
// the reduction effectively inactive and every row keeps just its first lane's
// partial sum. The type is wide, the value stays 32 lanes.
constexpr unsigned long long kWaveMask = 0xFFFFFFFFull;

std::uint64_t us_since(clock_type::time_point start) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(clock_type::now() - start).count());
}

// ---------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------

// Reduce the 32 lanes that share one output row. A shared-memory tree is used
// rather than __shfl_down_sync: on this platform the shuffle builtin silently
// left each row holding only its first lane's partial sum (parity error ~1.0,
// i.e. 1/32 of the true result), while the identical tree in rmsnorm_kernel
// measured 7e-8. Correctness first; the shuffle can be revisited behind a probe.
//
// Every thread must reach the barriers, so callers use an `active` predicate
// instead of an early return.
__device__ inline void row_sum_shared(float* sred, int lane, float acc, float* out) {
    sred[threadIdx.x] = acc;
    __syncthreads();
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        if (lane < s) {
            sred[threadIdx.x] += sred[threadIdx.x + s];
        }
        __syncthreads();
    }
    *out = sred[threadIdx.x];
}

// One wave per output row. Lane `l` owns element `l` of every 32-wide block, so
// the per-block scale folds into that lane's accumulator and no cross-lane work
// is needed until the final row sum.
__global__ void gemv_q8_0_kernel(const std::uint8_t* __restrict__ qbuf,
                                 const float* __restrict__ sbuf,
                                 const float* __restrict__ xvec, float* __restrict__ yvec,
                                 int rows, int blocks_per_row) {
    __shared__ float sred[kThreadsPerBlock];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int local_row = static_cast<int>(threadIdx.x) >> 5;
    const int row = static_cast<int>(blockIdx.x) * static_cast<int>(kRowsPerBlock) + local_row;
    const bool active = row < rows;

    float acc = 0.0f;
    if (active) {
        const std::uint8_t* qrow =
            qbuf + static_cast<std::size_t>(row) * blocks_per_row * kElemsPerBlock;
        const float* srow = sbuf + static_cast<std::size_t>(row) * blocks_per_row;
        // Every block is visited by ALL 32 lanes, each lane taking its own element.
        // Striding b by the lane index instead would give each lane one element of a
        // single block, and the row sum would then cover only `blocks_per_row` of the
        // `blocks_per_row * 32` elements - which measures as block count, not dot.
        for (int b = 0; b < blocks_per_row; ++b) {
            const int qv = static_cast<int>(
                static_cast<std::int8_t>(qrow[static_cast<std::size_t>(b) * kElemsPerBlock + lane]));
            acc += static_cast<float>(qv) * xvec[static_cast<std::size_t>(b) * kElemsPerBlock + lane] *
                   srow[b];
        }
    }

    float total = 0.0f;
    row_sum_shared(sred, lane, acc, &total);
    if (lane == 0 && active) {
        yvec[row] = total;
    }
}

// Batched product for prefill. Same ownership as gemv: one wave per row, lane
// owns one element of each block, and the token loop is the inner accumulator
// dimension so a weight block is read once for the whole tile.
__global__ void gemm_q8_0_kernel(const std::uint8_t* __restrict__ qbuf,
                                 const float* __restrict__ sbuf,
                                 const float* __restrict__ xmat, float* __restrict__ ymat,
                                 int rows, int blocks_per_row, int token_count) {
    __shared__ float sred[kThreadsPerBlock];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int local_row = static_cast<int>(threadIdx.x) >> 5;
    const int row = static_cast<int>(blockIdx.x) * static_cast<int>(kRowsPerBlock) + local_row;
    const bool active = row < rows;

    const int cols = blocks_per_row * static_cast<int>(kElemsPerBlock);
    const std::uint8_t* qrow =
        qbuf + static_cast<std::size_t>(active ? row : 0) * blocks_per_row * kElemsPerBlock;
    const float* srow = sbuf + static_cast<std::size_t>(active ? row : 0) * blocks_per_row;

    for (int t0 = 0; t0 < token_count; t0 += static_cast<int>(kTokenTile)) {
        float acc[kTokenTile];
#pragma unroll
        for (unsigned t = 0; t < kTokenTile; ++t) {
            acc[t] = 0.0f;
        }

        if (active) {
            // Same ownership as gemv: all lanes walk every block, one element each.
            for (int b = 0; b < blocks_per_row; ++b) {
                const int qv = static_cast<int>(static_cast<std::int8_t>(
                    qrow[static_cast<std::size_t>(b) * kElemsPerBlock + lane]));
                const float scale = srow[b];
                const int column = b * static_cast<int>(kElemsPerBlock) + lane;
#pragma unroll
                for (unsigned t = 0; t < kTokenTile; ++t) {
                    const int tok = t0 + static_cast<int>(t);
                    if (tok < token_count) {
                        acc[t] += static_cast<float>(qv) *
                                  xmat[static_cast<std::size_t>(tok) * cols + column] * scale;
                    }
                }
            }
        }

#pragma unroll
        for (unsigned t = 0; t < kTokenTile; ++t) {
            float sum = 0.0f;
            row_sum_shared(sred, lane, acc[t], &sum);
            const int tok = t0 + static_cast<int>(t);
            if (lane == 0 && active && tok < token_count) {
                ymat[static_cast<std::size_t>(tok) * rows + row] = sum;
            }
        }
    }
}

// One workgroup per row. Mirrors rmsnorm.hlsl, including the clamping of the
// row index so every thread reaches the barriers.
__global__ void rmsnorm_kernel(const float* __restrict__ x, const float* __restrict__ w,
                               float* __restrict__ out, int columns, float epsilon) {
    extern __shared__ float partial[];
    const int row = static_cast<int>(blockIdx.x);
    const std::size_t base = static_cast<std::size_t>(row) * columns;

    float ss = 0.0f;
    for (int i = static_cast<int>(threadIdx.x); i < columns; i += static_cast<int>(blockDim.x)) {
        const float v = x[base + i];
        ss += v * v;
    }
    partial[threadIdx.x] = ss;
    __syncthreads();
    for (int s = static_cast<int>(blockDim.x) / 2; s > 0; s >>= 1) {
        if (static_cast<int>(threadIdx.x) < s) {
            partial[threadIdx.x] += partial[threadIdx.x + s];
        }
        __syncthreads();
    }

    const float scale = rsqrtf(partial[0] / static_cast<float>(columns) + epsilon);
    for (int i = static_cast<int>(threadIdx.x); i < columns; i += static_cast<int>(blockDim.x)) {
        out[base + i] = x[base + i] * scale * w[i];
    }
}

// SiLU(gate) * up, elementwise.
__global__ void silu_mul_kernel(const float* __restrict__ gate, const float* __restrict__ up,
                                float* __restrict__ out, int count) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < count) {
        const float g = gate[i];
        out[i] = (g / (1.0f + __expf(-g))) * up[i];
    }
}

// ---------------------------------------------------------------------------
// Host state
// ---------------------------------------------------------------------------

// One allocation per tensor: [quants | 256B pad | scales]. Keeping both regions in
// a single hipMalloc measurably cut allocation cost on the D3D12 side, where every
// committed resource carried its own implicit heap.
struct WeightEntry {
    void* buffer{};
    std::uint64_t buffer_bytes{};
    std::uint64_t scale_offset{};
    std::size_t rows{};
    std::size_t columns{};
};

struct State {
    std::mutex mutex;
    bool ready{};
    bool failed{};
    AdapterInfo adapter{};
    Stats stats{};

    hipStream_t stream{};
    int device{};

    void* xbuf{};
    std::size_t x_cap{};
    void* ybuf{};
    std::size_t y_cap{};
    void* fbuf{};
    std::size_t f_cap{};
    float* yread{};        // pinned host landing buffer
    std::size_t yread_cap{};
    std::uint8_t* stage{}; // host staging, reused across every upload
    std::size_t stage_cap{};

    bool trace{};
    hipEvent_t ev_start{};
    hipEvent_t ev_stop{};

    std::unordered_map<const void*, WeightEntry> weights;
    std::uint64_t weight_bytes{};
};

State& state() {
    static State s;
    return s;
}

bool opted_in() {
    static const bool value = [] {
        const char* backend = std::getenv("ZEN_BACKEND");
        if (backend) {
            std::string name(backend);
            for (char& c : name) {
                c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            }
            if (name == "hip" || name == "rocm" || name == "amd") {
                return true;
            }
        }
        const char* flag = std::getenv("ZEN_HIP");
        return flag && flag[0] == '1';
    }();
    return value;
}

bool gemm_opted_in() {
    static const bool value = [] {
        const char* flag = std::getenv("ZEN_HIP_GEMM_EXPERIMENTAL");
        return flag && flag[0] == '1';
    }();
    return value;
}

// fp16 -> fp32, identical semantics to the D3D12 repack so the two backends
// consume byte-identical weight data.
float half_to_float(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    const std::uint32_t exp = (h >> 10) & 0x1Fu;
    const std::uint32_t mant = h & 0x3FFu;
    std::uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            std::uint32_t m = mant;
            int e = -1;
            do {
                m <<= 1;
                ++e;
            } while ((m & 0x400u) == 0);
            bits = sign | (static_cast<std::uint32_t>(127 - 15 - e) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

void repack_rows(const std::uint8_t* weights, std::size_t row_begin, std::size_t row_end,
                 std::size_t blocks_per_row, std::uint8_t* q_dst, float* s_dst) {
    for (std::size_t r = row_begin; r < row_end; ++r) {
        const std::uint8_t* src = weights + r * blocks_per_row * kRawBlockBytes;
        std::uint8_t* qout = q_dst + r * blocks_per_row * kElemsPerBlock;
        float* sout = s_dst + r * blocks_per_row;
        for (std::size_t b = 0; b < blocks_per_row; ++b) {
            const std::uint8_t* block = src + b * kRawBlockBytes;
            sout[b] = half_to_float(static_cast<std::uint16_t>(block[0] | (block[1] << 8)));
            std::memcpy(qout + b * kElemsPerBlock, block + 2, kElemsPerBlock);
        }
    }
}

void repack_q8_0(const std::uint8_t* weights, std::size_t rows, std::size_t blocks_per_row,
                 std::uint8_t* q_dst, float* s_dst) {
    const unsigned hardware = std::max(1u, std::thread::hardware_concurrency());
    const std::size_t workers = std::min<std::size_t>(hardware, std::max<std::size_t>(1, rows / 32));
    if (workers == 1) {
        repack_rows(weights, 0, rows, blocks_per_row, q_dst, s_dst);
        return;
    }
    // Process-wide persistent pool rather than a fresh thread per tensor. This runs
    // once per tensor and spawning threads for each one cost the D3D12 backend
    // ~281ms of a 557ms model load against a few tens of ms of actual copying.
    zen::ComputePool::instance().parallel_for(
        rows, static_cast<unsigned>(workers),
        [&](std::size_t begin, std::size_t end) {
            repack_rows(weights, begin, end, blocks_per_row, q_dst, s_dst);
        });
}

bool reserve_scratch(State& s, std::size_t x_bytes, std::size_t y_bytes) {
    if (x_bytes > s.x_cap) {
        if (s.xbuf) {
            (void)hipFree(s.xbuf);
            s.xbuf = nullptr;
        }
        if (hipMalloc(&s.xbuf, x_bytes) != hipSuccess) {
            s.x_cap = 0;
            return false;
        }
        s.x_cap = x_bytes;
    }
    if (y_bytes > s.y_cap) {
        if (s.ybuf) {
            (void)hipFree(s.ybuf);
            s.ybuf = nullptr;
        }
        if (hipMalloc(&s.ybuf, y_bytes) != hipSuccess) {
            s.y_cap = 0;
            return false;
        }
        s.y_cap = y_bytes;
    }
    if (y_bytes > s.yread_cap) {
        if (s.yread) {
            (void)hipHostFree(s.yread);
            s.yread = nullptr;
        }
        // Pinned, because the result of every matvec is copied back to the host and
        // a pageable destination would serialise that copy.
        if (hipHostMalloc(reinterpret_cast<void**>(&s.yread), y_bytes) != hipSuccess) {
            s.yread_cap = 0;
            return false;
        }
        s.yread_cap = y_bytes;
    }
    return true;
}

bool init_locked(State& s) {
    if (s.ready) {
        return true;
    }
    if (s.failed) {
        return false;
    }
    auto fail = [&s] {
        s.failed = true;
        return false;
    };

    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || devices < 1) {
        return fail();
    }
    if (hipSetDevice(0) != hipSuccess) {
        return fail();
    }
    s.device = 0;

    hipDeviceProp_t props{};
    if (hipGetDeviceProperties(&props, 0) != hipSuccess) {
        return fail();
    }
    std::snprintf(s.adapter.name, sizeof(s.adapter.name), "%s", props.name);
    std::snprintf(s.adapter.arch, sizeof(s.adapter.arch), "%s", props.gcnArchName);
    s.adapter.dedicated_vram = props.totalGlobalMem;
    s.adapter.shared_memory = props.totalGlobalMem;
    s.adapter.sm_count = props.multiProcessorCount;
    s.adapter.warp_size = props.warpSize;

    if (hipStreamCreate(&s.stream) != hipSuccess) {
        return fail();
    }

    if (s.trace) {
        if (hipEventCreate(&s.ev_start) != hipSuccess || hipEventCreate(&s.ev_stop) != hipSuccess) {
            return fail();
        }
    }

    s.ready = true;
    return true;
}

// Launch the shared gemv shape: 8 rows per workgroup, one wave each.
inline void launch_gemv(const std::uint8_t* q, const float* s, const float* x, float* y,
                        std::size_t rows, std::size_t blocks_per_row, hipStream_t stream) {
    const unsigned groups =
        static_cast<unsigned>((rows + kRowsPerBlock - 1) / kRowsPerBlock);
    gemv_q8_0_kernel<<<groups, kThreadsPerBlock, 0, stream>>>(
        q, s, x, y, static_cast<int>(rows), static_cast<int>(blocks_per_row));
}

WeightEntry* lookup_weights(State& s, const std::uint8_t* weights, std::size_t rows,
                            std::size_t columns);

bool upload_weights(State& s, const std::uint8_t* weights, std::size_t rows, std::size_t columns,
                    WeightEntry& entry) {
    const auto upload_start = clock_type::now();
    const std::size_t blocks_per_row = (columns + kElemsPerBlock - 1) / kElemsPerBlock;
    const std::uint64_t quant_bytes =
        static_cast<std::uint64_t>(rows) * blocks_per_row * kElemsPerBlock;
    const std::uint64_t scale_bytes =
        static_cast<std::uint64_t>(rows) * blocks_per_row * sizeof(float);

    if (s.adapter.dedicated_vram) {
        const std::uint64_t budget = s.adapter.dedicated_vram / 100 * kVRAMBudgetPercent;
        if (s.weight_bytes + quant_bytes + scale_bytes > budget) {
            // Evict the whole residency set rather than thrash: the loader re-uploads
            // in the same order every run, so a partial eviction would just churn.
            for (auto& kv : s.weights) {
                if (kv.second.buffer) {
                    (void)hipFree(kv.second.buffer);
                }
            }
            s.weights.clear();
            s.weight_bytes = 0;
        }
        if (quant_bytes + scale_bytes > budget) {
            return false;
        }
    }

    const std::uint64_t scale_offset = (quant_bytes + 255u) & ~static_cast<std::uint64_t>(255u);
    const std::uint64_t needed = scale_offset + scale_bytes;

    if (hipMalloc(&entry.buffer, needed) != hipSuccess) {
        entry.buffer = nullptr;
        return false;
    }
    entry.buffer_bytes = needed;
    entry.scale_offset = scale_offset;
    s.stats.alloc_us += us_since(upload_start);

    if (needed > s.stage_cap) {
        const auto stage_start = clock_type::now();
        std::free(s.stage);
        s.stage = static_cast<std::uint8_t*>(std::malloc(static_cast<std::size_t>(needed)));
        if (!s.stage) {
            s.stage_cap = 0;
            return false;
        }
        s.stage_cap = static_cast<std::size_t>(needed);
        s.stats.stage_us += us_since(stage_start);
    }

    auto* q_dst = s.stage;
    auto* s_dst = reinterpret_cast<float*>(s.stage + scale_offset);

    const auto repack_start = clock_type::now();
    repack_q8_0(weights, rows, blocks_per_row, q_dst, s_dst);
    s.stats.repack_us += us_since(repack_start);

    const auto transfer_start = clock_type::now();
    // One transfer: the staging buffer already carries [quants | pad | scales].
    if (hipMemcpyAsync(entry.buffer, s.stage, needed, hipMemcpyHostToDevice, s.stream) !=
        hipSuccess) {
        return false;
    }
    if (hipStreamSynchronize(s.stream) != hipSuccess) {
        return false;
    }
    s.stats.transfer_us += us_since(transfer_start);

    s.weight_bytes += needed;
    s.stats.upload_bytes += needed;
    s.stats.upload_us += us_since(upload_start);
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
    // A different shape at the same address means the loader reused the storage.
    if (it != s.weights.end()) {
        if (it->second.buffer) {
            (void)hipFree(it->second.buffer);
        }
        s.weight_bytes -= it->second.buffer_bytes;
        s.weights.erase(it);
    }
    WeightEntry fresh;
    if (!upload_weights(s, weights, rows, columns, fresh)) {
        return nullptr;
    }
    auto& slot = s.weights[weights];
    slot = fresh;
    return &slot;
}

void sync_and_count(State& s, clock_type::time_point launch_start) {
    const auto wait_start = clock_type::now();
    if (s.trace) {
        (void)hipEventRecord(s.ev_stop, s.stream);
    }
    (void)hipStreamSynchronize(s.stream);
    s.stats.wait_us += us_since(wait_start);

    if (s.trace) {
        float ms = 0.0f;
        if (hipEventElapsedTime(&ms, s.ev_start, s.ev_stop) == hipSuccess && ms > 0.0f) {
            s.stats.gpu_us += static_cast<std::uint64_t>(ms * 1000.0f);
        }
    }
    s.stats.launch_us += us_since(launch_start);
    ++s.stats.submits;
}

bool gemv_locked(State& s, const std::uint8_t* weights, const float* x, float* y,
                 std::size_t rows, std::size_t columns) {
    if (columns % kElemsPerBlock != 0) {
        return false;
    }
    WeightEntry* entry = lookup_weights(s, weights, rows, columns);
    if (!entry) {
        return false;
    }
    if (!reserve_scratch(s, columns * sizeof(float), rows * sizeof(float))) {
        return false;
    }
    const std::size_t blocks_per_row = columns / kElemsPerBlock;

    const auto start = clock_type::now();
    if (hipMemcpyAsync(s.xbuf, x, columns * sizeof(float), hipMemcpyHostToDevice, s.stream) !=
        hipSuccess) {
        return false;
    }
    launch_gemv(static_cast<const std::uint8_t*>(entry->buffer),
                reinterpret_cast<const float*>(static_cast<std::uint8_t*>(entry->buffer) +
                                               entry->scale_offset),
                static_cast<const float*>(s.xbuf), static_cast<float*>(s.ybuf), rows,
                blocks_per_row, s.stream);
    const hipError_t launched = hipGetLastError();
    if (launched != hipSuccess) {
        return false;
    }
    sync_and_count(s, start);

    const auto read_start = clock_type::now();
    if (hipMemcpyAsync(s.yread, s.ybuf, rows * sizeof(float), hipMemcpyDeviceToHost, s.stream) !=
        hipSuccess) {
        return false;
    }
    if (hipStreamSynchronize(s.stream) != hipSuccess) {
        return false;
    }
    std::memcpy(y, s.yread, rows * sizeof(float));
    s.stats.readback_us += us_since(read_start);
    ++s.stats.dispatches;
    return true;
}

bool gemv_batch_locked(State& s, const float* x, const GemvTask* tasks, std::size_t count) {
    if (count == 0 || count > kMaxBatch) {
        return false;
    }
    std::size_t total_rows = 0;
    std::size_t cols = tasks[0].columns;
    for (std::size_t i = 0; i < count; ++i) {
        if (tasks[i].columns != cols || cols % kElemsPerBlock != 0) {
            return false;
        }
        total_rows += tasks[i].rows;
    }

    // Every lookup first, then one launch each: a residency eviction mid-batch must
    // not free a buffer a previous launch is still reading.
    WeightEntry* entries[kMaxBatch]{};
    std::size_t offsets[kMaxBatch]{};
    std::size_t span = 0;
    for (std::size_t i = 0; i < count; ++i) {
        entries[i] = lookup_weights(s, tasks[i].weights, tasks[i].rows, tasks[i].columns);
        if (!entries[i]) {
            return false;
        }
        offsets[i] = span;
        span += tasks[i].rows;
    }

    if (!reserve_scratch(s, cols * sizeof(float), span * sizeof(float))) {
        return false;
    }

    const auto start = clock_type::now();
    if (hipMemcpyAsync(s.xbuf, x, cols * sizeof(float), hipMemcpyHostToDevice, s.stream) !=
        hipSuccess) {
        return false;
    }
    const std::size_t blocks_per_row = cols / kElemsPerBlock;
    auto* ybase = static_cast<float*>(s.ybuf);
    for (std::size_t i = 0; i < count; ++i) {
        launch_gemv(static_cast<const std::uint8_t*>(entries[i]->buffer),
                    reinterpret_cast<const float*>(static_cast<std::uint8_t*>(entries[i]->buffer) +
                                                   entries[i]->scale_offset),
                    static_cast<const float*>(s.xbuf), ybase + offsets[i], tasks[i].rows,
                    blocks_per_row, s.stream);
    }
    if (hipGetLastError() != hipSuccess) {
        return false;
    }
    sync_and_count(s, start);

    const auto read_start = clock_type::now();
    if (hipMemcpyAsync(s.yread, s.ybuf, span * sizeof(float), hipMemcpyDeviceToHost, s.stream) !=
        hipSuccess) {
        return false;
    }
    if (hipStreamSynchronize(s.stream) != hipSuccess) {
        return false;
    }
    for (std::size_t i = 0; i < count; ++i) {
        std::memcpy(tasks[i].y, s.yread + offsets[i], tasks[i].rows * sizeof(float));
    }
    s.stats.readback_us += us_since(read_start);
    s.stats.dispatches += count;
    ++s.stats.batches;
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
    if (!reserve_scratch(s, tokens * columns * sizeof(float), tokens * rows * sizeof(float))) {
        return false;
    }

    const auto start = clock_type::now();
    if (hipMemcpyAsync(s.xbuf, input, tokens * columns * sizeof(float), hipMemcpyHostToDevice,
                       s.stream) != hipSuccess) {
        return false;
    }
    const unsigned groups = static_cast<unsigned>((rows + kRowsPerBlock - 1) / kRowsPerBlock);
    gemm_q8_0_kernel<<<groups, kThreadsPerBlock, 0, s.stream>>>(
        static_cast<const std::uint8_t*>(entry->buffer),
        reinterpret_cast<const float*>(static_cast<std::uint8_t*>(entry->buffer) +
                                       entry->scale_offset),
        static_cast<const float*>(s.xbuf), static_cast<float*>(s.ybuf), static_cast<int>(rows),
        static_cast<int>(blocks_per_row), static_cast<int>(tokens));
    if (hipGetLastError() != hipSuccess) {
        return false;
    }
    sync_and_count(s, start);

    const auto read_start = clock_type::now();
    if (hipMemcpyAsync(s.yread, s.ybuf, tokens * rows * sizeof(float), hipMemcpyDeviceToHost,
                       s.stream) != hipSuccess) {
        return false;
    }
    if (hipStreamSynchronize(s.stream) != hipSuccess) {
        return false;
    }
    std::memcpy(output, s.yread, tokens * rows * sizeof(float));
    s.stats.readback_us += us_since(read_start);
    ++s.stats.dispatches;
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

    const auto start = clock_type::now();
    if (hipMemcpyAsync(s.xbuf, x, bytes, hipMemcpyHostToDevice, s.stream) != hipSuccess) {
        return false;
    }
    // The weight vector is tiny and changes only per layer, so it rides in a plain
    // device allocation sized to the widest row seen.
    if (s.fbuf == nullptr || s.f_cap < columns * sizeof(float)) {
        if (s.fbuf) {
            (void)hipFree(s.fbuf);
            s.fbuf = nullptr;
        }
        if (hipMalloc(&s.fbuf, columns * sizeof(float)) != hipSuccess) {
            s.f_cap = 0;
            return false;
        }
        s.f_cap = columns * sizeof(float);
    }
    if (hipMemcpyAsync(s.fbuf, weight, columns * sizeof(float), hipMemcpyHostToDevice, s.stream) !=
        hipSuccess) {
        return false;
    }

    const unsigned threads = 256;
    rmsnorm_kernel<<<static_cast<unsigned>(rows), threads, threads * sizeof(float), s.stream>>>(
        static_cast<const float*>(s.xbuf), static_cast<const float*>(s.fbuf),
        static_cast<float*>(s.ybuf), static_cast<int>(columns), epsilon);
    if (hipGetLastError() != hipSuccess) {
        return false;
    }
    sync_and_count(s, start);

    const auto read_start = clock_type::now();
    if (hipMemcpyAsync(s.yread, s.ybuf, bytes, hipMemcpyDeviceToHost, s.stream) != hipSuccess) {
        return false;
    }
    if (hipStreamSynchronize(s.stream) != hipSuccess) {
        return false;
    }
    std::memcpy(y, s.yread, bytes);
    s.stats.readback_us += us_since(read_start);
    ++s.stats.dispatches;
    return true;
}

bool gemv_ffn_locked(State& s, const std::uint8_t* gate_w, const std::uint8_t* up_w,
                     const std::uint8_t* down_w, const float* x, float* y,
                     std::size_t ffn_rows, std::size_t columns, std::size_t out_rows) {
    if (ffn_rows == 0 || out_rows == 0 || columns % kElemsPerBlock != 0) {
        return false;
    }
    // Resolve every entry up front so an eviction cannot invalidate a buffer that an
    // already-queued launch is reading (the D3D12 F-09 failure mode).
    WeightEntry* gate = lookup_weights(s, gate_w, ffn_rows, columns);
    if (!gate) {
        return false;
    }
    WeightEntry* up = lookup_weights(s, up_w, ffn_rows, columns);
    if (!up) {
        return false;
    }
    WeightEntry* down = lookup_weights(s, down_w, out_rows, ffn_rows);
    if (!down) {
        return false;
    }

    const std::size_t ffn_bytes = ffn_rows * sizeof(float);
    const std::size_t out_bytes = out_rows * sizeof(float);
    if (!reserve_scratch(s, columns * sizeof(float), ffn_bytes + ffn_bytes + out_bytes)) {
        return false;
    }
    if (s.fbuf == nullptr || s.f_cap < ffn_bytes) {
        if (s.fbuf) {
            (void)hipFree(s.fbuf);
            s.fbuf = nullptr;
        }
        if (hipMalloc(&s.fbuf, ffn_bytes) != hipSuccess) {
            s.f_cap = 0;
            return false;
        }
        s.f_cap = ffn_bytes;
    }

    auto* ybase = static_cast<float*>(s.ybuf);
    float* up_slot = ybase + ffn_rows;
    float* out_slot = ybase + 2 * ffn_rows;
    float* silu_slot = static_cast<float*>(s.fbuf);

    const auto start = clock_type::now();
    if (hipMemcpyAsync(s.xbuf, x, columns * sizeof(float), hipMemcpyHostToDevice, s.stream) !=
        hipSuccess) {
        return false;
    }

    const std::size_t ffn_blocks = columns / kElemsPerBlock;
    const std::size_t out_blocks = ffn_rows / kElemsPerBlock;
    // gate and up read the same activation; SiLU combines them in place and ffn_down
    // consumes the result, all on one stream with a single host synchronisation.
    launch_gemv(static_cast<const std::uint8_t*>(gate->buffer),
                reinterpret_cast<const float*>(static_cast<std::uint8_t*>(gate->buffer) +
                                               gate->scale_offset),
                static_cast<const float*>(s.xbuf), ybase, ffn_rows, ffn_blocks, s.stream);
    launch_gemv(static_cast<const std::uint8_t*>(up->buffer),
                reinterpret_cast<const float*>(static_cast<std::uint8_t*>(up->buffer) +
                                               up->scale_offset),
                static_cast<const float*>(s.xbuf), up_slot, ffn_rows, ffn_blocks, s.stream);

    const unsigned silu_threads = 256;
    const unsigned silu_groups = static_cast<unsigned>((ffn_rows + silu_threads - 1) / silu_threads);
    silu_mul_kernel<<<silu_groups, silu_threads, 0, s.stream>>>(ybase, up_slot, silu_slot,
                                                               static_cast<int>(ffn_rows));

    launch_gemv(static_cast<const std::uint8_t*>(down->buffer),
                reinterpret_cast<const float*>(static_cast<std::uint8_t*>(down->buffer) +
                                               down->scale_offset),
                silu_slot, out_slot, out_rows, out_blocks, s.stream);

    if (hipGetLastError() != hipSuccess) {
        return false;
    }
    sync_and_count(s, start);

    const auto read_start = clock_type::now();
    if (hipMemcpyAsync(s.yread, out_slot, out_bytes, hipMemcpyDeviceToHost, s.stream) !=
        hipSuccess) {
        return false;
    }
    if (hipStreamSynchronize(s.stream) != hipSuccess) {
        return false;
    }
    std::memcpy(y, s.yread, out_bytes);
    s.stats.readback_us += us_since(read_start);
    s.stats.dispatches += 4;
    ++s.stats.batches;
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

bool enabled() noexcept {
    if (!opted_in()) {
        return false;
    }
    return init();
}

bool init() noexcept {
    if (!opted_in()) {
        return false;
    }
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    const char* trace = std::getenv("ZEN_HIP_TRACE");
    s.trace = trace && trace[0] == '1';
    return init_locked(s);
}

void shutdown() noexcept {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.ready && s.stream) {
        (void)hipStreamSynchronize(s.stream);
    }
    for (auto& kv : s.weights) {
        if (kv.second.buffer) {
            (void)hipFree(kv.second.buffer);
        }
    }
    s.weights.clear();
    s.weight_bytes = 0;
    if (s.xbuf) { (void)hipFree(s.xbuf); s.xbuf = nullptr; }
    if (s.ybuf) { (void)hipFree(s.ybuf); s.ybuf = nullptr; }
    if (s.fbuf) { (void)hipFree(s.fbuf); s.fbuf = nullptr; }
    if (s.yread) { (void)hipHostFree(s.yread); s.yread = nullptr; }
    std::free(s.stage);
    s.stage = nullptr;
    s.x_cap = s.y_cap = s.f_cap = s.yread_cap = s.stage_cap = 0;
    if (s.ev_start) { (void)hipEventDestroy(s.ev_start); s.ev_start = nullptr; }
    if (s.ev_stop) { (void)hipEventDestroy(s.ev_stop); s.ev_stop = nullptr; }
    if (s.stream) {
        (void)hipStreamDestroy(s.stream);
        s.stream = nullptr;
    }
    s.ready = false;
    // A failed init latches this; clearing it here is what makes shutdown() a real
    // reset instead of a one-way trip into permanent failure.
    s.failed = false;
    s.trace = false;
}

const AdapterInfo* info() noexcept {
    State& s = state();
    if (!s.ready) {
        return nullptr;
    }
    return &s.adapter;
}

const Stats& stats() noexcept { return state().stats; }

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

bool gemv_q8_0_batch(const float* x, const GemvTask* tasks, std::size_t count) noexcept {
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
        // Prefill stays on the CPU unless the experimental flag is set, but pay the
        // residency cost here so the first decoded token does not absorb it.
        (void)lookup_weights(s, weights, rows, columns);
        return false;
    }
    if (!gemm_locked(s, weights, input, output, tokens, rows, columns)) {
        ++s.stats.fallbacks;
        return false;
    }
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

}  // namespace zen::hip
