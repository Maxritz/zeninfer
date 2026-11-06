#pragma once
#include <cstddef>
#include <cstdint>

// AMD HIP (ROCm) compute backend for the Q8_0 matrix kernels.
// Only compiled into the build when ZEN_HAS_HIP is defined.
//
// This header intentionally contains no <hip/...> includes. The whole backend -
// host logic and device kernels alike - is compiled by hipcc into a single object
// file, and MSVC translation units (quant.cpp, the parity tools) only ever see
// these plain declarations. Mixing Clang and MSVC inside one CMake configuration
// is rejected outright by CMake's Windows-Clang guard, so the boundary has to be
// at the translation-unit level rather than at the header level.
namespace zen::hip {

struct AdapterInfo {
    char name[128]{};
    std::uint64_t dedicated_vram{};
    std::uint64_t shared_memory{};
    int sm_count{};    // compute units reported by the runtime
    int warp_size{};   // 32 on RDNA
    char arch[64]{};   // gcnArchName, e.g. gfx1201
};

struct Stats {
    std::uint64_t dispatches{};
    std::uint64_t submits{};
    std::uint64_t batches{};
    std::uint64_t upload_bytes{};
    std::uint64_t weight_bytes{};
    std::uint64_t upload_us{};
    std::uint64_t alloc_us{};      // device allocation + staging growth
    std::uint64_t stage_us{};
    std::uint64_t repack_us{};
    std::uint64_t transfer_us{};
    std::uint64_t wait_us{};       // stream synchronisation only
    std::uint64_t launch_us{};     // kernel launch + argument marshalling
    std::uint64_t readback_us{};
    std::uint64_t gpu_us{};        // event-measured dispatch time, trace only
    std::uint64_t fallbacks{};
};

// True when the process asked for this backend (ZEN_BACKEND=hip|rocm|amd or
// ZEN_HIP=1) *and* a usable device exists. Initialisation is lazy, cached and
// thread-safe; a failed init is remembered so later calls stay on the CPU.
bool enabled() noexcept;
bool init() noexcept;
void shutdown() noexcept;
const AdapterInfo* info() noexcept;
const Stats& stats() noexcept;

// Weights stay resident in VRAM, keyed by the host pointer the loader handed out
// plus the (rows, columns) shape. The contract is the same as the D3D12 backend's:
// a tensor keeps its host allocation for the model's lifetime, so a distinct tensor
// always arrives at a distinct address. Freeing weights and reusing the same address
// for different data of the same shape would return the stale residency.
// Q8_0 rows are repacked on upload because a raw GGML block is 34 bytes, which
// leaves every odd block index misaligned for a 16-byte vector load.
bool gemv_q8_0(const std::uint8_t* weights, const float* x, float* y,
               std::size_t rows, std::size_t columns) noexcept;
bool gemm_q8_0(const std::uint8_t* weights, const float* input, float* output,
               std::size_t tokens, std::size_t rows, std::size_t columns) noexcept;

// One matvec target for a batched submission. All tasks in a batch share the
// input vector, which is exactly the Q|K|V and gate|up pattern in a block.
struct GemvTask {
    const std::uint8_t* weights;
    float* y;
    std::size_t rows;
    std::size_t columns;
};
constexpr std::size_t kMaxBatch = 8;

bool gemv_q8_0_batch(const float* x, const GemvTask* tasks, std::size_t count) noexcept;

// Root-mean-square normalisation on the device.
bool rms_norm(const float* x, const float* weight, float* y, std::size_t rows,
              std::size_t columns, float epsilon) noexcept;

// gate|up|SiLU|down as a single submission, activation staying in VRAM.
bool gemv_q8_0_ffn(const std::uint8_t* gate, const std::uint8_t* up, const std::uint8_t* down,
                   const float* x, float* y, std::size_t ffn_rows, std::size_t columns,
                   std::size_t out_rows) noexcept;

}  // namespace zen::hip
