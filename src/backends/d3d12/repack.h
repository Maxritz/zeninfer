#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <thread>

#include <zen/compute_pool.h>

// Q8_0 rows arrive as 34-byte blocks (fp16 scale, then 32 signed bytes). A
// ByteAddressBuffer load of 32 bytes is only aligned when the block index is even,
// so the upload splits the block into an aligned 32-byte quant stream and a
// separate fp32 scale stream. Repacking the whole model is ~1.3 GB of memory
// traffic, which is why it is spread across cores rather than done inline.
namespace zen::d3d12::detail {

inline float half_to_float(std::uint16_t h) {
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

constexpr std::size_t kElemsPerBlock = 32;
constexpr std::size_t kRawBlockBytes = 34;

inline void repack_rows(const std::uint8_t* weights, std::size_t row_begin,
                        std::size_t row_end, std::size_t blocks_per_row,
                        std::uint8_t* q_dst, float* s_dst) {
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

inline void repack_q8_0(const std::uint8_t* weights, std::size_t rows,
                        std::size_t blocks_per_row, std::uint8_t* q_dst, float* s_dst) {
    const unsigned hardware = std::max(1u, std::thread::hardware_concurrency());
    // Below ~32 rows per thread the hand-off costs more than the copy.
    const std::size_t workers =
        std::min<std::size_t>(hardware, std::max<std::size_t>(1, rows / 32));
    if (workers == 1) {
        repack_rows(weights, 0, rows, blocks_per_row, q_dst, s_dst);
        return;
    }

    // The process-wide persistent pool rather than a fresh std::thread per chunk.
    // This runs once per tensor - 112 times for a 1B model - and spawning ~16 threads
    // for each one cost roughly 281ms of a 557ms model load, against a few tens of ms
    // of actual copying. parallel_for hands the caller (begin,end) row ranges, so the
    // chunking comes from the pool and needs no join here.
    zen::ComputePool::instance().parallel_for(
        rows, static_cast<unsigned>(workers),
        [&](std::size_t begin, std::size_t end) {
            repack_rows(weights, begin, end, blocks_per_row, q_dst, s_dst);
        });
}

}  // namespace zen::d3d12::detail
