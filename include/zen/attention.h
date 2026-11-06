#pragma once
#include <cstdint>
namespace zen::attention {
// ALiBi head slopes as defined by the reference power-of-two construction.
float alibi_head_slope(std::uint32_t head, std::uint32_t head_count) noexcept;
// Additive causal bias for query position q and key position k (k <= q).
float alibi_bias(std::uint32_t head, std::uint32_t head_count,
                 std::uint32_t query_position, std::uint32_t key_position) noexcept;
}
