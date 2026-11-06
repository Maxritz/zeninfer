#pragma once
#include <cstdint>
namespace zen {
struct ResourceSnapshot { unsigned logical_cpus{}, physical_cpus{}; std::uint64_t total_ram_bytes{}, available_ram_bytes{}, process_memory_limit_bytes{}; };
struct ResourcePolicy { unsigned worker_threads{1}, prefetch_workers{}; std::uint64_t cache_budget_bytes{}, transient_budget_bytes{}; const char* memory_tier{"constrained"}; };
class ResourceGovernor { public: ResourceSnapshot snapshot() const noexcept; ResourcePolicy policy(const ResourceSnapshot&) const noexcept; ResourcePolicy current_policy() const noexcept; };
}
