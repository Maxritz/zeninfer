#pragma once
#include <cstddef>
#include <functional>

namespace zen {
// Process-wide persistent CPU worker pool. The calling thread participates in
// each range so small jobs do not pay a queue-only dispatch penalty.
class ComputePool {
public:
    static ComputePool& instance();
    ComputePool(const ComputePool&) = delete;
    ComputePool& operator=(const ComputePool&) = delete;
    unsigned worker_count() const noexcept;
    void parallel_for(std::size_t count, unsigned requested_workers,
                      const std::function<void(std::size_t,std::size_t)>& fn);
private:
    ComputePool();
    ~ComputePool();
    struct Impl;
    Impl* impl_{};
};
}
