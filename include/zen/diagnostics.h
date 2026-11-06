#pragma once
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string_view>

namespace zen::diag {

enum class Phase : std::uint8_t { host, compute, io, wait, other };

struct Aggregate {
    std::uint64_t calls{};
    double total_us{}; // exclusive time, excludes nested scopes
    double inclusive_total_us{};
    double min_us{};
    double max_us{};
};

// Runtime-switchable operation tracing. Disabled scopes perform one atomic load.
void set_trace_enabled(bool enabled) noexcept;
bool trace_enabled() noexcept;
void reset_trace() noexcept;
void record_operation(std::string_view component, Phase phase, double exclusive_us, double inclusive_us) noexcept;
void print_trace(std::ostream& out);
void write_trace_csv(const char* path);
double measure_instrumentation_floor_us(std::size_t samples = 256) noexcept;

class Scope final {
public:
    explicit Scope(std::string_view component, Phase phase = Phase::compute) noexcept;
    ~Scope();
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
private:
    std::string_view component_;
    Phase phase_;
    std::uint64_t start_ns_{};
    std::uint64_t child_ns_{};
    Scope* parent_{};
    bool active_{};
};

// Breadcrumbs are deliberately compact: fixed-size global atomic ring, no allocation.
// Use stable numeric event IDs and small integer arguments in hot paths.
struct Breadcrumb {
    std::uint64_t sequence{};
    std::uint64_t timestamp_ns{};
    std::uint32_t event{};
    std::int64_t arg0{};
    std::int64_t arg1{};
};

void set_breadcrumbs_enabled(bool enabled) noexcept;
bool breadcrumbs_enabled() noexcept;
void breadcrumb(std::uint32_t event, std::int64_t arg0 = 0,
                std::int64_t arg1 = 0) noexcept;
void dump_breadcrumbs(std::ostream& out, std::size_t most_recent = 64);
void reset_breadcrumbs() noexcept;

} // namespace zen::diag
