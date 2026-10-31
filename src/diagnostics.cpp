#include "zen/diagnostics.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace zen::diag {
namespace {
using clock_type = std::chrono::steady_clock;
std::atomic<bool> g_trace{false};
std::atomic<bool> g_breadcrumbs{false};
struct Key {
    std::string component;
    Phase phase;
    bool operator==(const Key& o) const noexcept { return phase == o.phase && component == o.component; }
};
struct KeyHash {
    std::size_t operator()(const Key& k) const noexcept {
        return std::hash<std::string>{}(k.component) ^ (static_cast<std::size_t>(k.phase) << 1);
    }
};
std::mutex g_mutex;
std::unordered_map<Key, Aggregate, KeyHash> g_aggregates;
std::uint64_t now_ns() noexcept {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(clock_type::now().time_since_epoch()).count());
}
const char* phase_name(Phase p) noexcept {
    switch (p) { case Phase::host: return "host"; case Phase::compute: return "compute";
        case Phase::io: return "io"; case Phase::wait: return "wait"; default: return "other"; }
}
constexpr std::size_t kBreadcrumbCapacity = 4096;
struct BreadcrumbSlot {
    std::atomic<std::uint64_t> sequence{UINT64_MAX};
    std::atomic<std::uint64_t> timestamp_ns{0};
    std::atomic<std::uint32_t> event{0};
    std::atomic<std::int64_t> arg0{0}, arg1{0};
};
std::array<BreadcrumbSlot, kBreadcrumbCapacity> g_breadcrumbs_ring{};
std::atomic<std::uint64_t> g_breadcrumb_next{0};
thread_local Scope* t_scope_top = nullptr;
}

void set_trace_enabled(bool v) noexcept { g_trace.store(v, std::memory_order_relaxed); }
bool trace_enabled() noexcept { return g_trace.load(std::memory_order_relaxed); }
void reset_trace() noexcept { std::lock_guard lock(g_mutex); g_aggregates.clear(); }
void record_operation(std::string_view name, Phase phase, double exclusive_us, double inclusive_us) noexcept {
    try {
        std::lock_guard lock(g_mutex);
        auto& a = g_aggregates[Key{std::string(name), phase}];
        if (a.calls == 0) a.min_us = exclusive_us; else a.min_us = std::min(a.min_us, exclusive_us);
        a.max_us = std::max(a.max_us, exclusive_us);
        a.total_us += exclusive_us;
        a.inclusive_total_us += inclusive_us;
        ++a.calls;
    } catch (...) { /* diagnostics must never terminate inference */ }
}
Scope::Scope(std::string_view c, Phase p) noexcept : component_(c), phase_(p), active_(trace_enabled()) {
    if (active_) { parent_ = t_scope_top; t_scope_top = this; start_ns_ = now_ns(); }
}
Scope::~Scope() {
    if (active_) {
        const auto elapsed = now_ns() - start_ns_;
        if (parent_) parent_->child_ns_ += elapsed;
        t_scope_top = parent_;
        const auto exclusive = elapsed > child_ns_ ? elapsed - child_ns_ : 0;
        record_operation(component_, phase_, double(exclusive) / 1000.0, double(elapsed) / 1000.0);
    }
}
void print_trace(std::ostream& out) {
    std::vector<std::pair<Key, Aggregate>> rows;
    { std::lock_guard lock(g_mutex); for (const auto& kv : g_aggregates) rows.push_back(kv); }
    double total = 0; for (const auto& r : rows) total += r.second.total_us;
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second.total_us > b.second.total_us; });
    out << "\nSherlock CPU trace (instrumented scope time; not device time)\n";
    out << std::left << std::setw(22) << "component" << std::setw(10) << "phase" << std::right
        << std::setw(9) << "ops" << std::setw(9) << "%total" << std::setw(14) << "total us"
        << std::setw(13) << "avg us" << std::setw(13) << "min us" << std::setw(13) << "max us"
        << std::setw(16) << "inclusive us" << '\n';
    for (const auto& r : rows) {
        const auto& a = r.second;
        out << std::left << std::setw(22) << r.first.component << std::setw(10) << phase_name(r.first.phase)
            << std::right << std::setw(9) << a.calls << std::setw(8) << std::fixed << std::setprecision(1)
            << (total ? 100.0 * a.total_us / total : 0.0) << '%' << std::setw(14) << std::setprecision(2)
            << a.total_us << std::setw(13) << a.total_us / double(a.calls) << std::setw(13) << a.min_us
            << std::setw(13) << a.max_us << std::setw(16) << a.inclusive_total_us << '\n';
    }
    out << "instrumented scope sum: " << std::fixed << std::setprecision(2) << total << " us\n";
}
void write_trace_csv(const char* path) {
    std::ofstream f(path); if (!f) return;
    f << "component,phase,ops,exclusive_total_us,inclusive_total_us,avg_exclusive_us,min_exclusive_us,max_exclusive_us\n";
    std::lock_guard lock(g_mutex);
    for (const auto& r : g_aggregates) {
        const auto& a = r.second;
        f << '"' << r.first.component << "\"," << phase_name(r.first.phase) << ',' << a.calls << ','
          << a.total_us << ',' << a.inclusive_total_us << ',' << a.total_us / double(a.calls) << ',' << a.min_us << ',' << a.max_us << '\n';
    }
}
double measure_instrumentation_floor_us(std::size_t n) noexcept {
    if (!n) return 0;
    const bool was_enabled = trace_enabled();
    set_trace_enabled(true);
    const auto t0 = now_ns();
    for (std::size_t i = 0; i < n; ++i) { Scope scope("instrumentation.floor", Phase::other); }
    const auto elapsed = now_ns() - t0;
    set_trace_enabled(was_enabled);
    return double(elapsed) / 1000.0 / double(n);
}
void set_breadcrumbs_enabled(bool v) noexcept { g_breadcrumbs.store(v, std::memory_order_relaxed); }
bool breadcrumbs_enabled() noexcept { return g_breadcrumbs.load(std::memory_order_relaxed); }
void breadcrumb(std::uint32_t event, std::int64_t a, std::int64_t b) noexcept {
    if (!breadcrumbs_enabled()) return;
    const auto seq = g_breadcrumb_next.fetch_add(1, std::memory_order_relaxed);
    auto& slot = g_breadcrumbs_ring[seq % kBreadcrumbCapacity];
    slot.sequence.store(UINT64_MAX, std::memory_order_release);
    slot.timestamp_ns.store(now_ns(), std::memory_order_relaxed);
    slot.event.store(event, std::memory_order_relaxed);
    slot.arg0.store(a, std::memory_order_relaxed);
    slot.arg1.store(b, std::memory_order_relaxed);
    slot.sequence.store(seq, std::memory_order_release);
}
void reset_breadcrumbs() noexcept {
    g_breadcrumb_next.store(0, std::memory_order_relaxed);
    for (auto& slot : g_breadcrumbs_ring) slot.sequence.store(UINT64_MAX, std::memory_order_relaxed);
}
void dump_breadcrumbs(std::ostream& out, std::size_t count) {
    const auto end = g_breadcrumb_next.load(std::memory_order_acquire);
    const auto limit = std::min(count, kBreadcrumbCapacity);
    const auto begin = end > limit ? end - limit : 0;
    out << "Breadcrumbs (global, cross-thread, oldest first; event IDs are defined by caller)\n";
    for (auto seq = begin; seq < end; ++seq) {
        const auto& slot = g_breadcrumbs_ring[seq % kBreadcrumbCapacity];
        const auto before = slot.sequence.load(std::memory_order_acquire);
        if (before != seq) continue;
        Breadcrumb b{seq, slot.timestamp_ns.load(std::memory_order_relaxed),
                     slot.event.load(std::memory_order_relaxed),
                     slot.arg0.load(std::memory_order_relaxed), slot.arg1.load(std::memory_order_relaxed)};
        if (slot.sequence.load(std::memory_order_acquire) != seq) continue;
        out << "seq=" << b.sequence << " t_ns=" << b.timestamp_ns << " event=" << b.event
            << " arg0=" << b.arg0 << " arg1=" << b.arg1 << '\n';
    }
}
} // namespace zen::diag
