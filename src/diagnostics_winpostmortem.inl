#if defined(_WIN32)
#include <cstdio>
#include <cstdlib>
#include <windows.h>

static void zen_diag_write_str_literal(HANDLE f, const char* s) noexcept {
    if (!f) return;
    for (std::size_t i = 0; s[i] != '\0'; ++i) {
        char c = s[i];
        DWORD written = 0;
        WriteFile(f, &c, 1, &written, nullptr);
    }
}

static void zen_diag_dump_breadcrumbs_text(HANDLE f) noexcept {
    if (!breadcrumbs_enabled()) {
        zen_diag_write_str_literal(f, "\n[r] breadcrumbs disabled at dump time\n");
        return;
    }
    const std::uint64_t end = g_breadcrumb_next.load(std::memory_order_acquire);
    const std::uint64_t limit = (std::min)(static_cast<std::uint64_t>(256), kBreadcrumbCapacity);
    const std::uint64_t begin = (end > limit) ? (end - limit) : 0;
    char buf[384];
    std::size_t n = 0;
    n += std::sprintf(buf + n, "\nBreadcrumbs (global atomic ring; oldest first; event IDs are defined by caller)\n");
    for (std::uint64_t seq = begin; seq < end; ++seq) {
        const BreadcrumbSlot& slot = g_breadcrumbs_ring[seq % kBreadcrumbCapacity];
        const std::uint64_t before = slot.sequence.load(std::memory_order_acquire);
        if (before != seq) continue;
        const std::uint64_t ts = slot.timestamp_ns.load(std::memory_order_relaxed);
        const std::uint32_t ev = slot.event.load(std::memory_order_relaxed);
        const std::int64_t a0 = slot.arg0.load(std::memory_order_relaxed);
        const std::int64_t a1 = slot.arg1.load(std::memory_order_relaxed);
        n += std::sprintf(buf + n, "seq=%llu t_ns=%llu event=%u arg0=%lld arg1=%lld\n",
                          static_cast<unsigned long long>(seq),
                          static_cast<unsigned long long>(ts),
                          static_cast<unsigned int>(ev),
                          static_cast<long long>(a0),
                          static_cast<long long>(a1));
        if (++seq >= end) break;
    }
    DWORD written = 0;
    WriteFile(f, buf, static_cast<DWORD>(n), &written, nullptr);
}

static void zen_diag_dump_path(char* path, std::size_t path_cap) noexcept {
    const char* dir = std::getenv("ZEN_DIAG_CRASH_DUMP_DIR");
    if (dir && dir[0] && dir[0] != ' ') {
        std::size_t dlen = 0;
        while (dir[dlen] != '\0') ++dlen;
        if (dlen < path_cap - 64) {
            std::sprintf(path, "%s\\zen_diag_crash_breadcrumbs.txt", dir);
            return;
        }
    }
    std::sprintf(path, "zen_diag_crash_breadcrumbs.txt");
}

static void zen_diag_dump_breadcrumbs_safe(HANDLE f) noexcept {
    zen_diag_write_str_literal(f, "ZenInfer process-exit breadcrumb dump\n");
    char buf[128];
    std::size_t n = 0;
    n += std::sprintf(buf + n, "process_id=%lu\n", static_cast<unsigned long>(GetCurrentProcessId()));
    n += std::sprintf(buf + n, "tick_ms64=%llu\n", static_cast<unsigned long long>(GetTickCount64()));
    DWORD written = 0;
    WriteFile(f, buf, static_cast<DWORD>(n), &written, nullptr);
    zen_diag_dump_breadcrumbs_text(f);
    zen_diag_write_str_literal(f, "\n[end]\n");
}

static void zen_diag_atexit_dump() noexcept {
    char path[512];
    zen_diag_dump_path(path, sizeof(path));
    HANDLE f = CreateFileA(path, GENERIC_WRITE, 0, nullptr, OPEN_ALWAYS,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) {
        f = CreateFileA("zen_diag_crash_breadcrumbs.txt", GENERIC_WRITE, 0, nullptr, OPEN_ALWAYS,
                        FILE_ATTRIBUTE_NORMAL, nullptr);
        if (f == INVALID_HANDLE_VALUE) return;
    }
    zen_diag_write_str_literal(f, "ZenInfer atexit breadcrumb dump\n");
    zen_diag_dump_breadcrumbs_safe(f);
    CloseHandle(f);
}

static LONG CALLBACK zen_diag_seh_filter(EXCEPTION_POINTERS* ep) noexcept {
    if (ep && ep->ExceptionRecord) {
        zen::diag::breadcrumb(2,
            static_cast<std::int64_t>(ep->ExceptionRecord->ExceptionCode),
            reinterpret_cast<std::int64_t>(ep->ExceptionRecord->ExceptionAddress));
    } else {
        zen::diag::breadcrumb(2, -2, 0);
    }
    char path[512];
    zen_diag_dump_path(path, sizeof(path));
    HANDLE f = CreateFileA(path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) {
        f = CreateFileA("zen_diag_crash_breadcrumbs.txt", GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                        FILE_ATTRIBUTE_NORMAL, nullptr);
        if (f == INVALID_HANDLE_VALUE) return EXCEPTION_EXECUTE_HANDLER;
    }
    zen_diag_write_str_literal(f, "ZenInfer SEH exception caught in filter\n");
    char buf[256];
    std::size_t n = 0;
    n += std::sprintf(buf + n, "exception_code=%08x\n",
                      static_cast<unsigned int>(ep->ExceptionRecord->ExceptionCode));
    n += std::sprintf(buf + n, "exception_address=%p\n",
                      ep->ExceptionRecord->ExceptionAddress);
    n += std::sprintf(buf + n, "process_id=%lu\n",
                      static_cast<unsigned long>(GetCurrentProcessId()));
    n += std::sprintf(buf + n, "tick_ms64=%llu\n",
                      static_cast<unsigned long long>(GetTickCount64()));
    n += std::sprintf(buf + n, "\n");
    DWORD written = 0;
    WriteFile(f, buf, static_cast<DWORD>(n), &written, nullptr);
    CloseHandle(f);
    return EXCEPTION_EXECUTE_HANDLER;
}

static void zen_diag_terminate_handler() noexcept {
    zen::diag::breadcrumb(3, 0, 0);
    char path[512];
    zen_diag_dump_path(path, sizeof(path));
    HANDLE f = CreateFileA(path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) {
        f = CreateFileA("zen_diag_crash_breadcrumbs.txt", GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                        FILE_ATTRIBUTE_NORMAL, nullptr);
        if (f == INVALID_HANDLE_VALUE) return;
    }
    zen_diag_dump_breadcrumbs_safe(f);
    CloseHandle(f);
}

static void zen_diag_init_once() noexcept {
    zen::diag::breadcrumb(1, 1, 0);
    SetUnhandledExceptionFilter(reinterpret_cast<LPTOP_LEVEL_EXCEPTION_FILTER>(zen_diag_seh_filter));
    std::set_terminate(&zen_diag_terminate_handler);
    std::atexit(&zen_diag_atexit_dump);
}

static const bool zen_diag_postmortem_init = (zen_diag_init_once(), true);

#endif
