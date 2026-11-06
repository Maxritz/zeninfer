// TEMP harness (not shipped, not in CMake): shared-buffer read bandwidth.
// 16 threads sequentially scan one shared buffer (uint loads + sum).
// Sizes mirror the T=171 prefill working set:
//   1.4 MB = 171*2048 floats (gate/up input tile)
//   5.6 MB = 171*8192 floats (down-proj input / gate output tile)
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

static double bench_passes(const uint32_t* buf, size_t n_u32, int nthreads, int passes) {
    std::atomic<bool> go{false};
    std::vector<uint64_t> sums(nthreads, 0);
    std::vector<std::thread> th;
    th.reserve(nthreads);
    for (int t = 0; t < nthreads; ++t)
        th.emplace_back([&, t] {
            while (!go.load(std::memory_order_acquire)) {}
            uint64_t s = 0;
            for (int p = 0; p < passes; ++p)
                for (size_t i = 0; i < n_u32; ++i) s += buf[i];
            sums[t] = s;
        });
    auto t0 = std::chrono::steady_clock::now();
    go.store(true, std::memory_order_release);
    for (auto& x : th) x.join();
    auto t1 = std::chrono::steady_clock::now();
    uint64_t total = 0; for (auto s : sums) total += s;
    printf("  checksum=%llu (DCE guard)\n", (unsigned long long)total);
    return std::chrono::duration<double>(t1 - t0).count();
}

int main() {
    const size_t sizes[] = {171*2048*4, 171*8192*4};
    const char* names[] = {"FFN-small 1.4MB (171x2048 fp32)", "FFN-down 5.6MB (171x8192 fp32)"};
    for (int c = 0; c < 2; ++c) {
        size_t bytes = sizes[c];
        size_t n_u32 = bytes / 4;
        std::vector<uint32_t, std::allocator<uint32_t>> buf(n_u32);
        // 64B align not guaranteed by vector; good enough for BW scout.
        for (size_t i = 0; i < n_u32; ++i) buf[i] = (uint32_t)(i * 2654435761u + 97u);
        printf("%s\n", names[c]);
        for (int nt : {1, 8, 16}) {
            int passes = (nt == 16 && c == 0) ? 800 : (nt == 16 && c == 1) ? 200 : (nt == 1 && c == 0) ? 2000 : 500;
            double best = 1e30, med = 0;
            double samples[3] = {0,0,0};
            for (int rep = 0; rep < 3; ++rep) {
                double sec = bench_passes(buf.data(), n_u32, nt, passes);
                samples[rep] = sec;
                if (sec < best) best = sec;
            }
            // median
            double a=samples[0],b=samples[1],d=samples[2];
            med = a<b ? (b<d?b:d) : (a<d?a:d);
            double total_bytes = (double)bytes * passes * nt;
            double agg = total_bytes / med / 1e9;
            printf("  threads=%2d passes=%d median=%.3fs agg=%.1f GB/s per-thread=%.2f GB/s (best %.1f GB/s)\n",
                   nt, passes, med, agg, agg / nt, total_bytes / best / 1e9);
        }
        printf("\n");
    }
    return 0;
}
