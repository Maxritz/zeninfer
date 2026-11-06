#include "zen/kernels.h"
#include "zen/diagnostics.h"
#include "zen/resource_governor.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

using clock_type = std::chrono::steady_clock;

static double bench_ms(const std::function<void()>& fn, int warmup, int reps) {
    for (int i=0; i<warmup; ++i) fn();
    const auto t0=clock_type::now();
    for (int i=0; i<reps; ++i) fn();
    const auto t1=clock_type::now();
    return std::chrono::duration<double,std::milli>(t1-t0).count()/reps;
}

static bool close_enough(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size()!=b.size()) return false;
    for (std::size_t i=0;i<a.size();++i) {
        const float d=std::fabs(a[i]-b[i]);
        if (d > 2e-3f * std::max(1.0f,std::fabs(a[i]))) return false;
    }
    return true;
}

int main(int argc, char** argv) {
    int reps=5;
    if (argc>1) reps=std::max(1,std::atoi(argv[1]));
    zen::diag::set_trace_enabled(true);
    zen::diag::set_breadcrumbs_enabled(true);
    zen::ResourceGovernor governor;
    const auto resources = governor.snapshot();
    const auto policy = governor.policy(resources);
    std::cout << "Detected logical CPUs=" << resources.logical_cpus
              << " physical cores=" << resources.physical_cpus
              << " RAM available=" << (resources.available_ram_bytes / (1024.0*1024.0)) << " MiB"
              << " process memory limit=" << (resources.process_memory_limit_bytes / (1024.0*1024.0)) << " MiB"
              << " tier=" << policy.memory_tier
              << " policy workers=" << policy.worker_threads
              << " cache budget=" << (policy.cache_budget_bytes / (1024.0*1024.0)) << " MiB\n";
    std::mt19937 rng(1234567);
    std::uniform_real_distribution<float> dist(-1.0f,1.0f);

    const auto scalar=zen::select_kernels(true);
    const auto selected=zen::select_kernels(false);
    std::cout << "ZenInfer benchmark foundation " << ZEN_VERSION << "\n";
    std::cout << "AVX2+FMA3 runtime support: " << (zen::cpu_has_avx2_fma()?"yes":"no") << "\n";
    std::cout << "Selected kernels: " << selected.name << "\n\n";
    std::cout << std::fixed << std::setprecision(3);

    struct Shape { std::size_t m,k; };
    const Shape gemv_shapes[]={{4096,4096},{8192,4096},{4096,8192},{16384,4096}};
    bool all_checks_passed=true;double min_speedup_observed=std::numeric_limits<double>::infinity();
    std::cout << "GEMV (row-major FP32)\n";
    for (const auto s: gemv_shapes) {
        std::vector<float> a(s.m*s.k), x(s.k), ys(s.m), yv(s.m);
        for (auto& v:a) v=dist(rng);
        for (auto& v:x) v=dist(rng);
        const double ts=bench_ms([&]{zen::diag::Scope scope("gemv.scalar"); scalar.gemv(a.data(),x.data(),ys.data(),s.m,s.k);},1,reps);
        double tv=ts;
        if (selected.gemv!=scalar.gemv)
            tv=bench_ms([&]{zen::diag::Scope scope("gemv.selected"); selected.gemv(a.data(),x.data(),yv.data(),s.m,s.k);},1,reps);
        else yv=ys;
        const bool ok=close_enough(ys,yv); all_checks_passed = all_checks_passed && ok;
        const double speedup=ts/tv;min_speedup_observed=std::min(min_speedup_observed,speedup);
        const double gflops=(2.0*double(s.m)*double(s.k))/(tv*1.0e6);
        const double bytes=double(s.m)*double(s.k)*sizeof(float)+double(s.k)*sizeof(float);
        const double gbps=bytes/(tv*1.0e6);
        std::cout << "  "<<s.m<<"x"<<s.k<<" scalar="<<ts<<" ms selected="<<tv
                  <<" ms speedup="<<speedup<<"x approx="<<gflops<<" GFLOP/s"
                  <<" weight+input="<<gbps<<" GB/s check="<<(ok?"PASS":"FAIL")<<"\n";
    }

    struct GShape { std::size_t m,n,k; };
    const GShape gemm_shapes[]={{128,128,128},{256,256,256},{512,256,512}};
    std::cout << "\nGEMM (row-major FP32, simple 4x8 AVX2 kernel)\n";
    for (const auto s: gemm_shapes) {
        std::vector<float> a(s.m*s.k), b(s.k*s.n), cs(s.m*s.n), cv(s.m*s.n);
        for (auto& v:a) v=dist(rng);
        for (auto& v:b) v=dist(rng);
        const double ts=bench_ms([&]{zen::diag::Scope scope("gemm.scalar"); scalar.gemm(a.data(),b.data(),cs.data(),s.m,s.n,s.k);},1,reps);
        double tv=ts;
        if (selected.gemm!=scalar.gemm)
            tv=bench_ms([&]{zen::diag::Scope scope("gemm.selected"); selected.gemm(a.data(),b.data(),cv.data(),s.m,s.n,s.k);},1,reps);
        else cv=cs;
        const bool ok=close_enough(cs,cv);all_checks_passed=all_checks_passed&&ok;
        const double speedup=ts/tv;min_speedup_observed=std::min(min_speedup_observed,speedup);
        const double ops=2.0*double(s.m)*double(s.n)*double(s.k);
        std::cout << "  "<<s.m<<"x"<<s.n<<"x"<<s.k<<" scalar="<<ts<<" ms selected="<<tv
                  <<" ms speedup="<<speedup<<"x approx="<<(ops/(tv*1.0e6))
                  <<" GFLOP/s check="<<(ok?"PASS":"FAIL")<<"\n";
    }
    if(const char* gate=std::getenv("ZEN_PERF_MIN_SPEEDUP")){char* end=nullptr;const double threshold=std::strtod(gate,&end);if(end&&*end=='\0'&&threshold>0.0&&selected.gemv!=scalar.gemv&&min_speedup_observed<threshold){std::cerr<<"Performance regression gate failed: minimum speedup "<<min_speedup_observed<<"x < "<<threshold<<"x\n";all_checks_passed=false;}else if(selected.gemv==scalar.gemv)std::cout<<"Performance regression gate skipped: selected backend is scalar\n";}
    std::cout << "\nInstrumentation floor (empty timestamp pair): " << zen::diag::measure_instrumentation_floor_us() << " us/op\n";
    zen::diag::print_trace(std::cout);
    zen::diag::write_trace_csv("sherlock_trace.csv");
    zen::diag::breadcrumb(1, static_cast<std::int64_t>(resources.logical_cpus), static_cast<std::int64_t>(policy.worker_threads));
    zen::diag::dump_breadcrumbs(std::cout);
    return all_checks_passed ? 0 : 2;
}
