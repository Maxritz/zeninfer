#include "zen/execution_planner.h"
#include "zen/resource_governor.h"
#include <algorithm>

namespace zen::planner {
Plan ExecutionPlanner::select(Phase phase,bool quantized,std::size_t tokens,std::size_t rows,std::size_t columns) const noexcept {
    static const ResourcePolicy policy=[](){ResourceGovernor governor;return governor.current_policy();}();
    const bool batch=phase==Phase::prefill&&tokens>1;
    Plan p;p.phase=phase;
    p.kernel=batch?(quantized?Kernel::quantized_gemm:Kernel::fp32_gemm):(quantized?Kernel::quantized_gemv:Kernel::fp32_gemv);
    const std::size_t parallel_units=batch?rows:rows;
    // ZI-P5: batched prefill is latency-bound inside the Q8_0 GEMM (measured
    // 322 GF/s at 16 physical workers vs 437 GF/s at 32 logical ones), so the
    // batch path uses every logical CPU while decode keeps the memory-aware
    // physical-core policy.
    static const unsigned logical_workers=[](){
        ResourceGovernor governor;const unsigned n=governor.snapshot().logical_cpus;
        return std::max(1u,std::min(n,32u));
    }();
    const unsigned budget=std::max(policy.worker_threads,logical_workers);
    p.workers=(parallel_units>=32&&columns>=128)?std::max(1u,std::min<unsigned>(budget,static_cast<unsigned>(std::min<std::size_t>(parallel_units,512)))):1u;
    return p;
}
}
