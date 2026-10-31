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
    p.workers=(parallel_units>=32&&columns>=128)?std::max(1u,std::min<unsigned>(policy.worker_threads,static_cast<unsigned>(std::min<std::size_t>(parallel_units,512)))):1u;
    return p;
}
}
