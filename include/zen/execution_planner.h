#pragma once
#include <cstddef>

namespace zen::planner {
enum class Phase { prefill, decode };
enum class Kernel { fp32_gemv, quantized_gemv, fp32_gemm, quantized_gemm };
struct Plan { Phase phase{Phase::decode}; Kernel kernel{Kernel::fp32_gemv}; unsigned workers{1}; };
class ExecutionPlanner {
public:
    Plan select(Phase phase,bool quantized,std::size_t tokens,std::size_t rows,std::size_t columns) const noexcept;
};
}
