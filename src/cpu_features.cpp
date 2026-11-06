#include "zen/kernels.h"

#if defined(_MSC_VER)
  #include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
  #include <cpuid.h>
#endif

namespace zen {
bool cpu_has_avx2_fma() noexcept {
#if defined(_MSC_VER) && defined(_M_X64)
    int regs[4]{};
    __cpuidex(regs, 1, 0);
    const bool osxsave = (regs[2] & (1 << 27)) != 0;
    const bool avx = (regs[2] & (1 << 28)) != 0;
    const bool fma = (regs[2] & (1 << 12)) != 0;
    if (!(osxsave && avx && fma)) return false;
    const unsigned __int64 xcr0 = _xgetbv(0);
    if ((xcr0 & 0x6) != 0x6) return false;
    __cpuidex(regs, 7, 0);
    return (regs[1] & (1 << 5)) != 0;
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    unsigned eax=0, ebx=0, ecx=0, edx=0;
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return false;
    const bool osxsave = (ecx & bit_OSXSAVE) != 0;
    const bool avx = (ecx & bit_AVX) != 0;
    const bool fma = (ecx & bit_FMA) != 0;
    if (!(osxsave && avx && fma)) return false;
    unsigned xcr0_lo=0, xcr0_hi=0;
    __asm__ volatile("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
    if ((xcr0_lo & 0x6) != 0x6) return false;
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return false;
    return (ebx & bit_AVX2) != 0;
#else
    return false;
#endif
}

KernelSet select_kernels(bool force_scalar) noexcept {
#ifdef ZEN_HAS_AVX2_KERNELS
    if (!force_scalar && cpu_has_avx2_fma())
        return {"avx2_fma3", gemv_avx2, gemm_avx2};
#else
    (void)force_scalar;
#endif
    return {"scalar", gemv_scalar, gemm_scalar};
}
}
