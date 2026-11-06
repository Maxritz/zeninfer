# ZenInfer build status

The latest recorded verification is ZenInfer 0.4.5-alpha. See:

- `BUILD_VERIFIED.txt` for the short verification summary.
- `BUILD_STATUS_0.4.5.md` for compiler, platform, fixture and CTest details.
- `PERFORMANCE_0.4.5-alpha.md` for prefill and token-generation throughput.
- `REMAINING_WORK.md` for production features that are still incomplete.

The Linux GCC 14.2 Release AVX2 and scalar-only builds each passed 6/6 CTests.
Windows/MSVC has not been executed in this environment. Synthetic architecture
micro-model tests do not establish production model compatibility or reference
parity.
