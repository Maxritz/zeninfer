# ZI-006: software IEEE binary16 conversion

## Scope

Implement and verify software conversion in both directions between IEEE-754 binary16 bit patterns and C++ `float` (binary32). This does not add native FP16 arithmetic and does not change the excluded AVX-512/ARM scope.

## Plan and RED

- Public API: `zen::quant::encode_fp16` alongside the existing `decode_fp16`.
- Acceptance: all 65,536 half encodings decode and round-trip; finite values and signed zero preserve exact bits; infinities remain infinities; NaNs remain NaNs; float-to-half conversion follows round-to-nearest, ties-to-even at normal, subnormal and overflow boundaries.
- RED command: `cmake --build build-scalar --target zen_core_tests -j2` after adding the API declaration and tests but before implementation.
- RED result: linker failure, `undefined reference to zen::quant::encode_fp16(float const*, unsigned short*, unsigned long)`. This proves the test reaches the missing implementation.

## Implementation

- Added `encode_fp16` to `include/zen/quant.h`.
- Added software float32-to-binary16 conversion in `src/quant.cpp`.
- Handles signed zero, normal values, subnormals, round-to-nearest/ties-to-even, overflow to infinity, infinities and NaN payloads.
- Existing decode path remains software binary16-to-float32.
- No AVX-512 or ARM/NEON code added.

## GREEN results

Environment: Linux x86_64, C++20, CMake project.

- `cmake -S . -B build-scalar -DZEN_BUILD_AVX2=OFF -DZEN_BUILD_TESTS=ON -DZEN_BUILD_BENCH=ON -DZEN_BUILD_SERVER=ON`
- `cmake --build build-scalar -j2`
- `ctest --test-dir build-scalar --output-on-failure`
- Result: 8/8 passed.
- `cmake -S . -B build-avx2 -DZEN_BUILD_AVX2=ON -DZEN_BUILD_TESTS=ON -DZEN_BUILD_BENCH=ON -DZEN_BUILD_SERVER=ON`
- `cmake --build build-avx2 -j4`
- `ctest --test-dir build-avx2 --output-on-failure`
- Result: 8/8 passed.
- `zen_core_tests` now checks exhaustive binary16 decode/encode round-trip across all 65,536 patterns (with NaN class preservation) and selected ties-to-even/subnormal/overflow cases.

## Review and limitations

- This is software conversion, not native FP16 arithmetic. Zen 3 and Meteor Lake do not gain general packed FP16 arithmetic from this change.
- No throughput claim is made.
- Windows/MSVC was not run in this environment.
- The exhaustive test validates round-trip and selected rounding boundaries. A separate independently generated full float32-to-half differential corpus remains useful.

## Conclusion

ZI-006 is **PARTIAL**: software binary16 conversion and deterministic boundary/exhaustive round-trip tests are implemented. Native arithmetic remains outside the supported hardware scope unless a non-excluded ISA path is later added.
