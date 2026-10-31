# ZI-011: persistent compute-pool concurrency checks

## Scope

Exercise the existing persistent worker pool under simultaneous callers and verify worker exceptions propagate to the initiating call. No pool architecture rewrite is included.

## Acceptance tests added

- Four caller threads concurrently invoke `parallel_for` with four requested partitions each.
- Each caller has an isolated atomic visit counter; each of the four counters must equal 1,000 after all callers join.
- A worker partition throws `std::runtime_error`; the exception must be rethrown to the caller.
- Existing exact-once partition coverage and deterministic single-worker sum remain.

## GREEN results

- Scalar build: `cmake --build build-scalar -j2`; `ctest --test-dir build-scalar --output-on-failure`; 8/8 passed.
- AVX2 build: `cmake --build build-avx2 -j4`; `ctest --test-dir build-avx2 --output-on-failure`; 8/8 passed.
- Focused core test also passed directly in both configurations.

## Review and limitations

- The current pool uses a process-wide queue and per-call completion state. The new concurrent-caller test confirms isolation for this exercised workload.
- Exception propagation is verified for a worker-partition exception.
- Cancellation is not implemented. Shutdown/lifecycle, queue saturation, race-detector and scaling tests across a broad worker-count matrix remain open.
- No ThreadSanitizer run was completed. Windows/MSVC was not run.

## Conclusion

ZI-011 remains **PARTIAL**. Concurrent invocation and worker exception propagation now have direct regression coverage; cancellation, broader stress/race testing and scaling evidence remain outstanding.
