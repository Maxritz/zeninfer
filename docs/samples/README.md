# Sherlock diagnostic samples

This directory contains small, human-readable examples of ZenInfer diagnostic output.

- `sherlock_trace_smoke.csv` is a schema/example trace, not a performance benchmark.
- Generate a fresh trace by running `zen_bench` from a writable working directory. The benchmark writes `sherlock_trace.csv` in its current directory.
- Trace timings include instrumentation overhead. Compare runs on the same host, build configuration and workload; do not treat the sample as a performance guarantee.
