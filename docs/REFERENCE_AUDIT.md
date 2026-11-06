# ZenInfer reference audit

Date: 2026-10-30
Scope: Windows x64 CPU-only inference, Zen 3 target, AVX2/FMA3, RAM-resident hot MoE experts and NVMe-backed cold experts.

## Directly actionable references

### Goto & van de Geijn, Anatomy of High-Performance Matrix Multiplication
- https://www.cs.utexas.edu/~pingali/CS378/2008sp/papers/gotoPaper.pdf
- Use: layered memory model, packed panels, register microkernel, cache/TLB-aware blocking.
- Do not copy historical block sizes. Derive sizes from the target Zen 3 topology and benchmark.
- Design consequence: separate macro-scheduling, panel packing, and microkernel implementation. GEMM prefill should use packed panels and register tiles; decode GEMV needs a different path and should not pay GEMM packing costs.

### BLIS
- https://github.com/flame/blis
- https://github.com/flame/blis/blob/master/docs/KernelsHowTo.md
- https://github.com/flame/blis/blob/master/docs/FAQ.md
- Use: microkernel contract, packed micropanels, control-tree separation, runtime context for kernel/blocksize selection, edge handling.
- Do not take a dependency on BLIS for the first implementation. Reimplement only the needed concepts in our own runtime.
- Important distinction: BLIS GEMM microkernel is an outer-product update of a small MR x NR tile; the macro-kernel handles packing and cache blocks.

### Fast MMM on CPU (Siboehm)
- https://siboehm.com/articles/22/Fast-MMM-on-CPU
- Use: loop-order effects, register blocking, vectorisation, compiler-output inspection and controlled benchmark progression.
- Treat its measured speedups as specific to its CPU/compiler/shapes, not Zen 3 performance claims.

### Par4All / polyhedral loop transformation
- https://github.com/Par4All/par4all
- Related foundational locality/parallelism work: https://doi.org/10.1145/1379022.1375595
- Use selectively: dependence analysis, loop interchange, tiling, fusion/fission and parallel scheduling for regular kernels.
- Do not add Par4All as a runtime dependency. First use hand-written kernels and compiler diagnostics; consider source generation only after a stable operator IR exists.

## References that are informative but not implementation blueprints

### arXiv:2412.08902 — HC-SpMM
- https://arxiv.org/abs/2412.08902
- This is a sparse matrix-matrix method for graph workloads using hybrid GPU cores (Tensor cores + CUDA cores).
- CPU read-across: classify sparse tiles by density/regularity, group work by structure, reduce irregular memory access, and use a measured cost model.
- Not transferable: CUDA/Tensor-core partitioning and reported GPU speedups. It is not a Zen 3 CPU kernel design.

### arXiv:2409.10661 — A Study of Performance Programming of CPU, GPU accelerated Computers and SIMD Architecture
- https://arxiv.org/abs/2409.10661
- Broad survey/background. It does not establish a concrete kernel algorithm or target-specific performance claim for ZenInfer.

### arXiv:2507.19723 — Accelerating Matrix Multiplication: A Performance Comparison Between Multi-Core CPU and GPU
- https://arxiv.org/abs/2507.19723
- The paper compares sequential C++, OpenMP CPU and CUDA GPU implementations on one heterogeneous platform and square matrix sizes.
- Use only as a reminder to separate sequential, threaded CPU and GPU baselines. Its speedup figures are not applicable to Zen 3, GEMV, quantised inference or MoE paging.

### Quanta: matrix multiplication closer to ideal
- https://www.quantamagazine.org/new-breakthrough-brings-matrix-multiplication-closer-to-ideal-20240307/
- Theoretical/asymptotic matrix multiplication is not automatically useful for the small/skinny and quantised matrix-vector operations dominating batch-one decode. Revisit only if a practical algorithm has a favourable crossover on target shapes and precision.

### LAPACK / ScaLAPACK
- https://en.wikipedia.org/wiki/LAPACK
- https://en.wikipedia.org/wiki/ScaLAPACK
- Borrow blocked algorithm organisation, workspace discipline, and distributed data-layout concepts where useful. Do not port their broad solver stacks into an inference runtime.

### FFT
- https://en.wikipedia.org/wiki/Fast_Fourier_transform
- Not a replacement for transformer GEMV/GEMM. Investigate only for explicitly structured transforms (e.g. Walsh-Hadamard/butterfly rotations) if a model format or quantisation method requires them.

### DirectX WARP
- https://learn.microsoft.com/en-us/windows/win32/direct3darticles/directx-warp
- Software rasterisation implementation for Direct3D. Not a general CPU matrix/inference backend. Excluded from the CPU-only engine.

## Quake-engine read-across

References:
- https://doomgate.gamers.org/dEngine/quake/papers/ddjpvs.html
- https://www.fabiensanglard.net/quakeSource/quakeSourceRendition.php

Quake's PVS is precomputed from static map visibility and stored as compressed bitsets. This is not equivalent to MoE routing, which depends on live hidden states and can change token by token.

Potentially reusable systems ideas:
- compressed bitset representation for expert co-occurrence/working-set hints;
- precomputed adjacency or transition statistics from recorded router traces;
- two-stage candidate prefetch: cheap candidate set, then exact live router decision;
- skip prefetching experts outside a conservative candidate set.

Hard correctness rule: prediction may affect I/O timing only. It must not change the model's exact router output or expert selection. Prefetch misses must fall back to the exact expert on demand.

## User-supplied ACM DOI

`10.1145/3789692.3789756` could not be verified from the accessible ACM page during this audit (HTTP access denied), and search did not return a reliable title/abstract. Do not attribute a technique to it until the PDF or bibliographic record is available. The separately mentioned local file `1-s2.0-S1877050919318484-main.pdf.txt` is not present in this workspace; attach it for review.

## Consequences for ZenInfer

1. Maintain distinct decode and prefill paths. Batch-one decode is GEMV/skinny-GEMM territory; prefill benefits from panel packing and GEMM/grouped GEMM as token count rises.
2. Dispatch based on measured shape/cost, not a fixed slogan. GEMV is the decode default, not a universal winner.
3. For MoE prefill, group tokens by exact selected expert, then run grouped expert work. Keep the chat-template/instruction boundary intact if any approximation is ever considered; default implementation must preserve exact routing.
4. For MoE decode, model expert cost as bytes moved + storage latency + compute + scheduler overhead. Report bytes/token and cache misses, not just tokens/s.
5. Separate logical expert routing from physical RAM residency and prefetch state. Disk timing must not affect model outputs.
6. Build the CPU microbenchmark suite before selecting tile sizes, prefetch depth, worker count or cache policy.
