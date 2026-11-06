# DirectX 12 Backend Optimization Plan for RDNA4 (RX 9070 XT)

## Current State Analysis

**Measured Performance (from docs):**
- Decode: 122.1 tok/s (theoretical ceiling at Q8_0: 484 tok/s)
- GPU utilization: ~23% of peak bandwidth (149 GB/s of 644 GB/s)
- Submission overhead: ~70 μs per submit, 49 submits/token
- Fence wait: 185 μs average, 52.7% idle time
- 13 dispatches per layer in `layer_decode`

**Root Causes:**
1. **Per-submission overhead dominates** - 70μs fixed cost × 49 submits = 3.4ms/token
2. **Barrier overhead** - Legacy ResourceBarrier with full transitions
3. **No Work Graphs** - Can't fuse entire layer into single GPU work
4. **Staging buffer copies** - CPU→UPLOAD→DEFAULT adds latency
5. **No Shader Execution Reordering** - Independent dispatches serialize
6. **Suboptimal dispatch sizes** - Some kernels underutilize CUs

---

## Optimization Targets (RDNA4 Features)

### 1. Enhanced Barriers (D3D12_FEATURE_D3D12_OPTIONS12)
- Replace `ResourceBarrier` with `Barrier` API
- Use `D3D12_BARRIER_LAYOUT` and `D3D12_BARRIER_SYNC` for precise synchronization
- Eliminate redundant transitions, enable split barriers
- **Expected gain:** 20-30% reduction in barrier overhead

### 2. GPU Upload Heaps (D3D12_HEAP_TYPE_GPU_UPLOAD from OPTIONS16)
- Enable `ZEN_D3D12_GPU_UPLOAD=1` by default when supported
- Small buffers (params, activations, cos/sin tables) stay in VRAM
- Eliminate PCIe crossing for per-token data
- **Expected gain:** 5-10% decode throughput

### 3. Work Graphs (D3D12_WORK_GRAPHS_TIER from OPTIONS21)
- Fuse entire `layer_decode` (13 dispatches) into single Work Graph
- Single submission per layer instead of 1
- GPU-driven scheduling, no CPU round-trips between dispatches
- **Expected gain:** 2-3x decode throughput (target 250-350 tok/s)

### 4. Shader Execution Reordering (OPTIONS22)
- Enable `D3D12_SHADER_EXECUTION_REORDERING_ACTUALLY_REORDERS`
- Independent dispatches (gate/up, rope q/k) run concurrently
- **Expected gain:** 10-15% on FFN and attention

### 5. ExecuteIndirect Improvements (OPTIONS21)
- Use `ExecuteIndirect` for multi-dispatch patterns
- Reduce CPU command recording overhead
- **Expected gain:** 5-10% CPU overhead reduction

### 6. Compute-Only Custom Heaps (OPTIONS19)
- `ComputeOnlyCustomHeapSupported` for weight buffers
- Better memory placement control
- **Expected gain:** Better VRAM utilization

---

## Implementation Phases

### Phase 1: Foundation (Week 1)
- [ ] Query and cache all D3D12_OPTIONS12-22 features
- [ ] Implement Enhanced Barrier abstraction layer
- [ ] Add GPU Upload Heap support (opt-in → default when ReBAR live)
- [ ] Fix Work Graph capability detection

### Phase 2: Work Graphs Implementation (Week 2-3)
- [ ] Design Work Graph for `layer_decode` (13 nodes)
- [ ] Implement node shaders (RMSNorm, GEMV, RoPE, Attention, SiLU, Addn)
- [ ] Build Work Graph at init, record once, dispatch per token
- [ ] Add fallback to command-list path when Work Graphs unavailable

### Phase 3: Kernel Optimizations (Week 3-4)
- [ ] Rewrite GEMV for wave64-native reduction (no groupshared)
- [ ] Optimize Attention kernel for RDNA4 dual-issue
- [ ] Add FP16/BF16 paths for future quantization
- [ ] Tune dispatch sizes for 64 CUs (RX 9070 XT)

### Phase 4: RDNA2 Compatibility (Week 4+)
- [ ] Feature-level fallbacks for all new APIs
- [ ] Test on RX 6800/6900 XT
- [ ] Maintain parity test coverage

---

## Success Metrics

| Metric | Current | Target (RDNA4) |
|--------|---------|----------------|
| Decode tok/s (Llama-3.2-1B-Q8_0) | 122 | 300+ |
| Submits/token | 49 | 1-2 (Work Graph) |
| GPU utilization | 23% | 60%+ |
| Fence idle % | 53% | <15% |
| First-token latency | ~9ms | <3ms |

---

## Key Files to Modify

1. `src/backends/d3d12/d3d12_backend.cpp` - Main backend
2. `src/backends/d3d12/repack.h` - Weight upload
3. `shaders/*.hlsl` - Kernel shaders
4. New: `src/backends/d3d12/work_graph.hlsl` - Work Graph nodes
5. New: `src/backends/d3d12/enhanced_barriers.h` - Barrier abstraction
6. `tools/d3d12_probe.cpp` - Feature reporting