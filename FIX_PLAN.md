# Fix Plan for RDNA4 D3D12 Backend

## Critical Issues (Must Fix First)

### 1. Backend Initialization Missing
- **Problem**: No call to `zen::d3d12::init()` in runtime, so `enabled()` returns false
- **Fix**: Add explicit initialization call in `DenseTransformerRuntime::forward` before first GPU check

### 2. Permanent Failure Latch in `device_attention_available()`
- **Problem**: Static `disabled` flag permanently disables GPU attention after first failure
- **Fix**: Remove static latch, allow retries

### 3. CPU Fallback Doesn't Execute
- **Problem**: FlashAttention returns false but CPU attention code is in else block - never executes
- **Fix**: Move CPU attention code to execute when flash fails

### 4. FlashAttention Shader Correctness
- **Problem**: Accumulator arrays not per-query-token, online softmax incorrect
- **Fix**: Separate accumulators per query token, correct online softmax

### 5. Shared-Memory Bounds Insufficient
- **Problem**: C++ allows `kv_heads * head_dim <= 2048`, shader needs `<= 512`
- **Fix**: Update validation to `kv_heads * head_dim <= 512`

### 6. Wave Size Not Constrained
- **Problem**: Shader reductions assume 32 lanes, wave64 breaks them
- **Fix**: Add `[wave32]` attribute or implement wave-size-independent reduction

### 7. Source Code Bugs
- **Problem**: Duplicate `std::lock_guard<std::mutex> lock(s.mutex);` and `s.trace` used before `State& s` declaration
- **Fix**: Remove duplicate, reorder declarations

## Implementation Order

1. Fix source code bugs (duplicate mutex, declaration order)
2. Add explicit `zen::d3d12::init()` call in runtime
3. Remove static `disabled` latch in `device_attention_available()`
4. Fix CPU fallback path
5. Fix FlashAttention shader correctness
6. Update shared-memory bounds validation
7. Add wave size constraint
8. Run parity tests with GPU enabled
9. Benchmark

## Acceptance Criteria

- [ ] Backend initializes successfully
- [ ] `enabled() == true` after init
- [ ] KV cache GPU contents match CPU
- [ ] FlashAttention dispatch confirmed in trace
- [ ] FlashAttention parity passes (FP32 CPU comparison)
- [ ] CPU fallback produces correct output
- [ ] No out-of-bounds shared memory access
- [ ] Wave reductions correct
- [ ] All 5 D3D12 parity tests EXECUTED (not skipped)
- [ ] Decode >= 450 tok/s (Q4 1B model)
- [ ] Prefill >= 8000 tok/s