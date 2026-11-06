# DirectX 12 Backend Fixes - Implementation Tracker

## P1 - Critical Fixes (Must Fix First)

- [ ] **T-001**: Fix `gemv_ffn_locked` DP4a cbuffer population (DX12-003)
- [ ] **T-002**: Add `#include "q8_0_gemv_fp32_cso.h"` and fallback PSO selection (DX12-004)
- [ ] **T-003**: Guard resource transitions for GPU upload heap resources (DX12-012)

## P1 - Code Cleanup

- [ ] **T-004**: Remove corrupted `build_work_graph` and `layer_decode_gpu` stub (DX12-001, DX12-002, DX12-019)

## P2 - Material Fixes

- [ ] **T-005**: Fix VRAM budget integer division (DX12-034)
- [ ] **T-006**: Add SER macro fallback constant for older Agility SDK (DX12-006)
- [ ] **T-007**: Fix `state_to_layout/sync` exhaustive mapping or assert (DX12-007)
- [ ] **T-008**: Add `ZEN_D3D12_FORCE_GPU_UPLOAD` env var bypass (DX12-008)
- [ ] **T-009**: Fix `attn_upload` GPU upload heap transition bug (DX12-012 - part of T-003)
- [x] **T-010**: Pre-allocate command allocator pool for pipeline (DX12-013) — DONE, runtime-validated
- [x] **T-011**: Fix pipeline arena sizing for heterogeneous layers (DX12-014) — DONE, no fixed-shape bug present in restored code
- [ ] **T-012**: Fix `half_to_float` in HLSL (DX12-028)

## P3 - Polish / Low Risk

- [ ] **T-013**: Deduplicate `half_to_float` implementation (DX12-038)
- [ ] **T-014**: Update Work Graph CMake comments (DX12-020)
- [ ] **T-015**: Add `static_assert` for Comp enum count (DX12-029)
- [ ] **T-016**: Fix pointer arithmetic in `kv_cache_append_batched` (DX12-031)
- [ ] **T-017**: Fix integer division in VRAM budget (DX12-034)
- [ ] **T-018**: Add ARM64EC guard for `_mm_pause()` (DX12-037)
- [ ] **T-019**: Fix `quantize_fp32_to_int8` clamp in HLSL (DX12-028)
- [ ] **T-020**: Share cos/sin table in `LayerDecodeParams` (DX12-033)

---

## Implementation Order

1. **T-002** - Missing include (blocks build on non-DP4a)
2. **T-001** - DP4a cbuffer in fused FFN (correctness)
3. **T-003** - GPU upload heap transitions (device removal risk)
4. **T-004** - Clean up dead code
5. **T-005** - VRAM budget fix
6. **T-006** - SER macro fallback
7. **T-007** - Exhaustive state mapping
8. **T-008** - Force GPU upload env var
9. **T-010** - Allocator pool
10. **T-011** - Pipeline arena sizing
11. **T-012/T-019** - HLSL clamp fix
12. **T-013** - Deduplicate half_to_float
13. **T-014** - CMake comments
14. **T-015** - static_assert
15. **T-016** - Pointer arithmetic
16. **T-018** - ARM64EC guard
17. **T-020** - Shared cos/sin

---

## Progress Log

| Task | Status | Notes |
|------|--------|-------|
| T-002 | DONE | Added missing q8_0_gemv_fp32_cso.h include |
| T-001 | DONE | Fixed DP4a cbuffer population in gemv_ffn_locked, ffn_residual_fused_locked, layer_decode, layer_decode_pipeline |
| T-003 | DONE | Guard resource transitions for GPU upload heap resources |
| T-004 | DONE | Remove corrupted build_work_graph and layer_decode_gpu stub |
| T-005 | DONE | Fix VRAM budget integer division |
| T-006 | DONE | SER macro fallback constant |
| T-007 | DONE | Exhaustive state mapping |
| T-008 | DONE | ZEN_D3D12_FORCE_GPU_UPLOAD env var |
| T-010 | DONE | Allocator pool: grows once, `Reset()` per run, lists `Close()`d at pool creation (fresh lists are recording); runtime-validated `fallbacks=0` over all decode tokens |
| T-011 | DONE | Pipeline arena sizing: grow-on-demand caps (`reserve_scratch`/`ensure_upload`/`attn_upload`/`attn_device`), per-call span offsets recomputed from current shape; no fixed-shape assumption found in restored code |
| T-012 | DONE | HLSL clamp fix |
| T-013 | PENDING | Deduplicate half_to_float |
| T-021 | DONE | Batched prefill GEMM (`gemm_q8_0_batch` + `tensor_gemm_batch`): collapses Q|K|V (3) and gate|up (2) into single command lists per layer; **2.3x prefill speedup** (509→219 ms), 100% GPU route (647/647 gemm), 10/10 stable runs, no TDR |
| T-014 | DONE | CMake comments |
| T-015 | DONE | static_assert |
| T-016 | DONE | Pointer arithmetic |
| T-018 | DONE | ARM64EC guard |
| T-020 | DONE | Shared cos/sin |