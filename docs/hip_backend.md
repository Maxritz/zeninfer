# AMD HIP (ROCm) compute backend

The HIP backend runs the Q8_0 matrix kernels on an AMD GPU through ROCm. It hooks
the same four `zen::quant` entry points as the D3D12 backend and needs no changes to
the transformer code.

```sh
build-hip.bat                                    # build
ZEN_BACKEND=hip zen_generate_ids.exe model.gguf   # run
```

## Verified environment

| | |
|---|---|
| GPU | AMD Radeon RX 9070 XT |
| Arch | `gfx1201` (RDNA4 / Navi 48) |
| ROCm | 7.16.26332 at `G:\ROCM10RT-gfx1201` |
| Wavefront | 32 lanes |
| Compute units | 32 |
| Memory | 256-bit, 1259 MHz → 644.6 GB/s |
| Runtime | `amdhip64_7.dll` resolved from `System32` |

Native Windows ROCm is used directly; WSL is not required.

## Build integration

CMake's `Windows-Clang` platform module refuses to mix Clang (which HIP requires)
with MSVC in one configuration: *"Use either Clang or MSVC as the compiler for all
of C, C++, and/or HIP."* The project is MSVC throughout, so `enable_language(HIP)`
is not an option.

Instead `hipcc` is driven as a **custom command**, the same shape the D3D12 backend
uses for `dxc`, and the resulting COFF object is linked into `zen_core`:

```cmake
add_custom_command(OUTPUT "${ZEN_HIP_OBJ}"
  COMMAND "${ZEN_HIPCC_EXECUTABLE}" --offload-arch=${ZEN_HIP_ARCH} -O3
          -fms-runtime-lib=dll -I"${CMAKE_CURRENT_SOURCE_DIR}/include"
          -c "src/backends/hip/hip_backend.hip" -o "${ZEN_HIP_OBJ}" ...)
set_source_files_properties("${ZEN_HIP_OBJ}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
```

Two flags are load-bearing:

- `-fms-runtime-lib=dll` — hipcc otherwise defaults to the static CRT and the link
  fails `LNK2038: mismatch detected for 'RuntimeLibrary'`.
- `--offload-arch=gfx1201` — without an explicit arch the object contains no code
  for the card.

`include/zen/hip_backend.h` contains no `<hip/...>` includes: the entire backend,
host logic and kernels, lives in one hipcc-compiled translation unit, so MSVC
translation units only ever see plain declarations.

## Design

- **One wavefront per weight row.** Lane `l` owns element `l` of every 32-element
  Q8_0 block, so all 32 lanes walk every block together and the per-block scale
  folds into the lane's accumulator. The row sum is a 32-entry tree in `__shared__`.
- **No repack alignment requirement.** Because each lane loads a single byte,
  the quant stream needs no 16-byte alignment; the 34-byte GGML block is still split
  into a quant stream plus a separate fp32 scale stream on upload.
- **Single stream, one synchronisation per call.** Work is queued on one
  `hipStream_t`; the D3D12 backend pays a fence round trip per command list instead.
- **Residency cache** keyed by `(host pointer, rows, columns)` with an 80% VRAM
  budget. The contract matches D3D12's: a tensor keeps its host allocation for the
  model's lifetime, so a distinct tensor always arrives at a distinct address.
  Freeing weights and reusing that address for different data of the same shape
  would return the stale residency.
- **FFN fusion.** `gemv_q8_0_ffn` runs gate, up, SiLU and down on one stream, so the
  activation never round-trips to the host.

### The `__shfl_down_sync` caveat

The reduction deliberately uses a shared-memory tree rather than
`__shfl_down_sync`. On this platform the shuffle builtin left every row holding
only its first lane's partial sum (parity error ~1.0), while the identical tree in
`rmsnorm_kernel` measured 7e-08. Note that ROCm also static-asserts that the mask
*type* is 64-bit, even though RDNA waves are 32 wide.

## Kernel ownership bug found during bring-up

The first working version produced a partial sum with the right sign and a random
magnitude. An all-ones isolation test (16 rows × 64 cols, q=1, scale=1, x=1, so each
row must equal 64) returned **2.0** — exactly the block count. The loop had been
written as

```cpp
for (int b = lane; b < blocks_per_row; b += 32)   // WRONG
```

which gives each lane one element of a single block instead of one element of every
block, so the row sum covered only `blocks_per_row` of the `blocks_per_row * 32`
elements. Iterating every block while keeping `lane` as the element index is
correct.

## Verification

| Check | Result |
|---|---|
| `zen_hip_gemv_parity` | 1.66e-07 … 2.11e-07 across 4 shapes |
| `zen_hip_gemv_parity` (batch) | 1.10e-07, 1.67e-07 |
| `zen_hip_gemm_parity` | 1.34e-07 … 1.64e-07 across 4 shapes |
| `zen_hip_gemm_parity` (rms_norm) | 7.02e-08 |
| ctest | 11/11 |
| End-to-end ids, Llama-3.2-1B-Instruct-Q8_0 | identical to CPU |

Both harnesses exit 77 when no device is present, and `SKIP_RETURN_CODE 77` makes
that a ctest *skip* rather than a pass, so a machine without the hardware cannot
report this suite as verified.

## Current performance

Llama-3.2-1B-Instruct-Q8_0, 128 new tokens, "The quick brown fox":

| Backend | tokengen tok/s |
|---|---|
| CPU (AVX2) | 20.2 |
| HIP | 46.5 |

Decode streams 1.313 GB of weights per token, so the bandwidth roofline at
644.6 GB/s is ~490 tok/s. The backend is therefore at roughly 9% of its ceiling and
is **latency-bound, not bandwidth-bound**: the measured per-submit cost is ~463 µs
across ~48 submissions per token, because `quant::gemv` is synchronous and every
call synchronises the stream and copies results back to the host.

### Remaining work

1. **Remove the per-call host synchronisation.** The runtime API returns results on
   the host, so this needs a GPU-resident layer pipeline (attention + KV cache on
   device) before the stream can stay unsynchronised across a layer. This is the
   single largest win available and matches the D3D12 finding that ~190 µs of every
   submit is pure round-trip latency.
2. **The batched GEMM is opt-in** (`ZEN_HIP_GEMM_EXPERIMENTAL=1`) and has not been
   tuned; prefill currently stays on the CPU, and the backend absorbs the residency
   upload during prefill so the first decoded token does not.
3. **Prefill throughput** is below the CPU path (6.3 vs 22.9 tok/s) for the same
   reason: the GEMM is disabled by default, so prefill pays the 1.3 GB upload and
   then computes on the CPU.
