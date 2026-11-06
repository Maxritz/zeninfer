# Direct3D 12.1 compute backend

Optional GPU path for the Q8_0 matrix kernels. Off by default; the CPU code paths are
untouched and always remain the fallback.

## Enabling it

```
cmake -S . -B build-dx12 -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DZEN_BUILD_D3D12=ON -DZEN_DXC_DIR=H:/llamadx/dxc/bin/x64 -DZEN_BUILD_TESTS=ON
cmake --build build-dx12

build-dx12/zen_d3d12_probe.exe            # what does this machine support?
ZEN_BACKEND=d3d12 zen_generate_ids.exe ...  # run the model on the GPU
```

Diagnostics: `ZEN_D3D12_TRACE=1` collects GPU timestamps and per-phase host timing,
`ZEN_D3D12_BENCH_SUBMIT=1` measures the queue round trip, `ZEN_D3D12_NO_READBACK=1`
isolates the readback cost.

`build-dx12.bat` wraps the MSVC environment, pins the Windows SDK to 10.0.26100.0
(the 10.0.28000.0 kit's `mt.exe` fails to load on this machine) and appends
Strawberry to `PATH` rather than prepending it, because prepending shadows SDK tools.

## How it is wired

The backend hooks `zen::quant::gemv()` / `gemm()` / `gemv_batch()`. Transformer code is
only changed where consecutive matvecs share an input, and every hook falls back to the
existing AVX2 code on any failure.

Weights are cached in VRAM keyed by the host pointer the loader handed out, so a model is
repacked and uploaded once and then read from VRAM for the rest of the process. Q8_0
blocks are 34 bytes, which leaves every odd block index 2-byte aligned — illegal for a
4-byte `ByteAddressBuffer` load — so upload repacks each row into an aligned 32-byte
quant stream plus a separate fp32 scale stream. Shaders compile to DXIL at build time and
are embedded as byte arrays by `scripts/bin2header.py`.

## Sherlock trace: why decode was only ~65 tok/s

Measured per call on Llama-3.2-1B-Q8_0, 64-token prompt (`ZEN_D3D12_TRACE=1`):

| phase | us/call |
|---|---|
| prep (memcpy x, params) | 1.0 |
| allocator + list reset | 1.8 |
| root signature + args | 0.02 |
| record (dispatch, barriers, copy) | 3.4 |
| **execute + wait** | **122.1** |
| readback (Map + memcpy) | 1.0 |
| **GPU dispatch (timestamps)** | **23.4** |

Host recording is ~6 us; the GPU does 23 us of work; the remaining ~95 us is queue round
trip. Three experiments located it precisely:

| experiment | result |
|---|---|
| empty command list, submit + fence | 16.4 us |
| one trivial dispatch, submit + fence | 73.0 us |
| 2 dispatches in one submission | 94.3 us |
| 4 / 8 / 16 dispatches in one submission | 76 / 85 / 85 us |
| same, with the readback copy removed | 124.7 us (was 132.0) |

Conclusion: the readback costs ~7 us and the work is ~25 us, but **each submission
carries a ~70 us fixed cost** while extra dispatches in the same submission cost only
~1-2 us. Decode issued 113 separate dependent submissions per token, so ~8 ms of every
token was pure queue latency.

### Fix: batch the matvecs that share an input

Q|K|V and gate|up both consume the same normed activation, so `zen::quant::gemv_batch()`
now dispatches them into a single command list (one transition, one copy, one fence).
The runtime uses it in `forward()`; every other site is unchanged.

The FFN chain goes further: `gemv_q8_0_ffn()` runs gate, up, a device-side SiLU and
down in one command list, so the activation never comes back to the CPU between them.
That is 4 submissions per layer instead of 5, and 49 per token instead of 65.

| | before | after |
|---|---|---|
| submits per token | 113 | ~49 |
| decode, 24 new tokens | 65.9 tok/s | **113.4 tok/s** |
| decode, 64-token prompt | 60.2 tok/s | **100.9 tok/s** |

## Measured results

AMD Radeon RX 9070 XT (RDNA4, 15.36 GB, feature level 12_2, shader model 6.8),
Llama-3.2-1B-Instruct-Q8_0 (1.321 GB):

| scenario | CPU (AVX2) | D3D12 | change |
|---|---|---|---|
| decode, 3-token prompt, 24 new | 19.6 tok/s | **113.4 tok/s** | **5.8x** |
| decode, 64-token prompt, 32 new | 19.7 tok/s | **100.9 tok/s** | **5.1x** |
| prefill, 64 prompt tokens | **203.6 tok/s** | (CPU path) | - |

Decode output is token-for-token identical to the CPU path. Kernel parity against a
double-precision reference: **1.26e-07 relative error** (`zen_d3d12_gemv_parity`).

One-time cost to make a model VRAM-resident, measured at ~555 ms for 1.321 GB: repack
264 ms, PCIe 74 ms, remainder VRAM allocation. Prefill is the right place to pay it;
`gemm` warms residency even when prefill stays on the CPU.

## Limitations

- **Prefill matmuls stay on the CPU by default.** The GEMM kernel now stages each
  256-column activation chunk through groupshared once per workgroup (the earlier
  design re-read the tile per output row and measured 582 us per dispatch against
  203 tok/s on AVX2). Its four parity shapes pass at <=1.7e-07 against a
  double-precision reference and `ZEN_D3D12_GEMM_EXPERIMENTAL=1` produces end-to-end
  ids identical to the CPU path, but it is still opt-in because the first prefill pays
  the ~555 ms cold residency upload, and on a 6-token prompt it measured 7.9 tok/s
  against 20.4 tok/s on AVX2. It needs to earn a default slot on throughput first.
- **Decode is still queue-latency bound.** At 113.4 tok/s the backend streams
  1.321 GB/token = 149 GB/s, about 23% of peak, with ~49 submissions per token costing
  ~69 us each. The hard ceiling at Q8_0 is 640 GB/s / 1.321 GB = **484 tok/s**, so a
  600 tok/s target is not reachable on this model regardless of kernel quality. Batching took submits from 113 to 65; going further needs either a lower
  fixed submission cost or fewer, larger submissions (fusing attention and activation
  into the GPU pipeline so a whole layer is one submit).
- Matvecs below 256 Ki elements stay on the CPU; the submit round trip costs more than
  the arithmetic.
- Only Q8_0 with `scale == 1.0f` is routed to the GPU.

## On int8 / fp8 and data movement

Q8_0 already stores weights as 8-bit values behind one fp16 scale per 32 elements:
1.321 GB for 1.24 B parameters is **1.07 bytes per parameter**. So:

- **fp8 (E4M3)** would store 1.0 byte per parameter plus a scale — roughly 3% fewer bytes
  than Q8_0, in exchange for re-quantisation error on weights that are already int8.
- **int8 dot products** (`dot4add_i8packed`, available here since the device reports
  shader model 6.8) do not change weight traffic at all; they raise arithmetic
  throughput, which matters only once a kernel is compute-bound.

Neither addresses the measured bottleneck, which is submission latency: the GPU is busy
23 us out of every ~130 us of a matvec. Halving weight bytes would shave the transfer
term from ~25 us to ~13 us; removing submissions would be worth multiples of that.
