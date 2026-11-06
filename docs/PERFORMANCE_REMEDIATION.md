# CPU/GPU performance remediation: current status

This is an incremental fix, not a claim that all formats or the RX 9070 XT are
fully optimized. Real model throughput and Windows/D3D12/HIP parity have not
been measured in this environment.

## Changes

- Q4_K and Q5_K decode GEMV now have fused AVX2/FMA CPU kernels, using the
  existing scalar decoder for unsupported CPUs. They decode weights in registers
  rather than repeatedly materializing blocks and reducing each block. Set
  `ZEN_CPU_KQUANT_GEMV=0` before starting the process to A/B the previous path.
  This changes FP summation order, so expect small numerical differences.
- The execution planner now actually applies its physical-core/memory-aware
  worker policy to decode; batched prefill still uses the logical-core budget.
- D3D12 fused layer, FFN and LM-head paths now require Q8_0 weights. Previously
  they could interpret mapped Q4_K/Q5_K or mixed-format tensors as Q8_0. The
  pipeline also falls back for more than 64 layers and only skips the CPU layer
  loop after a successful submission.
- CPU-only and scalar builds had unrelated compilation failures; those are
  fixed. The Q8 repack test is now run only when its AVX2 kernel is available.

## Evidence and limitations

`zen_kquant_gemv_bench` compares one thread of the previous block-wise CPU GEMV
against the new route on deterministic synthetic weights (1024 rows, 4096
columns, 7-run median, warm buffers). On an Intel Xeon virtual CPU with GCC 12
Release, one run measured Q4_K 4.454 -> 0.802 ms (5.56x) and Q5_K 6.158 ->
1.102 ms (5.59x). This is a kernel microbenchmark, not full-model speed; cache,
threading, hardware and model mix can change the outcome.

Randomized multi-block Q4_K/Q5_K tests compare both serial and parallel dispatch
with scalar dequantization and a double-precision dot, including unaligned data,
scale/min fields, Q5 high bits and FP16 subnormal scales. AVX2 and scalar Linux
builds passed core, IQ differential, fuzz, Bonsai hybrid parity, benchmark
parity and HTTP smoke tests. A full CTest run also had three failures because
`models/synthetic` fixtures are absent from this checkout (Qwen2 numerical and
runtime tests, architecture micro-models). D3D12 runtime code was syntax-checked
with the D3D12 define on Linux, but this is not an MSVC build or GPU test.

## Outstanding work

- An experimental D3D12 Q4_K/Q5_K GEMV/GEMM path has been added; see below.
  It is off by default and has not yet been verified on Windows or a GPU. HIP
  still has no Q4_K/Q5_K matrix path. Other formats may still fall back to CPU;
  mixed-format models can take a mix of routes.
- Qwen3 Q/K normalization prevents using the existing fully fused Q8_0 decode
  layer pipeline. Its low reported prefill score has not been reproduced or
  diagnosed; it needs per-stage and per-shape measurement on the 9070 XT.
- Q4_K/Q5_K prefill GEMM remains separate from the new decode kernel. The
  existing Q4_K prefill optimization is opt-in (`ZEN_CPU_FAST_Q4K=1`); Q5_K
  still uses generic CPU prefill. More formats, real-model parity and
  end-to-end tuning remain necessary.

## Windows A/B and diagnostics

Build a Release binary with `ZEN_BUILD_D3D12=ON` and a working `dxc`; check
CMake output to ensure the backend was actually enabled. For each model, use
identical token IDs, context length, generation length and environment between
runs. Start a fresh process for each arm and record warm as well as cold runs;
do not compare a prefix-cache hit to a full prefill. Example PowerShell:

```powershell
$env:ZEN_BACKEND = 'd3d12'
$env:ZEN_QGEMM_TRACE = '1'
$env:ZEN_D3D12_TRACE = '1'
.\build-dx12\Release\zen_generate_ids.exe model.gguf --prompt-ids 1,2,3,4 --max-new 32
$env:ZEN_CPU_KQUANT_GEMV = '0'
.\build-dx12\Release\zen_generate_ids.exe model.gguf --prompt-ids 1,2,3,4 --max-new 32
Remove-Item Env:ZEN_CPU_KQUANT_GEMV
```

Replace the example IDs with the same realistic 128-token prompt for both
runs. Use `ZEN_BACKEND=cpu` for an additional CPU-only comparison. Route
counters are aggregate and do not count every fused or batched GPU operation;
GPU dispatches alone do not prove all weights or an entire model ran on the
GPU. Capture model filenames/types, build configuration, driver, prompt and
output lengths, generated IDs/logit parity and prefill/decode times alongside
D3D12 traces. Test each reported model before claiming improvement.

## Experimental D3D12 K-quant path

`ZEN_D3D12_KQUANT=1` opts Q4_K/Q5_K GEMV (decode) and GEMM (prefill) into a
D3D12 shader using raw GGML blocks and FP32 activations. Q4_K/Q5_K weights are
cached in VRAM with format-tagged entries, and runtime model reload/destruction
releases those entries. The API for external callers requires immutable weight
buffers while resident; call `release_kquant_weights` before changing or freeing
one. Failed/unsupported dispatches use the existing CPU paths. The shader tiles
four tokens and performs host input/output transfers on each call; it is not a
fully GPU-resident transformer pipeline, and its speedup is not established.

This is experimental: **the shader, Windows build, GPU parity and performance
have not been run in this environment**. Do not treat it as production-validated.
HIP support, fused K-quant layer/FFN submissions, and real-model parity and
performance remain outstanding. CTest on a D3D12 build adds
`zen_d3d12_kquant_parity`, covering decode/prefill, odd rows and token tails,
Q5 high bits, packed scales/mins, FP16 subnormals, residency replacement and
actual production dispatch (not a CPU fallback). It skips only when device
initialization fails; shader or dispatch failure is a test failure.

On a Windows machine with the D3D12 build prerequisites:

```powershell
cmake -S . -B build-dx12 -G "Visual Studio 17 2022" -A x64 -DZEN_BUILD_D3D12=ON -DZEN_DXC_DIR="C:/path/to/dxc"
cmake --build build-dx12 --config Release --parallel
ctest --test-dir build-dx12 -C Release -R zen_d3d12_kquant_parity --output-on-failure
$env:ZEN_BACKEND = 'd3d12'
$env:ZEN_D3D12_KQUANT = '1'
$env:ZEN_QGEMM_TRACE = '1'
.\build-dx12\Release\zen_generate_ids.exe model.gguf --prompt-ids 1,2,3,4 --max-new 32
```

Run parity first and use a realistic prompt for performance comparisons. A
`gpu_ok` route count for an individual GEMV/GEMM proves that dispatch succeeded,
not that an entire mixed-format model ran on the GPU or that the results are
correct. Compare real-model logits/outputs against the CPU, test cache pressure,
and measure prefill and decode separately on the 9070 XT before relying on it.
