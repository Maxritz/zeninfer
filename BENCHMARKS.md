# Benchmarks

Measured 2026-11-08 on AMD Ryzen 9 5900XT (16 cores / 32 threads, 3.3 GHz,
DDR4 ≈ 51 GB/s peak) and AMD Radeon RX 9070 XT (RDNA4, GDDR6 ≈ 614 GB/s,
D3D12.1, ReBAR live). All runs use `zen_sherlock model.gguf 32 8`
(prefill 32 tokens, decode 8 tokens) unless noted; `zen_generate_ids`
cold-start numbers include one-off VRAM upload (~4.3 GB ≈ 675 ms on this box).

`zen_sherlock` warms the model once (page cache + VRAM residency) before
measuring, so its prefill number is the steady-state figure. Its decode wall
re-runs the 32-token prompt through the prefix cache — true per-token decode
is the `zen_generate_ids` `decode_ms` / steps figure, quoted below where it
differs. Route proof: `ZEN_QGEMM_TRACE=1` prints `[qgemm] gemv gpu_try=N
gpu_ok=N` / `cpu_avx2=N` at exit.

## Warm (resident) steady state

| Model | Backend | Prefill | Decode |
|---|---|---|---|
| Qwen3-4B Q8_0 (4.0 GB) | CPU `ZEN_BACKEND=cpu` | 215 ms · 148 tok/s | 176 ms/token · 5.7 tok/s (true ~135 ms ≈ 7.4 tok/s) |
| Qwen3-4B Q8_0 | GPU (default) | 117 ms · 272 tok/s | 27 ms/token · 37 tok/s (true ~12 ms ≈ 85 tok/s) |
| Qwen3-8B Q4_K_M (~5.5 GB) | CPU `ZEN_BACKEND=cpu` | 913 ms · 35 tok/s | 323 ms/token · 3.1 tok/s |
| Qwen3-8B Q4_K_M | GPU, K-quant default-on | 367 ms · 87 tok/s | 99 ms/token · 10.1 tok/s |
| Llama-3.2-1B Q8_0 | CPU | 76 ms · 421 tok/s | 53 ms/token · 18.8 tok/s |
| Llama-3.2-1B Q8_0 | GPU | 41 ms · 785 tok/s | 9 ms/token · 107 tok/s |
| Phi-4-reasoning-plus Q3_K_M | CPU / GPU | 1735 ms / 1743 ms · 18 tok/s | 508 / 512 ms/token · 2.0 tok/s |
| DeepSeek-Coder-V2-Lite Q4_K_M | CPU | 3315 ms · 9.7 tok/s | 599 ms/token · 1.7 tok/s |
| GLM-4.7-Flash Q4_K_M | CPU / GPU | 7665 / 7660 ms · 4.2 tok/s | 1456 / 1415 ms/token · 0.7 tok/s |
| K2-Horizon-36B MoVA Q4_K_M | CPU / GPU | 6300 / 5984 ms · 5 tok/s | 1164 / 1132 ms/token · 0.9 tok/s |
| flint-q4_1 (MoE 256 experts) | CPU | 9866 ms / 8 tokens · 0.8 tok/s | ~203 ms/token · 4.9 tok/s |

## Cold start (`zen_generate_ids`, first run)

| Model | Backend | Load | Prefill (32 tok) | Decode (per token) |
|---|---|---|---|---|
| Qwen3-4B Q8_0 | CPU (MinGW) | 1218 ms (incl. K-block repack) | 535 ms · 60 tok/s | ~135 ms · 7.4 tok/s |
| Qwen3-4B Q8_0 | GPU (default) | 1400 ms | 948 ms · 33 tok/s (≈675 ms is one-off upload) | ~12 ms · 85 tok/s |

GPU tokens are greedy-identical to CPU on these runs (e.g. Qwen3-4B:
`198,75902,198,67512,198,698,785,914` on both backends).

## Defaults that produce these numbers

- D3D12 GPU is **default-on** when a 12.1 device opens; opt out with
  `ZEN_BACKEND=cpu` / `ZEN_D3D12=0`.
- K-quant GPU (`ZEN_D3D12_KQUANT=0` opts out) — without it Q4_K/Q5_K models
  silently run host AVX2 even with the GPU enabled.
- CPU prefill K-block + repack (`ZEN_CPU_Q8_KBLK=0` /
  `ZEN_CPU_Q8_REPACK_KBLK=0` opt out) — Qwen3-4B CPU prefill 923 ms → 457 ms,
  token-identical.
- GPU decode pipeline (`ZEN_D3D12_PIPELINE=0` opts out): all 36 layers in one
  submit+wait per token.

Reproduce:
`ZEN_MAP_TRACE=0 [ZEN_BACKEND=cpu] zen_sherlock model.gguf 32 8`
