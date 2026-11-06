# ZenInfer 0.4.5-alpha performance and quality report

## Measurement method

- Five measured runs per model; median throughput is reported.
- Each run uses 128 prompt tokens and requests 32 new tokens.
- A distinct prompt is used for warm-up. Each timed prompt differs from the others, so the one-entry exact-prefix cache does not affect timed results.
- Prefill tok/s = 128 prompt tokens / prefill wall time.
- Tokengen tok/s = decode-forward steps after the first token / decode-forward wall time. Sampling, detokenisation and HTTP overhead are excluded.
- Values are this container's measurements, not a prediction for the user's CPU.

## Real GGUF throughput

Environment: AMD EPYC 9V74 virtual CPU allocation, 5 vCPUs, Linux x86-64, GCC 14.2, Release.

| GGUF fixture | Backend | Prompt tokens | New tokens | Prefill median tok/s | Tokengen median tok/s |
|---|---|---:|---:|---:|---:|
| `Tiny-LLM.Q4_K_M(2).gguf` | AVX2 | 128 | 32 | 8,371.09 | 661.50 |
| `Tiny-LLM.Q4_K_M(2).gguf` | Scalar | 128 | 32 | 2,705.41 | 218.84 |
| `tinyllama-15M-stories-Q2_K.gguf` | AVX2 | 128 | 32 | 1,074.30 | 121.89 |
| `tinyllama-15M-stories-Q2_K.gguf` | Scalar | 128 | 32 | 382.67 | 79.84 |

These are five-run medians from the CTest runtime harness. The 15M fixture's model metadata identifies it as `ModelCloud Tinyllama 15M Stories`, size label `15M`. It is not the TinyLlama 1.1B checkpoint.

## Tinyllama 15M quality observation

A separate 200-token generation was run with prompt `Once upon a time`. The runtime returned 200 token IDs and completed. The decoded text was repetitive and incoherent in places (repeated bear/alone/unknown phrases). This is a qualitative observation, not a 1/200 accuracy score: no target continuation or scoring dataset was supplied. The throughput values above measure speed only and do not indicate output quality.

## Independent tokenizer parity

The previous Llama tokenizer used BPE-like pair merging, which was incorrect for the GGUF's SentencePiece unigram vocabulary. The implementation now uses score-based Viterbi segmentation with byte fallback. Google's SentencePiece 0.2.1, reconstructed from the GGUF vocabulary/scores with `nmt_nfkc`, matched ZenInfer on 8/8 test strings, including English, ROCm/RDNA terms, Malayalam and newline handling. This does not yet establish GPT-2/Hugging Face parity or all Unicode normalisation behaviour.

Reproduce with:

```bash
python3 scripts/sentencepiece_reference_parity.py \
  build/zen_tokenizer_export build/zen_generate_ids \
  /path/to/tinyllama-15M-stories-Q2_K.gguf
```

## Synthetic architecture kernels

Qwen3 MoE, BitNet and Bonsai `.zenmodel` fixtures still report architecture-kernel microbench rates only. They do not implement full transformer inference and are not comparable with the real GGUF numbers above.
