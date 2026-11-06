# ZenInfer 0.4.5-alpha

## Correctness

- Corrected Llama SentencePiece tokenisation from pair-merging to unigram Viterbi segmentation.
- Added byte fallback transitions and whitespace handling aligned with the supplied GGUF's SentencePiece normaliser behaviour.
- Added an exporter and reproducible independent SentencePiece parity script.
- Added five fixture-specific SentencePiece golden-ID cases to runtime tests.

## Benchmarks

- Runtime CTest now performs five distinct-prompt runs and reports median prefill and decode-forward tok/s.
- Exact-prefix KV cache hits are excluded from timed runs.
- Added Windows PowerShell real-GGUF smoke test script.

## Verification

- AVX2: 6/6 CTests PASS.
- Scalar-only: 6/6 CTests PASS.
- SentencePiece 0.2.1 external parity: 8/8 PASS on the supplied Tinyllama 15M Stories fixture in both builds.
- Windows/MSVC remains unverified in this environment.

## Still outstanding

Production Qwen3 MoE, BitNet BitLinear, Bonsai GGUF/custom layout, full Jinja, application-managed NVMe eviction, JSON Schema grammar, native FP16 hardware arithmetic, Qwen2 independent parity and GPT-2/HF tokenizer parity remain open.
