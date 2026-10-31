# ZenInfer 0.3.1-alpha

## GGUF fixture-driven fixes

- Fixed GGUF Boolean metadata parsing to consume exactly one byte and reject invalid Boolean values.
- Added GGML IQ4_NL tensor storage sizing (type 20, block 32, 18 bytes).
- Registered Q5_0 and Q6_K as recognised but not decoded formats.
- Added regression coverage for Boolean metadata and an IQ4_NL tensor descriptor.
- Audited two real user-supplied GGUF v3 files. Both now pass `zen_model_inspect` with zero unresolved tensor sizes.

## Not included

No new tensor dequantisation kernels, tokenizer, transformer forward pass, executable model adapter, generation path, or throughput benchmark is included in this patch.
