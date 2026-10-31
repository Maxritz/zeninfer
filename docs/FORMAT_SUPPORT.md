# ZenInfer GGML/GGUF format support (0.4.4-alpha)

A parser recognising a storage layout does not mean that a decoder or runtime kernel exists for that format. The three levels are deliberately separate:

1. **Size-known**: GGUF tensor directory can be bounds-checked using the canonical GGML type ID, block element count and bytes per block.
2. **Decoder implemented**: tensor payload can be converted to FP32 by ZenInfer.
3. **Runtime projection path**: weights can be used directly from mapped storage by quantised GEMV/GEMM.

## Runtime decoder formats

| Format | GGML ID | Elements/block | Bytes/block | Decoder | Runtime projection |
|---|---:|---:|---:|---|---|
| F32 | 0 | 1 | 4 | Yes | Mapped GEMV/GEMM |
| F16 | 1 | 1 | 2 | Yes | Mapped row conversion + GEMV/GEMM |
| Q4_0 | 2 | 32 | 18 | Yes | Block-streaming GEMV/GEMM |
| Q5_0 | 6 | 32 | 22 | Yes | Block-streaming GEMV/GEMM |
| Q8_0 | 8 | 32 | 34 | Yes | Fused AVX2 GEMV when available; block-streaming GEMM |
| Q3_K | 11 | 256 | 110 | Yes | Block-streaming GEMV/GEMM |
| Q6_K | 14 | 256 | 210 | Yes | Block-streaming GEMV/GEMM |
| IQ4_NL | 20 | 32 | 18 | Yes | Block-streaming GEMV/GEMM |
| TQ1_0 | 34 | 256 | 54 | Yes | Block-streaming GEMV/GEMM |
| TQ2_0 | 35 | 256 | 66 | Yes | Block-streaming GEMV/GEMM |

The runtime keeps supported matrix weights in a file-backed GGUF mapping. Quantised GEMV/GEMM decodes one block at a time into bounded scratch storage and accumulates the dot products. It does not expand the entire matrix into FP32 RAM.

## Parser-known, decoder not implemented

The parser knows the canonical storage sizes for Q4_1 (3), Q5_1 (7), Q8_1 (9), Q2_K (10), Q4_K (12), Q5_K (13), Q8_K (15), IQ2_XXS (16), IQ2_XS (17), IQ3_XXS (18), IQ1_S (19), IQ3_S (21), IQ2_S (22), IQ4_XS (23), I8/I16/I32/I64/F64 (24-28), IQ1_M (29), BF16 (30), MXFP4 (39). IDs 40 and above are not part of the upstream GGML type enum used by this parser; fork-specific extensions are rejected unless explicitly added with a verified storage contract.

These formats are **not runtime-decodable** in this release unless listed in the runtime decoder table above. A tensor using one of them is no longer rejected merely because the parser lacks its block-size entry, but loading it for inference still fails closed with an explicit unsupported decoder error.

GGML IDs 4, 5, 31-33 and 36-38 are removed/deprecated storage types and are rejected. They are not assigned guessed block sizes.

## ZenInfer-specific reference encodings

`Q1_SIGNED` through `Q18_SIGNED` and `TERNARY2` are ZenInfer reference encodings. They are not GGUF quantisation IDs and must not be confused with GGML's Q1_0/Q2_0 types.

## Correctness policy

Decoder tests include synthetic block-layout checks and runtime fixture tests. The two supplied GGUF fixtures are exercised by optional CTest runtime tests. Independent reference parity was measured for the current dense Llama forward path on those two fixtures; broader format/model parity remains outstanding.


## BitNet decoder scope added in 0.4.4-alpha

These are BitNet ternary storage decoders. TQ2_0 uses the GGML interleaved
4-by-32 layout and maps 2-bit codes to -1, 0, +1, +2 times the block scale.
A trained ternary checkpoint generally uses the first three values. Decoder
support alone does not mean a complete BitNet architecture adapter exists.
