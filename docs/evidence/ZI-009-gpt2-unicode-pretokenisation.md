# ZI-009 GPT-2 Unicode pre-tokenisation increment

Date: 2026-10-31

## Change

- Replaced the byte-based heuristic that treated every non-ASCII byte as a letter.
- Added a generated static Unicode category table for General_Category L, General_Category N and whitespace, based on Unicode database 15.1.0.
- Added UTF-8 codepoint classification and GPT-2-style pre-token spans for contractions, optional leading ASCII space, letter runs, number runs, punctuation/symbol runs and whitespace runs.
- Kept the runtime free of ICU and Python dependencies. The generated table is compiled into the tokenizer.
- Added a regression test in which a Unicode letter followed by punctuation must not merge across the pre-token boundary, even when the vocabulary contains a high-priority combined token.
- Refined newline whitespace handling so trailing horizontal whitespace after a newline remains a separate span.

## Verification

Clean copied workspace, Linux GCC:

- Scalar: `ZEN_BUILD_AVX2=OFF`, 6/6 CTest passed.
- AVX2: `ZEN_BUILD_AVX2=ON`, 6/6 CTest passed.
- Both configurations include `zen_core_tests`, NumPy Qwen2 synthetic parity, architecture micro-models, benchmark parity, synthetic GGUF runtime and HTTP smoke tests.

## Limits

This is a correctness improvement to GPT-2-style pre-tokenisation, not a claim of complete Hugging Face GPT-2 tokenizer parity. The implementation still requires a reference token-ID corpus generated from a named independent implementation (for example, Hugging Face tokenizers or tiktoken with the matching vocabulary/merge ranks), and edge-case parity for every supported tokenizer variant. It does not close ZI-009.
