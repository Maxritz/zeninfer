# ZI-016 — Security hardening: GGUF parse caps, u64-width gate, JSON depth cap, in-memory parse surface

Date: 2026-10-31
Scope: Port the two stack-overflow DoS fixes and the buffer-backed parse
surface from the security-hardened 0.4.1-alpha tree (commit `57523c5` and
`0cd1175`) into the completion tree.

## Threat model

A GGUF file or an HTTP JSON body is untrusted input. Both surfaces use
recursive descent parsers; unbounded nesting converts a few hundred kB of
input into a stack overflow (0xC00000FD). Metadata u64 values were cast to
u32 config fields, silently truncating hostile values into model geometry.

## Changes

- `src/gguf.cpp`:
  - `kMaxMetaDepth = 64` recursion cap in `skip_value` (verified upstream: a
    2 MB file with 180k nested one-element arrays crashed the parser);
  - `skip_value` templated on the reader type (`parse_gguf<R>` free-function
    parse core) so the file path and the in-memory path share one
    implementation — a member template cannot be partially specialised;
  - u64-width gate: `set()` only applies values that fit `uint32_t`; the file
    remains structurally valid but the hostile value is not applied
    (`embedding_length == 0` for 2^32+4096 instead of truncated 4096);
  - `BufferReader` — in-memory reader with the same interface as the file
    reader (bounds-checked `read`/`bytes`, 16 MiB string cap);
  - `inspect_bytes(data,size)` — public in-memory parse surface;
  - MXFP4 (GGML type 39) added to the `read_tensor_f32` decoder map.
- `include/zen/gguf.h`: `inspect_bytes` declaration with contract comment.
- `src/openai_server.cpp`: JSON parser `val()` recursion capped at 128 levels
  (verified upstream: a 400 kB body of `[` killed the server process).
- `tests/fuzz_gguf.cpp`: structured GGUF fuzzer copied from Tree 1
  (`ZINFUZZ_STANDALONE` mode: 4096 LCG cases + the 3 regression cases).
- `tests/core_tests.cpp`: three hostile-input regression blocks added:
  1. deep-array file/memory agreement — 80 nested arrays must be rejected by
     BOTH surfaces with identical error strings;
  2. u64-width gate — structurally valid image, value not applied;
  3. hostile buffers — overlong declared string length, null/empty buffer,
     2-byte truncated header: all fail closed; a valid minimal in-memory image
     parses with the correct architecture and aligned tensor offset.
- `CMakeLists.txt`: `zen_fuzz_gguf` target + `zen_fuzz_gguf` ctest (security
  label, 300 s timeout).

## Commands and results

```
H:\ZenInfer\ZenInfer-completion-work\build\zen_fuzz_gguf.exe
standalone fuzz: 4096 random cases + 3 regression cases, no crash/hang
```

ctest AVX2 and scalar (both trees): 7/7 passed, including
`zen_fuzz_gguf` (labels: security;fuzz;gguf) and the hostile-input blocks in
`zen_core_tests`. Full logs in `docs/evidence/ZI-012-kquant-iq-mxfp4-decoders.md`.

## Known limitations

- This is a structured fuzzer over GGUF-shaped inputs, not coverage-guided
  libFuzzer/ASan instrumentation; a clang+libFuzzer run remains open.
- No sanitizer (ASan/UBSan/TSan) builds yet on this toolchain.
- Windows/MSVC validation remains a separate workstream (ZI-017).

## Status

ZI-016 moves from PARTIAL: the two known DoS paths are closed, the u64-width
gate is enforced, the in-memory parse surface exists and is fuzz-tested
standalone, and hostile-input regressions are wired into the default test
suite. Sanitizer and coverage-guided fuzzing remain open.
