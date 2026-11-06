# ZenInfer model-neutral runtime boundary

## Intent

ZenInfer is an inference engine, not a Llama clone. Frontends and services must depend on model-neutral contracts. Model architecture, weight representation, tokenisation and CPU execution are separate concerns.

## Current runtime ownership

```text
OpenAI-compatible HTTP server
            |
            v
      ModelBackend
            |
            v
      ModelRuntime  <---- generic runtime contract
            |
            v
  load_model_runtime()
            |
            +---- GGUF architecture dispatch
                       |
                       +---- zen_model_adapters library
                                  |
                                  +---- DenseTransformerAdapter
                                             |
                                             v
                                    DenseTransformerRuntime
                                    (current dense execution)
```

`ModelRuntime` is defined in `include/zen/model_runtime.h`. It contains no reference to a particular architecture implementation. Shared generation/runtime value types live in `include/zen/runtime_types.h`. The generic `zen_core` library does not compile or link the dense transformer implementation. Concrete model adapters are built in the separate `zen_model_adapters` library.

`ModelBackend` is the HTTP-facing backend. It owns a `ModelRuntime` selected by the model factory. The server executable does not instantiate a Llama-named class.

`DenseTransformerAdapter` currently wraps the existing dense transformer implementation and is compiled under `src/adapters/`. `DenseTransformerRuntime` is the concrete implementation for the currently supported dense Llama and experimental Qwen2 GGUF layouts. It is an adapter implementation, not the public engine abstraction.

`ModelTokenizer` is the model-neutral tokenizer service name. The former `LlamaSentencePiece` type remains as a compatibility alias only; new code must use `ModelTokenizer`.

`ArchitectureDescriptor` in the model IR describes recognised metadata and tensor roles. It is deliberately not called an executable adapter. Executable dispatch is performed by `create_model_runtime(const ModelConfig&)`.

## Dispatch and support policy

The factory uses the complete GGUF `general.architecture` string, not only the coarse family enum:

| GGUF architecture | Factory behaviour | Runtime status |
|---|---|---|
| `llama` | Dense transformer adapter | Existing execution path |
| `qwen2` | Dense transformer adapter | Experimental existing execution path |
| `qwen2moe` | Reject | No MoE execution |
| `qwen3` | Reject | No Qwen3 adapter |
| `qwen3moe` | Reject | No Qwen3 MoE adapter |
| BitNet family | Reject | No BitNet adapter |
| Bonsai family | Reject | No Bonsai adapter |
| Unknown | Reject | No adapter |

Recognition is not support. Unsupported architectures fail in the factory with an explicit architecture-specific error rather than entering a superficially compatible forward pass.

## Extension contract

A new architecture implementation must:

1. Implement `ModelRuntime` or a reusable adapter around it.
2. Validate architecture-specific GGUF metadata and tensor names/shapes before execution.
3. Own its tensor mapping, attention/FFN/router semantics and positional encoding rules.
4. Reuse model-neutral tokenisation, generation options, diagnostics, resource management and HTTP interfaces where contracts match.
5. Register only after executable load and generation tests exist for representative fixtures.
6. Pass architecture-specific parity tests against an independent reference implementation.

Weight formats are not architecture identifiers. GGUF quantisation types must be decoded by the quantisation layer and supplied to the selected architecture adapter without embedding architecture assumptions in the decoder.

## Verified in this source revision

- Generic `ModelRuntime` contract is independent of `DenseTransformerRuntime` headers.
- HTTP server depends on `ModelBackend`, which owns a factory-selected `ModelRuntime`.
- Full architecture-string dispatch distinguishes `qwen2` from `qwen3` and `qwen2moe`.
- Runtime fixture tests load through `load_model_runtime`, not by directly constructing the dense implementation.
- Existing dense Llama fixtures and the existing Qwen2 experimental path remain subject to their current limitations.

## Not yet implemented

This refactor does not claim completion of the following requested capabilities:

- Qwen3 dense execution, Qwen3 MoE routing/expert execution, BitNet execution or Bonsai execution.
- General Jinja2-compatible chat-template interpretation. The current HTTP backend still has legacy template heuristics.
- Native CPU FP16 arithmetic. F16 storage/decode support is not equivalent to native FP16 arithmetic.
- Application-managed NVMe page eviction and streaming. Existing memory mapping/prefetch is not an application-controlled page cache.
- General grammar/JSON-schema constrained sampling.
- Stochastic speculative decoding. Existing speculative path is greedy-only.
- Independent external tokenizer parity for all supported tokenizer families.
- Independent Qwen2 reference parity and verified Windows/MSVC execution.

These remain implementation work, not documentation-only features.
