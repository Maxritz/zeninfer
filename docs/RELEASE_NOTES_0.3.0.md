# ZenInfer 0.3.0-alpha

This release moves the project towards a model-neutral inference architecture.

Added:
- Architecture-neutral model IR and initial model-family recognition.
- Tensor role classification.
- Quantisation format registry with explicit implemented/unimplemented states.
- Reference decoders for F32, F16, Q4_0, Q8_0, ternary2 and generic signed Q1-Q18.
- Scalar quantised GEMV for implemented encodings.
- GGUF v2/v3 directory inspector.
- `zen_model_inspect` command-line utility.
- Core unit tests and CTest integration.
- Format support and architecture design documentation.

Not included:
- Complete GGUF model loading, tokenizer, executable model adapters or actual generation.
- Full requested Q1-Q18 family-specific formats. Generic Q1_SIGNED ... Q18_SIGNED are ZenInfer reference encodings only.
- All GGUF quantisations. Several common formats are registered as recognised but unimplemented.
