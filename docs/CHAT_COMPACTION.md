# Opt-in chat context compaction (prototype)

Context compaction can reduce the working prompt for long chats. It is **off by
default** and does not guarantee better speed or answer quality. It uses the
loaded model to generate a lossy summary, then performs a separate final
inference. No model weights or original client messages are edited.

## Use

For `/v1/chat/completions` only, include the optional `context_compaction`
object. Its presence enables the feature; absence leaves existing behavior
unchanged. Settings are per request:

```json
{
  "model": "my-model",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Earlier conversation goes here."},
    {"role": "assistant", "content": "Earlier answer."},
    {"role": "user", "content": "Recent question."}
  ],
  "max_tokens": 128,
  "context_compaction": {
    "trigger_tokens": 4096,
    "target_tokens": 2048,
    "summary_max_tokens": 256,
    "keep_last_messages": 2
  }
}
```

Defaults are trigger 4096, target 2048, summary output 256, and 4 recent
messages. Token thresholds use the model tokenizer on the **rendered chat
prompt**, not characters or messages, and exclude the requested final output.
A prompt of exactly `trigger_tokens` is not compacted. Target must not exceed
trigger; all values are positive integers. Trigger/target are capped at
1,048,576, summary output at 4096 and recent messages at 1024. The model's
own context limit also applies.

If triggered, the server preserves the initial system-message prefix and the
last `keep_last_messages` user/assistant messages; all intervening messages
are summarized once at temperature 0. The summary is inserted as a **user**
message labeled as generated, potentially incomplete context—not as a system
instruction. The original history stays with the client. The summary call does
not use the request's sampling constraints, stop strings, bias, allowed-token
list, or speculative draft settings; the final generation still uses them.

The JSON response contains `zeninfer_compaction` for every opted-in request:
`applied`, `original_prompt_tokens`, `final_prompt_tokens`,
`summary_prompt_tokens`, `summary_completion_tokens`, `compacted_messages`, and
`summary` (empty when not applied). Regular `usage` describes **only the final
generation**; add the two summary counts to measure compaction's extra work.
The summary is returned to the same caller and can contain sensitive
information; protect response logs accordingly. For streaming responses a
separate SSE `data:` event contains `zeninfer_compaction` before the final
completion chunk and `[DONE]`. Normal streaming output contains only the final
answer, not the summary.

This server does not store or retrieve conversations. If the client sends the
full original history on every request, it may pay the summary cost repeatedly.
A client may retain the full history separately and explicitly send an earlier
summary plus recent messages on later turns, but must handle lost detail and
refreshing that summary itself. There is no built-in retrieval tool or
cross-request compaction cache.

## Failure behavior and limits

- The operation fails rather than silently dropping more messages when system
  messages are interleaved with conversation turns, there are no eligible older
  messages, summary generation fails/returns empty, the summary reaches its
  output limit without stopping, or the compacted prompt exceeds the requested
  target or model context after reserving the final output budget.
- This first version summarizes the older messages in **one** call. If the
  rendered summary input plus its reserved output exceeds the model context, it
  returns an error; compact earlier or use a larger-context model. There is no
  chunked/hierarchical summarization yet.
- Errors are HTTP 400 with code `context_compaction_error` for non-streaming
  generation; once streaming headers have been sent, errors appear in an SSE
  event followed by `[DONE]`. Invalid options are HTTP 400 before streaming.
- Summaries are lossy and may omit, invent or misstate facts. Treat history and
  the summary as untrusted input; the instruction to treat history as data is
  not a prompt-injection defense. Preserve and selectively retrieve original
  records where exact facts or auditability matter.
- No trained model or target CPU performance/quality comparison was available
  for this prototype. The included tests use a deterministic fake model and
  HTTP mock; they do not establish summary quality or real-model speedup.

## Measure before using in production

On the target machine, compare identical chats and generation settings with the
extension absent and present. Measure wall time for the entire request, summary
and final prompt/completion tokens, final decode latency and memory—not merely
the shorter final prompt. Include cold and warm/prefix-cache runs, several
history lengths, and tasks that require exact earlier facts. Repeat for the
intended GGUF/model and worker settings. A shorter final context can save decode
work but extra summarization, tokenization and re-prefill can outweigh that
saving, particularly for short outputs or repeated stateless requests.

See also: [You don't need long context](https://arxiv.org/html/2611.05306v1).

## LLVM IR and CPU performance

Hand-writing LLVM IR is not inherently faster than optimized C++. Clang already
lowers C++ to LLVM IR, while the current Windows MSVC path uses a different
compiler backend. Explicit AVX2/FMA kernels, memory access, quant decoding,
threading and measured bottlenecks are generally more important than expressing
the whole runtime in IR. LLVM IR would add platform/build maintenance and still
need scalar fallbacks, numerical tests and runtime CPU feature dispatch.

If a hot kernel remains slow, first benchmark optimized Clang versus MSVC on the
same CPU/model/quant, inspect generated assembly and profiling counters, then
compare a narrowly scoped intrinsic or LLVM-IR experiment against the existing
kernel with parity and end-to-end tests. No LLVM-versus-MSVC comparison was run
for this change.
