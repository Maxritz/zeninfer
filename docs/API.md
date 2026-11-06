# ZenInfer OpenAI-compatible HTTP API

| Method | Route | Behaviour |
|---|---|---|
| `GET` | `/health` | Server/backend health |
| `GET` | `/v1/models` | Loaded model list |
| `POST` | `/v1/chat/completions` | Text Chat Completions, JSON or SSE |
| `POST` | `/v1/completions` | Text Completions, JSON or SSE |
| `GET` | `/metrics` | Resource Governor metrics |
| `GET` | `/debug/sherlock` | Aggregated Sherlock trace |
| `GET` | `/debug/breadcrumbs` | Recent Breadcrumb events |

## Start

```bash
./build/zen_server --model ./model.gguf --host 127.0.0.1 --port 8000 --trace
```

Windows:

```powershell
.\build\Release\zen_server.exe --model .\model.gguf --host 127.0.0.1 --port 8000 --trace
```

Options include `--model`, `--model-id`, `--host`, `--port`, `--api-key`, `--max-connections`, `--draft-model`, `--trace` and `--breadcrumbs`. Environment variables `ZEN_HOST`, `ZEN_PORT` and `ZEN_API_KEY` are supported. Breadcrumbs default on in the server; trace aggregation is opt-in.

If no model is loaded, completion requests return HTTP 503 `model_not_loaded`.

## Supported request fields

Both completion endpoints support:

- `max_tokens` (integer 1-65536; `max_completion_tokens` is also accepted)
- `temperature` (0-5; zero selects greedy sampling)
- `top_k` (integer 0-32000; zero disables top-k truncation)
- `top_p` (exclusive lower bound 0, inclusive upper bound 1)
- `seed` (non-negative safe integer for reproducible sampling)
- `stop` (one string or up to four non-empty strings)
- `logit_bias` (object of token-ID keys to finite biases in [-100, 100])
- `allowed_token_ids` (non-empty array of token IDs; a simple constrained-token allow-list)
- `speculative_tokens` (0-16; greedy speculative decoding proposals, requires `--draft-model`)
- `stream` (boolean; SSE supported by both endpoints)

Chat messages accept `system`, `user` and `assistant` roles with string content. Common Llama 2, Llama 3 and ChatML prompt layouts are selected from the GGUF chat-template markers. Speculative decoding is greedy-only and requires the draft model to use the exact same token vocabulary as the target. This is not a general Jinja-template interpreter.

## Diagnostics and resource policy

- `--trace` enables Sherlock aggregation. Query `/debug/sherlock` for inclusive/exclusive timings.
- Breadcrumbs use a 4096-entry atomic ring. Query `/debug/breadcrumbs` for recent events.
- `/metrics` reports CPU affinity, available memory, cgroup v2 quota/memory limits where available, worker count and memory budgets.
- Model matrices use file-backed mapping and OS-managed demand paging. Prefetch hints use `madvise(MADV_WILLNEED)` on Linux and dynamically resolved `PrefetchVirtualMemory` on Windows.
- KV cache allocation is checked against the Resource Governor transient budget. A one-entry exact-prompt prefix cache is retained when it fits the cache budget.

## Validation and operational limits

- Request body size is capped (4 MiB by default); prompt and message sizes and sampling fields are type/range checked.
- Optional bearer-key authentication applies to all routes, including health and diagnostics.
- Defaults to loopback. No TLS, rate limiting, request deadlines or production-grade abuse protection.
- Tools/function calling, multimodal content, JSON-schema grammar constraints, logprobs and the Responses API are not implemented.

For API contract testing without a model, build `zen_mock_server` and run the `zen_http_api_smoke` CTest. Mock responses are not model inference.
