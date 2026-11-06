import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

exe = sys.argv[1]
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
proc = subprocess.Popen([exe, str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
base = f"http://127.0.0.1:{port}"

def request(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=5)

try:
    for _ in range(80):
        try:
            if request("/health").status == 200:
                break
        except Exception:
            time.sleep(0.05)
    else:
        raise RuntimeError("mock server did not become ready")

    completion = {"model": "zeninfer-mock-test", "prompt": "hello", "max_tokens": 32, "stream": True}
    with request("/v1/completions", completion) as response:
        body = response.read().decode()
    if "text_completion" not in body or '"text":' not in body or "[DONE]" not in body:
        raise RuntimeError("/v1/completions SSE payload is malformed")

    chat = {"model": "zeninfer-mock-test", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 32, "stream": True}
    with request("/v1/chat/completions", chat) as response:
        body = response.read().decode()
    if "chat.completion.chunk" not in body or '"delta":' not in body or "[DONE]" not in body:
        raise RuntimeError("/v1/chat/completions SSE payload is malformed")

    invalid = {"model": "zeninfer-mock-test", "prompt": "hello", "temperature": "hot"}
    try:
        request("/v1/completions", invalid)
    except urllib.error.HTTPError as error:
        if error.code != 400:
            raise
    else:
        raise RuntimeError("invalid temperature was accepted")

    stochastic_spec = {"model": "zeninfer-mock-test", "prompt": "hello", "max_tokens": 2,
                       "temperature": 0.8, "top_k": 8, "top_p": 0.9, "speculative_tokens": 2}
    with request("/v1/completions", stochastic_spec) as response:
        if response.status != 200:
            raise RuntimeError("stochastic speculative request was not accepted by the API")

    # The extension is opt-in, validates its settings, and reports summary work.
    opts = {"trigger_tokens": 512, "target_tokens": 256, "summary_max_tokens": 32, "keep_last_messages": 2}
    chat_base = {"model": "zeninfer-mock-test", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 32}
    with request("/v1/chat/completions", chat_base) as response:
        if "zeninfer_compaction" in json.load(response):
            raise RuntimeError("default request unexpectedly included compaction")
    with request("/v1/chat/completions", {**chat_base, "context_compaction": opts}) as response:
        data = json.load(response)
    info = data["zeninfer_compaction"]
    if not info["applied"] or info["original_prompt_tokens"] != 512 or info["final_prompt_tokens"] != 256 or info["compacted_messages"] != 2 or info["summary"] != 'Mock "summary"':
        raise RuntimeError("compaction metadata or option propagation is wrong")
    with request("/v1/chat/completions", {**chat_base, "context_compaction": opts, "stream": True}) as response:
        events = [line[6:] for line in response.read().decode().splitlines() if line.startswith("data: ")]
    decoded = [json.loads(event) for event in events if event != "[DONE]"]
    if events[-1] != "[DONE]" or not any(e.get("zeninfer_compaction", {}).get("applied") for e in decoded):
        raise RuntimeError("streaming compaction metadata is missing")
    for invalid_opts in (None, True, {"target_tokens": 1000, "trigger_tokens": 1}, {"keep_last_messages": 0}, {"summary_max_tokens": "32"}, {"summary_max_tokens": 4097}, {"unknown": 1}):
        try:
            request("/v1/chat/completions", {**chat_base, "context_compaction": invalid_opts})
        except urllib.error.HTTPError as error:
            if error.code != 400: raise
        else:
            raise RuntimeError(f"invalid compaction settings accepted: {invalid_opts!r}")
    try:
        request("/v1/completions", {"model": "zeninfer-mock-test", "prompt": "hi", "context_compaction": opts})
    except urllib.error.HTTPError as error:
        if error.code != 400: raise
    else:
        raise RuntimeError("compaction was accepted for text completions")
    failed = {**chat_base, "messages": [{"role": "user", "content": "FAIL_COMPACTION"}], "context_compaction": opts}
    try:
        request("/v1/chat/completions", failed)
    except urllib.error.HTTPError as error:
        data = json.load(error)
        if error.code != 400 or data["error"]["code"] != "context_compaction_error": raise
    else:
        raise RuntimeError("compaction failure was not returned")
    with request("/v1/chat/completions", {**failed, "stream": True}) as response:
        events = [line[6:] for line in response.read().decode().splitlines() if line.startswith("data: ")]
    decoded = [json.loads(event) for event in events if event != "[DONE]"]
    if events[-1] != "[DONE]" or not any(e.get("error", {}).get("code") == "context_compaction_error" for e in decoded):
        raise RuntimeError("streaming compaction failure was not returned")

    # Unsupported backends must reject the extension rather than silently ignore it.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        other_port = sock.getsockname()[1]
    other = subprocess.Popen([exe, str(other_port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             env={**os.environ, "ZEN_MOCK_NO_COMPACTION": "1"})
    try:
        url = f"http://127.0.0.1:{other_port}"
        for _ in range(80):
            try:
                with urllib.request.urlopen(url + "/health", timeout=1): break
            except Exception: time.sleep(0.05)
        payload = json.dumps({**chat_base, "context_compaction": opts}).encode()
        try:
            urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", data=payload,
                headers={"Content-Type": "application/json"}), timeout=5)
        except urllib.error.HTTPError as error:
            if error.code != 400: raise
        else:
            raise RuntimeError("unsupported backend silently accepted compaction")
    finally:
        other.terminate()
        try: other.wait(timeout=5)
        except subprocess.TimeoutExpired:
            other.kill()
            other.wait(timeout=5)

    print("HTTP API smoke: PASS")
finally:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
