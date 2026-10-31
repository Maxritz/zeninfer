import json
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

    print("HTTP API smoke: PASS")
finally:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
