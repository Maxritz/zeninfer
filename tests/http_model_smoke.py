#!/usr/bin/env python3
"""HTTP end-to-end with a deterministic synthetic GGUF, not a mock backend."""
import json
from concurrent.futures import ThreadPoolExecutor
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

exe, model=sys.argv[1:]
with socket.socket() as sock:
    sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
proc=subprocess.Popen([exe,'--model',model,'--model-id','synthetic','--host','127.0.0.1','--port',str(port)],
                      stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,env={**os.environ,'ZEN_MAP_TRACE':'0'})
base=f'http://127.0.0.1:{port}'
def req(path,payload=None):
    data=None if payload is None else json.dumps(payload).encode()
    request=urllib.request.Request(base+path,data=data,headers={'Content-Type':'application/json'})
    return urllib.request.urlopen(request,timeout=15)
try:
    for _ in range(150):
        if proc.poll() is not None: raise RuntimeError(f'model server exited {proc.returncode}')
        try:
            with req('/health') as response:
                if json.load(response)['engine_loaded']: break
        except (OSError,urllib.error.URLError): time.sleep(.05)
    else: raise RuntimeError('model server did not become ready')
    with req('/v1/models') as response:
        assert json.load(response)['data'][0]['id']=='synthetic'
    base_request={'model':'synthetic','prompt':'Once upon a time','temperature':0,
                  'allowed_token_ids':[42]}
    with req('/v1/completions',{**base_request,'max_tokens':1}) as response:
        short=json.load(response)
    with req('/v1/completions',{**base_request,'max_tokens':6}) as response:
        long=json.load(response)
    assert short['choices'][0]['text']=='G' and long['choices'][0]['text']=='GGGGGG', (short,long)
    assert long['usage']['completion_tokens']==6
    with req('/v1/completions',{**base_request,'max_tokens':6,'stream':True}) as response:
        events=[line[6:] for line in response.read().decode().splitlines() if line.startswith('data: ')]
    assert events[-1]=='[DONE]'
    pieces=[json.loads(e)['choices'][0].get('text','') for e in events[:-1]]
    assert ''.join(pieces)=='GGGGGG', pieces
    chat={'model':'synthetic','messages':[{'role':'user','content':'Hello'}],
          'temperature':0,'max_tokens':4,'allowed_token_ids':[42]}
    with req('/v1/chat/completions',chat) as response:
        result=json.load(response)
    assert result['choices'][0]['message']['content']=='GGGG', result
    # Several simultaneous real inference requests with mixed lengths and
    # repeated/different prompts exercise request-local state and prefix reuse.
    def parallel_request(i):
        n=1+i%7
        with req('/v1/completions',{**base_request,'prompt':'Once upon a time' if i%2 else 'Different prompt', 'max_tokens':n}) as response:
            result=json.load(response)
        if result['choices'][0]['text']!='G'*n or result['usage']['completion_tokens']!=n:
            raise AssertionError(result)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(parallel_request,range(12)))
    try: req('/v1/completions',{**base_request,'model':'missing','max_tokens':1})
    except urllib.error.HTTPError as err:
        if err.code!=404: raise
    else: raise AssertionError('wrong model was accepted')
    print('Real server + synthetic GGUF: health/models, cache-growth, completion, SSE, chat and concurrency PASS')
finally:
    proc.terminate()
    try: proc.wait(timeout=5)
    except subprocess.TimeoutExpired: proc.kill();proc.wait(timeout=5)
