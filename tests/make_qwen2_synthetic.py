#!/usr/bin/env python3
"""Create a tiny deterministic Qwen2-layout GGUF (not a trained checkpoint).

Uses only the Python standard library; regenerated into the CMake build tree.
The F32 weights use a fixed seed so CPU runtime and NumPy reference tests can
run without downloading or checking in a large model.
"""
import random
import struct
import sys
from pathlib import Path

U32, F32, BOOL, STRING, ARRAY, I32 = 4, 6, 7, 8, 9, 5
E, L, H, KH, F, V = 32, 2, 4, 2, 64, 128
rng = random.Random(31036)

def string(s):
    b=s.encode('utf-8'); return struct.pack('<Q',len(b))+b

def value(t,v):
    if t==STRING: return string(v)
    if t==U32: return struct.pack('<I',v)
    if t==I32: return struct.pack('<i',v)
    if t==F32: return struct.pack('<f',v)
    if t==BOOL: return struct.pack('<B',v)
    if t==ARRAY:
        subtype, items=v
        return struct.pack('<IQ',subtype,len(items))+b''.join(value(subtype,x) for x in items)
    raise ValueError(t)

tokens=['<unk>','<s>','</s>','▁']+[chr(i) for i in range(33,127)]
tokens += [f'<unused-{i}>' for i in range(V-len(tokens))]
types=[2,3,3]+[1]*(V-3)
meta=[('general.architecture',STRING,'qwen2'),('general.name',STRING,'synthetic-qwen2-f32'),
      ('qwen2.embedding_length',U32,E),('qwen2.block_count',U32,L),
      ('qwen2.attention.head_count',U32,0 if '--invalid-heads' in sys.argv[2:] else H),('qwen2.attention.head_count_kv',U32,KH),
      ('qwen2.feed_forward_length',U32,F),('qwen2.context_length',U32,256),
      ('qwen2.rope.dimension_count',U32,E//H),('qwen2.rope.freq_base',F32,10000.0),
      ('qwen2.attention.layer_norm_rms_epsilon',F32,1e-5),
      ('tokenizer.ggml.model',STRING,'llama'),('tokenizer.ggml.tokens',ARRAY,(STRING,tokens)),
      ('tokenizer.ggml.token_type',ARRAY,(I32,types)),('tokenizer.ggml.bos_token_id',U32,1),
      ('tokenizer.ggml.eos_token_id',U32,2),('tokenizer.ggml.unknown_token_id',U32,0),
      ('tokenizer.ggml.add_bos_token',BOOL,1),('tokenizer.ggml.add_space_prefix',BOOL,1)]
tensors=[]
def tensor(name, dims, norm=False):
    n=1
    for d in dims: n*=d
    data=[rng.uniform(0.85,1.15) if norm else rng.uniform(-0.12,0.12) for _ in range(n)]
    tensors.append((name,dims,struct.pack('<'+str(n)+'f',*data)))

tensor('token_embd.weight',[E,V]); tensor('output.weight',[E,V]); tensor('output_norm.weight',[E],True)
for l in range(L):
    p=f'blk.{l}.'
    tensor(p+'attn_norm.weight',[E],True)
    tensor(p+'attn_q.weight',[E,E]);tensor(p+'attn_k.weight',[E,E*KH//H]);tensor(p+'attn_v.weight',[E,E*KH//H])
    tensor(p+'attn_output.weight',[E,E]);tensor(p+'ffn_norm.weight',[E],True)
    tensor(p+'ffn_gate.weight',[E,F]);tensor(p+'ffn_up.weight',[E,F]);tensor(p+'ffn_down.weight',[F,E])

out=bytearray(b'GGUF'+struct.pack('<IQQ',3,len(tensors),len(meta)))
for key,t,v in meta: out += string(key)+struct.pack('<I',t)+value(t,v)
offsets=[];offset=0
for _,_,data in tensors:
    offset=(offset+31)&~31;offsets.append(offset);offset+=len(data)
for (name,dims,data),offset in zip(tensors,offsets):
    out+=string(name)+struct.pack('<I',len(dims))+struct.pack('<'+'Q'*len(dims),*dims)+struct.pack('<IQ',0,offset)
out+=bytes((-len(out))%32);start=len(out)
for (_,_,data),offset in zip(tensors,offsets):
    out+=bytes(start+offset-len(out));out+=data
path=Path(sys.argv[1]);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(out)
print(f'wrote deterministic synthetic Qwen2-layout GGUF: {path} ({len(out)} bytes)')
