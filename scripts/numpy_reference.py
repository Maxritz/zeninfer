#!/usr/bin/env python3
"""Independent NumPy FP32 reference for ZenInfer's supported dense Llama path.

The exporter supplies already dequantised GGUF tensors. This isolates transformer
math from the GGUF decoder and compares the C++ runtime's final prompt logits.
"""
import argparse, json, struct
from pathlib import Path
import numpy as np


def read_export(path):
    with open(path, 'rb') as f:
        if f.read(8) != b'ZNPAR01\0': raise ValueError('bad ZNPAR magic')
        E,L,H,KH,F,V,C,RD = struct.unpack('<8I', f.read(32))
        rope,eps = struct.unpack('<2f', f.read(8))
        nt, = struct.unpack('<I', f.read(4))
        tensors={}; dims={}
        for _ in range(nt):
            nl,=struct.unpack('<I',f.read(4)); name=f.read(nl).decode()
            nd,=struct.unpack('<I',f.read(4)); ds=struct.unpack('<'+'Q'*nd,f.read(8*nd))
            n,=struct.unpack('<Q',f.read(8)); raw=f.read(4*n)
            if len(raw)!=4*n: raise ValueError('truncated tensor '+name)
            # GGUF dimension 0 is contiguous. For matrix tensors this means
            # [input, output] in GGUF and [output, input] for NumPy matvec.
            a=np.frombuffer(raw,dtype='<f4').copy()
            if nd==2: a=a.reshape((ds[1],ds[0]))
            else: a=a.reshape(tuple(reversed(ds)))
            tensors[name]=a; dims[name]=ds
        if f.read(1): raise ValueError('trailing export bytes')
    return (E,L,H,KH,F,V,C,RD,rope,eps),tensors


def matvec(w,x):
    # Accumulate in float64 like ZenInfer's scalar C++ reference, then round to FP32.
    return np.sum(w.astype(np.float64)*x.astype(np.float64)[None,:],axis=1,dtype=np.float64).astype(np.float32)


def rms(x,w,eps):
    ss=np.sum(x.astype(np.float64)*x.astype(np.float64),dtype=np.float64)
    inv=np.float32(1.0/np.sqrt(np.float32(ss/len(x))+np.float32(eps)))
    return (x*inv)*w


def rope(x,heads,head_dim,rotary_dim,pos,base):
    y=x.copy(); rotary_dim=min(rotary_dim,head_dim); rotary_dim-=rotary_dim%2
    half=rotary_dim//2
    for h in range(heads):
        off=h*head_dim
        for i in range(half):
            freq=np.float32(np.power(np.float32(base),np.float32(-2.0*float(i)/float(rotary_dim))))
            angle=np.float32(np.float32(pos)*freq)
            c=np.float32(np.cos(angle)); s=np.float32(np.sin(angle))
            a=y[off+i]; b=y[off+i+half]
            y[off+i]=np.float32(a*c-b*s); y[off+i+half]=np.float32(a*s+b*c)
    return y


def forward(token,pos,kv_k,kv_v,cfg,t):
    E,L,H,KH,F,V,C,RD,rope_base,eps=cfg
    D=E//H
    x=t['token_embd.weight'][token].copy()
    for layer in range(L):
        p=f'blk.{layer}.'
        norm=rms(x,t[p+'attn_norm.weight'],eps)
        q=matvec(t[p+'attn_q.weight'],norm)
        k=matvec(t[p+'attn_k.weight'],norm)
        v=matvec(t[p+'attn_v.weight'],norm)
        q=rope(q,H,D,RD,pos,rope_base)
        k=rope(k,KH,D,RD,pos,rope_base)
        kv_k[layer][pos]=k.reshape(KH,D); kv_v[layer][pos]=v.reshape(KH,D)
        att=np.zeros(E,dtype=np.float32); scores=np.empty(pos+1,dtype=np.float32)
        group=H//KH; inv_scale=np.float32(1.0/np.sqrt(np.float32(D)))
        for h in range(H):
            kh=h//group; qh=q[h*D:(h+1)*D]
            mx=np.float32(-np.inf)
            for step in range(pos+1):
                dot=np.sum(qh.astype(np.float64)*kv_k[layer][step,kh].astype(np.float64),dtype=np.float64)
                scores[step]=np.float32(np.float32(dot)*inv_scale)
                mx=max(mx,scores[step])
            denom=0.0
            for step in range(pos+1):
                scores[step]=np.float32(np.exp(np.float32(scores[step]-mx)))
                denom+=float(scores[step])
            inv=np.float32(1.0/denom)
            out=att[h*D:(h+1)*D]
            for step in range(pos+1):
                a=np.float32(scores[step]*inv)
                out += a*kv_v[layer][step,kh]
        tmp=matvec(t[p+'attn_output.weight'],att)
        x=x+tmp
        norm=rms(x,t[p+'ffn_norm.weight'],eps)
        gate=matvec(t[p+'ffn_gate.weight'],norm)
        up=matvec(t[p+'ffn_up.weight'],norm)
        silu=(gate/(np.float32(1.0)+np.exp(-gate,dtype=np.float32))).astype(np.float32)
        ffn=silu*up
        tmp=matvec(t[p+'ffn_down.weight'],ffn)
        x=x+tmp
    norm=rms(x,t['output_norm.weight'],eps)
    out=t.get('output.weight',t['token_embd.weight'])
    return matvec(out,norm)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('export');ap.add_argument('probe_logits');ap.add_argument('--prompt-ids',required=True)
    ap.add_argument('--json-out');args=ap.parse_args()
    cfg,t=read_export(args.export); ids=[int(x) for x in args.prompt_ids.split(',') if x]
    E,L,H,KH,F,V,C,RD,rb,eps=cfg
    kv_k=np.zeros((L,C,KH,E//H),dtype=np.float32); kv_v=np.zeros_like(kv_k)
    ref=None
    for pos,token in enumerate(ids): ref=forward(token,pos,kv_k,kv_v,cfg,t)
    actual=np.fromfile(args.probe_logits,dtype='<f4')
    if actual.size!=V: raise ValueError(f'probe logits count {actual.size}, expected {V}')
    diff=np.abs(ref.astype(np.float64)-actual.astype(np.float64))
    denom=np.maximum(np.abs(ref.astype(np.float64)),1e-6)
    report={
      'prompt_ids':ids,'vocab':V,'layers':L,
      'max_abs_error':float(diff.max()),'mean_abs_error':float(diff.mean()),
      'rms_error':float(np.sqrt(np.mean(diff*diff))),
      'max_relative_error':float(np.max(diff/denom)),
      'reference_argmax':int(np.argmax(ref)),'zeninfer_argmax':int(np.argmax(actual)),
      'argmax_match':bool(np.argmax(ref)==np.argmax(actual)),
      'all_finite':bool(np.isfinite(ref).all() and np.isfinite(actual).all()),
      'top5_reference':np.argsort(ref)[-5:][::-1].tolist(),
      'top5_zeninfer':np.argsort(actual)[-5:][::-1].tolist()
    }
    print(json.dumps(report,indent=2))
    if args.json_out: Path(args.json_out).write_text(json.dumps(report,indent=2)+'\n')
    # Tight tolerance for same dequantised weights and same algorithm. Small FP32
    # implementation-order differences are expected; a large error is a failure.
    if not report['all_finite'] or not report['argmax_match'] or report['max_abs_error']>2e-3:
        raise SystemExit(1)

if __name__=='__main__': main()
