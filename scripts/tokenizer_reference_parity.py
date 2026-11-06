#!/usr/bin/env python3
"""Compare GGUF tokenizer IDs with an independent Hugging Face tokenizer.

Requires transformers. This is deliberately an external oracle, not a test that
reuses ZenInfer's own tokenizer implementation. The model GGUF and HF tokenizer
must describe the same vocabulary/revision. Set HF_TOKEN if the reference is gated.
"""
from __future__ import annotations
import argparse, json, re, subprocess, sys
from pathlib import Path

CORPUS = [
    "Hello, world!", "The quick brown fox jumps over the lazy dog.",
    "ROCm RDNA4 Vulkan compute", "മലയാളം ഭാഷാ പരിശോധന",
    "नमस्ते दुनिया", "中文分词测试", "emoji: 🧠🚀",
    " leading and trailing ", "line one\nline two", "JSON: {\"ok\":true,\"n\":3}",
]

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("zen_generate_ids",type=Path)
    ap.add_argument("gguf",type=Path)
    ap.add_argument("hf_tokenizer",help="HF model ID or local tokenizer directory")
    ap.add_argument("--output",type=Path,default=Path("tokenizer-parity.json"))
    args=ap.parse_args()
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        print(f"ERROR: install transformers to run independent parity: {exc}",file=sys.stderr);return 2
    tok=AutoTokenizer.from_pretrained(args.hf_tokenizer,trust_remote_code=False)
    rows=[]; failed=0
    for text in CORPUS:
        ref=tok.encode(text,add_special_tokens=True)
        run=subprocess.run([str(args.zen_generate_ids),str(args.gguf),"--prompt-text",text,"--max-new","1"],capture_output=True,text=True)
        if run.returncode:
            rows.append({"text":text,"status":"runtime_error","stderr":run.stderr[-2000:]});failed+=1;continue
        match=re.search(r"^prompt_ids=(.*)$",run.stdout,re.M)
        if not match:
            rows.append({"text":text,"status":"missing_prompt_ids"});failed+=1;continue
        got=[int(x) for x in match.group(1).split(",") if x]
        # Both implementations must agree on special-token policy. If BOS/EOS
        # differs, the report makes that explicit instead of silently trimming.
        ok=got==ref
        rows.append({"text":text,"status":"pass" if ok else "mismatch","zeninfer_ids":got,"huggingface_ids":ref})
        failed+=not ok
    report={"reference":"Hugging Face AutoTokenizer","model":args.hf_tokenizer,"gguf":str(args.gguf),"cases":len(rows),"failed":failed,"results":rows}
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("reference","model","cases","failed")}))
    return 1 if failed else 0
if __name__=="__main__": raise SystemExit(main())
