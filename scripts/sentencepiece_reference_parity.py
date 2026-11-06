#!/usr/bin/env python3
"""Independent SentencePiece parity check for GGUF Llama tokenizers.

Requires the sentencepiece Python package. ZenInfer exports GGUF vocabulary,
scores and token types; Google's SentencePiece processor is reconstructed as
an independent tokenizer oracle. This is a reproducible parity test, not a
claim that GGUF contains the original SentencePiece normaliser model.
"""
from __future__ import annotations
import argparse, json, re, subprocess, sys, tempfile
from pathlib import Path

CORPUS = [
    "Once upon a time", "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "ROCm RDNA4 Vulkan compute",
    "Once upon a time there was a little girl.",
    " leading and trailing ", "line one\nline two", "മലയാളം ഭാഷാ പരിശോധന",
]

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("exporter",type=Path);ap.add_argument("generator",type=Path);ap.add_argument("gguf",type=Path)
    ap.add_argument("--output",type=Path,default=Path("sentencepiece-parity.json"));args=ap.parse_args()
    try:
        import sentencepiece as spm
        from sentencepiece import sentencepiece_model_pb2 as pb
    except Exception as exc:
        print(f"ERROR: install sentencepiece: {exc}",file=sys.stderr);return 2
    with tempfile.TemporaryDirectory() as td:
        exported=Path(td)/"tokenizer.json";model_path=Path(td)/"reference.model"
        run=subprocess.run([str(args.exporter),str(args.gguf),str(exported)],capture_output=True,text=True)
        if run.returncode: print(run.stderr,file=sys.stderr);return run.returncode
        data=json.loads(exported.read_text(encoding="utf-8"))
        if data["tokenizer_model"]!="llama": print("ERROR: GGUF tokenizer is not SentencePiece/llama",file=sys.stderr);return 2
        proto=pb.ModelProto();proto.trainer_spec.model_type=pb.TrainerSpec.UNIGRAM
        proto.trainer_spec.vocab_size=len(data["tokens"]);proto.trainer_spec.unk_id=data["unk"]
        proto.trainer_spec.bos_id=data["bos"] if data["add_bos"] else -1
        proto.trainer_spec.eos_id=data["eos"] if data["add_eos"] else -1
        proto.trainer_spec.pad_id=-1;proto.trainer_spec.byte_fallback=any(t==6 for t in data["types"])
        proto.normalizer_spec.name="nmt_nfkc";proto.normalizer_spec.add_dummy_prefix=data["add_space_prefix"]
        proto.normalizer_spec.remove_extra_whitespaces=True;proto.normalizer_spec.escape_whitespaces=True
        for i,(piece,score,kind) in enumerate(zip(data["tokens"],data["scores"],data["types"])):
            p=proto.pieces.add();p.piece=piece;p.score=score
            p.type=(pb.ModelProto.SentencePiece.UNKNOWN if i==data["unk"] else kind)
        model_path.write_bytes(proto.SerializeToString());reference=spm.SentencePieceProcessor(model_file=str(model_path))
        rows=[];failed=0
        for text in CORPUS:
            expected=reference.encode(text,out_type=int)
            if data["add_bos"]: expected.insert(0,data["bos"])
            if data["add_eos"]: expected.append(data["eos"])
            got_run=subprocess.run([str(args.generator),str(args.gguf),"--prompt-text",text,"--max-new","1"],capture_output=True,text=True)
            match=re.search(r"^prompt_ids=(.*)$",got_run.stdout,re.M)
            got=[int(v) for v in match.group(1).split(",") if v] if match else []
            ok=got_run.returncode==0 and got==expected;failed+=not ok
            rows.append({"text":text,"status":"pass" if ok else "mismatch","expected":expected,"actual":got,"stderr":got_run.stderr[-1000:]})
        report={"reference":"Google SentencePiece","sentencepiece_version":spm.__version__,"gguf":str(args.gguf),"cases":len(rows),"failed":failed,"results":rows}
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps({k:report[k] for k in ("reference","sentencepiece_version","cases","failed")}))
        return 1 if failed else 0
if __name__=="__main__":raise SystemExit(main())
