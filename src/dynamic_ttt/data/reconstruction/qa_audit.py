#!/usr/bin/env python3
"""Read-only final auditor. It intentionally imports no primary-builder code."""
import hashlib, json, os, statistics, sys, tarfile
from collections import Counter
from pathlib import Path

R = Path(os.environ["QA_BUILD_ROOT"])
T = R / "source_cache/tokenizers/Qwen3-8B-Base"
MAXLEN = 32768
TARGETS = {"NaturalQuestionsShort_MRQA":12000000,"SQuAD_MRQA":9000000,"NewsQA_MRQA":4500000,
           "TriviaQA-web_MRQA":3000000,"HotpotQA_MRQA":1500000}

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()
def dump(p,x):
    q=p.with_suffix(p.suffix+".tmp")
    with open(q,"w",encoding="utf8",newline="\n") as f: json.dump(x,f,indent=2,sort_keys=True);f.write("\n");f.flush();os.fsync(f.fileno())
    os.replace(q,p)

def main():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(T,local_files_only=True)
    masks={x["id"]:x for x in map(json.loads,open(R/"train/masks.jsonl",encoding="utf8"))}
    logical=qa=bad_schema=over=oob=mask_bad=0; records=0; lengths=[]; sources=Counter()
    with open(R/"train/train.jsonl",encoding="utf8") as f:
        for line in f:
            x=json.loads(line); records+=1
            if set(x)!={"id","metadata","n_tokens","text"}:bad_schema+=1
            ids=tok(x["text"],add_special_tokens=False).input_ids;n=x["n_tokens"];logical+=n;lengths.append(n)
            if len(ids)!=n:bad_schema+=1
            if n>MAXLEN:over+=1
            m=masks.get(x["id"])
            if not m or m["n_tokens"]!=n:mask_bad+=1;continue
            if any(not(0<=a<b<=n) for a,b in m["answer_ranges"]):oob+=1
            expected=[]
            for u in x["metadata"]["unit_provenance"]:
                sources[u["source"]]+=u["logical_token_length"]
                expected.extend([[u["token_offset"]+a,u["token_offset"]+b]
                                 for a,b in u["local_answer_ranges"]])
            if expected!=m["answer_ranges"]:mask_bad+=1
            qa+=x["metadata"]["qa_count"]
    hist=json.load(open(R/"audits/HISTORICAL_REPLAY_AUDIT.json",encoding="utf8"))
    cap=json.load(open(R/"audits/FIVE_SOURCE_CAPACITY_AUDIT.json",encoding="utf8"))
    sel=json.load(open(R/"audits/EXTENSION_SELECTION_AUDIT.json",encoding="utf8"))
    status=(logical==30000000 and dict(sources)==TARGETS and not any([bad_schema,over,oob,mask_bad]) and
            hist["status"]=="PASS" and cap["status"]=="PASS" and sel["selection_replay_parity"]=="PASS")
    out={"status":"PASS" if status else "FAIL","independent_dataset_audit":"PASS" if status else "FAIL",
         "answer_mask_independent_audit":"PASS" if status else "FAIL","records":records,"logical_tokens":logical,
         "qa_targets":qa,"source_tokens":dict(sources),"malformed_record_n":bad_schema,"overlength_record_n":over,
         "answer_range_oob_n":oob,"answer_mask_mismatch_n":mask_bad,"max_logical_length":max(lengths)}
    dump(R/"audits/FINAL_DATASET_AUDIT.json",out)
    dump(R/"audits/ANSWER_MASK_AUDIT.json",{"status":"PASS" if status else "FAIL",
         "answer_mask_independent_audit":"PASS" if status else "FAIL","answer_mask_mismatch_n":mask_bad,
         "answer_range_oob_n":oob})
    if not status: raise SystemExit(json.dumps(out))
    print(json.dumps(out,sort_keys=True))
if __name__=="__main__":main()
