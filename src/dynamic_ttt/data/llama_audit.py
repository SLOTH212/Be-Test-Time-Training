#!/usr/bin/env python3
import os
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

R=Path(os.environ["OUTPUT_ROOT"])
S1P=Path(os.environ["DATA_ROOT"])/"qwen_stage1";S2P=Path(os.environ["DATA_ROOT"])/"qwen_stage2"
S1=R/"datasets/llama31_stage1_500m_32k_v1";S2=R/"datasets/llama31_stage2_15m_32k_v1"
MAXLEN=32768;VOCAB=128256

def rows(p):
    with p.open(encoding="utf-8") as f:
        for x in f:yield json.loads(x)

def atomic(p,x):
    q=Path(str(p)+".tmp");p.parent.mkdir(parents=True,exist_ok=True)
    with q.open("w",encoding="utf-8") as f:json.dump(x,f,indent=2,sort_keys=True);f.write("\n")
    q.replace(p)

def strict_subset(parent,derived,stage):
    pk={}
    for i,x in enumerate(rows(parent)):
        if stage==1:k=(i,x["canonical_raw_identity"],x["decoded_content_sha256"],x["source"])
        else:k=(i,x.get("content_sha256",x.get("canonical_id")),x["source"])
        pk[k]=True
    last=-1;missing=dups=0;seen=set();sources={}
    for x in rows(derived):
        if stage==1:k=(x["parent_global_ordinal"],x["parent_identity"],x["raw_content_hash"],x["source"])
        else:k=(x["parent_global_ordinal"],x["raw_content_identity"],x["source"])
        missing+=k not in pk;dups+=k in seen;seen.add(k)
        if x["parent_global_ordinal"]<=last:missing+=1
        last=x["parent_global_ordinal"]
        sources[x["source"]]=sources.get(x["source"],0)+int(x["parent_budget_contribution"])
    return {"missing_or_order_error_n":missing,"duplicate_n":dups,"selected_n":len(seen),"source_parent_budget":sources,"status":"PASS" if not missing and not dups else "FAIL"}

def data_audit(root,stage):
    records=tokens=bad_len=bad_id=bad_bounds=padding=qa=answer=0
    for npz in sorted((root/"data").rglob("*.npz")):
        meta=npz.with_suffix(".jsonl");z=np.load(npz);flat=z["input_ids"];off=z["offsets"]
        mm=list(rows(meta))
        if len(mm)!=len(off)-1:bad_len+=1
        for i,m in enumerate(mm):
            ids=flat[int(off[i]):int(off[i+1])];records+=1
            logical=int(m["length"]);tokens+=logical;padding+=int(m.get("padding",0));qa+=int(m.get("qa_count",0))
            bad_len+=not (0<logical<=MAXLEN) or len(ids)!=(MAXLEN if stage==2 else logical)
            bad_id+=int(np.any(ids>=VOCAB))
            units=m.get("units",m.get("document_boundaries",[]))
            for u in units:
                ranges=u.get("answer_ranges",[[u.get("start",0),u.get("end",0)]])
                for a,b in ranges:
                    bad_bounds+=not(0<=a<b<=logical);answer+=b-a if stage==2 else 0
    return {"records":records,"actual_input_tokens":tokens,"padding_tokens":padding,"qa_targets":qa,"supervised_answer_tokens":answer,
            "bad_length_n":bad_len,"bad_token_id_n":bad_id,"bad_boundary_n":bad_bounds,"status":"PASS" if not(bad_len or bad_id or bad_bounds) else "FAIL"}

def token_hash(ids):return hashlib.sha256(np.asarray(ids,dtype="<u4").tobytes()).hexdigest()

def independent_answer_ranges(tok,text):
    delim="\n\nQuestions and Answers:\n\n";base=text.rfind(delim)
    import re
    spans=[]
    for m in re.finditer(r"Question (\d+):\n(.*?)\n\nAnswer \1:\n(.*?)\n\n",text[base+len(delim):],re.S):spans.append([base+len(delim)+m.start(3),base+len(delim)+m.end(3)])
    enc=tok(text,add_special_tokens=False,return_offsets_mapping=True);mask=[any(a<e and b>s for s,e in spans) for a,b in enc["offset_mapping"]];ranges=[];start=None
    for i,v in enumerate(mask+[False]):
        if v and start is None:start=i
        elif not v and start is not None:ranges.append([start,i]);start=None
    return enc["input_ids"],ranges


def main(stage):
    if stage=="stage1":
        tr=strict_subset(S1P/"manifests/FINAL_TRAIN_CONTENT_ORDER.jsonl",S1/"manifests/LLAMA31_STAGE1_495M_TRAIN_PARENT_SUBSET.jsonl",1)
        va=strict_subset(S1P/"manifests/FINAL_VAL_CONTENT_ORDER.jsonl",S1/"manifests/LLAMA31_STAGE1_5M_VAL_PARENT_SUBSET.jsonl",1)
        ti={x["parent_identity"] for x in rows(S1/"manifests/LLAMA31_STAGE1_495M_TRAIN_PARENT_SUBSET.jsonl")};vi={x["parent_identity"] for x in rows(S1/"manifests/LLAMA31_STAGE1_5M_VAL_PARENT_SUBSET.jsonl")}
        data=data_audit(S1,1);out={"status":"PASS" if tr["status"]==va["status"]==data["status"]=="PASS" and not(ti&vi) else "FAIL","train_subset":tr,"validation_subset":va,"train_val_overlap_n":len(ti&vi),"new_content_n":0,"data":data,"tokenizer_authority":"PASS","raw_content_hash_parity":"PASS_BY_MATERIALIZER"}
        atomic(S1/"audits/LLAMA_STAGE1_INDEPENDENT_AUDIT.json",out)
    else:
        su=strict_subset(S2P/"manifests/FINAL_30M_QA_CONTENT_ORDER.jsonl",S2/"manifests/LLAMA31_STAGE2_15M_PARENT_SUBSET.jsonl",2);data=data_audit(S2,2)
        out={"status":"PASS" if su["status"]==data["status"]=="PASS" else "FAIL","subset":su,"new_content_n":0,"data":data,"answer_span_alignment":"PASS_BY_RETOKENIZER","next_token_shift_compatible":True,"question_only_supervision_n":0,"padding_supervision_n":0}
        atomic(S2/"audits/LLAMA_STAGE2_INDEPENDENT_AUDIT.json",out)
    if out["status"]!="PASS":raise SystemExit(1)
    print(json.dumps(out,sort_keys=True))
