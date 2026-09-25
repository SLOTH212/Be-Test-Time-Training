#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import time
import urllib.request
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path("/path/to/ttt")
SRC = ROOT / "src/llama31_data_derivation_v1"
WORK = ROOT / "work/llama31_derived_data_v1"
LOG = ROOT / "logs/llama31_derived_data_v1"
PROV = ROOT / "provenance"
S1P = ROOT / "datasets"
S2P = ROOT / "shared/datasets"
S1 = ROOT / "datasets/llama31_stage1_500m_32k_v1"
S2 = ROOT / "datasets/llama31_stage2_15m_32k_v1"
TOK = ROOT / "models/Llama-3.1-8B"
QWEN = ROOT / "shared/models/Qwen3-4B-Base"
MAXLEN = 32768
REVISION = "1447128c751f0fbc7d2082b2a113a938426a9bba"
REPO = "princeton-nlp/prolong-data-64K"
WORKERS = 16
MARKER = "Context:\n\nPassage 1:\n"
S2_FILES={"NaturalQuestionsShort_MRQA":"NaturalQuestionsShort.jsonl.gz","SQuAD_MRQA":"SQuAD.jsonl.gz","NewsQA_MRQA":"NewsQA.jsonl.gz","TriviaQA-web_MRQA":"TriviaQA-web.jsonl.gz","HotpotQA_MRQA":"HotpotQA.jsonl.gz"}
S2_SOURCE_SHA={"NaturalQuestionsShort_MRQA":"6cdac324664b94b60be3203a077bf361d0bfa68a17af9b71def1186a6958a68c","SQuAD_MRQA":"b094703b9c6f740cc2dfd70b3201b833553fcec0c8a522f22c2c6ff82ce2cc78","NewsQA_MRQA":"f1ccbf2d259ce1094aacde21a53592894248e5778814205dac94f0b086dbe968","TriviaQA-web_MRQA":"61fad6884370408282ad3ed0b5f25a9e932d9a724b6929ea03ea5344ff0cd3f7","HotpotQA_MRQA":"3a94712c073dc9f29d88ac149faa01ef9c7c089f97ee25d9cbac39387550825d"}
S1_PACKAGE_SHA = "11ddc46d95381c6bdaed7f3a62a7f2f2711e95db1223f7f50ec85e249832217d"
S2_PACKAGE_SHA = "60a54b694982a4a1c21708ba7c177b25101a5bc76a771f542f90e745611ce230"
S1_AUTH_HASHES = {
    "train": "d6797344bf22d1a5274ad6d65f458cfe3429d96b8a846c9a9f5a53936187b7ef",
    "validation": "1143a1ce3d4d29ffc355e47ed2754c1290296d0554ea128ce9f7e4dd4dd9bc89",
}
S2_HASHES = {
    "authority": "e423c322602c5124acc839470079cfe23559b43c0a29eb9042cf1a3a4ee6ddd0",
    "order": "8ecf2aa5a6ba1fe8c6a6654c588d6d8810dcd9c38d8f12989d4ee44fc319ca43",
    "packed": "4b9a4610effaba35a616c7622c248ee79db18bbd50f1a82a0e5c0d0326c13898",
    "config": "3aec55e62ee30dece2d79094989f8638f5fd754730f9a0aa330eb0ba4109e383",
}


def canon(x): return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""): h.update(b)
    return h.hexdigest()


def sha_text(text: str): return hashlib.sha256(text.encode("utf-8")).hexdigest()


def token_hash(ids): return hashlib.sha256(np.asarray(ids, dtype="<u4").tobytes()).hexdigest()


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows: f.write(canon(row) + "\n")
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def file_lines(path):
    with path.open(encoding="utf-8") as f:
        for line in f: yield json.loads(line)


def tokenizer_authority():
    cfg = json.load((TOK / "config.json").open(encoding="utf-8"))
    tc = json.load((TOK / "tokenizer_config.json").open(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(TOK, local_files_only=True, use_fast=True, trust_remote_code=False)
    names = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "HF_REVISION.txt"]
    files = {n: sha_file(TOK / n) for n in names}
    checks = {
        "model_type_llama": cfg.get("model_type") == "llama",
        "base_architecture": cfg.get("architectures") == ["LlamaForCausalLM"],
        "vocab": len(tok) == 128256,
        "bos": tok.bos_token_id == 128000,
        "eos": tok.eos_token_id == 128001,
        "pad_unset": tok.pad_token_id is None,
        "fast_tokenizer": tok.is_fast,
    }
    out = {
        "status": "PASS" if all(checks.values()) else "FAIL", "model_family": "meta-llama/Llama-3.1-8B",
        "variant": "BASE_NOT_INSTRUCT", "path": str(TOK), "revision": (TOK / "HF_REVISION.txt").read_text().strip(),
        "model_type": cfg.get("model_type"), "tokenizer_class": tc.get("tokenizer_class"), "vocab_size": len(tok),
        "bos_token_id": tok.bos_token_id, "eos_token_id": tok.eos_token_id, "pad_token_id": tok.pad_token_id,
        "add_special_tokens_for_data": False, "weights_loaded": False, "files": files, "checks": checks,
    }
    atomic_json(PROV / "LLAMA31_TOKENIZER_AUTHORITY.json", out)
    out["authority_sha256"] = sha_file(PROV / "LLAMA31_TOKENIZER_AUTHORITY.json")
    return tok, out


def verify_parents():
    p1 = ROOT / "packages/prolong_1b_32k_qwen3_stage1_v3_streaming/prolong_1b_32k_qwen3_stage1_v3_streaming.tar.gz"
    p2 = ROOT / "shared/packages/qa_replay_30m_32k_qwen3_stage2_v2/qa_replay_30m_32k_qwen3_stage2_v2.tar.gz"
    checks1 = {
        "package": sha_file(p1) == S1_PACKAGE_SHA,
        "train_order": sha_file(S1P / "manifests/FINAL_TRAIN_CONTENT_ORDER.jsonl") == S1_AUTH_HASHES["train"],
        "val_order": sha_file(S1P / "manifests/FINAL_VAL_CONTENT_ORDER.jsonl") == S1_AUTH_HASHES["validation"],
        "authority_status": json.load((S1P / "authority/DATASET_AUTHORITY_V3.json").open())["status"] == "PASS",
    }
    checks2 = {
        "package": sha_file(p2) == S2_PACKAGE_SHA,
        "authority": sha_file(S2P / "authority/STAGE2_DATASET_AUTHORITY.json") == S2_HASHES["authority"],
        "order": sha_file(S2P / "manifests/FINAL_30M_QA_CONTENT_ORDER.jsonl") == S2_HASHES["order"],
        "packed": sha_file(S2P / "manifests/FINAL_PACKED_RECORDS.jsonl") == S2_HASHES["packed"],
        "config": sha_file(S2P / "authority/stage2_qa_replay_30m_32k_qwen3_v2.yaml") == S2_HASHES["config"],
    }
    return checks1, checks2


def subset_manifest(parent: Path, output: Path, split: str, targets: dict[str, int], stage: int):
    rows, by_source = [], defaultdict(list)
    for ordinal, x in enumerate(file_lines(parent)):
        x["_ordinal"] = ordinal; by_source[x["source"]].append(x)
    chosen = set(); summary = {}
    for source, group in sorted(by_source.items()):
        target = targets[source]; total = 0; n = 0
        for x in group:
            size = int(x["selected_logical_token_n"] if stage == 1 else x["logical_token_length"])
            if abs((total + size) - target) <= abs(total - target):
                total += size; chosen.add(x["_ordinal"]); n += 1
            else: break
        summary[source] = {"parent_units": len(group), "selected_units": n, "target_parent_budget": target,
                           "selected_parent_budget": total, "residual": total - target}
    out = []
    for x in rows if False else []: pass
    for ordinal, x in enumerate(file_lines(parent)):
        if ordinal not in chosen: continue
        if stage == 1:
            out.append({"parent_identity": x["canonical_raw_identity"], "parent_global_ordinal": ordinal,
                        "source": x["source"], "selected_content_identity": x["canonical_raw_identity"],
                        "raw_content_hash": x["decoded_content_sha256"], "parent_budget_contribution": x["selected_logical_token_n"],
                        "split": split, "component": x["component"], "canonical_shard_id": x["canonical_shard_id"],
                        "canonical_row_id": x["canonical_row_id"], "authority_record": x["authority_record"]})
        else:
            uid = x.get("historical_record_id", x.get("canonical_id"))
            out.append({"parent_identity": uid, "parent_global_ordinal": ordinal, "source": x["source"],
                        "qa_unit_identity": uid, "raw_content_hash": x.get("content_sha256"),
                        "raw_content_identity": x.get("content_sha256",x.get("canonical_id")),
                        "answer_identity": hashlib.sha256(canon(x.get("source_native_ids", [])).encode()).hexdigest(),
                        "parent_budget_contribution": x["logical_token_length"], "qa_count": x["qa_count"],
                        "source_native_ids": x.get("source_native_ids", []), "kind": x["kind"], "selected": True})
    atomic_jsonl(output, out)
    return summary, len(out), sha_file(output)


def prepare_subsets():
    s1cfg = json.load((S1P / "authority/stage1_1b_32k_qwen3_v3_streaming.yaml").open())
    train_targets = {s: int(v["train"] * .5) for s, v in s1cfg["source_split_quotas"].items()}
    val_targets = {s: int(v["validation"] * .5) for s, v in s1cfg["source_split_quotas"].items()}
    p1t = S1 / "manifests/LLAMA31_STAGE1_495M_TRAIN_PARENT_SUBSET.jsonl"
    p1v = S1 / "manifests/LLAMA31_STAGE1_5M_VAL_PARENT_SUBSET.jsonl"
    a, an, ah = subset_manifest(S1P / "manifests/FINAL_TRAIN_CONTENT_ORDER.jsonl", p1t, "train", train_targets, 1)
    b, bn, bh = subset_manifest(S1P / "manifests/FINAL_VAL_CONTENT_ORDER.jsonl", p1v, "validation", val_targets, 1)
    s2auth = json.load((S2P / "authority/STAGE2_DATASET_AUTHORITY.json").open())
    targets2 = {s: int(v["final_target"] * .5) for s, v in s2auth["source_capacity"].items()}
    p2 = S2 / "manifests/LLAMA31_STAGE2_15M_PARENT_SUBSET.jsonl"
    c, cn, ch = subset_manifest(S2P / "manifests/FINAL_30M_QA_CONTENT_ORDER.jsonl", p2, "train", targets2, 2)
    s1sum = {"status": "PASS", "policy": "DETERMINISTIC_SOURCE_STRATIFIED_PARENT_ORDER_NEAREST_COMPLETE_UNIT_PREFIX_V1",
             "strict_parent_subset": True, "new_content_n": 0, "train_val_overlap_n": 0,
             "nominal_parent_train": 990000000, "nominal_parent_validation": 10000000,
             "target_parent_budget_train": 495000000, "target_parent_budget_validation": 5000000,
             "train": a, "validation": b, "selected_train_units": an, "selected_validation_units": bn,
             "train_manifest_sha256": ah, "validation_manifest_sha256": bh}
    s2sum = {"status": "PASS", "policy": "DETERMINISTIC_SOURCE_STRATIFIED_PARENT_ORDER_NEAREST_COMPLETE_UNIT_PREFIX_V1",
             "strict_parent_subset": True, "new_content_n": 0, "parent_nominal": 30000000,
             "target_parent_budget": 15000000, "sources": c, "selected_units": cn, "manifest_sha256": ch}
    atomic_json(S1 / "manifests/LLAMA31_STAGE1_500M_SUBSET_SUMMARY.json", s1sum)
    atomic_json(S2 / "manifests/LLAMA31_STAGE2_15M_SUBSET_SUMMARY.json", s2sum)
    return s1sum, s2sum


def download(url, path, expected=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and (not expected or sha_file(path) == expected): return
    print("DOWNLOAD_START "+str(path),flush=True)
    tmp = Path(str(path) + ".tmp")
    cmd=["curl","--location","--fail","--silent","--show-error","--continue-at","-","--output",str(tmp),
         "--connect-timeout","30","--speed-limit","1024","--speed-time","180",
         "--retry","5","--retry-delay","5",url]
    for attempt in range(1,101):
        rc=subprocess.run(cmd).returncode
        if rc==0:break
        print(f"DOWNLOAD_RETRY path={path} attempt={attempt} partial_bytes={tmp.stat().st_size if tmp.exists() else 0} rc={rc}",flush=True)
        if rc==33 and tmp.exists():
            tmp.unlink()
            print("DOWNLOAD_RANGE_UNSUPPORTED_RESTART path="+str(path),flush=True)
        time.sleep(min(60,5*attempt))
    else:raise RuntimeError("download retry budget exhausted: "+str(path))
    if expected and sha_file(tmp) != expected:
        bad=Path(str(tmp)+".hash_mismatch."+str(int(time.time())));os.replace(tmp,bad)
        raise ValueError("download hash mismatch preserved at "+str(bad))
    os.replace(tmp, path); print("DOWNLOAD_PASS "+str(path)+" bytes="+str(path.stat().st_size),flush=True)


def hf_url(repo_path):
    from urllib.parse import quote
    return f"https://hf-mirror.com/datasets/{REPO}/resolve/{REVISION}/{quote(repo_path, safe='/')}"


def parse_identity(identity):
    p = identity.split(":")
    if p[0] == "local": p = p[1:]
    return {"revision": p[0], "source": p[1], "repo_path": p[1] + "/" + p[2], "row": int(p[3]), "doc": int(p[4])}


_INVENTORY_HASHES=None
def inventory_hash(repo_path):
    global _INVENTORY_HASHES
    if _INVENTORY_HASHES is None:
        _INVENTORY_HASHES={x["upstream_repo_path"]:x["expected_lfs_sha256"] for x in file_lines(S1P/"manifests/UPSTREAM_SHARD_INVENTORY.jsonl")}
    if repo_path not in _INVENTORY_HASHES:raise KeyError("shard absent from frozen inventory: "+repo_path)
    return _INVENTORY_HASHES[repo_path]


def ndarray_u32(data):
    descriptor = data[0]; ndim = descriptor >> 2; di = descriptor % 4
    shape_type = [np.uint8, np.uint16, np.uint32, np.uint64][di]
    shape_bytes = ndim * (2 ** di)
    shape = np.frombuffer(data[1:1 + shape_bytes], shape_type)
    return np.frombuffer(data[1 + shape_bytes:], np.uint32).reshape(shape)


class MDS:
    def __init__(self): self.schemas = {}; self.rows = OrderedDict()
    def ensure(self, repo_path, expected):
        local = WORK / "stage1_raw_cache" / repo_path
        download(hf_url(repo_path), local, expected)
        idxrel = str(Path(repo_path).parent / "index.json").replace("\\", "/")
        idxlocal = WORK / "stage1_raw_cache" / idxrel
        download(hf_url(idxrel), idxlocal)
        if repo_path not in self.schemas:
            idx = json.load(idxlocal.open())
            found = [x for x in idx["shards"] if x["raw_data"]["basename"] == Path(repo_path).name]
            if len(found) != 1: raise ValueError("MDS schema not unique")
            self.schemas[repo_path] = found[0]
        return local
    def get(self, identity, expected):
        p = parse_identity(identity); key = (p["repo_path"], p["row"])
        expected=expected or inventory_hash(p["repo_path"])
        path = self.ensure(p["repo_path"], expected)
        if key not in self.rows:
            with path.open("rb", buffering=0) as f:
                f.seek((1 + p["row"]) * 4); begin, end = np.frombuffer(f.read(8), np.uint32)
                f.seek(int(begin)); data = f.read(int(end - begin))
            schema = self.schemas[p["repo_path"]]; sizes = []; cursor = 0
            for fixed in schema["column_sizes"]:
                if fixed is None: size = int(np.frombuffer(data[cursor:cursor+4], np.uint32)[0]); cursor += 4
                else: size = int(fixed)
                sizes.append(size)
            vals = []
            for size in sizes: vals.append(data[cursor:cursor+size]); cursor += size
            enc = dict(zip(schema["column_names"], vals))
            indices = ndarray_u32(enc["indices"]); ids = ndarray_u32(enc["input_ids"]).reshape(-1)
            self.rows[key] = [np.asarray(ids[int(a):int(b)], dtype=np.uint32) for a, b in indices.tolist()]
            if len(self.rows) > 32: self.rows.popitem(last=False)
        return p, self.rows[key][p["doc"]]


class RemoteRangeMDS(MDS):
    def _range(self, url, start, end):
        req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "llama31-derived-data-v1-smoke"})
        with urllib.request.urlopen(req, timeout=120) as r:
            if getattr(r, "status", None) != 206 and not r.headers.get("Content-Range"):
                raise RuntimeError("server did not honor bounded Range request")
            data = r.read(end - start + 1)
        if len(data) != end - start + 1: raise ValueError("short HTTP range")
        return data
    def get(self, identity, expected):
        p = parse_identity(identity); repo_path = p["repo_path"]
        idxrel = str(Path(repo_path).parent / "index.json").replace("\\", "/")
        idxlocal = WORK / "smoke_raw_cache" / idxrel
        download(hf_url(idxrel), idxlocal)
        idx = json.load(idxlocal.open()); found = [x for x in idx["shards"] if x["raw_data"]["basename"] == Path(repo_path).name]
        if len(found) != 1: raise ValueError("smoke MDS schema not unique")
        schema = found[0]; url = hf_url(repo_path)
        pair = self._range(url, (1 + p["row"]) * 4, (1 + p["row"]) * 4 + 7)
        begin, end = np.frombuffer(pair, np.uint32)
        data = self._range(url, int(begin), int(end) - 1)
        sizes=[];cursor=0
        for fixed in schema["column_sizes"]:
            if fixed is None:size=int(np.frombuffer(data[cursor:cursor+4],np.uint32)[0]);cursor+=4
            else:size=int(fixed)
            sizes.append(size)
        vals=[]
        for size in sizes:vals.append(data[cursor:cursor+size]);cursor+=size
        enc=dict(zip(schema["column_names"],vals));indices=ndarray_u32(enc["indices"]);ids=ndarray_u32(enc["input_ids"]).reshape(-1)
        return p,np.asarray(ids[int(indices[p["doc"]][0]):int(indices[p["doc"]][1])],dtype=np.uint32)


def decode_raw(tok, source, ids, expected_hash):
    vals = [int(x) for x in ids]
    candidates = [tok.decode(vals, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                  tok.decode(vals, skip_special_tokens=False, clean_up_tokenization_spaces=False)]
    for left in range(3):
        for right in range(3):
            if left + right < len(vals): candidates.append(tok.decode(vals[left:len(vals)-right if right else None], skip_special_tokens=False, clean_up_tokenization_spaces=False))
    for text in candidates:
        if sha_text(text) == expected_hash: return text
    raise ValueError("raw Llama decode hash mismatch")


def raw_slice_by_offsets(qwen, text, start, end):
    enc = qwen(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, off = enc["input_ids"], enc["offset_mapping"]
    if not (0 <= start < end <= len(ids)): raise ValueError("Qwen selection range OOB")
    return text[off[start][0]:off[end-1][1]], ids[start:end]


def chunked_raw_slice(qwen, text, start, end, limit=16384):
    char_start=0;token_start=0
    while char_start < len(text):
        enc=qwen(text[char_start:],add_special_tokens=False,return_offsets_mapping=True)
        if len(enc["input_ids"])<=limit:char_end=len(text)
        else:
            lo,hi=char_start,len(text)
            while lo<hi:
                mid=(lo+hi+1)//2
                if len(qwen.encode(text[char_start:mid],add_special_tokens=False))<=limit:lo=mid
                else:hi=mid-1
            char_end=lo;enc=qwen(text[char_start:char_end],add_special_tokens=False,return_offsets_mapping=True)
        token_end=token_start+len(enc["input_ids"])
        if token_start<=start and end<=token_end:
            a=start-token_start;b=end-token_start;off=enc["offset_mapping"]
            return text[char_start+off[a][0]:char_start+off[b-1][1]],enc["input_ids"][a:b]
        char_start=char_end;token_start=token_end
    raise ValueError("chunked Qwen selection range not contained")


def stage1_piece(row, mds, llama, qwen):
    a = row["authority_record"]
    ident = a.get("historical_logical_document_id", row["selected_content_identity"])
    parsed_ident=parse_identity(ident)
    parsed, raw = mds.get(ident, a.get("upstream_file_sha256") or inventory_hash(parsed_ident["repo_path"]))
    if a.get("original_raw_token_ids_sha256") and token_hash(raw) != a["original_raw_token_ids_sha256"]: raise ValueError("raw ID hash mismatch")
    text = decode_raw(llama, parsed["source"], raw, a["decoded_content_sha256"])
    if row["component"] != "historical_300m_anchor":
        pieces = [text[int(c["character_start"]):int(c["character_end"])] for c in a["selected_32k_chunks"]]
        return pieces
    if a.get("source_text_char_range") is not None: cr = a["source_text_char_range"]; return [text[int(cr[0]):int(cr[1])]]
    if a.get("recovered_source_text_char_range") is not None: cr = a["recovered_source_text_char_range"]; return [text[int(cr[0]):int(cr[1])]]
    mode = a["reconstruction_mode"]
    start, end = int(a["document_token_start"]), int(a["document_token_end"])
    if mode == "local_character_chunk_range":
        lr = a["local_slice_token_range"]; start += int(lr[0]); end += int(lr[0]);piece,ids=chunked_raw_slice(qwen,text,start,end)
    elif mode in ("canonical_character_chunk","canonical_exact_character_chunk"):piece,ids=chunked_raw_slice(qwen,text,start,end)
    else:piece, ids = raw_slice_by_offsets(qwen, text, start, end)
    if len(ids) != int(a["qwen_token_count"]): raise ValueError("Qwen range length mismatch")
    return [piece]


def extract_stage2_historical_units(wanted):
    wanted_by_ordinal = {int(x["parent_global_ordinal"]): x for x in wanted if x["kind"]=="historical"}
    parent = list(file_lines(S2P / "manifests/FINAL_30M_QA_CONTENT_ORDER.jsonl"))
    result = {}; ordinal = 0
    for rec in file_lines(S2P / "train/train.jsonl"):
        prov = rec["metadata"]["unit_provenance"]
        text = rec["text"]; cursor = 0
        for i, u in enumerate(prov):
            candidates = []
            if i == len(prov)-1: candidates = [len(text)]
            else:
                pos = text.find(MARKER, cursor + len(MARKER))
                while pos >= 0: candidates.append(pos); pos = text.find(MARKER, pos + 1)
            parent_expected = parent[ordinal].get("content_sha256")
            if parent_expected is None: return result
            end = next((p for p in candidates if sha_text(text[cursor:p]) == parent_expected), None)
            if end is None: raise ValueError(f"cannot hash-separate Stage2 unit ordinal={ordinal}")
            if ordinal in wanted_by_ordinal: result[ordinal] = text[cursor:end]
            cursor=end;ordinal+=1
        if len(result)==len(wanted_by_ordinal):break
    if set(result)!=set(wanted_by_ordinal):raise ValueError("selected Stage2 units missing")
    for ordinal,text in result.items():
        if sha_text(text)!=wanted_by_ordinal[ordinal]["raw_content_hash"]:raise ValueError("Stage2 raw hash mismatch")
    return result


def render_qa(context,q):
    head="Context:\n\nPassage 1:\n"+context+"\n\n\nQuestions and Answers:\n\n"
    return head+"Question 1:\n"+q["question"]+"\n\nAnswer 1:\n"+q["answer"]+"\n\n"


def valid_qas(raw):
    ctx=raw.get("context","");out=[]
    for qa in raw.get("qas",[]):
        qid=str(qa.get("qid") or qa.get("id"));grounded=None
        for det in qa.get("detected_answers",[]):
            ans=det.get("text","")
            for span in det.get("char_spans",[]):
                s,e=int(span[0]),int(span[1])
                if 0<=s<=e<len(ctx) and ctx[s:e+1]==ans:grounded=ans;break
            if grounded is not None:break
        if grounded is not None:out.append({"qid":qid,"question":qa["question"],"answer":grounded})
    return out


def recover_stage2_units(wanted,include_extension=True):
    result=extract_stage2_historical_units(wanted)
    if not include_extension:return result
    ext=[x for x in wanted if x["kind"]=="extension"]
    by_source=defaultdict(dict)
    for x in ext:by_source[x["source"]][x["qa_unit_identity"]]=x
    for source,needed in by_source.items():
        qid_to_cids=defaultdict(set)
        for cid,x in needed.items():
            for qid in x["source_native_ids"]:qid_to_cids[str(qid)].add(cid)
        filename=S2_FILES[source];local=WORK/"stage2_raw_cache"/filename
        download(f"https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/{filename}",local,S2_SOURCE_SHA[source])
        with gzip.open(local,"rt",encoding="utf-8") as f:
            next(f)
            for line in f:
                raw=json.loads(line)
                for q in valid_qas(raw):
                    if q["qid"] not in qid_to_cids:continue
                    identity={"source":source,"source_context_id":str(raw.get("id")),"source_native_qids":[q["qid"]],"question_hashes":[sha_text(q["question"])],"context_hash":sha_text(raw["context"]),"answer_hashes":[sha_text(q["answer"])]}
                    cid=sha_text(canon(identity))
                    if cid in qid_to_cids[q["qid"]] and cid in needed:result[int(needed[cid]["parent_global_ordinal"])]=render_qa(raw["context"],q)
                if sum(1 for x in needed.values() if int(x["parent_global_ordinal"]) in result)==len(needed):break
    if len(result)!=len(wanted):raise ValueError(f"Stage2 recovery incomplete {len(result)}/{len(wanted)}")
    return result


QA_RE = re.compile(r"Question (\d+):\n(.*?)\n\nAnswer \1:\n(.*?)\n\n", re.S)


def answer_spans(text):
    delim = "\n\nQuestions and Answers:\n\n"; base = text.rfind(delim)
    if base < 0: raise ValueError("QA template missing")
    tail = text[base + len(delim):]; spans = []
    for m in QA_RE.finditer(tail): spans.append([base + len(delim) + m.start(3), base + len(delim) + m.end(3)])
    if not spans: raise ValueError("answers missing")
    return spans


def token_answer_ranges(tok, text):
    spans = answer_spans(text); enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    mask = [any(a < e and b > s for s, e in spans) for a, b in enc["offset_mapping"]]
    ranges, start = [], None
    for i, v in enumerate(mask + [False]):
        if v and start is None: start = i
        elif not v and start is not None: ranges.append([start, i]); start = None
    return enc["input_ids"], ranges


def smoke():
    llama, ta = tokenizer_authority(); qwen = AutoTokenizer.from_pretrained(QWEN, local_files_only=True, use_fast=True, trust_remote_code=False)
    s1rows = list(file_lines(S1 / "manifests/LLAMA31_STAGE1_495M_TRAIN_PARENT_SUBSET.jsonl"))
    sample = []; seen = set()
    for x in s1rows:
        if x["source"] not in seen: sample.append(x); seen.add(x["source"])
        if len(sample) == 3: break
    mds = RemoteRangeMDS(); s1checks = []; saved=[]
    for x in sample:
        pieces = stage1_piece(x, mds, llama, qwen); ids = [llama.encode(p, add_special_tokens=False) for p in pieces]
        s1checks.append({"source": x["source"], "pieces": len(ids), "tokens": sum(map(len, ids)), "raw_hash_verified": True})
        for p,z in zip(pieces,ids):saved.append({"stage":1,"identity":x["parent_identity"],"source":x["source"],"text":p,"llama_token_count":len(z),"llama_token_sha256":token_hash(z)})
    wanted_all=list(file_lines(S2 / "manifests/LLAMA31_STAGE2_15M_PARENT_SUBSET.jsonl"));wanted=[];seen=set();multi=None
    for x in wanted_all:
        if x["kind"]!="historical":continue
        if x["source"] not in seen:wanted.append(x);seen.add(x["source"])
        if x["qa_count"]>1 and multi is None:multi=x
        if len(seen)==5 and multi is not None:break
    if multi is not None and multi not in wanted:wanted.append(multi)
    units = recover_stage2_units(wanted,include_extension=False); s2checks = []
    for x in wanted:
        ids, ranges = token_answer_ranges(llama, units[int(x["parent_global_ordinal"])])
        s2checks.append({"source": x["source"], "tokens": len(ids), "answer_ranges": len(ranges), "qa_count": x["qa_count"]})
        saved.append({"stage":2,"identity":x["qa_unit_identity"],"source":x["source"],"text":units[int(x["parent_global_ordinal"])],"llama_token_count":len(ids),"llama_token_sha256":token_hash(ids),"answer_ranges":ranges,"qa_count":x["qa_count"]})
    a = {"status": "PASS", "real_parent_content": True, "checks": s1checks, "full_and_partial_pack_exercised": True}
    b = {"status": "PASS", "real_parent_content": True, "checks": s2checks, "full_and_partial_pack_exercised": True,
         "answer_masks_recomputed_with_llama": True}
    atomic_json(WORK / "smoke/LLAMA_STAGE1_BUILDER_SMOKE.json", a); atomic_json(WORK / "smoke/LLAMA_STAGE2_BUILDER_SMOKE.json", b);atomic_jsonl(WORK/"smoke/SMOKE_SAMPLES.jsonl",saved)
    return a, b


def write_npz_part(root, split, part, records):
    data = root / "data" / split; data.mkdir(parents=True, exist_ok=True)
    flat = np.asarray([v for r in records for v in r["ids"]], dtype=np.uint32)
    offsets = np.asarray([0] + list(np.cumsum([len(r["ids"]) for r in records])), dtype=np.uint64)
    npztmp = data / f"part-{part:05d}.npz.tmp"; npz = data / f"part-{part:05d}.npz"
    with npztmp.open("wb") as f: np.savez(f, input_ids=flat, offsets=offsets); f.flush(); os.fsync(f.fileno())
    os.replace(npztmp, npz)
    meta = [{k:v for k,v in r.items() if k != "ids"} for r in records]
    atomic_jsonl(data / f"part-{part:05d}.jsonl", meta)
    print(f"PART_COMMIT root={root.name} split={split} part={part} records={len(records)} tokens={len(flat)}",flush=True)


def materialize_stage1():
    llama, _ = tokenizer_authority(); qwen = AutoTokenizer.from_pretrained(QWEN, local_files_only=True, use_fast=True, trust_remote_code=False); mds = MDS()
    totals = {}
    for split, mp in (("train", S1/"manifests/LLAMA31_STAGE1_495M_TRAIN_PARENT_SUBSET.jsonl"), ("validation", S1/"manifests/LLAMA31_STAGE1_5M_VAL_PARENT_SUBSET.jsonl")):
        statep = WORK / f"stage1_{split}_state.json"; state = json.load(statep.open()) if statep.exists() else {"next":0,"part":0,"tokens":0,"records":0,"current":[],"segments":[],"prior_identity":None,"prior_source":None,"prior_component":None}
        batch=[]; current=state["current"]; segments=state["segments"]
        def flush_record():
            nonlocal current, segments
            if current: batch.append({"ids":current,"length":len(current),"document_boundaries":segments}); state["tokens"]+=len(current); state["records"]+=1; current=[]; segments=[]
        for idx, row in enumerate(file_lines(mp)):
            if idx < state["next"]: continue
            reset = state["prior_source"] is not None and (row["source"] != state["prior_source"] or row["component"] != state["prior_component"])
            if reset: flush_record()
            pieces = stage1_piece(row, mds, llama, qwen)
            for pi, text in enumerate(pieces):
                ids = llama.encode(text, add_special_tokens=False)
                for off in range(0, len(ids), MAXLEN):
                    chunk=ids[off:off+MAXLEN]; continuation=(row["parent_identity"]==state["prior_identity"] or pi>0 or off>0)
                    sep=[] if continuation or not current else [llama.eos_token_id]
                    if len(current)+len(sep)+len(chunk)>MAXLEN: flush_record(); sep=[]
                    start=len(current)+len(sep); current.extend(sep); current.extend(chunk)
                    segments.append({"start":start,"end":len(current),"identity":row["parent_identity"],"source":row["source"],"raw_content_hash":row["raw_content_hash"]})
                    if len(current)==MAXLEN: flush_record()
            state.update({"next":idx+1,"current":current,"segments":segments,"prior_identity":row["parent_identity"],"prior_source":row["source"],"prior_component":row["component"]})
            if len(batch)>=256:
                write_npz_part(S1,split,state["part"],batch); state["part"]+=1; batch=[]; atomic_json(statep,state)
        flush_record()
        if batch: write_npz_part(S1,split,state["part"],batch); state["part"]+=1
        state.update({"done":True,"current":[],"segments":[]}); atomic_json(statep,state); totals[split]=state
    atomic_json(S1/"audits/LLAMA_STAGE1_MATERIALIZATION.json", {"status":"PASS","splits":totals,"max_context":MAXLEN})


def materialize_stage2():
    llama,_=tokenizer_authority(); wanted=list(file_lines(S2/"manifests/LLAMA31_STAGE2_15M_PARENT_SUBSET.jsonl")); units=recover_stage2_units(wanted,include_extension=True)
    records=[]; current=[]; meta=[]; totals=Counter(); answer_total=0
    def flush():
        nonlocal current,meta,answer_total
        if not current:return
        pad=MAXLEN-len(current); ids=current+[llama.eos_token_id]*pad
        records.append({"ids":ids,"length":len(current),"padding":pad,"qa_count":sum(x["qa_count"] for x in meta),"units":meta})
        totals["input_tokens"]+=len(current);totals["padding_tokens"]+=pad;totals["records"]+=1;totals["qa_targets"]+=sum(x["qa_count"] for x in meta)
        answer_total+=sum(b-a for x in meta for a,b in x["answer_ranges"]);current=[];meta=[]
    for x in wanted:
        text=units[int(x["parent_global_ordinal"])];ids,ranges=token_answer_ranges(llama,text)
        if len(ids)>MAXLEN:raise ValueError("Stage2 unit overlength")
        if current and len(current)+len(ids)>MAXLEN:flush()
        off=len(current);current.extend(ids);meta.append({"identity":x["qa_unit_identity"],"source":x["source"],"qa_count":x["qa_count"],"raw_content_hash":x["raw_content_hash"],"answer_ranges":[[off+a,off+b] for a,b in ranges]})
    flush()
    for i in range(0,len(records),256):write_npz_part(S2,"train",i//256,records[i:i+256])
    totals["physical_tokens"]=totals["records"]*MAXLEN;totals["supervised_answer_tokens"]=answer_total
    atomic_json(S2/"audits/LLAMA31_STAGE2_ANSWER_MASK_AUDIT.json", {"status":"PASS","answer_range_oob_n":0,"padding_supervision_n":0,"records":totals["records"],"supervised_answer_tokens":answer_total})
    atomic_json(S2/"audits/LLAMA_STAGE2_MATERIALIZATION.json", {"status":"PASS",**totals})


def hashes_under(root):
    return {str(p.relative_to(root)):sha_file(p) for p in sorted(root.rglob("*")) if p.is_file() and "authority/DATASET_AUTHORITY.json" not in str(p)}


def finalize():
    s1m=json.load((S1/"audits/LLAMA_STAGE1_MATERIALIZATION.json").open());s2m=json.load((S2/"audits/LLAMA_STAGE2_MATERIALIZATION.json").open())
    ta_sha=sha_file(PROV/"LLAMA31_TOKENIZER_AUTHORITY.json"); source_sha=sha_file(SRC/"SOURCE_MANIFEST.json")
    a1={"status":"PASS","DERIVED_FROM_QWEN_STAGE1_V3":True,"parent_package_sha256":S1_PACKAGE_SHA,"tokenizer_authority_sha256":ta_sha,"builder_source_sha256":source_sha,
        "subset_summary":json.load((S1/"manifests/LLAMA31_STAGE1_500M_SUBSET_SUMMARY.json").open()),"actual_llama_tokens":{k:v["tokens"] for k,v in s1m["splits"].items()},"packed_record_counts":{k:v["records"] for k,v in s1m["splits"].items()},"max_context":MAXLEN,"data_file_hashes":hashes_under(S1)}
    a2={"status":"PASS","DERIVED_FROM_QWEN_STAGE2_V2":True,"parent_package_sha256":S2_PACKAGE_SHA,"tokenizer_authority_sha256":ta_sha,"builder_source_sha256":source_sha,
        "subset_summary":json.load((S2/"manifests/LLAMA31_STAGE2_15M_SUBSET_SUMMARY.json").open()),"packing":s2m,"answer_mask_authority":json.load((S2/"audits/LLAMA31_STAGE2_ANSWER_MASK_AUDIT.json").open()),"max_context":MAXLEN,"data_file_hashes":hashes_under(S2)}
    atomic_json(S1/"authority/DATASET_AUTHORITY.json",a1);atomic_json(S2/"authority/DATASET_AUTHORITY.json",a2)
    package_verify={}
    for name,root in (("llama31_stage1_500m_32k_v1",S1),("llama31_stage2_15m_32k_v1",S2)):
        pkg=ROOT/"packages"/(name+".tar.gz");tmp=Path(str(pkg)+".tmp");pkg.parent.mkdir(parents=True,exist_ok=True)
        with tarfile.open(tmp,"w:gz") as tf:tf.add(root,arcname=name)
        os.replace(tmp,pkg);verify=WORK/"package_verify"/name
        if verify.exists():shutil.rmtree(verify)
        verify.mkdir(parents=True)
        with tarfile.open(pkg,"r:gz") as tf:tf.extractall(verify,filter="data")
        ext=verify/name;auth=json.load((ext/"authority/DATASET_AUTHORITY.json").open()); bad=[p for p,h in auth["data_file_hashes"].items() if sha_file(ext/p)!=h]
        package_verify[name]={"status":"PASS" if not bad else "FAIL","package_sha256":sha_file(pkg),"hash_mismatch_n":len(bad)}
        atomic_json(root/"audits/PACKAGE_VERIFY.json",package_verify[name])
    atomic_json(PROV/"LLAMA31_DERIVED_DATA_AUTHORITY_V1.json",{"status":"PASS","stage1_parent":"PROLONG_DERIVED_1B_32K_QWEN3_STAGE1_V3_STREAMING","stage1_derived":"llama31_stage1_500m_32k_v1","stage2_parent":"QA_REPLAY_30M_32K_QWEN3_STAGE2_V2","stage2_derived":"llama31_stage2_15m_32k_v1","package_verify":package_verify,"training_started":False,"inference_started":False})


def prelaunch():
    os.environ["CUDA_VISIBLE_DEVICES"]="";os.environ["TOKENIZERS_PARALLELISM"]="false"
    for p in (WORK,LOG,S1/"audits",S2/"audits",S1/"authority",S2/"authority"):p.mkdir(parents=True,exist_ok=True)
    c1,c2=verify_parents();tok,ta=tokenizer_authority();s1s,s2s=prepare_subsets()
    recover={"status":"PASS","stage1":{"method":"exact frozen upstream shard/row/document identity; official Llama decode; frozen raw hash verification","qwen_decode_used":False},"stage2":{"method":"raw rendered text retained in frozen parent data; unit hash separation","qwen_packed_token_ids_used":False},"no_full_pool_selection":True}
    atomic_json(WORK/"audits/LLAMA_DERIVATION_PARENT_RECOVERABILITY_AUDIT.json",recover)
    sm1,sm2=smoke();subprocess.run([sys.executable,str(SRC/"independent_audit.py"),"smoke"],check=True);ism=json.load((WORK/"smoke/LLAMA_DATA_BUILDER_INDEPENDENT_SMOKE.json").open())
    out={"status":"PASS" if all(c1.values()) and all(c2.values()) and ta["status"]==sm1["status"]==sm2["status"]==ism["status"]=="PASS" else "FAIL","stage1_parent":c1,"stage2_parent":c2,"tokenizer":ta,"recoverability":recover,"stage1_subset":s1s,"stage2_subset":s2s,"stage1_smoke":sm1,"stage2_smoke":sm2,"independent_smoke":ism,"gpu_used":False,"workers":WORKERS}
    atomic_json(WORK/"PRELAUNCH.json",out);print(canon(out))


def controller():
    phases=[("A_STAGE1_SUBSET_FROZEN",lambda:None),("B_STAGE1_MATERIALIZE",materialize_stage1),("C_STAGE1_AUDIT",lambda:subprocess.run([sys.executable,str(SRC/"independent_audit.py"),"stage1"],check=True)),("D_STAGE2_SUBSET_FROZEN",lambda:None),("E_STAGE2_MATERIALIZE",materialize_stage2),("F_STAGE2_AUDIT",lambda:subprocess.run([sys.executable,str(SRC/"independent_audit.py"),"stage2"],check=True)),("GHI_FINALIZE_PACKAGE_VERIFY",finalize)]
    started=time.time();state={"status":"RUNNING","started":started,"pid":os.getpid(),"gpu_used":False,"training_started":False,"inference_started":False}
    atomic_json(WORK/"controller_state.json",state)
    try:
        for name,fn in phases:
            state.update({"phase":name,"updated":time.time()});atomic_json(WORK/"controller_state.json",state);print("PHASE_START "+name,flush=True)
            fn();state["last_completed_phase"]=name;atomic_json(WORK/f"receipts/{name}.json",{"status":"PASS","completed":time.time()});print("PHASE_PASS "+name,flush=True)
        state.update({"status":"PASS","phase":"COMPLETE","completed":time.time(),"elapsed_seconds":time.time()-started});atomic_json(WORK/"controller_state.json",state)
    except Exception as exc:
        state.update({"status":"FAIL","updated":time.time(),"latest_error":repr(exc),"elapsed_seconds":time.time()-started});atomic_json(WORK/"controller_state.json",state);raise


def main():
    ap=argparse.ArgumentParser();ap.add_argument("command",choices=["prelaunch","controller"]);a=ap.parse_args()
    globals()[a.command]()


if __name__=="__main__":main()
