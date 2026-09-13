"""Independent read-only audit of every final V3 JSONL record.

This verifier intentionally imports no selector or materializer code.  It is
restartable at final data-file boundaries and freezes per-file hashes.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.path.expandvars("${DATA_ROOT}/prolong_1b_32k_qwen3_v3_streaming"))
V1 = Path(os.path.expandvars("${DATA_ROOT}/prolong_1b_32k_qwen3_v1"))
V2 = Path(os.path.expandvars("${DATA_ROOT}/prolong_1b_32k_qwen3_v2"))
from transformers import AutoTokenizer

QWEN = Path(os.path.expandvars("${MODEL_ROOT}/Qwen3-4B-Base"))
STATE = ROOT / "tmp" / "independent_final_audit_state.json"
AUDIT = ROOT / "audits" / "INDEPENDENT_DATASET_AUDIT.json"
HASHES = ROOT / "manifests" / "FINAL_DATA_FILE_HASHES.jsonl"
EXPECTED_ANCHOR = "017a74186b419b2c3f566f568e6c2d120723e3ddafb69024138eb6b321c90ae8"
EXPECTED_TOKENIZER = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
EXPECTED_LOGICAL = {"train": 990_000_000, "validation": 10_000_000}
REQUIRED_FIELDS = {"text","sample_id","split","category","domain","source_name","source_revision","source_shard","source_row_ids_json","logical_document_ids_json","document_boundaries_qwen_json","provenance_json","token_count"}


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); temp = path.with_name(path.name + ".partial")
    with temp.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n"); f.flush(); os.fsync(f.fileno())
    os.replace(str(temp), str(path))


def load_json(path): return json.loads(path.read_text(encoding="utf-8"))


def percentile(values, fraction):
    if not values: return None
    ordered = sorted(values); pos = (len(ordered)-1)*fraction; lo = int(pos); hi = min(lo+1,len(ordered)-1); w = pos-lo
    return ordered[lo]*(1-w)+ordered[hi]*w


def audit_file(path, split, tokenizer):
    malformed = 0; lengths = []; roundtrip_mismatch = 0; max_context_violation = 0
    with path.open("r", encoding="utf-8") as f:
        for line_n, line in enumerate(f, 1):
            try:
                if not line.endswith("\n"): raise ValueError("non-LF line")
                row = json.loads(line)
                if set(row) != REQUIRED_FIELDS or row["split"] != split: raise ValueError("schema/split")
                declared = int(row["token_count"])
                ids = tokenizer.encode(row["text"], add_special_tokens=False)
                if declared != len(ids): roundtrip_mismatch += 1
                if declared <= 0 or declared > 32768: max_context_violation += 1
                bounds = json.loads(row["document_boundaries_qwen_json"])
                json.loads(row["source_row_ids_json"]); json.loads(row["logical_document_ids_json"]); json.loads(row["provenance_json"])
                if not bounds or any(len(x)!=2 or int(x[0])<0 or int(x[1])<=int(x[0]) or int(x[1])>declared for x in bounds): raise ValueError("boundaries")
                lengths.append(declared)
            except Exception:
                malformed += 1
    return {"path":str(path.relative_to(ROOT)).replace("\\","/"),"bytes":path.stat().st_size,"sha256":file_hash(path),"record_n":len(lengths),"physical_tokens":sum(lengths),"malformed_n":malformed,"roundtrip_mismatch_n":roundtrip_mismatch,"max_context_violation_n":max_context_violation,"lengths":lengths}


def main():
    material = load_json(ROOT/"audits"/"MATERIALIZATION_32K_AUDIT.json")
    gates = {
        "materialization_pass": material.get("status") == "PASS",
        "historical_anchor_sha": file_hash(V2/"manifests"/"HISTORICAL_300M_RAW_ANCHOR.jsonl") == EXPECTED_ANCHOR,
        "tokenizer_sha": file_hash(QWEN/"tokenizer.json") == EXPECTED_TOKENIZER,
        "scope_pass": load_json(ROOT/"audits"/"UPSTREAM_REPOSITORY_FILE_SCOPE_AUDIT.json").get("status") == "PASS",
        "inventory_primary_pass": load_json(ROOT/"audits"/"UPSTREAM_SHARD_INVENTORY_AUDIT.json").get("status") == "PASS",
        "inventory_independent_pass": load_json(ROOT/"audits"/"UPSTREAM_SHARD_INVENTORY_INDEPENDENT_AUDIT_V2.json").get("status") == "PASS",
        "selection_pass": load_json(ROOT/"audits"/"V3_EXTENSION_SELECTION_AUDIT.json").get("status") == "PASS",
        "receipt_replay_pass": load_json(ROOT/"audits"/"STREAM_RECEIPT_REPLAY_AUDIT.json").get("status") == "PASS",
        "content_order_pass": load_json(ROOT/"audits"/"FINAL_CONTENT_ORDER_AUDIT.json").get("status") == "PASS",
    }
    if not all(gates.values()): raise SystemExit("independent final audit prerequisite failed")
    files = [(split,p) for split in ("train","validation") for p in sorted((ROOT/split).glob("part-*.jsonl"))]
    if any(any((ROOT/s).glob("*.partial")) for s in ("train","validation")): raise SystemExit("partial final output exists")
    state = load_json(STATE) if STATE.is_file() else {"version":1,"files":{}}
    tok = AutoTokenizer.from_pretrained(QWEN,local_files_only=True,use_fast=True,trust_remote_code=False)
    for split,path in files:
        key = str(path.relative_to(ROOT)).replace("\\","/")
        prior = state["files"].get(key)
        if prior and prior["bytes"] == path.stat().st_size and prior["sha256"] == file_hash(path): continue
        state["files"][key] = audit_file(path,split,tok); atomic_json(STATE,state)
    rows = [state["files"][str(p.relative_to(ROOT)).replace("\\","/")] for _,p in files]
    by_split = {}
    for split in ("train","validation"):
        selected = [r for r in rows if r["path"].startswith(split+"/")]; lengths = [n for r in selected for n in r["lengths"]]
        by_split[split] = {"file_n":len(selected),"record_n":len(lengths),"physical_tokens":sum(lengths),"bytes":sum(r["bytes"] for r in selected),
            "min":min(lengths),"p25":percentile(lengths,.25),"median":percentile(lengths,.5),"mean":sum(lengths)/len(lengths),"p75":percentile(lengths,.75),"p95":percentile(lengths,.95),"p99":percentile(lengths,.99),"max":max(lengths),
            "malformed_n":sum(r["malformed_n"] for r in selected),"roundtrip_mismatch_n":sum(r["roundtrip_mismatch_n"] for r in selected),"max_context_violation_n":sum(r["max_context_violation_n"] for r in selected)}
    expected_records = {s:int(material["splits"][s]["record_index"]) for s in by_split}
    expected_physical = {s:int(material["splits"][s]["physical_tokens"]) for s in by_split}
    selection = load_json(ROOT/"audits"/"V3_EXTENSION_SELECTION_AUDIT.json")
    final_order = load_json(ROOT/"audits"/"FINAL_CONTENT_ORDER_AUDIT.json")
    checks = {
        "all_files_audited":len(rows)==152 and len(state["files"])==152,
        "malformed_zero":all(by_split[s]["malformed_n"]==0 for s in by_split),
        "roundtrip_mismatch_zero":all(by_split[s]["roundtrip_mismatch_n"]==0 for s in by_split),
        "max_context_32768":all(by_split[s]["max_context_violation_n"]==0 and by_split[s]["max"]<=32768 for s in by_split),
        "record_count_parity":all(by_split[s]["record_n"]==expected_records[s] for s in by_split),
        "physical_token_parity":all(by_split[s]["physical_tokens"]==expected_physical[s] for s in by_split),
        "logical_train_990m":material["splits"]["train"]["logical_input_tokens"]==EXPECTED_LOGICAL["train"],
        "logical_validation_10m":material["splits"]["validation"]["logical_input_tokens"]==EXPECTED_LOGICAL["validation"],
        "logical_total_1b":sum(material["splits"][s]["logical_input_tokens"] for s in by_split)==1_000_000_000,
        "historical_extension_identity_overlap_zero":selection.get("historical_extension_identity_overlap_n")==0,
        "historical_extension_content_overlap_zero":selection.get("historical_extension_content_overlap_n")==0,
        "extension_train_val_identity_overlap_zero":selection.get("extension_train_val_identity_overlap_n")==0,
        "extension_train_val_content_overlap_zero":selection.get("extension_train_val_content_overlap_n")==0,
        "final_train_val_identity_overlap_zero":final_order.get("final_train_val_identity_overlap_n")==0,
        "source_quota_parity":final_order.get("checks",{}).get("source_split_quota_parity") is True,
    }
    temp = HASHES.with_name(HASHES.name+".partial")
    with temp.open("w",encoding="utf-8",newline="\n") as f:
        for r in rows:
            slim={k:v for k,v in r.items() if k!="lengths"}; f.write(json.dumps(slim,sort_keys=True,separators=(",",":"))+"\n")
        f.flush(); os.fsync(f.fileno())
    os.replace(str(temp),str(HASHES))
    audit={"audit_id":"PROLONG_V3_INDEPENDENT_DATASET_AUDIT_V1","status":"PASS" if all(checks.values()) else "FAIL","independent_of_primary_code":True,"gates":gates,"checks":checks,
        "logical_tokens":EXPECTED_LOGICAL,"splits":by_split,"final_data_file_hashes_sha256":file_hash(HASHES),"shard_order_sha256":{p.name:file_hash(p) for p in sorted((ROOT/"manifests").glob("SHARD_ORDER_*.jsonl"))},
        "stream_receipt_sha256":file_hash(ROOT/"manifests"/"STREAM_SCAN_RECEIPT.jsonl"),"extension_manifest_sha256":file_hash(ROOT/"manifests"/"V3_EXTENSION_700M_SELECTION.jsonl"),
        "final_train_content_order_sha256":file_hash(ROOT/"manifests"/"FINAL_TRAIN_CONTENT_ORDER.jsonl"),"final_val_content_order_sha256":file_hash(ROOT/"manifests"/"FINAL_VAL_CONTENT_ORDER.jsonl")}
    atomic_json(AUDIT,audit)
    if audit["status"]!="PASS": raise SystemExit("independent dataset audit FAIL")
    print(json.dumps(audit,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
