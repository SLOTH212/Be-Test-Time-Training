"""Crash-safe V3 32K materialization from frozen raw-content authorities.

The exact 1B logical budget is an input authority.  Physical 32K records are
repacked with the frozen first-fit/reset policy, so their retokenized token
count (including recomputed EOS) is reported separately.
"""

import hashlib
import json
import msvcrt
import os
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path

V1 = Path(os.path.expandvars("${DATA_ROOT}/prolong_1b_32k_qwen3_v1"))
V2 = Path(os.path.expandvars("${DATA_ROOT}/prolong_1b_32k_qwen3_v2"))
ROOT = Path(os.path.expandvars("${DATA_ROOT}/prolong_1b_32k_qwen3_v3_streaming"))

import numpy as np
from transformers import AutoTokenizer

REVISION = "1447128c751f0fbc7d2082b2a113a938426a9bba"
MAX_CONTEXT = 32768
EOS = 151643
RECORDS_PER_SHARD = 256
QWEN = Path(os.path.expandvars("${MODEL_ROOT}/Qwen3-4B-Base"))
LLAMA = V1 / "source_cache" / "hf" / "models--NousResearch--Meta-Llama-3.1-8B" / "snapshots" / "1f47e50cdbe801ad8a5174156ec3a0655108fb9f"
INDEX_ROOT = V1 / "source_cache" / "hf" / "datasets--princeton-nlp--prolong-data-64K" / "snapshots" / REVISION
FINAL = {
    "train": ROOT / "manifests" / "FINAL_TRAIN_CONTENT_ORDER.jsonl",
    "validation": ROOT / "manifests" / "FINAL_VAL_CONTENT_ORDER.jsonl",
}
EXPECTED_FINAL_SHA = {
    "train": "d6797344bf22d1a5274ad6d65f458cfe3429d96b8a846c9a9f5a53936187b7ef",
    "validation": "1143a1ce3d4d29ffc355e47ed2754c1290296d0554ea128ce9f7e4dd4dd9bc89",
}
EXPECTED_LOGICAL = {"train": 990_000_000, "validation": 10_000_000}
WORK = ROOT / "tmp" / "materialization_32k"
STATE = WORK / "state.json"
LOCK = WORK / "materializer.lock"
AUDIT = ROOT / "audits" / "MATERIALIZATION_32K_AUDIT.json"
HIST_DOWNLOAD = V2 / "audits" / "HISTORICAL_SHARD_DOWNLOAD.json"
INVENTORY = ROOT / "manifests" / "UPSTREAM_SHARD_INVENTORY.jsonl"
EXT_CACHE = ROOT / "tmp" / "stream_selection" / "source_shards"
GATE_FILES = [
    ROOT / "audits" / "UPSTREAM_REPOSITORY_FILE_SCOPE_AUDIT.json",
    ROOT / "audits" / "UPSTREAM_SHARD_INVENTORY_AUDIT.json",
    ROOT / "audits" / "UPSTREAM_SHARD_INVENTORY_INDEPENDENT_AUDIT_V2.json",
    ROOT / "audits" / "V3_EXTENSION_SELECTION_AUDIT.json",
    ROOT / "audits" / "STREAM_RECEIPT_REPLAY_AUDIT.json",
    ROOT / "audits" / "RAW_SOURCE_PROBE_AUDIT.json",
    ROOT / "audits" / "FINAL_CONTENT_ORDER_AUDIT.json",
]
CANONICAL_RE = re.compile(r"^(?P<revision>[^:]+):(?P<source>[^:]+):(?P<shard>[^:]+):(?P<row>\d+):(?P<doc>\d+)$")
LOCAL_RE = re.compile(r"^local:(?P<revision>[^:]+):(?P<source>[^:]+):(?P<shard>[^:]+):(?P<row>\d+):(?P<doc>\d+):(?P<slice_start>\d+):(?P<slice_end>\d+)$")


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def token_hash(ids):
    return hashlib.sha256(np.asarray(ids, dtype="<u4").tobytes()).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    with temp.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        f.flush(); os.fsync(f.fileno())
    os.replace(str(temp), str(path))


def acquire_lock():
    WORK.mkdir(parents=True, exist_ok=True)
    f = LOCK.open("a+b")
    if f.seek(0, os.SEEK_END) == 0:
        f.write(b"0"); f.flush()
    f.seek(0)
    try:
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        raise SystemExit("another V3 materializer holds the process lock")
    f.seek(0); f.write(str(os.getpid()).encode("ascii").ljust(32, b" ")); f.flush()
    return f


def parse_identity(identity):
    m = LOCAL_RE.match(identity); local = m is not None
    if m is None: m = CANONICAL_RE.match(identity)
    if m is None: raise ValueError("bad logical identity: " + identity)
    d = m.groupdict()
    for k in ("row", "doc", "slice_start", "slice_end"):
        if d.get(k) is not None: d[k] = int(d[k])
    d["local"] = local
    d["repo_path"] = d["source"] + "/" + d["shard"]
    return d


def ndarray_u32(data):
    descriptor = data[0]; ndim = descriptor >> 2; di = descriptor % 4
    shape_type = [np.uint8, np.uint16, np.uint32, np.uint64][di]
    shape_bytes = ndim * (2 ** di)
    shape = np.frombuffer(data[1:1 + shape_bytes], shape_type)
    return np.frombuffer(data[1 + shape_bytes:], np.uint32).reshape(shape)


def read_mds_row(path, row_index, schema):
    with path.open("rb", buffering=0) as f:
        f.seek((1 + row_index) * 4); offsets = f.read(8)
        if len(offsets) != 8: raise ValueError(f"missing offsets {path}:{row_index}")
        begin, end = np.frombuffer(offsets, np.uint32)
        f.seek(int(begin)); data = f.read(int(end - begin))
    sizes = []; cursor = 0
    for fixed in schema["column_sizes"]:
        if fixed is None:
            size = int(np.frombuffer(data[cursor:cursor + 4], np.uint32)[0]); cursor += 4
        else: size = int(fixed)
        sizes.append(size)
    values = []
    for size in sizes:
        values.append(data[cursor:cursor + size]); cursor += size
    encoded = dict(zip(schema["column_names"], values))
    return ndarray_u32(encoded["indices"]), ndarray_u32(encoded["input_ids"]).reshape(-1)


class RawCache:
    def __init__(self, paths, max_rows=24):
        self.paths = paths; self.max_rows = max_rows; self.rows = OrderedDict(); self.schemas = {}
    def schema(self, repo_path):
        if repo_path not in self.schemas:
            rel = Path(repo_path)
            idx = json.loads((INDEX_ROOT / rel.parent / "index.json").read_text(encoding="utf-8"))
            found = [x for x in idx["shards"] if x["raw_data"]["basename"] == rel.name]
            if len(found) != 1: raise ValueError("schema not unique: " + repo_path)
            self.schemas[repo_path] = found[0]
        return self.schemas[repo_path]
    def get(self, identity):
        p = parse_identity(identity); key = (p["repo_path"], p["row"])
        if key not in self.rows:
            indices, all_ids = read_mds_row(self.paths[p["repo_path"]], p["row"], self.schema(p["repo_path"]))
            self.rows[key] = [np.asarray(all_ids[int(a):int(b)], dtype=np.uint32) for a, b in indices.tolist()]
            if len(self.rows) > self.max_rows: self.rows.popitem(last=False)
        else: self.rows.move_to_end(key)
        return p, self.rows[key][p["doc"]]


def decode_document(llama, parsed, raw_ids, removed=None):
    removed = removed or {}; values = [int(x) for x in raw_ids]
    leading = [int(x) for x in removed.get("leading", [])]
    trailing = [int(x) for x in removed.get("trailing", [])]
    if leading or trailing:
        if leading and values[:len(leading)] != leading: raise ValueError("leading specials mismatch")
        if trailing and values[-len(trailing):] != trailing: raise ValueError("trailing specials mismatch")
        values = values[len(leading):len(values)-len(trailing) if trailing else len(values)]
        skip = False
    elif parsed["source"] == "tuluv2":
        if values and values[0] == 128000: values = values[1:]
        if len(values) >= 2 and values[-2:] == [128009, 128001]: values = values[:-2]
        elif values and values[-1] == 128001: values = values[:-1]
        skip = False
    else: skip = True
    return llama.decode(values, skip_special_tokens=skip, clean_up_tokenization_spaces=False)


class CharacterChunks:
    def __init__(self, tokenizer): self.tok = tokenizer; self.cache = OrderedDict()
    def maximal(self, text, start, limit=16384):
        enc = self.tok(text[start:], add_special_tokens=False, return_offsets_mapping=True)
        if len(enc["input_ids"]) <= limit: return len(text), enc["input_ids"]
        lo, hi = start, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if len(self.tok.encode(text[start:mid], add_special_tokens=False)) <= limit: lo = mid
            else: hi = mid - 1
        return lo, self.tok.encode(text[start:lo], add_special_tokens=False)
    def contained(self, key, text, start, end):
        if key not in self.cache: self.cache[key] = [0, 0, []]
        e = self.cache[key]
        while e[1] < end:
            ce, ids = self.maximal(text, e[0]); ts, te = e[1], e[1] + len(ids)
            e[2].append((ts, te, ids)); e[0], e[1] = ce, te
        matches = [(a, b, ids) for a, b, ids in e[2] if a <= start and end <= b]
        if len(matches) != 1: raise ValueError(f"range {start}:{end} not in character chunks")
        a, _, ids = matches[0]
        return ids[start-a:end-a]


class HistoricalRecords:
    def __init__(self): self.state = {}
    def get(self, path, index):
        key = str(path)
        if key not in self.state: self.state[key] = [path.open("r", encoding="utf-8"), -1, None]
        s = self.state[key]
        if index < s[1]: raise ValueError("historical record order regressed")
        while s[1] < index:
            line = s[0].readline()
            if not line: raise IndexError(f"historical index {index} missing")
            s[1] += 1; s[2] = json.loads(line)
        return s[2]


def historical_ids(row, raw_cache, hist_records, chunks, qwen, llama):
    a = row["authority_record"]
    old = hist_records.get(Path(a["provenance_file"]), int(a["historical_record_index"]))
    span = json.loads(old["provenance_json"])["spans"][int(a["span_index"])]
    parsed, raw = raw_cache.get(a["historical_logical_document_id"])
    if token_hash(raw) != a["original_raw_token_ids_sha256"]: raise ValueError("historical raw hash mismatch")
    text = decode_document(llama, parsed, raw, span.get("removed_source_special_ids"))
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != a["decoded_content_sha256"]: raise ValueError("historical decoded hash mismatch")
    mode = a["reconstruction_mode"]; start = int(a["document_token_start"]); end = int(a["document_token_end"])
    if mode == "explicit_source_text_char_range":
        cr = a["source_text_char_range"]; piece_text = text[int(cr[0]):int(cr[1])]
        ids = qwen.encode(piece_text, add_special_tokens=False)[:int(a["qwen_token_count"])]
    elif mode == "local_character_chunk_range":
        lr = a["local_slice_token_range"]; base = chunks.contained(a["decoded_content_sha256"], text, int(lr[0]), int(lr[1])); ids = base[start:end]
    elif mode in ("canonical_direct_qwen_slice", "canonical_direct_qwen_slice_window_trim"):
        ids = qwen.encode(text, add_special_tokens=False)[start:end]
    elif mode in ("canonical_character_chunk", "canonical_exact_character_chunk"):
        ids = chunks.contained(a["decoded_content_sha256"], text, start, end)
    elif mode == "single_span_exact_raw_substring":
        cr = a.get("recovered_source_text_char_range")
        chosen = text[int(cr[0]):int(cr[1])] if cr else old["text"]
        ids = qwen.encode(chosen, add_special_tokens=False)[:int(a["qwen_token_count"])]
    else: raise ValueError("unsupported historical mode: " + mode)
    if len(ids) != int(a["qwen_token_count"]) or token_hash(ids) != a["selected_token_ids_sha256"]: raise ValueError("historical selected IDs mismatch")
    if mode != "explicit_source_text_char_range" or len(qwen.encode(piece_text, add_special_tokens=False)) != len(ids):
        piece_text = qwen.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return [piece_text]


def extension_ids(row, raw_cache, qwen, llama):
    a = row["authority_record"]; parsed, raw = raw_cache.get(a["canonical_raw_identity"])
    text = decode_document(llama, parsed, raw)
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != a["decoded_content_sha256"]: raise ValueError("extension decoded hash mismatch")
    result = []; logical_n = 0
    for chunk in a["selected_32k_chunks"]:
        piece_text = text[int(chunk["character_start"]):int(chunk["character_end"])]
        ids = qwen.encode(piece_text, add_special_tokens=False)
        if len(ids) != int(chunk["qwen_token_n"]) or token_hash(ids) != chunk["qwen_token_ids_sha256"]: raise ValueError("extension chunk mismatch")
        logical_n += len(ids); result.append(piece_text)
    if logical_n != int(a["selected_document_token_n"]): raise ValueError("extension selected length mismatch")
    return result


def continuity_key(row):
    a = row["authority_record"]
    if row["component"] != "historical_300m_anchor": return None
    if a.get("source_text_char_range") is not None: return ("char", tuple(a["source_text_char_range"]))
    if a.get("recovered_source_text_char_range") is not None: return ("char", tuple(a["recovered_source_text_char_range"]))
    if a.get("local_slice_token_range") is not None:
        lr = a["local_slice_token_range"]
        return ("localtok", (int(lr[0]) + int(a["document_token_start"]), int(lr[0]) + int(a["document_token_end"])))
    return ("tok", (int(a["document_token_start"]), int(a["document_token_end"])))


def initial_state():
    return {"version": 1, "authority_sha256": EXPECTED_FINAL_SHA, "completed": {}, "active_split": "train"}


def split_state():
    return {"next_content_index": 0, "shard_index": 0, "record_index": 0, "logical_input_tokens": 0,
            "physical_tokens": 0, "full_records": 0, "short_records": 0, "recomputed_eos": 0,
            "roundtrip_checked": 0, "current_ids": [], "current_text": "", "segments": [], "prior_identity": None,
            "prior_range": None, "prior_component": None, "prior_source": None, "done": False}


def write_shard(split, ss, records):
    if not records: return
    out = ROOT / split / f"part-{ss['shard_index']:05d}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True); temp = out.with_name(out.name + ".partial")
    with temp.open("w", encoding="utf-8", newline="\n") as f:
        for row in records: f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        f.flush(); os.fsync(f.fileno())
    os.replace(str(temp), str(out)); ss["shard_index"] += 1


def materialize_split(split, state, raw_cache, qwen, llama):
    ss = state.setdefault(split, split_state()); records = []
    current = ss["current_ids"]; current_text = ss.get("current_text", ""); segments = ss["segments"]
    def flush():
        nonlocal current, current_text, segments
        if not current: return
        replay = qwen.encode(current_text, add_special_tokens=False)
        if replay != current: raise ValueError(f"packed roundtrip mismatch split={split} record={ss['record_index']}")
        sources = sorted({x["source"] for x in segments}); shards = sorted({x["canonical_shard_id"] for x in segments})
        bounds = [[x["start"], x["end"]] for x in segments]
        provenance = {"packing":"recovered_first_fit_32768_v1", "segments":segments}
        rec = {"text":current_text, "sample_id":f"prolong-v3-32k:{split}:{ss['record_index']:06d}", "split":split,
               "category":"prolong", "domain":sources[0], "source_name":"princeton-nlp/prolong-data-64K",
               "source_revision":REVISION, "source_shard":shards[0] if len(shards)==1 else "MULTIPLE",
               "source_row_ids_json":json.dumps([x["canonical_raw_identity"] for x in segments], ensure_ascii=False, separators=(",", ":")),
               "logical_document_ids_json":json.dumps([x["canonical_raw_identity"] for x in segments], ensure_ascii=False, separators=(",", ":")),
               "document_boundaries_qwen_json":json.dumps(bounds, separators=(",", ":")),
               "provenance_json":json.dumps(provenance, ensure_ascii=False, separators=(",", ":")), "token_count":len(current)}
        records.append(rec); ss["record_index"] += 1; ss["physical_tokens"] += len(current); ss["roundtrip_checked"] += 1
        if len(current) == MAX_CONTEXT: ss["full_records"] += 1
        else: ss["short_records"] += 1
        current = []; current_text = ""; segments = []; ss["current_ids"] = current; ss["current_text"] = current_text; ss["segments"] = segments
    def append_piece(piece_text, row, continuation):
        nonlocal current, current_text, segments
        piece_ids = qwen.encode(piece_text, add_special_tokens=False)
        if len(piece_ids) > MAX_CONTEXT: raise ValueError("piece exceeds 32K after trainer retokenization")
        separator = qwen.eos_token if (not continuation and current) else ""
        candidate_text = current_text + separator + piece_text
        candidate_ids = qwen.encode(candidate_text, add_special_tokens=False)
        if not continuation and current:
            if len(candidate_ids) <= MAX_CONTEXT: ss["recomputed_eos"] += 1
            else:
                flush(); separator = ""; candidate_text = piece_text; candidate_ids = piece_ids
        elif len(candidate_ids) > MAX_CONTEXT:
            flush(); candidate_text = piece_text; candidate_ids = piece_ids
        prefix_text = current_text + separator
        start = len(qwen.encode(prefix_text, add_special_tokens=False))
        current_text = candidate_text; current = candidate_ids
        segments.append({"start":start,"end":len(current),"source":row["source"],"component":row["component"],
                         "canonical_raw_identity":row["canonical_raw_identity"],"canonical_shard_id":row["canonical_shard_id"]})
    path = FINAL[split]
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < ss["next_content_index"]: continue
            row = json.loads(line)
            reset = ss["prior_component"] is not None and (row["component"] != ss["prior_component"] or row["source"] != ss["prior_source"])
            if reset: flush()
            rk = continuity_key(row)
            continuation = (not reset and row["canonical_raw_identity"] == ss["prior_identity"] and rk is not None and ss["prior_range"] is not None and rk[0] == ss["prior_range"][0] and rk[1][0] == ss["prior_range"][1][1])
            pieces = historical_ids(row, raw_cache, HIST_RECORDS, HIST_CHUNKS, qwen, llama) if row["component"] == "historical_300m_anchor" else extension_ids(row, raw_cache, qwen, llama)
            for pi, piece_text in enumerate(pieces): append_piece(piece_text, row, continuation or pi > 0)
            ss["logical_input_tokens"] += int(row["selected_logical_token_n"])
            ss["prior_identity"] = row["canonical_raw_identity"]; ss["prior_range"] = rk
            ss["prior_component"] = row["component"]; ss["prior_source"] = row["source"]
            ss["next_content_index"] = idx + 1; ss["current_ids"] = current; ss["current_text"] = current_text; ss["segments"] = segments
            if len(records) >= RECORDS_PER_SHARD:
                write_shard(split, ss, records); records = []; atomic_json(STATE, state)
    flush(); write_shard(split, ss, records)
    ss["current_ids"] = []; ss["current_text"] = ""; ss["segments"] = []; ss["done"] = True; atomic_json(STATE, state)


def load_paths():
    paths = {}
    download = json.loads(HIST_DOWNLOAD.read_text(encoding="utf-8"))
    for item in download["files"]: paths[item["repo_path"]] = Path(item["local_path"])
    with INVENTORY.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line); repo = row["upstream_repo_path"]
            candidate = EXT_CACHE / Path(repo)
            if candidate.is_file(): paths[repo] = candidate
            elif row.get("local_cached_path") and Path(row["local_cached_path"]).is_file(): paths[repo] = Path(row["local_cached_path"])
    return paths


def main():
    global HIST_CHUNKS
    lock = acquire_lock()
    gates = {str(p): json.loads(p.read_text(encoding="utf-8")).get("status") == "PASS" for p in GATE_FILES}
    for split in FINAL: gates["sha:" + split] = sha256_file(FINAL[split]) == EXPECTED_FINAL_SHA[split]
    failed = [k for k, v in gates.items() if not v]
    if failed: raise SystemExit("materialization gates failed: " + ", ".join(failed))
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else initial_state()
    if state.get("authority_sha256") != EXPECTED_FINAL_SHA: raise SystemExit("state authority mismatch")
    qwen = AutoTokenizer.from_pretrained(QWEN, local_files_only=True, use_fast=True, trust_remote_code=False)
    llama = AutoTokenizer.from_pretrained(LLAMA, local_files_only=True, use_fast=True, trust_remote_code=False)
    if qwen.eos_token_id != EOS: raise ValueError("Qwen EOS mismatch")
    HIST_CHUNKS = CharacterChunks(qwen)
    raw_cache = RawCache(load_paths())
    for split in ("train", "validation"):
        if not state.get(split, {}).get("done"): materialize_split(split, state, raw_cache, qwen, llama)
    checks = {split + "_done": state[split]["done"] for split in FINAL}
    checks.update({split + "_logical": state[split]["logical_input_tokens"] == EXPECTED_LOGICAL[split] for split in FINAL})
    checks["roundtrip_all"] = all(state[s]["roundtrip_checked"] == state[s]["record_index"] for s in FINAL)
    status = "PASS" if all(checks.values()) else "FAIL"
    audit = {"audit_id":"PROLONG_V3_32K_MATERIALIZATION_V1","status":status,"gates":gates,"checks":checks,
             "packing_rule":"recovered first-fit; reset at component/source-run boundaries; no cross-source/component records",
             "logical_budget_tokens":sum(EXPECTED_LOGICAL.values()),
             "splits":{s:{k:v for k,v in state[s].items() if k not in ("current_ids","segments","prior_range")} for s in FINAL},
             "output_shards":{s:state[s]["shard_index"] for s in FINAL}}
    atomic_json(AUDIT, audit)
    if status != "PASS": raise SystemExit("materialization audit FAIL")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    lock.close()


HIST_RECORDS = HistoricalRecords()
HIST_CHUNKS = None

if __name__ == "__main__":
    main()
