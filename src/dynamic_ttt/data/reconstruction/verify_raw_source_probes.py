"""Independent deterministic raw-source probes for frozen V3 selections."""

import hashlib
import json
import os
import sys
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path


V1 = Path(os.path.expandvars("${DATA_ROOT}/prolong_1b_32k_qwen3_v1"))
ROOT = Path(os.path.expandvars("${DATA_ROOT}/prolong_1b_32k_qwen3_v3_streaming"))
import numpy as np
from transformers import AutoTokenizer

REVISION = "1447128c751f0fbc7d2082b2a113a938426a9bba"
MAX_CONTEXT = 32768
QWEN = Path(os.path.expandvars("${MODEL_ROOT}/Qwen3-4B-Base"))
LLAMA = V1 / "source_cache" / "hf" / "models--NousResearch--Meta-Llama-3.1-8B" / "snapshots" / "1f47e50cdbe801ad8a5174156ec3a0655108fb9f"
INDEX_ROOT = V1 / "source_cache" / "hf" / "datasets--princeton-nlp--prolong-data-64K" / "snapshots" / REVISION
INVENTORY = ROOT / "manifests" / "UPSTREAM_SHARD_INVENTORY.jsonl"
SELECTION = ROOT / "manifests" / "V3_EXTENSION_700M_SELECTION.jsonl"
SELECTION_AUDIT = ROOT / "audits" / "V3_EXTENSION_SELECTION_AUDIT.json"
REPLAY_AUDIT = ROOT / "audits" / "STREAM_RECEIPT_REPLAY_AUDIT.json"
CACHE = ROOT / "tmp" / "stream_selection" / "source_shards"
OUTPUT = ROOT / "audits" / "RAW_SOURCE_PROBE_AUDIT.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_id(path):
    digest = hashlib.sha1()
    digest.update(("blob %d\0" % path.stat().st_size).encode("ascii"))
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def ndarray(data, expected_dtype):
    descriptor = data[0]
    ndim, dtype_index = descriptor >> 2, descriptor % 4
    shape_dtype = [np.uint8, np.uint16, np.uint32, np.uint64][dtype_index]
    shape_bytes = ndim * (2 ** dtype_index)
    shape = np.frombuffer(data[1:1 + shape_bytes], shape_dtype)
    return np.frombuffer(data[1 + shape_bytes:], expected_dtype).reshape(shape)


def read_row(stream, row_index, schema):
    stream.seek((1 + row_index) * 4)
    begin, end = np.frombuffer(stream.read(8), np.uint32)
    stream.seek(int(begin))
    data = stream.read(int(end - begin))
    sizes, cursor = [], 0
    for fixed in schema["column_sizes"]:
        if fixed is None:
            size = int(np.frombuffer(data[cursor:cursor + 4], np.uint32)[0])
            cursor += 4
        else:
            size = int(fixed)
        sizes.append(size)
    values = []
    for size in sizes:
        values.append(data[cursor:cursor + size])
        cursor += size
    encoded = dict(zip(schema["column_names"], values))
    return ndarray(encoded["indices"], np.uint32), ndarray(encoded["input_ids"], np.uint32).reshape(-1)


def decode_ids(llama, source, ids):
    values = [int(value) for value in ids]
    if source == "tuluv2":
        if values and values[0] == 128000:
            values = values[1:]
        if len(values) >= 2 and values[-2:] == [128009, 128001]:
            values = values[:-2]
        elif values and values[-1] == 128001:
            values = values[:-1]
        return llama.decode(values, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return llama.decode(values, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def token_ids_sha256(ids):
    return hashlib.sha256(np.asarray(ids, dtype="<u4").tobytes()).hexdigest()


def maximal_character_prefix(tokenizer, text, start, limit=MAX_CONTEXT, preencoded=None):
    encoded = preencoded or tokenizer(text[start:], add_special_tokens=False, return_offsets_mapping=True)
    if len(encoded["input_ids"]) <= limit:
        return len(text), encoded["input_ids"]
    ends = [int(pair[1]) for pair in encoded["offset_mapping"]]
    low, high = start, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        approximate = bisect_right(ends, middle - start)
        size = approximate if abs(approximate - limit) > 128 else len(tokenizer.encode(text[start:middle], add_special_tokens=False))
        if size <= limit:
            low = middle
        else:
            high = middle - 1
    ids = tokenizer.encode(text[start:low], add_special_tokens=False)
    if len(ids) > limit or (low < len(text) and len(tokenizer.encode(text[start:low + 1], add_special_tokens=False)) <= limit):
        raise ValueError("maximal prefix invariant failed")
    return low, ids


def qwen_plan(tokenizer, text):
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if not encoded["input_ids"]:
        return []
    if len(encoded["input_ids"]) <= MAX_CONTEXT:
        return [{"start": 0, "end": len(text), "ids": encoded["input_ids"]}]
    chunks, start, first = [], 0, encoded
    while start < len(text):
        end, ids = maximal_character_prefix(tokenizer, text, start, preencoded=first)
        first = None
        chunks.append({"start": start, "end": end, "ids": ids})
        start = end
    return chunks


def load_inventory():
    rows = {}
    with INVENTORY.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            rows[row["canonical_shard_id"]] = row
    return rows


def schema_for(shard_id):
    source, proc, basename = shard_id.split("/", 2)
    index = json.loads((INDEX_ROOT / source / proc / "index.json").read_text(encoding="utf-8"))
    matches = [row for row in index["shards"] if row["raw_data"]["basename"] == basename]
    if len(matches) != 1:
        raise ValueError("schema lookup not unique: " + shard_id)
    return matches[0]


def selected_probes(rows):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(row["source"], row["split"])].append(index)
    chosen = set()
    for key, indices in groups.items():
        values = [rows[index] for index in indices]
        mandatory = {0, len(values) // 2, len(values) - 1}
        ranked = sorted(
            range(len(values)),
            key=lambda i: hashlib.sha256(("PROLONG_V3_RAW_PROBE_V1" + values[i]["canonical_raw_identity"]).encode("utf-8")).hexdigest(),
        )
        chosen.update(indices[i] for i in mandatory | set(ranked[:10]))
    by_source = defaultdict(list)
    for index, row in enumerate(rows):
        by_source[row["source"]].append(index)
    for source, indices in by_source.items():
        selected_n = sum(index in chosen for index in indices)
        if selected_n >= 20:
            continue
        supplements = sorted(
            (index for index in indices if index not in chosen),
            key=lambda index: hashlib.sha256(
                ("PROLONG_V3_RAW_PROBE_SOURCE_SUPPLEMENT_V1" + rows[index]["canonical_raw_identity"]).encode("utf-8")
            ).hexdigest(),
        )
        chosen.update(supplements[:20 - selected_n])
    return [rows[index] for index in sorted(chosen)]


def main():
    primary = json.loads(SELECTION_AUDIT.read_text(encoding="utf-8"))
    replay = json.loads(REPLAY_AUDIT.read_text(encoding="utf-8"))
    if primary.get("status") != "PASS" or replay.get("status") != "PASS":
        raise SystemExit("selection/replay gates are not PASS")
    if primary.get("extension_700m_selection_sha256") != sha256_file(SELECTION):
        raise ValueError("selection hash mismatch")
    inventory = load_inventory()
    rows = [json.loads(line) for line in SELECTION.read_text(encoding="utf-8").splitlines()]
    probes = selected_probes(rows)
    llama = AutoTokenizer.from_pretrained(LLAMA, local_files_only=True, use_fast=True, trust_remote_code=False)
    qwen = AutoTokenizer.from_pretrained(QWEN, local_files_only=True, use_fast=True, trust_remote_code=False)
    results, mismatch = [], Counter()
    for ordinal, row in enumerate(probes):
        shard_id = row["canonical_shard_id"]
        authority = inventory[shard_id]
        path = Path(authority["local_cached_path"]) if authority.get("local_cached_path") else CACHE / shard_id
        if not path.is_file():
            raise ValueError("selected probe shard missing from retained cache: " + shard_id)
        size_ok = path.stat().st_size == int(authority["expected_byte_size"])
        authority_ok = sha256_file(path) == authority["expected_lfs_sha256"] if authority.get("expected_lfs_sha256") else git_blob_id(path) == authority["expected_git_blob_sha1"]
        mds_row, document_position = (int(value) for value in str(row["canonical_row_id"]).split(":"))
        with path.open("rb", buffering=0) as stream:
            indices, input_ids = read_row(stream, mds_row, schema_for(shard_id))
        begin, end = indices.tolist()[document_position]
        text = decode_ids(llama, row["source"], input_ids[int(begin):int(end)])
        content_ok = hashlib.sha256(text.encode("utf-8")).hexdigest() == row["decoded_content_sha256"]
        plan = qwen_plan(qwen, text)
        full_logical = sum(len(chunk["ids"]) for chunk in plan)
        full_ok = full_logical == int(row["qwen_full_token_n"])
        selected_ok = True
        selected_tokens = 0
        for selected in row["selected_32k_chunks"]:
            ids = qwen.encode(text[int(selected["character_start"]):int(selected["character_end"])], add_special_tokens=False)
            selected_tokens += len(ids)
            if len(ids) != int(selected["qwen_token_n"]) or token_ids_sha256(ids) != selected["qwen_token_ids_sha256"]:
                selected_ok = False
        contribution_ok = selected_tokens == int(row["selected_document_token_n"]) and selected_tokens + int(row["eos_contribution"]) == int(row["selected_logical_token_n"])
        checks = {"size": size_ok, "authority_hash": authority_ok, "identity": row["canonical_raw_identity"] == "%s:%s:%s:%d:%d" % (REVISION, row["source"], shard_id.split("/", 1)[1], mds_row, document_position), "content_hash": content_ok, "full_tokenization": full_ok, "selected_chunk_tokenization": selected_ok, "accepted_contribution": contribution_ok}
        failed = [key for key, value in checks.items() if not value]
        for key in failed:
            mismatch[key] += 1
        results.append({"probe_ordinal": ordinal, "source": row["source"], "split": row["split"], "canonical_raw_identity": row["canonical_raw_identity"], "checks": checks, "status": "PASS" if not failed else "FAIL"})
        print("[%d/%d] %s" % (ordinal + 1, len(probes), results[-1]["status"]), flush=True)
    per_source = Counter(row["source"] for row in results)
    per_source_split = Counter((row["source"], row["split"]) for row in results)
    status = "PASS" if not mismatch and all(per_source[source] >= 20 for source in per_source) and all(value >= 3 for value in per_source_split.values()) else "FAIL"
    audit = {"audit_id": "V3_RAW_SOURCE_PROBE_V1", "status": status, "RAW_SOURCE_PROBE_N": len(results), "RAW_SOURCE_PROBE_MISMATCH_N": sum(mismatch.values()), "mismatch_counts": dict(sorted(mismatch.items())), "selection_sha256": sha256_file(SELECTION), "probe_rule": "first, middle, last plus deterministic SHA-ranked entries; at least 10 per source/split and 20 per source where feasible", "per_source_probe_n": dict(sorted(per_source.items())), "per_source_split_probe_n": {"%s/%s" % key: value for key, value in sorted(per_source_split.items())}, "probes": results}
    atomic_json(OUTPUT, audit)
    print(json.dumps({key: value for key, value in audit.items() if key != "probes"}, ensure_ascii=False, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
