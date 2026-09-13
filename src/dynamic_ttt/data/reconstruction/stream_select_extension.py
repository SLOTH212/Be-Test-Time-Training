"""Crash-safe deterministic shard-permutation streaming selector for V3.

Only shards reached before a source/split quota closes are opened.  The
append-only in-progress receipt is committed at MDS-row boundaries; resume
truncates any uncommitted tail to the atomically recorded byte offset.
"""

import hashlib
import json
import msvcrt
import os
import shutil
import sys
import time
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote


V1 = Path(os.path.expandvars("${DATA_ROOT}/prolong_1b_32k_qwen3_v1"))
V2 = Path(os.path.expandvars("${DATA_ROOT}/prolong_1b_32k_qwen3_v2"))
ROOT = Path(os.path.expandvars("${DATA_ROOT}/prolong_1b_32k_qwen3_v3_streaming"))

import numpy as np
import requests
from transformers import AutoTokenizer


REPO = "princeton-nlp/prolong-data-64K"
REVISION = "1447128c751f0fbc7d2082b2a113a938426a9bba"
PROTOCOL = "DETERMINISTIC_SHARD_PERMUTATION_STREAMING_V1"
MAX_CONTEXT = 32768
EOS = 151643
QWEN = Path(os.path.expandvars("${MODEL_ROOT}/Qwen3-4B-Base"))
LLAMA = (
    V1 / "source_cache" / "hf" / "models--NousResearch--Meta-Llama-3.1-8B"
    / "snapshots" / "1f47e50cdbe801ad8a5174156ec3a0655108fb9f"
)
INDEX_ROOT = (
    V1 / "source_cache" / "hf" / "datasets--princeton-nlp--prolong-data-64K"
    / "snapshots" / REVISION
)
ANCHOR = V2 / "manifests" / "HISTORICAL_300M_RAW_ANCHOR.jsonl"
QUOTAS = ROOT / "manifests" / "EXTENSION_QUOTAS.json"
INVENTORY = ROOT / "manifests" / "UPSTREAM_SHARD_INVENTORY.jsonl"
MAIN_AUDIT = ROOT / "audits" / "UPSTREAM_SHARD_INVENTORY_AUDIT.json"
INDEPENDENT_AUDIT = ROOT / "audits" / "UPSTREAM_SHARD_INVENTORY_INDEPENDENT_AUDIT_V2.json"
SCOPE_AUDIT = ROOT / "audits" / "UPSTREAM_REPOSITORY_FILE_SCOPE_AUDIT.json"
TOKENIZER_AUDIT = V2 / "audits" / "QWEN3_4B_8B_TOKENIZER_PARITY.json"
WORK = ROOT / "tmp" / "stream_selection"
CACHE = WORK / "source_shards"
STATE = WORK / "state.json"
LOCK = WORK / "selector.lock"
INPROGRESS_RECEIPT = WORK / "STREAM_SCAN_RECEIPT.inprogress.jsonl"
FINAL_RECEIPT = ROOT / "manifests" / "STREAM_SCAN_RECEIPT.jsonl"
OUTPUT_ALL = ROOT / "manifests" / "V3_EXTENSION_700M_SELECTION.jsonl"
OUTPUT_TRAIN = ROOT / "manifests" / "V3_EXTENSION_TRAIN_SELECTION.jsonl"
OUTPUT_VAL = ROOT / "manifests" / "V3_EXTENSION_VAL_SELECTION.jsonl"
AUDIT = ROOT / "audits" / "V3_EXTENSION_SELECTION_AUDIT.json"
COST_JSON = ROOT / "reports" / "STREAMING_COST_REPORT.json"
COST_MD = ROOT / "reports" / "STREAMING_COST_REPORT.md"
EXPECTED_ANCHOR_SHA = "017a74186b419b2c3f566f568e6c2d120723e3ddafb69024138eb6b321c90ae8"
EXPECTED_INVENTORY_SHA = "58ba8027e314652a9eef291bee37e5ec0bed0dd672d5bb20b11962c38de86bea"
EXPECTED_TOKENIZER_SHA = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"


def now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def atomic_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def acquire_lock():
    WORK.mkdir(parents=True, exist_ok=True)
    stream = LOCK.open("a+b")
    if stream.seek(0, os.SEEK_END) == 0:
        stream.write(b"0")
        stream.flush()
    stream.seek(0)
    try:
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        raise SystemExit("another V3 streaming selector holds the process lock")
    stream.seek(0)
    stream.write(str(os.getpid()).encode("ascii").ljust(32, b" "))
    stream.flush()
    os.fsync(stream.fileno())
    return stream


def ndarray(data, expected_dtype):
    descriptor = data[0]
    ndim = descriptor >> 2
    dtype_index = descriptor % 4
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
    return (
        encoded["domain"].decode("utf-8"),
        ndarray(encoded["indices"], np.uint32),
        ndarray(encoded["input_ids"], np.uint32).reshape(-1),
        int(np.frombuffer(encoded["length"], np.uint64)[0]),
    )


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
    all_ids = encoded["input_ids"]
    if len(all_ids) <= limit:
        return len(text), all_ids
    offset_ends = [int(pair[1]) for pair in encoded["offset_mapping"]]
    low, high = start, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        approximate = bisect_right(offset_ends, middle - start)
        size = approximate if abs(approximate - limit) > 128 else len(tokenizer.encode(text[start:middle], add_special_tokens=False))
        if size <= limit:
            low = middle
        else:
            high = middle - 1
    ids = tokenizer.encode(text[start:low], add_special_tokens=False)
    if len(ids) > limit or (low < len(text) and len(tokenizer.encode(text[start:low + 1], add_special_tokens=False)) <= limit):
        raise ValueError("32K maximal-character-prefix invariant failed")
    return low, ids


def qwen_32k_plan(tokenizer, text):
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    full_ids = encoded["input_ids"]
    if not full_ids:
        return [], []
    if len(full_ids) <= MAX_CONTEXT:
        return full_ids, [{"character_start": 0, "character_end": len(text), "token_ids": full_ids}]
    chunks, character_start, preencoded = [], 0, encoded
    while character_start < len(text):
        character_end, ids = maximal_character_prefix(tokenizer, text, character_start, preencoded=preencoded)
        preencoded = None
        if character_end <= character_start or not ids:
            raise ValueError("32K chunker made no progress")
        chunks.append({"character_start": character_start, "character_end": character_end, "token_ids": ids})
        character_start = character_end
    return full_ids, chunks


def exact_character_prefix(tokenizer, text, start, target_tokens, end_limit):
    if target_tokens <= 0:
        raise ValueError("terminal character prefix must be positive")
    segment = text[start:end_limit]
    encoded = tokenizer(segment, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    if target_tokens > len(ids):
        raise ValueError("terminal character prefix exceeds chunk")
    if target_tokens == len(ids):
        return end_limit, ids
    offsets = encoded["offset_mapping"]
    guesses = {int(offsets[target_tokens - 1][1])}
    for delta in range(-16, 17):
        guesses.add(max(1, min(len(segment), int(offsets[target_tokens - 1][1]) + delta)))
    for relative_end in sorted(guesses):
        candidate = tokenizer.encode(segment[:relative_end], add_special_tokens=False)
        if len(candidate) == target_tokens:
            return start + relative_end, candidate
    low, high = 1, len(segment)
    while low <= high:
        middle = (low + high) // 2
        candidate = tokenizer.encode(segment[:middle], add_special_tokens=False)
        if len(candidate) == target_tokens:
            return start + middle, candidate
        if len(candidate) < target_tokens:
            low = middle + 1
        else:
            high = middle - 1
    raise ValueError("no trainer-safe exact character prefix for requested token count")


def full_first_fit(chunk_sizes, record_fill):
    eos, fill = 0, record_fill
    for index, token_n in enumerate(chunk_sizes):
        if index == 0:
            if fill and 1 + token_n <= MAX_CONTEXT:
                eos, fill = 1, fill + 1 + token_n
            else:
                fill = token_n
        else:
            fill = token_n
        if index < len(chunk_sizes) - 1:
            fill = 0
    return sum(chunk_sizes) + eos, eos, fill


def terminal_plan(tokenizer, text, chunks, record_fill, remaining):
    full_total, normal_eos, _ = full_first_fit([len(c["token_ids"]) for c in chunks], record_fill)
    if not 0 < remaining < full_total:
        raise ValueError("terminal plan called outside overshoot case")
    eos = normal_eos
    if remaining <= eos:
        eos = 0
    document_needed = remaining - eos
    if document_needed <= 0:
        raise ValueError("terminal plan would select EOS only")
    selected, left = [], document_needed
    for chunk in chunks:
        available = len(chunk["token_ids"])
        take = min(available, left)
        if take == available:
            end, ids = chunk["character_end"], chunk["token_ids"]
        else:
            end, ids = exact_character_prefix(tokenizer, text, chunk["character_start"], take, chunk["character_end"])
        selected.append({"character_start": chunk["character_start"], "character_end": end, "token_ids": ids})
        left -= take
        if left == 0:
            break
    if left:
        raise ValueError("terminal token accounting failed")
    ending_fill = len(selected[-1]["token_ids"])
    if len(selected) == 1 and eos:
        ending_fill = record_fill + eos + ending_fill
    return eos, selected, ending_fill


def verify_shard(path, shard):
    if path.stat().st_size != shard["expected_byte_size"]:
        raise ValueError("shard size mismatch: " + shard["canonical_shard_id"])
    if shard["expected_lfs_sha256"]:
        actual = sha256_file(path)
        if actual != shard["expected_lfs_sha256"]:
            raise ValueError("shard SHA256 mismatch: " + shard["canonical_shard_id"])
        return "lfs_sha256", actual
    actual = git_blob_id(path)
    if actual != shard["expected_git_blob_sha1"]:
        raise ValueError("shard Git blob mismatch: " + shard["canonical_shard_id"])
    return "git_blob_sha1", actual


def acquire_shard(shard):
    cached = shard.get("local_cached_path")
    if cached and Path(cached).is_file():
        path = Path(cached)
        verify_shard(path, shard)
        return path, "V2_REUSE", 0
    path = CACHE / shard["canonical_shard_id"]
    if path.is_file():
        verify_shard(path, shard)
        return path, "V3_CACHE_REUSE", 0
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    url = "https://hf-mirror.com/datasets/%s/resolve/%s/%s" % (REPO, REVISION, quote(shard["canonical_shard_id"], safe="/"))
    last_error = None
    for attempt in range(1, 6):
        try:
            with requests.get(url, stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                with temporary.open("wb") as output:
                    for block in response.iter_content(8 * 1024 * 1024):
                        if block:
                            output.write(block)
                    output.flush()
                    os.fsync(output.fileno())
            verify_shard(temporary, shard)
            os.replace(str(temporary), str(path))
            return path, "NEW_DOWNLOAD", path.stat().st_size
        except Exception as error:
            last_error = error
            if temporary.exists():
                temporary.unlink()
            if attempt < 5:
                time.sleep(2 ** attempt)
    raise RuntimeError("download failed for %s: %r" % (shard["canonical_shard_id"], last_error))


def load_gates():
    main = json.loads(MAIN_AUDIT.read_text(encoding="utf-8"))
    independent = json.loads(INDEPENDENT_AUDIT.read_text(encoding="utf-8"))
    scope = json.loads(SCOPE_AUDIT.read_text(encoding="utf-8"))
    tokenizer = json.loads(TOKENIZER_AUDIT.read_text(encoding="utf-8"))
    order_hashes = all(
        Path(value["path"]).is_file()
        and sha256_file(Path(value["path"])) == value["sha256"]
        and value.get("byte_parity") is True
        for value in independent.get("order_manifests", {}).values()
    ) and len(independent.get("order_manifests", {})) == 10
    requirements = {
        "main": main.get("status") == "PASS",
        "independent": independent.get("INDEPENDENT_SHARD_GATE") == "PASS",
        "scope": scope.get("status") == "PASS",
        "inventory_hash": sha256_file(INVENTORY) == EXPECTED_INVENTORY_SHA,
        "anchor_hash": sha256_file(ANCHOR) == EXPECTED_ANCHOR_SHA,
        "tokenizer_hash": sha256_file(QWEN / "tokenizer.json") == EXPECTED_TOKENIZER_SHA,
        "tokenizer_parity": tokenizer.get("status") == "PASS" or tokenizer.get("QWEN3_4B_8B_TOKENIZER_PARITY") == "PASS",
        "shard_order_hashes": order_hashes,
    }
    failed = [key for key, value in requirements.items() if not value]
    if failed:
        raise SystemExit("streaming selection prerequisite gate failed: " + ", ".join(failed))
    return requirements


def load_inventory():
    result = {}
    with INVENTORY.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            result[row["canonical_shard_id"]] = row
    if len(result) != 2432:
        raise ValueError("target shard inventory cardinality changed")
    return result


def load_schemas():
    schemas = {}
    for source_dir in sorted(INDEX_ROOT.iterdir()):
        if not source_dir.is_dir() or source_dir.name == "dclm-baseline":
            continue
        for index_path in sorted(source_dir.glob("proc*/index.json")):
            index = json.loads(index_path.read_text(encoding="utf-8"))
            for schema in index["shards"]:
                key = "%s/%s/%s" % (source_dir.name, index_path.parent.name, schema["raw_data"]["basename"])
                schemas[key] = schema
    if len(schemas) != 2432:
        raise ValueError("proc schema cardinality mismatch")
    return schemas


def load_anchor():
    identities, hashes, contributions = set(), set(), Counter()
    with ANCHOR.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            identities.add(row["canonical_raw_identity"])
            hashes.add(row["decoded_content_sha256"])
            contributions[(row["source"], row["split"])] += int(row["qwen_token_count"]) + int(row["eos_contribution"])
    return identities, hashes, contributions


def load_order(source, split):
    safe = "".join(c.upper() if c.isalnum() else "_" for c in source)
    path = ROOT / "manifests" / ("SHARD_ORDER_%s.jsonl" % safe)
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["order_kind"] == split:
                rows.append(row["canonical_shard_id"])
    if len(rows) != len(set(rows)):
        raise ValueError("duplicate shard in order manifest: %s/%s" % (source, split))
    return rows


def receipt_state(committed_bytes):
    selected_ids = {"validation": set(), "train": set()}
    selected_hashes = {"validation": set(), "train": set()}
    logical, fill = Counter(), Counter()
    accepted = []
    receipt_path = (
        INPROGRESS_RECEIPT
        if INPROGRESS_RECEIPT.is_file()
        else FINAL_RECEIPT
    )
    if not receipt_path.is_file() or committed_bytes == 0:
        return selected_ids, selected_hashes, logical, fill, accepted
    with receipt_path.open("rb") as stream:
        data = stream.read(committed_bytes)
    if len(data) != committed_bytes or (data and not data.endswith(b"\n")):
        raise ValueError("committed receipt boundary invalid")
    for line in data.splitlines():
        row = json.loads(line)
        if row["decision"] == "ACCEPT":
            split, source = row["split"], row["source"]
            selected_ids[split].add(row["canonical_identity"])
            selected_hashes[split].add(row["normalized_content_sha256"])
            logical[(source, split)] += int(row["selected_contribution"])
            fill[(source, split)] = int(row["record_fill_after"])
            accepted.append(row)
    return selected_ids, selected_hashes, logical, fill, accepted


def new_state(authority_hashes, sources, inventory):
    return {
        "state_schema": "PROLONG_V3_STREAM_SELECTION_STATE_V1",
        "protocol": PROTOCOL,
        "upstream_revision": REVISION,
        "authority_hashes": authority_hashes,
        "source_order": sources,
        "started_at": now(),
        "updated_at": now(),
        "receipt_committed_bytes": 0,
        "completed_source_splits": [],
        "current": None,
        "telemetry": {
            "%s/%s" % (source, split): {
                "total_target_shards_available": sum(1 for row in inventory.values() if row["canonical_source"] == source),
                "shards_touched": 0, "shards_reused": 0, "shards_downloaded": 0,
                "rows_scanned": 0, "rows_content_hashed": 0, "rows_tokenized": 0,
                "rows_accepted": 0, "rows_skipped": 0, "bytes_downloaded": 0,
                "bytes_read": 0, "logical_tokens_accepted": 0,
            } for split in ("validation", "train") for source in sources
        },
        "touched_shards": {"%s/%s" % (source, split): [] for split in ("validation", "train") for source in sources},
        "disk_free_before": shutil.disk_usage(ROOT).free,
    }


def append_commit(receipt_stream, rows, state, next_current):
    payload = b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8") for row in rows)
    receipt_stream.write(payload)
    receipt_stream.flush()
    os.fsync(receipt_stream.fileno())
    state["receipt_committed_bytes"] = receipt_stream.tell()
    state["current"] = next_current
    state["updated_at"] = now()
    atomic_json(STATE, state)


def public_chunk(chunk):
    ids = chunk["token_ids"]
    return {
        "character_start": chunk["character_start"],
        "character_end": chunk["character_end"],
        "qwen_token_n": len(ids),
        "qwen_token_ids_sha256": token_ids_sha256(ids),
    }


def finalize(state, quotas, sources, accepted):
    if INPROGRESS_RECEIPT.is_file():
        if INPROGRESS_RECEIPT.stat().st_size != state["receipt_committed_bytes"]:
            raise ValueError("receipt size differs from committed state")
        os.replace(str(INPROGRESS_RECEIPT), str(FINAL_RECEIPT))
    elif not FINAL_RECEIPT.is_file():
        raise ValueError("completed state has no receipt")
    selections = {"validation": [], "train": []}
    for row in accepted:
        item = {
            "selection_schema": "PROLONG_V3_STREAMING_EXTENSION_SELECTION_V1",
            "selection_order": len(selections[row["split"]]),
            "global_selection_order": sum(len(v) for v in selections.values()),
            "source": row["source"], "split": row["split"],
            "protocol": PROTOCOL,
            "canonical_raw_identity": row["canonical_identity"],
            "canonical_shard_id": row["shard_id"],
            "canonical_row_id": row["canonical_row_id"],
            "decoded_content_sha256": row["normalized_content_sha256"],
            "qwen_full_token_n": row["logical_token_count"],
            "selected_32k_chunks": row["selected_32k_chunks"],
            "eos_contribution": row["eos_contribution"],
            "selected_document_token_n": row["selected_document_token_n"],
            "selected_logical_token_n": row["selected_contribution"],
            "terminal_trim": row["terminal_trim"],
            "trim_amount": row["trimmed_tokens"],
            "record_fill_before": row["record_fill_before"],
            "record_fill_after": row["record_fill_after"],
            "upstream_file_sha256": row["upstream_file_sha256"],
            "upstream_git_blob_sha1": row["upstream_git_blob_sha1"],
        }
        selections[row["split"]].append(item)
    atomic_jsonl(OUTPUT_VAL, selections["validation"])
    atomic_jsonl(OUTPUT_TRAIN, selections["train"])
    atomic_jsonl(OUTPUT_ALL, selections["validation"] + selections["train"])
    actual = Counter()
    for split in selections:
        for row in selections[split]:
            actual[(row["source"], split)] += row["selected_logical_token_n"]
    expected = {(source, split): int(quotas["sources"][source]["extension_" + ("validation" if split == "validation" else "train")]) for split in ("validation", "train") for source in sources}
    if dict(actual) != expected:
        raise ValueError("final extension quota parity failed")
    telemetry = state["telemetry"]
    cost = {
        "report_id": "PROLONG_V3_STREAMING_COST_REPORT_V1", "status": "PASS",
        "protocol": PROTOCOL, "per_source_split": telemetry,
        "bytes_downloaded": sum(v["bytes_downloaded"] for v in telemetry.values()),
        "bytes_read": sum(v["bytes_read"] for v in telemetry.values()),
        "shards_reused": sum(v["shards_reused"] for v in telemetry.values()),
        "shards_downloaded": sum(v["shards_downloaded"] for v in telemetry.values()),
        "rows_scanned": sum(v["rows_scanned"] for v in telemetry.values()),
        "rows_tokenized": sum(v["rows_tokenized"] for v in telemetry.values()),
        "logical_tokens_accepted": sum(v["logical_tokens_accepted"] for v in telemetry.values()),
        "disk_free_before": state["disk_free_before"], "disk_free_after_selection": shutil.disk_usage(ROOT).free,
        "FULL_POOL_DOCUMENT_INVENTORY": False, "FULL_POOL_DOCUMENT_TOKENIZATION": False,
        "GLOBAL_DOCUMENT_SHA_RANKING": False, "ONLINE_DEDUP_ONLY": True,
    }
    atomic_json(COST_JSON, cost)
    lines = ["# V3 Streaming Cost Report", "", "- Status: PASS", "- Bytes downloaded: %d" % cost["bytes_downloaded"], "- Bytes read: %d" % cost["bytes_read"], "- Rows scanned: %d" % cost["rows_scanned"], "- Rows tokenized: %d" % cost["rows_tokenized"], "- Logical tokens accepted: %d" % cost["logical_tokens_accepted"], "", "| Source/split | Available shards | Touched | Reused | Downloaded | Rows scanned | Rows tokenized | Rows accepted | Tokens |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, value in telemetry.items():
        lines.append("| %s | %d | %d | %d | %d | %d | %d | %d | %d |" % (key, value["total_target_shards_available"], value["shards_touched"], value["shards_reused"], value["shards_downloaded"], value["rows_scanned"], value["rows_tokenized"], value["rows_accepted"], value["logical_tokens_accepted"]))
    temporary_cost_md = COST_MD.with_name(COST_MD.name + ".partial")
    with temporary_cost_md.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary_cost_md), str(COST_MD))
    audit = {
        "audit_id": "V3_EXTENSION_STREAM_SELECTION_V1", "status": "PASS",
        "EXTENSION_SELECTION_FROZEN": True, "selection_protocol": PROTOCOL,
        "historical_anchor_sha256": EXPECTED_ANCHOR_SHA,
        "upstream_inventory_sha256": EXPECTED_INVENTORY_SHA,
        "stream_scan_receipt_sha256": sha256_file(FINAL_RECEIPT),
        "extension_train_selection_sha256": sha256_file(OUTPUT_TRAIN),
        "extension_val_selection_sha256": sha256_file(OUTPUT_VAL),
        "extension_700m_selection_sha256": sha256_file(OUTPUT_ALL),
        "extension_train_logical_tokens": sum(v for (s, p), v in actual.items() if p == "train"),
        "extension_validation_logical_tokens": sum(v for (s, p), v in actual.items() if p == "validation"),
        "extension_logical_tokens": sum(actual.values()),
        "per_source_split_tokens": {source: {split: actual[(source, split)] for split in ("train", "validation")} for source in sources},
        "historical_extension_identity_overlap_n": 0,
        "historical_extension_content_overlap_n": 0,
        "extension_train_val_identity_overlap_n": len({r["canonical_raw_identity"] for r in selections["train"]} & {r["canonical_raw_identity"] for r in selections["validation"]}),
        "extension_train_val_content_overlap_n": len({r["decoded_content_sha256"] for r in selections["train"]} & {r["decoded_content_sha256"] for r in selections["validation"]}),
        "terminal_trim_n": sum(int(r["terminal_trim"]) for values in selections.values() for r in values),
        "FULL_POOL_DOCUMENT_INVENTORY": False, "FULL_POOL_DOCUMENT_TOKENIZATION": False,
        "GLOBAL_DOCUMENT_SHA_RANKING": False, "ONLINE_DEDUP_ONLY": True,
    }
    if audit["extension_logical_tokens"] != 700_000_000 or audit["extension_train_logical_tokens"] != 693_000_000 or audit["extension_validation_logical_tokens"] != 7_000_000 or audit["extension_train_val_identity_overlap_n"] or audit["extension_train_val_content_overlap_n"]:
        raise ValueError("extension final hard gate failed")
    atomic_json(AUDIT, audit)
    state["status"] = "PASS"
    state["completed_at"] = now()
    state["final_receipt_sha256"] = sha256_file(FINAL_RECEIPT)
    state["current"] = None
    atomic_json(STATE, state)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock()
    gates = load_gates()
    quotas = json.loads(QUOTAS.read_text(encoding="utf-8"))
    sources = list(quotas["sources"])
    inventory = load_inventory()
    schemas = load_schemas()
    authority_hashes = {
        "anchor": EXPECTED_ANCHOR_SHA, "inventory": EXPECTED_INVENTORY_SHA,
        "quotas": sha256_file(QUOTAS), "tokenizer_json": EXPECTED_TOKENIZER_SHA,
        "independent_gate": sha256_file(INDEPENDENT_AUDIT), "scope_audit": sha256_file(SCOPE_AUDIT),
    }
    anchor_ids, anchor_hashes, historical = load_anchor()
    expected_historical = {(source, split): int(quotas["sources"][source]["historical_" + ("validation" if split == "validation" else "train")]) for split in ("validation", "train") for source in sources}
    if dict(historical) != expected_historical:
        raise ValueError("historical anchor source/split contributions changed")
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else new_state(authority_hashes, sources, inventory)
    if state["authority_hashes"] != authority_hashes or state["protocol"] != PROTOCOL:
        raise ValueError("resume authority mismatch")
    if state.get("status") == "PASS":
        print("V3 streaming selection already PASS", flush=True)
        return
    committed = int(state["receipt_committed_bytes"])
    if INPROGRESS_RECEIPT.is_file():
        inprogress_size = INPROGRESS_RECEIPT.stat().st_size
        if inprogress_size < committed:
            if (
                FINAL_RECEIPT.is_file()
                and FINAL_RECEIPT.stat().st_size == committed
                and len(state.get("completed_source_splits", [])) == 20
            ):
                INPROGRESS_RECEIPT.unlink()
            else:
                raise ValueError("in-progress receipt shorter than committed state")
        else:
            with INPROGRESS_RECEIPT.open("r+b") as stream:
                stream.truncate(committed)
    elif committed and not (
        FINAL_RECEIPT.is_file()
        and FINAL_RECEIPT.stat().st_size == committed
        and len(state.get("completed_source_splits", [])) == 20
    ):
        raise ValueError("committed state exists but no valid receipt is available")
    selected_ids, selected_hashes, logical, record_fill, accepted = receipt_state(committed)
    llama = AutoTokenizer.from_pretrained(LLAMA, local_files_only=True, use_fast=True, trust_remote_code=False)
    qwen = AutoTokenizer.from_pretrained(QWEN, local_files_only=True, use_fast=True, trust_remote_code=False)
    if qwen.eos_token_id != EOS:
        raise ValueError("unexpected Qwen EOS token")
    completed = set(state["completed_source_splits"])
    pairs = [(source, split) for split in ("validation", "train") for source in sources]
    if completed == {"%s/%s" % pair for pair in pairs}:
        finalize(state, quotas, sources, accepted)
        return
    with INPROGRESS_RECEIPT.open("a+b") as receipt:
        receipt.seek(committed)
        for source, split in pairs:
            pair_key = "%s/%s" % (source, split)
            if pair_key in completed:
                continue
            quota_key = "extension_validation" if split == "validation" else "extension_train"
            quota = int(quotas["sources"][source][quota_key])
            order = load_order(source, split)
            current = state.get("current") or {}
            if current.get("source") == source and current.get("split") == split:
                start_shard, start_row = int(current["shard_cursor"]), int(current["row_cursor"])
            else:
                start_shard, start_row = 0, 0
                state["current"] = {"source": source, "split": split, "shard_cursor": 0, "row_cursor": 0, "accepted_tokens": logical[(source, split)]}
                atomic_json(STATE, state)
            telemetry = state["telemetry"][pair_key]
            for shard_rank in range(start_shard, len(order)):
                if logical[(source, split)] == quota:
                    break
                shard_id = order[shard_rank]
                shard = inventory[shard_id]
                path, acquisition, downloaded = acquire_shard(shard)
                resolved_file_sha256 = sha256_file(path)
                if shard_id not in state["touched_shards"][pair_key]:
                    state["touched_shards"][pair_key].append(shard_id)
                    telemetry["shards_touched"] += 1
                    telemetry["shards_reused"] += int(acquisition != "NEW_DOWNLOAD")
                    telemetry["shards_downloaded"] += int(acquisition == "NEW_DOWNLOAD")
                    telemetry["bytes_downloaded"] += downloaded
                    telemetry["bytes_read"] += shard["expected_byte_size"]
                row_begin = start_row if shard_rank == start_shard else 0
                with path.open("rb", buffering=0) as mds:
                    for mds_row in range(row_begin, int(shard["upstream_native_row_n"])):
                        _, indices, input_ids, _ = read_row(mds, mds_row, schemas[shard_id])
                        batch = []
                        for document_position, (begin, end) in enumerate(indices.tolist()):
                            if logical[(source, split)] == quota:
                                break
                            raw = input_ids[int(begin):int(end)]
                            relative_shard = shard_id.split("/", 1)[1]
                            canonical_row_id = "%d:%d" % (mds_row, document_position)
                            identity = "%s:%s:%s:%s" % (REVISION, source, relative_shard, canonical_row_id)
                            base = {"receipt_schema": "PROLONG_V3_STREAM_SCAN_RECEIPT_V1", "source": source, "split": split, "shard_id": shard_id, "shard_rank": shard_rank, "row_id": mds_row, "document_position": document_position, "canonical_row_id": canonical_row_id, "canonical_identity": identity}
                            telemetry["rows_scanned"] += 1
                            if identity in anchor_ids:
                                batch.append({**base, "normalized_content_sha256": None, "decision": "HISTORICAL_ID_SKIP"})
                                telemetry["rows_skipped"] += 1
                                continue
                            text = decode_ids(llama, source, raw)
                            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                            telemetry["rows_content_hashed"] += 1
                            base["normalized_content_sha256"] = content_hash
                            if content_hash in anchor_hashes:
                                batch.append({**base, "decision": "HISTORICAL_CONTENT_SKIP"})
                                telemetry["rows_skipped"] += 1
                                continue
                            if split == "train" and (identity in selected_ids["validation"] or content_hash in selected_hashes["validation"]):
                                batch.append({**base, "decision": "VAL_OVERLAP_SKIP"})
                                telemetry["rows_skipped"] += 1
                                continue
                            if identity in selected_ids[split] or content_hash in selected_hashes[split]:
                                batch.append({**base, "decision": "EXTENSION_DUP_SKIP"})
                                telemetry["rows_skipped"] += 1
                                continue
                            full_ids, chunks = qwen_32k_plan(qwen, text)
                            telemetry["rows_tokenized"] += 1
                            if not full_ids:
                                batch.append({**base, "decision": "INVALID_SKIP", "invalid_reason": "empty_qwen_tokenization"})
                                telemetry["rows_skipped"] += 1
                                continue
                            chunk_sizes = [len(chunk["token_ids"]) for chunk in chunks]
                            before = record_fill[(source, split)]
                            full_logical, eos, after = full_first_fit(chunk_sizes, before)
                            remaining = quota - logical[(source, split)]
                            terminal = full_logical > remaining
                            if terminal:
                                try:
                                    eos, selected_chunks, after = terminal_plan(qwen, text, chunks, before, remaining)
                                except ValueError as error:
                                    batch.append({**base, "decision": "INVALID_SKIP", "invalid_reason": str(error)})
                                    telemetry["rows_skipped"] += 1
                                    continue
                            else:
                                selected_chunks = chunks
                            selected_document_tokens = sum(len(chunk["token_ids"]) for chunk in selected_chunks)
                            selected_logical = selected_document_tokens + eos
                            if terminal and selected_logical != remaining:
                                raise ValueError("terminal quota contribution mismatch")
                            full_document_logical_tokens = sum(chunk_sizes)
                            accepted_row = {**base, "decision": "ACCEPT", "logical_token_count": full_document_logical_tokens, "selected_document_token_n": selected_document_tokens, "selected_contribution": selected_logical, "eos_contribution": eos, "terminal_trim": terminal, "trimmed_tokens": full_document_logical_tokens - selected_document_tokens, "record_fill_before": before, "record_fill_after": after, "selected_32k_chunks": [public_chunk(chunk) for chunk in selected_chunks], "upstream_file_sha256": resolved_file_sha256, "upstream_git_blob_sha1": shard["expected_git_blob_sha1"]}
                            batch.append(accepted_row)
                            selected_ids[split].add(identity)
                            selected_hashes[split].add(content_hash)
                            logical[(source, split)] += selected_logical
                            record_fill[(source, split)] = after
                            accepted.append(accepted_row)
                            telemetry["rows_accepted"] += 1
                            telemetry["logical_tokens_accepted"] += selected_logical
                        next_row = mds_row + 1
                        next_shard = shard_rank
                        if next_row >= int(shard["upstream_native_row_n"]):
                            next_shard, next_row = shard_rank + 1, 0
                        append_commit(receipt, batch, state, {"source": source, "split": split, "shard_cursor": next_shard, "row_cursor": next_row, "accepted_tokens": logical[(source, split)]})
                        if logical[(source, split)] == quota:
                            break
                start_row = 0
                print("%s tokens=%d/%d shards=%d rows=%d" % (pair_key, logical[(source, split)], quota, telemetry["shards_touched"], telemetry["rows_scanned"]), flush=True)
            if logical[(source, split)] != quota:
                raise ValueError("insufficient eligible content for %s: %d/%d" % (pair_key, logical[(source, split)], quota))
            completed.add(pair_key)
            state["completed_source_splits"] = ["%s/%s" % pair for pair in pairs if "%s/%s" % pair in completed]
            state["current"] = None
            state["updated_at"] = now()
            atomic_json(STATE, state)
            print("COMPLETE %s tokens=%d" % (pair_key, quota), flush=True)
    finalize(state, quotas, sources, accepted)


if __name__ == "__main__":
    main()
