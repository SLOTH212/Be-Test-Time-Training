"""Independent metadata replay of the frozen V3 streaming scan receipt.

This verifier imports no selector code and does not open or tokenize MDS data.
Raw-byte grounding is performed by the separate deterministic probe auditor.
"""

import hashlib
import json
import os
from collections import Counter
from pathlib import Path


V2 = Path(os.path.expandvars("${DATA_ROOT}/prolong_1b_32k_qwen3_v2"))
ROOT = Path(os.path.expandvars("${DATA_ROOT}/prolong_1b_32k_qwen3_v3_streaming"))
ANCHOR = V2 / "manifests" / "HISTORICAL_300M_RAW_ANCHOR.jsonl"
QUOTAS = ROOT / "manifests" / "EXTENSION_QUOTAS.json"
RECEIPT = ROOT / "manifests" / "STREAM_SCAN_RECEIPT.jsonl"
SELECTION = ROOT / "manifests" / "V3_EXTENSION_700M_SELECTION.jsonl"
SELECTION_TRAIN = ROOT / "manifests" / "V3_EXTENSION_TRAIN_SELECTION.jsonl"
SELECTION_VAL = ROOT / "manifests" / "V3_EXTENSION_VAL_SELECTION.jsonl"
PRIMARY_AUDIT = ROOT / "audits" / "V3_EXTENSION_SELECTION_AUDIT.json"
OUTPUT = ROOT / "audits" / "STREAM_RECEIPT_REPLAY_AUDIT.json"
EXPECTED_ANCHOR_SHA = "017a74186b419b2c3f566f568e6c2d120723e3ddafb69024138eb6b321c90ae8"
DECISIONS = {
    "ACCEPT", "HISTORICAL_ID_SKIP", "HISTORICAL_CONTENT_SKIP",
    "EXTENSION_DUP_SKIP", "VAL_OVERLAP_SKIP", "INVALID_SKIP",
}


def sha256_file(path):
    digest = hashlib.sha256()
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


def load_anchor():
    identities, hashes = set(), set()
    with ANCHOR.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            identities.add(row["canonical_raw_identity"])
            hashes.add(row["decoded_content_sha256"])
    return identities, hashes


def load_orders(sources):
    result = {}
    for source in sources:
        safe = "".join(c.upper() if c.isalnum() else "_" for c in source)
        path = ROOT / "manifests" / ("SHARD_ORDER_%s.jsonl" % safe)
        by_split = {"validation": [], "train": []}
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["order_kind"] in by_split:
                    by_split[row["order_kind"]].append(row["canonical_shard_id"])
        for split, values in by_split.items():
            result[(source, split)] = values
    return result


def load_selection(path):
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            rows.append(json.loads(line))
    return rows


def main():
    if sha256_file(ANCHOR) != EXPECTED_ANCHOR_SHA:
        raise ValueError("historical anchor SHA changed")
    quotas = json.loads(QUOTAS.read_text(encoding="utf-8"))
    sources = list(quotas["sources"])
    orders = load_orders(sources)
    anchor_ids, anchor_hashes = load_anchor()
    selected_ids = {"validation": set(), "train": set()}
    selected_hashes = {"validation": set(), "train": set()}
    logical = Counter()
    accepted = []
    prior_position = {}
    decision_counts = Counter()
    mismatch = Counter()
    line_n = 0
    pair_closed = set()
    pair_sequence = [(source, split) for split in ("validation", "train") for source in sources]
    pair_rank = {pair: index for index, pair in enumerate(pair_sequence)}
    last_pair_rank = -1

    with RECEIPT.open("r", encoding="utf-8") as stream:
        for line_n, line in enumerate(stream, 1):
            row = json.loads(line)
            source, split = row["source"], row["split"]
            pair = (source, split)
            decision = row.get("decision")
            if pair not in pair_rank or decision not in DECISIONS:
                mismatch["schema_or_enum"] += 1
                continue
            if pair_rank[pair] < last_pair_rank:
                mismatch["pair_order"] += 1
            if pair_rank[pair] > last_pair_rank:
                last_pair_rank = pair_rank[pair]
            shard_rank = int(row["shard_rank"])
            order = orders[pair]
            if not 0 <= shard_rank < len(order) or order[shard_rank] != row["shard_id"]:
                mismatch["shard_order"] += 1
            position = (shard_rank, int(row["row_id"]), int(row["document_position"]))
            if pair in prior_position and position <= prior_position[pair]:
                mismatch["native_row_order"] += 1
            prior_position[pair] = position
            identity = row["canonical_identity"]
            content_hash = row.get("normalized_content_sha256")
            expected_identity = "%s:%s:%s:%d:%d" % (
                "1447128c751f0fbc7d2082b2a113a938426a9bba", source,
                row["shard_id"].split("/", 1)[1], int(row["row_id"]),
                int(row["document_position"]),
            )
            if identity != expected_identity:
                mismatch["canonical_identity"] += 1
            if pair in pair_closed:
                mismatch["rows_after_quota"] += 1

            if decision == "HISTORICAL_ID_SKIP":
                if identity not in anchor_ids or content_hash is not None:
                    mismatch["historical_id_decision"] += 1
            elif decision == "HISTORICAL_CONTENT_SKIP":
                if identity in anchor_ids or content_hash not in anchor_hashes:
                    mismatch["historical_content_decision"] += 1
            elif decision == "VAL_OVERLAP_SKIP":
                if split != "train" or not (
                    identity in selected_ids["validation"]
                    or content_hash in selected_hashes["validation"]
                ):
                    mismatch["validation_overlap_decision"] += 1
            elif decision == "EXTENSION_DUP_SKIP":
                if not (
                    identity in selected_ids[split]
                    or content_hash in selected_hashes[split]
                ):
                    mismatch["extension_duplicate_decision"] += 1
            elif decision == "INVALID_SKIP":
                if not row.get("invalid_reason"):
                    mismatch["invalid_skip_reason"] += 1
            elif decision == "ACCEPT":
                if (
                    identity in anchor_ids or content_hash in anchor_hashes
                    or identity in selected_ids[split]
                    or content_hash in selected_hashes[split]
                    or (split == "train" and (
                        identity in selected_ids["validation"]
                        or content_hash in selected_hashes["validation"]
                    ))
                ):
                    mismatch["accepted_overlap"] += 1
                chunks = row.get("selected_32k_chunks", [])
                chunk_tokens = sum(int(chunk["qwen_token_n"]) for chunk in chunks)
                selected_document = int(row["selected_document_token_n"])
                selected_logical = int(row["selected_contribution"])
                eos = int(row["eos_contribution"])
                before, after = int(row["record_fill_before"]), int(row["record_fill_after"])
                if not chunks or any(not 0 < int(c["qwen_token_n"]) <= 32768 for c in chunks):
                    mismatch["chunk_plan"] += 1
                if chunk_tokens != selected_document or selected_logical != selected_document + eos or eos not in (0, 1):
                    mismatch["logical_accounting"] += 1
                quota_key = "extension_validation" if split == "validation" else "extension_train"
                quota = int(quotas["sources"][source][quota_key])
                remaining = quota - logical[pair]
                if not 0 < selected_logical <= remaining:
                    mismatch["quota_overshoot"] += 1
                terminal = bool(row["terminal_trim"])
                if terminal != (selected_logical == remaining and int(row["trimmed_tokens"]) > 0):
                    mismatch["terminal_flag"] += 1
                if int(row["logical_token_count"]) - selected_document != int(row["trimmed_tokens"]):
                    mismatch["trim_accounting"] += 1
                expected_after = int(chunks[-1]["qwen_token_n"])
                if len(chunks) == 1 and eos:
                    expected_after = before + eos + expected_after
                if after != expected_after:
                    mismatch["record_fill"] += 1
                logical[pair] += selected_logical
                selected_ids[split].add(identity)
                selected_hashes[split].add(content_hash)
                accepted.append(row)
                if logical[pair] == quota:
                    pair_closed.add(pair)
            decision_counts[decision] += 1

    expected_tokens = {
        (source, split): int(quotas["sources"][source]["extension_validation" if split == "validation" else "extension_train"])
        for source, split in pair_sequence
    }
    if dict(logical) != expected_tokens:
        mismatch["source_split_quota"] += 1
    if pair_closed != set(pair_sequence):
        mismatch["pair_completion"] += 1

    selection = load_selection(SELECTION)
    selection_val = load_selection(SELECTION_VAL)
    selection_train = load_selection(SELECTION_TRAIN)
    accepted_signature = [
        (r["source"], r["split"], r["canonical_identity"], int(r["selected_contribution"]))
        for r in accepted
    ]
    selection_signature = [
        (r["source"], r["split"], r["canonical_raw_identity"], int(r["selected_logical_token_n"]))
        for r in selection
    ]
    if accepted_signature != selection_signature:
        mismatch["accepted_selection_order"] += 1
    if selection != selection_val + selection_train:
        mismatch["combined_selection_order"] += 1
    primary = json.loads(PRIMARY_AUDIT.read_text(encoding="utf-8"))
    if primary.get("status") != "PASS" or primary.get("stream_scan_receipt_sha256") != sha256_file(RECEIPT):
        mismatch["primary_audit"] += 1

    status = "PASS" if not mismatch else "FAIL"
    audit = {
        "audit_id": "V3_STREAM_RECEIPT_INDEPENDENT_REPLAY_V1",
        "status": status,
        "STREAM_RECEIPT_REPLAY_PARITY": status,
        "primary_selector_imported": False,
        "raw_source_data_opened": False,
        "full_unused_pool_replayed": False,
        "historical_anchor_sha256": sha256_file(ANCHOR),
        "stream_scan_receipt_sha256": sha256_file(RECEIPT),
        "receipt_line_n": line_n,
        "decision_counts": dict(sorted(decision_counts.items())),
        "accepted_record_n": len(accepted),
        "per_source_split_tokens": {
            source: {split: logical[(source, split)] for split in ("train", "validation")}
            for source in sources
        },
        "extension_logical_tokens": sum(logical.values()),
        "historical_extension_identity_overlap_n": len((selected_ids["train"] | selected_ids["validation"]) & anchor_ids),
        "historical_extension_content_overlap_n": len((selected_hashes["train"] | selected_hashes["validation"]) & anchor_hashes),
        "extension_train_val_identity_overlap_n": len(selected_ids["train"] & selected_ids["validation"]),
        "extension_train_val_content_overlap_n": len(selected_hashes["train"] & selected_hashes["validation"]),
        "selection_sha256": sha256_file(SELECTION),
        "selection_train_sha256": sha256_file(SELECTION_TRAIN),
        "selection_val_sha256": sha256_file(SELECTION_VAL),
        "mismatch_counts": dict(sorted(mismatch.items())),
    }
    atomic_json(OUTPUT, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
