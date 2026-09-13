"""Freeze final split content order: historical anchor then V3 extension."""

import hashlib
import json
import os
from collections import Counter
from pathlib import Path


V2 = Path(os.path.expandvars("${DATA_ROOT}/prolong_1b_32k_qwen3_v2"))
ROOT = Path(os.path.expandvars("${DATA_ROOT}/prolong_1b_32k_qwen3_v3_streaming"))
ANCHOR = V2 / "manifests" / "HISTORICAL_300M_RAW_ANCHOR.jsonl"
EXT_TRAIN = ROOT / "manifests" / "V3_EXTENSION_TRAIN_SELECTION.jsonl"
EXT_VAL = ROOT / "manifests" / "V3_EXTENSION_VAL_SELECTION.jsonl"
QUOTAS = ROOT / "manifests" / "EXTENSION_QUOTAS.json"
SELECTION_AUDIT = ROOT / "audits" / "V3_EXTENSION_SELECTION_AUDIT.json"
REPLAY_AUDIT = ROOT / "audits" / "STREAM_RECEIPT_REPLAY_AUDIT.json"
PROBE_AUDIT = ROOT / "audits" / "RAW_SOURCE_PROBE_AUDIT.json"
OUTPUT_TRAIN = ROOT / "manifests" / "FINAL_TRAIN_CONTENT_ORDER.jsonl"
OUTPUT_VAL = ROOT / "manifests" / "FINAL_VAL_CONTENT_ORDER.jsonl"
OUTPUT_AUDIT = ROOT / "audits" / "FINAL_CONTENT_ORDER_AUDIT.json"
SCHEMA = "PROLONG_V3_FINAL_CONTENT_ORDER_V1"
EXPECTED_ANCHOR_SHA = "017a74186b419b2c3f566f568e6c2d120723e3ddafb69024138eb6b321c90ae8"


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


def canonical_line(row):
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def main():
    gates = {
        "extension_selection": json.loads(SELECTION_AUDIT.read_text(encoding="utf-8")).get("status") == "PASS",
        "stream_receipt_replay": json.loads(REPLAY_AUDIT.read_text(encoding="utf-8")).get("status") == "PASS",
        "raw_source_probe": json.loads(PROBE_AUDIT.read_text(encoding="utf-8")).get("status") == "PASS",
        "anchor_sha": sha256_file(ANCHOR) == EXPECTED_ANCHOR_SHA,
    }
    failed = [key for key, passed in gates.items() if not passed]
    if failed:
        raise SystemExit("final content order gates failed: " + ", ".join(failed))
    quotas = json.loads(QUOTAS.read_text(encoding="utf-8"))
    streams = {}
    temporary = {
        "train": OUTPUT_TRAIN.with_name(OUTPUT_TRAIN.name + ".partial"),
        "validation": OUTPUT_VAL.with_name(OUTPUT_VAL.name + ".partial"),
    }
    for split, path in temporary.items():
        streams[split] = path.open("w", encoding="utf-8", newline="\n")
    split_order = Counter()
    logical = Counter()
    identities = {"train": set(), "validation": set()}
    hashes = {"train": set(), "validation": set()}
    component_n = Counter()
    try:
        with ANCHOR.open("r", encoding="utf-8") as anchor:
            for component_order, line in enumerate(anchor):
                original = json.loads(line)
                split = original["split"]
                document_tokens = int(original["qwen_token_count"])
                eos = int(original["eos_contribution"])
                row = {
                    "final_content_schema": SCHEMA,
                    "split_content_order": split_order[split],
                    "component": "historical_300m_anchor",
                    "component_order": component_order,
                    "split": split,
                    "source": original["source"],
                    "canonical_raw_identity": original["canonical_raw_identity"],
                    "canonical_shard_id": original["canonical_shard_id"],
                    "canonical_row_id": original["canonical_row_id"],
                    "decoded_content_sha256": original["decoded_content_sha256"],
                    "selected_document_token_n": document_tokens,
                    "eos_contribution": eos,
                    "selected_logical_token_n": document_tokens + eos,
                    "terminal_trim": bool(original["terminal_trim"]),
                    "authority_record": original,
                }
                streams[split].write(canonical_line(row))
                split_order[split] += 1
                logical[(row["source"], split)] += row["selected_logical_token_n"]
                identities[split].add(row["canonical_raw_identity"])
                hashes[split].add(row["decoded_content_sha256"])
                component_n[row["component"]] += 1
        for split, path in (("validation", EXT_VAL), ("train", EXT_TRAIN)):
            with path.open("r", encoding="utf-8") as extension:
                for component_order, line in enumerate(extension):
                    original = json.loads(line)
                    if original["split"] != split:
                        raise ValueError("extension split mismatch")
                    row = {
                        "final_content_schema": SCHEMA,
                        "split_content_order": split_order[split],
                        "component": "v3_streaming_700m_extension",
                        "component_order": component_order,
                        "split": split,
                        "source": original["source"],
                        "canonical_raw_identity": original["canonical_raw_identity"],
                        "canonical_shard_id": original["canonical_shard_id"],
                        "canonical_row_id": original["canonical_row_id"],
                        "decoded_content_sha256": original["decoded_content_sha256"],
                        "selected_document_token_n": int(original["selected_document_token_n"]),
                        "eos_contribution": int(original["eos_contribution"]),
                        "selected_logical_token_n": int(original["selected_logical_token_n"]),
                        "terminal_trim": bool(original["terminal_trim"]),
                        "authority_record": original,
                    }
                    if row["selected_document_token_n"] + row["eos_contribution"] != row["selected_logical_token_n"]:
                        raise ValueError("extension logical accounting mismatch")
                    streams[split].write(canonical_line(row))
                    split_order[split] += 1
                    logical[(row["source"], split)] += row["selected_logical_token_n"]
                    identities[split].add(row["canonical_raw_identity"])
                    hashes[split].add(row["decoded_content_sha256"])
                    component_n[row["component"]] += 1
    finally:
        for stream in streams.values():
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
    expected = {
        (source, split): int(quotas["sources"][source]["final_validation" if split == "validation" else "final_train"])
        for source in quotas["sources"] for split in ("train", "validation")
    }
    actual = dict(logical)
    checks = {
        "source_split_quota_parity": actual == expected,
        "train_logical_tokens": sum(v for (s, p), v in actual.items() if p == "train") == 990_000_000,
        "validation_logical_tokens": sum(v for (s, p), v in actual.items() if p == "validation") == 10_000_000,
        "train_val_identity_overlap_zero": not (identities["train"] & identities["validation"]),
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise ValueError("final content order checks failed: " + ", ".join(failed))
    os.replace(str(temporary["train"]), str(OUTPUT_TRAIN))
    os.replace(str(temporary["validation"]), str(OUTPUT_VAL))
    audit = {
        "audit_id": "PROLONG_V3_FINAL_CONTENT_ORDER_V1",
        "status": "PASS",
        "gates": gates,
        "checks": checks,
        "historical_anchor_sha256": sha256_file(ANCHOR),
        "extension_train_sha256": sha256_file(EXT_TRAIN),
        "extension_val_sha256": sha256_file(EXT_VAL),
        "final_train_content_order_sha256": sha256_file(OUTPUT_TRAIN),
        "final_val_content_order_sha256": sha256_file(OUTPUT_VAL),
        "final_train_content_record_n": split_order["train"],
        "final_validation_content_record_n": split_order["validation"],
        "component_record_n": dict(component_n),
        "train_logical_tokens": 990_000_000,
        "validation_logical_tokens": 10_000_000,
        "total_logical_tokens": 1_000_000_000,
        "per_source_split_tokens": {source: {split: logical[(source, split)] for split in ("train", "validation")} for source in quotas["sources"]},
        "final_train_val_identity_overlap_n": len(identities["train"] & identities["validation"]),
        "final_train_val_content_overlap_n": len(hashes["train"] & hashes["validation"]),
        "content_overlap_policy": "reported separately; identity overlap is the hard exclusion gate",
        "canonical_order_rule": "historical split rows in exact anchor order, then V3 extension rows in frozen streaming selection order",
    }
    atomic_json(OUTPUT_AUDIT, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
