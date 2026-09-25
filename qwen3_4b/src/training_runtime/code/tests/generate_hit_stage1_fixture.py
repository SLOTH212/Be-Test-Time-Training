#!/usr/bin/env python3
import os
"""Generate four deterministic, exact-32K Stage1 records with real Qwen tokenization."""
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
MODEL = Path(os.path.expandvars("${TTT_MODEL_ROOT}/Qwen3-4B-Base"))
OUTPUT = ROOT / "fixtures/stage1_32k_debug.jsonl"
MANIFEST = ROOT / "audits/hit_l3_l4_v1/SYNTHETIC_FIXTURE_MANIFEST.json"
CONTEXT = 32768
RECORDS = 4


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def exact_record(tokenizer, index):
    line = (
        f"HIT synthetic Stage1 record {index:02d}; deterministic gated NTP debug fixture. "
        "No formal corpus content is present.\n"
    )
    source = line * 20000
    encoded = tokenizer(source, add_special_tokens=False).input_ids
    if len(encoded) < CONTEXT:
        raise RuntimeError("SYNTHETIC_SOURCE_TOO_SHORT")
    target_ids = encoded[: CONTEXT - 1] + [int(tokenizer.eos_token_id)]
    text = tokenizer.decode(
        target_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    roundtrip = tokenizer(text, add_special_tokens=False).input_ids
    if roundtrip != target_ids or len(roundtrip) != CONTEXT:
        raise RuntimeError(
            f"TOKENIZER_ROUNDTRIP_FAILED record={index} "
            f"expected={CONTEXT} actual={len(roundtrip)}"
        )
    return {
        "sample_id": f"hit-synthetic-32k-{index:04d}",
        "text": text,
        "token_count": CONTEXT,
        "document_boundaries_qwen_json": json.dumps([[0, CONTEXT]]),
        "metadata": {
            "document_boundaries_qwen_json": json.dumps([[0, CONTEXT]]),
            "synthetic": True,
            "formal_data": False,
            "eos_token_id": int(tokenizer.eos_token_id),
        },
    }


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    if tokenizer.eos_token_id is None:
        raise RuntimeError("TOKENIZER_EOS_MISSING")
    rows = [exact_record(tokenizer, index) for index in range(RECORDS)]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "status": "PASS",
        "synthetic_data_only": True,
        "formal_stage1_dataset_used": False,
        "path": str(OUTPUT),
        "sha256": digest(OUTPUT),
        "record_count": len(rows),
        "record_ids": [row["sample_id"] for row in rows],
        "token_count_each": CONTEXT,
        "total_logical_tokens": CONTEXT * len(rows),
        "eos_token_id": int(tokenizer.eos_token_id),
        "document_boundaries_each": [[0, CONTEXT]],
        "schema_fields": sorted(rows[0]),
        "tokenizer_path": str(MODEL),
        "tokenizer_local_files_only": True,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
