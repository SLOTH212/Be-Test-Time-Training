"""Independently verify source aggregate indices against proc-level indices."""

import copy
import hashlib
import json
import os
from pathlib import Path


V1 = Path(os.path.expandvars("${DATA_ROOT}/prolong_1b_32k_qwen3_v1"))
ROOT = Path(os.path.expandvars("${DATA_ROOT}/prolong_1b_32k_qwen3_v3_streaming"))
REVISION = "1447128c751f0fbc7d2082b2a113a938426a9bba"
INDEX_ROOT = (
    V1
    / "source_cache"
    / "hf"
    / "datasets--princeton-nlp--prolong-data-64K"
    / "snapshots"
    / REVISION
)
SOURCES = [
    "arxiv",
    "book-65536",
    "dolmawiki",
    "fineweb-2023-50",
    "fineweb-edu",
    "openwebmath",
    "stackexchange",
    "textbooks",
    "thestackv1_concat_by_repo-65536",
    "tuluv2",
]
OUTPUT = ROOT / "audits" / "INDEX_AGGREGATION_PARITY_AUDIT.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".partial")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def normalized(entry, basename):
    value = copy.deepcopy(entry)
    value["raw_data"]["basename"] = basename.replace("\\", "/")
    return value


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main():
    per_source = {}
    mismatch_examples = []
    total_aggregate_shards = 0
    total_proc_shards = 0
    total_proc_indices = 0

    for source in SOURCES:
        source_root = INDEX_ROOT / source
        aggregate_path = source_root / "index.json"
        if not aggregate_path.is_file():
            raise FileNotFoundError(aggregate_path)
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        aggregate_rows = {}
        duplicate_aggregate = []
        for entry in aggregate["shards"]:
            key = entry["raw_data"]["basename"].replace("\\", "/")
            if key in aggregate_rows:
                duplicate_aggregate.append(key)
            aggregate_rows[key] = normalized(entry, key)

        proc_rows = {}
        duplicate_proc = []
        proc_paths = sorted(source_root.glob("proc*/index.json"))
        proc_hash_framing = hashlib.sha256()
        for proc_path in proc_paths:
            relative = proc_path.relative_to(INDEX_ROOT).as_posix()
            digest = sha256_file(proc_path)
            proc_hash_framing.update(relative.encode("utf-8"))
            proc_hash_framing.update(b"\0")
            proc_hash_framing.update(digest.encode("ascii"))
            proc_hash_framing.update(b"\n")
            proc = json.loads(proc_path.read_text(encoding="utf-8"))
            if proc.get("version") != aggregate.get("version"):
                mismatch_examples.append(
                    {
                        "source": source,
                        "kind": "index_version",
                        "proc": relative,
                        "aggregate_version": aggregate.get("version"),
                        "proc_version": proc.get("version"),
                    }
                )
            for entry in proc["shards"]:
                key = "%s/%s" % (
                    proc_path.parent.name,
                    entry["raw_data"]["basename"],
                )
                key = key.replace("\\", "/")
                if key in proc_rows:
                    duplicate_proc.append(key)
                proc_rows[key] = normalized(entry, key)

        missing_from_proc = sorted(set(aggregate_rows) - set(proc_rows))
        missing_from_aggregate = sorted(set(proc_rows) - set(aggregate_rows))
        metadata_mismatches = []
        for key in sorted(set(aggregate_rows) & set(proc_rows)):
            if canonical(aggregate_rows[key]) != canonical(proc_rows[key]):
                metadata_mismatches.append(key)
                if len(mismatch_examples) < 20:
                    mismatch_examples.append(
                        {
                            "source": source,
                            "kind": "shard_metadata",
                            "shard": key,
                            "aggregate": aggregate_rows[key],
                            "proc": proc_rows[key],
                        }
                    )
        for kind, values in (
            ("missing_from_proc", missing_from_proc),
            ("missing_from_aggregate", missing_from_aggregate),
            ("duplicate_aggregate", duplicate_aggregate),
            ("duplicate_proc", duplicate_proc),
        ):
            for key in values:
                if len(mismatch_examples) < 20:
                    mismatch_examples.append(
                        {"source": source, "kind": kind, "shard": key}
                    )

        source_pass = not (
            missing_from_proc
            or missing_from_aggregate
            or metadata_mismatches
            or duplicate_aggregate
            or duplicate_proc
        )
        per_source[source] = {
            "status": "PASS" if source_pass else "FAIL",
            "aggregate_index_path": str(aggregate_path),
            "aggregate_index_sha256": sha256_file(aggregate_path),
            "aggregate_shard_n": len(aggregate_rows),
            "proc_index_n": len(proc_paths),
            "proc_index_hash_framing": "SHA256(relative UTF-8 path || NUL || lowercase file SHA256 ASCII || LF), proc path order",
            "proc_indices_sha256": proc_hash_framing.hexdigest(),
            "proc_shard_n": len(proc_rows),
            "missing_from_proc_n": len(missing_from_proc),
            "missing_from_aggregate_n": len(missing_from_aggregate),
            "metadata_mismatch_n": len(metadata_mismatches),
            "duplicate_aggregate_n": len(duplicate_aggregate),
            "duplicate_proc_n": len(duplicate_proc),
        }
        total_aggregate_shards += len(aggregate_rows)
        total_proc_shards += len(proc_rows)
        total_proc_indices += len(proc_paths)

    all_index_paths = list(INDEX_ROOT.rglob("index.json"))
    status = (
        "PASS"
        if len(all_index_paths) == 650
        and total_proc_indices == 640
        and total_aggregate_shards == 2432
        and total_proc_shards == 2432
        and all(value["status"] == "PASS" for value in per_source.values())
        else "FAIL"
    )
    audit = {
        "audit_id": "INDEX_AGGREGATION_PARITY_V3_STREAMING_V1",
        "status": status,
        "upstream_revision": REVISION,
        "source_n": len(SOURCES),
        "all_index_json_n": len(all_index_paths),
        "source_aggregate_index_n": len(SOURCES),
        "proc_level_index_n": total_proc_indices,
        "aggregate_shard_n": total_aggregate_shards,
        "proc_shard_n": total_proc_shards,
        "per_source": per_source,
        "mismatch_example_n": len(mismatch_examples),
        "mismatch_examples": mismatch_examples,
        "primary_shard_inventory_builder_imported": False,
    }
    atomic_json(OUTPUT, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
