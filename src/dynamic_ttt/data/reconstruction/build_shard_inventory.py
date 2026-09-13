"""Build the cheap V3 shard inventory and deterministic shard orders.

This builder never opens an MDS shard for row/document processing.  It reads
only the frozen repository file inventory and the immutable revision's MDS
index.json files.  Existing local shard cache entries are hashed solely to
prove safe reuse.
"""

import hashlib
import json
import os
from collections import Counter
from pathlib import Path


V1 = Path(os.path.expandvars("${DATA_ROOT}/prolong_1b_32k_qwen3_v1"))
V2 = Path(os.path.expandvars("${DATA_ROOT}/prolong_1b_32k_qwen3_v2"))
ROOT = Path(os.path.expandvars("${DATA_ROOT}/prolong_1b_32k_qwen3_v3_streaming"))
REPO = "princeton-nlp/prolong-data-64K"
REVISION = "1447128c751f0fbc7d2082b2a113a938426a9bba"
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
BASE_NAMESPACE = "PROLONG_1B_32K_QWEN3_V3_STREAMING_SHARDS_V1"
VAL_NAMESPACE = "PROLONG_1B_32K_QWEN3_V3_VAL_SHARDS_V1"
TRAIN_NAMESPACE = "PROLONG_1B_32K_QWEN3_V3_TRAIN_SHARDS_V1"
FILE_INVENTORY = V1 / "manifests" / "PROLONG_UPSTREAM_FILE_INVENTORY.jsonl"
INDEX_ROOT = (
    V1
    / "source_cache"
    / "hf"
    / "datasets--princeton-nlp--prolong-data-64K"
    / "snapshots"
    / REVISION
)
HISTORICAL_DOWNLOAD = V2 / "audits" / "HISTORICAL_SHARD_DOWNLOAD.json"
V2_CURRENT_MDS = V2 / "tmp" / "upstream_document_inventory_work_v3" / "current.mds"
OUTPUT = ROOT / "manifests" / "UPSTREAM_SHARD_INVENTORY.jsonl"
AUDIT = ROOT / "audits" / "UPSTREAM_SHARD_INVENTORY_AUDIT.json"
CONFIG = ROOT / "authority" / "stage1_1b_32k_qwen3_v3_streaming.yaml"
INDEX_PARITY_AUDIT = ROOT / "audits" / "INDEX_AGGREGATION_PARITY_AUDIT.json"
SCHEMA = "PROLONG_UPSTREAM_SHARD_INVENTORY_V3_STREAMING_V1"


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


def atomic_text(path, content):
    temporary = path.with_name(path.name + ".partial")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def jsonl_bytes(rows):
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    )


def load_file_authority():
    if not FILE_INVENTORY.is_file():
        raise FileNotFoundError(FILE_INVENTORY)
    authority = {}
    repository_file_n = 0
    with FILE_INVENTORY.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            repository_file_n += 1
            if row["path"].endswith(".mds"):
                authority[row["path"]] = {
                    "size": int(row["size"]),
                    "lfs_sha256": row["lfs"]["sha256"] if row["lfs"] else None,
                    "git_blob_sha1": row["blob_id"],
                }
    if repository_file_n != 3608:
        raise ValueError(
            "frozen repository file count disagreement: %d" % repository_file_n
        )
    return authority, repository_file_n


def enumerate_shards(authority):
    shards = []
    index_count = 0
    for source in SOURCES:
        source_root = INDEX_ROOT / source
        index_paths = sorted(source_root.glob("proc*/index.json"))
        if not index_paths:
            raise ValueError("no index files for source: %s" % source)
        for index_path in index_paths:
            index_count += 1
            index = json.loads(index_path.read_text(encoding="utf-8"))
            proc = index_path.parent.name
            for entry in index["shards"]:
                basename = entry["raw_data"]["basename"]
                repo_path = "%s/%s/%s" % (source, proc, basename)
                if repo_path not in authority:
                    raise ValueError("shard missing from frozen authority: %s" % repo_path)
                frozen = authority[repo_path]
                index_size = int(entry["raw_data"]["bytes"])
                if index_size != frozen["size"]:
                    raise ValueError("index/authority size mismatch: %s" % repo_path)
                shards.append(
                    {
                        "canonical_source": source,
                        "canonical_shard_id": repo_path,
                        "upstream_repo_path": repo_path,
                        "expected_byte_size": frozen["size"],
                        "expected_lfs_sha256": frozen["lfs_sha256"],
                        "expected_git_blob_sha1": frozen["git_blob_sha1"],
                        "upstream_native_row_n": int(entry["samples"]),
                        "index_repo_path": "%s/%s/index.json" % (source, proc),
                    }
                )
    if len(shards) != 2432:
        raise ValueError("expected 2432 MDS shards, found %d" % len(shards))
    if index_count != 640:
        raise ValueError("expected 640 proc-level MDS indices, found %d" % index_count)
    all_index_count = len(list(INDEX_ROOT.rglob("index.json")))
    if all_index_count != 650:
        raise ValueError("expected 650 total index.json files, found %d" % all_index_count)
    return shards, index_count, all_index_count


def require_index_parity():
    if not INDEX_PARITY_AUDIT.is_file():
        raise FileNotFoundError(INDEX_PARITY_AUDIT)
    audit = json.loads(INDEX_PARITY_AUDIT.read_text(encoding="utf-8"))
    required = {
        "status": "PASS",
        "upstream_revision": REVISION,
        "source_n": 10,
        "all_index_json_n": 650,
        "source_aggregate_index_n": 10,
        "proc_level_index_n": 640,
        "aggregate_shard_n": 2432,
        "proc_shard_n": 2432,
        "primary_shard_inventory_builder_imported": False,
    }
    for key, expected in required.items():
        if audit.get(key) != expected:
            raise ValueError(
                "index aggregation parity gate mismatch for %s: %r != %r"
                % (key, audit.get(key), expected)
            )
    return sha256_file(INDEX_PARITY_AUDIT)


def verify_cached(path, shard):
    if path.stat().st_size != shard["expected_byte_size"]:
        raise ValueError("cached shard size mismatch: %s" % shard["canonical_shard_id"])
    if shard["expected_lfs_sha256"]:
        actual = sha256_file(path)
        if actual != shard["expected_lfs_sha256"]:
            raise ValueError("cached shard SHA256 mismatch: %s" % shard["canonical_shard_id"])
        return "lfs_sha256", actual
    actual = git_blob_id(path)
    if actual != shard["expected_git_blob_sha1"]:
        raise ValueError("cached shard Git blob mismatch: %s" % shard["canonical_shard_id"])
    return "git_blob_sha1", actual


def discover_cache(shards):
    by_id = {row["canonical_shard_id"]: row for row in shards}
    cache = {}
    if HISTORICAL_DOWNLOAD.is_file():
        receipt = json.loads(HISTORICAL_DOWNLOAD.read_text(encoding="utf-8"))
        if receipt.get("status") != "PASS":
            raise ValueError("historical shard cache receipt is not PASS")
        for item in receipt["files"]:
            repo_path = item["repo_path"]
            path = Path(item["local_path"])
            if repo_path not in by_id or not path.is_file():
                continue
            hash_kind, actual_hash = verify_cached(path, by_id[repo_path])
            cache[repo_path] = {
                "local_cached_path": str(path),
                "local_cache_origin": "V2_HISTORICAL_SHARD_DOWNLOAD",
                "local_cache_hash_kind": hash_kind,
                "local_cache_verified_hash": actual_hash,
            }

    if V2_CURRENT_MDS.is_file():
        size = V2_CURRENT_MDS.stat().st_size
        digest = sha256_file(V2_CURRENT_MDS)
        matches = [
            row
            for row in shards
            if row["expected_byte_size"] == size
            and row["expected_lfs_sha256"] == digest
        ]
        if len(matches) != 1:
            raise ValueError(
                "V2 current.mds did not map uniquely to frozen LFS authority: %d matches"
                % len(matches)
            )
        repo_path = matches[0]["canonical_shard_id"]
        cache[repo_path] = {
            "local_cached_path": str(V2_CURRENT_MDS),
            "local_cache_origin": "V2_STOPPED_INVENTORY_CURRENT_MDS",
            "local_cache_hash_kind": "lfs_sha256",
            "local_cache_verified_hash": digest,
        }
    return cache


def order_key(namespace, source, canonical_shard_id):
    framing = (namespace + source + canonical_shard_id).encode("utf-8")
    return hashlib.sha256(framing).hexdigest()


def build_order_rows(source, source_shards):
    result = []
    for order_kind, namespace in (
        ("base", BASE_NAMESPACE),
        ("validation", VAL_NAMESPACE),
        ("train", TRAIN_NAMESPACE),
    ):
        ranked = sorted(
            (
                order_key(namespace, source, row["canonical_shard_id"]),
                row["canonical_shard_id"],
            )
            for row in source_shards
        )
        for rank, (key, shard_id) in enumerate(ranked):
            result.append(
                {
                    "order_schema": "PROLONG_V3_DETERMINISTIC_SHARD_ORDER_V1",
                    "source": source,
                    "order_kind": order_kind,
                    "namespace": namespace,
                    "framing": "UTF-8 direct concatenation with no delimiter: namespace || source || canonical_shard_id",
                    "rank": rank,
                    "shard_order_key": key,
                    "canonical_shard_id": shard_id,
                }
            )
    return result


def safe_source_name(source):
    return "".join(character.upper() if character.isalnum() else "_" for character in source)


def main():
    index_parity_audit_sha256 = require_index_parity()
    authority, repository_file_n = load_file_authority()
    shards, proc_index_count, all_index_count = enumerate_shards(authority)
    cache = discover_cache(shards)
    counts = Counter(row["canonical_source"] for row in shards)

    inventory_rows = []
    for ordinal, row in enumerate(shards):
        value = {
            "inventory_schema": SCHEMA,
            "upstream_repo": REPO,
            "upstream_revision": REVISION,
            "inventory_ordinal": ordinal,
            **row,
            "local_cached_path": None,
            "local_cache_origin": None,
            "local_cache_hash_kind": None,
            "local_cache_verified_hash": None,
        }
        value.update(cache.get(row["canonical_shard_id"], {}))
        inventory_rows.append(value)
    atomic_text(OUTPUT, jsonl_bytes(inventory_rows))

    order_manifests = {}
    for source in SOURCES:
        source_shards = [
            row for row in shards if row["canonical_source"] == source
        ]
        order_path = (
            ROOT / "manifests" / ("SHARD_ORDER_%s.jsonl" % safe_source_name(source))
        )
        order_rows = build_order_rows(source, source_shards)
        atomic_text(order_path, jsonl_bytes(order_rows))
        order_manifests[source] = {
            "path": str(order_path),
            "sha256": sha256_file(order_path),
            "row_n": len(order_rows),
            "shard_n": len(source_shards),
        }

    audit = {
        "audit_id": "UPSTREAM_SHARD_INVENTORY_V3_STREAMING_V1",
        "status": "PASS",
        "dataset_id": "PROLONG_DERIVED_1B_32K_QWEN3_STAGE1_V3_STREAMING",
        "upstream_repo": REPO,
        "upstream_revision": REVISION,
        "repository_file_inventory_path": str(FILE_INVENTORY),
        "repository_file_inventory_sha256": sha256_file(FILE_INVENTORY),
        "repository_file_n": repository_file_n,
        "all_index_json_n": all_index_count,
        "source_aggregate_index_n": len(SOURCES),
        "proc_level_mds_index_n": proc_index_count,
        "mds_shard_n": len(shards),
        "per_source_shard_n": {source: counts[source] for source in SOURCES},
        "inventory_path": str(OUTPUT),
        "inventory_sha256": sha256_file(OUTPUT),
        "inventory_row_n": len(inventory_rows),
        "cached_shard_n": len(cache),
        "cached_shards": sorted(cache),
        "order_key_framing": "UTF-8 direct concatenation with no delimiter: namespace || source || canonical_shard_id",
        "base_namespace": BASE_NAMESPACE,
        "validation_namespace": VAL_NAMESPACE,
        "train_namespace": TRAIN_NAMESPACE,
        "order_manifests": order_manifests,
        "config_path": str(CONFIG),
        "config_sha256": sha256_file(CONFIG),
        "index_aggregation_parity_audit_path": str(INDEX_PARITY_AUDIT),
        "index_aggregation_parity_audit_sha256": index_parity_audit_sha256,
        "index_aggregation_parity_status": "PASS",
        "FULL_POOL_DOCUMENT_INVENTORY": False,
        "FULL_POOL_DOCUMENT_TOKENIZATION": False,
        "GLOBAL_DOCUMENT_SHA_RANKING": False,
        "mds_opened_for_row_processing_n": 0,
        "document_row_enumeration_n": 0,
        "document_tokenization_n": 0,
        "ON_DEMAND_SHARD_DOWNLOAD": True,
        "ONLINE_DEDUP_ONLY": True,
    }
    atomic_json(AUDIT, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
