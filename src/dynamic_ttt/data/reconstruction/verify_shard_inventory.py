"""Independent byte-for-byte verifier for the V3 shard inventory.

Unlike the primary builder, this verifier enumerates shards from each source's
aggregate index.json.  It imports no primary implementation.
"""

import hashlib
import json
import os
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
NAMESPACES = (
    ("base", "PROLONG_1B_32K_QWEN3_V3_STREAMING_SHARDS_V1"),
    ("validation", "PROLONG_1B_32K_QWEN3_V3_VAL_SHARDS_V1"),
    ("train", "PROLONG_1B_32K_QWEN3_V3_TRAIN_SHARDS_V1"),
)
SCHEMA = "PROLONG_UPSTREAM_SHARD_INVENTORY_V3_STREAMING_V1"
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
INVENTORY = ROOT / "manifests" / "UPSTREAM_SHARD_INVENTORY.jsonl"
PRIMARY_AUDIT = ROOT / "audits" / "UPSTREAM_SHARD_INVENTORY_AUDIT.json"
SCOPE_AUDIT = ROOT / "audits" / "UPSTREAM_REPOSITORY_FILE_SCOPE_AUDIT.json"
OUTPUT = ROOT / "audits" / "UPSTREAM_SHARD_INVENTORY_INDEPENDENT_AUDIT_V2.json"
EXPECTED_PRIMARY_INVENTORY_SHA256 = (
    "58ba8027e314652a9eef291bee37e5ec0bed0dd672d5bb20b11962c38de86bea"
)


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


def jsonl_bytes(rows):
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    ).encode("utf-8")


def safe_source_name(source):
    return "".join(character.upper() if character.isalnum() else "_" for character in source)


def load_authority():
    authority = {}
    row_n = 0
    repository_mds_n = 0
    excluded_dclm_mds_n = 0
    unexpected_mds = []
    with FILE_INVENTORY.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            row_n += 1
            if row["path"].endswith(".mds"):
                repository_mds_n += 1
                repo_path = row["path"].replace("\\", "/")
                source = repo_path.split("/", 1)[0]
                if source == "dclm-baseline":
                    excluded_dclm_mds_n += 1
                    continue
                if source not in SOURCES:
                    unexpected_mds.append(repo_path)
                    continue
                authority[repo_path] = {
                    "size": int(row["size"]),
                    "lfs_sha256": row["lfs"]["sha256"] if row["lfs"] else None,
                    "git_blob_sha1": row["blob_id"],
                }
    if (
        row_n != 3608
        or repository_mds_n != 2880
        or len(authority) != 2432
        or excluded_dclm_mds_n != 448
        or unexpected_mds
    ):
        raise ValueError("frozen file authority cardinality mismatch")
    return authority, {
        "repository_file_n": row_n,
        "repository_mds_total": repository_mds_n,
        "v3_target_mds_total": len(authority),
        "excluded_dclm_baseline_mds": excluded_dclm_mds_n,
        "unexpected_mds_n": len(unexpected_mds),
    }


def aggregate_shards(authority):
    rows = []
    for source in SOURCES:
        aggregate = json.loads(
            (INDEX_ROOT / source / "index.json").read_text(encoding="utf-8")
        )
        for entry in aggregate["shards"]:
            relative = entry["raw_data"]["basename"].replace("\\", "/")
            proc, basename = relative.split("/", 1)
            repo_path = "%s/%s" % (source, relative)
            frozen = authority[repo_path]
            if int(entry["raw_data"]["bytes"]) != frozen["size"]:
                raise ValueError("aggregate index/authority size mismatch: %s" % repo_path)
            rows.append(
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
    if len(rows) != 2432 or len({row["canonical_shard_id"] for row in rows}) != 2432:
        raise ValueError("aggregate inventory cardinality/uniqueness mismatch")
    return rows


def cache_map(shards):
    by_id = {row["canonical_shard_id"]: row for row in shards}
    result = {}
    receipt = json.loads(HISTORICAL_DOWNLOAD.read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS":
        raise ValueError("historical download receipt is not PASS")
    for item in receipt["files"]:
        repo_path = item["repo_path"]
        path = Path(item["local_path"])
        if repo_path not in by_id or not path.is_file():
            continue
        shard = by_id[repo_path]
        if path.stat().st_size != shard["expected_byte_size"]:
            raise ValueError("cached size mismatch: %s" % repo_path)
        if shard["expected_lfs_sha256"]:
            hash_kind = "lfs_sha256"
            digest = sha256_file(path)
            expected = shard["expected_lfs_sha256"]
        else:
            hash_kind = "git_blob_sha1"
            digest = git_blob_id(path)
            expected = shard["expected_git_blob_sha1"]
        if digest != expected:
            raise ValueError("cached authority hash mismatch: %s" % repo_path)
        result[repo_path] = {
            "local_cached_path": str(path),
            "local_cache_origin": "V2_HISTORICAL_SHARD_DOWNLOAD",
            "local_cache_hash_kind": hash_kind,
            "local_cache_verified_hash": digest,
        }
    if V2_CURRENT_MDS.is_file():
        digest = sha256_file(V2_CURRENT_MDS)
        size = V2_CURRENT_MDS.stat().st_size
        matches = [
            row
            for row in shards
            if row["expected_byte_size"] == size
            and row["expected_lfs_sha256"] == digest
        ]
        if len(matches) != 1:
            raise ValueError("stopped V2 current.mds authority match is not unique")
        repo_path = matches[0]["canonical_shard_id"]
        result[repo_path] = {
            "local_cached_path": str(V2_CURRENT_MDS),
            "local_cache_origin": "V2_STOPPED_INVENTORY_CURRENT_MDS",
            "local_cache_hash_kind": "lfs_sha256",
            "local_cache_verified_hash": digest,
        }
    return result


def order_rows(source, shards):
    rows = []
    for kind, namespace in NAMESPACES:
        ranked = []
        for shard in shards:
            shard_id = shard["canonical_shard_id"]
            key = hashlib.sha256((namespace + source + shard_id).encode("utf-8")).hexdigest()
            ranked.append((key, shard_id))
        ranked.sort()
        for rank, (key, shard_id) in enumerate(ranked):
            rows.append(
                {
                    "order_schema": "PROLONG_V3_DETERMINISTIC_SHARD_ORDER_V1",
                    "source": source,
                    "order_kind": kind,
                    "namespace": namespace,
                    "framing": "UTF-8 direct concatenation with no delimiter: namespace || source || canonical_shard_id",
                    "rank": rank,
                    "shard_order_key": key,
                    "canonical_shard_id": shard_id,
                }
            )
    return rows


def main():
    authority, population = load_authority()
    shards = aggregate_shards(authority)
    cache = cache_map(shards)
    expected_inventory = []
    for ordinal, shard in enumerate(shards):
        row = {
            "inventory_schema": SCHEMA,
            "upstream_repo": REPO,
            "upstream_revision": REVISION,
            "inventory_ordinal": ordinal,
            **shard,
            "local_cached_path": None,
            "local_cache_origin": None,
            "local_cache_hash_kind": None,
            "local_cache_verified_hash": None,
        }
        row.update(cache.get(shard["canonical_shard_id"], {}))
        expected_inventory.append(row)
    inventory_expected_bytes = jsonl_bytes(expected_inventory)
    inventory_actual_bytes = INVENTORY.read_bytes()
    inventory_bytes_match = inventory_actual_bytes == inventory_expected_bytes
    primary_inventory_sha256 = sha256_file(INVENTORY)
    actual_rows = [json.loads(line) for line in inventory_actual_bytes.decode("utf-8").splitlines()]
    expected_by_id = {row["canonical_shard_id"]: row for row in expected_inventory}
    actual_by_id = {row["canonical_shard_id"]: row for row in actual_rows}
    missing_target_shards = sorted(set(expected_by_id) - set(actual_by_id))
    extra_target_shards = sorted(set(actual_by_id) - set(expected_by_id))
    metadata_mismatches = []
    for shard_id in sorted(set(expected_by_id) & set(actual_by_id)):
        if expected_by_id[shard_id] != actual_by_id[shard_id]:
            metadata_mismatches.append(shard_id)

    order_results = {}
    all_order_match = True
    for source in SOURCES:
        source_shards = [row for row in shards if row["canonical_source"] == source]
        path = ROOT / "manifests" / ("SHARD_ORDER_%s.jsonl" % safe_source_name(source))
        expected = jsonl_bytes(order_rows(source, source_shards))
        actual = path.read_bytes()
        match = actual == expected
        all_order_match = all_order_match and match
        order_results[source] = {
            "path": str(path),
            "byte_parity": match,
            "row_n": 3 * len(source_shards),
            "sha256": sha256_file(path),
        }

    primary = json.loads(PRIMARY_AUDIT.read_text(encoding="utf-8"))
    scope = json.loads(SCOPE_AUDIT.read_text(encoding="utf-8"))
    status = (
        "PASS"
        if inventory_bytes_match
        and not missing_target_shards
        and not extra_target_shards
        and not metadata_mismatches
        and all_order_match
        and scope.get("status") == "PASS"
        and scope.get("repository_file_n") == 3608
        and scope.get("repository_mds_total") == 2880
        and scope.get("v3_target_mds_total") == 2432
        and scope.get("excluded_dclm_baseline_mds") == 448
        and primary.get("status") == "PASS"
        and primary.get("inventory_sha256") == primary_inventory_sha256
        and primary_inventory_sha256 == EXPECTED_PRIMARY_INVENTORY_SHA256
        and primary.get("mds_shard_n") == 2432
        and primary.get("FULL_POOL_DOCUMENT_INVENTORY") is False
        and primary.get("FULL_POOL_DOCUMENT_TOKENIZATION") is False
        and primary.get("GLOBAL_DOCUMENT_SHA_RANKING") is False
        else "FAIL"
    )
    audit = {
        "audit_id": "UPSTREAM_SHARD_INVENTORY_INDEPENDENT_V3_STREAMING_V2",
        "status": status,
        "INDEPENDENT_SHARD_GATE": status,
        "primary_builder_imported": False,
        "independent_enumeration_source": "10 source-level aggregate index.json files",
        "target_source_n": len(SOURCES),
        **population,
        "inventory_row_n": len(expected_inventory),
        "inventory_byte_parity": inventory_bytes_match,
        "inventory_sha256": primary_inventory_sha256,
        "required_primary_inventory_sha256": EXPECTED_PRIMARY_INVENTORY_SHA256,
        "missing_target_shard_n": len(missing_target_shards),
        "extra_target_shard_n": len(extra_target_shards),
        "metadata_mismatch_n": len(metadata_mismatches),
        "missing_target_shards": missing_target_shards,
        "extra_target_shards": extra_target_shards,
        "metadata_mismatch_shards": metadata_mismatches,
        "cached_shard_n": len(cache),
        "order_manifest_byte_parity": all_order_match,
        "order_manifests": order_results,
        "primary_audit_path": str(PRIMARY_AUDIT),
        "primary_audit_sha256": sha256_file(PRIMARY_AUDIT),
        "primary_audit_status": primary.get("status"),
        "repository_scope_audit_path": str(SCOPE_AUDIT),
        "repository_scope_audit_sha256": sha256_file(SCOPE_AUDIT),
        "repository_scope_audit_status": scope.get("status"),
        "FULL_POOL_DOCUMENT_INVENTORY": False,
        "FULL_POOL_DOCUMENT_TOKENIZATION": False,
        "GLOBAL_DOCUMENT_SHA_RANKING": False,
    }
    atomic_json(OUTPUT, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
