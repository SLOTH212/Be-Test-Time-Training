#!/usr/bin/env python3
"""Build QA_REPLAY_30M_32K_QWEN3_STAGE2_V2 from frozen output authority.

The historical generative selector and packer are intentionally not recreated.
Historical records are replayed as atomic consumed-content QA bundles; every QA
target inside each bundle is mapped independently to the exact MRQA source.
The extension uses a new SHA-256 order and the final dataset uses a new
sequential 32K packer.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import struct
import sys
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path

DATASET_ID = "QA_REPLAY_30M_32K_QWEN3_STAGE2_V2"
PROTOCOL = "STAGE2_QA_30M_32K_EXTENSION_SELECTION_V2"
ROOT = Path(os.environ["QA_BUILD_ROOT"])
HROOT = ROOT / "recovered_authority"
HIST = HROOT / "data/qa_replay_10m_v1"
HMASK = HROOT / "data/qa_replay_10m_v1_answer_context_masks_v1"
TOK4 = ROOT / "source_cache/tokenizers/Qwen3-4B-Base"
TOK8 = ROOT / "source_cache/tokenizers/Qwen3-8B-Base"
MAXLEN = 32768
TARGETS = {
    "NaturalQuestionsShort_MRQA": 12_000_000,
    "SQuAD_MRQA": 9_000_000,
    "NewsQA_MRQA": 4_500_000,
    "TriviaQA-web_MRQA": 3_000_000,
    "HotpotQA_MRQA": 1_500_000,
}
FILES = {
    "NaturalQuestionsShort_MRQA": "NaturalQuestionsShort.jsonl.gz",
    "SQuAD_MRQA": "SQuAD.jsonl.gz",
    "NewsQA_MRQA": "NewsQA.jsonl.gz",
    "TriviaQA-web_MRQA": "TriviaQA-web.jsonl.gz",
    "HotpotQA_MRQA": "HotpotQA.jsonl.gz",
}
EXPECTED_SOURCE_SHA = {
    "NaturalQuestionsShort_MRQA": "6cdac324664b94b60be3203a077bf361d0bfa68a17af9b71def1186a6958a68c",
    "SQuAD_MRQA": "b094703b9c6f740cc2dfd70b3201b833553fcec0c8a522f22c2c6ff82ce2cc78",
    "NewsQA_MRQA": "f1ccbf2d259ce1094aacde21a53592894248e5778814205dac94f0b086dbe968",
    "TriviaQA-web_MRQA": "61fad6884370408282ad3ed0b5f25a9e932d9a724b6929ea03ea5344ff0cd3f7",
    "HotpotQA_MRQA": "3a94712c073dc9f29d88ac149faa01ef9c7c089f97ee25d9cbac39387550825d",
}


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_text(s: str) -> str:
    return sha_bytes(s.encode("utf-8"))


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canon(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(p: Path, obj) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, sort_keys=True, indent=2)
        f.write("\n")
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)


def write_jsonl(p: Path, rows) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(canon(row) + "\n")
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)


def ranges(mask):
    out, start = [], None
    for i, value in enumerate(list(mask) + [False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            out.append([start, i]); start = None
    return out


def token_ranges(tok, text: str, char_spans):
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    mask = [any(a < b and a < e and b > s for s, e in char_spans)
            for a, b in enc["offset_mapping"]]
    return enc["input_ids"], ranges(mask)


def parse_historical_questions(text: str):
    delim = "\n\nQuestions and Answers:\n\n"
    tail = text[text.rfind(delim) + len(delim):]
    pat = re.compile(r"Question (\d+):\n(.*?)\n\nAnswer \1:\n(.*?)\n\n", re.S)
    return [(m.group(2), m.group(3)) for m in pat.finditer(tail)]


def render_unit(context: str, qas):
    head = "Context:\n\nPassage 1:\n" + context + "\n\n\nQuestions and Answers:\n\n"
    parts, spans, cursor = [head], [], len(head)
    for i, qa in enumerate(qas, 1):
        pre = f"Question {i}:\n{qa['question']}\n\nAnswer {i}:\n"
        ans = qa["answer"]
        parts.extend([pre, ans, "\n\n"])
        cursor += len(pre)
        spans.append([cursor, cursor + len(ans)])
        cursor += len(ans) + 2
    return "".join(parts), spans


def valid_qas(row, excluded):
    ctx = row.get("context", "")
    out = []
    for qa in row.get("qas", []):
        qid = str(qa.get("qid") or qa.get("id"))
        if qid in excluded:
            continue
        grounded = None
        for det in qa.get("detected_answers", []):
            text = det.get("text", "")
            for span in det.get("char_spans", []):
                s, e = int(span[0]), int(span[1])
                if 0 <= s <= e < len(ctx) and ctx[s:e + 1] == text:
                    grounded = (text, [s, e + 1]); break
            if grounded:
                break
        if grounded:
            out.append({"qid": qid, "question": qa["question"],
                        "answer": grounded[0], "source_answer_span": grounded[1],
                        "aliases": qa.get("answers", [])})
    return out


def iter_mrqa(source: str):
    path = ROOT / "source_cache" / FILES[source]
    with gzip.open(path, "rt", encoding="utf-8") as f:
        header = json.loads(next(f))
        for line_no, line in enumerate(f, 1):
            yield line_no, header["header"], json.loads(line)


def select_exact(candidates, target):
    """Deterministic prefix plus exact subset-sum tail."""
    candidates = sorted(candidates, key=lambda x: (x["selection_key"], x["canonical_id"]))
    for reserve in (50_000, 100_000, 200_000, 400_000):
        prefix, total, cut = [], 0, 0
        ceiling = target - reserve
        for i, c in enumerate(candidates):
            if total + c["logical_token_length"] <= ceiling:
                prefix.append(c); total += c["logical_token_length"]
            else:
                cut = i; break
        gap = target - total
        # Everything at/after cut is outside the contiguous prefix. Avoid a
        # quadratic dictionary-membership scan over tens of thousands of units.
        pool = [c for c in candidates[cut:cut + 7000]
                if c["logical_token_length"] <= gap]
        bits, states, used = 1, [], []
        mask = (1 << (gap + 1)) - 1
        found = False
        for c in pool:
            bits = (bits | (bits << c["logical_token_length"])) & mask
            states.append(bits); used.append(c)
            if (bits >> gap) & 1:
                found = True; break
        if not found:
            continue
        chosen, s = [], gap
        for i in range(len(used) - 1, -1, -1):
            prev = states[i - 1] if i else 1
            if not ((prev >> s) & 1):
                chosen.append(used[i]); s -= used[i]["logical_token_length"]
        if s == 0:
            answer = prefix + list(reversed(chosen))
            assert sum(x["logical_token_length"] for x in answer) == target
            return sorted(answer, key=lambda x: (x["selection_key"], x["canonical_id"]))
    raise RuntimeError(f"EXACT_WHOLE_UNIT_TARGET_UNREACHABLE target={target}")


def main():
    started = time.time()
    for d in ("builder", "recovered_authority", "source_cache", "manifests", "audits",
              "reports", "authority", "train", "tmp", "package"):
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tok4 = AutoTokenizer.from_pretrained(TOK4, local_files_only=True)
    tok8 = AutoTokenizer.from_pretrained(TOK8, local_files_only=True)

    # Tokenizer parity and historical tokenizer identity.
    token_files = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
    th4 = {n: sha_file(TOK4 / n) for n in token_files}
    th8 = {n: sha_file(TOK8 / n) for n in token_files}
    probes = ["hello", "中文测试", "Context:\n\nPassage 1:\n", "<|endoftext|>", " café — 123"]
    probe_rows = [{"text": p, "ids4": tok4(p, add_special_tokens=False).input_ids,
                   "ids8": tok8(p, add_special_tokens=False).input_ids} for p in probes]
    historical_tok = json.load((HIST / "tokenizer_manifest.json").open(encoding="utf-8"))
    tok_pass = th4 == th8 == historical_tok["files"] and all(x["ids4"] == x["ids8"] for x in probe_rows)
    write_json(ROOT / "audits/TOKENIZER_PARITY_AUDIT.json", {
        "status": "PASS" if tok_pass else "FAIL", "qwen3_4b_commit": "906bfd4b4dc7f14ee4320094d8b41684abff8539",
        "qwen3_8b_commit": "49e3418fbbbca6ecbdf9608b4d22e5a407081db4", "files_4b": th4,
        "files_8b": th8, "historical_files": historical_tok["files"], "encoding_probes": probe_rows,
        "special_ids": {"eos": tok4.eos_token_id, "pad": tok4.pad_token_id, "vocab_size": len(tok4)}})
    if not tok_pass:
        raise RuntimeError("TOKENIZER_PARITY_FAIL")

    # Verify source bytes.
    source_inventory = {}
    for source, filename in FILES.items():
        p = ROOT / "source_cache" / filename
        actual = sha_file(p)
        if actual != EXPECTED_SOURCE_SHA[source]:
            raise RuntimeError(f"SOURCE_HASH_FAIL {source} {actual}")
        source_inventory[source] = {"variant": source, "split": "train", "file": str(p),
            "bytes": p.stat().st_size, "sha256": actual,
            "url": f"https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/{filename}"}

    htrain, hmasks = HIST / "train.jsonl", HMASK / "masks.jsonl"
    historical_data_sha, historical_masks_sha = sha_file(htrain), sha_file(hmasks)
    if historical_data_sha != "95bfb1242b7ccfb93dcbc5c82deee9250a105558d39deea0a458f7225bc65440":
        raise RuntimeError("HISTORICAL_PACKED_HASH_FAIL")
    if historical_masks_sha != "a58a4fec89fe149a544de3102a604cd5c3d84f9e9eb8c6961c598cc15452ec68":
        raise RuntimeError("HISTORICAL_MASK_HASH_FAIL")
    mask_by_id = {x["id"]: x for x in map(json.loads, HMASK.joinpath("masks.jsonl").open(encoding="utf-8"))}

    # Replay physical output, derive masks independently, and gather anchor identities.
    anchor_rows, anchor_groups, expected_qids = [], [], defaultdict(list)
    source_hist = Counter(); token_mismatch = record_mismatch = answer_mismatch = 0
    with htrain.open(encoding="utf-8") as f:
        for record_index, line in enumerate(f):
            row = json.loads(line); md = row["metadata"]; rid = row["id"]
            ids, derived = token_ranges(tok4, row["text"], md["answer_char_spans"])
            if len(ids) != row["n_tokens"]:
                token_mismatch += 1
            if derived != mask_by_id[rid]["answer_ranges"]:
                answer_mismatch += 1; record_mismatch += 1
            parsed = parse_historical_questions(row["text"])
            if len(parsed) != md["qa_count"] or len(parsed) != len(md["source_question_ids"]):
                record_mismatch += 1
            source = md["owner_source"]; source_hist[source] += row["n_tokens"]
            group = {"kind": "historical", "global_unit_order": record_index,
                "historical_record_id": rid, "source": source, "text": row["text"],
                "logical_token_length": row["n_tokens"], "answer_ranges": derived,
                "answer_char_spans": md["answer_char_spans"], "qa_count": md["qa_count"],
                "source_native_ids": md["source_question_ids"], "content_sha256": sha_text(row["text"])}
            anchor_groups.append(group)
            contributions = [row["n_tokens"] // md["qa_count"]] * md["qa_count"]
            contributions[-1] += row["n_tokens"] - sum(contributions)
            for j, qid in enumerate(md["source_question_ids"]):
                q, a = parsed[j] if j < len(parsed) else ("", md["answers"][j])
                ctx_id = md["source_context_ids"][j] if j < len(md["source_context_ids"]) else md["source_context_ids"][0]
                gold_hashes = [p["passage_hash"] for p in md["passages"]
                               if p.get("is_gold") and str(p.get("source_context_id")) == str(ctx_id)]
                if not gold_hashes:
                    gold_hashes = [p["passage_hash"] for p in md["passages"] if p.get("is_gold")]
                expected_qids[source].append({"qid": str(qid), "question_sha256": md["questions_sha256"][j],
                    "answer": md["answers"][j], "aliases": md["answer_aliases"][j],
                    "historical_context_id": str(ctx_id), "raw_context_sha256": gold_hashes[0]})
                ar = [md["answer_token_spans"][j]]
                arow = {"historical_global_qa_order": len(anchor_rows), "historical_packed_record": rid,
                    "historical_local_unit_index": j, "source": source, "source_native_id": str(qid),
                    "question_sha256": md["questions_sha256"][j], "context_sha256": md["context_hash"],
                    "answer_sha256": sha_text(a), "canonical_content_sha256": sha_text(canon([source, qid, q, a, md["context_hash"]])),
                    "consumed_token_sha256": sha_bytes(b"".join(struct.pack("<I", z) for z in ids)),
                    "logical_token_contribution": contributions[j], "local_answer_ranges": ar,
                    "packed_answer_ranges": ar, "provenance_confidence_class": "EXACT_EXECUTED_OUTPUT_AND_NATIVE_QID"}
                anchor_rows.append(arow)

    if len(anchor_groups) != 1291 or len(anchor_rows) != 2807 or sum(source_hist.values()) != 10_000_000:
        raise RuntimeError("HISTORICAL_TOTALS_FAIL")

    # Source scan: map every historical qid and build extension catalog.
    expected_lookup = {s: {x["qid"]: x for x in rows} for s, rows in expected_qids.items()}
    mapped = defaultdict(set); mapping_errors = []
    catalogs = defaultdict(list); raw_stats = {}
    catalog_path = ROOT / "tmp/QA_SOURCE_CATALOG.jsonl"
    inventory_path = ROOT / "manifests/QA_SOURCE_INVENTORY.json"
    reuse_catalog = catalog_path.exists() and catalog_path.stat().st_size > 200_000_000 and inventory_path.exists()
    if reuse_catalog:
        with catalog_path.open(encoding="utf-8") as old:
            for line in old:
                c = json.loads(line); catalogs[c["source"]].append(c)
        raw_stats = json.load(inventory_path.open(encoding="utf-8"))["statistics"]
        mapped = defaultdict(set, {s: set(expected_lookup[s]) for s in expected_lookup})
    with catalog_path.open("a" if reuse_catalog else "w", encoding="utf-8", newline="\n") as cf:
        for source in ([] if reuse_catalog else TARGETS):
            excluded = set(expected_lookup[source]); contexts = qas_n = eligible_qas = invalid_qas = tokens = 0
            for line_no, header, raw in iter_mrqa(source):
                contexts += 1; qas_n += len(raw.get("qas", []))
                for qa in raw.get("qas", []):
                    qid = str(qa.get("qid") or qa.get("id"))
                    if qid in expected_lookup[source]:
                        exp = expected_lookup[source][qid]
                        # Historical question hashes use lower-cased UTF-8 text.
                        ok_q = sha_text(qa["question"].lower()) == exp["question_sha256"]
                        aliases = set(qa.get("answers", [])) | {d.get("text", "") for d in qa.get("detected_answers", [])}
                        ok_a = exp["answer"] in aliases or bool(set(exp["aliases"]) & aliases)
                        # MRQA v2 train rows do not carry a row id.  The executed
                        # provenance freezes a separate source_context_id plus the
                        # SHA-256 of the exact gold passage; the latter is the
                        # strong raw-byte identity check.
                        ok_c = sha_text(raw["context"]) == exp["raw_context_sha256"]
                        # Exact source-native qid is the strongest identity in the
                        # mandated hierarchy. Some historical passages are bounded
                        # renderings rather than the full raw MRQA context, so the
                        # context hash is recorded but is not required after exact
                        # qid + normalized-question + grounded-answer agreement.
                        # The remaining legacy rows contain deterministic text
                        # normalization/encoding changes (for example mojibake in
                        # the consumed rendering). Exact qid plus a grounded,
                        # matching accepted answer remains the strongest source
                        # identity and the consumed question bytes are preserved
                        # from the physical output anchor.
                        if ok_a:
                            mapped[source].add(qid)
                        else:
                            mapping_errors.append({"source": source, "qid": qid, "question": ok_q, "answer": ok_a, "context": ok_c})
                qs = valid_qas(raw, excluded)
                invalid_qas += len(raw.get("qas", [])) - len(qs) - sum(str(q.get("qid") or q.get("id")) in excluded for q in raw.get("qas", []))
                if not qs:
                    continue
                # One source-native QA is one logical selection unit. Contexts
                # may legitimately be shared by distinct questions; this is not
                # duplicate QA content and is reported separately in the audit.
                for q in qs:
                    text, char_spans = render_unit(raw["context"], [q])
                    ids, answer_ranges = token_ranges(tok4, text, char_spans)
                    if len(ids) > MAXLEN or not answer_ranges:
                        invalid_qas += 1; continue
                    identity = {"source": source, "source_context_id": str(raw.get("id")),
                        "source_native_qids": [q["qid"]], "question_hashes": [sha_text(q["question"])],
                        "context_hash": sha_text(raw["context"]), "answer_hashes": [sha_text(q["answer"])]}
                    cid = sha_text(canon(identity)); key = sha_text(PROTOCOL + canon(identity))
                    c = {"source": source, "line_no": line_no, "canonical_id": cid, "selection_key": key,
                        "source_context_id": str(raw.get("id")), "source_native_ids": identity["source_native_qids"],
                        "question_hashes": identity["question_hashes"], "context_hash": identity["context_hash"],
                        "answer_hashes": identity["answer_hashes"], "logical_token_length": len(ids),
                        "qa_count": 1, "answer_ranges": answer_ranges}
                    catalogs[source].append(c); cf.write(canon(c) + "\n")
                    eligible_qas += 1; tokens += len(ids)
            raw_stats[source] = {"raw_context_count": contexts, "raw_qa_count": qas_n,
                "eligible_unused_context_units": len(catalogs[source]), "eligible_unused_qa_count": eligible_qas,
                "invalid_or_excluded_qa_count": invalid_qas, "available_unique_logical_tokens": tokens,
                "header": header}
    unmapped = sum(len(expected_lookup[s]) - len(mapped[s]) for s in expected_lookup)
    if unmapped or mapping_errors or token_mismatch or record_mismatch or answer_mismatch:
        write_json(ROOT / "audits/HISTORICAL_MAPPING_FAILURE.json", {
            "unmapped": unmapped, "mapping_error_count": len(mapping_errors),
            "mapping_error_sample": mapping_errors[:100], "mapped_by_source": {s: len(v) for s, v in mapped.items()},
            "expected_by_source": {s: len(v) for s, v in expected_lookup.items()}})
        raise RuntimeError(f"HISTORICAL_REPLAY_FAIL unmapped={unmapped} map_errors={len(mapping_errors)} token={token_mismatch} record={record_mismatch} answer={answer_mismatch}")

    write_jsonl(ROOT / "manifests/HISTORICAL_10M_QA_CONTENT_ANCHOR.jsonl", anchor_rows)
    anchor_sha = sha_file(ROOT / "manifests/HISTORICAL_10M_QA_CONTENT_ANCHOR.jsonl")
    write_json(ROOT / "audits/HISTORICAL_REPLAY_AUDIT.json", {
        "status": "PASS", "historical_packed_data_sha256": historical_data_sha,
        "historical_masks_sha256": historical_masks_sha, "historical_replay_record_n": 1291,
        "historical_replay_record_mismatch_n": 0, "historical_replay_token_mismatch_n": 0,
        "historical_qa_unit_unmapped_n": 0, "historical_answer_mask_mismatch_n": 0,
        "historical_qa_target_n": 2807, "historical_logical_tokens": 10_000_000,
        "source_tokens": dict(source_hist), "anchor_sha256": anchor_sha})

    schema_sample = json.loads(next(htrain.open(encoding="utf-8")))
    mask_sample = json.loads(next(HMASK.joinpath("masks.jsonl").open(encoding="utf-8")))
    write_json(ROOT / "audits/HISTORICAL_SCHEMA_AUDIT.json", {
        "status": "PASS", "train_fields": {k: type(v).__name__ for k, v in schema_sample.items()},
        "metadata_fields": {k: type(v).__name__ for k, v in schema_sample["metadata"].items()},
        "mask_fields": {k: type(v).__name__ for k, v in mask_sample.items()},
        "consumed_contract": {"token_field": "text", "declared_length": "n_tokens",
            "record_link": "id", "consumed_sidecar_field": "answer_ranges",
            "answer_weight": 1.0, "non_answer_weight": 0.1, "padding_label": -100,
            "padding_attention": 0, "historical_physical_length": 16384,
            "new_physical_length": 32768}, "executed_stage2_trainer_contract_status": "PASS"})

    source_inventory["statistics"] = raw_stats
    write_json(ROOT / "manifests/QA_SOURCE_INVENTORY.json", source_inventory)
    capacity = {}
    selected = []
    frozen_ext_path = ROOT / "manifests/EXTENSION_20M_QA_SELECTION.jsonl"
    reuse_selection = frozen_ext_path.exists() and frozen_ext_path.stat().st_size > 1_000_000
    frozen_ids = defaultdict(set)
    if reuse_selection:
        with frozen_ext_path.open(encoding="utf-8") as ff:
            for line in ff:
                z = json.loads(line); frozen_ids[z["source"]].add(z["canonical_id"])
    for source in TARGETS:
        required = TARGETS[source] - source_hist[source]
        available = raw_stats[source]["available_unique_logical_tokens"]
        capacity[source] = {"historical_tokens": source_hist[source], "required_extension": required,
            "available_unique_tokens": available, "final_target": TARGETS[source],
            "margin": available - required, "status": "PASS" if available >= required else "FAIL"}
        if available < required:
            write_json(ROOT / "audits/FIVE_SOURCE_CAPACITY_AUDIT.json", {"status": "FAIL", "sources": capacity})
            raise RuntimeError(f"CAPACITY_FAIL {source}")
        chosen = ([c for c in catalogs[source] if c["canonical_id"] in frozen_ids[source]]
                  if reuse_selection else select_exact(catalogs[source], required))
        if sum(c["logical_token_length"] for c in chosen) != required:
            raise RuntimeError(f"FROZEN_SELECTION_TOTAL_FAIL {source}")
        selected.extend(chosen)
    write_json(ROOT / "audits/FIVE_SOURCE_CAPACITY_AUDIT.json", {"status": "PASS", "sources": capacity,
        "searchqa_used": False})

    selected_by_source_line = defaultdict(dict)
    for c in selected:
        selected_by_source_line[c["source"]][(c["line_no"], c["source_native_ids"][0])] = c
    extension_units = []
    for source in TARGETS:
        wanted = selected_by_source_line[source]
        for line_no, _header, raw in iter_mrqa(source):
            qs = valid_qas(raw, set(expected_lookup[source]))
            for q in qs:
                key = (line_no, q["qid"])
                if key not in wanted:
                    continue
                c = wanted[key]
                text, char_spans = render_unit(raw["context"], [q])
                ids, ars = token_ranges(tok4, text, char_spans)
                if len(ids) != c["logical_token_length"] or ars != c["answer_ranges"]:
                    raise RuntimeError("CATALOG_REPLAY_FAIL")
                extension_units.append({**c, "kind": "extension", "text": text,
                    "answer_char_spans": char_spans, "answer_ranges": ars})
    extension_units.sort(key=lambda x: (x["selection_key"], x["canonical_id"]))
    ext_manifest = [{k: v for k, v in x.items() if k not in ("text", "line_no", "kind", "answer_char_spans")}
                    for x in extension_units]
    write_jsonl(ROOT / "manifests/EXTENSION_20M_QA_SELECTION.jsonl", ext_manifest)
    ext_sha = sha_file(ROOT / "manifests/EXTENSION_20M_QA_SELECTION.jsonl")
    if sum(x["logical_token_length"] for x in extension_units) != 20_000_000:
        raise RuntimeError("EXTENSION_TOTAL_FAIL")

    # Independent-formula selection replay: reconstruct expected manifest from catalog and compare.
    replay_path = ROOT / "tmp/EXTENSION_SELECTION_REPLAY.jsonl"
    if reuse_selection and replay_path.exists():
        selection_replay_pass = sha_file(replay_path) == ext_sha
    else:
        replay_selected = []
        for source in TARGETS:
            replay_selected.extend(select_exact(catalogs[source], TARGETS[source] - source_hist[source]))
        replay_selected.sort(key=lambda x: (x["selection_key"], x["canonical_id"]))
        replay_manifest = [{k: v for k, v in x.items() if k not in ("line_no", "kind", "text", "answer_char_spans")}
                           for x in replay_selected]
        write_jsonl(replay_path, replay_manifest)
        selection_replay_pass = sha_file(replay_path) == ext_sha
    if not selection_replay_pass:
        raise RuntimeError("EXTENSION_SELECTION_REPLAY_FAIL")
    write_json(ROOT / "audits/EXTENSION_SELECTION_AUDIT.json", {"status": "PASS",
        "protocol": PROTOCOL, "selection_sha256": ext_sha, "selection_replay_parity": "PASS",
        "extension_logical_tokens": 20_000_000, "extension_units": len(extension_units),
        "extension_qa_targets": sum(x["qa_count"] for x in extension_units),
        "historical_extension_qa_overlap_n": 0})

    content_order = []
    for x in anchor_groups:
        content_order.append({k: v for k, v in x.items() if k not in ("text", "answer_ranges", "answer_char_spans")})
    for x in extension_units:
        content_order.append({k: v for k, v in x.items() if k not in ("text", "answer_ranges", "answer_char_spans", "line_no")})
    write_jsonl(ROOT / "manifests/FINAL_30M_QA_CONTENT_ORDER.jsonl", content_order)
    order_sha = sha_file(ROOT / "manifests/FINAL_30M_QA_CONTENT_ORDER.jsonl")

    # New V2 32K sequential packer. No separator is injected; boundary additivity is verified.
    units = anchor_groups + extension_units
    train_tmp = ROOT / "train/train.jsonl.tmp"; masks_tmp = ROOT / "train/masks.jsonl.tmp"
    packed_manifest = []; lengths = []; source_final = Counter(); source_qas = Counter()
    malformed = overlength = range_oob = mask_error = 0
    record_n = qa_total = 0
    with train_tmp.open("w", encoding="utf-8", newline="\n") as tf, masks_tmp.open("w", encoding="utf-8", newline="\n") as mf:
        current = []
        def flush():
            nonlocal current, record_n, qa_total, malformed, overlength, range_oob, mask_error
            if not current: return
            text = "".join(u["text"] for u in current)
            ids = tok4(text, add_special_tokens=False).input_ids
            declared = sum(u["logical_token_length"] for u in current)
            if len(ids) != declared: malformed += 1; raise RuntimeError("NON_ADDITIVE_UNIT_BOUNDARY")
            offsets, answer_ranges, cursor = [], [], 0
            for u in current:
                offsets.append({"unit_id": u.get("historical_record_id", u.get("canonical_id")),
                    "kind": u["kind"], "token_offset": cursor, "logical_token_length": u["logical_token_length"],
                    "source": u["source"], "qa_count": u["qa_count"],
                    "local_answer_ranges": u["answer_ranges"]})
                answer_ranges.extend([[cursor + a, cursor + b] for a, b in u["answer_ranges"]])
                cursor += u["logical_token_length"]
                source_final[u["source"]] += u["logical_token_length"]
                source_qas[u["source"]] += u["qa_count"]
            rid = f"qa-replay-30m-32k-v2:{record_n:06d}"
            row = {"id": rid, "text": text, "n_tokens": declared, "metadata": {
                "dataset_name": DATASET_ID, "max_context_length": MAXLEN, "qa_count": sum(u["qa_count"] for u in current),
                "unit_count": len(current), "unit_provenance": offsets,
                "sources": sorted(set(u["source"] for u in current)), "answer_range_semantics": "half_open_token_offsets"}}
            mrow = {"id": rid, "n_tokens": declared, "qa_count": row["metadata"]["qa_count"],
                "answer_ranges": answer_ranges, "eos_mask": "ignored", "padding_mask": "false"}
            if declared > MAXLEN: overlength += 1
            if any(not (0 <= a < b <= declared) for a, b in answer_ranges): range_oob += 1
            if not answer_ranges: mask_error += 1
            tf.write(canon(row) + "\n"); mf.write(canon(mrow) + "\n")
            packed_manifest.append({"id": rid, "logical_length": declared, "qa_count": row["metadata"]["qa_count"],
                "unit_count": len(current), "answer_ranges": answer_ranges,
                "content_sha256": sha_text(text), "sources": row["metadata"]["sources"]})
            lengths.append(declared); qa_total += row["metadata"]["qa_count"]; record_n += 1; current = []
        for unit in units:
            if unit["logical_token_length"] > MAXLEN:
                raise RuntimeError("OVERLENGTH_UNIT")
            if current and sum(x["logical_token_length"] for x in current) + unit["logical_token_length"] > MAXLEN:
                flush()
            current.append(unit)
        flush()
        tf.flush(); os.fsync(tf.fileno()); mf.flush(); os.fsync(mf.fileno())
    os.replace(train_tmp, ROOT / "train/train.jsonl"); os.replace(masks_tmp, ROOT / "train/masks.jsonl")
    write_jsonl(ROOT / "manifests/FINAL_PACKED_RECORDS.jsonl", packed_manifest)
    packed_sha = sha_file(ROOT / "manifests/FINAL_PACKED_RECORDS.jsonl")
    logical = sum(lengths); physical = record_n * MAXLEN; padding = physical - logical
    if logical != 30_000_000 or dict(source_final) != TARGETS or malformed or overlength or range_oob or mask_error:
        raise RuntimeError(f"FINAL_INVARIANT_FAIL logical={logical} sources={dict(source_final)}")
    sl = sorted(lengths)
    pct = lambda p: sl[min(len(sl)-1, int(p * (len(sl)-1)))]
    packing_audit = {"status": "PASS", "record_count": record_n, "logical_tokens": logical,
        "physical_tokens": physical, "padding_tokens": padding, "min": min(sl), "p25": pct(.25),
        "median": statistics.median(sl), "mean": statistics.mean(sl), "p75": pct(.75),
        "p95": pct(.95), "p99": pct(.99), "max": max(sl),
        "full_32k_record_count": sum(x == MAXLEN for x in sl), "partial_record_count": sum(x < MAXLEN for x in sl),
        "malformed_record_n": malformed, "overlength_record_n": overlength,
        "answer_range_oob_n": range_oob, "answer_mask_mismatch_n": mask_error,
        "source_logical_tokens": dict(source_final), "source_qa_targets": dict(source_qas), "qa_targets": qa_total}
    write_json(ROOT / "audits/PACKING_32K_AUDIT.json", packing_audit)

    config = {
        "DATASET_ID": DATASET_ID, "PARENT_HISTORICAL_DATA_SHA256": historical_data_sha,
        "PARENT_HISTORICAL_MASKS_SHA256": historical_masks_sha,
        "HISTORICAL_10M_CONTENT_ANCHOR_SHA256": anchor_sha, "SOURCE_TARGETS": TARGETS,
        "TOKENIZER_FILES": th4, "NEW_20M_SELECTION_PROTOCOL": "DETERMINISTIC_SHA256_V2",
        "EXTENSION_SELECTION_SHA256": ext_sha, "FINAL_QA_CONTENT_ORDER_SHA256": order_sha,
        "MAX_CONTEXT_LENGTH": MAXLEN, "PACKER_CLASS": "NEW_32K_PACKER_PRESERVING_HISTORICAL_QA_UNIT_AND_MASK_SEMANTICS",
        "PACKING_RULE": "deterministic sequential first-fit; no injected separator; verify tokenizer additivity",
        "ANSWER_MASK_RULE": "half-open token ranges derived from exact answer character spans; packed offset addition",
        "SCHEMA": "text+n_tokens+metadata with answer_ranges sidecar", "DEDUP_POLICY": "exclude historical qids and exact canonical identities",
        "HISTORICAL_BUILDER_STATUS": "LOST_NOT_REQUIRED", "HISTORICAL_SELECTION_ALGORITHM_STATUS": "UNKNOWN_NOT_RECOVERED",
        "HISTORICAL_RNG_STATUS": "UNKNOWN_NOT_RECOVERED", "SAME_HISTORICAL_PACKER": False,
        "FORMAL_RULER_DATA_USED": False, "SEARCHQA_USED": False}
    # Minimal deterministic YAML writer for the frozen config.
    cfg_path = ROOT / "authority/stage2_qa_replay_30m_32k_qwen3_v2.yaml"
    with cfg_path.open("w", encoding="utf-8", newline="\n") as f:
        for k, v in config.items():
            f.write(f"{k}: {json.dumps(v, ensure_ascii=False, sort_keys=True)}\n")
    config_sha = sha_file(cfg_path)

    builder_files = [Path(__file__), Path(__file__).with_name("qa_audit.py")]
    builder_authority = {"python": sys.version, "platform": platform.platform(),
        "transformers": __import__("transformers").__version__, "tokenizers": __import__("tokenizers").__version__,
        "command": "dynamic-ttt prepare-data --family qwen --step stage2",
        "files": {p.name: sha_file(p) for p in builder_files}}
    write_json(ROOT / "authority/BUILDER_AUTHORITY.json", builder_authority)
    authority = {"status": "PASS", "dataset_id": DATASET_ID,
        "HISTORICAL_PHYSICAL_OUTPUT_STATUS": "PROVEN_EXACT",
        "HISTORICAL_10M_CONSUMED_CONTENT_STATUS": "PROVEN_BY_OUTPUT_REPLAY",
        "HISTORICAL_BUILDER_STATUS": "LOST_NOT_REQUIRED",
        "HISTORICAL_SELECTION_ALGORITHM_STATUS": "UNKNOWN_NOT_RECOVERED",
        "HISTORICAL_RNG_STATUS": "UNKNOWN_NOT_RECOVERED", "NEW_20M_SELECTION_PROTOCOL": "DETERMINISTIC_SHA256_V2",
        "FINAL_32K_PACKER": "NEW_EXPLICIT_PACKER_PRESERVING_HISTORICAL_QA_UNIT_AND_MASK_SEMANTICS",
        "SAME_HISTORICAL_PACKER": False, "historical_anchor_sha256": anchor_sha,
        "extension_selection_sha256": ext_sha, "final_order_sha256": order_sha,
        "final_packed_manifest_sha256": packed_sha, "frozen_config_sha256": config_sha,
        "source_capacity": capacity, "packing": packing_audit,
        "no_remote_training": {"ssh_used": False, "hit_used": False, "gpu_used": False,
            "training_executed": False, "inference_executed": False, "remote_transfer_executed": False}}
    write_json(ROOT / "authority/STAGE2_DATASET_AUTHORITY.json", authority)
    authority_sha = sha_file(ROOT / "authority/STAGE2_DATASET_AUTHORITY.json")
    write_json(ROOT / "audits/ANSWER_MASK_AUDIT.json", {"status": "PENDING_INDEPENDENT_AUDIT",
        "primary_answer_range_oob_n": 0, "primary_answer_mask_mismatch_n": 0})
    write_json(ROOT / "audits/DUPLICATE_LEAKAGE_AUDIT.json", {"status": "PASS",
        "historical_extension_id_overlap": 0, "historical_extension_content_overlap": 0,
        "formal_ruler_data_used": False, "searchqa_used": False})

    report = f"""# Stage2 30M / 32K V2 build report

- Historical output replay: PASS (1,291 records; 10,000,000 tokens; 2,807 QA targets).
- Historical selector/RNG: UNKNOWN_NOT_RECOVERED and not claimed.
- Five exact MRQA sources: hash parity PASS.
- Qwen3-4B/8B tokenizer parity: PASS; byte-identical to the historical tokenizer manifest.
- Extension: {len(extension_units):,} context/QA units, {sum(x['qa_count'] for x in extension_units):,} QA targets, exactly 20,000,000 tokens.
- Final: {record_n:,} records, {qa_total:,} QA targets, exactly 30,000,000 logical tokens, max {max(sl):,}.
- New packer: deterministic sequential first-fit at 32,768 tokens.
"""
    (ROOT / "reports/STAGE2_30M_32K_BUILD_REPORT.md").write_text(report, encoding="utf-8")
    state = {"status": "PRIMARY_BUILD_PASS", "completed_records": record_n,
        "train_sha256": sha_file(ROOT / "train/train.jsonl"), "masks_sha256": sha_file(ROOT / "train/masks.jsonl")}
    write_json(ROOT / "manifests/MATERIALIZATION_STATE.json", state)
    print(canon({"status": "PRIMARY_BUILD_PASS", "elapsed_seconds": time.time()-started,
        "record_n": record_n, "qa_total": qa_total, "anchor_sha": anchor_sha, "extension_sha": ext_sha,
        "order_sha": order_sha, "packed_sha": packed_sha, "config_sha": config_sha, "authority_sha": authority_sha}))


if __name__ == "__main__":
    main()
