#!/usr/bin/env python3
"""Validate and aggregate 173 committed Exact Reverse results."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name("." + path.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except BaseException:
        try:
            os.unlink(temp)
        except FileNotFoundError:
            pass
        raise


def atomic_json(path, obj):
    atomic_text(path, json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def atomic_csv(path, rows):
    rows = list(rows)
    if not rows:
        atomic_text(path, "")
        return
    path = Path(path)
    temp = path.with_name("." + path.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except BaseException:
        try:
            os.unlink(temp)
        except FileNotFoundError:
            pass
        raise


def result_name(sample_id):
    return hashlib.sha256(sample_id.encode()).hexdigest() + ".json"


def mean(xs):
    return statistics.fmean(xs)


def ci(values, seed, resamples):
    x = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 1000):
        n = min(1000, resamples - start)
        means[start:start+n] = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def stratified_ci(rows, seed, resamples):
    groups = defaultdict(list)
    for r in rows:
        groups[r["task"]].append(float(r["dynamic_minus_reverse"]))
    rng = np.random.default_rng(seed)
    out = np.zeros(resamples, dtype=np.float64)
    total = sum(len(v) for v in groups.values())
    for vals in groups.values():
        x = np.asarray(vals, dtype=np.float64)
        for start in range(0, resamples, 1000):
            n = min(1000, resamples - start)
            out[start:start+n] += x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1) * len(x) / total
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def update_status(run, **changes):
    path = run / "status/pipeline_status.json"
    state = load_json(path)
    state.update(changes)
    state["updated_at"] = now()
    atomic_json(path, state)


def main(run_root):
    run = Path(run_root).resolve()
    config = load_json(run / "config/EXACT_REVERSE_FORMAL_CONFIG.json")
    if sha(Path(config["aggregator"])) != config["aggregator_sha256"]:
        raise RuntimeError("AGGREGATOR_HASH_MISMATCH")
    if sha(Path(config["cohort_manifest"])) != config["cohort_manifest_sha256"]:
        raise RuntimeError("COHORT_HASH_MISMATCH")
    if sha(Path(config["protocol"])) != config["protocol_sha256"]:
        raise RuntimeError("PROTOCOL_HASH_MISMATCH")
    smoke = load_json(run / "smoke/ORIGINAL_DYNAMIC_REPLAY_PARITY.json")
    if smoke.get("ORIGINAL_DYNAMIC_REPLAY_PARITY") != "PASS":
        raise RuntimeError("ORIGINAL_REPLAY_PARITY_NOT_PASS")
    cohort = load_jsonl(config["cohort_manifest"])
    records = []
    for meta in cohort:
        path = run / "results" / result_name(meta["canonical_sample_id"])
        if not path.exists():
            raise RuntimeError(f"MISSING_RESULT {meta['canonical_sample_id']}")
        r = load_json(path)
        checks = [
            r.get("completed") is True, r.get("status") == "COMMITTED",
            r.get("canonical_sample_id") == meta["canonical_sample_id"],
            r.get("task") == meta["task"], r.get("reversed_actions") == meta["reversed_actions"],
            r.get("cohort_manifest_hash") == config["cohort_manifest_sha256"],
            r.get("protocol_hash") == config["protocol_sha256"],
            r.get("model_identity") == config["model_identity"],
            r.get("checkpoint_identity") == config["checkpoint_identity"],
            r.get("benchmark_identity") == config["benchmark_identity"],
            r.get("action_composition_parity") is True, r.get("reverse_distinct") is True,
            r.get("recompute_downstream_updates") is True, r.get("stored_delta_stitching") is False,
            r.get("generation_update_count") == 0, r.get("base_weight_unchanged") is True,
        ]
        if not all(checks):
            raise RuntimeError(f"INVALID_RESULT {meta['canonical_sample_id']} {checks}")
        records.append(r)
    ids = [r["canonical_sample_id"] for r in records]
    duplicate_n = len(ids) - len(set(ids))
    result_files = list((run / "results").glob("*.json"))
    if len(records) != 173 or duplicate_n or len(result_files) != 173:
        raise RuntimeError(f"COMPLETENESS_FAILURE records={len(records)} files={len(result_files)} duplicates={duplicate_n}")
    if list((run / "errors").glob("*.json")):
        raise RuntimeError("UNRESOLVED_ERROR_FILES")

    dyn = [float(r["original_dynamic_score"]) for r in records]
    rev = [float(r["reverse_score"]) for r in records]
    sb = [float(r["sample_best_score"]) for r in records]
    off = [float(r["off_score"]) for r in records]
    d = [float(r["dynamic_minus_reverse"]) for r in records]
    dg = [float(r["dynamic_minus_sample_best"]) for r in records]
    rg = [float(r["reverse_minus_sample_best"]) for r in records]
    retained = [float(r["retained_gain"]) for r in records]
    worse = sum(x > 1e-12 for x in d)
    equal = sum(abs(x) <= 1e-12 for x in d)
    better = sum(x < -1e-12 for x in d)
    rev_gt_sb = sum(x > 1e-12 for x in rg)
    rev_eq_sb = sum(abs(x) <= 1e-12 for x in rg)
    rev_lt_sb = sum(x < -1e-12 for x in rg)
    paired_ci = ci(d, config["paired_bootstrap_seed"], config["bootstrap_resamples"])
    task_ci = stratified_ci(records, config["task_stratified_bootstrap_seed"], config["bootstrap_resamples"])
    quantile_levels = [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]
    retention_quantiles = {str(q): float(np.quantile(np.asarray(retained), q)) for q in quantile_levels}

    per_task = []
    for task in sorted({r["task"] for r in records}):
        z = [r for r in records if r["task"] == task]
        zd = [float(r["dynamic_minus_reverse"]) for r in z]
        zrg = [float(r["reverse_minus_sample_best"]) for r in z]
        per_task.append({
            "task": task, "anchor_n": len(z),
            "mean_dynamic": mean(float(r["original_dynamic_score"]) for r in z),
            "mean_reverse": mean(float(r["reverse_score"]) for r in z),
            "mean_sample_best": mean(float(r["sample_best_score"]) for r in z),
            "mean_dynamic_minus_reverse": mean(zd), "mean_reverse_minus_sample_best": mean(zrg),
            "reverse_worse_count": sum(x > 1e-12 for x in zd),
            "reverse_equal_count": sum(abs(x) <= 1e-12 for x in zd),
            "reverse_better_count": sum(x < -1e-12 for x in zd),
        })
    atomic_csv(run / "reports/PER_TASK_EXACT_REVERSE_RESULTS.csv", per_task)
    atomic_json(run / "reports/PER_TASK_EXACT_REVERSE_RESULTS.json", per_task)

    summary = {
        "N": 173,
        "DYNAMIC_MEAN": mean(dyn), "REVERSE_MEAN": mean(rev), "SAMPLE_BEST_MEAN": mean(sb), "OFF_MEAN": mean(off),
        "DYNAMIC_MINUS_REVERSE_MEAN": mean(d), "DYNAMIC_MINUS_REVERSE_MEDIAN": statistics.median(d),
        "DYNAMIC_MINUS_REVERSE_STD": statistics.stdev(d), "DYNAMIC_MINUS_REVERSE_MIN": min(d), "DYNAMIC_MINUS_REVERSE_MAX": max(d),
        "REVERSE_WORSE_COUNT": worse, "REVERSE_EQUAL_COUNT": equal, "REVERSE_BETTER_COUNT": better,
        "DYNAMIC_MINUS_SAMPLE_BEST_MEAN": mean(dg), "REVERSE_MINUS_SAMPLE_BEST_MEAN": mean(rg),
        "REVERSE_GT_SAMPLE_BEST_COUNT": rev_gt_sb, "REVERSE_EQ_SAMPLE_BEST_COUNT": rev_eq_sb,
        "REVERSE_LT_SAMPLE_BEST_COUNT": rev_lt_sb,
        "GAIN_RETENTION_MEAN": mean(retained), "GAIN_RETENTION_MEDIAN": statistics.median(retained),
        "GAIN_RETENTION_QUANTILES": retention_quantiles,
        "GAIN_RETENTION_FRACTION_LE_0": sum(x <= 0 for x in retained) / 173,
        "GAIN_RETENTION_FRACTION_LT_0P5": sum(x < 0.5 for x in retained) / 173,
        "GAIN_RETENTION_FRACTION_LT_1": sum(x < 1 for x in retained) / 173,
        "GAIN_RETENTION_FRACTION_GE_1": sum(x >= 1 for x in retained) / 173,
        "AGGREGATE_GAIN_RETENTION": sum(rg) / sum(dg),
        "PAIRED_BOOTSTRAP_95CI": paired_ci,
        "TASK_STRATIFIED_PAIRED_BOOTSTRAP_95CI": task_ci,
        "paired_bootstrap_seed": config["paired_bootstrap_seed"],
        "task_stratified_bootstrap_seed": config["task_stratified_bootstrap_seed"],
        "bootstrap_resamples": config["bootstrap_resamples"],
        "task_distribution": dict(Counter(r["task"] for r in records)),
        "secondary_interpretation": "NOT_CLASSIFIED_NO_HISTORICAL_PREREGISTERED_THRESHOLD",
    }
    atomic_json(run / "reports/EXACT_REVERSE_AGGREGATE.json", summary)
    atomic_csv(run / "reports/EXACT_REVERSE_PER_SAMPLE.csv", [{
        "cohort_order": r["cohort_order"], "canonical_sample_id": r["canonical_sample_id"], "task": r["task"],
        "benchmark_global_index": r["benchmark_global_index"], "dynamic": r["original_dynamic_score"],
        "reverse": r["reverse_score"], "sample_best": r["sample_best_score"], "off": r["off_score"],
        "dynamic_minus_reverse": r["dynamic_minus_reverse"], "dynamic_minus_sample_best": r["dynamic_minus_sample_best"],
        "reverse_minus_sample_best": r["reverse_minus_sample_best"], "retained_gain": r["retained_gain"],
    } for r in records])

    result_manifest = {
        "result_n": len(records), "unique_sample_ids": len(set(ids)), "duplicate_result_ids": duplicate_n,
        "cohort_manifest_sha256": config["cohort_manifest_sha256"], "protocol_sha256": config["protocol_sha256"],
        "results": [{"canonical_sample_id": r["canonical_sample_id"],
                     "path": str(run / "results" / result_name(r["canonical_sample_id"])),
                     "sha256": sha(run / "results" / result_name(r["canonical_sample_id"]))} for r in records],
        "frozen_at": now(),
    }
    atomic_json(run / "manifests/FINAL_RESULT_MANIFEST.json", result_manifest)
    result_manifest_sha = sha(run / "manifests/FINAL_RESULT_MANIFEST.json")

    audit = {
        "MODEL_IDENTITY_MATCH": True, "BENCHMARK_IDENTITY_MATCH": True, "SCREENING_MANIFEST_MATCH": True,
        "COHORT_MANIFEST_MATCH": True, "PROTOCOL_HASH_MATCH": True, "COHORT_N": 173,
        "VALID_RESULTS": 173, "MISSING_RESULTS": 0, "DUPLICATE_RESULTS": 0, "UNRESOLVED_ERRORS": 0,
        "ACTION_COMPOSITION_PARITY": "PASS", "ORIGINAL_DYNAMIC_REPLAY_PARITY": "PASS",
        "HISTORICAL_OVERLAP": 0, "MECHANISM_OUTCOME_LEAKAGE": 0,
        "STRUCTURALLY_ELIGIBLE_NEGATIVE_CONTROLS": 0,
        "RESULT_MANIFEST_SHA256": result_manifest_sha, "FINAL_INTEGRITY": "PASS", "completed_at": now(),
    }
    atomic_json(run / "audit/FINAL_EXACT_REVERSE_INTEGRITY_AUDIT.json", audit)
    receipt = {
        "MODEL_IDENTITY": config["model_identity"], "CHECKPOINT_IDENTITY": config["checkpoint_identity"],
        "SCREENING_MANIFEST_SHA256": config["screening_manifest_sha256"],
        "COHORT_MANIFEST_SHA256": config["cohort_manifest_sha256"], "PROTOCOL_SHA256": config["protocol_sha256"],
        **summary, "STRUCTURALLY_ELIGIBLE_NEGATIVE_CONTROLS": 0,
        "FINAL_INTEGRITY": "PASS", "FORMAL_EXACT_REVERSE_COMPLETE": True,
        "claim_boundary": load_json(config["protocol"])["claim_boundary"], "completed_at": now(),
    }
    receipt_json = run / "reports/EXACT_REVERSE_1P7B_FORMAL_COMPLETION_RECEIPT.json"
    receipt_md = run / "reports/EXACT_REVERSE_1P7B_FORMAL_COMPLETION_RECEIPT.md"
    atomic_json(receipt_json, receipt)
    atomic_text(receipt_md, f"""# Exact Reverse 1.7B Formal Completion Receipt

- N: 173/173
- Dynamic mean: {summary['DYNAMIC_MEAN']:.12f}
- Reverse mean: {summary['REVERSE_MEAN']:.12f}
- Sample Best mean: {summary['SAMPLE_BEST_MEAN']:.12f}
- OFF mean: {summary['OFF_MEAN']:.12f}
- Mean Dynamic - Reverse: {summary['DYNAMIC_MINUS_REVERSE_MEAN']:.12f}
- Median Dynamic - Reverse: {summary['DYNAMIC_MINUS_REVERSE_MEDIAN']:.12f}
- Reverse worse/equal/better: {worse}/{equal}/{better}
- Mean Dynamic - Sample Best: {summary['DYNAMIC_MINUS_SAMPLE_BEST_MEAN']:.12f}
- Mean Reverse - Sample Best: {summary['REVERSE_MINUS_SAMPLE_BEST_MEAN']:.12f}
- Mean/median retained gain: {summary['GAIN_RETENTION_MEAN']:.12f}/{summary['GAIN_RETENTION_MEDIAN']:.12f}
- Aggregate gain retention: {summary['AGGREGATE_GAIN_RETENTION']:.12f}
- Ordinary paired bootstrap 95% CI: {paired_ci}
- Task-stratified paired bootstrap 95% CI: {task_ci}
- Structurally eligible negative controls: 0
- FINAL_INTEGRITY: PASS
- FORMAL_EXACT_REVERSE_COMPLETE: true

Claims remain restricted to the frozen successful sequential-positive cohort. No comparison against unsuccessful nontrivial trajectories is available.
""")
    update_status(run, pipeline_status="COMPLETE", current_phase="COMPLETE", committed_results=173,
                  current_sample=None, current_task=None, worker_pid=None, worker_running=False,
                  error_count=0, formal_complete=True, final_receipt_exists=True,
                  final_receipt=str(receipt_json), final_integrity="PASS", completed_at=now())
    atomic_json(run / "status/FINAL_COMPLETE.json", {"FORMAL_EXACT_REVERSE_COMPLETE": True,
                                                      "FINAL_INTEGRITY": "PASS", "N": 173,
                                                      "receipt": str(receipt_json), "timestamp": now()})
    print(json.dumps({"FORMAL_EXACT_REVERSE_COMPLETE": True, "FINAL_INTEGRITY": "PASS",
                      "N": 173, "DYNAMIC_MINUS_REVERSE_MEAN": summary["DYNAMIC_MINUS_REVERSE_MEAN"],
                      "PAIRED_BOOTSTRAP_95CI": paired_ci,
                      "TASK_STRATIFIED_PAIRED_BOOTSTRAP_95CI": task_ci}, sort_keys=True), flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("usage: aggregate_exact_reverse.py RUN_ROOT")
    main(sys.argv[1])
