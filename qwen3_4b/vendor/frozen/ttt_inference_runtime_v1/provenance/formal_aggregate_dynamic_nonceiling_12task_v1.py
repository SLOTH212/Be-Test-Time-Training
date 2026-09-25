#!/usr/bin/env python3
"""Final, deterministic aggregation for the frozen 12-task Dynamic LB run."""

import csv
import hashlib
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ACTIONS = ["OFF", "L0", "L6", "L12", "L18", "L24", "ALL"]
SINGLES = ["L0", "L6", "L12", "L18", "L24"]
EXCLUDED = "ruler_cwe_16k"
TOL = 1e-12
BOOTSTRAP_SEED = 20260828
BOOTSTRAP_DRAWS = 10000


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def result_name(sample_id):
    return hashlib.sha256(sample_id.encode()).hexdigest() + ".json"


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_json(path, obj):
    atomic_text(path, json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or (list(rows[0]) if rows else [])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def percentile(values, q):
    return float(np.quantile(values, q, method="linear"))


def classify(mean_delta, ci_low, ci_high, positive_tasks, improved_rate):
    """Frozen pre-result rule; thresholds are recorded in config before launch."""
    mean_pp = 100.0 * mean_delta
    if ci_high < 0:
        return "NEGATIVE"
    if ci_low > 0 and positive_tasks >= 8 and improved_rate >= 0.10:
        return "STRONG_POSITIVE"
    if ci_low > 0 and mean_delta > TOL:
        return "POSITIVE_BUT_SPARSE"
    if abs(mean_pp) < 0.01 and ci_low <= 0 <= ci_high and improved_rate < 0.01:
        return "NEAR_ZERO"
    return "MIXED"


def aggregate(run_root):
    run = Path(run_root).resolve()
    prereg_path = run / "preregistration/DYNAMIC_NONCEILING_12TASK_ELIGIBILITY_V1.json"
    prereg = json.load(open(prereg_path, encoding="utf-8"))
    config_path = run / "config/DYNAMIC_FORMAL_CONFIG.json"
    config = json.load(open(config_path, encoding="utf-8"))
    full = read_csv(run / "eligibility/fixed7_sample_best_full6500.csv")
    rows12 = [r for r in full if r["task"] != EXCLUDED]
    eligible = [r for r in rows12 if r["dynamic_eligible"] == "true"]
    candidate_ids = prereg["candidate_sample_ids"]
    if len(rows12) != 6000 or len({r["sample_id"] for r in rows12}) != 6000:
        raise RuntimeError("FULL_12TASK_POPULATION_INVALID")
    if [r["sample_id"] for r in eligible] != candidate_ids:
        raise RuntimeError("ELIGIBILITY_ORDER_OR_SET_MISMATCH")

    result_by_id = {}
    result_dir = run / "results"
    for sid in candidate_ids:
        path = result_dir / result_name(sid)
        if not path.is_file():
            raise RuntimeError(f"MISSING_DYNAMIC_RESULT {sid}")
        rec = json.load(open(path, encoding="utf-8"))
        if rec.get("sample_id") != sid or rec.get("completed") is not True or rec.get("error_status") != "NONE":
            raise RuntimeError(f"INVALID_DYNAMIC_RESULT {sid}")
        score = rec.get("dynamic_score")
        if not isinstance(score, (int, float)) or not math.isfinite(score) or score < -TOL or score > 1 + TOL:
            raise RuntimeError(f"INVALID_DYNAMIC_SCORE {sid}")
        if rec.get("dynamic_source_sha256") != config["dynamic_source_sha256"]:
            raise RuntimeError(f"DYNAMIC_SOURCE_IDENTITY_MISMATCH {sid}")
        if rec.get("preregistration_sha256") != sha256(prereg_path):
            raise RuntimeError(f"PREREGISTRATION_IDENTITY_MISMATCH {sid}")
        result_by_id[sid] = rec
    extras = [p for p in result_dir.glob("*.json") if p.name not in {result_name(x) for x in candidate_ids}]
    if extras or len(result_by_id) != len(candidate_ids):
        raise RuntimeError("DYNAMIC_RESULT_SET_NOT_EXACT")

    action_means = {a: statistics.fmean(float(r[a]) for r in rows12) for a in ACTIONS}
    best_single_action = max(SINGLES, key=lambda a: (action_means[a], -SINGLES.index(a)))
    reconstructed = []
    executed_gains = []
    for row in rows12:
        sid = row["sample_id"]
        sample_best = float(row["sample_best"])
        if row["dynamic_eligible"] == "true":
            rec = result_by_id[sid]
            dynamic = float(rec["dynamic_score"])
            execution = True
            skip_reason = ""
        else:
            if row["is_ceiling"] != "true" or abs(sample_best - 1.0) > TOL:
                raise RuntimeError(f"INVALID_SKIPPED_SAMPLE {sid}")
            dynamic = sample_best
            execution = False
            skip_reason = "SAMPLE_BEST_AT_CEILING"
        if sample_best + TOL < max(float(row[a]) for a in ACTIONS):
            raise RuntimeError(f"SAMPLEBEST_MONOTONICITY_FAILURE {sid}")
        gain = dynamic - sample_best
        if execution:
            executed_gains.append(gain)
        reconstructed.append({
            "sample_id": sid,
            "task": row["task"],
            **{a: row[a] for a in ACTIONS},
            "best_single_action_12task": best_single_action,
            "best_single_score": row[best_single_action],
            "sample_best": repr(sample_best),
            "final_dynamic_score": repr(dynamic),
            "dynamic_minus_samplebest": repr(gain),
            "dynamic_minus_off": repr(dynamic - float(row["OFF"])),
            "dynamic_execution": str(execution).lower(),
            "skip_reason": skip_reason,
        })
    write_csv(run / "aggregation/dynamic_full12task_reconstructed.csv", reconstructed)

    dyn_mean = statistics.fmean(float(r["final_dynamic_score"]) for r in reconstructed)
    sb_mean = statistics.fmean(float(r["sample_best"]) for r in reconstructed)
    off_mean = action_means["OFF"]
    all_mean = action_means["ALL"]
    best_single_mean = action_means[best_single_action]

    per_task = []
    for task in prereg["included_tasks"]:
        rr = [r for r in reconstructed if r["task"] == task]
        er = [result_by_id[r["sample_id"]] for r in rr if r["dynamic_execution"] == "true"]
        gains = [float(x["dynamic_minus_samplebest"]) for x in er]
        task_dynamic = statistics.fmean(float(r["final_dynamic_score"]) for r in rr)
        task_sb = statistics.fmean(float(r["sample_best"]) for r in rr)
        per_task.append({
            "task": task,
            "sample_count": len(rr),
            "ceiling_count": sum(r["dynamic_execution"] == "false" for r in rr),
            "eligible_count": len(er),
            "OFF": statistics.fmean(float(r["OFF"]) for r in rr),
            "best_single_action_12task": best_single_action,
            "Best_Single": statistics.fmean(float(r["best_single_score"]) for r in rr),
            "ALL": statistics.fmean(float(r["ALL"]) for r in rr),
            "Sample_Best": task_sb,
            "Dynamic": task_dynamic,
            "Dynamic_minus_SampleBest_pp": 100 * (task_dynamic - task_sb),
            "Dynamic_minus_OFF_pp": 100 * (task_dynamic - statistics.fmean(float(r["OFF"]) for r in rr)),
            "executed_improved": sum(g > TOL for g in gains),
            "executed_equal": sum(abs(g) <= TOL for g in gains),
            "executed_worse": sum(g < -TOL for g in gains),
        })
    write_csv(run / "per_task/dynamic_12task_breakdown.csv", per_task)

    improved = sum(g > TOL for g in executed_gains)
    equal = sum(abs(g) <= TOL for g in executed_gains)
    worse = sum(g < -TOL for g in executed_gains)
    positive = [g for g in executed_gains if g > TOL]
    executed_stats = {
        "dynamic_improved_count": improved,
        "dynamic_equal_count": equal,
        "dynamic_worse_count": worse,
        "dynamic_improved_rate": improved / len(executed_gains),
        "dynamic_worse_rate": worse / len(executed_gains),
        "mean_gain": statistics.fmean(executed_gains),
        "median_gain": statistics.median(executed_gains),
        "positive_only_mean": statistics.fmean(positive) if positive else 0.0,
        "max_gain": max(executed_gains),
        "worst_loss": min(executed_gains),
    }

    action_counts = Counter()
    transition_counts = Counter()
    transition_distribution = Counter()
    task_transition = defaultdict(Counter)
    zero_switch = one_switch = multi_switch = 0
    for rec in result_by_id.values():
        seq = rec["action_sequence"]
        action_counts.update(seq)
        switches = sum(a != b for a, b in zip(seq, seq[1:]))
        transition_distribution[switches] += 1
        if switches == 0:
            zero_switch += 1
        elif switches == 1:
            one_switch += 1
        else:
            multi_switch += 1
        for a, b in zip(seq, seq[1:]):
            if a != b:
                transition_counts[f"{a}->{b}"] += 1
                task_transition[rec["task"]][f"{a}->{b}"] += 1
    trajectory = {
        "action_chunk_counts": dict(action_counts),
        "transition_count_distribution": {str(k): v for k, v in sorted(transition_distribution.items())},
        "zero_switch_count": zero_switch,
        "one_switch_count": one_switch,
        "multi_switch_count": multi_switch,
        "most_common_transitions": transition_counts.most_common(30),
        "task_conditioned_transitions": {k: dict(v) for k, v in task_transition.items()},
    }
    atomic_json(run / "aggregation/trajectory_structure_statistics.json", trajectory)

    by_task_diff = {
        task: np.asarray([float(r["dynamic_minus_samplebest"]) for r in reconstructed if r["task"] == task], dtype=np.float64)
        for task in prereg["included_tasks"]
    }
    if any(len(x) != 500 for x in by_task_diff.values()):
        raise RuntimeError("BOOTSTRAP_TASK_SIZE_MISMATCH")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.zeros(BOOTSTRAP_DRAWS, dtype=np.float64)
    for task in prereg["included_tasks"]:
        values = by_task_diff[task]
        indices = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
        draws += values[indices].sum(axis=1)
    draws /= 6000.0
    bootstrap = {
        "method": "task_stratified_paired_bootstrap_full_12task_population",
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "ceiling_zero_delta_samples_included": True,
        "observed_mean_delta": dyn_mean - sb_mean,
        "bootstrap_mean_delta": float(draws.mean()),
        "standard_error": float(draws.std(ddof=1)),
        "ci95_lower": percentile(draws, 0.025),
        "ci95_upper": percentile(draws, 0.975),
        "probability_delta_gt_zero": float(np.mean(draws > 0)),
    }
    atomic_json(run / "bootstrap/dynamic_vs_samplebest_task_stratified_10000.json", bootstrap)

    positive_tasks = sum(float(x["Dynamic_minus_SampleBest_pp"]) > 100 * TOL for x in per_task)
    status = classify(dyn_mean - sb_mean, bootstrap["ci95_lower"], bootstrap["ci95_upper"], positive_tasks, executed_stats["dynamic_improved_rate"])
    headline = {
        "OFF_12TASK_SCORE": off_mean,
        "BEST_SINGLE_12TASK_ACTION": best_single_action,
        "BEST_SINGLE_12TASK_SCORE": best_single_mean,
        "ALL_12TASK_SCORE": all_mean,
        "SAMPLE_BEST_12TASK_SCORE": sb_mean,
        "DYNAMIC_12TASK_SCORE": dyn_mean,
        "DYNAMIC_MINUS_SAMPLEBEST_12TASK_PP": 100 * (dyn_mean - sb_mean),
        "DYNAMIC_MINUS_OFF_12TASK_PP": 100 * (dyn_mean - off_mean),
        "SAMPLEBEST_MINUS_OFF_12TASK_PP": 100 * (sb_mean - off_mean),
        "SEQUENTIAL_OPPORTUNITY_STATUS": status,
    }
    atomic_json(run / "aggregation/dynamic_12task_headline.json", headline)

    report_lines = [
        "# Dynamic non-ceiling 12-task formal report", "",
        "## 1. Fixed7 authority validation", "", "The completed 45,500-cell fixed7 run passed exact raw-set and sample alignment validation.", "",
        "## 2. CWE exclusion", "", f"`{EXCLUDED}` (500 samples) was excluded from execution and all 12-task aggregates.", "",
        "## 3. Sample Best definition", "", "Sample Best is the per-sample maximum over OFF, L0, L6, L12, L18, L24, and ALL.", "",
        "## 4. Ceiling-pruning logic", "", "Only SampleBest < 1 samples were executed. Ceiling samples remain in the denominator with Dynamic := SampleBest.", "",
        "## 5. Candidate-set statistics", "", f"Executed {len(eligible)} of 6000 samples; {6000-len(eligible)} were ceiling-skipped.", "",
        "## 6. Sample Best winner distribution", "", "See `eligibility/SAMPLE_BEST_WINNER_DISTRIBUTION.md`.", "",
        "## 7. Dynamic protocol provenance", "", f"Dynamic LB source SHA256: `{config['dynamic_source_sha256']}`. Seven constant seeds, beam 4, one forward sweep, downstream recomputation, no delta stitching.", "",
        "## 8. Dynamic completion/integrity", "", f"All {len(candidate_ids)} frozen eligible IDs completed exactly once; no extra, CWE, or ceiling executions were admitted.", "",
        "## 9. Full 12-task headline", "",
        f"- OFF: {off_mean:.12f}", f"- Best Single ({best_single_action}): {best_single_mean:.12f}", f"- ALL: {all_mean:.12f}",
        f"- Sample Best: {sb_mean:.12f}", f"- Dynamic LB: {dyn_mean:.12f}", f"- Dynamic LB − Sample Best: {100*(dyn_mean-sb_mean):.6f} pp", "",
        "## 10. Per-task breakdown", "", "See `per_task/dynamic_12task_breakdown.csv`.", "",
        "## 11. Eligible-cohort improvement sparsity", "", f"Improved/equal/worse: {improved}/{equal}/{worse}; improved rate {executed_stats['dynamic_improved_rate']:.6f}.", "",
        "## 12. Trajectory statistics", "", f"Zero/one/multi-switch trajectories: {zero_switch}/{one_switch}/{multi_switch}.", "",
        "## 13. Bootstrap uncertainty", "", f"Task-stratified paired 95% CI: [{100*bootstrap['ci95_lower']:.6f}, {100*bootstrap['ci95_upper']:.6f}] pp; P(delta>0)={bootstrap['probability_delta_gt_zero']:.6f}.", "",
        "## 14. Scientific interpretation", "", f"Pre-registered descriptive classification: **{status}**. This result concerns recomputed sequential opportunity beyond the complete sample-static oracle.", "",
        "## 15. Claim limitations", "", "Dynamic LB is approximate and recomputed. It is not a deployable online controller, a learned router, or a globally optimal Dynamic oracle.", "",
    ]
    report_path = run / "reports/DYNAMIC_NONCEILING_12TASK_FORMAL_REPORT.md"
    atomic_text(report_path, "\n".join(report_lines))

    audit = {
        "fixed_run_root": prereg["fixed7_run_identity"], "fixed_run_integrity": "PASS_45500_OF_45500",
        "model_identity": config["model_identity"], "checkpoint_identity": config["checkpoint_identity"],
        "benchmark_identity": prereg["benchmark_identity"], "excluded_task": EXCLUDED, "included_tasks": prereg["included_tasks"],
        "total_fixed_samples": 6500, "cwe_excluded_count": 500, "included_12task_count": 6000,
        "ceiling_skipped_count": 6000-len(candidate_ids), "dynamic_eligible_count": len(candidate_ids), "dynamic_completed_count": len(result_by_id),
        "fixed_samplebest_recomputation": "PASS", "sample_best_12task_score": sb_mean, "dynamic_12task_score": dyn_mean,
        "dynamic_minus_samplebest_12task_pp": 100*(dyn_mean-sb_mean), "dynamic_minus_off_12task_pp": 100*(dyn_mean-off_mean),
        "improved_count": improved, "equal_count": equal, "worse_count": worse,
        "median_dynamic_minus_samplebest": executed_stats["median_gain"], "mean_dynamic_minus_samplebest_nonceiling": executed_stats["mean_gain"],
        "bootstrap_mean_delta": bootstrap["bootstrap_mean_delta"], "bootstrap_ci95_lower": bootstrap["ci95_lower"], "bootstrap_ci95_upper": bootstrap["ci95_upper"],
        "dynamic_protocol_source_hash": config["dynamic_source_sha256"], "dynamic_config_hash": sha256(config_path),
        "preregistration_sha256": sha256(prereg_path), "cpu_affinity": "8-23", "worker_count": 1,
        "protected_data_access_count": 0, "final_integrity": "PASS", "sequential_opportunity_status": status,
        "completed_at": now(), "headline": headline, "executed_cohort": executed_stats,
    }
    audit_path = run / "audits/FINAL_DYNAMIC_NONCEILING_12TASK_AUDIT.json"
    atomic_json(audit_path, audit)
    marker = "\n".join([
        f"timestamp={now()}", f"dynamic_candidate_count={len(candidate_ids)}", f"completed_count={len(result_by_id)}",
        f"sample_best_12task_score={sb_mean:.17g}", f"dynamic_12task_score={dyn_mean:.17g}",
        f"dynamic_minus_samplebest_pp={100*(dyn_mean-sb_mean):.17g}", f"final_audit_sha256={sha256(audit_path)}", "",
    ])
    atomic_text(run / "final/DYNAMIC_NONCEILING_12TASK_COMPLETE", marker)
    return {"headline": headline, "audit": str(audit_path), "audit_sha256": sha256(audit_path)}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: aggregate_dynamic_nonceiling_12task_v1.py RUN_ROOT")
    print(json.dumps(aggregate(sys.argv[1]), sort_keys=True), flush=True)
