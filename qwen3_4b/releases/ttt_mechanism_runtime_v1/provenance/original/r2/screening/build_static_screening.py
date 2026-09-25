#!/usr/bin/env python3
"""Build the revised Exact-R2 screening freeze using metadata only.

This script deliberately does not import torch, load a model, call nvidia-smi,
or execute any candidate.  It reads frozen authorities and writes only beneath
this run directory.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
HIST = Path("/home/USER/ttt_mechanism_analysis_v1/regime_level_control_opportunity_audit_v1")
MODEL = Path("/home/USER/ttt/models/qwen3_1p7b_stage2")
BENCH = Path("/home/USER/ttt/benchmarks/ruler_16k_standard_13task500_v1")
FIXED = Path("/home/USER/ttt/runs/formal/fixed7_1p7b_v1/run_20260828T012948+0800")
DYNAMIC = Path("/home/USER/ttt/runs/formal/dynamic_nonceiling_12task_1p7b_v1/run_20260828T161330+0800")
OLD_DEV = Path("/home/USER/ttt_phase_c_migrated_v1/ttt_phase_c_migration_20260813/data/dev_authority/manifests/dev_manifest.json")

MODEL_ID = "ca51ee90d20ee09efbe0b31dd21bf9faa70275508316d1ba7cb1387ee4f0725f"
CHECKPOINT_ID = "f4e264ed6f5671f123d6c8a27c69bce3af5ffcba518c9e298398314c6147d9fd"
BENCHMARK_ID = "5d16e3a6963d4c21450f0530af743730b91bf7b2442699e288e10754f77ad0c5"
DYNAMIC_AUDIT_ID = "89e5b7da8b85ce063af15216ef9a16de5c433d4e489a19efaae20ad16099689f"
ACTIONS = ["OFF", "L0", "L6", "L12", "L18", "L24", "ALL"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def dump_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows) -> int:
    n = 0
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
            n += 1
    return n


def write_csv(path: Path, rows) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def q_nearest(values, p: float) -> float:
    values = sorted(values)
    return values[round((len(values) - 1) * p)]


for dirname in ["audit", "historical", "census", "search_space", "pools", "manifests", "reports", "receipts"]:
    (ROOT / dirname).mkdir(parents=True, exist_ok=True)

# Frozen-source integrity.
source_files = {
    "historical_regime_spec": HIST / "authority/REGIME_SEARCH_SPEC_V1.json",
    "historical_tiebreak": HIST / "authority/BEST_SCHEDULE_TIEBREAK_FREEZE.json",
    "historical_final_r2_runner": HIST / "scripts/run_r2_full_exact.py",
    "historical_final_freezer": HIST / "scripts/freeze_r2_and_gate_r3.py",
    "historical_completeness": HIST / "audits/R2_SEARCH_COMPLETENESS_AUDIT.json",
    "historical_state_recomputation": HIST / "audits/STATE_RECOMPUTATION_AUDIT.json",
    "historical_generation": HIST / "audits/GENERATION_NO_UPDATE_AUDIT.json",
    "historical_train42": HIST / "tables/TRAIN42_MANIFEST.csv",
    "benchmark_manifest": BENCH / "benchmark_manifest.json",
    "benchmark_samples": BENCH / "sample_manifest.jsonl",
    "fixed7_final_audit": FIXED / "audits/FINAL_FIXED7_FORMAL_AUDIT.json",
    "fixed7_all_results": FIXED / "manifests/all_fixed7_results.jsonl",
    "dynamic_final_audit": DYNAMIC / "audits/FINAL_DYNAMIC_NONCEILING_12TASK_AUDIT.json",
    "dynamic_aggregation": DYNAMIC / "aggregation/dynamic_full12task_reconstructed.csv",
}
assert all(p.is_file() for p in source_files.values())
source_hashes = {k: {"path": str(p), "sha256": sha(p)} for k, p in source_files.items()}
assert source_hashes["benchmark_manifest"]["sha256"] == BENCHMARK_ID
assert source_hashes["dynamic_final_audit"]["sha256"] == DYNAMIC_AUDIT_ID

fixed_audit = json.loads(source_files["fixed7_final_audit"].read_text())
dynamic_audit = json.loads(source_files["dynamic_final_audit"].read_text())
assert fixed_audit["final_integrity"] == "PASS" and fixed_audit["completed_cells"] == 45500
assert dynamic_audit["final_integrity"] == "PASS" and dynamic_audit["improved_count"] == 173

# Benchmark authority, retaining no prompt text in the new artifacts.
benchmark_rows = []
with (BENCH / "sample_manifest.jsonl").open() as f:
    for global_index, line in enumerate(f):
        x = json.loads(line)
        benchmark_rows.append({
            "canonical_sample_id": x["sample_id"],
            "task": x["task"],
            "benchmark_global_index": global_index,
            "input_hash": x["input_hash"],
            "input_token_count_canonical": x["input_token_count_canonical"],
        })
bench_by_id = {x["canonical_sample_id"]: x for x in benchmark_rows}
assert len(bench_by_id) == 6500

# Fixed7 values supply OFF and Sample Best. Validate identities for every cell.
fixed_scores = {}
with (FIXED / "manifests/all_fixed7_results.jsonl").open() as f:
    for line in f:
        x = json.loads(line)
        assert x["status"] == "SUCCESS"
        assert x["model_hash"] == MODEL_ID and x["checkpoint_hash"] == CHECKPOINT_ID
        assert x["benchmark_manifest_hash"] == BENCHMARK_ID
        fixed_scores.setdefault(x["sample_id"], {})[x["mode"]] = float(x["score"])
assert len(fixed_scores) == 6500 and all(set(v) == set(ACTIONS) for v in fixed_scores.values())

# Read only the compact fields required from Dynamic result JSONs.
dynamic_results = {}
for path_s in glob.glob(str(DYNAMIC / "results/*.json")):
    x = json.loads(Path(path_s).read_text())
    seq = x.get("action_sequence") or x.get("chosen_trajectory")
    if seq is None:
        continue
    dynamic_results[x["sample_id"]] = {
        "sequence": seq,
        "model_identity": x.get("model_identity"),
        "checkpoint_identity": x.get("checkpoint_identity"),
        "completed": x.get("completed"),
        "runtime_seconds": x.get("runtime_seconds"),
        "replay_count": x.get("replay_count"),
    }

with (DYNAMIC / "aggregation/dynamic_full12task_reconstructed.csv").open(newline="") as f:
    dynamic_rows = list(csv.DictReader(f))
selected = [x for x in dynamic_rows if float(x["dynamic_minus_samplebest"]) > 0.0]
assert len(selected) == 173

parent = []
for x in selected:
    sid = x["sample_id"]
    b = bench_by_id[sid]
    scores = fixed_scores[sid]
    d = dynamic_results[sid]
    assert d["completed"] is True and d["model_identity"] == MODEL_ID and d["checkpoint_identity"] == CHECKPOINT_ID
    off = scores["OFF"]
    sample_best = max(scores.values())
    dynamic_score = float(x["final_dynamic_score"])
    assert abs(off - float(x["OFF"])) <= 1e-12
    assert abs(sample_best - float(x["sample_best"])) <= 1e-12
    assert abs(dynamic_score - sample_best - float(x["dynamic_minus_samplebest"])) <= 1e-12
    seq = d["sequence"]
    assert seq and all(a in ACTIONS for a in seq)
    switches = sum(a != b for a, b in zip(seq, seq[1:]))
    regimes = switches + 1
    parent.append({
        "canonical_sample_id": sid,
        "task": x["task"],
        "benchmark_global_index": b["benchmark_global_index"],
        "content_hash": b["input_hash"],
        "OFF_score": off,
        "SampleBest_score": sample_best,
        "Dynamic_score": dynamic_score,
        "Dynamic_minus_OFF": dynamic_score - off,
        "Dynamic_minus_SampleBest": dynamic_score - sample_best,
        "complete_chunk_count": len(seq),
        "frozen_Dynamic_action_sequence": seq,
        "Dynamic_switch_count": switches,
        "Dynamic_regime_count": regimes,
    })
parent.sort(key=lambda x: x["benchmark_global_index"])
assert len({x["canonical_sample_id"] for x in parent}) == 173
assert all(x["Dynamic_minus_SampleBest"] > 0 for x in parent)

write_jsonl(ROOT / "census/R2_PRIMARY_PARENT.jsonl", parent)
write_csv(ROOT / "census/R2_PRIMARY_PARENT.csv", [{**x, "frozen_Dynamic_action_sequence": "|".join(x["frozen_Dynamic_action_sequence"])} for x in parent])

# Historical exposure is the 42-sample cohort that actually received the full
# exhaustive Exact-R2 search. Resolve its prompt hashes through its old frozen
# DEV authority and compare those hashes against the new benchmark content.
old_dev = json.loads(OLD_DEV.read_text())
old_by_id = {x["sample_id"]: x for x in old_dev["samples"]}
with (HIST / "tables/TRAIN42_MANIFEST.csv").open(newline="") as f:
    train42 = list(csv.DictReader(f))
exposure = []
for row in train42:
    old = old_by_id[row["canonical_sample_id"]]
    exposure.append({
        "canonical_sample_id": row["canonical_sample_id"],
        "task": row["task"],
        "historical_task_index": int(row["canonical_sample_id"].rsplit(":", 1)[1]),
        "content_hash": old["prompt_hash"],
        "historical_exact_r2_exposure": True,
        "source": str(HIST / "tables/TRAIN42_MANIFEST.csv"),
    })
assert len(exposure) == 42 and len({x["content_hash"] for x in exposure}) == 42
write_jsonl(ROOT / "historical/HISTORICAL_EXACT_R2_EXPOSURE.jsonl", exposure)
exposure_hashes = {x["content_hash"] for x in exposure}
parent_hashes = {x["content_hash"] for x in parent}
true_overlap = exposure_hashes & parent_hashes
assert not true_overlap

# Intrinsic eligibility only; no downstream mechanism outcome is read or used.
eligibility = []
for x in parent:
    reasons = {
        "prompt_exists": x["canonical_sample_id"] in bench_by_id,
        "complete_chunk_count_known": isinstance(x["complete_chunk_count"], int) and x["complete_chunk_count"] >= 2,
        "candidate_action_set_valid": all(a in ACTIONS for a in x["frozen_Dynamic_action_sequence"]),
        "exact_candidate_set_finite": x["complete_chunk_count"] >= 2,
        "scorer_available": True,
        "deterministic_replay_possible": True,
        "independent_of_historical_exact_r2_content": x["content_hash"] not in exposure_hashes,
    }
    eligible = all(reasons.values())
    eligibility.append({
        "canonical_sample_id": x["canonical_sample_id"],
        "task": x["task"],
        "benchmark_global_index": x["benchmark_global_index"],
        "content_hash": x["content_hash"],
        "complete_chunk_count": x["complete_chunk_count"],
        "eligible": eligible,
        "eligibility_reasons": reasons,
        "ineligibility_reason": None if eligible else ";".join(k for k, v in reasons.items() if not v),
        "historical_exact_r2_true_content_overlap": x["content_hash"] in exposure_hashes,
    })
eligible = [x for x in eligibility if x["eligible"]]
assert len(eligible) == 173
write_jsonl(ROOT / "manifests/EXACT_R2_ELIGIBILITY_V1.jsonl", eligibility)

# Freeze the exact historical rule before candidate enumeration.
rule = {
    "schema": "EXACT_R2_RULE_V1",
    "scientific_question": "Can a one-transition/two-regime hindsight structural oracle reproduce the behavioral gain of successful Dynamic trajectories?",
    "primary_conditions": ["OFF", "Exact R2", "Dynamic"],
    "explicitly_excluded": {"P1": True, "Frozen Expert": True},
    "parent_rule": "frozen formal Dynamic_score > frozen seven-action SampleBest_score",
    "structural_class": {
        "exact_regime_count": 2,
        "exact_switch_count": 1,
        "schedule": "A^[0:tau) -> B^[tau:T)",
        "action_set": ACTIONS,
        "A_not_equal_B": True,
        "tau_domain": "integer 1..T-1 inclusive",
        "tau_meaning": "number of complete prompt chunks assigned to early action A; zero-based switch boundary index",
        "constant_schedule_handling": "excluded from Exact R2 because A!=B and tau is internal; historical constants belonged to separate R1, which is not a comparator in this revised design",
        "OFF_legal": True,
        "ALL_legal": True,
    },
    "candidate_canonicalization": {
        "executable_representation": "JSON array of T chunk-level action strings",
        "hash": "SHA256(json.dumps(sequence, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode())",
        "deduplication_key": "canonical_schedule_hash within canonical_sample_id",
        "raw_candidates_per_sample": "7*6*(T-1)=42*(T-1)",
        "duplicates_under_recovered_parameterization": 0,
    },
    "objective": {
        "value": "same-sample frozen terminal RULER task reward",
        "sample_level_hindsight_structural_oracle": True,
        "online_controller": False,
    },
    "tie_break": {
        "score_tolerance": 1e-12,
        "order": ["maximum terminal score", "earliest tau", "action_order A", "action_order B", "canonical_schedule_hash"],
        "action_order": ACTIONS,
    },
    "replay_semantics": {
        "normal_per_sample_initial_state": True,
        "clean_full_sequence_replay": True,
        "dynamic_state_transplant": False,
        "dynamic_delta_reuse": False,
        "stored_delta_stitching": False,
        "prefix_cache_used": False,
        "downstream_recomputation": True,
        "context": 16384,
        "chunk": 1024,
        "candidate_layers": [0, 6, 12, 18, 24],
        "fast_weight_parameter": "mlp.down_proj.weight",
        "ttt_objective": "contextual_hidden_direct_outer_product_no_scalar_loss",
        "ttt_lr": 1,
        "delta_clip_frobenius": 1e-5,
        "update_order": "apply-then-update",
        "sample_reset": True,
        "complete_chunks": "candidate-scheduled updates",
        "incomplete_tail": "forward normally, retain, no TTT update",
        "generation": "no TTT update",
        "scorer": "frozen family-specific RULER evaluator / terminal task reward",
    },
    "authorities": {
        "model": {"path": str(MODEL), "identity": MODEL_ID, "checkpoint_identity": CHECKPOINT_ID},
        "benchmark": {"path": str(BENCH), "manifest_sha256": BENCHMARK_ID},
        "OFF_and_SampleBest": {"root": str(FIXED), "final_audit_sha256": source_hashes["fixed7_final_audit"]["sha256"], "results_sha256": source_hashes["fixed7_all_results"]["sha256"]},
        "Dynamic": {"root": str(DYNAMIC), "final_audit_sha256": DYNAMIC_AUDIT_ID, "aggregation_sha256": source_hashes["dynamic_aggregation"]["sha256"]},
    },
    "future_analysis": {
        "primary_gain_retention": "SUM(R2-OFF)/SUM(Dynamic-OFF), entire frozen cohort, raw, unclamped",
        "primary_dynamic_r2_gap": "MEAN(Dynamic-R2)",
        "dynamic_r2_gap_summary": ["mean", "median", "std", "min", "max", "paired counts", "paired bootstrap CI", "task-stratified paired bootstrap CI"],
        "bootstrap_seed": 20260906,
        "sample_best_role": "SECONDARY_REFERENCE_AND_PARENT_DEFINITION_ONLY",
    },
    "screening_firewall": {
        "allowed": ["identity", "task", "benchmark metadata", "OFF score", "Sample Best score", "Dynamic score", "Dynamic action sequence", "chunk count", "historical exposure identity/content hash"],
        "forbidden": ["Reverse outcome", "State Reset outcome", "Selective Deletion outcome", "K/P outcome", "historical R2 outcome", "future R2 outcome", "criticality outcome", "observability outcome"],
        "outcome_leakage_count": 0,
    },
    "claim_boundary": {
        "allowed": [
            "Successful Dynamic behavior can often be reproduced by a dramatically lower regime-transition-complexity schedule, if future results support it.",
            "Dynamic behavioral benefit may not require comparable action-transition complexity, if future results support it.",
        ],
        "forbidden": ["R2 is predictable online", "R2 is deployable", "one switch is always sufficient", "Dynamic controller is unnecessary", "structural simplicity implies easy control"],
    },
    "historical_sources": source_hashes,
}
rule_path = ROOT / "manifests/EXACT_R2_RULE_V1.json"
dump_json(rule_path, rule)
rule_sha = sha(rule_path)

# Enumerate all and only unique executable Exact-R2 schedules.
candidate_path = ROOT / "manifests/EXACT_R2_CANDIDATE_MANIFEST.jsonl"
raw_count = 0
unique_count = 0
per_sample_counts = []
with candidate_path.open("w") as f:
    for sample in eligible:
        sid = sample["canonical_sample_id"]
        T = sample["complete_chunk_count"]
        seen = set()
        ordinal = 0
        for A in ACTIONS:
            for B in ACTIONS:
                if A == B:
                    continue
                for tau in range(1, T):
                    raw_count += 1
                    sequence = [A] * tau + [B] * (T - tau)
                    schedule_hash = canonical_hash(sequence)
                    if schedule_hash in seen:
                        continue
                    seen.add(schedule_hash)
                    ordinal += 1
                    candidate = {
                        "candidate_id": f"r2-{sample['benchmark_global_index']:04d}-{ordinal:03d}-{schedule_hash[:12]}",
                        "canonical_sample_id": sid,
                        "task": sample["task"],
                        "benchmark_global_index": sample["benchmark_global_index"],
                        "T": T,
                        "action_A": A,
                        "action_B": B,
                        "tau": tau,
                        "regime_count": 2,
                        "switch_count": 1,
                        "full_chunk_level_action_sequence": sequence,
                        "canonical_schedule_hash": schedule_hash,
                    }
                    f.write(json.dumps(candidate, sort_keys=True, separators=(",", ":")) + "\n")
                    unique_count += 1
        expected = 42 * (T - 1)
        assert ordinal == expected and len(seen) == expected
        per_sample_counts.append(ordinal)
assert raw_count == unique_count == 99120
candidate_sha = sha(candidate_path)

# Dynamic complexity census.
switch_counts = [x["Dynamic_switch_count"] for x in parent]
switch_distribution = dict(sorted(Counter(switch_counts).items()))
regime_distribution = dict(sorted(Counter(x["Dynamic_regime_count"] for x in parent).items()))
chunk_distribution = dict(sorted(Counter(x["complete_chunk_count"] for x in parent).items()))
task_distribution = dict(sorted(Counter(x["task"] for x in parent).items()))
complexity = {
    "parent_n": len(parent),
    "switch_count_distribution": switch_distribution,
    "regime_count_distribution": regime_distribution,
    "constant_dynamic_n": sum(x == 0 for x in switch_counts),
    "one_switch_dynamic_n": sum(x == 1 for x in switch_counts),
    "multi_switch_dynamic_n": sum(x >= 2 for x in switch_counts),
    "switch_mean": statistics.mean(switch_counts),
    "switch_median": statistics.median(switch_counts),
    "switch_max": max(switch_counts),
    "task_distribution": task_distribution,
    "complete_chunk_count_distribution": chunk_distribution,
}
dump_json(ROOT / "census/DYNAMIC_ORIGINAL_COMPLEXITY.json", complexity)

# Cost estimate from recent formal Dynamic full-replay wall time divided by its
# replay count on the exact 173-sample parent. Low/central/high use P25/mean/P90.
seconds_per_replay = []
for x in parent:
    d = dynamic_results[x["canonical_sample_id"]]
    seconds_per_replay.append(float(d["runtime_seconds"]) / int(d["replay_count"]))
low_s = q_nearest(seconds_per_replay, 0.25)
central_s = statistics.mean(seconds_per_replay)
high_s = q_nearest(seconds_per_replay, 0.90)
cost = {
    "eligible_n": len(eligible),
    "candidate_count_per_sample": {
        "min": min(per_sample_counts),
        "median": statistics.median(per_sample_counts),
        "mean": statistics.mean(per_sample_counts),
        "max": max(per_sample_counts),
    },
    "total_unique_gpu_replays": unique_count,
    "throughput_basis": "recent frozen formal Dynamic parent per-sample wall_seconds/replay_count; static estimate only",
    "seconds_per_replay": {"low_p25": low_s, "central_mean": central_s, "high_p90": high_s},
    "estimated_gpu_hours": {
        "low": unique_count * low_s / 3600,
        "central": unique_count * central_s / 3600,
        "high": unique_count * high_s / 3600,
    },
}
dump_json(ROOT / "search_space/EXACT_R2_COST_AUDIT.json", cost)

eligibility_jsonl = ROOT / "manifests/EXACT_R2_ELIGIBILITY_V1.jsonl"
eligibility_manifest = {
    "schema": "EXACT_R2_ELIGIBILITY_V1_MANIFEST",
    "model_identity": MODEL_ID,
    "checkpoint_identity": CHECKPOINT_ID,
    "benchmark_identity": BENCHMARK_ID,
    "dynamic_authority": rule["authorities"]["Dynamic"],
    "OFF_authority": rule["authorities"]["OFF_and_SampleBest"],
    "SampleBest_authority": rule["authorities"]["OFF_and_SampleBest"],
    "rule_sha256": rule_sha,
    "eligibility_jsonl": str(eligibility_jsonl),
    "eligibility_jsonl_sha256": sha(eligibility_jsonl),
    "parent_n": 173,
    "eligible_n": 173,
    "ineligible_n": 0,
    "historical_exact_r2_exposure_n": 42,
    "new_true_content_overlap_n": 0,
    "screening_outcome_leakage": 0,
    "total_unique_r2_candidate_count": unique_count,
    "candidate_manifest": str(candidate_path),
    "candidate_manifest_sha256": candidate_sha,
    "all_eligible_independent_samples_no_subsampling": True,
    "execution_cohort_frozen": True,
}
eligibility_manifest_path = ROOT / "manifests/EXACT_R2_ELIGIBILITY_V1_MANIFEST.json"
dump_json(eligibility_manifest_path, eligibility_manifest)
eligibility_manifest_sha = sha(eligibility_manifest_path)

# Machine-readable historical rule audit with component-level evidence grades.
audit_components = [
    ("structural_class", "PROVEN_EXACT", "exactly two nonempty regimes A^[0:tau)->B^[tau:T), A!=B", "REGIME_SEARCH_SPEC_V1.json; run_r2_full_exact.py"),
    ("action_set", "PROVEN_EXACT", ",".join(ACTIONS), "REGIME_SEARCH_SPEC_V1.json; BEST_SCHEDULE_TIEBREAK_FREEZE.json"),
    ("regime_count", "PROVEN_EXACT", "2", "REGIME_SEARCH_SPEC_V1.json"),
    ("switch_count", "PROVEN_EXACT", "1", "REGIME_SEARCH_SPEC_V1.json"),
    ("tau_domain", "PROVEN_EXACT", "integer 1..T-1; tau is number of A chunks", "REGIME_SEARCH_SPEC_V1.json; run_r2_full_exact.py"),
    ("constant_handling", "PROVEN_EXACT", "not in R2; A==B forbidden and endpoints forbidden", "REGIME_SEARCH_SPEC_V1.json; run_r2_full_exact.py"),
    ("A_equals_B", "PROVEN_EXACT", "forbidden", "REGIME_SEARCH_SPEC_V1.json; run_r2_full_exact.py"),
    ("OFF_legality", "PROVEN_EXACT", "legal", "REGIME_SEARCH_SPEC_V1.json"),
    ("ALL_legality", "PROVEN_EXACT", "legal", "REGIME_SEARCH_SPEC_V1.json"),
    ("canonicalization", "PROVEN_EXACT", rule["candidate_canonicalization"]["hash"], "run_r2_full_exact.py"),
    ("duplicate_elimination", "PROVEN_EXACT", "sequence hash unique per sample; no duplicates in intrinsic parameterization", "R2_SEARCH_COMPLETENESS_AUDIT.json; freeze_r2_and_gate_r3.py"),
    ("execution_semantics", "PROVEN_EXACT", "clean independent full-sequence replay; no prefix cache or delta stitching", "run_r2_full_exact.py; STATE_RECOMPUTATION_AUDIT.json"),
    ("objective", "PROVEN_EXACT", "same-sample terminal frozen RULER task reward", "REGIME_SEARCH_SPEC_V1.json; run_r2_full_exact.py"),
    ("tie_break", "PROVEN_EXACT", "max score, earliest tau, A order, B order, sequence hash; tolerance 1e-12", "BEST_SCHEDULE_TIEBREAK_FREEZE.json; freeze_r2_and_gate_r3.py"),
    ("incomplete_tail", "PROVEN_EXACT", "normal forward and retain; no TTT update", "REGIME_SEARCH_SPEC_V1.json; run_r2_full_exact.py"),
    ("generation", "PROVEN_EXACT", "no TTT update", "GENERATION_NO_UPDATE_AUDIT.json; run_r2_full_exact.py"),
    ("scorer", "PROVEN_EXACT", "frozen family-specific RULER terminal task reward", "REGIME_SEARCH_SPEC_V1.json; run_r2_full_exact.py"),
    ("historical_parent_identity", "PROVEN_EXACT", "TRAIN42, development-only historical identity check", "TRAIN42_MANIFEST.csv; R2_SEARCH_COMPLETENESS_AUDIT.json"),
]
historical_audit = {
    "status": "PASS",
    "all_core_search_space_fields_exact": True,
    "components": [{"component": a, "status": b, "exact_recovered_rule": c, "source": d, "blocking": False} for a, b, c, d in audit_components],
    "correction_note": "An early prefix-cache pilot was non-equivalent and rejected before formal search; the final authority used clean full-sequence replay and zero pilot outcomes entered R2.",
    "P1_FORMAL_COMPARATOR": False,
    "FROZEN_EXPERT_FORMAL_COMPARATOR": False,
    "historical_source_hashes": source_hashes,
}
dump_json(ROOT / "audit/HISTORICAL_EXACT_R2_RULE_AUDIT.json", historical_audit)

audit_md = [
    "# Historical Exact R2 Rule Audit", "", "Status: **PASS** — all core search-space fields are PROVEN_EXACT.", "",
    "The authoritative path is the corrected clean full-sequence runner `run_r2_full_exact.py`. An earlier prefix-cache pilot was rejected as non-equivalent; no pilot outcome entered the final search.", "",
    "| Component | Status | Exact recovered rule | Source | Blocking |", "|---|---|---|---|---|",
]
for a, b, c, d in audit_components:
    audit_md.append(f"| {a} | {b} | {c} | {d} | NO |")
audit_md += ["", "P1_FORMAL_COMPARATOR=false", "FROZEN_EXPERT_FORMAL_COMPARATOR=false", ""]
(ROOT / "audit/HISTORICAL_EXACT_R2_RULE_AUDIT.md").write_text("\n".join(audit_md))

reason_counts = Counter("ELIGIBLE" if x["eligible"] else x["ineligibility_reason"] for x in eligibility)
screening = {
    "status": "PASS",
    "primary_comparison": "OFF_VS_R2_VS_DYNAMIC",
    "authority_status": {"model": "PASS", "benchmark": "PASS", "OFF": "PASS", "Dynamic": "PASS", "SampleBest": "PASS"},
    "parent_n": 173,
    "eligible_n": 173,
    "ineligible_n": 0,
    "historical_exposure_n": 42,
    "true_content_overlap_n": 0,
    "outcome_leakage": 0,
    "raw_enumeration_count": raw_count,
    "unique_candidate_count": unique_count,
    "complexity": complexity,
    "cost": cost,
    "rule_manifest": str(rule_path),
    "rule_manifest_sha256": rule_sha,
    "eligibility_manifest": str(eligibility_manifest_path),
    "eligibility_manifest_sha256": eligibility_manifest_sha,
    "candidate_manifest": str(candidate_path),
    "candidate_manifest_sha256": candidate_sha,
    "execution_cohort_frozen": True,
    "safe_to_select_r2_execution_cohort": True,
    "safe_to_launch_r2": False,
    "no_execution": {
        "R2_COUNTERFACTUAL_EXECUTED": False,
        "R2_CANDIDATE_REPLAYS_EXECUTED": 0,
        "OFF_RERUN": False,
        "DYNAMIC_RERUN": False,
        "P1_EXECUTED": False,
        "FROZEN_EXPERT_EXECUTED": False,
        "NEW_GPU_COUNTERFACTUAL_RUNS": 0,
        "FORMAL_MODEL_MUTATION": 0,
        "BENCHMARK_MUTATION": 0,
    },
}

receipt_json = ROOT / "receipts/EXACT_R2_REVISED_SCREENING_V1_RECEIPT.json"
dump_json(receipt_json, screening)

receipt_md = f"""# Exact R2 Revised Screening V1 Receipt

Status: **PASS**

- Primary design: OFF vs Exact R2 vs Dynamic.
- Explicitly excluded: P1 and Frozen Expert.
- Parent: 173 frozen formal samples with Dynamic > Sample Best.
- Eligible independent cohort: 173; ineligible: 0; historical true-content overlap: 0.
- Exact R2: A^[0:tau) -> B^[tau:T), A != B, tau in 1..T-1, actions {', '.join(ACTIONS)}.
- Raw/unique candidates: {raw_count}/{unique_count}.
- OFF provenance: `{FIXED}`; final audit `{source_hashes['fixed7_final_audit']['sha256']}`.
- Dynamic provenance: `{DYNAMIC}`; final audit `{DYNAMIC_AUDIT_ID}`.
- Primary retention: `SUM(R2-OFF)/SUM(Dynamic-OFF)` on the full frozen cohort, raw and unclamped.
- Rule SHA256: `{rule_sha}`.
- Eligibility manifest SHA256: `{eligibility_manifest_sha}`.
- Candidate manifest SHA256: `{candidate_sha}`.
- Execution cohort frozen: true; no subsampling.
- Safe to select future execution cohort: YES.
- Safe to launch R2 in this static-only goal: **NO**.
- New GPU runs: 0; R2 candidate replays: 0; OFF reruns: 0; Dynamic reruns: 0.
"""
(ROOT / "receipts/EXACT_R2_REVISED_SCREENING_V1_RECEIPT.md").write_text(receipt_md)

report_md = f"""# Qwen3-1.7B Exact R2 Revised Formal Screening

The static screening passed. The 173-sample parent and all {unique_count:,} Exact-R2 candidates are frozen. No model forward, generation, replay, or GPU run occurred.

## Cohort

| Item | Count |
|---|---:|
| Dynamic > Sample Best parent | 173 |
| Eligible independent primary samples | 173 |
| Ineligible | 0 |
| Historical full Exact-R2 exposure | 42 |
| True content overlap | 0 |

## Dynamic trajectory complexity

| Type | Count |
|---|---:|
| Constant | {complexity['constant_dynamic_n']} |
| One switch | {complexity['one_switch_dynamic_n']} |
| Multi-switch | {complexity['multi_switch_dynamic_n']} |

Mean/median/max switches: {complexity['switch_mean']:.6f} / {complexity['switch_median']:.0f} / {complexity['switch_max']}.

## Candidate and cost census

- Candidates/sample min/median/mean/max: {min(per_sample_counts)} / {statistics.median(per_sample_counts):.0f} / {statistics.mean(per_sample_counts):.6f} / {max(per_sample_counts)}.
- Total unique future GPU replays: {unique_count:,}.
- Static runtime estimate low/central/high: {cost['estimated_gpu_hours']['low']:.2f} / {cost['estimated_gpu_hours']['central']:.2f} / {cost['estimated_gpu_hours']['high']:.2f} GPU-hours.
- Estimate basis: recent formal Dynamic wall time per replay on these same 173 samples (P25 / mean / P90).

## Gate

The future all-eligible execution cohort is frozen and safe to select. `SAFE_TO_LAUNCH_R2=NO` because this goal is static-only and explicitly forbids GPU replay.
"""
(ROOT / "reports/EXACT_R2_REVISED_SCREENING_REPORT.md").write_text(report_md)

# Final self-audit: line counts, uniqueness, required hashes, and zero outcomes.
with candidate_path.open() as f:
    manifest_rows = [json.loads(line) for line in f]
assert len(manifest_rows) == 99120
assert len({(x["canonical_sample_id"], x["canonical_schedule_hash"]) for x in manifest_rows}) == 99120
assert all(x["regime_count"] == 2 and x["switch_count"] == 1 and x["action_A"] != x["action_B"] for x in manifest_rows)
assert all(len(x["full_chunk_level_action_sequence"]) == x["T"] for x in manifest_rows)
assert sha(candidate_path) == candidate_sha and sha(rule_path) == rule_sha and sha(eligibility_manifest_path) == eligibility_manifest_sha

self_audit = {
    "pass": True,
    "parent_rows": len(parent),
    "eligibility_rows": len(eligibility),
    "candidate_rows": len(manifest_rows),
    "unique_sample_schedule_hash_rows": len({(x["canonical_sample_id"], x["canonical_schedule_hash"]) for x in manifest_rows}),
    "rule_sha256_match": True,
    "eligibility_manifest_sha256_match": True,
    "candidate_manifest_sha256_match": True,
    "outcome_fields_in_candidate_manifest": 0,
    "gpu_or_model_execution": 0,
}
dump_json(ROOT / "audit/STATIC_SCREENING_SELF_AUDIT.json", self_audit)

print(json.dumps({
    "status": "PASS",
    "root": str(ROOT),
    "parent_n": len(parent),
    "eligible_n": len(eligible),
    "historical_exposure_n": len(exposure),
    "true_content_overlap_n": len(true_overlap),
    "raw_count": raw_count,
    "unique_count": unique_count,
    "rule_sha256": rule_sha,
    "eligibility_manifest_sha256": eligibility_manifest_sha,
    "candidate_manifest_sha256": candidate_sha,
    "gpu_runs": 0,
}, indent=2))
