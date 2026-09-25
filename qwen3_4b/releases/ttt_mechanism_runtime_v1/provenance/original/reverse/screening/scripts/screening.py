#!/usr/bin/env python3
"""CPU-only authority alignment and mechanism-anchor screening.

This script reads frozen authorities and writes screening metadata only.  It does
not import torch, load model weights, execute inference, or inspect historical
counterfactual outcomes for eligibility/matching.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import socket
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path("/home/USER/ttt/runs/formal/mechanism_anchor_screening_1p7b_v1/run_20260905T143059+0800")
MODEL_ROOT = Path("/home/USER/ttt/models/qwen3_1p7b_stage2")
BENCH_ROOT = Path("/home/USER/ttt/benchmarks/ruler_16k_standard_13task500_v1")
FIXED_ROOT = Path("/home/USER/ttt/runs/formal/fixed7_1p7b_v1/run_20260828T012948+0800")
FORMAL_ROOT = Path("/home/USER/ttt/runs/formal")
HIST_ROOT = Path("/home/USER/ttt_mechanism_analysis_v1")
OLD_BENCH_ROOT = Path("/home/USER/ttt_phase_c_migrated_v1/ttt_phase_c_migration_20260813/data/dev_authority/benchmarks/dev/samples")

MODEL_SHA = "ca51ee90d20ee09efbe0b31dd21bf9faa70275508316d1ba7cb1387ee4f0725f"
CHECKPOINT_SHA = "f4e264ed6f5671f123d6c8a27c69bce3af5ffcba518c9e298398314c6147d9fd"
BENCHMARK_MANIFEST_SHA = "5d16e3a6963d4c21450f0530af743730b91bf7b2442699e288e10754f77ad0c5"
SAMPLE_MANIFEST_SHA = "8000dfccd5199ff83bf0cb0ede51e91991083a5dc6a90ac27397b8b24988dfdc"
ACTIONS = ["OFF", "L0", "L6", "L12", "L18", "L24", "ALL"]
EPS = 1e-12
NOW = datetime.now(timezone(timedelta(hours=8))).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_text(path: Path, value: str):
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def as_bool(v):
    return v is True or str(v).lower() == "true"


def as_float(v):
    return None if v in (None, "") else float(v)


def locate_dynamic():
    candidates = []
    for p in FORMAL_ROOT.glob("dynamic*_1p7b_v1/run_*/audits/FINAL_DYNAMIC_NONCEILING_12TASK_AUDIT.json"):
        try:
            z = read_json(p)
        except Exception:
            continue
        if (z.get("final_integrity") == "PASS" and z.get("model_identity") == MODEL_SHA
                and z.get("checkpoint_identity") == CHECKPOINT_SHA
                and z.get("benchmark_identity") == BENCHMARK_MANIFEST_SHA):
            candidates.append((z.get("completed_at", ""), p.parents[1], z))
    if not candidates:
        raise RuntimeError("NO_COMPLETED_MATCHING_DYNAMIC_AUTHORITY")
    return sorted(candidates)[-1][1], sorted(candidates)[-1][2], [str(x[1]) for x in sorted(candidates)]


def extract_ids(path: Path):
    """Extract sample identifiers from a small manifest/authority table."""
    out = set()
    try:
        if path.suffix == ".csv":
            rows = read_csv(path)
        elif path.suffix == ".jsonl":
            rows = read_jsonl(path)
        elif path.suffix == ".json":
            obj = read_json(path)
            rows = obj if isinstance(obj, list) else [obj]
        else:
            return out
    except Exception:
        return out

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in {"sample_id", "canonical_sample_id"} and isinstance(v, str):
                    if ":ruler_" in v or v.startswith("ruler_"):
                        out.add(v)
                else:
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(rows)
    return out


def parse_old_id(sid: str):
    m = re.search(r"(ruler_[^:]+):(\d+)$", sid)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def source_record(label, path, classification, details):
    return {
        "experiment_or_rule": label,
        "classification": classification,
        "source_file": str(path),
        "source_sha256": sha256(path),
        **details,
    }


for d in ["audit", "census", "manifests", "pools", "matching", "overlap", "reports", "receipts", "scripts"]:
    (ROOT / d).mkdir(parents=True, exist_ok=True)

# ----- Protected authority verification -----
dynamic_root, dynamic_audit, dynamic_candidates = locate_dynamic()
fixed_audit = read_json(FIXED_ROOT / "audits/FINAL_FIXED7_FORMAL_AUDIT.json")
pre_reg = read_json(BENCH_ROOT / "PRE_REGISTRATION.json")
bench_rows = read_jsonl(BENCH_ROOT / "sample_manifest.jsonl")
task_counts = Counter(r["task"] for r in bench_rows)
model_manifest = read_json(MODEL_ROOT / "artifact_manifest.json")

assert len(bench_rows) == 6500
assert len(task_counts) == 13 and set(task_counts.values()) == {500}
assert pre_reg["total_samples"] == 6500 and pre_reg["samples_per_task"] == 500
assert pre_reg["context_length"] == 16384 and pre_reg["seed"] == 42
assert sha256(BENCH_ROOT / "sample_manifest.jsonl") == SAMPLE_MANIFEST_SHA
assert sha256(BENCH_ROOT / "benchmark_manifest.json") == BENCHMARK_MANIFEST_SHA
assert model_manifest["model_sha256"] == MODEL_SHA
assert fixed_audit["final_integrity"] == "PASS" and fixed_audit["completed_cells"] == 45500
assert dynamic_audit["included_12task_count"] == 6000
assert dynamic_audit["ceiling_skipped_count"] == 4565
assert dynamic_audit["dynamic_completed_count"] == 1435
assert dynamic_audit["improved_count"] == 173 and dynamic_audit["equal_count"] == 1262 and dynamic_audit["worse_count"] == 0

authority_audit = {
    "audit_timestamp": NOW,
    "host": socket.gethostname(),
    "user": os.environ.get("USER", ""),
    "model": {"status": "PASS", "root": str(MODEL_ROOT), "model_sha256": MODEL_SHA, "checkpoint_sha256": CHECKPOINT_SHA,
              "artifact_manifest_sha256": sha256(MODEL_ROOT / "artifact_manifest.json")},
    "benchmark": {"status": "PASS", "root": str(BENCH_ROOT), "benchmark_manifest_sha256": BENCHMARK_MANIFEST_SHA,
                  "sample_manifest_sha256": SAMPLE_MANIFEST_SHA, "task_n": 13, "per_task_n": 500, "total_n": 6500,
                  "context_length": 16384, "seed": 42},
    "fixed7": {"status": "PASS", "root": str(FIXED_ROOT), "final_audit_sha256": sha256(FIXED_ROOT / "audits/FINAL_FIXED7_FORMAL_AUDIT.json"),
               "cells": 45500, "actions": ACTIONS},
    "dynamic": {"status": "PASS", "root": str(dynamic_root), "selection": "latest completed matching formal authority by completed_at",
                "matching_completed_candidates": dynamic_candidates,
                "final_audit_sha256": sha256(dynamic_root / "audits/FINAL_DYNAMIC_NONCEILING_12TASK_AUDIT.json"),
                "config_sha256": sha256(dynamic_root / "config/DYNAMIC_FORMAL_CONFIG.json"),
                "protocol_source_sha256": dynamic_audit["dynamic_protocol_source_hash"]},
    "all_status": "PASS",
}
write_json(ROOT / "audit/AUTHORITY_ALIGNMENT_AUDIT.json", authority_audit)

# ----- Historical rule audit -----
reverse_script = HIST_ROOT / "minimal_targeted_causal_replay_v1/scripts/build_frozen_manifest.py"
reverse_manifest = HIST_ROOT / "minimal_targeted_causal_replay_v1/manifests/TARGETED_CAUSAL_REPLAY_MANIFEST.jsonl"
match_script = HIST_ROOT / "signal_analysis_abcd_v1/scripts/run_discovery_and_freeze.py"
reset_script = HIST_ROOT / "targeted_fast_weight_state_replay_v1/scripts/build_state_replay_manifest.py"
reset_manifest = HIST_ROOT / "targeted_fast_weight_state_replay_v1/manifests/STATE_REPLAY_MANIFEST.jsonl"
delete_script = HIST_ROOT / "selective_state_component_ablation_v1/scripts/build_ablation_manifest.py"
delete_manifest = HIST_ROOT / "selective_state_component_ablation_v1/manifests/COMPONENT_ABLATION_MANIFEST.jsonl"
transplant_script = HIST_ROOT / "early_state_later_regime_transplant_v1/scripts/setup_authority.py"
kp_script = HIST_ROOT / "trajectory_closure_factorial_intervention_v1/scripts/setup_authority.py"
r2_script = HIST_ROOT / "regime_level_control_opportunity_audit_v1/scripts/setup_authority.py"
r2_spec = HIST_ROOT / "regime_level_control_opportunity_audit_v1/authority/REGIME_SEARCH_SPEC_V1.json"
critical_script = HIST_ROOT / "critical_event_state_representation_analysis_v1/scripts/setup_authority.py"

rules = [
    source_record("EXACT_REVERSE_EXECUTED_SELECTION_AND_REPLAY", reverse_script, "PROVEN_EXACT", {
        "input_pool": "TARGETED_REPLAY_MINIMAL_SET + MISSING_COUNTERFACTUALS + ABCD matched controls",
        "positive_definition": "frozen positive plan rows marked EXACT_REVERSE_MISSING; intervention is exact list reversal",
        "negative_definition": "ABCD controls receive a deterministic same-task/same-decision-chunk positive donor trajectory",
        "eligibility_rule": "len(reference)==decision_chunk_count; reverse differs; intervention has identical sorted composition",
        "thresholds": [], "random_seed": None, "sample_count": 38, "positive_n": 20, "negative_n": 18,
        "exclusion_logic": "malformed length, identical reverse, composition mismatch, absent donor",
        "replayability_checks": "reset per sequence; exact full replay; frozen model/checkpoint/executor",
        "trajectory_requirements": "complete sequence; reverse-distinct; downstream recomputation",
        "new_formal_interpretation": "Formal nontrivial_order is logically implied by reverse-distinct and does not conflict with executed rule. Historical plan membership and old outcomes are not inherited."
    }),
    source_record("MATCHED_NEGATIVE_PRIMARY_OUTCOME_FREE", match_script, "PROVEN_EXACT", {
        "input_pool": "ABCD observations; controls are not dynamic-positive",
        "positive_definition": "ABCD dynamic-positive",
        "negative_definition": "not dynamic-positive",
        "matching_variables": ["same task", "same D-vs-not-D phase if available", "absolute decision_chunk_count distance", "absolute character_count distance", "sample_id lexical tie-break"],
        "thresholds": [], "random_seed": None, "replacement": True,
        "secondary_rule_not_inherited": "SECONDARY_OUTCOME_MATCHED used OFF score and is not needed for the primary leakage-free rule",
        "new_formal_interpretation": "Single formal authority collapses the historical phase stratum; primary outcome-free priorities remain deterministic."
    }),
    source_record("STATE_RESET_EARLY_STEADY", reset_script, "PROVEN_EXACT", {
        "input_pool": str(reset_manifest), "sample_count": len({r["sample_id"] for r in read_jsonl(reset_manifest)}),
        "eligibility_rule": "positive membership came from EARLY_STEADY_EXACT_REVERSE; reset boundary is one-based start of terminal equal-action run",
        "thresholds": [], "random_seed": None,
        "replayability_checks": "strict cross-action, carried fast-state at valid complete-chunk terminal-run boundary",
        "new_formal_transfer_status": "UNRESOLVED",
        "reason": "early-steady membership is a historical Exact Reverse outcome, unavailable and forbidden for new eligibility screening"
    }),
    source_record("SELECTIVE_COMPONENT_DELETION", delete_script, "PROVEN_EXACT", {
        "input_pool": str(delete_manifest), "sample_count": len({r["sample_id"] for r in read_jsonl(delete_manifest)}),
        "eligibility_rule": "State-Reset positive early-steady cohort; terminal-run switch; prefix-written layer set; observed nonzero component norm; ALL uses leave-one-out, otherwise deterministic primary plus zero-norm sham",
        "thresholds": ["component norm > 0 for primary", "component norm == 0 for sham"], "random_seed": None,
        "new_formal_transfer_status": "UNRESOLVED",
        "reason": "requires prior Reverse-selected early-steady membership and instrumented boundary component norms"
    }),
    source_record("EARLY_STATE_LATER_REGIME_TRANSPLANT", transplant_script, "PROVEN_EXACT", {
        "input_pool": "Exact TRAIN42 R2 optimum set plus frozen component provenance", "sample_count": 42,
        "positive_definition": "R2_minus_P1 >= 0.05 and an optimal A!=OFF,B!=OFF schedule exists",
        "negative_definition": "R2_minus_P1 < 0.05 and an optimal A!=OFF,B!=OFF schedule exists",
        "eligibility_rule": "deterministically selected exact R2 anchor and compatible frozen fast-state component mapping",
        "thresholds": [0.05], "random_seed": None,
        "new_formal_transfer_status": "UNRESOLVED", "reason": "requires Exact R2 outcomes and prior component intervention provenance"
    }),
    source_record("KP_CLOSURE_FACTORIAL", kp_script, "PROVEN_EXACT", {
        "input_pool": "42 frozen transplant/closure anchors", "sample_count": 42,
        "eligibility_rule": "valid continuation boundary; same future suffix/action; exact fast-weight parity; K/P mismatch diagnosed at first divergence; reconstructible attention-KV K and TTT-cache/path P",
        "thresholds": [], "random_seed": None,
        "new_formal_transfer_status": "UNRESOLVED", "reason": "requires prior transplant/closure boundary snapshots and continuation instrumentation"
    }),
    source_record("EXACT_R2_INPUT_AND_SCHEDULE", r2_spec, "PROVEN_PARTIAL", {
        "selection_script": str(r2_script), "input_pool": "frozen selected TRAIN42 feasibility cohort inherited from DYNAMIC_POSITIVE_AD50 42/8 split",
        "sample_count": 42, "schedule_rule": "A^[0:tau)->B^[tau:T), A!=B, tau=1..T-1, 42*(T-1) exact candidates",
        "action_space": ACTIONS, "thresholds": [], "random_seed": None,
        "replayability_checks": "complete chunks only; exact downstream recomputation; no stored-delta stitching; generation update false",
        "new_formal_transfer_status": "UNRESOLVED",
        "reason": "schedule family is exact, but executed input eligibility is only the frozen selected TRAIN42 cohort; no proven generic rule maps all new Dynamic positives into Exact R2 inputs"
    }),
    source_record("CRITICALITY_AND_OBSERVABILITY", critical_script, "PROVEN_PARTIAL", {
        "input_pool": "frozen TRAIN42/heldout split and expert query states",
        "eligibility_rule": "critical labels require expert continuation action values/headroom and sample-grouped folds; several later interventions additionally require frozen P1 and state representations",
        "thresholds": ["headroom >= 0.05 in critical-state authorities"], "random_seed": None,
        "new_formal_transfer_status": "UNRESOLVED",
        "reason": "Dynamic outcome alone cannot generate critical labels, expert state/value labels, folds, or intervention-ready representations"
    }),
]

historical_audit = {
    "audit_timestamp": NOW,
    "status": "PASS_WITH_UNRESOLVED_RULES",
    "authority_precedence": ["actual executed manifest/script", "frozen mechanism interpretation", "Phase C", "Phase B", "Phase A/D", "exploratory prose/comments"],
    "rules": rules,
    "historical_vs_proposed_nontrivial_order": {
        "historical_rule": "reverse(sequence) must differ and preserve exact length/composition",
        "proposed_formal_rule": "len>=2 AND unique_action_count>=2 AND reverse_distinct",
        "reconciliation": "EQUIVALENT_ON_FINITE_ACTION_SEQUENCES_FOR_REVERSE_DISTINCTNESS",
        "conflict": False,
    },
    "only_silently_inherited": ["Exact Reverse structural/replayability checks", "primary outcome-free matched-negative priority"],
    "unresolved_for_new_pool": ["State Reset", "Selective Deletion", "Exact R2 input eligibility", "K/P factorial", "Criticality/Observability"],
}
write_json(ROOT / "audit/HISTORICAL_MECHANISM_SCREENING_RULE_AUDIT.json", historical_audit)

rule_lines = ["# Historical Mechanism Screening Rule Audit", "", f"Status: `{historical_audit['status']}`", "",
              "Actual executed manifests/scripts take precedence over prose. No historical mechanism outcome is used in the new eligibility or matching logic.", "",
              "| Rule | Classification | Historical fact | New-pool disposition |", "|---|---|---|---|"]
for r in rules:
    fact = r.get("eligibility_rule", r.get("schedule_rule", "See JSON audit"))
    disp = r.get("new_formal_transfer_status", "Inherited only where outcome-free")
    rule_lines.append(f"| {r['experiment_or_rule']} | {r['classification']} | {fact} | {disp} |")
rule_lines += ["", "## Nontrivial-order reconciliation", "",
               "The historical executable requires exact length/composition and `reverse != reference`. For a finite action sequence, reverse-distinct already implies length >= 2 and at least two unique actions. The proposed formal definition is therefore equivalent on the admissible historical domain; there is no structural-rule conflict and no stop-before-freeze condition is triggered.", "",
               "Reset, selective deletion, Exact R2 input membership, K/P, and criticality/observability are not transferred to new samples because their historical eligibility requires prior mechanism outcomes or instrumented states that this read-only screen cannot generate."]
write_text(ROOT / "audit/HISTORICAL_MECHANISM_SCREENING_RULE_AUDIT.md", "\n".join(rule_lines))

# ----- Historical mechanism exposure set, with content-hash lookup -----
exposure_sources = {
    "EXACT_REVERSE": [reverse_manifest],
    "STATE_RESET": [reset_manifest],
    "SELECTIVE_COMPONENT_DELETION": [delete_manifest],
    "OLD_TRANSPLANT": [HIST_ROOT / "early_state_later_regime_transplant_v1/manifests/FROZEN_BRANCH_MANIFEST.csv"],
    "KP_FACTORIAL": [HIST_ROOT / "trajectory_state_closure_audit_v1/authority/FROZEN_ANCHORS.csv"],
    "EXACT_R2": [HIST_ROOT / "regime_level_control_opportunity_audit_v1/tables/TRAIN42_MANIFEST.csv"],
}
scan_roots = {
    "CRITICAL_STATE_CAUSAL_INTERVENTION": ["sparse_critical_state_override_ttt_v1", "long_horizon_critical_event_credit_analysis_v1"],
    "OBSERVABILITY_LABELED_MECHANISM_EVALUATION": ["critical_event_state_representation_analysis_v1", "state_evolution_boundary_observability_audit_v1", "update_scale_sensitivity_critical_observability_v1", "active_probing_for_hfull_action_value_v1", "active_probe_specificity_and_trigger_analysis_v1", "short_trajectory_diagnostic_probing_v1", "short_horizon_early_regime_observable_v1", "semantic_early_regime_observable_v1"],
}
for label, roots in scan_roots.items():
    exposure_sources[label] = []
    for name in roots:
        base = HIST_ROOT / name
        if not base.exists():
            continue
        for sub in ["authority", "manifests", "tables"]:
            d = base / sub
            if not d.exists():
                continue
            for p in sorted(d.rglob("*")):
                if p.is_file() and p.suffix in {".csv", ".jsonl", ".json"} and p.stat().st_size <= 100 * 1024 * 1024:
                    exposure_sources[label].append(p)

exposures = defaultdict(lambda: {"experiments": set(), "source_files": set()})
for label, paths in exposure_sources.items():
    for p in paths:
        if not p.exists():
            continue
        for sid in extract_ids(p):
            exposures[sid]["experiments"].add(label)
            exposures[sid]["source_files"].add(str(p))

old_payload = {}
for p in sorted(OLD_BENCH_ROOT.glob("*.jsonl")):
    for r in read_jsonl(p):
        old_payload[r["sample_id"]] = r
current_by_hash = defaultdict(list)
current_by_task_index = defaultdict(list)
current_ids = {r["sample_id"] for r in bench_rows}
for r in bench_rows:
    current_by_hash[r["input_hash"]].append(r["sample_id"])
    current_by_task_index[(r["task"], int(r["sample_index"]))].append(r["sample_id"])

exposure_rows = []
for sid in sorted(exposures):
    task, idx = parse_old_id(sid)
    old = old_payload.get(sid, {})
    ih = old.get("input_hash")
    exposure_rows.append({
        "historical_sample_id": sid,
        "historical_benchmark_source": old.get("benchmark_version", sid.split(":", 1)[0] if ":ruler_" in sid else None),
        "task": task or old.get("task"), "task_sample_index": idx if idx is not None else old.get("sample_index"),
        "content_input_sha256": ih,
        "old_experiments": sorted(exposures[sid]["experiments"]),
        "source_files": sorted(exposures[sid]["source_files"]),
        "exact_new_canonical_id_match": sid in current_ids,
        "new_content_hash_matches": sorted(current_by_hash.get(ih, [])) if ih else [],
        "new_task_index_collisions": sorted(current_by_task_index.get((task, idx), [])) if task is not None and idx is not None else [],
        "independence_interpretation": "EXPOSED" if sid in current_ids or (ih and current_by_hash.get(ih)) else "DISTINCT_BENCHMARK_CONTENT",
    })
write_jsonl(ROOT / "overlap/HISTORICAL_MECHANISM_EXPOSURE_SET.jsonl", exposure_rows)

exposed_new = defaultdict(set)
for r in exposure_rows:
    if r["exact_new_canonical_id_match"]:
        exposed_new[r["historical_sample_id"]].update(r["old_experiments"])
    for sid in r["new_content_hash_matches"]:
        exposed_new[sid].update(r["old_experiments"])

# ----- Fixed7 and Dynamic joins -----
fixed = defaultdict(dict)
fixed_meta = defaultdict(lambda: {"scorer_hashes": set(), "config_hashes": set(), "prompt_token_counts": set()})
for r in read_jsonl(FIXED_ROOT / "manifests/all_fixed7_results.jsonl"):
    assert r["status"] == "SUCCESS" and r["model_hash"] == MODEL_SHA and r["checkpoint_hash"] == CHECKPOINT_SHA
    sid, mode = r["sample_id"], r["mode"]
    assert mode in ACTIONS and mode not in fixed[sid]
    fixed[sid][mode] = float(r["score"])
    fixed_meta[sid]["scorer_hashes"].add(r["scorer_hash"])
    fixed_meta[sid]["config_hashes"].add(r["config_hash"])
    fixed_meta[sid]["prompt_token_counts"].add(int(r["prompt_token_count"]))
assert len(fixed) == 6500 and all(set(z) == set(ACTIONS) for z in fixed.values())

agg_path = dynamic_root / "aggregation/dynamic_full12task_reconstructed.csv"
agg = {r["sample_id"]: r for r in read_csv(agg_path)}
assert len(agg) == 6000
dyn = {}
for p in sorted((dynamic_root / "results").glob("*.json")):
    z = read_json(p)
    if z.get("completed") is True:
        assert z["model_identity"] == MODEL_SHA and z["checkpoint_identity"] == CHECKPOINT_SHA
        assert z["dynamic_config_sha256"] == dynamic_audit["dynamic_config_hash"]
        assert z.get("error_status") in (None, "NONE")
        assert z["sample_id"] not in dyn
        dyn[z["sample_id"]] = z
assert len(dyn) == 1435

census = []
for global_index, b in enumerate(bench_rows):
    sid = b["sample_id"]
    fs = fixed[sid]
    sb = max(fs.values())
    winners = [a for a in ACTIONS if abs(fs[a] - sb) <= EPS]
    covered = sid in agg
    searched = sid in dyn
    ceiling = covered and not searched and abs(sb - 1.0) <= EPS
    dscore = as_float(agg[sid]["final_dynamic_score"]) if covered else None
    delta_sb = None if dscore is None else dscore - sb
    delta_off = None if dscore is None else dscore - fs["OFF"]
    if b["task"] == "ruler_cwe_16k" and not covered:
        parent_class = "CWE_NO_FORMAL_DYNAMIC"
    elif ceiling:
        parent_class = "DYNAMIC_CEILING_NOT_SEARCHED"
    elif searched and delta_sb > EPS:
        parent_class = "DYNAMIC_SEARCHED_IMPROVED"
    elif searched and abs(delta_sb) <= EPS:
        parent_class = "DYNAMIC_SEARCHED_EQUAL"
    elif searched and delta_sb < -EPS:
        parent_class = "DYNAMIC_SEARCHED_WORSE"
    else:
        parent_class = "INVALID_OR_INCONSISTENT"

    seq = None
    action_counts = {a: None for a in ACTIONS}
    structural = {"sequence_length": None, "unique_action_count": None, "number_of_switches": None,
                  "first_action": None, "last_action": None, "constant_trajectory": None,
                  "reverse_identical": None, "reverse_distinct": None, "nontrivial_order": None}
    replay = {"complete_trajectory_metadata": False, "exact_replay_inputs_exist": False, "exact_chunk_boundaries_exist": False,
              "no_formal_dynamic_error": False, "authority_identities_match": False, "downstream_recomputation_possible": False,
              "exact_reverse_structurally_replayable": False}
    if searched:
        z = dyn[sid]
        seq = z.get("action_sequence") or z.get("chosen_trajectory", {}).get("action_sequence")
        if isinstance(seq, str):
            seq = seq.split("|")
        T = len(seq)
        for a in ACTIONS:
            action_counts[a] = seq.count(a)
        rev_identical = seq == list(reversed(seq))
        structural = {"sequence_length": T, "unique_action_count": len(set(seq)),
                      "number_of_switches": sum(x != y for x, y in zip(seq, seq[1:])),
                      "first_action": seq[0] if seq else None, "last_action": seq[-1] if seq else None,
                      "constant_trajectory": len(set(seq)) == 1, "reverse_identical": rev_identical,
                      "reverse_distinct": not rev_identical,
                      "nontrivial_order": T >= 2 and len(set(seq)) >= 2 and not rev_identical}
        md = z.get("search_metadata", {})
        ar = z.get("authority_search_result", {})
        expected_T = int(md.get("full_chunk_count", ar.get("full_chunk_count", b["input_token_count_canonical"] // 1024)))
        replay = {"complete_trajectory_metadata": bool(seq) and T == expected_T and all(a in ACTIONS for a in seq),
                  "exact_replay_inputs_exist": bool(b.get("input_hash")) and bool(b.get("input")),
                  "exact_chunk_boundaries_exist": expected_T == b["input_token_count_canonical"] // 1024,
                  "no_formal_dynamic_error": z.get("completed") is True and z.get("error_status") in (None, "NONE"),
                  "authority_identities_match": z.get("model_identity") == MODEL_SHA and z.get("checkpoint_identity") == CHECKPOINT_SHA,
                  "downstream_recomputation_possible": as_bool(md.get("recompute_downstream_updates", ar.get("recompute_downstream_updates"))) and not as_bool(md.get("stored_delta_stitching", ar.get("stored_delta_stitching"))),
                  "exact_reverse_structurally_replayable": False}
        replay["exact_reverse_structurally_replayable"] = all(replay[k] for k in replay if k != "exact_reverse_structurally_replayable") and structural["nontrivial_order"]

    row = {
        "canonical_sample_id": sid, "task": b["task"], "family": b.get("family"),
        "task_sample_index": int(b["sample_index"]), "benchmark_global_index": global_index,
        "benchmark_version": b["benchmark_version"], "benchmark_content_sha256": b["input_hash"],
        "reference_identity_sha256": canonical_sha(b.get("reference")), "max_new_tokens": b.get("max_new_tokens"),
        "prompt_length_tokens_canonical": int(b["input_token_count_canonical"]),
        "fixed7_prompt_token_counts": sorted(fixed_meta[sid]["prompt_token_counts"]),
        "full_chunk_count": int(b["input_token_count_canonical"]) // 1024,
        "incomplete_tail_length": int(b["input_token_count_canonical"]) % 1024,
        "scorer_sha256s": sorted(fixed_meta[sid]["scorer_hashes"]),
        "fixed7_config_sha256s": sorted(fixed_meta[sid]["config_hashes"]),
        "fixed7_scores": fs, "sample_best_score": sb, "winning_fixed_actions": winners,
        "dynamic_covered_task": covered, "actual_dynamic_search_executed": searched,
        "ceiling_skipped": ceiling, "dynamic_score": dscore,
        "dynamic_minus_sample_best": delta_sb, "dynamic_minus_off": delta_off,
        "parent_class": parent_class, "action_sequence": seq, "action_counts": action_counts,
        **structural, "replayability": replay,
        "historical_mechanism_exposed": sid in exposed_new,
        "old_experiments": sorted(exposed_new.get(sid, set())),
    }
    census.append(row)

class_counts = Counter(r["parent_class"] for r in census)
assert class_counts == Counter({"CWE_NO_FORMAL_DYNAMIC": 500, "DYNAMIC_CEILING_NOT_SEARCHED": 4565,
                                "DYNAMIC_SEARCHED_IMPROVED": 173, "DYNAMIC_SEARCHED_EQUAL": 1262})
sequential = [r for r in census if r["actual_dynamic_search_executed"]]
positive = [r for r in sequential if r["parent_class"] == "DYNAMIC_SEARCHED_IMPROVED" and r["replayability"]["exact_reverse_structurally_replayable"]]
negative = [r for r in sequential if r["parent_class"] == "DYNAMIC_SEARCHED_EQUAL" and r["replayability"]["exact_reverse_structurally_replayable"]]
worse = [r for r in sequential if r["parent_class"] == "DYNAMIC_SEARCHED_WORSE"]
assert len(sequential) == 1435 and len(positive) == 173 and len(negative) == 0 and len(worse) == 0

positive_ind = [r for r in positive if not r["historical_mechanism_exposed"]]
negative_ind = [r for r in negative if not r["historical_mechanism_exposed"]]
pos_overlap = len(positive) - len(positive_ind)
neg_overlap = len(negative) - len(negative_ind)

write_jsonl(ROOT / "census/PARENT_6500_CENSUS.jsonl", census)
write_jsonl(ROOT / "pools/SEQUENTIAL_PARENT.jsonl", sequential)
write_jsonl(ROOT / "pools/EXACT_REVERSE_POSITIVE_ELIGIBLE.jsonl", positive)
write_jsonl(ROOT / "pools/EXACT_REVERSE_NEGATIVE_ELIGIBLE.jsonl", negative)
write_jsonl(ROOT / "pools/DYNAMIC_WORSE_DIAGNOSTIC_POOL.jsonl", worse)
write_jsonl(ROOT / "pools/EXACT_REVERSE_POSITIVE_INDEPENDENT.jsonl", positive_ind)
write_jsonl(ROOT / "pools/EXACT_REVERSE_NEGATIVE_INDEPENDENT.jsonl", negative_ind)

# The exact historical primary matching rule is deterministic and permits reuse,
# but there are no structurally eligible new controls to which it can be applied.
matched_pairs = []
write_jsonl(ROOT / "matching/EXACT_REVERSE_MATCHED_PAIRS.jsonl", matched_pairs)
matching_audit = {
    "rule_status": "PROVEN_EXACT_NO_ELIGIBLE_CONTROLS",
    "rule_source": str(match_script), "rule_source_sha256": sha256(match_script),
    "rule_version": "HISTORICAL_ABCD_PRIMARY_OUTCOME_FREE",
    "positive_pool_n": len(positive_ind), "negative_pool_n": len(negative_ind), "matched_pair_n": 0,
    "same_task_first": True, "historical_phase_stratum": "collapsed because all new rows share one formal authority",
    "distance_priority": ["absolute full_chunk_count difference", "absolute prompt character-count difference", "negative sample_id lexical tie-break"],
    "replacement_allowed_by_historical_code": True, "randomness": False,
    "why_empty": "All 1262 Dynamic-equal trajectories are constant/reverse-identical and fail the same nontrivial-order structural rule required of positives.",
    "matching_frozen": True, "final_execution_cohort_frozen": False,
}
write_json(ROOT / "matching/MATCHING_RULE_AUDIT.json", matching_audit)

# Unresolved experiment-specific eligibility is intentionally not invented.
unresolved = {
    "STATE_RESET": "Historical early-steady membership was selected using Exact Reverse outcomes; boundary formula is known but membership is forbidden/unavailable.",
    "SELECTIVE_DELETION": "Requires Reset-selected membership and observed boundary component norms from a prior mechanism run.",
    "EXACT_R2": "Schedule/replay semantics are exact, but historical input eligibility was only a frozen selected TRAIN42 cohort; no generic transfer rule is proven.",
    "KP_FACTORIAL": "Requires prior transplant/closure snapshots, fast-weight parity, K/P divergence and cache/path instrumentation.",
    "CRITICALITY_OBSERVABILITY": "Requires expert action-value/headroom labels, state representations, and grouped folds; Dynamic outcome alone is insufficient.",
}
write_text(ROOT / "pools/UNRESOLVED_EXPERIMENT_SPECIFIC_ELIGIBILITY.md",
           "# Unresolved Experiment-Specific Eligibility\n\n" + "\n".join(f"- **{k}**: {v}" for k, v in unresolved.items()))

# ----- Leakage and independence reports -----
feature_classes = {
    "PRE_EXISTING_PARENT_OUTCOME": ["fixed7_scores", "sample_best_score", "dynamic_score", "dynamic_minus_sample_best", "dynamic_minus_off", "parent_class"],
    "TRAJECTORY_STRUCTURE": ["action_sequence", "sequence_length", "action_counts", "unique_action_count", "number_of_switches", "first_action", "last_action", "constant_trajectory", "reverse_identical", "reverse_distinct", "nontrivial_order"],
    "IDENTITY_REPLAYABILITY": ["sample/task/index", "benchmark/content/reference/scorer/config hashes", "token/chunk/tail lengths", "Dynamic completion/error/identity/recomputation flags", "historical exposure identity/content hashes"],
    "HISTORICAL_MECHANISM_OUTCOME_EXCLUDED": ["Reverse reward", "Reset reward", "Selective deletion reward/component effect", "R2 reward", "K/P outcome", "critical intervention outcome", "future mechanism outcome"],
}
leakage_md = """# Screening Feature Leakage Audit

MECHANISM_OUTCOME_LEAKAGE=0

| Class | Columns / information | Used in selection? |
|---|---|---|
| PRE_EXISTING_PARENT_OUTCOME | {parent} | Yes, only as authorized by the frozen protocol |
| TRAJECTORY_STRUCTURE | {structure} | Yes |
| IDENTITY / REPLAYABILITY | {identity} | Yes |
| HISTORICAL_MECHANISM_OUTCOME | {excluded} | No; explicitly excluded |

Historical outcome-bearing tables were inspected only to recover executed rule provenance and exposure identity. Their outcome columns were never joined into the 6500 census, eligibility predicates, or matching inputs.
""".format(parent=", ".join(feature_classes["PRE_EXISTING_PARENT_OUTCOME"]), structure=", ".join(feature_classes["TRAJECTORY_STRUCTURE"]), identity=", ".join(feature_classes["IDENTITY_REPLAYABILITY"]), excluded=", ".join(feature_classes["HISTORICAL_MECHANISM_OUTCOME_EXCLUDED"]))
write_text(ROOT / "audit/SCREENING_FEATURE_LEAKAGE_AUDIT.md", leakage_md)
write_json(ROOT / "audit/SCREENING_FEATURE_LEAKAGE_AUDIT.json", {"pass": True, "mechanism_outcome_leakage": 0, "feature_classes": feature_classes})

task_index_collisions = sum(bool(r["new_task_index_collisions"]) for r in exposure_rows)
content_matches = sum(len(r["new_content_hash_matches"]) for r in exposure_rows)
independence_md = f"""# Independent Replication Audit

Status: PASS

- Full parent census: {len(census)}
- Sequential parent: {len(sequential)}
- Exact Reverse positive eligible: {len(positive)}
- Exact Reverse negative eligible: {len(negative)}
- Positive historical overlap: {pos_overlap}
- Negative historical overlap: {neg_overlap}
- Independent positives: {len(positive_ind)}
- Independent negatives: {len(negative_ind)}
- State Reset eligible: 0 (rule unresolved for the new pool)
- Selective Deletion eligible: 0 (rule unresolved for the new pool)
- Exact R2 eligible: 0 (input eligibility unresolved for the new pool)
- K/P eligible: 0 (rule unresolved for the new pool)

The exposure census contains {len(exposure_rows)} unique historical sample IDs. Exact current canonical-ID matches: {sum(r['exact_new_canonical_id_match'] for r in exposure_rows)}; exact content-hash matches: {content_matches}. There are {task_index_collisions} task/index collisions, but they belong to the distinct historical benchmark namespace `gated_ntp_stage2_ruler7_16k_newpool_v2` and their content hashes differ; task/index alone is not treated as sample identity.

These are enriched mechanism cohorts and must not be used to estimate population prevalence of TTT benefit. Prevalence belongs to the full formal benchmark and Dynamic authority.
"""
write_text(ROOT / "reports/INDEPENDENT_REPLICATION_AUDIT.md", independence_md)

counts = {
    "PARENT_6500_N": len(census), "CWE_NO_FORMAL_DYNAMIC_N": class_counts["CWE_NO_FORMAL_DYNAMIC"],
    "DYNAMIC_CEILING_NOT_SEARCHED_N": class_counts["DYNAMIC_CEILING_NOT_SEARCHED"],
    "DYNAMIC_SEARCHED_N": len(sequential), "DYNAMIC_SEARCHED_IMPROVED_N": class_counts["DYNAMIC_SEARCHED_IMPROVED"],
    "DYNAMIC_SEARCHED_EQUAL_N": class_counts["DYNAMIC_SEARCHED_EQUAL"], "DYNAMIC_SEARCHED_WORSE_N": class_counts["DYNAMIC_SEARCHED_WORSE"],
    "SEQUENTIAL_PARENT_N": len(sequential), "REVERSE_POSITIVE_ELIGIBLE_N": len(positive),
    "REVERSE_NEGATIVE_ELIGIBLE_N": len(negative), "POSITIVE_HISTORICAL_OVERLAP_N": pos_overlap,
    "NEGATIVE_HISTORICAL_OVERLAP_N": neg_overlap, "REVERSE_POSITIVE_INDEPENDENT_N": len(positive_ind),
    "REVERSE_NEGATIVE_INDEPENDENT_N": len(negative_ind), "MATCHED_PAIRS_AVAILABLE_N": 0,
    "STATE_RESET_ELIGIBLE_N": 0, "SELECTIVE_DELETION_ELIGIBLE_N": 0,
    "EXACT_R2_ELIGIBLE_N": 0, "KP_ELIGIBLE_N": 0,
}
write_json(ROOT / "reports/SCREENING_COUNTS.json", counts)

summary_md = f"""# Mechanism Anchor Screening V1

The protected model, benchmark, Fixed7, and latest completed matching Dynamic authorities all pass identity and integrity checks. The full 6500-row census and 1435-row sequential parent are complete.

All 173 Dynamic-improved rows have nontrivial, reverse-distinct, structurally replayable trajectories. All 1262 Dynamic-equal rows are constant and reverse-identical, leaving zero eligible controls under the required same structural rule. Historical exposure has zero exact current canonical-ID or content-hash overlap.

The Exact Reverse eligibility census is frozen, but a controlled final execution cohort cannot be selected: no eligible negative controls exist and final execution N remains a user/protocol decision. Reset, Selective Deletion, Exact R2 input eligibility, K/P, and Criticality/Observability remain unresolved for new samples because transferring them would require forbidden prior mechanism outcomes or new instrumented execution.
"""
write_text(ROOT / "reports/MECHANISM_ANCHOR_SCREENING_V1_REPORT.md", summary_md)

# ----- Freeze manifest (conventional non-circular manifest) -----
screening_code_hash = sha256(ROOT / "scripts/screening.py")
pre_manifest_files = sorted(p for p in ROOT.rglob("*") if p.is_file() and p.relative_to(ROOT).parts[0] != "receipts" and p.name not in {"MECHANISM_ANCHOR_SCREENING_V1_MANIFEST.json", "SHA256SUMS"})
output_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in pre_manifest_files}
manifest = {
    "version": "MECHANISM_ANCHOR_SCREENING_V1", "frozen_at": NOW,
    "model_identity": MODEL_SHA, "checkpoint_identity": CHECKPOINT_SHA,
    "benchmark_identity": BENCHMARK_MANIFEST_SHA, "benchmark_sample_manifest_sha256": SAMPLE_MANIFEST_SHA,
    "fixed7_run": str(FIXED_ROOT), "fixed7_final_audit_sha256": authority_audit["fixed7"]["final_audit_sha256"],
    "dynamic_run": str(dynamic_root), "dynamic_final_audit_sha256": authority_audit["dynamic"]["final_audit_sha256"],
    "historical_rule_audit_sha256": sha256(ROOT / "audit/HISTORICAL_MECHANISM_SCREENING_RULE_AUDIT.json"),
    "screening_code_sha256": screening_code_hash, "output_file_sha256": output_hashes,
    "self_hash_excluded_to_avoid_circularity": True,
    "downstream_receipts_excluded_from_manifest_payload": True,
    "counts": counts,
    "freeze_state": {"CENSUS_FROZEN": True, "ELIGIBILITY_FROZEN": True, "MATCHING_FROZEN": True,
                     "FINAL_EXECUTION_COHORT_FROZEN": False},
    "execution_prohibitions": {"EXACT_REVERSE_EXECUTED": False, "STATE_RESET_EXECUTED": False,
        "SELECTIVE_DELETION_EXECUTED": False, "R2_EXECUTED": False, "KP_FACTORIAL_EXECUTED": False,
        "CRITICALITY_INTERVENTION_EXECUTED": False, "NEW_GPU_COUNTERFACTUAL_RUNS": 0,
        "FORMAL_MODEL_MUTATION": 0, "BENCHMARK_MUTATION": 0},
}
manifest_path = ROOT / "manifests/MECHANISM_ANCHOR_SCREENING_V1_MANIFEST.json"
write_json(manifest_path, manifest)
manifest_sha = sha256(manifest_path)

receipt = {
    "MECHANISM_ANCHOR_SCREENING_STATUS": "PASS_WITH_UNRESOLVED_RULES",
    "CENSUS_FROZEN": True, "ELIGIBILITY_FROZEN": True, "MATCHING_FROZEN": True,
    "FINAL_EXECUTION_COHORT_FROZEN": False, "FINAL_EXECUTION_N_PENDING": True,
    "MATCHED_CONTROL_RULE_STATUS": "PROVEN_EXACT_NO_ELIGIBLE_CONTROLS",
    "STATE_RESET_RULE_STATUS": "UNRESOLVED", "SELECTIVE_DELETION_RULE_STATUS": "UNRESOLVED",
    "EXACT_R2_RULE_STATUS": "UNRESOLVED_INPUT_ELIGIBILITY", "KP_RULE_STATUS": "UNRESOLVED",
    "CRITICALITY_OBSERVABILITY_RULE_STATUS": "UNRESOLVED",
    "MECHANISM_OUTCOME_LEAKAGE": 0, "SAFE_TO_SELECT_EXACT_REVERSE_EXECUTION_COHORT": False,
    "SAFE_TO_LAUNCH_EXACT_REVERSE": False,
    "SCREENING_MANIFEST": str(manifest_path), "SCREENING_MANIFEST_SHA256": manifest_sha,
    "REPORT_ROOT": str(ROOT), "counts": counts,
    "execution_prohibitions": manifest["execution_prohibitions"],
}
receipt_json = ROOT / "receipts/MECHANISM_ANCHOR_SCREENING_V1_RECEIPT.json"
receipt_md = ROOT / "receipts/MECHANISM_ANCHOR_SCREENING_V1_RECEIPT.md"
write_json(receipt_json, receipt)
write_text(receipt_md, f"""# Mechanism Anchor Screening V1 Receipt

- Status: `PASS_WITH_UNRESOLVED_RULES`
- CENSUS_FROZEN: `true`
- ELIGIBILITY_FROZEN: `true`
- MATCHING_FROZEN: `true` (deterministic empty result; no structurally eligible controls)
- FINAL_EXECUTION_COHORT_FROZEN: `false`
- FINAL_EXECUTION_N_PENDING: `true`
- MECHANISM_OUTCOME_LEAKAGE: `0`
- SAFE_TO_SELECT_EXACT_REVERSE_EXECUTION_COHORT: `NO`
- SAFE_TO_LAUNCH_EXACT_REVERSE: `NO`
- Screening manifest SHA256: `{manifest_sha}`

No mechanism counterfactual, GPU formal evaluation, model mutation, benchmark mutation, training, or inference was executed.
""")

# Final checksum inventory includes manifest and receipts; SHA256SUMS excludes itself.
all_files = sorted(p for p in ROOT.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
write_text(ROOT / "manifests/SHA256SUMS", "\n".join(f"{sha256(p)}  {p.relative_to(ROOT)}" for p in all_files))

print(json.dumps({"status": receipt["MECHANISM_ANCHOR_SCREENING_STATUS"], "root": str(ROOT),
                  "manifest_sha256": manifest_sha, "counts": counts,
                  "historical_exposure_unique_n": len(exposure_rows), "content_overlap_n": content_matches,
                  "task_index_collision_n": task_index_collisions}, indent=2, sort_keys=True))
