#!/usr/bin/env python3
"""Freeze the positive-only Exact Reverse formal protocol before new outcomes."""
from __future__ import annotations

import hashlib
import json
import os
import socket
from collections import Counter
from datetime import datetime
from pathlib import Path

RUN = Path("/home/USER/ttt/runs/formal/exact_reverse_1p7b_v1/run_20260905T144633+0800")
SCREEN = Path("/home/USER/ttt/runs/formal/mechanism_anchor_screening_1p7b_v1/run_20260905T143059+0800")
MODEL = Path("/home/USER/ttt/models/qwen3_1p7b_stage2")
BENCH = Path("/home/USER/ttt/benchmarks/ruler_16k_standard_13task500_v1")
FIXED = Path("/home/USER/ttt/runs/formal/fixed7_1p7b_v1/run_20260828T012948+0800")
EXECUTOR = Path("/home/USER/ttt_dynamic_mining_v1/phase_d_enriched_dynamic_mining_v1/runtime/code/dynamic_runner.py")
SCORER = Path("/home/USER/ttt/code/ruler/tools/common.py")
MODEL_SHA = "ca51ee90d20ee09efbe0b31dd21bf9faa70275508316d1ba7cb1387ee4f0725f"
CHECKPOINT_SHA = "f4e264ed6f5671f123d6c8a27c69bce3af5ffcba518c9e298398314c6147d9fd"
SCREEN_SHA = "c5e04276887da0dda6d1b4599da01e70d4dd88ca8a0cf96f3b9db58ee08f2a65"
BENCH_SHA = "5d16e3a6963d4c21450f0530af743730b91bf7b2442699e288e10754f77ad0c5"
SAMPLE_SHA = "8000dfccd5199ff83bf0cb0ede51e91991083a5dc6a90ac27397b8b24988dfdc"
EXECUTOR_SHA = "293886005cb8f78c943f0affe3e3b383d6de238f4bc825a5aea28a6e6dc66f6e"
SCORER_SHA = "cee3f5cb3cad1709f81abe3dc32b364ca97322e7ebd91747a8b119aecb0ce556"
ACTIONS = ["OFF", "L0", "L6", "L12", "L18", "L24", "ALL"]


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
    temp = path.with_name("." + path.name + f".{os.getpid()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(temp, flags, 0o644)
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


def atomic_jsonl(path, rows):
    atomic_text(path, "".join(json.dumps(r, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows))


def result_name(sample_id):
    return hashlib.sha256(sample_id.encode()).hexdigest() + ".json"


for name in ["preregistration", "audit", "config", "manifests", "logs", "claims", "commits", "errors", "status", "smoke", "results", "reports", "scripts"]:
    (RUN / name).mkdir(parents=True, exist_ok=True)

screen_manifest_path = SCREEN / "manifests/MECHANISM_ANCHOR_SCREENING_V1_MANIFEST.json"
screen_receipt_path = SCREEN / "receipts/MECHANISM_ANCHOR_SCREENING_V1_RECEIPT.json"
screen_pool_path = SCREEN / "pools/EXACT_REVERSE_POSITIVE_INDEPENDENT.jsonl"
assert sha(screen_manifest_path) == SCREEN_SHA
screen_manifest = load_json(screen_manifest_path)
screen_receipt = load_json(screen_receipt_path)
assert screen_receipt["MECHANISM_ANCHOR_SCREENING_STATUS"] == "PASS_WITH_UNRESOLVED_RULES"
assert screen_receipt["counts"]["REVERSE_POSITIVE_INDEPENDENT_N"] == 173
assert screen_receipt["counts"]["REVERSE_NEGATIVE_ELIGIBLE_N"] == 0
assert screen_receipt["counts"]["POSITIVE_HISTORICAL_OVERLAP_N"] == 0
assert screen_receipt["MECHANISM_OUTCOME_LEAKAGE"] == 0

assert sha(MODEL / "model.safetensors") == MODEL_SHA
assert load_json(MODEL / "artifact_manifest.json")["model_sha256"] == MODEL_SHA
assert sha(BENCH / "benchmark_manifest.json") == BENCH_SHA
assert sha(BENCH / "sample_manifest.jsonl") == SAMPLE_SHA
assert sha(EXECUTOR) == EXECUTOR_SHA
assert sha(SCORER) == SCORER_SHA
fixed_audit = load_json(FIXED / "audits/FINAL_FIXED7_FORMAL_AUDIT.json")
assert fixed_audit["final_integrity"] == "PASS" and fixed_audit["completed_cells"] == 45500

dynamic_root = Path(screen_manifest["dynamic_run"])
dynamic_audit_path = dynamic_root / "audits/FINAL_DYNAMIC_NONCEILING_12TASK_AUDIT.json"
dynamic_audit = load_json(dynamic_audit_path)
assert dynamic_audit["final_integrity"] == "PASS"
assert dynamic_audit["model_identity"] == MODEL_SHA
assert dynamic_audit["checkpoint_identity"] == CHECKPOINT_SHA
assert dynamic_audit["benchmark_identity"] == BENCH_SHA
assert dynamic_audit["improved_count"] == 173 and dynamic_audit["equal_count"] == 1262 and dynamic_audit["worse_count"] == 0

bench_rows = load_jsonl(BENCH / "sample_manifest.jsonl")
bench_by_id = {r["sample_id"]: r for r in bench_rows}
assert len(bench_rows) == len(bench_by_id) == 6500
source_rows = load_jsonl(screen_pool_path)
assert len(source_rows) == len({r["canonical_sample_id"] for r in source_rows}) == 173

cohort = []
for source in sorted(source_rows, key=lambda r: r["benchmark_global_index"]):
    sid = source["canonical_sample_id"]
    assert source["historical_mechanism_exposed"] is False
    assert source["parent_class"] == "DYNAMIC_SEARCHED_IMPROVED"
    assert source["nontrivial_order"] is True and source["reverse_distinct"] is True
    assert source["replayability"]["exact_reverse_structurally_replayable"] is True
    original = list(source["action_sequence"])
    reverse = list(reversed(original))
    assert original != reverse and Counter(original) == Counter(reverse)
    assert len(original) == source["full_chunk_count"]
    dyn_path = dynamic_root / "results" / result_name(sid)
    dyn = load_json(dyn_path)
    assert dyn["completed"] is True and dyn["error_status"] == "NONE"
    assert dyn["action_sequence"] == original
    assert abs(float(dyn["dynamic_score"]) - float(source["dynamic_score"])) <= 1e-12
    b = bench_by_id[sid]
    cohort.append({
        "cohort_order": len(cohort) + 1,
        "canonical_sample_id": sid,
        "task": source["task"],
        "task_sample_index": source["task_sample_index"],
        "benchmark_global_index": source["benchmark_global_index"],
        "benchmark_content_sha256": source["benchmark_content_sha256"],
        "prompt_length_tokens_canonical": source["prompt_length_tokens_canonical"],
        "full_chunk_count": source["full_chunk_count"],
        "incomplete_tail_length": source["incomplete_tail_length"],
        "original_dynamic_actions": original,
        "reversed_actions": reverse,
        "action_count_vector": {action: original.count(action) for action in ACTIONS},
        "switch_count_original": source["number_of_switches"],
        "switch_count_reversed": sum(a != b for a, b in zip(reverse, reverse[1:])),
        "original_dynamic_score": source["dynamic_score"],
        "sample_best_score": source["sample_best_score"],
        "off_score": source["fixed7_scores"]["OFF"],
        "dynamic_minus_sample_best": source["dynamic_minus_sample_best"],
        "original_dynamic_prediction_sha256": dyn["authority_search_result"]["best_prediction_hash"],
        "dynamic_result_path": str(dyn_path),
        "dynamic_result_sha256": sha(dyn_path),
        "model_identity": MODEL_SHA,
        "checkpoint_identity": CHECKPOINT_SHA,
        "benchmark_identity": BENCH_SHA,
        "historical_mechanism_exposed": False,
        "exact_reverse_structurally_replayable": True,
        "max_new_tokens": b["max_new_tokens"],
    })

cohort_path = RUN / "manifests/EXACT_REVERSE_173_COHORT.jsonl"
atomic_jsonl(cohort_path, cohort)
cohort_sha = sha(cohort_path)

amendment = {
    "version": "EXACT_REVERSE_PROTOCOL_AMENDMENT_V1",
    "frozen_at": now(),
    "MATCHED_NEGATIVE_ORIGINAL_DESIGN": "NOT_EXECUTABLE_ON_FORMAL_PARENT",
    "REASON": "NO_STRUCTURALLY_ELIGIBLE_NEGATIVE_CONTROLS",
    "AMENDMENT_TIMING": "BEFORE_NEW_EXACT_REVERSE_OUTCOMES",
    "NEW_PRIMARY_DESIGN": "WITHIN_SAMPLE_PAIRED_POSITIVE_REPLICATION",
    "EXECUTION_COHORT": "ALL_173_INDEPENDENT_ELIGIBLE_POSITIVES",
    "POSTHOC_SAMPLE_SELECTION": False,
    "STRUCTURALLY_ELIGIBLE_NEGATIVE_CONTROLS": 0,
    "allowed_claim": "successful sequential trajectories are order-sensitive",
    "disallowed_claim": "successful trajectories are more order-sensitive than unsuccessful nontrivial trajectories",
    "screening_manifest_sha256": SCREEN_SHA,
    "cohort_manifest_sha256": cohort_sha,
}
amendment_json = RUN / "preregistration/EXACT_REVERSE_PROTOCOL_AMENDMENT_V1.json"
atomic_json(amendment_json, amendment)
atomic_text(RUN / "preregistration/EXACT_REVERSE_PROTOCOL_AMENDMENT_V1.md", f"""# Exact Reverse Protocol Amendment V1

Frozen before any new Exact Reverse outcome at `{amendment['frozen_at']}`.

- MATCHED_NEGATIVE_ORIGINAL_DESIGN=NOT_EXECUTABLE_ON_FORMAL_PARENT
- REASON=NO_STRUCTURALLY_ELIGIBLE_NEGATIVE_CONTROLS
- AMENDMENT_TIMING=BEFORE_NEW_EXACT_REVERSE_OUTCOMES
- NEW_PRIMARY_DESIGN=WITHIN_SAMPLE_PAIRED_POSITIVE_REPLICATION
- EXECUTION_COHORT=ALL_173_INDEPENDENT_ELIGIBLE_POSITIVES
- POSTHOC_SAMPLE_SELECTION=false

All 1262 Dynamic-equal trajectories are constant/reverse-identical. They cannot serve as genuine temporal-order interventions. Structural eligibility is not relaxed and no negative control is manufactured.
""")

worker_path = RUN / "scripts/exact_reverse_worker.py"
aggregate_path = RUN / "scripts/aggregate_exact_reverse.py"
controller_path = RUN / "scripts/run_exact_reverse_controller.sh"
for p in [worker_path, aggregate_path, controller_path]:
    if not p.is_file():
        raise RuntimeError(f"MISSING_EXECUTION_SCRIPT {p}")

protocol = {
    "version": "EXACT_REVERSE_1P7B_FORMAL_V1",
    "frozen_at": now(),
    "model": {"root": str(MODEL), "model_identity": MODEL_SHA, "checkpoint_identity": CHECKPOINT_SHA},
    "benchmark": {"root": str(BENCH), "benchmark_identity": BENCH_SHA, "sample_manifest_sha256": SAMPLE_SHA,
                  "context_length": 16384, "task_n": 13, "samples_per_task": 500, "total_n": 6500, "seed": 42},
    "parent": {"screening_root": str(SCREEN), "screening_manifest_sha256": SCREEN_SHA,
               "dynamic_root": str(dynamic_root), "dynamic_final_audit_sha256": sha(dynamic_audit_path),
               "fixed7_root": str(FIXED)},
    "cohort": {"manifest": str(cohort_path), "manifest_sha256": cohort_sha, "execution_n": 173,
               "unique_sample_ids": 173, "historical_overlap_n": 0, "posthoc_sample_selection": False},
    "primary_intervention": "exact list reversal; temporal action order only",
    "primary_outcome": "D_i = Dynamic_i - Reverse_i; headline mean(D_i)",
    "secondary_outcomes": ["median(D_i)", "worse/equal/better counts", "Reverse-SampleBest", "unclamped retained_gain"],
    "statistics": {"paired_bootstrap_resamples": 20000, "paired_bootstrap_seed": 20260905,
                   "task_stratified_bootstrap_resamples": 20000, "task_stratified_bootstrap_seed": 20260906,
                   "ci": 0.95, "sign_test": "NOT_ADDED", "classification": "NO_HISTORICAL_THRESHOLD; raw effect first"},
    "scientific_semantics": {"actions": ACTIONS, "layers": [0, 6, 12, 18, 24], "context_length": 16384,
        "chunk_size": 1024, "fast_weight": "mlp.down_proj.weight",
        "objective": "contextual_hidden_direct_outer_product_no_scalar_loss", "ttt_lr": 1.0,
        "delta_clip_frobenius": 1e-5, "execution": "apply-then-update", "reset": "per sample",
        "complete_prompt_chunk_update": True, "incomplete_tail_update": False, "generation_update": False,
        "formal_execution": "full-prompt layer-major", "downstream_recomputation": True,
        "stored_delta_stitching": False, "workers": 1, "cpu_affinity": "8-23"},
    "code_authority": {"frozen_dynamic_executor": str(EXECUTOR), "frozen_dynamic_executor_sha256": EXECUTOR_SHA,
        "scorer": str(SCORER), "scorer_sha256": SCORER_SHA, "worker_sha256": sha(worker_path),
        "aggregator_sha256": sha(aggregate_path), "controller_sha256": sha(controller_path)},
    "claim_boundary": {"allowed": ["temporal ordering is causally important for the successful frozen Dynamic trajectories if Reverse damages them", "action composition alone is insufficient for those trajectories if Reverse damages them"],
                       "not_allowed": ["all TTT trajectories are order-sensitive", "successful trajectories are more order-sensitive than unsuccessful nontrivial trajectories", "mathematical operators are proven non-commutative", "Dynamic is globally optimal"]},
    "other_mechanisms": {"state_reset": False, "selective_deletion": False, "exact_r2": False,
                         "kp_factorial": False, "criticality": False, "observability": False},
}
protocol_path = RUN / "preregistration/EXACT_REVERSE_1P7B_FORMAL_V1.json"
atomic_json(protocol_path, protocol)
protocol_sha = sha(protocol_path)
atomic_text(RUN / "preregistration/EXACT_REVERSE_1P7B_FORMAL_V1.md", f"""# Qwen3-1.7B Formal Exact Reverse Replication V1

Frozen before smoke and formal outcomes at `{protocol['frozen_at']}`.

- Design: within-sample paired positive replication on all 173 frozen independent anchors.
- Intervention: reverse the complete action list; change temporal order only.
- Primary outcome: mean Dynamic minus Reverse.
- Ordinary paired bootstrap: 20,000 resamples, seed 20260905.
- Task-stratified paired bootstrap: 20,000 resamples, seed 20260906.
- Retained gain is not clamped; aggregate gain retention is also reported.
- No sign-test headline and no outcome-dependent classification threshold are introduced.
- No matched negative controls exist; no negative controls will be fabricated.
- No other mechanism experiment may be launched.

Protocol SHA256 is recorded in the frozen config after this document is written.
""")

config = {
    "version": "EXACT_REVERSE_1P7B_FORMAL_V1",
    "run_root": str(RUN), "model_root": str(MODEL), "model_identity": MODEL_SHA,
    "checkpoint_identity": CHECKPOINT_SHA, "benchmark_root": str(BENCH), "benchmark_identity": BENCH_SHA,
    "benchmark_sample_manifest_sha256": SAMPLE_SHA, "fixed7_root": str(FIXED), "dynamic_root": str(dynamic_root),
    "screening_root": str(SCREEN), "screening_manifest_sha256": SCREEN_SHA,
    "cohort_manifest": str(cohort_path), "cohort_manifest_sha256": cohort_sha,
    "protocol": str(protocol_path), "protocol_sha256": protocol_sha,
    "protocol_amendment": str(amendment_json), "protocol_amendment_sha256": sha(amendment_json),
    "frozen_executor": str(EXECUTOR), "frozen_executor_sha256": EXECUTOR_SHA,
    "scorer": str(SCORER), "scorer_sha256": SCORER_SHA,
    "worker": str(worker_path), "worker_sha256": sha(worker_path),
    "aggregator": str(aggregate_path), "aggregator_sha256": sha(aggregate_path),
    "controller": str(controller_path), "controller_sha256": sha(controller_path),
    "python": "/home/USER/conda_envs/ttt_phase_c_v1/bin/python",
    "tmux": "/home/USER/ttt_dynamic_mining_v1/phase_d_enriched_dynamic_mining_v1/runtime/tools/tmux",
    "tmux_session": "exact_reverse_1p7b_v1", "workers": 1, "cpu_affinity": "8-23",
    "expected_results": 173, "smoke_cohort_orders": [1, 2, 3],
    "paired_bootstrap_seed": 20260905, "task_stratified_bootstrap_seed": 20260906,
    "bootstrap_resamples": 20000, "tolerance": 1e-12,
}
config_path = RUN / "config/EXACT_REVERSE_FORMAL_CONFIG.json"
atomic_json(config_path, config)

historical_runner = Path("/home/USER/ttt_mechanism_analysis_v1/minimal_targeted_causal_replay_v1/scripts/run_targeted_replay.py")
historical_manifest_builder = Path("/home/USER/ttt_mechanism_analysis_v1/minimal_targeted_causal_replay_v1/scripts/build_frozen_manifest.py")
historical_manifest = Path("/home/USER/ttt_mechanism_analysis_v1/minimal_targeted_causal_replay_v1/manifests/TARGETED_CAUSAL_REPLAY_MANIFEST.jsonl")
fields = [
    ("trajectory representation", "PROVEN_EXACT", "Python list of one action label per complete 1024-token prompt chunk"),
    ("action indexing", "PROVEN_EXACT", "sequence[j] controls one-based chunk_index j+1; actions map to layers OFF/0/6/12/18/24/ALL"),
    ("chunk indexing", "PROVEN_EXACT", "full_chunks=floor(tokenized_prompt_length/1024); event chunk_index is one-based"),
    ("reverse operation", "PROVEN_EXACT", "list(reversed(reference_sequence)); exact length/composition equality and nonidentity checked"),
    ("model reset", "PROVEN_EXACT", "one loaded eval model; every generate starts from base weights and local past_w; base hash checked before/after"),
    ("fast-weight handling", "PROVEN_EXACT", "TTTState initialized from down_proj.weight; action-selected layer updates apply-then-update with clip 1e-5"),
    ("cache/context handling", "PROVEN_EXACT", "fresh generate cache per sequence; use_cache=True; full prompt evaluated layer-major"),
    ("downstream recomputation", "PROVEN_EXACT", "new full generate per schedule; subsequent_recomputation=true; no delta stitching"),
    ("incomplete tail", "PROVEN_EXACT", "forwarded with terminal fast weight; no TTT update"),
    ("generation", "PROVEN_EXACT", "greedy generate, num_beams=1, no generation TTT update"),
    ("scoring", "PROVEN_EXACT", "frozen RULER per_sample_score(family,prediction,reference)"),
    ("result schema/commit", "PROVEN_EXACT", "one JSON per intervention; completed=true only after atomic os.replace"),
]
audit_lines = ["# Historical Exact Reverse Implementation Audit", "", "Status: `PASS`", "",
               f"Executed runner: `{historical_runner}` (`{sha(historical_runner)}`)",
               f"Manifest builder: `{historical_manifest_builder}` (`{sha(historical_manifest_builder)}`)",
               f"Executed manifest: `{historical_manifest}` (`{sha(historical_manifest)}`)",
               f"Frozen Dynamic executor: `{EXECUTOR}` (`{EXECUTOR_SHA}`)", "",
               "| Semantic field | Classification | Recovered behavior |", "|---|---|---|"]
audit_lines.extend(f"| {a} | {b} | {c} |" for a, b, c in fields)
audit_lines += ["", "No core semantic field is UNKNOWN. The new worker calls the same frozen Evaluator and changes only the supplied sequence to its exact reversal."]
atomic_text(RUN / "audit/HISTORICAL_EXACT_REVERSE_IMPLEMENTATION_AUDIT.md", "\n".join(audit_lines))

prelaunch = {
    "status": "PASS", "timestamp": now(), "host": socket.gethostname(), "user": os.environ.get("USER"),
    "MODEL_IDENTITY_MATCH": True, "CHECKPOINT_IDENTITY_MATCH": True, "BENCHMARK_IDENTITY_MATCH": True,
    "SCREENING_MANIFEST_MATCH": True, "DYNAMIC_AUTHORITY_MATCH": True, "FIXED7_AUTHORITY_MATCH": True,
    "COHORT_FROZEN": True, "COHORT_N": 173, "UNIQUE_SAMPLE_IDS": 173, "HISTORICAL_MECHANISM_OVERLAP": 0,
    "ACTION_COMPOSITION_PARITY": "PASS", "REVERSE_DISTINCT_173_OF_173": True,
    "NO_STRUCTURALLY_ELIGIBLE_NEGATIVE_CONTROLS": True,
    "HISTORICAL_EXACT_REVERSE_AUDIT_STATUS": "PASS", "CORE_UNKNOWN_FIELD_N": 0,
    "PROTOCOL_AMENDMENT_STATUS": "PASS", "PROTOCOL_SHA256": protocol_sha,
    "COHORT_MANIFEST_SHA256": cohort_sha, "FORMAL_LAUNCHED": False,
}
atomic_json(RUN / "audit/PRELAUNCH_AUTHORITY_AND_COHORT_AUDIT.json", prelaunch)
atomic_json(RUN / "status/pipeline_status.json", {
    "pipeline_status": "PREREGISTERED", "current_phase": "READY_FOR_SMOKE", "total_expected": 173,
    "committed_results": 0, "error_count": 0, "controller_pid": None, "worker_pid": None,
    "current_sample": None, "current_task": None, "last_commit_timestamp": None,
    "cohort_manifest_sha256": cohort_sha, "protocol_sha256": protocol_sha,
    "updated_at": now(), "last_error": "NONE",
})
atomic_json(RUN / "manifests/PRELAUNCH_FREEZE.json", {
    "frozen_at": now(), "cohort_manifest_sha256": cohort_sha, "protocol_sha256": protocol_sha,
    "protocol_amendment_sha256": sha(amendment_json), "config_sha256": sha(config_path),
    "historical_audit_sha256": sha(RUN / "audit/HISTORICAL_EXACT_REVERSE_IMPLEMENTATION_AUDIT.md"),
    "worker_sha256": sha(worker_path), "aggregator_sha256": sha(aggregate_path),
    "controller_sha256": sha(controller_path), "formal_outcome_file_count_at_freeze": len(list((RUN / "results").glob("*.json"))),
})
print(json.dumps({"status": "PASS", "run_root": str(RUN), "cohort_n": len(cohort),
                  "cohort_manifest_sha256": cohort_sha, "protocol_sha256": protocol_sha,
                  "dynamic_root": str(dynamic_root)}, indent=2, sort_keys=True))
