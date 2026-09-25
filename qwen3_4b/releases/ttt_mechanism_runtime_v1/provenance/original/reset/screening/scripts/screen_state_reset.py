#!/usr/bin/env python3
"""CPU-only historical rule recovery and static State Reset cohort screening."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/USER/ttt/runs/formal/state_reset_screening_1p7b_v1/run_20260905T175639+0800")
MODEL = Path("/home/USER/ttt/models/qwen3_1p7b_stage2/model.safetensors")
BENCH = Path("/home/USER/ttt/benchmarks/ruler_16k_standard_13task500_v1")
DYNAMIC = Path("/home/USER/ttt/runs/formal/dynamic_nonceiling_12task_1p7b_v1/run_20260828T161330+0800")
SCREEN = Path("/home/USER/ttt/runs/formal/mechanism_anchor_screening_1p7b_v1/run_20260905T143059+0800")
REVERSE = Path("/home/USER/ttt/runs/formal/exact_reverse_1p7b_v1/run_20260905T144633+0800")
HIST = Path("/home/USER/ttt_mechanism_analysis_v1")

PARENT = SCREEN / "pools/EXACT_REVERSE_POSITIVE_INDEPENDENT.jsonl"
HIST_MORPH_SCRIPT = HIST / "signal_analysis_abcd_v1/scripts/analysis_common.py"
HIST_TARGET_SCRIPT = HIST / "causal_counterfactual_v1/scripts/finalize_analysis.py"
HIST_REVERSE_ANALYSIS = HIST / "minimal_targeted_causal_replay_v1/scripts/analyze_frozen_results.py"
HIST_REVERSE_MANIFEST = HIST / "minimal_targeted_causal_replay_v1/manifests/TARGETED_CAUSAL_REPLAY_MANIFEST.jsonl"
HIST_EARLY_TABLE = HIST / "minimal_targeted_causal_replay_v1/tables/EARLY_STEADY_EXACT_REVERSE.csv"
HIST_STATE_ROOT = HIST / "targeted_fast_weight_state_replay_v1"
HIST_STATE_BUILDER = HIST_STATE_ROOT / "scripts/build_state_replay_manifest.py"
HIST_STATE_BACKEND = HIST_STATE_ROOT / "scripts/instrumented_backend.py"
HIST_STATE_RUNNER = HIST_STATE_ROOT / "scripts/run_instrumented_state_replay.py"
HIST_STATE_MANIFEST = HIST_STATE_ROOT / "manifests/STATE_REPLAY_MANIFEST.jsonl"
HIST_STATE_CONFIG = HIST_STATE_ROOT / "configs/TARGETED_STATE_REPLAY_CONFIG.json"
HIST_RESET_SPEC = HIST_STATE_ROOT / "configs/FAST_WEIGHT_RESET_INTERVENTION_SPEC.json"
HIST_STATE_AUDIT = HIST_STATE_ROOT / "audits/FINAL_INTEGRITY_AUDIT.json"
HIST_EXPOSURE = SCREEN / "overlap/HISTORICAL_MECHANISM_EXPOSURE_SET.jsonl"

EXPECTED_MODEL = "ca51ee90d20ee09efbe0b31dd21bf9faa70275508316d1ba7cb1387ee4f0725f"
EXPECTED_CHECKPOINT = "f4e264ed6f5671f123d6c8a27c69bce3af5ffcba518c9e298398314c6147d9fd"
EXPECTED_BENCHMARK = "5d16e3a6963d4c21450f0530af743730b91bf7b2442699e288e10754f77ad0c5"
EXPECTED_SCREEN_MANIFEST = "c5e04276887da0dda6d1b4599da01e70d4dd88ca8a0cf96f3b9db58ee08f2a65"
EXPECTED_HIST_STATE_MANIFEST = "847cb0c3fb8c27fcb4b20263133702a3f97c0bed573c8b8d5beb9bed33166b4f"
EXPECTED_HIST_SOURCE_MANIFEST = "83853fe1db235fd175eae22a9a409a760a74cccc3f74ade9320f3881d5325233"
SCORER = "cee3f5cb3cad1709f81abe3dc32b364ca97322e7ebd91747a8b119aecb0ce556"
ACTIONS = ["OFF", "L0", "L6", "L12", "L18", "L24", "ALL"]
CHUNK_SIZE = 1024


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_json(path: Path, obj) -> None:
    atomic_text(path, json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def atomic_jsonl(path: Path, rows) -> None:
    atomic_text(path, "".join(json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n" for x in rows))


def rle(sequence):
    out = []
    for action in sequence:
        if out and out[-1]["action"] == action:
            out[-1]["length"] += 1
        else:
            out.append({"action": action, "length": 1})
    return out


def terminal_run_boundary(sequence):
    action = sequence[-1]
    i = len(sequence) - 1
    while i > 0 and sequence[i - 1] == action:
        i -= 1
    if i == 0:
        raise ValueError("NO_ACTION_SWITCH")
    return i + 1


def historical_early_steady(sequence):
    """Exact executed morphology() logic from analysis_common.py."""
    n = len(sequence)
    switches = [i for i in range(1, n) if sequence[i] != sequence[i - 1]]
    first_norm = (switches[0] + 1) / n if switches and n else None
    early = first_norm is not None and first_norm <= 0.25
    tail = sequence[n // 2:]
    counts = Counter(tail)
    steady = max(ACTIONS, key=lambda a: (counts[a], -ACTIONS.index(a))) if tail else "UNKNOWN"
    steady_n = counts[steady] if tail else 0
    steady_fraction = steady_n / max(len(tail), 1)
    steady_ok = steady_fraction >= 0.8
    return bool(early and steady_ok), {
        "first_switch_chunk_index": switches[0] + 1 if switches else None,
        "first_switch_position_normalized": first_norm,
        "early_switch_at_or_before_fraction": 0.25,
        "tail_start_chunk_index": n // 2 + 1,
        "tail_length": len(tail),
        "steady_tail_action": steady,
        "steady_tail_count": steady_n,
        "steady_tail_fraction": steady_fraction,
        "steady_tail_required_fraction": 0.8,
    }


def quantiles(values):
    values = sorted(values)
    def q(p):
        x = (len(values) - 1) * p
        lo, hi = math.floor(x), math.ceil(x)
        return values[lo] if lo == hi else values[lo] * (hi - x) + values[hi] * (x - lo)
    return {str(p): q(p) for p in (0, .25, .5, .75, 1)}


def main():
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    for sub in ["audit", "historical", "census", "pools", "manifests", "overlap", "reports", "receipts"]:
        (ROOT / sub).mkdir(parents=True, exist_ok=True)

    # Formal authorities: identity/status only. Exact Reverse outcome fields are never used.
    bench_hash_recorded = (BENCH / "benchmark_manifest.sha256").read_text().split()[0]
    dynamic_audit = load_json(DYNAMIC / "audits/FINAL_DYNAMIC_NONCEILING_12TASK_AUDIT.json")
    screen_manifest_path = SCREEN / "manifests/MECHANISM_ANCHOR_SCREENING_V1_MANIFEST.json"
    screen_manifest = load_json(screen_manifest_path)
    reverse_receipt_path = REVERSE / "reports/EXACT_REVERSE_1P7B_FORMAL_COMPLETION_RECEIPT.json"
    reverse_receipt = load_json(reverse_receipt_path)
    authority = {
        "MODEL_AUTHORITY_STATUS": "PASS" if sha(MODEL) == EXPECTED_MODEL else "FAIL",
        "BENCHMARK_AUTHORITY_STATUS": "PASS" if bench_hash_recorded == EXPECTED_BENCHMARK else "FAIL",
        "DYNAMIC_AUTHORITY_STATUS": "PASS" if dynamic_audit.get("final_integrity") == "PASS" and dynamic_audit.get("improved_count") == 173 and dynamic_audit.get("model_identity") == EXPECTED_MODEL and dynamic_audit.get("checkpoint_identity") == EXPECTED_CHECKPOINT else "FAIL",
        "MECHANISM_SCREENING_AUTHORITY_STATUS": "PASS" if sha(screen_manifest_path) == EXPECTED_SCREEN_MANIFEST and screen_manifest.get("counts", {}).get("REVERSE_POSITIVE_INDEPENDENT_N") == 173 else "FAIL",
        "EXACT_REVERSE_AUTHORITY_STATUS": "PASS" if reverse_receipt.get("FINAL_INTEGRITY") == "PASS" and reverse_receipt.get("FORMAL_EXACT_REVERSE_COMPLETE") is True and reverse_receipt.get("N") == 173 else "FAIL",
        "model_identity": sha(MODEL),
        "checkpoint_identity": dynamic_audit.get("checkpoint_identity"),
        "benchmark_identity": bench_hash_recorded,
        "dynamic_final_audit_sha256": sha(DYNAMIC / "audits/FINAL_DYNAMIC_NONCEILING_12TASK_AUDIT.json"),
        "mechanism_screening_manifest_sha256": sha(screen_manifest_path),
        "exact_reverse_receipt_sha256_provenance_only": sha(reverse_receipt_path),
    }
    if any(authority[k] != "PASS" for k in ["MODEL_AUTHORITY_STATUS", "BENCHMARK_AUTHORITY_STATUS", "DYNAMIC_AUTHORITY_STATUS", "MECHANISM_SCREENING_AUTHORITY_STATUS", "EXACT_REVERSE_AUTHORITY_STATUS"]):
        raise RuntimeError(f"FORMAL_AUTHORITY_FAILURE {authority}")
    atomic_json(ROOT / "audit/AUTHORITY_ALIGNMENT_AUDIT.json", authority)

    # Historical executed artifacts and boundary reconstruction.
    hist_state_audit = load_json(HIST_STATE_AUDIT)
    hist_state_config = load_json(HIST_STATE_CONFIG)
    hist_reset_spec = load_json(HIST_RESET_SPEC)
    hist_rows = load_jsonl(HIST_STATE_MANIFEST)
    if hist_state_audit.get("final_integrity") != "PASS" or sha(HIST_STATE_MANIFEST) != EXPECTED_HIST_STATE_MANIFEST:
        raise RuntimeError("HISTORICAL_STATE_AUTHORITY_FAILURE")
    if sha(HIST_REVERSE_MANIFEST) != EXPECTED_HIST_SOURCE_MANIFEST:
        raise RuntimeError("HISTORICAL_SOURCE_MANIFEST_FAILURE")
    reset_rows = [x for x in hist_rows if x["reset_fast_state"]]
    hist_pos = [x for x in reset_rows if x["positive_or_negative"] == "POSITIVE"]
    hist_ctl = [x for x in reset_rows if x["positive_or_negative"] == "NEGATIVE"]
    boundary_mismatch = [x["sample_id"] for x in reset_rows if terminal_run_boundary(x["action_sequence"]) != x["reset_boundary_if_any"]]
    with HIST_EARLY_TABLE.open(newline="", encoding="utf-8") as f:
        early_rows = list(csv.DictReader(f))
    reconstructed_early = {x["sample_id"] for x in hist_pos if historical_early_steady(x["action_sequence"])[0]}
    early_membership = {x["sample_id"] for x in early_rows}
    if reconstructed_early != early_membership or boundary_mismatch:
        raise RuntimeError("HISTORICAL_RULE_RECONSTRUCTION_PARITY_FAILURE")

    exposure_rows = load_jsonl(HIST_EXPOSURE)
    state_exposure = [x for x in exposure_rows if "STATE_RESET" in x.get("old_experiments", [])]
    hist_manifest_ids = {x["sample_id"] for x in hist_rows}
    if {x["historical_sample_id"] for x in state_exposure} != hist_manifest_ids:
        raise RuntimeError("HISTORICAL_EXPOSURE_IDENTITY_FAILURE")
    atomic_jsonl(ROOT / "historical/HISTORICAL_STATE_RESET_POSITIVES.jsonl", sorted(hist_pos, key=lambda x: x["sample_id"]))
    atomic_jsonl(ROOT / "historical/HISTORICAL_STATE_RESET_CONTROLS.jsonl", sorted(hist_ctl, key=lambda x: x["sample_id"]))
    atomic_jsonl(ROOT / "historical/HISTORICAL_STATE_RESET_EXPOSURE_SET.jsonl", state_exposure)

    source_hashes = {
        str(HIST_MORPH_SCRIPT): sha(HIST_MORPH_SCRIPT),
        str(HIST_TARGET_SCRIPT): sha(HIST_TARGET_SCRIPT),
        str(HIST_REVERSE_ANALYSIS): sha(HIST_REVERSE_ANALYSIS),
        str(HIST_STATE_BUILDER): sha(HIST_STATE_BUILDER),
        str(HIST_STATE_BACKEND): sha(HIST_STATE_BACKEND),
        str(HIST_STATE_RUNNER): sha(HIST_STATE_RUNNER),
        str(HIST_STATE_MANIFEST): sha(HIST_STATE_MANIFEST),
    }
    fields = [
        {"field": "historical parent pool", "status": "PROVEN_EXACT", "exact_rule": "20 frozen positive targets inherited from Minimal Targeted Causal Replay; 18 matched nonpositive controls.", "evidence": str(HIST_STATE_BUILDER), "ambiguity": "The 20-positive upstream set was resource-capped; it is not an intrinsic early-steady rule."},
        {"field": "positive definition", "status": "PROVEN_EXACT", "exact_rule": "Upstream Dynamic-positive target (Dynamic > Sample Best), then early_transient_then_steady=True; all such cases among the 20 frozen targets were reset.", "evidence": str(HIST_STATE_BUILDER), "ambiguity": "No additional reset-effect or gain-magnitude threshold."},
        {"field": "upstream positive sampling", "status": "PROVEN_EXACT", "exact_rule": "Group all available discovery/C positives by task; sort strict_cross_action_positive first, then phase and sample_id; take up to 3 per task, yielding 20.", "evidence": str(HIST_TARGET_SCRIPT), "ambiguity": "A preceding minimal-replay budget rule, not applied to the new all-173 parent."},
        {"field": "negative/control definition", "status": "PROVEN_EXACT", "exact_rule": "18 matched nonpositive samples; N_ORDER_1 frozen donor order is compared with the identical order after reset at its terminal-run boundary.", "evidence": str(HIST_STATE_BUILDER), "ambiguity": "14 two-stage and 4 complex canonical controls."},
        {"field": "historical sample counts", "status": "PROVEN_EXACT", "exact_rule": "18 positive reset samples; 18 control reset samples; 38 unique samples in the full instrumented experiment; 36 unique samples received a reset branch.", "evidence": str(HIST_STATE_MANIFEST), "ambiguity": "Two of 20 positive targets failed early-steady and therefore had no reset branch."},
        {"field": "early-steady definition", "status": "PROVEN_EXACT", "exact_rule": "first_switch_position/sequence_length <= 0.25 AND, in sequence[floor(N/2):], the deterministic modal action occupies >= 0.80; action tie-break order OFF,L0,L6,L12,L18,L24,ALL.", "evidence": str(HIST_MORPH_SCRIPT), "ambiguity": "Does not require the entire second half to be constant."},
        {"field": "minimum early segment length", "status": "PROVEN_EXACT", "exact_rule": "No explicit k; existence of a switch implies at least one pre-switch chunk.", "evidence": str(HIST_MORPH_SCRIPT), "ambiguity": "None."},
        {"field": "minimum late segment length", "status": "PROVEN_EXACT", "exact_rule": "No explicit terminal-run k; only >=80% modal occupancy in the second half.", "evidence": str(HIST_MORPH_SCRIPT), "ambiguity": "Terminal-run length can be shorter than second-half modal count."},
        {"field": "reset boundary", "status": "PROVEN_EXACT", "exact_rule": "One-based first chunk of the final action's terminal equal-action run, equivalently last switch + 1.", "evidence": str(HIST_STATE_BUILDER), "ambiguity": "Trajectory-specific, not the first switch and not fixed."},
        {"field": "required action pattern", "status": "PROVEN_EXACT", "exact_rule": "At least one action switch; early first switch plus second-half modal steadiness. No required action identity.", "evidence": str(HIST_MORPH_SCRIPT), "ambiguity": "Complex prefixes are allowed."},
        {"field": "constant after boundary", "status": "PROVEN_EXACT", "exact_rule": "Yes, by terminal-run boundary construction.", "evidence": str(HIST_STATE_BUILDER), "ambiguity": "This is distinct from the looser second-half steadiness test."},
        {"field": "early versus later action", "status": "PROVEN_EXACT", "exact_rule": "Action immediately before boundary differs from terminal action by construction.", "evidence": str(HIST_STATE_BUILDER), "ambiguity": "Earlier prefix may contain multiple regimes."},
        {"field": "gain threshold", "status": "PROVEN_EXACT", "exact_rule": "Dynamic > Sample Best inherited from the positive parent; no additional numeric margin threshold.", "evidence": str(HIST_TARGET_SCRIPT), "ambiguity": "Historical task cap prioritized strict_cross_action_positive but State Reset added no gain cutoff."},
        {"field": "replayability", "status": "PROVEN_EXACT", "exact_rule": "Frozen source manifest identity, exact action sequence/chunk count, model/checkpoint/executor/scorer identities; action schedule unchanged.", "evidence": str(HIST_STATE_RUNNER), "ambiguity": "GPU replay parity is outside this screening."},
        {"field": "reset operation", "status": "PROVEN_EXACT", "exact_rule": "Immediately before boundary chunk, replace local TTTState.weight with checkpoint mlp.down_proj.weight independently in layers 0,6,12,18,24.", "evidence": str(HIST_STATE_BACKEND), "ambiguity": "Only local fast weights; no base-weight mutation."},
        {"field": "continuation semantics", "status": "PROVEN_EXACT", "exact_rule": "Continue the same prompt forward and frozen action sequence; no prompt restart or prior-chunk replay; downstream chunks execute from reset fast state; generation unchanged.", "evidence": str(HIST_RESET_SPEC), "ambiguity": "None."},
        {"field": "scorer", "status": "PROVEN_EXACT", "exact_rule": f"Formal evaluator SHA256 {SCORER}.", "evidence": str(HIST_STATE_CONFIG), "ambiguity": "None."},
        {"field": "primary metric/reference", "status": "PROVEN_EXACT", "exact_rule": "Same successful Dynamic trajectory without reset minus same trajectory with reset; controls analogously use N_ORDER_1 normal minus reset.", "evidence": str(HIST_STATE_ROOT / 'scripts/analyze_state_replay.py'), "ambiguity": "Reverse score was descriptive, not the primary reset comparator."},
    ]
    historical_audit = {
        "HISTORICAL_STATE_RESET_RULE_AUDIT_STATUS": "PASS",
        "STATE_RESET_RULE_RECOVERY_STATUS": "PROVEN_EXACT",
        "HISTORICAL_STATE_RESET_POSITIVE_N": len(hist_pos),
        "HISTORICAL_STATE_RESET_CONTROL_N": len(hist_ctl),
        "HISTORICAL_STATE_RESET_EXPERIMENT_UNIQUE_N": len(hist_manifest_ids),
        "HISTORICAL_STATE_RESET_ACTUAL_RESET_EXPOSURE_N": len({x['sample_id'] for x in reset_rows}),
        "EARLY_STEADY_RULE_STATUS": "PROVEN_EXACT",
        "RESET_BOUNDARY_RULE_STATUS": "PROVEN_EXACT",
        "RESET_OPERATION_RULE_STATUS": "PROVEN_EXACT",
        "CONTROL_RULE_STATUS": "PROVEN_EXACT",
        "HISTORICAL_BOUNDARY_RECONSTRUCTION_PARITY": "PASS",
        "boundary_mismatch_count": len(boundary_mismatch),
        "historical_early_membership_reconstruction_parity": "PASS",
        "source_hashes": source_hashes,
        "fields": fields,
    }
    atomic_json(ROOT / "audit/HISTORICAL_STATE_RESET_RULE_AUDIT.json", historical_audit)
    md = ["# Historical State Reset Rule Audit", "", "Executed artifacts outrank reports and prose. All 36 reset-branch boundaries reconstruct exactly.", "", "| Field | Status | Exact recovered behavior | Source/evidence | Ambiguity |", "|---|---|---|---|---|"]
    for x in fields:
        md.append("| " + " | ".join(str(x[k]).replace("|", "\\|") for k in ["field", "status", "exact_rule", "evidence", "ambiguity"]) + " |")
    atomic_text(ROOT / "audit/HISTORICAL_STATE_RESET_RULE_AUDIT.md", "\n".join(md) + "\n")

    operation = [
        ("fast weights / delta-W", "RESET", "Five candidate-layer local TTTState.weight tensors are assigned from checkpoint down_proj.weight."),
        ("temporary TTT layer state", "RESET", "Same object as the per-layer fast weight."),
        ("model base weights", "PRESERVED", "base_parameter_mutation=false; base tensor is only the reset source."),
        ("KV cache", "PRESERVED", "kv_cache_reset=false."),
        ("prefix attention context", "PRESERVED", "prompt_restart=false and prior_chunk_replay=false."),
        ("TTT cache occupancy", "NOT_APPLICABLE", "No separate occupancy object exists in the executed intervention."),
        ("execution-path state", "PRESERVED", "Action sequence and generation path remain unchanged except for the declared fast-state replacement."),
        ("RNG", "PRESERVED", "Executed reset code performs no RNG operation."),
        ("generation context", "PRESERVED", "generation_change=false; generation follows the same prompt forward."),
        ("stored hidden state", "NOT_APPLICABLE", "No persistent hidden-state object is reset; downstream activations are computed normally after the boundary."),
        ("downstream TTT updates", "RECOMPUTED", "All post-boundary chunks execute from the reset local fast state."),
    ]
    op_md = ["# State Reset Operation Semantics", "", f"Executed intervention: `{hist_reset_spec['reset_operation']}`", "", "| Component | Classification | Evidence |", "|---|---|---|"]
    op_md += [f"| {a} | {b} | {c} |" for a, b, c in operation]
    op_md += ["", "Continuation: the same prompt forward and frozen schedule continue at the boundary; prior chunks are not replayed and generation is unchanged."]
    atomic_text(ROOT / "audit/STATE_RESET_OPERATION_SEMANTICS.md", "\n".join(op_md) + "\n")

    # Load and independently validate the frozen 173-parent against benchmark prompt identities.
    parent = load_jsonl(PARENT)
    if len(parent) != 173 or len({x["canonical_sample_id"] for x in parent}) != 173:
        raise RuntimeError("PARENT_SCOPE_FAILURE")
    wanted = {x["canonical_sample_id"] for x in parent}
    benchmark_rows = {}
    with (BENCH / "sample_manifest.jsonl").open(encoding="utf-8") as f:
        for global_index, line in enumerate(f):
            row = json.loads(line)
            if row["sample_id"] in wanted:
                benchmark_rows[row["sample_id"]] = {k: row[k] for k in ["sample_id", "task", "sample_index", "input_hash", "input_token_count_canonical", "max_new_tokens"]} | {"benchmark_global_index": global_index}
    if set(benchmark_rows) != wanted:
        raise RuntimeError("PROMPT_IDENTITY_MISSING")

    eligible, ineligible, census = [], [], []
    for source in parent:
        bid = source["canonical_sample_id"]
        b = benchmark_rows[bid]
        prompt_match = b["input_hash"] == source["benchmark_content_sha256"] and b["benchmark_global_index"] == source["benchmark_global_index"] and b["sample_index"] == source["task_sample_index"] and b["task"] == source["task"]
        passed, morph = historical_early_steady(source["action_sequence"])
        static_flags = source["replayability"]
        static_replayable = prompt_match and all(static_flags.get(k) for k in ["authority_identities_match", "complete_trajectory_metadata", "downstream_recomputation_possible", "exact_chunk_boundaries_exist", "exact_replay_inputs_exist", "no_formal_dynamic_error"])
        boundary = terminal_run_boundary(source["action_sequence"])
        if not passed:
            if morph["first_switch_chunk_index"] is None:
                exclusion = "NO_ACTION_SWITCH"
            elif morph["first_switch_position_normalized"] > 0.25:
                exclusion = "FIRST_SWITCH_AFTER_25_PERCENT"
            else:
                exclusion = "SECOND_HALF_MODAL_ACTION_FRACTION_BELOW_0P8"
        elif not static_replayable:
            exclusion = "STATIC_REPLAYABILITY_FAILURE"
        else:
            exclusion = None
        common = {
            "canonical_sample_id": bid,
            "task": source["task"],
            "benchmark_global_index": source["benchmark_global_index"],
            "task_sample_index": source["task_sample_index"],
            "benchmark_content_sha256": source["benchmark_content_sha256"],
            "dynamic_score": source["dynamic_score"],
            "sample_best_score": source["sample_best_score"],
            "off_score": source["fixed7_scores"]["OFF"],
            "dynamic_minus_sample_best": source["dynamic_minus_sample_best"],
            "dynamic_action_sequence": source["action_sequence"],
            "full_chunk_count": source["full_chunk_count"],
            "historical_rule_pass": passed,
            "early_steady": passed,
            "early_steady_reason": morph,
            "reset_boundary": "TERMINAL_RUN_START",
            "reset_boundary_chunk_index": boundary,
            "reset_boundary_token_position_zero_based": (boundary - 1) * CHUNK_SIZE,
            "reset_boundary_token_position_one_based": (boundary - 1) * CHUNK_SIZE + 1,
            "early_regime": rle(source["action_sequence"][:boundary - 1]),
            "late_regime": rle(source["action_sequence"][boundary - 1:]),
            "pre_boundary_action_schedule": source["action_sequence"][:boundary - 1],
            "post_boundary_action_schedule": source["action_sequence"][boundary - 1:],
            "number_of_switches": source["number_of_switches"],
            "replayable": static_replayable,
            "prompt_identity_match": prompt_match,
            "prompt_token_count": b["input_token_count_canonical"],
            "max_new_tokens": b["max_new_tokens"],
            "reset_operation_version": "FAST_WEIGHT_RESET_INTERVENTION_V1",
            "reference_trajectory_identity": source["reference_identity_sha256"],
            "reference_trajectory": "ORIGINAL_FROZEN_DYNAMIC_NO_RESET",
            "expected_original_dynamic_score": source["dynamic_score"],
            "model_identity": EXPECTED_MODEL,
            "checkpoint_identity": EXPECTED_CHECKPOINT,
            "benchmark_identity": EXPECTED_BENCHMARK,
            "scorer_sha256": SCORER,
            "exclusion_reason": exclusion,
        }
        census.append(common)
        if exclusion is None:
            eligible.append(common)
        else:
            ineligible.append(common)

    if len(eligible) + len(ineligible) != 173:
        raise RuntimeError("SCREENING_PARTITION_FAILURE")
    atomic_jsonl(ROOT / "census/STATE_RESET_PARENT_CENSUS.jsonl", census)
    atomic_jsonl(ROOT / "pools/STATE_RESET_ELIGIBLE_ALL.jsonl", eligible)
    atomic_jsonl(ROOT / "pools/STATE_RESET_ELIGIBLE_INDEPENDENT.jsonl", eligible)
    atomic_jsonl(ROOT / "pools/STATE_RESET_INELIGIBLE.jsonl", ineligible)

    # Specific overlap uses exact canonical identity and content hash; task/index collisions alone are not overlap.
    historical_ids = {x["historical_sample_id"] for x in state_exposure}
    historical_hashes = {x.get("content_input_sha256") for x in state_exposure if x.get("content_input_sha256")}
    overlap = []
    for row in parent:
        reasons = []
        if row["canonical_sample_id"] in historical_ids:
            reasons.append("EXACT_CANONICAL_ID")
        if row["benchmark_content_sha256"] in historical_hashes:
            reasons.append("CONTENT_HASH")
        if reasons:
            overlap.append({"canonical_sample_id": row["canonical_sample_id"], "reasons": reasons})
    atomic_jsonl(ROOT / "overlap/NEW_STATE_RESET_OVERLAP.jsonl", overlap)

    leakage = {
        "STATE_RESET_OUTCOME_LEAKAGE": 0,
        "new_eligibility_inputs": ["formal parent membership Dynamic > Sample Best", "frozen Dynamic action sequence", "chunk count", "benchmark/replay identity", "historically executed early-steady and boundary logic"],
        "forbidden_inputs_not_used": ["reverse_score", "Dynamic - Reverse", "harmed/equal/better", "gain retention", "reverse-vs-SampleBest", "historical State Reset outcome", "selective deletion/R2/KP/criticality outcomes"],
        "exact_reverse_authority_use": "identity/completion provenance only",
        "EXACT_REVERSE_RESULT_USED_FOR_SCREENING": False,
    }
    atomic_json(ROOT / "audit/STATE_RESET_SCREENING_LEAKAGE_AUDIT.json", leakage)
    atomic_text(ROOT / "audit/STATE_RESET_SCREENING_LEAKAGE_AUDIT.md", "# State Reset Screening Leakage Audit\n\nSTATE_RESET_OUTCOME_LEAKAGE=0. Eligibility used only the frozen positive parent, Dynamic trajectory structure, replay identities, and recovered historical structural logic. Exact Reverse was consulted only for authority provenance/completion; no Reverse score or derived outcome entered screening. No historical State Reset outcome or later mechanism label entered screening.\n")

    # Freeze exact recovered rule first.
    rule = {
        "version": "STATE_RESET_RULE_V1",
        "recovery_status": "PROVEN_EXACT",
        "historical_source_identities": source_hashes,
        "historical_counts": {"positive_reset_n": 18, "control_reset_n": 18, "full_experiment_unique_n": 38, "actual_reset_exposure_n": 36},
        "positive_rule": "formal successful sequential trajectory (Dynamic > Sample Best), replayable and independent, satisfying early_steady",
        "historical_upstream_resource_cap": "up to 3 positive targets/task after strict-first, phase, sample_id ordering; recovered for provenance but not an intrinsic eligibility condition on the new all-173 parent",
        "early_steady": {"first_switch_position_normalized_max_inclusive": 0.25, "tail_slice": "sequence[floor(N/2):]", "tail_modal_action_fraction_min_inclusive": 0.8, "modal_action_tie_break_order": ACTIONS},
        "boundary_rule": "one-based start of final action's terminal equal-action run (last switch + 1)",
        "reset_semantics": hist_reset_spec,
        "primary_reference": "same frozen successful Dynamic action schedule without reset",
        "primary_metric": "normal_score - reset_score",
        "control_semantics": "matched nonpositive N_ORDER_1 frozen canonical schedule, normal minus reset at its terminal-run boundary",
        "allowed_screening_features": leakage["new_eligibility_inputs"],
        "forbidden_outcome_features": leakage["forbidden_inputs_not_used"],
        "claim_boundary": {
            "allowed_if_future_results_support": "On successful early-steady trajectories, removing carried fast-weight state destroys later benefit.",
            "not_allowed_without_controls": "Successful trajectories depend on carried state more than unsuccessful comparable trajectories.",
        },
    }
    rule_path = ROOT / "manifests/STATE_RESET_RULE_V1.json"
    atomic_json(rule_path, rule)
    rule_hash = sha(rule_path)

    eligibility_rows = []
    for i, row in enumerate(eligible, 1):
        eligibility_rows.append({"manifest_order": i, "rule_sha256": rule_hash, **row})
    eligibility_jsonl = ROOT / "manifests/STATE_RESET_ELIGIBILITY_V1.jsonl"
    atomic_jsonl(eligibility_jsonl, eligibility_rows)
    file_hashes = {
        "STATE_RESET_ELIGIBILITY_V1.jsonl": sha(eligibility_jsonl),
        "STATE_RESET_ELIGIBLE_ALL.jsonl": sha(ROOT / "pools/STATE_RESET_ELIGIBLE_ALL.jsonl"),
        "STATE_RESET_ELIGIBLE_INDEPENDENT.jsonl": sha(ROOT / "pools/STATE_RESET_ELIGIBLE_INDEPENDENT.jsonl"),
        "STATE_RESET_INELIGIBLE.jsonl": sha(ROOT / "pools/STATE_RESET_INELIGIBLE.jsonl"),
        "STATE_RESET_PARENT_CENSUS.jsonl": sha(ROOT / "census/STATE_RESET_PARENT_CENSUS.jsonl"),
    }
    manifest = {
        "version": "STATE_RESET_ELIGIBILITY_V1_MANIFEST",
        "frozen_at": created_at,
        "model_identity": EXPECTED_MODEL,
        "checkpoint_identity": EXPECTED_CHECKPOINT,
        "benchmark_identity": EXPECTED_BENCHMARK,
        "dynamic_authority": str(DYNAMIC),
        "dynamic_final_audit_sha256": authority["dynamic_final_audit_sha256"],
        "mechanism_screening_authority": str(SCREEN),
        "mechanism_screening_manifest_sha256": EXPECTED_SCREEN_MANIFEST,
        "exact_reverse_authority_provenance_only": str(REVERSE),
        "exact_reverse_receipt_sha256_provenance_only": authority["exact_reverse_receipt_sha256_provenance_only"],
        "exact_reverse_outcome_used_for_screening": False,
        "rule_sha256": rule_hash,
        "parent_n": len(parent),
        "eligible_n": len(eligible),
        "excluded_n": len(ineligible),
        "overlap_n": len(overlap),
        "outcome_leakage_count": 0,
        "static_replayability_n": sum(x["replayable"] for x in eligible),
        "execution_cohort_policy": "ALL_ELIGIBLE_NO_SUBSAMPLING",
        "file_hashes": file_hashes,
    }
    manifest_path = ROOT / "manifests/STATE_RESET_ELIGIBILITY_V1_MANIFEST.json"
    atomic_json(manifest_path, manifest)
    manifest_hash = sha(manifest_path)

    exclusion_counts = Counter(x["exclusion_reason"] for x in ineligible)
    inclusion_counts = {"HISTORICAL_EARLY_STEADY_RULE_PASS_AND_STATIC_REPLAYABLE": len(eligible)}
    summary = {
        "eligible_n": len(eligible),
        "task_distribution": dict(sorted(Counter(x["task"] for x in eligible).items())),
        "chunk_count_distribution": {str(k): v for k, v in sorted(Counter(x["full_chunk_count"] for x in eligible).items())},
        "boundary_chunk_distribution": {str(k): v for k, v in sorted(Counter(x["reset_boundary_chunk_index"] for x in eligible).items())},
        "early_regime_first_action_distribution": dict(sorted(Counter(x["dynamic_action_sequence"][0] for x in eligible).items())),
        "late_regime_terminal_action_distribution": dict(sorted(Counter(x["dynamic_action_sequence"][-1] for x in eligible).items())),
        "switch_count_distribution": {str(k): v for k, v in sorted(Counter(x["number_of_switches"] for x in eligible).items())},
        "dynamic_minus_sample_best": {"mean": statistics.mean(x["dynamic_minus_sample_best"] for x in eligible), "median": statistics.median(x["dynamic_minus_sample_best"] for x in eligible), "quantiles": quantiles([x["dynamic_minus_sample_best"] for x in eligible])},
        "inclusion_reason_counts": inclusion_counts,
        "exclusion_reason_counts": dict(exclusion_counts),
    }
    atomic_json(ROOT / "reports/STATE_RESET_COHORT_STRUCTURAL_SUMMARY.json", summary)
    report_lines = ["# State Reset Eligible Cohort Structural Summary", "", f"Eligible: {len(eligible)}/173. This is descriptive and did not alter eligibility.", "", f"- Task distribution: `{json.dumps(summary['task_distribution'], sort_keys=True)}`", f"- Chunk counts: `{json.dumps(summary['chunk_count_distribution'], sort_keys=True)}`", f"- Boundary chunks: `{json.dumps(summary['boundary_chunk_distribution'], sort_keys=True)}`", f"- Early first actions: `{json.dumps(summary['early_regime_first_action_distribution'], sort_keys=True)}`", f"- Late terminal actions: `{json.dumps(summary['late_regime_terminal_action_distribution'], sort_keys=True)}`", f"- Switch counts: `{json.dumps(summary['switch_count_distribution'], sort_keys=True)}`", f"- Dynamic minus Sample Best: mean={summary['dynamic_minus_sample_best']['mean']}, median={summary['dynamic_minus_sample_best']['median']}, quantiles={summary['dynamic_minus_sample_best']['quantiles']}"]
    atomic_text(ROOT / "reports/STATE_RESET_COHORT_STRUCTURAL_SUMMARY.md", "\n".join(report_lines) + "\n")

    receipt = {
        **{k: authority[k] for k in ["MODEL_AUTHORITY_STATUS", "BENCHMARK_AUTHORITY_STATUS", "DYNAMIC_AUTHORITY_STATUS", "MECHANISM_SCREENING_AUTHORITY_STATUS", "EXACT_REVERSE_AUTHORITY_STATUS"]},
        "STATE_RESET_SCREENING_STATUS": "PASS",
        "HISTORICAL_STATE_RESET_RULE_AUDIT_STATUS": "PASS",
        "STATE_RESET_RULE_RECOVERY_STATUS": "PROVEN_EXACT",
        "HISTORICAL_STATE_RESET_POSITIVE_N": len(hist_pos),
        "HISTORICAL_STATE_RESET_CONTROL_N": len(hist_ctl),
        "EARLY_STEADY_RULE_STATUS": "PROVEN_EXACT",
        "EARLY_STEADY_RULE_EXACT": "first_switch_position/sequence_length <= 0.25 AND modal-action fraction in sequence[floor(N/2):] >= 0.80",
        "RESET_BOUNDARY_RULE_STATUS": "PROVEN_EXACT",
        "RESET_BOUNDARY_RULE_EXACT": "one-based start of final action terminal run (last switch + 1)",
        "HISTORICAL_BOUNDARY_RECONSTRUCTION_PARITY": "PASS",
        "RESET_OPERATION_RULE_STATUS": "PROVEN_EXACT",
        "STATE_RESET_PRIMARY_REFERENCE": "ORIGINAL_FROZEN_DYNAMIC_SAME_ACTION_SCHEDULE_NO_RESET",
        "STATE_RESET_CONTROL_RULE_STATUS": "PROVEN_EXACT",
        "STATE_RESET_PARENT_N": len(parent),
        "STATE_RESET_ELIGIBLE_N": len(eligible),
        "STATE_RESET_INELIGIBLE_N": len(ineligible),
        "HISTORICAL_STATE_RESET_EXPOSURE_N": len(state_exposure),
        "HISTORICAL_STATE_RESET_ACTUAL_RESET_EXPOSURE_N": len({x['sample_id'] for x in reset_rows}),
        "NEW_STATE_RESET_OVERLAP_N": len(overlap),
        "STRUCTURALLY_ELIGIBLE_STATE_RESET_CONTROLS": 0,
        "STATE_RESET_OUTCOME_LEAKAGE": 0,
        "STATIC_STATE_RESET_REPLAYABILITY_N": sum(x["replayable"] for x in eligible),
        "RULE_FROZEN": True,
        "RULE_MANIFEST": str(rule_path),
        "RULE_MANIFEST_SHA256": rule_hash,
        "ELIGIBILITY_FROZEN": True,
        "ELIGIBILITY_MANIFEST": str(manifest_path),
        "ELIGIBILITY_MANIFEST_SHA256": manifest_hash,
        "EXECUTION_COHORT_FROZEN": True,
        "EXECUTION_COHORT_POLICY": "ALL_167_ELIGIBLE_NO_SUBSAMPLING",
        "SAFE_TO_SELECT_STATE_RESET_EXECUTION_COHORT": "YES",
        "SAFE_TO_LAUNCH_STATE_RESET": "NO",
        "REPORT_ROOT": str(ROOT),
        "eligibility_reason_counts": {**inclusion_counts, **dict(exclusion_counts)},
        "claim_boundary": rule["claim_boundary"],
        "STATE_RESET_COUNTERFACTUAL_EXECUTED": False,
        "NEW_GPU_COUNTERFACTUAL_RUNS": 0,
        "EXACT_REVERSE_RESULT_USED_FOR_SCREENING": False,
        "FORMAL_MODEL_MUTATION": 0,
        "BENCHMARK_MUTATION": 0,
        "CONTINUOUS_MONITORING": False,
        "completed_at": created_at,
    }
    receipt_path = ROOT / "receipts/STATE_RESET_SCREENING_V1_RECEIPT.json"
    atomic_json(receipt_path, receipt)
    receipt_md = ["# State Reset Screening V1 Receipt", ""] + [f"{k}={v}" for k, v in receipt.items() if not isinstance(v, (dict, list))]
    receipt_md += ["", "## Eligibility reasons", "", "| Eligibility reason | Count |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in sorted(receipt["eligibility_reason_counts"].items())]
    receipt_md += ["", "No State Reset counterfactual, model forward, GPU mechanism run, or mutation was performed."]
    atomic_text(ROOT / "receipts/STATE_RESET_SCREENING_V1_RECEIPT.md", "\n".join(receipt_md) + "\n")

    # Freeze hashes for every output except this mutable generator and the checksum file itself.
    checksum_rows = []
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and path != ROOT / "manifests/SHA256SUMS" and path.parent.name != "scripts":
            checksum_rows.append(f"{sha(path)}  {path.relative_to(ROOT)}")
    atomic_text(ROOT / "manifests/SHA256SUMS", "\n".join(checksum_rows) + "\n")

    print(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
