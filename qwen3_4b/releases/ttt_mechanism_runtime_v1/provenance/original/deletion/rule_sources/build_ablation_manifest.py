#!/usr/bin/env python3
"""Freeze provenance-defined selective component ablation branches."""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

WS = Path("/home/USER/ttt_mechanism_analysis_v1/selective_state_component_ablation_v1")
ROOT = Path("/home/USER/ttt_mechanism_analysis_v1")
STATE = ROOT / "targeted_fast_weight_state_replay_v1"
MINIMAL = ROOT / "minimal_targeted_causal_replay_v1"
COMPAT = ROOT / "action_conditioned_compatibility_probe_v1"
LAYERS = [0, 6, 12, 18, 24]
ACTION_LAYERS = {"OFF": [], "L0": [0], "L6": [6], "L12": [12],
                 "L18": [18], "L24": [24], "ALL": LAYERS}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(obj):
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


def load_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def atomic_text(path, text):
    path = Path(path)
    tmp = path.with_name("." + path.name + f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_json(path, obj):
    atomic_text(path, json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    atomic_text(path, "".join(json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n" for x in rows))


def category(prefix, next_action):
    written = set().union(*(ACTION_LAYERS[a] for a in prefix)) if prefix else set()
    post = set(ACTION_LAYERS[next_action])
    if "ALL" in prefix:
        return "ALL_PREFIX"
    if "OFF" in prefix or next_action == "OFF":
        return "OFF_INVOLVING"
    if len(written) == 1 and len(post) == 1 and written.isdisjoint(post):
        return "SINGLE_LAYER_DISJOINT"
    if written & post:
        return "OVERLAPPING_LAYER"
    return "COMPLEX_PREFIX"


def first_real_layer(prefix):
    for action in prefix:
        layers = ACTION_LAYERS[action]
        if action == "ALL":
            raise RuntimeError("ALL_PREFIX_MUST_USE_LOO")
        if layers:
            return layers[0]
    return None


def main():
    state_manifest = load_jsonl(STATE / "manifests/STATE_REPLAY_MANIFEST.jsonl")
    result_files = sorted((STATE / "instrumented_results").glob("*.json"))
    results = [json.load(path.open(encoding="utf-8")) for path in result_files]
    result_by_trajectory = {x["trajectory_id"]: x for x in results}
    reset_positive = {x["sample_id"]: x for x in state_manifest if x["trajectory_role"] == "P_SUCCESS_RESET_AT_SWITCH"}
    normal_positive = {x["sample_id"]: x for x in state_manifest if x["trajectory_role"] == "P_SUCCESS" and x["sample_id"] in reset_positive}
    reverse_positive = {x["sample_id"]: x for x in state_manifest if x["trajectory_role"] == "P_REVERSE" and x["sample_id"] in reset_positive}
    normal_negative = {x["sample_id"]: x for x in state_manifest if x["trajectory_role"] == "N_ORDER_1"}
    reset_negative = {x["sample_id"]: x for x in state_manifest if x["trajectory_role"] == "N_CANONICAL_RESET_AT_BOUNDARY"}
    if not all(len(x) == 18 for x in [reset_positive, normal_positive, reverse_positive, normal_negative, reset_negative]):
        raise RuntimeError("FROZEN_18_PLUS_18_SCOPE_MISMATCH")

    branches = []
    cohort_rows = []
    category_counts = Counter()
    switch_match = 0

    def add_branch(row, cohort, role, layers, component_source, norms, category_name):
        branch = {
            "manifest_order": len(branches) + 1, "cohort": cohort,
            "intervention_role": role, "sample_id": row["sample_id"], "phase": row["phase"],
            "task": row["task"], "source_trajectory_id": row["trajectory_id"],
            "source_manifest_order": row["manifest_order"], "action_sequence": row["action_sequence"],
            "decision_chunk_count": row["decision_chunk_count"],
            "switch_boundary": row["switch_boundary_if_any"], "ablate_layers": list(layers),
            "component_source_rule": component_source, "transition_category": category_name,
            "pre_ablation_component_norms": {str(layer): norms[layer] for layer in layers},
            "outcome_adaptive_selection": False, "same_successful_or_control_suffix": True,
            "generation_update_count": 0, "new_search": False,
        }
        branch["intervention_id"] = canonical({"version": "SELECTIVE_STATE_COMPONENT_ABLATION_V1",
                                                "sample_id": row["sample_id"], "role": role,
                                                "layers": list(layers), "boundary": row["switch_boundary_if_any"]})[:24]
        branches.append(branch)

    for cohort, mapping in [("POSITIVE", normal_positive), ("NEGATIVE", normal_negative)]:
        for sid, row in sorted(mapping.items(), key=lambda z: (z[1]["phase"], z[1]["task"], z[0])):
            boundary = int(row["switch_boundary_if_any"])
            prefix = row["action_sequence"][:boundary - 1]
            next_action = row["action_sequence"][boundary - 1]
            written = sorted(set().union(*(ACTION_LAYERS[a] for a in prefix)) if prefix else set())
            post = sorted(ACTION_LAYERS[next_action])
            cat = category(prefix, next_action)
            authority = result_by_trajectory[row["trajectory_id"]]
            events = authority["state_events"]
            norms = {layer: float(next(e["fast_weight_delta_norm_before"] for e in events
                                       if e["layer"] == layer and e["chunk_index"] == boundary)) for layer in LAYERS}
            nonzero = [layer for layer in LAYERS if norms[layer] > 0]
            zero = [layer for layer in LAYERS if norms[layer] == 0]
            if set(nonzero) != set(written):
                raise RuntimeError(f"PROVENANCE_STATE_MISMATCH {sid} written={written} nonzero={nonzero}")
            category_counts[(cohort, cat)] += 1
            if cohort == "POSITIVE":
                reset_row = reset_positive[sid]
                if (reset_row["switch_boundary_if_any"] == boundary and reset_row["reset_boundary_if_any"] == boundary
                        and reset_row["action_sequence"] == row["action_sequence"]):
                    switch_match += 1
                else:
                    raise RuntimeError("SWITCH_BOUNDARY_IDENTITY_MISMATCH")
            primary_layers = []
            if cat == "ALL_PREFIX":
                for layer in nonzero:
                    add_branch(row, cohort, f"{cohort}_ALL_PREFIX_LOO", [layer],
                               "LEAVE_ONE_NONZERO_COMPONENT_OUT", norms, cat)
            else:
                first = first_real_layer(prefix)
                if first is not None:
                    primary_layers = [first]
                    if norms[first] <= 0:
                        raise RuntimeError("PRIMARY_COMPONENT_NOT_NONZERO")
                    add_branch(row, cohort, f"{cohort}_PRIMARY_SELECTIVE", [first],
                               "FIRST_REAL_EARLY_ACTION_PROVENANCE", norms, cat)
            sham_layer = min(zero) if zero and cohort == "POSITIVE" else None
            if sham_layer is not None:
                add_branch(row, cohort, "POSITIVE_ZERO_STATE_SHAM", [sham_layer],
                           "LOWEST_ZERO_STATE_LAYER_NOT_IN_PREFIX_WRITTEN_SET", norms, cat)
            cohort_rows.append({
                "cohort": cohort, "sample_id": sid, "phase": row["phase"], "task": row["task"],
                "source_trajectory_id": row["trajectory_id"], "switch_boundary": boundary,
                "prefix_sequence": prefix, "suffix_sequence": row["action_sequence"][boundary - 1:],
                "prefix_written_layer_set": written, "post_switch_primary_action": next_action,
                "post_switch_primary_action_layer_set": post, "transition_category": cat,
                "primary_prefix_component_set": primary_layers,
                "nonzero_component_set_at_switch": nonzero, "zero_component_set_at_switch": zero,
                "sham_layer": sham_layer, "selective_intervention_available": bool(primary_layers) or cat == "ALL_PREFIX",
                "normal_score_authority_trajectory_id": row["trajectory_id"],
                "full_reset_authority_trajectory_id": (reset_positive[sid]["trajectory_id"] if cohort == "POSITIVE"
                                                        else reset_negative[sid]["trajectory_id"]),
                "exact_reverse_authority_trajectory_id": reverse_positive[sid]["trajectory_id"] if cohort == "POSITIVE" else None,
            })

    if switch_match != 18:
        raise RuntimeError("SWITCH_BOUNDARY_IDENTITY_COUNT")
    if category_counts[("POSITIVE", "SINGLE_LAYER_DISJOINT")] != 9 or category_counts[("POSITIVE", "ALL_PREFIX")] != 7 or category_counts[("POSITIVE", "OVERLAPPING_LAYER")] != 2:
        raise RuntimeError("POSITIVE_CATEGORY_SCOPE_MISMATCH")
    if len(branches) != 89:
        raise RuntimeError(f"INTERVENTION_BRANCH_COUNT_MISMATCH {len(branches)}")

    write_jsonl(WS / "manifests/COMPONENT_ABLATION_MANIFEST.jsonl", branches)
    write_jsonl(WS / "manifests/COMPONENT_PROVENANCE.jsonl", cohort_rows)
    hypothesis = {
        "version": "COMPONENT_ABLATION_HYPOTHESIS_REGISTRY_V1",
        "frozen_before_new_ablation_scores": True,
        "H1": "Selective early component ablation hurts positives.",
        "H2": "Effect is strong in disjoint transitions.",
        "H3": "Matched negatives show much weaker effect.",
        "H4": "Zero-state sham ablation has no effect.",
        "H5": "Selective ablation explains a substantial fraction of full-reset loss.",
        "primary_unit": "canonical_sample_id", "bootstrap_reps": 10000, "bootstrap_seed": 20260820,
        "all_prefix_aggregation": "MEAN_ACROSS_FIVE_PREDECLARED_SINGLE_COMPONENT_LOO_BRANCHES",
        "negative_no_component_rule": "retain in cohort; mark selective effect unavailable; never invent OFF component",
        "pairwise_component_ablation": False, "outcome_adaptive_selection": False,
    }
    atomic_json(WS / "configs/COMPONENT_ABLATION_HYPOTHESIS_REGISTRY.json", hypothesis)
    source_audits = {
        "minimal_targeted_causal_replay_v1": MINIMAL / "audits/FINAL_INTEGRITY_AUDIT.json",
        "targeted_fast_weight_state_replay_v1": STATE / "audits/FINAL_INTEGRITY_AUDIT.json",
        "action_conditioned_compatibility_probe_v1": COMPAT / "audits/FINAL_INTEGRITY_AUDIT.json",
    }
    authority = {name: {"path": str(path), "sha256": sha(path)} for name, path in source_audits.items()}
    authority.update({
        "state_replay_manifest": {"path": str(STATE / "manifests/STATE_REPLAY_MANIFEST.jsonl"),
                                  "sha256": sha(STATE / "manifests/STATE_REPLAY_MANIFEST.jsonl")},
        "state_replay_raw_result_manifest": {"path": str(STATE / "audits/RAW_RESULT_MANIFEST.sha256"),
                                             "sha256": sha(STATE / "audits/RAW_RESULT_MANIFEST.sha256")},
    })
    atomic_json(WS / "authority/SOURCE_AUTHORITY_SNAPSHOT.json", authority)
    config = {
        "version": "SELECTIVE_STATE_COMPONENT_ABLATION_V1", "cpuset": "8-23", "workers": 1,
        "candidate_layers": LAYERS, "fast_weight": "mlp.down_proj.weight", "ttt_lr": 1.0,
        "delta_clip_frobenius": 1e-5, "batch_size": 1, "apply_then_update": True,
        "complete_chunk_size": 1024, "generation_update": False,
        "model_identity": "ca51ee90d20ee09efbe0b31dd21bf9faa70275508316d1ba7cb1387ee4f0725f",
        "checkpoint_identity": "f4e264ed6f5671f123d6c8a27c69bce3af5ffcba518c9e298398314c6147d9fd",
        "formal_executor_sha256": "293886005cb8f78c943f0affe3e3b383d6de238f4bc825a5aea28a6e6dc66f6e",
        "instrumented_backend": str(STATE / "scripts/instrumented_backend.py"),
        "instrumented_backend_sha256": sha(STATE / "scripts/instrumented_backend.py"),
        "new_dynamic_search_count": 0, "router_training_count": 0, "new_mining_count": 0,
        "new_final_protected_path_access_count": 0,
    }
    atomic_json(WS / "configs/SELECTIVE_COMPONENT_ABLATION_CONFIG.json", config)
    freeze = {
        "manifest_frozen_before_new_scores": True,
        "component_ablation_manifest_sha256": sha(WS / "manifests/COMPONENT_ABLATION_MANIFEST.jsonl"),
        "component_provenance_sha256": sha(WS / "manifests/COMPONENT_PROVENANCE.jsonl"),
        "component_ablation_hypothesis_sha256": sha(WS / "configs/COMPONENT_ABLATION_HYPOTHESIS_REGISTRY.json"),
        "positive_primary_n": 18, "negative_control_n": 18, "negative_selective_eligible_n": 17,
        "switch_boundary_identity_match": True, "switch_boundary_identity_match_count": 18,
        "intervention_branch_count": len(branches),
        "branch_role_counts": dict(Counter(x["intervention_role"] for x in branches)),
        "positive_category_counts": {k[1]: v for k, v in category_counts.items() if k[0] == "POSITIVE"},
        "new_ablation_result_read_before_freeze": False,
    }
    atomic_json(WS / "audits/PRE_FORWARD_MANIFEST_FREEZE.json", freeze)
    print("POSITIVE_PRIMARY_N=18")
    print("NEGATIVE_CONTROL_N=18")
    print("INTERVENTION_BRANCH_COUNT=" + str(len(branches)))
    print("COMPONENT_ABLATION_MANIFEST_SHA256=" + freeze["component_ablation_manifest_sha256"])
    print("COMPONENT_ABLATION_HYPOTHESIS_SHA256=" + freeze["component_ablation_hypothesis_sha256"])


if __name__ == "__main__":
    main()
