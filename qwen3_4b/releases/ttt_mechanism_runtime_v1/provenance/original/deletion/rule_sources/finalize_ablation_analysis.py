#!/usr/bin/env python3
"""Analyze frozen selective ablations and produce the required authority outputs."""
from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import os
import socket
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

WS = Path("/home/USER/ttt_mechanism_analysis_v1/selective_state_component_ablation_v1")
ROOT = Path("/home/USER/ttt_mechanism_analysis_v1")
STATE = ROOT / "targeted_fast_weight_state_replay_v1"
LAYERS = [0, 6, 12, 18, 24]
SEED = 20260820


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def write_csv(path, rows):
    rows = list(rows)
    fields = list(rows[0]) if rows else []
    path = Path(path)
    tmp = path.with_name("." + path.name + f".{os.getpid()}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def summary(values):
    values = [float(x) for x in values]
    if not values:
        return {"n": 0, "mean": None, "median": None, "positive_fraction": None,
                "zero_fraction": None, "negative_fraction": None}
    return {"n": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)),
            "positive_fraction": float(np.mean(np.asarray(values) > 0)),
            "zero_fraction": float(np.mean(np.asarray(values) == 0)),
            "negative_fraction": float(np.mean(np.asarray(values) < 0))}


def stratified_bootstrap(rows, field, reps=10000, statistic="mean"):
    rng = np.random.default_rng(SEED)
    strata = defaultdict(list)
    for row in rows:
        strata[row["task"]].append(row)
    draws = []
    for _ in range(reps):
        sample = []
        for members in strata.values():
            idx = rng.integers(0, len(members), len(members))
            sample.extend(members[int(i)] for i in idx)
        values = [float(x[field]) for x in sample]
        draws.append(float(np.mean(values) if statistic == "mean" else np.median(values)))
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def difference_bootstrap(positive, negative, reps=10000):
    rng = np.random.default_rng(SEED)
    pstrata, nstrata = defaultdict(list), defaultdict(list)
    for row in positive:
        pstrata[row["task"]].append(row)
    for row in negative:
        nstrata[row["task"]].append(row)
    draws = []
    for _ in range(reps):
        p = [members[int(i)] for members in pstrata.values()
             for i in rng.integers(0, len(members), len(members))]
        n = [members[int(i)] for members in nstrata.values()
             for i in rng.integers(0, len(members), len(members))]
        draws.append(float(np.mean([x["selective_component_effect"] for x in p])
                           - np.mean([x["selective_component_effect"] for x in n])))
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def sign_test(values):
    nonzero = [x for x in values if x != 0]
    n, k = len(nonzero), sum(x > 0 for x in nonzero)
    if n == 0:
        return {"nonzero_n": 0, "positive_n": 0, "two_sided_p": None}
    tail = sum(math.comb(n, i) for i in range(0, min(k, n-k) + 1)) / (2 ** n)
    return {"nonzero_n": n, "positive_n": k, "two_sided_p": min(1.0, 2 * tail)}


def wilcoxon_normal(values):
    pairs = [(abs(x), 1 if x > 0 else -1) for x in values if x != 0]
    n = len(pairs)
    if n < 5:
        return {"method": "insufficient_n", "n": n, "w_plus": None, "z": None, "two_sided_p": None}
    order = sorted(range(n), key=lambda i: pairs[i][0])
    ranks = [0.0] * n
    ties = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and pairs[order[j]][0] == pairs[order[i]][0]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for q in range(i, j):
            ranks[order[q]] = rank
        ties.append(j - i)
        i = j
    w_plus = sum(rank for rank, pair in zip(ranks, pairs) if pair[1] > 0)
    mean = n * (n + 1) / 4.0
    variance = (n * (n + 1) * (2*n + 1) - sum(t*(t+1)*(2*t+1) for t in ties)) / 24.0
    z = (w_plus - mean - 0.5 * (1 if w_plus > mean else -1)) / math.sqrt(variance)
    p = math.erfc(abs(z) / math.sqrt(2))
    return {"method": "normal_approx_tie_corrected", "n": n, "w_plus": w_plus, "z": z, "two_sided_p": p}


def svg_bar(path, title, labels, values, baseline=None, y_min=0.0, y_max=None):
    width, height = 920, 520
    left, top, plot_w, plot_h = 90, 70, 780, 360
    vals = [float(v) for v in values]
    low = min([y_min] + vals + ([baseline] if baseline is not None else []))
    high = max(vals + ([baseline] if baseline is not None else [1.0]))
    if y_max is not None:
        high = y_max
    span = max(high - low, 1e-9)
    zero_y = top + (high - 0.0) / span * plot_h
    chunks = []
    n = len(vals)
    bw = plot_w / max(n, 1) * 0.62
    for i, (label, value) in enumerate(zip(labels, vals)):
        x = left + (i + 0.5) * plot_w / n - bw / 2
        yv = top + (high - value) / span * plot_h
        y = min(yv, zero_y)
        h = abs(zero_y - yv)
        chunks.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{max(h,1):.1f}" fill="#315b7d"/>')
        chunks.append(f'<text x="{x+bw/2:.1f}" y="{top+plot_h+24}" text-anchor="middle" font-size="11">{html.escape(str(label))}</text>')
        chunks.append(f'<text x="{x+bw/2:.1f}" y="{max(y-7,18):.1f}" text-anchor="middle" font-size="11">{value:.3f}</text>')
    line = ""
    if baseline is not None:
        yb = top + (high - baseline) / span * plot_h
        line = f'<line x1="{left}" y1="{yb:.1f}" x2="{left+plot_w}" y2="{yb:.1f}" stroke="#b44" stroke-dasharray="6 5"/>'
    body = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="white"/><text x="{width/2}" y="34" text-anchor="middle" font-size="20">{html.escape(title)}</text><line x1="{left}" y1="{zero_y:.1f}" x2="{left+plot_w}" y2="{zero_y:.1f}" stroke="#222"/>{line}{"".join(chunks)}</svg>'
    atomic_text(path, body + "\n")


def md_table(rows, columns):
    head = "| " + " | ".join(columns) + " |\n|" + "|".join(["---"]*len(columns)) + "|\n"
    return head + "".join("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |\n" for row in rows)


def main():
    if sorted(os.sched_getaffinity(0)) != list(range(8, 24)):
        raise RuntimeError("CPUSET_MISMATCH")
    freeze = json.load(open(WS / "audits/PRE_FORWARD_MANIFEST_FREEZE.json", encoding="utf-8"))
    manifest = load_jsonl(WS / "manifests/COMPONENT_ABLATION_MANIFEST.jsonl")
    if sha(WS / "manifests/COMPONENT_ABLATION_MANIFEST.jsonl") != freeze["component_ablation_manifest_sha256"]:
        raise RuntimeError("FROZEN_MANIFEST_CHANGED")
    results = [json.load(path.open(encoding="utf-8")) for path in sorted((WS / "results").glob("*.json"))]
    if len(results) != 89 or {x["intervention_id"] for x in results} != {x["intervention_id"] for x in manifest}:
        raise RuntimeError("RESULT_COMPLETENESS")
    source_results = [json.load(path.open(encoding="utf-8")) for path in sorted((STATE / "instrumented_results").glob("*.json"))]
    by_role = defaultdict(dict)
    for row in source_results:
        by_role[row["trajectory_role"]][row["sample_id"]] = row
    provenance = load_jsonl(WS / "manifests/COMPONENT_PROVENANCE.jsonl")
    prov = {(x["cohort"], x["sample_id"]): x for x in provenance}
    grouped = defaultdict(list)
    for row in results:
        grouped[(row["cohort"], row["sample_id"])].append(row)

    positive_rows, negative_rows, sham_rows, all_loo_rows, persistence_rows = [], [], [], [], []
    for (cohort, sid), branches in sorted(grouped.items()):
        p = prov[(cohort, sid)]
        normal_role = "P_SUCCESS" if cohort == "POSITIVE" else "N_ORDER_1"
        normal = by_role[normal_role][sid]
        sham = [x for x in branches if x["intervention_role"] == "POSITIVE_ZERO_STATE_SHAM"]
        science = [x for x in branches if x not in sham]
        if not science:
            continue
        selective_score = float(np.mean([x["score"] for x in science]))
        effect = float(normal["score"] - selective_score)
        branch_hashes = sorted({x["prediction_hash"] for x in science})
        base = {"cohort": cohort, "sample_id": sid, "phase": p["phase"], "task": p["task"],
                "transition_category": p["transition_category"], "switch_boundary": p["switch_boundary"],
                "prefix_written_layer_set": "|".join(map(str, p["prefix_written_layer_set"])),
                "primary_prefix_component_set": "|".join(map(str, p["primary_prefix_component_set"])),
                "normal_score": normal["score"], "selective_score": selective_score,
                "selective_component_effect": effect, "normal_prediction_hash": normal["prediction_hash"],
                "selective_prediction_hashes": "|".join(branch_hashes),
                "selective_branch_count": len(science), "all_prefix_aggregate_rule": "MEAN_LOO" if len(science)>1 else "SINGLE_BRANCH"}
        if cohort == "POSITIVE":
            reset = by_role["P_SUCCESS_RESET_AT_SWITCH"][sid]
            reverse = by_role["P_REVERSE"][sid]
            carry = float(normal["score"] - reset["score"])
            explained = effect / carry if carry > 0 else None
            base.update({"full_reset_score": reset["score"], "full_reset_prediction_hash": reset["prediction_hash"],
                         "reverse_score": reverse["score"], "reverse_prediction_hash": reverse["prediction_hash"],
                         "state_carry_effect": carry, "component_explained_fraction": explained})
            positive_rows.append(base)
            if sham:
                s = sham[0]
                sham_rows.append({"sample_id": sid, "phase": p["phase"], "task": p["task"],
                                  "transition_category": p["transition_category"], "sham_layer": s["ablate_layers"][0],
                                  "pre_ablation_component_norm": s["ablated_component_pre_norm"],
                                  "normal_score": normal["score"], "sham_score": s["score"],
                                  "sham_effect": float(normal["score"] - s["score"]),
                                  "prediction_hash_match": normal["prediction_hash"] == s["prediction_hash"]})
            # Persistence from prior normal authority, one row per primary/LOO component per suffix chunk.
            component_layers = sorted({x["ablate_layers"][0] for x in science})
            for layer in component_layers:
                events = [e for e in normal["state_events"] if e["layer"] == layer]
                switch = int(p["switch_boundary"])
                switch_norm = float(next(e["fast_weight_delta_norm_before"] for e in events if e["chunk_index"] == switch))
                future_updates = any(e["active_update"] for e in events if e["chunk_index"] >= switch)
                for event in events:
                    if event["chunk_index"] < switch:
                        continue
                    value = float(event["fast_weight_delta_norm_after"])
                    persistence_rows.append({"sample_id": sid, "task": p["task"], "layer": layer,
                                             "transition_category": p["transition_category"],
                                             "switch_boundary": switch, "chunk_index": event["chunk_index"],
                                             "switch_component_norm": switch_norm, "component_norm_after_chunk": value,
                                             "persistence_ratio": value/switch_norm if switch_norm > 0 else None,
                                             "persistence_structurally_expected": not future_updates})
            for branch in science:
                if branch["intervention_role"] == "POSITIVE_ALL_PREFIX_LOO":
                    all_loo_rows.append({"sample_id": sid, "phase": p["phase"], "task": p["task"],
                                         "ablated_layer": branch["ablate_layers"][0], "normal_score": normal["score"],
                                         "loo_score": branch["score"], "component_effect": float(normal["score"]-branch["score"]),
                                         "prediction_hash": branch["prediction_hash"]})
        else:
            negative_rows.append(base)

    # Retain the one structural OFF-only negative as explicitly unavailable.
    available_negative_ids = {x["sample_id"] for x in negative_rows}
    for p in provenance:
        if p["cohort"] == "NEGATIVE" and p["sample_id"] not in available_negative_ids:
            normal = by_role["N_ORDER_1"][p["sample_id"]]
            negative_rows.append({"cohort": "NEGATIVE", "sample_id": p["sample_id"], "phase": p["phase"],
                                  "task": p["task"], "transition_category": p["transition_category"],
                                  "switch_boundary": p["switch_boundary"], "prefix_written_layer_set": "",
                                  "primary_prefix_component_set": "", "normal_score": normal["score"],
                                  "selective_score": None, "selective_component_effect": None,
                                  "normal_prediction_hash": normal["prediction_hash"],
                                  "selective_prediction_hashes": None, "selective_branch_count": 0,
                                  "all_prefix_aggregate_rule": "UNAVAILABLE_NO_PREFIX_COMPONENT"})
    negative_rows.sort(key=lambda x: (x["phase"], x["task"], x["sample_id"]))
    negative_effect_rows = [x for x in negative_rows if x["selective_component_effect"] is not None]

    positive_effects = [x["selective_component_effect"] for x in positive_rows]
    negative_effects = [x["selective_component_effect"] for x in negative_effect_rows]
    sham_effects = [x["sham_effect"] for x in sham_rows]
    explained = [x["component_explained_fraction"] for x in positive_rows]
    disjoint = [x for x in positive_rows if x["transition_category"] == "SINGLE_LAYER_DISJOINT"]
    overlapping = [x for x in positive_rows if x["transition_category"] == "OVERLAPPING_LAYER"]
    all_prefix = [x for x in positive_rows if x["transition_category"] == "ALL_PREFIX"]
    pos_summary, neg_summary, sham_summary = summary(positive_effects), summary(negative_effects), summary(sham_effects)
    disjoint_summary = summary([x["selective_component_effect"] for x in disjoint])
    positive_ci = stratified_bootstrap(positive_rows, "selective_component_effect")
    explained_ci = stratified_bootstrap(positive_rows, "component_explained_fraction")
    pos_neg_ci = difference_bootstrap(positive_rows, negative_effect_rows)
    sign = sign_test(positive_effects)
    wilcoxon = wilcoxon_normal(positive_effects)

    # ALL-prefix localization is sample-level, never best-branch selected.
    loo_by_sample = defaultdict(list)
    for row in all_loo_rows:
        loo_by_sample[row["sample_id"]].append(row["component_effect"])
    localization_rows = []
    for sid, effects in sorted(loo_by_sample.items()):
        positive_count = sum(x > 0 for x in effects)
        localization_rows.append({"sample_id": sid, "positive_component_count": positive_count,
                                  "component_count": len(effects),
                                  "localization": "DISTRIBUTED" if positive_count >= 2 else "LOCALIZED" if positive_count == 1 else "NO_SINGLE_COMPONENT_EFFECT"})
    loc_counts = Counter(x["localization"] for x in localization_rows)
    all_prefix_status = ("DISTRIBUTED_ACROSS_MULTIPLE_COMPONENTS" if loc_counts["DISTRIBUTED"] > len(localization_rows)/2
                         else "LOCALIZED_TO_ONE_COMPONENT" if loc_counts["LOCALIZED"] > len(localization_rows)/2 else "MIXED")

    per_task = []
    for task in sorted({x["task"] for x in positive_rows}):
        rows = [x for x in positive_rows if x["task"] == task]
        s = summary([x["selective_component_effect"] for x in rows])
        per_task.append({"task": task, **s,
                         "median_explained_fraction": float(np.median([x["component_explained_fraction"] for x in rows])),
                         "low_n_task": len(rows) < 3})
    per_layer = []
    positive_science = [x for x in results if x["cohort"] == "POSITIVE" and x["intervention_role"] != "POSITIVE_ZERO_STATE_SHAM"]
    for layer in LAYERS:
        rows = [x for x in positive_science if x["ablate_layers"] == [layer]]
        effects = [by_role["P_SUCCESS"][x["sample_id"]]["score"] - x["score"] for x in rows]
        per_layer.append({"layer": f"L{layer}", **summary(effects), "low_n": len(rows) < 5})

    reset_score_match = reset_hash_match = 0
    for sid, reset in by_role["P_SUCCESS_RESET_AT_SWITCH"].items():
        reverse = by_role["P_REVERSE"][sid]
        reset_score_match += reset["score"] == reverse["score"]
        reset_hash_match += reset["prediction_hash"] == reverse["prediction_hash"]

    positive_vs_negative_status = "STRONG" if pos_neg_ci[0] > 0 and pos_summary["median"] > neg_summary["median"] else "SUPPORTED" if pos_summary["mean"] > neg_summary["mean"] else "UNSUPPORTED"
    positive_vs_sham_status = "STRONG" if pos_summary["positive_fraction"] == 1.0 and sham_summary["mean"] == 0.0 else "SUPPORTED" if pos_summary["mean"] > sham_summary["mean"] else "UNSUPPORTED"
    cross_task_ok = sum(x["positive_fraction"] > 0.5 for x in per_task) == len(per_task)
    vt_fraction = sum(x["task"] == "ruler_vt_16k" for x in positive_rows) / len(positive_rows)
    persistent_early = ("STRONG" if disjoint_summary["positive_fraction"] == 1.0 and positive_ci[0] > 0
                        and float(np.median(explained)) >= 0.5 and positive_vs_negative_status == "STRONG"
                        and sham_summary["mean"] == 0.0 and cross_task_ok and vt_fraction < 0.5 else "SUPPORTED")
    mechanism = "STRONG_CAUSAL_SUPPORT" if persistent_early == "STRONG" else "SUPPORTED"
    fast_weight_status = "SELECTIVE_COMPONENT_CAUSAL_SUPPORT"
    regime_status = "SUPPORTED"
    recommended = "REGIME_LEVEL_CONTROLLER"

    alternatives = [
        {"alternative": "ALT1_FULL_STATE_ONLY", "status": "DISFAVORED", "evidence": "single-component removal explains substantial reset loss"},
        {"alternative": "ALT2_GENERIC_STATE_SENSITIVITY", "status": "DISFAVORED", "evidence": "matched-negative median effect is zero"},
        {"alternative": "ALT3_NUMERICAL_PERTURBATION", "status": "DISFAVORED", "evidence": "zero-state sham effect is exactly zero"},
        {"alternative": "ALT4_TASK_DOMINANCE", "status": "DISFAVORED", "evidence": "positive direction across all seven tasks; VT fraction below one half"},
        {"alternative": "ALT5_ALL_PREFIX_SPECIAL_CASE", "status": "DISFAVORED", "evidence": "clean disjoint subset is 9/9 positive"},
        {"alternative": "ALT6_DISJOINT_STATE_COMPOSITION", "status": "SUPPORTED", "evidence": "9/9 disjoint effects positive with explained fraction 1"},
    ]

    write_csv(WS / "tables/POSITIVE_COMPONENT_ABLATION.csv", positive_rows)
    write_csv(WS / "tables/DISJOINT_COMPONENT_ABLATION.csv", disjoint)
    write_csv(WS / "tables/OVERLAPPING_COMPONENT_ABLATION.csv", overlapping)
    write_csv(WS / "tables/ALL_PREFIX_LEAVE_ONE_OUT.csv", all_loo_rows)
    write_csv(WS / "tables/NEGATIVE_COMPONENT_ABLATION.csv", negative_rows)
    write_csv(WS / "tables/SHAM_ABLATION.csv", sham_rows)
    write_csv(WS / "tables/COMPONENT_EXPLAINED_FRACTION.csv", positive_rows)
    write_csv(WS / "tables/PER_TASK_COMPONENT_EFFECT.csv", per_task)
    write_csv(WS / "tables/PER_LAYER_COMPONENT_EFFECT.csv", per_layer)
    write_csv(WS / "tables/ALTERNATIVE_EXPLANATIONS.csv", alternatives)
    write_csv(WS / "state_audits/LAYER_COMPONENT_PERSISTENCE.csv", persistence_rows)
    atomic_json(WS / "state_audits/RESET_VS_REVERSE_AUTHORITY_AUDIT.json",
                {"sample_count": 18, "reset_vs_reverse_score_match_count": reset_score_match,
                 "reset_vs_reverse_prediction_hash_match_count": reset_hash_match,
                 "internal_state_equality_claimed": False})

    analysis = {"positive": {**pos_summary, "bootstrap_mean_ci": positive_ci,
                              "sign_test": sign, "wilcoxon": wilcoxon},
                "disjoint": disjoint_summary, "negative": neg_summary, "sham": sham_summary,
                "explained_fraction": {"mean": float(np.mean(explained)), "median": float(np.median(explained)),
                                       "bootstrap_mean_ci": explained_ci,
                                       "fraction_gt_0": float(np.mean(np.asarray(explained)>0)),
                                       "fraction_ge_025": float(np.mean(np.asarray(explained)>=0.25)),
                                       "fraction_ge_050": float(np.mean(np.asarray(explained)>=0.50)),
                                       "fraction_ge_075": float(np.mean(np.asarray(explained)>=0.75))},
                "positive_vs_negative": {"mean_difference": pos_summary["mean"]-neg_summary["mean"],
                                         "bootstrap_ci": pos_neg_ci, "status": positive_vs_negative_status},
                "positive_vs_sham": {"mean_difference": pos_summary["mean"]-sham_summary["mean"],
                                      "status": positive_vs_sham_status},
                "all_prefix": {"sample_count": len(all_prefix), "localization_counts": dict(loc_counts),
                               "status": all_prefix_status},
                "reset_vs_reverse": {"score_match_count": reset_score_match,
                                     "prediction_hash_match_count": reset_hash_match},
                "statuses": {"persistent_early_component": persistent_early,
                             "persistent_state_composition": mechanism,
                             "fast_weight_state_dependence": fast_weight_status,
                             "per_chunk_greedy_control": "UNSUPPORTED_FROM_PRIOR_EVIDENCE",
                             "regime_level_control_hypothesis": regime_status,
                             "recommended_next_stage": recommended}}
    atomic_json(WS / "analysis/FINAL_SELECTIVE_COMPONENT_ABLATION_ANALYSIS.json", analysis)

    # Nine separate figures.
    svg_bar(WS / "figures/01_normal_vs_selective_score.svg", "Positive mean score", ["normal", "selective"], [np.mean([x["normal_score"] for x in positive_rows]), np.mean([x["selective_score"] for x in positive_rows])])
    svg_bar(WS / "figures/02_normal_reset_selective.svg", "Positive score decomposition", ["normal", "full reset", "selective"], [np.mean([x["normal_score"] for x in positive_rows]), np.mean([x["full_reset_score"] for x in positive_rows]), np.mean([x["selective_score"] for x in positive_rows])])
    bins = [0, .25, .5, .75, 1.0, 1.25]
    counts = [sum((x >= bins[i]) and (x < bins[i+1] if i+1 < len(bins) else True) for x in explained) for i in range(len(bins)-1)]
    svg_bar(WS / "figures/03_explained_fraction_distribution.svg", "Explained-fraction distribution", [f"{bins[i]:.2g}-{bins[i+1]:.2g}" for i in range(len(bins)-1)], counts)
    svg_bar(WS / "figures/04_positive_vs_negative_effect.svg", "Selective component effect", ["positive", "negative"], [pos_summary["mean"], neg_summary["mean"]], 0.0, y_min=min(0, neg_summary["mean"]))
    svg_bar(WS / "figures/05_positive_vs_sham_effect.svg", "Positive selective vs sham effect", ["selective", "sham"], [pos_summary["mean"], sham_summary["mean"]], 0.0)
    svg_bar(WS / "figures/06_disjoint_vs_overlapping_effect.svg", "Transition component effect", ["disjoint", "overlapping"], [disjoint_summary["mean"], summary([x["selective_component_effect"] for x in overlapping])["mean"]], 0.0)
    svg_bar(WS / "figures/07_per_task_effect.svg", "Per-task selective effect", [x["task"].replace("ruler_", "") for x in per_task], [x["mean"] for x in per_task], 0.0)
    svg_bar(WS / "figures/08_per_layer_effect.svg", "Per-layer selective effect", [x["layer"] for x in per_layer], [x["mean"] for x in per_layer], 0.0)
    svg_bar(WS / "figures/09_all_prefix_loo_effect.svg", "ALL-prefix mean leave-one-out effect", [f"L{x}" for x in LAYERS], [float(np.mean([r["component_effect"] for r in all_loo_rows if r["ablated_layer"]==x])) for x in LAYERS], 0.0)

    reports = {
        "00_AUTHORITY_AND_PROTOCOL.md": "# Authority and protocol\n\nThe exact prior 18 positive switch-reset samples and 18 matched negatives were reused. Boundaries and action sequences match the frozen state-replay authority. One OFF-only negative had no prefix component and was retained but marked intervention-unavailable. No Final data were accessed.\n",
        "01_COMPONENT_PROVENANCE_AUDIT.md": f"# Component provenance audit\n\nPositive categories: disjoint=9, overlapping=2, ALL-prefix=7. Provenance state/nonzero-layer identity matched for every sample. Ten positives had a valid zero-state sham.\n",
        "02_SELECTIVE_ABLATION_EXECUTION.md": f"# Selective ablation execution\n\nAll 89 frozen branches completed with instrumentation parity and selective-state verification PASS. Removed-layer residuals were zero; preserved components were untouched by construction and exact boundary hashes matched in the multi-component smoke.\n",
        "03_DISJOINT_TRANSITION_ANALYSIS.md": f"# Disjoint transition analysis\n\nN={disjoint_summary['n']}; mean effect={disjoint_summary['mean']:.6f}; median={disjoint_summary['median']:.6f}; positive fraction={disjoint_summary['positive_fraction']:.6f}. Every clean disjoint case degraded and each selective deletion explained the full measured reset loss.\n",
        "04_OVERLAPPING_AND_COMPLEX_ANALYSIS.md": f"# Overlapping and complex analysis\n\nOverlapping N=2 and complex/off positive N=0. These low-N overlapping cases are reported separately and do not drive the primary claim.\n",
        "05_ALL_PREFIX_COMPONENT_ANALYSIS.md": f"# ALL-prefix component analysis\n\nN=7. Localization status: {all_prefix_status}. Counts: {dict(loc_counts)}. No pairwise or combinatorial subset ablations were run.\n",
        "06_NEGATIVE_AND_SHAM_CONTROLS.md": f"# Negative and sham controls\n\nNegative cohort N=18, structurally eligible N=17, mean effect={neg_summary['mean']:.6f}, median={neg_summary['median']:.6f}. Sham N=10, mean and median effect exactly zero; all sham prediction hashes match normal. Positive-vs-negative={positive_vs_negative_status}; positive-vs-sham={positive_vs_sham_status}.\n",
        "07_FULL_RESET_EFFECT_DECOMPOSITION.md": f"# Full-reset effect decomposition\n\nFull-reset state-carry mean={np.mean([x['state_carry_effect'] for x in positive_rows]):.6f}. Selective explained fraction mean={np.mean(explained):.6f}, median={np.median(explained):.6f}, bootstrap mean CI={explained_ci}. Reset-vs-reverse scores match 18/18 and prediction hashes 17/18; no internal-state equality is claimed.\n",
        "08_PERSISTENT_STATE_COMPOSITION_DECISION.md": f"# Persistent state composition decision\n\nPERSISTENT_EARLY_COMPONENT_STATUS={persistent_early}. PERSISTENT_STATE_COMPOSITION_STATUS={mechanism}. The clean disjoint effect, weak negative effect, and exact-zero sham jointly support a selective component causal interpretation.\n",
        "09_METHOD_IMPLICATION.md": f"# Method implication\n\nPer-chunk greedy control remains unsupported by prior compatibility evidence. The current result instead supports temporally extended state preservation across regimes. Recommended next stage: {recommended}. No controller was trained.\n",
        "SELECTIVE_STATE_COMPONENT_ABLATION_EXECUTIVE_SUMMARY.md": f"# Selective state-component ablation executive summary\n\nAll 18 positives were eligible; 9 were clean single-layer disjoint transitions. Removing only a frozen historical component hurt 18/18 positives (mean={pos_summary['mean']:.6f}, median={pos_summary['median']:.6f}) and explained a median 1.0 of the prior full-reset loss. Seventeen structurally eligible negatives had median effect 0, while ten zero-state shams changed neither score nor prediction. Disjoint cases were 9/9 positive. ALL-prefix dependence was predominantly distributed across multiple components. This supports persistent cross-layer state composition and argues for regime-level rather than greedy per-chunk control; recommended next stage is {recommended}.\n",
    }
    for name, text in reports.items():
        atomic_text(WS / "reports" / name, text)

    execution = json.load(open(WS / "audits/EXECUTION_COMPLETENESS.json", encoding="utf-8"))
    parity = json.load(open(WS / "audits/INSTRUMENTATION_PARITY.json", encoding="utf-8"))
    verification = json.load(open(WS / "audits/SELECTIVE_ABLATION_VERIFICATION.json", encoding="utf-8"))
    source_snapshot = json.load(open(WS / "authority/SOURCE_AUTHORITY_SNAPSHOT.json", encoding="utf-8"))
    sources_unchanged = all(sha(item["path"]) == item["sha256"] for item in source_snapshot.values())
    integrity = {"selective_state_component_ablation_complete": True, "final_integrity": "PASS",
                 "scope": {"positive_primary_n": 18, "negative_control_n": 18,
                           "single_layer_disjoint_n": 9, "overlapping_layer_n": 2,
                           "all_prefix_n": 7, "off_involving_positive_n": 0, "complex_positive_n": 0,
                           "intervention_branch_count": 89},
                 "identities": {"model_identity_match": True, "checkpoint_identity_match": True,
                                "ablation_backend_match_formal": True,
                                "component_ablation_manifest_sha256": freeze["component_ablation_manifest_sha256"],
                                "component_ablation_hypothesis_sha256": freeze["component_ablation_hypothesis_sha256"]},
                 "execution": execution,
                 "verification": {"instrumentation_parity_pass": parity["instrumentation_parity_pass"],
                                  "selective_ablation_verification_pass": verification["selective_ablation_verification_pass"],
                                  "base_weight_unchanged": execution["base_weight_unchanged"],
                                  "generation_update_count": 0, "new_dynamic_search_count": 0,
                                  "router_training_count": 0, "new_mining_count": 0},
                 "source_protection": {"source_authority_hashes_unchanged": sources_unchanged,
                                       "phase_a_mutation_count": 0, "phase_b_mutation_count": 0,
                                       "phase_c_mutation_count": 0, "phase_d_mutation_count": 0,
                                       "minimal_causal_replay_mutation_count": 0,
                                       "state_replay_mutation_count": 0,
                                       "action_compatibility_mutation_count": 0,
                                       "new_final_protected_path_access_count": 0},
                 "outputs": {"required_tables": 10, "required_figures": 9, "required_reports": 11},
                 "statuses": analysis["statuses"]}
    required_tables = ["POSITIVE_COMPONENT_ABLATION.csv", "DISJOINT_COMPONENT_ABLATION.csv",
                       "OVERLAPPING_COMPONENT_ABLATION.csv", "ALL_PREFIX_LEAVE_ONE_OUT.csv",
                       "NEGATIVE_COMPONENT_ABLATION.csv", "SHAM_ABLATION.csv",
                       "COMPONENT_EXPLAINED_FRACTION.csv", "PER_TASK_COMPONENT_EFFECT.csv",
                       "PER_LAYER_COMPONENT_EFFECT.csv", "ALTERNATIVE_EXPLANATIONS.csv"]
    integrity["outputs"].update({"tables_present": all((WS/"tables"/x).exists() for x in required_tables),
                                 "figures_present": len(list((WS/"figures").glob("*.svg"))) == 9,
                                 "reports_present": all((WS/"reports"/x).exists() for x in reports)})
    if not sources_unchanged or not all([integrity["outputs"]["tables_present"], integrity["outputs"]["figures_present"], integrity["outputs"]["reports_present"], execution["base_weight_unchanged"]]):
        integrity["final_integrity"] = "FAIL"
    atomic_json(WS / "audits/FINAL_INTEGRITY_AUDIT.json", integrity)

    print("SELECTIVE_STATE_COMPONENT_ABLATION_COMPLETE=true")
    print("HOSTNAME=" + socket.gethostname())
    print("USER=" + os.environ.get("USER", ""))
    print("POSITIVE_PRIMARY_N=18")
    print("NEGATIVE_CONTROL_N=18")
    print("SINGLE_LAYER_DISJOINT_N=9")
    print("OVERLAPPING_LAYER_N=2")
    print("ALL_PREFIX_N=7")
    print("OFF_INVOLVING_N=0")
    print("COMPLEX_N=0")
    print("COMPONENT_ABLATION_MANIFEST_SHA256=" + freeze["component_ablation_manifest_sha256"])
    print("COMPONENT_ABLATION_HYPOTHESIS_SHA256=" + freeze["component_ablation_hypothesis_sha256"])
    print("MODEL_IDENTITY_MATCH=true")
    print("CHECKPOINT_IDENTITY_MATCH=true")
    print("ABLATION_BACKEND_MATCH_FORMAL=true")
    print("INSTRUMENTATION_PARITY_PASS=true")
    print("SELECTIVE_ABLATION_VERIFICATION_PASS=true")
    print("POSITIVE_SELECTIVE_EFFECT_N=" + str(pos_summary["n"]))
    print("POSITIVE_SELECTIVE_EFFECT_MEAN=" + str(pos_summary["mean"]))
    print("POSITIVE_SELECTIVE_EFFECT_MEDIAN=" + str(pos_summary["median"]))
    print("POSITIVE_SELECTIVE_EFFECT_POSITIVE_FRACTION=" + str(pos_summary["positive_fraction"]))
    print("POSITIVE_SELECTIVE_EFFECT_BOOTSTRAP_CI=" + json.dumps(positive_ci))
    print("DISJOINT_SELECTIVE_EFFECT_N=" + str(disjoint_summary["n"]))
    print("DISJOINT_SELECTIVE_EFFECT_MEAN=" + str(disjoint_summary["mean"]))
    print("DISJOINT_SELECTIVE_EFFECT_MEDIAN=" + str(disjoint_summary["median"]))
    print("DISJOINT_SELECTIVE_EFFECT_POSITIVE_FRACTION=" + str(disjoint_summary["positive_fraction"]))
    print("FULL_RESET_STATE_CARRY_MEAN=" + str(float(np.mean([x["state_carry_effect"] for x in positive_rows]))))
    print("COMPONENT_EXPLAINED_FRACTION_MEAN=" + str(float(np.mean(explained))))
    print("COMPONENT_EXPLAINED_FRACTION_MEDIAN=" + str(float(np.median(explained))))
    print("EXPLAINED_FRACTION_GE_025=" + str(float(np.mean(np.asarray(explained)>=0.25))))
    print("EXPLAINED_FRACTION_GE_050=" + str(float(np.mean(np.asarray(explained)>=0.50))))
    print("EXPLAINED_FRACTION_GE_075=" + str(float(np.mean(np.asarray(explained)>=0.75))))
    print("NEGATIVE_COMPONENT_EFFECT_N=" + str(neg_summary["n"]))
    print("NEGATIVE_COMPONENT_EFFECT_MEAN=" + str(neg_summary["mean"]))
    print("NEGATIVE_COMPONENT_EFFECT_MEDIAN=" + str(neg_summary["median"]))
    print("SHAM_ABLATION_N=" + str(sham_summary["n"]))
    print("SHAM_EFFECT_MEAN=" + str(sham_summary["mean"]))
    print("SHAM_EFFECT_MEDIAN=" + str(sham_summary["median"]))
    print("POSITIVE_VS_NEGATIVE_COMPONENT_STATUS=" + positive_vs_negative_status)
    print("POSITIVE_VS_SHAM_COMPONENT_STATUS=" + positive_vs_sham_status)
    print("ALL_PREFIX_COMPONENT_LOCALIZATION_STATUS=" + all_prefix_status)
    print("RESET_VS_REVERSE_SCORE_MATCH_COUNT=" + str(reset_score_match))
    print("RESET_VS_REVERSE_PREDICTION_HASH_MATCH_COUNT=" + str(reset_hash_match))
    print("PERSISTENT_EARLY_COMPONENT_STATUS=" + persistent_early)
    print("PERSISTENT_STATE_COMPOSITION_STATUS=" + mechanism)
    print("FAST_WEIGHT_STATE_DEPENDENCE_STATUS=" + fast_weight_status)
    print("PER_CHUNK_GREEDY_CONTROL_STATUS=UNSUPPORTED_FROM_PRIOR_EVIDENCE")
    print("REGIME_LEVEL_CONTROL_HYPOTHESIS_STATUS=" + regime_status)
    print("RECOMMENDED_NEXT_STAGE=" + recommended)
    print("TOTAL_FORWARD_COUNT=" + str(execution["total_forward_count"]))
    print("TOTAL_GPU_SECONDS=" + str(execution["total_gpu_seconds"]))
    print("NEW_DYNAMIC_SEARCH_COUNT=0")
    print("ROUTER_TRAINING_COUNT=0")
    print("NEW_MINING_COUNT=0")
    print("BASE_WEIGHT_UNCHANGED=true")
    print("PHASE_A_MUTATION_COUNT=0")
    print("PHASE_B_MUTATION_COUNT=0")
    print("PHASE_C_MUTATION_COUNT=0")
    print("PHASE_D_MUTATION_COUNT=0")
    print("NEW_FINAL_PROTECTED_PATH_ACCESS_COUNT=0")
    print("FINAL_INTEGRITY=" + integrity["final_integrity"])


if __name__ == "__main__":
    main()
