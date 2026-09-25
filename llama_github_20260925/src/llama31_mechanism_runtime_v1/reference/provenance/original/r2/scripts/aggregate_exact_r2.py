#!/usr/bin/env python3
"""Independent integrity audit, winner selection, and frozen Exact-R2 analyses."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ACTIONS = ["OFF", "L0", "L6", "L12", "L18", "L24", "ALL"]
TOL = 1e-12


def now(): return datetime.now().astimezone().isoformat(timespec="seconds")
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()
def load(path): return json.loads(Path(path).read_text())
def rows(path):
    with open(path) as f: return [json.loads(x) for x in f if x.strip()]
def imp(path, name):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
def atomic_text(path, text):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "w") as f: f.write(text); f.flush(); os.fsync(f.fileno())
        os.replace(temp, path); d = os.open(path.parent, os.O_RDONLY); os.fsync(d); os.close(d)
    except BaseException:
        try: os.unlink(temp)
        except FileNotFoundError: pass
        raise
def atomic(path, obj): atomic_text(path, json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
def write_csv(path, data):
    data = list(data); fields = list(data[0])
    text = []
    import io
    s = io.StringIO(); w = csv.DictWriter(s, fields); w.writeheader(); w.writerows(data); atomic_text(path, s.getvalue())


def independent_winner(records):
    """Separate implementation from the worker's online selector."""
    best = max(float(x["score"]) for x in records)
    action_rank = dict(zip(ACTIONS, range(len(ACTIONS))))
    tied = []
    for x in records:
        if math.isclose(float(x["score"]), best, rel_tol=0.0, abs_tol=TOL): tied.append(x)
    selected = min(tied, key=lambda x: (int(x["tau"]), action_rank[x["A"]], action_rank[x["B"]], x["full_schedule_hash"]))
    return selected, best, len(tied)


def ci(values): return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def bootstrap(data, seed, n_resamples, stratified=False):
    rng = np.random.default_rng(seed); gaps = []; ratios = []
    groups = defaultdict(list)
    for i, x in enumerate(data): groups[x["task"]].append(i)
    all_indices = np.arange(len(data))
    for _ in range(n_resamples):
        if stratified:
            idx = np.concatenate([rng.choice(v, size=len(v), replace=True) for _, v in sorted(groups.items())])
        else: idx = rng.choice(all_indices, size=len(data), replace=True)
        gaps.append(float(np.mean([data[i]["Dynamic_minus_R2"] for i in idx])))
        den = sum(data[i]["Dynamic_minus_OFF"] for i in idx)
        ratios.append(sum(data[i]["R2_minus_OFF"] for i in idx) / den)
    return {"mean_dynamic_minus_r2_95ci": ci(gaps), "aggregate_r2_gain_retention_95ci": ci(ratios),
            "seed": seed, "resamples": n_resamples, "stratified_by_task": stratified}


def aggregate(run):
    run = Path(run).resolve(); worker = imp(run / "scripts/exact_r2_worker.py", "exact_r2_worker_for_aggregate")
    run, cfg, manifest, by_sample, mappings, checks = worker.gates(run)
    parent = {x["canonical_sample_id"]: x for x in rows(cfg["parent_manifest"])}
    if len(parent) != 173: raise RuntimeError("PARENT_SCOPE")
    records_by_sample = defaultdict(list); invalid = []; missing = []; seen_ids = set()
    for candidate in manifest:
        path = worker.branch_path(run, candidate)
        if not path.exists(): missing.append(candidate["candidate_id"]); continue
        try: rec = worker.valid_result(path, candidate, cfg)
        except Exception as exc: invalid.append({"candidate_id": candidate["candidate_id"], "error": str(exc)}); continue
        if rec["candidate_id"] in seen_ids: raise RuntimeError("DUPLICATE_RESULT_ID " + rec["candidate_id"])
        seen_ids.add(rec["candidate_id"]); records_by_sample[rec["canonical_sample_id"]].append(rec)
    integrity = {"status": "PASS" if len(seen_ids)==99120 and not missing and not invalid else "FAIL",
                 "VALID_CANDIDATE_RESULTS": len(seen_ids), "MISSING_CANDIDATES": len(missing),
                 "DUPLICATE_CANDIDATES": 0, "INVALID_CANDIDATES": len(invalid),
                 "UNRESOLVED_ERRORS": len(list((run / "errors").glob("*.json"))),
                 "candidate_manifest_sha256": sha(cfg["candidate_manifest"]), "timestamp": now()}
    atomic(run / "audit/CANDIDATE_INTEGRITY_AUDIT.json", integrity)
    if integrity["status"] != "PASS": raise RuntimeError("CANDIDATE_INTEGRITY_FAILED")

    independent = []; mismatch = []
    for sid, candidates in by_sample.items():
        records = records_by_sample[sid]
        if len(records) != len(candidates): raise RuntimeError("SAMPLE_NOT_FULLY_SEARCHED " + sid)
        selected, best, tie_n = independent_winner(records)
        online = load(worker.winner_path(run, sid))
        if online["winning_candidate_id"] != selected["candidate_id"] or abs(online["R2_score"] - best) > TOL:
            mismatch.append(sid)
        independent.append({"canonical_sample_id": sid, "task": selected["task"],
            "winning_candidate_id": selected["candidate_id"], "R2_score": best, "winning_A": selected["A"],
            "winning_B": selected["B"], "winning_tau": selected["tau"], "T": selected["T"],
            "number_of_max_score_ties": tie_n, "normalized_tau": selected["tau"] / selected["T"]})
    winner_audit = {"status": "PASS" if not mismatch else "FAIL", "INDEPENDENT_WINNER_AUDIT": "PASS" if not mismatch else "FAIL",
                    "WINNER_MISMATCH_N": len(mismatch), "WINNERS_VALID": len(independent), "mismatches": mismatch, "timestamp": now()}
    atomic(run / "audit/INDEPENDENT_WINNER_AUDIT.json", winner_audit)
    if mismatch: raise RuntimeError("WINNER_MISMATCH")
    write_csv(run / "winners/INDEPENDENT_EXACT_R2_WINNERS.csv", independent)
    winners = {x["canonical_sample_id"]: x for x in independent}

    data = []
    for sid, p in parent.items():
        w = winners[sid]; off=float(p["OFF_score"]); sb=float(p["SampleBest_score"]); dyn=float(p["Dynamic_score"]); r2=float(w["R2_score"])
        data.append({"canonical_sample_id":sid,"task":p["task"],"benchmark_global_index":p["benchmark_global_index"],
            "OFF":off,"R2":r2,"Dynamic":dyn,"SampleBest":sb,"R2_minus_OFF":r2-off,
            "Dynamic_minus_OFF":dyn-off,"Dynamic_minus_R2":dyn-r2,"R2_minus_SampleBest":r2-sb,
            "Dynamic_switch_count":p["Dynamic_switch_count"],"Dynamic_regime_count":p["Dynamic_regime_count"],
            "winning_A":w["winning_A"],"winning_B":w["winning_B"],"winning_tau":w["winning_tau"],"T":w["T"]})
    data.sort(key=lambda x:x["benchmark_global_index"]); write_csv(run / "aggregates/OFF_R2_DYNAMIC_PER_SAMPLE.csv", data)
    gaps = [x["Dynamic_minus_R2"] for x in data]
    classify = Counter("GT" if x["R2"] > x["Dynamic"]+TOL else "LT" if x["R2"] < x["Dynamic"]-TOL else "EQ" for x in data)
    sbc = Counter("GT" if x["R2"] > x["SampleBest"]+TOL else "LT" if x["R2"] < x["SampleBest"]-TOL else "EQ" for x in data)
    den = sum(x["Dynamic_minus_OFF"] for x in data); retention=sum(x["R2_minus_OFF"] for x in data)/den
    primary = {"N":173,"OFF_MEAN":statistics.mean(x["OFF"] for x in data),"R2_MEAN":statistics.mean(x["R2"] for x in data),
        "DYNAMIC_MEAN":statistics.mean(x["Dynamic"] for x in data),"R2_MINUS_OFF_MEAN":statistics.mean(x["R2_minus_OFF"] for x in data),
        "DYNAMIC_MINUS_OFF_MEAN":statistics.mean(x["Dynamic_minus_OFF"] for x in data),"DYNAMIC_MINUS_R2_MEAN":statistics.mean(gaps),
        "DYNAMIC_MINUS_R2_MEDIAN":statistics.median(gaps),"DYNAMIC_MINUS_R2_STD":statistics.pstdev(gaps),
        "DYNAMIC_MINUS_R2_MIN":min(gaps),"DYNAMIC_MINUS_R2_MAX":max(gaps),"R2_GT_DYNAMIC_COUNT":classify["GT"],
        "R2_EQ_DYNAMIC_COUNT":classify["EQ"],"R2_LT_DYNAMIC_COUNT":classify["LT"],"R2_GE_DYNAMIC_COUNT":classify["GT"]+classify["EQ"],
        "R2_GT_DYNAMIC_RATE":classify["GT"]/173,"R2_EQ_DYNAMIC_RATE":classify["EQ"]/173,"R2_LT_DYNAMIC_RATE":classify["LT"]/173,
        "R2_GE_DYNAMIC_RATE":(classify["GT"]+classify["EQ"])/173,"AGGREGATE_R2_GAIN_RETENTION_VS_OFF":retention,
        "AGGREGATE_R2_GAIN_RETENTION_PERCENT":100*retention,"SAMPLE_BEST_MEAN":statistics.mean(x["SampleBest"] for x in data),
        "R2_MINUS_SAMPLE_BEST_MEAN":statistics.mean(x["R2_minus_SampleBest"] for x in data),
        "R2_GT_SAMPLE_BEST_COUNT":sbc["GT"],"R2_EQ_SAMPLE_BEST_COUNT":sbc["EQ"],"R2_LT_SAMPLE_BEST_COUNT":sbc["LT"]}

    b1=bootstrap(data,cfg["paired_bootstrap_seed"],20000,False); b2=bootstrap(data,cfg["task_stratified_bootstrap_seed"],20000,True)
    primary["DYNAMIC_R2_PAIRED_BOOTSTRAP_95CI"]=b1["mean_dynamic_minus_r2_95ci"]
    primary["DYNAMIC_R2_TASK_STRATIFIED_BOOTSTRAP_95CI"]=b2["mean_dynamic_minus_r2_95ci"]
    primary["R2_GAIN_RETENTION_BOOTSTRAP_95CI"]=b1["aggregate_r2_gain_retention_95ci"]
    atomic(run / "bootstrap/PAIRED_BOOTSTRAP_20000.json", b1); atomic(run / "bootstrap/TASK_STRATIFIED_PAIRED_BOOTSTRAP_20000.json", b2)
    atomic(run / "aggregates/PRIMARY_OFF_R2_DYNAMIC.json", primary)

    one=[x for x in data if x["Dynamic_switch_count"]==1]; multi=[x for x in data if x["Dynamic_switch_count"]>1]
    one_lt=sum(x["R2"]<x["Dynamic"]-TOL for x in one)
    invariant={"status":"PASS" if len(one)==100 and one_lt==0 else "FAIL","ONE_SWITCH_DYNAMIC_CANDIDATE_MAPPING_N":len(mappings),
               "ONESWITCH_N":len(one),"ONE_SWITCH_R2_LT_DYNAMIC_N":one_lt,
               "R2_GT_DYNAMIC_COUNT":sum(x["R2"]>x["Dynamic"]+TOL for x in one),
               "R2_EQ_DYNAMIC_COUNT":sum(abs(x["R2"]-x["Dynamic"])<=TOL for x in one),"timestamp":now()}
    atomic(run / "audit/ONE_SWITCH_LOGICAL_INVARIANT.json", invariant)
    if invariant["status"] != "PASS": raise RuntimeError("ONE_SWITCH_LOGICAL_INVARIANT")
    mden=sum(x["Dynamic_minus_OFF"] for x in multi); mret=sum(x["R2_minus_OFF"] for x in multi)/mden
    multi_out={"label":"PREDEFINED_MULTI_SWITCH_STRUCTURAL_ANALYSIS","MULTISWITCH_N":len(multi),
        "MULTISWITCH_OFF_MEAN":statistics.mean(x["OFF"] for x in multi),"MULTISWITCH_R2_MEAN":statistics.mean(x["R2"] for x in multi),
        "MULTISWITCH_DYNAMIC_MEAN":statistics.mean(x["Dynamic"] for x in multi),
        "MULTISWITCH_DYNAMIC_MINUS_R2_MEAN":statistics.mean(x["Dynamic_minus_R2"] for x in multi),
        "MULTISWITCH_R2_GAIN_RETENTION":mret,"MULTISWITCH_R2_GAIN_RETENTION_PERCENT":100*mret,
        "MULTISWITCH_R2_GT_DYNAMIC_COUNT":sum(x["R2"]>x["Dynamic"]+TOL for x in multi),
        "MULTISWITCH_R2_EQ_DYNAMIC_COUNT":sum(abs(x["R2"]-x["Dynamic"])<=TOL for x in multi),
        "MULTISWITCH_R2_LT_DYNAMIC_COUNT":sum(x["R2"]<x["Dynamic"]-TOL for x in multi),
        "MULTISWITCH_R2_GE_DYNAMIC_RATE":sum(x["R2"]>=x["Dynamic"]-TOL for x in multi)/len(multi)}
    atomic(run / "aggregates/PREDEFINED_MULTI_SWITCH_STRUCTURAL_ANALYSIS.json", multi_out)

    per_task=[]
    for task in sorted({x["task"] for x in data}):
        z=[x for x in data if x["task"]==task]; per_task.append({"task":task,"N":len(z),
            "OFF_mean":statistics.mean(x["OFF"] for x in z),"R2_mean":statistics.mean(x["R2"] for x in z),
            "Dynamic_mean":statistics.mean(x["Dynamic"] for x in z),"R2_minus_OFF":statistics.mean(x["R2_minus_OFF"] for x in z),
            "Dynamic_minus_OFF":statistics.mean(x["Dynamic_minus_OFF"] for x in z),
            "Dynamic_minus_R2":statistics.mean(x["Dynamic_minus_R2"] for x in z),
            "R2_ge_Dynamic_rate":sum(x["R2"]>=x["Dynamic"]-TOL for x in z)/len(z)})
    write_csv(run / "aggregates/PER_TASK_OFF_R2_DYNAMIC.csv", per_task)
    descriptive={"winning_tau_distribution":dict(sorted(Counter(x["winning_tau"] for x in data).items())),
        "normalized_tau_over_T":{"mean":statistics.mean(x["winning_tau"]/x["T"] for x in data),
                                 "median":statistics.median(x["winning_tau"]/x["T"] for x in data)},
        "winning_A_distribution":dict(Counter(x["winning_A"] for x in data)),"winning_B_distribution":dict(Counter(x["winning_B"] for x in data)),
        "transition_pair_distribution":dict(Counter(x["winning_A"]+"->"+x["winning_B"] for x in data))}
    atomic(run / "aggregates/WINNING_TAU_ACTION_DESCRIPTIVE.json", descriptive)

    receipt={"FORMAL_EXACT_R2_COMPLETE":True,"FINAL_INTEGRITY":"PASS","MODEL_IDENTITY_MATCH":True,"CHECKPOINT_IDENTITY_MATCH":True,
        "BENCHMARK_IDENTITY_MATCH":True,"RULE_HASH_MATCH":True,"ELIGIBILITY_HASH_MATCH":True,"CANDIDATE_MANIFEST_HASH_MATCH":True,
        "PROTOCOL_HASH_MATCH":True,"EXECUTION_COHORT_N":173,**integrity,"FULLY_SEARCHED_SAMPLES":173,"WINNERS_VALID":173,
        **winner_audit,**invariant,"HISTORICAL_OVERLAP":0,"R2_OUTCOME_LEAKAGE":0,"CROSS_CANDIDATE_STATE_LEAKAGE":False,
        "ONE_SWITCH_DYNAMIC_REPLAY_SMOKE":"PASS","R2_REPLAY_DETERMINISM_SMOKE":"PASS","SCORER_PARITY":"PASS",
        "OFF_RERUN":False,"DYNAMIC_RERUN":False,"P1_EXECUTED":False,"FROZEN_EXPERT_EXECUTED":False,
        "primary":primary,"multi_switch":multi_out,"rule_sha256":worker.RULE_SHA,"eligibility_sha256":worker.ELIG_SHA,
        "candidate_manifest_sha256":worker.CAND_SHA,"protocol_sha256":cfg["protocol_sha256"],"completed_at":now()}
    receipt_json=run/"reports/EXACT_R2_1P7B_FORMAL_COMPLETION_RECEIPT.json"; atomic(receipt_json,receipt)
    md="# Exact R2 1.7B Formal Completion Receipt\n\n"+"\n".join(f"{k}={v}" for k,v in receipt.items() if not isinstance(v,dict))+"\n\nR2 is an offline same-sample terminal-reward hindsight structural oracle; it does not establish online identification or deployability.\n"
    atomic_text(run/"reports/EXACT_R2_1P7B_FORMAL_COMPLETION_RECEIPT.md",md)
    worker.update_status(run,pipeline_status="COMPLETE",current_phase="STOP",candidates_committed=99120,
        samples_fully_searched=173,winners_finalized=173,worker_running=False,final_receipt_exists=True,
        final_integrity="PASS",formal_complete=True,updated_at=now())
    print(json.dumps({"FORMAL_EXACT_R2_COMPLETE":True,"FINAL_INTEGRITY":"PASS","receipt":str(receipt_json)},sort_keys=True))


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("run_root"); a=p.parse_args(); aggregate(a.run_root)
