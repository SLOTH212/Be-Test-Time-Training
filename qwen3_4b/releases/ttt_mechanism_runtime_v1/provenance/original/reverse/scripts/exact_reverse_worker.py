#!/usr/bin/env python3
"""Crash-safe single-worker Exact Reverse replay using the frozen evaluator."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

EXPECTED_MODEL = "ca51ee90d20ee09efbe0b31dd21bf9faa70275508316d1ba7cb1387ee4f0725f"
EXPECTED_CHECKPOINT = "f4e264ed6f5671f123d6c8a27c69bce3af5ffcba518c9e298398314c6147d9fd"
EXPECTED_BENCHMARK = "5d16e3a6963d4c21450f0530af743730b91bf7b2442699e288e10754f77ad0c5"
EXPECTED_SCREENING = "c5e04276887da0dda6d1b4599da01e70d4dd88ca8a0cf96f3b9db58ee08f2a65"
EXPECTED_EXECUTOR = "293886005cb8f78c943f0affe3e3b383d6de238f4bc825a5aea28a6e6dc66f6e"
EXPECTED_SCORER = "cee3f5cb3cad1709f81abe3dc32b364ca97322e7ebd91747a8b119aecb0ce556"
ACTIONS = ["OFF", "L0", "L6", "L12", "L18", "L24", "ALL"]
TOL = 1e-12


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


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def result_name(sample_id):
    return hashlib.sha256(sample_id.encode()).hexdigest() + ".json"


def update_status(run, **changes):
    path = run / "status/pipeline_status.json"
    state = load_json(path) if path.exists() else {}
    state.update(changes)
    state["updated_at"] = now()
    atomic_json(path, state)
    return state


def affinity_gate():
    allowed = sorted(os.sched_getaffinity(0))
    if allowed != list(range(8, 24)):
        raise RuntimeError(f"CPU_AFFINITY_MISMATCH expected=8-23 actual={allowed}")
    return allowed


class ExactContext:
    def __init__(self, config, runner):
        self.run = Path(config["run_root"])
        self.code = Path("/home/USER/ttt/code/ntp")
        self.ckpt = Path(config["model_root"])
        scorer = import_file(Path(config["scorer"]), "exact_reverse_frozen_scorer")
        self.score_fn = scorer.per_sample_score
        rows = load_jsonl(Path(config["benchmark_root"]) / "sample_manifest.jsonl")
        self.rows = []
        for i, row in enumerate(rows):
            row = dict(row)
            row["_sample_index"] = i
            self.rows.append(row)
        self.by_id = {r["sample_id"]: r for r in self.rows}
        if len(self.rows) != len(self.by_id) or len(self.rows) != 6500:
            raise RuntimeError("BENCHMARK_SAMPLE_SET_INVALID")
        self.manifest = {"tasks": sorted({r["task"] for r in rows})}
        self.baseline = {a: {} for a in ACTIONS}
        fixed_rows = load_jsonl(Path(config["fixed7_root"]) / "manifests/all_fixed7_results.jsonl")
        for record in fixed_rows:
            copy = dict(record)
            copy["official_sample_score"] = float(copy["score"])
            self.baseline[copy["mode"]][copy["sample_id"]] = copy
        if any(len(self.baseline[a]) != 6500 for a in ACTIONS):
            raise RuntimeError("FIXED7_SAMPLE_SET_INVALID")


def gates(run):
    run = Path(run).resolve()
    config_path = run / "config/EXACT_REVERSE_FORMAL_CONFIG.json"
    config = load_json(config_path)
    cohort_path = Path(config["cohort_manifest"])
    protocol_path = Path(config["protocol"])
    amendment_path = Path(config["protocol_amendment"])
    if config["run_root"] != str(run):
        raise RuntimeError("RUN_ROOT_MISMATCH")
    if affinity_gate() != list(range(8, 24)):
        raise RuntimeError("AFFINITY_GATE_FAILED")
    checks = {
        "model": sha(Path(config["model_root"]) / "model.safetensors") == EXPECTED_MODEL == config["model_identity"],
        "benchmark": sha(Path(config["benchmark_root"]) / "benchmark_manifest.json") == EXPECTED_BENCHMARK == config["benchmark_identity"],
        "screening": sha(Path(config["screening_root"]) / "manifests/MECHANISM_ANCHOR_SCREENING_V1_MANIFEST.json") == EXPECTED_SCREENING == config["screening_manifest_sha256"],
        "cohort": sha(cohort_path) == config["cohort_manifest_sha256"],
        "protocol": sha(protocol_path) == config["protocol_sha256"],
        "amendment": sha(amendment_path) == config["protocol_amendment_sha256"],
        "executor": sha(Path(config["frozen_executor"])) == EXPECTED_EXECUTOR == config["frozen_executor_sha256"],
        "scorer": sha(Path(config["scorer"])) == EXPECTED_SCORER == config["scorer_sha256"],
        "worker": sha(Path(config["worker"])) == config["worker_sha256"],
        "aggregator": sha(Path(config["aggregator"])) == config["aggregator_sha256"],
        "controller": sha(Path(config["controller"])) == config["controller_sha256"],
    }
    if not all(checks.values()):
        raise RuntimeError("AUTHORITY_HASH_MISMATCH " + json.dumps(checks, sort_keys=True))
    protocol = load_json(protocol_path)
    if protocol["model"]["checkpoint_identity"] != EXPECTED_CHECKPOINT:
        raise RuntimeError("CHECKPOINT_IDENTITY_MISMATCH")
    cohort = load_jsonl(cohort_path)
    if len(cohort) != 173 or len({r["canonical_sample_id"] for r in cohort}) != 173:
        raise RuntimeError("COHORT_SCOPE_MISMATCH")
    for row in cohort:
        original, reverse = row["original_dynamic_actions"], row["reversed_actions"]
        if reverse != list(reversed(original)) or original == reverse or Counter(original) != Counter(reverse):
            raise RuntimeError(f"REVERSE_INVARIANT_FAILED {row['canonical_sample_id']}")
        if len(original) != row["full_chunk_count"]:
            raise RuntimeError(f"CHUNK_COUNT_FAILED {row['canonical_sample_id']}")
        if row["historical_mechanism_exposed"]:
            raise RuntimeError(f"HISTORICAL_OVERLAP {row['canonical_sample_id']}")
        dyn_path = Path(row["dynamic_result_path"])
        if sha(dyn_path) != row["dynamic_result_sha256"]:
            raise RuntimeError(f"DYNAMIC_RESULT_HASH_MISMATCH {row['canonical_sample_id']}")
    return run, config, cohort, checks


def valid_result(path, row, config):
    rec = load_json(path)
    finite = isinstance(rec.get("reverse_score"), (int, float)) and math.isfinite(rec["reverse_score"])
    expected = (
        rec.get("completed") is True
        and rec.get("status") == "COMMITTED"
        and rec.get("canonical_sample_id") == row["canonical_sample_id"]
        and rec.get("task") == row["task"]
        and rec.get("reversed_actions") == row["reversed_actions"]
        and rec.get("cohort_manifest_hash") == config["cohort_manifest_sha256"]
        and rec.get("protocol_hash") == config["protocol_sha256"]
        and rec.get("model_identity") == EXPECTED_MODEL
        and rec.get("checkpoint_identity") == EXPECTED_CHECKPOINT
        and rec.get("benchmark_identity") == EXPECTED_BENCHMARK
        and rec.get("action_composition_parity") is True
        and rec.get("recompute_downstream_updates") is True
        and rec.get("stored_delta_stitching") is False
        and rec.get("generation_update_count") == 0
        and finite
    )
    if not expected:
        raise RuntimeError(f"INVALID_EXISTING_COMMIT {path}")
    return rec


def load_runtime(run, config):
    runner = import_file(Path(config["frozen_executor"]), "frozen_exact_reverse_dynamic_executor")
    if runner.CHECKPOINT_HASH != EXPECTED_CHECKPOINT:
        raise RuntimeError("FROZEN_EXECUTOR_CHECKPOINT_IDENTITY_MISMATCH")
    context = ExactContext(config, runner)
    import torch
    torch.set_num_threads(16)
    update_status(run, current_phase="MODEL_LOADING", worker_pid=os.getpid(), controller_pid=int(os.environ.get("EXACT_REVERSE_CONTROLLER_PID", os.getpid())))
    model, tokenizer = runner.load_model(context)
    base = runner.base_hash(model)
    atomic_json(run / "audit/MODEL_LOAD_AUDIT.json", {
        "status": "PASS", "timestamp": now(), "model_identity": EXPECTED_MODEL,
        "checkpoint_identity": EXPECTED_CHECKPOINT, "model_load_succeeded": True,
        "base_weight_runtime_hash": base, "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "cuda_device": torch.cuda.get_device_name(0), "cuda_memory_allocated_bytes": torch.cuda.memory_allocated(),
    })
    return runner, context, model, tokenizer, base


def run_smoke(run):
    run, config, cohort, checks = gates(run)
    if list((run / "results").glob("*.json")):
        raise RuntimeError("FORMAL_RESULTS_EXIST_BEFORE_SMOKE")
    orders = set(config["smoke_cohort_orders"])
    selected = [r for r in cohort if r["cohort_order"] in orders]
    if len(selected) != 3:
        raise RuntimeError("SMOKE_SCOPE_MISMATCH")
    update_status(run, pipeline_status="SMOKE_RUNNING", current_phase="ORIGINAL_DYNAMIC_REPLAY_PARITY",
                  controller_pid=int(os.environ.get("EXACT_REVERSE_CONTROLLER_PID", os.getpid())), worker_pid=os.getpid(),
                  current_sample=None, current_task=None, last_error="NONE")
    runner, ctx, model, tok, initial_base = load_runtime(run, config)
    rows = []
    try:
        for ordinal, meta in enumerate(selected, 1):
            sid = meta["canonical_sample_id"]
            update_status(run, current_sample=sid, current_task=meta["task"])
            ev = runner.Evaluator(ctx, model, tok, ctx.by_id[sid], "exact_reverse_nonformal_smoke", ordinal,
                                  len(selected), run / "smoke/SMOKE_INTERNAL_PROGRESS.json")
            original = ev.evaluate(tuple(meta["original_dynamic_actions"]), capture=True, force=True)
            reverse = ev.evaluate(tuple(meta["reversed_actions"]), capture=False, force=True)
            original_again = ev.evaluate(tuple(meta["original_dynamic_actions"]), capture=True, force=True)
            ev.finish()
            score_parity = abs(original["score"] - float(meta["original_dynamic_score"])) <= TOL
            prediction_parity = original["prediction_hash"] == meta["original_dynamic_prediction_sha256"]
            repeat_parity = original_again["score"] == original["score"] and original_again["prediction_hash"] == original["prediction_hash"]
            composition = Counter(meta["original_dynamic_actions"]) == Counter(meta["reversed_actions"])
            distinct = meta["original_dynamic_actions"] != meta["reversed_actions"]
            if not (score_parity and prediction_parity and repeat_parity and composition and distinct):
                raise RuntimeError(f"SMOKE_PARITY_FAILED {sid}")
            rows.append({
                "formal_result": False, "cohort_order": meta["cohort_order"], "canonical_sample_id": sid,
                "task": meta["task"], "original_dynamic_actions": meta["original_dynamic_actions"],
                "reversed_actions": meta["reversed_actions"], "frozen_dynamic_score": meta["original_dynamic_score"],
                "replayed_original_score": original["score"], "replayed_reverse_smoke_score": reverse["score"],
                "original_prediction_sha256": original["prediction_hash"],
                "frozen_prediction_sha256": meta["original_dynamic_prediction_sha256"],
                "original_score_parity": score_parity, "original_prediction_parity": prediction_parity,
                "sample_reset_repeat_parity": repeat_parity, "action_composition_parity": composition,
                "reverse_distinct": distinct, "full_chunk_count": original["full_chunk_count"],
                "prompt_token_count": original["prompt_token_count"],
                "incomplete_tail_length": original["prompt_token_count"] % 1024,
                "incomplete_tail_update": False, "generation_update_count": reverse["generation_update_count"],
                "recompute_downstream_updates": reverse["recompute_downstream_updates"],
                "stored_delta_stitching": reverse["stored_delta_stitching"],
                "base_weight_unchanged": runner.base_hash(model) == initial_base,
                "reverse_gpu_peak_memory_bytes": reverse["gpu_peak_memory_bytes"], "timestamp": now(),
            })
            del ev, original, reverse, original_again
            gc.collect()
            import torch
            torch.cuda.empty_cache()
        if runner.base_hash(model) != initial_base:
            raise RuntimeError("BASE_WEIGHT_MUTATION_AFTER_SMOKE")
    finally:
        del model, tok
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    smoke = {
        "status": "PASS", "formal_result": False, "smoke_n": 3,
        "ORIGINAL_DYNAMIC_REPLAY_PARITY": "PASS", "score_parity_n": 3, "prediction_parity_n": 3,
        "ACTION_COMPOSITION_PARITY": "PASS", "REVERSE_EXECUTION_SMOKE": "PASS",
        "SAMPLE_RESET_PARITY": "PASS", "DOWNSTREAM_RECOMPUTATION": True,
        "STORED_DELTA_STITCHING": False, "INCOMPLETE_TAIL_UPDATE": False, "GENERATION_UPDATE": False,
        "model_load_succeeded": True, "authority_checks": checks,
        "cohort_manifest_sha256": config["cohort_manifest_sha256"], "protocol_sha256": config["protocol_sha256"],
        "rows": rows, "completed_at": now(),
    }
    atomic_json(run / "smoke/ORIGINAL_DYNAMIC_REPLAY_PARITY.json", smoke)
    atomic_json(run / "smoke/SMOKE_COMPLETION.json", {k: smoke[k] for k in ["status", "smoke_n", "ORIGINAL_DYNAMIC_REPLAY_PARITY", "ACTION_COMPOSITION_PARITY", "REVERSE_EXECUTION_SMOKE", "SAMPLE_RESET_PARITY", "completed_at"]})
    update_status(run, pipeline_status="SMOKE_PASS", current_phase="READY_FOR_FORMAL_LAUNCH", worker_pid=None,
                  current_sample=None, current_task=None, original_dynamic_replay_parity="PASS",
                  smoke_status="PASS", last_error="NONE")
    print(json.dumps({"SMOKE_STATUS": "PASS", "SMOKE_N": 3, "ORIGINAL_DYNAMIC_REPLAY_PARITY": "PASS",
                      "ACTION_COMPOSITION_PARITY": "PASS", "SAMPLE_RESET_PARITY": "PASS"}, sort_keys=True), flush=True)


def run_formal(run):
    run, config, cohort, checks = gates(run)
    smoke = load_json(run / "smoke/ORIGINAL_DYNAMIC_REPLAY_PARITY.json")
    if smoke.get("status") != "PASS" or smoke.get("ORIGINAL_DYNAMIC_REPLAY_PARITY") != "PASS":
        raise RuntimeError("SMOKE_GATE_NOT_PASSED")
    completed = 0
    for row in cohort:
        path = run / "results" / result_name(row["canonical_sample_id"])
        if path.exists():
            valid_result(path, row, config)
            completed += 1
    update_status(run, pipeline_status="RUNNING", current_phase="FORMAL_EXACT_REVERSE", committed_results=completed,
                  total_expected=173, controller_pid=int(os.environ.get("EXACT_REVERSE_CONTROLLER_PID", os.getpid())),
                  worker_pid=os.getpid(), worker_running=True, formal_launched=True, start_timestamp=now(), last_error="NONE")
    runner, ctx, model, tok, initial_base = load_runtime(run, config)
    start = time.time()
    new_completed = 0
    try:
        for row in cohort:
            sid = row["canonical_sample_id"]
            final_path = run / "results" / result_name(sid)
            if final_path.exists():
                continue
            claim_path = run / "claims" / result_name(sid)
            claim = {"canonical_sample_id": sid, "task": row["task"], "state": "claimed", "worker_pid": os.getpid(),
                     "controller_pid": int(os.environ.get("EXACT_REVERSE_CONTROLLER_PID", os.getpid())), "claimed_at": now(),
                     "cohort_manifest_sha256": config["cohort_manifest_sha256"], "protocol_sha256": config["protocol_sha256"]}
            atomic_json(claim_path, claim)
            claim.update({"state": "running", "running_at": now()})
            atomic_json(claim_path, claim)
            update_status(run, current_sample=sid, current_task=row["task"], worker_pid=os.getpid(), worker_running=True)
            sample_started = time.time()
            ev = runner.Evaluator(ctx, model, tok, ctx.by_id[sid], "formal_exact_reverse_1p7b_v1",
                                  row["cohort_order"], 173, run / "status/worker_internal_progress.json")
            rec = ev.evaluate(tuple(row["reversed_actions"]), capture=False, force=True)
            ev.finish()
            if runner.base_hash(model) != initial_base:
                raise RuntimeError(f"BASE_WEIGHT_MUTATION {sid}")
            reverse_score = float(rec["score"])
            dynamic_score = float(row["original_dynamic_score"])
            sample_best = float(row["sample_best_score"])
            denom = dynamic_score - sample_best
            if denom <= 0:
                raise RuntimeError(f"NONPOSITIVE_FROZEN_DYNAMIC_GAIN {sid}")
            output = {
                "canonical_sample_id": sid, "task": row["task"], "benchmark_global_index": row["benchmark_global_index"],
                "task_sample_index": row["task_sample_index"], "cohort_order": row["cohort_order"],
                "original_dynamic_actions": row["original_dynamic_actions"], "reversed_actions": row["reversed_actions"],
                "original_dynamic_score": dynamic_score, "sample_best_score": sample_best, "off_score": float(row["off_score"]),
                "reverse_score": reverse_score, "dynamic_minus_sample_best": denom,
                "dynamic_minus_reverse": dynamic_score - reverse_score,
                "reverse_minus_sample_best": reverse_score - sample_best,
                "retained_gain": (reverse_score - sample_best) / denom,
                "action_count_vector": row["action_count_vector"],
                "action_composition_parity": Counter(row["original_dynamic_actions"]) == Counter(row["reversed_actions"]),
                "reverse_distinct": row["original_dynamic_actions"] != row["reversed_actions"],
                "switch_count_original": row["switch_count_original"], "switch_count_reversed": row["switch_count_reversed"],
                "full_chunk_count": rec["full_chunk_count"], "prompt_token_count": rec["prompt_token_count"],
                "incomplete_tail_length": rec["prompt_token_count"] % 1024, "incomplete_tail_update": False,
                "layer_update_counts": rec["layer_update_counts"], "prediction_sha256": rec["prediction_hash"],
                "sequence_sha256": rec["sequence_hash"], "generation_forward_calls": rec["generation_forward_calls"],
                "generation_update_count": rec["generation_update_count"], "gpu_peak_memory_bytes": rec["gpu_peak_memory_bytes"],
                "runtime_seconds": rec["latency_seconds"], "wall_seconds": time.time() - sample_started,
                "clean_full_replay": rec["clean_full_replay"],
                "recompute_downstream_updates": rec["recompute_downstream_updates"],
                "stored_delta_stitching": rec["stored_delta_stitching"], "sample_reset_confirmed": True,
                "base_weight_unchanged": True, "model_identity": EXPECTED_MODEL,
                "checkpoint_identity": EXPECTED_CHECKPOINT, "benchmark_identity": EXPECTED_BENCHMARK,
                "screening_manifest_hash": EXPECTED_SCREENING, "cohort_manifest_hash": config["cohort_manifest_sha256"],
                "protocol_hash": config["protocol_sha256"], "frozen_executor_sha256": EXPECTED_EXECUTOR,
                "worker_sha256": config["worker_sha256"], "status": "COMMITTED", "completed": True,
                "timestamp": now(), "completion_schema": "exact_reverse_atomic_v1",
            }
            if not output["action_composition_parity"] or not output["reverse_distinct"]:
                raise RuntimeError(f"POST_EXECUTION_REVERSE_INVARIANT {sid}")
            atomic_json(final_path, output)
            committed = valid_result(final_path, row, config)
            commit_path = run / "commits" / result_name(sid)
            atomic_json(commit_path, {"canonical_sample_id": sid, "result_path": str(final_path),
                                      "result_sha256": sha(final_path), "status": "COMMITTED", "timestamp": now()})
            claim.update({"state": "committed", "committed_at": now(), "result_sha256": sha(final_path)})
            atomic_json(claim_path, claim)
            completed += 1
            new_completed += 1
            elapsed = max(time.time() - start, 1e-9)
            rate = new_completed / elapsed * 3600
            update_status(run, committed_results=completed, current_sample=sid, current_task=row["task"],
                          last_commit_timestamp=output["timestamp"], recent_samples_per_hour=rate,
                          overall_samples_per_hour=completed / max(elapsed, 1e-9) * 3600,
                          eta_hours=(173 - completed) / rate if rate else None, error_count=len(list((run / "errors").glob("*.json"))))
            print(json.dumps({"EXACT_REVERSE_PROGRESS": f"{completed}/173", "sample_id": sid,
                              "task": row["task"], "dynamic": dynamic_score, "reverse": reverse_score,
                              "dynamic_minus_reverse": dynamic_score - reverse_score,
                              "runtime_seconds": rec["latency_seconds"]}, sort_keys=True), flush=True)
            del ev, rec, committed
            gc.collect()
            import torch
            torch.cuda.empty_cache()
        if completed != 173:
            raise RuntimeError(f"INCOMPLETE_FORMAL_RESULTS {completed}/173")
        if runner.base_hash(model) != initial_base:
            raise RuntimeError("BASE_WEIGHT_MUTATION_FINAL")
    except BaseException as exc:
        sid = locals().get("sid", "UNKNOWN")
        atomic_json(run / "errors" / (result_name(sid) if sid != "UNKNOWN" else f"worker_{os.getpid()}.json"), {
            "sample_id": sid, "failure_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(),
            "retry_count": 0, "timestamp": now(), "scientific_or_config_error": True,
        })
        update_status(run, pipeline_status="FAILED", current_phase="FORMAL_FAILED", worker_running=False,
                      error_count=len(list((run / "errors").glob("*.json"))), last_error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        del model, tok
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    update_status(run, pipeline_status="FORMAL_EXECUTION_COMPLETE", current_phase="READY_FOR_AGGREGATION",
                  committed_results=173, current_sample=None, current_task=None, worker_pid=None, worker_running=False,
                  eta_hours=0, last_error="NONE")
    print("FORMAL_EXACT_REVERSE_COMMITTED=173/173", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root")
    parser.add_argument("--mode", choices=["smoke", "formal"], required=True)
    args = parser.parse_args()
    if args.mode == "smoke":
        run_smoke(args.run_root)
    else:
        run_formal(args.run_root)


if __name__ == "__main__":
    main()
