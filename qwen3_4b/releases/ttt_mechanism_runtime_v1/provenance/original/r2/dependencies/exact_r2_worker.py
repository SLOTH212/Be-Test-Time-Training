#!/usr/bin/env python3
"""Single-worker, crash-safe, manifest-authoritative Exact-R2 executor."""
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
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

MODEL_SHA = "ca51ee90d20ee09efbe0b31dd21bf9faa70275508316d1ba7cb1387ee4f0725f"
CKPT_SHA = "f4e264ed6f5671f123d6c8a27c69bce3af5ffcba518c9e298398314c6147d9fd"
BENCH_SHA = "5d16e3a6963d4c21450f0530af743730b91bf7b2442699e288e10754f77ad0c5"
RULE_SHA = "afeb7f0ceb42cfaa753ea04d926866ba346f83015be92ff8ad5138d5c32f8378"
ELIG_SHA = "f966cbfd65f8f9fbc0d079e44cd5e46fb24ca7da72130f5200c84be6faee3a42"
CAND_SHA = "737a5244a561e103c660ae33aff7a94e9c53e195d1fc05fc1006a1a7b4979739"
DYNAMIC_SHA = "89e5b7da8b85ce063af15216ef9a16de5c433d4e489a19efaae20ad16099689f"
EXECUTOR_SHA = "293886005cb8f78c943f0affe3e3b383d6de238f4bc825a5aea28a6e6dc66f6e"
SCORER_SHA = "cee3f5cb3cad1709f81abe3dc32b364ca97322e7ebd91747a8b119aecb0ce556"
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


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
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


def atomic(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def imp(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def update_status(run, **changes):
    path = run / "status/pipeline_status.json"
    state = load(path) if path.exists() else {}
    state.update(changes)
    state["updated_at"] = now()
    atomic(path, state)


def sample_key(sample_id):
    return hashlib.sha256(sample_id.encode()).hexdigest()[:20]


def branch_path(run, candidate):
    return run / "branches" / sample_key(candidate["canonical_sample_id"]) / f"{candidate['candidate_id']}.json"


def winner_path(run, sample_id):
    return run / "winners" / f"{sample_key(sample_id)}.json"


class Context:
    def __init__(self, config):
        self.run = Path(config["run_root"])
        self.code = Path("/home/USER/ttt/code/ntp")
        self.ckpt = Path(config["model_root"])
        self.score_fn = imp(config["scorer"], "exact_r2_frozen_scorer").per_sample_score
        all_rows = rows(Path(config["benchmark_root"]) / "sample_manifest.jsonl")
        self.rows = []
        for i, row in enumerate(all_rows):
            row = dict(row)
            row["_sample_index"] = i
            self.rows.append(row)
        self.by_id = {r["sample_id"]: r for r in self.rows}
        self.manifest = {"tasks": sorted({r["task"] for r in self.rows})}
        self.baseline = {}
        if len(self.rows) != 6500 or len(self.by_id) != 6500:
            raise RuntimeError("BENCHMARK_SAMPLE_SET_INVALID")


def gates(run):
    run = Path(run).resolve()
    config_path = run / "config/EXACT_R2_FORMAL_CONFIG.json"
    config = load(config_path)
    authority = load(run / "manifests/FORMAL_EXECUTION_AUTHORITY.json")
    protocol_path = Path(config["protocol"])
    candidate_path = Path(config["candidate_manifest"])
    eligibility_path = Path(config["eligibility_manifest"])
    rule_path = Path(config["rule_manifest"])
    checks = {
        "run_root": config["run_root"] == str(run),
        "model": sha(Path(config["model_root"]) / "model.safetensors") == MODEL_SHA == config["model_identity"],
        "checkpoint": config["checkpoint_identity"] == CKPT_SHA and authority["model"]["checkpoint_identity"] == CKPT_SHA,
        "benchmark": sha(Path(config["benchmark_root"]) / "benchmark_manifest.json") == BENCH_SHA == config["benchmark_identity"],
        "rule": sha(rule_path) == RULE_SHA == config["rule_sha256"],
        "eligibility": sha(eligibility_path) == ELIG_SHA == config["eligibility_sha256"],
        "candidate": sha(candidate_path) == CAND_SHA == config["candidate_manifest_sha256"],
        "dynamic": sha(config["dynamic_final_audit"]) == DYNAMIC_SHA,
        "executor": sha(config["frozen_executor"]) == EXECUTOR_SHA,
        "scorer": sha(config["scorer"]) == SCORER_SHA,
        "protocol": sha(protocol_path) == config["protocol_sha256"],
        "mapping": sha(config["one_switch_mapping"]) == config["one_switch_mapping_sha256"],
        "worker": sha(config["worker"]) == config["worker_sha256"],
        "aggregator": sha(config["aggregator"]) == config["aggregator_sha256"],
        "controller": sha(config["controller"]) == config["controller_sha256"],
        "authority": sha(run / "manifests/FORMAL_EXECUTION_AUTHORITY.json") == config["execution_authority_sha256"],
        "authority_internal": authority["candidate_manifest"]["sha256"] == CAND_SHA,
        "affinity": sorted(os.sched_getaffinity(0)) == list(range(8, 24)),
    }
    if not all(checks.values()):
        raise RuntimeError("AUTHORITY_OR_AFFINITY_MISMATCH " + json.dumps(checks, sort_keys=True))
    manifest = rows(candidate_path)
    if len(manifest) != 99120 or len({r["candidate_id"] for r in manifest}) != 99120:
        raise RuntimeError("CANDIDATE_MANIFEST_SCOPE")
    by_sample = defaultdict(list)
    seen_schedule = set()
    for candidate in manifest:
        sid = candidate["canonical_sample_id"]
        seq = candidate["full_chunk_level_action_sequence"]
        expected = [candidate["action_A"]] * candidate["tau"] + [candidate["action_B"]] * (candidate["T"] - candidate["tau"])
        if (candidate["action_A"] == candidate["action_B"] or not 1 <= candidate["tau"] < candidate["T"]
                or seq != expected or canonical(seq) != candidate["canonical_schedule_hash"]
                or candidate["regime_count"] != 2 or candidate["switch_count"] != 1):
            raise RuntimeError("CANDIDATE_ROW_INVALID " + candidate["candidate_id"])
        key = (sid, candidate["canonical_schedule_hash"])
        if key in seen_schedule:
            raise RuntimeError("DUPLICATE_SAMPLE_SCHEDULE " + candidate["candidate_id"])
        seen_schedule.add(key)
        by_sample[sid].append(candidate)
    counts = sorted(len(v) for v in by_sample.values())
    if len(by_sample) != 173 or min(counts) != 462 or max(counts) != 588 or statistics_median(counts) != 588:
        raise RuntimeError("PER_SAMPLE_CANDIDATE_COUNTS")
    mappings = rows(config["one_switch_mapping"])
    if len(mappings) != 100:
        raise RuntimeError("ONE_SWITCH_MAPPING_SCOPE")
    return run, config, manifest, by_sample, mappings, checks


def statistics_median(values):
    values = sorted(values)
    n = len(values)
    return values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2


def valid_result(path, candidate, config):
    record = load(path)
    finite = isinstance(record.get("score"), (int, float)) and math.isfinite(record["score"])
    valid = (
        record.get("SUCCESS") is True
        and record.get("execution_status") == "SUCCESS"
        and record.get("candidate_id") == candidate["candidate_id"]
        and record.get("canonical_sample_id") == candidate["canonical_sample_id"]
        and record.get("task") == candidate["task"]
        and record.get("A") == candidate["action_A"]
        and record.get("B") == candidate["action_B"]
        and record.get("tau") == candidate["tau"]
        and record.get("T") == candidate["T"]
        and record.get("full_schedule_hash") == candidate["canonical_schedule_hash"]
        and record.get("model_identity") == MODEL_SHA
        and record.get("checkpoint_identity") == CKPT_SHA
        and record.get("benchmark_identity") == BENCH_SHA
        and record.get("rule_sha256") == RULE_SHA
        and record.get("eligibility_sha256") == ELIG_SHA
        and record.get("candidate_manifest_sha256") == CAND_SHA
        and record.get("protocol_sha256") == config["protocol_sha256"]
        and record.get("clean_full_replay") is True
        and record.get("candidate_prefix_reuse") is False
        and record.get("stored_delta_stitching") is False
        and record.get("generation_update_count") == 0
        and isinstance(record.get("prediction"), str)
        and record.get("prediction_sha256") == hashlib.sha256(record["prediction"].encode()).hexdigest()
        and finite
    )
    if not valid:
        raise RuntimeError("INVALID_EXISTING_FORMAL_RESULT " + str(path))
    return record


def load_runtime(run, config):
    import torch
    torch.set_num_threads(16)
    runner = imp(config["frozen_executor"], "exact_r2_frozen_dynamic_executor")
    if runner.CHECKPOINT_HASH != CKPT_SHA:
        raise RuntimeError("EXECUTOR_CHECKPOINT_MISMATCH")
    context = Context(config)
    update_status(run, current_phase="MODEL_LOADING", worker_pid=os.getpid(), worker_running=True)
    model, tokenizer = runner.load_model(context)
    model.requires_grad_(False)
    base = runner.base_hash(model)
    atomic(run / "audit/MODEL_LOAD_AUDIT.json", {
        "status": "PASS", "timestamp": now(), "model_identity": MODEL_SHA,
        "checkpoint_identity": CKPT_SHA, "base_weight_runtime_hash": base,
        "cuda_device": torch.cuda.get_device_name(0), "cpu_affinity": sorted(os.sched_getaffinity(0)),
    })
    return runner, context, model, tokenizer, base


def execute(runner, context, model, tokenizer, bench_row, sequence, stage, ordinal, total, progress_path):
    """Use frozen evaluator while capturing decoded text without enabling state hashing."""
    captured = {}
    original_decode = tokenizer.decode
    def capture_decode(*args, **kwargs):
        text = original_decode(*args, **kwargs)
        captured["prediction"] = text
        return text
    tokenizer.decode = capture_decode
    try:
        ev = runner.Evaluator(context, model, tokenizer, bench_row, stage, ordinal, total, progress_path)
        for layer in runner.LAYERS:
            mlp = model.model.layers[layer].mlp
            mlp._dynamic_sequence = None
            mlp._branch_id = None
            mlp._capture_identity = False
            mlp._events = []
            mlp._generation_forward_calls = 0
            mlp._captured_final_state = None
        rec = ev.evaluate(tuple(sequence), capture=False, force=True)
        ev.finish()
    finally:
        tokenizer.decode = original_decode
    prediction = captured.get("prediction")
    if prediction is None or hashlib.sha256(prediction.encode()).hexdigest() != rec["prediction_hash"]:
        raise RuntimeError("PREDICTION_CAPTURE_MISMATCH")
    if not rec["clean_full_replay"] or not rec["recompute_downstream_updates"] or rec["stored_delta_stitching"]:
        raise RuntimeError("REPLAY_SEMANTICS_MISMATCH")
    return rec, prediction


def choose_winner(records):
    max_score = max(float(r["score"]) for r in records)
    ties = [r for r in records if abs(float(r["score"]) - max_score) <= TOL]
    order = {a: i for i, a in enumerate(ACTIONS)}
    ties.sort(key=lambda r: (int(r["tau"]), order[r["A"]], order[r["B"]], r["full_schedule_hash"]))
    return ties[0], max_score, len(ties)


def run_smoke(run):
    run, config, manifest, by_sample, mappings, checks = gates(run)
    if any((run / "branches").rglob("*.json")):
        raise RuntimeError("FORMAL_RESULTS_EXIST_BEFORE_SMOKE")
    smoke_ids = config["one_switch_smoke_candidate_ids"]
    mapping_by_candidate = {x["candidate_id"]: x for x in mappings}
    if len(smoke_ids) != 3 or any(x not in mapping_by_candidate for x in smoke_ids):
        raise RuntimeError("ONE_SWITCH_SMOKE_SCOPE")
    update_status(run, pipeline_status="SMOKE_RUNNING", current_phase="ONE_SWITCH_PARITY_SMOKE",
                  worker_pid=os.getpid(), worker_running=True, last_error="NONE")
    runner, context, model, tokenizer, base = load_runtime(run, config)
    parity_rows = []
    try:
        for ordinal, candidate_id in enumerate(smoke_ids, 1):
            mapping = mapping_by_candidate[candidate_id]
            candidate = next(x for x in by_sample[mapping["canonical_sample_id"]] if x["candidate_id"] == candidate_id)
            sid = candidate["canonical_sample_id"]
            update_status(run, current_sample=sid, current_task=candidate["task"], current_candidate=candidate_id,
                          current_A=candidate["action_A"], current_B=candidate["action_B"], current_tau=candidate["tau"])
            rec, prediction = execute(runner, context, model, tokenizer, context.by_id[sid],
                                      candidate["full_chunk_level_action_sequence"], "exact_r2_one_switch_smoke",
                                      ordinal, 3, run / "smoke/ONE_SWITCH_INTERNAL_PROGRESS.json")
            score_match = abs(float(rec["score"]) - float(mapping["frozen_dynamic_score"])) <= TOL
            prediction_match = rec["prediction_hash"] == mapping["frozen_dynamic_prediction_sha256"]
            if not score_match or not prediction_match:
                raise RuntimeError("ONE_SWITCH_DYNAMIC_REPLAY_PARITY_FAILED " + sid)
            parity_rows.append({
                "candidate_id": candidate_id, "canonical_sample_id": sid, "task": candidate["task"],
                "A": candidate["action_A"], "B": candidate["action_B"], "tau": candidate["tau"],
                "frozen_dynamic_score": mapping["frozen_dynamic_score"], "replay_score": rec["score"],
                "frozen_dynamic_prediction_sha256": mapping["frozen_dynamic_prediction_sha256"],
                "replay_prediction_sha256": rec["prediction_hash"], "score_match": score_match,
                "prediction_match": prediction_match, "formal_result": False,
            })
            del rec, prediction
            gc.collect()
            import torch
            torch.cuda.empty_cache()

        det = config["determinism_smoke"]
        candidates = by_sample[det["canonical_sample_id"]]
        first = next(x for x in candidates if x["candidate_id"] == det["candidate_id"])
        intervening = next(x for x in candidates if x["candidate_id"] == det["intervening_candidate_id"])
        update_status(run, current_phase="R2_DETERMINISM_SMOKE", current_sample=det["canonical_sample_id"],
                      current_task=first["task"], current_candidate=first["candidate_id"])
        r1, p1 = execute(runner, context, model, tokenizer, context.by_id[det["canonical_sample_id"]],
                         first["full_chunk_level_action_sequence"], "exact_r2_determinism_smoke_A1", 1, 3,
                         run / "smoke/DETERMINISM_INTERNAL_PROGRESS.json")
        rb, pb = execute(runner, context, model, tokenizer, context.by_id[det["canonical_sample_id"]],
                         intervening["full_chunk_level_action_sequence"], "exact_r2_determinism_smoke_B", 2, 3,
                         run / "smoke/DETERMINISM_INTERNAL_PROGRESS.json")
        r2, p2 = execute(runner, context, model, tokenizer, context.by_id[det["canonical_sample_id"]],
                         first["full_chunk_level_action_sequence"], "exact_r2_determinism_smoke_A2", 3, 3,
                         run / "smoke/DETERMINISM_INTERNAL_PROGRESS.json")
        integrity_fields = ["sequence_hash", "score", "prediction_hash", "prompt_token_count", "full_chunk_count",
                            "layer_update_counts", "generation_update_count", "generation_forward_calls",
                            "clean_full_replay", "recompute_downstream_updates", "stored_delta_stitching"]
        deterministic = p1 == p2 and all(r1[k] == r2[k] for k in integrity_fields)
        if not deterministic:
            raise RuntimeError("R2_REPLAY_DETERMINISM_FAILED")
        if runner.base_hash(model) != base:
            raise RuntimeError("BASE_WEIGHT_MUTATION_AFTER_SMOKE")
        determinism = {
            "status": "PASS", "formal_result": False, "canonical_sample_id": det["canonical_sample_id"],
            "candidate_id": first["candidate_id"], "intervening_candidate_id": intervening["candidate_id"],
            "pattern": "A_B_A", "score": r1["score"], "prediction_sha256": r1["prediction_hash"],
            "candidate_result_integrity_exact": True, "cross_candidate_state_leakage": False,
        }
        smoke = {
            "status": "PASS", "formal_result": False, "ONE_SWITCH_DYNAMIC_REPLAY_SMOKE": "PASS",
            "R2_REPLAY_DETERMINISM_SMOKE": "PASS", "SCORER_PARITY": "PASS",
            "one_switch_smoke_n": 3, "authority_checks": checks, "parity_rows": parity_rows,
            "determinism": determinism, "scorer_sha256": SCORER_SHA, "base_weight_unchanged": True,
            "completed_at": now(),
        }
        atomic(run / "smoke/ONE_SWITCH_DYNAMIC_REPLAY_SMOKE.json", {"status": "PASS", "rows": parity_rows, "formal_result": False})
        atomic(run / "smoke/R2_REPLAY_DETERMINISM_SMOKE.json", determinism)
        atomic(run / "smoke/SMOKE_GATE.json", smoke)
    except BaseException as exc:
        atomic(run / "errors/smoke_failure.json", {"failure_type": type(exc).__name__, "error": str(exc),
               "traceback": traceback.format_exc(), "timestamp": now()})
        update_status(run, pipeline_status="FAILED", current_phase="SMOKE_FAILED", worker_running=False,
                      last_error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        del model, tokenizer
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    update_status(run, pipeline_status="SMOKE_PASS", current_phase="READY_FOR_FORMAL_LAUNCH",
                  worker_pid=None, worker_running=False, current_sample=None, current_task=None,
                  current_candidate=None, one_switch_dynamic_replay_smoke="PASS",
                  r2_replay_determinism_smoke="PASS", scorer_parity="PASS", last_error="NONE")
    print(json.dumps({"ONE_SWITCH_DYNAMIC_REPLAY_SMOKE": "PASS", "R2_REPLAY_DETERMINISM_SMOKE": "PASS",
                      "SCORER_PARITY": "PASS"}, sort_keys=True), flush=True)


def run_formal(run):
    run, config, manifest, by_sample, mappings, checks = gates(run)
    smoke = load(run / "smoke/SMOKE_GATE.json")
    if any(smoke.get(k) != "PASS" for k in ["ONE_SWITCH_DYNAMIC_REPLAY_SMOKE", "R2_REPLAY_DETERMINISM_SMOKE", "SCORER_PARITY"]):
        raise RuntimeError("SMOKE_GATE_NOT_PASSED")
    completed = 0
    completed_by_sample = Counter()
    existing = {}
    for candidate in manifest:
        path = branch_path(run, candidate)
        if path.exists():
            record = valid_result(path, candidate, config)
            completed += 1
            completed_by_sample[candidate["canonical_sample_id"]] += 1
            existing[candidate["candidate_id"]] = record
    update_status(run, pipeline_status="RUNNING", current_phase="FORMAL_EXACT_R2_EXECUTION",
                  formal_launched=True, candidates_committed=completed, candidates_expected=99120,
                  samples_fully_searched=sum(completed_by_sample[s] == len(by_sample[s]) for s in by_sample),
                  worker_pid=os.getpid(), worker_running=True,
                  controller_pid=int(os.environ.get("EXACT_R2_CONTROLLER_PID", os.getpid())), last_error="NONE")
    runner, context, model, tokenizer, base = load_runtime(run, config)
    started = time.time()
    newly_completed = 0
    try:
        for sample_ordinal, (sid, candidates) in enumerate(by_sample.items(), 1):
            records = []
            for c in candidates:
                path = branch_path(run, c)
                if path.exists():
                    records.append(existing.get(c["candidate_id"]) or valid_result(path, c, config))
                    continue
                started_at = now()
                update_status(run, current_sample=sid, current_task=c["task"], current_candidate=c["candidate_id"],
                              current_A=c["action_A"], current_B=c["action_B"], current_tau=c["tau"],
                              current_sample_candidates_completed=len(records),
                              current_sample_candidates_expected=len(candidates))
                wall_start = time.time()
                rec, prediction = execute(runner, context, model, tokenizer, context.by_id[sid],
                                          c["full_chunk_level_action_sequence"], "formal_exact_r2_1p7b_v1",
                                          sample_ordinal, 173, run / "status/worker_internal_progress.json")
                if runner.base_hash(model) != base:
                    raise RuntimeError("BASE_WEIGHT_MUTATION " + c["candidate_id"])
                output = {
                    "candidate_id": c["candidate_id"], "canonical_sample_id": sid, "task": c["task"],
                    "benchmark_global_index": c["benchmark_global_index"], "A": c["action_A"],
                    "B": c["action_B"], "tau": c["tau"], "T": c["T"],
                    "regime_count": 2, "switch_count": 1,
                    "full_schedule_hash": c["canonical_schedule_hash"], "score": float(rec["score"]),
                    "prediction": prediction, "prediction_sha256": rec["prediction_hash"],
                    "execution_status": "SUCCESS", "SUCCESS": True,
                    "model_identity": MODEL_SHA, "checkpoint_identity": CKPT_SHA, "benchmark_identity": BENCH_SHA,
                    "rule_sha256": RULE_SHA, "eligibility_sha256": ELIG_SHA,
                    "candidate_manifest_sha256": CAND_SHA, "protocol_sha256": config["protocol_sha256"],
                    "executor_sha256": EXECUTOR_SHA, "scorer_sha256": SCORER_SHA,
                    "clean_full_replay": True, "normal_per_sample_initial_state": True,
                    "candidate_prefix_reuse": False, "dynamic_state_transplant": False,
                    "cross_candidate_ttt_state_reuse": False, "stored_delta_stitching": False,
                    "recompute_downstream_updates": True, "generation_update_count": rec["generation_update_count"],
                    "layer_update_counts": rec["layer_update_counts"],
                    "prompt_token_count": rec["prompt_token_count"], "incomplete_tail_length": rec["prompt_token_count"] % 1024,
                    "gpu_peak_memory_bytes": rec["gpu_peak_memory_bytes"], "runtime_seconds": rec["latency_seconds"],
                    "wall_seconds": time.time() - wall_start, "started_at": started_at, "completed_at": now(),
                }
                atomic(path, output)
                valid_result(path, c, config)
                records.append(output)
                completed += 1
                newly_completed += 1
                completed_by_sample[sid] += 1
                elapsed = max(time.time() - started, 1e-9)
                rate = newly_completed / elapsed * 3600
                update_status(run, candidates_committed=completed, current_sample_candidates_completed=len(records),
                              last_commit_timestamp=output["completed_at"], recent_candidates_per_hour=rate,
                              eta_hours=(99120 - completed) / rate if rate else None)
                if completed % 10 == 0 or len(records) == len(candidates):
                    print(json.dumps({"R2_PROGRESS": f"{completed}/99120", "sample": sid,
                                      "sample_progress": f"{len(records)}/{len(candidates)}",
                                      "candidate": c["candidate_id"], "score": output["score"]}, sort_keys=True), flush=True)
                del rec, prediction
                if completed % 25 == 0:
                    gc.collect()
                    import torch
                    torch.cuda.empty_cache()
            if len(records) != len(candidates):
                raise RuntimeError("INCOMPLETE_SAMPLE " + sid)
            winning, best_score, tie_n = choose_winner(records)
            winner = {
                "canonical_sample_id": sid, "task": candidates[0]["task"],
                "winning_candidate_id": winning["candidate_id"], "R2_score": best_score,
                "winning_A": winning["A"], "winning_B": winning["B"], "winning_tau": winning["tau"],
                "T": winning["T"], "switch_count": 1, "regime_count": 2,
                "number_of_max_score_ties": tie_n,
                "tie_break_trace": ["maximum score", "earliest tau", "action order A", "action order B", "sequence SHA256"],
                "score_tolerance": TOL, "candidate_count": len(candidates),
                "candidate_manifest_sha256": CAND_SHA, "protocol_sha256": config["protocol_sha256"],
                "completed_at": now(),
            }
            atomic(winner_path(run, sid), winner)
            update_status(run, samples_fully_searched=sum(completed_by_sample[s] == len(by_sample[s]) for s in by_sample),
                          winners_finalized=len(list((run / "winners").glob("*.json"))))
            del records
            gc.collect()
            import torch
            torch.cuda.empty_cache()
        if completed != 99120:
            raise RuntimeError(f"INCOMPLETE_CANDIDATES {completed}/99120")
        if runner.base_hash(model) != base:
            raise RuntimeError("BASE_WEIGHT_MUTATION_FINAL")
    except BaseException as exc:
        key = locals().get("c", {}).get("candidate_id", "worker") if isinstance(locals().get("c"), dict) else "worker"
        atomic(run / "errors" / f"{canonical(key + now())}.json", {
            "candidate_id": key, "failure_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(), "timestamp": now(), "scientific_or_config_error": True,
        })
        update_status(run, pipeline_status="FAILED", current_phase="FORMAL_FAILED", worker_running=False,
                      last_error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        del model, tokenizer
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    update_status(run, pipeline_status="FORMAL_EXECUTION_COMPLETE", current_phase="READY_FOR_AGGREGATION",
                  candidates_committed=99120, samples_fully_searched=173, winners_finalized=173,
                  current_sample=None, current_task=None, current_candidate=None, worker_pid=None,
                  worker_running=False, eta_hours=0, last_error="NONE")
    print("FORMAL_EXACT_R2_COMMITTED=99120/99120", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root")
    parser.add_argument("--mode", choices=["smoke", "formal"], required=True)
    args = parser.parse_args()
    if args.mode == "smoke":
        run_smoke(args.run_root)
    else:
        run_formal(args.run_root)
