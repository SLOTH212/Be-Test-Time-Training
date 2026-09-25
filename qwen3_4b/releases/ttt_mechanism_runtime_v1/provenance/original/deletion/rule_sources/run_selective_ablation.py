#!/usr/bin/env python3
"""Execute frozen per-layer component removals with the formal replay backend."""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

WS = Path("/home/USER/ttt_mechanism_analysis_v1/selective_state_component_ablation_v1")
FORMAL = Path("/home/USER/ttt_dynamic_mining_v1/phase_d_enriched_dynamic_mining_v1/runtime/code")
EXECUTOR = FORMAL / "dynamic_runner.py"
INSTRUMENT = Path("/home/USER/ttt_mechanism_analysis_v1/targeted_fast_weight_state_replay_v1/scripts/instrumented_backend.py")
MODEL = Path("/home/USER/ttt_phase_c_migrated_v1/ttt_phase_c_migration_20260813/checkpoint/model.safetensors")
MANIFEST = WS / "manifests/COMPONENT_ABLATION_MANIFEST.jsonl"
EXPECTED_MANIFEST = "d5e972a2c5b34145b512c5a66c73c39ae3514daeda8ee5a079db32793176cbb4"
EXPECTED_EXECUTOR = "293886005cb8f78c943f0affe3e3b383d6de238f4bc825a5aea28a6e6dc66f6e"
EXPECTED_MODEL = "ca51ee90d20ee09efbe0b31dd21bf9faa70275508316d1ba7cb1387ee4f0725f"
EXPECTED_CHECKPOINT = "f4e264ed6f5671f123d6c8a27c69bce3af5ffcba518c9e298398314c6147d9fd"
LAYERS = [0, 6, 12, 18, 24]


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(finite(obj), f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite(item) for item in value]
    return value



def configure(model, ablate_layers, boundary):
    selected = set(ablate_layers)
    for layer in LAYERS:
        module = model.model.layers[layer].mlp
        module._reset_boundary_chunk = int(boundary) if layer in selected else None
        module._instrumentation_enabled = True
        module._state_events = []
        module._reset_events = []


def collect(model, backend):
    state_events, reset_events = [], []
    for layer in LAYERS:
        module = model.model.layers[layer].mlp
        state_events.extend(backend.materialize_scalar(list(module._state_events)))
        reset_events.extend(backend.materialize_scalar(list(module._reset_events)))
    return state_events, reset_events


def evaluate(runner, backend, ctx, model, tok, row, stage, ablate_layers):
    configure(model, ablate_layers, row["switch_boundary"])
    evaluator = runner.Evaluator(ctx, model, tok, ctx.by_id[row["sample_id"]], stage,
                                 int(row.get("manifest_order", 0)), 1, WS / "logs/FORMAL_PROGRESS.json")
    record = evaluator.evaluate(tuple(row["action_sequence"]), capture=True, force=True)
    evaluator.finish()
    state_events, reset_events = collect(model, backend)
    expected = 5 * int(row["decision_chunk_count"])
    if len(state_events) != expected:
        raise RuntimeError(f"STATE_EVENT_COUNT got={len(state_events)} want={expected}")
    if len(reset_events) != len(ablate_layers):
        raise RuntimeError(f"RESET_EVENT_COUNT got={len(reset_events)} want={len(ablate_layers)}")
    if any(event["chunk_index"] != row["switch_boundary"] for event in reset_events):
        raise RuntimeError("RESET_BOUNDARY_MISMATCH")
    if any(event["layer"] not in ablate_layers for event in reset_events):
        raise RuntimeError("RESET_LAYER_MISMATCH")
    return record, state_events, reset_events


def boundary_hashes(record, boundary):
    return {int(event["layer"]): event["pre_state_hash"] for event in record["chunk_layer_events"]
            if int(event["chunk_index"]) == int(boundary)}


def result_path(row):
    return WS / "results" / f"{int(row['manifest_order']):03d}_{row['intervention_id']}.json"


def valid(path, row, controller_hash):
    try:
        obj = json.load(open(path, encoding="utf-8"))
    except Exception:
        return False
    return (obj.get("completed") is True and obj.get("intervention_id") == row["intervention_id"]
            and obj.get("manifest_sha256") == EXPECTED_MANIFEST
            and obj.get("controller_sha256") == controller_hash)


def main():
    if sorted(os.sched_getaffinity(0)) != list(range(8, 24)):
        raise RuntimeError("CPUSET_MISMATCH")
    if sha(MANIFEST) != EXPECTED_MANIFEST or sha(EXECUTOR) != EXPECTED_EXECUTOR or sha(MODEL) != EXPECTED_MODEL:
        raise RuntimeError("PRE_FORWARD_IDENTITY_MISMATCH")
    freeze = json.load(open(WS / "audits/PRE_FORWARD_MANIFEST_FREEZE.json", encoding="utf-8"))
    if not freeze.get("manifest_frozen_before_new_scores") or freeze["component_ablation_manifest_sha256"] != EXPECTED_MANIFEST:
        raise RuntimeError("MANIFEST_FREEZE_MISSING")
    rows = load_jsonl(MANIFEST)
    if len(rows) != 89:
        raise RuntimeError("MANIFEST_SCOPE_MISMATCH")
    controller_hash = sha(Path(__file__))
    spec = importlib.util.spec_from_file_location("formal_runner", EXECUTOR)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    if runner.CHECKPOINT_HASH != EXPECTED_CHECKPOINT:
        raise RuntimeError("CHECKPOINT_IDENTITY_MISMATCH")
    ispec = importlib.util.spec_from_file_location("instrumented_backend", INSTRUMENT)
    backend = importlib.util.module_from_spec(ispec)
    ispec.loader.exec_module(backend)
    ctx = runner.Context(WS)
    ctx.code = FORMAL
    import torch
    torch.set_num_threads(16)
    model, tok = runner.load_model(ctx)
    initial_base = runner.base_hash(model)
    total_seconds = 0.0
    total_forwards = 0
    try:
        # Formal instrumentation-OFF replay.
        smoke = next(row for row in rows if row["intervention_role"] == "POSITIVE_ALL_PREFIX_LOO")
        off_eval = runner.Evaluator(ctx, model, tok, ctx.by_id[smoke["sample_id"]], "component_parity_off", 0, 1,
                                    WS / "logs/PARITY_OFF_PROGRESS.json")
        off_record = off_eval.evaluate(tuple(smoke["action_sequence"]), capture=True, force=True)
        off_eval.finish()
        total_seconds += off_record["latency_seconds"]
        total_forwards += 1
        if runner.base_hash(model) != initial_base:
            raise RuntimeError("BASE_MUTATION_AFTER_PARITY_OFF")

        # Semantic instrumentation twin, with no intervention.
        backend.install_instrumented_dynamic_forward(model, runner.tensor_hash)
        on_record, on_events, on_resets = evaluate(runner, backend, ctx, model, tok, smoke, "component_parity_on", [])
        total_seconds += on_record["latency_seconds"]
        total_forwards += 1
        parity_pass = (off_record["score"] == on_record["score"]
                       and off_record["prediction_hash"] == on_record["prediction_hash"] and not on_resets)
        parity = {"instrumentation_parity_pass": parity_pass,
                  "score_match": off_record["score"] == on_record["score"],
                  "prediction_hash_match": off_record["prediction_hash"] == on_record["prediction_hash"],
                  "off_score": off_record["score"], "on_score": on_record["score"],
                  "off_prediction_hash": off_record["prediction_hash"],
                  "on_prediction_hash": on_record["prediction_hash"],
                  "state_event_count": len(on_events), "base_weight_unchanged": runner.base_hash(model) == initial_base,
                  "model_identity_match": True, "checkpoint_identity_match": True,
                  "ablation_backend_match_formal": True}
        atomic_json(WS / "audits/INSTRUMENTATION_PARITY.json", parity)
        if not parity_pass:
            raise RuntimeError("INSTRUMENTATION_PARITY_FAILED")

        # Frozen engineering branch verifies selective, not-full-reset semantics.
        smoke_record, smoke_events, smoke_resets = evaluate(
            runner, backend, ctx, model, tok, smoke, "selective_verification_smoke", smoke["ablate_layers"])
        total_seconds += smoke_record["latency_seconds"]
        total_forwards += 1
        residual = max(float(x["post_reset_delta_norm"]) for x in smoke_resets)
        max_abs_residual = max(float(x["post_reset_max_abs_residual"]) for x in smoke_resets)
        normal_hash = boundary_hashes(on_record, smoke["switch_boundary"])
        selective_hash = boundary_hashes(smoke_record, smoke["switch_boundary"])
        preserved = [layer for layer in LAYERS if layer not in smoke["ablate_layers"]]
        preserved_hash_match = all(normal_hash[layer] == selective_hash[layer] for layer in preserved)
        preserved_nonzero = sum(
            float(next(event["fast_weight_delta_norm_before"] for event in on_events
                       if event["layer"] == layer and event["chunk_index"] == smoke["switch_boundary"])) > 0
            for layer in preserved)
        verification_pass = residual == 0.0 and max_abs_residual == 0.0 and preserved_hash_match
        verification = {"selective_ablation_verification_pass": verification_pass,
                        "sample_id": smoke["sample_id"], "switch_boundary": smoke["switch_boundary"],
                        "ablated_layers": smoke["ablate_layers"], "ablated_component_residual_norm": residual,
                        "ablated_component_max_abs_residual": max_abs_residual,
                        "preserved_layer_boundary_hash_exact_match": preserved_hash_match,
                        "max_preserved_component_abs_diff": 0.0 if preserved_hash_match else None,
                        "preserved_nonzero_component_count": preserved_nonzero,
                        "selective_not_full_reset": preserved_nonzero >= 1,
                        "base_weight_unchanged": runner.base_hash(model) == initial_base}
        atomic_json(WS / "audits/SELECTIVE_ABLATION_VERIFICATION.json", verification)
        if not verification_pass or not verification["selective_not_full_reset"]:
            raise RuntimeError("SELECTIVE_ABLATION_VERIFICATION_FAILED")
        print("INSTRUMENTATION_PARITY_PASS=true", flush=True)
        print("SELECTIVE_ABLATION_VERIFICATION_PASS=true", flush=True)

        completed = skipped = generation_calls = 0
        science_seconds = 0.0
        reset_residual_max = 0.0
        started = time.time()
        for row in rows:
            path = result_path(row)
            if valid(path, row, controller_hash):
                obj = json.load(open(path, encoding="utf-8"))
                completed += 1
                skipped += 1
                science_seconds += obj["runtime_seconds"]
                generation_calls += obj["generation_forward_calls"]
                reset_residual_max = max(reset_residual_max, obj["ablated_component_residual_norm"])
                continue
            atomic_json(WS / "logs/MONITOR.json", {"stage": "SELECTIVE_COMPONENT_ABLATION",
                        "completed": completed, "expected": len(rows), "intervention_id": row["intervention_id"],
                        "cohort": row["cohort"], "role": row["intervention_role"],
                        "sample_id": row["sample_id"], "native_crash": False, "updated_at": now()})
            if runner.base_hash(model) != initial_base:
                raise RuntimeError("BASE_WEIGHT_PRECONDITION")
            torch.cuda.reset_peak_memory_stats()
            record, state_events, reset_events = evaluate(
                runner, backend, ctx, model, tok, row, "selective_component_ablation", row["ablate_layers"])
            residual = max(float(x["post_reset_delta_norm"]) for x in reset_events)
            max_abs = max(float(x["post_reset_max_abs_residual"]) for x in reset_events)
            if residual != 0.0 or max_abs != 0.0:
                raise RuntimeError("ABLATION_RESIDUAL_NONZERO")
            preserved_nonzero = sum(
                float(next(e["fast_weight_delta_norm_before"] for e in state_events
                           if e["layer"] == layer and e["chunk_index"] == row["switch_boundary"])) > 0
                for layer in LAYERS if layer not in row["ablate_layers"])
            pre_norm = max(float(x["pre_reset_delta_norm"]) for x in reset_events)
            if row["intervention_role"] == "POSITIVE_ZERO_STATE_SHAM" and pre_norm != 0.0:
                raise RuntimeError("SHAM_PRE_ABLATION_COMPONENT_NONZERO")
            out = {**row, "score": record["score"], "prediction": record["prediction"],
                   "prediction_hash": record["prediction_hash"], "sequence_hash": record["sequence_hash"],
                   "runtime_seconds": record["latency_seconds"], "generation_forward_calls": record["generation_forward_calls"],
                   "generation_update_count": record["generation_update_count"], "state_events": state_events,
                   "reset_events": reset_events, "ablated_component_pre_norm": pre_norm,
                   "ablated_component_residual_norm": residual, "ablated_component_max_abs_residual": max_abs,
                   "preserved_nonzero_component_count": preserved_nonzero,
                   "selective_not_full_reset": preserved_nonzero >= 1 if row["intervention_role"] != "POSITIVE_ZERO_STATE_SHAM" else True,
                   "base_weight_unchanged": runner.base_hash(model) == initial_base,
                   "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
                   "manifest_sha256": EXPECTED_MANIFEST, "controller_sha256": controller_hash,
                   "instrumented_backend_sha256": sha(INSTRUMENT), "formal_executor_sha256": EXPECTED_EXECUTOR,
                   "model_identity": EXPECTED_MODEL, "checkpoint_identity": EXPECTED_CHECKPOINT,
                   "new_dynamic_search_count": 0, "router_training_count": 0, "new_mining_count": 0,
                   "completed": True, "completed_at": now()}
            atomic_json(path, out)
            completed += 1
            science_seconds += out["runtime_seconds"]
            generation_calls += out["generation_forward_calls"]
            reset_residual_max = max(reset_residual_max, residual)
            if completed == 1 or completed % 10 == 0 or completed == len(rows):
                print(f"ABLATION_PROGRESS={completed}/{len(rows)} cohort={row['cohort']} role={row['intervention_role']} seconds={out['runtime_seconds']:.4f}", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
        total_seconds += science_seconds
        total_forwards += completed
        audit = {"expected_intervention_count": 89, "complete_intervention_count": completed,
                 "skipped_valid_resume_count": skipped, "engineering_forward_count": 3,
                 "scientific_forward_count": completed, "total_forward_count": total_forwards,
                 "engineering_gpu_seconds": total_seconds - science_seconds,
                 "scientific_gpu_seconds": science_seconds, "total_gpu_seconds": total_seconds,
                 "generation_forward_calls": generation_calls,
                 "generation_update_count": 0, "new_dynamic_search_count": 0,
                 "router_training_count": 0, "new_mining_count": 0,
                 "ablation_residual_max": reset_residual_max,
                 "selective_ablation_verification_pass": verification_pass,
                 "instrumentation_parity_pass": parity_pass,
                 "base_weight_unchanged": runner.base_hash(model) == initial_base,
                 "model_identity_match": True, "checkpoint_identity_match": True,
                 "ablation_backend_match_formal": True, "gpu_model": torch.cuda.get_device_name(0),
                 "workers": 1, "cpuset": "8-23", "native_sigsegv_detected": False,
                 "wall_seconds": time.time() - started, "finished_at": now()}
        atomic_json(WS / "audits/EXECUTION_COMPLETENESS.json", audit)
        print("SELECTIVE_INTERVENTION_COMPLETE=" + str(completed))
        print("TOTAL_FORWARD_COUNT=" + str(total_forwards))
        print("TOTAL_GPU_SECONDS=" + str(total_seconds))
    except BaseException as exc:
        atomic_json(WS / "logs/EXECUTION_FAILURE.json",
                    {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "time": now()})
        raise
    finally:
        del model, tok
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
