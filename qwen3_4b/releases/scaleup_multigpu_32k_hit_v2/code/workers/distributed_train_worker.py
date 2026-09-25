#!/usr/bin/env python3
"""Selective FSDP2 restoration. This does not replace the audited 1-GPU worker."""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.distributed_checkpoint import (DistributedCheckpointRotation, load_model_only,
                                        preflight_checkpoint_metadata, save_model_only)
from lib.distributed_grad import clip_grad_norm_fsdp2
from lib.distributed_runtime import barrier, initialize, rank0_mkdir, shutdown
from lib.distributed_semantics import BatchPlan, assert_batch_equation, differentiable_empty_step
from lib.fsdp2_runtime import apply_fsdp2, make_device_mesh
from lib.model_factory import build_pretrained
from lib.pipeline_core import atomic_json, load_yaml


def seed_rank(seed, rank):
    random.seed(seed + rank); np.random.seed(seed + rank); torch.manual_seed(seed + rank)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed + rank)


def windows(path: Path, start: int, stop: int, width: int):
    window = []
    with path.open(encoding="utf-8") as handle:
        for position, line in enumerate(handle):
            if position < start: continue
            if position >= stop: break
            window.append((position, json.loads(line)))
            if len(window) == width:
                yield window; window = []
    if window: yield window


def answer_ranges(path):
    with Path(path).open(encoding="utf-8") as handle:
        return {row["id"]: row["answer_ranges"] for row in map(json.loads, handle)}


def masks_for(row, ranges):
    n = int(row["n_tokens"]); answer = torch.zeros(n - 1, dtype=torch.bool)
    for start, stop in ranges[row["id"]]:
        # A label at token j is predicted from hidden state j-1.
        answer[max(1, int(start)) - 1:max(1, int(stop)) - 1] = True
    context = ~answer
    if not answer.any() or (answer & context).any() or not torch.equal(answer | context, torch.ones_like(answer)):
        raise RuntimeError("STAGE2_MASK_CONTRACT")
    return answer, context


def local_rows(window, plan, rank, slot):
    return [(position, row) for position, row in window
            if plan.owner(position)[1:3] == (slot, rank)]


def count_window(window, plan, rank, stage, ranges, device):
    answer = context = tokens = 0
    for position, row in window:
        if plan.owner(position)[2] != rank: continue
        n = int(row["token_count"] if stage == 1 else row["n_tokens"])
        tokens += n
        if stage == 1: context += n - 1
        else:
            a, c = masks_for(row, ranges); answer += int(a.sum()); context += int(c.sum())
    counts = torch.tensor([answer, context, tokens], dtype=torch.int64, device=device)
    if dist.is_initialized(): dist.all_reduce(counts)
    return tuple(int(x) for x in counts.tolist())


def stage1_document_boundaries(row, token_count):
    raw = row.get("document_boundaries_qwen_json")
    if raw is None:
        raw = row.get("metadata", {}).get("document_boundaries_qwen_json")
    if raw is None:
        raw = [[0, token_count]]
    bounds = json.loads(raw) if isinstance(raw, str) else raw
    bounds = [(int(start), int(stop)) for start, stop in bounds]
    if not bounds or any(not 0 <= start < stop <= token_count for start, stop in bounds):
        raise RuntimeError("STAGE1_DOCUMENT_BOUNDARY_INVALID")
    if bounds[0][0] != 0 or bounds[-1][1] != token_count:
        raise RuntimeError("STAGE1_DOCUMENT_BOUNDARY_COVERAGE")
    if any(left[1] != right[0] for left, right in zip(bounds, bounds[1:])):
        raise RuntimeError("STAGE1_DOCUMENT_BOUNDARY_GAP_OR_OVERLAP")
    return bounds


def fused_sum(model, ids, selected=None, document_boundaries=None):
    try:
        from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
    except ImportError as error:
        raise RuntimeError("LIGER_FUSED_LOSS_REQUIRED_FOR_32K_TRAINING") from error
    boundaries = document_boundaries or [(0, ids.shape[1])]
    hidden_forward = getattr(model, "_fsdp_hidden_forward", model.model)
    hidden = hidden_forward(
        input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
        return_dict=True, document_boundaries=boundaries,
    ).last_hidden_state[:, :-1]
    labels = ids[:, 1:]
    if selected is not None:
        hidden = hidden[:, selected, :]; labels = labels[:, selected]
    loss_fn = LigerFusedLinearCrossEntropyLoss(reduction="sum", accum_dtype=torch.float32)
    return loss_fn(model.lm_head.weight, hidden.reshape(-1, hidden.shape[-1]), labels.reshape(-1))


def validate_stage1(model, tokenizer, path, limit, context):
    versions = {name: parameter._version for name, parameter in model.named_parameters()}
    numerator = torch.zeros((), dtype=torch.float64, device=context.device); denominator = 0
    model.eval()
    with torch.no_grad(), Path(path).open(encoding="utf-8") as handle:
        for position, line in enumerate(handle):
            if position >= limit: break
            if position % context.world_size != context.rank: continue
            row = json.loads(line)
            ids = tokenizer(row["text"], add_special_tokens=False, return_tensors="pt").input_ids.to(context.device)
            if ids.shape[1] != int(row["token_count"]): raise RuntimeError("VALIDATION_TOKEN_ALIGNMENT")
            bounds = stage1_document_boundaries(row, ids.shape[1])
            numerator += fused_sum(model, ids, document_boundaries=bounds).detach().double()
            denominator += ids.shape[1] - 1
    pair = torch.stack((numerator, torch.tensor(float(denominator), device=context.device, dtype=torch.float64)))
    if dist.is_initialized(): dist.all_reduce(pair)
    model.train()
    if any(parameter._version != versions[name] for name, parameter in model.named_parameters()):
        raise RuntimeError("VALIDATION_MUTATED_WEIGHTS")
    return {"records": limit, "global_mean_loss": float(pair[0] / pair[1]),
            "optimizer_steps": 0, "weights_unchanged": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    parser.add_argument("--config", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-from"); parser.add_argument("--parent-artifact")
    parser.add_argument("--max-records", type=int)
    args = parser.parse_args(); cfg = load_yaml(args.config); context = initialize()
    try:
        expected_world = int(cfg.get("world_size", context.world_size))
        if context.world_size != expected_world:
            raise RuntimeError(
                f"WORLD_SIZE_MISMATCH expected={expected_world} actual={context.world_size}"
            )
        allowlist = str(cfg.get("physical_gpu_allowlist", "2,3"))
        if os.environ.get("CUDA_VISIBLE_DEVICES") != allowlist:
            raise RuntimeError(
                f"PHYSICAL_GPU_ALLOWLIST_VIOLATION expected={allowlist} "
                f"actual={os.environ.get('CUDA_VISIBLE_DEVICES')}"
            )
        if context.device.type != "cuda" or torch.cuda.device_count() != expected_world:
            raise RuntimeError(
                f"CUDA_VISIBLE_DEVICE_COUNT_MISMATCH expected={expected_world} "
                f"actual={torch.cuda.device_count()}"
            )
        seed_rank(int(cfg.get("seed", 42)), context.rank)
        plan = BatchPlan(context.world_size, int(cfg["micro_batch_size"]), int(cfg["gradient_accumulation"]))
        assert_batch_equation(int(cfg["global_batch_size"]), plan.micro_batch, plan.dp_size,
                              plan.gradient_accumulation)
        output = Path(args.output_dir); rank0_mkdir(context, output)
        batch_metadata = {"global_batch_size": plan.global_batch, "micro_batch_size": plan.micro_batch,
                          "gradient_accumulation": plan.gradient_accumulation}
        if args.resume_from:
            preflight_checkpoint_metadata(
                args.resume_from, context.world_size, [context.world_size], batch_metadata
            )
        model, _ = build_pretrained(
            cfg["model_base"],
            context.device,
            train_clip_none=True,
            ttt_layers=list(cfg["ttt_layers"]),
            ttt_chunk=int(cfg["ttt_chunk_size"]),
        )
        mesh = make_device_mesh(context.device.type, context.world_size)
        inventory = apply_fsdp2(model, mesh=mesh)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["outer_lr"]),
            betas=tuple(cfg["betas"]), eps=float(cfg["eps"]), weight_decay=float(cfg["weight_decay"]),
            fused=context.device.type == "cuda")
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        checkpoint_root = Path(args.resume_from) if args.resume_from else output / "checkpoints"
        rotation = DistributedCheckpointRotation(checkpoint_root, context.world_size,
                                                  [context.world_size], batch_metadata)
        resumed = rotation.load(model, optimizer, scheduler)
        if args.stage == 2 and resumed is None:
            if not args.parent_artifact: raise RuntimeError("STAGE1_MODEL_ONLY_HANDOFF_REQUIRED")
            load_model_only(model, args.parent_artifact)
            # Stage 2 is a new correctness extension: fresh AdamW/scheduler/step/RNG.
            seed_rank(int(cfg.get("seed", 42)), context.rank)
        cursor = int(resumed["record_cursor"]) if resumed else 0
        cumulative_tokens = int(resumed["cumulative_tokens"]) if resumed else 0
        update_step = int(resumed["update_step"]) if resumed else 0
        final_cursor = int(cfg["record_cursor_final"])
        stop = min(final_cursor, cursor + args.max_records) if args.max_records else final_cursor
        tokenizer = AutoTokenizer.from_pretrained(cfg["model_base"], local_files_only=True)
        ranges = answer_ranges(cfg["answer_mask_sidecar"]) if args.stage == 2 else None
        for window in windows(Path(cfg["train_data"]), cursor, stop, plan.global_batch):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            step_started = time.perf_counter()
            local_numerator = torch.zeros((), dtype=torch.float64, device=context.device)
            local_sample_ids = []
            global_answer, global_context, window_tokens = count_window(
                window, plan, context.rank, args.stage, ranges, context.device)
            optimizer.zero_grad(set_to_none=True)
            for slot in range(plan.gradient_accumulation):
                if hasattr(model, "set_requires_gradient_sync"):
                    model.set_requires_gradient_sync(slot == plan.gradient_accumulation - 1)
                rows = local_rows(window, plan, context.rank, slot)
                loss = None
                for _, row in rows:
                    local_sample_ids.append(row.get("sample_id", row.get("id")))
                    ids = tokenizer(row["text"], add_special_tokens=False, return_tensors="pt").input_ids.to(context.device)
                    declared = int(row["token_count"] if args.stage == 1 else row["n_tokens"])
                    if ids.shape[1] != declared or declared > int(cfg["context_length"]):
                        raise RuntimeError("TOKEN_ALIGNMENT_ERROR")
                    if args.stage == 1:
                        bounds = stage1_document_boundaries(row, declared)
                        raw_numerator = fused_sum(model, ids, document_boundaries=bounds)
                        local_numerator += raw_numerator.detach().double()
                        term = raw_numerator * (
                            context.world_size / global_context
                        )
                    else:
                        a, c = masks_for(row, ranges); a = a.to(context.device); c = c.to(context.device)
                        term = fused_sum(model, ids, a) * (context.world_size / global_answer)
                        term = term + float(cfg["context_group_weight"]) * fused_sum(model, ids, c) * (context.world_size / global_context)
                    loss = term if loss is None else loss + term
                if loss is None: loss = differentiable_empty_step(model.parameters(), context.device)
                loss.backward()
            global_numerator = local_numerator.clone()
            if args.stage == 1:
                dist.all_reduce(global_numerator, op=dist.ReduceOp.SUM)
            global_mean_loss = float(global_numerator.item() / global_context) if args.stage == 1 else None
            if args.stage == 1 and not math.isfinite(global_mean_loss):
                raise RuntimeError("NONFINITE_GLOBAL_LOSS")
            l30_parameters = [
                (name, parameter) for name, parameter in model.named_parameters()
                if ".layers.30.mlp.ttt_" in name
            ]
            if not l30_parameters:
                raise RuntimeError("L30_TTT_PARAMETERS_MISSING_AFTER_FSDP2")
            l30_gradient = {}
            for name, parameter in l30_parameters:
                grad = parameter.grad
                expects_gradient = not name.endswith("ttt_conv.weight")
                if grad is None:
                    if expects_gradient:
                        raise RuntimeError(f"L30_TTT_GRADIENT_MISSING {name}")
                    l30_gradient[name] = {
                        "present": False,
                        "expected_present": False,
                        "finite": True,
                        "local_norm": 0.0,
                        "reason": "gated_next_position_hidden_does_not_read_conv_kernel",
                    }
                    continue
                local_grad = grad.to_local() if hasattr(grad, "to_local") else grad
                grad_finite = bool(torch.isfinite(local_grad).all().item())
                if not grad_finite:
                    raise RuntimeError(f"L30_TTT_GRADIENT_NONFINITE {name}")
                l30_gradient[name] = {
                    "present": True,
                    "expected_present": expects_gradient,
                    "finite": True,
                    "local_norm": float(torch.linalg.vector_norm(local_grad.float()).item()),
                }
            grad_norm = clip_grad_norm_fsdp2(model.parameters(), float(cfg["outer_grad_clip"]))
            grad_clip = float(cfg["outer_grad_clip"])
            clipped_grad_norm = min(grad_norm, grad_clip)
            optimizer.step(); scheduler.step(); update_step += 1
            torch.cuda.synchronize()
            step_seconds = time.perf_counter() - step_started
            peak_allocated_bytes = torch.cuda.max_memory_allocated()
            peak_reserved_bytes = torch.cuda.max_memory_reserved()
            optimizer_parameter_ids = {
                id(parameter) for group in optimizer.param_groups for parameter in group["params"]
            }
            l30_optimizer_state = {}
            for name, parameter in l30_parameters:
                owned = id(parameter) in optimizer_parameter_ids
                if not owned:
                    raise RuntimeError(f"L30_OPTIMIZER_OWNERSHIP_MISSING {name}")
                state = optimizer.state.get(parameter, {})
                expects_state = l30_gradient[name]["present"]
                state_keys = sorted(state.keys())
                if expects_state and not {"step", "exp_avg", "exp_avg_sq"}.issubset(state):
                    raise RuntimeError(f"L30_OPTIMIZER_STATE_MISSING {name}")
                l30_optimizer_state[name] = {
                    "owned": owned,
                    "state_expected": expects_state,
                    "state_keys": state_keys,
                }
            weights_finite = True
            for parameter in model.parameters():
                local_parameter = parameter.to_local() if hasattr(parameter, "to_local") else parameter
                if not bool(torch.isfinite(local_parameter).all().item()):
                    weights_finite = False
                    break
            if not weights_finite:
                raise RuntimeError("NONFINITE_MODEL_WEIGHT_AFTER_STEP")
            local_runtime = {
                "rank": context.rank,
                "local_rank": context.local_rank,
                "logical_device": str(context.device),
                "physical_gpu": 2 + context.local_rank,
                "sample_ids": local_sample_ids,
                "local_numerator": float(local_numerator.item()),
                "step_seconds": step_seconds,
                "peak_allocated_bytes": peak_allocated_bytes,
                "peak_reserved_bytes": peak_reserved_bytes,
                "l30_gradient": l30_gradient,
                "l30_optimizer_state": l30_optimizer_state,
                "weights_finite": weights_finite,
            }
            gathered_runtime = [None] * context.world_size
            dist.all_gather_object(gathered_runtime, local_runtime)
            cursor = window[-1][0] + 1; cumulative_tokens += window_tokens
            local_valid = sum(int(row["token_count"] if args.stage == 1 else row["n_tokens"]) - 1
                              for position, row in window if plan.owner(position)[2] == context.rank)
            global_valid = global_context if args.stage == 1 else global_answer + global_context
            progress = {"stage": args.stage, "record_cursor": cursor,
                        "cumulative_tokens": cumulative_tokens, "global_valid_tokens": global_valid,
                        "local_valid_tokens": local_valid,
                        "update_step": update_step, "accumulation_phase": 0,
                        "dataloader_state": {"canonical_next_position": cursor},
                        "rank_assignment_state": {"rank": context.rank, "world_size": context.world_size,
                                                  "mapping": "canonical_modulo_with_accumulation_v1"}}
            save_points = set(int(x) for x in cfg.get("save_steps", []))
            if update_step in save_points or cursor == stop:
                rotation.save(model, optimizer, progress, scheduler)
            if context.is_rank0:
                with (output / "training.jsonl").open("a", encoding="utf-8") as handle:
                    record = {
                        **progress,
                        "global_answer_positions": global_answer,
                        "global_context_positions": global_context,
                        "global_numerator": float(global_numerator.item()),
                        "global_mean_loss": global_mean_loss,
                        "grad_norm_before_clip": grad_norm,
                        "grad_norm_after_clip": clipped_grad_norm,
                        "learning_rate": scheduler.get_last_lr()[0],
                        "rank_runtime": gathered_runtime,
                    }
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
        if cursor != stop: raise RuntimeError("CURSOR_STOP_MISMATCH")
        if args.stage == 1 and cursor == final_cursor and cfg.get("stage1_model_only_artifact"):
            validation = validate_stage1(model, tokenizer, cfg["validation_data"],
                                         int(cfg["validation_records"]), context)
            artifact = Path(cfg["stage1_model_only_artifact"])
            if not artifact.is_absolute(): artifact = ROOT / artifact
            save_model_only(model, artifact,
                            {"source_stage": 1, "record_cursor": cursor, "ttt_lr": 1.0,
                             "validation": validation, "immutable_parent": True})
        if context.is_rank0:
            atomic_json(output / "complete.json", {"complete": True, "stage": args.stage,
                "record_cursor": cursor, "cumulative_tokens": cumulative_tokens,
                "fsdp2_wrapped_blocks": inventory.decoder_blocks})
        if context.is_rank0: print("FINAL_BARRIER_ENTER", flush=True)
        barrier()
        if context.is_rank0: print("FINAL_BARRIER_PASS", flush=True)
    finally:
        shutdown()
        print(f"RANK_{context.rank}_SHUTDOWN_PASS", flush=True)


if __name__ == "__main__": main()
