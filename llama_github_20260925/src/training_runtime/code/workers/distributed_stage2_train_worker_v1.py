#!/usr/bin/env python3
"""Selective FSDP2 restoration. This does not replace the audited 1-GPU worker."""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

# Fail before importing CUDA/JIT libraries when an implicit cache would escape.
_CACHE_ROOT = Path("/path/to/ttt")
for _key in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "TORCH_EXTENSIONS_DIR", "CUDA_CACHE_PATH", "TMPDIR"):
    _value = os.environ.get(_key)
    if not _value or not Path(_value).resolve().is_relative_to(_CACHE_ROOT):
        raise RuntimeError(f"RUNTIME_CACHE_ROOT_POLICY {_key}={_value}")
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
    """Current canonical loader, extended to the frozen ordered shard inventory."""
    path = Path(path)
    paths = [path]
    if path.is_dir():
        manifest = path.parent / "manifests/FINAL_DATA_FILE_HASHES.jsonl"
        entries = [json.loads(line) for line in manifest.read_text().splitlines()]
        paths = [path.parent / row["path"] for row in entries
                 if Path(row["path"]).parent.as_posix() == path.name]
        if not paths or paths != sorted(paths):
            raise RuntimeError("STAGE1_SHARD_ORDER_INVALID")
    window = []; position = 0
    for shard in paths:
        with shard.open(encoding="utf-8") as handle:
            for line in handle:
                if position >= stop:
                    if window: yield window
                    return
                if position >= start:
                    window.append((position, json.loads(line)))
                    if len(window) == width:
                        yield window; window = []
                position += 1
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


def stage1_document_boundaries(row, token_count, input_ids=None):
    raw = row.get("document_boundaries_qwen_json")
    if raw is None:
        raw = row.get("metadata", {}).get("document_boundaries_qwen_json")
    if raw is None:
        raise RuntimeError("STAGE1_REQUIRED_DOCUMENT_METADATA_MISSING")
    bounds = json.loads(raw) if isinstance(raw, str) else raw
    bounds = [(int(start), int(stop)) for start, stop in bounds]
    if not bounds or any(not 0 <= start < stop <= token_count for start, stop in bounds):
        raise RuntimeError("STAGE1_DOCUMENT_BOUNDARY_INVALID")
    if bounds[0][0] != 0 or bounds[-1][1] != token_count:
        raise RuntimeError("STAGE1_DOCUMENT_BOUNDARY_COVERAGE")
    for left, right in zip(bounds, bounds[1:]):
        if left[1] > right[0]:
            raise RuntimeError("STAGE1_DOCUMENT_BOUNDARY_OVERLAP")
        if left[1] < right[0]:
            if input_ids is None:
                raise RuntimeError("STAGE1_EOS_GAP_REQUIRES_TOKEN_VERIFICATION")
            gap = input_ids.reshape(-1)[left[1]:right[0]]
            if gap.numel() != 1 or int(gap.item()) != 151643:
                raise RuntimeError("STAGE1_NON_EOS_DOCUMENT_GAP")
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
    ).last_hidden_state
    if ids.shape[1] == 1:
        # One real input token has no shifted LM target, but must traverse the
        # same FSDP forward/backward collectives and remain in token accounting.
        return hidden.sum() * 0.0 + model.lm_head.weight.reshape(-1)[:1].sum() * 0.0
    hidden = hidden[:, :-1]
    labels = ids[:, 1:]
    if selected is not None:
        hidden = hidden[:, selected, :]; labels = labels[:, selected]
    loss_fn = LigerFusedLinearCrossEntropyLoss(reduction="sum", accum_dtype=torch.float32)
    return loss_fn(model.lm_head.weight, hidden.reshape(-1, hidden.shape[-1]), labels.reshape(-1))


def exact_sharded_parameter_fingerprint(model):
    """Normalize public FSDP representations, then hash exact owned bytes."""
    from torch.distributed.fsdp import FSDPModule
    for module in reversed(list(model.modules())):
        if isinstance(module, FSDPModule):
            module.reshard()
    torch.cuda.synchronize()
    result = {}
    for name, parameter in model.named_parameters():
        value = parameter.to_local() if hasattr(parameter, "to_local") else parameter
        cpu = value.detach().cpu().contiguous()
        result[name] = {
            "shape": list(cpu.shape), "dtype": str(cpu.dtype),
            "sha256": hashlib.sha256(cpu.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
        }
    return result


def validate_stage1(model, tokenizer, path, limit, context):
    before_fingerprint = exact_sharded_parameter_fingerprint(model)
    numerator = torch.zeros((), dtype=torch.float64, device=context.device); denominator = 0
    model.eval()
    torch.cuda.synchronize()
    validation_started = time.perf_counter()
    observed_records = 0
    with torch.no_grad():
        for window in windows(Path(path), 0, limit, context.world_size):
            rows = [(position, row) for position, row in window if position % context.world_size == context.rank]
            observed_records = window[-1][0] + 1
            owned = bool(rows)
            position, row = rows[0] if owned else window[0]
            ids = tokenizer(row["text"], add_special_tokens=False, return_tensors="pt").input_ids.to(context.device)
            if ids.shape[1] != int(row["token_count"]): raise RuntimeError("VALIDATION_TOKEN_ALIGNMENT")
            bounds = stage1_document_boundaries(row, ids.shape[1], ids)
            value = fused_sum(model, ids, document_boundaries=bounds).detach().double()
            numerator += value if owned else value * 0.0
            denominator += (ids.shape[1] - 1) if owned else 0
    if observed_records != limit:
        raise RuntimeError("VALIDATION_RECORD_COUNT_MISMATCH")
    pair = torch.stack((numerator, torch.tensor(float(denominator), device=context.device, dtype=torch.float64)))
    if dist.is_initialized(): dist.all_reduce(pair)
    if not bool(torch.isfinite(pair).all()) or float(pair[1]) <= 0:
        raise RuntimeError("VALIDATION_NONFINITE_OR_ZERO_SUPERVISION")
    if not math.isfinite(float(pair[0] / pair[1])):
        raise RuntimeError("VALIDATION_NONFINITE_MEAN_LOSS")
    model.train()
    after_fingerprint = exact_sharded_parameter_fingerprint(model)
    changed = sorted(name for name in set(before_fingerprint) | set(after_fingerprint)
                     if before_fingerprint.get(name) != after_fingerprint.get(name))
    changed_by_rank = [None] * context.world_size
    dist.all_gather_object(changed_by_rank, changed)
    if any(changed_by_rank):
        raise RuntimeError(f"VALIDATION_MUTATED_WEIGHT_BYTES {changed_by_rank}")
    torch.cuda.synchronize()
    return {"records": limit, "global_mean_loss": float(pair[0] / pair[1]),
            "wall_seconds": time.perf_counter()-validation_started,
            "optimizer_steps": 0, "weights_unchanged": True,
            "weights_unchanged_method": "exact_sha256_local_shards_after_recursive_reshard",
            "verified_parameter_shards_per_rank": len(before_fingerprint)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    parser.add_argument("--config", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-from"); parser.add_argument("--parent-artifact")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--debug-forward-only", action="store_true")
    parser.add_argument("--debug-fail-after-save-step", type=int)
    parser.add_argument("--debug-start-record", type=int, default=0)
    args = parser.parse_args(); cfg = load_yaml(args.config)
    # Reject unauthorized visibility before initialize() can create contexts.
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3":
        raise RuntimeError("PREINIT_PHYSICAL_GPU_ALLOWLIST_VIOLATION")
    if int(os.environ.get("WORLD_SIZE", "0")) != 2 or int(os.environ.get("LOCAL_RANK", "-1")) not in (0, 1):
        raise RuntimeError("PREINIT_TWO_RANK_GPU_CONTRACT")
    for _path in [args.output_dir, args.resume_from, cfg.get("stage1_model_only_artifact")]:
        if _path and not Path(_path).resolve().is_relative_to(_CACHE_ROOT):
            raise RuntimeError("RUNTIME_OUTPUT_ROOT_POLICY")
    if args.stage != 2:
        raise RuntimeError("STAGE2_CANDIDATE_ONLY")
    lineage_path = _CACHE_ROOT / "audits/STAGE2_LOSS_LINEAGE_AUDIT.json"
    if not lineage_path.is_file() or json.loads(lineage_path.read_text()).get("CORRECTED_STAGE2_LOSS_LINEAGE") != "PASS":
        raise RuntimeError("CORRECTED_STAGE2_LOSS_LINEAGE_NOT_PROVEN_PRECUDA")
    context = initialize()
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
        batch_metadata["identity"] = {
            **cfg.get("input_identity", {}),
            "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        }
        if args.debug_fail_after_save_step and cfg.get("formal_training"):
            raise RuntimeError("FAILURE_INJECTION_DEBUG_ONLY")
        declared_identity = cfg.get("input_identity", {})
        if declared_identity.get("worker_sha256") != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
            raise RuntimeError("CURRENT_WORKER_IDENTITY_MISMATCH")
        dataset_root = Path(cfg["train_data"]).parent if args.stage == 1 else Path(cfg["train_data"]).parent.parent
        dataset_checks = (("dataset_authority_sha256", "authority/DATASET_AUTHORITY_V3.json"),
                          ("data_manifest_sha256", "manifests/FINAL_DATA_FILE_HASHES.jsonl")) if args.stage == 1 else (
                          ("dataset_authority_sha256", "authority/STAGE2_DATASET_AUTHORITY.json"),
                          ("data_manifest_sha256", "manifests/FINAL_PACKED_RECORDS.jsonl"),
                          ("content_order_sha256", "manifests/FINAL_30M_QA_CONTENT_ORDER.jsonl"))
        for key, relative in dataset_checks:
            if hashlib.sha256((dataset_root / relative).read_bytes()).hexdigest() != declared_identity.get(key):
                raise RuntimeError("CURRENT_DATASET_IDENTITY_MISMATCH")
        if args.resume_from:
            if not (Path(args.resume_from) / "latest.json").is_file():
                raise RuntimeError("EXPLICIT_RESUME_CHECKPOINT_MISSING")
            preflight_checkpoint_metadata(
                args.resume_from, context.world_size, [context.world_size], batch_metadata
            )
        if context.is_rank0 and (output / "complete.json").exists():
            os.replace(output / "complete.json", output / f"complete.previous.{time.time_ns()}.json")
        barrier()
        model, _ = build_pretrained(
            cfg["model_base"],
            context.device,
            train_clip_none=True,
            ttt_layers=list(cfg["ttt_layers"]),
            ttt_chunk=int(cfg["ttt_chunk_size"]),
        )
        parameter_count = sum(p.numel() for p in model.parameters())
        if parameter_count != 4061881856 or len(model.model.layers) != 36:
            raise RuntimeError("QWEN3_4B_TTT_MODEL_PARAMETER_IDENTITY")
        mesh = make_device_mesh(context.device.type, context.world_size)
        inventory = apply_fsdp2(model, mesh=mesh)
        if inventory.decoder_blocks != 36 or not all(hasattr(block, "set_requires_gradient_sync") for block in model.model.layers):
            raise RuntimeError("FORMAL_FSDP2_OWNERSHIP")
        if context.is_rank0:
            atomic_json(output / "model_build.json", {"parameter_count":parameter_count,"layers":36,
                        "ttt_layers":cfg["ttt_layers"],"fsdp2_blocks":inventory.decoder_blocks})
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
        if args.debug_start_record and (cfg.get("formal_training") or args.resume_from):
            raise RuntimeError("DEBUG_START_FORBIDDEN_FORMAL_OR_RESUME")
        if args.debug_start_record % plan.global_batch:
            raise RuntimeError("DEBUG_START_MUST_ALIGN_UPDATE_WINDOW")
        cursor = int(resumed["record_cursor"]) if resumed else args.debug_start_record
        cumulative_tokens = int(resumed["cumulative_tokens"]) if resumed else 0
        update_step = int(resumed["update_step"]) if resumed else 0
        cumulative_answer_positions = int(resumed.get("cumulative_answer_positions",0)) if resumed else 0
        cumulative_context_positions = int(resumed.get("cumulative_context_positions",0)) if resumed else 0
        final_cursor = int(cfg["record_cursor_final"])
        stop = min(final_cursor, cursor + args.max_records) if args.max_records else final_cursor
        if stop != final_cursor and stop % plan.global_batch:
            raise RuntimeError("NONTERMINAL_DEBUG_STOP_MUST_ALIGN_OPTIMIZER_WINDOW")
        if cursor % plan.global_batch and cursor != final_cursor:
            raise RuntimeError("NONTERMINAL_RESUME_CURSOR_MUST_ALIGN_OPTIMIZER_WINDOW")
        tokenizer = AutoTokenizer.from_pretrained(cfg["model_base"], local_files_only=True)
        ranges = answer_ranges(cfg["answer_mask_sidecar"]) if args.stage == 2 else None
        iterator = iter(windows(Path(cfg["train_data"]), cursor, stop, plan.global_batch))
        while True:
            iteration_started = time.perf_counter()
            try: window = next(iterator)
            except StopIteration: break
            loader_wait_seconds = time.perf_counter() - iteration_started
            tokenizer_seconds = 0.0
            local_boundary_trace = []
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            step_started = time.perf_counter()
            local_numerator = torch.zeros((), dtype=torch.float64, device=context.device)
            local_sample_ids = []
            global_answer, global_context, window_tokens = count_window(
                window, plan, context.rank, args.stage, ranges, context.device)
            if args.stage == 1 and global_context <= 0:
                raise RuntimeError("ZERO_SUPERVISION_WINDOW_REJECTED_BEFORE_FORWARD")
            optimizer.zero_grad(set_to_none=True)
            for slot in range(plan.gradient_accumulation):
                if hasattr(model, "set_requires_gradient_sync"):
                    model.set_requires_gradient_sync(slot == plan.gradient_accumulation - 1)
                rows = local_rows(window, plan, context.rank, slot)
                loss = None
                for _, row in rows:
                    local_sample_ids.append(row.get("sample_id", row.get("id")))
                    tokenization_started = time.perf_counter()
                    ids = tokenizer(row["text"], add_special_tokens=False, return_tensors="pt").input_ids.to(context.device)
                    tokenizer_seconds += time.perf_counter() - tokenization_started
                    declared = int(row["token_count"] if args.stage == 1 else row["n_tokens"])
                    if ids.shape[1] != declared or declared > int(cfg["context_length"]):
                        raise RuntimeError("TOKEN_ALIGNMENT_ERROR")
                    if args.stage == 1:
                        bounds = stage1_document_boundaries(row, declared, ids)
                        raw_numerator = fused_sum(model, ids, document_boundaries=bounds)
                        trace = {}
                        for layer_id in cfg["ttt_layers"]:
                            groups = model.model.layers[layer_id].mlp.last_ttt_stats
                            expected = [(b-a+int(cfg["ttt_chunk_size"])-1)//int(cfg["ttt_chunk_size"]) for a,b in bounds]
                            actual = [len(group) for group in groups]
                            if actual != expected:
                                raise RuntimeError(f"REAL_DOCUMENT_CHUNK_SCHEDULE_MISMATCH layer={layer_id}")
                            pair_counts = [sum(int(stat.valid_tokens) for stat in group) for group in groups]
                            expected_pairs = [b-a-chunks for (a,b),chunks in zip(bounds,expected)]
                            if pair_counts != expected_pairs:
                                raise RuntimeError("REAL_DOCUMENT_NTP_PAIR_COUNT_MISMATCH")
                            trace[str(layer_id)] = {"chunks_per_document":actual,"ntp_pairs_per_document":pair_counts}
                        local_boundary_trace.append({"sample_id":row["sample_id"],"boundaries":bounds,"layers":trace})
                        local_numerator += raw_numerator.detach().double()
                        term = raw_numerator * (
                            context.world_size / global_context
                        )
                    else:
                        a, c = masks_for(row, ranges); a = a.to(context.device); c = c.to(context.device)
                        term = fused_sum(model, ids, a) * (context.world_size / global_answer)
                        term = term + float(cfg["context_group_weight"]) * fused_sum(model, ids, c) * (context.world_size / global_context)
                        local_numerator += term.detach().double()
                    loss = term if loss is None else loss + term
                if loss is None:
                    # A missing final rank slot still participates in all FSDP
                    # collectives via a real-record zero contribution, not a new
                    # sample, token, or TTT state carried across documents.
                    dummy = window[0][1]
                    dummy_ids = tokenizer(dummy["text"], add_special_tokens=False, return_tensors="pt").input_ids.to(context.device)
                    dummy_bounds = (stage1_document_boundaries(dummy, int(dummy["token_count"]), dummy_ids)
                                    if args.stage == 1 else [(0, int(dummy["n_tokens"]))])
                    if args.stage == 2:
                        dummy_a,dummy_c = masks_for(dummy,ranges)
                        loss = fused_sum(model,dummy_ids,dummy_a.to(context.device)) * 0.0
                        loss = loss + fused_sum(model,dummy_ids,dummy_c.to(context.device)) * 0.0
                    else:
                        loss = fused_sum(model, dummy_ids, document_boundaries=dummy_bounds) * 0.0
                if args.debug_forward_only:
                    if cfg.get("formal_training"): raise RuntimeError("DEBUG_FORWARD_ONLY_NOT_FORMAL")
                    torch.cuda.synchronize()
                    forward_receipt = {"rank":context.rank,"local_sample_ids":local_sample_ids,
                        "local_loss_scaled":float(loss.detach()),"global_answer_positions":global_answer,
                        "global_context_positions":global_context,"window_tokens":window_tokens,
                        "peak_memory":torch.cuda.max_memory_reserved(),"optimizer_steps":0,"DEBUG_ONLY":True}
                    forward_rows = [None]*context.world_size
                    dist.all_gather_object(forward_rows,forward_receipt)
                    if context.is_rank0: atomic_json(output/"forward_only.json",{"status":"PASS","ranks":forward_rows})
                    return
                loss.backward()
            global_numerator = local_numerator.clone()
            dist.all_reduce(global_numerator, op=dist.ReduceOp.SUM)
            global_mean_loss = float(global_numerator.item() / global_context) if args.stage == 1 else float(global_numerator.item()/context.world_size)
            if args.stage == 1 and not math.isfinite(global_mean_loss):
                raise RuntimeError("NONFINITE_GLOBAL_LOSS")
            l30_parameters = [
                (name, parameter) for name, parameter in model.named_parameters()
                if any(f".layers.{layer_id}.mlp.ttt_" in name for layer_id in cfg["ttt_layers"])
            ]
            if not l30_parameters:
                raise RuntimeError("L30_TTT_PARAMETERS_MISSING_AFTER_FSDP2")
            eligible_ttt_update = any(any(int(b)-int(a) > int(cfg["ttt_chunk_size"])
                for a,b in ((json.loads(row["document_boundaries_qwen_json"])
                            if isinstance(row["document_boundaries_qwen_json"], str)
                            else row["document_boundaries_qwen_json"])
                            if args.stage == 1 else [(0, int(row["n_tokens"]))])) for _,row in window)
            l30_gradient = {}
            for name, parameter in l30_parameters:
                grad = parameter.grad
                expects_gradient = eligible_ttt_update and not name.endswith("ttt_conv.weight")
                if grad is None:
                    if expects_gradient:
                        raise RuntimeError(f"L30_TTT_GRADIENT_MISSING {name}")
                    l30_gradient[name] = {
                        "present": False,
                        "expected_present": False,
                        "finite": True,
                        "local_norm": 0.0,
                        "reason": "conv_unused_or_no_later_chunk_reads_updated_fast_weight",
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
            gradient_coverage = {"trainable":0,"present":0,"missing":[],"nonfinite":[]}
            for pname,parameter in model.named_parameters():
                if not parameter.requires_grad: continue
                gradient_coverage["trainable"] += 1
                if parameter.grad is None: gradient_coverage["missing"].append(pname);continue
                gradient_coverage["present"] += 1
                pg=parameter.grad.to_local() if hasattr(parameter.grad,"to_local") else parameter.grad
                if not bool(torch.isfinite(pg).all()): gradient_coverage["nonfinite"].append(pname)
            if gradient_coverage["nonfinite"]: raise RuntimeError("STAGE2_NONFINITE_GRADIENT")
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
                "boundary_trace": local_boundary_trace,
                "local_tokens": sum(int(row["token_count"] if args.stage == 1 else row["n_tokens"]) for pos,row in window if plan.owner(pos)[2] == context.rank),
                "loader_wait_seconds": loader_wait_seconds,
                "tokenizer_seconds": tokenizer_seconds,
                "local_numerator": float(local_numerator.item()),
                "step_seconds": step_seconds,
                "peak_allocated_bytes": peak_allocated_bytes,
                "peak_reserved_bytes": peak_reserved_bytes,
                "gradient_coverage": gradient_coverage,
                "l30_gradient": l30_gradient,
                "l30_optimizer_state": l30_optimizer_state,
                "weights_finite": weights_finite,
            }
            gathered_runtime = [None] * context.world_size
            dist.all_gather_object(gathered_runtime, local_runtime)
            cursor = window[-1][0] + 1; cumulative_tokens += window_tokens
            cumulative_answer_positions += global_answer
            cumulative_context_positions += global_context
            local_valid = sum(int(row["token_count"] if args.stage == 1 else row["n_tokens"]) - 1
                              for position, row in window if plan.owner(position)[2] == context.rank)
            global_valid = global_context if args.stage == 1 else global_answer + global_context
            progress = {"stage": args.stage, "record_cursor": cursor,
                        "cumulative_tokens": cumulative_tokens, "global_valid_tokens": global_valid,
                        "cumulative_answer_positions":cumulative_answer_positions,
                        "cumulative_context_positions":cumulative_context_positions,
                        "local_valid_tokens": local_valid,
                        "update_step": update_step, "accumulation_phase": 0,
                        "dataloader_state": {"canonical_next_position": cursor},
                        "input_identity": batch_metadata["identity"],
                        "rank_assignment_state": {"rank": context.rank, "world_size": context.world_size,
                                                  "mapping": "canonical_modulo_with_accumulation_v1"}}
            save_points = set(int(x) for x in cfg.get("save_steps", []))
            checkpoint_seconds = 0.0
            if update_step in save_points or cursor == stop:
                checkpoint_started = time.perf_counter()
                rotation.save(model, optimizer, progress, scheduler)
                checkpoint_seconds = time.perf_counter() - checkpoint_started
                if args.debug_fail_after_save_step == update_step:
                    print(f"CONTROLLED_SOFTWARE_FAILURE_AFTER_VALID_DCP rank={context.rank} step={update_step}", flush=True)
                    if context.rank == 1: raise RuntimeError("CONTROLLED_DEBUG_SOFTWARE_FAILURE")
                    dist.barrier()
            torch.cuda.synchronize()
            iteration_seconds = time.perf_counter() - iteration_started
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
                        "iteration_seconds_excluding_checkpoint": iteration_seconds-checkpoint_seconds,
                        "checkpoint_seconds": checkpoint_seconds,
                        "logged_at_unix": time.time(),
                        "rank_runtime": gathered_runtime,
                    }
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
        if cursor != stop: raise RuntimeError("CURSOR_STOP_MISMATCH")
        if cfg.get("formal_training") and cursor == final_cursor and cumulative_tokens != int(cfg["train_tokens"]):
            raise RuntimeError("FORMAL_FINAL_INPUT_TOKEN_TOTAL_MISMATCH")
        if args.stage == 1 and cfg.get("debug_validation_records"):
            validation = validate_stage1(model, tokenizer, cfg["validation_data"],
                                        int(cfg["debug_validation_records"]), context)
            if context.is_rank0: atomic_json(output / "validation_smoke.json", validation)
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
