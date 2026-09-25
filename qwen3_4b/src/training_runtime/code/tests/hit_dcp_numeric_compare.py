#!/usr/bin/env python3
"""Numerically compare two real two-rank DCP checkpoints tensor by tensor."""
import hashlib
import json
import os
import sys
from pathlib import Path

if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3":
    raise RuntimeError("PHYSICAL_GPU_ALLOWLIST_VIOLATION")

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))
from lib.distributed_checkpoint import DistributedCheckpointRotation
from lib.distributed_runtime import barrier, initialize, shutdown
from lib.fsdp2_runtime import apply_fsdp2, make_device_mesh
from lib.model_factory import build_pretrained

MODEL_PATH = Path(os.path.expandvars("${TTT_MODEL_ROOT}/Qwen3-4B-Base"))
A_ROOT = ROOT / "runs/l4_uninterrupted_v1/checkpoints"
B_ROOT = ROOT / "runs/l4_exact_a_step1_snapshot/checkpoints"
AUDIT = ROOT / "audits/hit_l3_l4_v1/L4_NUMERIC_STATE_COMPARE.json"
TTT_LAYERS = [0, 6, 12, 18, 24, 30]
BATCH = {"global_batch_size": 2, "micro_batch_size": 1, "gradient_accumulation": 1}


def local_cpu(value):
    value = value.to_local() if hasattr(value, "to_local") else value
    return value.detach().cpu().clone().contiguous()


def tensor_sha256(value):
    value = value.reshape(-1).view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def capture(model, optimizer):
    named = sorted(model.named_parameters(), key=lambda item: item[0])
    model_state = {name: local_cpu(parameter) for name, parameter in named}
    parameter_names = {id(parameter): name for name, parameter in named}
    optimizer_state = {}
    for parameter, state in optimizer.state.items():
        name = parameter_names[id(parameter)]
        for key, value in state.items():
            if torch.is_tensor(value) or hasattr(value, "to_local"):
                optimizer_state[f"{name}.{key}"] = local_cpu(value)
    return model_state, optimizer_state


def current_maps(model, optimizer):
    named = sorted(model.named_parameters(), key=lambda item: item[0])
    model_state = {name: parameter for name, parameter in named}
    parameter_names = {id(parameter): name for name, parameter in named}
    optimizer_state = {}
    for parameter, state in optimizer.state.items():
        name = parameter_names[id(parameter)]
        for key, value in state.items():
            if torch.is_tensor(value) or hasattr(value, "to_local"):
                optimizer_state[f"{name}.{key}"] = value
    return model_state, optimizer_state


def compare(reference, current, rtol, atol):
    if set(reference) != set(current):
        return {"key_sets_equal": False, "missing": sorted(set(reference) - set(current))[:8],
                "extra": sorted(set(current) - set(reference))[:8], "allclose": False}
    exact_count = allclose_count = 0
    max_abs = sum_sq_diff = sum_sq_reference = 0.0
    total_numel = 0
    worst_key = None
    digest_reference = hashlib.sha256()
    digest_current = hashlib.sha256()
    for name in sorted(reference):
        left = reference[name]
        right = local_cpu(current[name])
        if left.shape != right.shape or left.dtype != right.dtype:
            return {"key_sets_equal": True, "shape_dtype_equal": False,
                    "mismatch_key": name, "allclose": False}
        digest_reference.update(name.encode()); digest_reference.update(tensor_sha256(left).encode())
        digest_current.update(name.encode()); digest_current.update(tensor_sha256(right).encode())
        exact = torch.equal(left, right)
        close = torch.allclose(left, right, rtol=rtol, atol=atol, equal_nan=False)
        exact_count += int(exact); allclose_count += int(close)
        difference = left.float() - right.float()
        item_max = float(difference.abs().max().item()) if difference.numel() else 0.0
        if item_max > max_abs:
            max_abs = item_max; worst_key = name
        sum_sq_diff += float(difference.double().pow(2).sum().item())
        sum_sq_reference += float(left.double().pow(2).sum().item())
        total_numel += left.numel()
    rms_diff = (sum_sq_diff / max(1, total_numel)) ** 0.5
    reference_rms = (sum_sq_reference / max(1, total_numel)) ** 0.5
    return {
        "key_sets_equal": True,
        "shape_dtype_equal": True,
        "tensor_count": len(reference),
        "exact_tensor_count": exact_count,
        "allclose_tensor_count": allclose_count,
        "allclose": allclose_count == len(reference),
        "rtol": rtol,
        "atol": atol,
        "max_abs": max_abs,
        "worst_key": worst_key,
        "rms_diff": rms_diff,
        "reference_rms": reference_rms,
        "relative_rms": rms_diff / max(reference_rms, 1e-30),
        "canonical_sha256_A": digest_reference.hexdigest(),
        "canonical_sha256_B": digest_current.hexdigest(),
    }


def main():
    context = initialize()
    try:
        torch.manual_seed(42 + context.rank); torch.cuda.manual_seed_all(42 + context.rank)
        model, _ = build_pretrained(MODEL_PATH, context.device, train_clip_none=True,
                                    ttt_layers=TTT_LAYERS, ttt_chunk=4096)
        apply_fsdp2(model, mesh=make_device_mesh(context.device.type, context.world_size))
        optimizer = torch.optim.AdamW(model.parameters(), lr=5.0e-6, betas=(0.9, 0.95),
                                      eps=1.0e-8, weight_decay=0.1, fused=True)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        progress_a = DistributedCheckpointRotation(A_ROOT, 2, [2], BATCH).load(
            model, optimizer, scheduler)
        model_a, optimizer_a = capture(model, optimizer)
        scheduler_a = json.dumps(scheduler.state_dict(), sort_keys=True, default=str)
        progress_b = DistributedCheckpointRotation(B_ROOT, 2, [2], BATCH).load(
            model, optimizer, scheduler)
        model_b, optimizer_b = current_maps(model, optimizer)
        model_comparison = compare(model_a, model_b, rtol=1.0e-2, atol=1.0e-5)
        optimizer_comparison = compare(optimizer_a, optimizer_b, rtol=1.0e-2, atol=1.0e-6)
        local = {
            "rank": context.rank,
            "physical_gpu": 2 + context.local_rank,
            "model": model_comparison,
            "optimizer": optimizer_comparison,
            "scheduler_exact": scheduler_a == json.dumps(
                scheduler.state_dict(), sort_keys=True, default=str),
            "progress_equal": progress_a == progress_b,
        }
        gathered = [None] * context.world_size
        dist.all_gather_object(gathered, local)
        passed = all(item["model"]["allclose"] and item["optimizer"]["allclose"]
                     and item["scheduler_exact"] and item["progress_equal"]
                     for item in gathered)
        if context.rank == 0:
            report = {"status": "PASS" if passed else "FAIL",
                      "comparison": "per_tensor_numeric_local_shards", "ranks": gathered}
            AUDIT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            print(json.dumps(report, sort_keys=True), flush=True)
        if not passed:
            raise RuntimeError("L4_NUMERIC_STATE_MISMATCH")
        barrier()
    finally:
        shutdown()


if __name__ == "__main__":
    main()
