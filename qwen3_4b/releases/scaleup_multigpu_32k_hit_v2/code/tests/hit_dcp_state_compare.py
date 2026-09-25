#!/usr/bin/env python3
"""Compare two real two-rank DCP checkpoints without executing a train step."""
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

MODEL_PATH = Path("/home/USER/data/USER/models/Qwen3-4B-Base")
A_ROOT = ROOT / "runs/l4_uninterrupted_v1/checkpoints"
B_ROOT = ROOT / "runs/l4_exact_a_step1_snapshot/checkpoints"
AUDIT = ROOT / "audits/hit_l3_l4_v1/L4_STATE_COMPARE.json"
TTT_LAYERS = [0, 6, 12, 18, 24, 30]
BATCH = {"global_batch_size": 2, "micro_batch_size": 1, "gradient_accumulation": 1}


def update_tensor(digest, name, value):
    local = value.to_local() if hasattr(value, "to_local") else value
    local = local.detach().contiguous()
    digest.update(name.encode())
    digest.update(str(local.dtype).encode())
    digest.update(str(tuple(local.shape)).encode())
    digest.update(local.reshape(-1).view(torch.uint8).cpu().numpy().tobytes())


def state_digests(model, optimizer, scheduler):
    model_digest = hashlib.sha256()
    named_parameters = sorted(model.named_parameters(), key=lambda item: item[0])
    for name, parameter in named_parameters:
        update_tensor(model_digest, name, parameter)

    optimizer_digest = hashlib.sha256()
    parameter_names = {id(parameter): name for name, parameter in named_parameters}
    for parameter, state in sorted(
        optimizer.state.items(), key=lambda item: parameter_names[id(item[0])]
    ):
        parameter_name = parameter_names[id(parameter)]
        for key, value in sorted(state.items()):
            item_name = f"{parameter_name}.{key}"
            if torch.is_tensor(value) or hasattr(value, "to_local"):
                update_tensor(optimizer_digest, item_name, value)
            else:
                optimizer_digest.update(item_name.encode())
                optimizer_digest.update(repr(value).encode())
    for index, group in enumerate(optimizer.param_groups):
        optimizer_digest.update(f"group.{index}".encode())
        optimizer_digest.update(json.dumps(
            {key: value for key, value in group.items() if key != "params"},
            sort_keys=True, default=str,
        ).encode())
        optimizer_digest.update(json.dumps(
            [parameter_names[id(parameter)] for parameter in group["params"]]
        ).encode())
    scheduler_digest = hashlib.sha256(
        json.dumps(scheduler.state_dict(), sort_keys=True, default=str).encode()
    ).hexdigest()
    return {
        "model_sha256": model_digest.hexdigest(),
        "optimizer_sha256": optimizer_digest.hexdigest(),
        "scheduler_sha256": scheduler_digest,
        "optimizer_state_parameter_count": len(optimizer.state),
    }


def main():
    context = initialize()
    try:
        if context.world_size != 2 or torch.cuda.device_count() != 2:
            raise RuntimeError("TWO_GPU_RUNTIME_CONTRACT")
        torch.manual_seed(42 + context.rank)
        torch.cuda.manual_seed_all(42 + context.rank)
        model, _ = build_pretrained(
            MODEL_PATH, context.device, train_clip_none=True,
            ttt_layers=TTT_LAYERS, ttt_chunk=4096,
        )
        mesh = make_device_mesh(context.device.type, context.world_size)
        apply_fsdp2(model, mesh=mesh)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=5.0e-6, betas=(0.9, 0.95), eps=1.0e-8,
            weight_decay=0.1, fused=True,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

        a_rotation = DistributedCheckpointRotation(A_ROOT, 2, [2], BATCH)
        progress_a = a_rotation.load(model, optimizer, scheduler)
        digest_a = state_digests(model, optimizer, scheduler)

        b_rotation = DistributedCheckpointRotation(B_ROOT, 2, [2], BATCH)
        progress_b = b_rotation.load(model, optimizer, scheduler)
        digest_b = state_digests(model, optimizer, scheduler)
        local = {
            "rank": context.rank,
            "physical_gpu": 2 + context.local_rank,
            "A": digest_a,
            "B": digest_b,
            "model_exact": digest_a["model_sha256"] == digest_b["model_sha256"],
            "optimizer_exact": digest_a["optimizer_sha256"] == digest_b["optimizer_sha256"],
            "scheduler_exact": digest_a["scheduler_sha256"] == digest_b["scheduler_sha256"],
            "progress_equal": progress_a == progress_b,
        }
        gathered = [None] * context.world_size
        dist.all_gather_object(gathered, local)
        passed = all(
            item["model_exact"] and item["optimizer_exact"]
            and item["scheduler_exact"] and item["progress_equal"]
            for item in gathered
        )
        if context.rank == 0:
            report = {
                "status": "PASS" if passed else "FAIL",
                "comparison": "exact_per_rank_local_shard_sha256",
                "A": str(A_ROOT),
                "B": str(B_ROOT),
                "ranks": gathered,
            }
            AUDIT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            print(json.dumps(report, sort_keys=True), flush=True)
        if not passed:
            raise RuntimeError("L4_STATE_DIGEST_MISMATCH")
        barrier()
    finally:
        shutdown()


if __name__ == "__main__":
    main()
