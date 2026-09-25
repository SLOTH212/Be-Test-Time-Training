#!/usr/bin/env python3
"""Strict two-rank NCCL smoke for physical GPUs 2 and 3 only."""
import json
import os
import socket
from pathlib import Path

if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3":
    raise RuntimeError(
        f"PHYSICAL_GPU_ALLOWLIST_VIOLATION expected=2,3 actual={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "audits/hit_l3_l4_v1/L3_NCCL_REPORT.json"


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2 or torch.cuda.device_count() != 2:
        raise RuntimeError(
            f"TWO_GPU_CONTRACT world_size={world_size} visible={torch.cuda.device_count()}"
        )
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    value = torch.tensor(float(rank + 1), device=f"cuda:{local_rank}")
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    if float(value.item()) != 3.0:
        raise RuntimeError(f"ALL_REDUCE_MISMATCH value={value.item()}")
    dist.barrier()
    props = torch.cuda.get_device_properties(local_rank)
    local = {
        "rank": rank,
        "local_rank": local_rank,
        "logical_device": f"cuda:{local_rank}",
        "physical_gpu": 2 + local_rank,
        "device_name": props.name,
        "all_reduce_value": float(value.item()),
        "barrier": "PASS",
    }
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local)
    if rank == 0:
        report = {
            "status": "PASS",
            "hostname": socket.gethostname(),
            "backend": dist.get_backend(),
            "world_size": world_size,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "physical_gpu_allowlist": [2, 3],
            "unauthorized_gpu_used": False,
            "ranks": gathered,
            "collective": "PASS",
        }
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
