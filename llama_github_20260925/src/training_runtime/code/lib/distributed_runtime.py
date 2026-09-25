"""Minimal torchrun process lifecycle with rank-zero side-effect ownership."""
from dataclasses import dataclass
from datetime import timedelta
import os
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: str

    @property
    def is_rank0(self) -> bool:
        return self.rank == 0


def initialize() -> DistributedContext:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"
    device = torch.device("cuda", local_rank) if use_cuda else torch.device("cpu")
    if use_cuda:
        torch.cuda.set_device(device)
    if world_size > 1 and not dist.is_initialized():
        kwargs = {"backend": backend, "rank": rank, "world_size": world_size,
                  "timeout": timedelta(minutes=5)}
        if use_cuda: kwargs["device_id"] = device
        dist.init_process_group(**kwargs)
    return DistributedContext(rank, local_rank, world_size, device, backend)


def rank0_mkdir(context: DistributedContext, path: Path) -> None:
    if context.is_rank0:
        path.mkdir(parents=True, exist_ok=True)
    barrier()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def any_rank_failed(local_failed: bool, device: torch.device) -> bool:
    flag = torch.tensor(int(local_failed), device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def broadcast_from_rank0(value):
    if not (dist.is_available() and dist.is_initialized()):
        return value
    payload = [value if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def shutdown() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
