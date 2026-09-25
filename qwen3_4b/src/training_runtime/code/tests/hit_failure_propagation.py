#!/usr/bin/env python3
"""Controlled one-rank failure propagation smoke; no model or training work."""
import os
import sys
from pathlib import Path

if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3":
    raise RuntimeError("PHYSICAL_GPU_ALLOWLIST_VIOLATION")

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.distributed_runtime import barrier, initialize, shutdown


def main():
    context = initialize()
    if context.world_size != 2 or torch.cuda.device_count() != 2:
        raise RuntimeError("TWO_GPU_RUNTIME_CONTRACT")
    barrier()
    if context.rank == 0:
        print("INJECTED_RANK0_FAILURE", flush=True)
        raise RuntimeError("CONTROLLED_FAILURE_PROPAGATION_TEST")
    print("RANK1_WAITING_FOR_COLLECTIVE", flush=True)
    try:
        dist.barrier(device_ids=[torch.cuda.current_device()])
        print("RANK1_PROGRESS_AFTER_FAILURE", flush=True)
    finally:
        shutdown()


if __name__ == "__main__":
    main()
