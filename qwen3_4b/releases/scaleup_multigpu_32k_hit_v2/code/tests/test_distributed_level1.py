import json
import os
import random
import socket
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.distributed_checkpoint import DistributedCheckpointRotation, load_model_only, save_model_only
from lib.distributed_grad import clip_local_shards_reference
from lib.distributed_runtime import any_rank_failed, broadcast_from_rank0
from lib.distributed_semantics import BatchPlan, globally_scaled_loss


def free_port():
    sock = socket.socket(); sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]; sock.close(); return port


def worker(rank, world, port, root):
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    try:
        plan = BatchPlan(world, 1, 2)
        local = [p for p in range(11) if plan.owner(p)[2] == rank]
        gathered = [None] * world; dist.all_gather_object(gathered, local)
        if rank == 0:
            assert sorted(x for shard in gathered for x in shard) == list(range(11))
        assert broadcast_from_rank0({"owner": 0} if rank == 0 else None) == {"owner": 0}
        assert any_rank_failed(rank == 1, torch.device("cpu"))

        accumulated = torch.tensor(1.0, requires_grad=True)
        for coefficient in (rank * 2.0 + 1.0, rank * 2.0 + 2.0):
            (accumulated * coefficient * (world / 4.0)).backward()
        accumulated_grad = accumulated.grad.clone(); dist.all_reduce(accumulated_grad); accumulated_grad /= world
        assert torch.allclose(accumulated_grad, torch.tensor(2.5))

        local_tokens = torch.tensor(10 + rank, dtype=torch.int64); dist.all_reduce(local_tokens)
        assert int(local_tokens) == 21
        owner_file = Path(root) / "rank0-owned.txt"
        if rank == 0: owner_file.write_text("rank0")
        dist.barrier(); assert owner_file.read_text() == "rank0"

        parameter = torch.tensor(2.0, requires_grad=True)
        local_count = rank + 2
        local_coeff = 3.0 if rank == 0 else 5.0
        scaled, denominator = globally_scaled_loss(parameter * local_coeff, local_count)
        scaled.backward()
        grad = parameter.grad.clone(); dist.all_reduce(grad); grad /= world
        assert denominator == 5
        assert torch.allclose(grad, torch.tensor(8.0 / 5.0))

        shard = torch.nn.Parameter(torch.zeros(2)); shard.grad = torch.tensor([3.0 + rank, 4.0])
        before = clip_local_shards_reference([shard], 1.0, torch.device("cpu"))
        expected = ((3.0 ** 2 + 4.0 ** 2) + (4.0 ** 2 + 4.0 ** 2)) ** 0.5
        assert abs(before - expected) < 1e-6

        # Replicated logical state (as reconstructed by FSDP), rank-local RNG.
        torch.manual_seed(777)
        model = torch.nn.Linear(3, 2); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        original = {k: v.detach().clone() for k, v in model.state_dict().items()}
        torch.manual_seed(123 + rank); random.seed(123 + rank); np.random.seed(123 + rank)
        rotation = DistributedCheckpointRotation(Path(root) / "rotation", world, [world],
                                                  {"global_batch_size": 4, "micro_batch_size": 1,
                                                   "gradient_accumulation": 2})
        progress = {"record_cursor": 11, "cumulative_tokens": 99, "update_step": 3}
        rotation.save(model, optimizer, progress)
        if rank == 0:
            latest = json.loads((Path(root) / "rotation/latest.json").read_text())
            manifest = json.loads((Path(root) / "rotation" / latest["slot"] / "manifest.json").read_text())
            assert manifest["complete"] and manifest["world_size"] == world and manifest["files"]
        rank_payload = torch.load(Path(root) / "rotation" / rotation.latest()["slot"] /
                                  f"rank_{rank:05d}.pt", map_location="cpu", weights_only=False)
        assert {"python_rng", "numpy_rng", "torch_cpu_rng", "torch_cuda_rng"} <= set(rank_payload)
        expected_rng = (random.random(), float(np.random.rand()), float(torch.rand(())))
        for p in model.parameters(): p.data.add_(10)
        loaded = rotation.load(model, optimizer)
        actual_rng = (random.random(), float(np.random.rand()), float(torch.rand(())))
        assert loaded == progress and expected_rng == actual_rng
        for key, value in model.state_dict().items(): assert torch.equal(value, original[key])

        handoff = Path(root) / "handoff"
        save_model_only(model, handoff, {"source_stage": 1})
        for p in model.parameters(): p.data.zero_()
        metadata = load_model_only(model, handoff)
        assert metadata["source_stage"] == 1
        for key, value in model.state_dict().items(): assert torch.equal(value, original[key])
    finally:
        dist.destroy_process_group()


class Level1(unittest.TestCase):
    def test_two_rank_gloo_semantics_checkpoint_resume(self):
        with tempfile.TemporaryDirectory() as root:
            mp.spawn(worker, args=(2, free_port(), root), nprocs=2, join=True)
            rotation = DistributedCheckpointRotation(Path(root) / "rotation", 1, [1],
                {"global_batch_size": 4, "micro_batch_size": 1, "gradient_accumulation": 2})
            model = torch.nn.Linear(3, 2); optimizer = torch.optim.AdamW(model.parameters())
            with self.assertRaisesRegex(RuntimeError, "WORLD_SIZE_OR_MESH_MISMATCH"):
                rotation.load(model, optimizer)


if __name__ == "__main__": unittest.main(verbosity=2)
