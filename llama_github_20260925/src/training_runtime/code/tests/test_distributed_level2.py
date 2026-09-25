import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.distributed_checkpoint import DistributedCheckpointRotation, load_model_only, save_model_only
from lib.distributed_grad import clip_grad_norm_fsdp2
from lib.fsdp2_runtime import apply_fsdp2, make_device_mesh
from lib.model_factory import build_tiny_authority


def materialize(value):
    if hasattr(value, "full_tensor"): value = value.full_tensor()
    return value.detach().float().cpu()


class Level2(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "local CUDA unavailable; defer to HIT")
    def test_world_size_one_matches_unwrapped_authority(self):
        sock = socket.socket(); sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]; sock.close()
        dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)
        try:
            device = torch.device("cuda", 0); torch.cuda.set_device(device)
            torch.manual_seed(42); authority, _ = build_tiny_authority(device)
            torch.manual_seed(42); candidate, _ = build_tiny_authority(device)
            authority = authority.to(torch.bfloat16).train(); candidate = candidate.to(torch.bfloat16).train()
            authority.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            candidate.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            inventory = apply_fsdp2(candidate, make_device_mesh("cuda", 1))
            self.assertEqual(inventory.decoder_blocks, 28)
            optimizer_a = torch.optim.AdamW(authority.parameters(), lr=5e-6, betas=(0.9, 0.95),
                                              eps=1e-8, weight_decay=0.1)
            optimizer_b = torch.optim.AdamW(candidate.parameters(), lr=5e-6, betas=(0.9, 0.95),
                                              eps=1e-8, weight_decay=0.1)
            scheduler_a = torch.optim.lr_scheduler.LambdaLR(optimizer_a, lambda _: 1.0)
            scheduler_b = torch.optim.lr_scheduler.LambdaLR(optimizer_b, lambda _: 1.0)
            generator = torch.Generator(device=device).manual_seed(9001)
            inputs = torch.randint(0, 128, (1, 24), generator=generator, device=device)
            kwargs = {"input_ids": inputs, "labels": inputs, "attention_mask": torch.ones_like(inputs),
                      "use_cache": False, "document_boundaries": [(0, 24)]}
            loss_a = authority(**kwargs).loss; loss_b = candidate(**kwargs).loss
            self.assertTrue(torch.allclose(loss_a, loss_b, atol=2e-2, rtol=2e-2))
            loss_a.backward(); loss_b.backward()
            norm_a = float(torch.nn.utils.clip_grad_norm_(authority.parameters(), 1.0))
            norm_b = clip_grad_norm_fsdp2(candidate.parameters(), 1.0)
            self.assertAlmostEqual(norm_a, norm_b, delta=max(2e-2, abs(norm_a) * 2e-2))
            optimizer_a.step(); scheduler_a.step(); optimizer_b.step(); scheduler_b.step()
            states_a, states_b = authority.state_dict(), candidate.state_dict()
            self.assertEqual(set(states_a), set(states_b))
            for name in states_a:
                self.assertTrue(torch.allclose(materialize(states_a[name]), materialize(states_b[name]),
                                               atol=3e-2, rtol=3e-2), name)
            self.assertEqual(scheduler_a.state_dict(), scheduler_b.state_dict())
            self.assertEqual(24, inputs.numel())  # identical record/token authority

            with tempfile.TemporaryDirectory() as root:
                rotation = DistributedCheckpointRotation(Path(root) / "rotation", 1, [1],
                    {"global_batch_size": 1, "micro_batch_size": 1, "gradient_accumulation": 1})
                progress = {"record_cursor": 1, "cumulative_tokens": 24, "update_step": 1,
                            "accumulation_phase": 0}
                rotation.save(candidate, optimizer_b, progress, scheduler_b)
                loaded = rotation.load(candidate, optimizer_b, scheduler_b)
                self.assertEqual(loaded, progress)
                save_model_only(candidate, Path(root) / "parent", {"source_stage": 1})
                torch.manual_seed(17); stage2, _ = build_tiny_authority(device)
                stage2 = stage2.to(torch.bfloat16); apply_fsdp2(stage2, make_device_mesh("cuda", 1))
                self.assertEqual(load_model_only(stage2, Path(root) / "parent")["source_stage"], 1)
                for name, value in candidate.state_dict().items():
                    self.assertTrue(torch.equal(materialize(value), materialize(stage2.state_dict()[name])), name)
        finally:
            dist.destroy_process_group()


if __name__ == "__main__": unittest.main(verbosity=2)
