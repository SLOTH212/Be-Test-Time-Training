import ast
import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.distributed_semantics import BatchPlan, assert_batch_equation
from lib.fsdp2_runtime import apply_fsdp2
from lib.model_factory import EXPECTED, DEFAULT_TTT_LAYERS as TTT_LAYERS, VENDOR, build_tiny_authority, verify_frozen_vendor


class Block:
    pass


class FakeModel:
    def __init__(self, count=28):
        self.model = type("Body", (), {"layers": [Block() for _ in range(count)]})()


class Level0(unittest.TestCase):
    def test_frozen_source_hashes(self):
        verify_frozen_vendor()
        self.assertEqual(len(EXPECTED), 6)

    def test_all_python_parses(self):
        for path in list((ROOT / "lib").glob("*.py")) + list((ROOT / "workers").glob("*.py")):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_exact_fsdp2_inventory(self):
        calls = []
        inventory = apply_fsdp2(FakeModel(), mesh="mesh", fully_shard_impl=lambda obj, **kw: calls.append((obj, kw)))
        self.assertEqual(inventory.decoder_blocks, 28)
        self.assertEqual(len(calls), 29)
        self.assertTrue(inventory.root_wrapped)
        self.assertEqual((inventory.mixed_precision, inventory.reduce_precision), ("bf16", "fp32"))

    def test_recovered_builder_and_ttt_parameter_inventory(self):
        import torch
        model, config = build_tiny_authority(torch.device("cpu"))
        self.assertEqual(len(model.model.layers), 28)
        self.assertEqual(config.ttt_layers, TTT_LAYERS)
        self.assertEqual((config.ttt_lr, config.ttt_chunk), (1.0, 1024))
        self.assertEqual(config.ttt_target_type, "gated_next_position_hidden")
        names = dict(model.named_parameters())
        self.assertEqual(len(names), len({id(p) for p in names.values()}))
        for layer in TTT_LAYERS:
            prefix = f"model.layers.{layer}.mlp"
            projection = names[f"{prefix}.ttt_proj.weight"]
            gate = names[f"{prefix}.ttt_ntp_gate"]
            convolution = names[f"{prefix}.ttt_conv.weight"]
            self.assertEqual(tuple(projection.shape), (32, 32))
            self.assertTrue(projection.requires_grad and gate.requires_grad and convolution.requires_grad)
            self.assertEqual(float(gate.detach().abs().max()), 0.0)
            self.assertEqual(float(convolution.detach().abs().max()), 0.0)

    def test_canonical_partition_complete_disjoint(self):
        plan = BatchPlan(4, 2, 3)
        owned = [[p for p in range(103) if plan.owner(p)[2] == rank] for rank in range(4)]
        self.assertEqual(sorted(p for shard in owned for p in shard), list(range(103)))
        self.assertEqual(len(set(p for shard in owned for p in shard)), 103)
        self.assertEqual(plan.global_batch, 24)

    def test_batch_equation_fail_closed(self):
        assert_batch_equation(4, 1, 4, 1)
        with self.assertRaisesRegex(RuntimeError, "GLOBAL_BATCH_EQUATION"):
            assert_batch_equation(8, 1, 4, 1)

    def test_no_fsdp1_or_hardcoded_worker_cuda(self):
        sources = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "lib").glob("*.py"))
        worker = (ROOT / "workers/distributed_train_worker.py").read_text(encoding="utf-8")
        self.assertNotIn("FullyShardedDataParallel", sources + worker)
        self.assertNotIn(".cuda()", worker)
        self.assertNotIn("cuda:0", worker)
        self.assertIn("fully_shard", sources)

    def test_torchrun_rendering_and_optimizer_after_wrap(self):
        launches = "\n".join(p.read_text() for p in (ROOT / "bin").glob("launch_*_multigpu.sh"))
        self.assertEqual(launches.count("torchrun"), 2)
        worker = (ROOT / "workers/distributed_train_worker.py").read_text()
        self.assertLess(worker.index("apply_fsdp2(model"), worker.index("torch.optim.AdamW(model.parameters()"))
        for forbidden in ("shuffle(", "Cosine", "warmup", "global_batch_size: 64"):
            self.assertNotIn(forbidden, worker + launches +
                             (ROOT / "configs/stage1_1b_32k_multigpu_v1.yaml").read_text())

    def test_configs_preserve_recipe_and_label_extension(self):
        s1 = (ROOT / "configs/stage1_1b_32k_multigpu_v1.yaml").read_text()
        s2 = (ROOT / "configs/stage2_qa_multigpu_v1.yaml").read_text()
        for text in (s1, s2):
            for token in ("outer_lr: 5.0e-6", "betas: [0.9, 0.95]", "eps: 1.0e-8",
                          "weight_decay: 0.1", "ttt_lr: 1.0", "ttt_chunk_size: 1024"):
                self.assertIn(token, text)
        self.assertIn("stage2_new_correctness_extension", s2)
        self.assertIn("separate_global_group_means", s2)
        self.assertIn("HIT_CAPACITY_TUNING_PENDING", s1)
        self.assertIn("HIT_CAPACITY_TUNING_PENDING", s2)


if __name__ == "__main__": unittest.main(verbosity=2)
