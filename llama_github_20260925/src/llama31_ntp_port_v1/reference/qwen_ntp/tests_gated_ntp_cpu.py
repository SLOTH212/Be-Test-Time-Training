import copy
import unittest

import torch

from hf_models.hf_qwen3.configuration_qwen3 import Qwen3Config
from hf_models.hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3MLP
from hf_models.hf_qwen3.ttt_state_core import (
    GATED_NTP_TARGET,
    build_chunk_local_ntp_pairs,
    run_document,
    run_sequence_with_boundaries,
)


def tiny_config(**overrides):
    values = dict(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        ttt_layers=[0],
        ttt_mode=True,
        ttt_proj=True,
        ttt_lr=1.0,
        ttt_chunk=1024,
        ttt_target="hidden_states",
        ttt_target_type=GATED_NTP_TARGET,
        ttt_ntp_gate=True,
        ttt_ntp_gate_init="zero",
        ttt_ntp_within_chunk_only=True,
        ttt_ntp_cross_chunk=False,
        ttt_ntp_cross_record=False,
        ttt_ntp_cross_document=False,
        ttt_document_metadata_required=True,
        ttt_backend="authoritative_document_chunk_loop",
        optimizer_betas=(0.9, 0.95),
    )
    values.update(overrides)
    return Qwen3Config(**values)


def core_args(length, hidden_size=2):
    hidden = torch.arange(1, length * hidden_size + 1, dtype=torch.float32).reshape(1, length, hidden_size)
    target = hidden.flip(-1).requires_grad_()
    base = torch.zeros(hidden_size, hidden_size)
    projection = torch.eye(hidden_size, requires_grad=True)
    conv = torch.zeros(hidden_size, 1, 5, requires_grad=True)
    valid = torch.ones(1, length, dtype=torch.bool)
    return hidden, target, base, projection, conv, valid


class GatedNTPCPUTests(unittest.TestCase):
    def test_shift_alignment(self):
        source = torch.tensor([[[0.0], [1.0], [2.0], [3.0]]])
        target, pair = build_chunk_local_ntp_pairs(source, torch.ones(1, 4, dtype=torch.bool))
        self.assertEqual(target.flatten().tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(pair.flatten().tolist(), [True, True, True])

    def test_chunk_boundary(self):
        source = torch.arange(8.0).reshape(1, 8, 1)
        pairs = []
        for start in (0, 4):
            target, valid = build_chunk_local_ntp_pairs(
                source[:, start:start + 4], torch.ones(1, 4, dtype=torch.bool)
            )
            pairs.extend((start + index, int(target[0, index, 0])) for index in range(valid.shape[1]) if valid[0, index])
        self.assertEqual(pairs, [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7)])

    def test_tail_chunk(self):
        args = core_args(6)
        _, _, stats = run_document(
            *args, chunk_size=4, learning_rate=1.0, target_type=GATED_NTP_TARGET,
            target_gate=torch.ones(2), allow_test_override=True,
        )
        self.assertEqual([stat.valid_tokens for stat in stats], [3, 1])

    def test_padding_mask(self):
        source = torch.arange(6.0).reshape(1, 6, 1)
        valid = torch.tensor([[True, True, True, True, True, False]])
        _, pair = build_chunk_local_ntp_pairs(source, valid)
        self.assertEqual(pair.flatten().tolist(), [True, True, True, True, False])

    def test_document_boundary_reset(self):
        args = core_args(6)
        _, stats = run_sequence_with_boundaries(
            *args, document_boundaries=[(0, 3), (3, 6)], chunk_size=4,
            learning_rate=1.0, target_type=GATED_NTP_TARGET,
            target_gate=torch.ones(2), allow_test_override=True,
        )
        self.assertEqual([[s.valid_tokens for s in document] for document in stats], [[2], [2]])

    def test_gate_exact_zero_means_exact_zero_write(self):
        args = core_args(8)
        output, state, _ = run_document(
            *args, chunk_size=4, learning_rate=1.0, target_type=GATED_NTP_TARGET,
            target_gate=torch.zeros(2), allow_test_override=True,
        )
        self.assertTrue(torch.equal(state.weight, args[2]))
        self.assertTrue(torch.equal(output, torch.nn.functional.linear(args[0], args[2])))

    def test_pretrained_initializer_rezeros_materialized_gate(self):
        model = Qwen3ForCausalLM(tiny_config())
        mlp = model.model.layers[0].mlp
        mlp.ttt_ntp_gate.data.fill_(float("nan"))
        model._init_weights(mlp)
        self.assertTrue(torch.equal(mlp.ttt_ntp_gate, torch.zeros_like(mlp.ttt_ntp_gate)))

    def test_gate_gradient_and_conv_no_gradient(self):
        torch.manual_seed(7)
        mlp = Qwen3MLP(tiny_config(), layer_idx=0)
        self.assertTrue(torch.equal(mlp.ttt_ntp_gate, torch.zeros(8)))
        calls = []
        handle = mlp.ttt_conv.register_forward_hook(lambda *unused: calls.append(1))
        x = torch.randn(1, 2048, 8, requires_grad=True)
        source = torch.randn(1, 2048, 8, requires_grad=True)
        output = mlp(
            x, t=source, valid_mask=torch.ones(1, 2048, dtype=torch.bool),
            document_boundaries=[(0, 2048)],
        )
        output.square().mean().backward()
        handle.remove()
        self.assertEqual(calls, [])
        self.assertIsNone(mlp.ttt_conv.weight.grad)
        self.assertIsNotNone(mlp.ttt_ntp_gate.grad)
        self.assertTrue(torch.isfinite(mlp.ttt_ntp_gate.grad).all())
        self.assertGreater(float(mlp.ttt_ntp_gate.grad.abs().sum()), 0.0)
        self.assertIsNotNone(mlp.ttt_proj.weight.grad)
        self.assertTrue(torch.isfinite(mlp.ttt_proj.weight.grad).all())
        self.assertTrue(torch.equal(mlp.ttt_proj.weight.grad, torch.zeros_like(mlp.ttt_proj.weight.grad)))

    def test_projection_gradient_after_gate_opens(self):
        hidden, source, base, projection, conv, valid = core_args(8)
        output, _, _ = run_document(
            hidden, source, base, projection, conv, valid,
            chunk_size=4, learning_rate=1.0, target_type=GATED_NTP_TARGET,
            target_gate=torch.ones(2), allow_test_override=True,
        )
        output.square().sum().backward()
        self.assertTrue(torch.isfinite(projection.grad).all())
        self.assertGreater(float(projection.grad.abs().sum()), 0.0)
        self.assertIsNone(conv.grad)

    def test_apply_then_update_exclusive_prefix(self):
        hidden = torch.ones(1, 12, 1)
        source = torch.arange(1.0, 13.0).reshape(1, 12, 1)
        output, state, _ = run_document(
            hidden, source, torch.zeros(1, 1), torch.ones(1, 1), torch.zeros(1, 1, 5),
            torch.ones(1, 12, dtype=torch.bool), chunk_size=4, learning_rate=1.0,
            target_type=GATED_NTP_TARGET, target_gate=torch.ones(1), allow_test_override=True,
        )
        self.assertEqual(output[0, :4, 0].tolist(), [0.0] * 4)
        self.assertEqual(output[0, 4:8, 0].tolist(), [9.0] * 4)
        self.assertEqual(output[0, 8:12, 0].tolist(), [30.0] * 4)
        self.assertEqual(float(state.weight), 63.0)

    def test_off_parity_and_initialization(self):
        torch.manual_seed(19)
        legacy = Qwen3MLP(tiny_config(ttt_target_type="conv", ttt_ntp_gate=False, ttt_lr=0.3), layer_idx=0)
        torch.manual_seed(19)
        gated = Qwen3MLP(tiny_config(), layer_idx=0)
        for name, value in legacy.state_dict().items():
            self.assertTrue(torch.equal(value, gated.state_dict()[name]), name)
        x = torch.randn(1, 7, 8)
        self.assertTrue(torch.equal(legacy(x, t=None), gated(x, t=None)))

    def test_state_dict_strict_roundtrip(self):
        model = Qwen3MLP(tiny_config(), layer_idx=0)
        state = copy.deepcopy(model.state_dict())
        self.assertIn("ttt_ntp_gate", state)
        self.assertEqual(tuple(state["ttt_ntp_gate"].shape), (8,))
        self.assertIn("ttt_proj.weight", state)
        self.assertIn("ttt_conv.weight", state)
        reloaded = Qwen3MLP(tiny_config(), layer_idx=0)
        result = reloaded.load_state_dict(state, strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])

    def test_optimizer_contract_and_conv_unchanged_without_grad(self):
        model = Qwen3MLP(tiny_config(), layer_idx=0)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=5e-6, betas=(0.9, 0.95), eps=1e-8,
            weight_decay=0.1, fused=False,
        )
        group = optimizer.param_groups[0]
        self.assertEqual(group["betas"], (0.9, 0.95))
        self.assertEqual(group["lr"], 5e-6)
        self.assertEqual(group["weight_decay"], 0.1)
        self.assertTrue(any(param is model.ttt_ntp_gate for param in group["params"]))
        before = model.ttt_conv.weight.detach().clone()
        model.ttt_ntp_gate.grad = torch.ones_like(model.ttt_ntp_gate)
        optimizer.step()
        self.assertTrue(torch.equal(before, model.ttt_conv.weight))

    def test_formal_contract_hard_fails(self):
        invalid = (
            {"ttt_ntp_gate": False},
            {"ttt_ntp_gate_init": "one"},
            {"ttt_lr": 0.3},
            {"ttt_ntp_cross_chunk": True},
            {"ttt_ntp_cross_record": True},
            {"ttt_ntp_cross_document": True},
            {"ttt_backend": "vectorized_record_level_direct"},
            {"optimizer_betas": (0.9, 0.999)},
            {"router_enabled": True},
        )
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(ValueError):
                tiny_config(**change)


if __name__ == "__main__":
    unittest.main(verbosity=2)
