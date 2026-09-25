#!/usr/bin/env python3
"""Restricted equivalence checks for legacy record NTP and gated chunk-local NTP.

This is a validation-only harness.  It performs no optimization, training,
generation, dataset access, profiling, or network access.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import torch


def load_old_core(old_repo: Path):
    path = old_repo / "ttt_core" / "record_level.py"
    spec = importlib.util.spec_from_file_location("legacy_record_level_for_validation", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def metric(a: torch.Tensor, b: torch.Tensor) -> dict:
    af = a.detach().float()
    bf = b.detach().float()
    diff = (af - bf).abs()
    denom = torch.linalg.vector_norm(bf).clamp_min(torch.finfo(torch.float32).tiny)
    return {
        "shape": list(a.shape),
        "max_abs": float(diff.max()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean()) if diff.numel() else 0.0,
        "relative_l2": float(torch.linalg.vector_norm(af - bf) / denom),
        "bitwise_equal": bool(torch.equal(a, b)),
    }


def close(a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float) -> bool:
    return bool(torch.allclose(a, b, atol=atol, rtol=rtol))


def tensor_sha256(value: torch.Tensor) -> str:
    raw = value.detach().contiguous().cpu()
    if raw.dtype == torch.bfloat16:
        raw = raw.view(torch.uint16)
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def cpu_checks(old_repo: Path, new_repo: Path) -> dict:
    sys.path.insert(0, str(new_repo))
    from hf_models.hf_qwen3.ttt_state_core import (
        GATED_NTP_TARGET,
        build_chunk_local_ntp_pairs,
        run_document,
    )

    old = load_old_core(old_repo)
    torch.manual_seed(20260802)
    sequence, input_size, hidden_size = 16, 7, 5
    originals = (
        torch.randn(1, sequence, input_size),
        torch.randn(1, sequence, hidden_size),
        torch.randn(hidden_size, input_size),
        torch.randn(hidden_size, hidden_size),
        torch.randn(hidden_size) + 0.25,
    )
    valid = torch.ones(1, sequence, dtype=torch.bool)

    def make_values():
        return tuple(x.detach().clone().requires_grad_(True) for x in originals)

    oh, osrc, obase, oproj, ogate = make_values()
    legacy = old.vectorized_record_level_direct(
        oh, osrc, obase, oproj, valid,
        chunk_size=sequence, learning_rate=1.0, target_gate=ogate,
    )
    legacy_loss = legacy.output.square().mean() + legacy.final_weight.square().mean()
    legacy_grads = torch.autograd.grad(legacy_loss, (oh, osrc, obase, oproj, ogate))

    nh, nsrc, nbase, nproj, ngate = make_values()
    new_output, new_state, new_stats = run_document(
        nh, nsrc, nbase, nproj,
        torch.zeros(hidden_size, 1, 5), valid,
        chunk_size=sequence, learning_rate=1.0,
        target_type=GATED_NTP_TARGET, target_gate=ngate,
        allow_test_override=True,
    )
    new_loss = new_output.square().mean() + new_state.weight.square().mean()
    new_grads = torch.autograd.grad(new_loss, (nh, nsrc, nbase, nproj, ngate))
    new_target, new_pair = build_chunk_local_ntp_pairs(nsrc, valid)

    comparisons = {
        "target": metric(legacy.target[:, :-1], new_target),
        "pair_valid": metric(legacy.pair_valid[:, :-1], new_pair),
        "output": metric(legacy.output, new_output),
        "update": metric(legacy.updates[0, 0], new_state.weight - nbase),
        "final_weight": metric(legacy.final_weight[0], new_state.weight),
        "loss": metric(legacy_loss.reshape(1), new_loss.reshape(1)),
    }
    grad_names = ("hidden", "target_source", "base_weight", "projection", "gate")
    comparisons["gradients"] = {
        name: metric(left, right)
        for name, left, right in zip(grad_names, legacy_grads, new_grads)
    }
    values_to_check = [
        comparisons["target"], comparisons["pair_valid"], comparisons["output"],
        comparisons["update"], comparisons["final_weight"], comparisons["loss"],
        *comparisons["gradients"].values(),
    ]
    component_pass = all(item["max_abs"] <= 2e-5 and item["relative_l2"] <= 2e-6 for item in values_to_check)

    # Backend-only equivalence: feed exactly the same already-computed updates
    # into the legacy exclusive prefix and an explicit authoritative recurrence.
    torch.manual_seed(8)
    chunks = torch.randn(1, 4, 3, input_size)
    base = torch.randn(hidden_size, input_size)
    updates = torch.randn(1, 4, hidden_size, input_size) * 0.01
    legacy_prefix = old.exclusive_record_prefix_scan(updates)
    legacy_chunk_output, legacy_effective = old.apply_batched_effective_weights(
        chunks, base, legacy_prefix
    )
    current = base.clone()
    recurrence_outputs, recurrence_effective = [], []
    for index in range(updates.shape[1]):
        recurrence_effective.append(current)
        recurrence_outputs.append(torch.nn.functional.linear(chunks[0, index], current))
        current = current + updates[0, index]
    recurrence_output = torch.stack(recurrence_outputs).unsqueeze(0)
    recurrence_weights = torch.stack(recurrence_effective).unsqueeze(0)
    backend_comparisons = {
        "effective_weights": metric(legacy_effective, recurrence_weights),
        "chunk_output": metric(legacy_chunk_output, recurrence_output),
        "final_weight": metric(base + updates.sum(1)[0], current),
    }
    backend_pass = all(
        item["max_abs"] <= 2e-6 and item["relative_l2"] <= 2e-6
        for item in backend_comparisons.values()
    )

    def semantic_case(length: int, chunk_size: int) -> dict:
        hidden = torch.ones(1, length, 1)
        source = torch.arange(1.0, length + 1).reshape(1, length, 1)
        base = torch.zeros(1, 1)
        projection = torch.ones(1, 1)
        gate = torch.ones(1)
        valid_local = torch.ones(1, length, dtype=torch.bool)
        old_result = old.vectorized_record_level_direct(
            hidden, source, base, projection, valid_local,
            chunk_size=chunk_size, learning_rate=1.0, target_gate=gate,
        )
        new_out, new_state, new_chunk_stats = run_document(
            hidden, source, base, projection, torch.zeros(1, 1, 5), valid_local,
            chunk_size=chunk_size, learning_rate=1.0,
            target_type=GATED_NTP_TARGET, target_gate=gate,
            allow_test_override=True,
        )
        old_pairs = [int(i) for i in old_result.pair_valid[0].nonzero().flatten()]
        new_pairs = []
        for start in range(0, length, chunk_size):
            _, pair = build_chunk_local_ntp_pairs(
                source[:, start : start + chunk_size], valid_local[:, start : start + chunk_size]
            )
            new_pairs.extend(start + int(i) for i in pair[0].nonzero().flatten())
        return {
            "length": length,
            "chunk_size": chunk_size,
            "old_pair_source_positions": old_pairs,
            "new_pair_source_positions": new_pairs,
            "old_updates": old_result.updates.flatten().tolist(),
            "new_valid_pairs_per_chunk": [s.valid_tokens for s in new_chunk_stats],
            "old_output": old_result.output.flatten().tolist(),
            "new_output": new_out.flatten().tolist(),
            "old_final_weight": float(old_result.final_weight),
            "new_final_weight": float(new_state.weight),
            "outputs_intentionally_different": not torch.equal(old_result.output, new_out),
            "final_weights_intentionally_different": not torch.equal(old_result.final_weight[0], new_state.weight),
        }

    full = semantic_case(8, 4)
    tail = semantic_case(6, 4)
    semantic_pass = (
        full["old_pair_source_positions"] == [0, 1, 2, 3, 4, 5, 6]
        and full["new_pair_source_positions"] == [0, 1, 2, 4, 5, 6]
        and full["old_final_weight"] == 35.0
        and full["new_final_weight"] == 30.0
        and tail["old_pair_source_positions"] == [0, 1, 2, 3, 4]
        and tail["new_pair_source_positions"] == [0, 1, 2, 4]
        and tail["old_final_weight"] == 14.0
        and tail["new_final_weight"] == 15.0
        and full["outputs_intentionally_different"]
        and tail["outputs_intentionally_different"]
    )

    return {
        "status": "passed" if component_pass and backend_pass and semantic_pass else "failed",
        "environment": {"device": "cpu", "dtype": "float32", "torch": torch.__version__},
        "single_chunk_component_equivalence": {
            "status": "passed" if component_pass else "failed",
            "conditions": {
                "batch": 1, "sequence": sequence, "padding": False,
                "records": 1, "documents": 1, "chunks": 1,
                "learning_rate": 1.0, "gate_nonzero": bool(torch.count_nonzero(originals[-1]) == hidden_size),
            },
            "new_update_valid_tokens": new_stats[0].valid_tokens,
            "comparisons": comparisons,
        },
        "backend_same_precomputed_updates": {
            "status": "passed" if backend_pass else "failed",
            "comparisons": backend_comparisons,
        },
        "multi_chunk_expected_difference": {
            "status": "passed" if semantic_pass else "failed",
            "full_two_chunks": full,
            "tail_chunk": tail,
        },
    }


def cuda_real_model_check(old_repo: Path, new_repo: Path, model_path: Path) -> dict:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        return {"status": "failed", "error": "CUDA BF16 support is required"}
    sys.path.insert(0, str(new_repo))
    from hf_models.hf_qwen3.configuration_qwen3 import Qwen3Config
    from hf_models.hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM
    from hf_models.hf_qwen3.ttt_state_core import GATED_NTP_TARGET, run_document

    old = load_old_core(old_repo)
    torch.manual_seed(20260802)
    torch.cuda.manual_seed_all(20260802)
    config = Qwen3Config.from_pretrained(model_path, local_files_only=True)
    config.use_cache = False
    config.ttt_mode = True
    config.ttt_layers = [0]
    config.ttt_chunk = 1024
    config.ttt_proj = True
    config.ttt_lr = 1.0
    config.ttt_target = "hidden_states"
    config.ttt_target_type = GATED_NTP_TARGET
    config.ttt_ntp_gate = True
    config.ttt_ntp_gate_init = "zero"
    config.ttt_ntp_within_chunk_only = True
    config.ttt_ntp_cross_chunk = False
    config.ttt_ntp_cross_record = False
    config.ttt_ntp_cross_document = False
    config.ttt_document_metadata_required = True
    config.ttt_backend = "authoritative_document_chunk_loop"
    config.optimizer_betas = [0.9, 0.95]
    config.router_enabled = False
    config.signal_capture_enabled = False
    config.calibration_assignment_enabled = False
    model = Qwen3ForCausalLM.from_pretrained(
        model_path, config=config, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).cuda().eval()
    mlp = model.model.layers[0].mlp
    gate_zero_at_initialization = bool(torch.equal(mlp.ttt_ntp_gate, torch.zeros_like(mlp.ttt_ntp_gate)))
    captured = {}

    def capture(_module, args, kwargs):
        captured["x"] = args[0].detach()
        captured["target"] = kwargs["t"].detach()

    hook = mlp.register_forward_pre_hook(capture, with_kwargs=True)
    length = 128
    ids = torch.arange(1000, 1000 + length, device="cuda", dtype=torch.long).unsqueeze(0)
    valid = torch.ones_like(ids, dtype=torch.bool)
    with torch.no_grad():
        model.model(
            input_ids=ids, attention_mask=valid,
            document_boundaries=[(0, length)], use_cache=False,
        )
    hook.remove()
    if set(captured) != {"x", "target"}:
        raise RuntimeError("failed to capture real layer-0 hidden and contextual target")
    with torch.no_grad():
        update_hidden = mlp.act_fn(mlp.gate_proj(captured["x"])) * mlp.up_proj(captured["x"])
        target = captured["target"]
        gate = torch.linspace(0.25, 1.25, config.hidden_size, device="cuda", dtype=torch.float32).to(torch.bfloat16)
        old_result = old.vectorized_record_level_direct(
            update_hidden, target, mlp.down_proj.weight, mlp.ttt_proj.weight, valid,
            chunk_size=length, learning_rate=1.0, target_gate=gate,
        )
        new_output, new_state, stats = run_document(
            update_hidden, target, mlp.down_proj.weight, mlp.ttt_proj.weight,
            mlp.ttt_conv.weight, valid, chunk_size=length, learning_rate=1.0,
            target_type=GATED_NTP_TARGET, target_gate=gate,
            allow_test_override=True,
        )
    output_metric = metric(old_result.output, new_output)
    update_metric = metric(old_result.updates[0, 0], new_state.weight - mlp.down_proj.weight)
    final_metric = metric(old_result.final_weight[0], new_state.weight)
    # BF16 paths use different but algebraically equivalent contraction forms.
    # Use a dtype-appropriate tolerance and retain raw errors in the report.
    output_close = close(old_result.output, new_output, atol=0.125, rtol=0.02)
    update_close = close(old_result.updates[0, 0], new_state.weight - mlp.down_proj.weight, atol=0.125, rtol=0.02)
    final_close = close(old_result.final_weight[0], new_state.weight, atol=0.125, rtol=0.02)
    passed = gate_zero_at_initialization and output_close and update_close and final_close and stats[0].valid_tokens == length - 1
    result = {
        "status": "passed" if passed else "failed",
        "model": str(model_path),
        "model_class": type(model).__name__,
        "layer": 0,
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "dtype": "bfloat16",
        "sequence": length,
        "single_chunk": True,
        "padding": False,
        "record_count": 1,
        "document_count": 1,
        "learning_rate": 1.0,
        "gate_zero_at_initialization": gate_zero_at_initialization,
        "test_gate_nonzero_count": int(torch.count_nonzero(gate)),
        "valid_pairs": stats[0].valid_tokens,
        "same_inputs": {
            "update_hidden_sha256": tensor_sha256(update_hidden),
            "contextual_target_sha256": tensor_sha256(target),
            "base_weight_sha256": tensor_sha256(mlp.down_proj.weight),
            "projection_sha256": tensor_sha256(mlp.ttt_proj.weight),
            "gate_sha256": tensor_sha256(gate),
        },
        "tolerances": {"atol": 0.125, "rtol": 0.02},
        "comparisons": {
            "output": {**output_metric, "within_tolerance": output_close},
            "update": {**update_metric, "within_tolerance": update_close},
            "final_weight": {**final_metric, "within_tolerance": final_close},
        },
    }
    del old_result, new_output, new_state, update_hidden, target, captured, mlp, model, ids, valid
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-repo", type=Path, required=True)
    parser.add_argument("--new-repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "scope": "restricted old record-level NTP vs new gated NTP equivalence",
        "prohibitions_observed": [
            "no_training", "no_ruler", "no_profiler", "no_download",
            "no_old_repo_modification", "no_method_semantics_change",
            "no_shutdown", "no_process_kill",
        ],
        "host": {"python": platform.python_version(), "platform": platform.platform()},
        "cpu": cpu_checks(args.old_repo, args.new_repo),
        "cuda_bf16_real_model": cuda_real_model_check(args.old_repo, args.new_repo, args.model),
    }
    result["status"] = "passed" if result["cpu"]["status"] == "passed" and result["cuda_bf16_real_model"]["status"] == "passed" else "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
