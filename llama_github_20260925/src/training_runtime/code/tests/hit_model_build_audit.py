#!/usr/bin/env python3
"""Build Qwen3-4B, inventory HIT TTT layers, apply FSDP2, and stop before forward."""
import json
import os
import sys
import time
from pathlib import Path

if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3":
    raise RuntimeError("PHYSICAL_GPU_ALLOWLIST_VIOLATION")

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))
from lib.distributed_runtime import initialize, shutdown
from lib.fsdp2_runtime import apply_fsdp2, make_device_mesh
from lib.model_factory import build_pretrained

MODEL_PATH = Path(os.path.expandvars("${TTT_MODEL_ROOT}/Qwen3-4B-Base"))
AUDIT = ROOT / "audits/hit_l3_l4_v1"
TTT_LAYERS = [0, 6, 12, 18, 24, 30]


def module_inventory(model):
    result = {}
    for index in TTT_LAYERS:
        mlp = model.model.layers[index].mlp
        present = {
            "ttt_proj": getattr(mlp, "ttt_proj", None) is not None,
            "ttt_ntp_gate": hasattr(mlp, "ttt_ntp_gate"),
            "ttt_conv": hasattr(mlp, "ttt_conv"),
            "down_proj": hasattr(mlp, "down_proj"),
            "ttt_chunk": getattr(mlp, "ttt_chunk", None),
            "forward_stats_present": hasattr(mlp, "last_ttt_stats"),
        }
        result[str(index)] = present
    return result


def main():
    context = initialize()
    try:
        if context.world_size != 2 or torch.cuda.device_count() != 2:
            raise RuntimeError("TWO_GPU_RUNTIME_CONTRACT")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        model, config = build_pretrained(
            MODEL_PATH,
            context.device,
            train_clip_none=True,
            ttt_layers=TTT_LAYERS,
            ttt_chunk=4096,
        )
        torch.cuda.synchronize()
        build_seconds = time.perf_counter() - started
        pre = module_inventory(model)
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameters = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        decoder_ids = [id(block) for block in model.model.layers]
        if len(decoder_ids) != int(config.num_hidden_layers) or len(set(decoder_ids)) != len(decoder_ids):
            raise RuntimeError("DECODER_BLOCK_INVENTORY_MISMATCH")
        if any(not all(values[key] for key in ("ttt_proj", "ttt_ntp_gate", "ttt_conv", "down_proj"))
               or values["ttt_chunk"] != 4096 for values in pre.values()):
            raise RuntimeError("TTT_MODULE_INVENTORY_MISMATCH")
        mesh = make_device_mesh(context.device.type, context.world_size)
        fsdp = apply_fsdp2(model, mesh=mesh)
        post = module_inventory(model)
        blocks_wrapped = [
            hasattr(block, "set_requires_gradient_sync") for block in model.model.layers
        ]
        root_wrapped = hasattr(model, "set_requires_gradient_sync")
        if not root_wrapped or not all(blocks_wrapped):
            raise RuntimeError("FSDP2_RUNTIME_WRAP_MISSING")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=5.0e-6, betas=(0.9, 0.95),
            eps=1.0e-8, weight_decay=0.1, fused=True,
        )
        optimizer_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        l30 = {
            name: {
                "shape": list(parameter.shape),
                "requires_grad": parameter.requires_grad,
                "optimizer_owned": id(parameter) in optimizer_ids,
            }
            for name, parameter in model.model.layers[30].mlp.named_parameters()
            if name.startswith(("ttt_proj", "ttt_ntp_gate", "ttt_conv"))
        }
        if not l30 or not all(item["requires_grad"] and item["optimizer_owned"] for item in l30.values()):
            raise RuntimeError("L30_OPTIMIZER_PARAMETER_COVERAGE")
        local = {
            "rank": context.rank,
            "local_rank": context.local_rank,
            "logical_device": str(context.device),
            "physical_gpu": 2 + context.local_rank,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "build_seconds": build_seconds,
        }
        gathered = [None] * context.world_size
        dist.all_gather_object(gathered, local)
        if context.is_rank0:
            model_report = {
                "status": "PASS",
                "path": str(MODEL_PATH),
                "architecture": config.architectures,
                "hidden_size": config.hidden_size,
                "intermediate_size": config.intermediate_size,
                "num_attention_heads": config.num_attention_heads,
                "num_key_value_heads": config.num_key_value_heads,
                "num_hidden_layers": config.num_hidden_layers,
                "vocab_size": config.vocab_size,
                "dtype_metadata": str(config.torch_dtype),
                "total_parameter_count": total_parameters,
                "trainable_parameter_count": trainable_parameters,
                "ttt_layer_set": TTT_LAYERS,
                "l30_validation": "PASS",
            }
            coverage = {
                "status": "BUILD_AND_FSDP_PARAMETER_PASS_FORWARD_BACKWARD_PENDING",
                "requested_layers": TTT_LAYERS,
                "pre_fsdp": pre,
                "post_fsdp": post,
                "l30_parameters": l30,
                "l30_ttt_parameter_coverage": "PARAMETER_AND_OPTIMIZER_PASS_RUNTIME_FORWARD_BACKWARD_PENDING",
            }
            fsdp_report = {
                "status": "BUILD_WRAP_PASS_FORWARD_BACKWARD_PENDING",
                "wrapped_block_n": fsdp.decoder_blocks,
                "expected_block_n": config.num_hidden_layers,
                "all_decoder_blocks_wrapped": all(blocks_wrapped),
                "each_decoder_block_unique": len(set(decoder_ids)) == len(decoder_ids),
                "root_wrapped": root_wrapped,
                "optimizer_constructed_after_wrapping": True,
                "mixed_precision": fsdp.mixed_precision,
                "reduce_precision": fsdp.reduce_precision,
                "ranks": gathered,
            }
            AUDIT.mkdir(parents=True, exist_ok=True)
            (AUDIT / "MODEL_CONFIG_AUDIT.json").write_text(
                json.dumps(model_report, indent=2, sort_keys=True) + "\n"
            )
            (AUDIT / "TTT_LAYER_COVERAGE.json").write_text(
                json.dumps(coverage, indent=2, sort_keys=True) + "\n"
            )
            (AUDIT / "FSDP2_RUNTIME_AUDIT.json").write_text(
                json.dumps(fsdp_report, indent=2, sort_keys=True) + "\n"
            )
            print(json.dumps({"model": model_report, "fsdp": fsdp_report, "coverage": coverage}, sort_keys=True))
        dist.barrier()
    finally:
        shutdown()


if __name__ == "__main__":
    main()
