#!/usr/bin/env python3
"""Two-GPU FSDP2 forward-only validation on exact 32K synthetic Stage1 records."""
import json
import math
import os
import sys
import time
from pathlib import Path

if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3":
    raise RuntimeError("PHYSICAL_GPU_ALLOWLIST_VIOLATION")

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))
from lib.distributed_runtime import initialize, shutdown
from lib.fsdp2_runtime import apply_fsdp2, make_device_mesh
from lib.model_factory import build_pretrained
from workers.distributed_train_worker import fused_sum, stage1_document_boundaries

MODEL_PATH = Path(os.path.expandvars("${TTT_MODEL_ROOT}/Qwen3-4B-Base"))
FIXTURE = ROOT / "fixtures/stage1_32k_debug.jsonl"
AUDIT = ROOT / "audits/hit_l3_l4_v1"
TTT_LAYERS = [0, 6, 12, 18, 24, 30]
CONTEXT = 32768
CHUNK = 4096


def flatten_stats(value):
    return [item for document in value for item in document]


def main():
    context = initialize()
    try:
        if context.world_size != 2 or torch.cuda.device_count() != 2:
            raise RuntimeError("TWO_GPU_RUNTIME_CONTRACT")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        model, config = build_pretrained(
            MODEL_PATH, context.device, train_clip_none=True,
            ttt_layers=TTT_LAYERS, ttt_chunk=CHUNK,
        )
        mesh = make_device_mesh(context.device.type, context.world_size)
        inventory = apply_fsdp2(model, mesh=mesh)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        rows = [json.loads(line) for line in FIXTURE.read_text().splitlines()]
        row = rows[context.rank]
        ids = tokenizer(
            row["text"], add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(context.device)
        if ids.shape != (1, CONTEXT) or int(row["token_count"]) != CONTEXT:
            raise RuntimeError(f"FORWARD_TOKEN_ALIGNMENT shape={tuple(ids.shape)}")
        bounds = stage1_document_boundaries(row, CONTEXT)
        model.eval()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            local_numerator = fused_sum(
                model, ids, document_boundaries=bounds
            ).detach().double()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        pair = torch.stack((
            local_numerator,
            torch.tensor(float(CONTEXT - 1), dtype=torch.float64, device=context.device),
        ))
        dist.all_reduce(pair, op=dist.ReduceOp.SUM)
        global_loss = float((pair[0] / pair[1]).item())
        if not math.isfinite(global_loss):
            raise RuntimeError("NONFINITE_FORWARD_LOSS")
        traces = {}
        for index in TTT_LAYERS:
            mlp = model.model.layers[index].mlp
            if not hasattr(mlp, "last_ttt_stats"):
                raise RuntimeError(f"TTT_FORWARD_NOT_EXECUTED layer={index}")
            stats = flatten_stats(mlp.last_ttt_stats)
            traces[str(index)] = [
                {
                    "chunk_index": chunk_index,
                    "update_count": stat.update_count,
                    "valid_pairs": stat.valid_tokens,
                    "pre_clip_norm": stat.pre_clip_norm,
                    "post_clip_norm": stat.post_clip_norm,
                    "was_clipped": stat.was_clipped,
                }
                for chunk_index, stat in enumerate(stats)
            ]
            if len(stats) != 8 or any(stat.update_count != 1 for stat in stats):
                raise RuntimeError(f"TTT_CHUNK_TRACE_MISMATCH layer={index}")
        local = {
            "rank": context.rank,
            "local_rank": context.local_rank,
            "logical_device": str(context.device),
            "physical_gpu": 2 + context.local_rank,
            "sample_id": row["sample_id"],
            "local_numerator": float(local_numerator.item()),
            "local_denominator": CONTEXT - 1,
            "wall_seconds": elapsed,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "ttt_trace": traces,
        }
        gathered = [None] * context.world_size
        dist.all_gather_object(gathered, local)
        if context.is_rank0:
            report = {
                "status": "PASS",
                "world_size": context.world_size,
                "cuda_visible_devices": "2,3",
                "context_length": CONTEXT,
                "ttt_chunk_size": CHUNK,
                "ttt_layers": TTT_LAYERS,
                "fsdp2_wrapped_blocks": inventory.decoder_blocks,
                "fsdp2_root_wrapped": inventory.root_wrapped,
                "global_numerator": float(pair[0].item()),
                "global_denominator": int(pair[1].item()),
                "global_loss": global_loss,
                "ranks": gathered,
                "l30_forward_execution": "PASS",
                "formal_training": False,
                "optimizer_steps": 0,
            }
            AUDIT.mkdir(parents=True, exist_ok=True)
            (AUDIT / "L3_32K_FORWARD_REPORT.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n"
            )
            chunk_report = {
                "status": "RUNTIME_PASS",
                "context_length": CONTEXT,
                "chunk_size": CHUNK,
                "chunk_count": 8,
                "tail_tokens": 0,
                "apply_then_update": True,
                "ttt_layers": TTT_LAYERS,
                "rank0_trace": gathered[0]["ttt_trace"],
            }
            (AUDIT / "CHUNK_TRACE.json").write_text(
                json.dumps(chunk_report, indent=2, sort_keys=True) + "\n"
            )
            print(json.dumps({
                "status": "PASS",
                "global_loss": global_loss,
                "global_denominator": int(pair[1].item()),
                "rank_memory": [
                    {
                        "physical_gpu": item["physical_gpu"],
                        "peak_allocated_bytes": item["peak_allocated_bytes"],
                        "peak_reserved_bytes": item["peak_reserved_bytes"],
                        "wall_seconds": item["wall_seconds"],
                    }
                    for item in gathered
                ],
                "l30_forward_execution": "PASS",
            }, sort_keys=True))
        dist.barrier()
    finally:
        shutdown()


if __name__ == "__main__":
    main()
