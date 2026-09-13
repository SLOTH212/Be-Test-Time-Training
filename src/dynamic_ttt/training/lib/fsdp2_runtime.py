"""Composable FSDP2 policy for the recovered Qwen3 authority."""
from dataclasses import dataclass
from types import MethodType
from typing import Callable, Optional

import torch


@dataclass(frozen=True)
class FSDP2Inventory:
    decoder_blocks: int
    root_wrapped: bool
    mixed_precision: str
    reduce_precision: str


def apply_fsdp2(model, mesh=None, fully_shard_impl: Optional[Callable] = None) -> FSDP2Inventory:
    if fully_shard_impl is None:
        from torch.distributed.fsdp import (
            MixedPrecisionPolicy,
            fully_shard,
            register_fsdp_forward_method,
        )
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        fully_shard_impl = fully_shard
    else:
        policy = {"param_dtype": torch.bfloat16, "reduce_dtype": torch.float32}
        register_fsdp_forward_method = None
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None or len(layers) == 0:
        raise RuntimeError("FSDP2_DECODER_BLOCK_DISCOVERY_FAILED")
    for block in layers:
        fully_shard_impl(block, mesh=mesh, mp_policy=policy)
    fully_shard_impl(model, mesh=mesh, mp_policy=policy)
    if register_fsdp_forward_method is not None:
        def _fsdp_hidden_forward(self, *args, **kwargs):
            """Run the decoder through root FSDP hooks without materializing logits."""
            return self.model(*args, **kwargs)

        model._fsdp_hidden_forward = MethodType(_fsdp_hidden_forward, model)
        register_fsdp_forward_method(model, "_fsdp_hidden_forward")
    return FSDP2Inventory(len(layers), True, "bf16", "fp32")


def make_device_mesh(device_type: str, world_size: int):
    from torch.distributed.device_mesh import init_device_mesh
    return init_device_mesh(device_type, (world_size,), mesh_dim_names=("dp",))
