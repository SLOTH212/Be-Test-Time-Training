"""DTensor-aware clipping plus a sharded-reference implementation for tests."""
import math
import torch
import torch.distributed as dist


def clip_grad_norm_fsdp2(parameters, max_norm: float) -> float:
    parameters = [p for p in parameters if p.grad is not None]
    if not parameters:
        return 0.0
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    if hasattr(norm, "full_tensor"):
        norm = norm.full_tensor()
    value = float(norm.detach().float().cpu())
    if not math.isfinite(value):
        raise RuntimeError("NONFINITE_GLOBAL_GRADIENT_NORM")
    return value


def clip_local_shards_reference(parameters, max_norm: float, device: torch.device) -> float:
    local_sq = torch.zeros((), dtype=torch.float64, device=device)
    grads = []
    for parameter in parameters:
        if parameter.grad is not None:
            grads.append(parameter.grad)
            local_sq += parameter.grad.detach().double().pow(2).sum()
    if dist.is_initialized():
        dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
    norm = local_sq.sqrt()
    coefficient = min(1.0, max_norm / (float(norm) + 1e-6))
    for grad in grads:
        grad.mul_(coefficient)
    return float(norm)
