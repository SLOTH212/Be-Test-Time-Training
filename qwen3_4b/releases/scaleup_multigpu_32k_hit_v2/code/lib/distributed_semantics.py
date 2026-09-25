"""World-size independent record ownership and globally correct loss scaling."""
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class BatchPlan:
    dp_size: int
    micro_batch: int
    gradient_accumulation: int

    def __post_init__(self):
        if min(self.dp_size, self.micro_batch, self.gradient_accumulation) < 1:
            raise ValueError("batch dimensions must be positive")

    @property
    def global_batch(self) -> int:
        return self.dp_size * self.micro_batch * self.gradient_accumulation

    def owner(self, canonical_position: int) -> Tuple[int, int, int, int]:
        """Return update, accumulation slot, rank, local micro index."""
        if canonical_position < 0:
            raise ValueError("canonical_position must be non-negative")
        update, offset = divmod(canonical_position, self.global_batch)
        accumulation_slot, within_slot = divmod(offset, self.dp_size * self.micro_batch)
        rank, local_micro_index = divmod(within_slot, self.micro_batch)
        return update, accumulation_slot, rank, local_micro_index

    def rank_positions(self, rank: int, start: int, stop: int) -> Iterator[Tuple[int, int, int]]:
        """Yield (position, update, accumulation_slot), without shuffle or duplication."""
        if not 0 <= rank < self.dp_size or not 0 <= start <= stop:
            raise ValueError("invalid rank or interval")
        for position in range(start, stop):
            update, slot, owner, _ = self.owner(position)
            if owner == rank:
                yield position, update, slot


def assert_batch_equation(global_batch: int, micro_batch: int, dp_size: int, accumulation: int) -> None:
    actual = micro_batch * dp_size * accumulation
    if global_batch != actual:
        raise RuntimeError(f"GLOBAL_BATCH_EQUATION expected={global_batch} actual={actual}")


def global_count(local_count: int, device: torch.device) -> int:
    value = torch.tensor(int(local_count), dtype=torch.int64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return int(value.item())


def globally_scaled_loss(local_numerator: torch.Tensor, local_count: int,
                         dp_size: Optional[int] = None) -> Tuple[torch.Tensor, int]:
    """Scale a local sum so FSDP's averaged grads equal the global token mean.

    The denominator is reduced without autograd. Multiplication by dp_size
    compensates for the gradient average performed by data parallelism.
    """
    world = dp_size or (dist.get_world_size() if dist.is_initialized() else 1)
    denominator = global_count(local_count, local_numerator.device)
    if denominator <= 0:
        raise RuntimeError("GLOBAL_LOSS_DENOMINATOR_ZERO")
    return local_numerator * (world / denominator), denominator


def differentiable_empty_step(parameters, device: torch.device) -> torch.Tensor:
    """A zero loss touching every parameter for an unfilled final rank slot."""
    zero = torch.zeros((), device=device)
    for parameter in parameters:
        zero = zero + parameter.reshape(-1)[:1].float().sum() * 0.0
    return zero
