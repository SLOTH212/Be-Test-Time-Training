"""Reward-blind BF16 represented-displacement calibration.

No model, trajectory, reward or scorer is accepted by this module. Randomness is
local to a device-specific torch.Generator; global RNG states are untouched.
Reproducibility is defined within the same frozen PyTorch/device environment.
"""

from __future__ import annotations

import hashlib
import json
import math

import torch


ALGORITHM_VERSION = "gaussian_bf16_scalar_bracket_binary_v1"
MAX_BRACKET_STEPS = 24
MAX_BINARY_STEPS = 32
SEARCH_RELATIVE_TOLERANCE = 0.001
ACCEPTANCE_RELATIVE_TOLERANCE = 0.01


def derive_subseed(sample_id, layer, seed):
    """Stable unsigned 63-bit seed, independent of Python hash randomization."""
    payload = json.dumps(
        [ALGORITHM_VERSION, int(seed), str(sample_id), int(layer)],
        ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1), digest.hex()


@torch.no_grad()
def calibrate(native_bf16, ckpt_bf16, sample_id, layer, seed):
    """Return BF16 perturbed weight and JSON-safe diagnostics.

    Target and realized Frobenius norms use FP32 subtraction and accumulation.
    The Gaussian direction is generated and normalized in FP32 on the input
    device. Only its scalar multiplier changes during the search. Selection is
    by absolute norm error, with first-evaluated candidate breaking exact ties.
    At most 1 + MAX_BRACKET_STEPS + MAX_BINARY_STEPS trials are evaluated. A
    zero target is an explicit error: no nonzero direction can match it.
    """
    if native_bf16.dtype != torch.bfloat16 or ckpt_bf16.dtype != torch.bfloat16:
        raise TypeError("Both effective native and checkpoint weights must be BF16")
    if native_bf16.shape != ckpt_bf16.shape or native_bf16.device != ckpt_bf16.device:
        raise ValueError("Native/checkpoint weights must have identical shape/device")
    native = native_bf16.float()
    target = float(torch.linalg.vector_norm((ckpt_bf16.float() - native).reshape(-1)))
    if not math.isfinite(target) or target <= 0:
        raise ValueError(f"BLOCKED_MAGNITUDE_MATCHING: nonpositive/nonfinite target {target}")
    subseed, seed_digest = derive_subseed(sample_id, layer, seed)
    generator = torch.Generator(device=native.device)
    generator.manual_seed(subseed)
    direction = torch.randn(native.shape, dtype=torch.float32, device=native.device,
                            generator=generator)
    direction_norm = float(torch.linalg.vector_norm(direction.reshape(-1)))
    if not math.isfinite(direction_norm) or direction_norm <= 0:
        raise ValueError("BLOCKED_MAGNITUDE_MATCHING: invalid random direction norm")
    direction.div_(direction_norm)
    evaluations = 0
    best_alpha, best_norm, best_error = 0.0, 0.0, float("inf")

    def evaluate(alpha):
        nonlocal evaluations, best_alpha, best_norm, best_error
        represented = torch.add(native, direction, alpha=alpha).to(torch.bfloat16)
        realized = float(torch.linalg.vector_norm((represented.float() - native).reshape(-1)))
        evaluations += 1
        if not math.isfinite(realized):
            raise ValueError("BLOCKED_MAGNITUDE_MATCHING: nonfinite represented norm")
        error = abs(realized - target)
        # Never select a represented-zero candidate as a valid perturbation.
        if realized > 0 and error < best_error:
            best_alpha, best_norm, best_error = alpha, realized, error
        return realized

    initial_alpha = target
    initial_norm = evaluate(initial_alpha)
    lower, upper = 0.0, initial_alpha
    bracket_steps = 0
    binary_steps = 0
    upper_norm = initial_norm
    while upper_norm < target and bracket_steps < MAX_BRACKET_STEPS:
        if best_error / target <= SEARCH_RELATIVE_TOLERANCE:
            break
        lower = upper
        upper *= 2.0
        upper_norm = evaluate(upper)
        bracket_steps += 1
    bracketed = upper_norm >= target
    if bracketed:
        while binary_steps < MAX_BINARY_STEPS and best_error / target > SEARCH_RELATIVE_TOLERANCE:
            midpoint = (lower + upper) / 2.0
            if midpoint == lower or midpoint == upper:
                break
            realized = evaluate(midpoint)
            binary_steps += 1
            if realized < target:
                lower = midpoint
            else:
                upper = midpoint
    if not math.isfinite(best_error) or best_norm <= 0:
        raise ValueError("BLOCKED_MAGNITUDE_MATCHING: no represented-nonzero candidate")
    result = torch.add(native, direction, alpha=best_alpha).to(torch.bfloat16)
    final_norm = float(torch.linalg.vector_norm((result.float() - native).reshape(-1)))
    if final_norm != best_norm:
        raise RuntimeError("BLOCKED_MAGNITUDE_MATCHING: chosen candidate not reproducible")
    relative_error = abs(final_norm - target) / target
    diagnostics = dict(
        algorithm_version=ALGORITHM_VERSION,
        sample_id=str(sample_id), layer=int(layer), seed=int(seed),
        subseed=subseed, seed_payload_sha256=seed_digest,
        shape=list(native.shape), device=str(native.device), execution_dtype="bfloat16",
        norm_dtype="float32", requested_norm=target, target_norm=target,
        initial_proposed_norm=target,
        initial_realized_norm=initial_norm,
        initial_relative_norm_error=abs(initial_norm - target) / target,
        calibrated_alpha=best_alpha,
        calibrated_proposed_norm=float(torch.linalg.vector_norm(direction.reshape(-1))) * best_alpha,
        realized_norm=final_norm, relative_norm_error=relative_error,
        calibration_iterations=evaluations, bracket_steps=bracket_steps,
        binary_steps=binary_steps, bracketed=bracketed,
        max_bracket_steps=MAX_BRACKET_STEPS, max_binary_steps=MAX_BINARY_STEPS,
        search_relative_tolerance=SEARCH_RELATIVE_TOLERANCE,
        matched_within_1pct=relative_error <= ACCEPTANCE_RELATIVE_TOLERANCE,
        represented_nonzero=final_norm > 0,
        reward_used=False,
    )
    return result, diagnostics


def self_test():
    """Small CPU-only check of magnitude, local RNG and repeat determinism."""
    rng = torch.Generator(device="cpu").manual_seed(8675309)
    native = (torch.randn((256, 256), generator=rng) * 0.1).to(torch.bfloat16)
    checkpoint = (native.float() + torch.randn(native.shape, generator=rng) * 0.002).to(torch.bfloat16)
    state = torch.random.get_rng_state().clone()
    first, info = calibrate(native, checkpoint, "self-test", 6, 20260926)
    second, info2 = calibrate(native, checkpoint, "self-test", 6, 20260926)
    other, _ = calibrate(native, checkpoint, "self-test", 6, 20260927)
    assert torch.equal(state, torch.random.get_rng_state()), "Global RNG mutated"
    assert torch.equal(first, second) and info == info2, "Not deterministic"
    assert not torch.equal(first, native), "Represented-zero perturbation"
    assert not torch.equal(first, other), "Seed has no effect"
    assert info["matched_within_1pct"], info
    try:
        calibrate(native, native, "zero-target", 6, 20260926)
    except ValueError as exc:
        assert "BLOCKED_MAGNITUDE_MATCHING" in str(exc)
    else:
        raise AssertionError("Zero target did not block")
    return dict(status="PASS", checks=["cpu_bf16_magnitude", "exact_repeat",
                "global_rng_unchanged", "seed_changes_direction", "nonzero", "zero_target_blocks"],
                diagnostics=info)


if __name__ == "__main__":
    print(json.dumps(self_test(), indent=2, allow_nan=False))
