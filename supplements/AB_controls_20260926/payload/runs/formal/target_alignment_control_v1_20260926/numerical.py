"""Reward-blind scalar calibration and chunk-local target randomization.

The calibration returns the closest *evaluated* candidate; it makes no global
optimality or monotonicity claim for the BF16 clipping/addition pipeline.
"""
import hashlib
import json
import math

import torch


def original_apply(raw, current, clip):
    """Apply exactly the frozen core's clipping and state-add operation order."""
    pre = torch.linalg.vector_norm(raw.float())
    clipped = raw
    was_clipped = False
    if clip is not None:
        scale = (float(clip) / pre.clamp_min(torch.finfo(pre.dtype).tiny)).clamp(max=1.)
        was_clipped = bool((scale < 1).item())
        clipped = raw * scale.to(raw.dtype)
    post = torch.linalg.vector_norm(clipped.float())
    after = current + clipped
    represented = torch.linalg.vector_norm(after.float() - current.float())
    return after, clipped, float(pre), float(post), float(represented), was_clipped


@torch.no_grad()
def calibrate(raw, current, target_norm, clip):
    """Select a nonnegative scalar using only write norms, never task reward."""
    target = float(target_norm)
    if not math.isfinite(target) or target < 0:
        raise ValueError('target_norm must be finite and nonnegative')
    raw_norm = float(torch.linalg.vector_norm(raw.float()))
    if not math.isfinite(raw_norm):
        raise ValueError('raw update has a nonfinite norm')
    if clip is not None and (not math.isfinite(float(clip)) or float(clip) < 0):
        raise ValueError('clip must be finite and nonnegative')

    evaluated = {}
    invalid = []

    def evaluate(alpha):
        alpha = float(alpha)
        if alpha in evaluated or alpha in invalid:
            return
        if not math.isfinite(alpha) or alpha < 0:
            invalid.append(alpha)
            return
        # Do not cast alpha to a separately selected dtype or scale post-clip.
        after, clipped, pre, post, represented, was_clipped = original_apply(
            raw * alpha, current, clip)
        # Keep scalar metadata only; no evaluated candidate tensor is retained.
        del after, clipped
        if not all(math.isfinite(v) for v in (pre, post, represented)):
            invalid.append(alpha)
            return
        evaluated[alpha] = dict(raw_norm=pre, clipped_norm=post,
                                represented_norm=represented,
                                was_clipped=was_clipped,
                                absolute_error=abs(represented - target))

    evaluate(0.)
    rounds = 0
    if target != 0. and raw_norm != 0.:
        evaluate(1.)
        alpha_clip = float(clip) / raw_norm if clip is not None else 1.
        for exponent in range(-16, 17):
            evaluate(math.ldexp(alpha_clip, exponent))
        for _ in range(12):
            rounds += 1
            best = min(evaluated, key=lambda a: (evaluated[a]['absolute_error'], a))
            ordered = sorted(evaluated)
            index = ordered.index(best)
            proposals = []
            if index:
                lower = ordered[index - 1]
                proposals.append(best / 2. if lower == 0.
                                 else math.exp((math.log(lower) + math.log(best)) / 2.))
            if index + 1 < len(ordered):
                upper = ordered[index + 1]
                proposals.append(upper / 2. if best == 0.
                                 else math.exp((math.log(best) + math.log(upper)) / 2.))
            for alpha in proposals:
                evaluate(alpha)
    if not evaluated:
        raise RuntimeError('No finite calibration candidate, including zero')
    best = min(evaluated, key=lambda a: (evaluated[a]['absolute_error'], a))
    after, clipped, pre, post, represented, was_clipped = original_apply(raw * best, current, clip)
    if represented != evaluated[best]['represented_norm']:
        raise RuntimeError('Final scalar candidate did not reproduce its evaluated norm')
    relative_error = abs(represented - target) / target if target else (0. if represented == 0. else math.inf)
    ordered = sorted(evaluated)
    index = ordered.index(best)
    diagnostics = dict(
        alpha=best, relative_error=relative_error,
        matchedwithin=relative_error <= .01,
        matched_within_1pct=relative_error <= .01,
        iterations=rounds, n_evaluations=len(evaluated),
        n_invalid_candidates=len(invalid), invalid_alphas=invalid,
        target_norm=target, shuffle_raw_norm=raw_norm,
        scaled_raw_norm=pre, clipped_norm=post, represented_norm=represented,
        was_clipped=was_clipped, clip_saturated=was_clipped,
        observedmax=max(v['represented_norm'] for v in evaluated.values()),
        observed_min=min(v['represented_norm'] for v in evaluated.values()),
        zero_native=target == 0., zero_shuffle=raw_norm == 0.,
        alpha_min=ordered[0], alpha_max=ordered[-1],
        lower_neighbor=ordered[index-1] if index else None,
        upper_neighbor=ordered[index+1] if index+1 < len(ordered) else None,
        selection='closest_evaluated_candidate; ties_smaller_alpha',
        unattainable_reason=('zero_shuffle_nonzero_native' if raw_norm == 0. and target > 0.
                            else 'not_matched_in_frozen_bounded_search' if relative_error > .01 else None),
        evaluated_candidates=[dict(alpha=a, **evaluated[a]) for a in ordered],
    )
    if target == 0. and represented != 0.:
        raise RuntimeError('Native-zero calibration introduced a nonzero write')
    return after, clipped, diagnostics


def deterministic_permutation(n, sample_id, layer, chunk_index, seed):
    """Use a local CPU generator, with no process-global RNG state changes."""
    n = int(n)
    if n < 0:
        raise ValueError('n_targets cannot be negative')
    key = json.dumps(['target_alignment_control_v1', str(sample_id), int(layer),
                      int(chunk_index), int(seed), n], separators=(',', ':'),
                     ensure_ascii=False).encode('utf-8')
    derived_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], 'little') & ((1 << 63)-1)
    generator = torch.Generator(device='cpu').manual_seed(derived_seed)
    identity = torch.arange(n, dtype=torch.long, device='cpu')
    attempts = 0
    fallback = False
    perm = identity
    if n > 1:
        for attempts in range(1, 129):
            perm = torch.randperm(n, generator=generator, device='cpu')
            if not bool((perm == identity).any()):
                break
        else:
            perm = torch.roll(identity, shifts=1)
            fallback = True
    fixed_points = int((perm == identity).sum())
    # Explicit little-endian signed int64 hash contract, independent of host.
    payload = b''.join(int(value).to_bytes(8, 'little', signed=True) for value in perm.tolist())
    diagnostics = dict(permutation_hash=hashlib.sha256(payload).hexdigest(),
                       n_targets=n, fixed_points=fixed_points,
                       derived_seed=derived_seed, attempts=attempts,
                       cyclic_fallback=fallback,
                       hash_encoding='little_endian_int64',
                       singleton_identity=n <= 1)
    return perm, diagnostics


def self_test():
    """Small CPU-only checks for exact arithmetic and deterministic randomization."""
    current = torch.tensor([[.0, .25], [-.125, .5]], dtype=torch.bfloat16)
    raw = torch.tensor([[.03125, -.0625], [.125, -.25]], dtype=torch.bfloat16)
    clip = .05
    pre = torch.linalg.vector_norm(raw.float())
    scale = (clip / pre.clamp_min(torch.finfo(pre.dtype).tiny)).clamp(max=1.)
    expected_clip = raw * scale.to(raw.dtype)
    after, clipped, _, _, target, _ = original_apply(raw, current, clip)
    assert torch.equal(clipped, expected_clip)
    assert torch.equal(after, current + expected_clip)
    first, _, diag = calibrate(raw, current, target, clip)
    second, _, repeat = calibrate(raw, current, target, clip)
    assert torch.equal(first, second) and diag == repeat
    assert diag['relative_error'] == 0.
    zero, _, zero_diag = calibrate(raw, current, 0., clip)
    assert torch.equal(zero, current) and zero_diag['alpha'] == 0.
    _, _, unreachable = calibrate(torch.zeros_like(raw), current, target, clip)
    assert unreachable['unattainable_reason'] == 'zero_shuffle_nonzero_native'
    rng_before = torch.get_rng_state().clone()
    for n in (0, 1, 2, 31, 1023):
        perm, info = deterministic_permutation(n, 'sample', 6, 3, 20260926)
        repeated, repeated_info = deterministic_permutation(n, 'sample', 6, 3, 20260926)
        assert torch.equal(perm, repeated) and info == repeated_info
        assert torch.equal(torch.sort(perm).values, torch.arange(n))
        assert info['fixed_points'] == (n if n <= 1 else 0)
    assert torch.equal(rng_before, torch.get_rng_state())
    return dict(status='PASS', device='cpu', exact_original_arithmetic=True,
                deterministic_calibration=True, native_zero_exact=True,
                permutation_deterministic=True, global_rng_unchanged=True)


if __name__ == '__main__':
    print(json.dumps(self_test(), sort_keys=True))
