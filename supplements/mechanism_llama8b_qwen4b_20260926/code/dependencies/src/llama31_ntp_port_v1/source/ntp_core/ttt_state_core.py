"""Functional per-document Static In-Place TTT state transition."""
from dataclasses import dataclass
import torch
import torch.nn.functional as F

GATED_NTP_TARGET = "gated_next_position_hidden"
AUTHORITATIVE_BACKEND = "authoritative_document_chunk_loop"

@dataclass
class TTTState:
    weight: torch.Tensor

@dataclass
class TTTStats:
    update_count: int
    valid_tokens: int
    pre_clip_norm: float
    post_clip_norm: float
    was_clipped: bool

def compute_lm_aligned_target(x, kernel, valid_mask):
    """Official per-chunk LM target using Conv1d cross-correlation.

    For K=5,padding=2, output[t] = sum_j kernel[j] * x[t+j-2].
    Thus tap j=3 is the next-token target x[t+1].  The caller supplies one
    chunk/document segment, so unavailable future positions are zero and no
    later chunk or document can leak into this target.
    """
    mask = valid_mask.to(x.dtype).unsqueeze(-1)
    source = (x * mask).transpose(1, 2)
    pad = kernel.shape[-1] // 2
    result = F.conv1d(source, kernel, padding=pad, groups=x.shape[-1]).transpose(1, 2)
    return result * mask

# Backward-compatible name; semantics are LM-aligned, not past-causal.
causal_depthwise_target = compute_lm_aligned_target

def build_chunk_local_ntp_pairs(target_source, valid_mask):
    """Return document/chunk-local ``t -> t+1`` targets and pair validity.

    The caller supplies exactly one document-local chunk. Returning the
    ``[..., 1:, :]`` view excludes the chunk's last token without allocating a
    record-wide shifted target.
    """
    if target_source.ndim != 3 or valid_mask.shape != target_source.shape[:2]:
        raise ValueError("target_source/valid_mask must be [B,T,H]/[B,T]")
    valid = valid_mask.to(device=target_source.device, dtype=torch.bool)
    if target_source.shape[1] < 2:
        return target_source[:, :0], valid[:, :0]
    return target_source[:, 1:], valid[:, :-1] & valid[:, 1:]

def ttt_chunk_step(hidden, target_source, base_weight, target_projection,
                   conv_kernel, valid_mask, state=None, learning_rate=3.0,
                   enabled=True, delta_clip_norm=None, target_type="conv",
                   target_gate=None):
    if enabled and hidden.shape[0] != 1:
        raise ValueError("TTT-on currently requires batch_size=1 because fast weights are per sample")
    current = base_weight if state is None else state.weight
    output = F.linear(hidden, current)  # apply before this chunk's update
    valid = valid_mask.bool()
    if not enabled or not bool(valid.any()):
        return output, TTTState(current), TTTStats(0, int(valid.sum()), 0., 0., False)
    if target_type == "conv":
        target = compute_lm_aligned_target(target_source, conv_kernel, valid)
        update_hidden = hidden
        update_valid = valid
    elif target_type == GATED_NTP_TARGET:
        if target_gate is None or target_gate.ndim != 1 or target_gate.shape[0] != target_source.shape[-1]:
            raise ValueError("gated NTP requires target_gate with shape [hidden_size]")
        target, update_valid = build_chunk_local_ntp_pairs(target_source, valid)
        target = target * target_gate.to(device=target.device, dtype=target.dtype)
        update_hidden = hidden[:, :-1]
    else:
        raise ValueError(f"unknown TTT target_type: {target_type!r}")
    mask = update_valid.to(hidden.dtype).unsqueeze(-1)
    delta = torch.einsum(
        "bti,btd,de->ei", update_hidden * mask, target * mask, target_projection
    ) * learning_rate
    pre = torch.linalg.vector_norm(delta.float())
    clipped = False
    if delta_clip_norm is not None:
        scale = (float(delta_clip_norm) / pre.clamp_min(torch.finfo(pre.dtype).tiny)).clamp(max=1.)
        clipped = bool((scale < 1).item())
        delta = delta * scale.to(delta.dtype)
    post = torch.linalg.vector_norm(delta.float())
    return output, TTTState(current + delta), TTTStats(
        1, int(update_valid.sum()), float(pre.detach()), float(post.detach()), clipped)

def run_document(hidden, target_source, base_weight, target_projection,
                 conv_kernel, valid_mask, chunk_size=1024, learning_rate=3.0,
                 enabled=True, delta_clip_norm=None, initial_state=None,
                 target_type="conv", target_gate=None,
                 allow_test_override=False):
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if enabled and hidden.shape[0] != 1:
        raise ValueError("TTT-on currently requires batch_size=1 because fast weights are per sample")
    state, outputs, stats = (TTTState(base_weight) if initial_state is None else initial_state), [], []
    for start in range(0, hidden.shape[1], chunk_size):
        end = min(start + chunk_size, hidden.shape[1])
        out, state, stat = ttt_chunk_step(
            hidden[:, start:end], target_source[:, start:end], base_weight,
            target_projection, conv_kernel, valid_mask[:, start:end], state,
            learning_rate, enabled, delta_clip_norm,
            target_type, target_gate)
        outputs.append(out); stats.append(stat)
    return torch.cat(outputs, 1), state, stats

def run_sequence_with_boundaries(hidden, target_source, base_weight,
                                 target_projection, conv_kernel, valid_mask,
                                 document_boundaries=None, **kwargs):
    """Run independent documents; each [start,end) resets to checkpoint W0."""
    if hidden.shape[0] != 1:
        raise ValueError("TTT-on currently requires batch_size=1 because fast weights are per sample")
    boundaries = document_boundaries or [(0, hidden.shape[1])]
    output = torch.nn.functional.linear(hidden, base_weight)
    all_stats = []
    for start, end in boundaries:
        if not (0 <= start <= end <= hidden.shape[1]):
            raise ValueError("invalid document boundary")
        doc_out, _, stats = run_document(
            hidden[:, start:end], target_source[:, start:end], base_weight,
            target_projection, conv_kernel, valid_mask[:, start:end], **kwargs)
        output = torch.cat((output[:, :start], doc_out, output[:, end:]), dim=1)
        all_stats.append(stats)
    return output, all_stats
