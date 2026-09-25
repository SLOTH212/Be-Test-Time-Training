#!/usr/bin/env python3
"""Read-only state instrumentation for the frozen formal dynamic backend.

Scientific tensor computation is copied from ``install_dynamic_forward`` in the
formal Phase-D executor.  The only intervention is an explicit local TTTState
replacement at a predeclared chunk boundary.  Metrics are detached reductions;
no full dense state or update tensor is serialized.
"""
from __future__ import annotations

import types

LAYERS = [0, 6, 12, 18, 24]
ACTION_LAYERS = {
    "OFF": [], "L0": [0], "L6": [6], "L12": [12], "L18": [18],
    "L24": [24], "ALL": LAYERS,
}


def _scalar_stats(torch, value):
    detached = value.detach().float()
    return {
        "norm": torch.linalg.vector_norm(detached),
        "mean": detached.mean(),
        "std": detached.std(unbiased=False),
        "max_abs": detached.abs().max(),
    }


def _norm(torch, value):
    return torch.linalg.vector_norm(value.detach().float())


def _dot(torch, left, right):
    return torch.sum(left.detach().float() * right.detach().float())


def install_instrumented_dynamic_forward(model, tensor_hash):
    import torch
    import torch.nn.functional as F
    from hf_models.hf_qwen3.ttt_state_core import TTTState, TTTStats, ttt_chunk_step

    for layer in LAYERS:
        mlp = model.model.layers[layer].mlp
        mlp._dynamic_sequence = None
        mlp._branch_id = None
        mlp._capture_identity = False
        mlp._events = []
        mlp._state_events = []
        mlp._reset_events = []
        mlp._generation_forward_calls = 0
        mlp._captured_final_state = None
        mlp._instrumentation_enabled = True
        mlp._reset_boundary_chunk = None

        def patched(self, x, t=None, past_w=None, enabled=True):
            # Exact formal computation begins.
            h = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
            if not hasattr(self, "ttt_conv") or not enabled:
                result = self.down_proj(h)
                # Layers unused by an entire trajectory take this exact formal
                # non-TTT path.  Hooks only summarize its already-computed data.
                # Never-selected layers have no target; multi-token means prompt.
                if self._instrumentation_enabled and self._dynamic_sequence is not None and x.shape[1] >= self.ttt_chunk:
                    full_len = (x.shape[1] // self.ttt_chunk) * self.ttt_chunk
                    n = full_len // self.ttt_chunk
                    if len(self._dynamic_sequence) != n:
                        raise RuntimeError(f"ACTION_SEQUENCE_LENGTH layer={self.layer_idx} n={n}")
                    self._state_events = []
                    self._reset_events = []
                    for j, start in enumerate(range(0, full_len, self.ttt_chunk)):
                        end = start + self.ttt_chunk
                        action = self._dynamic_sequence[j]
                        event = {
                            "layer": int(self.layer_idx), "chunk_index": j + 1,
                            "selected_action": action, "active_update": False,
                            "reset_applied": bool(self._reset_boundary_chunk == j + 1),
                            "fast_weight_delta_norm_before": torch.zeros((), device=x.device),
                            "fast_weight_delta_norm_after": torch.zeros((), device=x.device),
                            "fast_weight_state_growth": torch.zeros((), device=x.device),
                            "raw_update_fro_norm": torch.zeros((), device=x.device),
                            "clipped_update_fro_norm": torch.zeros((), device=x.device),
                            "clip_ratio": torch.ones((), device=x.device),
                            "update_dot_cumulative_state": None,
                            "update_cos_cumulative_state": None,
                            "update_dot_previous_same_layer": None,
                            "update_cos_previous_same_layer": None,
                            "cumulative_state_norm_before": torch.zeros((), device=x.device),
                            "cumulative_state_norm_after": torch.zeros((), device=x.device),
                            "hidden_input_stats": _scalar_stats(torch, x[:, start:end]),
                            "mlp_intermediate_stats": _scalar_stats(torch, h[:, start:end]),
                            "mlp_output_stats": _scalar_stats(torch, result[:, start:end]),
                            "target_stats": _scalar_stats(torch, t[:, start:end]) if t is not None else None,
                            "gate_stats": None,
                        }
                        self._state_events.append(event)
                        if self._reset_boundary_chunk == j + 1:
                            zero = torch.zeros((), device=x.device)
                            self._reset_events.append({
                                "layer": int(self.layer_idx), "chunk_index": j + 1,
                                "pre_reset_delta_norm": zero, "post_reset_delta_norm": zero,
                                "post_reset_max_abs_residual": zero,
                                "unused_layer_implicitly_at_base": True,
                            })
                return result
            if x.shape[0] != 1:
                raise ValueError("TTT-on requires batch_size=1")
            present = self.down_proj.weight if past_w is None else past_w
            if t is None:
                self._generation_forward_calls += 1
                return F.linear(h, present, self.down_proj.bias), present
            full_len = (x.shape[1] // self.ttt_chunk) * self.ttt_chunk
            n = full_len // self.ttt_chunk
            if self._dynamic_sequence is None or len(self._dynamic_sequence) != n:
                raise RuntimeError(f"ACTION_SEQUENCE_LENGTH layer={self.layer_idx} n={n}")
            outputs = []
            state = TTTState(present)
            stats = []
            events = []
            state_events = []
            reset_events = []
            version = 0
            previous_applied_update = None
            projection = self.ttt_proj.weight if self.ttt_proj is not None else torch.eye(
                self.hidden_size, dtype=x.dtype, device=x.device)
            base = self.down_proj.weight
            gate_summary = _scalar_stats(torch, self.ttt_ntp_gate) if hasattr(self, "ttt_ntp_gate") else None
            for j, start in enumerate(range(0, full_len, self.ttt_chunk)):
                end = start + self.ttt_chunk
                action = self._dynamic_sequence[j]
                active = int(self.layer_idx) in ACTION_LAYERS[action]
                reset_applied = self._reset_boundary_chunk == j + 1
                pre_reset_delta = None
                if reset_applied:
                    pre_reset_delta = state.weight - base
                    state = TTTState(base)
                    residual = state.weight - base
                    reset_events.append({
                        "layer": int(self.layer_idx), "chunk_index": j + 1,
                        "pre_reset_delta_norm": _norm(torch, pre_reset_delta),
                        "post_reset_delta_norm": _norm(torch, residual),
                        "post_reset_max_abs_residual": residual.detach().abs().max().float(),
                        "unused_layer_implicitly_at_base": False,
                    })
                current = state.weight
                cumulative_before = current - base
                norm_before = _norm(torch, cumulative_before)
                pre_hash = tensor_hash(current) if self._capture_identity else None
                pre_identity = f"{self._branch_id}:L{self.layer_idx}:v{version}"
                if active:
                    out, proposal, stat = ttt_chunk_step(
                        h[:, start:end], t[:, start:end], base, projection, self.ttt_conv.weight,
                        torch.ones((1, self.ttt_chunk), dtype=torch.bool, device=x.device),
                        state, self.ttt_lr, True, self.ttt_update_clip_norm,
                        self.ttt_target_type, getattr(self, "ttt_ntp_gate", None))
                    applied_update = proposal.weight - current
                    cumulative_after = proposal.weight - base
                    norm_after = _norm(torch, cumulative_after)
                    update_norm = _norm(torch, applied_update)
                    dot_state = _dot(torch, applied_update, cumulative_before)
                    denom_state = update_norm * norm_before
                    cos_state = torch.where(denom_state > 0, dot_state / denom_state, torch.full_like(dot_state, float("nan")))
                    if previous_applied_update is None:
                        dot_previous = None
                        cos_previous = None
                    else:
                        previous_norm = _norm(torch, previous_applied_update)
                        dot_previous = _dot(torch, applied_update, previous_applied_update)
                        denom_previous = update_norm * previous_norm
                        cos_previous = torch.where(denom_previous > 0, dot_previous / denom_previous, torch.full_like(dot_previous, float("nan")))
                    previous_applied_update = applied_update.detach()
                    state = proposal
                    version += 1
                    raw_norm = torch.as_tensor(stat.pre_clip_norm, dtype=torch.float32, device=x.device)
                    clipped_nominal = torch.as_tensor(stat.post_clip_norm, dtype=torch.float32, device=x.device)
                    clip_ratio = torch.where(raw_norm > 0, clipped_nominal / raw_norm, torch.ones_like(raw_norm))
                else:
                    out = F.linear(h[:, start:end], current, self.down_proj.bias)
                    stat = TTTStats(0, self.ttt_chunk - 1, 0.0, 0.0, False)
                    cumulative_after = cumulative_before
                    norm_after = norm_before
                    update_norm = torch.zeros((), dtype=torch.float32, device=x.device)
                    raw_norm = torch.zeros((), dtype=torch.float32, device=x.device)
                    clipped_nominal = torch.zeros((), dtype=torch.float32, device=x.device)
                    clip_ratio = torch.ones((), dtype=torch.float32, device=x.device)
                    dot_state = None
                    cos_state = None
                    dot_previous = None
                    cos_previous = None
                post_hash = tensor_hash(state.weight) if self._capture_identity else None
                events.append({
                    "layer": int(self.layer_idx), "chunk_index": j + 1,
                    "selected_action": action, "active_update": active,
                    "pre_state_identity": pre_identity,
                    "post_state_identity": f"{self._branch_id}:L{self.layer_idx}:v{version}",
                    "pre_state_hash": pre_hash, "post_state_hash": post_hash,
                    "post_clip_delta_norm": float(stat.post_clip_norm),
                    "subsequent_recomputation": True,
                })
                state_events.append({
                    "layer": int(self.layer_idx), "chunk_index": j + 1,
                    "selected_action": action, "active_update": active,
                    "reset_applied": reset_applied,
                    "fast_weight_delta_norm_before": norm_before,
                    "fast_weight_delta_norm_after": norm_after,
                    "fast_weight_state_growth": norm_after - norm_before,
                    "raw_update_fro_norm": raw_norm,
                    "clipped_update_fro_norm": clipped_nominal,
                    "actual_applied_update_fro_norm": update_norm,
                    "clip_ratio": clip_ratio,
                    "update_dot_cumulative_state": dot_state,
                    "update_cos_cumulative_state": cos_state,
                    "update_dot_previous_same_layer": dot_previous,
                    "update_cos_previous_same_layer": cos_previous,
                    "cumulative_state_norm_before": norm_before,
                    "cumulative_state_norm_after": norm_after,
                    "hidden_input_stats": _scalar_stats(torch, x[:, start:end]),
                    "mlp_intermediate_stats": _scalar_stats(torch, h[:, start:end]),
                    "mlp_output_stats": _scalar_stats(torch, out),
                    "target_stats": _scalar_stats(torch, t[:, start:end]),
                    "gate_stats": gate_summary if active else None,
                })
                outputs.append(out)
                stats.append(stat)
            if full_len < x.shape[1]:
                outputs.append(F.linear(h[:, full_len:], state.weight, self.down_proj.bias))
            self.last_ttt_stats = stats
            self._events = events
            self._state_events = state_events
            self._reset_events = reset_events
            self._captured_final_state = state.weight.detach()
            return torch.cat(outputs, dim=1), state.weight

        mlp.forward = types.MethodType(patched, mlp)


def materialize_scalar(value):
    """Convert detached scalar tensors recursively, preserving missing values."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {key: materialize_scalar(item) for key, item in value.items()}
    if isinstance(value, list):
        return [materialize_scalar(item) for item in value]
    try:
        return float(value.detach().cpu().item())
    except AttributeError:
        return value
