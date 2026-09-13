"""Adapter around the frozen recovered NTP source authority.

The vendored builder is evidence and is never edited or executed: its final
`.cuda()` is intentionally replaced here by an explicit torchrun device.
"""
import hashlib
import sys
from pathlib import Path

import torch

from .scaleup_debug_adapter import enable_debug_chunk_size

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TTT_LAYERS = [0, 6, 12, 18, 24]






def apply_ntp_contract(config, train_clip_none: bool = True, ttt_layers=None, ttt_chunk: int = 1024):
    layers = list(DEFAULT_TTT_LAYERS if ttt_layers is None else ttt_layers)
    if not layers or len(layers) != len(set(layers)):
        raise ValueError("TTT_LAYER_SET_EMPTY_OR_DUPLICATED")
    if any(not isinstance(layer, int) or layer < 0 or layer >= int(config.num_hidden_layers) for layer in layers):
        raise ValueError(
            f"TTT_LAYER_INDEX_OUT_OF_RANGE layers={layers} num_hidden_layers={config.num_hidden_layers}"
        )
    if int(ttt_chunk) <= 0:
        raise ValueError("TTT_CHUNK_MUST_BE_POSITIVE")
    values = {
        "ttt_mode": True, "ttt_layers": layers, "ttt_proj": True,
        "ttt_lr": 1.0, "ttt_chunk": int(ttt_chunk), "ttt_target": "hidden_states",
        "ttt_target_type": "gated_next_position_hidden", "ttt_ntp_gate": True,
        "ttt_ntp_gate_init": "zero", "ttt_ntp_within_chunk_only": True,
        "ttt_ntp_cross_chunk": False, "ttt_ntp_cross_record": False,
        "ttt_ntp_cross_document": False, "ttt_document_metadata_required": True,
        "ttt_backend": "authoritative_document_chunk_loop",
        "optimizer_betas": [0.9, 0.95], "router_enabled": False,
        "signal_capture_enabled": False, "calibration_assignment_enabled": False,
        "ttt_update_clip_norm": None if train_clip_none else 1e-5,
        "use_cache": False,
    }
    for key, value in values.items():
        setattr(config, key, value)
    return config


def _classes():
    from dynamic_ttt.models.training.hf_qwen3.configuration_qwen3 import Qwen3Config
    from dynamic_ttt.models.training.hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM
    return Qwen3Config, Qwen3ForCausalLM


def build_pretrained(path, device: torch.device, train_clip_none: bool = True,
                     ttt_layers=None, ttt_chunk: int = 1024):
    Qwen3Config, Qwen3ForCausalLM = _classes()
    enable_debug_chunk_size(ttt_chunk)
    config = apply_ntp_contract(
        Qwen3Config.from_pretrained(path, local_files_only=True),
        train_clip_none,
        ttt_layers=ttt_layers,
        ttt_chunk=ttt_chunk,
    )
    model = Qwen3ForCausalLM.from_pretrained(
        path, config=config, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True,
    ).to(device).train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model, config


def build_tiny_authority(device: torch.device):
    """Small 28-block authority model for structure tests; no pretrained assets."""
    Qwen3Config, Qwen3ForCausalLM = _classes()
    config = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=28, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=8, max_position_embeddings=256)
    apply_ntp_contract(config)
    return Qwen3ForCausalLM(config).to(device), config
