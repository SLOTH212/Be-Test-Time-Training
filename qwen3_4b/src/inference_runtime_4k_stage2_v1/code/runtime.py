#!/usr/bin/env python3
"""Config, authority, action-space, replay-budget, and atomic ownership primitives."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

TOL = 1e-12


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_config(path):
    # Templates are JSON-compatible YAML, intentionally parseable without PyYAML.
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_config(cfg)
    return cfg


def fixed_actions(layers):
    layers = [int(x) for x in layers]
    if not layers or len(layers) != len(set(layers)) or layers != sorted(layers) or min(layers) < 0:
        raise ValueError("ttt_layers must be unique, sorted, nonnegative, and nonempty")
    return ["OFF", *[f"L{x}" for x in layers], "ALL"]


def action_layers(layers):
    actions = fixed_actions(layers)
    return {actions[0]: [], **{f"L{x}": [x] for x in layers}, actions[-1]: list(layers)}


def complete_chunks(prompt_tokens, chunk_size):
    if prompt_tokens < 0 or chunk_size <= 0:
        raise ValueError("invalid token/chunk count")
    return prompt_tokens // chunk_size


def replay_budget(action_count, complete_chunk_count, beam=4, sweeps=1):
    """Maximum unique candidate replays under the exact formal search.

    Constant seeds contribute A. On the first position all A seed branches add
    A*(A-1) non-constant candidates. Each later position/sweep adds at most
    B*(A-1) new candidates. The final identity/trace capture is separate.
    """
    if action_count < 2 or complete_chunk_count < 0 or beam < 1 or sweeps < 1:
        raise ValueError("invalid replay-budget parameters")
    if complete_chunk_count == 0:
        return action_count
    return action_count + action_count * (action_count - 1) + (complete_chunk_count * sweeps - 1) * beam * (action_count - 1)


def sample_best(scores, layers):
    actions = fixed_actions(layers)
    if set(scores) != set(actions):
        raise ValueError(f"fixed action mismatch: expected {actions}")
    values = {k: float(v) for k, v in scores.items()}
    if any(not math.isfinite(x) for x in values.values()):
        raise ValueError("nonfinite fixed score")
    best = max(values.values())
    return best, [a for a in actions if abs(values[a] - best) <= TOL]


def best_single(rows, layers):
    singles = [f"L{x}" for x in layers]
    means = {a: sum(float(r[a]) for r in rows) / len(rows) for a in singles}
    winner = max(singles, key=lambda a: (means[a], -singles.index(a)))
    return winner, means[winner]


def dynamic_required(sample_best_score):
    return abs(float(sample_best_score) - 1.0) > TOL


def validate_layers(layers, num_hidden_layers):
    fixed_actions(layers)
    invalid = [x for x in layers if x >= int(num_hidden_layers)]
    if invalid:
        raise ValueError(f"TTT layers outside model architecture: {invalid}; num_hidden_layers={num_hidden_layers}")


def validate_config(cfg):
    required = ["classification", "context_length", "ttt_chunk_size", "ttt_layers", "model_path", "benchmark_root", "run_root", "dynamic"]
    missing = [x for x in required if x not in cfg]
    if missing:
        raise ValueError(f"missing config fields: {missing}")
    if int(cfg["context_length"]) <= 0 or int(cfg["ttt_chunk_size"]) <= 0:
        raise ValueError("context/chunk must be positive")
    fixed_actions(cfg["ttt_layers"])
    dynamic = cfg["dynamic"]
    if dynamic != {"beam": 4, "sweeps": 1, "position_order": "forward", "downstream_recomputation": True, "stored_delta_stitching": False}:
        raise ValueError("Dynamic config differs from formal authority")
    if cfg.get("cuda_visible_devices", "INHERIT") != "INHERIT":
        raise ValueError("runtime must inherit CUDA_VISIBLE_DEVICES")


def validate_model_authority(model_path, layers, expected=None):
    root = Path(model_path)
    config_path = root / "config.json"
    tokenizer_candidates = [root / "tokenizer.json", root / "tokenizer_config.json"]
    if not config_path.is_file() or not any(p.is_file() for p in tokenizer_candidates):
        raise ValueError("model config/tokenizer authority missing")
    model_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    validate_layers(layers, model_cfg["num_hidden_layers"])
    if expected and sha256(config_path) != expected:
        raise ValueError("model config hash mismatch")
    return {"config_sha256": sha256(config_path), "num_hidden_layers": model_cfg["num_hidden_layers"], "tokenizer_files": [p.name for p in tokenizer_candidates if p.is_file()]}


def validate_benchmark_authority(root, expected_context, expected_hash=None):
    root = Path(root)
    manifest_path = root / "benchmark_manifest.json"
    sample_index = root / "sample_manifest.jsonl"
    if not manifest_path.is_file() or not sample_index.is_file():
        raise ValueError("benchmark authority/index missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = manifest.get("task_names") or [x["name"] for x in manifest.get("tasks", [])]
    if len(tasks) != 13 or manifest.get("total_samples") != 6500 or manifest.get("context_length") != expected_context:
        raise ValueError("benchmark inventory/context mismatch")
    actual = sha256(manifest_path)
    frozen = (root / "benchmark_manifest.sha256").read_text().split()[0] if (root / "benchmark_manifest.sha256").exists() else None
    if expected_hash and actual != expected_hash:
        raise ValueError("benchmark expected hash mismatch")
    if frozen and actual != frozen:
        raise ValueError("benchmark self-authority hash mismatch")
    return {"manifest_sha256": actual, "tasks": tasks, "samples": 6500, "context_length": expected_context}


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def exclusive_claim(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(fd, (json.dumps(payload, sort_keys=True) + "\n").encode())
        os.fsync(fd)
    finally:
        os.close(fd)
    return path
