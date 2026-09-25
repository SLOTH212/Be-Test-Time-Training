"""Atomic A/B distributed checkpoints with exact same-world-size resume."""
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .pipeline_core import atomic_json, fsync_dir


FORMAT = "gated_ntp_fsdp2_dcp_v1"


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _world() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def capture_rank_state(progress: dict, scheduler=None) -> dict:
    state = {
        "progress": progress,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_cpu_rng": torch.get_rng_state(),
        "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    return state


def restore_rank_state(state: dict) -> None:
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_cpu_rng"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda_rng"])


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def preflight_checkpoint_metadata(root, world_size: int, mesh_shape, batch_metadata: dict):
    """Fail closed on resume metadata before model construction or GPU training."""
    root = Path(root)
    latest_path = root / "latest.json"
    if not latest_path.is_file():
        return None
    latest = json.loads(latest_path.read_text())
    slot = root / latest["slot"]
    manifest_path = slot / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if _digest(manifest_path) != latest["manifest_sha256"] or not manifest.get("complete"):
        raise RuntimeError("DCP_MANIFEST_INVALID")
    if manifest["world_size"] != int(world_size) or manifest["mesh_shape"] != list(mesh_shape):
        raise RuntimeError("RESUME_WORLD_SIZE_OR_MESH_MISMATCH")
    if manifest["batch_metadata"] != dict(batch_metadata):
        raise RuntimeError("RESUME_BATCH_METADATA_MISMATCH")
    return manifest


class DistributedCheckpointRotation:
    def __init__(self, root, world_size: int, mesh_shape, batch_metadata: dict):
        self.root = Path(root)
        self.world_size = int(world_size)
        self.mesh_shape = list(mesh_shape)
        self.batch_metadata = dict(batch_metadata)
        if _rank() == 0:
            self.root.mkdir(parents=True, exist_ok=True)
        _barrier()

    def latest(self):
        path = self.root / "latest.json"
        return json.loads(path.read_text()) if path.is_file() else None

    def _validate_resume_metadata(self, manifest):
        if manifest["world_size"] != self.world_size or manifest["mesh_shape"] != self.mesh_shape:
            raise RuntimeError("RESUME_WORLD_SIZE_OR_MESH_MISMATCH")
        if manifest["batch_metadata"] != self.batch_metadata:
            raise RuntimeError("RESUME_BATCH_METADATA_MISMATCH")

    def save(self, model, optimizer, progress: dict, scheduler=None) -> Path:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

        latest = self.latest()
        slot = "slot_B" if latest and latest["slot"] == "slot_A" else "slot_A"
        partial, final = self.root / f"{slot}.partial", self.root / slot
        if _rank() == 0:
            if partial.exists(): shutil.rmtree(partial)
            if final.exists(): shutil.rmtree(final)
            partial.mkdir(parents=True)
        _barrier()
        # AdamW creates state lazily; untouched trainable parameters legitimately
        # have no optimizer entry at early checkpoints.
        options = StateDictOptions(strict=False)
        model_state, optimizer_state = get_state_dict(model, optimizer, options=options)
        dcp.save({"model": model_state, "optimizer": optimizer_state}, checkpoint_id=partial / "dcp")
        torch.save(capture_rank_state(progress, scheduler), partial / f"rank_{_rank():05d}.pt")
        _barrier()
        if _rank() == 0:
            files = {str(p.relative_to(partial)): _digest(p) for p in sorted(partial.rglob("*")) if p.is_file()}
            manifest = {
                "format": FORMAT, "complete": True, "slot": slot,
                "world_size": self.world_size, "mesh_shape": self.mesh_shape,
                "batch_metadata": self.batch_metadata, "progress": progress,
                "files": files, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            atomic_json(partial / "manifest.json", manifest)
            fsync_dir(partial)
            os.replace(partial, final)
            fsync_dir(self.root)
            atomic_json(self.root / "latest.json", {"slot": slot, "progress": progress,
                                                       "manifest_sha256": _digest(final / "manifest.json")})
        _barrier()
        return final

    def load(self, model, optimizer, scheduler=None):
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict

        latest = self.latest()
        if latest is None:
            return None
        slot = self.root / latest["slot"]
        manifest = json.loads((slot / "manifest.json").read_text())
        if _digest(slot / "manifest.json") != latest["manifest_sha256"] or not manifest.get("complete"):
            raise RuntimeError("DCP_MANIFEST_INVALID")
        self._validate_resume_metadata(manifest)
        for name, digest in manifest["files"].items():
            if _digest(slot / name) != digest:
                raise RuntimeError(f"DCP_FILE_HASH_MISMATCH {name}")
        options = StateDictOptions(strict=False)
        model_state, optimizer_state = get_state_dict(model, optimizer, options=options)
        state = {"model": model_state, "optimizer": optimizer_state}
        reader = dcp.FileSystemReader(slot / "dcp")
        checkpoint_keys = set(reader.read_metadata().state_dict_metadata)
        expected_model_keys = {f"model.{name}" for name in model_state}
        missing_model_keys = sorted(expected_model_keys - checkpoint_keys)
        if missing_model_keys:
            raise RuntimeError(f"DCP_MODEL_STATE_KEYS_MISSING {missing_model_keys[:8]}")
        planner = dcp.DefaultLoadPlanner(allow_partial_load=True)
        dcp.load(state, storage_reader=reader, planner=planner)
        set_state_dict(model, optimizer, model_state_dict=state["model"],
                       optim_state_dict=state["optimizer"], options=options)
        rank_state = torch.load(slot / f"rank_{_rank():05d}.pt", map_location="cpu", weights_only=False)
        if scheduler is not None:
            if "scheduler" not in rank_state: raise RuntimeError("DCP_SCHEDULER_STATE_MISSING")
            scheduler.load_state_dict(rank_state["scheduler"])
        restore_rank_state(rank_state)
        return rank_state["progress"]


def save_model_only(model, destination, metadata: dict) -> None:
    """Collectively consolidate a hashed Stage-1 model-only parent."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    destination = Path(destination)
    partial = destination.with_name(destination.name + ".partial")
    if _rank() == 0:
        if destination.exists():
            raise RuntimeError("REFUSE_STAGE1_MODEL_ONLY_OVERWRITE")
        if partial.exists(): shutil.rmtree(partial)
        partial.mkdir(parents=True)
    _barrier()
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    model_state = get_model_state_dict(model, options=options)
    if _rank() == 0:
        torch.save(model_state, partial / "model.pt")
        with (partial / "model.pt").open("rb") as handle: os.fsync(handle.fileno())
    _barrier()
    if _rank() == 0:
        files = {str(p.relative_to(partial)): _digest(p) for p in sorted(partial.rglob("*")) if p.is_file()}
        atomic_json(partial / "manifest.json", {"format": "gated_ntp_stage1_model_only_dcp_v1",
            "complete": True, "world_size": _world(), "metadata": metadata, "files": files})
        os.replace(partial, destination); fsync_dir(destination.parent)
    _barrier()


def load_model_only(model, source):
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
    source = Path(source); manifest = json.loads((source / "manifest.json").read_text())
    if not manifest.get("complete") or manifest.get("format") != "gated_ntp_stage1_model_only_dcp_v1":
        raise RuntimeError("STAGE1_MODEL_ONLY_MANIFEST_INVALID")
    for name, digest in manifest["files"].items():
        if _digest(source / name) != digest: raise RuntimeError(f"STAGE1_MODEL_ONLY_HASH_MISMATCH {name}")
    state = torch.load(source / "model.pt", map_location="cpu", weights_only=True)
    options = StateDictOptions(full_state_dict=True, strict=True)
    set_model_state_dict(model, state, options=options)
    return manifest["metadata"]
