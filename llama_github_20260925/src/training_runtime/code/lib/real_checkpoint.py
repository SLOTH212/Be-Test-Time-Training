import hashlib, json, os, random, shutil, time
from pathlib import Path

import numpy as np
import torch

from .pipeline_core import atomic_json, fsync_dir, sha256

REQUIRED = {
    "model", "optimizer", "scheduler", "global_step", "micro_step",
    "gradient_accumulation_phase", "record_cursor", "cumulative_tokens",
    "dataloader_state", "sampler_state", "python_rng", "numpy_rng",
    "torch_cpu_rng", "torch_cuda_rng", "config", "tokenizer_sha256",
    "project_head", "dataset_manifest_sha256", "complete",
}

def capture_rng():
    return {
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_cpu_rng": torch.get_rng_state(),
        "torch_cuda_rng": torch.cuda.get_rng_state_all(),
    }

def restore_rng(payload):
    random.setstate(payload["python_rng"])
    np.random.set_state(payload["numpy_rng"])
    torch.set_rng_state(payload["torch_cpu_rng"])
    torch.cuda.set_rng_state_all(payload["torch_cuda_rng"])

def optimizer_to_cuda(optimizer):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.cuda()

class RealCheckpointRotation:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def latest(self):
        path = self.root / "latest.json"
        return json.load(open(path)) if path.exists() else None

    def _next_slot(self):
        latest = self.latest()
        return "slot_B" if latest and latest["slot"] == "slot_A" else "slot_A"

    def save(self, payload):
        missing = sorted(REQUIRED - payload.keys())
        if missing or payload.get("complete") is not True:
            raise RuntimeError(f"BLOCKED_REAL_CHECKPOINT_ADAPTER missing={missing}")
        slot = self._next_slot()
        partial = self.root / f"{slot}.partial"
        final = self.root / slot
        latest = self.latest()
        if final.exists():
            # The selected target is, by construction, the non-latest slot. It
            # is safe to reclaim before writing the new partial and avoids a
            # transient third full checkpoint copy.
            if latest and latest["slot"] == slot:
                raise RuntimeError("BLOCKED_REAL_CHECKPOINT_ADAPTER latest overwrite")
            shutil.rmtree(final)
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir()
        checkpoint = partial / "checkpoint.pt"
        torch.save(payload, checkpoint)
        with open(checkpoint, "rb") as handle:
            os.fsync(handle.fileno())
        digest = sha256(checkpoint)
        # A genuine CPU reload must succeed before promotion/latest update.
        reloaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if sorted(REQUIRED - reloaded.keys()) or reloaded["record_cursor"] != payload["record_cursor"]:
            raise RuntimeError("BLOCKED_REAL_CHECKPOINT_ADAPTER reload validation")
        del reloaded
        manifest = {
            "format": "gated_ntp_real_checkpoint_v1", "complete": True,
            "slot": slot, "checkpoint_sha256": digest,
            "checkpoint_id": f"{payload['stage']}-cursor-{payload['record_cursor']}",
            "record_cursor": payload["record_cursor"],
            "cumulative_tokens": payload["cumulative_tokens"],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        atomic_json(partial / "checkpoint_manifest.json", manifest)
        fsync_dir(partial)
        os.replace(partial, final)
        fsync_dir(self.root)
        atomic_json(self.root / "latest.json", {
            "slot": slot, "checkpoint_id": manifest["checkpoint_id"],
            "record_cursor": payload["record_cursor"],
            "cumulative_tokens": payload["cumulative_tokens"],
            "manifest_sha256": sha256(final / "checkpoint_manifest.json"),
        })
        return final

    def load(self, model, optimizer, scheduler):
        latest = self.latest()
        if not latest:
            return None
        slot = self.root / latest["slot"]
        manifest = json.load(open(slot / "checkpoint_manifest.json"))
        if sha256(slot / "checkpoint_manifest.json") != latest["manifest_sha256"]:
            raise RuntimeError("BLOCKED_REAL_CHECKPOINT_ADAPTER manifest hash")
        checkpoint = slot / "checkpoint.pt"
        if not manifest["complete"] or sha256(checkpoint) != manifest["checkpoint_sha256"]:
            raise RuntimeError("BLOCKED_REAL_CHECKPOINT_ADAPTER checkpoint hash")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        optimizer_to_cuda(optimizer)
        scheduler.load_state_dict(payload["scheduler"])
        restore_rng(payload)
        return payload
