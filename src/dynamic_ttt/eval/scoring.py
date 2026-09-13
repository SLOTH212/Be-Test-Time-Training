#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path



def canonical_json(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()

def sha256_file(path, chunk=16 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()

def input_hash(prompt):
    return sha256_bytes(prompt.encode("utf-8"))

def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)



def weight_manifest(model_path):
    root = Path(model_path)
    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise RuntimeError(f"no safetensors under {root}")
    entries = [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in files]
    aggregate = sha256_bytes("".join(f"{x['name']}\0{x['bytes']}\0{x['sha256']}\n" for x in entries).encode())
    return {"files": entries, "aggregate_sha256": aggregate}

def per_sample_score(family, prediction, reference):
    pred = prediction.lower()
    refs = reference if isinstance(reference, list) else [reference]
    if family == "qa":
        return float(max((r.lower() in pred) for r in refs))
    return sum(float(r.lower() in pred) for r in refs) / len(refs)
