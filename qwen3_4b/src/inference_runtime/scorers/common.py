#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path

VERSION = "ruler_16k_13task_1000sample_v1"
CONTEXT_LENGTH = 16384
SAMPLES_PER_TASK = 1000
SEED = 42
BENCHMARK_ROOT = Path(os.path.expandvars("${TTT_BENCHMARK_ROOT}")) / VERSION
MODEL_SPECS = {
    "base": os.path.expandvars("${TTT_MODEL_ROOT}/Qwen3-1.7B-Base"),
    "ckpt1000": os.path.expandvars("${TTT_CHECKPOINT_ROOT}/legacy/global_step_1000/hf_ckpt"),
    "ckpt500": os.path.expandvars("${TTT_CHECKPOINT_ROOT}/legacy/global_step_500/hf_ckpt"),
}

TASKS = [
    dict(name="ruler_cwe_16k", family="cwe", max_new_tokens=120, evaluator="RulerCweEvaluator", kwargs=dict(freq_cw=30, freq_ucw=3, num_cw=10)),
    dict(name="ruler_fwe_16k", family="fwe", max_new_tokens=50, evaluator="RulerFweEvaluator", generation_context_budget=16078, kwargs=dict(alpha=2.0, coded_wordlen=6)),
    dict(name="ruler_niah_single_1_16k", family="niah", max_new_tokens=128, evaluator="RulerNiahEvaluator", kwargs=dict(type_haystack="repeat", type_needle_k="words", type_needle_v="numbers", num_needle_k=1, num_needle_v=1, num_needle_q=1)),
    dict(name="ruler_niah_single_2_16k", family="niah", max_new_tokens=128, evaluator="RulerNiahEvaluator", kwargs=dict(type_haystack="essay", type_needle_k="words", type_needle_v="numbers", num_needle_k=1, num_needle_v=1, num_needle_q=1)),
    dict(name="ruler_niah_single_3_16k", family="niah", max_new_tokens=128, evaluator="RulerNiahEvaluator", kwargs=dict(type_haystack="essay", type_needle_k="words", type_needle_v="uuids", num_needle_k=1, num_needle_v=1, num_needle_q=1)),
    dict(name="ruler_niah_multikey_1_16k", family="niah", max_new_tokens=128, evaluator="RulerNiahEvaluator", kwargs=dict(type_haystack="essay", type_needle_k="words", type_needle_v="numbers", num_needle_k=4, num_needle_v=1, num_needle_q=1)),
    dict(name="ruler_niah_multikey_2_16k", family="niah", max_new_tokens=128, evaluator="RulerNiahEvaluator", kwargs=dict(type_haystack="needle", type_needle_k="words", type_needle_v="numbers", num_needle_k=1, num_needle_v=1, num_needle_q=1)),
    dict(name="ruler_niah_multikey_3_16k", family="niah", max_new_tokens=128, evaluator="RulerNiahEvaluator", kwargs=dict(type_haystack="needle", type_needle_k="uuids", type_needle_v="uuids", num_needle_k=1, num_needle_v=1, num_needle_q=1)),
    dict(name="ruler_niah_multivalue_16k", family="niah", max_new_tokens=128, evaluator="RulerNiahEvaluator", kwargs=dict(type_haystack="essay", type_needle_k="words", type_needle_v="numbers", num_needle_k=1, num_needle_v=4, num_needle_q=1)),
    dict(name="ruler_niah_multiquery_16k", family="niah", max_new_tokens=128, evaluator="RulerNiahEvaluator", kwargs=dict(type_haystack="essay", type_needle_k="words", type_needle_v="numbers", num_needle_k=1, num_needle_v=1, num_needle_q=4)),
    dict(name="ruler_qa_squad_16k", family="qa", max_new_tokens=50, evaluator="RulerQaEvaluator", kwargs=dict(dataset="squad", path=os.path.expandvars("${TTT_OPENCOMPASS_ROOT}/.cache/opencompass/data/ruler/dev-v2.0.json"))),
    dict(name="ruler_qa_hotpotqa_16k", family="qa", max_new_tokens=50, evaluator="RulerQaEvaluator", kwargs=dict(dataset="hotpotqa", path=os.path.expandvars("${TTT_OPENCOMPASS_ROOT}/.cache/opencompass/data/ruler/hotpotqa.json"))),
    dict(name="ruler_vt_16k", family="vt", max_new_tokens=30, evaluator="RulerVtEvaluator", kwargs=dict(num_chains=1, num_hops=4)),
]

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

def read_manifest_verified(root=BENCHMARK_ROOT):
    root = Path(root)
    manifest_path = root / "benchmark_manifest.json"
    expected = (root / "benchmark_manifest.sha256").read_text().split()[0]
    actual = sha256_file(manifest_path)
    if actual != expected:
        raise RuntimeError(f"benchmark manifest checksum mismatch: {actual} != {expected}")
    return json.loads(manifest_path.read_text()), actual

def task_rows(task_name, root=BENCHMARK_ROOT):
    path = Path(root) / "samples" / f"{task_name}.jsonl"
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

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
