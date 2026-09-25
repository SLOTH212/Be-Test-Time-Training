#!/usr/bin/env python3
"""Generic formal Fixed_(N+2) evaluator; dry-run never imports torch or loads weights."""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from code.runtime import action_layers, atomic_json, fixed_actions, load_config, sha256, validate_benchmark_authority, validate_model_authority


def plan(cfg):
    actions = fixed_actions(cfg["ttt_layers"])
    result = {"status": "DRY_RUN_PASS", "actions": actions, "fixed_action_n": len(actions), "context_length": cfg["context_length"], "ttt_chunk_size": cfg["ttt_chunk_size"], "ttt_layers": cfg["ttt_layers"], "workers": cfg.get("workers", 1), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "INHERITED_UNSET"), "cwe_policy": "INCLUDED"}
    model_path, benchmark = cfg["model_path"], cfg["benchmark_root"]
    if model_path != "MODEL_PATH":
        result["model_authority"] = validate_model_authority(model_path, cfg["ttt_layers"], cfg.get("model_config_sha256"))
    if benchmark != "BENCHMARK_ROOT":
        result["benchmark_authority"] = validate_benchmark_authority(benchmark, cfg["context_length"], cfg.get("benchmark_manifest_sha256"))
    return result


def execute(cfg):
    # This is the historical mode-major evaluator generalized only over layers/actions.
    import torch
    from transformers import AutoTokenizer
    sys.path.insert(0, str(ROOT / "code"))
    from inference_model.hf_qwen3.configuration_qwen3 import Qwen3Config
    from inference_model.hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM
    scorer_path = ROOT / "scorers/common.py"
    spec = importlib.util.spec_from_file_location("formal_ruler_scorer", scorer_path)
    scorer = importlib.util.module_from_spec(spec); spec.loader.exec_module(scorer)
    bench, model_path, run = Path(cfg["benchmark_root"]), Path(cfg["model_path"]), Path(cfg["run_root"])
    rows = []
    manifest = json.loads((bench / "benchmark_manifest.json").read_text())
    for task in manifest["tasks"]:
        with (bench / "samples" / f"{task['name']}.jsonl").open(encoding="utf-8") as stream:
            rows.extend((task, json.loads(line)) for line in stream if line.strip())
    layers, mapping = cfg["ttt_layers"], action_layers(cfg["ttt_layers"])
    model_cfg = Qwen3Config.from_pretrained(model_path, local_files_only=True)
    model_cfg.ttt_mode=True; model_cfg.ttt_layers=layers; model_cfg.ttt_chunk=cfg["ttt_chunk_size"]; model_cfg.ttt_lr=1.0
    model_cfg.ttt_target='hidden_states'; model_cfg.ttt_target_type='gated_next_position_hidden'; model_cfg.ttt_proj=True; model_cfg.ttt_ntp_gate=True; model_cfg.ttt_update_clip_norm=1e-5; model_cfg.use_cache=True
    tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True); tok.padding_side='left'
    model=Qwen3ForCausalLM.from_pretrained(model_path,config=model_cfg,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval()
    config_id=hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    for action in fixed_actions(layers):
        outdir=run/'fixed'/action/'commits'; outdir.mkdir(parents=True,exist_ok=True)
        used=set(mapping[action]); previous=model.model.ttt_mode; stashed={}
        for layer in layers:
            if layer in used: continue
            mlp=model.model.layers[layer].mlp; stashed[layer]={}
            for name in ('ttt_conv','ttt_proj','ttt_ntp_gate'):
                if hasattr(mlp,name):stashed[layer][name]=getattr(mlp,name);delattr(mlp,name)
        if not used:model.model.ttt_mode=False
        try:
            for index,(task,row) in enumerate(rows):
                output=outdir/f"{index:06d}_{row['task']}_{row['sample_index']:04d}.json"
                if output.exists(): continue
                encoded0=tok(row['input'],return_tensors='pt',add_special_tokens=True,truncation=False); n=encoded0.input_ids.shape[1]; encoded={k:v.cuda() for k,v in encoded0.items()}; started=time.perf_counter()
                with torch.inference_mode(): generated=model.generate(**encoded,max_new_tokens=row['max_new_tokens'],do_sample=False,num_beams=1,use_cache=True,pad_token_id=tok.eos_token_id)
                pred=tok.decode(generated[0,n:],skip_special_tokens=True); score=float(scorer.per_sample_score(task['family'],pred,row['reference']))
                atomic_json(output,{"sample_id":row['sample_id'],"task":row['task'],"sample_index":row['sample_index'],"action":action,"score":score,"prediction":pred,"status":"SUCCESS","elapsed_seconds":time.perf_counter()-started,"config_identity":config_id,"model_identity":sha256(model_path/'model.safetensors'),"benchmark_identity":sha256(bench/'benchmark_manifest.json')})
        finally:
            model.model.ttt_mode=previous
            for layer,attrs in stashed.items():
                for name,value in attrs.items():setattr(model.model.layers[layer].mlp,name,value)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--dry-run',action='store_true');args=parser.parse_args()
    cfg=load_config(args.config); result=plan(cfg)
    if args.dry_run: print(json.dumps(result,sort_keys=True)); return
    if any(cfg[x] in ("MODEL_PATH","BENCHMARK_ROOT","RUN_ROOT") for x in ["model_path","benchmark_root","run_root"]): raise SystemExit("replace template paths before execution")
    execute(cfg)


if __name__=='__main__':main()
