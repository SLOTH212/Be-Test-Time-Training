#!/usr/bin/env python3
import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code/ntp'))
from inference_model.hf_qwen3.configuration_qwen3 import Qwen3Config
from inference_model.hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM

MODES = ['OFF','L0','L6','L12','L18','L24','ALL']
LAYER_MAP = {'OFF':[], 'L0':[0], 'L6':[6], 'L12':[12], 'L18':[18], 'L24':[24], 'ALL':[0,6,12,18,24]}
CHECKPOINT_ID = 'f4e264ed6f5671f123d6c8a27c69bce3af5ffcba518c9e298398314c6147d9fd'
MODEL_HASH = 'ca51ee90d20ee09efbe0b31dd21bf9faa70275508316d1ba7cb1387ee4f0725f'

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(16<<20),b''): h.update(b)
    return h.hexdigest()

def canon(obj): return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def load_rows(bench):
    manifest=json.load(open(bench/'benchmark_manifest.json'))
    for task in manifest['tasks']:
        with open(bench/'samples'/f"{task['name']}.jsonl",encoding='utf-8') as f:
            for line in f: yield task,json.loads(line)

def base_hash(model):
    h=hashlib.sha256()
    for layer in [0,6,12,18,24]:
        h.update(model.model.layers[layer].mlp.down_proj.weight.detach().contiguous().view(torch.uint16).cpu().numpy().tobytes())
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--benchmark-root',required=True); ap.add_argument('--checkpoint',default=str(ROOT/'models/qwen3_1p7b_stage2'))
    ap.add_argument('--output',required=True); ap.add_argument('--modes',default=','.join(MODES)); ap.add_argument('--sample-id'); ap.add_argument('--limit',type=int)
    ap.add_argument('--non-scientific-smoke',action='store_true'); ap.add_argument('--formal-inference-authorized',action='store_true'); args=ap.parse_args()
    modes=args.modes.split(',')
    if args.non_scientific_smoke:
        if args.limit not in (None,1): raise SystemExit('smoke limit may not exceed one sample')
        if not modes or len(modes) != len(set(modes)) or any(mode not in MODES for mode in modes):
            raise SystemExit('smoke modes must be a unique ordered subset of the frozen action set')
    elif modes != MODES:
        raise SystemExit('frozen launcher requires modes in exact order: '+','.join(MODES))
    elif not args.formal_inference_authorized or args.limit is not None or args.sample_id is not None:
        raise SystemExit('formal inference requires explicit authorization and forbids sample filters')
    bench=Path(args.benchmark_root).resolve(); ckpt=Path(args.checkpoint).resolve(); output=Path(args.output).resolve()
    if output.exists(): raise SystemExit(f'refusing to overwrite {output}')
    if sha(ckpt/'model.safetensors') != MODEL_HASH: raise RuntimeError('model hash mismatch')
    manifest_hash=sha(bench/'benchmark_manifest.json'); expected=(bench/'benchmark_manifest.sha256').read_text().split()[0]
    if manifest_hash != expected: raise RuntimeError('benchmark manifest hash mismatch')
    scorer_path=ROOT/'code/ruler/tools/common.py'; spec=importlib.util.spec_from_file_location('frozen_scorer',scorer_path)
    scorer=importlib.util.module_from_spec(spec); spec.loader.exec_module(scorer)
    rows=list(load_rows(bench))
    if args.sample_id: rows=[x for x in rows if x[1]['sample_id']==args.sample_id]
    if args.non_scientific_smoke: rows=rows[:1]
    if not rows: raise RuntimeError('no selected rows')
    cfg=Qwen3Config.from_pretrained(ckpt,local_files_only=True); cfg.ttt_mode=True; cfg.ttt_layers=[0,6,12,18,24]
    cfg.ttt_chunk=1024; cfg.ttt_lr=1.0; cfg.ttt_target='hidden_states'; cfg.ttt_target_type='gated_next_position_hidden'
    cfg.ttt_proj=True; cfg.ttt_ntp_gate=True; cfg.ttt_update_clip_norm=1e-5; cfg.use_cache=True
    tok=AutoTokenizer.from_pretrained(ckpt,local_files_only=True); tok.padding_side='left'
    model=Qwen3ForCausalLM.from_pretrained(ckpt,config=cfg,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval()
    output.parent.mkdir(parents=True,exist_ok=True); tmp=output.with_name('.'+output.name+'.tmp'); base=base_hash(model)
    with tmp.open('x',encoding='utf-8') as f:
        for task,row in rows:
            encoded0=tok(row['input'],return_tensors='pt',add_special_tokens=True,truncation=False); n=int(encoded0.input_ids.shape[1]); encoded={k:v.cuda() for k,v in encoded0.items()}
            for mode in modes:
                used=set(LAYER_MAP[mode]); stashed={}; previous=model.model.ttt_mode
                for layer in [0,6,12,18,24]:
                    if layer in used: continue
                    mlp=model.model.layers[layer].mlp; stashed[layer]={}
                    for name in ('ttt_conv','ttt_proj','ttt_ntp_gate'):
                        if hasattr(mlp,name): stashed[layer][name]=getattr(mlp,name); delattr(mlp,name)
                if not used: model.model.ttt_mode=False
                torch.cuda.reset_peak_memory_stats(); started=time.perf_counter()
                try:
                    with torch.inference_mode():
                        generated=model.generate(**encoded,max_new_tokens=row['max_new_tokens'],do_sample=False,num_beams=1,use_cache=True,pad_token_id=tok.eos_token_id)
                finally:
                    model.model.ttt_mode=previous
                    for layer,attrs in stashed.items():
                        for name,value in attrs.items(): setattr(model.model.layers[layer].mlp,name,value)
                torch.cuda.synchronize(); pred=tok.decode(generated[0,n:],skip_special_tokens=True)
                score=float(scorer.per_sample_score(task['family'],pred,row['reference']))
                if not math.isfinite(score) or base_hash(model)!=base: raise RuntimeError('nonfinite score or base weight mutation')
                rec={'sample_id':row['sample_id'],'task':row['task'],'sample_index':row.get('sample_index',int(row['sample_id'].rsplit(':',1)[1])),
                     'mode':mode,'prediction':pred,'reference':row['reference'],'score':score,'status':'ok',
                     'model_hash':MODEL_HASH,'checkpoint_hash':CHECKPOINT_ID,'dataset_manifest_hash':manifest_hash,
                     'evaluator_config_hash':sha(scorer_path),'decoding_config_hash':canon({'do_sample':False,'num_beams':1,'use_cache':True,'max_new_tokens':row['max_new_tokens']}),
                     'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'prompt_token_count':n,'gpu_peak_memory_bytes':int(torch.cuda.max_memory_allocated()),
                     'ttt_layers':LAYER_MAP[mode],'ttt_chunk':1024,'ttt_lr':None if mode=='OFF' else 1.0,'delta_clip_frobenius':None if mode=='OFF' else 1e-5}
                f.write(json.dumps(rec,ensure_ascii=False,separators=(',',':'))+'\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,output); print(json.dumps({'status':'PASS','rows':len(rows)*len(modes),'samples':len(rows),'output':str(output)}))

if __name__=='__main__': main()
