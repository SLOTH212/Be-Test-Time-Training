#!/usr/bin/env python3
"""Generic wrapper around the byte-exact formal 1.7B Dynamic LB search authority."""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from filelock import FileLock
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
from dynamic_ttt.eval.runtime import action_layers, atomic_json, dynamic_required, exclusive_claim, fixed_actions, load_config, replay_budget, sha256, validate_benchmark_authority, validate_model_authority


def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def plan(cfg):
    actions=fixed_actions(cfg['ttt_layers']);chunks=cfg['context_length']//cfg['ttt_chunk_size']
    result={"status":"DRY_RUN_PASS","name":"Dynamic Recomputed Lower Bound","actions":actions,"beam":4,"sweeps":1,"position_order":"forward","complete_chunk_capacity":chunks,"max_unique_candidate_replays":replay_budget(len(actions),chunks,4,1),"final_trace_capture_replays":1,"tail_update":False,"generation_update":False,"downstream_recomputation":True,"stored_delta_stitching":False,"cwe_policy":"EXCLUDED_FOR_HISTORICAL_12_TASK_AUTHORITY"}
    if cfg['model_path']!='MODEL_PATH':result['model_authority']=validate_model_authority(cfg['model_path'],cfg['ttt_layers'],cfg.get('model_config_sha256'))
    if cfg['benchmark_root']!='BENCHMARK_ROOT':result['benchmark_authority']=validate_benchmark_authority(cfg['benchmark_root'],cfg['context_length'],cfg.get('benchmark_manifest_sha256'))
    return result


class Context:
    def __init__(self,cfg,rows,baselines):
        self.config=cfg
        self.run=Path(cfg['run_root']);self.code=ROOT/'code';self.ckpt=Path(cfg['model_path']);self.rows=rows;self.by_id={r['sample_id']:r for r in rows};self.manifest={'tasks':list(dict.fromkeys(r['task'] for r in rows))};self.baseline=baselines
        scorer=load_module(ROOT/'eval/scoring.py',f"scorer_{os.getpid()}");self.score_fn=scorer.per_sample_score


def execute(cfg,worker_id,workers):
    actions=fixed_actions(cfg['ttt_layers']);mapping=action_layers(cfg['ttt_layers']);runner=__import__('dynamic_ttt.dynamic.configured',fromlist=['runner']).runner(cfg)
    runner.ACTIONS=actions;runner.LAYERS=list(cfg['ttt_layers']);runner.ACTION_LAYERS=mapping;runner.ACTION_MODE={a:a for a in actions};runner.CHECKPOINT_HASH=sha256(Path(cfg['model_path'])/'model.safetensors')
    bench=Path(cfg['benchmark_root']);fixed=Path(cfg['fixed_results_root']);run=Path(cfg['run_root']);rows=[]
    global_index={json.loads(line)['sample_id']:i for i,line in enumerate((bench/'sample_manifest.jsonl').read_text(encoding='utf-8').splitlines()) if line.strip()}
    manifest=json.loads((bench/'benchmark_manifest.json').read_text())
    excluded=set(cfg.get('dynamic_excluded_tasks',['ruler_cwe_16k']))
    for task in manifest['tasks']:
        if task['name'] in excluded:continue
        with (bench/'samples'/f"{task['name']}.jsonl").open(encoding='utf-8') as stream:
            for line in stream:
                row=json.loads(line);row['_sample_index']=len(rows);rows.append(row)
    baselines={a:{} for a in actions}
    for row in rows:
        for action in actions:
            path=fixed/action/'commits'/f"{global_index[row['sample_id']]:06d}_{row['task']}_{row['sample_index']:04d}.json"
            record=json.loads(path.read_text());record['official_sample_score']=float(record['score']);baselines[action][row['sample_id']]=record
    eligible=[]
    for row in rows:
        sb=max(baselines[a][row['sample_id']]['official_sample_score'] for a in actions)
        if dynamic_required(sb):eligible.append(row)
    identity=hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    owned=[r for i,r in enumerate(eligible) if i%workers==worker_id];ctx=Context(cfg,owned,baselines);model,tok=runner.load_model(ctx)
    for ordinal,row in enumerate(owned,1):
        name=hashlib.sha256(row['sample_id'].encode()).hexdigest();result_path=run/'dynamic/results'/f'{name}.json'
        claim=run/'dynamic/claims'/f'{name}.lock';claim.parent.mkdir(parents=True,exist_ok=True)
        with FileLock(str(claim),timeout=0):
            if result_path.exists():
                saved=json.loads(result_path.read_text())
                if saved.get('config_identity')!=identity or saved.get('sample_id')!=row['sample_id'] or saved.get('completed') is not True:
                    raise ValueError('INVALID_RESUME_RESULT')
                continue
            raw=runner.search_sample(ctx,model,tok,row,'formal_dynamic',ordinal,len(owned),run/'dynamic/status'/f'worker{worker_id}.json')
            if raw['candidate_count']>replay_budget(len(actions),raw['full_chunk_count'],4,1):raise ValueError('REPLAY_BUDGET_EXCEEDED')
            atomic_json(result_path,{"sample_id":row['sample_id'],"task":row['task'],"sample_best":raw['sample_best_score'],"dynamic_score":raw['dynamic_search_score'],"action_sequence":raw['best_sequence'],"replay_count":raw['candidate_count'],"chunk_count":raw['full_chunk_count'],"ceiling":False,"search_status":"SUCCESS","errors":[],"prediction_hash":raw['best_prediction_hash'],"dynamic_source_sha256":sha256(ROOT/'dynamic/search.py'),"downstream_recomputation":True,"stored_delta_stitching":False,"config_identity":identity,"completed":True})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--dry-run',action='store_true');parser.add_argument('--worker-id',type=int,default=0);parser.add_argument('--workers',type=int,default=1);args=parser.parse_args();cfg=load_config(args.config)
    if args.dry_run:print(json.dumps(plan(cfg),sort_keys=True));return
    for key in ['model_path','benchmark_root','run_root']:
        if cfg[key] in ['MODEL_PATH','BENCHMARK_ROOT','RUN_ROOT']:raise SystemExit('replace template paths')
    if 'fixed_results_root' not in cfg:raise SystemExit('fixed_results_root required')
    if args.workers<1 or not 0<=args.worker_id<args.workers:raise SystemExit('invalid worker shard')
    execute(cfg,args.worker_id,args.workers)


if __name__=='__main__':main()
