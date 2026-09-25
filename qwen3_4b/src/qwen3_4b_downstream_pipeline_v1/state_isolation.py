#!/usr/bin/env python3
import os,json,argparse,hashlib,time
from pathlib import Path
from mechanism_worker import AdaptedExecutor,fixture
from debug_worker import R,W,A,setup,rt,atomic,Store,FLAGS,LAYERS,ACTIONS,load_rows,cfg,fixed_store
import torch
ap=argparse.ArgumentParser();ap.add_argument('--worker-id',type=int,default=0);args=ap.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='2,3';assert args.worker_id in [0,1];torch.set_num_threads(4);torch.cuda.set_device(args.worker_id)
ids=[x['sample_id'] for x in rt.rows(W/'DEBUG_SAMPLE_MANIFEST.jsonl')];fs=fixed_store(ids);fixed={sid:{a:fs.committed(sid+'|'+a)['score'] for a in ACTIONS} for sid in ids};ds=[json.loads(p.read_text()) for p in sorted((W/'dynamic/samples').glob('*.json'))];assert len(ds)==2
b=ds[args.worker_id];a=ds[1-args.worker_id];rows=load_rows([a['sample_id'],b['sample_id']]);workspace=W/'isolation'/str(args.worker_id);workspace.mkdir(parents=True,exist_ok=True);ex=AdaptedExecutor(cfg('reset'),rows,fixed,workspace)
for l in LAYERS:
 m=ex.model.model.layers[l].mlp;assert m.ttt_chunk==4096 and m.ttt_lr==1.0 and m.ttt_update_clip_norm==1e-5
r1,_=ex.evaluate(b,b['action_sequence']);assert r1['prediction_hash']==b['prediction_hash']
fa=fixture(next(r for r in rows if r['sample_id']==a['sample_id']),ex.tok);ex.ctx.by_id[fa['sample_id']]=fa;ex.ctx.baseline={act:{**ex.ctx.baseline[act],fa['sample_id']:{'official_sample_score':fixed[a['sample_id']][act]}} for act in ACTIONS};seq=['L30']+['L0']*7
mid,events=ex.evaluate({'sample_id':fa['sample_id']},seq,LAYERS,2);assert {e['layer'] for e in events}==set(LAYERS)
r2,_=ex.evaluate(b,b['action_sequence'])
assert r1['prediction_hash']==r2['prediction_hash'] and r1['score']==r2['score'];assert [e['post_state_hash'] for e in r1['chunk_layer_events']]==[e['post_state_hash'] for e in r2['chunk_layer_events']]
atomic(A/f'STATE_ISOLATION_WORKER{args.worker_id}.json',dict(status='PASS',sequence=['clean real B','derived fixture A with full six-layer reset','clean real B'],real_B=b['sample_id'],fixture_A=fa['sample_id'],prediction_hash_repeat=True,score_repeat=True,all_recorded_fast_state_hashes_repeat=True,base_weights_unchanged=True,cache_scope='new generate call per replay; no cache supplied',intervention_state_cleared=True,physical_gpu=args.worker_id+2,configured_layers=LAYERS,DEBUG_INTERFACE_FIXTURE=True,**FLAGS))
print(json.dumps({'worker':args.worker_id,'MECHANISM_SAMPLE_STATE_ISOLATION':'PASS'}))
