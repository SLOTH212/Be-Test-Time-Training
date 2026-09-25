#!/usr/bin/env python3
"""Bounded integration worker; frozen model/scorer/search imported without edits."""
import os,sys,json,hashlib,time,types,math,traceback,argparse,fcntl,contextlib
from pathlib import Path
R=Path('/path/to/ttt'); W=R/'work/qwen3_4b_downstream_pipeline_debug_v1'; A=R/'audits/qwen3_4b_downstream_debug_v1'
setup=json.loads((W/'SETUP.json').read_text()); M=Path(setup['mechanism']);sys.path.insert(0,str(M))
import runtime as rt
from worker import Store,atomic
import execution
sys.path.insert(0,setup['adapt'])
# Preload the explicit 4K modules before any frozen mechanism path insertion.
import inference_model.hf_qwen3.modeling_qwen3 as qm
import hf_models.hf_qwen3.ttt_state_core as core
assert Path(core.__file__).resolve().is_relative_to(Path(setup['adapt']))
FLAGS=dict(DEBUG_ONLY=True,PAPER_RESULT=False,FORMAL_SCIENTIFIC_RESULT=False)
LAYERS=setup['layers']; ACTIONS=setup['actions']; BENCH=Path(setup['benchmark'])
def sha(p):return rt.sha(p)
def index():return rt.rows(BENCH/'manifests/SAMPLE_INDEX.jsonl')
def load_rows(ids):
 idx=index(); by={x['project_sample_id']:x for x in idx}; gen=json.loads((BENCH/'authority/RULER_32K_GENERATION_CONFIG.json').read_text()); tasks={x['name']:x for x in gen['tasks']};out=[]
 for sid in ids:
  m=by[sid]
  with (BENCH/m['official_jsonl_path']).open() as f:
   for i,line in enumerate(f,1):
    if i==m['source_line']:x=json.loads(line);break
  assert hashlib.sha256(x['prompt'].encode()).hexdigest()==m['input_sha256']
  assert rt.canonical(x['answer'])==m['reference_sha256']
  out.append(dict(sample_id=sid,task=m['task'],sample_index=m['source_line']-1,_sample_index=m['global_index'],input=x['prompt'],reference=x['answer'],family=tasks[m['task']]['family'],max_new_tokens=m['tokens_to_generate'],expected_prompt_tokens=m['input_token_length'],source=m))
 return out

def cfg(mechanism='reverse'):
 return dict(classification='DEBUG_PIPELINE_INTEGRATION',model_family='qwen3',mechanism=mechanism,configured_ttt_layers=LAYERS,actions=ACTIONS,chunk_size=4096,context_length=32768,model_path=setup['model'],model_identity=setup['model_identity'],benchmark_root=setup['benchmark'],benchmark_identity=setup['benchmark_identity'],**FLAGS)

def identity(phase,ids):return dict(phase=phase,ids=ids,setup=setup,worker_source=sha(__file__),mechanism_sources={p.name:sha(p) for p in M.glob('*.py')},**FLAGS)
def fixed_store(ids):return Store(W/'fixed',identity('fixed',ids))
def make_context(rows,base,workspace):
 ns={'__name__':'frozen_debug_scorer'};p=Path(setup['base'])/'scorers/common.py';exec(compile(p.read_text(),str(p),'exec'),ns)
 return types.SimpleNamespace(run=workspace,code=Path(setup['adapt']),ckpt=Path(setup['model']),rows=rows,by_id={x['sample_id']:x for x in rows},manifest={'tasks':list(dict.fromkeys(x['task'] for x in rows))},baseline=base,score_fn=ns['per_sample_score'])

def native_fixed(ctx,model,tok,row,action):
 import torch
 used=set(rt.mapping(LAYERS)[action]);prev=model.model.ttt_mode; nstep=[]; original_step=core.ttt_chunk_step
 # Runtime wrapper records only scalar instrumentation; leaves tensor operations untouched.
 def traced(*args,**kwargs):
  out,proposal,stat=original_step(*args,**kwargs)
  assert bool(torch.isfinite(proposal.weight).all()) and bool(torch.isfinite(out).all())
  nstep.append(dict(valid_pairs=int(stat.valid_pairs) if hasattr(stat,'valid_pairs') else int(stat.valid_tokens),post_clip_norm=float(stat.post_clip_norm)))
  return out,proposal,stat
 core.ttt_chunk_step=traced
 enc0=tok(row['input'],return_tensors='pt',add_special_tokens=True,truncation=False);n=enc0.input_ids.shape[1];assert n==row['expected_prompt_tokens'];assert n+row['max_new_tokens']<=32768
 sys.path.insert(0,str(Path('/path/to/ttt/src/reference_executor_v2_port')))
 from adapter import prepare_fixed
 prepare_fixed(model,LAYERS,core,action,n)
 enc={k:v.to(next(model.parameters()).device) for k,v in enc0.items()};started=time.time();(torch.cuda.reset_peak_memory_stats() if next(model.parameters()).device.type=='cuda' else 0);before=ctx.dr.base_hash(model)
 # Generation calls with t=None must never call the update primitive.
 generation_steps=[];hooks=[]
 def hook(m,args,out):
  if len(args)>1 and args[1] is None:generation_steps.append(len(nstep))
 for l in LAYERS:hooks.append(model.model.layers[l].mlp.register_forward_hook(hook))
 try:
  with torch.inference_mode():out=model.generate(**enc,max_new_tokens=row['max_new_tokens'],do_sample=False,num_beams=1,use_cache=True,pad_token_id=tok.eos_token_id)
  (torch.cuda.synchronize() if next(model.parameters()).device.type=='cuda' else 0);pred=tok.decode(out[0,n:],skip_special_tokens=True);score=float(ctx.score_fn(row['family'],pred,row['reference']))
  counts={str(l):sum(int(s.update_count) for s in model.model.layers[l].mlp.last_ttt_stats) for l in LAYERS}
  for l in LAYERS:assert counts[str(l)]==(n//4096 if l in used else 0),(l,counts)
  assert len(nstep)==(n//4096)*len(used)
  assert all(s['valid_pairs']==4095 for s in nstep)
  assert all(math.isfinite(s['post_clip_norm']) and s['post_clip_norm']<=1e-5*(1+2**-7)**2 for s in nstep)
  assert not generation_steps or min(generation_steps)==max(generation_steps)==len(nstep)
  assert ctx.dr.base_hash(model)==before and math.isfinite(score)
  return dict(sample_id=row['sample_id'],task=row['task'],sample_index=row['sample_index'],action=action,score=score,prediction=pred,prediction_hash=hashlib.sha256(pred.encode()).hexdigest(),prompt_token_count=n,complete_chunks=n//4096,tail_tokens=n%4096,update_count=len(nstep),post_clip_norms=[s['post_clip_norm'] for s in nstep],configured_clip=1e-5,clip_rounding_bound='BF16 scalar and product rounding: (1+2^-7)^2',layer_update_counts=counts,valid_pair_count_per_update=4095,generation_update_count=0,base_unchanged=True,latency_seconds=time.time()-started,gpu_peak_memory_bytes=(torch.cuda.max_memory_allocated() if next(model.parameters()).device.type=='cuda' else 0),model_identity=setup['model_identity'],benchmark_identity=setup['benchmark_identity'],parent_authority_sha256=setup['parent_authority_sha256'],inference_config_sha256=setup['config_sha256'],chunk_size=4096,context_length=32768,configured_layers=LAYERS,**FLAGS)
 finally:
  core.ttt_chunk_step=original_step
  for h in hooks:h.remove()
  model.model.ttt_mode=prev

def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['sanity','fixed','dynamic'],required=True);p.add_argument('--worker-id',type=int,default=0);p.add_argument('--workers',type=int,default=1);args=p.parse_args()
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='2,3';assert args.workers in [1,2] and 0<=args.worker_id<args.workers
 import torch
 torch.set_num_threads(4);torch.cuda.set_device(args.worker_id)
 expected=['GPU-fb7ba9a8-ce12-65a4-a9cf-5b5b7506b0ed','GPU-351e23d4-250b-6a70-2c66-d2fb0221a25a'][args.worker_id]
 assert str(torch.cuda.get_device_properties(args.worker_id).uuid)==expected.removeprefix('GPU-')
 ids=[x['sample_id'] for x in rt.rows(W/'DEBUG_SAMPLE_MANIFEST.jsonl')];assert 1<=len(ids)<=32
 fs=fixed_store(ids);rows=load_rows(ids);rows=rows[:1] if args.phase=='sanity' else rows
 baseline={a:{} for a in ACTIONS}
 if args.phase=='dynamic':
  for row in rows:
   for a in ACTIONS:
    c=fs.committed(row['sample_id']+'|'+a);assert c is not None
    baseline[a][row['sample_id']]={**c,'official_sample_score':c['score']}
  rows=[r for r in rows if max(baseline[a][r['sample_id']]['score'] for a in ACTIONS)<1-1e-12 and r['task']!='ruler_cwe_16k'][:2]
 owned=[r for i,r in enumerate(rows) if i%args.workers==args.worker_id];workspace=W/args.phase/'workers'/str(args.worker_id);workspace.mkdir(parents=True,exist_ok=True)
 ds=Store(W/'dynamic',identity('dynamic',[r['sample_id'] for r in rows])) if args.phase=='dynamic' else None
 pending=[r for r in owned if (ds.committed(r['sample_id']) is None if ds else any(fs.committed(r['sample_id']+'|'+a) is None for a in ACTIONS))]
 if not pending:print(json.dumps({'phase':args.phase,'worker':args.worker_id,'resumed_no_model_load':True,'owned':len(owned)}));return
 ctx=make_context(rows,baseline,workspace);dr=execution.runner(cfg(),Path(setup['base']));ctx.dr=dr
 model,tok=dr.load_model(ctx)
 if args.phase!='dynamic':
  for l in LAYERS:delattr(model.model.layers[l].mlp,'forward')
 atomic(workspace/'WORKER.json',dict(pid=os.getpid(),phase=args.phase,physical_gpu=args.worker_id+2,logical_gpu=args.worker_id,model_loads=1,active_sample_per_worker=1,**FLAGS))
 for row in pending:
  if ds:
   with ds.own(row['sample_id']):
    if ds.committed(row['sample_id']) is not None:continue
    raw=dr.search_sample(ctx,model,tok,row,'DEBUG_DYNAMIC_LB',1,len(pending),workspace/'progress.json')
    c=raw['full_chunk_count'];budget=8+8*7+(c-1)*4*7;assert raw['candidate_count']<=budget
    payload=dict(task=row['task'],sample_best=raw['fixed_oracle_score'],dynamic_score=raw['dynamic_search_score'],action_sequence=raw['best_sequence'],chunk_count=c,replay_count=raw['candidate_count'],replay_budget=budget,final_capture_replays=1,search_status='SUCCESS',prediction_hash=raw['best_prediction_hash'],completed=True,native_prediction_sha256=raw['best_prediction_hash'],model_identity=setup['model_identity'],benchmark_identity=setup['benchmark_identity'],inference_config_sha256=setup['config_sha256'],parent_fixed_config=fs.identity,downstream_recomputation=True,stored_delta_stitching=False,chunk_size=4096,context_length=32768,actions=ACTIONS,errors=[],raw=raw,**FLAGS)
    ds.commit(row['sample_id'],payload);print(json.dumps({'phase':'dynamic','sample':row['sample_id'],'replays':raw['candidate_count'],'committed':True}),flush=True)
  else:
   for action in ACTIONS:
    key=row['sample_id']+'|'+action
    with fs.own(key):
     if fs.committed(key) is not None:continue
     result=native_fixed(ctx,model,tok,row,action);result['benchmark_sample_id']=row['sample_id'];fs.commit(key,result)
     print(json.dumps({'phase':args.phase,'sample':row['sample_id'],'action':action,'score_DEBUG_ONLY':result['score'],'seconds':result['latency_seconds'],'committed':True}),flush=True)
 atomic(workspace/'DONE.json',dict(phase=args.phase,worker=args.worker_id,owned_samples=len(owned),status='PASS',**FLAGS))
if __name__=='__main__':
 try:main()
 except Exception as e:
  kind='RESOURCE_RUNTIME_FAILURE' if any(s in str(e).lower() for s in ['out of memory','cuda error','cuda failure']) else 'INTEGRATION_FAILURE'
  atomic(W/'errors'/f'{os.getpid()}.json',dict(status=kind,error=str(e),traceback=traceback.format_exc(),**FLAGS));raise
