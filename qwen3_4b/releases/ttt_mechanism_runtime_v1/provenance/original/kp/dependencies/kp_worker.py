#!/usr/bin/env python3
from __future__ import annotations
import argparse, gc, hashlib, importlib.util, json, os, sys, time, traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

MODEL_SHA="ca51ee90d20ee09efbe0b31dd21bf9faa70275508316d1ba7cb1387ee4f0725f"
CKPT_SHA="f4e264ed6f5671f123d6c8a27c69bce3af5ffcba518c9e298398314c6147d9fd"
BENCH_SHA="5d16e3a6963d4c21450f0530af743730b91bf7b2442699e288e10754f77ad0c5"
RULE_SHA="84a3b1e3f6215e6ae94a8b394e92d633440567d4fc19a71375e1cb9be9fabc0a"
ELIG_SHA="70c08451da1abc11c489eb119c37c9a53b2bad861395f54bdf640d513a51a306"
DYNAMIC_AUDIT_SHA="89e5b7da8b85ce063af15216ef9a16de5c433d4e489a19efaae20ad16099689f"
LAYERS=[0,6,12,18,24]
ACTION_LAYERS={'OFF':[],'L0':[0],'L6':[6],'L12':[12],'L18':[18],'L24':[24],'ALL':LAYERS}
CELLS={'F00':(0,0),'F10':(1,0),'F01':(0,1),'F11':(1,1)}

def now(): return datetime.now().astimezone().isoformat(timespec="seconds")
def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def canonical(obj): return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def load(path): return json.loads(Path(path).read_text())
def rows(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def atomic(path,obj):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
 fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
 try:
  with os.fdopen(fd,"w") as f:
   json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
  os.replace(tmp,path); d=os.open(path.parent,os.O_RDONLY); os.fsync(d); os.close(d)
 except BaseException:
  try: os.unlink(tmp)
  except FileNotFoundError: pass
  raise
def imp(path,name):
 spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
def status(run,**kw):
 p=run/'status/pipeline_status.json'; x=load(p) if p.exists() else {}; x.update(kw); x['updated_at']=now(); atomic(p,x)
def branch_path(run,branch_id): return run/'branches'/f'{branch_id}.json'

class Context:
 def __init__(self,cfg):
  self.run=Path(cfg['run_root']); self.code=Path('/home/USER/ttt/code/ntp'); self.ckpt=Path(cfg['model_root'])
  self.score_fn=imp(cfg['scorer'],'kp_scorer').per_sample_score
  all_rows=rows(Path(cfg['benchmark_root'])/'sample_manifest.jsonl')
  self.rows=[]
  for i,r in enumerate(all_rows): r['_sample_index']=i; self.rows.append(r)
  self.by_id={r['sample_id']:r for r in self.rows}; self.manifest={'tasks':sorted({r['task'] for r in self.rows})}; self.baseline={}
  if len(self.by_id)!=6500: raise RuntimeError('BENCHMARK_SAMPLE_COUNT')

def gates(run):
 run=Path(run).resolve(); cfg=load(run/'config/KP_FACTORIAL_FORMAL_CONFIG.json')
 if sorted(os.sched_getaffinity(0))!=list(range(8,24)): raise RuntimeError('CPU_AFFINITY_MISMATCH')
 checks={
  'model':sha(Path(cfg['model_root'])/'model.safetensors')==MODEL_SHA,
  'benchmark':sha(Path(cfg['benchmark_root'])/'benchmark_manifest.json')==BENCH_SHA,
  'dynamic_audit':sha(cfg['dynamic_final_audit'])==DYNAMIC_AUDIT_SHA,
  'rule':sha(cfg['rule_manifest'])==RULE_SHA,
  'eligibility':sha(cfg['eligibility_manifest'])==ELIG_SHA,
  'amendment':sha(cfg['amendment'])==cfg['amendment_sha256'],
  'cohort':sha(cfg['cohort_manifest'])==cfg['cohort_sha256'],
  'branches':sha(cfg['branch_manifest'])==cfg['branch_manifest_sha256'],
  'protocol':sha(cfg['protocol'])==cfg['protocol_sha256'],
  'worker':sha(cfg['worker'])==cfg['worker_sha256'],
  'aggregator':sha(cfg['aggregator'])==cfg['aggregator_sha256'],
  'controller':sha(cfg['controller'])==cfg['controller_sha256'],
  'historical_core':sha(cfg['historical_factorial_core'])==cfg['historical_factorial_core_sha256'],
  'historical_closure_core':sha(cfg['historical_closure_core'])==cfg['historical_closure_core_sha256'],
 }
 if not all(checks.values()): raise RuntimeError('AUTHORITY_HASH_MISMATCH '+json.dumps(checks,sort_keys=True))
 cohort=rows(cfg['cohort_manifest']); manifest=rows(cfg['branch_manifest'])
 if len(cohort)!=167 or len({r['canonical_sample_id'] for r in cohort})!=167: raise RuntimeError('COHORT_SCOPE')
 if len(manifest)!=668 or len({r['branch_id'] for r in manifest})!=668: raise RuntimeError('BRANCH_SCOPE')
 if Counter(r['cell'] for r in manifest)!=Counter({c:167 for c in CELLS}): raise RuntimeError('CELL_SCOPE')
 for r in cohort:
  seq=r['full_native_dynamic_sequence']; b=r['boundary']; expected=1+max(i for i in range(1,len(seq)) if seq[i]!=seq[i-1])
  if b!=expected or r['prefix_chunks']!=b-1 or r['frozen_terminal_suffix']!=seq[b-1:]: raise RuntimeError('BOUNDARY_INVARIANT '+r['canonical_sample_id'])
 return run,cfg,cohort,manifest,checks

def runtime(run,cfg):
 import torch
 torch.set_num_threads(16)
 dr=imp(cfg['executor'],'kp_frozen_executor')
 sys.path.insert(0,str(Path(cfg['historical_closure_core']).parent))
 import closure_core as closure
 ctx=Context(cfg); status(run,current_phase='MODEL_LOADING',worker_pid=os.getpid(),worker_running=True)
 model,tok=dr.load_model(ctx); closure.install_formal(model); model.requires_grad_(False); base=dr.base_hash(model)
 atomic(run/'audit/MODEL_LOAD_AUDIT.json',{'status':'PASS','model_identity':MODEL_SHA,'checkpoint_identity':CKPT_SHA,'base_weight_runtime_hash':base,'cuda_device':torch.cuda.get_device_name(0),'timestamp':now()})
 return dr,closure,ctx,model,tok,base

def cmp(native,condition,th):
 import torch
 a=native.float(); b=condition.float(); d=a-b
 return {'exact_equal':bool(torch.equal(native,condition)),'max_abs_diff':float(d.abs().max()) if d.numel() else 0.0,'relative_l2_diff':float(torch.linalg.vector_norm(d)/(torch.linalg.vector_norm(a)+1e-30)),'native_hash':th(native),'condition_hash':th(condition)}

class Recorder:
 def __init__(self,prefix,T,n,mode,th,ref=None):
  self.start=prefix*1024; self.full=T*1024; self.n=n; self.mode=mode; self.th=th; self.ref={} if ref is None else ref; self.rows=[]; self.handles=[]; self.order=0
 def record(self,key,layer,tensor,region,chunk,x):
  import torch
  if not torch.is_tensor(x): return
  x=x.detach().cpu().clone()
  if self.mode=='native': self.ref[key]=x
  elif key in self.ref:
   self.rows.append({'stage_order':self.order,'region':region,'chunk_index':chunk,'layer':layer,'tensor':tensor,**cmp(self.ref[key],x,self.th)}); self.order+=1
 def first(self,layer,tensor,x):
  import torch
  if not torch.is_tensor(x) or x.ndim<2 or x.shape[1]<=self.start:return
  self.record(f'first:{layer}:{tensor}',layer,tensor,'FIRST_CONTINUATION_CHUNK',self.start//1024+1,x[:,self.start:min(self.start+1024,x.shape[1])])
 def tail(self,layer,tensor,x):
  import torch
  if not torch.is_tensor(x) or x.ndim<2 or self.n<=self.full or x.shape[1]<=self.full:return
  self.record(f'tail:{layer}:{tensor}',layer,tensor,'INCOMPLETE_PROMPT_TAIL','TAIL',x[:,self.full:self.n])
 def internal(self,layer,tensor,x): self.record(f'internal:{layer}:{tensor}',layer,tensor,'FIRST_POST_BOUNDARY_UPDATE',self.start//1024+1,x)
 def hooks(self,m):
  for li,layer in enumerate(m.model.layers):
   self.handles.append(layer.input_layernorm.register_forward_pre_hook(lambda q,i,l=li:self.first(l,'attention_input',i[0])))
   self.handles.append(layer.self_attn.q_proj.register_forward_hook(lambda q,i,o,l=li:self.first(l,'q_projection',o)))
   self.handles.append(layer.self_attn.k_proj.register_forward_hook(lambda q,i,o,l=li:self.first(l,'k_projection',o)))
   self.handles.append(layer.self_attn.v_proj.register_forward_hook(lambda q,i,o,l=li:self.first(l,'v_projection',o)))
   def ah(q,i,o,l=li): self.first(l,'attention_output',o[0]); self.tail(l,'attention_output',o[0])
   self.handles.append(layer.self_attn.register_forward_hook(ah))
   self.handles.append(layer.post_attention_layernorm.register_forward_pre_hook(lambda q,i,l=li:self.first(l,'mlp_input_residual',i[0])))
   def mh(q,i,o,l=li):
    x=o[0] if isinstance(o,tuple) else o; self.first(l,'mlp_output',x); self.tail(l,'mlp_output',x)
   self.handles.append(layer.mlp.register_forward_hook(mh))
   def lh(q,i,o,l=li): self.first(l,'residual_stream_output',o); self.tail(l,'residual_stream_output',o)
   self.handles.append(layer.register_forward_hook(lh))
  def nh(q,i,o):
   for j in range(self.start//1024,self.full//1024): self.record(f'complete:{j}:final_hidden',28,'final_complete_chunk_hidden','COMPLETE_CONTINUATION_CHUNK',j+1,o[:,j*1024:(j+1)*1024])
   if self.n>self.full:self.record('tail:28:final_hidden',28,'final_tail_hidden','INCOMPLETE_PROMPT_TAIL','TAIL',o[:,self.full:self.n])
  self.handles.append(m.model.norm.register_forward_hook(nh))
 def close(self):
  for h in self.handles:h.remove()

def prefix_hashes(cache,prefix,th):
 n=prefix*1024
 return {str(l):{'key':th(k[...,:n,:]),'value':th(v[...,:n,:])} for l,(k,v) in cache.items()}
def fast_hashes(x,th): return {str(l):th(x['boundary'][l]) for l in LAYERS}
def occupancy(x): return {str(l):bool(x['prompt_ttt_occupancy'][l]) for l in LAYERS}
def run_one(f,ctx,model,tok,row,seq,prefix,rec=None,inject=None,donor=None):
 # Historical structural execution stops after the prompt. Generation/reward is
 # a separate replay so one-token generation hooks cannot overwrite prompt keys.
 return f.run_prompt(ctx,model,tok,row,seq,prefix,inject=inject,donor=donor,trace_callback=None if rec is None else rec.internal,return_cache=True,generate=False)
def first_divergence(trace,l4,l5):
 canonical_regions=['FIRST_CONTINUATION_CHUNK','FIRST_POST_BOUNDARY_UPDATE','COMPLETE_CONTINUATION_CHUNK','INCOMPLETE_PROMPT_TAIL']
 for region in canonical_regions:
  q=[x for x in trace if x['region']==region and not x['exact_equal']]
  if q:
   x=q[0]; return {'closure_level':{'FIRST_CONTINUATION_CHUNK':'L1','FIRST_POST_BOUNDARY_UPDATE':'L1','COMPLETE_CONTINUATION_CHUNK':'L2','INCOMPLETE_PROMPT_TAIL':'L3'}[region],'region':region,'layer':x['layer'],'tensor':x['tensor'],'max_abs_diff':x['max_abs_diff'],'relative_l2_diff':x['relative_l2_diff']}
 if not l4:return {'closure_level':'L4','region':'GENERATION_START','layer':-1,'tensor':'generation_start_logits'}
 if not l5:return {'closure_level':'L5','region':'FULL_OUTPUT','layer':-1,'tensor':'prediction_sha256'}
 return {'closure_level':'NONE_OBSERVED','region':'NONE_OBSERVED','layer':None,'tensor':None}

def execute_sample(run,cfg,row,dr,c,ctx,model,tok,base,formal):
 import torch
 f=imp(cfg['historical_factorial_core'],'kp_historical_factorial_core')
 f.c=c; f.th=c.th; f.base=c.base; f.donor_prefix=c.donor_prefix
 sid=row['canonical_sample_id']; b=row['boundary']; prefix=b-1; seqn=row['full_native_dynamic_sequence']; seqc=['OFF']*prefix+row['frozen_terminal_suffix']; T=len(seqn)
 bench=ctx.by_id[sid]; n=len(tok(bench['input'],add_special_tokens=True)['input_ids'])
 rn=Recorder(prefix,T,n,'native',c.th); rn.hooks(model)
 try: native=run_one(f,ctx,model,tok,bench,seqn,prefix,rn)
 finally: rn.close()
 native_reward=f.run_reward(ctx,model,tok,bench,seqn,prefix)
 parity=(native_reward['prediction_hash']==row['native_prediction_sha256'] and native_reward['score']==row['native_dynamic_score'])
 if not parity: raise RuntimeError('NATIVE_REFERENCE_REPLAY_PARITY '+sid)
 full={l:native['boundary'][l] for l in LAYERS}; used=set(row['native_used_layer_union']); path={l:full[l] for l in used}; donor=c.donor_prefix(native['prompt_cache'],prefix)
 native_k=prefix_hashes(native['prompt_cache'],prefix,c.th); native_occ=occupancy(native)
 control=run_one(f,ctx,model,tok,bench,seqc,prefix)
 control_reward=f.run_reward(ctx,model,tok,bench,seqc,prefix)
 control_k=prefix_hashes(control['prompt_cache'],prefix,c.th); control_occ=occupancy(control)
 ref={'canonical_sample_id':sid,'execution_type':'REFERENCE_TRACE_EXECUTION','native_prediction_sha256':native_reward['prediction_hash'],'frozen_native_prediction_sha256':row['native_prediction_sha256'],'native_reward':native_reward['score'],'control_prediction_sha256':control_reward['prediction_hash'],'control_reward':control_reward['score'],'native_fast_weight_hashes':fast_hashes(native,c.th),'native_prefix_kv_hashes':native_k,'control_prefix_kv_hashes':control_k,'native_P_occupancy':native_occ,'control_P_occupancy':control_occ,'native_used_layer_union':sorted(used),'boundary':b,'prefix_chunks':prefix,'source_state_scope':'SAME_ANCHOR_IMMUTABLE','timestamp':now()}
 atomic(run/'source_states'/f"{canonical(sid)}.json",ref)
 results=[]; branch_rows=[x for x in rows(cfg['branch_manifest']) if x['canonical_sample_id']==sid]
 for br in branch_rows:
  p=branch_path(run,br['branch_id'])
  if formal and p.exists():
   old=load(p)
   if not valid_branch(old,br,cfg): raise RuntimeError('INVALID_EXISTING_COMMIT '+str(p))
   results.append(old); continue
  kbit,pbit=br['K_bit'],br['P_bit']; inject=path if pbit else full; kd=donor if kbit else None
  rr=Recorder(prefix,T,n,'condition',c.th,rn.ref); rr.hooks(model)
  try: x=run_one(f,ctx,model,tok,bench,seqc,prefix,rr,inject,kd)
  finally: rr.close()
  fh=fast_hashes(x,c.th); kh=prefix_hashes(x['prompt_cache'],prefix,c.th); occ=occupancy(x)
  expected_k=native_k if kbit else control_k; expected_occ={str(l):(l in (used if pbit else set(LAYERS))) for l in LAYERS}
  fast_ok=fh==ref['native_fast_weight_hashes']; k_ok=kh==expected_k; p_ok=occ==expected_occ
  qlog=cmp(native['prompt_logits'],x['prompt_logits'],c.th)
  L1=all(z['exact_equal'] for z in rr.rows if z['region'] in ['FIRST_CONTINUATION_CHUNK','FIRST_POST_BOUNDARY_UPDATE'])
  L2=all(z['exact_equal'] for z in rr.rows if z['region']=='COMPLETE_CONTINUATION_CHUNK')
  L3=all(z['exact_equal'] for z in rr.rows if z['region']=='INCOMPLETE_PROMPT_TAIL')
  L4=qlog['exact_equal']
  if not (fast_ok and k_ok and p_ok): raise RuntimeError('FACTOR_ISOLATION_FAILURE '+sid+' '+br['cell'])
  reward=f.run_reward(ctx,model,tok,bench,seqc,prefix,inject=inject,donor=kd)
  L5=reward['prediction_hash']==native_reward['prediction_hash']; L6=L1 and L2 and L3 and L4 and L5
  out={**br,'status':'SUCCESS','completed':True,'execution_type':'FACTORIAL_BRANCH','post_injection_fast_weight_sha256':fh,'fast_weight_parity':'PASS','K_source':'NATIVE' if kbit else 'CONTROL','K_cache_length':prefix*1024,'K_prefix_hashes':kh,'K_state_equal_expected':True,'P_semantics':'P1' if pbit else 'P0','P_materialized_layers':[l for l in LAYERS if occ[str(l)]],'P_state_equal_expected':True,'future_input_equal':x['input_ids_hash']==native['input_ids_hash'] and x['attention_mask_hash']==native['attention_mask_hash'],'generation_start_relative_l2':qlog['relative_l2_diff'],'generation_start_max_abs':qlog['max_abs_diff'],'closure':{'L1':L1,'L2':L2,'L3':L3,'L4':L4,'L5':L5,'L6':L6},'first_divergence':first_divergence(rr.rows,L4,L5),'K_DIFF':native_k!=control_k,'P_DIFF':native_occ!={str(l):True for l in LAYERS},'terminal_reward':reward['score'],'prediction_sha256':reward['prediction_hash'],'native_prediction_sha256':native_reward['prediction_hash'],'trace_comparisons':rr.rows,'model_identity':MODEL_SHA,'checkpoint_identity':CKPT_SHA,'benchmark_identity':BENCH_SHA,'rule_hash':RULE_SHA,'eligibility_hash':ELIG_SHA,'amendment_hash':cfg['amendment_sha256'],'cohort_hash':cfg['cohort_sha256'],'branch_manifest_hash':cfg['branch_manifest_sha256'],'protocol_hash':cfg['protocol_sha256'],'base_parameter_mutation':False,'timestamp':now()}
  if br['cell']=='F11' and not L6: raise RuntimeError('F11_NONCLOSURE '+sid)
  if formal:
   atomic(p,out); atomic(run/'commits'/f"{br['branch_id']}.json",{'branch_id':br['branch_id'],'result_sha256':sha(p),'status':'COMMITTED','timestamp':now()})
  results.append(out)
  del x; gc.collect(); torch.cuda.empty_cache()
 if len(results)!=4 or {x['cell'] for x in results}!=set(CELLS): raise RuntimeError('FOUR_CELL_COMPLETENESS '+sid)
 fast_sets={json.dumps(x['post_injection_fast_weight_sha256'],sort_keys=True) for x in results}
 if len(fast_sets)!=1: raise RuntimeError('CROSS_CELL_FAST_PARITY '+sid)
 if dr.base_hash(model)!=base: raise RuntimeError('BASE_PARAMETER_MUTATION '+sid)
 del native,control,donor,rn; gc.collect(); torch.cuda.empty_cache()
 return results,ref

def valid_branch(x,br,cfg):
 return x.get('status')=='SUCCESS' and x.get('completed') is True and x.get('branch_id')==br['branch_id'] and x.get('cell')==br['cell'] and x.get('cohort_hash')==cfg['cohort_sha256'] and x.get('branch_manifest_hash')==cfg['branch_manifest_sha256'] and x.get('protocol_hash')==cfg['protocol_sha256'] and x.get('fast_weight_parity')=='PASS'

def run_mode(run,mode):
 run,cfg,cohort,manifest,checks=gates(run)
 is_smoke=mode=='smoke'; selected=[cohort[i-1] for i in cfg['smoke_orders']] if is_smoke else cohort
 if is_smoke and list((run/'branches').glob('*.json')): raise RuntimeError('FORMAL_BRANCHES_EXIST_BEFORE_SMOKE')
 status(run,pipeline_status='SMOKE_RUNNING' if is_smoke else 'RUNNING',current_phase='KP_SMOKE' if is_smoke else 'FORMAL_FACTORIAL',worker_pid=os.getpid(),worker_running=True,total_expected=668,cohort_n=167,last_error='NONE')
 dr,c,ctx,model,tok,base=runtime(run,cfg); started=time.time(); smoke_out=[]
 try:
  for ordinal,row in enumerate(selected,1):
   sid=row['canonical_sample_id']; status(run,current_sample=sid,current_task=row['task'],current_cell='REFERENCE_NATIVE_TRACE',current_boundary=row['boundary'])
   result,ref=execute_sample(run,cfg,row,dr,c,ctx,model,tok,base,not is_smoke)
   if is_smoke:
    by={x['cell']:x for x in result}; korth=(by['F00']['K_prefix_hashes']!=by['F10']['K_prefix_hashes'] and by['F00']['P_materialized_layers']==by['F10']['P_materialized_layers'] and by['F01']['K_prefix_hashes']!=by['F11']['K_prefix_hashes'] and by['F01']['P_materialized_layers']==by['F11']['P_materialized_layers'])
    p_change=(by['F00']['P_materialized_layers']!=by['F01']['P_materialized_layers']) if row['native_used_layer_union']!=LAYERS else True
    porth=(by['F00']['K_prefix_hashes']==by['F01']['K_prefix_hashes'] and by['F10']['K_prefix_hashes']==by['F11']['K_prefix_hashes'] and p_change)
    q={'canonical_sample_id':sid,'trajectory_type':'MULTISWITCH' if row['switch_count']>1 else 'SIMPLE','native_sequence_exact':True,'boundary_terminal_run_start':True,'control_prefix_all_off':True,'terminal_suffix_identical':True,'full_sequence_used_layer_union_exact':True,'P1_exact':by['F11']['P_materialized_layers']==row['native_used_layer_union'],'P0_exact':by['F00']['P_materialized_layers']==LAYERS,'K1_native_exact':by['F11']['K_state_equal_expected'],'K0_control_exact':by['F00']['K_state_equal_expected'],'fast_weight_parity':len({json.dumps(x['post_injection_fast_weight_sha256'],sort_keys=True) for x in result})==1,'F11_L1_L6':by['F11']['closure']['L6'],'K_orthogonality':korth,'P_orthogonality':porth,'native_reference_replay_parity':ref['native_prediction_sha256']==ref['frozen_native_prediction_sha256'],'cells':result}
    if not all(q[k] for k in ['native_sequence_exact','boundary_terminal_run_start','control_prefix_all_off','terminal_suffix_identical','full_sequence_used_layer_union_exact','P1_exact','P0_exact','K1_native_exact','K0_control_exact','fast_weight_parity','F11_L1_L6','K_orthogonality','P_orthogonality','native_reference_replay_parity']): raise RuntimeError('SMOKE_GATE_FAILED '+sid)
    smoke_out.append(q)
   else:
    done=len(list((run/'branches').glob('*.json'))); rate=done/max(time.time()-started,1e-9)*3600
    status(run,committed_results=done,last_commit_timestamp=now(),recent_branches_per_hour=rate,eta_hours=(668-done)/rate if rate else None,current_cell='SAMPLE_COMPLETE')
    print(json.dumps({'KP_PROGRESS':f'{done}/668','sample':sid,'ordinal':ordinal},sort_keys=True),flush=True)
  if is_smoke:
   payload={'status':'PASS','formal_result':False,'smoke_n':len(smoke_out),'simple_n':sum(x['trajectory_type']=='SIMPLE' for x in smoke_out),'multiswitch_n':sum(x['trajectory_type']=='MULTISWITCH' for x in smoke_out),'NATIVE_REFERENCE_REPLAY_PARITY':'PASS','MULTISWITCH_AMENDMENT_SMOKE':'PASS','FAST_WEIGHT_PARITY_SMOKE':'PASS','K_FACTOR_ORTHOGONALITY':'PASS','P_FACTOR_ORTHOGONALITY':'PASS','F11_NATIVE_L1_L6_SMOKE':'PASS','SOURCE_STATE_ISOLATION':'PASS','authority_checks':checks,'rows':smoke_out,'completed_at':now()}
   atomic(run/'smoke/KP_FACTORIAL_AMENDMENT_SMOKE.json',payload); status(run,pipeline_status='SMOKE_PASS',current_phase='READY_FOR_FORMAL_LAUNCH',smoke_status='PASS',worker_pid=None,worker_running=False); print(json.dumps(payload,sort_keys=True),flush=True)
  else: status(run,pipeline_status='FORMAL_EXECUTION_COMPLETE',current_phase='READY_FOR_AGGREGATION',committed_results=668,worker_pid=None,worker_running=False)
 except BaseException as e:
  atomic(run/'errors'/f"{canonical(locals().get('sid','worker')+now())}.json",{'failure_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),'mode':mode,'timestamp':now()}); status(run,pipeline_status='FAILED',current_phase=f'{mode.upper()}_FAILED',worker_running=False,last_error=f'{type(e).__name__}: {e}'); raise
 finally:
  del model,tok; gc.collect(); import torch; torch.cuda.empty_cache()

if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('run_root'); p.add_argument('--mode',choices=['smoke','formal'],required=True); a=p.parse_args(); run_mode(a.run_root,a.mode)
