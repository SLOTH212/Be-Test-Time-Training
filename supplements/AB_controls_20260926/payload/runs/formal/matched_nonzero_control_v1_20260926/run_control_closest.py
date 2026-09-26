#!/usr/bin/env python3
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
import sys,json,hashlib,importlib.util,types,traceback,time,platform
from pathlib import Path
from collections import Counter
O=Path(__file__).resolve().parent
H=Path('/home/zonghan/ttt/runs/formal/mechanism_old1k_v2_20260921')
LAYERS=[0,6,12,18,24]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n');os.replace(t,p)
def jl(p,xs):Path(p).write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in xs))
def readjl(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def sidkey(s):return hashlib.sha256(s.encode()).hexdigest()
def historical():return {x['sample_id']:x for x in (json.loads(p.read_text()) for p in (H/'core').glob('*.json'))}
def prepare():
 assert not (O/'config.json').exists(),'Config already frozen'
 co=json.loads((H/'COHORT.json').read_text());hist=historical();selected=[c for c in co if c['nontrivial_reset']];assert len(selected)==157
 branches=[]
 for c in co:
  seq=c['sequence'];T=len(seq);sw=[i for i in range(1,T) if seq[i]!=seq[i-1]];late=seq[T//2:]
  assert c['tau']==sw[-1] and c['boundary']==sw[-1]+1
  elig=(sw[0]+1)/T<=.25 and max(Counter(late).values())/len(late)>=.8
  assert c['reset_eligible']==elig and c['nontrivial_reset']==(elig and any(z!='OFF' for z in seq[:c['tau']]))
  if not c['nontrivial_reset']:continue
  p=seq[:c['tau']];targets=LAYERS if 'ALL' in p else [int(next(a for a in p if a!='OFF')[1:])]
  assert targets==c['deletion_targets'];ds=hist[c['sample_id']]['deletion'];assert [d['layer'] for d in ds]==targets
  nh={e['layer']:e['before'] for e in hist[c['sample_id']]['K']['native']['interventions']}
  for d in ds:
   for e in d['result']['interventions']:
    assert e['before']==nh[e['layer']]
    assert e['reset']==(e['layer']==d['layer'])
    if e['layer']!=d['layer']:assert e['after']==nh[e['layer']]
    else:assert e['before']!=e['after']
   branches.append(dict(sample_id=c['sample_id'],task=c['task'],layer=d['layer'],tau=c['tau'],boundary=c['boundary'],deletion_score=d['result']['score'],deletion_prediction_hash=d['result']['prediction_hash']))
 assert len(branches)==349
 jl(O/'cohort_manifest.jsonl',selected);jl(O/'branch_manifest.jsonl',branches)
 frozen={str(p):sha(p) for p in sorted(set(list(H.rglob('*.py'))+list((H/'core').glob('*.json'))+[H/n for n in ['COHORT.json','PROTOCOL.json','SUMMARY.json','samples.jsonl','SOURCE_HASHES.json']]))}
 # Also freeze upstream winning trajectory artifacts explicitly referenced in the frozen cohort.
 for c in selected:
  p=Path(c['dynamic_source']['path']);assert sha(p)==c['dynamic_source']['sha256'];frozen[str(p)]=sha(p)
 put(O/'historical_source_hashes.json',frozen)
 cfg=dict(version='MAGNITUDE_MATCHED_NONZERO_CONTROL_V1',historical_root=str(H),expected_cohort_n=157,branch_n=349,random_evaluation_n=1745,seeds=[20260926,20260927,20260928,20260929,20260930],bootstrap_seed=20260921,bootstrap_replicates=20000,comparison_tolerance=1e-12,smoke_ids=[c['sample_id'] for c in selected[:5]],cpu_sets=['8-15','16-23'],chunk=1024,model_path=json.loads((H/'PROTOCOL.json').read_text())['model_path'],checkpoint_sha256=json.loads((H/'PROTOCOL.json').read_text())['model_sha256'],execution='native sequence clean full replay; existing target-only boundary inject; transparent prefix KV hash observer',random_direction='local SHA256-derived generator; FP32 isotropic Gaussian; scalar BF16 calibration; no reward access',calibration_relative_tolerance=.01,calibration_max_bracket_steps=24,calibration_max_binary_steps=32,calibration_max_evaluations=57,calibration_search_relative_tolerance=.001,reward_blind_construction_reviewed=True,cohort_manifest_sha256=sha(O/'cohort_manifest.jsonl'),branch_manifest_sha256=sha(O/'branch_manifest.jsonl'),historical_hash_manifest_sha256=sha(O/'historical_source_hashes.json'),A1_frozen=json.loads((H/'SUMMARY.json').read_text())['results']['deletion'])
 put(O/'config.json',cfg)
 local={str(O/n):sha(O/n) for n in ['run_control.py','pipeline.py','calibration.py','summarize_new.py','IMPLEMENTATION_PLAN.md','config.json','cohort_manifest.jsonl','branch_manifest.jsonl','historical_source_hashes.json']}
 put(O/'experiment_source_hashes.json',local)
 print(json.dumps({'prepared':True,'samples':157,'branches':349,'evaluations':1745}),flush=True)
def integrity():
 for name in ['historical_source_hashes.json','experiment_source_hashes.json','amendment_source_hashes.json']:
  for p,h in json.loads((O/name).read_text()).items():assert sha(p)==h,('SOURCE_CHANGED',p)
 am=json.loads((O/'PROTOCOL_AMENDMENT_01.json').read_text())
 for name,h in am['retained_results'].items():assert sha(O/'results'/name)==h,('RETAINED_RESULT_CHANGED',name)
 return True

def load_runtime(worker):
 os.sched_setaffinity(0,set(range(8+8*worker,16+8*worker)))
 saved=sys.argv;sys.argv=[str(H/'run.py')]
 try:
  spec=importlib.util.spec_from_file_location('historical_mechanism',H/'run.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 finally:sys.argv=saved
 os.sched_setaffinity(0,set(range(8+8*worker,16+8*worker)))
 mod.R=O;mod.W=worker
 import torch
 torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.manual_seed(20260921);torch.cuda.manual_seed_all(20260921)
 ctx=types.SimpleNamespace(code=H/'runtime/code',ckpt=Path(mod.P['model_path']),run=O,score_fn=mod.scorer.per_sample_score,training_chunk=1024,chunk=1024,mechanism_cache=None)
 model,tok=mod.ex.load_model(ctx)
 env=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),numpy=__import__('numpy').__version__,transformers=__import__('transformers').__version__,platform=platform.platform(),cpu_affinity=sorted(os.sched_getaffinity(0)),config_sha256=sha(O/'config.json'))
 put(O/f'environment_{worker}.json',env)
 return mod,ctx,model,tok

def process_one(c,mod,ctx,m,tok,row,hist,cfg,smoke=False):
 import torch
 from calibration import calibrate
 from inference_model.hf_qwen3.modeling_qwen3 import TTTDynamicCache
 ex=mod.ex;tau=c['tau'];seq=c['sequence'];prefix=tau*1024
 class ObservingCache(TTTDynamicCache):
  def __init__(self,expected=None):super().__init__(config=m.config);self.expected=expected;self.prefix_hashes={}
  def update(self,k,v,layer_idx,cache_kwargs=None):
   k,v=super().update(k,v,layer_idx,cache_kwargs)
   if layer_idx not in self.prefix_hashes and k.shape[-2]>1:
    assert k.shape[-2]>=prefix
    z=dict(key=ex.tensor_hash(k[...,:prefix,:]),value=ex.tensor_hash(v[...,:prefix,:]),prefix_tokens=prefix)
    if self.expected is not None:assert z==self.expected[layer_idx],('PREFIX_KV_NOT_NATIVE',c['sample_id'],layer_idx)
    self.prefix_hashes[layer_idx]=z
   return k,v
 def branch(label,inject=None,reset=(),expected=None):
  cache=ObservingCache(expected)
  out,weights=mod.eval_branch(ctx,m,tok,row,seq,tau,reset=reset,inject=inject,cache=cache,trace=False,label=label)
  assert set(cache.prefix_hashes)==set(range(len(m.model.layers)))
  assert all(cache.ttt_states[l][2] is not None for l in LAYERS)
  assert out['sequence']==seq and out['generation_update_count']==0 and out['clean_full_replay']
  hashes=cache.prefix_hashes;del cache
  return out,weights,hashes
 native,W,kv=branch('matched_native')
 assert native['score']==c['native_score'] and native['prediction_hash']==c['native_prediction_hash']
 old=hist[c['sample_id']];oldn={e['layer']:e['before'] for e in old['K']['native']['interventions']}
 nh={l:ex.tensor_hash(W[l]) for l in LAYERS};assert nh==oldn
 ds={d['layer']:d['result'] for d in old['deletion']};assert list(ds)==c['deletion_targets']
 def check(out,target,after):
  ev=out['interventions'];assert len(ev)==5 and {e['layer'] for e in ev}==set(LAYERS)
  for e in ev:
   l=e['layer'];assert e['chunk_index']==tau+1 and e['before']==nh[l]
   assert e['after']==(after if l==target else nh[l]),('OTHER_LAYER_CHANGED',l)
  assert out['sequence']==seq
 records=[];det_records=[];del_records=[]
 for layer in c['deletion_targets']:
  wn=W[layer].to(device='cuda');wc=m.model.layers[layer].mlp.down_proj.weight.detach()
  if smoke:
   z,ww,kk=branch(f'deletion_validation_L{layer}',reset=[layer],expected=kv);del ww
   assert z['score']==ds[layer]['score'] and z['prediction_hash']==ds[layer]['prediction_hash']
   check(z,layer,ex.tensor_hash(wc));del_records.append(dict(layer=layer,score=z['score'],prediction_hash=z['prediction_hash'],matched=True))
  seeds=[]
  for seed in cfg['seeds']:
   pert,diag=calibrate(wn,wc,c['sample_id'],layer,seed)
   assert diag['realized_norm']>0,'ZERO_PERTURBATION'
   if not diag['matched_within_1pct']:
    assert diag['bracketed'] and diag['binary_steps']==32 and diag['calibration_iterations']==1+diag['bracket_steps']+diag['binary_steps'], 'UNVERIFIED_CLOSEST_FALLBACK'
    put(O/'quantization_exceptions'/f'{sidkey(c["sample_id"])}_L{layer}_S{seed}.json',dict(sample_id=c['sample_id'],layer=layer,seed=seed,diagnostics=diag,policy='frozen deterministic closest; user permitted quantization fallback'))
   ph=ex.tensor_hash(pert)
   z,ww,kk=branch(f'matched_L{layer}_S{seed}',inject={layer:pert},expected=kv);del ww
   check(z,layer,ph)
   audit=dict(A6=True,A7=True,A8=True,A9=True,A10=True)
   rec=dict(seed=seed,score=z['score'],prediction_hash=z['prediction_hash'],diagnostics=diag,checks=audit,perturbed_weight_hash=ph,interventions=z['interventions'],prefix_kv_hashes=kk,latency_seconds=z['latency_seconds'])
   seeds.append(rec)
   if smoke and seed==cfg['seeds'][0]:
    p2,d2=calibrate(wn,wc,c['sample_id'],layer,seed);assert ex.tensor_hash(p2)==ph and d2==diag
    zz,www,kkk=branch(f'determinism_L{layer}_S{seed}',inject={layer:p2},expected=kv);del www,p2
    check(zz,layer,ph);assert zz['score']==z['score'] and zz['prediction_hash']==z['prediction_hash']
    det_records.append(dict(layer=layer,seed=seed,weight_exact=True,prediction_exact=True,score_exact=True))
   del pert
  records.append(dict(layer=layer,deletion_score=ds[layer]['score'],deletion_prediction_hash=ds[layer]['prediction_hash'],seeds=seeds))
  del wn
 again,ww,kk=branch('matched_native_repeat',expected=kv);del ww
 assert again['score']==native['score'] and again['prediction_hash']==native['prediction_hash']
 r=dict(sample_id=c['sample_id'],task=c['task'],tau=tau,boundary=c['boundary'],sequence=seq,config_sha256=sha(O/'config.json'),native_score=native['score'],native_prediction_hash=native['prediction_hash'],native_replay_pass=True,native_repeat_pass=True,native_boundary_hashes=nh,native_prefix_kv_hashes=kv,deletion_score=sum(ds[l]['score'] for l in ds)/len(ds),random_score=sum(sum(s['score'] for s in b['seeds'])/len(b['seeds']) for b in records)/len(records),branches=records,checks={f'A{i}':True for i in range(1,11)},smoke=smoke,determinism_records=det_records,deletion_replay_records=del_records)
 del W;return r

def run(phase,worker):
 integrity();cfg=json.loads((O/'config.json').read_text());co=readjl(O/'cohort_manifest.jsonl');hist=historical()
 if phase=='formal':assert json.loads((O/'SMOKE_RESULT.json').read_text())['status']=='PASS'
 mod,ctx,m,tok=load_runtime(worker);rows={r['sample_id']:r for r in readjl(H/'samples.jsonl')}
 selected=[c for c in co if c['sample_id'] in cfg['smoke_ids']] if phase=='smoke' else co[worker::2]
 dest=O/('smoke' if phase=='smoke' else 'results');dest.mkdir(exist_ok=True)
 done=0;started=time.time()
 for c in selected:
  p=dest/(sidkey(c['sample_id'])+'.json')
  if p.exists():
   old=json.loads(p.read_text());assert old['config_sha256']==sha(O/'config.json');done+=1;continue
  print(json.dumps(dict(event='start',phase=phase,worker=worker,sample_id=c['sample_id'],done=done,total=len(selected))),flush=True)
  r=process_one(c,mod,ctx,m,tok,rows[c['sample_id']],hist,cfg,smoke=phase=='smoke');put(p,r);done+=1
  put(O/f'progress_{phase}_{worker}.json',dict(phase=phase,worker=worker,done=done,total=len(selected),sample_id=c['sample_id'],elapsed_seconds=time.time()-started))
  print(json.dumps(dict(event='completed',phase=phase,worker=worker,sample_id=c['sample_id'],done=done,total=len(selected),elapsed_seconds=time.time()-started)),flush=True)
 integrity()
 if phase=='smoke':
  rr=[json.loads((dest/(sidkey(c['sample_id'])+'.json')).read_text()) for c in selected]
  assert len(rr)==5 and all(r['determinism_records'] for r in rr)
  put(O/'SMOKE_RESULT.json',dict(status='PASS',sample_n=5,branch_n=sum(len(r['branches']) for r in rr),random_evaluation_n=sum(len(b['seeds']) for r in rr for b in r['branches']),determinism_pass=True,determinism_repeat_n=sum(len(r['determinism_records']) for r in rr),deletion_replay_n=sum(len(r['deletion_replay_records']) for r in rr),sample_ids=[r['sample_id'] for r in rr],elapsed_seconds=time.time()-started))
 else:put(O/f'WORKER_{worker}_DONE.json',dict(status='PASS',sample_n=done,elapsed_seconds=time.time()-started))

if __name__=='__main__':
 os.sched_setaffinity(0,set(range(8,24)))
 phase=sys.argv[1];worker=int(sys.argv[2]) if len(sys.argv)>2 else 0
 try:
  if phase=='prepare':prepare()
  elif phase=='integrity':integrity();put(O/'source_integrity_check.json',dict(status='PASS',historical_unchanged=True,experiment_sources_unchanged=True,reward_blind_construction=True,amendment_checked=True))
  else:run(phase,worker)
 except Exception as e:
  put(O/f'FAILURE_closest_{phase}_{worker}.json',dict(status='FAIL',phase=phase,error=str(e),traceback=traceback.format_exc()))
  if 'MAGNITUDE' in str(e) or 'ZERO_PERTURBATION' in str(e) or 'ZERO_TARGET' in str(e):
   (O/'BLOCKED_MAGNITUDE_MATCHING_REPORT.md').write_text('# BLOCKED_MAGNITUDE_MATCHING\n\n'+traceback.format_exc()+'\nNo cohort or seed changes. See magnitude_failure.json if present. Formal execution halted.\n')
  raise
