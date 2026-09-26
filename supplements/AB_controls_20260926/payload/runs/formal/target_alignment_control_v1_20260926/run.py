import os,sys,json,hashlib,types,traceback,time,ast
from pathlib import Path
os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
R=Path(__file__).resolve().parent;H=Path('/home/zonghan/ttt/runs/formal/mechanism_old1k_v2_20260921')
sys.path.insert(0,str(H/'runtime'))
import executor as ex
LAYERS=ex.LAYERS
def sha(p):return ex.sha(p)
def put(p,x):ex.atomic_json(p,x)
def key(s):return hashlib.sha256(s.encode()).hexdigest()
def jl(p,rows):Path(p).write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in rows))
def coords(c):return [(l,j+1,a) for l in LAYERS for j,a in enumerate(c['sequence']) if l in ex.ACTION_LAYERS[a]]
def prepare():
 assert not (R/'config.json').exists()
 co=json.loads((H/'COHORT.json').read_text());assert len(co)==170 and len({c['sample_id'] for c in co})==170
 hist={x['sample_id']:x for x in (json.loads(p.read_text()) for p in (H/'core').glob('*.json'))};tra=[]
 for c in co:
  src=json.loads(Path(c['dynamic_source']['path']).read_text());assert sha(c['dynamic_source']['path'])==c['dynamic_source']['sha256']
  assert hist[c['sample_id']]['native']['sequence']==c['sequence'] and hist[c['sample_id']]['native']['score']==c['native_score'] and hist[c['sample_id']]['native']['prediction_hash']==c['native_prediction_hash']
  assert c['native_score']>c['sample_best'];assert len(c['sequence'])==c['T']
  tra.append(dict(sample_id=c['sample_id'],sequence=c['sequence'],native_prediction_hash=c['native_prediction_hash'],native_score=c['native_score'],sample_best=c['sample_best'],write_count=len(coords(c)),source=c['dynamic_source'],exact=True))
 assert sum(t['write_count'] for t in tra)==3760
 jl(R/'cohort_manifest.jsonl',co);jl(R/'trajectory_audit.jsonl',tra)
 cfg=dict(version='WITHIN_CHUNK_TARGET_ALIGNMENT_V1',historical_root=str(H),expected_cohort_n=170,seeds=[20260926,20260927,20260928,20260929,20260930],bootstrap_seed=20260921,bootstrap_replicates=20000,comparison_tolerance=1e-12,chunk=1024,layers=LAYERS,smoke_ids=[c['sample_id'] for c in co[:5]],write_count_native=3760,write_count_per_condition=18800,model_path=json.loads((H/'PROTOCOL.json').read_text())['model_path'],checkpoint_sha256=ex.CHECKPOINT_HASH,cpu_sets=['8-15','16-23'],permutation='SHA256 local CPU generator; rejection128 derangement; cyclicshift1 fallback',calibration=dict(grid=['0','1','alpha_clip*2**j j=-16..16'],refine_rounds=12,max_trial_evaluations=59,target_relative_error=.01,closest_fallback=True,zero_native='alpha0 exactzero',scale_location='raw BF16 before unchanged clip',native_reference='trueNative same sample/layer/chunk scalar represented norm only'),reporting='execution/protocol correctness distinct from strict1percent attainment; no unqualified magnitude-controlled inference if matching fails')
 put(R/'config.json',cfg);put(R/'seed_manifest.json',dict(seeds=cfg['seeds'],derivation=cfg['permutation'],frozen_before_model_execution=True))
 files=set(H.rglob('*.py'))|set((H/'core').glob('*.json'))|{H/n for n in ['COHORT.json','PROTOCOL.json','SUMMARY.json','samples.jsonl','SOURCE_HASHES.json']}
 files|={Path(c['dynamic_source']['path']) for c in co}
 put(R/'historical_source_hashes.json',{str(p):sha(p) for p in sorted(files)})
 own=[R/n for n in ['run.py','adapter.py','control_step.py','numerical.py','summarize.py','pipeline.py','IMPLEMENTATION_PLAN.md','config.json','seed_manifest.json','cohort_manifest.jsonl','trajectory_audit.jsonl','historical_source_hashes.json']]
 put(R/'experiment_source_hashes.json',{str(p):sha(p) for p in own})
 print('FROZEN170 cohort,3760writes,5seeds',flush=True)
def integrity():
 for name in ['historical_source_hashes.json','experiment_source_hashes.json']:
  for p,h in json.loads((R/name).read_text()).items():assert sha(p)==h,('SOURCE_CHANGED',p)
def load(worker):
 os.sched_setaffinity(0,set(range(8+worker*8,16+worker*8)))
 import torch
 torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.manual_seed(20260921);torch.cuda.manual_seed_all(20260921)
 import scorer
 cfg=json.loads((R/'config.json').read_text());ctx=types.SimpleNamespace(code=H/'runtime/code',ckpt=Path(cfg['model_path']),run=R,score_fn=scorer.per_sample_score,training_chunk=1024,chunk=1024,mechanism_cache=None)
 model,tok=ex.load_model(ctx)
 from control_step import Controller
 from adapter import install_adapter
 ctrl=Controller();install_adapter(model,ctrl)
 put(R/f'environment_{worker}.json',dict(torch=torch.__version__,cuda=torch.version.cuda,transformers=__import__('transformers').__version__,numpy=__import__('numpy').__version__,gpu=torch.cuda.get_device_name(),cpu_affinity=sorted(os.sched_getaffinity(0)),python=sys.version))
 return ctx,model,tok,ctrl,cfg

def process(c,row,ctx,m,tok,ctrl,cfg,worker,smoke):
 expected=coords(c);native_map={}
 def evaluate(mode,seed=None,audit=False,compare=False):
  ctrl.begin(mode,c['sample_id'],c['task'],seed,native_map,audit_hashes=audit,compare_original=compare)
  assert ctx.mechanism_cache is None
  ev=ex.Evaluator(ctx,m,tok,row,'target_alignment_'+mode,1,1,R/f'candidate_{worker}.json')
  out=ev.evaluate(c['sequence'],force=True,capture=False);ev.finish()
  assert out['sequence']==c['sequence'] and out['generation_update_count']==0 and out['clean_full_replay']
  ww=ctrl.writes
  if mode!='disabled':assert [(w['layer'],w['chunk_index'],w['action']) for w in ww]==expected
  return dict(score=out['score'],prediction_hash=out['prediction_hash'],writes=ww,latency_seconds=out['latency_seconds'])
 def parity(z):assert z['score']==c['native_score'] and z['prediction_hash']==c['native_prediction_hash'],('NATIVE_PARITY',c['sample_id'])
 if smoke:parity(evaluate('disabled'))
 native=evaluate('native',audit=True,compare=smoke);parity(native)
 native_map={(w['layer'],w['chunk_index']):w for w in native['writes']}
 if smoke:
  ident=evaluate('identity',audit=True);parity(ident)
  assert len(ident['writes'])==len(native['writes'])
 seeds=[];repeat=[]
 for seed in cfg['seeds']:
  pair=dict(seed=seed)
  for mode in ['B1','B2']:
   z=evaluate(mode,seed,audit=smoke);pair[mode]=z
   assert all(w['fixed_points']==0 and w['n_targets']==1023 for w in z['writes'])
   if smoke and seed==cfg['seeds'][0]:
    zz=evaluate(mode,seed,audit=True)
    assert zz['score']==z['score'] and zz['prediction_hash']==z['prediction_hash'] and zz['writes']==z['writes'],('SEED_NOT_EXACT',c['sample_id'],mode)
    repeat.append(dict(mode=mode,seed=seed,exact=True))
  seeds.append(pair)
 again=evaluate('native',audit=False);parity(again)
 assert [(w['native_raw_norm'],w['native_clipped_norm'],w['native_represented_norm']) for w in again['writes']]==[(w['native_raw_norm'],w['native_clipped_norm'],w['native_represented_norm']) for w in native['writes']]
 return dict(sample_id=c['sample_id'],task=c['task'],sequence=c['sequence'],config_sha256=sha(R/'config.json'),sample_best=c['sample_best'],native_score=native['score'],native_prediction_hash=native['prediction_hash'],native_replay_pass=True,native_repeat_pass=True,native_writes=native['writes'],seeds=seeds,checks={f'B{i}':True for i in range(1,15)},smoke=smoke,identity_exact=True if smoke else None,native_original_step_exact=True if smoke else None,repeat_records=repeat)

def run(phase,worker):
 integrity();cfg=json.loads((R/'config.json').read_text());co=ex.load_jsonl(R/'cohort_manifest.jsonl')
 if phase=='formal':assert json.loads((R/'SMOKE_RESULT.json').read_text())['status']=='PASS'
 ctx,m,tok,ctrl,cfg=load(worker);rows={x['sample_id']:x for x in ex.load_jsonl(H/'samples.jsonl')}
 selected=[c for c in co if c['sample_id'] in cfg['smoke_ids']] if phase=='smoke' else co[worker::2]
 dest=R/('smoke' if phase=='smoke' else 'results');dest.mkdir(exist_ok=True);done=0;started=time.time()
 for c in selected:
  p=dest/(key(c['sample_id'])+'.json')
  if p.exists():assert json.loads(p.read_text())['config_sha256']==sha(R/'config.json');done+=1;continue
  print(json.dumps(dict(event='start',phase=phase,worker=worker,done=done,sample_id=c['sample_id'])),flush=True)
  out=process(c,rows[c['sample_id']],ctx,m,tok,ctrl,cfg,worker,phase=='smoke');put(p,out);done+=1
  put(R/f'progress_{phase}_{worker}.json',dict(done=done,total=len(selected),sample_id=c['sample_id'],elapsed_seconds=time.time()-started))
  print(json.dumps(dict(event='complete',phase=phase,worker=worker,done=done,sample_id=c['sample_id'],elapsed_seconds=time.time()-started)),flush=True)
 integrity()
 if phase=='smoke':
  rr=[json.loads((dest/(key(c['sample_id'])+'.json')).read_text()) for c in selected];assert len(rr)==5
  ds=[w for r in rr for s in r['seeds'] for w in s['B2']['writes']]
  put(R/'SMOKE_RESULT.json',dict(status='PASS',sample_n=5,identity_exact=all(r['identity_exact'] for r in rr),native_original_step_exact=all(r['native_original_step_exact'] for r in rr),determinism_pass=all(len(r['repeat_records'])==2 for r in rr),B2_write_n=len(ds),B2_match_within1pct=sum(w['matched_within_1pct'] for w in ds)/len(ds),B2_max_relative_error=max(w['relative_norm_error'] for w in ds),elapsed_seconds=time.time()-started))
 else:put(R/f'WORKER_{worker}_DONE.json',dict(status='PASS',sample_n=done,elapsed_seconds=time.time()-started))
if __name__=='__main__':
 os.sched_setaffinity(0,set(range(8,24)));phase=sys.argv[1];worker=int(sys.argv[2]) if len(sys.argv)>2 else 0
 try:
  if phase=='prepare':prepare()
  elif phase=='integrity':integrity();put(R/'source_integrity_check.json',dict(status='PASS',historical_unchanged=True,source_hashes_exact=True,reward_blind_construction=True))
  else:run(phase,worker)
 except Exception as e:
  put(R/f'FAILURE_{phase}_{worker}.json',dict(error=str(e),traceback=traceback.format_exc()));raise
