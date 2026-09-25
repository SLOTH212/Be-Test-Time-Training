"""Llama contracts and provenance-checked immutable mechanism primitives."""
import importlib.util,json,hashlib,os,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
LAYERS=[0,6,12,18,24,30]
ACTIONS=['OFF','L0','L6','L12','L18','L24','L30','ALL']
REVISION='d04e592bb4f6aa9cfee91e2e20afa771667e1d4b'
INFERENCE_SHA='84dab6b19a522e6f7c38e3732210c4d861bb0e231ed204482089b46490876557'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load_module(name,p):
 spec=importlib.util.spec_from_file_location(name,p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
def reference():
 manifest=json.loads((ROOT/'provenance/immutable_reference.json').read_text())
 for p,h in manifest['files'].items():
  if sha(ROOT/'reference'/p)!=h:raise ValueError('IMMUTABLE_REFERENCE_HASH '+p)
 return load_module('llama_immutable_mechanism_reference',ROOT/'reference/runtime.py')
rt=reference();v2=load_module('llama_reset_v2',ROOT/'reference/reset_v2.py')
assert rt.actions(LAYERS)==ACTIONS
MAPPING=rt.mapping(LAYERS)
def action_layers(a):
 if a not in ACTIONS:raise ValueError('UNKNOWN_ACTION')
 return list(MAPPING[a])
def validate_config(c):
 for k,v in {'model_family':'llama','model_revision':REVISION,'configured_ttt_layers':LAYERS,'actions':ACTIONS,'context_length':32768,'chunk_size':4096,'ttt_lr':1.,'delta_clip':1e-5,'physical_gpu_allowlist':[0,1,2,3],'apply_then_update':True,'per_sample_reset':True,'downstream_recomputation':True,'inference_authority_sha256':INFERENCE_SHA}.items():
  if c.get(k)!=v:raise ValueError('CONFIG_AUTHORITY '+k)
 return c
def validate_trajectory(row):
 for k in ['sample_id','task','prompt_token_count','chunk_size','complete_decision_chunks','action_sequence','actions','ttt_layers','model_revision','inference_authority_sha256','prompt_ids_sha256']:
  if k not in row:raise ValueError('MISSING '+k)
 if not isinstance(row['sample_id'],str) or not row['sample_id'] or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in row['sample_id']):raise ValueError('SAMPLE_ID')
 if not row['task'] or row['actions']!=ACTIONS or row['ttt_layers']!=LAYERS:raise ValueError('ACTION_AUTHORITY')
 if row['model_revision']!=REVISION or row['inference_authority_sha256']!=INFERENCE_SHA:raise ValueError('MODEL_OR_INFERENCE_AUTHORITY')
 n=row['prompt_token_count'];chunks=row['complete_decision_chunks']
 if type(n) is not int or not 4096<=n<=32768 or row['chunk_size']!=4096 or type(chunks) is not int or chunks!=n//4096 or len(row['action_sequence'])!=chunks:raise ValueError('CHUNK_GEOMETRY')
 for a in row['action_sequence']:action_layers(a)
 if row.get('native_score') is not None and (type(row['native_score']) not in (int,float) or not math.isfinite(row['native_score'])):raise ValueError('NATIVE_SCORE')
 if len(row['prompt_ids_sha256'])!=64:raise ValueError('PROMPT_HASH')
 return row
def reset_plan(seq):
 for a in seq:action_layers(a)
 p=v2.screen(seq,ACTIONS);p['boundary']=rt.reset_rules(LAYERS).terminal_run_boundary(seq) if p['eligible'] else None
 p['classification']='V2_CANDIDATE_ENGINEERING_ONLY';return p
def kp_plan(seq):
 p=reset_plan(seq)
 if not p['eligible']:raise ValueError('RESET_INELIGIBLE')
 return rt.kp_plan(seq,p['boundary'],LAYERS)
def deletion_target(seq,chunk,layer):
 if type(chunk) is not int or not 0<=chunk<len(seq) or layer not in action_layers(seq[chunk]):raise ValueError('DELETION_TARGET')
 return {'chunk_index_0':chunk,'layer':layer,'omit_update_only':True,'action_sequence':list(seq)}
def r2_candidates(n):return list(rt.r2_candidates(n,LAYERS))
def owner(sample_id):return int(hashlib.sha256(sample_id.encode()).hexdigest(),16)%4
def atomic(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name('.'+p.name+'.'+str(os.getpid())+'.tmp')
 with tmp.open('w') as f:json.dump(x,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
class Receipts:
 def __init__(self,root,authority):self.root=Path(root);self.authority=authority
 def path(self,sid):
  if not sid or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in sid):raise ValueError('SAMPLE_ID')
  return self.root/(sid+'.json')
 def completed(self,sid):
  p=self.path(sid)
  if not p.exists():return None
  r=json.loads(p.read_text())
  if r.get('status')!='COMMITTED' or r.get('authority')!=self.authority or r.get('sample_id')!=sid or rt.canonical(r.get('result'))!=r.get('result_sha256'):raise ValueError('INVALID_COMMITTED_RECEIPT')
  return r
 def commit(self,sid,result):
  old=self.completed(sid)
  if old:
   if old['result_sha256']!=rt.canonical(result):raise ValueError('CONFLICTING_RESULT')
   return old
  r={'status':'COMMITTED','sample_id':sid,'authority':self.authority,'result':result,'result_sha256':rt.canonical(result)};atomic(self.path(sid),r);return r
FIXTURES={'A':['L12']*4,'B':['L0','L6','L12','L18'],'C':['L6']*2+['L24']*4,'D':['L12']+['L0']*5,'E':['L12']+['L6']*6,'F':['OFF','L30','L30','L30'],'G':['L0','ALL','L24','L30']}
def fixture(sid,seq,ids):
 return validate_trajectory({'sample_id':sid,'task':'synthetic_neutral_diagnostic','prompt_token_count':len(ids),'chunk_size':4096,'complete_decision_chunks':len(seq),'action_sequence':list(seq),'native_score':None,'actions':list(ACTIONS),'ttt_layers':list(LAYERS),'model_revision':REVISION,'inference_authority_sha256':INFERENCE_SHA,'prompt_ids_sha256':rt.canonical(ids),'DEBUG_ONLY':True,'PAPER_RESULT':False})
def handoff(stage_path,trajectory_authority,execute=False):
 if not execute or not stage_path or not trajectory_authority:raise ValueError('EXPLICIT_EXECUTE_AND_FORMAL_AUTHORITIES_REQUIRED')
 stage=json.loads(Path(stage_path).read_text());a=json.loads(Path(trajectory_authority).read_text())
 if stage.get('schema')!='LLAMA_STAGE2_FINAL_AUTHORITY_V1' or stage.get('status')!='FINAL' or stage.get('model_revision')!=REVISION or stage.get('DEBUG_ONLY') is not False:raise ValueError('INVALID_STAGE2_FINAL')
 model=Path(stage['model_path']).resolve()
 if not stage.get('files') or any(not (model/p).resolve().is_relative_to(model) or sha(model/p)!=h for p,h in stage['files'].items()):raise ValueError('FINAL_FILES')
 if a.get('schema')!='LLAMA_DYNAMIC_TRAJECTORY_AUTHORITY_V1' or a.get('DEBUG_ONLY') is not False or a.get('FORMAL_RESULTS') is not True or a.get('parent_stage2_sha256')!=sha(stage_path):raise ValueError('INVALID_FORMAL_DYNAMIC')
 if a.get('chain')!=['Stage2_FINAL','Fixed8','SampleBest','Dynamic_LB','improved_trajectories']:raise ValueError('FORMAL_CHAIN')
 p=Path(a['trajectory_path'])
 if sha(p)!=a['trajectory_sha256']:raise ValueError('TRAJECTORY_FILE_HASH')
 rows=[validate_trajectory(json.loads(line)) for line in p.read_text().splitlines() if line.strip()]
 if len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('DUPLICATE_SAMPLE')
 for r in rows:
  if r.get('DEBUG_ONLY') is not False or r.get('parent_stage2_sha256')!=sha(stage_path) or r.get('completed') is not True:raise ValueError('DEBUG_OR_UNCOMMITTED_TRAJECTORY')
 return stage,a,rows
