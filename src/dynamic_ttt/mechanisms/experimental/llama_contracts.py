"""Explicit Llama engineering contracts; these are not frozen paper protocols."""
import importlib.util,json,hashlib,os,math
from pathlib import Path
from dynamic_ttt.mechanisms import runtime as rt
from dynamic_ttt.mechanisms.experimental import reset_resolution as v2
LAYERS=[0,6,12,18,24,30]
ACTIONS=['OFF','L0','L6','L12','L18','L24','L30','ALL']
MAPPING=rt.mapping(LAYERS)
REVISION='d04e592bb4f6aa9cfee91e2e20afa771667e1d4b'
INFERENCE_SHA='84dab6b19a522e6f7c38e3732210c4d861bb0e231ed204482089b46490876557'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()


def load_module(name,p):
 spec=importlib.util.spec_from_file_location(name,p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def action_layers(a):
 if a not in ACTIONS:raise ValueError('UNKNOWN_ACTION')
 return list(MAPPING[a])


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
