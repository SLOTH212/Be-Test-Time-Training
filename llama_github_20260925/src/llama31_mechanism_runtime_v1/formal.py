"""Prepared future interface; no formal job is launched by engineering tests."""
import json,os,sys,subprocess,fcntl,math
from pathlib import Path
import mechanism as m
def validate_execution_inputs(stage,authority,rows,out):
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='0,1,2,3':raise ValueError('GPU_POLICY')
 if not out.name.startswith('llama') or out.exists() and not out.is_dir():raise ValueError('SEPARATE_LLAMA_OUTPUT_ROOT')
 dcp=Path(stage['inference_dcp']).resolve();root=Path(stage['model_path']).resolve()
 if not dcp.is_relative_to(root):raise ValueError('DCP_OUTSIDE_AUTHORIZED_FINAL')
 actual=[p for p in dcp.rglob('*') if p.is_file()]
 if not actual or any(str(p.relative_to(root)) not in stage['files'] for p in actual):raise ValueError('UNBOUND_DCP_FILES')
 if authority.get('mechanisms')!=['reverse','reset','kp','deletion','r2']:raise ValueError('MECHANISM_CONTRACT')
 scorer=Path(authority['scorer_path'])
 if m.sha(scorer)!=authority['scorer_sha256']:raise ValueError('SCORER_AUTHORITY')
 for row in rows:
  ids=json.loads(Path(row['prompt_ids_path']).read_text())
  if m.rt.canonical(ids)!=row['prompt_ids_sha256'] or len(ids)!=row['prompt_token_count'] or any(type(x) is not int or not 0<=x<128256 for x in ids):raise ValueError('PROMPT_TOKEN_AUTHORITY')
  if type(row.get('max_new_tokens')) is not int or row['max_new_tokens']<=0:raise ValueError('GENERATION_CONFIG')
  if row.get('native_score') is None or not row.get('native_prediction_sha256'):raise ValueError('NATIVE_PARITY_AUTHORITY')
 return dcp
def execute(cfg,stage,authority,rows,out):
 dcp=validate_execution_inputs(stage,authority,rows,out)
 locks=[]
 for rank in range(4):
  q=Path('/path/to/ttt/work/llama31_mechanism_runtime_v1')/f'gpu{rank}.lock';f=q.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 free=subprocess.check_output(['nvidia-smi','-i','0,1,2,3','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
 if any(float(x)<44000 for x in free.splitlines()):raise ValueError('GPU_RESOURCE_UNAVAILABLE')
 out.mkdir(parents=True,exist_ok=True)
 job={'config':{**cfg,'debug_dcp':str(dcp)},'stage':stage,'authority':authority,'rows':rows};m.atomic(out/'job.json',job)
 processes=[];logs=[]
 for rank in range(4):
  log=(out/f'worker{rank}.log').open('a');logs.append(log);processes.append(subprocess.Popen([sys.executable,'-B',__file__,str(rank),str(out)],stdout=log,stderr=subprocess.STDOUT))
 m.atomic(out/'workers.json',{'pids':[p.pid for p in processes],'physical_gpus':[0,1,2,3]})
 codes=[p.wait() for p in processes]
 for log in logs:log.close()
 if codes!=[0]*4:raise RuntimeError('FORMAL_WORKER_FAILED '+str(codes))
def worker(rank,out):
 from backend import Backend
 job=json.loads((out/'job.json').read_text());a=job['authority'];rows=[r for r in job['rows'] if m.owner(r['sample_id'])==rank];receipt=m.Receipts(out/'completed',m.rt.canonical(job));todo=[r for r in rows if receipt.completed(r['sample_id']) is None]
 if not todo:return
 b=Backend(job['config'],rank);scorer=m.load_module('authorized_formal_scorer',a['scorer_path']).score
 for row in todo:
  m.atomic(out/'status'/f'rank{rank}.json',{'rank':rank,'pid':os.getpid(),'state':'RUNNING','sample_id':row['sample_id'],'mechanism':'native'})
  sid=row['sample_id'];ids=json.loads(Path(row['prompt_ids_path']).read_text());b.ids=lambda n:ids if n==len(ids) else (_ for _ in ()).throw(ValueError('PROMPT_LENGTH'))
  b.max_new_tokens=row['max_new_tokens'];seq=row['action_sequence'];tail=len(ids)%4096
  def run(sequence,**kwargs):return b.run(sequence,tail=tail,**kwargs)
  def score(r):
   value=float(scorer(row,r))
   if not math.isfinite(value):raise ValueError('NONFINITE_SCORE')
   return value
  native,_=run(seq)
  if abs(score(native)-row['native_score'])>1e-12 or native['prediction_hash']!=row['native_prediction_sha256']:raise ValueError('NATIVE_REPLAY_PARITY')
  result={'native':native,'reverse':run(m.rt.reverse(seq))[0]};plan=m.reset_plan(seq)
  if plan['eligible']:
   tau=plan['boundary']-1;used={l for a0 in seq for l in m.MAPPING[a0]};result['reset']=run(seq,tau=tau,inject={l:b.fw.base(l) for l in used})[0]
   p=m.kp_plan(seq);native_boundary,keep=run(seq,tau=tau,retain=True);cells={}
   for name,c in p['cells'].items():cells[name]=run(p['control_sequence'],tau=tau,inject={l:keep['boundary'][l] for l in c['injection_layers']},donor=keep['donor'] if c['K_bit'] else None)[0]
   if cells['F11']['prediction_hash']!=native_boundary['prediction_hash'] or cells['F11']['final']!=native_boundary['final']:raise ValueError('F11_CLOSURE')
   result['kp']=cells;del keep
  else:result['reset']=result['kp']={'status':'INELIGIBLE','plan':plan}
  deletions=[]
  for d in row.get('deletion_targets',[]):
   m.deletion_target(seq,d['chunk_index_0'],d['layer']);r,_=run(seq,delete=(d['chunk_index_0'],d['layer']));deletions.append({'target':d,'result':r})
  if not deletions:raise ValueError('DECLARED_DELETION_TARGETS_REQUIRED')
  result['deletion']=deletions;candidates=[]
  for c in m.r2_candidates(len(seq)):
   m.atomic(out/'status'/f'rank{rank}.json',{'rank':rank,'pid':os.getpid(),'state':'RUNNING','sample_id':sid,'mechanism':'r2','candidates_complete':len(candidates)})
   r,_=run(c['sequence']);candidates.append({**c,'score':score(r),'result':r})
  winner,best,ties=m.rt.r2_winner(candidates,m.LAYERS);result['r2']={'candidates':candidates,'winner':winner,'score':best,'ties':ties}
  if b.parameter_hash()!=b.base_hash:raise ValueError('BASE_MUTATION')
  receipt.commit(sid,result)
 m.atomic(out/'status'/f'rank{rank}.json',{'rank':rank,'pid':os.getpid(),'state':'COMPLETE','completed_samples':len(rows)})
if __name__=='__main__':
 try:worker(int(sys.argv[1]),Path(sys.argv[2]))
 except BaseException as e:
  m.atomic(Path(sys.argv[2])/'status'/f'rank{sys.argv[1]}.json',{'rank':int(sys.argv[1]),'pid':os.getpid(),'state':'ERROR','error':repr(e)});raise
