"""Prepared engineering job interface; supply a local job.json and bound inputs."""
import json,os,math
from pathlib import Path
from dynamic_ttt.mechanisms.experimental import llama_contracts as m
def worker(rank,out):
 from dynamic_ttt.mechanisms.experimental.llama_backend import Backend
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
