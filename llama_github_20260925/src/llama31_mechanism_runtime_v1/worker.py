import os,json,time,sys,gc,traceback
from pathlib import Path
import mechanism as m
assert os.environ['CUDA_VISIBLE_DEVICES']=='0,1,2,3'
rank=int(sys.argv[1]);assert rank in range(4)
cfg=m.validate_config(json.loads(Path(sys.argv[2]).read_text()));out=Path(sys.argv[3]);out.mkdir(parents=True,exist_ok=True)
from backend import Backend
def save(name,result):m.atomic(out/(name+'.json'),result);print('PASS',rank,name,flush=True)
def same(a,b):
 for k in ['sequence','prompt_ids_sha256','chunk_boundaries','initial','final','updates','logits_hash','prediction_hash','state_events','calls']:
  assert a[k]==b[k],('REPLAY',k)
def chain(r):
 assert r['downstream_recomputation'] and not r['stored_delta_stitching'];assert len(r['chunk_logits_hashes'])==len(r['sequence'])
 assert r['generation_updates']==0
def equivalent_prefix(native,other,boundary):
 assert [e for e in native['state_events'] if e['chunk']<boundary]==[e for e in other['state_events'] if e['chunk']<boundary]
try:
 b=Backend(cfg,rank);save('loaded',{'status':'PASS','rank':rank,'physical_gpu':rank,'parameter_hash_before':b.base_hash,'strict_debug_dcp':cfg['debug_dcp'],'pid':os.getpid()})
 results={}
 if rank==0:
  for name in ['B','G']:
   seq=m.FIXTURES[name];native,_=b.run(seq);replay,_=b.run(seq);same(native,replay);reverse,_=b.run(m.rt.reverse(seq));chain(reverse)
   assert reverse['initial']==native['initial'];assert reverse['prompt_ids_sha256']==native['prompt_ids_sha256'];assert sorted(reverse['sequence'])==sorted(seq)
   results[name]={'native':native,'repeat':replay,'reverse':reverse};save('replay_reverse_'+name,results[name])
 elif rank==1:
  assert b.adapter_test()=='PASS'
  seq=['L30']+['L12']*5;p=m.reset_plan(seq);assert p['eligible'];tau=p['boundary']-1
  native,_=b.run(seq,tau=tau)
  # Unused layers are implicitly at base per original Reset contract; preserve occupancy.
  used={l for a in seq for l in m.MAPPING[a]};inject={l:b.fw.base(l) for l in used}
  reset,_=b.run(seq,tau=tau,inject=inject);equivalent_prefix(native,reset,tau)
  assert reset['boundary']==reset['initial'];assert native['boundary']['30']!=native['initial']['30'];assert reset['sequence']==native['sequence'];assert reset['prefix_hashes']==native['prefix_hashes'];assert reset['occupancy']==native['occupancy'];chain(reset)
  results={'native':native,'reset':reset,'plan':p,'all_six_at_checkpoint_boundary':True,'unused_layers_implicitly_at_base':True,'adapter':'PASS','prompt_restart':False,'prior_chunk_replay':False,'explicit_kv_invalidation_or_copy':False};save('reset',results)
 elif rank==2:
  seq=['L30']+['L6']*6;p=m.kp_plan(seq);tau=p['prefix_chunks'];native,kept=b.run(seq,tau=tau,retain=True);control,_=b.run(p['control_sequence'],tau=tau)
  cells={}
  for name,cell in p['cells'].items():
   inject={l:kept['boundary'][l] for l in cell['injection_layers']};donor=kept['donor'] if cell['K_bit'] else None
   r,_=b.run(p['control_sequence'],tau=tau,inject=inject,donor=donor);chain(r)
   assert r['boundary']==native['boundary'];assert r['prefix_hashes']==(native if cell['K_bit'] else control)['prefix_hashes'];assert [l for l in m.LAYERS if r['occupancy'][str(l)]]==cell['injection_layers'];assert r['initial']==native['initial']
   if name=='F11':
    assert r['logits_hash']==native['logits_hash'] or r['chunk_logits_hashes'][tau:]==native['chunk_logits_hashes'][tau:]
    assert r['prediction_hash']==native['prediction_hash'];assert r['final']==native['final']
    # Prefix intentionally remains Control before the injected boundary.
    assert [e for e in r['state_events'] if e['chunk']>=tau]==[dict(e,action=p['control_sequence'][e['chunk']]) for e in native['state_events'] if e['chunk']>=tau]
   cells[name]=r;save('kp_'+name,{'plan':cell,'result':r})
  results={'native':native,'control':control,'cells':cells,'plan':p,'F11_native_continuation_closure':'PASS'};del kept;save('kp',results)
 else:
  seq=['L0','L30','L12','L18'];target=m.deletion_target(seq,1,30);native,_=b.run(seq);deleted,_=b.run(seq,delete=(1,30));equivalent_prefix(native,deleted,1)
  assert native['updates']-deleted['updates']==1;assert sum(e['omitted'] for e in deleted['calls'])==1
  e=next(e for e in deleted['calls'] if e['omitted']);n=next(e for e in native['calls'] if e['layer']==30 and e['chunk']==1)
  assert e['state_before']==n['state_before']==e['state_after'];assert n['state_after']!=e['state_after'];chain(deleted)
  r2=[]
  for c in [c for c in m.r2_candidates(4) if (c['A'],c['B'],c['tau']) in [('L30','OFF',1),('OFF','L30',2),('L0','ALL',3)]]:
   r,_=b.run(c['sequence']);chain(r);assert r['initial']==native['initial'];r2.append({'candidate':c,'result':r})
  assert len(r2)==3;results={'native':native,'deletion':deleted,'target':target,'r2':r2,'full_cpu_candidate_count':168,'gpu_subset_count':3};save('deletion_r2',results)
 assert b.parameter_hash()==b.base_hash,'PARAMETER_HASH_CHANGED'
 sid=next(f'worker{rank}_fixture_{i}' for i in range(100) if m.owner(f'worker{rank}_fixture_{i}')==rank)
 receipt=m.Receipts(out/'completed',m.sha(sys.argv[2]));receipt.commit(sid,{'rank':rank,'results':results,'DEBUG_ONLY':True,'PAPER_RESULT':False})
 save('complete',{'status':'PASS','rank':rank,'physical_gpu':rank,'sample_id':sid,'owner':m.owner(sid),'parameter_hash_before':b.base_hash,'parameter_hash_after':b.base_hash,'peak_allocated':b.torch.cuda.max_memory_allocated(),'peak_reserved':b.torch.cuda.max_memory_reserved(),'DEBUG_ONLY':True,'PAPER_RESULT':False})
except BaseException as e:
 save('error',{'status':'FAIL','error':repr(e),'traceback':traceback.format_exc()});raise
