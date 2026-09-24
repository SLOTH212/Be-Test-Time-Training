#!/usr/bin/env python3
import os
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
from pathlib import Path
import sys,json,csv,hashlib,time,types,importlib.util,gc,traceback
O=Path(__file__).resolve().parent;LAYERS=[0,6,12,18,24];A=['OFF','L0','L6','L12','L18','L24','ALL'];bridge='--bridge' in sys.argv
import torch
torch.set_num_threads(16);os.sched_setaffinity(0,range(8,24))
def j(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def canon(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def put(n,x):
 p=O/n;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(p)
def append(n,x):
 p=O/n;exists=p.exists()
 with p.open('a',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(x));
  if not exists:w.writeheader()
  w.writerow(x)
def imp(p,n):
 spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);sys.modules[n]=m;spec.loader.exec_module(m);return m
pre=j(O/'PRE_FLIGHT_V2.json');cohort=j(O/'V2_STAGE1_COHORT.json');sources=j(O/'V2_SOURCE_SHA256.json');env=j(O/'V2_ENVIRONMENT.json');assert pre['status']=='PASS' and j(O/'V2_PRECHECK_RESULTS.json')['status']=='PASS_EXACT';assert sha(O/'V2_STAGE1_COHORT.json')==pre['cohort_sha256']
for p,h in sources['files'].items():assert sha(p)==h,('SOURCE_HASH',p)
for p,h in pre['source_hashes'].items():assert sha(p)==h,('AUTHORITY_HASH',p)
if not bridge:
 protocol=j(O/'FROZEN_V2_STAGE1_PROTOCOL.json');assert sha(__file__)==protocol['formal_driver_sha256'];assert sha(O/'V2_STAGE1_COHORT.json')==protocol['cohort_sha256'];assert j(O/'V2_INSTRUMENTATION_BRIDGE.json')['status']=='PASS_EXACT';assert not (O/'V2_TRAJECTORIES.jsonl').exists(),'NO_SILENT_RESTART_OR_DUPLICATES'
else:assert not (O/'V2_INSTRUMENTATION_BRIDGE.json').exists(),'NO_SILENT_BRIDGE_RESTART'
BEN=Path('/home/zonghan/ttt/benchmarks/ruler_16k_standard_13task500_v1');by={x['sample_id']:x for x in cohort['samples']};rows={}
for line in (BEN/'sample_manifest.jsonl').open():
 x=json.loads(line)
 if x['sample_id'] in by:rows[x['sample_id']]=x
runner=imp(O/'source/reference_executor_v2.py','reference_executor_v2');scorer=imp('/home/zonghan/ttt/code/ruler/tools/common.py','v2_frozen_scorer');ctx=types.SimpleNamespace(run=O,code=Path('/home/zonghan/ttt/code/ntp'),ckpt=Path(env['model_path']),score_fn=scorer.per_sample_score)
print('V2_MODEL_LOAD',('BRIDGE' if bridge else 'FORMAL173'),flush=True);model,tok=runner.load_model(ctx);base=runner.base_hash(model);isolation=[];records=[];runordinal=0
if bridge:
 plan=[(sid,'NATIVE') for sid in j(O/'V2_PRECHECK_PROTOCOL.json')['repeat_samples']]
else:plan=[(x['sample_id'],a) for x in cohort['samples'] for a in A+['NATIVE']]
for sid,action in plan:
 x=by[sid];r=rows[sid];seq=x['native_sequence'] if action=='NATIVE' else x['fixed'][action]['sequence'];runordinal+=1;runid=('BRIDGE' if bridge else 'FORMAL')+'_'+str(runordinal).zfill(4)+'_'+hashlib.sha256((sid+'|'+action).encode()).hexdigest()[:12];before=runner.base_hash(model);start=time.time()
 try:
  assert before==base
  assert hashlib.sha256(r['input'].encode()).hexdigest()==x['input_sha256'] and r['max_new_tokens']==x['max_new_tokens']
  assert len(seq)==x['T'];expected=x['native_sequence_sha256'] if action=='NATIVE' else x['fixed'][action]['sequence_sha256'];assert canon(seq)==expected
  for l in LAYERS:
   m=model.model.layers[l].mlp;m._captured_final_state=None;m._events=[];m._dynamic_sequence=None
  torch.manual_seed(20260915);torch.cuda.manual_seed_all(20260915)
  ev=runner.Evaluator(ctx,model,tok,r,'v2_stage1',runordinal,len(plan),O/'unused_candidate_progress.json')
  assert hashlib.sha256(ev.enc0.input_ids.numpy().tobytes()).hexdigest()==x['input_token_sha256'];assert hashlib.sha256(ev.enc0.attention_mask.numpy().tobytes()).hexdigest()==x['attention_mask_sha256'];assert ev.full_chunks==x['T']
  rec=ev.evaluate(tuple(seq),capture=False,force=True);assert not ev.records;ev.finish();del ev
  assert rec['sequence']==seq and rec['sequence_hash']==expected and rec['generation_update_count']==0 and rec['clean_full_replay'] and rec['recompute_downstream_updates'] and not rec['stored_delta_stitching']
  after=runner.base_hash(model);assert before==after==base
  for l in LAYERS:
   m=model.model.layers[l].mlp;m._captured_final_state=None;m._events=[];m._dynamic_sequence=None
  iso={'run_id':runid,'sample_id':sid,'action':action,'checkpoint_unchanged':before==after==base,'pre_checkpoint_digest':before,'post_checkpoint_digest':after,'state_reset_pass':True,'new_evaluator':True,'fresh_generate_cache':True,'candidate_cache_used':False,'retry':False,'status':'SUCCESS'};isolation.append(iso);append('BRIDGE_STATE_ISOLATION.csv' if bridge else 'V2_STATE_ISOLATION.csv',iso)
  z={'run_id':runid,'sample_id':sid,'task':x['task'],'action':action,'T':x['T'],'sequence':seq,'sequence_sha256':expected,'historical_sequence_identity_match':True,'input_sha256':x['input_sha256'],'input_token_sha256':x['input_token_sha256'],'attention_mask_sha256':x['attention_mask_sha256'],'reference_sha256':canon(r['reference']),'generation_budget':r['max_new_tokens'],'score':rec['score'],'prediction_hash':rec['prediction_hash'],'historical_score':x['historical_native_score'] if action=='NATIVE' else x['fixed'][action]['score'],'historical_prediction_hash':x['historical_native_prediction_hash'] if action=='NATIVE' else x['fixed'][action]['prediction_hash'],'materialized_layers':LAYERS,'functional_layers':LAYERS,'layer_update_counts':rec['layer_update_counts'],'generation_update_count':0,'tail_update_count':0,'checkpoint_unchanged':True,'state_reset_pass':True,'runtime_seconds':time.time()-start,'gpu_peak_memory_bytes':rec['gpu_peak_memory_bytes'],'status':'SUCCESS','source_sha256':sources['reference_executor_v2_sha256']};records.append(z)
  if bridge:
   old=j(O/'precheck/run_metadata'/('V2_REPEAT_A_'+sid.replace(':','_')+'.json'))['result'];z['bridge_prediction_match']=z['prediction_hash']==old['prediction_hash'];z['bridge_reward_match']=z['score']==old['reward'];assert z['bridge_prediction_match'] and z['bridge_reward_match'],'INSTRUMENTATION_BRIDGE_MISMATCH';put('bridge_runs/'+runid+'.json',z)
  else:
   put('formal_runs/'+runid+'.json',z)
   with (O/'V2_TRAJECTORIES.jsonl').open('a') as f:f.write(json.dumps(z,sort_keys=True,allow_nan=False)+'\n')
   row={k:v for k,v in z.items() if k not in ['sequence','materialized_layers','functional_layers','layer_update_counts']};append('V2_NATIVE_173.csv' if action=='NATIVE' else 'V2_FIXED7_173.csv',row)
  print('V2_RUN_COMPLETE',runordinal,'/',len(plan),sid,action,f'{time.time()-start:.2f}s',flush=True);del rec;gc.collect()
 except Exception as e:
  err={'run_id':runid,'sample_id':sid,'action':action,'error_type':type(e).__name__,'traceback':traceback.format_exc(),'attempt':1,'retry_performed':False,'completed_before_error':len(records),'pre_checkpoint_digest':before,'post_checkpoint_digest':runner.base_hash(model)};put('errors/'+runid+'.json',err);put('V2_STAGE1_ERROR.json',{'status':'BLOCKED_EXECUTION_ERROR','error':err,'bridge':bridge});raise
if bridge:put('V2_INSTRUMENTATION_BRIDGE.json',{'status':'PASS_EXACT','trajectory_n':4,'prediction_match_n':4,'reward_match_n':4,'checkpoint_unchanged_n':4,'records':records})
else:
 assert len(records)==1384 and len({(x['sample_id'],x['action']) for x in records})==1384;put('V2_FORMAL_COMPLETION.json',{'status':'ALL_1384_COMPLETED','trajectory_n':len(records),'checkpoint_unchanged_n':len(isolation),'state_reset_pass_n':len(isolation),'error_n':0,'retry_n':0,'protocol_sha256':sha(O/'FROZEN_V2_STAGE1_PROTOCOL.json')})
print('V2_BRIDGE_PASS_EXACT' if bridge else 'V2_FORMAL_REPLAY_COMPLETE',flush=True)
