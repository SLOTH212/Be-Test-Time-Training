import json, hashlib, math, time
from pathlib import Path
R=Path('/path/to/ttt');A=R/'audits/v3';RUN=R/'runs/qwen3_4b_stage1_resume_v3/attempt2'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
P=A/'RESUME_NUMERICAL_EQUIVALENCE_PROTOCOL_V3.json';protocol=read(P)
assert sha(P)==P.with_suffix('.sha256').read_text().split()[0]
for p,h in protocol['input_file_sha256'].items():assert sha(p)==h,p
amend=read(A/'OBSERVER_FIX1_AMENDMENT.json')
for p,h in amend['new_file_sha256'].items():assert sha(p)==h,p
receipts=read(A/'execution2/REPLICATE_CONTROLLER_RECEIPTS.json')
all_receipts=read(A/'REPLICATE_CONTROLLER_RECEIPTS.json')+receipts
checks=[];trajectories={};reference={};names=[]
def check(name,truth,detail=None):
 checks.append({'check':name,'pass':bool(truth),'detail':detail})
 if not truth:raise AssertionError((name,detail))
def identical(k,v):
 if k not in reference:reference[k]=v
 check(k,v==reference[k])
for n in [f'{c}{i}' for c in 'UR' for i in range(1,6)]:
 roles=['U'] if n[0]=='U' else ['R_PRE','R_POST']
 rs=[next((x for x in receipts if x['replica']==n and x['role']==role),None) for role in roles]
 if not all(x and x['status']=='VALID_COMPLETED_PHASE' for x in rs):continue
 names.append(n);steps=[]
 for role,receipt in zip(roles,rs):
  out=Path(receipt['output']);expected_steps=[1,2] if role=='R_PRE' else [3,4,5] if role=='R_POST' else [1,2,3,4,5]
  check(n+role+' protocol frozen before GPU',receipt['started_at']>P.stat().st_mtime and receipt['started_at']>amend['created_at'])
  check(n+role+' exact protocol',receipt['protocol_sha256']==sha(P))
  lines=[json.loads(x) for x in (out/'training.jsonl').read_text().splitlines()]
  check(n+role+' step count',[x['update_step'] for x in lines]==expected_steps)
  norms=[[json.loads(x) for x in (out/f'parameter_norm_rank{k}.jsonl').read_text().splitlines()] for k in (0,1)]
  check(n+role+' observer steps',all([x['update_step'] for x in xs]==expected_steps for xs in norms))
  for rank in (0,1):
   initial=read(out/f'initial_rank{rank}.json');parent=read(R/f'runs/qwen3_4b_stage1_closure_v2/U1/initial_rank{rank}.json')
   check(n+role+str(rank)+' exact initial parent',initial==parent)
   check(n+role+str(rank)+' clean exit',read(out/f'process_exit_rank{rank}.json')['status']=='PASS')
   rt=read(out/f'process_start_rank{rank}.json')['runtime']
   identical('runtime excluding IDs',{k:v for k,v in rt.items() if k not in ['pid','rank','replica','role']})
   check('GPU allowlist',rt['CUDA_VISIBLE_DEVICES']=='2,3' and rt['WORLD_SIZE']==2 and not rt['deterministic'])
   if role=='R_PRE':check(n+str(rank)+' save preserves raw state',read(out/f'boundary_save_rank{rank}.json')['raw_state_preserved'])
   if role=='R_POST':
    loaded=read(out/f'load_completeness_rank{rank}.json');check(n+str(rank)+' exact raw fresh load',loaded['status']=='PASS' and not loaded['differences'])
    before=read(RUN/(n+'_pre')/f'process_exit_rank{rank}.json');after=read(out/f'process_start_rank{rank}.json')
    check(n+str(rank)+' fresh process after clean exit',before['time']<after['time'] and before['runtime']['pid']!=after['runtime']['pid'])
  for idx,line in enumerate(lines):
   step=line['update_step'];ids=[]
   for rank in (0,1):
    pre=read(out/f'preforward_step{step}_rank{rank}.json');check('post load RNG preserved',pre['post_load_rng_preserved'])
    identical(f'step{step} rank{rank} canonical IDs',pre['sample_ids']);ids=pre['sample_ids']
   identical(f'step{step} progress',{k:line[k] for k in ['cumulative_tokens','record_cursor','update_step','dataloader_state','input_identity','accumulation_phase','learning_rate']})
   norm=math.sqrt(sum(xs[idx]['local_unique_parameter_squared_norm'] for xs in norms))
   obs={'step':step,'sample_ids':ids,'loss':line['global_mean_loss'],'gradient_norm_before_clip':line['grad_norm_before_clip'],'gradient_norm_after_clip':line['grad_norm_after_clip'],'parameter_norm':norm}
   check('finite observed scalar values',all(math.isfinite(obs[k]) for k in ['loss','gradient_norm_before_clip','gradient_norm_after_clip','parameter_norm']))
   steps.append(obs)
  if role!='R_PRE':
   for rank in (0,1):
    final=read(out/f'final_rank{rank}.json');identical('final scheduler',final['scheduler']);identical('final optimizer groups',final['param_groups'])
    check(n+' final contract',final['progress']['update_step']==5 and final['progress']['record_cursor']==10 and final['progress']['cumulative_tokens']==286544)
 trajectories[n]=steps
check('minimum4+4',sum(x[0]=='U' for x in names)>=4 and sum(x[0]=='R' for x in names)>=4)
payload={'status':'PASS','REPLICATE_INPUT_PARITY':'PASS','ORIGINAL_MODE_RESUME_STATE_INTEGRITY':'PASS','REMAINING_RESUME_STATE_BUG':False,'names':names,'checks':checks,'trajectories':trajectories,'excluded_or_incomplete_receipts':[x for x in all_receipts if x['status']!='VALID_COMPLETED_PHASE'],'all_attempt_optimizer_updates':sum(x['optimizer_updates_observed'] for x in all_receipts),'all_attempt_torchrun_wall_seconds':sum(x['wall_seconds'] for x in all_receipts),'gpu_hours':sum(x['wall_seconds'] for x in all_receipts)*2/3600,'FORMAL_STAGE1_TRAINING_STARTED':False,'completed_at':time.time()}
(A/'REPLICATE_INPUT_PARITY.json').write_text(json.dumps(payload,indent=2)+'\n')
print('INPUT_PARITY_PASS',names,len(checks))
