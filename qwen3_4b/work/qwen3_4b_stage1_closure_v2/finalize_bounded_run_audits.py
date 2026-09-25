from pathlib import Path
import json,hashlib,math,shutil,subprocess
R=Path('/path/to/ttt');A=R/'audits';RUN=R/'runs/qwen3_4b_stage1_closure_v2'
def read(p):return json.loads(p.read_text())
def write(n,v):(A/n).write_text(json.dumps(v,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(n):return [json.loads(x) for x in (RUN/n/'training.jsonl').read_text().splitlines()]
def ids(rr):return [i for x in rr for rank in x['rank_runtime'] for i in rank['sample_ids']]
def runcheck(n,cursor,tokens,steps):
 rr=rows(n);c=read(RUN/n/'complete.json');log=R/'logs'/('qwen3_4b_stage1_closure_v2_'+n+'.log');text=log.read_text()
 assert len(rr)==steps and c['complete'] and c['record_cursor']==cursor and c['cumulative_tokens']==tokens
 assert all(x in text for x in ['FINAL_BARRIER_PASS','RANK_0_SHUTDOWN_PASS','RANK_1_SHUTDOWN_PASS'])
 assert all(math.isfinite(x['global_mean_loss']) and x['global_valid_tokens']>0 and all(y['weights_finite'] for y in x['rank_runtime']) for x in rr)
 assert all(sorted(y['physical_gpu'] for y in x['rank_runtime'])==[2,3] for x in rr)
 cp=RUN/n/'checkpoints';cp=RUN/'R_checkpoints' if n=='R_recovered' else cp
 if n=='one_token_resume_fixed':cp=RUN/'one_token_edge_fixed/checkpoints'
 latest=read(cp/'latest.json')
 if n=='one_token_edge_fixed':latest={'slot':'slot_A','progress':read(cp/'slot_A/manifest.json')['progress'],'manifest_sha256':sha(cp/'slot_A/manifest.json')}
 assert latest['progress']['record_cursor']==cursor and latest['progress']['cumulative_tokens']==tokens
 return rr,{'status':'PASS','complete':c,'steps_observed':steps,'final_update_step':rr[-1]['update_step'],'sample_ids':ids(rr),'last_good_dcp':latest,'training_sha256':sha(RUN/n/'training.jsonl'),'log_sha256':sha(log),'both_ranks_clean_shutdown':True,'finite_loss_and_weights':True,'PHYSICAL_GPU_ALLOWLIST':[2,3],'UNAUTHORIZED_GPU_USED':False}
rr,receipt=runcheck('R_recovered',10,286544,3);u1=rows('U1')
assert [(x['update_step'],x['record_cursor'],x['cumulative_tokens'],x['learning_rate']) for x in rr]==[(x['update_step'],x['record_cursor'],x['cumulative_tokens'],x['learning_rate']) for x in u1[2:]]
assert ids(rr)==ids(u1[2:]) and len(set(ids(u1[:2]+rr)))==10
for rank in [0,1]:assert read(RUN/'R_recovered'/f'load_completeness_rank{rank}.json')['status']=='PASS'
f=read(A/'REAL_DATA_FAILURE_RECOVERY_V2.json');assert f['LAST_GOOD_DCP_SURVIVED'] and f['partial_not_promoted'] and f['FALSE_COMPLETE_PREVENTED']
assert sha(RUN/'R_checkpoints/slot_A/manifest.json')==f['last_good_manifest_sha256'] and not (RUN/'R_checkpoints/slot_B.partial').exists()
f.update(status='PASS',REAL_DATA_FAILURE_RECOVERY='PASS',RESUME_AFTER_FAILURE='PASS',recovery=receipt,resumed_sequence_exact=True,resumed_counters_steps_lr_scheduler_exact=True,recovery_note='Uncommitted failed step3 is replayed from step2; committed trajectory contains records0..9 exactly once.');write('REAL_DATA_FAILURE_RECOVERY_V2.json',f)
t=read(A/'CHECKPOINT_RETENTION_RUNTIME_V2.json');t.pop('remaining',None);t.update(status='PASS',CHECKPOINT_RETENTION_RUNTIME='PASS',current_partial_failure_preservation='PASS',current_recovery_publication='PASS',latest_slot='slot_B',latest_step=5,previous_known_good_slot='slot_A',previous_step=2);write('CHECKPOINT_RETENTION_RUNTIME_V2.json',t)
one,o=runcheck('one_token_edge_fixed',15222,32769,1);assert ids(one)==['prolong-v3-32k:train:015220','prolong-v3-32k:train:015221'];assert one[0]['global_valid_tokens']==32767
assert [x['local_tokens'] for x in one[0]['rank_runtime']]==[1,32768]
o.update(REAL_ONE_TOKEN_GPU_EDGE='PASS',real_one_token_position=15220,global_supervised_positions=32767,zero_supervised_rank0_loss_contribution=one[0]['rank_runtime'][0]['local_numerator'],no_invalid_shift_or_unintended_zero_denominator=True)
assert o['zero_supervised_rank0_loss_contribution']==0
write('REAL_ONE_TOKEN_GPU_EDGE_V2.json',o)
app=read(A/'STAGE1_ONE_TOKEN_APPLICABILITY_V2.json');app['REAL_ONE_TOKEN_GPU_EDGE']='PASS';app['actual_GPU_evidence']=str(A/'REAL_ONE_TOKEN_GPU_EDGE_V2.json');write('STAGE1_ONE_TOKEN_APPLICABILITY_V2.json',app)
odd,o=runcheck('odd_final_edge',38335,76941,2);assert ids(odd)==['prolong-v3-32k:train:038332','prolong-v3-32k:train:038333','prolong-v3-32k:train:038334'];assert odd[-1]['rank_runtime'][1]['sample_ids']==[] and odd[-1]['rank_runtime'][1]['local_tokens']==0
assert sum(x['global_valid_tokens'] for x in odd)==76938
assert odd[-1]['update_step']==2 and len(set(ids(odd)))==3
o.update(REAL_ODD_FINAL_GPU_EDGE='PASS',tail_only_debug=True,canonical_start=38332,canonical_stop=38335,real_records=3,actual_input_tokens=76941,supervised_positions=76938,empty_final_rank=1,real_final_rank=0,formal_updates_derived=19168,formal_input_tokens_derived=989920030+76941,explicit_dummy_protocol='Original worker forwards a real window record at zero loss weight on empty rank; no sample ID or input tokens counted for dummy.',completion_ownership='Original worker rank0-only complete.json followed by successful both-rank final barrier')
write('REAL_ODD_FINAL_GPU_EDGE_V2.json',o)
a=read(A/'FINAL_DISTRIBUTED_EXHAUSTION_ACCOUNTING_V2.json');a.update(status='PASS',FINAL_DISTRIBUTED_EXHAUSTION_ACCOUNTING='PASS',GPU_tail_evidence=str(A/'REAL_ODD_FINAL_GPU_EDGE_V2.json'));write('FINAL_DISTRIBUTED_EXHAUSTION_ACCOUNTING_V2.json',a)
resumed_edge,re=runcheck('one_token_resume_fixed',15224,65565,1)
assert ids(resumed_edge)==['prolong-v3-32k:train:015222','prolong-v3-32k:train:015223'] and resumed_edge[0]['update_step']==2 and resumed_edge[0]['global_valid_tokens']==32794
re.update(status='PASS',source_checkpoint='one_token_edge_fixed/checkpoints/slot_A',source_step=1,source_cursor=15222,source_tokens=32769,final_step=2,final_cursor=15224,final_tokens=65565,short_rank_after_resume=1)
write('STAGE1_POST_SPARSE_FIX_RESUME_V2.json',re)
repair=read(A/'STAGE1_SPARSE_GRAD_REPAIR_V2.json');repair.update(status='PASS_GPU_REGRESSION',one_token='PASS',opposite_rank_short_record_after_resume='PASS',odd_final='PASS');write('STAGE1_SPARSE_GRAD_REPAIR_V2.json',repair)
boundary=[]
for row in rr+odd+resumed_edge:
 for rank in row['rank_runtime']:
  for trace in rank['boundary_trace']:
   lengths=[b-a for a,b in trace['boundaries']]
   for layer,x in trace['layers'].items():
    assert x['chunks_per_document']==[(n+4095)//4096 for n in lengths]
    assert x['ntp_pairs_per_document']==[n-(n+4095)//4096 for n in lengths]
   boundary.append({'sample_id':trace['sample_id'],'documents':len(lengths),'layers':sorted(map(int,trace['layers']))})
assert any(x['documents']>1 for x in boundary)
write('STAGE1_MINIMAL_POST_FIX_REGRESSION_V2.json',{'status':'PASS','real_two_gpu_step':'PASS','document_boundary_representative':'PASS','token_accounting':'PASS','DCP':'PASS','resume':'PASS','odd_final':'PASS','one_token':'PASS','boundary_trace_checks':boundary,'scope':'Only affected bounded closure tests; no full scientific suite or throughput benchmark rerun.'})
storage=read(A/'CHECKPOINT_STORAGE_PREFLIGHT_V2.json')
for p,v in storage['paths'].items():v['free_bytes_final']=shutil.disk_usage(p).free
storage['CHECKPOINT_FREE_SPACE_BYTES']=min(v['free_bytes_final'] for v in storage['paths'].values());assert storage['CHECKPOINT_FREE_SPACE_BYTES']>=storage['CHECKPOINT_REQUIRED_MARGIN_BYTES']
storage['df_h_final']=subprocess.check_output(['df','-h',str(R/'checkpoints'),str(R/'runs')],text=True);storage['df_B1_final']=subprocess.check_output(['df','-B1',str(R/'checkpoints'),str(R/'runs')],text=True);write('CHECKPOINT_STORAGE_PREFLIGHT_V2.json',storage)
print('BOUNDED_RUN_AUDITS_PASS')
