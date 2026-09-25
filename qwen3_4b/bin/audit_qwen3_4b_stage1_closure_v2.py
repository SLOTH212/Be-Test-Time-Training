import json,hashlib,math,sys,shutil,subprocess,importlib.util
from pathlib import Path
R=Path('/path/to/ttt');A=R/'audits';RUN=R/'runs/qwen3_4b_stage1_closure_v2'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
checks={};evidence={}
def check(name,value):checks[name]=bool(value)
def audited(name):
 p=A/name;evidence[str(p)]=sha(p);return read(p)
def rows(name):return [json.loads(x) for x in (RUN/name/'training.jsonl').read_text().splitlines()]
def ids(rr):return [i for x in rr for rank in x['rank_runtime'] for i in rank['sample_ids']]
v1p=R/'provenance/QWEN3_4B_STAGE1_1B_32K_GPU23_PRELAUNCH_AUTHORITY_V1.json';v1=read(v1p)
check('V1_authority_unchanged',sha(v1p)=='1007b91903ce9812140655aa92779f93749cb43da99ac6c4180260f5944f9490')
check('V1_bound_evidence_unchanged',all(sha(p)==h for p,h in v1['file_sha256'].items()))
import yaml
cfgp=R/'configs/qwen3_4b_stage1_1b_32k_gpu23_formal_v1.yaml';cfg=yaml.safe_load(cfgp.read_text())
check('scientific_config_byte_exact',sha(cfgp)=='75740a4513f3f74302f276e1be38314a98ac095ed68d5ede458f4f8774576ff8')
check('GPU_allowlist_exact',cfg['physical_gpu_allowlist']=='2,3' and cfg['world_size']==2)
oldmanifest=read(cfg['runtime_source_manifest']);check('original_runtime_unchanged',all(sha(R/'src/training_runtime'/p)==h for p,h in oldmanifest.items()))
newmanifest=audited('STAGE1_CLOSURE_RUNTIME_SOURCE_MANIFEST_V2.json')
check('effective_runtime_source_hashes',all(sha(p)==h for p,h in newmanifest.items()))
model=read(cfg['model_identity_receipt']);check('model_weights_unchanged',all(sha(Path(cfg['model_base'])/p)==v['sha256'] for p,v in model['weights'].items()))
ds=Path(cfg['train_data']).parent;entries=[json.loads(x) for x in (ds/'manifests/FINAL_DATA_FILE_HASHES.jsonl').read_text().splitlines()]
check('dataset_actual_shards_unchanged',all(sha(ds/x['path'])==x['sha256'] for x in entries))
identity=audited('STAGE1_CLOSURE_IDENTITY_AND_POLICY_V2.json')
state=audited('RESUME_STATE_COMPLETENESS_V2.json');check('resume_state_complete',state['MISSING_MUTABLE_STATE_N']==0 and state['STRUCTURAL_ORIGINAL_MODE_RESUME_STATE_INTEGRITY']=='PASS')
check('original_mode_resume_launch_gate',state['ORIGINAL_MODE_RESUME_STATE_INTEGRITY']=='PASS')
for name in ['R_failure','R_recovered']:
 for rank in [0,1]:
  d=read(RUN/name/f'load_completeness_rank{rank}.json');check(name+'_rank'+str(rank)+'_raw_load_exact',d['status']=='PASS' and not d['differences'] and d['expected_optimizer_states']==d['loaded_optimizer_states']==410)
baseline=audited('ORIGINAL_MODE_NONDETERMINISM_BASELINE_V2.json');check('original_numerical_baseline',baseline['status']=='PASS_NUMERICALLY_COMPARABLE' and baseline['abnormal_tensor_count']==0)
check('no_deterministic_mode',baseline['deterministic_mode_enabled'] is False)
recomputed=[];flags_exact=True
for group,atol in [('model',1e-5),('optimizer',1e-6)]:
 for t in baseline['comparisons'][group]['tensors']:
  mm=t['metrics'];base=mm['U1_U2']
  ok=all(mm[k]['old_allclose'] for k in ['U1_R','U2_R']) or all(mm[k]['max_abs']<=base['max_abs']*1.01+atol and mm[k]['rms_diff']<=base['rms_diff']*1.01+atol for k in ['U1_R','U2_R'])
  flags_exact=flags_exact and ok==t['within_predeclared_empirical_envelope_or_old_tolerance']
  if not ok:recomputed.append((group,t['key']))
check('numerical_decision_recomputed_without_relaxation',flags_exact and len(recomputed)==baseline['abnormal_tensor_count'] and recomputed==[(t['group'],t['key']) for t in baseline['abnormal_tensors']])
u1=rows('U1');u2=rows('U2');rr=rows('R_recovered')
check('trajectory_IDs_exact',ids(u1)==ids(u2)==ids(u1[:2]+rr) and len(set(ids(u1)))==10)
check('resume_progress_exact',[(x['update_step'],x['record_cursor'],x['cumulative_tokens']) for x in rr]==[(x['update_step'],x['record_cursor'],x['cumulative_tokens']) for x in u1[2:]])
for name in ['U1','U2','R_recovered','one_token_edge_fixed','odd_final_edge','one_token_resume_fixed']:
 l=(R/'logs'/('qwen3_4b_stage1_closure_v2_'+name+'.log')).read_text();check(name+'_both_shutdown',all(x in l for x in ['FINAL_BARRIER_PASS','RANK_0_SHUTDOWN_PASS','RANK_1_SHUTDOWN_PASS']))
 c=read(RUN/name/'complete.json');check(name+'_complete',c['complete'] is True)
 for row in rows(name):
  check(name+'_step'+str(row['update_step'])+'_finite',math.isfinite(row['global_mean_loss']) and row['global_valid_tokens']>0)
  check(name+'_step'+str(row['update_step'])+'_GPU_only',sorted(x['physical_gpu'] for x in row['rank_runtime'])==[2,3])
odd=audited('REAL_ODD_FINAL_GPU_EDGE_V2.json');one=audited('REAL_ONE_TOKEN_GPU_EDGE_V2.json');app=audited('STAGE1_ONE_TOKEN_APPLICABILITY_V2.json')
check('real_odd_edge',odd['REAL_ODD_FINAL_GPU_EDGE']=='PASS')
check('real_one_edge',one['REAL_ONE_TOKEN_GPU_EDGE']=='PASS')
ot=rows('odd_final_edge');oo=rows('one_token_edge_fixed')
check('odd_final_exact_IDs',ids(ot)==['prolong-v3-32k:train:038332','prolong-v3-32k:train:038333','prolong-v3-32k:train:038334'])
check('odd_final_exact_counters',len(ot)==2 and ot[-1]['record_cursor']==38335 and ot[-1]['cumulative_tokens']==76941 and ot[-1]['update_step']==2)
check('odd_final_empty_rank',ot[-1]['rank_runtime'][1]['sample_ids']==[] and ot[-1]['rank_runtime'][1]['local_tokens']==0)
check('one_exact_IDs',ids(oo)==['prolong-v3-32k:train:015220','prolong-v3-32k:train:015221'])
check('one_exact_counters',len(oo)==1 and oo[0]['cumulative_tokens']==32769 and oo[0]['global_valid_tokens']==32767 and oo[0]['record_cursor']==15222)
account=audited('FINAL_DISTRIBUTED_EXHAUSTION_ACCOUNTING_V2.json');check('formal_tail_accounting',account['FINAL_DISTRIBUTED_EXHAUSTION_ACCOUNTING']=='PASS' and 989920030+ot[-1]['cumulative_tokens']==989996971 and 38335//2+1==19168)
failure=audited('REAL_DATA_FAILURE_RECOVERY_V2.json');check('failure_recovery',failure['REAL_DATA_FAILURE_RECOVERY']=='PASS' and failure['LAST_GOOD_DCP_SURVIVED'] and failure['FALSE_COMPLETE_PREVENTED'])
check('failure_no_complete',not (RUN/'R_failure/complete.json').exists())
cp=RUN/'R_checkpoints';latest=read(cp/'latest.json');check('recovery_published_step5',latest['slot']=='slot_B' and latest['progress']['update_step']==5 and not (cp/'slot_B.partial').exists())
for slot in ['slot_A','slot_B']:
 m=read(cp/slot/'manifest.json');check(slot+'_actual_hashes',m['complete'] and all(sha(cp/slot/p)==h for p,h in m['files'].items()))
check('step2_preserved',sha(cp/'slot_A/manifest.json')==failure['last_good_manifest_sha256'])
storage=audited('CHECKPOINT_STORAGE_PREFLIGHT_V2.json');retention=audited('CHECKPOINT_RETENTION_RUNTIME_V2.json')
check('storage_gate',storage['CHECKPOINT_STORAGE_PREFLIGHT']=='PASS')
check('storage_live',min(shutil.disk_usage(R/p).free for p in ['runs','checkpoints'])>=106828846148)
check('retention_gate',retention['CHECKPOINT_RETENTION_RUNTIME']=='PASS')
check('formal_not_started',not (R/'runs/qwen3_4b_stage1_1b_32k_gpu23_formal_v1').exists() and not (R/'runs/qwen3_4b_stage1_1b_32k_gpu23_formal_v1.process.json').exists())
sparse=audited('STAGE1_SPARSE_GRAD_REPAIR_V2.json');check('sparse_grad_fix_GPU',sparse['status']=='PASS_GPU_REGRESSION')
post=audited('STAGE1_POST_SPARSE_FIX_RESUME_V2.json');check('final_candidate_resume',post['status']=='PASS' and post['final_cursor']==15224 and post['final_tokens']==65565)
policy=audited('V2_LAUNCH_GUARD_POLICY_TESTS.json');check('guard_policy_tests',policy['status']=='PASS')
result={'status':'PASS' if all(checks.values()) else 'FAIL','FORMAL_PRELAUNCH_AUDIT_V2':'PASS' if all(checks.values()) else 'FAIL','scope':'Independent read-only CPU audit of actual files, saved raw states, run ledgers, shutdown markers, DCP hashes and prior evidence. Does not launch GPU work or training.','checks':checks,'failed_checks':[k for k,v in checks.items() if not v],'evidence_sha256':evidence,'auditor_sha256':sha(__file__),'FORMAL_STAGE1_TRAINING_STARTED':False}
print(json.dumps(result,indent=2))
sys.exit(0 if all(checks.values()) else 1)
