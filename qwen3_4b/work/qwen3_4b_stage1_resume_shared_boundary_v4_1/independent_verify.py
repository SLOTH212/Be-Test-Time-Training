"""Read-only verification of inputs/evidence; only writes its own audit receipt."""
import ast,json,hashlib,itertools,math,statistics,time
from pathlib import Path
import numpy as np,torch
R=Path('/path/to/ttt');tag='qwen3_4b_stage1_resume_shared_boundary_v4_1';W=R/'work'/tag;A=R/'audits'/tag;RUN=R/'runs'/tag
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(16<<20),b''):h.update(block)
 return h.hexdigest()
checks=0
def check(x,label):
 global checks
 checks+=1
 if not x:raise AssertionError(label)
assert not torch.cuda.is_initialized();torch.set_num_threads(4)
P=A/'MISSING_CAPTURE_CLOSURE_PROTOCOL_V4_1.json';p=read(P);c=read(A/'COMPARISON.json');v4=read(p['prior_v4_protocol_path']);freeze=read(A/'PROTOCOL_FREEZE_RECEIPT.json')
check(sha(P)==c['protocol_sha256']==freeze['sha256'],'protocol hash')
for k in ['decision','metrics','scientific_contract','boundary']:check(p[k]==v4[k],'unchanged '+k)
for k in ['V4_DECISION_RULE_CHANGED','V4_NUMERICAL_FLOORS_CHANGED','V4_THRESHOLDS_CHANGED','V4_SHARED_BOUNDARY_DESIGN_CHANGED']:check(p[k] is False,k)
for path,h in read(A/'PRIOR_AUTHORITIES_IMMUTABLE.json')['files'].items():check(sha(path)==h,'prior immutable '+path)
for k,file in [('observer_sha256','entry.py'),('controller_sha256','run_branches.py'),('metric_source_sha256','frozen_metrics.py')]:check(sha(W/file)==p[k],k)
trees=[ast.parse((R/'work/qwen3_4b_stage1_resume_shared_boundary_v4/analyze.py').read_text()),ast.parse((W/'frozen_metrics.py').read_text())]
fn=[next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name=='metrics') for t in trees];check(ast.dump(fn[0])==ast.dump(fn[1]),'exact original metric AST')
check(read(A/'CPU_PRECHECK.json')['status']=='PASS' and read(A/'CAPTURE_PATCH_AUDIT.json')['status']=='PASS','diagnostic-only source checks')
coverage=read(R/'audits/qwen3_4b_stage1_resume_shared_boundary_v4/MEASUREMENT_COVERAGE_AUDIT.json');keys=[x['full_tensor_key'] for x in coverage['missing_full_states']]
check(keys==p['promoted_keys'] and len(keys)==3 and coverage['original_48_full_target_capture_complete'],'promoted keys exactly match missing states')
names=['A','B1','B2','B3'];receipts=read(A/'RUN_RECEIPTS.json');check(receipts==c['receipts'] and [r['branch'] for r in receipts]==names,'four receipts')
check(sum(r['optimizer_updates_observed'] for r in receipts)==6,'six updates')
for prev,nxt in zip(receipts,receipts[1:]):check(prev['ended_at']<nxt['started_at'],'sequential process groups')
check(len({r['pid'] for r in receipts})==4,'fresh launcher pids')
dcp=read(A/'SHARED_DCP_IDENTITY.json');root=Path(dcp['root']);latest=read(root/'latest.json');slot=root/latest['slot'];check(sha(root/'latest.json')==dcp['latest_sha256'] and sha(slot/'manifest.json')==dcp['manifest_sha256'],'shared DCP identity')
for name,h in dcp['files'].items():check(sha(slot/name)==h,'real DCP payload '+name)
raw={};metas={};traces={}
for n,receipt in zip(names,receipts):
 check(receipt['status']=='VALID' and receipt['returncode']==0 and receipt['started_at']>freeze['frozen_at'],'valid/frozen '+n)
 check(receipt['protocol_sha256']==sha(P) and receipt['observer_sha256']==p['observer_sha256'],'receipt binding '+n)
 allowed_controller_hashes={p['controller_sha256']}
 technical=A/'TECHNICAL_CONTROLLER_RECOVERY.json'
 if technical.exists():
  recovery=read(technical);check(recovery['interrupted_branch_updates']==0 and recovery['completed_updates']==3,'technical recovery did not rerun updates')
  check(sha(W/'resume_remaining_technical.py')==recovery['resume_controller_sha256'],'technical resume source hash')
  allowed_controller_hashes.add(recovery['resume_controller_sha256'])
 check(receipt['controller_sha256'] in allowed_controller_hashes,'controller identity including logged technical continuation')
 check(sha(receipt['log'])==receipt['log_sha256'],'log hash '+n)
 check(receipt['environment']['CUDA_VISIBLE_DEVICES']=='2,3' and receipt['environment']['WORLD_SIZE']=='2','GPU allowlist '+n)
 if n!='A':check(receipt['load_identity']==dcp,'same load DCP '+n)
 out=RUN/n;raw[n]=[];metas[n]=[]
 training=[json.loads(x) for x in (out/'training.jsonl').read_text().splitlines()];traces[n]=training[-1]
 check(len(training)==(3 if n=='A' else 1),'exact branch length')
 for r in (0,1):
  saved=read(root/f'presave_step2_rank{r}.json');check(saved==read(RUN/'A'/f'boundary_post_save_rank{r}.json')==read(out/f'boundary_contract_rank{r}.json'),'raw save/boundary exact')
  check(read(RUN/'A'/f'boundary_save_rank{r}.json')['status']=='PASS','save nonmutation')
  check(read(out/f'initial_parent_check_rank{r}.json')['status']=='PASS','initial parent')
  if n!='A':
   load=read(out/f'load_completeness_rank{r}.json');check(load['status']=='PASS' and load['actual']==saved,'exact restore')
  pre=read(out/f'preforward_step3_rank{r}.json');ref=read(RUN/'A'/f'preforward_step3_rank{r}.json')
  for k in ['rng','sample_ids','cursor']:check(pre[k]==ref[k],'preforward '+k)
  check(pre['post_load_rng_preserved'],'no construction RNG mutation')
  m=read(out/f'final_rank{r}.json');check(sha(out/f'final_rank{r}.pt')==m['sha256'],'raw shard hash');values=torch.load(out/f'final_rank{r}.pt',weights_only=True,map_location='cpu')
  check(set(values)==set(m['tensors'])==set(keys),'exact three captures')
  for k in keys:check(list(values[k].shape)==[1280] and m['tensors'][k]['global_shape']==[2560] and m['tensors'][k]['placement']=='(Shard(dim=0),)' and bool(torch.isfinite(values[k]).all()),'full valid shard')
  check(m['runtime']['CUDA_VISIBLE_DEVICES']=='2,3' and m['runtime']['WORLD_SIZE']==2 and not m['runtime']['deterministic'],'runtime policy')
  check(m['progress']['update_step']==3 and m['progress']['record_cursor']==6,'post step3')
  check(read(out/f'process_exit_rank{r}.json')['status']=='PASS','clean process exit')
  def finite(v):
   if isinstance(v,dict):
    if 'finite_count' in v:check(v['numel']==v['finite_count'],'whole compact finite')
    for x in v.values():finite(x)
   elif isinstance(v,list):
    for x in v:finite(x)
  finite(read(out/f'whole_state_light_rank{r}.json'))
  raw[n].append(values);metas[n].append(m)
 for field in ['update_step','record_cursor','cumulative_tokens','global_valid_tokens','input_identity']:check(traces[n][field]==traces['A'][field],'same training input '+field)
 check([x['sample_ids'] for x in traces[n]['rank_runtime']]==[x['sample_ids'] for x in traces['A']['rank_runtime']],'canonical input IDs')
 for x in traces[n]['rank_runtime']:check(x['physical_gpu'] in [2,3] and x['weights_finite'],'GPU/finite')
 check(metas[n][0]['clip_observation']==metas[n][1]['clip_observation'],'rank scalar parity')
 sanity=c['sanity'][n];check(sanity['loss']==traces[n]['global_mean_loss'] and sanity['pre_clip_global_norm']==traces[n]['grad_norm_before_clip'] and sanity['actual_clip_coefficient']==metas[n][0]['clip_observation']['actual_clamped_coefficient']['samples'][0],'sanity actual values')
check(len(c['full_tensor_files'])==12,'12 complete files');seen=set()
for file in c['full_tensor_files']:
 n=file['branch'];k=file['key'];check((n,k) not in seen,'unique capture');seen.add((n,k));check(sha(file['path'])==file['sha256'],'full file hash')
 tensor=torch.load(file['path'],weights_only=True,map_location='cpu');check(torch.equal(tensor,torch.cat([raw[n][r][k] for r in (0,1)])) and list(tensor.shape)==file['shape']==[2560],'complete concatenation')
# Independent arithmetic, without importing analysis or frozen_metrics.
pairwise={}
for a,b in itertools.combinations(names,2):
 rows={}
 for k in keys:
  sums={name:0. for name in ['L1','sq','aa','bb','sa','sb']};mx=0.;nz=closebad=n=0
  for rank in (0,1):
   x=raw[a][rank][k].float().numpy();y=raw[b][rank][k].float().numpy();xx=x.astype(np.float64);yy=y.astype(np.float64);d=np.abs(x-y).astype(np.float64)
   mx=max(mx,float(d.max(initial=0)));nz+=int(np.count_nonzero(d));n+=x.size;closebad+=int(np.count_nonzero(np.abs(x-y)>(1e-6+0.01*np.abs(y))))
   for name,value in [('L1',d.sum()),('sq',np.dot(d,d)),('aa',np.dot(xx,xx)),('bb',np.dot(yy,yy)),('sa',xx.sum()),('sb',yy.sum())]:sums[name]+=float(value)
  rows[k]={'numel':n,'exact':nz==0,'max_abs':mx,'mean_abs':sums['L1']/n,'L1':sums['L1'],'L2':math.sqrt(sums['sq']),'left_norm':math.sqrt(sums['aa']),'right_norm':math.sqrt(sums['bb']),'left_signed_mean':sums['sa']/n,'right_signed_mean':sums['sb']/n,'fraction_differing':nz/n,'finite':True,'legacy_allclose':closebad==0,'legacy_mismatch_count':closebad}
 pairwise[a+'_'+b]=rows
systematic_n=0
for k in keys:
 ref=max(statistics.median([pairwise['A_'+n][k]['right_norm'] for n in names[1:]]),math.sqrt(2560)*1e-6,np.finfo(np.float64).tiny)
 for pair,rows in pairwise.items():
  rows[k]['relative_L2']=rows[k]['L2']/ref;rows[k]['common_reference_norm']=ref
  check(rows[k]==c['pairwise_metrics'][pair][k],'exact independent metrics '+pair+' '+k)
 calculations={}
 for metric,floor in {'max_abs':1e-6,'mean_abs':1e-6,'L1':2560*1e-6,'L2':math.sqrt(2560)*1e-6,'relative_L2':math.sqrt(2560)*1e-6/ref}.items():
  D=max(pairwise[a+'_'+b][k][metric] for a,b in itertools.combinations(names[1:],2));N=min(pairwise['A_'+b][k][metric] for b in names[1:]);M=max(pairwise['A_'+b][k][metric] for b in names[1:])
  calc={'D':D,'N':N,'M':M,'floor':floor,'2D_plus_floor':2*D+floor,'D_plus_floor':D+floor,'max_condition_pass':M<=2*D+floor,'nearest_condition_pass':N<=D+floor,'consistent':M<=2*D+floor and N<=D+floor,'systematic_metric':N>2*D+floor};check(calc==c['promoted_results'][k]['calculations'][metric],'frozen decision '+metric);calculations[metric]=calc
 separated=calculations['max_abs']['systematic_metric'] and calculations['L2']['systematic_metric'];systematic_n+=int(separated)
 result='SYSTEMATIC_A_VS_B_SEPARATION' if separated else 'WITHIN_FRESH_LOAD_VARIABILITY' if all(x['consistent'] for x in calculations.values()) else 'MIXED_NOT_CONFIRMED'
 check(result==c['promoted_results'][k]['promotion_result'],'full state classification')
check(c['systematic_promoted_state_n']==systematic_n,'systematic count');classification='PROCESS_RESTART_OR_RESUME_SPECIFIC_EFFECT' if systematic_n else 'SHARED_BOUNDARY_RESUME_CONSISTENT';check(c['classification']==classification,'user closure decision')
formal=R/'runs/qwen3_4b_stage1_1b_32k_gpu23_formal_v1';check(not formal.exists() and not formal.with_suffix('.process.json').exists(),'no formal output or process receipt')
result={'status':'PASS','checked_at':time.time(),'checks':checks,'classification':classification,'protocol_sha256':sha(P),'comparison_sha256':sha(A/'COMPARISON.json'),'auditor_sha256':sha(Path(__file__)),'read_only_evidence':True,'cuda_initialized':torch.cuda.is_initialized(),'full_tensor_files_verified':12,'pairwise_full_states_recomputed':18,'DCP_payload_hashes_verified':True,'all_prior_files_immutable':True,'formal_training_started':False,'scope':'Independently recomputed all three complete tensors and frozen distances/calculations; prior 48-vector evidence hash-verified and reused. PASS here validates evidence, not automatic causal PASS.'}
(A/'INDEPENDENT_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
