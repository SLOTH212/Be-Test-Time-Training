import json,hashlib,math,itertools,time
from pathlib import Path
import torch,numpy as np
R=Path('/path/to/ttt');W=R/'work/qwen3_4b_stage1_resume_shared_boundary_v4';A=R/'audits/qwen3_4b_stage1_resume_shared_boundary_v4';RUN=R/'runs/qwen3_4b_stage1_resume_shared_boundary_v4'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();checks=[]
def check(name,ok):
 checks.append({'check':name,'pass':bool(ok)})
 if not ok:raise AssertionError(name)
torch.set_num_threads(8);check('CPU only',not torch.cuda.is_initialized())
p=A/'SHARED_BOUNDARY_CAUSAL_PROTOCOL_V4.json';ph=sha(p);protocol=read(p);freeze=read(A/'PROTOCOL_FREEZE_RECEIPT.json');check('Protocol content matches preregistration',freeze['sha256']==ph)
for path,h in read(A/'PRIOR_AUTHORITIES_IMMUTABLE.json')['files'].items():check('Prior immutable '+path,sha(Path(path))==h)
check('Scientific config unchanged',sha(Path(protocol['config']))==protocol['config_sha256']);check('Observer frozen',sha(W/'entry.py')==protocol['observer_sha256'])
check('V3 target set exact',read(A/'EXACT_V3_TARGETS.json')==read(Path(protocol['target_source'])))
receipts=read(A/'RUN_RECEIPTS.json');names=[x['branch'] for x in receipts];check('Branch order and maximum',names in [['A','B1','B2'],['A','B1','B2','B3']]);count=sum(x['optimizer_updates_observed'] for x in receipts);check('Exactly requested update count',count==(5 if len(names)==3 else 6))
for i,r in enumerate(receipts):
 check('Valid '+r['branch'],r['status']=='VALID' and r['returncode']==0 and r['protocol_sha256']==ph and freeze['frozen_at']<r['started_at'])
 check('GPU allowlist '+r['branch'],r['environment']['CUDA_VISIBLE_DEVICES']=='2,3' and r['environment']['WORLD_SIZE']=='2')
 check('Config command '+r['branch'],r['command'][r['command'].index('--config')+1]==protocol['config'])
 if i:check('Previous process exits before next branch '+r['branch'],receipts[i-1]['ended_at']<=r['started_at'])
 check('Observer receipt hash '+r['branch'],r['observer_sha256']==protocol['observer_sha256'])
 if i:check('Same DCP receipt '+r['branch'],r['load_identity']==read(A/'SHARED_DCP_IDENTITY.json'))
id=read(A/'SHARED_DCP_IDENTITY.json');root=Path(id['root']);latest=read(root/'latest.json');check('DCP latest immutable',sha(root/'latest.json')==id['latest_sha256']);check('DCP manifest immutable',sha(root/latest['slot']/'manifest.json')==id['manifest_sha256'])
meta={n:[read(RUN/n/f'final_rank{r}.json') for r in (0,1)] for n in names}
for rank in (0,1):
 saved=read(root/f'presave_step2_rank{rank}.json');check('Save nonmutation rank'+str(rank),saved==read(RUN/'A'/f'boundary_post_save_rank{rank}.json'))
 for n in names:
  check('Shared boundary exact '+n+str(rank),read(RUN/n/f'boundary_contract_rank{rank}.json')==saved)
  if n!='A':check('Fresh load exact '+n+str(rank),read(RUN/n/f'load_completeness_rank{rank}.json')['actual']==saved)
  pf=read(RUN/n/f'preforward_step3_rank{rank}.json');a=read(RUN/'A'/f'preforward_step3_rank{rank}.json');check('RNG and canonical records '+n+str(rank),all(pf[k]==a[k] for k in ['rng','sample_ids','cursor']))
  runtime=meta[n][rank]['runtime'];check('Original runtime '+n+str(rank),runtime['CUDA_VISIBLE_DEVICES']=='2,3' and runtime['WORLD_SIZE']==2 and runtime['deterministic'] is False)
  check('Read-only guards executed '+n+str(rank),meta[n][rank]['observer_read_checks']>=567)
# Producer reused audited V3 observation math; independently check source AST identity.
import ast
v3=ast.parse((R/'work/qwen3_4b_stage1_resume_v3/localization_final/entry.py').read_text());v4=ast.parse((W/'entry.py').read_text());defs=lambda t:{n.name:ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,ast.FunctionDef)}
a,b=defs(v3),defs(v4)
for name in ['capture_value','capture_state','tensor_fingerprint','install_forward_hooks','observed_clip','observed_foreach_reduce','observed_fused','record_clip','rng_only']:check('Read-only function inherited '+name,a[name]==b[name])
check('CPU metric examples pass',read(A/'ANALYSIS_CPU_CHECKS.json')['status']=='PASS')
# Independent torch reductions over the full targeted post-step and gradient arrays.
data={n:[torch.load(RUN/n/f'final_rank{r}.pt',mmap=True,map_location='cpu',weights_only=True) for r in (0,1)] for n in names};allpairs=read(A/'ALL_COMPARISON_METRICS.json')['pairs']
for pair,rows in allpairs.items():
 left,right=pair.split('_')
 for key,row in rows.items():
  if key not in data[left][0] or key.startswith('post_reduce.'):continue
  rep=meta[left][0]['tensors'][key]['replicated'] and not key.startswith('pre_reduce.');ranks=[0] if rep else [0,1];mx=l1=sq=aa=bb=0.;num=nz=0
  for rank in ranks:
   check('Raw schema '+pair+' '+key+str(rank),data[left][rank][key].shape==data[right][rank][key].shape and data[left][rank][key].dtype==data[right][rank][key].dtype)
   x=data[left][rank][key].reshape(-1);y=data[right][rank][key].reshape(-1)
   for off in range(0,x.numel(),1<<20):
    a=x[off:off+(1<<20)].float();b=y[off:off+(1<<20)].float();d=(a-b).abs().double();num+=d.numel();nz+=int(torch.count_nonzero(d));mx=max(mx,float(d.max()) if d.numel() else 0.);l1+=float(d.sum());sq+=float(d.square().sum());aa+=float(a.double().square().sum());bb+=float(b.double().square().sum())
  expected={'max_abs':mx,'mean_abs':l1/max(num,1),'L1':l1,'L2':math.sqrt(sq),'left_norm':math.sqrt(aa),'right_norm':math.sqrt(bb)}
  check('Full independent metrics '+pair+' '+key,row['numel']==num and row['exact']==(nz==0) and all(math.isclose(row[k],v,rel_tol=2e-12,abs_tol=1e-20) for k,v in expected.items()))
 print('INDEPENDENT_PAIR_OK',pair,flush=True)

# Independently reconstruct compact features, finite/schema invariants, and their declared floors.
compact={n:{} for n in names};bases={}
for n in names:
 tr=[json.loads(x) for x in (RUN/n/'training.jsonl').read_text().splitlines()][-1]
 compact[n]['loss']=[tr['global_mean_loss']];bases['loss']=1e-5
 for rank in (0,1):
  for event in meta[n][rank]['forward_trace']:
   for j,v in enumerate(event['outputs']):
    check('Forward finite '+n+str(rank)+event['module']+str(j),v['finite_count']==v['numel'])
    prefix=f"forward.rank{rank}.{event['module']}.call{event['call']}.out{j}"
    compact[n][prefix+'.samples']=v['samples'];bases[prefix+'.samples']=1e-5
    for k in ['norm','mean','max_abs']:
     compact[n][prefix+'.'+k]=[v[k]];bases[prefix+'.'+k]=1e-5*(math.sqrt(v['numel']) if k=='norm' else 1)
  whole=read(RUN/n/f'whole_state_light_rank{rank}.json')
  def collect(x,path):
   if isinstance(x,dict) and 'sha256' in x and 'norm' in x:
    check('Whole finite '+n+str(rank)+path,x['finite_count']==x['numel'])
    for k in ['norm','mean','max_abs']:
     key=f'whole.rank{rank}.{path}.{k}';compact[n][key]=[x[k]];bases[key]=(1e-6 if path.startswith('optimizer.') else 1e-5)*(math.sqrt(x['numel']) if k=='norm' else 1)
   elif isinstance(x,dict):
    for k,v in x.items():collect(v,path+'.'+k if path else k)
  collect({k:whole[k] for k in ['model','optimizer','buffers']},'')
 for k in ['global_norm','actual_clamped_coefficient']:
  compact[n]['clip.'+k]=meta[n][0]['clip_observation'][k]['samples'];bases['clip.'+k]=1e-6
for pair,rows in allpairs.items():
 left,right=pair.split('_');check('Same compact feature set '+pair,set(compact[left])==set(compact[right]))
 for key,arr in compact[left].items():
  x=torch.tensor(arr,dtype=torch.float64);y=torch.tensor(compact[right][key],dtype=torch.float64);d=(x-y).abs();vals={'max_abs':float(d.max()) if len(d) else 0.,'mean_abs':float(d.mean()) if len(d) else 0.,'L1':float(d.sum()),'L2':float(torch.linalg.vector_norm(d))};row=rows[key]
  check('Compact independently measured '+pair+' '+key,all(math.isclose(row[k],v,rel_tol=2e-12,abs_tol=1e-20) for k,v in vals.items()))

dec=read(A/('ABC_DECISION.json' if len(names)==3 else 'ABCD_DECISION.json'));check('Decision bound to protocol',dec['protocol_sha256']==ph)
# Check every declared rule using independent distance table reconstruction, including floors.
fresh=names[1:];ab=['A_'+n for n in fresh];bb=[x+'_'+y for x,y in itertools.combinations(fresh,2)];failed=systematic=0
for item in dec['all_observable_decisions']:
 key=item['key'];ok=True;sep={}
 n=allpairs[ab[0]][key]['numel'];base=bases.get(key,1e-5 if key.startswith('post_step.parameter.') else 1e-6)
 import statistics
 scale=max(statistics.median(allpairs['A_'+b][key]['right_norm'] for b in fresh),base*math.sqrt(n),np.finfo(float).tiny)
 for pair in allpairs:check('Independent relative L2 '+pair+' '+key,math.isclose(allpairs[pair][key]['relative_L2'],allpairs[pair][key]['L2']/scale,rel_tol=2e-12,abs_tol=1e-20))
 expected_floors={'max_abs':base,'mean_abs':base,'L1':n*base,'L2':math.sqrt(n)*base,'relative_L2':math.sqrt(n)*base/scale}
 for metric,v in item['metrics'].items():
  check('Preregistered floor '+key+' '+metric,math.isclose(v['floor'],expected_floors[metric],rel_tol=1e-15,abs_tol=0.))
  D=max(allpairs[p][key][metric] for p in bb);N=min(allpairs[p][key][metric] for p in ab);M=max(allpairs[p][key][metric] for p in ab);F=v['floor'];good=(M<=2*D+F and N<=D+F);separated=N>2*D+F
  check('Rule inputs and outcome '+key+' '+metric,v['fresh_diameter']==D and v['nearest_A']==N and v['maximum_A']==M and v['consistent']==good and v['all_A_separated']==separated);ok=ok and good;sep[metric]=separated
 check('Observable decision '+key,item['consistent']==ok and item['systematic_candidate']==(sep['max_abs'] and sep['L2']))
 failed+=not ok;systematic+=sep['max_abs'] and sep['L2']
if len(names)==4:check('B3 preregistered trigger',read(A/'ABC_DECISION.json')['run_B3'] and read(A/'ABC_DECISION.json')['failed_observable_n']>0)
expected='SHARED_BOUNDARY_RESUME_CONSISTENT' if not failed else 'PENDING_PREREGISTERED_B3' if len(names)==3 else 'PROCESS_RESTART_OR_RESUME_SPECIFIC_EFFECT' if systematic else 'INCONCLUSIVE'
check('Final classification follows frozen rule',dec['classification']==expected and dec['failed_observable_n']==failed and dec['systematic_candidate_n']==systematic)
op=read(A/'OPERATOR_AUDIT.json');check('Operators independently reconstructed',op['all_reductions_match_fp32_two_rank_average'] and op['all_clips_match_actual_recorded_coefficient'] and len(op['results'])==47*len(names))
check('All steps 2 to 3',all(x['optimizer_step_before']==[2.,2.] and x['optimizer_step_after']==[3.,3.] for x in op['results']))
coverage=read(A/'MEASUREMENT_COVERAGE_AUDIT.json');missing=set()
for item in dec['failed_observables']:
 key=item['key']
 if not key.startswith('whole.'):continue
 rank=int(key.split('.')[1][-1]);full='post_step.'+key.split('.',2)[2].rsplit('.',1)[0]
 if any(full not in data[n][rank] for n in names):missing.add(full)
check('Required non-target promotion availability independently verified',missing==set(x['full_tensor_key'] for x in coverage['missing_full_states']) and len(missing)==coverage['missing_full_state_n'])
check('Numeric candidate preserved',coverage['numeric_rule_output_preserved']==expected)
check('Missing-capture rule preregistered',protocol['decision']['missing'].startswith('Missing required captures/resource failure -> CASE_D'))
final_classification='INCONCLUSIVE' if missing else expected
check('Final evidence coverage classification',coverage['final_classification']==final_classification)

check('No CUDA initialized in auditor',not torch.cuda.is_initialized())
result={'DEBUG_ONLY':True,'status':'PASS','checks':checks,'check_count':len(checks),'classification':final_classification,'numeric_rule_classification':expected,'required_measurements_complete':not bool(missing),'missing_full_state_n':len(missing),'protocol_sha256':ph,'source_sha256':sha(Path(__file__)),'completed_at':time.time(),'scope':'Read-only independent full target metric reconstruction and causal-rule audit; PASS verifies evidence, not a population confidence assertion.'};(A/'INDEPENDENT_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print('V4_INDEPENDENT_AUDIT_PASS',len(checks),flush=True)
