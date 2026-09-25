"""Independent read-only recomputation from pair metrics, weighted order statistics.

Does not import v3_statistics or trust its confidence intervals / decisions.
"""
import json, itertools, math, hashlib, time
from pathlib import Path
import numpy as np
R=Path('/path/to/ttt');A=R/'audits/v3';D=A/'analysis'
read=lambda p:json.loads(Path(p).read_text())
M=['max_abs','mean_abs','L1','L2','relative_L2']
checks=[]
def ck(label,truth):
 checks.append({'check':label,'pass':bool(truth)})
 if not truth:raise AssertionError(label)
def close(a,b):return np.allclose(a,b,rtol=2e-11,atol=1e-16,equal_nan=False)
def main():
 protocol=read(A/'RESUME_NUMERICAL_EQUIVALENCE_PROTOCOL_V3.json');inp=read(A/'REPLICATE_INPUT_PARITY.json');report=read(D/'ALL_TENSOR_INFERENCE.json');bykey={x['key']:x for x in report};design=read(D/'RESAMPLING_DESIGN.json');summary=read(A/'STATISTICAL_RESULT_V3.json');names=inp['names'];N=len(names);nu=sum(n[0]=='U' for n in names);nr=N-nu
 ck('input audit PASS',inp['status']=='PASS' and all(x['pass'] for x in inp['checks']))
 protocol_hash=hashlib.sha256((A/'RESUME_NUMERICAL_EQUIVALENCE_PROTOCOL_V3.json').read_bytes()).hexdigest()
 ck('original protocol exact frozen SHA',protocol_hash=='174696cef1781e00c6e503155e7b1a4caf6de2f5d2c5fd04b30e4b9ac43a7983' and protocol_hash==(A/'RESUME_NUMERICAL_EQUIVALENCE_PROTOCOL_V3.sha256').read_text().split()[0])
 for path,digest in protocol['input_file_sha256'].items():ck('immutable '+path,hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest)
 amendment=read(A/'OBSERVER_FIX1_AMENDMENT.json')
 for path,digest in amendment['new_file_sha256'].items():ck('observer correction immutable '+path,hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest)
 receipts=read(A/'execution2/REPLICATE_CONTROLLER_RECEIPTS.json')
 ck('preregistered before all GPU starts',all(x['started_at']>(A/'RESUME_NUMERICAL_EQUIVALENCE_PROTOCOL_V3.json').stat().st_mtime and x['started_at']>amendment['created_at'] for x in receipts))
 ck('minimum valid counts',nu>=4 and nr>=4)
 ck('no formal workload in receipts',all('--max-records' in x['command'] and x['command'][x['command'].index('--max-records')+1] in ['4','6','10'] and x['environment']['CUDA_VISIBLE_DEVICES']=='2,3' for x in receipts))
 original_inputs={};original_runtime=None;original_scheduler=None;original_groups=None
 for name in names:
  for role in (['U'] if name[0]=='U' else ['R_PRE','R_POST']):
   receipt=next(x for x in receipts if x['replica']==name and x['role']==role);p=Path(receipt['output']);ck('actual valid receipt '+name+role,receipt['status']=='VALID_COMPLETED_PHASE' and receipt['protocol_sha256']==protocol_hash)
   for rank in (0,1):
    ck('actual raw initial equality '+name+role+str(rank),read(p/f'initial_rank{rank}.json')==read(R/f'runs/qwen3_4b_stage1_closure_v2/U1/initial_rank{rank}.json'))
    rt=read(p/f'process_start_rank{rank}.json')['runtime'];rt={k:v for k,v in rt.items() if k not in ['pid','role','replica','rank']}
    if original_runtime is None:original_runtime=rt
    ck('actual original-mode runtime '+name+role+str(rank),rt==original_runtime and rt['CUDA_VISIBLE_DEVICES']=='2,3' and rt['WORLD_SIZE']==2 and not rt['deterministic'])
    ck('actual clean exit '+name+role+str(rank),read(p/f'process_exit_rank{rank}.json')['status']=='PASS')
    if role=='R_POST':
     loaded=read(p/f'load_completeness_rank{rank}.json');pre=p.parent/(name+'_pre');ck('actual raw DCP load '+name+str(rank),loaded['status']=='PASS' and loaded['actual']==read(pre/f'checkpoints/presave_step2_rank{rank}.json'))
     ck('fresh independent reconstruction '+name+str(rank),read(pre/f'process_exit_rank{rank}.json')['time']<read(p/f'process_start_rank{rank}.json')['time'] and read(pre/f'process_start_rank{rank}.json')['runtime']['pid']!=read(p/f'process_start_rank{rank}.json')['runtime']['pid'])
   for line in map(json.loads,(p/'training.jsonl').read_text().splitlines()):
    step=line['update_step'];contract={k:line[k] for k in ['record_cursor','cumulative_tokens','dataloader_state','input_identity','learning_rate','accumulation_phase']}
    if step not in original_inputs:original_inputs[step]=contract
    ck('actual same step input contract '+name+str(step),contract==original_inputs[step])
    for rank in (0,1):
     f=read(p/f'preforward_step{step}_rank{rank}.json');ck('canonical sample IDs '+name+str(step)+str(rank),f['sample_ids']==[f'prolong-v3-32k:train:{i:06d}' for i in range((step-1)*2,step*2)] and f['post_load_rng_preserved'])
   if role!='R_PRE':
    f=read(p/'final_rank0.json')
    if original_scheduler is None:original_scheduler=f['scheduler'];original_groups=f['param_groups']
    ck('actual final state contract '+name,f['status']=='PASS' and f['scheduler']==original_scheduler and f['param_groups']==original_groups and f['progress']['update_step']==5 and f['progress']['record_cursor']==10 and f['progress']['cumulative_tokens']==286544)
 pairs=list(itertools.combinations(range(N),2));K=len(pairs);lookup={p:k for k,p in enumerate(pairs)}
 def weights(us,rs):
  pools=[list(itertools.combinations(us,2)),list(itertools.product(us,rs)),list(itertools.combinations(rs,2))]
  return [np.bincount([lookup[tuple(sorted((int(i),int(j))))] for i,j in pool if i!=j],minlength=K) for pool in pools]
 rng=np.random.default_rng(20260911);draws=[];rejected=0
 while len(draws)<4096:
  us=rng.choice(np.arange(nu),nu,replace=True);rs=rng.choice(np.arange(nu,N),nr,replace=True)
  if len(set(us))<2 or len(set(rs))<2:rejected+=1;continue
  draws.append([us.tolist(),rs.tolist()])
 ck('bootstrap identities and rejection count',draws==design['bootstrap_draws'] and rejected==design['rejected'])
 bw=np.array([weights(*draw) for draw in draws]).transpose(1,0,2)
 assignments=[[list(us),[i for i in range(N) if i not in us]] for us in itertools.combinations(range(N),nu)]
 ck('full label permutation enumeration',assignments==design['permutation_assignments'])
 pw=np.array([weights(*draw) for draw in assignments]).transpose(1,0,2)
 ow=np.array(weights(range(nu),range(nu,N)))[:,None,:]
 def wq(values,w,q):
  order=np.argsort(values,kind='stable');v=values[order];counts=w[...,order];cum=counts.cumsum(axis=-1);pos=(counts.sum(axis=-1)-1)*q;lo=np.floor(pos);hi=np.ceil(pos)
  li=np.argmax(cum>lo[...,None],axis=-1);ri=np.argmax(cum>hi[...,None],axis=-1)
  return v[li]+(v[ri]-v[li])*(pos-lo)
 rows=[json.loads(x) for x in (D/'ALL_TENSOR_PAIR_METRICS.jsonl').read_text().splitlines()]
 ck('all417+1230 tensor items',len(rows)==1647 and sum(x['group']=='model' for x in rows)==417)
 target={x['group']+x['canonical_state_name'] for x in read(A/'V2_FAILED_76_TENSOR_INVENTORY.json')['rows']}
 v2=read(R/'audits/ORIGINAL_MODE_NONDETERMINISM_BASELINE_V2.json')
 ck('76 names directly from preserved V2 artifact',target=={x['group']+x['key'] for x in v2['abnormal_tensors']} and v2['ORIGINAL_MODE_RESUME_NUMERICAL_PARITY']=='FAIL')
 ck('exact76 original identities',target=={x['key'] for x in rows if x['primary76']} and len(target)==76)
 for step in (3,4,5):
  vs=np.array([next(x['loss'] for x in inp['trajectories'][n] if x['step']==step) for n in names]);scale=max(float(np.median(np.abs(vs[:nu]))),1e-5)
  rows.append({'key':f'loss.step{step}','group':'loss','primary76':False,'numel':1,'atol':1e-5,'U_relative_scale':scale,'run_stats':{n:{'signed_mean':float(v)} for n,v in zip(names,vs)},'pairs':[{m:float(abs(vs[i]-vs[j])/(scale if m=='relative_L2' else 1)) for m in M} for i,j in pairs]})
 maxdisp=np.zeros(len(assignments));maxdir=np.zeros(len(assignments));audit_rows=[];pbm=[];pbmean=[]
 for row in rows:
  rep=bykey[row['key']];n=row['numel'];a=row['atol'];floors=[a,a,n*a,math.sqrt(n)*a,math.sqrt(n)*a/row['U_relative_scale']];values=np.array([[p[m] for p in row['pairs']] for m in M]);bsmed=[];bsmean=[];flags=[];disp=[]
  if row['group']!='loss':
   ck(row['key']+' exact and legacy allclose counts',rep['exact_all_pairs']==all(p['exact'] for p in row['pairs']) and rep['legacy_allclose_all_pairs']==all(p['old_allclose'] for p in row['pairs']))
   if 'U1' in names and 'U2' in names:
    ai=lookup[(names.index('U1'),names.index('U2'))];base=row['pairs'][ai];old={}
    for rn in names[nu:]:
     ps=[row['pairs'][lookup[tuple(sorted((names.index(un),names.index(rn))))]] for un in ['U1','U2']]
     old[rn]=all(p['old_allclose'] for p in ps) or all(p['max_abs']<=base['max_abs']*1.01+a and p['legacy_rms_float32_square']<=base['legacy_rms_float32_square']*1.01+a for p in ps)
    ck(row['key']+' exact old V2 two-anchor rule',old==rep['V2_two_anchor_rule'])
  for k,m in enumerate(M):
   v=values[k];f=floors[k];med=wq(v,ow,.5)[:,0];p95=wq(v,ow,.95)[:,0];mx=wq(v,ow,1)[:,0]
   bm=wq(v,bw,.5);b95=wq(v,bw,.95)[0];delta=bm[1:]-b95[None,:]-f;ci=np.quantile(delta,[.05,.95],axis=-1,method='linear');point=med[1:]-p95[0]-f
   ck(row['key']+' '+m+' bootstrap CI recomputed',close(ci[:,0],rep['metrics'][m]['RU_delta_CI90']) and close(ci[:,1],rep['metrics'][m]['RR_delta_CI90']))
   ck(row['key']+' '+m+' distributions recomputed',all(close([med[j],p95[j],mx[j]],list(rep['metrics'][m][c].values())) for j,c in enumerate(['UU','RU','RR'])))
   nearest=[min(v[lookup[(i,j)]] for i in range(nu)) for j in range(nu,N)];ext=mx[1]<=2*mx[0]+f and mx[2]<=2*mx[0]+f and max(nearest)<=mx[0]+f;ni=bool(np.all(ci[1]<=0))
   ck(row['key']+' '+m+' gates recomputed',ext==rep['metrics'][m]['extreme_pass'] and ni==rep['metrics'][m]['noninferiority_pass'])
   pm=wq(v,pw,.5);pbase=wq(v,pw,.95)[0];den=np.quantile(v,.95)+f;stat=np.maximum(pm[1:]-pbase[None,:]-f,0)/den;maxdisp=np.maximum(maxdisp,stat.max(axis=0));disp.append(np.maximum(point,0)/den)
   flags.append((ni,ext));bsmed.append(bm/(p95[0]+f));bsmean.append((bw@v/bw.sum(axis=-1))/(p95[0]+f))
  if row['primary76']:pbm.append(np.array(bsmed).transpose(1,0,2));pbmean.append(np.array(bsmean).transpose(1,0,2))
  # Centered Gram representation reconstructed from pairwise squared distances.
  dsq=np.zeros((N,N))
  for z,(i,j) in enumerate(pairs):dsq[i,j]=dsq[j,i]=values[3,z]**2
  H=np.eye(N)-np.ones((N,N))/N;G=-.5*H@dsq@H
  means=np.array([row['run_stats'][n]['signed_mean'] for n in names]);vden=np.sqrt(np.mean(values[3]**2))+floors[3];sden=np.std(means,ddof=1)+a
  dst=[];amplitudes=[]
  for us,rs in assignments:
   c=np.zeros(N);c[us]=-1/nu;c[rs]=1/nr;gram_squared=float(c@G@c)
   direct_squared=float(dsq[np.ix_(us,rs)].mean()-.5*dsq[np.ix_(us,us)].mean()-.5*dsq[np.ix_(rs,rs)].mean())
   # Verify the independent Gram identity at squared-distance precision.
   # Taking sqrt first amplifies cancellation near a truly zero centroid.
   rounding_bound=100*np.finfo(float).eps*max(float(np.max(dsq)),np.finfo(float).tiny)
   assert abs(gram_squared-direct_squared)<=rounding_bound
   # Recompute the documented pair-distance formula for the exact float64
   # permutation statistic; no change to producer statistics or decision gates.
   amps=[math.sqrt(max(0,direct_squared)),abs(float(means[rs].mean()-means[us].mean()))];dst.append([amps[0]/vden,amps[1]/sden]);amplitudes.append(amps)
  dst=np.array(dst);maxdir=np.maximum(maxdir,dst.max(axis=1));obs=assignments.index([list(range(nu)),list(range(nu,N))]);ck(row['key']+' directional Gram statistic',close(dst[obs],rep['direction']['observed_statistics']))
  audit_rows.append((row,flags,np.array(disp).T,dst[obs],amplitudes[obs]))
  if len(audit_rows)%50==0:print('INDEPENDENT_RECOMPUTE',len(audit_rows),flush=True)
 ck('whole-state maxT dispersion',close(maxdisp,design['dispersion_maxT']));ck('whole-state maxT direction',close(maxdir,design['drift_maxT']))
 def adj(v,mx):return float(np.mean(mx>=v-100*np.finfo(float).eps*max(1,abs(v))))
 for row,flags,disp,ds,amps in audit_rows:
  rep=bykey[row['key']];dp=np.array([[adj(v,maxdisp) for v in xs] for xs in disp]);dr=[adj(v,maxdir) for v in ds];ck(row['key']+' adjusted p-values',close(dp,rep['dispersion_maxT_adjusted_p']) and close(dr,rep['direction']['maxT_adjusted_p']))
  floors=[math.sqrt(row['numel'])*row['atol'],row['atol']];drift=any(p<=.05 and amp>f for p,amp,f in zip(dr,amps,floors));below=any(p<=.05 and 0<amp<=f for p,amp,f in zip(dr,amps,floors));abnormal=drift or bool(np.any((disp>0)&(dp<=.05)));passes=all(a and b for a,b in flags) and not drift and not below;trigger=abnormal or any(not b for _,b in flags)
  ck(row['key']+' classification',abnormal==rep['confirmed_abnormal'] and passes==rep['empirical_equivalence_pass'] and trigger==rep['localization_trigger'])
 distribution=read(A/'DISTRIBUTION_LEVEL_NONINFERIORITY.json')
 for raw,key,reducer in [(pbm,'target_median_normalized',np.median),(pbmean,'pooled_mean_normalized',np.mean)]:
  x=reducer(np.array(raw),axis=0)
  for j,m in enumerate(M):
   for k,c in [(1,'RU'),(2,'RR')]:
    delta=x[k,j]-x[0,j];ratio=np.divide(x[k,j],x[0,j],out=np.ones_like(x[k,j]),where=x[0,j]!=0)
    ck(key+m+c+' fixed76 distribution CIs',close(np.quantile(delta,[.05,.95]),distribution[key][m][c]['difference_CI90']) and close(np.quantile(ratio,[.05,.95]),distribution[key][m][c]['ratio_CI90']))
 payload={'status':'PASS','RESUME_EQUIVALENCE_INDEPENDENT_AUDIT':'PASS','method':'Separate weighted-order-statistic bootstrap and centered-Gram directional recomputation; all whole-state tensor metrics, all4096 whole-replicate draws, all full label permutations, all maxT adjusted p-values, fixed76 distribution CIs and per-tensor gates recomputed without importing producer code. Exact input evidence, immutable hashes and pre-registration checked. This audits numerical evidence; conditional localization and final authority remain separately required.','checks':checks,'check_count':len(checks),'statistical_result_sha256':hashlib.sha256((A/'STATISTICAL_RESULT_V3.json').read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'completed_at':time.time()}
 (A/'RESUME_EQUIVALENCE_INDEPENDENT_AUDIT.json').write_text(json.dumps(payload,indent=2)+'\n');print('INDEPENDENT_AUDIT_PASS',len(checks),flush=True)
if __name__=='__main__':main()
