"""Frozen-protocol replicate-unit inference. No GPU, no training mutation."""
import json, math, itertools, hashlib, time
from pathlib import Path
import numpy as np
R=Path('/path/to/ttt');A=R/'audits/v3';OUT=A/'analysis'
METRICS=['max_abs','mean_abs','L1','L2','relative_L2']

def q(a,p,axis=-1):return np.quantile(a,p,axis=axis,method='linear')
def nq(a,p):
 # Quantile over last axis, variable within-class multiplicities, NaNs last.
 s=np.sort(a,axis=-1);n=np.sum(~np.isnan(s),axis=-1);assert np.all(n>0)
 pos=(n-1)*p;lo=np.floor(pos).astype(int);hi=np.ceil(pos).astype(int)
 x=np.take_along_axis(s,lo[...,None],-1)[...,0];y=np.take_along_axis(s,hi[...,None],-1)[...,0]
 return x+(y-x)*(pos-lo)

def design(names):
 u=np.array([i for i,n in enumerate(names) if n[0]=='U']);r=np.array([i for i,n in enumerate(names) if n[0]=='R']);n=len(names)
 pairs=list(itertools.combinations(range(n),2));pi={tuple(sorted(x)):i for i,x in enumerate(pairs)}
 def indices(us,rs):
  get=lambda a,b:pi[tuple(sorted((int(a),int(b))))] if a!=b else len(pairs)
  return ([get(a,b) for a,b in itertools.combinations(us,2)],[get(a,b) for a in us for b in rs],[get(a,b) for a,b in itertools.combinations(rs,2)])
 rng=np.random.default_rng(20260911);draws=[];reject=0
 while len(draws)<4096:
  us=rng.choice(u,len(u),replace=True);rs=rng.choice(r,len(r),replace=True)
  if len(set(us))<2 or len(set(rs))<2:reject+=1;continue
  draws.append((us.tolist(),rs.tolist()))
 bs=[np.array([indices(*draw)[k] for draw in draws]) for k in range(3)]
 assignments=[(list(us),[i for i in range(n) if i not in us]) for us in itertools.combinations(range(n),len(u))]
 perms=[np.array([indices(*draw)[k] for draw in assignments]) for k in range(3)]
 return u,r,pairs,indices(u,r),draws,reject,bs,assignments,perms

def main():
 inp=json.loads((A/'REPLICATE_INPUT_PARITY.json').read_text());assert inp['status']=='PASS';names=inp['names']
 rows=[json.loads(s) for s in (OUT/'ALL_TENSOR_PAIR_METRICS.jsonl').read_text().splitlines()]
 manifest=json.loads((OUT/'PAIR_METRIC_MANIFEST.json').read_text());assert names==manifest['names']
 u,r,pairs,observed,draws,reject,bs,assignments,perms=design(names);B=len(draws);P=len(assignments);N=len(names)
 for step in (3,4,5):
  values=[next(x['loss'] for x in inp['trajectories'][name] if x['step']==step) for name in names]
  scale=max(float(np.median(np.abs(np.array(values)[u]))),1e-5)
  rows.append({'key':f'loss.step{step}','group':'loss','primary76':False,'shape':[],'numel':1,'dtype':'float64','atol':1e-5,'U_relative_scale':scale,'run_stats':{n:{'signed_mean':v,'norm':abs(v)} for n,v in zip(names,values)},'pairs':[dict(left=names[i],right=names[j],exact=values[i]==values[j],old_allclose=abs(values[i]-values[j])<=1e-5+.01*abs(values[j]),**{m:abs(values[i]-values[j])/(scale if m=='relative_L2' else 1) for m in METRICS}) for i,j in pairs]})
 outputs=[];global_drift=np.zeros(P);global_disp=np.zeros(P);primary_boot=[];primary_points=[]
 for row in rows:
  n=row['numel'];atol=row['atol'];floor=np.array([atol,n*0+atol,n*atol,math.sqrt(n)*atol,math.sqrt(n)*atol/row['U_relative_scale']])
  d=np.array([[x[m] for x in row['pairs']] for m in METRICS]);assert d.shape==(5,len(pairs)) and np.isfinite(d).all()
  padded=np.concatenate([d,np.full((5,1),np.nan)],axis=1)
  pools=[d[:,idx] for idx in observed];uu,ru,rr=pools
  baseline95=q(uu,.95);medians=np.stack([q(x,.5) for x in pools]);points=np.stack([medians[1]-baseline95-floor,medians[2]-baseline95-floor])
  bootstrap=[padded[:,ix] for ix in bs]
  bm=[nq(x,.5) for x in bootstrap];b95=nq(bootstrap[0],.95)
  deltas=np.stack([bm[1]-b95-floor[:,None],bm[2]-b95-floor[:,None]])
  cis=np.quantile(deltas,[.05,.95],axis=-1,method='linear') # bound, contrast, metric
  norm=baseline95+floor
  if row['primary76']:
   primary_boot.append({'medians':np.stack(bm)/norm[None,:,None], 'means':np.stack([np.nanmean(x,axis=-1) for x in bootstrap])/norm[None,:,None]})
   primary_points.append({'medians':medians/norm[None,:],'means':np.stack([x.mean(axis=-1) for x in pools])/norm[None,:]})
  maximum=np.stack([x.max(axis=-1) for x in pools]);nearest=np.array([[min(d[k,pairs.index(tuple(sorted((int(i),int(j)))))] for j in u) for i in r] for k in range(5)])
  extreme=(maximum[1]<=2*maximum[0]+floor)&(maximum[2]<=2*maximum[0]+floor)&np.all(nearest<=maximum[0,:,None]+floor[:,None],axis=-1)
  pp=[d[:,ix] for ix in perms];p95=q(pp[0],.95);pdeltas=np.stack([q(pp[1],.5)-p95-floor[:,None],q(pp[2],.5)-p95-floor[:,None]])
  denom=q(d,.95)+floor
  pstats=np.maximum(pdeltas,0)/denom[None,:,None];global_disp=np.maximum(global_disp,pstats.max(axis=(0,1)))
  sq=np.zeros((N,N));
  for k,(i,j) in enumerate(pairs):sq[i,j]=sq[j,i]=d[3,k]**2
  signed=np.array([row['run_stats'][name]['signed_mean'] for name in names]);pooled_vector=math.sqrt(float(np.mean(d[3]**2)))+floor[3];pooled_scalar=float(np.std(signed,ddof=1))+atol
  def direction(us,rs):
   us=np.array(us);rs=np.array(rs)
   cent2=float(sq[np.ix_(us,rs)].mean()-.5*sq[np.ix_(us,us)].mean()-.5*sq[np.ix_(rs,rs)].mean());cent=math.sqrt(max(0,cent2));scalar=abs(float(signed[rs].mean()-signed[us].mean()))
   return [cent/pooled_vector,scalar/pooled_scalar],[cent,scalar]
  permdir=np.array([direction(us,rs)[0] for us,rs in assignments]);global_drift=np.maximum(global_drift,permdir.max(axis=1));dstats,amplitudes=direction(u,r)
  dc=sq[np.ix_(r,u)].mean(axis=1)-.5*sq[np.ix_(u,u)].mean();cos=[]
  for i,j in itertools.combinations(range(len(r)),2):
   denomcos=math.sqrt(max(0,float(dc[i]*dc[j])))
   cos.append({'left':names[r[i]],'right':names[r[j]],'cosine':float(np.clip((dc[i]+dc[j]-sq[r[i],r[j]])/(2*denomcos),-1,1)) if denomcos>0 else None})
  signed_deltas=signed[r]-signed[u].mean()
  legacy_anchor=None
  if 'U1' in names and 'U2' in names:
   i=names.index('U1');j=names.index('U2');anchor_index=pairs.index((i,j));anchor=d[:,anchor_index];legacy_anchor={}
   legacy_rms=lambda z:row['pairs'][z].get('legacy_rms_float32_square',d[3,z]/math.sqrt(n))
   for k in r:
    ids=[pairs.index(tuple(sorted((a,int(k))))) for a in (i,j)];legacy_anchor[names[k]]=all(row['pairs'][z]['old_allclose'] for z in ids) or all(d[0,z]<=anchor[0]*1.01+atol and legacy_rms(z)<=legacy_rms(anchor_index)*1.01+atol for z in ids)
  out={'key':row['key'],'group':row['group'],'primary76':row['primary76'],'numel':n,'exact_all_pairs':bool(np.all(d==0)),'legacy_allclose_all_pairs':all(x['old_allclose'] for x in row['pairs']),'legacy_pair_pass_counts':{c:sum(row['pairs'][idx]['old_allclose'] for idx in ix) for c,ix in zip(['UU','RU','RR'],observed)},'V2_two_anchor_rule':legacy_anchor,'metrics':{},'direction':{'observed_statistics':dstats,'amplitudes':amplitudes,'floors':[float(floor[3]),atol],'signed_mean_R_minus_U_centroid':dict(zip([names[i] for i in r],map(float,signed_deltas))),'same_nonzero_sign':bool(np.all(signed_deltas>0) or np.all(signed_deltas<0)),'delta_cosines':cos},'dispersion_observed_statistics':(np.maximum(points,0)/denom[None,:]).tolist()}
  for k,m in enumerate(METRICS):
   out['metrics'][m]={'floor':float(floor[k]),'UU':dict(zip(['median','p95','max'],map(float,[medians[0,k],baseline95[k],maximum[0,k]]))),'RU':dict(zip(['median','p95','max'],map(float,[medians[1,k],q(ru[k],.95),maximum[1,k]]))),'RR':dict(zip(['median','p95','max'],map(float,[medians[2,k],q(rr[k],.95),maximum[2,k]]))),'RU_delta':float(points[0,k]),'RR_delta':float(points[1,k]),'RU_delta_CI90':cis[:,0,k].tolist(),'RR_delta_CI90':cis[:,1,k].tolist(),'noninferiority_pass':bool(np.all(cis[1,:,k]<=0)),'extreme_pass':bool(extreme[k]),'nearest_U_per_R':nearest[k].tolist()}
  outputs.append(out)
  if len(outputs)%25==0:print('INFERENCE',len(outputs),len(rows),flush=True)
 def adj(v,maximum):return float(np.mean(maximum>=v-100*np.finfo(float).eps*max(1,abs(v))))
 for out in outputs:
  di=out['direction'];di['maxT_adjusted_p']=[adj(v,global_drift) for v in di['observed_statistics']];di['significant_above_floor']=any(p<=.05 and amp>fl for p,amp,fl in zip(di['maxT_adjusted_p'],di['amplitudes'],di['floors']));di['significant_below_floor']=any(p<=.05 and 0<amp<=fl for p,amp,fl in zip(di['maxT_adjusted_p'],di['amplitudes'],di['floors']))
  out['dispersion_maxT_adjusted_p']=[[adj(v,global_disp) for v in contrast] for contrast in out['dispersion_observed_statistics']]
  out['confirmed_abnormal']=di['significant_above_floor'] or any(out['metrics'][m][c+'_delta']>0 and out['dispersion_maxT_adjusted_p'][i][j]<=.05 for i,c in enumerate(['RU','RR']) for j,m in enumerate(METRICS))
  out['empirical_equivalence_pass']=all(x['noninferiority_pass'] and x['extreme_pass'] for x in out['metrics'].values()) and not di['significant_above_floor'] and not di['significant_below_floor']
  out['localization_trigger']=out['confirmed_abnormal'] or any(not x['extreme_pass'] for x in out['metrics'].values())
 distribution={}
 for agg in ['medians','means']:
  arr=np.stack([x[agg] for x in primary_boot]);point=np.stack([x[agg] for x in primary_points]);assert len(arr)==76
  reducer=np.median if agg=='medians' else np.mean
  b=reducer(arr,axis=0);p=reducer(point,axis=0)
  report={}
  for j,m in enumerate(METRICS):
   report[m]={}
   for i,c in [(1,'RU'),(2,'RR')]:
    delta=b[i,j]-b[0,j];ratio=np.divide(b[i,j],b[0,j],out=np.ones_like(b[i,j]),where=b[0,j]!=0)
    report[m][c]={'normalized_UU':float(p[0,j]),'normalized_other':float(p[i,j]),'difference':float(p[i,j]-p[0,j]),'difference_CI90':q(delta,[.05,.95]).tolist(),'ratio':float(p[i,j]/p[0,j]) if p[0,j] else None,'ratio_CI90':q(ratio,[.05,.95]).tolist(),'zero_denominator_bootstrap_draws':int(np.count_nonzero(b[0,j]==0))}
  distribution['target_median_normalized' if agg=='medians' else 'pooled_mean_normalized']=report
 any_abnormal=any(x['confirmed_abnormal'] for x in outputs);any_drift=any(x['direction']['significant_above_floor'] for x in outputs);allpass=all(x['empirical_equivalence_pass'] for x in outputs)
 summary={'statistical_case':'CASE_C' if any_abnormal else 'CASE_B_CANDIDATE' if allpass else 'CASE_D','NUMERICAL_EQUIVALENCE_RESULT':'FAIL' if any_abnormal else 'PASS_PENDING_INDEPENDENT_AUDIT' if allpass else 'INCONCLUSIVE','RESUME_DIRECTIONAL_DRIFT_STATUS':'PRESENT' if any_drift else 'ABSENT' if allpass else 'INCONCLUSIVE','localization_required':[x['key'] for x in outputs if x['localization_trigger']],'confirmed_abnormal':[x['key'] for x in outputs if x['confirmed_abnormal']],'significant_below_floor':[x['key'] for x in outputs if x['direction']['significant_below_floor']],'primary76_pass_count':sum(x['primary76'] and x['empirical_equivalence_pass'] for x in outputs),'counts':{g:{'total':sum(x['group']==g for x in outputs),'exact':sum(x['group']==g and x['exact_all_pairs'] for x in outputs),'legacy_allclose':sum(x['group']==g and x['legacy_allclose_all_pairs'] for x in outputs),'empirical':sum(x['group']==g and x['empirical_equivalence_pass'] for x in outputs)} for g in ['model','optimizer','loss']},'names':names,'bootstrap_draws':B,'bootstrap_rejected_draws':reject,'permutation_assignments':P,'legacy_V2_result':'FAIL','FORMAL_STAGE1_TRAINING_STARTED':False,'SAFE_TO_LAUNCH':'NO_PENDING_CLOSURE','completed_at':time.time()}
 (OUT/'ALL_TENSOR_INFERENCE.json').write_text(json.dumps(outputs,indent=2)+'\n');(A/'PRIMARY_76_EQUIVALENCE_TABLE.json').write_text(json.dumps([x for x in outputs if x['primary76']],indent=2)+'\n');(A/'DISTRIBUTION_LEVEL_NONINFERIORITY.json').write_text(json.dumps(distribution,indent=2)+'\n');(A/'STATISTICAL_RESULT_V3.json').write_text(json.dumps(summary,indent=2)+'\n');(OUT/'RESAMPLING_DESIGN.json').write_text(json.dumps({'names':names,'bootstrap_draws':draws,'rejected':reject,'permutation_assignments':assignments,'drift_maxT':global_drift.tolist(),'dispersion_maxT':global_disp.tolist()},indent=2)+'\n')
 print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
