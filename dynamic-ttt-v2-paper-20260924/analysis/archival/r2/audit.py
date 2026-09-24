"""Offline CPU audit. No torch/model/runtime imports; source inputs opened read-only."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='none',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import json,csv,hashlib,math,sys,itertools
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
OUT=Path(__file__).resolve().parent
A=['OFF','L0','L6','L12','L18','L24','ALL']; EPS=1e-12; SEED=20260923
provenance={}; checks=Counter()
def read(p):
 p=Path(p); b=p.read_bytes();provenance[str(p)]={'sha256':hashlib.sha256(b).hexdigest(),'size':len(b)};return b
def j(p):return json.loads(read(p))
def jl(p):return [json.loads(x) for x in read(p).splitlines() if x.strip()]
def check(k,v):
 if not v: raise AssertionError(k)
 checks[k]+=1
def canon(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def save(name,x): (OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvout(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def mean(x):return float(np.mean(x)) if len(x) else None
def desc(x):
 return {'n':len(x),'mean':mean(x),'q25':float(np.quantile(x,.25)),'median':float(np.median(x)),'q75':float(np.quantile(x,.75))} if len(x) else {'n':0}
def modes(s):
 c=Counter(s);return {a for a,n in c.items() if n==max(c.values())}
def segments(ts):
 out=[]
 for t in sorted(ts):
  if out and t==out[-1][-1]+1:out[-1].append(t)
  else:out.append([t])
 return out
def ranks(x):
 x=np.asarray(x);ix=np.argsort(x,kind='stable');r=np.empty(len(x));k=0
 while k<len(x):
  e=k+1
  while e<len(x) and x[ix[e]]==x[ix[k]]:e+=1
  r[ix[k:e]]=(k+e-1)/2;k=e
 return r
def rho(x,y):
 x=ranks(x);y=ranks(y)
 return float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else None
def bootmean(rows,key):
 rng=np.random.default_rng(SEED);v=np.zeros(4000)
 for task in sorted({r['task'] for r in rows}):
  x=np.array([r[key] for r in rows if r['task']==task],float)
  v+=x[rng.integers(len(x),size=(4000,len(x)))].sum(1)
 return [float(x) for x in np.quantile(v/len(rows),[.025,.975])]
M=Path('/home/zonghan/ttt/audits/dynamic_old1k_v2_merge_20260921')
B=Path('/home/zonghan/ttt/transfer/dynamic_old1k_v2_noncwe_half_20260918')
MR=Path('/home/zonghan/ttt/runs/formal/mechanism_old1k_v2_20260921')
mp=j(MR/'PROTOCOL.json'); mph=provenance[str(MR/'PROTOCOL.json')]['sha256']; cohort=j(MR/'COHORT.json'); mc={x['sample_id']:x for x in cohort}; r2total=0; reusedtotal=0; r2coverage=[]
sources=j(M/'RESULT_SOURCES.json'); prior=j(M/'FINAL_AUDIT.json');protocol=j(B/'PROTOCOL.json');ph=provenance[str(B/'PROTOCOL.json')]['sha256'];refs=j(B/'data/fixed_reference.json')
fixed=[x for x in jl(B/'data/fixed_all6500.jsonl') if x['task']!='ruler_cwe_16k'];fx={x['sample_id']:x for x in fixed}
check('mechanism_source_protocol',mp['source_protocol_sha256']==ph and mp['model_sha256']==protocol['model_sha256']);check('latest_protocol',ph==prior['protocol_sha256']);check('full_cohort',len(fx)==len(fixed)==6000)
check('searched_cohort',set(sources)=={x['sample_id'] for x in fixed if x['sample_best']<1} and len(sources)==1433)
rows=[];land=[];align=[];search=[];reverse=[];coverage=[];full=[];machine=Counter();candidate_total=0;pairmass=[]
for index,sid in enumerate(sorted(sources)):
 src=sources[sid];d=j(src['path']);check('source_hash',provenance[src['path']]['sha256']==src['sha256'])
 check('sample_identity',d['sample_id']==sid and d['task']==fx[sid]['task']);check('protocol',d['protocol_hash']==ph)
 for k in ['model_sha256','executor_sha256']:check(k,d[k]==protocol[k])
 check('fixed_reference',d['fixed_reference_match'] and d['fixed_scores']=={a:v['score'] for a,v in refs[sid].items()})
 w=d['winner'];seq=w['sequence'];T=len(seq);sb=d['sample_best'];native=d['dynamic_score'];cs=d['candidates'];machine[d['machine']]+=1;candidate_total+=len(cs)
 check('scores',abs(sb-max(d['fixed_scores'].values()))<=EPS and sb==fx[sid]['sample_best'] and native>=sb-EPS and abs(d['gain']-(native-sb))<=EPS)
 check('winner',w['score']==native and w['ttt_chunk_size']==1024 and T==w['full_chunk_count'] and w['checkpoint_hash']==protocol['model_sha256'])
 check('candidate_count',len(cs)==d['candidate_count'] and max(c['score'] for c in cs)==native)
 check('winner_candidate',any(c['sequence']==seq and c['prediction_hash']==w['prediction_hash'] for c in cs))
 unique={};dups=0;conflicts=0
 for c in cs:
  q=tuple(c['sequence']);check('candidate_sequence',len(q)==T and set(q)<=set(A) and canon(list(q))==c['sequence_hash'])
  check('candidate_metadata',c['sample_id']==sid and c['task']==d['task'] and c['ttt_chunk_size']==1024 and c['checkpoint_hash']==protocol['model_sha256'] and c['generation_update_count']==0 and c['clean_full_replay'] and c['recompute_downstream_updates'] and not c['stored_delta_stitching'] and math.isfinite(c['score']))
  if q in unique:
   dups+=1;conflicts+=int(c['score']!=unique[q]['score'] or c['prediction_hash']!=unique[q]['prediction_hash'])
  else:unique[q]=c
 check('no_conflicting_duplicates',conflicts==0)
 one={}
 for q,c in unique.items():
  swc=[i for i in range(1,T) if q[i]!=q[i-1]]
  if len(swc)==1:one[q[0],q[-1],swc[0]]=c
 coverage.append({'sample_id':sid,'task':d['task'],'improved':native>sb+EPS,'T':T,'candidate_n':len(cs),'duplicate_sequence_n':dups,'observed_one_switch_n':len(one),'full_R2_expected_n':42*(T-1),'R2_coverage':len(one)/(42*(T-1)),'complete_R2':len(one)==42*(T-1),'reverse_available_n':sum((b,a,T-t) in one for a,b,t in one)})
 full.append({'sample_id':sid,'task':d['task'],'gain':native-sb,'sample_best':sb,'dynamic_score':native})
 if native<=sb+EPS:continue
 cp=mc[sid];check('mechanism_dynamic_source',cp['dynamic_source']==src and cp['sequence']==seq and cp['native_score']==native and cp['native_prediction_hash']==w['prediction_hash'] and cp['sample_best']==sb)
 rd=j(MR/'r2'/(hashlib.sha256(sid.encode()).hexdigest()+'.json'));check('r2_protocol',rd['protocol_sha256']==mph and rd['sample_id']==sid and rd['task']==d['task'] and rd['native_score']==native and rd['sample_best']==sb)
 one={}
 for c in rd['candidates']:
  aa,bb,tt=c['A'],c['B'],c['tau'];key=(aa,bb,tt);q=tuple([aa]*tt+[bb]*(T-tt))
  check('r2_unique',key not in one);check('r2_schedule',aa in A and bb in A and aa!=bb and 1<=tt<T and canon(list(q))==c['sequence_hash'] and math.isfinite(c['score']) and len(c['prediction_hash'])==64)
  if c['reused']:check('r2_reuse_parity',q in unique and c['score']==unique[q]['score'] and c['prediction_hash']==unique[q]['prediction_hash'])
  one[key]=c
 check('r2_full_family',set(one)=={(aa,bb,tt) for aa in A for bb in A if aa!=bb for tt in range(1,T)} and len(one)==rd['candidate_count']==cp['r2_candidates'])
 check('r2_reuse_count',sum(c['reused'] for c in one.values())==rd['reused'] and len(one)-rd['reused']==rd['executed_new'])
 r2total+=len(one);reusedtotal+=rd['reused'];r2coverage.append({'sample_id':sid,'expected':42*(T-1),'observed':len(one),'reused':rd['reused'],'new':rd['executed_new'],'complete':True})
 sw=[i for i in range(1,T) if seq[i]!=seq[i-1]];runs=[(a,len(list(g))) for a,g in itertools.groupby(seq)];counts=Counter(seq);e=modes(seq[:T//2]);l=modes(seq[T//2:])
 best=max((c['score'] for c in one.values()),default=None);matches=[(k,c) for k,c in one.items() if abs(c['score']-native)<=EPS];opts=[(k,c) for k,c in one.items() if abs(c['score']-best)<=EPS]
 check('one_switch_available',bool(one));o=min(opts,key=lambda kc:(kc[0][2],A.index(kc[0][0]),A.index(kc[0][1])));a,b,t=o[0]
 check('r2_best_score',best==rd['r2_score']);check('r2_best_tie_rule',(a,b,t)==(rd['best']['A'],rd['best']['B'],rd['best']['tau']));check('r2_prediction_parity',o[1]['prediction_hash']==rd['winner']['prediction_hash'] and rd['winner']['score']==best)
 pairrows=[]
 for a0 in A:
  for b0 in A:
   if a0==b0:continue
   zz={t0:c for (aa,bb,t0),c in one.items() if (aa,bb)==(a0,b0)};mt=[t0 for t0,c in zz.items() if abs(c['score']-native)<=EPS];sg=segments(mt)
   q={'sample_id':sid,'task':d['task'],'A':a0,'B':b0,'T':T,'observed_tau_n':len(zz),'complete_pair':len(zz)==T-1,'match_n':len(mt),'matching_taus':json.dumps(sorted(mt)),'earliest_r2_match':min(mt) if mt else None,'latest_r2_match':max(mt) if mt else None,'r2_match_span':max(mt)-min(mt) if mt else None,'observed_contiguous_max_n':max(map(len,sg),default=0),'r2_match_segment_n':len(sg),'score_by_tau_null_means_unobserved':json.dumps([zz[t0]['score'] if t0 in zz else None for t0 in range(1,T)])};land.append(q);pairrows.append(q)
 rv=[]
 for k,c in one.items():
  aa,bb,t0=k;rk=(bb,aa,T-t0)
  if rk in one and A.index(aa)<A.index(bb):
   rc=one[rk];v={'sample_id':sid,'task':d['task'],'A':aa,'B':bb,'tau':t0,'reverse_tau':T-t0,'forward_score':c['score'],'reverse_score':rc['score'],'delta':c['score']-rc['score']};reverse.append(v);rv.append(v)
 if matches:
  al={'sample_id':sid,'task':d['task'],'matching_candidate_n':len(matches),'early_alignment':mean([k[0] in e for k,c in matches]),'late_alignment':mean([k[1] in l for k,c in matches]),'joint_alignment':mean([k[0] in e and k[1] in l for k,c in matches]),'joint_any':any(k[0] in e and k[1] in l for k,c in matches),'joint_all':all(k[0] in e and k[1] in l for k,c in matches),'modal_sets_overlap':bool(e&l),'early_modes':json.dumps(sorted(e)),'late_modes':json.dumps(sorted(l)),'all_observed_joint_baseline':mean([k[0] in e and k[1] in l for k in one])};align.append(al)
  for k,c in matches:pairmass.append({'sample_id':sid,'task':d['task'],'A':k[0],'B':k[1],'weight':1/len(matches)})
 r={'sample_id':sid,'task':d['task'],'T':T,'sample_best':sb,'native':native,'gain':native-sb,'switch_count':len(sw),'distinct_actions':len(counts),'longest_run':max(n for a0,n in runs),'mean_run_length':T/len(runs),'max_run_fraction':max(n for a0,n in runs)/T,'entropy':-sum(n/T*math.log2(n/T) for n in counts.values()),'return_count':len(runs)-len(counts),'aba_n':sum(runs[i][0]==runs[i-2][0] for i in range(2,len(runs))),'first_switch_norm':sw[0]/T,'last_switch_norm':sw[-1]/T,'r2_score':best,'r2_recovery':(best-sb)/(native-sb),'at_most_one_switch_recovery':(max(best,sb)-sb)/(native-sb),'r2_match':bool(matches),'r2_match_n':len(matches),'r2_best_n':len(opts),'r2_matching_pair_n':sum(q['match_n']>0 for q in pairrows),'r2_max_plateau_n':max(q['observed_contiguous_max_n'] for q in pairrows),'r2_coverage':len(one)/(42*(T-1)),'best_A':a,'best_B':b,'best_tau':t,'best_tau_norm':t/T,'best_mean_distance_first_norm':mean([abs(k[2]-sw[0])/T for k,c in opts]),'best_mean_distance_last_norm':mean([abs(k[2]-sw[-1])/T for k,c in opts]),'available_reverse_pairs':len(rv),'available_reverse_mean_abs_delta':mean([abs(q['delta']) for q in rv]),'available_reverse_unequal_fraction':mean([abs(q['delta'])>EPS for q in rv]),'best_reverse_delta_if_observed':best-one[b,a,T-t]['score'] if (b,a,T-t) in one else None,'selected_best_prediction_match':bool(matches) and o[1]['prediction_hash']==w['prediction_hash'],'any_matching_prediction_match':any(c['prediction_hash']==w['prediction_hash'] for k,c in matches),'matching_prediction_fraction':mean([c['prediction_hash']==w['prediction_hash'] for k,c in matches]),'trajectory':json.dumps(seq)};rows.append(r)
 search.append({'sample_id':sid,'task':d['task'],'candidate_n':len(cs),'first_improvement_list_index':next(i+1 for i,c in enumerate(cs) if c['score']>sb+EPS),'first_final_score_list_index':next(i+1 for i,c in enumerate(cs) if abs(c['score']-native)<=EPS),'first_selected_sequence_list_index':next(i+1 for i,c in enumerate(cs) if c['sequence']==seq),'explicit_replay_index_n':sum('replay_index' in c for c in cs),'parent_field_n':sum('parent' in c for c in cs)})
 if len(rows)%40==0:print('CPU improved samples analyzed',len(rows),flush=True)
check('improved_cohort',len(rows)==prior['full6000']['improved']==170 and {r['sample_id'] for r in rows}==set(mc));check('r2_total',r2total==mp['counts']['r2_candidates']);csvout('r2_candidate_coverage.csv',r2coverage)
for f in fixed:
 if f['sample_id'] not in sources:full.append({'sample_id':f['sample_id'],'task':f['task'],'gain':0,'sample_best':f['sample_best'],'dynamic_score':f['sample_best']})
check('full_gain',abs(mean([r['gain'] for r in full])*100-prior['full6000']['gain_pp'])<EPS)
for name,data in [('per_sample_improved.csv',rows),('candidate_coverage_all_searched.csv',coverage),('pair_landscapes.csv',land),('modal_alignment.csv',align),('count_matched_reversals.csv',reverse),('search_log_coverage.csv',search)]:csvout(name,data)
subsets={'improved170':rows,'native_one_switch':[r for r in rows if r['switch_count']==1],'native_multi_switch':[r for r in rows if r['switch_count']>1],'multi_matched':[r for r in rows if r['switch_count']>1 and r['r2_match']],'multi_unmatched':[r for r in rows if r['switch_count']>1 and not r['r2_match']]}
metrics=['gain','switch_count','distinct_actions','return_count','entropy','max_run_fraction','r2_recovery','at_most_one_switch_recovery','r2_match_n','r2_max_plateau_n','r2_coverage','best_tau_norm','best_mean_distance_first_norm','best_mean_distance_last_norm']
summary={}
for name,rr in subsets.items():
 ids={r['sample_id'] for r in rr};aa=[q for q in align if q['sample_id'] in ids];vv=[r for r in rr if r['available_reverse_pairs']]
 summary[name]={'n':len(rr),'r2_match_n':sum(r['r2_match'] for r in rr),'revisit_n':sum(r['return_count']>0 for r in rr),'aba_n':sum(r['aba_n']>0 for r in rr),'metrics':{k:desc([r[k] for r in rr]) for k in metrics},'recovery_ci95':bootmean(rr,'at_most_one_switch_recovery') if rr else None,'single_r2_match_n':sum(r['r2_match_n']==1 for r in rr),'multiple_r2_match_n':sum(r['r2_match_n']>1 for r in rr),'adjacent_matching_tau_n':sum(r['r2_max_plateau_n']>=2 for r in rr),'selected_prediction_match_n':sum(r['selected_best_prediction_match'] for r in rr),'any_matching_prediction_match_n':sum(r['any_matching_prediction_match'] for r in rr),'alignment_sample_n':len(aa),'alignment':{k:mean([q[k] for q in aa]) for k in ['early_alignment','late_alignment','joint_alignment','joint_any','joint_all','all_observed_joint_baseline']},'reverse_sample_n':len(vv),'available_reverse_mean_abs_delta':mean([r['available_reverse_mean_abs_delta'] for r in vv]),'available_reverse_mean_unequal_fraction':mean([r['available_reverse_unequal_fraction'] for r in vv])}
print('CPU computing correlations',flush=True)
correlations=[]
for name in ['improved170','native_multi_switch']:
 rr=subsets[name];groups=[np.array([i for i,r in enumerate(rr) if r['task']==t0]) for t0 in sorted({r['task'] for r in rr})]
 for key in ['switch_count','distinct_actions','entropy','first_switch_norm','last_switch_norm','max_run_fraction']:
  x=np.array([r['gain'] for r in rr]);y=np.array([r[key] for r in rr]);rng=np.random.default_rng(SEED);bs=[]
  for _ in range(2000):
   ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]);v=rho(x[ix],y[ix])
   if v is not None:bs.append(v)
  correlations.append({'cohort':name,'metric':key,'n':len(rr),'rho':rho(x,y),'ci_low':float(np.quantile(bs,.025)) if bs else None,'ci_high':float(np.quantile(bs,.975)) if bs else None,'valid_bootstrap_n':len(bs)})
for task in sorted({r['task'] for r in rows}):
 rr=[r for r in rows if r['task']==task]
 for key in ['switch_count','distinct_actions','entropy','max_run_fraction']:
  correlations.append({'cohort':task,'metric':key,'n':len(rr),'rho':rho([r['gain'] for r in rr],[r[key] for r in rr]),'ci_low':None,'ci_high':None,'valid_bootstrap_n':0})
csvout('gain_complexity_correlations.csv',correlations)
tasks=[]
for task in sorted({r['task'] for r in full}):
 rr=[r for r in rows if r['task']==task];ff=[r for r in full if r['task']==task]
 tasks.append({'task':task,'full_n':len(ff),'improved_n':len(rr),'full_gain_pp':mean([r['gain'] for r in ff])*100,'r2_match_n':sum(r['r2_match'] for r in rr),**{k:mean([r[k] for r in rr]) for k in ['switch_count','at_most_one_switch_recovery','r2_coverage','r2_max_plateau_n']}})
csvout('per_task.csv',tasks)
bins=[]
for n in sorted({r['switch_count'] for r in rows}):
 rr=[r for r in rows if r['switch_count']==n];bins.append({'switch_count':n,'n':len(rr),'mean_gain':mean([r['gain'] for r in rr]),'r2_match_n':sum(r['r2_match'] for r in rr)})
csvout('gain_by_switch_count.csv',bins)
trans=[]
for name,rr in subsets.items():
 ids={r['sample_id'] for r in rr if r['r2_match']};obs=Counter();freq=Counter()
 for q in pairmass:
  if q['sample_id'] in ids:obs[q['A'],q['B']]+=q['weight']
 for r in rr:
  if r['sample_id'] in ids:
   for a0,n in Counter(json.loads(r['trajectory'])).items():freq[a0]+=n/r['T']/len(ids)
 den=sum(freq[a0]*freq[b0] for a0 in A for b0 in A if a0!=b0)
 for a0 in A:
  for b0 in A:
   if a0!=b0:
    ex=len(ids)*freq[a0]*freq[b0]/den if den else 0;trans.append({'cohort':name,'A':a0,'B':b0,'matching_samples':len(ids),'observed_sample_mass':obs[a0,b0],'native_marginal_expected':ex,'descriptive_enrichment':obs[a0,b0]/ex if ex else None})
csvout('transition_enrichment.csv',trans)
save('summary.json',summary)
save('integrity.json',{'status':'PASS','protocol_sha256':ph,'sources_verified':len(sources),'candidate_records_checked':candidate_total,'machine_counts':dict(machine),'full6000_gain_pp':mean([r['gain'] for r in full])*100,'complete_R2_improved_samples':len(r2coverage),'r2_candidate_n':r2total,'r2_reused_n':reusedtotal,'mechanism_protocol_sha256':mph,'CPU_only':True,'model_imported':False,'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'numpy':np.__version__,'checks':dict(checks),'bootstrap_seed':SEED,'mean_bootstrap_resamples':4000,'correlation_bootstrap_resamples':2000})
save('input_provenance.json',provenance)
print(json.dumps({n:{'n':s['n'],'match':s['r2_match_n'],'revisit':s['revisit_n'],'recovery':s['metrics']['at_most_one_switch_recovery'],'coverage':s['metrics']['r2_coverage'],'prediction_any':s['any_matching_prediction_match_n'],'alignment':s['alignment']} for n,s in summary.items()},indent=2),flush=True)
