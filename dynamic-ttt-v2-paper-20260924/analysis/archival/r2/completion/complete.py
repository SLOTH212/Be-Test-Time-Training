import os
os.environ.update(CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='none',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLBACKEND='Agg',MPLCONFIGDIR='/tmp/r2_cpu_mplconfig')
import json,csv,hashlib,math
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
P=Path(__file__).resolve().parent; BASE=P.parent; R=Path('/home/zonghan/ttt/runs/formal/mechanism_old1k_v2_20260921')
A=['OFF','L0','L6','L12','L18','L24','ALL'];EPS=1e-12;SEED=20260923
prov={}
def j(path):
 b=Path(path).read_bytes();prov[str(path)]=hashlib.sha256(b).hexdigest();return json.loads(b)
def save(n,x):(P/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def csvout(n,rows):
 with (P/n).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def mean(x):return float(np.mean(x)) if len(x) else None
def modes(x):
 c=Counter(x);return {a for a,n in c.items() if n==max(c.values())}
def boot(rs,k):
 rg=np.random.default_rng(SEED);v=np.zeros(4000)
 for t in sorted({r['task'] for r in rs}):
  x=np.array([r[k] for r in rs if r['task']==t]);v+=x[rg.integers(len(x),size=(4000,len(x)))].sum(1)
 return np.quantile(v/len(rs),[.025,.975]).tolist()
def typ(a,b):
 if a=='OFF':return 'OFF_to_ALL' if b=='ALL' else 'OFF_to_layer'
 if b=='OFF':return 'ALL_to_OFF' if a=='ALL' else 'layer_to_OFF'
 if a=='ALL':return 'ALL_to_layer'
 if b=='ALL':return 'layer_to_ALL'
 return 'shallow_to_deep' if int(a[1:])<int(b[1:]) else 'deep_to_shallow'
TYPES=sorted({typ(a,b) for a in A for b in A if a!=b})
cohort=j(R/'COHORT.json');cp={r['sample_id']:r for r in cohort}; prior=j(BASE/'input_provenance.json');data={};sets=[];mass=[];pairnull=[];checks=Counter()
for path in sorted((R/'r2').glob('*.json')):
 d=j(path);sid=d['sample_id'];assert prov[str(path)]==prior[str(path)]['sha256'];c=cp[sid];seq=c['sequence'];T=c['T'];z={(x['A'],x['B'],x['tau']):x for x in d['candidates']};assert len(z)==42*(T-1)
 best=max(x['score'] for x in z.values());native=c['native_score'];e=modes(seq[:T//2]);l=modes(seq[T//2:]);e2=modes(seq[:(T+1)//2]);l2=modes(seq[(T+1)//2:]);sw=[i for i in range(1,T) if seq[i]!=seq[i-1]]
 data[sid]=(d,c,z)
 freq={a:seq.count(a)/T for a in A};den=sum(freq[a]*freq[b] for a in A for b in A if a!=b)
 for a in A:
  for b in A:
   if a!=b:pairnull.append({'sample_id':sid,'task':c['task'],'A':a,'B':b,'native_occupancy_null':freq[a]*freq[b]/den})
 for label,target in [('best',best),('match',native)]:
  zz=[x for x in z.values() if abs(x['score']-target)<=EPS]
  if not zz:continue
  deltas=[x['score']-z[x['B'],x['A'],T-x['tau']]['score'] for x in zz]
  assert all(v>=-EPS for v in deltas) # This dataset has no R2 score above Native.
  rec={'sample_id':sid,'task':c['task'],'set':label,'native_switches':len(sw),'native_matched':abs(best-native)<=EPS,'candidate_n':len(zz),'early':mean([x['A'] in e for x in zz]),'late':mean([x['B'] in l for x in zz]),'joint':mean([x['A'] in e and x['B'] in l for x in zz]),'joint_any':any(x['A'] in e and x['B'] in l for x in zz),'joint_all':all(x['A'] in e and x['B'] in l for x in zz),'joint_odd_split':mean([x['A'] in e2 and x['B'] in l2 for x in zz]),'modal_overlap':bool(e&l),'uniform_42_pair_joint_baseline':sum(a in e and b in l for a in A for b in A if a!=b)/42,'reverse_mean_delta':mean(deltas),'reverse_positive_fraction':mean([v>EPS for v in deltas]),'reverse_equal_fraction':mean([abs(v)<=EPS for v in deltas]),'reverse_min_delta':min(deltas),'reverse_max_delta':max(deltas),'reverse_all_positive':all(v>EPS for v in deltas),'tau_mean_norm':mean([x['tau']/T for x in zz]),'distance_first_mean_norm':mean([abs(x['tau']-sw[0])/T for x in zz]),'distance_last_mean_norm':mean([abs(x['tau']-sw[-1])/T for x in zz])};sets.append(rec)
  for x in zz:mass.append({'sample_id':sid,'task':c['task'],'set':label,'A':x['A'],'B':x['B'],'type':typ(x['A'],x['B']),'weight':1/len(zz)})
 checks['r2_sample_verified']+=1
csvout('setwise_alignment_and_reversal.csv',sets)
agg=[]
for label in ['best','match']:
 for scope,pred in [('all',lambda x:True),('native_multi',lambda x:x['native_switches']>1),('native_multi_matched',lambda x:x['native_switches']>1 and x['native_matched']),('native_unmatched',lambda x:not x['native_matched'])]:
  rr=[x for x in sets if x['set']==label and pred(x)]
  if not rr:continue
  out={'set':label,'scope':scope,'n':len(rr),'joint_any_n':sum(x['joint_any'] for x in rr),'modal_overlap_n':sum(x['modal_overlap'] for x in rr),'reverse_all_positive_n':sum(x['reverse_all_positive'] for x in rr)}
  for key in ['early','late','joint','joint_odd_split','reverse_mean_delta','reverse_positive_fraction','reverse_equal_fraction','tau_mean_norm','distance_first_mean_norm','distance_last_mean_norm']:
   out[key]=mean([x[key] for x in rr]);lo,hi=boot(rr,key);out[key+'_ci_low']=lo;out[key+'_ci_high']=hi
  agg.append(out)
csvout('setwise_summary.csv',agg)
# Every sample contributes total observed mass 1; expected mass from its own Native occupancy.
tr=[];tt=[];tasklist=sorted({c['task'] for c in cohort})
for label in ['best','match']:
 for task in ['ALL_TASKS']+tasklist:
  rr=[x for x in sets if x['set']==label and (task=='ALL_TASKS' or x['task']==task)];ids={x['sample_id'] for x in rr};obs=Counter();ex=Counter()
  for m in mass:
   if m['set']==label and m['sample_id'] in ids:obs[m['A'],m['B']]+=m['weight']
  for q in pairnull:
   if q['sample_id'] in ids:ex[q['A'],q['B']]+=q['native_occupancy_null']
  assert abs(sum(obs.values())-len(ids))<1e-8 and abs(sum(ex.values())-len(ids))<1e-8
  for a in A:
   for b in A:
    if a!=b:tr.append({'set':label,'task':task,'n':len(ids),'A':a,'B':b,'observed_mass':obs[a,b],'observed_fraction':obs[a,b]/len(ids),'expected_mass':ex[a,b],'enrichment':obs[a,b]/ex[a,b] if ex[a,b]>EPS else None})
  for t in TYPES:
   v=sum(v for (a,b),v in obs.items() if typ(a,b)==t);ev=sum(v for (a,b),v in ex.items() if typ(a,b)==t)
   tt.append({'set':label,'task':task,'n':len(ids),'type':t,'observed_mass':v,'observed_fraction':v/len(ids),'expected_mass':ev,'enrichment':v/ev if ev>EPS else None})
csvout('task_transition_enrichment.csv',tr);csvout('task_transition_types.csv',tt)
cons=[];dom=[]
for label in ['best','match']:
 for task in tasklist:
  zs=[x for x in tt if x['set']==label and x['task']==task];mx=max(x['observed_mass'] for x in zs);tops=[x['type'] for x in zs if abs(x['observed_mass']-mx)<EPS]
  dom.append({'set':label,'task':task,'n':zs[0]['n'],'dominant_type_ties':json.dumps(tops),'dominant_fraction':mx/zs[0]['n']})
 for t in TYPES:
  zs=[x for x in tt if x['set']==label and x['type']==t and x['task']!='ALL_TASKS'];valid=[x for x in zs if x['enrichment'] is not None]
  cons.append({'set':label,'type':t,'tasks_n':len(zs),'tasks_with_defined_null':len(valid),'tasks_enriched_gt1':sum(x['enrichment']>1+EPS for x in valid),'tasks_observed_positive':sum(x['observed_mass']>EPS for x in zs),'macro_task_fraction':mean([x['observed_fraction'] for x in zs]),'micro_sample_fraction':next(x['observed_fraction'] for x in tt if x['set']==label and x['type']==t and x['task']=='ALL_TASKS')})
csvout('dominant_transition_by_task.csv',dom);csvout('cross_task_direction_consistency.csv',cons)
# Binned descriptions and within-task centered rank association (complements pooled Spearman).
rs=list(csv.DictReader((BASE/'per_sample_improved.csv').open()))
def rank(v):
 v=np.array(v);return np.array([sum(v<a)+(sum(v==a)-1)/2 for a in v],float)
def corr(x,y):return float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else None
within=[]
for key in ['switch_count','distinct_actions','entropy','first_switch_norm','last_switch_norm','max_run_fraction']:
 xx=[];yy=[]
 for t in tasklist:
  rr=[r for r in rs if r['task']==t];x=rank([float(r['gain']) for r in rr]);y=rank([float(r[key]) for r in rr]);xx.extend(x-x.mean());yy.extend(y-y.mean())
 within.append({'metric':key,'within_task_centered_rank_correlation':corr(xx,yy),'n':len(xx)})
csvout('within_task_rank_correlations.csv',within)
# Freeze all inputs and numerical artifacts before plotting.
save('input_provenance.json',prov);save('checks.json',{'status':'PASS','checks':dict(checks),'CPU_only':True,'setwise_rows':len(sets),'weighting':'uniform candidates within sample, equal samples within scope','bootstrap':'4000 sample bootstrap within task; seed20260923','expected_transition':'per-sample Native occupancy product conditioned on unequal actions','no_new_model_execution':True})
print('Numerical supplement complete',flush=True)
import sys
sys.path.insert(0,'/tmp/r2_cpu_plot_deps')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
FIG=P/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':9,'svg.fonttype':'none','pdf.fonttype':42})
# Complete landscapes, all 170 samples, 42 ordered pairs per sample. Missing data impossible by gates.
pairs=[(a,b) for a in A for b in A if a!=b]
with PdfPages(FIG/'all_170_reward_landscapes.pdf') as pdf:
 for i,(sid,(d,c,z)) in enumerate(sorted(data.items())):
  T=c['T'];v=np.array([[z[a,b,t]['score'] for t in range(1,T)] for a,b in pairs]);fig,ax=plt.subplots(figsize=(10,11));im=ax.imshow(v,aspect='auto',vmin=0,vmax=1,cmap='viridis',extent=(.5,T-.5,41.5,-.5));ax.set_yticks(range(42),[a+' → '+b for a,b in pairs],fontsize=7);ax.set_xticks(range(1,T));ax.set_xlabel('Switch boundary tau (number of initial A chunks)');ax.set_ylabel('Ordered action pair');ax.set_title(f'{sid} | Native={c["native_score"]:.3f}, R2={d["r2_score"]:.3f}\nWhite dots: score equals Native; complete 42 × (T−1) grid')
  yy,xx=np.where(abs(v-c['native_score'])<=EPS);ax.scatter(xx+1,yy,s=5,color='white');fig.colorbar(im,ax=ax,label='Reward');fig.tight_layout();pdf.savefig(fig);plt.close(fig)
  if (i+1)%40==0:print('CPU landscape pages',i+1,flush=True)
# Six deterministic examples, lowest ID in each structural category; all curves retained in full tables/PDF.
categories=[('Multi-switch unmatched',lambda r:float(r['switch_count'])>1 and r['r2_match']=='False'),('Multi-switch unique match',lambda r:float(r['switch_count'])>1 and int(r['r2_match_n'])==1),('Multi-switch isolated multiple matches',lambda r:float(r['switch_count'])>1 and int(r['r2_match_n'])>1 and int(r['r2_max_plateau_n'])==1),('Multi-switch adjacent matches',lambda r:float(r['switch_count'])>1 and int(r['r2_max_plateau_n'])>=2),('One-switch unique match',lambda r:float(r['switch_count'])==1 and int(r['r2_match_n'])==1),('One-switch broad matches',lambda r:float(r['switch_count'])==1 and int(r['r2_max_plateau_n'])>=3)]
examples=[]
with PdfPages(FIG/'example_pair_curves.pdf') as pdf:
 for title,pred in categories:
  rr=sorted([r for r in rs if pred(r)],key=lambda r:r['sample_id']);
  if not rr:continue
  sid=rr[0]['sample_id'];d,c,z=data[sid];T=c['T'];examples.append({'category':title,'sample_id':sid,'selection':'lexicographically first qualifying sample'})
  fig,axes=plt.subplots(4,2,figsize=(11,12),sharex=True,sharey=True)
  for ax,a in zip(axes.flat,A):
   for b in A:
    if a!=b:ax.plot(range(1,T),[z[a,b,t]['score'] for t in range(1,T)],marker='.',lw=.9,label=b)
   ax.axhline(c['native_score'],color='black',ls='--',lw=.8);ax.set_title('Initial action '+a);ax.set_ylim(-.05,1.05);ax.set_ylabel('Reward');ax.set_xlabel('tau');ax.legend(ncol=3,fontsize=6)
  axes.flat[-1].axis('off');fig.suptitle(f'{title}: {sid}\nAll 42 ordered pairs; dashed line = Native');fig.tight_layout(rect=[0,0,1,.95]);pdf.savefig(fig);plt.close(fig)
csvout('figure_example_selection.csv',examples)
# Task-stratified scatter with deterministic display jitter only for overlapping discrete switch counts.
fig,axes=plt.subplots(3,3,figsize=(12,10));rg=np.random.default_rng(SEED)
for ax,t in zip(axes.flat,tasklist):
 rr=[r for r in rs if r['task']==t];x=np.array([float(r['switch_count']) for r in rr]);y=[float(r['gain']) for r in rr];col=['#167c80' if r['r2_match']=='True' else '#c04e37' for r in rr];ax.scatter(x+rg.uniform(-.10,.10,len(x)),y,c=col,alpha=.7,s=20);ax.set_title(t.replace('ruler_','').replace('_16k','')+f' (n={len(rr)})');ax.set_xlabel('Native switch count (display jitter ±0.1)');ax.set_ylabel('Dynamic − SampleBest');ax.set_ylim(-.04,1.04)
fig.suptitle('Improved samples only | teal: R2 matched; orange: unmatched');fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(FIG/'task_gain_complexity.png',dpi=160);fig.savefig(FIG/'task_gain_complexity.pdf');plt.close(fig)
# Transition matrices: sample-normalized match mass and native occupancy enrichment.
for label in ['best','match']:
 fig,axes=plt.subplots(1,2,figsize=(11,5));zs=[x for x in tr if x['set']==label and x['task']=='ALL_TASKS'];obs=np.full((7,7),np.nan);en=np.full((7,7),np.nan)
 for q in zs:
  i,k=A.index(q['A']),A.index(q['B']);obs[i,k]=q['observed_fraction'];en[i,k]=q['enrichment'] if q['enrichment'] is not None else np.nan
 for ax,v,title in [(axes[0],obs,'Sample-normalized transition mass'),(axes[1],en,'Enrichment vs per-sample Native occupancy')]:
  im=ax.imshow(v,cmap='viridis');ax.set_xticks(range(7),A);ax.set_yticks(range(7),A);ax.set_xlabel('Late action B');ax.set_ylabel('Early action A');ax.set_title(title);fig.colorbar(im,ax=ax,shrink=.75)
 fig.suptitle(label+' R2 set | undefined nulls and diagonal blank');fig.tight_layout();fig.savefig(FIG/(label+'_transition_matrix.png'),dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(13,6));v=np.array([[next(q['observed_fraction'] for q in tt if q['set']=='match' and q['task']==t and q['type']==ty) for ty in TYPES] for t in tasklist]);im=ax.imshow(v,vmin=0,vmax=1,cmap='viridis',aspect='auto');ax.set_xticks(range(len(TYPES)),TYPES,rotation=35,ha='right');ax.set_yticks(range(len(tasklist)),[t.replace('ruler_','').replace('_16k','') for t in tasklist]);ax.set_title('Matching R2 transition types by task | sample-normalized mass');fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(FIG/'task_transition_types.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(13,4));multi=[r for r in rs if float(r['switch_count'])>1];axes[0].hist([float(r['r2_recovery']) for r in multi],bins=[-.05,.05,.95,1.05],color='#167c80');axes[0].set_title('Recovery | multi-switch n=78');axes[0].set_xlabel('(R2−SB)/(Native−SB)');matched=[r for r in multi if r['r2_match']=='True'];axes[1].hist([int(r['r2_match_n']) for r in matched],bins=15,color='#167c80');axes[1].set_title('Number of matching R2 | n=62');axes[1].set_xlabel('Matching candidates');counts=Counter(int(r['r2_max_plateau_n']) for r in matched);axes[2].bar(list(counts),list(counts.values()),color='#167c80');axes[2].set_title('Longest contiguous match run | n=62');axes[2].set_xlabel('Number of adjacent tau values');fig.tight_layout();fig.savefig(FIG/'recovery_multiplicity_timing.png',dpi=160);plt.close(fig)
for path,h in prov.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
save('source_preservation.json',{'status':'PASS','inputs_rehashed':len(prov),'changed':[]})
print('CPU figures and source preservation complete',flush=True)
