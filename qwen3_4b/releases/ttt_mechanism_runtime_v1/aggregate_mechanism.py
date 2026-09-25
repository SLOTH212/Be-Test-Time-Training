#!/usr/bin/env python3
"""Aggregate portable sample commits with the frozen experiment-specific procedures."""
import argparse,json,statistics
from collections import defaultdict
from pathlib import Path
import runtime as rt
from worker import Store,atomic

def procedures(mechanism):
 import numpy as np
 p=next((rt.ROOT/'provenance/original'/mechanism/'scripts').glob('aggregate*.py'))
 import ast
 t=ast.parse(p.read_text());names=[x.name for x in t.body if isinstance(x,ast.FunctionDef) and x.name not in ['main','aggregate']]
 return rt.source_functions(str(p.relative_to(rt.ROOT)),names,{'np':np,'statistics':statistics,'ACTIONS':[],'TOL':1e-12,'CELLS':['F00','F10','F01','F11'],'LEVELS':['L1','L2','L3','L4','L5','L6']})
def aggregate(records,mechanism,seeds):
 if not records:raise ValueError('EMPTY_COHORT')
 if len({r['sample_id'] for r in records})!=len(records):raise ValueError('DUPLICATE_SAMPLE')
 m=procedures(mechanism);B=seeds['resamples'];N=len(records);out={'N':N,'mechanism':mechanism,'statistical_procedure':seeds}
 if mechanism in ['reverse','reset']:
  d=[r['parent_dynamic_score']-r['intervention_score'] for r in records];ng=[r['parent_dynamic_score']-r['sample_best'] for r in records];rg=[r['intervention_score']-r['sample_best'] for r in records]
  if any(g<=0 for g in ng):raise ValueError('PARENT_NOT_IMPROVED')
  field='dynamic_minus_reverse' if mechanism=='reverse' else 'native_minus_reset';rs=[{'task':r['task'],field:d[i]} for i,r in enumerate(records)]
  out.update(native_mean=statistics.fmean(r['parent_dynamic_score'] for r in records),intervention_mean=statistics.fmean(r['intervention_score'] for r in records),effect_mean=statistics.fmean(d),effect_median=statistics.median(d),effect_std=statistics.stdev(d) if N>1 else None,worse=sum(x>(1e-12 if mechanism=='reverse' else 0) for x in d),equal=sum(abs(x)<=1e-12 if mechanism=='reverse' else x==0 for x in d),better=sum(x<(-1e-12 if mechanism=='reverse' else 0) for x in d),retained_gain_mean=statistics.fmean(a/b for a,b in zip(rg,ng)),aggregate_retained_gain=sum(rg)/sum(ng),paired_ci=m.ci(d,seeds['paired'],B),task_stratified_ci=(m.stratified_ci if mechanism=='reverse' else m.strat)(rs,seeds['stratified'],B))
 elif mechanism=='deletion':
  samples=[];shams=[]
  for r in records:
   details=r['details']['deletion'];branches=details['selective_branches'];score=statistics.fmean(x['result']['score'] for x in branches)
   if score!=r['intervention_score']:raise ValueError('DELETION_BRANCH_MEAN')
   normal=r['parent_dynamic_score'];den=normal-details['full_reset_score'];samples.append({'task':r['task'],'normal_minus_selective':normal-score,'normal_minus_full_reset':den,'transition_category':details['plan']['transition_category']})
   if details['sham'] is not None:shams.append(normal-details['sham']['result']['score'])
  pos=[x for x in samples if x['normal_minus_full_reset']>0];d=[x['normal_minus_selective'] for x in samples]
  out.update(effect_mean=statistics.fmean(d),paired_ci=m.ci(d,seeds['paired'],B),task_stratified_ci=m.strat(samples,seeds['stratified'],B),positive_denominator_n=len(pos),nonpositive_denominator_n=N-len(pos),explained_fraction_mean=statistics.fmean(x['normal_minus_selective']/x['normal_minus_full_reset'] for x in pos) if pos else None,aggregate_explained_fraction=sum(x['normal_minus_selective'] for x in pos)/sum(x['normal_minus_full_reset'] for x in pos) if pos else None,sham_n=len(shams),sham_mean_effect=statistics.fmean(shams) if shams else None)
  dis=[x['normal_minus_selective'] for x in samples if x['transition_category']=='SINGLE_LAYER_DISJOINT'];out['single_layer_disjoint_ci']=m.ci(dis,seeds['disjoint'],B) if dis else None
 elif mechanism=='kp':
  es=[m.effects({c:r['details']['kp']['cells'][c]['generation_start_relative_l2'] for c in m.CELLS}) for r in records]
  out['paired_factorial_bootstrap']={k:m.boot([x[k] for x in es],seeds['paired'],B) for k in es[0]}
  out['closure']={c:{l:sum(r['details']['kp']['cells'][c]['closure'][l] for r in records) for l in m.LEVELS} for c in m.CELLS}
  out['reward_means']={c:statistics.fmean(r['details']['kp']['cells'][c]['terminal_reward'] for r in records) for c in m.CELLS}
 elif mechanism=='r2':
  data=[]
  for r in records:
   off=r['details']['r2']['off_score'];native=r['parent_dynamic_score'];v=r['intervention_score'];data.append({'task':r['task'],'Dynamic_minus_R2':native-v,'Dynamic_minus_OFF':native-off,'R2_minus_OFF':v-off})
  den=sum(x['Dynamic_minus_OFF'] for x in data)
  if den<=0:raise ValueError('RETENTION_DENOMINATOR')
  out.update(dynamic_minus_r2_mean=statistics.mean(x['Dynamic_minus_R2'] for x in data),r2_mean=statistics.mean(r['intervention_score'] for r in records),native_mean=statistics.mean(r['parent_dynamic_score'] for r in records),aggregate_gain_retention_vs_off=sum(x['R2_minus_OFF'] for x in data)/den,paired_bootstrap=m.bootstrap(data,seeds['paired'],B,False),task_stratified_bootstrap=m.bootstrap(data,seeds['stratified'],B,True),equal=sum(abs(x['Dynamic_minus_R2'])<=1e-12 for x in data),gap_n=sum(x['Dynamic_minus_R2']>1e-12 for x in data))
 else:raise ValueError('UNKNOWN_MECHANISM')
 return out

def main():
 a=argparse.ArgumentParser();a.add_argument('--run-root',required=True);args=a.parse_args();root=Path(args.run_root);identity=json.loads((root/'CONFIG_IDENTITY.json').read_text())['config'];cfg=identity['config'];s=Store(root,identity);rs=[]
 for p in sorted((root/'samples').glob('*.json')):
  sid=json.loads(p.read_text())['sample_id'];rs.append(rt.validate_result(s.committed(sid),committed=True))
 # Canonical cohort ordering is preserved for seeded, task-stratified resampling.
 rs.sort(key=lambda r:r['benchmark_global_index'])
 if len(rs)!=identity['cohort_n']:raise ValueError('INCOMPLETE_COHORT')
 out=aggregate(rs,cfg['mechanism'],cfg['statistics']);atomic(root/'MECHANISM_AGGREGATE.json',out);print(json.dumps(out,sort_keys=True))
if __name__=='__main__':main()
