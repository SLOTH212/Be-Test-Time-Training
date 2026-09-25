"""Compare portable aggregation to historical scalar outputs and frozen procedures."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import runtime as rt
from aggregate_mechanism import aggregate
R=rt.ROOT

def main():
 audit=json.loads((R/'audits/HISTORICAL_MECHANISM_MASTER_REPLAY.json').read_text());parents={x['sample_id']:x for x in rt.rows(R/'provenance/IMPROVED_173_PARENT_COHORT.jsonl')}
 for name,file,metric in [('reverse','reverse','reverse_score'),('reset','state_reset','reset_score')]:
  cfg=rt.load_config(R/f'configs/qwen3_1p7b_{file}_formal_reference.yaml');raw=rt.rows(R/f'tests/reference/{name}_scalar_results.jsonl')
  rs=[{'sample_id':x['canonical_sample_id'],'task':x['task'],'parent_dynamic_score':parents[x['canonical_sample_id']]['dynamic_score'],'sample_best':x['sample_best_score'],'intervention_score':x[metric]} for x in raw]
  got=aggregate(rs,name,cfg['statistics']);expected=audit[name]['summary'];assert got['paired_ci']==expected['PAIRED_BOOTSTRAP_95CI'] and got['task_stratified_ci']==expected['TASK_STRATIFIED_PAIRED_BOOTSTRAP_95CI'];assert got['aggregate_retained_gain']==expected['AGGREGATE_GAIN_RETENTION'];assert got['retained_gain_mean']==expected['GAIN_RETENTION_MEAN']
 raw=rt.rows(R/'tests/reference/deletion_scalar_results.jsonl');co=rt.rows(next((R/'provenance/original/deletion/manifests').glob('*COHORT.jsonl')));rs=[]
 for s in co:
  z=[x for x in raw if x['canonical_sample_id']==s['canonical_sample_id']];sel=[{'result':{'score':x['score']}} for x in z if x['intervention_type']=='SELECTIVE'];sh=[x for x in z if x['intervention_type']=='SHAM'];import statistics
  rs.append({'sample_id':s['canonical_sample_id'],'task':s['task'],'parent_dynamic_score':s['normal_score'],'intervention_score':statistics.fmean(x['result']['score'] for x in sel),'details':{'deletion':{'selective_branches':sel,'full_reset_score':s['full_reset_score'],'sham':{'result':{'score':sh[0]['score']}} if sh else None,'plan':{'transition_category':s['transition_category']}}}})
 cfg=rt.load_config(R/'configs/qwen3_1p7b_selective_deletion_formal_reference.yaml');got=aggregate(rs,'deletion',cfg['statistics']);e=audit['deletion']['summary'];assert got['aggregate_explained_fraction']==e['AGGREGATE_EXPLAINED_FRACTION'] and got['explained_fraction_mean']==e['EXPLAINED_FRACTION_MEAN'];assert got['paired_ci']==e['PAIRED_BOOTSTRAP_95CI'] and got['task_stratified_ci']==e['TASK_STRATIFIED_PAIRED_BOOTSTRAP_95CI']
 rs=[{'sample_id':x['canonical_sample_id'],'task':x['task'],'parent_dynamic_score':x['Dynamic'],'intervention_score':x['R2'],'details':{'r2':{'off_score':x['OFF']}}} for x in rt.rows(R/'tests/reference/r2_scalar_results.jsonl')];cfg=rt.load_config(R/'configs/qwen3_1p7b_exact_r2_formal_reference.yaml');got=aggregate(rs,'r2',cfg['statistics']);e=audit['r2']['summary'];assert got['aggregate_gain_retention_vs_off']==e['AGGREGATE_R2_GAIN_RETENTION_VS_OFF'];assert got['equal']==160 and got['gap_n']==13;assert got['paired_bootstrap']['mean_dynamic_minus_r2_95ci']==e['DYNAMIC_R2_PAIRED_BOOTSTRAP_95CI']
 raw=rt.rows(R/'tests/reference/kp_scalar_results.jsonl');by={}
 for x in raw:by.setdefault(x['canonical_sample_id'],{})[x['cell']]=x
 co=rt.rows(next((R/'provenance/original/kp/manifests').glob('*COHORT.jsonl')))
 rs=[{'sample_id':x['canonical_sample_id'],'details':{'kp':{'cells':by[x['canonical_sample_id']]}}} for x in co]
 cfg=rt.load_config(R/'configs/qwen3_1p7b_kp_factorial_formal_reference.yaml');got=aggregate(rs,'kp',cfg['statistics']);e=json.loads((R/'tests/reference/expected/kp/aggregates/KP_PRIMARY_FACTORIAL_ANALYSIS.json').read_text())
 assert got['paired_factorial_bootstrap']==e['paired_bootstrap']
 assert {c:got['closure'][c]['L6'] for c in got['closure']}=={'F00':97,'F10':151,'F01':108,'F11':167}
 print(json.dumps({'PORTABLE_STATISTICS_PARITY':'PASS'}))
if __name__=='__main__':main()
