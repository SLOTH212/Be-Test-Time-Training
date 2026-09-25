#!/usr/bin/env python3
import csv, hashlib, json, math, os, statistics, sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ACTIONS = ['OFF','L0','L6','L12','L18','L24','ALL']
EXCLUDED = 'ruler_cwe_16k'
FIXED = Path('/home/USER/ttt/runs/formal/fixed7_1p7b_v1/run_20260828T012948+0800')
BENCH = Path('/home/USER/ttt/benchmarks/ruler_16k_standard_13task500_v1')
MODEL = Path('/home/USER/ttt/models/qwen3_1p7b_stage2')
SOURCE = Path('/home/USER/ttt_phase_c_migrated_v1/ttt_phase_c_migration_20260813/new_host_runtime/execution_copy/code/dynamic_runner.py')
SCIENTIFIC = Path('/home/USER/ttt_phase_c_migrated_v1/ttt_phase_c_migration_20260813/phase_c_run/config/PHASE_C_SCIENTIFIC_CONFIG.json')
EXPECTED_MODEL = 'ca51ee90d20ee09efbe0b31dd21bf9faa70275508316d1ba7cb1387ee4f0725f'
EXPECTED_CKPT = 'f4e264ed6f5671f123d6c8a27c69bce3af5ffcba518c9e298398314c6147d9fd'
TOL = 1e-12

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def atomic_text(path,text):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name('.'+path.name+'.tmp')
    with open(tmp,'w',encoding='utf-8') as f: f.write(text); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)
def atomic_json(path,obj): atomic_text(path,json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n')
def write_csv(path,rows,fields):
    path=Path(path); tmp=path.with_name('.'+path.name+'.tmp')
    with open(tmp,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def load_jsonl(path):
    with open(path,encoding='utf-8') as f:return [json.loads(x) for x in f if x.strip()]

def main():
    if len(sys.argv)!=2: raise SystemExit('usage: prepare_dynamic_nonceiling_12task_v1.py RUN_ROOT')
    run=Path(sys.argv[1]).resolve()
    if run.exists() and any(run.iterdir()): raise RuntimeError('REFUSE_NONEMPTY_RUN_ROOT')
    for d in ['preregistration','fixed_authority','eligibility','config','status','results','per_task','logs','checkpoints','aggregation','bootstrap','reports','audits','final','preflight']:
        (run/d).mkdir(parents=True,exist_ok=True)
    audit=json.load(open(FIXED/'audits/FINAL_FIXED7_FORMAL_AUDIT.json'))
    if not (audit.get('complete') and audit.get('completed_cells')==45500 and audit.get('failed_cells')==0 and audit.get('mode_counts')=={a:6500 for a in ACTIONS}):
        raise RuntimeError('FIXED7_COMPLETION_GATE')
    benchmark=json.load(open(BENCH/'benchmark_manifest.json'))
    tasks=benchmark['task_names']; included=[t for t in tasks if t!=EXCLUDED]
    if len(tasks)!=13 or len(included)!=12: raise RuntimeError('TASK_INVENTORY')
    bench_rows={}
    for task in tasks:
        rows=load_jsonl(BENCH/'samples'/f'{task}.jsonl')
        if len(rows)!=500: raise RuntimeError(f'BENCH_TASK_COUNT {task}')
        for r in rows:
            if r['sample_id'] in bench_rows or r['task']!=task: raise RuntimeError('BENCH_ALIGNMENT')
            bench_rows[r['sample_id']]=r
    modes={}
    for action in ACTIONS:
        rows=load_jsonl(FIXED/'modes'/action/'results.jsonl')
        by={}
        for r in rows:
            sid=r.get('sample_id');score=r.get('score')
            if sid in by or sid not in bench_rows or r.get('task')!=bench_rows[sid]['task'] or r.get('mode')!=action or r.get('status')!='SUCCESS': raise RuntimeError(f'FIXED_ROW_INVALID {action} {sid}')
            if not isinstance(score,(int,float)) or not math.isfinite(score) or score < -TOL or score > 1+TOL: raise RuntimeError('MALFORMED_SCORE')
            if r.get('model_hash')!=EXPECTED_MODEL or r.get('checkpoint_hash')!=EXPECTED_CKPT: raise RuntimeError('MODEL_IDENTITY_MISMATCH')
            by[sid]=r
        if len(by)!=6500 or set(by)!=set(bench_rows): raise RuntimeError(f'FIXED_MODE_SET {action}')
        modes[action]=by
    fields=['sample_id','task',*ACTIONS,'sample_best','best_action_set','best_action_count','score_ceiling','is_cwe','is_ceiling','dynamic_eligible','exclusion_reason']
    full=[]
    for sid,b in sorted(bench_rows.items(),key=lambda kv:(tasks.index(kv[1]['task']),kv[1].get('sample_index',0),kv[0])):
        scores={a:float(modes[a][sid]['score']) for a in ACTIONS}; sb=max(scores.values()); winners=[a for a in ACTIONS if abs(scores[a]-sb)<=TOL]
        if any(sb+TOL<s for s in scores.values()): raise RuntimeError('ORACLE_MONOTONICITY')
        cwe=b['task']==EXCLUDED; ceiling=abs(sb-1.0)<=TOL; eligible=not cwe and not ceiling
        reason='' if eligible else ('CWE_TASK_EXCLUDED' if cwe else 'SAMPLE_BEST_AT_CEILING')
        full.append({'sample_id':sid,'task':b['task'],**{a:repr(scores[a]) for a in ACTIONS},'sample_best':repr(sb),
          'best_action_set':'|'.join(winners),'best_action_count':len(winners),'score_ceiling':'1.0','is_cwe':str(cwe).lower(),
          'is_ceiling':str(ceiling).lower(),'dynamic_eligible':str(eligible).lower(),'exclusion_reason':reason})
    aggregate=sum(float(r['sample_best']) for r in full)/6500
    expected=float(json.load(open(FIXED/'aggregation/fixed7_summary.json'))['SAMPLE_BEST_SCORE'])
    if abs(aggregate-expected)>TOL: raise RuntimeError(f'SAMPLEBEST_RECOMPUTE {aggregate} {expected}')
    eligible=[r for r in full if r['dynamic_eligible']=='true']; skipped=[r for r in full if r['is_cwe']=='false' and r['is_ceiling']=='true']
    if sum(r['is_cwe']=='true' for r in full)!=500 or len(eligible)+len(skipped)!=6000: raise RuntimeError('ELIGIBILITY_PARTITION')
    full_path=run/'eligibility/fixed7_sample_best_full6500.csv'; eligible_path=run/'eligibility/dynamic_eligible_nonceiling_12task.csv'; skipped_path=run/'eligibility/dynamic_skipped_ceiling_12task.csv'
    write_csv(full_path,full,fields);write_csv(eligible_path,eligible,fields);write_csv(skipped_path,skipped,fields)
    per_task=[]
    for task in included:
        rr=[r for r in full if r['task']==task]; nc=sum(r['dynamic_eligible']=='true' for r in rr)
        per_task.append({'task':task,'total_count':500,'ceiling_count':500-nc,'nonceiling_count':nc,'nonceiling_prevalence':nc/500,
          'mean_off':sum(float(r['OFF']) for r in rr)/500,'mean_sample_best':sum(float(r['sample_best']) for r in rr)/500})
    write_csv(run/'eligibility/eligibility_by_task.csv',per_task,list(per_task[0]))
    noncwe=[r for r in full if r['is_cwe']=='false']; gains=[float(r['sample_best'])-float(r['OFF']) for r in noncwe]
    unique=Counter(); ties=0
    for r in noncwe:
        ws=r['best_action_set'].split('|')
        if len(ws)==1: unique[ws[0]]+=1
        else:ties+=1
    winner='\n'.join(['# Sample Best winner distribution','',*[f'- Unique {a}: {unique[a]}' for a in ACTIONS],f'- Multi-action winner sets: {ties}',f'- Total: {len(noncwe)}',''])
    atomic_text(run/'eligibility/SAMPLE_BEST_WINNER_DISTRIBUTION.md',winner)
    stats={'total_fixed_sample_count':6500,'cwe_excluded_count':500,'included_12task_count':6000,'samplebest_ceiling_count_12task':len(skipped),
      'samplebest_nonceiling_count_12task':len(eligible),'nonceiling_prevalence_12task':len(eligible)/6000,
      'sample_best_13task_recomputed':aggregate,'sample_best_12task_score':sum(float(r['sample_best']) for r in noncwe)/6000,
      'off_12task_score':sum(float(r['OFF']) for r in noncwe)/6000,'samplebest_greater_than_off_count':sum(g>TOL for g in gains),
      'samplebest_equal_off_count':sum(abs(g)<=TOL for g in gains),'samplebest_less_than_off_count':sum(g<-TOL for g in gains),
      'median_gain':statistics.median(gains),'positive_only_mean_gain':statistics.mean([g for g in gains if g>TOL]) if any(g>TOL for g in gains) else 0,
      'maximum_gain':max(gains),'unique_winner_counts':dict(unique),'multi_action_winner_set_count':ties,'per_task':per_task}
    atomic_json(run/'eligibility/ELIGIBILITY_AUDIT.json',stats)
    config={'version':'dynamic_nonceiling_12task_v1','dynamic_language':'DYNAMIC_RECOMPUTED_LOWER_BOUND','actions':ACTIONS,'layers':[0,6,12,18,24],
      'search_algorithm':'seven_constant_seeds_then_terminal_score_beam_coordinate_search','beam_size':4,'sweeps':1,'position_order':'forward',
      'max_replays_15chunks':385,'context_length':16384,'chunk_size':1024,'workers':1,'cpu_affinity':'8-23','reset_per_sample':True,
      'recompute_downstream_updates':True,'stored_delta_stitching':False,'generation_update':False,'incomplete_tail_update':False,
      'model_path':str(MODEL),'model_identity':EXPECTED_MODEL,'checkpoint_identity':EXPECTED_CKPT,'benchmark_root':str(BENCH),'fixed_run_root':str(FIXED),
      'dynamic_source_authority':str(SOURCE),'dynamic_source_sha256':sha(SOURCE),'scientific_config_authority':str(SCIENTIFIC),'scientific_config_sha256':sha(SCIENTIFIC)}
    atomic_json(run/'config/DYNAMIC_FORMAL_CONFIG.json',config)
    prereg={'version':'DYNAMIC_NONCEILING_12TASK_ELIGIBILITY_V1','created_at':datetime.now().astimezone().isoformat(),
      'fixed7_run_identity':str(FIXED),'fixed7_integrity':'PASS_45500_OF_45500','model_identity':EXPECTED_MODEL,'checkpoint_identity':EXPECTED_CKPT,
      'benchmark_identity':sha(BENCH/'benchmark_manifest.json'),'excluded_task':EXCLUDED,'included_tasks':included,
      'sample_best_definition':'PER_SAMPLE_MAX_OVER_ALL_SEVEN_FIXED_ACTIONS','ceiling_rule':'abs(sample_best-1.0)<=1e-12',
      'candidate_sample_ids':[r['sample_id'] for r in eligible],'candidate_count':len(eligible),'skipped_ceiling_ids':[r['sample_id'] for r in skipped],
      'eligibility_files':{str(p.relative_to(run)):sha(p) for p in [full_path,eligible_path,skipped_path]},'dynamic_protocol':config,
      'environment':'/home/USER/conda_envs/ttt_phase_c_v1','cpu_affinity':'8-23','worker_count':1,'fixed_sample_best_recomputation':'PASS'}
    prereg_path=run/'preregistration/DYNAMIC_NONCEILING_12TASK_ELIGIBILITY_V1.json';atomic_json(prereg_path,prereg)
    atomic_text(prereg_path.with_suffix('.json.sha256'),f'{sha(prereg_path)}  {prereg_path.name}\n')
    fixed_audit={'status':'PASS','pipeline_status':'COMPLETE','modes_complete':'7/7','completed_cells':'45500/45500','error_count':0,
      'mode_counts':{a:6500 for a in ACTIONS},'unique_sample_ids_per_mode':6500,'fixed_sample_best_recomputation':'PASS','oracle_monotonicity':'PASS'}
    atomic_json(run/'fixed_authority/FIXED7_COMPLETION_AND_ALIGNMENT_AUDIT.json',fixed_audit)
    status={'pipeline_status':'PREREGISTERED','current_phase':'PREFLIGHT','total_fixed_samples':6500,'cwe_excluded_count':500,
      'included_12task_count':6000,'ceiling_skipped_count':len(skipped),'dynamic_eligible_count':len(eligible),'dynamic_completed_count':0,
      'dynamic_remaining_count':len(eligible),'current_task':None,'current_sample_id':None,'start_timestamp':None,'last_progress_timestamp':None,
      'recent_rate_samples_per_hour':0,'estimated_remaining_hours':None,'controller_pid':None,'worker_pid':None,'worker_running':False,
      'gpu_utilization':None,'gpu_memory_used':None,'error_count':0,'last_error':'NONE'}
    atomic_json(run/'status/pipeline_status.json',status)
    print(json.dumps({'status':'PASS','run_root':str(run),'eligible':len(eligible),'ceiling':len(skipped),'sample_best_12task':stats['sample_best_12task_score'],'off_12task':stats['off_12task_score'],'prereg_sha256':sha(prereg_path)},sort_keys=True))

if __name__=='__main__': main()
