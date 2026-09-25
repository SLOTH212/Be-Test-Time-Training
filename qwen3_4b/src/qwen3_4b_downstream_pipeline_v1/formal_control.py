#!/usr/bin/env python3
"""Explicit launch/finalize/status; all formal parent chains fail closed."""
import os,sys,json,time,ast,types,statistics,collections,subprocess,argparse,fcntl,hashlib
from pathlib import Path
import contracts as ct
sys.path.insert(0,str(ct.M));import runtime as rt
from worker import Store,atomic
FLAGS=dict(executor_version='V2',DEBUG_ONLY=False,DEBUG_INTERFACE_FIXTURE=False,PAPER_RESULT=True,FORMAL_SCIENTIFIC_RESULT=True)
def fixed_aggregate(rows):
 ns={};exec(compile((ct.I/'runtime.py').read_text(),str(ct.I/'runtime.py'),'exec'),ns)
 p=ct.F/'fixed/aggregate_fixed.py';t=ast.parse(p.read_text());ns.update(defaultdict=collections.defaultdict,statistics=statistics);exec(compile(ast.Module(body=[x for x in t.body if isinstance(x,ast.FunctionDef) and x.name=='aggregate'],type_ignores=[]),str(p),'exec'),ns);return ns['aggregate'](rows,ct.LAYERS)
def default_root(phase):return ct.R/'runs'/('qwen3_4b_ruler32k_'+phase+'_4k_'+('executor_v2' if phase in ['fixed','dynamic'] else 'formal_v1'))
def sources():ct.verify_runtime();ct.benchmark();return ct.source_guard()
def prepare(args):
 phase=args.phase;config=ct.validate_config(ct.read(ct.config_path(phase)),phase);source=sources();idx=ct.benchmark();fixed=None;cohort=[];improved_n=0
 if phase=='fixed':
  ct.require(args.parent_authority is not None,'PARENT_AUTHORITY_REQUIRED');parent=ct.under(args.parent_authority);pa=ct.parent(parent);ids=[x['project_sample_id'] for x in idx];upstream=parent
 elif phase=='dynamic':
  ct.require(args.fixed_authority is not None,'FIXED_FINAL_AUTHORITY_REQUIRED');upstream=ct.under(args.fixed_authority);fixed,records=ct.final_authority(upstream,'fixed');parent=Path(fixed['model_parent_authority']);pa=ct.parent(parent);agg=ct.read(fixed['aggregate_path']);ids=[r['sample_id'] for r in agg['per_sample'] if r['task']!='ruler_cwe_16k' and abs(r['sample_best']-1)>1e-12]
 else:
  ct.require(args.dynamic_authority is not None,'DYNAMIC_FINAL_AUTHORITY_REQUIRED');upstream=ct.under(args.dynamic_authority);dynamic,dr=ct.final_authority(upstream,'dynamic');fixed,fr=ct.final_authority(dynamic['fixed_authority'],'fixed');parent=Path(fixed['model_parent_authority']);pa=ct.parent(parent);agg=ct.read(fixed['aggregate_path']);by={r['sample_id']:{a:r[a] for a in ct.ACTIONS} for r in agg['per_sample']};cohort=rt.improved_cohort(by,{r['sample_id']:r for r in dr},ct.LAYERS);order={x['project_sample_id']:x['global_index'] for x in idx}
  for r in cohort:r['benchmark_global_index']=order[r['sample_id']]
  cohort.sort(key=lambda r:r['benchmark_global_index']);improved_n=len(cohort)
  if args.experiment in ['reset','deletion','kp']:cohort=[r for r in cohort if rt.reset_screen(r['action_sequence'],ct.LAYERS)['eligible']]
  if args.experiment=='deletion':
   cohort=[r for r in cohort if rt.deletion_plan(r['action_sequence'],rt.reset_screen(r['action_sequence'],ct.LAYERS)['boundary'],ct.LAYERS)['eligible']]
   if cohort:
    rr=ct.under(args.run_root or default_root(phase))/'reset';reset={ct.committed(p)['sample_id']:ct.committed(p) for p in (rr/'samples').glob('*.json')}
    for r in cohort:
     q=reset[r['sample_id']];ct.require(q['parent_dynamic_trajectory']==r['action_sequence'] and q['model_identity']==pa['files']['model.safetensors'],'MATCHED_RESET_PARENT');r['full_reset_score']=q['intervention_score']
  ids=[r['sample_id'] for r in cohort]
 root=ct.under(args.run_root or default_root(phase));ct.require(root.is_relative_to((ct.R/'runs').resolve()),'FORMAL_RUN_ROOT_REQUIRED')
 if phase=='mechanism':root=root/args.experiment
 identity=dict(config=config,config_sha256=ct.sha(ct.config_path(phase)),upstream_authority=str(upstream),upstream_sha256=ct.sha(upstream),model_parent_authority=str(parent),model_parent_sha256=ct.sha(parent),source_manifest_sha256=source,sample_ids=ids,cohort=cohort,improved_cohort_n=improved_n,**FLAGS)
 job=dict(phase=phase,experiment=args.experiment,config=config,config_sha256=identity['config_sha256'],model_parent_authority=str(parent),run_root=str(root),sample_ids=ids,cohort=cohort,store_identity=identity,upstream_authority=str(upstream),execution_authorized=True,DEBUG_ONLY=False)
 if fixed:job.update(fixed_aggregate=fixed['aggregate_path'],fixed_commit_paths=list(fixed['commits']))
 ct.ready_guard(phase)
 return job

def launch(args):
 if args.phase=='mechanism' and args.experiment=='all':
  result=[]
  for ex in ['reverse','reset','deletion','kp','r2']:
   args.experiment=ex;result.append(launch(args))
  return result
 job=prepare(args);plan=dict(status='VALIDATED_NOT_EXECUTED',phase=args.phase,experiment=args.experiment,sample_n=len(job['sample_ids']),run_root=job['run_root'],zero_eligible_reason='ZERO_ELIGIBLE_BY_CHUNK_GEOMETRY' if args.phase=='mechanism' and args.experiment in ['reset','kp','deletion'] and not job['sample_ids'] else None)
 if not args.execute:print(json.dumps(plan));return plan
 root=Path(job['run_root']);store=Store(root,job['store_identity']);atomic(root/'JOB.json',job)
 if not job['sample_ids']:
  atomic(root/'PHASE_STATUS.json',{**plan,'status':'COMPLETE_EMPTY_COHORT','scientific_effect_estimate':None,**FLAGS});return plan
 lockroot=ct.R/'work/downstream_gpu_ownership';lockroot.mkdir(exist_ok=True);locks=[]
 for gpu in [2,3]:
  f=(lockroot/f'physical_gpu{gpu}.lock').open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='2,3',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',TTT_INFERENCE_RUNTIME_ROOT=str(ct.F),TTT_INFERENCE_RUNTIME_ARCHIVE=str(ct.R/'shared/packages/ttt_inference_runtime_v1/ttt_inference_runtime_v1.tar.gz'))
 started=time.time();ps=[];streams=[]
 for w in [0,1]:
  f=(root/f'worker{w}.log').open('a');streams.append(f);ps.append(subprocess.Popen([sys.executable,str(ct.S/'formal_worker.py'),'--job',str(root/'JOB.json'),'--worker-id',str(w)],stdout=f,stderr=subprocess.STDOUT,env=env))
 atomic(root/'PHASE_STATUS.json',dict(status='RUNNING',pids=[p.pid for p in ps],started=started,phase=args.phase,experiment=args.experiment,**FLAGS));codes=[p.wait() for p in ps];atomic(root/'PHASE_STATUS.json',dict(status='COMPLETE' if codes==[0,0] else 'FAILED',exit_codes=codes,elapsed=time.time()-started,**FLAGS))
 ct.require(codes==[0,0],'WORKER_FAILURE_NO_GPU_FALLBACK');atomic(root/('FIXED8_COMPLETED_AUTHORITY.json' if args.phase=='fixed' else 'COMPLETED_AUTHORITY.json'),dict(schema='COMPLETED_PHASE_AUTHORITY_V1',status='PASS',phase=args.phase,run_root=str(root),job_sha256=ct.sha(root/'JOB.json'),**FLAGS));return plan

def finalize(args):
 phase=args.phase;ct.require(phase in ['fixed','dynamic'],'SELECT_FIXED_OR_DYNAMIC');completed=ct.read(args.fixed_authority) if phase=='fixed' and args.fixed_authority else None
 if completed:ct.require(completed.get('schema')=='COMPLETED_PHASE_AUTHORITY_V1' and completed.get('status')=='PASS' and completed.get('DEBUG_ONLY') is False,'COMPLETED_FIXED_AUTHORITY_REQUIRED')
 root=ct.under(completed['run_root'] if completed else (args.run_root or default_root(phase)));job=ct.read(root/'JOB.json')
 if completed:ct.require(ct.sha(root/'JOB.json')==completed['job_sha256'],'COMPLETED_JOB_HASH')
 ct.require(job['DEBUG_ONLY'] is False,'DEBUG_JOB_FORBIDDEN');source=ct.source_guard();ct.require(source==job['store_identity']['source_manifest_sha256'],'SOURCE_CHANGED');store=Store(root,job['store_identity']);rs=[];commits={}
 for sid in job['sample_ids']:
  for key in ([sid+'|'+a for a in ct.ACTIONS] if phase=='fixed' else [sid]):
   r=store.committed(key);ct.require(r is not None,'INCOMPLETE_RESULT');rs.append(r);commits[str(store.path(key))]=ct.sha(store.path(key))
 ct.require(len(list((root/'samples').glob('*.json')))==len(rs),'EXTRA_RESULT')
 base=dict(schema=phase.upper()+'_FINAL_AUTHORITY_V1',status='PASS',actions=ct.ACTIONS,layers=ct.LAYERS,chunk_size=4096,context_length=32768,benchmark_identity=ct.BH,model_parent_authority=job['model_parent_authority'],model_identity=ct.read(job['model_parent_authority'])['files']['model.safetensors'],source_manifest_sha256=source,commits=commits,files={},**FLAGS)
 if phase=='fixed':
  ct.require(len(rs)==52000,'FIXED8_FORMAL_REQUIRES_52000_CELLS');agg=fixed_aggregate([{**r,'sample_id':r['benchmark_sample_id'],'status':'SUCCESS'} for r in rs]);ct.require(agg['sample_count']==6500,'FIXED_SAMPLE_COUNT');base['sample_n']=6500
 else:
  fa,fr=ct.final_authority(job['upstream_authority'],'fixed');fixed=ct.read(fa['aggregate_path']);p=ct.F/'dynamic/aggregate_dynamic.py';ns=dict(json=json,statistics=statistics,sys=sys,Path=Path,np=__import__('numpy'),TOL=1e-12,dynamic_required=lambda x:abs(float(x)-1)>1e-12,fixed_actions=lambda _:ct.ACTIONS);node=next(x for x in ast.parse(p.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='aggregate');exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),ns);agg=ns['aggregate'](fixed,root/'samples',ct.LAYERS,{'ruler_cwe_16k'},draws=10000,seed=20260828)
  base.update(population_n=agg['sample_count'],searched_n=agg['searched_count'],skipped_n=agg['ceiling_count'],fixed_authority=job['upstream_authority']);ct.require(base['population_n']==6000,'DYNAMIC_POPULATION')
  agg['best_single_action']=fixed['best_single_action'];agg['best_single_mean_on_12_tasks']=statistics.fmean(r[fixed['best_single_action']] for r in fixed['per_sample'] if r['task']!='ruler_cwe_16k')
  by={r['sample_id']:{a:r[a] for a in ct.ACTIONS} for r in fixed['per_sample']};improved=rt.improved_cohort(by,{r['sample_id']:r for r in rs},ct.LAYERS);cohort=dict(improved=improved,equal=[r['sample_id'] for r in rs if abs(r['dynamic_score']-r['sample_best'])<=1e-12],worse=[r['sample_id'] for r in rs if r['dynamic_score']<r['sample_best']-1e-12]);atomic(root/'COHORTS.json',{**cohort,**FLAGS});base['files'][str(root/'COHORTS.json')]=ct.sha(root/'COHORTS.json')
 out=root/(phase.upper()+'_AGGREGATE.json');atomic(out,{**agg,**FLAGS});base['aggregate_path']=str(out);base['files'][str(out)]=ct.sha(out);atomic(root/('FIXED8_FINAL_AUTHORITY.json' if phase=='fixed' else 'DYNAMIC_FINAL_AUTHORITY.json'),base);print(json.dumps({'status':'PASS','authority':str(root/('FIXED8_FINAL_AUTHORITY.json' if phase=='fixed' else 'DYNAMIC_FINAL_AUTHORITY.json'))}))

def status(args):
 root=ct.under(args.run_root or default_root(args.phase));roots=[root/x for x in ['reverse','reset','deletion','kp','r2']] if args.phase=='mechanism' else [root];out=[]
 for r in roots:
  j=ct.read(r/'JOB.json') if (r/'JOB.json').exists() else {};state=ct.read(r/'PHASE_STATUS.json') if (r/'PHASE_STATUS.json').exists() else {'status':'NOT_STARTED'};completed=len(list((r/'samples').glob('*.json')));errors=[]
  for p in r.glob('worker*.log'):
   lines=p.read_text(errors='replace').splitlines();errors.extend([l for l in lines[-60:] if any(x in l for x in ['Error','Traceback','FAILED'])])
  pids=state.get('pids',[]);live={str(p):Path('/proc',str(p)).exists() for p in pids};out.append(dict(root=str(r),experiment=r.name,parent_authority=j.get('upstream_authority'),cohort_n=j.get('store_identity',{}).get('improved_cohort_n',len(j.get('cohort',[]))),eligible_n=len(j.get('sample_ids',[])),completed_n=completed,errors=errors,worker_status=live,state=state,ETA=None))
 gpu=subprocess.run(['nvidia-smi','-i','2,3','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True);print(json.dumps(dict(experiments=out,gpu_status=gpu.stdout),indent=2))

def preflight(args):
 source=sources()
 for phase in ['fixed','dynamic','mechanism']:ct.validate_config(ct.read(ct.config_path(phase)),phase)
 ct.require(args.parent_authority is not None,'PARENT_AUTHORITY_REQUIRED');ct.parent(args.parent_authority,debug=args.debug_test);print(json.dumps(dict(status='PASS_STATIC_PREFLIGHT',parent_class='DEBUG_ONLY' if args.debug_test else 'FORMAL_STAGE2_FINAL',source_manifest_sha256=source,GPU_policy=[2,3],inference_executed=False,formal_reset_eligibility='ZERO_ELIGIBLE_BY_CHUNK_GEOMETRY')))

def dispatch(args):
 ct.require(args.through in ['fixed','dynamic','mechanism'],'EXPLICIT_THROUGH_REQUIRED');ct.require(args.parent_authority is not None,'PARENT_AUTHORITY_REQUIRED')
 if not args.execute:
  args.phase='fixed';prepare(args);print(json.dumps(dict(status='PLAN_ONLY',through=args.through,phases=['Fixed8','Fixed finalizer','Dynamic LB','Dynamic finalizer','Mechanism'],requires_execute_flag=True)));return
 for phase in ['fixed','dynamic','mechanism']:
  args.phase=phase;args.run_root=None
  if phase=='dynamic':args.fixed_authority=str(default_root('fixed')/'FIXED8_FINAL_AUTHORITY.json')
  if phase=='mechanism':args.dynamic_authority=str(default_root('dynamic')/'DYNAMIC_FINAL_AUTHORITY.json');args.experiment='all'
  launch(args)
  if phase!='mechanism':finalize(args)
  if args.through==phase:break

def main():
 ap=argparse.ArgumentParser();ap.add_argument('command',choices=['launch','finalize','status','preflight','dispatch']);ap.add_argument('--phase',choices=['fixed','dynamic','mechanism'],default='fixed');ap.add_argument('--parent-authority');ap.add_argument('--fixed-authority');ap.add_argument('--dynamic-authority');ap.add_argument('--run-root');ap.add_argument('--experiment',choices=['reverse','reset','deletion','kp','r2','all'],default='all');ap.add_argument('--through',choices=['fixed','dynamic','mechanism']);ap.add_argument('--execute',action='store_true');ap.add_argument('--debug-test',action='store_true');a=ap.parse_args();ct.require(not a.debug_test or a.command=='preflight','DEBUG_ONLY_ALLOWED_IN_PREFLIGHT');globals()[a.command](a)
if __name__=='__main__':main()
