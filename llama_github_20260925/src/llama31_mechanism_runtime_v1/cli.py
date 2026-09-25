import argparse,json,os,sys,subprocess,time,fcntl
from pathlib import Path
import mechanism as m
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['smoke','formal','status']);p.add_argument('--execute',action='store_true');p.add_argument('--dry-run',action='store_true');p.add_argument('--config',default=str(m.ROOT/'configs/llama31_8b_mechanism_runtime_4k_debug_v1.json'));p.add_argument('--stage2-authority');p.add_argument('--trajectory-authority');p.add_argument('--output-root');a=p.parse_args()
 cfg=m.validate_config(json.loads(Path(a.config).read_text()));out=Path(a.output_root or cfg['output_root'])
 if a.mode=='status':
  if (out/'job.json').exists():
   job=json.loads((out/'job.json').read_text());rows=job['rows'];receipts=list((out/'completed').glob('*.json'));progress=[json.loads(p.read_text()) for p in (out/'status').glob('*.json')];active=[]
   for r in progress:
    if Path('/proc/'+str(r['pid'])).exists() and r.get('state')=='RUNNING':active.append(r['rank'])
   print(json.dumps({'parent_model':job['stage']['model_path'],'trajectory_authority':job['authority'],'eligible_samples':sum(m.reset_plan(r['action_sequence'])['eligible'] for r in rows),'completed_samples':len(receipts),'active_workers':active,'per_mechanism_progress':progress,'errors':[r for r in progress if r.get('state')=='ERROR']},indent=2));return
  records=list(out.glob('rank*/complete.json'));errors=[json.loads(x.read_text()) for x in out.glob('rank*/error.json')];active=[]
  for f in out.glob('rank*/loaded.json'):
   r=json.loads(f.read_text())
   if Path('/proc/'+str(r['pid'])).exists():active.append(r['rank'])
  print(json.dumps({'parent_model':cfg['model_path'],'trajectory_authority':a.trajectory_authority,'eligible_samples':None,'completed_samples':len(records),'active_workers':active,'per_mechanism_progress':{str(x.relative_to(out)):json.loads(x.read_text()).get('status','RECORDED') for x in out.glob('rank*/*.json')},'errors':errors},indent=2));return
 if a.mode=='formal':
  stage,auth,rows=m.handoff(a.stage2_authority,a.trajectory_authority,a.execute)
  # Future formal execution uses the same schema-driven backend and separate output root.
  from formal import execute
  if not a.output_root:raise ValueError('FORMAL_OUTPUT_ROOT_REQUIRED')
  execute(cfg,stage,auth,rows,out);return
 if not a.execute:
  print(json.dumps({'status':'DRY_RUN_PASS','source_root':str(m.ROOT),'config':str(Path(a.config).resolve()),'physical_gpu_allowlist':[0,1,2,3],'workers':4,'DEBUG_ONLY':True,'PAPER_RESULT':False}));return
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='0,1,2,3':raise ValueError('GPU_POLICY')
 if cfg.get('DEBUG_ONLY') is not True or cfg.get('PAPER_RESULT') is not False:raise ValueError('DEBUG_CONFIG_REQUIRED')
 locks=[]
 for rank in range(4):
  q=Path('/path/to/ttt/work/llama31_mechanism_runtime_v1')/f'gpu{rank}.lock';f=q.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 gpu=subprocess.check_output(['nvidia-smi','-i','0,1,2,3','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
 if any(float(s.split(',')[-1])<44000 for s in gpu.strip().splitlines()):raise ValueError('GPU_RESOURCE_UNAVAILABLE')
 out.mkdir(parents=True,exist_ok=True);m.atomic(out/'gpu_before.json',{'csv':gpu});workers=[];logs=[];start=time.time()
 for rank in range(4):
  d=out/f'rank{rank}';d.mkdir(exist_ok=True)
  if (d/'complete.json').exists():raise ValueError('OUTPUT_ALREADY_COMPLETE_USE_FRESH_ROOT')
  log=(d/'worker.log').open('w');logs.append(log);workers.append(subprocess.Popen([sys.executable,'-B',str(m.ROOT/'worker.py'),str(rank),a.config,str(d)],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1'}))
 m.atomic(out/'workers.json',{'pids':[w.pid for w in workers],'source_root':str(m.ROOT),'started':start})
 while any(w.poll() is None for w in workers):
  with (out/'telemetry.jsonl').open('a') as f:
   csv=subprocess.check_output(['nvidia-smi','-i','0,1,2,3','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True);f.write(json.dumps({'time':time.time(),'csv':csv})+'\n')
  time.sleep(5)
 codes=[w.wait() for w in workers]
 for log in logs:log.close()
 result={'status':'PASS' if codes==[0]*4 else 'FAIL','exit_codes':codes,'wall_seconds':time.time()-start,'run_root':str(out),'DEBUG_ONLY':True,'PAPER_RESULT':False};m.atomic(out/'execution.json',result);print(json.dumps(result));assert result['status']=='PASS'
if __name__=='__main__':main()
