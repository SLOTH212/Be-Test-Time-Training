import os,sys,json,time,hashlib,subprocess
from pathlib import Path
R=Path('/path/to/ttt');W=R/'work/qwen3_4b_stage1_resume_v3';A=R/'audits/v3';D=A/'analysis'
plan=json.loads((A/'CPU_PARALLEL_PARTITIONS.json').read_text())
assert hashlib.sha256((W/'v3_metrics_parallel_worker.py').read_bytes()).hexdigest()==plan['worker_sha256']
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='8',PYTHONDONTWRITEBYTECODE='1')
receipts=[];running=[]
for i in range(3):
 e=dict(env,V3_CPU_PARTITION=str(i));log=R/'logs'/('qwen3_4b_stage1_resume_v3_cpu_part%d.log'%i);f=log.open('x')
 p=subprocess.Popen([sys.executable,'-B',str(W/'v3_metrics_parallel_worker.py')],env=e,cwd=R,stdout=f,stderr=subprocess.STDOUT)
 item={'partition':i,'pid':p.pid,'started_at':time.time(),'log':str(log)};receipts.append(item);running.append((p,f,item))
print('THREE_CPU_WORKERS_STARTED',flush=True)
for p,f,item in running:
 p.wait();f.close();item.update(returncode=p.returncode,ended_at=time.time());(A/'CPU_PARALLEL_RECEIPTS.json').write_text(json.dumps(receipts,indent=2)+'\n')
assert all(x['returncode']==0 for x in receipts),receipts
prefix=(D/'ALL_TENSOR_PAIR_METRICS.jsonl').read_bytes();assert hashlib.sha256(prefix).hexdigest()==plan['preserved_prefix_sha256']
rows=[json.loads(x) for x in prefix.splitlines()]
for i in range(3):
 part=[json.loads(x) for x in (D/('PARALLEL_PART_%d.jsonl'%i)).read_text().splitlines()]
 assert {x['key'] for x in part}==set(plan['partitions'][i]) and len(part)==len(plan['partitions'][i]);rows.extend(part)
keys=sorted(json.loads((R/'runs/qwen3_4b_stage1_resume_v3/attempt2/U1/final_rank0.json').read_text())['tensors'])
assert len(rows)==len(keys)==1647 and len({x['key'] for x in rows})==1647 and {x['key'] for x in rows}==set(keys)
rows.sort(key=lambda x:x['key']);assert sum(x['primary76'] for x in rows)==76
suffix=b''.join((json.dumps(x,allow_nan=False)+'\n').encode() for x in rows[plan['preserved_rows']:])
temp=D/'ALL_TENSOR_PAIR_METRICS_PARALLEL_MERGED.tmp';temp.write_bytes(prefix+suffix);temp.replace(D/'ALL_TENSOR_PAIR_METRICS.jsonl')
sha=hashlib.sha256(prefix+suffix).hexdigest()
(D/'PAIR_METRICS_COMPLETE.json').write_text(json.dumps({'status':'PASS','tensor_count':1647,'primary_count':76,'pairs':45,'names':['U1','U2','U3','U4','U5','R1','R2','R3','R4','R5'],'completed_at':time.time(),'metrics_sha256':sha},indent=2)+'\n')
(A/'CPU_PARALLEL_MERGE_AUDIT.json').write_text(json.dumps({'status':'PASS','all_keys_exactly_once':True,'original_order_preserved':True,'prefix_unchanged':True,'metrics_sha256':sha,'partition_plan_sha256':hashlib.sha256((A/'CPU_PARALLEL_PARTITIONS.json').read_bytes()).hexdigest(),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
print('FULL_METRICS_PARALLEL_MERGED_PASS',time.time(),flush=True)
for script in ['v3_statistics.py','v3_reporting_evidence.py','v3_independent_auditor.py','v3_reporting_independent_auditor.py']:
 item={'script':script,'source_sha256':hashlib.sha256((W/script).read_bytes()).hexdigest(),'started_at':time.time()};print('CPU_STAGE_START',script,item['started_at'],flush=True)
 with (R/'logs'/('qwen3_4b_stage1_resume_v3_'+script+'.log')).open('x') as f:
  p=subprocess.Popen([sys.executable,'-B',str(W/script)],env=env,cwd=R,stdout=f,stderr=subprocess.STDOUT);item['pid']=p.pid;p.wait()
 item.update(returncode=p.returncode,ended_at=time.time());receipts.append(item);(A/'CPU_PARALLEL_RECEIPTS.json').write_text(json.dumps(receipts,indent=2)+'\n')
 assert p.returncode==0,(script,p.returncode)
 print('CPU_STAGE_PASS',script,item['ended_at'],flush=True)
print('CPU_PIPELINE_COMPLETE',flush=True)
