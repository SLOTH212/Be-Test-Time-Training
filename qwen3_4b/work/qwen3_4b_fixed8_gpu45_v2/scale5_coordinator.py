import pathlib,sys,os,json,time,subprocess,fcntl,types,traceback
R=pathlib.Path('/path/to/ttt');W=R/'work/qwen3_4b_fixed8_gpu45_v2';r=R/'runs/qwen3_4b_ruler32k_fixed_4k_executor_v2_gpu45'
setup=(W/'entry.py').read_text().split("worker='--job' in sys.argv")[0];exec(compile(setup,str(W/'entry.py'),'exec'),globals())
import formal_control as fc
from worker import Store,atomic
locks=[]
for gpu in [0,2,3,4,5]:
 f=(R/'work/downstream_gpu_ownership'/('physical_gpu%d.lock'%gpu)).open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
j=ct.read(r/'JOB.json');ct.source_guard();ct.ready_guard('fixed');ct.parent(j['model_parent_authority']);store=Store(r,j['store_identity']);ids=j['sample_ids'];parts=[set(ids[i::5]) for i in range(5)];assert sum(map(len,parts))==len(set.union(*parts))==6500
existing=0
for sid in ids:
 for action in ct.ACTIONS:
  if store.committed(sid+'|'+action) is not None:existing+=1
receipt={'status':'STARTING','started':time.time(),'physical_gpu_allowlist':[0,2,3,4,5],'workers':5,'original_config_identity_retained':True,'resource_override_only':True,'model_parent_authority_sha256':ct.sha(j['model_parent_authority']),'source_manifest_sha256':ct.source_guard(),'worker_overlay_sha256':ct.sha(W/'scale5_worker.py'),'coordinator_sha256':ct.sha(__file__),'committed_verified_before':existing,'partition_sizes':[len(x) for x in parts],'continuous_monitoring':False}
atomic(r/'scale5_migration/RESOURCE_AUTHORITY.json',receipt);ps=[];streams=[];env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,2,3,4,5',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',TTT_INFERENCE_RUNTIME_ROOT=str(ct.F),TTT_INFERENCE_RUNTIME_ARCHIVE=str(R/'shared/packages/ttt_inference_runtime_v1/ttt_inference_runtime_v1.tar.gz'))
try:
 for i in range(5):
  f=(r/('worker_scale5_%d.log'%i)).open('a');streams.append(f);ps.append(subprocess.Popen([sys.executable,'-B',str(W/'scale5_worker.py'),'--job',str(r/'JOB.json'),'--worker-id',str(i)],stdout=f,stderr=subprocess.STDOUT,env=env))
 state={**fc.FLAGS,'status':'RUNNING','phase':'fixed','pids':[p.pid for p in ps],'physical_gpus':[0,2,3,4,5],'started':ct.read(r/'scale5_migration/previous_phase_status.json')['started'],'scale5_started':time.time()};atomic(r/'PHASE_STATUS.json',state);print(json.dumps(state),flush=True)
 codes=[p.wait() for p in ps];state.update(status='COMPLETE' if codes==[0]*5 else 'FAILED',exit_codes=codes,ended=time.time());atomic(r/'PHASE_STATUS.json',state);assert codes==[0]*5,codes
 atomic(r/'FIXED8_COMPLETED_AUTHORITY.json',dict(schema='COMPLETED_PHASE_AUTHORITY_V1',status='PASS',phase='fixed',run_root=str(r),job_sha256=ct.sha(r/'JOB.json'),resource_authority_sha256=ct.sha(r/'scale5_migration/RESOURCE_AUTHORITY.json'),**fc.FLAGS))
 fc.finalize(types.SimpleNamespace(phase='fixed',fixed_authority=None,run_root=str(r)))
except BaseException:
 traceback.print_exc();raise
