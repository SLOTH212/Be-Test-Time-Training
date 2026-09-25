import pathlib,sys,json,os,time,subprocess,fcntl,types,traceback
R=pathlib.Path('/path/to/ttt');W=R/'work/qwen3_4b_dynamic_gpu05_v2';S=R/'src/qwen3_4b_downstream_pipeline_v1';sys.path.insert(0,str(S));import contracts as ct
ct.source_guard();cfg=W/'config.json';config=ct.read(cfg);original_path=ct.config_path;original=ct.read(original_path('dynamic'));resource_keys={'physical_gpu_allowlist','workers','run_root'};assert {k:v for k,v in config.items() if k not in resource_keys}=={k:v for k,v in original.items() if k not in resource_keys}
original_validate=ct.validate_config
def validate(c,phase):
 if phase=='dynamic':
  assert c['physical_gpu_allowlist']==list(range(6)) and c['workers']==6
  original_validate({**c,'physical_gpu_allowlist':[2,3],'workers':2},phase);return c
 return original_validate(c,phase)
ct.validate_config=validate;ct.config_path=lambda phase:cfg if phase=='dynamic' else original_path(phase)
def replace(s,a,b):
 assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
E=R/'work/qwen3_4b_dynamic_gpu07_v2'
if '--job' in sys.argv:
 p=S/'formal_worker.py';source=p.read_text();source=replace(source,"os.environ.get('CUDA_VISIBLE_DEVICES')=='2,3' and args.worker_id in [0,1]","os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1,2,3,4,5,6,7' and args.worker_id in [0,1,2,3,4,5,6,7]")
 source=replace(source,"['fb7ba9a8-ce12-65a4-a9cf-5b5b7506b0ed','351e23d4-250b-6a70-2c66-d2fb0221a25a']","['c7982600-dea0-9d54-16b8-76db4288663e','8459f05d-4d22-5b39-cc1f-e25137d49389','fb7ba9a8-ce12-65a4-a9cf-5b5b7506b0ed','351e23d4-250b-6a70-2c66-d2fb0221a25a','ae0edae4-c3bc-cfb2-4b15-f8cdd888bc25','25896d0f-c552-9793-5c7a-ced116606c44','f377875e-f7da-0d49-cb45-b57a4acd2a0f','a8814bd0-844f-5804-faae-c6e74f3ae82d']")
 source=replace(source,"owned=j['sample_ids'][args.worker_id::2]","owned=ct.read(Path(j['run_root'])/'GPU07_RESUME_PLAN.json')['partitions'][args.worker_id]");source=replace(source,'physical_gpu=args.worker_id+2','physical_gpu=args.worker_id')
 ns={'__name__':'dynamic_gpu05_resource_overlay','__file__':str(p)};exec(compile(source,str(p),'exec'),ns);ns['main']()
else:
 import formal_control as fc
 from worker import Store,atomic
 lock=(E/'LAUNCH.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 root=pathlib.Path(config['run_root']);job=ct.read(root/'JOB.json');plan=ct.read(root/'GPU07_RESUME_PLAN.json');store=Store(root,job['store_identity'])
 assert ct.sha(root/'JOB.json')==plan['job_sha256']
 assert ct.sha(E/'entry.py')==plan['entry_sha256']
 ids=job['sample_ids'];parts=plan['partitions'];pending=set(x for a in parts for x in a)
 assert len(parts)==8 and sum(map(len,parts))==len(pending)
 assert pending.issubset(set(ids)) and all(store.committed(x) is not None for x in set(ids)-pending)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',TTT_INFERENCE_RUNTIME_ROOT=str(ct.F),TTT_INFERENCE_RUNTIME_ARCHIVE=str(R/'shared/packages/ttt_inference_runtime_v1/ttt_inference_runtime_v1.tar.gz'))
 ps=[];streams=[]
 for i in range(8):
  f=(E/('worker%d.log'%i)).open('x');streams.append(f);ps.append(subprocess.Popen([sys.executable,'-B',str(E/'entry.py'),'--job',str(root/'JOB.json'),'--worker-id',str(i)],stdout=f,stderr=subprocess.STDOUT,env=env))
 state={**fc.FLAGS,'status':'RUNNING','phase':'dynamic','pids':[p.pid for p in ps],'coordinator_pid':os.getpid(),'physical_gpus':list(range(8)),'started':plan['original_started'],'resumed_at':time.time(),'sample_n':1278,'population_n':6000,'skipped_n':4722,'resource_extension':str(root/'GPU07_RESUME_PLAN.json')};atomic(root/'PHASE_STATUS.json',state);print(json.dumps(state),flush=True)
 codes=[p.wait() for p in ps];state.update(status='COMPLETE' if codes==[0]*8 else 'FAILED',exit_codes=codes,ended=time.time());atomic(root/'PHASE_STATUS.json',state);assert codes==[0]*8,codes
 assert all(store.committed(x) is not None for x in ids)
 atomic(root/'COMPLETED_AUTHORITY.json',dict(schema='COMPLETED_PHASE_AUTHORITY_V1',status='PASS',phase='dynamic',run_root=str(root),job_sha256=ct.sha(root/'JOB.json'),resource_extension_sha256=ct.sha(root/'GPU07_RESUME_PLAN.json'),**fc.FLAGS));fc.finalize(types.SimpleNamespace(phase='dynamic',fixed_authority=None,run_root=str(root)))
