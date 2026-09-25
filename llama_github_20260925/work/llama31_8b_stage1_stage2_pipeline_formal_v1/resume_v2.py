"""One explicit resume dispatch, preserving the original run and audit trail."""
import sys
from core import *

def dispatch(c):
 from pipeline import env,move
 root=Path(c['state_root']);run=Path(c['stage1']['run_root']);intent=root/'STAGE1_RESUME_V3.json'
 with lock(root/'PIPELINE.lock'),lock(root/'STAGE1.lock'):
  require(not intent.exists(),'RESUME_ALREADY_DISPATCHED')
  require(read(root/'PIPELINE_STATE.json')['state']=='BLOCKED','RESUME_REQUIRES_BLOCKED')
  require(not trainer_snapshot(1,c)['active'],'ACTIVE_TRAINER')
  for key in ['TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','TORCH_EXTENSIONS_DIR','CUDA_CACHE_PATH','TMPDIR']:
   require(os.environ.get(key) and Path(os.environ[key]).resolve().is_relative_to(R),'RESUME_CACHE_ENV_'+key)
  preflight(c,live=True)
  slot,manifest=dcp(run,{'update_step':500,'record_cursor':2000,'cumulative_tokens':55557667})
  require(last(run)['update_step']==500,'UNEXPECTED_LOG_CURSOR')
  import yaml
  cfg=yaml.safe_load(Path(c['stage1']['config']).read_text())
  require(manifest['batch_metadata']['identity']=={**cfg['input_identity'],'config_sha256':sha(c['stage1']['config'])},'RESUME_IDENTITY')
  archive=run/'attempt_2_environment_failed';require(not archive.exists(),'ATTEMPT_ARCHIVE_EXISTS');archive.mkdir()
  for name in ['launcher.log','launcher_exit.json','process.json','startup_health.json']:
   if (run/name).exists():os.replace(run/name,archive/name)
  atomic(intent,{'phase':'INTENT','time':time.time(),'source_sha256':sha(SOURCE),'checkpoint_manifest_sha256':sha(slot/'manifest.json'),'progress':manifest['progress'],'automatic_retry':False})
  command=[sys.executable,'-B',str(W/'pipeline.py'),'trainer','--stage','1','--execute','--resume']
  # Bootstrap waits until dispatch releases both locks.
  bootstrap='import os,time;time.sleep(2);os.execv('+repr(sys.executable)+','+repr(command)+')'
  with (root/'stage1_manager.log').open('a') as log:
   proc=subprocess.Popen([sys.executable,'-c',bootstrap],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env(''),cwd=R)
  rec=read(intent);rec.update(phase='SPAWNED',manager=identity(proc.pid));atomic(intent,rec)
  move(c,'STAGE1_RUNNING','Explicit user-authorized resume from validated step500 with checkpoint I/O repair')
  print(json.dumps(rec))
