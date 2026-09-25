import pathlib,json,sys,os,time,subprocess,traceback,hashlib
R=pathlib.Path('/path/to/ttt');sys.path.insert(0,str(R/'work/qwen3_4b_stage1_stage2_pipeline_formal_v1'));import pipeline_core as core
c=core.config();P=pathlib.Path(c['state_root']);journal=P/'STAGE2_LAUNCH_RECEIPT.json';parent=pathlib.Path(c['stage1']['final_root'])/'STAGE1_FINAL_AUTHORITY.json';src=pathlib.Path(c['stage2']['controller'])
old="if idx not in (4,5) or free<48000:raise RuntimeError('GPU_MEMORY_CAPACITY_NOT_AVAILABLE')"
new="if idx not in (4,5):raise RuntimeError('GPU_ALLOWLIST')"
with core.lock(P/'PIPELINE.lock'):
 core.source_contract();assert not journal.exists();assert not pathlib.Path(c['stage2']['pointer']).exists();assert not list((R/'runs').glob('qwen3_4b_stage2_formal_*'))
 source=src.read_text();assert source.count(old)==1;patched=source.replace(old,new);assert patched.replace(new,old)==source
 receipt={'phase':'INTENT','stage':2,'created_at':time.time(),'explicit_user_instruction':'Directly attempt Stage2 launch with current GPU4/5 free capacity','only_override':'Remove fixed 48000 MiB free-memory gate; real CUDA allocation errors remain fatal','original_controller_sha256':core.digest(src),'executed_controller_sha256':hashlib.sha256(patched.encode()).hexdigest(),'override_script_sha256':core.digest(__file__),'source_authority_sha256':core.digest(core.SOURCE),'scientific_config_changed':False,'gpu_before':subprocess.check_output(['nvidia-smi','-i','4,5','--query-gpu=index,uuid,memory.free','--format=csv'],text=True)}
 core.atomic(journal,receipt)
 try:
  namespace={'__name__':'stage2_explicit_capacity_attempt','__file__':str(src)};exec(compile(patched,str(src),'exec'),namespace);sys.argv=[str(src),'launch','--parent-authority',str(parent)];namespace['main']()
  pointer=core.read(c['stage2']['pointer']);receipt.update(phase='LAUNCHER_RETURNED',returncode=0,ended_at=time.time(),observed=pointer);core.atomic(journal,receipt)
  core.atomic(P/'STAGE2_RESOURCE_WAIT.json',{'status':'USER_AUTHORIZED_DIRECT_ATTEMPT_LAUNCHED','timestamp':time.time(),'fixed_capacity_gate_overridden_for_this_launch':True})
  state=core.read(P/'PIPELINE_STATE.json');state=core.transition(state,'STAGE2_RUNNING',core.digest(c['stage2']['authority']),pointer['run_root'],'Explicit user-authorized direct allocation attempt launched; fixed free-memory gate bypassed once');core.atomic(P/'PIPELINE_STATE.json',state)
 except BaseException as e:
  receipt.update(phase='LAUNCH_FAILED',error=repr(e),ended_at=time.time());core.atomic(journal,receipt);raise
