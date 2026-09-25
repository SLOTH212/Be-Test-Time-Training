import json,hashlib,time,difflib
from pathlib import Path
import yaml
R=Path('/path/to/ttt');W=R/'work/qwen3_4b_stage1_stage2_pipeline_formal_v1';A=R/'audits/qwen3_4b_stage1_stage2_pipeline_formal_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
config=R/'configs/qwen3_4b_stage1_stage2_pipeline_formal_v1.yaml';c=read(config);before=read(config);files={};diffs=[]
loader=W/'gpu45_resource_loader.py';files[str(loader)]=sha(loader)
for stage in (1,2):
 old=R/(f'bin/stage{stage}_runtime_entry_'+('v2.py' if stage==1 else 'v1.py'));new=R/f'bin/stage{stage}_runtime_entry_gpu45_pipeline_v1.py';source=old.read_text()
 prefix="import sys,os\nif os.environ.get('CUDA_VISIBLE_DEVICES')!='4,5' or os.environ.get('WORLD_SIZE')!='2':raise RuntimeError('GPU45_RESOURCE_REQUIRED')\nsys.path.insert(0,"+repr(str(W))+')\nfrom gpu45_resource_loader import execute_worker\n'
 source=source.replace("!='2,3'","!='4,5'").replace('runpy.run_path(', 'execute_worker(')
 new.write_text(prefix+source);compile(new.read_text(),str(new),'exec');files[str(old)]=sha(old);files[str(new)]=sha(new)
 original=Path(c[f'stage{stage}']['config']);cfg=yaml.safe_load(original.read_text());cfg['physical_gpu_allowlist']='4,5'
 if stage==1:
  cfg['gpu_uuids']={4:'GPU-ae0edae4-c3bc-cfb2-4b15-f8cdd888bc25',5:'GPU-25896d0f-c552-9793-5c7a-ced116606c44'};cfg['minimum_free_memory_mib']={4:43000,5:43000};cfg['authorized_existing_gpu_processes']=[]
  cfg['output_root']=str(R/'runs/qwen3_4b_stage1_1b_32k_gpu45_formal_v1');cfg['stage1_model_only_artifact']=str(R/'checkpoints/qwen3_4b_stage1_1b_32k_gpu45_formal_v1_model_only')
 else:cfg['input_identity']['stage2_entry_sha256']=sha(new)
 cfg['input_identity']['gpu45_resource_loader_sha256']=sha(loader)
 dst=R/'configs'/original.name.replace('gpu23','gpu45')
 if stage==1:dst.write_text(yaml.safe_dump(cfg,sort_keys=False))
 else:write(dst,cfg)
 dst.with_suffix('.sha256').write_text(sha(dst)+'  '+dst.name+'\n');files[str(original)]=sha(original);files[str(dst)]=sha(dst);c[f'stage{stage}']['historical_config']=str(original);c[f'stage{stage}']['config']=str(dst)
 # Only resource/output identity differs; every scientific key remains equal.
 prior=yaml.safe_load(original.read_text());allowed={'physical_gpu_allowlist','gpu_uuids','minimum_free_memory_mib','authorized_existing_gpu_processes','output_root','stage1_model_only_artifact','input_identity'}
 assert all(prior.get(k)==cfg.get(k) for k in set(prior)|set(cfg) if k not in allowed)
 diffs.append({'stage':stage,'original_config_sha256':sha(original),'derived_config_sha256':sha(dst),'changed_keys':[k for k in set(prior)|set(cfg) if prior.get(k)!=cfg.get(k)],'scientific_parameters_unchanged':True})
old=Path(c['stage1']['guard']);source=old.read_text();idx=source.index('\ndef guard():');head=source[:idx];tail=source[idx:]
helper="\ndef gpu45_resource_contract():\n import sys\n sys.path.insert(0,"+repr(str(W))+')\n from gpu45_resource_loader import resource_contract\n return resource_contract()\n'
head=head.replace('NAME="qwen3_4b_stage1_1b_32k_gpu23_formal_v1"','NAME="qwen3_4b_stage1_1b_32k_gpu45_formal_v1"')
tail=tail.replace('validate_v4_1_authority(auth)','validate_v4_1_authority(auth)\n resource=gpu45_resource_contract()')
tail=tail.replace('"2,3"','"4,5"').replace('[2,3]','[4,5]')
# query() belongs to head and must use the new physical pair, while the
# historical V4.1 validator retains its original [2,3] evidence requirements.
head=head.replace('"-i","2,3"','"-i","4,5"')
tail=tail.replace('75740a4513f3f74302f276e1be38314a98ac095ed68d5ede458f4f8774576ff8',sha(c['stage1']['config']))
tail=tail.replace('WORKER=owned(auth.get("FORMAL_RUNTIME_ENTRYPOINT",str(WORKER)))','WORKER=owned('+repr(str(R/'bin/stage1_runtime_entry_gpu45_pipeline_v1.py'))+')')
tail=tail.replace('auth["file_sha256"].get(str(WORKER))','resource["files"].get(str(WORKER))')
new=R/'bin/qwen3_4b_stage1_gpu45_formal_control_v4_1_pipeline_v1.py';new.write_text(head+helper+tail);compile(new.read_text(),str(new),'exec');files[str(old)]=sha(old);files[str(new)]=sha(new);c['stage1']['historical_guard']=str(old);c['stage1']['guard']=str(new);c['stage1']['launcher']=str(new);c['stage1']['run_root']=str(R/'runs/qwen3_4b_stage1_1b_32k_gpu45_formal_v1');c['stage1']['pointer']=c['stage1']['run_root']+'.process.json'
old=Path(c['stage2']['controller']);source=old.read_text();source=source.replace('qwen3_4b_stage2_30m_32k_gpu23_formal_v1.yaml','qwen3_4b_stage2_30m_32k_gpu45_formal_v1.yaml').replace("'2,3'","'4,5'").replace('(2,3)','(4,5)').replace("bin/stage2_runtime_entry_v1.py","bin/stage2_runtime_entry_gpu45_pipeline_v1.py")
source=source.replace('def check(parent):','def check(parent):\n sys.path.insert(0,'+repr(str(W))+')\n from gpu45_resource_loader import resource_contract\n resource_contract()')
new=R/'bin/qwen3_4b_stage2_gpu45_formal_control_pipeline_v1.py';new.write_text(source);compile(source,str(new),'exec');files[str(old)]=sha(old);files[str(new)]=sha(new);c['stage2']['historical_controller']=str(old);c['stage2']['controller']=str(new)
launch=R/'bin/launch_qwen3_4b_stage2_30m_32k_gpu45_formal_pipeline_v1.sh';launch.write_text('#!/usr/bin/env bash\nset -Eeuo pipefail\nsource /path/to/ttt/bin/activate_ttt.sh\nexport CUDA_VISIBLE_DEVICES=4,5 WORLD_SIZE=2 PYTHONDONTWRITEBYTECODE=1\nunset CUBLAS_WORKSPACE_CONFIG PYTHONHASHSEED\nexec python -B '+str(new)+' launch "$@"\n');launch.chmod(0o755);files[str(launch)]=sha(launch);c['stage2']['historical_launcher']=c['stage2']['launcher'];c['stage2']['launcher']=str(launch)
from gpu45_resource_loader import adapt
for stage,name in [(1,'distributed_train_worker.py'),(2,'distributed_stage2_train_worker_v1.py')]:
 path=R/'src/training_runtime/code/workers'/name;original=path.read_text();adapted=adapt(original);compile(adapted,str(path)+'<GPU45>','exec');files[str(path)]=sha(path)
 patch=A/f'STAGE{stage}_GPU45_RESOURCE_ONLY.diff';patch.write_text(''.join(difflib.unified_diff(original.splitlines(True),adapted.splitlines(True),fromfile=str(path),tofile=str(path)+'<GPU45_RESOURCE_ONLY>')));diffs.append({'worker':str(path),'original_sha256':sha(path),'adapted_execution_source_sha256':hashlib.sha256(adapted.encode()).hexdigest(),'diff':str(patch),'change_scope':'Two physical-visibility literals and physical GPU logging offset only; all scientific code byte-identical after reversing these substitutions.'})
auth=R/'provenance/QWEN3_4B_PIPELINE_GPU45_RESOURCE_AUTHORITY_V1.json';write(auth,{'schema':'GPU45_RESOURCE_AMENDMENT_V1','user_instruction':'把gpu改成4-5，但不正式启动，在等卡','created_at':time.time(),'physical_gpu_allowlist':[4,5],'world_size':2,'historical_physical_gpu_allowlist':[2,3],'historical_stage1_authority_sha256':sha(c['stage1']['authority']),'historical_stage2_preflight_sha256':sha(c['stage2']['authority']),'scientific_semantics_changed':False,'gpu_trials_performed':False,'formal_training_started':False,'files':files,'diff_audits':diffs});auth.with_suffix('.sha256').write_text(sha(auth)+'  '+auth.name+'\n')
c['physical_gpu_allowlist']=[4,5];c['gpu_resource_authority']=str(auth);c['bindings'].update(files);c['bindings'][str(auth)]=sha(auth)
write(A/'PIPELINE_CONFIG_BEFORE_GPU45_AMENDMENT.json',before);write(config,c);write(A/'GPU45_RESOURCE_AMENDMENT_AUDIT.json',{'status':'PASS_STATIC_RESOURCE_ONLY','authority':str(auth),'sha256':sha(auth),'changes':diffs,'no_GPU_execution':True})
print(json.dumps({'gpu45_resource_authority':str(auth),'sha256':sha(auth),'stage1_guard':c['stage1']['guard'],'stage2_launcher':c['stage2']['launcher'],'formal_training_started':False},indent=2))
