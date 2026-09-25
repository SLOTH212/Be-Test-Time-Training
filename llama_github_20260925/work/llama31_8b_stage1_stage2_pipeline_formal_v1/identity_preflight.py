"""Read-only model, port, optimizer and mathematical source identity audit."""
import ast,json,hashlib,socket,getpass
from pathlib import Path
R=Path('/path/to/ttt');W=R/'work/llama31_8b_stage1_stage2_pipeline_formal_v1';C=R/'src/training_runtime/code'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
assert socket.gethostname()=='amax' and getpass.getuser()=='USER'
smoke=R/'provenance/LLAMA31_8B_TRAIN_INFERENCE_SMOKE_V1.json'
assert sha(smoke)=='84dab6b19a522e6f7c38e3732210c4d861bb0e231ed204482089b46490876557'
identity=read(R/'work/llama31_8b_compat_smoke_v1/model_identity.json');model=Path(identity['path']);files={}
for name,h in identity['files'].items():assert sha(model/name)==h;files[str(model/name)]=h
for name,e in identity['shards'].items():assert sha(model/name)==e['sha256'];files[str(model/name)]=e['sha256']
assert (model/'HF_REVISION.txt').read_text().strip()=='d04e592bb4f6aa9cfee91e2e20afa771667e1d4b'
assert [identity['config'][k] for k in ['num_hidden_layers','hidden_size','intermediate_size']]==[32,4096,14336]
port=R/'src/llama31_ntp_port_v1';archive=R/'packages/llama31_ntp_port_v1/llama31_ntp_port_v1.tar.gz'
assert sha(archive)=='3f1ca6c6cee8f8601fd5bfca396a067fe1876cbba3fbabecb273e75e19fd54bc'
for row in read(R/'work/llama31_8b_compat_smoke_v1/port_identity.json')['files']:
 p=port/row['path'];assert sha(p)==row['sha256'];files[str(p)]=row['sha256']
reference=read(W/'worker_reference_manifest.json')
for p,h in reference.items():assert sha(p)==h;files[p]=h
def definition(source,name):
 tree=ast.parse(Path(source).read_text());return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
for stage,source in [(1,R/'work/llama31_8b_compat_smoke_v1/worker.py'),(2,C/'workers/distributed_stage2_train_worker_v1.py')]:
 for name in ['fused_sum','masks_for','local_rows','count_window']:
  assert definition(source,name)==definition(W/f'worker_stage{stage}.py',name),(stage,name)
for name in ['distributed_semantics.py','distributed_grad.py','distributed_checkpoint.py','stage1_closure_checkpoint_v2.py','stage1_sparse_grad_v2.py','stage2_checkpoint_runtime.py','fsdp2_runtime.py']:
 files[str(C/'lib'/name)]=sha(C/'lib'/name)
import yaml
cfg=yaml.safe_load((R/'configs/llama31_8b_stage1_32k_gpu0123_debug_v1.yaml').read_text())
optimizer={k:cfg[k] for k in ['optimizer','outer_lr','betas','eps','weight_decay','outer_grad_clip','scheduler','precision','gradient_checkpointing_use_reentrant']}
assert optimizer=={'optimizer':'AdamW','outer_lr':5e-6,'betas':[0.9,0.95],'eps':1e-8,'weight_decay':0.1,'outer_grad_clip':1.0,'scheduler':'constant','precision':'bf16','gradient_checkpointing_use_reentrant':False}
result={'status':'PASS','model_revision':identity['revision'],'files':files,'port_package_sha256':sha(archive),'inherited_smoke_sha256':sha(smoke),'optimizer':optimizer,'unchanged_math_functions':['fused_sum','masks_for','local_rows','count_window'],'formal_training_started':False,'stage1_split_resolution':'PENDING_USER_RESPONSE'}
(W/'identity_cpu_preflight.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':'PASS','verified_files':len(files),'optimizer':optimizer,'unchanged_math_functions':result['unchanged_math_functions']}))
