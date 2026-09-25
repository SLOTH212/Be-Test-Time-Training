import os,json,hashlib,inspect,ast,importlib,time
from pathlib import Path
import torch,numpy as np
R=Path('/path/to/ttt');L=R/'audits/v3/localization_final';N=R/'work/qwen3_4b_stage1_resume_v3/localization_final'
assert not torch.cuda.is_initialized()
source=(N/'entry.py').read_text();tree=ast.parse(source);checks={}
checks['no_explicit_rng_restore_in_observer']='cp.restore_rank_state' not in source
checks['no_deterministic_switch']='use_deterministic_algorithms' not in source
checks['no_optimizer_or_forward_math_rewrite']='optimizer.step()' not in source and '.backward()' not in source
clip=importlib.import_module('torch.nn.utils.clip_grad');orig=inspect.getsource(clip._clip_grads_with_norm_);needle='    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)';changed=orig.replace(needle,needle+'\n    _v3_record_clip(total_norm, max_norm, clip_coef, clip_coef_clamped)');t=ast.parse(changed)
class RemoveObservation(ast.NodeTransformer):
 def visit_Expr(self,n):
  if isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='_v3_record_clip':return None
  return self.generic_visit(n)
checks['clip_AST_identical_after_removing_one_read_call']=ast.dump(RemoveObservation().visit(t))==ast.dump(ast.parse(orig))
seen=[];g=dict(clip.__dict__);g['_v3_record_clip']=lambda *v:seen.append([x.detach().clone() if isinstance(x,torch.Tensor) else x for x in v]);exec(compile(changed,'testclip','exec'),g)
for dtype in [torch.float32,torch.bfloat16]:
 for val in [.01,1.,41.5,41.75]:
  a=torch.nn.Parameter(torch.ones(17,dtype=dtype));b=torch.nn.Parameter(a.detach().clone());a.grad=torch.arange(17,dtype=dtype);b.grad=a.grad.clone();state=torch.get_rng_state().clone()
  clip._clip_grads_with_norm_([a],1.,torch.tensor(val,dtype=dtype));g['_clip_grads_with_norm_']([b],1.,torch.tensor(val,dtype=dtype))
  assert torch.equal(a.grad,b.grad) and torch.equal(state,torch.get_rng_state())
checks['original_and_observed_clip_exact_8_cases']=True
checks['coefficient_observed_from_actual_variable']=len(seen)==8
helpers=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['capture_value','tensor_fingerprint']]
g={'torch':torch,'np':np,'hashlib':hashlib,'observer_read_checks':0,'captured':{},'capture_meta':{},'rng_only':lambda:hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()}
exec(compile(ast.Module(body=helpers,type_ignores=[]),'helpers','exec'),g)
x=torch.arange(128,dtype=torch.float32).reshape(8,16);before=x.clone();state=torch.get_rng_state().clone();g['capture_value']('test',x,'cpu_check');fp=g['tensor_fingerprint'](x)
checks['capture_and_fingerprint_preserve_tensor_and_rng']=torch.equal(before,x) and torch.equal(state,torch.get_rng_state()) and fp['finite_count']==128 and torch.equal(g['captured']['test'],x)
importlib.import_module('torch.distributed.fsdp._fully_shard._fsdp_param_group')
checks['hooks_return_original_forward_reduce_clip_results']=all(x in source for x in ['return original_foreach_reduce(fsdp_params,unsharded_grads,*args,**kwargs)','result=original_clip(*args,**kwargs)','result=original_fused(*args,**kwargs)'])
checks['all_checks_before_cuda_initialization']=not torch.cuda.is_initialized()
assert all(checks.values()),checks
payload={'status':'PASS','checks':checks,'observer_sha256':hashlib.sha256(source.encode()).hexdigest(),'torch_clip_source_sha256':hashlib.sha256(orig.encode()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'review':'Prepared original observer had undefined final-export names and lacked forward/coefficient observations. New entry fixes diagnostics only, removes redundant RNG restore, guards observed versions/RNG, and retains original numeric operations. CPU copies may perturb timing; no claim of timing-neutral observation.','completed_at':time.time()}
(L/'OBSERVER_READ_ONLY_AUDIT.json').write_text(json.dumps(payload,indent=2)+'\n');print('OBSERVER_READ_ONLY_SEMANTICS_PASS',checks)
