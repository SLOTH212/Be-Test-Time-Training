import ast,json,math,hashlib
from pathlib import Path
import torch,numpy as np
W=Path('/path/to/ttt/work/qwen3_4b_stage1_resume_shared_boundary_v4');A=Path('/path/to/ttt/audits/qwen3_4b_stage1_resume_shared_boundary_v4')
source=(W/'analyze.py').read_text();tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='metrics');scope={'torch':torch,'np':np,'math':math};exec(compile(ast.Module(body=[node],type_ignores=[]),'<isolated_v4_metrics>','exec'),scope)
checks=[]
for dtype in [torch.bfloat16,torch.float32]:
 a=torch.tensor([0.,1e-7,-1.,2.,0.003,0.],dtype=dtype);b=torch.tensor([0.,-2e-7,-1.,2.02,0.0031,0.],dtype=dtype);v=scope['metrics']([(a,b)],1e-6);d=(a.float()-b.float()).abs().double();checks.append({'dtype':str(dtype),'metrics':v,'PASS':v['max_abs']==float(d.max()) and math.isclose(v['L2'],float(torch.linalg.vector_norm(d)),rel_tol=1e-14) and v['legacy_allclose']==bool(torch.isclose(a.float(),b.float(),atol=1e-6,rtol=.01).all())})
assert all(c['PASS'] for c in checks)
# Independent analytic examples exercise diameter/nearest rule and B3 criterion.
cases=[('all_equal',[0.,0.],0.,1e-6,True),('within_diameter',[.9,1.1],1.,.01,True),('one_far_outlier',[.5,2.5],1.,.01,False),('A_separated',[4.,4.1],1.,.01,False),('zero_B_variation',[1e-7,1e-7],0.,1e-6,True)]
for name,ab,diam,floor,expected in cases:assert (max(ab)<=2*diam+floor and min(ab)<=diam+floor)==expected
assert not torch.cuda.is_initialized()
(A/'ANALYSIS_CPU_CHECKS.json').write_text(json.dumps({'DEBUG_ONLY':True,'status':'PASS','metric_checks':checks,'causal_rule_examples':cases,'CUDA_initialized':False,'analysis_sha256':hashlib.sha256(source.encode()).hexdigest()},indent=2)+'\n');print('CPU_CHECKS_PASS')
