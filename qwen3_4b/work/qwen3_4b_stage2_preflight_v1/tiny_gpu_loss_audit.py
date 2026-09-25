import os,sys,json,ast,hashlib
from pathlib import Path
import torch
import torch.distributed as dist
from types import SimpleNamespace
R=Path('/path/to/ttt');A=R/'audits/qwen3_4b_stage2_30m_32k_gpu23_preflight_v1'
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='2,3'
rank=int(os.environ['LOCAL_RANK']);assert rank in (0,1)
torch.cuda.set_device(rank);assert torch.cuda.device_count()==2
dist.init_process_group('nccl');device=torch.device('cuda',rank)
sys.path.insert(0,str(R/'work/qwen3_4b_stage2_preflight_v1/recovered_0p6b/source/loss'))
from answer_context_fused_loss import single_decoder_fused_answer_context
P=R/'src/training_runtime/code/workers/distributed_stage2_train_worker_v1.py'
ns={'torch':torch,'Path':Path,'json':json}
tree=ast.parse(P.read_text())
for name in ('masks_for','fused_sum'):
 node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name)
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(P),'exec'),ns)
class Decoder(torch.nn.Module):
 def __init__(self):
  super().__init__();self.embedding=torch.nn.Embedding(16,8)
 def forward(self,input_ids,**kw):return SimpleNamespace(last_hidden_state=self.embedding(input_ids))
class Toy(torch.nn.Module):
 def __init__(self):
  super().__init__();self.model=Decoder();self.lm_head=torch.nn.Linear(8,16,bias=False)
torch.manual_seed(17);model=Toy().to(device)
ids=torch.tensor([[1,2,3,4,5,6,7,8]],device=device)
start,stop=(2,3) if rank==0 else (3,6)
answer=torch.zeros_like(ids,dtype=torch.bool);answer[:,start:stop]=True
context=torch.ones_like(answer)&~answer
batch={'input_ids':ids,'attention_mask':torch.ones_like(ids),'labels':ids,'document_boundaries':[(0,8)]}
total,al,cl,counts=single_decoder_fused_answer_context(model,batch,answer,context)
a,c=ns['masks_for']({'id':'fixture','n_tokens':8},{'fixture':[[start,stop]]});a=a.to(device);c=c.to(device)
assert torch.equal(a,answer[0,1:]);assert torch.equal(c,context[0,1:])
ce=torch.nn.functional.cross_entropy(model.lm_head(model.model(ids).last_hidden_state[:,:-1]).reshape(-1,16),ids[:,1:].reshape(-1),reduction='none')
expected=ce[a].mean()+0.1*ce[c].mean()
torch.testing.assert_close(total,expected,atol=2e-6,rtol=2e-6)
anum=ns['fused_sum'](model,ids,a);cnum=ns['fused_sum'](model,ids,c)
torch.testing.assert_close(anum,ce[a].sum(),atol=2e-6,rtol=2e-6)
torch.testing.assert_close(cnum,ce[c].sum(),atol=3e-6,rtol=3e-6)
counts_global=torch.tensor([int(a.sum()),int(c.sum())],device=device);dist.all_reduce(counts_global)
ga,gc=map(int,counts_global.tolist())
# Exact frozen worker expression, independent explicit CE reference.
local_scaled=anum*(2/ga)+0.1*cnum*(2/gc)
reported=local_scaled.detach().clone();dist.all_reduce(reported);reported/=2
num=torch.stack([ce[a].sum().detach(),ce[c].sum().detach()]);dist.all_reduce(num)
global_expected=num[0]/ga+0.1*num[1]/gc
torch.testing.assert_close(reported,global_expected,atol=3e-6,rtol=3e-6)
model.zero_grad();local_scaled.backward()
actual_grads=[p.grad.detach().clone() for p in model.parameters()]
for g in actual_grads:dist.all_reduce(g);g/=2
model.zero_grad()
ref=torch.zeros((),device=device)
for r in [0,1]:
 ra=torch.zeros(7,dtype=torch.bool,device=device);rs,re=(2,3) if r==0 else (3,6);ra[rs-1:re-1]=True
 vals=torch.nn.functional.cross_entropy(model.lm_head(model.model(ids).last_hidden_state[:,:-1]).reshape(-1,16),ids[:,1:].reshape(-1),reduction='none')
 ref=ref+vals[ra].sum()/ga+0.1*vals[~ra].sum()/gc
ref.backward()
for actual,p in zip(actual_grads,model.parameters()):torch.testing.assert_close(actual,p.grad,atol=3e-6,rtol=1e-5)
# Ignored padded tail and shifted first answer token with the actual historical helper.
padids=torch.cat([ids,torch.tensor([[0,0]],device=device)],1)
labels=padids.clone();labels[:,8:]=-100
pa=torch.zeros_like(padids,dtype=torch.bool);pa[:,start:stop]=True
pc=torch.ones_like(pa)&~pa;pc[:,8:]=False
pb={'input_ids':padids,'attention_mask':labels.ne(-100).long(),'labels':labels,'document_boundaries':[(0,8)]}
pt,*_=single_decoder_fused_answer_context(model,pb,pa,pc)
torch.testing.assert_close(pt,total,atol=2e-6,rtol=2e-6)
zero=[]
for typ,aa,cc in [('answer',torch.zeros_like(answer),torch.ones_like(context)),('context',torch.ones_like(answer),torch.zeros_like(context))]:
 try:single_decoder_fused_answer_context(model,batch,aa,cc);raise AssertionError('ZERO_GROUP_NOT_REJECTED')
 except RuntimeError as e:
  assert str(e)==('ZERO_ANSWER_DENOMINATOR' if typ=='answer' else 'ZERO_CONTEXT_DENOMINATOR');zero.append(str(e))
rec={'rank':rank,'physical_gpu':rank+2,'device_name':torch.cuda.get_device_name(rank),'answer_count':int(a.sum()),'context_count':int(c.sum()),'global_answer_count':ga,'global_context_count':gc,'loss':float(total),'global_loss':float(reported),'historical_actual_helper_shift_numerator_denominator':'PASS','padding_ignore':'PASS','active_fused_sum_equivalence':'PASS','unequal_global_loss_and_averaged_gradient':'PASS','zero_group_expected_rejection':zero,'zero_mask_training_continuation':'NOT_SUPPORTED_BY_FROZEN_HELPER','cuda_peak_memory':torch.cuda.max_memory_allocated(),'formal_training':False}
allrows=[None,None];dist.all_gather_object(allrows,rec)
if rank==0:
 (A/'STAGE2_TINY_GPU_LOSS_AUDIT.json').write_text(json.dumps({'status':'PASS_VALID_GROUPS_AND_AUTHORITY_DEFINED_REJECTION','ranks':allrows,'limitation':'Tiny real Liger kernel fixture; not full Qwen/FSDP Stage2 training. Zero-group continuation is not supported and not claimed.'},indent=2)+'\n')
 print(json.dumps(allrows),flush=True)
dist.barrier();dist.destroy_process_group()
