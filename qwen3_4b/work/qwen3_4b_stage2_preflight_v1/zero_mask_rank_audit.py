import os,sys,json
from pathlib import Path
from types import SimpleNamespace
import torch,torch.distributed as dist
R=Path('/path/to/ttt');A=R/'audits/qwen3_4b_stage2_30m_32k_gpu23_preflight_v1'
assert os.environ['CUDA_VISIBLE_DEVICES']=='2,3';rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);dist.init_process_group('nccl')
sys.path.insert(0,str(R/'work/qwen3_4b_stage2_preflight_v1/recovered_0p6b/source/loss'))
from answer_context_fused_loss import single_decoder_fused_answer_context
class D(torch.nn.Module):
 def __init__(self):super().__init__();self.e=torch.nn.Embedding(16,8)
 def forward(self,input_ids,**kw):return SimpleNamespace(last_hidden_state=self.e(input_ids))
class M(torch.nn.Module):
 def __init__(self):super().__init__();self.model=D();self.lm_head=torch.nn.Linear(8,16,bias=False)
torch.manual_seed(42);m=M().cuda(rank);ids=torch.tensor([[1,2,3,4,5,6]],device=rank);answer=torch.zeros_like(ids,dtype=torch.bool)
if rank==1:answer[:,2:4]=True
context=~answer;before={k:v.detach().clone() for k,v in m.state_dict().items()}
try:
 loss,*_=single_decoder_fused_answer_context(m,{'input_ids':ids,'attention_mask':torch.ones_like(ids),'labels':ids},answer,context);local={'rank':rank,'error':None,'finite':bool(torch.isfinite(loss))}
except RuntimeError as e:local={'rank':rank,'error':str(e),'finite':None}
rows=[None,None];dist.all_gather_object(rows,local)
assert rows[0]['error']=='ZERO_ANSWER_DENOMINATOR' and rows[1]['error'] is None and rows[1]['finite']
assert all(torch.equal(v,before[k]) for k,v in m.state_dict().items())
# Fixture treats a source-defined error as a collective rejected update. No backward/optimizer.
dist.barrier()
if rank==0:
 receipt={'status':'PASS_AUTHORITY_DEFINED_FAIL_CLOSED_SAFETY','rows':rows,'optimizer_steps':0,'no_nan_or_divide_by_zero':True,'collective_fixture_completed_without_deadlock':True,'zero_mask_continuation_supported':False,'scope':'Actual frozen helper on a zero-answer rank plus a valid rank; fixture coordinates expected rejection. Not a claim that current training worker accepts arbitrary corrupted zero-mask records. Frozen real dataset has no such records.'}
 (A/'STAGE2_ZERO_MASK_RANK_AUDIT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
dist.destroy_process_group()
