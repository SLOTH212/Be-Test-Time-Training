import torch
from dynamic_ttt.training.qwen_stage2 import masks_for,count_window
from dynamic_ttt.training.lib.distributed_semantics import BatchPlan
from dynamic_ttt.training.lib.pipeline_core import separate_group_loss
def test_shifted_answer_and_context_masks():
 a,c=masks_for({'id':'x','n_tokens':6},{'x':[[3,5]]})
 assert a.tolist()==[False,False,True,True,False]
 assert c.tolist()==[True,True,False,False,True]
 loss,al,cl,A,C=separate_group_loss([1,2,3,4,999],[False,False,True,True,False],[True,True,True,True,False])
 assert (A,C)==(2,2) and loss==3.5+.1*1.5
def test_optimizer_window_denominators():
 rows=[(0,{'id':'a','n_tokens':6}),(1,{'id':'b','n_tokens':4})]
 plan=BatchPlan(1,1,2)
 A,C,tokens=count_window(rows,plan,0,2,{'a':[[3,5]],'b':[[2,3]]},torch.device('cpu'))
 assert (A,C,tokens)==(3,5,10)
