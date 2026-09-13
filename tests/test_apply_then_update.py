import torch
from dynamic_ttt.ttt.state import ttt_chunk_step,GATED_NTP_TARGET
def test_outer_product_and_apply_order(tensors):
 h,x,w,p,g=tensors;mask=torch.tensor([[True,True,False,True]])
 out,state,stats=ttt_chunk_step(h,x,w,p,None,mask,learning_rate=1.,target_type=GATED_NTP_TARGET,target_gate=g)
 expected=w+torch.outer((x[0,1]*g)@p,h[0,0])
 torch.testing.assert_close(out,torch.nn.functional.linear(h,w),rtol=0,atol=0)
 torch.testing.assert_close(state.weight,expected,rtol=1e-12,atol=1e-12)
 assert stats.valid_tokens==1
def test_clip(tensors):
 h,x,w,p,g=tensors
 out,state,stats=ttt_chunk_step(h,x,w,p,None,torch.ones(1,4,dtype=torch.bool),learning_rate=1.,target_type=GATED_NTP_TARGET,target_gate=g,delta_clip_norm=1e-5)
 assert stats.was_clipped and abs(stats.post_clip_norm-1e-5)<1e-11
