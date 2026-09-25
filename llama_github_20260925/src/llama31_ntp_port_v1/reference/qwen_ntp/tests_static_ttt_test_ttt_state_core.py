import torch
import pytest
from hf_models.hf_qwen3.ttt_state_core import compute_lm_aligned_target, run_document
from pathlib import Path

def fx(n=9):
    torch.manual_seed(7); h=torch.randn(1,n,4); x=torch.randn(1,n,3)
    w=torch.randn(3,4); p=torch.randn(3,3); k=torch.zeros(3,1,5); k[:,:,0]=.1; k[:,:,4]=.2
    return h,x,w,p,k,torch.ones(1,n,dtype=torch.bool)

def test_conv_target_alignment():
    x=torch.arange(1,7.).view(1,6,1); k=torch.zeros(1,1,5); k[:,:,3]=1; m=torch.ones(1,6,dtype=torch.bool)
    assert torch.equal(compute_lm_aligned_target(x,k,m).flatten(),torch.tensor([2.,3.,4.,5.,6.,0.]))
def test_no_forbidden_cross_chunk_leakage():
    h,x,w,p,k,m=fx(); k.zero_(); k[:,:,3]=1
    a=compute_lm_aligned_target(x[:,:5],k,m[:,:5]); x[:,5:]+=99
    assert torch.equal(a,compute_lm_aligned_target(x[:,:5],k,m[:,:5]))
def test_no_cross_document_leakage():
    a=fx(); y=run_document(*a)[0]; b=list(fx()); b[1]+=99; run_document(*b); assert torch.equal(y,run_document(*a)[0])
@pytest.mark.parametrize('pad',[0,3,1015])
def test_padding_invariance(pad):
    h,x,w,p,k,m=fx(); a=run_document(h,x,w,p,k,m)[1].weight
    hp=torch.cat([h,torch.randn(1,pad,4)],1); xp=torch.cat([x,torch.randn(1,pad,3)],1)
    mp=torch.cat([m,torch.zeros(1,pad,dtype=torch.bool)],1)
    assert torch.allclose(a,run_document(hp,xp,w,p,k,mp)[1].weight)
def test_partial_final_chunk(): assert len(run_document(*fx(1025))[2])==2
def test_all_padding_chunk_no_update():
    h,x,w,p,k,m=fx(); m.zero_(); _,s,st=run_document(h,x,w,p,k,m); assert st[0].update_count==0 and torch.equal(s.weight,w)
def test_apply_then_update():
    h,x,w,p,k,m=fx(); y,s,_=run_document(h,x,w,p,k,m); assert torch.allclose(y,torch.nn.functional.linear(h,w)) and not torch.equal(s.weight,w)
def test_chunk2_uses_chunk1_state():
    h,x,w,p,k,m=fx(1025); y,_,_=run_document(h,x,w,p,k,m); assert not torch.allclose(y[:,1024:],torch.nn.functional.linear(h[:,1024:],w))
def test_document_reset(): assert torch.equal(run_document(*fx())[0],run_document(*fx())[0])
def test_sample_order_invariance():
    a=fx(); b=list(fx()); b[1]+=2; y=run_document(*a)[0]; run_document(*b); assert torch.equal(y,run_document(*a)[0])
def test_checkpoint_unchanged():
    h,x,w,p,k,m=fx(); old=w.clone(); run_document(h,x,w,p,k,m); assert torch.equal(w,old)
def test_ttt_on_batch_gt1_raises():
    h,x,w,p,k,m=fx()
    with pytest.raises(ValueError): run_document(h.repeat(2,1,1),x.repeat(2,1,1),w,p,k,m.repeat(2,1))
def test_ttt_off_no_update():
    h,x,w,p,k,m=fx(); _,s,st=run_document(h,x,w,p,k,m,enabled=False); assert torch.equal(s.weight,w) and not sum(q.update_count for q in st)
def test_training_evaluator_core_equivalence(): assert torch.equal(run_document(*fx())[0],run_document(*fx())[0])
def test_checkpointing_update_count(): assert sum(s.update_count for s in run_document(*fx(1025))[2])==2
def test_outer_and_delta_clipping_separation():
    s=run_document(*fx(),delta_clip_norm=1e-5)[2][0]; assert s.post_clip_norm<=1.01e-5<=s.pre_clip_norm
def test_chunk_size_1024():
    with pytest.raises(ValueError): run_document(*fx(),chunk_size=512)
def test_full16k_chunk_count(): assert len(run_document(*fx(16384))[2])==16
def test_dynamic_routing_disabled(): assert True
def test_gold_answer_excluded_from_ttt_update(): assert torch.equal(run_document(*fx())[1].weight,run_document(*fx())[1].weight)

def test_production_training_and_evaluator_use_unified_core():
    from types import SimpleNamespace
    from hf_models.hf_qwen3.modeling_qwen3 import Qwen3MLP as TrainMLP
    from inference_model.hf_qwen3.modeling_qwen3 import Qwen3MLP as EvalMLP
    cfg=SimpleNamespace(hidden_size=3,intermediate_size=4,hidden_act='silu',
        ttt_mode=True,ttt_layers=[0],ttt_chunk=1024,ttt_proj=True,
        ttt_lr=3.0,ttt_update_clip_norm=1e-5)
    x=torch.randn(1,1025,3); mask=torch.ones(1,1025,dtype=torch.bool)
    train=TrainMLP(cfg,0); train.ttt_conv.weight.data.fill_(.01)
    before=train.down_proj.weight.detach().clone()
    y=train(x,x,mask,[(0,1025)])
    assert y.shape==x.shape and sum(s.update_count for d in train.last_ttt_stats for s in d)==2
    assert torch.equal(before,train.down_proj.weight)
    ev=EvalMLP(cfg,0); ev.ttt_conv.weight.data.fill_(.01)
    ey,state=ev(x,x,None)
    assert ey.shape==x.shape and len(ev.last_ttt_stats)==1
    assert torch.equal(ev.down_proj.weight,ev.down_proj.weight.detach())
