import torch
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss

def unwrap_model(model):
    while hasattr(model,'module'): model=model.module
    return model

def single_decoder_fused_answer_context(model,batch,answer_mask,context_mask,ignore_index=-100,answer_weight=1.0,context_weight=0.1):
    core=unwrap_model(model); labels=batch['labels']; decoder_kwargs={k:v for k,v in batch.items() if k!='labels'}
    outputs=core.model(**decoder_kwargs,use_cache=False,return_dict=True)
    hidden=outputs.last_hidden_state
    sh=hidden[:,:-1,:].contiguous(); sl=labels[:,1:].contiguous(); am=(answer_mask[:,1:] & sl.ne(ignore_index)); cm=(context_mask[:,1:] & sl.ne(ignore_index))
    if torch.any(am & cm): raise RuntimeError('ANSWER_CONTEXT_MASK_OVERLAP')
    ac=am.sum();cc=cm.sum()
    if int(ac.detach())==0: raise RuntimeError('ZERO_ANSWER_DENOMINATOR')
    if int(cc.detach())==0: raise RuntimeError('ZERO_CONTEXT_DENOMINATOR')
    al=sl.masked_fill(~am,ignore_index);cl=sl.masked_fill(~cm,ignore_index)
    fused=LigerFusedLinearCrossEntropyLoss(ignore_index=ignore_index,reduction='sum',accum_dtype=torch.float32)
    af=fused(core.lm_head.weight,sh.reshape(-1,sh.size(-1)),al.reshape(-1));cf=fused(core.lm_head.weight,sh.reshape(-1,sh.size(-1)),cl.reshape(-1))
    a=af/ac.to(af.dtype);c=cf/cc.to(cf.dtype);total=answer_weight*a+context_weight*c
    return total,a,c,{'answer_positions':int(ac.detach()),'context_positions':int(cc.detach()),'decoder_forward_count':1,'output_head_calls':2,'loss_backend':'C_same_hidden_dual_fused_sum','full_logits_materialized':False,'maximum_logits_chunk_shape':None}
