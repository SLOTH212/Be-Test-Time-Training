

@torch.inference_mode()
def run_prompt(ctx, model, tok, row, sequence, tau, inject=None, donor=None,
               trace_callback=None, return_cache=False, generate=False):
    """Authoritative formal prompt execution, optionally followed by greedy generation.

    No scorer is invoked. This function is therefore safe for the pre-reward structural stage.
    """
    used, stashed = c.configure(model, sequence, tau, inject, trace_callback)
    old = model.model.ttt_mode
    if not used: model.model.ttt_mode = False
    enc = tok(row['input'], return_tensors='pt', add_special_tokens=True, truncation=False)
    ids = enc.input_ids.cuda(); n = ids.shape[1]
    mask = torch.ones((1,n), dtype=torch.long, device='cuda')
    cache = c.PrefixKVInjectCache(config=model.config, donor_prefix=donor, prefix_len=tau*1024)
    try:
        out = model(input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=True, logits_to_keep=1)
    finally:
        model.model.ttt_mode = old; c.restore(model, stashed)
    boundary = {l:(model.model.layers[l].mlp._cl_boundary['after'] if l in used else base(model,l)) for l in LAYERS}
    final = {l:(model.model.layers[l].mlp._cl_final if l in used else base(model,l)) for l in LAYERS}
    prompt_cache = {i:(cache.layers[i].keys.detach().clone(),cache.layers[i].values.detach().clone()) for i in range(len(cache.layers))} if return_cache else None
    occupancy = {l:cache.ttt_states[l][2] is not None for l in LAYERS}
    ttt_hash = {l:(th(cache.ttt_states[l][2]) if cache.ttt_states[l][2] is not None else None) for l in LAYERS}
    logits = out.logits.detach().cpu().clone()
    generation_trace=[]; generated=[]; pred_hash=''; next_id=int(out.logits[0,-1].argmax())
    generation_trace.append({'generation_step':1,'next_token_id':next_id,'logits_hash':th(out.logits),'logits_max':float(out.logits.max()),'logits_argmax':next_id})
    if generate:
        for _ in range(int(row['max_new_tokens'])):
            generated.append(next_id)
            if next_id==tok.eos_token_id: break
            one=torch.tensor([[next_id]],device='cuda'); am=torch.ones((1,n+len(generated)),dtype=torch.long,device='cuda')
            out=model(input_ids=one,attention_mask=am,past_key_values=cache,use_cache=True,logits_to_keep=1)
            next_id=int(out.logits[0,-1].argmax())
            generation_trace.append({'generation_step':len(generated)+1,'next_token_id':next_id,'logits_hash':th(out.logits),'logits_max':float(out.logits.max()),'logits_argmax':next_id})
        pred=tok.decode(generated,skip_special_tokens=True); pred_hash=hashlib.sha256(pred.encode()).hexdigest()
    return {
        'prompt_token_n':n, 'input_ids_hash':hashlib.sha256(ids.detach().cpu().numpy().tobytes()).hexdigest(),
        'attention_mask_hash':hashlib.sha256(mask.detach().cpu().numpy().tobytes()).hexdigest(),
        'boundary':boundary, 'final':final, 'prompt_cache':prompt_cache,
        'prompt_ttt_occupancy':occupancy, 'prompt_ttt_hash':ttt_hash,
        'prompt_logits':logits, 'prompt_logits_hash':th(logits),
        'generation_trace':generation_trace, 'prediction_hash':pred_hash,
        'cache_replacement_records':cache.replacement_records, 'used_layers':sorted(used),
        'generation_update_count':0,
    }


def run_reward(ctx, model, tok, row, sequence, tau, inject=None, donor=None):
    """Secondary terminal reward; call only after the structural freeze audit exists."""
    return c.run_branch(ctx,model,tok,row,sequence,tau,inject=inject,donor_prefix=donor,return_cache=False)
