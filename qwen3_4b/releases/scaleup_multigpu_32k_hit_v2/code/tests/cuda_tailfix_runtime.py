#!/usr/bin/env python3
import argparse, importlib.util, json, sys, types
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('eval_worker',ROOT/'workers/eval_worker.py')
worker=importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--output',required=True)
    args=ap.parse_args(); checkpoint=Path(args.checkpoint)
    model,cfg=worker.load_model(checkpoint,'on')
    assert cfg.ttt_lr==1 and abs(cfg.ttt_update_clip_norm-1e-5)<1e-12
    generation_deltas=[]
    originals={}
    for layer in worker.LAYERS:
        mlp=model.model.layers[layer].mlp; originals[layer]=mlp.forward
        original=mlp.forward
        def wrapped(self,x,t=None,past_w=None,enabled=True,_original=original,_layer=layer):
            if t is None and past_w is not None:
                generation_deltas.append({'layer':_layer,'norm':float(torch.linalg.vector_norm((past_w-self.down_proj.weight).float()).item())})
            return _original(x,t=t,past_w=past_w,enabled=enabled)
        mlp.forward=types.MethodType(wrapped,mlp)
    cases=[]
    for length in (1023,1024,1025,2048,2049):
        generation_deltas.clear()
        for layer in worker.LAYERS: model.model.layers[layer].mlp.last_ttt_stats=[]
        ids=torch.full((1,length),100,dtype=torch.long,device='cuda'); mask=torch.ones_like(ids)
        before=worker.base_fast_hash(model)
        with torch.no_grad():
            output=model.generate(input_ids=ids,attention_mask=mask,max_new_tokens=2,do_sample=False,num_beams=1,use_cache=True,
                                  pad_token_id=0,eos_token_id=-1)
        torch.cuda.synchronize(); audit=worker.stats(model,length,'on'); after=worker.base_fast_hash(model)
        expected=(length//1024)*len(worker.LAYERS)
        retained=(len(generation_deltas)==len(worker.LAYERS) and
                  (all(x['norm']>0 for x in generation_deltas) if expected else all(x['norm']==0 for x in generation_deltas)))
        passed=(audit['ttt_update_count']==expected and audit['generation_update_count']==0 and before==after and retained and output.shape[1]==length+2)
        cases.append({'prompt_tokens':length,'complete_chunks':length//1024,'tail_tokens':length%1024,
          'expected_update_count':expected,'actual_update_count':audit['ttt_update_count'],'generation_update_count':0,
          'generation_fast_weight_delta_norms':generation_deltas.copy(),'prompt_fast_weight_retained_and_used':retained,
          'base_weight_hash_unchanged':before==after,'max_post_clip_norm':audit['post_clip_delta_norm_max'],'passed':passed})
    result={'status':'passed' if all(x['passed'] for x in cases) else 'failed','ttt_lr':cfg.ttt_lr,
      'delta_clip_frobenius':cfg.ttt_update_clip_norm,'delta_clip_tolerance':worker.DELTA_CLIP_TOLERANCE,
      'generation_update_count':0,'conv_forward_count':model._ttt_conv_forward_count,'cases':cases}
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
    if result['status']!='passed': raise SystemExit(1)

if __name__=='__main__': main()
