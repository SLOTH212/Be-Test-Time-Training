#!/usr/bin/env python3
import os
import argparse, json, sys
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from workers import train_worker as worker
MODEL_BF16_TOLERANCE=2e-5
OPTIMIZER_BF16_TOLERANCE=5e-4

def compare_tensors(left,right):
    if left.shape!=right.shape or left.dtype!=right.dtype: return False,float('inf')
    if torch.equal(left.cpu(),right.cpu()): return True,0.0
    return False,float((left.detach().float().cpu()-right.detach().float().cpu()).abs().max())

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--output',required=True)
    args=ap.parse_args(); worker.seed_all(42)
    tokenizer=worker.AutoTokenizer.from_pretrained(worker.BASE,local_files_only=True)
    model,cfg=worker.formal.build(worker.BASE)
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-6,betas=(0.9,0.95),eps=1e-8,weight_decay=0.1,fused=True)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.0)
    data=Path(os.path.expandvars('${TTT_DATASET_ROOT}/continual_16k_300m_prolong_v3_full/train.jsonl'))
    ids_seen=[]
    for index,row in worker.json_rows(data):
        if index>=2: break
        optimizer.zero_grad(set_to_none=True)
        ids,bounds,_=worker.stage1_input(row,tokenizer)
        result=model(input_ids=ids,attention_mask=torch.ones_like(ids),labels=ids,use_cache=False,document_boundaries=bounds)
        if not torch.isfinite(result.loss): raise RuntimeError('NONFINITE_REFERENCE_LOSS')
        result.loss.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        if not torch.isfinite(norm): raise RuntimeError('NONFINITE_REFERENCE_GRADIENT')
        optimizer.step(); scheduler.step(); ids_seen.append(row['sample_id'])
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    max_model=0.0; model_equal=True
    for name,value in model.state_dict().items():
        equal,diff=compare_tensors(value,saved['model'][name]); model_equal &= equal; max_model=max(max_model,diff)
    reference_opt=optimizer.state_dict(); saved_opt=saved['optimizer']; optimizer_equal=reference_opt['param_groups']==saved_opt['param_groups']; max_optimizer=0.0
    for key,state in reference_opt['state'].items():
        other=saved_opt['state'][key]
        for field,value in state.items():
            if torch.is_tensor(value):
                equal,diff=compare_tensors(value,other[field]); optimizer_equal &= equal; max_optimizer=max(max_optimizer,diff)
            else: optimizer_equal &= value==other[field]
    rng=worker.capture_rng()
    rng_equal=bool(torch.equal(rng['torch_cpu_rng'],saved['torch_cpu_rng']) and
               all(torch.equal(a,b) for a,b in zip(rng['torch_cuda_rng'],saved['torch_cuda_rng'])) and
               rng['python_rng']==saved['python_rng'] and rng['numpy_rng'][0]==saved['numpy_rng'][0] and
               (rng['numpy_rng'][1]==saved['numpy_rng'][1]).all())
    within_tolerance=(max_model<=MODEL_BF16_TOLERANCE and max_optimizer<=OPTIMIZER_BF16_TOLERANCE)
    result={'status':'passed' if within_tolerance and scheduler.state_dict()==saved['scheduler'] and rng_equal else 'failed',
      'reference_record_ids':ids_seen,'saved_record_cursor':saved['record_cursor'],'model_bitwise_equal':model_equal,
      'model_max_abs_diff':max_model,'optimizer_bitwise_equal':optimizer_equal,'optimizer_max_abs_diff':max_optimizer,
      'scheduler_equal':scheduler.state_dict()==saved['scheduler'],'rng_equal':rng_equal,
      'model_bf16_tolerance':MODEL_BF16_TOLERANCE,'optimizer_bf16_tolerance':OPTIMIZER_BF16_TOLERANCE,
      'within_declared_tolerance':within_tolerance}
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))
    if result['status']!='passed': raise SystemExit(1)

if __name__=='__main__': main()
