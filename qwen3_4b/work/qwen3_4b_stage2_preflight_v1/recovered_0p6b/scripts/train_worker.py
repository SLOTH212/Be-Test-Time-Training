#!/usr/bin/env python3
import argparse, importlib.util, json, math, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'lib'),str(ROOT/'source/ntp'),str(ROOT/'source/loss')]
from common import *
from rolling_checkpoint import RollingCheckpoint, capture_rng
from answer_context_fused_loss import single_decoder_fused_answer_context

spec=importlib.util.spec_from_file_location('formal_gated',ROOT/'source/ntp/tools/cloud/formal_train_gated_ntp.py')
formal=importlib.util.module_from_spec(spec); spec.loader.exec_module(formal)


def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def json_rows(path,start=0):
    with open(path) as f:
        for i,line in enumerate(f):
            if i>=start: yield i,json.loads(line)


def stage1_input(row,tokenizer):
    ids=tokenizer(row['text'],add_special_tokens=False,return_tensors='pt').input_ids
    n=int(row['token_count'])
    if ids.shape[1]!=n or n>16384: raise RuntimeError('STAGE1_TOKEN_ALIGNMENT')
    bounds=json.loads(row['document_boundaries_qwen_json'])
    return ids.cuda(),[(int(a),int(b)) for a,b in bounds],n


def load_answers(path):
    return {r['id']:r['answer_ranges'] for r in map(json.loads,open(path))}


def stage2_input(row,tokenizer,answers):
    raw=tokenizer(row['text'],add_special_tokens=False,return_tensors='pt').input_ids[0]
    n=int(row['n_tokens'])
    if len(raw)!=n or n>16384: raise RuntimeError('STAGE2_ALIGNMENT_ERROR')
    answer=torch.zeros(n,dtype=torch.bool)
    for a,b in answers[row['id']]:
        if not 0<=a<b<=n: raise RuntimeError('STAGE2_ALIGNMENT_ERROR')
        answer[a:b]=True
    valid=torch.ones(n,dtype=torch.bool); context=valid & ~answer
    if not answer.any() or (answer&context).any() or not torch.equal(answer|context,valid):
        raise RuntimeError('MASK_CONTRACT')
    pad=16384-n
    batch={'input_ids':F.pad(raw,(0,pad),value=tokenizer.eos_token_id).unsqueeze(0).cuda(),
           'attention_mask':F.pad(valid.long(),(0,pad),value=0).unsqueeze(0).cuda(),
           'labels':F.pad(raw,(0,pad),value=-100).unsqueeze(0).cuda(),
           'document_boundaries':[(0,n)]}
    return batch,F.pad(answer,(0,pad)).unsqueeze(0).cuda(),F.pad(context,(0,pad)).unsqueeze(0).cuda(),n


def write_log(path,row):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(row,sort_keys=True)+'\n'); f.flush(); os.fsync(f.fileno())


def validate_stage1_live(model,tokenizer,path,limit,label,step,tokens):
    versions={n:p._version for n,p in model.named_parameters()}
    cpu_rng=torch.get_rng_state(); cuda_rng=torch.cuda.get_rng_state_all()
    was_training=model.training; model.eval(); losses=[]; count=0; started=time.time()
    try:
        with torch.no_grad():
            for _,row in json_rows(path):
                if count>=limit: break
                ids,bounds,_=stage1_input(row,tokenizer)
                decoded=model.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,
                                    return_dict=True,document_boundaries=bounds).last_hidden_state
                hidden=decoded[:,:-1,:].contiguous(); labels=ids[:,1:].contiguous()
                fused=LigerFusedLinearCrossEntropyLoss(reduction='sum',accum_dtype=torch.float32)
                summed=fused(model.lm_head.weight,hidden.reshape(-1,hidden.size(-1)),labels.reshape(-1))
                losses.append(float(summed/labels.numel())); count+=1
                del ids,decoded,hidden,labels,summed
    finally:
        torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state_all(cuda_rng)
        if was_training:model.train()
    if any(p._version!=versions[n] for n,p in model.named_parameters()):
        raise RuntimeError('VALIDATION_MUTATED_WEIGHTS')
    result={'label':label,'global_step':step,'cumulative_tokens':tokens,'records':count,
            'mean_loss':sum(losses)/len(losses),'optimizer_steps':0,'weights_saved':False,
            'weights_unchanged':True,'rng_restored':True,'early_stop_action':False,
            'elapsed_seconds':time.time()-started,'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    atomic_json(ROOT/f'periodic_validation/{label}.json',result)
    write_log(ROOT/'periodic_validation/metrics.jsonl',result)
    return result


def checkpoint_payload(stage,model,optimizer,scheduler,cursor,tokens,micro,config,config_sha,
                       tokenizer_sha,source_sha,dataset_sha,base_sha,metrics):
    return {'format_version':1,'run_id':RUN_ID,'stage':stage,'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
            'global_step':cursor,'micro_step':micro,'gradient_accumulation_phase':0,
            'record_cursor':cursor,'cumulative_tokens':tokens,
            'dataloader_state':{'record_cursor':cursor,'next_record_index':cursor},
            'sampler_state':{'sequential':True,'cursor':cursor},**capture_rng(),
            'config':config,'config_sha256':config_sha,'tokenizer_sha256':tokenizer_sha,
            'source_sha256':source_sha,'dataset_manifest_sha256':dataset_sha,
            'base_model_sha256':base_sha,'metrics':metrics,
            'ttt_contract':{'lr':1.0,'chunk':1024,'layers':LAYERS,'gate_saved':True,
                            'projection_saved':True,'operation_order':'apply_then_update'},
            'complete':True}


def should_save(stage,cursor,final_cursor,smoke):
    if smoke or cursor==final_cursor:return True
    if stage=='stage1':return cursor==100 or (cursor>=1000 and cursor%1000==0)
    return cursor in (50,500,1000)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',choices=('stage1','stage2'),required=True)
    ap.add_argument('--config',required=True); ap.add_argument('--output-dir',required=True)
    ap.add_argument('--state-file',required=True); ap.add_argument('--resume-root',required=True)
    ap.add_argument('--parent-checkpoint'); ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--formal-run',action='store_true'); ap.add_argument('--max-records',type=int)
    args=ap.parse_args()
    if args.smoke==args.formal_run: raise SystemExit('exactly one of --smoke/--formal-run required')
    if args.formal_run and args.max_records is not None: raise SystemExit('formal run forbids record limits')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA_REQUIRED')
    config=load_yaml(args.config); stage=args.stage; output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True)
    seed_all(int(config['seed'])); tokenizer=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    model,cfg=formal.build(BASE); cfg.ttt_lr=1.0; cfg.ttt_update_clip_norm=None
    if cfg.hidden_size!=1024 or cfg.intermediate_size!=3072 or cfg.num_hidden_layers!=28:
        raise RuntimeError('MODEL_SCALE_CONFIG_MISMATCH')
    conv_count={'value':0}
    hooks=[model.model.layers[l].mlp.ttt_conv.register_forward_hook(lambda m,i,o:conv_count.__setitem__('value',conv_count['value']+1)) for l in LAYERS]
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(config['outer_lr']),betas=tuple(config['betas']),
                                eps=float(config['eps']),weight_decay=float(config['weight_decay']),fused=True)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.0)
    config_sha=sha256(args.config); source_sha,_=source_identity(); dataset_sha=sha256(config['dataset_manifest'])
    tokenizer_sha=sha256(BASE/'tokenizer.json'); base_sha=sha256(BASE/'model.safetensors')
    identity={'run_id':RUN_ID,'stage':stage,'config_sha256':config_sha,
              'dataset_manifest_sha256':dataset_sha,'base_model_sha256':base_sha,'source_sha256':source_sha}
    rotation=RollingCheckpoint(args.resume_root,identity); resumed=rotation.load(model,optimizer,scheduler)
    if stage=='stage2' and resumed is None:
        if not args.parent_checkpoint: raise RuntimeError('STAGE1_PARENT_REQUIRED')
        parent_file=Path(args.parent_checkpoint)/'checkpoint.pt'
        if not parent_file.is_file(): raise RuntimeError('STAGE1_PARENT_MISSING')
        parent=torch.load(parent_file,map_location='cpu',weights_only=False)
        terminal_ok=(parent.get('record_cursor')==21002) if args.formal_run else (int(parent.get('record_cursor',0))>0)
        if parent.get('run_id')!=RUN_ID or parent.get('stage')!='stage1' or not terminal_ok:
            raise RuntimeError('STAGE1_PARENT_IDENTITY')
        model.load_state_dict(parent['model'],strict=True); del parent
    cursor=int(resumed['record_cursor']) if resumed else 0; tokens=int(resumed['cumulative_tokens']) if resumed else 0
    micro=int(resumed['micro_step']) if resumed else 0; metrics=dict(resumed.get('metrics',{})) if resumed else {}
    final_cursor=int(config['record_cursor_final']); stop=min(final_cursor,cursor+args.max_records) if args.max_records else final_cursor
    masks=load_answers(config['answer_mask_sidecar']) if stage=='stage2' else None
    initialization='resume_checkpoint' if resumed else ('base_model' if stage=='stage1' else 'stage1_terminal_weights')
    state={'state':stage.upper()+'_INITIALIZING','run_id':RUN_ID,'stage':stage,'worker_pid':os.getpid(),
           'initialization_source':initialization,'record_cursor':cursor,'cumulative_tokens':tokens,
           'global_step':cursor,'micro_step':micro,'expected_total_steps':final_cursor,'target_tokens':int(config.get('train_tokens',config.get('sequence_tokens'))),
           'outer_lr':float(config['outer_lr']),'ttt_lr':1.0,'ttt_chunk_size':1024,'ttt_layers':LAYERS,
           'context_length':16384,'precision':'bf16','formal_run':args.formal_run,'smoke':args.smoke}
    atomic_json(args.state_file,state); print(json.dumps({'runtime_event':'worker_initialized',**state}),flush=True)
    milestones={}
    if stage=='stage1':
        manifest=json.loads((ROOT/'manifests/periodic_validation_milestones.json').read_text())
        milestones={int(x['deterministic_step']):x for x in manifest['milestones']}
    answer_total=int(metrics.get('answer_tokens',0)); context_total=int(metrics.get('context_tokens',0))
    last_checkpoint=Path(args.resume_root)/'current' if resumed else None
    for index,row in json_rows(config['train_data'],cursor):
        if index>=stop:break
        optimizer.zero_grad(set_to_none=True); started=time.time(); torch.cuda.reset_peak_memory_stats()
        if stage=='stage1':
            ids,bounds,n=stage1_input(row,tokenizer)
            result=model(input_ids=ids,attention_mask=torch.ones_like(ids),labels=ids,use_cache=False,document_boundaries=bounds)
            loss=result.loss; answer_loss=context_loss=None; counts={}
        else:
            batch,answer,context,n=stage2_input(row,tokenizer,masks)
            loss,answer_loss,context_loss,counts=single_decoder_fused_answer_context(model,batch,answer,context)
            answer_total+=counts['answer_positions']; context_total+=counts['context_positions']
        if not torch.isfinite(loss):raise RuntimeError('NONFINITE_LOSS')
        loss.backward(); pre=torch.nn.utils.clip_grad_norm_(model.parameters(),float(config['outer_grad_clip']))
        if not torch.isfinite(pre):raise RuntimeError('NONFINITE_GRADIENT')
        optimizer.step();scheduler.step();cursor=index+1;tokens+=n;micro+=1;torch.cuda.synchronize()
        if conv_count['value']!=0:raise RuntimeError(f'CONV_FORWARD_COUNT:{conv_count["value"]}')
        log={'record_id':row.get('sample_id',row.get('id')),'global_step':cursor,'micro_step':micro,
             'record_cursor':cursor,'cumulative_tokens':tokens,'total_loss':float(loss.detach()),
             'answer_loss':None if answer_loss is None else float(answer_loss.detach()),
             'context_loss':None if context_loss is None else float(context_loss.detach()),**counts,
             'answer_tokens':answer_total,'context_tokens':context_total,'grad_norm_before_clip':float(pre),
             'grad_norm_after_clip':min(float(pre),float(config['outer_grad_clip'])),'learning_rate':scheduler.get_last_lr()[0],
             'step_time':time.time()-started,'gpu_allocated':torch.cuda.memory_allocated(),
             'gpu_reserved':torch.cuda.memory_reserved(),'gpu_peak_allocated':torch.cuda.max_memory_allocated(),
             'gpu_peak_reserved':torch.cuda.max_memory_reserved(),'optimizer_step_completed':True,
             'ttt_lr':1.0,'ttt_chunk_size':1024,'ttt_layers':LAYERS,'context_length':16384,'precision':'bf16','conv_forward_count':0}
        write_log(output/'training.jsonl',log); metrics={'last':log,'answer_tokens':answer_total,'context_tokens':context_total}
        if stage=='stage1' and cursor in milestones and not (ROOT/f"periodic_validation/{milestones[cursor]['label']}.json").exists():
            metrics['last_validation']=validate_stage1_live(model,tokenizer,config['validation_data'],int(config['validation_records']),
                                                             milestones[cursor]['label'],cursor,tokens)
        if should_save(stage,cursor,final_cursor,args.smoke):
            last_checkpoint=rotation.save(checkpoint_payload(stage,model,optimizer,scheduler,cursor,tokens,micro,config,config_sha,
                                                              tokenizer_sha,source_sha,dataset_sha,base_sha,metrics))
        runtime={**state,'state':stage.upper()+'_RUNNING','record_cursor':cursor,'global_step':cursor,'micro_step':micro,
                 'cumulative_tokens':tokens,'last_loss':log['total_loss'],'last_grad_norm':log['grad_norm_before_clip'],
                 'learning_rate':log['learning_rate'],'gpu_allocated':log['gpu_allocated'],'gpu_reserved':log['gpu_reserved'],
                 'last_progress_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'last_checkpoint':str(last_checkpoint) if last_checkpoint else None}
        atomic_json(args.state_file,runtime);print(json.dumps({'runtime_event':'optimizer_step',**log}),flush=True)
    if cursor!=stop:raise RuntimeError('CURSOR_STOP_MISMATCH')
    if args.smoke:
        print(json.dumps({'status':'SMOKE_PASS','stage':stage,'cursor':cursor,'tokens':tokens,
                          'checkpoint':str(last_checkpoint),'conv_forward_count':conv_count['value']}));return
    expected=int(config.get('train_tokens',config.get('sequence_tokens')))
    if cursor!=final_cursor or tokens!=expected:raise RuntimeError('FORMAL_FINAL_BOUNDARY_MISMATCH')
    if last_checkpoint is None:raise RuntimeError('FINAL_CHECKPOINT_MISSING')
    if stage=='stage1':
        terminal=milestones[final_cursor]
        if not (ROOT/f"periodic_validation/{terminal['label']}.json").exists():
            metrics['last_validation']=validate_stage1_live(model,tokenizer,config['validation_data'],int(config['validation_records']),
                                                             terminal['label'],cursor,tokens)
    elif (answer_total,context_total)!=(int(config['answer_tokens']),int(config['context_tokens'])):
        raise RuntimeError('STAGE2_LOSS_RUNTIME_TOTALS')
    atomic_json(args.state_file,{**state,'state':stage.upper()+'_COMPLETE','record_cursor':cursor,
                                'global_step':cursor,'cumulative_tokens':tokens,'last_checkpoint':str(last_checkpoint)})
    print(json.dumps({'status':'FORMAL_STAGE_COMPLETE','stage':stage,'cursor':cursor,'tokens':tokens}),flush=True)


if __name__=='__main__':main()
