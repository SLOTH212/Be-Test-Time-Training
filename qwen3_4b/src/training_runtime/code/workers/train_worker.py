#!/usr/bin/env python3
import argparse, importlib.util, json, math, os, random, shutil, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.expandvars('${TTT_PROJECT_ROOT}'))
BASE = Path(os.path.expandvars('${TTT_MODEL_ROOT}/Qwen3-1.7B-Base'))
sys.path[:0] = [str(ROOT), str(PROJECT), os.path.expandvars('${TTT_OPENCOMPASS_ROOT}/tools/autodl/v2_full_epoch_ruler_bs2')]
from lib.pipeline_core import atomic_json, load_yaml, sha256
from lib.real_checkpoint import RealCheckpointRotation, capture_rng
from answer_context_fused_loss import single_decoder_fused_answer_context
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss

spec = importlib.util.spec_from_file_location('formal_gated', PROJECT/'tools/cloud/formal_train_gated_ntp.py')
formal = importlib.util.module_from_spec(spec); spec.loader.exec_module(formal)

def json_rows(path, start=0):
    with open(path) as handle:
        for index, line in enumerate(handle):
            if index >= start:
                yield index, json.loads(line)

def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def stage1_input(row, tokenizer):
    ids = tokenizer(row['text'], add_special_tokens=False, return_tensors='pt').input_ids
    declared = int(row['token_count'])
    if ids.shape[1] != declared or declared > 16384:
        raise RuntimeError('STAGE1_TOKEN_ALIGNMENT')
    bounds = json.loads(row['document_boundaries_qwen_json'])
    return ids.cuda(), [(int(a), int(b)) for a,b in bounds], declared

def load_answers(path):
    return {row['id']: row['answer_ranges'] for row in map(json.loads, open(path))}

def stage2_input(row, tokenizer, answers):
    raw = tokenizer(row['text'], add_special_tokens=False, return_tensors='pt').input_ids[0]
    n = int(row['n_tokens'])
    if len(raw) != n or n > 16384:
        raise RuntimeError('STAGE2_ALIGNMENT_ERROR')
    answer = torch.zeros(n, dtype=torch.bool)
    for a,b in answers[row['id']]:
        if not 0 <= a < b <= n: raise RuntimeError('STAGE2_ALIGNMENT_ERROR')
        answer[a:b] = True
    valid = torch.ones(n, dtype=torch.bool)
    context = valid & ~answer
    if not answer.any(): raise RuntimeError('EMPTY_ANSWER_MASK')
    if (answer & context).any() or not torch.equal(answer | context, valid): raise RuntimeError('MASK_CONTRACT')
    pad = 16384 - n
    batch = {
        'input_ids': F.pad(raw, (0,pad), value=tokenizer.eos_token_id).unsqueeze(0).cuda(),
        'attention_mask': F.pad(valid.long(), (0,pad), value=0).unsqueeze(0).cuda(),
        'labels': F.pad(raw, (0,pad), value=-100).unsqueeze(0).cuda(),
        'document_boundaries': [(0,n)],
    }
    return batch, F.pad(answer,(0,pad)).unsqueeze(0).cuda(), F.pad(context,(0,pad)).unsqueeze(0).cuda(), n

def payload(stage, model, optimizer, scheduler, cursor, tokens, micro, config, tokenizer_hash, dataset_hash, metrics):
    return {
        'format_version': 1, 'stage': stage, 'model': model.state_dict(),
        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
        'global_step': cursor, 'micro_step': micro, 'gradient_accumulation_phase': 0,
        'record_cursor': cursor, 'cumulative_tokens': tokens,
        'dataloader_state': {'record_cursor': cursor, 'next_record_index': cursor},
        'sampler_state': {'sequential': True, 'cursor': cursor},
        **capture_rng(), 'config': config, 'tokenizer_sha256': tokenizer_hash,
        'project_head': subprocess.check_output(['git','-C',str(PROJECT),'rev-parse','HEAD'],text=True).strip(),
        'dataset_manifest_sha256': dataset_hash, 'metrics': metrics,
        'ttt_contract': {'lr':1.0,'chunk':1024,'layers':[0,6,12,18,24],
                         'gate_saved':True,'projection_saved':True}, 'complete': True,
    }

def write_log(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a') as handle:
        handle.write(json.dumps(row)+'\n'); handle.flush(); os.fsync(handle.fileno())

def should_save(stage, cursor, final_cursor, smoke):
    """Preserve the audited production cadence; smoke checkpoints every step."""
    if smoke or cursor == final_cursor:
        return True
    if stage == 1:
        return cursor == 100 or (cursor >= 1000 and cursor % 1000 == 0)
    return cursor in (50, 500, 1000)

def promote(model, tokenizer, checkpoint_dir, destination, stage, metrics):
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()): raise RuntimeError('REFUSE_FINAL_ARTIFACT_OVERWRITE')
    # Hard-link the complete resume checkpoint for Stage1; Stage2 needs inference weights only.
    if stage == 1:
        for name in ('checkpoint.pt','checkpoint_manifest.json'):
            os.link(checkpoint_dir/name, destination/name)
        model.config.to_json_file(destination/'model_config.json')
        tokenizer.save_pretrained(destination/'tokenizer')
        hf = None
    else:
        hf = destination
        model.config.ttt_lr = 1.0
        model.config.ttt_update_clip_norm = 1e-5
        model.save_pretrained(hf, safe_serialization=True)
        tokenizer.save_pretrained(hf)
    atomic_json(destination/'artifact_manifest.json', {'stage':stage,'complete':True,'metrics':metrics,
      'model_sha256':sha256(destination/'checkpoint.pt') if stage==1 else sha256(hf/'model.safetensors'),
      'model_storage':'full_resume_checkpoint' if stage==1 else 'hf_safetensors',
      'ttt_lr':1.0,'delta_clip_frobenius':1e-5 if stage==2 else None})
    with open(destination/'SHA256SUMS.txt','w') as out:
        for p in sorted(x for x in destination.rglob('*') if x.is_file() and x.name!='SHA256SUMS.txt'):
            out.write(f'{sha256(p)}  {p.relative_to(destination)}\n')
    if stage == 1:
        for p in destination.rglob('*'):
            os.chmod(p,0o555 if p.is_dir() else 0o444)
        os.chmod(destination,0o555)

def validate_stage1(model, tokenizer, path, limit):
    versions = {n:p._version for n,p in model.named_parameters()}
    optimizer_steps = 0; losses=[]; count=0
    model.eval()
    with torch.no_grad():
        for _,row in json_rows(path):
            if count >= limit: break
            ids,bounds,_ = stage1_input(row,tokenizer)
            decoded=model.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,
                                return_dict=True,document_boundaries=bounds).last_hidden_state
            hidden=decoded[:,:-1,:].contiguous(); labels=ids[:,1:].contiguous()
            fused=LigerFusedLinearCrossEntropyLoss(reduction='sum',accum_dtype=torch.float32)
            summed=fused(model.lm_head.weight,hidden.reshape(-1,hidden.size(-1)),labels.reshape(-1))
            losses.append(float(summed/labels.numel())); count += 1
            del decoded, hidden, labels, summed
    model.train()
    if any(p._version != versions[n] for n,p in model.named_parameters()):
        raise RuntimeError('VALIDATION_MUTATED_WEIGHTS')
    return {'records':count,'mean_loss':sum(losses)/len(losses),'optimizer_steps':optimizer_steps,'weights_unchanged':True}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage',type=int,choices=(1,2),required=True); ap.add_argument('--config',required=True)
    ap.add_argument('--resume-from'); ap.add_argument('--output-dir',required=True); ap.add_argument('--state-file',required=True)
    ap.add_argument('--smoke',action='store_true'); ap.add_argument('--max-records',type=int); ap.add_argument('--validation-records',type=int)
    ap.add_argument('--formal-run',action='store_true'); ap.add_argument('--parent-artifact'); ap.add_argument('--promote',action='store_true')
    args=ap.parse_args()
    if args.formal_run == args.smoke: raise SystemExit('exactly one of --formal-run/--smoke is required')
    if args.formal_run and args.max_records is not None: raise SystemExit('formal run forbids smoke record limits')
    config=load_yaml(args.config); output=Path(args.output_dir); state_file=Path(args.state_file)
    output.mkdir(parents=True,exist_ok=True); seed_all(42)
    tokenizer=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    init=BASE
    if args.stage==2 and not (Path(args.parent_artifact)/'checkpoint.pt').is_file(): raise RuntimeError('STAGE1_PARENT_MISSING')
    model,cfg=formal.build(init); cfg.ttt_lr=1.0; cfg.ttt_update_clip_norm=None
    conv_forward_count={'value':0}
    def count_conv(_module,_inputs,_output): conv_forward_count['value'] += 1
    conv_hooks=[model.model.layers[layer].mlp.ttt_conv.register_forward_hook(count_conv) for layer in (0,6,12,18,24)]
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-6,betas=(0.9,0.95),eps=1e-8,weight_decay=0.1,fused=True)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.0)
    rotation=RealCheckpointRotation(Path(args.resume_from) if args.resume_from else output/'checkpoints')
    resumed=rotation.load(model,optimizer,scheduler)
    if args.stage==2 and resumed is None:
        parent=torch.load(Path(args.parent_artifact)/'checkpoint.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(parent['model'],strict=True)
        del parent
    cursor=resumed['record_cursor'] if resumed else 0; tokens=resumed['cumulative_tokens'] if resumed else 0
    micro=resumed['micro_step'] if resumed else 0; metrics=resumed.get('metrics',{}) if resumed else {}
    resumed_slot=(rotation.root/rotation.latest()['slot']) if resumed else None
    data=Path(config['train_data']); final_cursor=int(config['record_cursor_final'])
    stop=min(final_cursor,cursor+args.max_records) if args.max_records is not None else final_cursor
    masks=load_answers(config['answer_mask_sidecar']) if args.stage==2 else None
    dataset_manifest=(Path(os.path.expandvars('${TTT_DATASET_ROOT}/continual_16k_300m_prolong_v3_full/dataset_manifest.json')) if args.stage==1 else Path(os.path.expandvars('${TTT_DATASET_ROOT}/qa_replay_10m_v1_answer_context_masks_v1/derivation_manifest.json')))
    tokenizer_hash=sha256(BASE/'tokenizer.json'); last_checkpoint=resumed_slot
    initialization_source=('resume_checkpoint' if resumed else ('base_model' if args.stage==1 else 'stage1_final_model_weights'))
    resume_checkpoint=(str(resumed_slot) if resumed_slot is not None else 'none')
    startup_state={'state':f'STAGE{args.stage}_INITIALIZING','initialization_source':initialization_source,
      'resume_checkpoint':resume_checkpoint,'base_model_path':str(BASE),'base_model_hash':sha256(BASE/'model.safetensors'),
      'initial_record_cursor':cursor,'initial_cumulative_tokens':tokens,'record_cursor':cursor,'cumulative_tokens':tokens,
      'global_step':cursor,'micro_step':micro,'outer_lr':5e-6,'ttt_lr':1.0,'ttt_chunk_size':1024,
      'ttt_layers':[0,6,12,18,24],'context_length':16384,'precision':'bf16','conv_forward_count':0,
      'formal_run':args.formal_run,'smoke':args.smoke}
    atomic_json(state_file,startup_state); print(json.dumps({'runtime_event':'worker_initialized',**startup_state}),flush=True)
    answer_total=int(metrics.get('answer_tokens',0)); context_total=int(metrics.get('context_tokens',0))
    for index,row in json_rows(data,cursor):
        if index >= stop: break
        optimizer.zero_grad(set_to_none=True); started=time.time()
        if args.stage==1:
            ids,bounds,n=stage1_input(row,tokenizer)
            result=model(input_ids=ids,attention_mask=torch.ones_like(ids),labels=ids,use_cache=False,document_boundaries=bounds)
            loss=result.loss; answer_loss=context_loss=None; counts={}
        else:
            batch,answer,context,n=stage2_input(row,tokenizer,masks)
            loss,answer_loss,context_loss,counts=single_decoder_fused_answer_context(model,batch,answer,context)
            answer_total += counts['answer_positions']; context_total += counts['context_positions']
        if not torch.isfinite(loss): raise RuntimeError('NONFINITE_LOSS')
        loss.backward(); pre=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        if not torch.isfinite(pre): raise RuntimeError('NONFINITE_GRADIENT')
        optimizer.step(); scheduler.step(); cursor=index+1; tokens += n; micro += 1
        if conv_forward_count['value'] != 0: raise RuntimeError(f'CONV_FORWARD_COUNT={conv_forward_count["value"]}')
        log={'record_id':row.get('sample_id',row.get('id')),'global_step':cursor,'micro_step':micro,
          'record_cursor':cursor,'cumulative_tokens':tokens,'total_loss':float(loss.detach()),
          'answer_loss':None if answer_loss is None else float(answer_loss.detach()),
          'context_loss':None if context_loss is None else float(context_loss.detach()),**counts,
          'answer_tokens':answer_total,'context_tokens':context_total,'empty_answer_mask_count':0,
          'mask_overlap_count':0,'alignment_error_count':0,'grad_norm_before_clip':float(pre),
          'grad_norm_after_clip':min(float(pre),1.0),'learning_rate':scheduler.get_last_lr()[0],
          'step_time':time.time()-started,'gpu_allocated':torch.cuda.memory_allocated(),'gpu_reserved':torch.cuda.memory_reserved(),
          'initialization_source':initialization_source,'resume_checkpoint':resume_checkpoint,'ttt_lr':1.0,
          'ttt_chunk_size':1024,'ttt_layers':[0,6,12,18,24],'context_length':16384,'precision':'bf16',
          'conv_forward_count':conv_forward_count['value'],'optimizer_step_completed':True}
        write_log(output/'training.jsonl',log); metrics={'last':log,'answer_tokens':answer_total,'context_tokens':context_total}
        if should_save(args.stage,cursor,final_cursor,args.smoke):
            last_checkpoint=rotation.save(payload(f'stage{args.stage}',model,optimizer,scheduler,cursor,tokens,micro,config,tokenizer_hash,sha256(dataset_manifest),metrics))
        runtime_state={**startup_state,'state':f'STAGE{args.stage}_RUNNING','record_cursor':cursor,'cumulative_tokens':tokens,
          'global_step':cursor,'micro_step':micro,'last_record_id':log['record_id'],'last_loss':log['total_loss'],
          'last_grad_norm':log['grad_norm_before_clip'],'learning_rate':log['learning_rate'],
          'optimizer_step_completed':True,'gpu_allocated':log['gpu_allocated'],'gpu_reserved':log['gpu_reserved']}
        atomic_json(state_file,runtime_state); print(json.dumps({'runtime_event':'optimizer_step',**log}),flush=True)
    if cursor != stop: raise RuntimeError('CURSOR_STOP_MISMATCH')
    if args.smoke:
        val=None
        if args.stage==1 and args.validation_records:
            val=validate_stage1(model,tokenizer,Path(config['validation_data']),args.validation_records)
        destination=output/'artifacts'/('stage1_final_resume' if args.stage==1 else 'stage2_final_eval') if args.promote else None
        if destination:
            promote(model,tokenizer,last_checkpoint,destination,args.stage,{**metrics,'validation':val})
        print(json.dumps({'status':'SMOKE_PASS','stage':args.stage,'cursor':cursor,'tokens':tokens,
                          'artifact':None if destination is None else str(destination)})); return
    expected_tokens=int(config['train_tokens'] if args.stage==1 else config['sequence_tokens'])
    if cursor!=final_cursor or tokens!=expected_tokens: raise RuntimeError('FORMAL_FINAL_BOUNDARY_MISMATCH')
    if last_checkpoint is None: raise RuntimeError('FORMAL_FINAL_CHECKPOINT_MISSING')
    if args.stage==1:
        metrics['validation']=validate_stage1(model,tokenizer,Path(config['validation_data']),220)
    else:
        if (answer_total,context_total)!=(13332,9985377): raise RuntimeError('STAGE2_LOSS_RUNTIME_TOTALS')
    destination=ROOT/'artifacts'/('stage1_final_resume' if args.stage==1 else 'stage2_final_eval')
    promote(model,tokenizer,last_checkpoint,destination,args.stage,metrics)
    atomic_json(state_file,{'state':f'STAGE{args.stage}_COMPLETE','record_cursor':cursor,'cumulative_tokens':tokens})

if __name__=='__main__': main()
