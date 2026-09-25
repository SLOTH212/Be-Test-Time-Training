#!/usr/bin/env python3
import argparse, gc, hashlib, importlib.util, json, os, sys, time
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.expandvars('${TTT_PROJECT_ROOT}'))
BENCH=Path(os.path.expandvars('${TTT_BENCHMARK_ROOT}/ruler_16k_13task_1000sample_v1'))
sys.path[:0]=[str(ROOT),str(PROJECT),os.path.expandvars('${TTT_OPENCOMPASS_ROOT}/tools/autodl/ruler_benchmark_v1')]
from lib.pipeline_core import atomic_json, load_yaml, sha256
from common import input_hash, per_sample_score, weight_manifest
from inference_model.hf_qwen3.configuration_qwen3 import Qwen3Config
from inference_model.hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM

LAYERS=[0,6,12,18,24]; EXPECTED_BENCH='f921d4bbf2641b1069c925e534beee07e1a3749c4c5f221328ffe4656ab3b977'
DELTA_CLIP_TOLERANCE=1e-7  # BF16 storage/norm accumulation tolerance at a 1e-5 threshold.

def rows():
    manifest=json.load(open(BENCH/'benchmark_manifest.json'))
    for task in manifest['tasks']:
        with open(BENCH/'samples'/f"{task['name']}.jsonl") as f:
            for line in f: yield task,json.loads(line)

def base_fast_hash(model):
    h=hashlib.sha256()
    for layer in LAYERS:
        raw=model.model.layers[layer].mlp.down_proj.weight.detach().contiguous().view(torch.uint16).cpu().numpy()
        h.update(raw.tobytes())
    return h.hexdigest()

def load_model(path, mode):
    cfg=Qwen3Config.from_pretrained(path,local_files_only=True)
    cfg.ttt_mode=True; cfg.ttt_layers=LAYERS; cfg.ttt_chunk=1024; cfg.ttt_lr=1.0
    cfg.ttt_target='hidden_states'; cfg.ttt_target_type='gated_next_position_hidden'
    cfg.ttt_proj=True; cfg.ttt_ntp_gate=True; cfg.ttt_update_clip_norm=1e-5
    model=Qwen3ForCausalLM.from_pretrained(path,config=cfg,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval()
    model._ttt_conv_forward_count=0
    def count_conv(_module, _inputs, _output):
        model._ttt_conv_forward_count += 1
    model._ttt_conv_hooks=[model.model.layers[layer].mlp.ttt_conv.register_forward_hook(count_conv) for layer in LAYERS]
    if mode=='off': model.model.ttt_mode=False
    actual=model.model.ttt_mode
    if mode=='on':
        assert actual is True and cfg.ttt_lr==1 and abs(cfg.ttt_update_clip_norm-1e-5)<1e-12
        assert cfg.ttt_chunk==1024 and cfg.ttt_layers==LAYERS
    else: assert actual is False
    return model,cfg

def stats(model,prompt_tokens,mode):
    if model._ttt_conv_forward_count != 0:
        raise RuntimeError(f'CONV_FORWARD_COUNT={model._ttt_conv_forward_count}')
    if mode=='off':
        return {'ttt_update_count':0,'expected_ttt_update_count':0,'generation_update_count':0,'conv_forward_count':0,
          'online_update_path_executed':False,'inner_update_call_count':0,'fast_weight_mutation_count':0,
          'pre_clip_delta_norm_min':0,'pre_clip_delta_norm_mean':0,'pre_clip_delta_norm_max':0,
          'post_clip_delta_norm_min':0,'post_clip_delta_norm_mean':0,'post_clip_delta_norm_max':0,'clip_activation_count':0,'clip_activation_ratio':0}
    raw=[]
    for layer in LAYERS:
        raw.extend(getattr(model.model.layers[layer].mlp,'last_ttt_stats',[]) or [])
    pre=[x.pre_clip_norm for x in raw]; post=[x.post_clip_norm for x in raw]; clipped=[x.was_clipped for x in raw]
    expected=(prompt_tokens//1024)*len(LAYERS)
    if sum(x.update_count for x in raw)!=expected: raise RuntimeError('TAILFIX_RUNTIME_UPDATE_COUNT')
    if any(x>1e-5+DELTA_CLIP_TOLERANCE for x in post):
        raise RuntimeError(f'BLOCKED_INFERENCE_DELTA_CLIP_MISMATCH max={max(post)} tolerance={DELTA_CLIP_TOLERANCE}')
    return {'ttt_update_count':expected,'expected_ttt_update_count':expected,'generation_update_count':0,'conv_forward_count':0,
      'online_update_path_executed':True,'inner_update_call_count':expected,'fast_weight_mutation_count':expected,
      'pre_clip_delta_norm_min':min(pre,default=0),'pre_clip_delta_norm_mean':sum(pre)/len(pre) if pre else 0,'pre_clip_delta_norm_max':max(pre,default=0),
      'post_clip_delta_norm_min':min(post,default=0),'post_clip_delta_norm_mean':sum(post)/len(post) if post else 0,'post_clip_delta_norm_max':max(post,default=0),
      'delta_clip_tolerance':DELTA_CLIP_TOLERANCE,'clip_activation_count':sum(clipped),'clip_activation_ratio':sum(clipped)/len(clipped) if clipped else 0}

def run_one(model,tokenizer,task,row,mode,checkpoint_hash):
    if input_hash(row['input'])!=row['input_hash']: raise RuntimeError('INPUT_HASH_MISMATCH')
    encoded=tokenizer(row['input'],return_tensors='pt',add_special_tokens=True,truncation=False)
    prompt_tokens=int(encoded.input_ids.shape[1]); encoded={k:v.cuda() for k,v in encoded.items()}
    before=base_fast_hash(model); started=time.perf_counter()
    with torch.no_grad():
        output=model.generate(**encoded,max_new_tokens=row['max_new_tokens'],do_sample=False,num_beams=1,use_cache=True,pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize(); continuation=output[0,prompt_tokens:]; prediction=tokenizer.decode(continuation,skip_special_tokens=True)
    audit=stats(model,prompt_tokens,mode); after=base_fast_hash(model)
    if before!=after: raise RuntimeError('BASE_FAST_WEIGHT_MUTATED')
    return {'mode':mode,'task':row['task'],'sample_id':row['sample_id'],'input_hash':row['input_hash'],
      'checkpoint_hash':checkpoint_hash,'benchmark_hash':EXPECTED_BENCH,'prediction':prediction,'target':row['reference'],
      'prompt_hash':row['input_hash'],'decode_config_hash':hashlib.sha256(json.dumps({'do_sample':False,'num_beams':1,'max_new_tokens':row['max_new_tokens']},sort_keys=True).encode()).hexdigest(),
      'score':per_sample_score(task['family'],prediction,row['reference']),'prompt_token_count':prompt_tokens,
      'complete_prompt_chunk_count':prompt_tokens//1024,'incomplete_prompt_tail_token_count':prompt_tokens%1024,
      'ttt_enabled':mode=='on','ttt_lr':1.0 if mode=='on' else None,'delta_clip_frobenius':1e-5 if mode=='on' else None,
      'batch_size':1 if mode=='on' else 2,'fast_weight_reset_before_sample':True,'fast_weight_reset_after_error':False,
      'latency_seconds':time.perf_counter()-started,'success':True,'error':None,'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S%z'),**audit}

def run_off_batch(model,tokenizer,pairs,checkpoint_hash):
    # The frozen OFF protocol is genuinely batch=2. Duplicate only a final odd
    # resume remainder as padding; it is discarded and never counted/written.
    real_count=len(pairs)
    if real_count==1: pairs=pairs+pairs
    assert len(pairs)==2 and model.model.ttt_mode is False
    for _,row in pairs:
        if input_hash(row['input'])!=row['input_hash']: raise RuntimeError('INPUT_HASH_MISMATCH')
    encoded=tokenizer([row['input'] for _,row in pairs],return_tensors='pt',padding=True,add_special_tokens=True,truncation=False)
    padded=int(encoded.input_ids.shape[1]); prompt_counts=[int(x) for x in encoded.attention_mask.sum(1)]
    encoded={k:v.cuda() for k,v in encoded.items()}; before=base_fast_hash(model); started=time.perf_counter()
    with torch.no_grad():
        output=model.generate(**encoded,max_new_tokens=max(row['max_new_tokens'] for _,row in pairs),do_sample=False,
                              num_beams=1,use_cache=True,pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize(); latency=time.perf_counter()-started
    if base_fast_hash(model)!=before: raise RuntimeError('OFF_FAST_WEIGHT_MUTATED')
    records=[]
    for i,(task,row) in enumerate(pairs[:real_count]):
        continuation=output[i,padded:]; prediction=tokenizer.decode(continuation,skip_special_tokens=True)
        records.append({'mode':'off','task':row['task'],'sample_id':row['sample_id'],'input_hash':row['input_hash'],
          'checkpoint_hash':checkpoint_hash,'benchmark_hash':EXPECTED_BENCH,'prediction':prediction,'target':row['reference'],
          'prompt_hash':row['input_hash'],'decode_config_hash':hashlib.sha256(json.dumps({'do_sample':False,'num_beams':1,'max_new_tokens':row['max_new_tokens']},sort_keys=True).encode()).hexdigest(),
          'score':per_sample_score(task['family'],prediction,row['reference']),'prompt_token_count':prompt_counts[i],
          'complete_prompt_chunk_count':prompt_counts[i]//1024,'incomplete_prompt_tail_token_count':prompt_counts[i]%1024,
          'ttt_enabled':False,'ttt_lr':None,'delta_clip_frobenius':None,'batch_size':2,
          'fast_weight_reset_before_sample':True,'fast_weight_reset_after_error':False,'latency_seconds':latency,
          'success':True,'error':None,'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S%z'),**stats(model,prompt_counts[i],'off')})
    return records

def append_sync(path,obj):
    with open(path,'a') as f: f.write(json.dumps(obj)+'\n'); f.flush(); os.fsync(f.fileno())

def isolation_test(model,tokenizer,samples,checkpoint_path):
    task_a,row_a=samples[0]; task_b,row_b=samples[1]; disk_before=sha256(checkpoint_path/'model.safetensors')
    b1=run_one(model,tokenizer,task_b,row_b,'on','isolation')['prediction']; base=base_fast_hash(model)
    run_one(model,tokenizer,task_a,row_a,'on','isolation'); reset=base_fast_hash(model)
    b2=run_one(model,tokenizer,task_b,row_b,'on','isolation')['prediction']
    try: raise RuntimeError('simulated sample A failure')
    except RuntimeError: pass
    finally: reset_error=base_fast_hash(model)
    b3=run_one(model,tokenizer,task_b,row_b,'on','isolation')['prediction']
    disk_after=sha256(checkpoint_path/'model.safetensors')
    result={'passed':b1==b2==b3 and base==reset==reset_error and disk_before==disk_after,
      'prediction_equal':b1==b2==b3,'base_fast_weight_hash':base,'reset_hash_equal':base==reset==reset_error,
      'checkpoint_hash_unchanged':disk_before==disk_after,'tolerance':0.0}
    if not result['passed']: raise RuntimeError('BLOCKED_ON_STATE_ISOLATION')
    return result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=('off','on'),required=True); ap.add_argument('--config',required=True)
    ap.add_argument('--common-config',required=True); ap.add_argument('--resume-from'); ap.add_argument('--output-dir',required=True)
    ap.add_argument('--state-file',required=True); ap.add_argument('--smoke',action='store_true'); ap.add_argument('--max-records',type=int)
    ap.add_argument('--formal-run',action='store_true'); ap.add_argument('--checkpoint',required=True); ap.add_argument('--run-isolation-test',action='store_true')
    args=ap.parse_args()
    if args.formal_run==args.smoke: raise SystemExit('exactly one of --formal-run/--smoke required')
    if args.formal_run and args.max_records is not None: raise SystemExit('formal run forbids limit')
    if sha256(BENCH/'benchmark_manifest.json')!=EXPECTED_BENCH: raise RuntimeError('BLOCKED_EVAL13K_BENCHMARK_HASH_MISMATCH')
    spec=load_yaml(args.config); common=load_yaml(args.common_config)
    assert spec['mode']==args.mode and spec['batch_size']==(1 if args.mode=='on' else 2)
    assert common['ttt']['chunk_size']==1024 and common['ttt']['layers']==LAYERS
    assert common['ttt']['update_order']=='apply_then_update' and common['ttt']['reset_per_sample'] is True
    assert common['tailfix']['generation_online_update'] is False and common['tailfix']['generation_append_ttt_tail'] is False
    if args.mode=='on': assert spec['ttt_lr']==1.0 and abs(spec['delta_clip_frobenius']-1e-5)<1e-12
    else: assert spec['ttt_enabled'] is False and spec['online_update_path'] is False
    checkpoint=Path(args.checkpoint); checkpoint_hash=weight_manifest(checkpoint)['aggregate_sha256']; disk_hash=sha256(checkpoint/'model.safetensors')
    tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True); tokenizer.padding_side='left'
    model,cfg=load_model(checkpoint,args.mode); output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True)
    partial=output/'predictions.jsonl.partial'; completed_path=output/'completed_sample_ids.jsonl'
    completed={json.loads(x)['sample_id'] for x in open(completed_path)} if completed_path.exists() else set()
    partial_records=[json.loads(x) for x in open(partial)] if partial.exists() else []
    partial_ids=[x['sample_id'] for x in partial_records if x.get('success')]
    if len(partial_ids)!=len(set(partial_ids)): raise RuntimeError('DUPLICATE_PARTIAL_SAMPLE_IDS')
    for sample_id in partial_ids:
        if sample_id not in completed:
            append_sync(completed_path,{'sample_id':sample_id}); completed.add(sample_id)
    all_rows=list(rows()); selected=all_rows[:args.max_records] if args.max_records is not None else all_rows
    pending=[(i,task,row) for i,(task,row) in enumerate(selected) if row['sample_id'] not in completed]
    groups=([pending[i:i+2] for i in range(0,len(pending),2)] if args.mode=='off' else [[x] for x in pending])
    for group in groups:
        try:
            produced=(run_off_batch(model,tokenizer,[(t,r) for _,t,r in group],checkpoint_hash) if args.mode=='off'
                      else [run_one(model,tokenizer,group[0][1],group[0][2],args.mode,checkpoint_hash)])
        except Exception as exc:
            for _,_,row in group: append_sync(output/'errors.jsonl',{'mode':args.mode,'task':row['task'],'sample_id':row['sample_id'],'success':False,'error':f'{type(exc).__name__}: {exc}'})
            raise
        for (index,_,row),rec in zip(group,produced):
            rec['sample_index']=index; append_sync(partial,rec); append_sync(completed_path,{'sample_id':row['sample_id']}); completed.add(row['sample_id'])
            atomic_json(output/'progress.json',{'mode':args.mode,'completed':len(completed),'last_sample_id':row['sample_id']})
    isolation=isolation_test(model,tokenizer,all_rows[:2],checkpoint) if args.mode=='on' and args.run_isolation_test else None
    if sha256(checkpoint/'model.safetensors')!=disk_hash: raise RuntimeError('CHECKPOINT_DISK_MUTATED')
    expected=len(selected)
    if len(completed)!=expected: raise RuntimeError('INFERENCE_COUNT_MISMATCH')
    final=output/'predictions.jsonl'
    if final.exists(): raise RuntimeError('REFUSE_PREDICTION_OVERWRITE')
    os.replace(partial,final)
    atomic_json(output/'manifest.json',{'mode':args.mode,'success_count':len(completed),'checkpoint_hash':checkpoint_hash,
      'benchmark_hash':EXPECTED_BENCH,'ttt_enabled':args.mode=='on','ttt_lr':1 if args.mode=='on' else None,
      'delta_clip_frobenius':1e-5 if args.mode=='on' else None,'batch_size':spec['batch_size'],'isolation':isolation})
    atomic_json(args.state_file,{'state':f'EVAL_{args.mode.upper()}_COMPLETE','success_count':len(completed)})
    print(json.dumps({'status':'PASS','mode':args.mode,'success_count':len(completed),'isolation':isolation}))
if __name__=='__main__': main()
