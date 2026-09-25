#!/usr/bin/env python3
"""Resume-safe single-mode wrapper around the validated fixed inference authority."""
import argparse, importlib.util, json, math, os, sys, time
from datetime import datetime, timezone
from pathlib import Path

AUTHORITY=Path('/home/USER/ttt/scripts/run_fixed_inference.py')
spec=importlib.util.spec_from_file_location('fixed_authority',AUTHORITY)
impl=importlib.util.module_from_spec(spec); spec.loader.exec_module(impl)
torch=impl.torch; AutoTokenizer=impl.AutoTokenizer; Qwen3Config=impl.Qwen3Config; Qwen3ForCausalLM=impl.Qwen3ForCausalLM
MODES=impl.MODES; LAYER_MAP=impl.LAYER_MAP

def now(): return datetime.now(timezone.utc).astimezone().isoformat()
def atomic_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name('.'+path.name+f'.tmp.{os.getpid()}')
    with tmp.open('x',encoding='utf-8') as f: json.dump(obj,f,ensure_ascii=False,sort_keys=True,separators=(',',':')); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)
    fd=os.open(path.parent,os.O_DIRECTORY); os.fsync(fd); os.close(fd)
def read_json(path,default=None):
    try:return json.loads(Path(path).read_text())
    except FileNotFoundError:return default
def commit_name(index,row): return f"{index:06d}_{row['task']}_{row['sample_index']:04d}.json"
def valid_commit(r,mode,expected,ids):
    required=['sample_id','task','task_sample_index','mode','prediction','reference','score','status','model_hash','checkpoint_hash','benchmark_hash','config_hash','scorer_hash']
    return all(k in r for k in required) and r['status']=='SUCCESS' and r['mode']==mode and r['sample_id']==expected['sample_id'] and r['task']==expected['task'] and r['task_sample_index']==expected['sample_index'] and isinstance(r['prediction'],str) and isinstance(r['score'],(int,float)) and math.isfinite(r['score']) and r['model_hash']==ids['model'] and r['checkpoint_hash']==ids['checkpoint'] and r['benchmark_hash']==ids['benchmark'] and r['config_hash']==ids['config'] and r['scorer_hash']==ids['scorer']
def update_status(path,updates):
    state=read_json(path,{}) or {}; state.update(updates); atomic_json(path,state)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',required=True,choices=MODES); ap.add_argument('--benchmark-root',required=True); ap.add_argument('--checkpoint',required=True); ap.add_argument('--mode-root',required=True); ap.add_argument('--pipeline-status',required=True); ap.add_argument('--mode-status',required=True); ap.add_argument('--config-hash',required=True); ap.add_argument('--benchmark-hash',required=True); ap.add_argument('--scorer-hash',required=True); ap.add_argument('--completed-cells-base',type=int,required=True); args=ap.parse_args()
    mode=args.mode; bench=Path(args.benchmark_root).resolve(); ckpt=Path(args.checkpoint).resolve(); mode_root=Path(args.mode_root).resolve(); commits=mode_root/'commits'; commits.mkdir(parents=True,exist_ok=True)
    if impl.sha(ckpt/'model.safetensors')!=impl.MODEL_HASH: raise RuntimeError('model hash mismatch')
    manifest_hash=impl.sha(bench/'benchmark_manifest.json'); expected_manifest=(bench/'benchmark_manifest.sha256').read_text().split()[0]
    if manifest_hash!=expected_manifest: raise RuntimeError('benchmark manifest mismatch')
    scorer_path=impl.ROOT/'code/ruler/tools/common.py'; scorer_spec=importlib.util.spec_from_file_location('frozen_scorer',scorer_path); scorer=importlib.util.module_from_spec(scorer_spec); scorer_spec.loader.exec_module(scorer)
    if impl.sha(scorer_path)!=args.scorer_hash: raise RuntimeError('scorer hash mismatch')
    pairs=list(impl.load_rows(bench)); assert len(pairs)==6500 and len({r['sample_id'] for _,r in pairs})==6500
    ids={'model':impl.MODEL_HASH,'checkpoint':impl.CHECKPOINT_ID,'benchmark':args.benchmark_hash,'config':args.config_hash,'scorer':args.scorer_hash}
    complete={}; invalid=[]
    for i,(task,row) in enumerate(pairs):
        p=commits/commit_name(i,row)
        if not p.exists():continue
        try:r=json.loads(p.read_text())
        except Exception as e:invalid.append((str(p),repr(e)));continue
        if valid_commit(r,mode,row,ids):complete[row['sample_id']]=r
        else:invalid.append((str(p),'contract mismatch'))
    if invalid: raise RuntimeError(f'invalid existing commits: {invalid[:3]}')
    if len(complete)!=len(list(commits.glob('*.json'))): raise RuntimeError('duplicate/unmapped commit files')
    started=now(); update_status(args.mode_status,{'mode':mode,'status':'RUNNING','completed':len(complete),'expected':6500,'errors':0,'start_time':read_json(args.mode_status,{}).get('start_time') or started,'last_progress_time':started,'end_time':None})
    update_status(args.pipeline_status,{'pipeline_status':'RUNNING','current_mode':mode,'mode_expected_samples':6500,'mode_completed_samples':len(complete),'worker_pid':os.getpid(),'last_progress_timestamp':started,'total_completed_cells':args.completed_cells_base+len(complete)})
    cfg=Qwen3Config.from_pretrained(ckpt,local_files_only=True); cfg.ttt_mode=True; cfg.ttt_layers=[0,6,12,18,24]; cfg.ttt_chunk=1024; cfg.ttt_lr=1.0; cfg.ttt_target='hidden_states'; cfg.ttt_target_type='gated_next_position_hidden'; cfg.ttt_proj=True; cfg.ttt_ntp_gate=True; cfg.ttt_update_clip_norm=1e-5; cfg.use_cache=True
    tok=AutoTokenizer.from_pretrained(ckpt,local_files_only=True); tok.padding_side='left'
    model=Qwen3ForCausalLM.from_pretrained(ckpt,config=cfg,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval(); base=impl.base_hash(model)
    used=set(LAYER_MAP[mode]); stashed={}; previous=model.model.ttt_mode
    for layer in [0,6,12,18,24]:
        if layer in used:continue
        mlp=model.model.layers[layer].mlp; stashed[layer]={}
        for name in ('ttt_conv','ttt_proj','ttt_ntp_gate'):
            if hasattr(mlp,name):stashed[layer][name]=getattr(mlp,name);delattr(mlp,name)
    if not used:model.model.ttt_mode=False
    for i,(task,row) in enumerate(pairs):
        if row['sample_id'] in complete:continue
        encoded0=tok(row['input'],return_tensors='pt',add_special_tokens=True,truncation=False); n=int(encoded0.input_ids.shape[1]); encoded={k:v.cuda() for k,v in encoded0.items()}; torch.cuda.reset_peak_memory_stats(); t0=time.perf_counter()
        with torch.inference_mode(): generated=model.generate(**encoded,max_new_tokens=row['max_new_tokens'],do_sample=False,num_beams=1,use_cache=True,pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize(); pred=tok.decode(generated[0,n:],skip_special_tokens=True); score=float(scorer.per_sample_score(task['family'],pred,row['reference']))
        if not math.isfinite(score):raise RuntimeError('nonfinite score')
        rec={'global_index':i,'sample_id':row['sample_id'],'task':row['task'],'task_sample_index':row['sample_index'],'mode':mode,'prediction':pred,'reference':row['reference'],'score':score,'status':'SUCCESS','model_hash':impl.MODEL_HASH,'checkpoint_hash':impl.CHECKPOINT_ID,'benchmark_hash':args.benchmark_hash,'benchmark_manifest_hash':manifest_hash,'config_hash':args.config_hash,'scorer_hash':args.scorer_hash,'decoding_config_hash':impl.canon({'do_sample':False,'num_beams':1,'use_cache':True,'max_new_tokens':row['max_new_tokens'],'pad_token_id':tok.eos_token_id}),'completion_timestamp':now(),'prompt_token_count':n,'elapsed_seconds':time.perf_counter()-t0,'gpu_peak_memory_bytes':int(torch.cuda.max_memory_allocated()),'ttt_layers':LAYER_MAP[mode],'ttt_chunk':1024,'ttt_lr':None if mode=='OFF' else 1.0,'delta_clip_frobenius':None if mode=='OFF' else 1e-5,'reset_scope':'PER_SAMPLE','cross_sample_state_carry':False}
        atomic_json(commits/commit_name(i,row),rec); complete[row['sample_id']]=rec
        update_status(args.mode_status,{'completed':len(complete),'last_progress_time':rec['completion_timestamp'],'last_successful_sample_id':row['sample_id'],'current_task':row['task']})
        update_status(args.pipeline_status,{'current_task':row['task'],'mode_completed_samples':len(complete),'total_completed_cells':args.completed_cells_base+len(complete),'last_progress_timestamp':rec['completion_timestamp'],'last_successful_sample_id':row['sample_id'],'worker_pid':os.getpid()})
        if len(complete)%25==0:print(json.dumps({'mode':mode,'completed':len(complete),'sample_id':row['sample_id']}),flush=True)
    if impl.base_hash(model)!=base:raise RuntimeError('base weight mutation')
    model.model.ttt_mode=previous
    for layer,attrs in stashed.items():
        for name,value in attrs.items():setattr(model.model.layers[layer].mlp,name,value)
    ordered=[]
    for i,(task,row) in enumerate(pairs):ordered.append(json.loads((commits/commit_name(i,row)).read_text()))
    if len(ordered)!=6500 or len({x['sample_id'] for x in ordered})!=6500:raise RuntimeError('mode completion validation failed')
    result=mode_root/'results.jsonl'; tmp=result.with_name('.'+result.name+f'.tmp.{os.getpid()}')
    with tmp.open('x',encoding='utf-8') as f:
        for r in ordered:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,result); ended=now(); update_status(args.mode_status,{'status':'COMPLETE','completed':6500,'last_progress_time':ended,'end_time':ended,'worker_pid':None}); update_status(args.pipeline_status,{'mode_completed_samples':6500,'total_completed_cells':args.completed_cells_base+6500,'last_progress_timestamp':ended,'worker_pid':None})
    print(json.dumps({'status':'PASS','mode':mode,'completed':6500,'output':str(result)}))
if __name__=='__main__':main()
