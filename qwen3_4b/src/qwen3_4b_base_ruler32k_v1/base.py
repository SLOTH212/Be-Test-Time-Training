"""Original Qwen3-4B-Base, frozen raw RULER prompts, no TTT or trained parent."""
import os,sys,json,time,statistics,collections,subprocess,fcntl,argparse
from pathlib import Path
R=Path('/path/to/ttt');S=R/'src/qwen3_4b_base_ruler32k_v1';C=R/'configs/qwen3_4b_base_ruler32k_v1.json'
sys.path.insert(0,str(S));import data_util as util
def config(full=False):
 c=util.read(C);assert util.sha(C)==C.with_suffix('.sha256').read_text().strip()
 for p,h in c['bindings'].items():assert util.sha(p)==h,('INPUT_CHANGED',p)
 if full:
  for p,h in c['model_files'].items():assert util.sha(p)==h,('BASE_MODEL_CHANGED',p)
 assert c['model_path']=='/home/USER/data/USER/models/Qwen3-4B-Base' and c['ttt_enabled'] is False and c['excluded_tasks']==[]
 return c
def worker(args):
 c=config();root=Path(c['smoke_root'] if args.smoke else c['output_root']);rows=util.load_rows(c)
 if args.smoke:
  # One real sample from every task; each chosen by longest prompt.
  tasks=sorted({x['task'] for x in rows});rows=[max((x for x in rows if x['task']==t),key=lambda x:x['prompt_tokens']) for t in tasks][args.rank::4]
 else:rows=rows[args.rank::4]
 import torch
 from transformers import AutoTokenizer,Qwen3ForCausalLM
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1,2,3' and args.rank in range(4);torch.set_num_threads(4);torch.cuda.set_device(args.rank)
 assert str(torch.cuda.get_device_properties(args.rank).uuid).removeprefix('GPU-')==util.UUIDS[args.rank]
 model,info=Qwen3ForCausalLM.from_pretrained(c['model_path'],local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',output_loading_info=True)
 assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']),info
 assert len(model.model.layers)==36 and not any('ttt_' in n for n in model.state_dict())
 assert all(bool(torch.isfinite(p).all()) for p in model.parameters());model.to('cuda:'+str(args.rank)).eval().requires_grad_(False)
 tok=AutoTokenizer.from_pretrained(c['model_path'],local_files_only=True,fix_mistral_regex=False);versions={n:p._version for n,p in model.named_parameters()};before=util.param_hash(model)
 util.atomic(root/f'worker{args.rank}_loaded.json',{'pid':os.getpid(),'strict_load':'PASS','ttt_parameters':0,'model_hash':before,'parameter_count':sum(p.numel() for p in model.parameters())})
 ns={};sp=Path(c['scorer']);exec(compile(sp.read_text(),str(sp),'exec'),ns);identity=util.canonical({'config':util.sha(C),'smoke':args.smoke})
 for i,row in enumerate(rows,1):
  p=root/'samples'/(row['sample_id'].replace(':','_')+'.json')
  if util.receipt(p,identity):continue
  enc=tok(row['prompt'],return_tensors='pt',add_special_tokens=False,truncation=False);n=enc.input_ids.shape[1];assert n==row['prompt_tokens'] and n+row['max_new_tokens']<=32768
  enc={k:v.to(next(model.parameters()).device) for k,v in enc.items()};start=time.time();torch.cuda.reset_peak_memory_stats()
  with torch.inference_mode():out=model.generate(**enc,max_new_tokens=row['max_new_tokens'],do_sample=False,num_beams=1,use_cache=True,pad_token_id=tok.eos_token_id)
  torch.cuda.synchronize();pred=tok.decode(out[0,n:],skip_special_tokens=True);score=float(ns['per_sample_score'](row['family'],pred,row['reference']));assert 0<=score<=1
  assert versions=={n:p._version for n,p in model.named_parameters()}
  d={'sample_id':row['sample_id'],'task':row['task'],'score':score,'prediction':pred,'prediction_hash':__import__('hashlib').sha256(pred.encode()).hexdigest(),'prompt_token_count':n,'generated_tokens':int(out.shape[1]-n),'latency_seconds':time.time()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'time':time.time(),'DEBUG_ONLY':args.smoke,'model_class':'ORIGINAL_BASE_NO_TTT'}
  util.commit(p,d,identity);util.atomic(root/f'worker{args.rank}_progress.json',{'completed':i,'total':len(rows),'sample_id':row['sample_id'],'time':time.time(),'seconds':d['latency_seconds']});print(json.dumps({k:d[k] for k in ['sample_id','score','latency_seconds']}),flush=True)
 assert util.param_hash(model)==before;util.atomic(root/f'worker{args.rank}_done.json',{'status':'PASS','base_unchanged':True,'samples':len(rows)})
def supervise(args):
 c=config(full=True);root=Path(c['smoke_root'] if args.smoke else c['output_root']);root.mkdir(parents=True,exist_ok=True);lock=(root/'manager.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (root/'PHASE_STATUS.json').exists()
 if not args.smoke:
  smoke=util.read(Path(c['smoke_root'])/'PHASE_STATUS.json');assert smoke['status']=='COMPLETE' and smoke['config_sha256']==util.sha(C)
 locks=[]
 for i in range(4):
  f=(R/'work/downstream_gpu_ownership'/f'physical_gpu{i}.lock').open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 import csv
 q=subprocess.check_output(['nvidia-smi','-i','0,1,2,3','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
 for line in csv.reader(q.splitlines()):assert line[1].strip().removeprefix('GPU-')==util.UUIDS[int(line[0])] and int(line[2])>=43000
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1,2,3',CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',PYTORCH_ALLOC_CONF='expandable_segments:True',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True');ps=[];logs=[]
 for i in range(4):
  f=(root/f'worker{i}.log').open('a');logs.append(f);ps.append(subprocess.Popen([sys.executable,'-B',str(S/'base.py'),'worker','--rank',str(i)]+(['--smoke'] if args.smoke else []),env=env,stdout=f,stderr=subprocess.STDOUT))
 util.atomic(root/'PHASE_STATUS.json',{'status':'RUNNING','manager_pid':os.getpid(),'pids':[p.pid for p in ps],'started':time.time(),'config_sha256':util.sha(C),'smoke':args.smoke})
 while any(p.poll() is None for p in ps):
  if any(p.poll() not in [None,0] for p in ps):
   for p in ps:
    if p.poll() is None:p.terminate()
   break
  time.sleep(5)
 codes=[p.wait() for p in ps]
 if codes==[0]*4:
  identity=util.canonical({'config':util.sha(C),'smoke':args.smoke});files=list((root/'samples').glob('*.json'));rows=[util.receipt(p,identity) for p in files];assert len(rows)==(13 if args.smoke else 6500) and len({r['sample_id'] for r in rows})==len(rows)
  tasks=collections.Counter(x['task'] for x in rows);assert len(tasks)==13 and set(tasks.values())==({1} if args.smoke else {500})
  if not args.smoke:assert {r['sample_id'] for r in rows}=={r['sample_id'] for r in util.load_rows(c)}
  ag={'mean_score':statistics.mean(x['score'] for x in rows),'per_task':{t:statistics.mean(x['score'] for x in rows if x['task']==t) for t in sorted(tasks)},'sample_n':len(rows),'DEBUG_ONLY':args.smoke,'per_sample':rows};util.atomic(root/'AGGREGATE.json',ag)
  if not args.smoke:util.atomic(root/'BASE_RULER_FINAL_AUTHORITY.json',{'status':'PASS','model_class':'ORIGINAL_BASE_NO_TTT','config_sha256':util.sha(C),'sample_n':6500,'included_cwe':True,'aggregate_sha256':util.sha(root/'AGGREGATE.json'),'commits':{str(p):util.sha(p) for p in files}})
 util.atomic(root/'PHASE_STATUS.json',{'status':'COMPLETE' if codes==[0]*4 else 'FAILED','exit_codes':codes,'ended':time.time(),'config_sha256':util.sha(C),'smoke':args.smoke});assert codes==[0]*4
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['worker','supervise']);p.add_argument('--rank',type=int);p.add_argument('--smoke',action='store_true');a=p.parse_args();worker(a) if a.mode=='worker' else supervise(a)
