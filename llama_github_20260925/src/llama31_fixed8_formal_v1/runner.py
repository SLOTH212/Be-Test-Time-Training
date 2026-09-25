"""Isolated Llama Fixed8 runner; immutable inputs and atomic per-cell receipts."""
import os, sys, json, hashlib, time, math, argparse, collections, subprocess, fcntl
from pathlib import Path
R = Path('/path/to/ttt')
S = R/'src/llama31_fixed8_formal_v1'
C = R/'configs/llama31_8b_ruler32k_fixed8_4k_executor_v2.json'
ACTIONS = ['OFF','L0','L6','L12','L18','L24','L30','ALL']
LAYERS = [0,6,12,18,24,30]
UUIDS = ['c7982600-dea0-9d54-16b8-76db4288663e','8459f05d-4d22-5b39-cc1f-e25137d49389','fb7ba9a8-ce12-65a4-a9cf-5b5b7506b0ed','351e23d4-250b-6a70-2c66-d2fb0221a25a']
def read(p): return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
def canonical(d): return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def atomic(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp.'+str(os.getpid()));q.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(q,p)
def verify(weights=False):
 c=read(C);assert sha(C)==(C.with_suffix('.sha256')).read_text().strip()
 assert c['actions']==ACTIONS and c['layers']==LAYERS and c['excluded_tasks']==[] and c['expected_cells']==52000
 for p,h in c['bindings'].items(): assert sha(p)==h,('INPUT_HASH',p)
 a=read(c['parent_authority']);assert a['artifact_class']=='STAGE2_FINAL' and a['integrity']=='PASS' and a['update_step']==92
 assert a['record_cursor']==457 and a['cumulative_tokens']==14390540 and a['DEBUG_ONLY'] is False
 if weights:
  for n,h in a['files'].items():assert sha(Path(c['model_path'])/n)==h,n
 return c
def receipt(p,identity):
 if not p.exists():return None
 d=read(p);h=d.pop('payload_sha256');assert canonical(d)==h and d['identity']==identity and d['status']=='COMMITTED';d['payload_sha256']=h;return d
def commit(p,d,identity):
 assert not p.exists();d={**d,'identity':identity,'status':'COMMITTED'};d['payload_sha256']=canonical(d);atomic(p,d)
def load_rows(c):
 idx=[json.loads(l) for l in Path(c['sample_index']).open()];assert len(idx)==6500
 lengths={x['sample_id']:x for x in read(c['length_index'])};gen=read(c['generation_config']);families={x['name']:x['family'] for x in gen['tasks']};files={};out=[]
 for m in idx:
  rel=m['official_jsonl_path']
  if rel not in files:files[rel]=[json.loads(l) for l in (Path(c['benchmark_root'])/rel).open()]
  x=files[rel][m['source_line']-1];assert hashlib.sha256(x['prompt'].encode()).hexdigest()==m['input_sha256'] and canonical(x['answer'])==m['reference_sha256']
  n=lengths[m['project_sample_id']]['llama_prompt_tokens']+1
  assert n+m['tokens_to_generate']<=32768
  out.append(dict(sample_id=m['project_sample_id'],task=m['task'],global_index=m['global_index'],prompt=x['prompt'],reference=x['answer'],family=families[m['task']],max_new_tokens=m['tokens_to_generate'],prompt_tokens=n))
 assert len({x['sample_id'] for x in out})==6500 and set(collections.Counter(x['task'] for x in out).values())=={500}
 return out
def model_load(c,rank):
 import torch
 from transformers import AutoTokenizer
 sys.path.insert(0,c['llama_source'])
 from inference_model.hf_llama.modeling_llama import LlamaForCausalLM
 import ntp_core.ttt_state_core as core
 torch.set_num_threads(4);torch.cuda.set_device(rank)
 assert str(torch.cuda.get_device_properties(rank).uuid).removeprefix('GPU-')==UUIDS[rank]
 model,info=LlamaForCausalLM.from_pretrained(c['model_path'],local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',output_loading_info=True)
 assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']),info
 assert len(model.model.layers)==32 and sum(p.numel() for p in model.parameters())==8131072000
 assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
 model.to('cuda:'+str(rank)).eval().requires_grad_(False)
 tok=AutoTokenizer.from_pretrained(c['model_path'],local_files_only=True,fix_mistral_regex=False)
 assert tok.encode('hello',add_special_tokens=True)==[128000]+tok.encode('hello',add_special_tokens=False)
 return model,tok,core
def param_hash(model):
 import torch
 h=hashlib.sha256()
 for n,p in sorted(model.named_parameters()):h.update(n.encode());h.update(p.detach().view(torch.uint8).cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def evaluate(model,tok,core,row,action,score_fn):
 import torch
 used=set(LAYERS if action=='ALL' else [] if action=='OFF' else [int(action[1:])]);old=model.model.ttt_mode;calls=[];hooks=[];generation=[]
 versions={n:p._version for n,p in model.named_parameters()};original=core.ttt_chunk_step
 def step(*args,**kwargs):
  out,proposal,stat=original(*args,**kwargs)
  assert bool(torch.isfinite(proposal.weight).all()) and bool(torch.isfinite(out).all())
  calls.append((int(stat.update_count),int(stat.valid_tokens),float(stat.post_clip_norm)))
  return out,proposal,stat
 core.ttt_chunk_step=step
 # V2 routing is installed after instrumentation, so only actual writes are counted.
 try:
  enc=tok(row['prompt'],return_tensors='pt',add_special_tokens=True,truncation=False);n=enc.input_ids.shape[1]
  assert n==row['prompt_tokens'] and n+row['max_new_tokens']<=32768
  sys.path.insert(0,str(R/'src/reference_executor_v2_port'))
  from adapter import prepare_fixed
  prepare_fixed(model,LAYERS,core,action,n)
  def hook(m,args,out):
   if len(args)>1 and args[1] is None:generation.append(len(calls))
  hooks=[model.model.layers[l].mlp.register_forward_hook(hook) for l in LAYERS]
  enc={k:v.to(next(model.parameters()).device) for k,v in enc.items()};(torch.cuda.reset_peak_memory_stats() if next(model.parameters()).device.type=='cuda' else 0);start=time.time()
  with torch.inference_mode():out=model.generate(**enc,max_new_tokens=row['max_new_tokens'],do_sample=False,num_beams=1,use_cache=True,pad_token_id=tok.eos_token_id)
  (torch.cuda.synchronize() if next(model.parameters()).device.type=='cuda' else 0);pred=tok.decode(out[0,n:],skip_special_tokens=True);score=float(score_fn(row['family'],pred,row['reference']));assert math.isfinite(score) and 0<=score<=1
  counts={str(l):sum(s.update_count for s in model.model.layers[l].mlp.last_ttt_stats) for l in LAYERS}
  assert all(counts[str(l)]==(n//4096 if l in used else 0) for l in LAYERS),(action,counts,n)
  assert len(calls)==n//4096*len(used) and all(u==1 and v==4095 and math.isfinite(d) and d<=1e-5*(1+2**-7)**2 for u,v,d in calls)
  assert not generation or min(generation)==max(generation)==len(calls)
  result=dict(sample_id=row['sample_id'],task=row['task'],action=action,score=score,prediction=pred,prediction_hash=hashlib.sha256(pred.encode()).hexdigest(),prompt_token_count=n,generated_tokens=int(out.shape[1]-n),complete_chunks=n//4096,tail_tokens=n%4096,layer_update_counts=counts,update_count=len(calls),generation_update_count=0,latency_seconds=time.time()-start,peak_allocated_bytes=(torch.cuda.max_memory_allocated() if next(model.parameters()).device.type=='cuda' else 0),peak_reserved_bytes=(torch.cuda.max_memory_reserved() if next(model.parameters()).device.type=='cuda' else 0),time=time.time())
 finally:
  core.ttt_chunk_step=original;model.model.ttt_mode=old
  for h in hooks:h.remove()
 assert versions=={n:p._version for n,p in model.named_parameters()},'BASE_PARAMETER_MUTATION'
 return result
def worker(args):
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1,2,3' and args.rank in range(4)
 c=verify();root=Path(c['smoke_root'] if args.smoke else c['output_root']);identity=canonical({'config':sha(C),'smoke':args.smoke});rows=load_rows(c)
 if args.smoke:
  # All actions on four real prompts: CWE, longest NIAH, FWE, VT.
  tasks=['ruler_cwe_16k','ruler_niah_single_1_16k','ruler_fwe_16k','ruler_vt_16k'];rows=[max((x for x in rows if x['task']==tasks[args.rank]),key=lambda x:x['prompt_tokens'])]
 else:rows=rows[args.rank::4]
 scorer={};p=Path(c['scorer']);exec(compile(p.read_text(),str(p),'exec'),scorer)
 model,tok,core=model_load(c,args.rank);before=param_hash(model)
 atomic(root/f'worker{args.rank}_loaded.json',dict(pid=os.getpid(),rank=args.rank,model_hash=before,strict_load='PASS',config_sha256=sha(C)))
 for row in rows:
  for action in ACTIONS:
   p=root/'samples'/(row['sample_id'].replace(':','_')+'__'+action+'.json')
   if receipt(p,identity):continue
   result=evaluate(model,tok,core,row,action,scorer['per_sample_score']);commit(p,{**result,'DEBUG_ONLY':args.smoke,'FORMAL_RESULT':not args.smoke},identity)
   atomic(root/f'worker{args.rank}_progress.json',dict(sample_id=row['sample_id'],action=action,latency_seconds=result['latency_seconds'],time=time.time(),pid=os.getpid()))
   print(json.dumps({k:result[k] for k in ['sample_id','action','score','latency_seconds']}),flush=True)
  if args.smoke:
   first=receipt(root/'samples'/(row['sample_id'].replace(':','_')+'__OFF.json'),identity)
   repeated=evaluate(model,tok,core,row,'OFF',scorer['per_sample_score']);assert repeated['prediction_hash']==first['prediction_hash'],'CROSS_ACTION_STATE_LEAK'
 assert param_hash(model)==before
 if args.smoke:
  from safetensors import safe_open
  import torch
  with safe_open(str(Path(c['model_path'])/'model.safetensors'),framework='pt',device='cpu') as source:
   assert set(source.keys())==set(model.state_dict())
   for n,p in model.state_dict().items():assert torch.equal(p.detach().cpu(),source.get_tensor(n)),('SOURCE_WEIGHT_CHANGED',n)
 atomic(root/f'worker{args.rank}_done.json',dict(status='PASS',base_unchanged=True,owned_samples=len(rows),time=time.time()))
def supervise(args):
 c=verify(weights=True);root=Path(c['smoke_root'] if args.smoke else c['output_root']);root.mkdir(parents=True,exist_ok=True)
 lock=(root/'manager.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if not args.smoke:
  smoke=read(Path(c['smoke_root'])/'PHASE_STATUS.json');assert smoke['status']=='COMPLETE' and smoke['config_sha256']==sha(C)
 assert not (root/'PHASE_STATUS.json').exists(),'EXPLICIT_NEW_RUN_REQUIRED'
 locks=[]
 for gpu in range(4):
  p=R/'work/downstream_gpu_ownership'/f'physical_gpu{gpu}.lock';p.parent.mkdir(exist_ok=True);f=p.open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 import csv
 g=subprocess.check_output(['nvidia-smi','-i','0,1,2,3','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
 for line in csv.reader(g.splitlines()):assert line[1].strip().removeprefix('GPU-')==UUIDS[int(line[0])] and int(line[2])>=43000
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1,2,3',CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',PYTORCH_ALLOC_CONF='expandable_segments:True',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
 ps=[];logs=[]
 for i in range(4):
  log=(root/f'worker{i}.log').open('a');logs.append(log);cmd=[sys.executable,'-B',str(S/'runner.py'),'worker','--rank',str(i)]+(['--smoke'] if args.smoke else []);ps.append(subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT))
 atomic(root/'PHASE_STATUS.json',dict(status='RUNNING',manager_pid=os.getpid(),pids=[p.pid for p in ps],started=time.time(),config_sha256=sha(C),smoke=args.smoke))
 while any(p.poll() is None for p in ps):
  if any(p.poll() not in [None,0] for p in ps):
   for p in ps:
    if p.poll() is None:p.terminate()
   break
  time.sleep(5)
 codes=[p.wait() for p in ps];status='COMPLETE' if codes==[0]*4 else 'FAILED'
 if status=='COMPLETE':
  identity=canonical({'config':sha(C),'smoke':args.smoke});rs=[receipt(p,identity) for p in (root/'samples').glob('*.json')];assert len(rs)==(32 if args.smoke else 52000)
  cells={(r['sample_id'],r['action']) for r in rs};assert len(cells)==len(rs)
  if not args.smoke:assert cells=={(r['sample_id'],a) for r in load_rows(c) for a in ACTIONS}
  scores={a:sum(r['score'] for r in rs if r['action']==a)/sum(r['action']==a for r in rs) for a in ACTIONS}
  per_sample={};per_task={}
  for r in rs:
   per_sample.setdefault(r['sample_id'],{'sample_id':r['sample_id'],'task':r['task']})[r['action']]=r['score']
  for row in per_sample.values():
   row['sample_best']=max(row[a] for a in ACTIONS);row['best_actions']=[a for a in ACTIONS if abs(row[a]-row['sample_best'])<=1e-12]
  for task in sorted({r['task'] for r in rs}):
   subset=[r for r in per_sample.values() if r['task']==task];per_task[task]={a:sum(r[a] for r in subset)/len(subset) for a in ACTIONS+['sample_best']}
  best=max(scores.values());aggregate=dict(action_means=scores,per_task=per_task,per_sample=list(per_sample.values()),best_single_actions=[a for a in ACTIONS if abs(scores[a]-best)<=1e-12],sample_best_mean=sum(r['sample_best'] for r in per_sample.values())/len(per_sample),cells=len(rs),config_sha256=sha(C),DEBUG_ONLY=args.smoke)
  atomic(root/'AGGREGATE.json',aggregate)
  if not args.smoke:
   atomic(root/'FIXED8_FINAL_AUTHORITY.json',dict(schema='LLAMA31_FIXED8_FINAL_AUTHORITY_V1',status='PASS',DEBUG_ONLY=False,model_parent_authority=c['parent_authority'],model_parent_sha256=sha(c['parent_authority']),config_sha256=sha(C),sample_n=6500,cells=52000,actions=ACTIONS,layers=LAYERS,chunk_size=4096,included_cwe=True,aggregate_path=str(root/'AGGREGATE.json'),aggregate_sha256=sha(root/'AGGREGATE.json'),commits={str(p):sha(p) for p in (root/'samples').glob('*.json')}))
 atomic(root/'PHASE_STATUS.json',dict(status=status,exit_codes=codes,ended=time.time(),config_sha256=sha(C),smoke=args.smoke))
 if status!='COMPLETE':raise RuntimeError('WORKER_FAILED_NO_AUTORETRY')
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['worker','supervise']);p.add_argument('--rank',type=int);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if a.mode=='worker':worker(a)
 else:supervise(a)
if __name__=='__main__':main()
