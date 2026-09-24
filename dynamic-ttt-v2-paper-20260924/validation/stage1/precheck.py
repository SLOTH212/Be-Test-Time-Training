#!/usr/bin/env python3
import os
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import sys,json,csv,hashlib,time,math,types,importlib.util,gc,copy
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent;O=ROOT/'precheck';O.mkdir(exist_ok=True);LAYERS=[0,6,12,18,24];AL={'OFF':[],'ALL':LAYERS,**{f'L{l}':[l] for l in LAYERS}}
torch.set_num_threads(16);os.sched_setaffinity(0,range(8,24))
def j(p):return json.loads(Path(p).read_text())
def jl(p):return [json.loads(l) for l in Path(p).open() if l.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(n,x):
 p=O/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def append(n,x):
 p=O/n;exists=p.exists()
 with p.open('a',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(x));
  if not exists:w.writeheader()
  w.writerow(x)
def imp(p,n):
 spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);sys.modules[n]=m;spec.loader.exec_module(m);return m
def digest_cpu(x):return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def CPU(x):return x.detach().to('cpu').contiguous()
def dh(x):return digest_cpu(CPU(x))
selection=j(Path('/home/zonghan/ttt/audits/formal_execution_validation_1p7b_20260915/FORMAL_SAMPLE_SELECTION.json'));formal=selection['samples'];F={x['sample_id']:x for x in formal};assets=j(ROOT/'PRE_FLIGHT_V2.json');assert assets['status']=='PASS'
Bench=Path('/home/zonghan/ttt/benchmarks/ruler_16k_standard_13task500_v1');rows={x['sample_id']:x for x in jl(Bench/'sample_manifest.jsonl')};K=Path('/home/zonghan/ttt/runs/formal/kp_factorial_1p7b_v1/run_20260905T235503+0800');co={x['canonical_sample_id']:x for x in jl(K/'manifests/KP_167_COHORT.jsonl')}
sys.path.insert(0,'/home/zonghan/ttt/code/ntp');core=imp(ROOT/'source/instrumented_ttt_state_core.py','audit_instrumented_core')
import hf_models.hf_qwen3.ttt_state_core as production_core
original_step=production_core.ttt_chunk_step;production_core.ttt_chunk_step=core.ttt_chunk_step
runner=imp(ROOT/'source/instrumented_reference_executor_v2.py','audit_instrumented_runner');scorer=imp('/home/zonghan/ttt/code/ruler/tools/common.py','audit_frozen_scorer')
ctx=types.SimpleNamespace(run=O,code=Path('/home/zonghan/ttt/code/ntp'),ckpt=Path('/home/zonghan/ttt/models/qwen3_1p7b_stage2'),score_fn=scorer.per_sample_score)
# CPU authority hash of actual input manifest and metadata before any model load.
bm=j(Bench/'benchmark_manifest.json');assert sha(Bench/'sample_manifest.jsonl')==bm['global_sample_manifest']['sha256']
for sid in [x['sample_id'] for x in formal]+selection['engineering_smoke_ids']:
 x=rows[sid];assert hashlib.sha256(x['input'].encode()).hexdigest()==x['input_hash']==co[sid]['benchmark_content_sha256']
print('AUTHORITY_FINAL_INPUT_GATE_PASS',flush=True)
class Trace:
 def __init__(self,label,reference=None,limit=None,metrics=None,force_zero=False,offset=0,capture_donor=False,boundary=None):
  self.label=label;self.ref=reference;self.limit=limit;self.metrics=metrics or {'mlp_input','u','mlp_output','hidden','kv_k','kv_v','W_pre','W_post','raw_delta','clipped_delta','actual_delta','target_v','logits'};self.force_zero=force_zero;self.offset=offset;self.data={};self.hashes={};self.seen=set();self.details=[];self.location_now=None;self.callstep={};self.genstep=0;self.weight_zero=True;self.max_actual=0.;self.capture_donor=capture_donor;self.boundary=boundary;self.boundary_W={};self.donor=None;self.finite=True;self.update_records=[]
 def location(self,l,c):self.location_now=(l,c+self.offset)
 def proposal(self,name,x):
  if self.location_now:self.add(name,*self.location_now,x)
 def add(self,site,l,c,x):
  if site not in self.metrics or (self.limit is not None and (c>self.limit or c<1)):return
  key=f'{site}/L{l}/C{c}';v=CPU(x);self.finite &= bool(torch.isfinite(v).all());h=digest_cpu(v);self.seen.add(key)
  if self.ref is None:self.data[key]=v;self.hashes[key]=h;return
  r=self.ref.data.get(key)
  if r is None:self.details.append({'site':key,'exact':False,'missing_reference':True,'shape':list(v.shape)});return
  exact=r.shape==v.shape and r.dtype==v.dtype and h==self.ref.hashes[key];q={'site':key,'shape':list(v.shape),'dtype':str(v.dtype),'hash_A':self.ref.hashes[key],'hash_B':h,'exact':exact,'max_abs_diff':0.,'mean_abs_diff':0.,'relative_l2':0.}
  if not exact:
   if r.shape!=v.shape:q['shape_mismatch']=True
   else:
    a=r.float();b=v.float();diff=b-a;q.update(max_abs_diff=float(diff.abs().max()),mean_abs_diff=float(diff.abs().mean()),relative_l2=float(torch.linalg.vector_norm(diff)/torch.linalg.vector_norm(a).clamp_min(1e-30)),norm_A=float(torch.linalg.vector_norm(a)),norm_B=float(torch.linalg.vector_norm(b)),diff_frobenius_norm=float(torch.linalg.vector_norm(diff)))
    neq=(r!=v).flatten();q['first_mismatch_flat_index']=int(neq.to(torch.uint8).argmax()) if bool(neq.any()) else None
  self.details.append(q)
 def enter(self,m,x,h,t,w):
  l=int(m.layer_idx);c=0 if x.shape[1]==1 else None
  if c==0:
   self.add('mlp_input',l,-self.genstep,x);self.add('u',l,-self.genstep,h)
  else:
   for start in range(0,x.shape[1],1024):
    ci=start//1024+1+self.offset;self.add('mlp_input',l,ci,x[:,start:start+1024]);self.add('u',l,ci,h[:,start:start+1024])
 def zerostate(self,m,c,w):
  self.add('W_pre',m.layer_idx,c,w);self.add('W_post',m.layer_idx,c,w)
  if {'raw_delta','clipped_delta','actual_delta'}&self.metrics:
   z=torch.zeros_like(w)
   for n in ['raw_delta','clipped_delta','actual_delta']:self.add(n,m.layer_idx,c,z)
 def standard(self,m,x,h,out):
  w=getattr(m,'_suite_fixed_weight',m.down_proj.weight)
  if x.shape[1]==1:self.add('mlp_output',m.layer_idx,-self.genstep,out);return
  for start in range(0,x.shape[1],1024):
   ci=start//1024+1+self.offset;self.add('mlp_output',m.layer_idx,ci,out[:,start:start+1024]);self.zerostate(m,ci,w)
   if self.capture_donor and self.boundary==ci+1:self.boundary_W[m.layer_idx]=w.detach().clone()
 def generation(self,m,x,h,out,w):self.add('mlp_output',m.layer_idx,-self.genstep,out)
 def chunk(self,m,c,x,h,out,pre,post,stat,active):
  c+=self.offset;self.add('mlp_output',m.layer_idx,c,out);self.add('W_pre',m.layer_idx,c,pre);self.add('W_post',m.layer_idx,c,post);actual=post-pre;an=float(torch.linalg.vector_norm(actual.float()));self.max_actual=max(self.max_actual,an);self.weight_zero &= dh(post)==dh(m.down_proj.weight)
  self.update_records.append({'layer':int(m.layer_idx),'chunk':c,'active':active,'raw_delta_norm':float(stat.pre_clip_norm),'clipped_proposal_norm':float(stat.post_clip_norm),'actual_applied_weight_diff_norm':an,'effective_W_before_sha256':dh(pre),'effective_W_after_sha256':dh(post)})
  self.add('actual_delta',m.layer_idx,c,actual)
  if not active:
   z=torch.zeros_like(pre);self.add('raw_delta',m.layer_idx,c,z);self.add('clipped_delta',m.layer_idx,c,z)
  if self.capture_donor and self.boundary==c+1:self.boundary_W[m.layer_idx]=post.detach().clone()
 def tail(self,m,x,h,out,w):
  c=len(m._dynamic_sequence)+1+self.offset;self.add('mlp_output',m.layer_idx,c,out);self.zerostate(m,c,w);self.weight_zero &= dh(w)==dh(m.down_proj.weight)
 def report(self):
  if self.ref is None:return {'trace_exact':None,'finite':self.finite,'tensor_n':len(self.data)}
  missing=sorted(set(self.ref.data)-self.seen);bad=[x for x in self.details if not x['exact']];first=bad[0] if bad else ({'site':missing[0],'missing_observation':True} if missing else None)
  return {'trace_exact':not bad and not missing,'finite':self.finite,'tensor_n':len(self.details),'exact_tensor_n':sum(x['exact'] for x in self.details),'missing_tensor_n':len(missing),'max_abs_diff':max([x.get('max_abs_diff',0.) for x in self.details] or [0.]),'relative_l2':max([x.get('relative_l2',0.) for x in self.details] or [0.]),'first_mismatch_site':first['site'] if first else None,'first_mismatch':first}
active=None
handles=[]
def attach(model):
 for i,layer in enumerate(model.model.layers):
  def attn_hook(mod,args,kw,out,l=i):
   tr=active
   if tr is None:return
   cache=kw.get('past_key_values');h=kw.get('hidden_states',args[0] if args else None)
   if cache is None or h is None:return
   if h.shape[1]==1:
    tr.add('kv_k',l,-tr.genstep,cache.layers[l].keys[...,-1:,:]);tr.add('kv_v',l,-tr.genstep,cache.layers[l].values[...,-1:,:]);return
   kk=cache.layers[l].keys;vv=cache.layers[l].values;n=h.shape[1];start_abs=tr.offset*1024
   for start in range(0,n,1024):
    ci=start//1024+1+tr.offset
    tr.add('kv_k',l,ci,kk[...,start_abs+start:start_abs+min(start+1024,n),:]);tr.add('kv_v',l,ci,vv[...,start_abs+start:start_abs+min(start+1024,n),:])
  handles.append(layer.self_attn.register_forward_hook(attn_hook,with_kwargs=True))
  def layer_hook(mod,args,kw,out,l=i):
   tr=active
   if tr is None:return
   h=out[0] if isinstance(out,tuple) else out
   if h.shape[1]==1:tr.add('hidden',l,-tr.genstep,h)
   else:
    for start in range(0,h.shape[1],1024):tr.add('hidden',l,start//1024+1+tr.offset,h[:,start:start+1024])
  handles.append(layer.register_forward_hook(layer_hook,with_kwargs=True))
 def model_pre(mod,args,kw):
  if active is not None:active.genstep+=1
 handles.append(model.register_forward_pre_hook(model_pre,with_kwargs=True))
 def model_post(mod,args,kw,out):
  tr=active
  if tr is None:return
  tr.add('logits',-1,-tr.genstep,out.logits[:,-1:])
  if tr.capture_donor and tr.genstep==1:
   tr.donor=out.past_key_values
 handles.append(model.register_forward_hook(model_post,with_kwargs=True))
print('ENGINEERING_MODEL_LOAD_STARTED',flush=True);model,tok=runner.load_model(ctx);attach(model)
base={l:dh(model.model.layers[l].mlp.down_proj.weight) for l in LAYERS};run_count=0;isolation=[]
def run(sid,seq,label,reference=None,limit=None,metrics=None,zero=False,gold=False,capture_donor=False,donor=None,standard_W=None):
 global active,run_count
 torch.manual_seed(20260915);torch.cuda.manual_seed_all(20260915)
 offset=0 if donor is None else donor['prefix_chunks'];tr=Trace(label,reference,limit,metrics,zero,offset,capture_donor,co[sid]['boundary']);active=tr;runner._suite_trace=tr;core._suite_trace=tr;before={l:dh(model.model.layers[l].mlp.down_proj.weight) for l in LAYERS};assert before==base
 row=dict(rows[sid]);
 if gold:row['reference']=['__EXECUTION_SUITE_ALTERNATE_REFERENCE__']
 start=time.time();old_forwards={};stashed={};oldmode=model.model.ttt_mode
 try:
  if donor is None:
   ev=runner.Evaluator(ctx,model,tok,row,'execution_validation',1,1,O/'progress.json');rec=ev.evaluate(tuple(seq),capture=False,force=True);ev.finish();del ev
  else:
   enc=tok(row['input'],return_tensors='pt',add_special_tokens=True,truncation=False);enc={k:v.cuda() for k,v in enc.items()}
   from inference_model.hf_qwen3.modeling_qwen3 import TTTDynamicCache
   cache=TTTDynamicCache(config=model.config)
   for l,(kk,vv) in donor['kv'].items():cache.update(kk.clone(),vv.clone(),l)
   for l in LAYERS:cache.ttt_states[l]=(None,None,donor['W'][l].clone())
   for l in LAYERS:
    m=model.model.layers[l].mlp;m._dynamic_sequence=list(seq);m._capture_identity=False;m._branch_id=label;m._events=[]
    if standard_W is not None:
     stashed[l]={name:getattr(m,name) for name in ['ttt_conv','ttt_proj','ttt_ntp_gate'] if hasattr(m,name)}
     for name in stashed[l]:delattr(m,name)
     m._suite_fixed_weight=donor['W'][l];old_forwards[l]=m.down_proj.forward
     def fixedforward(x,m=m):return torch.nn.functional.linear(x,m._suite_fixed_weight,m.down_proj.bias)
     m.down_proj.forward=fixedforward
   if standard_W is not None:model.model.ttt_mode=False
   with torch.inference_mode():out=model.generate(**enc,past_key_values=cache,max_new_tokens=row['max_new_tokens'],do_sample=False,num_beams=1,use_cache=True,pad_token_id=tok.eos_token_id)
   pred=tok.decode(out[0,enc['input_ids'].shape[1]:],skip_special_tokens=True);rec={'prediction_hash':hashlib.sha256(pred.encode()).hexdigest(),'score':float(ctx.score_fn(row['family'],pred,row['reference']))};del enc,out,cache
 finally:
  model.model.ttt_mode=oldmode
  for l,attrs in stashed.items():
   m=model.model.layers[l].mlp
   for name,val in attrs.items():setattr(m,name,val)
   m.down_proj.forward=old_forwards[l];delattr(m,'_suite_fixed_weight')
  active=None;runner._suite_trace=None;core._suite_trace=None
 after={l:dh(model.model.layers[l].mlp.down_proj.weight) for l in LAYERS};run_count+=1;isol={'run_id':label,'sample_id':sid,'checkpoint_unchanged':before==after==base,'clean_state':donor is None,'controlled_donor_fork':donor is not None,'pre_digest':json.dumps(before,sort_keys=True),'post_digest':json.dumps(after,sort_keys=True)};isolation.append(isol);append('checkpoint_isolation.csv',isol);assert isol['checkpoint_unchanged']
 rec.pop('prediction',None);tr.result={'sample_id':sid,'label':label,'sequence':seq,'used_layer_union':sorted({l for a in seq for l in AL[a]}),'prediction_hash':rec['prediction_hash'],'reward':rec['score'],'elapsed_seconds':time.time()-start,'gpu_peak_bytes':torch.cuda.max_memory_allocated(),'actual_weight_diff_max_norm':tr.max_actual,'all_effective_weights_checkpoint':tr.weight_zero,'input_hash':hashlib.sha256(tok(row['input'],return_tensors='pt',add_special_tokens=True,truncation=False).input_ids.numpy().tobytes()).hexdigest()};assert tr.finite
 put('run_metadata/'+label.replace(':','_')+'.json',{'result':tr.result,'updates':tr.update_records,'finite':tr.finite,'checkpoint_unchanged':isol['checkpoint_unchanged']})
 print('RUN_COMPLETE',label,sid,f'{time.time()-start:.2f}s',flush=True);return tr

def pairrow(sid,label,a,b,filename,extra=None):
 q=b.report();x={'sample_id':sid,'pair':label,**{k:v for k,v in q.items() if k!='first_mismatch'},'prediction_match':a.result['prediction_hash']==b.result['prediction_hash'],'reward_match':a.result['reward']==b.result['reward'],'prediction_hash_A':a.result['prediction_hash'],'prediction_hash_B':b.result['prediction_hash'],'reward_A':a.result['reward'],'reward_B':b.result['reward'],**(extra or {})};append(filename,x)
 put('pair_metrics/'+label.replace('/','_')+'_'+hashlib.sha256(sid.encode()).hexdigest()[:12]+'.json',{'pair':x,'run_A':a.result,'run_B':b.result,'trace_metrics':b.details})
 if not q['trace_exact']:
  put('failure_forensics/'+label.replace('/','_')+'_'+hashlib.sha256(sid.encode()).hexdigest()[:12]+'.json',{'pair':x,'first_mismatch':q['first_mismatch'],'run_A':a.result,'run_B':b.result,'protocol_sha256':sha(ROOT/'V2_PRECHECK_PROTOCOL.json'),'environment_sha256':sha(ROOT/'V2_ENVIRONMENT.json'),'trace_metrics':b.details})
 return x

PRE=j(ROOT/'V2_PRECHECK_PROTOCOL.json');counts={'repeat':0,'zero':0,'future':0,'apply':0}
def require(ok,label):
 if not ok:
  (ROOT/'V2_PRECHECK_RESULTS.json').write_text(json.dumps({'status':'BLOCKED_V2_VALIDATION_FAIL','failed_check':label,'counts':counts,'trajectory_n':run_count},indent=2)+'\n');raise RuntimeError('V2_PRECHECK_FAIL:'+label)
outputsites={'mlp_input','u','mlp_output','hidden','kv_k','kv_v','W_pre','W_post','logits'}
for sid in PRE['repeat_samples']:
 a=run(sid,F[sid]['native_sequence'],'V2_REPEAT_A_'+sid);b=run(sid,F[sid]['native_sequence'],'V2_REPEAT_B_'+sid,reference=a);q=pairrow(sid,'V2_REPEAT',a,b,'same_path_repeat.csv');require(q['trace_exact'] and q['prediction_match'] and q['reward_match'],'repeat:'+sid);counts['repeat']+=1;del a,b;gc.collect()
for x in formal:
 sid=x['sample_id'];T=x['T'];a=run(sid,['OFF']*T,'V2_Z_OFF_'+sid,metrics=outputsites)
 for label,seq in [('V2_Z_NATIVE',x['native_sequence']),('V2_Z_ALL',['ALL']*T)]:
  b=run(sid,seq,label+'_'+sid,reference=a,metrics=outputsites,zero=True);q=pairrow(sid,label,a,b,'zero_write.csv',{'weight_zero_A':a.weight_zero,'weight_zero_B':b.weight_zero,'max_actual_A':a.max_actual,'max_actual_B':b.max_actual});require(q['trace_exact'] and q['prediction_match'] and q['reward_match'] and a.weight_zero and b.weight_zero and a.max_actual==b.max_actual==0,'zero:'+label+':'+sid);counts['zero']+=1;del b;gc.collect()
 del a;gc.collect()
for p in PRE['future_pairs']:
 sid=p['sample_id'];label='V2_'+p['test'];a=run(sid,p['A'],label+'_A_'+sid,limit=p['t']);b=run(sid,p['B'],label+'_B_'+sid,reference=a,limit=p['t']);q=pairrow(sid,label,a,b,'future_prefix.csv',{'t':p['t']});require(q['trace_exact'],'future:'+label+':'+sid);counts['future']+=1;del a,b;gc.collect()
for p in PRE['apply_cases']:
 sid=p['sample_id'];l=p['layer'];a=run(sid,p['A'],f'V2_APPLY_L{l}_A_'+sid);b=run(sid,p['B'],f'V2_APPLY_L{l}_B_'+sid,reference=a);d={q['site']:q for q in b.details};entry=all(d[f'{k}/L{l}/C2']['exact'] for k in ['W_pre','mlp_input','u']);current=d[f'mlp_output/L{l}/C2']['exact'];append('apply_then_update.csv',{'sample_id':sid,'layer':l,'entry_exact':entry,'current_output_exact':current,'post_W_equal':d[f'W_post/L{l}/C2']['exact'],'next_output_exact':d[f'mlp_output/L{l}/C3']['exact']});put(f'apply_metrics/L{l}.json',b.details);require(entry and current,'apply:'+str(l));counts['apply']+=1;del a,b;gc.collect()
require(counts=={'repeat':4,'zero':64,'future':15,'apply':5},'coverage');require(all(x['checkpoint_unchanged'] and x['clean_state'] for x in isolation),'isolation')
result={'status':'PASS_EXACT','formal32_n':32,'counts':counts,'same_path_repeat_status':'PASS_EXACT','zero_write_schedule_invariance_status':'PASS_EXACT','future_action_prefix_status':'PASS_EXACT','apply_then_update_status':'PASS_EXACT','checkpoint_isolation_status':'PASS_EXACT','trajectory_n':run_count,'checkpoint_unchanged_n':run_count,'state_reset_pass_n':run_count,'precheck_protocol_sha256':sha(ROOT/'V2_PRECHECK_PROTOCOL.json')};(ROOT/'V2_PRECHECK_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n');print('V2_PRECHECK_PASS_EXACT',run_count,flush=True)
