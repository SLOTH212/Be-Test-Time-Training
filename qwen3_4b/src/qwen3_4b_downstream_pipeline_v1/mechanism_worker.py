#!/usr/bin/env python3
import sys,os,json,ast,types,time,hashlib,traceback,argparse
from pathlib import Path
from debug_worker import R,W,A,setup,rt,execution,Store,atomic,FLAGS,LAYERS,ACTIONS,load_rows,cfg,make_context,sha
import torch
class AdaptedExecutor(execution.Executor):
 pass
# External integration changes only dependency code path and deployed benchmark-authority layout.
init=next(x for x in next(x for x in ast.parse(Path(execution.__file__).read_text()).body if isinstance(x,ast.ClassDef) and x.name=='Executor').body if isinstance(x,ast.FunctionDef) and x.name=='__init__')
class Adapt(ast.NodeTransformer):
 def visit_Constant(self,n):
  return ast.copy_location(ast.Constant('authority/RULER_32K_AUTHORITY.json'),n) if n.value=='benchmark_manifest.json' else n
 def visit_BinOp(self,n):
  n=self.generic_visit(n)
  if isinstance(n.op,ast.Div) and isinstance(n.left,ast.Attribute) and isinstance(n.left.value,ast.Name) and n.left.value.id=='self' and n.left.attr=='dep' and isinstance(n.right,ast.Constant) and n.right.value=='code':
   return ast.copy_location(ast.Call(ast.Name('Path',ast.Load()),[ast.Constant(setup['adapt'])],[]),n)
  return n
ns=dict(execution.__dict__);exec(compile(ast.fix_missing_locations(ast.Module(body=[Adapt().visit(init)],type_ignores=[])),__file__,'exec'),ns);AdaptedExecutor.__init__=ns['__init__']

def fixture(row,tok):
 root=W/'DEBUG_INTERFACE_FIXTURE';root.mkdir(exist_ok=True);original=row['input'];before=len(tok(original,add_special_tokens=True)['input_ids']);n=32768-before
 for _ in range(12):
  text=original+' .'*n;got=len(tok(text,add_special_tokens=True)['input_ids'])
  if got==32768:break
  n+=32768-got
 else:raise RuntimeError('EXACT_FILLER_LENGTH_FAILED')
 assert text.startswith(original) and n>=0
 sid='DEBUG_INTERFACE_FIXTURE:'+row['sample_id'];out={**row,'sample_id':sid,'input':text,'expected_prompt_tokens':32768,'source_sample_id':row['sample_id'],'DEBUG_INTERFACE_FIXTURE':True,**FLAGS}
 out['fixture_provenance']=dict(original_prompt_sha256=hashlib.sha256(original.encode()).hexdigest(),fixture_prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),filler='deterministic repeated ASCII space+period',filler_repetitions=n,filler_uses_answers_or_other_samples=False,prompt_tokens=32768,complete_chunks=8,tail_tokens=0,original_prompt_preserved_as_prefix=True,frozen_benchmark_modified=False,scientific_eligibility_resolved=False)
 path=root/(hashlib.sha256(sid.encode()).hexdigest()+'.json')
 if path.exists():assert json.loads(path.read_text())==out
 else:atomic(path,out)
 return out

def direct_parent(ex,row,sequence,base):
 stub={'sample_id':row['sample_id']};native,_=ex.evaluate(stub,sequence)
 return dict(sample_id=row['sample_id'],task=row['task'],benchmark_global_index=row['_sample_index'],action_sequence=sequence,chunk_count=len(sequence),dynamic_score=native['score'],native_prediction_sha256=native['prediction_hash'],sample_best=max(base.values()),fixed_scores=base,completed=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--experiment',choices=['reverse','reset','deletion','kp','r2'],required=True);ap.add_argument('--worker-id',type=int,default=0);ap.add_argument('--workers',type=int,default=2);args=ap.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='2,3';assert args.workers in [1,2] and 0<=args.worker_id<args.workers;torch.set_num_threads(4);torch.cuda.set_device(args.worker_id)
 ids=[x['sample_id'] for x in rt.rows(W/'DEBUG_SAMPLE_MANIFEST.jsonl')];from debug_worker import fixed_store
 fs=fixed_store(ids);fixed={sid:{a:fs.committed(sid+'|'+a)['score'] for a in ACTIONS} for sid in ids}
 drows=[json.loads(p.read_text()) for p in sorted((W/'dynamic/samples').glob('*.json'))];assert drows
 if args.experiment in ['reverse','r2']:
  jobs=drows;input_class='REAL_CHAIN'
 else:
  jobs=drows[:2];input_class='DEBUG_INTERFACE_FIXTURE'
 owned=[x for i,x in enumerate(jobs) if i%args.workers==args.worker_id];root=W/'mechanism'/input_class/args.experiment
 identity=dict(experiment=args.experiment,input_class=input_class,dynamic_parent_hashes={x['sample_id']:x['payload_sha256'] for x in jobs},setup=setup,source_sha256=sha(__file__),inference4k=setup['config_sha256'],**FLAGS)
 store=Store(root,identity);workspace=root/'workers'/str(args.worker_id);workspace.mkdir(parents=True,exist_ok=True)
 ex=None
 for d in owned:
  sid=d['sample_id'] if input_class=='REAL_CHAIN' else 'DEBUG_INTERFACE_FIXTURE:'+d['sample_id']
  with store.own(sid):
   if store.committed(sid) is not None:continue
   row=load_rows([d['sample_id']])[0]
   if ex is None:
    ex=AdaptedExecutor(cfg(args.experiment),[row],{row['sample_id']:fixed[row['sample_id']]},workspace)
    atomic(workspace/'WORKER.json',dict(pid=os.getpid(),physical_gpu=args.worker_id+2,model_loads=1,**FLAGS))
   if input_class=='DEBUG_INTERFACE_FIXTURE':
    row=fixture(row,ex.tok);ex.ctx.by_id[sid]=row;ex.ctx.baseline={a:{sid:{'official_sample_score':fixed[d['sample_id']][a]}} for a in ACTIONS}
    sequence=['L30']+['L0']*7;screen=rt.reset_screen(sequence,LAYERS);assert screen['eligible'] and screen['boundary']==2
    if args.experiment=='kp':
     # Native fixture identity is measured by the same frozen reward path.
     c,f,_=ex.kp;nr=f.run_reward(ex.ctx,ex.model,ex.tok,row,sequence,1)
     parent=dict(sample_id=sid,task=row['task'],benchmark_global_index=row['_sample_index'],action_sequence=sequence,chunk_count=8,dynamic_score=nr['score'],native_prediction_sha256=nr['prediction_hash'],sample_best=max(fixed[d['sample_id']].values()),fixed_scores=fixed[d['sample_id']],completed=True)
    else:parent=direct_parent(ex,row,sequence,fixed[d['sample_id']])
   else:
    ex.ctx.by_id[sid]=row;parent={**d,'benchmark_global_index':row['_sample_index'],'fixed_scores':fixed[sid]}
   started=time.time()
   if args.experiment=='r2':
    native_parent,_=ex.evaluate(parent,parent['action_sequence']);assert native_parent['prediction_hash']==parent['native_prediction_sha256'] and native_parent['score']==parent['dynamic_score']
    T=len(parent['action_sequence']);allc=list(rt.r2_candidates(T,LAYERS));assert len(allc)==8*7*(T-1)
    sequences=[['OFF']*T,['L30']*T]+[x['sequence'] for x in [next(c for c in allc if c['A']=='L30' and c['B']=='L0' and c['tau']==1),next(c for c in allc if c['A']=='ALL' and c['B']=='OFF' and c['tau']==T-1)]]
    records=[]
    for seq in sequences:
     r,_=ex.evaluate(parent,seq);records.append(r)
    # Frozen selector applied only to the two genuine restricted-family candidates.
    subset=[]
    for r in records[2:]:
     c=next(c for c in allc if c['sequence']==r['sequence']);subset.append({**c,'score':r['score']})
    winner,score,ties=rt.r2_winner(subset,LAYERS)
    result=dict(schema='DEBUG_R2_BOUNDED_EXECUTION_V1',sample_id=sid,records=records,winner=winner,selection_score=score,tie_count=ties,enumerated_full_family_count=len(allc),executed_candidates=len(records),R2_FULL_DEBUG_SEARCH_EXECUTED=False,scientific_exact_R2_result=False,native_parent_replay_verified=True)
   else:
    result=ex.execute(parent)
    # Debug interface schemas never pretend to satisfy formal improved-parent validation.
    result['schema']='DEBUG_MECHANISM_INTERFACE_V1' if input_class=='DEBUG_INTERFACE_FIXTURE' else 'DEBUG_MECHANISM_REAL_CHAIN_V1'
   if args.experiment in ['reset','deletion']:
    # Same-sample replay after intervention must return the clean native prediction.
    again,_=ex.evaluate(parent,parent['action_sequence']);assert again['prediction_hash']==parent['native_prediction_sha256'] and again['score']==parent['dynamic_score']
    result['post_intervention_native_repeatability']=True
   if args.experiment=='deletion':
    resetfiles=list((W/'mechanism/DEBUG_INTERFACE_FIXTURE/reset/samples').glob('*.json'));matched=next(json.loads(p.read_text()) for p in resetfiles if json.loads(p.read_text())['sample_id']==sid)
    assert matched['parent_dynamic_trajectory']==parent['action_sequence'];result['matched_reset_payload_sha256']=matched['payload_sha256']
   result.update(input_class=input_class,DEBUG_INTERFACE_FIXTURE=input_class=='DEBUG_INTERFACE_FIXTURE',model_identity=setup['model_identity'],benchmark_identity=setup['benchmark_identity'],inference_config_sha256=setup['config_sha256'],source_dynamic_payload_sha256=d['payload_sha256'],context_length=32768,chunk_size=4096,configured_layers=LAYERS,physical_gpu=args.worker_id+2,wall_seconds=time.time()-started,**FLAGS)
   if input_class=='DEBUG_INTERFACE_FIXTURE':result['fixture_provenance']=row['fixture_provenance'];result['FORMAL_RESET_4B_32K_4K_ELIGIBILITY_STATUS']='ZERO_ELIGIBLE_BY_CHUNK_GEOMETRY'
   store.commit(sid,result);print(json.dumps({'experiment':args.experiment,'sample':sid,'input_class':input_class,'committed':True}),flush=True)
 atomic(workspace/'DONE.json',dict(status='PASS',owned=len(owned),**FLAGS))
if __name__=='__main__':
 try:main()
 except Exception as e:
  atomic(W/'errors'/f'mechanism_{os.getpid()}.json',dict(status='RESOURCE_RUNTIME_FAILURE' if 'out of memory' in str(e).lower() or 'cuda error' in str(e).lower() else 'INTEGRATION_FAILURE',error=str(e),traceback=traceback.format_exc(),**FLAGS));raise
