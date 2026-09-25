#!/usr/bin/env python3
"""Authority-prepared formal worker. No model work occurs without an explicit CLI job."""
import os,sys,json,ast,types,math,hashlib,time,traceback,argparse
from pathlib import Path
import contracts as ct
sys.path.insert(0,str(ct.M));import runtime as rt;import execution
from worker import Store,atomic
sys.path.insert(0,str(ct.I));import inference_model.hf_qwen3.modeling_qwen3 as qm;import hf_models.hf_qwen3.ttt_state_core as core
LAYERS=ct.LAYERS;ACTIONS=ct.ACTIONS;FLAGS=dict(executor_version='V2',DEBUG_ONLY=False,DEBUG_INTERFACE_FIXTURE=False,PAPER_RESULT=True,FORMAL_SCIENTIFIC_RESULT=True)
def extract_native(setup):
 p=ct.S/'debug_worker.py';node=next(x for x in ast.parse(p.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='native_fixed');ns={**globals(),'setup':setup};exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),ns);return ns['native_fixed']
def executor_class():
 node=next(x for x in next(x for x in ast.parse(Path(execution.__file__).read_text()).body if isinstance(x,ast.ClassDef) and x.name=='Executor').body if isinstance(x,ast.FunctionDef) and x.name=='__init__')
 class Adapter(ast.NodeTransformer):
  def visit_Constant(self,n):return ast.copy_location(ast.Constant('authority/RULER_32K_AUTHORITY.json'),n) if n.value=='benchmark_manifest.json' else n
  def visit_BinOp(self,n):
   n=self.generic_visit(n)
   if isinstance(n.op,ast.Div) and isinstance(n.left,ast.Attribute) and isinstance(n.left.value,ast.Name) and n.left.value.id=='self' and n.left.attr=='dep' and isinstance(n.right,ast.Constant) and n.right.value=='code':return ast.copy_location(ast.Call(ast.Name('Path',ast.Load()),[ast.Constant(str(ct.I))],[]),n)
   return n
 ns=dict(execution.__dict__);exec(compile(ast.fix_missing_locations(ast.Module(body=[Adapter().visit(node)],type_ignores=[])),__file__,'exec'),ns)
 return type('Formal4KExecutor',(execution.Executor,),{'__init__':ns['__init__']})
class BenchmarkRows:
 def __init__(self):
  self.meta={x['project_sample_id']:x for x in ct.benchmark()};self.tasks={x['name']:x for x in ct.read(ct.B/'authority/RULER_32K_GENERATION_CONFIG.json')['tasks']};self.offsets={}
 def get(self,sid):
  m=self.meta[sid];path=ct.B/m['official_jsonl_path']
  if path not in self.offsets:
   offsets=[]
   with path.open('rb') as f:
    while True:
     pos=f.tell();line=f.readline()
     if not line:break
     offsets.append(pos)
   self.offsets[path]=offsets
  with path.open('rb') as f:f.seek(self.offsets[path][m['source_line']-1]);x=json.loads(f.readline())
  ct.require(hashlib.sha256(x['prompt'].encode()).hexdigest()==m['input_sha256'] and ct.canonical(x['answer'])==m['reference_sha256'],'BENCHMARK_SAMPLE_CONTENT_HASH')
  return dict(sample_id=sid,task=m['task'],sample_index=m['source_line']-1,_sample_index=m['global_index'],input=x['prompt'],reference=x['answer'],family=self.tasks[m['task']]['family'],max_new_tokens=m['tokens_to_generate'],expected_prompt_tokens=m['input_token_length'])

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);ap.add_argument('--worker-id',type=int,required=True);args=ap.parse_args();j=ct.read(args.job);ct.require(j['DEBUG_ONLY'] is False and j['execution_authorized'] is True,'FORMAL_JOB_REQUIRED');ct.source_guard();ct.validate_config(j['config'],j['phase']);ct.ready_guard(j['phase']);ct.require(ct.sha(ct.config_path(j['phase']))==j['config_sha256'],'JOB_CONFIG_HASH');ct.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='2,3' and args.worker_id in [0,1],'GPU_POLICY')
 import torch
 torch.set_num_threads(4);torch.cuda.set_device(args.worker_id);uuid=['fb7ba9a8-ce12-65a4-a9cf-5b5b7506b0ed','351e23d4-250b-6a70-2c66-d2fb0221a25a'][args.worker_id];ct.require(str(torch.cuda.get_device_properties(args.worker_id).uuid).removeprefix('GPU-')==uuid,'GPU_UUID')
 root=ct.under(j['run_root']);phase=j['phase'];store=Store(root,j['store_identity']);owned=j['sample_ids'][args.worker_id::2];workspace=root/'workers'/str(args.worker_id);workspace.mkdir(parents=True,exist_ok=True);bench=BenchmarkRows();pa=ct.read(j['model_parent_authority']);modelroot=Path(j['model_parent_authority']).parent
 setup=dict(model_identity=pa['files']['model.safetensors'],benchmark_identity=ct.BH,parent_authority_sha256=ct.sha(j['model_parent_authority']),config_sha256=ct.IH)
 c=dict(classification='FORMAL_AUTHORITY_DRIVEN',model_family='qwen3',mechanism=j.get('experiment','reverse'),configured_ttt_layers=LAYERS,actions=ACTIONS,chunk_size=4096,context_length=32768,model_path=str(modelroot),model_identity=setup['model_identity'],benchmark_root=str(ct.B),benchmark_identity=ct.BH)
 scorer={};p=ct.F/'scorers/common.py';exec(compile(p.read_text(),str(p),'exec'),scorer)
 baselines={a:{} for a in ACTIONS};fixed={}
 if phase!='fixed':
  for r in ct.read(j['fixed_aggregate'])['per_sample']:
   fixed[r['sample_id']]={a:r[a] for a in ACTIONS}
  for p in j['fixed_commit_paths']:
   r=ct.committed(p);baselines[r['action']][r['benchmark_sample_id']]={**r,'official_sample_score':r['score']}
 ctx=types.SimpleNamespace(run=workspace,code=ct.I,ckpt=modelroot,rows=[],by_id={},manifest={'tasks':[]},baseline=baselines,score_fn=scorer['per_sample_score']);dr=execution.runner(c,ct.F);sys.path.insert(0,str(ct.R/'src/reference_executor_v2_port'));from adapter import bind_runner;bind_runner(dr,core.__name__,c['chunk_size']);ctx.dr=dr;model=tok=ex=None;native=extract_native(setup)
 parents={r['sample_id']:r for r in j.get('cohort',[])}
 for sid in owned:
  pending=any(store.committed(sid+'|'+a) is None for a in ACTIONS) if phase=='fixed' else store.committed(sid) is None
  if not pending:continue
  row=bench.get(sid);ctx.by_id[sid]=row
  if phase=='mechanism':
   with store.own(sid):
    if store.committed(sid) is not None:continue
    if ex is None:ex=executor_class()(c,[row],fixed,workspace)
    ex.ctx.by_id[sid]=row;r=rt.validate_result(ex.execute(parents[sid]))
    if j['experiment']=='deletion':r['details']['deletion']['full_reset_score']=parents[sid]['full_reset_score']
    store.commit(sid,{**r,**FLAGS,'chunk_size':4096,'inference_config_sha256':ct.IH})
  else:
   if model is None:
    model,tok=dr.load_model(ctx)
    if phase=='fixed':
     for l in LAYERS:delattr(model.model.layers[l].mlp,'forward')
   if phase=='fixed':
    for action in ACTIONS:
     key=sid+'|'+action
     with store.own(key):
      if store.committed(key) is not None:continue
      r=native(ctx,model,tok,row,action);store.commit(key,{**r,'benchmark_sample_id':sid})
   else:
    with store.own(sid):
     if store.committed(sid) is not None:continue
     raw=dr.search_sample(ctx,model,tok,row,'FORMAL_DYNAMIC_LB',1,len(owned),workspace/'progress.json');n=raw['full_chunk_count'];ct.require(raw['candidate_count']<=8+56+(n-1)*28,'REPLAY_BUDGET')
     store.commit(sid,dict(task=row['task'],sample_best=raw['fixed_oracle_score'],dynamic_score=raw['dynamic_search_score'],action_sequence=raw['best_sequence'],chunk_count=n,replay_count=raw['candidate_count'],final_capture_replays=1,search_status='SUCCESS',completed=True,prediction_hash=raw['best_prediction_hash'],native_prediction_sha256=raw['best_prediction_hash'],model_identity=setup['model_identity'],benchmark_identity=ct.BH,inference_config_sha256=ct.IH,chunk_size=4096,raw=raw,errors=[],**FLAGS))
  atomic(workspace/'progress_commit.json',dict(sample_id=sid,worker_id=args.worker_id,physical_gpu=args.worker_id+2,time=time.time(),**FLAGS))
 atomic(workspace/'DONE.json',dict(status='PASS',owned_n=len(owned),time=time.time(),**FLAGS))
if __name__=='__main__':main()
