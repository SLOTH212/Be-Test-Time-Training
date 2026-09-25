"""Qwen3 execution adapters consuming frozen inference and formal mechanism code.

This module performs no inference at import time. Model execution requires the
explicit CLI --execute flag. Original tensor operations are compiled unchanged;
the literal chunk size and configured layer/action globals are parameterized.
"""
import ast,hashlib,json,sys,time,types,statistics
from pathlib import Path
import runtime as rt

def runner(cfg,dep):
 p=dep/'dynamic/formal_dynamic_runner.py';tree=ast.parse(p.read_text())
 # Exclude historical Context and command entry points with fixed filesystem paths.
 selected=[x for x in tree.body if isinstance(x,(ast.Import,ast.ImportFrom,ast.FunctionDef,ast.ClassDef)) and getattr(x,'name',None) not in ['Context','main']]
 class Chunk(ast.NodeTransformer):
  def visit_Constant(self,n):return ast.copy_location(ast.Constant(cfg['chunk_size']),n) if type(n.value) is int and n.value==1024 else n
 tree=ast.fix_missing_locations(ast.Module(body=[Chunk().visit(n) for n in selected],type_ignores=[]));ns={'__name__':'mechanism_frozen_inference','TOL':1e-12,'CHECKPOINT_HASH':cfg['model_identity'],'LAYERS':cfg['configured_ttt_layers'],'ACTIONS':cfg['actions'],'ACTION_LAYERS':rt.mapping(cfg['configured_ttt_layers']),'ACTION_MODE':{a:a for a in cfg['actions']}}
 exec(compile(tree,str(p),'exec'),ns);return types.SimpleNamespace(**ns)

def backend(cfg):
 names=['_scalar_stats','_norm','_dot','install_instrumented_dynamic_forward','materialize_scalar']
 return rt.source_functions('provenance/original/reset/dependencies/instrumented_backend.py',names,{'LAYERS':cfg['configured_ttt_layers'],'ACTION_LAYERS':rt.mapping(cfg['configured_ttt_layers'])})

def kp_modules(cfg,dr,model):
 import torch
 import torch.nn.functional as F
 from inference_model.hf_qwen3.modeling_qwen3 import TTTDynamicCache
 from hf_models.hf_qwen3.ttt_state_core import TTTState,TTTStats,ttt_chunk_step,build_chunk_local_ntp_pairs
 env={'torch':torch,'F':F,'dr':dr,'TTTDynamicCache':TTTDynamicCache,'TTTState':TTTState,'TTTStats':TTTStats,'ttt_chunk_step':ttt_chunk_step,'build_chunk_local_ntp_pairs':build_chunk_local_ntp_pairs,'LAYERS':cfg['configured_ttt_layers'],'ACTION_LAYERS':rt.mapping(cfg['configured_ttt_layers'])}
 c=rt.source_functions('provenance/original/kp/dependencies/closure_core.py',['th','base','PrefixKVInjectCache','install_formal','configure','restore','run_branch','donor_prefix'],env,chunk=cfg['chunk_size'])
 f=rt.source_functions('provenance/original/kp/dependencies/factorial_core.py',['run_prompt','run_reward'],{'torch':torch,'c':c,'LAYERS':cfg['configured_ttt_layers'],'base':c.base,'th':c.th},chunk=cfg['chunk_size'])
 w=rt.source_functions('provenance/original/kp/scripts/kp_worker.py',['cmp','Recorder','prefix_hashes','first_divergence'],{},chunk=cfg['chunk_size'],integer_overrides={28:len(model.model.layers)})
 # Recorder's final-norm layer label is diagnostic only; adapt the historical 28.
 # All actual attention/MLP hooks enumerate model.model.layers directly.
 c.install_formal(model);return c,f,w

class Executor:
 def __init__(self,cfg,benchmark_rows,fixed,output_root):
  self.cfg=rt.validate_config(cfg);self.dep=rt.dependency();self.dr=runner(cfg,self.dep)
  sys.path.insert(0,str(self.dep/'code'))
  p=self.dep/'scorers/common.py';ns={'__name__':'mechanism_scorer'};exec(compile(p.read_text(),str(p),'exec'),ns)
  ctx=types.SimpleNamespace(run=Path(output_root),code=self.dep/'code',ckpt=Path(cfg['model_path']),rows=benchmark_rows,by_id={r['sample_id']:r for r in benchmark_rows},manifest={'tasks':sorted({r['task'] for r in benchmark_rows})},baseline={a:{sid:{'official_sample_score':s[a]} for sid,s in fixed.items()} for a in cfg['actions']},score_fn=ns['per_sample_score'])
  mc=json.loads((ctx.ckpt/'config.json').read_text())
  if max(cfg['configured_ttt_layers'])>=mc['num_hidden_layers']:raise ValueError('MODEL_LAYER_COVERAGE')
  if rt.sha(ctx.ckpt/'model.safetensors')!=cfg['model_identity']:raise ValueError('MODEL_IDENTITY')
  if rt.sha(Path(cfg['benchmark_root'])/'benchmark_manifest.json')!=cfg['benchmark_identity']:raise ValueError('BENCHMARK_IDENTITY')
  self.ctx=ctx;self.model,self.tok=self.dr.load_model(ctx);self.base=self.dr.base_hash(self.model)
  self.b=backend(cfg) if cfg['mechanism'] in ['reset','deletion'] else None
  if self.b:self.b.install_instrumented_dynamic_forward(self.model,self.dr.tensor_hash)
  self.kp=kp_modules(cfg,self.dr,self.model) if cfg['mechanism']=='kp' else None
 def evaluate(self,sample,sequence,reset_layers=(),boundary=None):
  if self.b:
   for l in self.cfg['configured_ttt_layers']:
    m=self.model.model.layers[l].mlp;m._reset_boundary_chunk=boundary if l in reset_layers else None;m._instrumentation_enabled=True;m._state_events=[];m._reset_events=[]
  ev=self.dr.Evaluator(self.ctx,self.model,self.tok,self.ctx.by_id[sample['sample_id']],'mechanism_'+self.cfg['mechanism'],1,1,self.ctx.run/'status/evaluator.json')
  try:r=ev.evaluate(tuple(sequence),capture=True,force=True)
  finally:ev.finish()
  if self.dr.base_hash(self.model)!=self.base:raise RuntimeError('BASE_WEIGHT_MUTATION')
  if r['generation_update_count']!=0 or not r['recompute_downstream_updates'] or r['stored_delta_stitching']:raise RuntimeError('REPLAY_SEMANTICS')
  events=[]
  if self.b:
   for l in self.cfg['configured_ttt_layers']:events+=self.b.materialize_scalar(self.model.model.layers[l].mlp._reset_events)
   if sorted(x['layer'] for x in events)!=sorted(reset_layers):raise RuntimeError('RESET_LAYER_COVERAGE')
   if any(x['chunk_index']!=boundary or x['post_reset_max_abs_residual']!=0 for x in events):raise RuntimeError('RESET_BOUNDARY_OR_BASE_PARITY')
  return r,events
 def execute(self,s):
  seq=s['action_sequence'];mech=self.cfg['mechanism'];layers=self.cfg['configured_ttt_layers'];started=time.time()
  if mech=='kp':details=self.execute_kp(s);score=details['cells']['F11']['terminal_reward']
  else:
   native,_=self.evaluate(s,seq)
   if native['score']!=s['dynamic_score'] or native['prediction_hash']!=s['native_prediction_sha256']:raise RuntimeError('NATIVE_PARENT_PARITY')
   if mech=='reverse':r,_=self.evaluate(s,rt.reverse(seq));score=r['score'];details={'transformed_sequence':rt.reverse(seq),'result':r}
   elif mech=='reset':
    p=rt.reset_screen(seq,layers)
    if not p['eligible']:raise ValueError('RESET_INELIGIBLE')
    r,events=self.evaluate(s,seq,layers,p['boundary']);score=r['score'];details={'screening':p,'reset_events':events,'result':r}
   elif mech=='deletion':
    b=rt.reset_screen(seq,layers)['boundary'];p=rt.deletion_plan(seq,b,layers)
    if not p['eligible']:raise ValueError('DELETION_INELIGIBLE')
    branches=[]
    for l in p['selected_component_layers']:
     r,e=self.evaluate(s,seq,[l],b)
     if e[0]['pre_reset_delta_norm']<=0:raise RuntimeError('TARGET_NOT_NONZERO')
     nh={int(e['layer']):e['pre_state_hash'] for e in native['chunk_layer_events'] if e['chunk_index']==b}
     rh={int(e['layer']):e['pre_state_hash'] for e in r['chunk_layer_events'] if e['chunk_index']==b}
     if any(nh[z]!=rh[z] for z in layers if z!=l):raise RuntimeError('NON_TARGET_BOUNDARY_PARITY')
     branches.append({'target_layer':l,'result':r,'reset_events':e,'non_target_boundary_parity':True})
    score=statistics.fmean(x['result']['score'] for x in branches);sham=None
    if p['sham_layer'] is not None:
     r,e=self.evaluate(s,seq,[p['sham_layer']],b)
     if e[0]['pre_reset_delta_norm']!=0 or r['score']!=native['score'] or r['prediction_hash']!=native['prediction_hash']:raise RuntimeError('SHAM_PARITY')
     sham={'target_layer':p['sham_layer'],'result':r,'reset_events':e}
    details={'plan':p,'selective_branches':branches,'sham':sham}
   elif mech=='r2':
    candidates=[]
    for c in rt.r2_candidates(len(seq),layers):
     r,_=self.evaluate(s,c['sequence']);candidates.append({**c,'score':r['score'],'result':r})
    winner,score,ties=rt.r2_winner(candidates,layers);details={'winner':winner,'tie_count':ties,'candidate_count':len(candidates),'off_score':s['fixed_scores']['OFF'],'candidates':candidates,'restricted_family':'exactly one transition; hindsight terminal reward; not global trajectory optimum'}
   else:raise ValueError('CONFIGURE_A_MECHANISM')
  return {'schema':'MECHANISM_SAMPLE_V1','model_identity':self.cfg['model_identity'],'benchmark_identity':self.cfg['benchmark_identity'],'sample_id':s['sample_id'],'task':s['task'],'benchmark_global_index':s['benchmark_global_index'],'parent_dynamic_trajectory':seq,'parent_dynamic_score':s['dynamic_score'],'sample_best':s['sample_best'],'intervention':mech,'intervention_score':score,'details':{mech:details},'errors':[],'timing':{'wall_seconds':time.time()-started}}
 def execute_kp(self,s):
  c,f,w=self.kp;layers=self.cfg['configured_ttt_layers'];seq=s['action_sequence'];b=rt.reset_screen(seq,layers)['boundary'];plan=rt.kp_plan(seq,b,layers);prefix=b-1;row=self.ctx.by_id[s['sample_id']];n=len(self.tok(row['input'],add_special_tokens=True)['input_ids'])
  def prompt(schedule,rec=None,inject=None,donor=None):return f.run_prompt(self.ctx,self.model,self.tok,row,schedule,prefix,inject=inject,donor=donor,trace_callback=None if rec is None else rec.internal,return_cache=True,generate=False)
  rn=w.Recorder(prefix,len(seq),n,'native',c.th);rn.hooks(self.model)
  try:native=prompt(seq,rn)
  finally:rn.close()
  nr=f.run_reward(self.ctx,self.model,self.tok,row,seq,prefix)
  if nr['score']!=s['dynamic_score'] or nr['prediction_hash']!=s['native_prediction_sha256']:raise RuntimeError('NATIVE_PARITY')
  control=prompt(plan['control_sequence']);full={l:native['boundary'][l] for l in layers};donor=c.donor_prefix(native['prompt_cache'],prefix);cells={}
  for name,p in plan['cells'].items():
   inject={l:full[l] for l in p['injection_layers']};kd=donor if p['K_bit'] else None;rr=w.Recorder(prefix,len(seq),n,'condition',c.th,rn.ref);rr.hooks(self.model)
   try:x=prompt(plan['control_sequence'],rr,inject,kd)
   finally:rr.close()
   if {l:c.th(x['boundary'][l]) for l in layers}!={l:c.th(full[l]) for l in layers}:raise RuntimeError('W_ISOLATION')
   expected=native if p['K_bit'] else control
   if w.prefix_hashes(x['prompt_cache'],prefix,c.th)!=w.prefix_hashes(expected['prompt_cache'],prefix,c.th):raise RuntimeError('K_ISOLATION')
   if [l for l in layers if x['prompt_ttt_occupancy'][l]]!=p['injection_layers']:raise RuntimeError('P_ISOLATION')
   if x['input_ids_hash']!=native['input_ids_hash'] or x['attention_mask_hash']!=native['attention_mask_hash']:raise RuntimeError('FUTURE_INPUT_PARITY')
   q=w.cmp(native['prompt_logits'],x['prompt_logits'],c.th);reward=f.run_reward(self.ctx,self.model,self.tok,row,plan['control_sequence'],prefix,inject=inject,donor=kd)
   levels={'L1':all(z['exact_equal'] for z in rr.rows if z['region'] in ['FIRST_CONTINUATION_CHUNK','FIRST_POST_BOUNDARY_UPDATE']),'L2':all(z['exact_equal'] for z in rr.rows if z['region']=='COMPLETE_CONTINUATION_CHUNK'),'L3':all(z['exact_equal'] for z in rr.rows if z['region']=='INCOMPLETE_PROMPT_TAIL'),'L4':q['exact_equal'],'L5':reward['prediction_hash']==nr['prediction_hash']};levels['L6']=all(levels.values())
   cells[name]={'closure':levels,'post_injection_fast_weight_sha256':{str(l):c.th(x['boundary'][l]) for l in layers},'K_prefix_hashes':w.prefix_hashes(x['prompt_cache'],prefix,c.th),'P_materialized_layers':p['injection_layers'],'future_input_equal':True,'generation_update_count':reward['generation_update_count'],'generation_start_relative_l2':q['relative_l2_diff'],'terminal_reward':reward['score'],'prediction_sha256':reward['prediction_hash'],'trace_comparisons':rr.rows,'first_divergence':w.first_divergence(rr.rows,levels['L4'],levels['L5']),**p}
  if not cells['F11']['closure']['L6']:raise RuntimeError('F11_NONCLOSURE')
  if self.dr.base_hash(self.model)!=self.base:raise RuntimeError('BASE_WEIGHT_MUTATION')
  return {'plan':plan,'cells':cells,'native_reward':nr['score']}
