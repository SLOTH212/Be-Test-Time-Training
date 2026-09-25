"""CPU tiny state/kernel tests and portable compact historical statistics replay."""
import copy,json,os,sys,types,statistics
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import runtime as rt
import execution
R=rt.ROOT;L=[0,6,12,18,24]
def compact():
 import replay_historical as h
 for k in ['reverse','reset']:
  rs=rt.rows(R/f'tests/reference/{k}_scalar_results.jsonl');c=h.CFG[k]
  result,_,_=h.fragment(k,'dyn' if k=='reverse' else 'native','summary',{'records':rs,'rs':rs,'config':c,'c':c,'run':Path('unused')})
  h.check(result,json.loads((R/'audits/HISTORICAL_MECHANISM_MASTER_REPLAY.json').read_text())[k]['summary'],k+' compact exact')
 rs=rt.rows(R/'tests/reference/deletion_scalar_results.jsonl');co=rt.rows(next((R/'provenance/original/deletion/manifests').glob('*COHORT.jsonl')))
 result,_,_=h.fragment('deletion','by','summary',{'got':rs,'co':co,'c':h.CFG['deletion'],'r':Path('unused')})
 h.check(result,json.loads((R/'audits/HISTORICAL_MECHANISM_MASTER_REPLAY.json').read_text())['deletion']['summary'],'deletion compact')
 m,_=h.module('kp');by={};refs={x['canonical_sample_id']:x for x in rt.rows(R/'tests/reference/kp_source_states.jsonl')}
 for x in rt.rows(R/'tests/reference/kp_scalar_results.jsonl'):by.setdefault(x['canonical_sample_id'],{})[x['cell']]=x
 co=rt.rows(next((R/'provenance/original/kp/manifests').glob('*COHORT.jsonl')));ids=[x['canonical_sample_id'] for x in co];expected=json.loads((R/'tests/reference/expected/kp/aggregates/KP_PRIMARY_FACTORIAL_ANALYSIS.json').read_text())
 h.check(m.group_summary(ids,by,refs),expected['all_167'],'KP compact summary');es=[m.effects({c:by[s][c]['generation_start_relative_l2'] for c in m.CELLS}) for s in ids]
 h.check({k:m.boot([x[k] for x in es]) for k in es[0]},expected['paired_bootstrap'],'KP compact bootstrap')
 m,_=h.module('r2');data=rt.rows(R/'tests/reference/r2_scalar_results.jsonl');cfg=h.CFG['r2'];result,_,_=h.fragment('r2','gaps','primary',{'data':data,'cfg':cfg})
 b1=m.bootstrap(data,cfg['paired_bootstrap_seed'],20000,False);b2=m.bootstrap(data,cfg['task_stratified_bootstrap_seed'],20000,True)
 result.update(DYNAMIC_R2_PAIRED_BOOTSTRAP_95CI=b1['mean_dynamic_minus_r2_95ci'],DYNAMIC_R2_TASK_STRATIFIED_BOOTSTRAP_95CI=b2['mean_dynamic_minus_r2_95ci'],R2_GAIN_RETENTION_BOOTSTRAP_95CI=b1['aggregate_r2_gain_retention_95ci'])
 h.check(result,json.loads((R/'audits/HISTORICAL_MECHANISM_MASTER_REPLAY.json').read_text())['r2']['summary'],'R2 compact stats')

def tiny():
 import torch
 import torch.nn.functional as F
 torch.set_num_threads(1);torch.manual_seed(19)
 dep=rt.dependency();sys.path.insert(0,str(dep/'code'))
 from hf_models.hf_qwen3.ttt_state_core import TTTState,TTTStats,ttt_chunk_step,build_chunk_local_ntp_pairs
 def fixture(layers):
  arr=[]
  for l in range(max(layers)+1):
   m=torch.nn.Module();m.gate_proj=torch.nn.Linear(3,4,bias=False);m.up_proj=torch.nn.Linear(3,4,bias=False);m.down_proj=torch.nn.Linear(4,3,bias=False);m.act_fn=torch.nn.SiLU();m.ttt_proj=torch.nn.Linear(3,3,bias=False);m.ttt_conv=torch.nn.Conv1d(3,3,5,groups=3,bias=False);m.ttt_ntp_gate=torch.nn.Parameter(torch.ones(3));m.layer_idx=l;m.hidden_size=3;m.ttt_chunk=2;m.ttt_lr=1.;m.ttt_update_clip_norm=1e-5;m.ttt_target_type='gated_next_position_hidden';arr.append(types.SimpleNamespace(mlp=m))
  return types.SimpleNamespace(model=types.SimpleNamespace(layers=arr,ttt_mode=True))
 def tensor_hash(t):return rt.canonical(t.detach().tolist())
 for layers in [L,L+[30]]:
  cfg={'configured_ttt_layers':layers};b=execution.backend(cfg);model=fixture(layers);b.install_instrumented_dynamic_forward(model,tensor_hash);seq=['ALL','L0','L0'];x=torch.randn(1,7,3);target=torch.randn(1,7,3)
  for l in layers:
   m=model.model.layers[l].mlp;base=m.down_proj.weight.detach().clone();m._dynamic_sequence=seq;m._branch_id='tiny';m._reset_boundary_chunk=2
   out,weight=m.forward(x,target);assert out.shape==(1,7,3);assert torch.equal(base,m.down_proj.weight);assert len(m._reset_events)==1;assert m._reset_events[0]['post_reset_max_abs_residual']==0;assert m._reset_events[0]['pre_reset_delta_norm']>0
   if l!=0:assert torch.equal(weight,m.down_proj.weight)
  # Exact native five-layer source and adapted five-layer source produce identical kernels.
  if layers==L:
   original=rt.source_functions('provenance/original/reset/dependencies/instrumented_backend.py',['_scalar_stats','_norm','_dot','install_instrumented_dynamic_forward','materialize_scalar'],{'LAYERS':L,'ACTION_LAYERS':rt.mapping(L)})
   other=copy.deepcopy(model);original.install_instrumented_dynamic_forward(other,tensor_hash)
   for l in L:
    a=model.model.layers[l].mlp;z=other.model.layers[l].mlp;z._dynamic_sequence=seq;z._branch_id='tiny';z._reset_boundary_chunk=2
    ao,aw=a.forward(x,target);zo,zw=z.forward(x,target);assert torch.equal(ao,zo) and torch.equal(aw,zw)
  # K/P actual configure/injection operations, including the sixth L30 state.
  env={'torch':torch,'F':F,'TTTState':TTTState,'TTTStats':TTTStats,'ttt_chunk_step':ttt_chunk_step,'build_chunk_local_ntp_pairs':build_chunk_local_ntp_pairs,'LAYERS':layers,'ACTION_LAYERS':rt.mapping(layers)}
  c=rt.source_functions('provenance/original/kp/dependencies/closure_core.py',['install_formal','configure','restore'],env)
  c.install_formal(model);sequence=[f'L{layers[-1]}','L0','L0'];native_used={0,layers[-1]};injected={l:torch.full_like(model.model.layers[l].mlp.down_proj.weight,0.25+l/100) for l in layers}
  for used_expected in [set(layers),native_used]:
   used,stash=c.configure(model,sequence,1,{l:injected[l] for l in used_expected});assert used==used_expected
   for l in layers:
    m=model.model.layers[l].mlp
    if l in used:
     m.forward(x,target);assert torch.equal(m._cl_boundary['after'],injected[l])
    else:assert not hasattr(m,'ttt_conv')
   c.restore(model,stash);assert all(hasattr(model.model.layers[l].mlp,'ttt_conv') for l in layers)
 # Test exact source prefix-KV replacement against a CPU cache fixture, not a model.
 class Cache:
  def __init__(self,config=None):pass
  def update(self,k,v,l,kw=None):return k.clone(),v.clone()
 c=rt.source_functions('provenance/original/kp/dependencies/closure_core.py',['PrefixKVInjectCache'],{'TTTDynamicCache':Cache,'th':tensor_hash})
 dk=torch.ones(1,2,3,4);dv=torch.full_like(dk,2);cache=c.PrefixKVInjectCache(donor_prefix={30:(dk,dv)},prefix_len=3);k=torch.zeros(1,2,8,4);v=torch.zeros_like(k);outk,outv=cache.update(k,v,30)
 assert torch.equal(outk[:,:,:3],dk) and torch.equal(outv[:,:,:3],dv);assert torch.equal(outk[:,:,3:],k[:,:,3:]);assert torch.equal(outv[:,:,3:],v[:,:,3:]);assert k.count_nonzero()==0
 # No model forward, generation, checkpoint loading, CUDA call, or GPU query occurs.

def main():
 tiny();compact();print(json.dumps({'MECHANISM_RUNTIME_LEVEL1':'PASS','tiny_backend_and_kp_state_tests':'PASS','L30_coverage':'PASS','compact_aggregation_and_bootstrap':'PASS','inference_executed':False,'gpu_used':False}))
if __name__=='__main__':main()
