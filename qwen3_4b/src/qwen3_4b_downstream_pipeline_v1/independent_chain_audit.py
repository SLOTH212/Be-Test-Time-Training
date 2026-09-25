#!/usr/bin/env python3
"""Read-only independent verification; writes only its own final audit receipts."""
import os,sys,json,hashlib,math,statistics,collections,socket,getpass,time
from pathlib import Path
import contracts as ct
R=ct.R;W=R/'work/qwen3_4b_downstream_pipeline_debug_v1';A=R/'audits/qwen3_4b_downstream_debug_v1';FLAGS=dict(DEBUG_ONLY=True,PAPER_RESULT=False,FORMAL_SCIENTIFIC_RESULT=False)
def read(p):return json.loads(Path(p).read_text())
def commits(root):
 out=[]
 for p in sorted((root/'samples').glob('*.json')):
  x=read(p);assert x['status']=='COMMITTED' and x['payload_sha256']==ct.canonical({k:v for k,v in x.items() if k!='payload_sha256'});assert x['DEBUG_ONLY'] is True and x['PAPER_RESULT'] is False and x['FORMAL_SCIENTIFIC_RESULT'] is False;out.append((p,x))
 return out

def main():
 assert socket.gethostname()=='amax' and getpass.getuser()=='USER';ct.verify_runtime();ct.benchmark();ct.source_guard();setup=read(W/'SETUP.json');ct.parent(Path(setup['model'])/'INFERENCE_PARENT_AUTHORITY.json',debug=True)
 ids=[x['sample_id'] for x in [json.loads(l) for l in (W/'DEBUG_SAMPLE_MANIFEST.jsonl').open()]];assert len(ids)==8 and len(set(ids))==8
 fixed=commits(W/'fixed');assert len(fixed)==64;cells={(x['benchmark_sample_id'],x['action']):x for _,x in fixed};assert set(cells)=={(sid,a) for sid in ids for a in ct.ACTIONS}
 for (sid,a),x in cells.items():
  assert x['sample_id']==sid+'|'+a and x['model_identity']==setup['model_identity'] and x['benchmark_identity']==ct.BH and x['inference_config_sha256']==ct.IH
  assert x['configured_layers']==ct.LAYERS and x['chunk_size']==4096 and x['context_length']==32768;assert x['complete_chunks']==x['prompt_token_count']//4096 and x['tail_tokens']==x['prompt_token_count']%4096
  assert x['generation_update_count']==0 and x['base_unchanged'];used=ct.LAYERS if a=='ALL' else ([] if a=='OFF' else [int(a[1:])]);assert x['layer_update_counts']=={str(l):x['complete_chunks']*int(l in used) for l in ct.LAYERS};assert math.isfinite(x['score'])
 agg=read(W/'FIXED8_DEBUG_AGGREGATE.json');assert len(agg['per_sample'])==8
 for x in agg['per_sample']:
  sid=x['sample_id'];sb=max(cells[sid,a]['score'] for a in ct.ACTIONS);assert sb==x['sample_best'];assert x['sample_best_actions']==[a for a in ct.ACTIONS if abs(cells[sid,a]['score']-sb)<=1e-12]
 ds=commits(W/'dynamic');assert len(ds)==2;dmap={x['sample_id']:x for _,x in ds};assert set(dmap)=={x['sample_id'] for x in agg['per_sample'] if x['sample_best']<1-1e-12}
 for sid,x in dmap.items():
  assert x['chunk_count']==cells[sid,'OFF']['complete_chunks'];assert x['replay_budget']==8+56+(x['chunk_count']-1)*28;raw=x['raw'];assert x['replay_count']==len(raw['candidates'])<=x['replay_budget'];assert len({tuple(c['sequence']) for c in raw['candidates']})==x['replay_count']
  assert all(c['clean_full_replay'] and c['recompute_downstream_updates'] and not c['stored_delta_stitching'] for c in raw['candidates'])
  assert x['sample_best']==max(cells[sid,a]['score'] for a in ct.ACTIONS);assert x['dynamic_score']==max(c['score'] for c in raw['candidates']);assert len(x['action_sequence'])==x['chunk_count'] and set(x['action_sequence'])<=set(ct.ACTIONS);assert x['final_capture_replays']==1
  assert x['parent_fixed_config']==read(W/'fixed/CONFIG_IDENTITY.json')['sha256'];assert x['model_identity']==setup['model_identity'] and x['benchmark_identity']==ct.BH and x['inference_config_sha256']==ct.IH
 cohort=read(W/'DEBUG_COHORT.json');assert {x['sample_id'] for x in cohort['improved']}=={sid for sid,x in dmap.items() if x['dynamic_score']>x['sample_best']+1e-12};assert not cohort['worse']
 mechs={};paths={}
 for ex,cl in [('reverse','REAL_CHAIN'),('r2','REAL_CHAIN'),('reset','DEBUG_INTERFACE_FIXTURE'),('kp','DEBUG_INTERFACE_FIXTURE'),('deletion','DEBUG_INTERFACE_FIXTURE')]:
  records=commits(W/'mechanism'/cl/ex);assert len(records)==2,(ex,len(records));mechs[ex]=[]
  for p,x in records:
   assert x['input_class']==cl and x['DEBUG_INTERFACE_FIXTURE']==(cl=='DEBUG_INTERFACE_FIXTURE');assert x['configured_layers']==ct.LAYERS and x['chunk_size']==4096 and x['context_length']==32768;assert x['physical_gpu'] in [2,3];assert x['model_identity']==setup['model_identity'] and x['benchmark_identity']==ct.BH and x['inference_config_sha256']==ct.IH
   real=x['sample_id'].removeprefix('DEBUG_INTERFACE_FIXTURE:');assert x['source_dynamic_payload_sha256']==dmap[real]['payload_sha256']
   if ex=='reverse':assert x['parent_dynamic_trajectory']==dmap[real]['action_sequence'] and x['details']['reverse']['transformed_sequence']==list(reversed(dmap[real]['action_sequence']))
   if ex=='reset':
    events=x['details']['reset']['reset_events'];assert {e['layer'] for e in events}==set(ct.LAYERS) and all(e['chunk_index']==2 and e['post_reset_max_abs_residual']==0 for e in events);assert x['post_intervention_native_repeatability']
   if ex=='kp':
    cells2=x['details']['kp']['cells'];assert set(cells2)=={'F00','F10','F01','F11'} and cells2['F11']['closure']['L6'];assert all(set(c['post_injection_fast_weight_sha256'])=={str(l) for l in ct.LAYERS} for c in cells2.values());assert all(c['generation_update_count']==0 for c in cells2.values())
   if ex=='deletion':
    dd=x['details']['deletion'];assert dd['plan']['selected_component_layers']==[30] and dd['sham']['target_layer']==0;assert dd['selective_branches'][0]['reset_events'][0]['pre_reset_delta_norm']>0 and dd['sham']['reset_events'][0]['pre_reset_delta_norm']==0;assert x['post_intervention_native_repeatability']
   if ex=='r2':assert x['executed_candidates']==4 and not x['R2_FULL_DEBUG_SEARCH_EXECUTED'] and not x['scientific_exact_R2_result'] and x['native_parent_replay_verified'];assert x['enumerated_full_family_count']==8*7*(dmap[real]['chunk_count']-1)
   if cl=='DEBUG_INTERFACE_FIXTURE':assert x['fixture_provenance']['prompt_tokens']==32768 and x['fixture_provenance']['original_prompt_preserved_as_prefix'] and not x['fixture_provenance']['filler_uses_answers_or_other_samples'];assert x['FORMAL_RESET_4B_32K_4K_ELIGIBILITY_STATUS']=='ZERO_ELIGIBLE_BY_CHUNK_GEOMETRY'
   mechs[ex].append(dict(path=str(p),payload_sha256=x['payload_sha256'],physical_gpu=x['physical_gpu'],input_class=cl));paths[str(p)]=ct.sha(p)
  assert {x['physical_gpu'] for _,x in records}=={2,3}
 for worker in [0,1]:assert read(A/f'STATE_ISOLATION_WORKER{worker}.json')['status']=='PASS'
 for name in ['fixed','dynamic','reverse','reset','deletion','kp','r2']:
  phase=read(W/(name+'_PHASE_STATE.json'));assert phase['status']=='PASS' and phase['exit_codes']==[0,0]
 for phase in ['fixed','dynamic','mechanism']:assert not (R/'runs'/('qwen3_4b_ruler32k_'+phase+'_4k_formal_v1')).exists()
 assert read(A/'DOWNSTREAM_NEGATIVE_CONTRACT_TESTS.json')['status']=='PASS';assert read(A/'RESUME_FINAL_AUDIT.json')['status']=='PASS'
 audit=dict(status='PASS',INFERENCE_MECHANISM_FULL_DEBUG_CHAIN='PASS',model_authority_continuity=True,benchmark_ids_verified=True,chunk4k_continuity=True,actions=ct.ACTIONS,layers=ct.LAYERS,debug_mislabeled_formal_n=0,fixed_cells=64,dynamic_samples=2,cohort_natural_improved_n=cohort['natural_improved_n'],mechanisms=mechs,FORMAL_RESET_4B_32K_4K_ELIGIBILITY_STATUS='ZERO_ELIGIBLE_BY_CHUNK_GEOMETRY',debug_fixture_resolves_formal_scientific_eligibility=False,FORMAL_FIXED8_STARTED=False,FORMAL_DYNAMIC_STARTED=False,FORMAL_MECHANISM_STARTED=False,**FLAGS)
 (A/'END_TO_END_DEBUG_CHAIN_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
 paths.update({str(p):ct.sha(p) for p,_ in fixed+ds});paths.update({str(p):ct.sha(p) for p in A.glob('*.json')});paths[str(R/'provenance/QWEN3_4B_DOWNSTREAM_SOURCE_MANIFEST_V1.json')]=ct.sha(R/'provenance/QWEN3_4B_DOWNSTREAM_SOURCE_MANIFEST_V1.json')
 authority=dict(schema='QWEN3_4B_DOWNSTREAM_PIPELINE_DEBUG_AUTHORITY_V1',status='PASS_WITH_WARNING',model_parent=str(Path(setup['model'])/'INFERENCE_PARENT_AUTHORITY.json'),model_parent_sha256=setup['parent_authority_sha256'],inference_config_sha256=ct.IH,benchmark_authority_sha256=ct.BH,mechanism_package_sha256=ct.MH,files=paths,chain_audit=str(A/'END_TO_END_DEBUG_CHAIN_AUDIT.json'),FORMAL_RESET_4B_32K_4K_ELIGIBILITY_STATUS='ZERO_ELIGIBLE_BY_CHUNK_GEOMETRY',debug_fixture_resolves_formal_scientific_eligibility=False,**FLAGS)
 p=R/'provenance/QWEN3_4B_DOWNSTREAM_PIPELINE_DEBUG_AUTHORITY_V1.json';p.write_text(json.dumps(authority,indent=2,sort_keys=True)+'\n');p.with_suffix('.sha256').write_text(ct.sha(p)+'  '+p.name+'\n');print(json.dumps(dict(status='PASS_WITH_WARNING',authority=str(p),sha256=ct.sha(p))))
if __name__=='__main__':main()
