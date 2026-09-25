"""CPU-only fail-closed identities for downstream execution."""
import json,hashlib,math,collections
from pathlib import Path
R=Path('/path/to/ttt');S=R/'src/qwen3_4b_downstream_pipeline_v1';B=R/'shared/benchmarks/ruler_32k_standard_13task500_v1';I=R/'src/inference_runtime_4k_stage2_v1/code';M=R/'releases/ttt_mechanism_runtime_v1';F=R/'vendor/frozen/ttt_inference_runtime_v1'
LAYERS=[0,6,12,18,24,30];ACTIONS=['OFF','L0','L6','L12','L18','L24','L30','ALL'];BH='b62966b2754fa0d1cfef2c9aa3cef8d4b2bbae2a664495eb733c2ed5e5a4bebd';IH='210db096574f2170c2f2194338f51ff767977b9abd2c56a031f60caa1d3617a9';FH='65a0d7bd9c8801c92fc30477ab8662d3eb52261612eff2acef0bbeb3edf6ba51';MH='ce9ee7c7066d0413e4fcf90b69543cfa605ac1761af8d849ddb2d61ccd6fb1a8'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(16<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def require(ok,why):
 if not ok:raise ValueError(why)
def under(p,root=R):
 p=Path(p).resolve();require(p.is_relative_to(root.resolve()),'PATH_OUTSIDE_AUTHORIZED_ROOT');return p
def config_path(phase):
 suffix='executor_v2' if phase in ['fixed','dynamic'] else 'formal_v1'
 return R/'configs'/f'qwen3_4b_ruler32k_{dict(fixed="fixed8",dynamic="dynamic_lb",mechanism="mechanism")[phase]}_4k_{suffix}.yaml' 
def validate_config(c,phase):
 if phase in ['fixed','dynamic']:require(c.get('executor_version')=='V2','V2_CONFIG_REQUIRED')
 expect=dict(context_length=32768,chunk_size=4096,layers=LAYERS,actions=ACTIONS,ttt_lr=1.0,delta_clip_frobenius=1e-5,update_order='APPLY_THEN_UPDATE',reset_scope='PER_SAMPLE',tail_update=False,generation_update=False,physical_gpu_allowlist=[2,3],workers=2,workers_per_gpu=1,active_sample_per_worker=1)
 for k,v in expect.items():require(c.get(k)==v,'CONFIG_'+k.upper())
 require(c['phase']==phase,'CONFIG_PHASE');require(c['inference_config_sha256']==IH and c['benchmark_authority_sha256']==BH,'CONFIG_DEPENDENCY_IDENTITY')
 if phase=='dynamic':require(c['dynamic']==dict(beam=4,sweeps=1,position_order='forward',downstream_recomputation=True,stored_delta_stitching=False),'DYNAMIC_PROTOCOL');require(c['excluded_tasks']==['ruler_cwe_16k'],'DYNAMIC_TASK_POLICY')
 if phase=='mechanism':require(c['experiments']==['reverse','reset','deletion','kp','r2'] and c['mechanism_package_sha256']==MH,'MECHANISM_PROTOCOL')
 return c
def verify_runtime():
 require(sha(R/'configs/qwen3_4b_stage2_inference_4k_v1.json')==IH,'INFERENCE_CONFIG_HASH');c=read(R/'configs/qwen3_4b_stage2_inference_4k_v1.json')
 for p,h in c['runtime_file_sha256'].items():require(sha(I/p)==h,'INFERENCE_SOURCE_HASH')
 require(sha(R/'shared/packages/ttt_inference_runtime_v1/ttt_inference_runtime_v1.tar.gz')==FH,'BASE_PACKAGE_HASH');require(sha(R/'packages/ttt_mechanism_runtime_v1.tar.gz')==MH,'MECHANISM_PACKAGE_HASH')
 for root in [F,M]:
  for p,h in read(root/'provenance/RELEASE_MANIFEST.json')['files'].items():require(sha(root/p)==h,'FROZEN_SOURCE_HASH')
 return c
def benchmark():
 p=B/'authority/RULER_32K_AUTHORITY.json';require(sha(p)==BH,'BENCHMARK_AUTHORITY_HASH');b=read(p);require((b['total_n'],b['task_n'],b['samples_per_task'],b['context_length'],b['seed'])==(6500,13,500,32768,42),'BENCHMARK_INVENTORY')
 require(sha(B/'manifests/SAMPLE_INDEX.jsonl')==b['sample_index_sha256'],'BENCHMARK_INDEX_HASH');rows=[json.loads(l) for l in (B/'manifests/SAMPLE_INDEX.jsonl').open()];require(len(rows)==6500 and len({x['project_sample_id'] for x in rows})==6500,'BENCHMARK_IDS');require(set(collections.Counter(x['task'] for x in rows).values())=={500},'BENCHMARK_TASK_COUNTS');return rows

def parent(path,debug=False,rehash_weights=True):
 path=under(path);a=read(path)
 if not debug:require(a.get('DEBUG_ONLY') is False and a.get('PARENT_CLASS')!='DEBUG_ONLY','DEBUG_PARENT_FORBIDDEN')
 if not debug:require(sha(path)==read(config_path('fixed'))['model_parent_authority_sha256'],'V2_MODEL_PARENT_BINDING')
 else:require(a.get('DEBUG_ONLY') is True,'EXPLICIT_DEBUG_PARENT_REQUIRED')
 require(a.get('model_identity')=='Qwen3-4B' and a.get('ttt_layers')==LAYERS,'PARENT_MODEL_SCHEMA');require(a.get('context_length')==32768 and a.get('inference_ttt_chunk_size')==4096 and a.get('inference_reference_sha256')==IH,'PARENT_INFERENCE_SCHEMA')
 root=path.parent;mc=read(root/'config.json');require(mc['num_hidden_layers']==36,'PARENT_LAYER_COUNT')
 require(all(n in a['files'] for n in ['config.json','tokenizer.json','tokenizer_config.json','state_dict_audit.json']),'PARENT_FILE_MANIFEST')
 require(any(n.endswith('.safetensors') for n in a['files']),'PARENT_MODEL_SHARDS')
 for n,h in a['files'].items():
  q=under(root/n,root);require(q.is_file(),'PARENT_FILE_MISSING')
  if rehash_weights or not n.endswith('.safetensors'):require(sha(q)==h,'PARENT_FILE_HASH:'+n)
 source=under(a['source_stage2_authority']);require(sha(source)==a['source_stage2_authority_sha256'],'STAGE2_PARENT_AUTHORITY_HASH');stage=read(source)
 if not debug:
  require(stage.get('DEBUG_ONLY') is not True and stage.get('STAGE2_FINAL_INTEGRITY')=='PASS' and stage.get('stage2_completed') is True,'STAGE2_FINAL_REQUIRED');require(stage.get('record_cursor')==925 and stage.get('cumulative_tokens')==30000000,'STAGE2_FINAL_COUNTERS')
  manifest=under(stage['checkpoint_path'])/'manifest.json';require(sha(manifest)==a['source_checkpoint_manifest_sha256']==stage['checkpoint_manifest_sha256'],'STAGE2_FINAL_MANIFEST_HASH');m=read(manifest);require(m['complete'] and m['progress']['stage']==2 and m['progress']['record_cursor']==925 and m['progress']['cumulative_tokens']==30000000,'STAGE2_FINAL_ACTUAL_PROGRESS')
 keys=read(root/'state_dict_audit.json')['keys']
 for l in LAYERS:
  for n in ['ttt_proj.weight','ttt_conv.weight','ttt_ntp_gate']:require(f'model.layers.{l}.mlp.{n}' in keys,'PARENT_TTT_KEYS')
 tok=read(B/'authority/TOKENIZER_AUTHORITY.json')
 for k,n in [('tokenizer_json_sha256','tokenizer.json'),('tokenizer_config_sha256','tokenizer_config.json'),('special_tokens_map_sha256','special_tokens_map.json')]:require(a['files'][n]==tok[k],'TOKENIZER_CONTRACT')
 return a

def committed(path):
 x=read(path);require(x.get('status')=='COMMITTED','UNCOMMITTED_RESULT');require(x.get('payload_sha256')==canonical({k:v for k,v in x.items() if k!='payload_sha256'}),'COMMIT_HASH');return x

def final_authority(path,phase,expected_ids=None):
 path=under(path);a=read(path);require(a.get('executor_version')=='V2' and a.get('source_manifest_sha256')==source_guard(),'V2_BASELINE_SOURCE_REQUIRED');require(a.get('schema')==phase.upper()+'_FINAL_AUTHORITY_V1' and a.get('status')=='PASS' and a.get('DEBUG_ONLY') is False,'FORMAL_FINAL_AUTHORITY_REQUIRED');require(a['actions']==ACTIONS and a['layers']==LAYERS and a['chunk_size']==4096 and a['benchmark_identity']==BH,'FINAL_PROTOCOL_IDENTITY')
 for p,h in a['files'].items():require(sha(under(p))==h,'FINAL_FILE_HASH')
 records=[]
 for p,h in a['commits'].items():require(sha(under(p))==h,'FINAL_COMMIT_FILE_HASH');x=committed(p);require(x.get('executor_version')=='V2','V2_RECEIPT_REQUIRED');require(x.get('DEBUG_ONLY') is False and not x.get('DEBUG_INTERFACE_FIXTURE'),'DEBUG_COMMIT_FORBIDDEN');require(x.get('model_identity')==a['model_identity'] and x.get('benchmark_identity')==BH and x.get('chunk_size')==4096 and x.get('inference_config_sha256')==IH,'COMMIT_PARENT_PROTOCOL_MISMATCH');require(all(math.isfinite(float(x[k])) for k in (['score'] if phase=='fixed' else ['sample_best','dynamic_score'])),'NONFINITE_RESULT');records.append(x)
 if phase=='fixed':
  require(len(records)==52000 and a['sample_n']==6500,'INCOMPLETE_FIXED8_AUTHORITY');cells={(x['benchmark_sample_id'],x['action']) for x in records};ids={x['project_sample_id'] for x in benchmark()} if expected_ids is None else set(expected_ids);require(cells=={(sid,act) for sid in ids for act in ACTIONS},'FIXED_CELL_COMPLETENESS')
 else:
  fixed,_=final_authority(a['fixed_authority'],'fixed');fa=read(fixed['aggregate_path']);eligible={x['sample_id']:x for x in fa['per_sample'] if x['task']!='ruler_cwe_16k' and abs(x['sample_best']-1)>1e-12};require({x['sample_id'] for x in records}==set(eligible),'DYNAMIC_SEARCHED_SET');require(all(abs(x['sample_best']-eligible[x['sample_id']]['sample_best'])<=1e-12 and x['dynamic_score']>=x['sample_best']-1e-12 for x in records),'DYNAMIC_FIXED_PARENT_SCORE');require(a.get('population_n')==6000 and a['searched_n']+a['skipped_n']==6000,'DYNAMIC_POPULATION');require(len(records)==a['searched_n'],'DYNAMIC_COMPLETENESS');require(len({x['sample_id'] for x in records})==len(records),'DUPLICATE_DYNAMIC')
  for x in records:require(x['search_status']=='SUCCESS' and x['completed'] and len(x['action_sequence'])==x['chunk_count'] and set(x['action_sequence'])<=set(ACTIONS),'DYNAMIC_RESULT_SCHEMA')
 return a,records

def source_guard():
 p=R/'provenance/QWEN3_4B_DOWNSTREAM_SOURCE_MANIFEST_V2.json'
 require(sha(p)==p.with_suffix('.sha256').read_text().strip(),'V2_MANIFEST_HASH')
 m=read(p);require(m.get('executor_version')=='V2','V2_MANIFEST_REQUIRED')
 for name,h in m['files'].items():require(sha(under(name))==h,'DOWNSTREAM_SOURCE_HASH:'+name)
 return sha(p)


def ready_guard(phase='fixed'):
 require(phase in ['fixed','dynamic'],'V2_MECHANISM_NOT_ADAPTED')
 source=source_guard();p=R/'provenance/QWEN3_4B_EXECUTOR_V2_GPU_ACCEPTANCE.json'
 require(p.is_file() and p.with_suffix('.sha256').is_file(),'V2_GPU_ACCEPTANCE_REQUIRED')
 require(sha(p)==p.with_suffix('.sha256').read_text().strip(),'V2_GPU_ACCEPTANCE_HASH')
 a=read(p);require(a.get('status')=='PASS' and a.get('executor_version')=='V2' and a.get('model_family')=='Qwen3-4B' and a.get('checkpoint_class')=='STAGE2_FINAL' and a.get('source_manifest_sha256')==source,'V2_GPU_ACCEPTANCE_IDENTITY')
 expected=read(config_path('fixed'))['model_parent_authority_sha256']
 require(a.get('model_parent_authority_sha256')==expected,'V2_GPU_CHECKPOINT_IDENTITY')
 return sha(p)
