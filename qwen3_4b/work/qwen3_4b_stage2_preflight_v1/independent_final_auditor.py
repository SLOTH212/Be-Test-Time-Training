import ast,hashlib,json,os,subprocess,tarfile
from pathlib import Path
R=Path('/path/to/ttt');W=R/'work/qwen3_4b_stage2_preflight_v1';A=R/'audits/qwen3_4b_stage2_30m_32k_gpu23_preflight_v1'
checks={};bound={}
def sha(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):
 p=Path(p);bound[str(p)]=sha(p);return bound[str(p)]
def j(p):
 p=Path(p);bind(p);return json.loads(p.read_text())
def require(name,cond):
 checks[name]=bool(cond)
 if not cond:raise RuntimeError('INDEPENDENT_AUDIT_FAILED '+name)
def receipt(name):
 v=j(A/name);require(name,str(v.get('status','')).startswith('PASS'));return v
packages=[
 ('scaleup_multigpu_32k_hit_v2/scaleup_multigpu_32k_hit_v2_package.tar.gz','79d89c95908ea787c1358a651c9003ca14eced1957853154b43041db845c162b','scaleup_multigpu_32k_hit_v2','training_runtime'),
 ('ttt_inference_runtime_v1/ttt_inference_runtime_v1.tar.gz','65a0d7bd9c8801c92fc30477ab8662d3eb52261612eff2acef0bbeb3edf6ba51','ttt_inference_runtime_v1','inference_runtime'),
 ('qa_replay_30m_32k_qwen3_stage2_v2/qa_replay_30m_32k_qwen3_stage2_v2.tar.gz','60a54b694982a4a1c21708ba7c177b25101a5bc76a771f542f90e745611ce230',None,None)]
for rel,h,prefix,deployed in packages:
 p=R/'shared/packages'/rel;require('frozen_package_'+rel,bind(p)==h)
 if prefix:
  with tarfile.open(p) as t:
   for member in t.getmembers():
    if member.isfile() and member.name.endswith(('modeling_qwen3.py','ttt_state_core.py','formal_train_gated_ntp.py')):
     target=R/'src'/deployed/member.name[len(prefix)+1:]
     require('frozen_core_'+str(target),hashlib.sha256(t.extractfile(member).read()).hexdigest()==bind(target))
lineage=j(R/'audits/STAGE2_LOSS_LINEAGE_AUDIT.json')
require('corrected_lineage',lineage['CORRECTED_STAGE2_LOSS_LINEAGE']=='PASS' and lineage['completed_run_counters_verified'])
for p,h in lineage['source_sha256'].items():require('historical_source_'+p,bind(W/'recovered_0p6b'/p)==h)
require('historical_completion',bind(W/'recovered_0p6b/stage2/state.json')==lineage['completed_run_state_sha256'])
original=R/'src/training_runtime/code/workers/distributed_train_worker.py';candidate=R/'src/training_runtime/code/workers/distributed_stage2_train_worker_v1.py'
require('stage1_worker_untouched',bind(original)=='b93593c2a176b0ba8ebc72e5422d20f562c905efefbe7286ef070b6fa52ec47c')
require('checkpoint_base_untouched',bind(R/'src/training_runtime/code/lib/distributed_checkpoint.py')=='48fd2a39d06619422d20fda97e79a538725e334e1d405a24f5033759d6ffc608')
trees=[ast.parse(p.read_text()) for p in (original,candidate)]
for name in ['masks_for','fused_sum','answer_ranges','count_window']:
 nodes=[next(x for x in ast.walk(t) if isinstance(x,ast.FunctionDef) and x.name==name) for t in trees]
 require('unchanged_loss_function_'+name,ast.dump(nodes[0],include_attributes=False)==ast.dump(nodes[1],include_attributes=False))
def loss_terms(tree):
 return [ast.dump(n,include_attributes=False) for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='term' for t in n.targets) and any(isinstance(x,ast.Name) and x.id in ('global_answer','global_context') for x in ast.walk(n)) and any(isinstance(x,ast.Name) and x.id=='fused_sum' for x in ast.walk(n))]
require('unchanged_loss_scaling_expressions',loss_terms(trees[0])==loss_terms(trees[1]) and len(loss_terms(trees[1]))==2)
cfgpath=R/'configs/qwen3_4b_stage2_30m_32k_gpu23_formal_v1.yaml';cfg=j(cfgpath)
require('formal_config_sidecar',bind(cfgpath)==cfgpath.with_suffix('.sha256').read_text().split()[0])
for p,h in cfg['data_file_sha256'].items():require('raw_dataset_'+p,bind(p)==h)
expected={'authority/STAGE2_DATASET_AUTHORITY.json':'e423c322602c5124acc839470079cfe23559b43c0a29eb9042cf1a3a4ee6ddd0','manifests/FINAL_30M_QA_CONTENT_ORDER.jsonl':'8ecf2aa5a6ba1fe8c6a6654c588d6d8810dcd9c38d8f12989d4ee44fc319ca43','manifests/FINAL_PACKED_RECORDS.jsonl':'4b9a4610effaba35a616c7622c248ee79db18bbd50f1a82a0e5c0d0326c13898','authority/stage2_qa_replay_30m_32k_qwen3_v2.yaml':'3aec55e62ee30dece2d79094989f8638f5fd754730f9a0aa330eb0ba4109e383'}
for p,h in expected.items():require('frozen_data_'+p,bind(R/'shared/datasets'/p)==h)
science=j(R/'provenance/STAGE2_SCIENTIFIC_CONTRACT.json')
require('scientific_config',all(cfg[k]==v for k,v in science['values'].items()))
require('formal_dimensions',cfg['context_length']==32768 and cfg['ttt_chunk_size']==4096 and cfg['ttt_layers']==[0,6,12,18,24,30] and cfg['world_size']==2 and cfg['micro_batch_size']==1 and cfg['gradient_accumulation']==1 and cfg['global_batch_size']==2 and cfg['record_cursor_final']==925 and cfg['train_tokens']==30000000)
for key,p in [('worker_sha256',candidate),('stage2_checkpoint_adapter_sha256',R/'src/training_runtime/code/lib/stage2_checkpoint_runtime.py'),('stage2_entry_sha256',R/'bin/stage2_runtime_entry_v1.py')]:require('bound_'+key,bind(p)==cfg['input_identity'][key])
d=receipt('REAL_STAGE2_DATASET_CPU_AUDIT.json')
require('all925_mask_counts',d['records']==925 and d['nonpadding_tokens']==30000000 and d['answer_predictions']==388806 and d['context_predictions']==29610269 and all(d[k]==0 for k in ['malformed','overlength','range_oob','mask_mismatch']))
for n in ['INDEPENDENT_RENDERED_ANSWER_MASK_AUDIT.json','STAGE2_CPU_SHIFT_LOSS_FIXTURE.json','STAGE2_TINY_GPU_LOSS_AUDIT.json','STAGE2_ZERO_MASK_RANK_AUDIT.json','STAGE2_SUPERVISION_DENSITY_AUDIT.json','STAGE2_FINAL_RECORD_EDGE_AUDIT.json','STAGE2_HF_STATE_DICT_AUDIT.json','FORMAL_INTERFACE_CPU_TESTS.json','FORMAL_NEGATIVE_CONTRACT_TESTS.json','FALSE_FINAL_AUTHORITY_REJECTION.json','STAGE2_LAZY_RESTORE_CPU_TEST.json','STAGE2_EXPORTED_TOKENIZER_IDENTITY.json']:receipt(n)
rt=receipt('STAGE2_DCP_EXACT_STATE_ROUNDTRIP.json')
require('exact_model_optimizer_rng_roundtrip',rt['ranks'][0]['model_exact'] and rt['ranks'][0]['optimizer_exact'] and all(v['rank_payload_scheduler_cursor_rng_exact'] for v in rt['ranks']))
res=j(A/'STAGE2_RESUME_COMPARISON.json');require('resume_state_and_order',res['STAGE2_RESUME_STATE_INTEGRITY']=='PASS' and res['canonical_record_sequence_exact'])
parent=j(A/'DEBUG_STAGE1_PARENT_AUTHORITY.json');pm=j(Path(parent['checkpoint_path'])/'manifest.json')
require('debug_parent_identity',bind(Path(parent['checkpoint_path'])/'model.pt')==parent['checkpoint_sha256'] and bind(Path(parent['checkpoint_path'])/'manifest.json')==parent['manifest_sha256'] and pm['metadata']['config_sha256']==parent['config_sha256'] and parent['PARENT_CLASS']=='DEBUG_ONLY')
for name,step,cursor,tokens in [('uninterrupted_A',13,26,849150),('resume_B',5,10,327667),('final_record_edge',1,925,24870)]:
 root=R/'runs/qwen3_4b_stage2_preflight_v1'/name;done=j(root/'complete.json');require(name+'_complete',done['complete'])
 log=root/'training.jsonl';bind(log);rows=list(map(json.loads,log.read_text().splitlines()));last=rows[-1]
 require(name+'_counters',last['update_step']==step and last['record_cursor']==cursor and last['cumulative_tokens']==tokens)
 for row in rows:
  for rank in row['rank_runtime']:
   require(name+'_step'+str(row['update_step'])+'_rank'+str(rank['rank']),rank['physical_gpu'] in [2,3] and rank['gradient_coverage']['present']==410 and not rank['gradient_coverage']['nonfinite'] and rank['weights_finite'])
 for p in (root/'checkpoints').glob('slot_*/manifest.json'):j(p)
fwd=j(R/'runs/qwen3_4b_stage2_preflight_v1/forward_only/forward_only.json');require('real_forward',fwd['status']=='PASS')
inf=receipt('STAGE2_INFERENCE_HANDOFF_AUDIT.json')
require('OFF_and_L0_actual_generation',{v['mode'] for v in inf['results']}=={'OFF','L0'} and all(v['status']=='PASS' and v['generated_tokens']==1 for v in inf['results']))
require('TTT_update_exercised',next(v for v in inf['results'] if v['mode']=='L0')['ttt_stats_count']>0 and inf['inference_chunk']==4096)
amend=j(R/'configs/qwen3_4b_stage2_inference_4k_v1.json')
require('user_authorized_4k_contract',amend['ttt_chunk_size']==4096 and amend['user_instruction']=='推理chunk改为4k' and not amend['training_loss_modified'] and not amend['frozen_runtime_modified'])
core_root=Path(amend['runtime_code_root'])
for n,h in amend['runtime_file_sha256'].items():
 require('inference_4k_file_'+n,bind(core_root/n)==h)
 if n!='hf_models/hf_qwen3/ttt_state_core.py':require('inference_other_source_unchanged_'+n,sha(R/'src/inference_runtime/code'/n)==h)
old=(R/'src/inference_runtime/code/hf_models/hf_qwen3/ttt_state_core.py').read_text()
expected_core=old.replace('chunk_size=1024','chunk_size=4096').replace('chunk_size != 1024','chunk_size != 4096').replace('frozen Static-TTT chunk_size must be 1024','authorized Qwen3-4B Static-TTT chunk_size must be 4096')
require('only_authorized_chunk_source_change',(core_root/'hf_models/hf_qwen3/ttt_state_core.py').read_text()==expected_core)
receipt('INFERENCE_4K_CORE_CPU_AUDIT.json');bind(W/'inference_4k_core_cpu_audit.py');bind(R/'configs/qwen3_4b_stage2_parent_authority_schema_v1.json');bind(A/'INFERENCE_4K_USER_AMENDMENT.diff')
require('actual_4k_gpu_pairs',next(v for v in inf['results'] if v['mode']=='L0')['ttt_valid_pair_counts']==[4095])
hf=Path(inf['checkpoint']);ha=j(hf/'INFERENCE_PARENT_AUTHORITY.json')
for n,h in ha['files'].items():require('hf_file_'+n,bind(hf/n)==h)
oldha=j(W/'debug_stage2_hf/INFERENCE_PARENT_AUTHORITY.json')
require('reexport_weights_identical',ha['files']['model.safetensors']==oldha['files']['model.safetensors'])
require('hf_immutable',not (hf.stat().st_mode & 0o222) and all(not(p.stat().st_mode & 0o222) for p in hf.iterdir() if p.is_file()))
for n in ['launch_qwen3_4b_stage2_30m_32k_gpu23_formal_v1.sh','status_qwen3_4b_stage2_30m_32k_gpu23_formal_v1.sh','finalize_qwen3_4b_stage2_to_inference_formal_v1.sh']:
 p=R/'bin'/n;bind(p);require('shell_syntax_'+n,subprocess.run(['bash','-n',str(p)],capture_output=True).returncode==0)
for n in ['qwen3_4b_stage2_formal_control_v1.py','qwen3_4b_stage2_export_v1.py']:
 p=R/'bin'/n;bind(p);compile(p.read_text(),str(p),'exec')
require('no_formal_run',not (R/'runs/qwen3_4b_stage2_formal_current.json').exists() and not list((R/'runs').glob('qwen3_4b_stage2_formal_*')))
for root in [W,A,R/'runs/qwen3_4b_stage2_preflight_v1']:
 require('output_owner_'+str(root),root.resolve().is_relative_to(R.resolve()) and all(p.stat().st_uid==os.getuid() and p.resolve().is_relative_to(R.resolve()) for p in root.rglob('*')))
require('shared_model_guard',subprocess.run(['bash',str(R/'bin/check_shared_assets.sh')],capture_output=True).returncode==0)
for p in [R/'provenance/STAGE2_LOSS_SEMANTICS_AUTHORITY.json',R/'audits/STAGE2_STOPPING_SEMANTICS.json',A/'STAGE2_THROUGHPUT_AUDIT.json',A/'STAGE2_INFERENCE_EXPORT_CONFIG_REPAIR.json',W/'inference_handoff_smoke.py',W/'resume_roundtrip_wrapper.py',W/'compare_stage2_resume.py',W/'zero_mask_rank_audit.py',W/'tiny_gpu_loss_audit.py',Path(__file__)]:bind(p)
print(json.dumps({'STAGE2_PRELAUNCH_AUDIT':'PASS','status':'PASS','auditor_mode':'Independent read-only source/receipt/hash checks; stdout only. No training mutation.','checks':checks,'file_sha256':bound,'scope_warning':'Zero-mask group uses frozen fail-closed rejection; future trajectory numerical differences reported, not deterministic equivalence. DEBUG parent only; no full30M run.'},indent=2))
