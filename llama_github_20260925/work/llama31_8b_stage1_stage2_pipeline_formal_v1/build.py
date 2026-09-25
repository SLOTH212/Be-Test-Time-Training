import json,sys,os,hashlib,subprocess,shutil
from pathlib import Path
from core import *
from data_adapter import records
import yaml
W.mkdir(parents=True,exist_ok=True)
data={};bindings=read(W/'identity_cpu_preflight.json')['files'];bindings[str(R/'provenance/LLAMA31_8B_TRAIN_INFERENCE_SMOKE_V1.json')]='84dab6b19a522e6f7c38e3732210c4d861bb0e231ed204482089b46490876557'
bindings[str(R/'packages/llama31_ntp_port_v1/llama31_ntp_port_v1.tar.gz')]='3f1ca6c6cee8f8601fd5bfca396a067fe1876cbba3fbabecb273e75e19fd54bc'
bindings[str(R/'provenance/LLAMA31_DERIVED_DATA_AUTHORITY_V1.json')]=sha(R/'provenance/LLAMA31_DERIVED_DATA_AUTHORITY_V1.json')
for stage,name in [(1,'llama31_stage1_500m_32k_v1'),(2,'llama31_stage2_15m_32k_v1')]:
 root=R/'datasets'/name;authority=root/'authority/DATASET_AUTHORITY.json';bindings[str(authority)]=sha(authority)
 for p,h in read(authority)['data_file_hashes'].items():bindings[str(root/p)]=h
 package=R/'packages'/(name+'.tar.gz');bindings[str(package)]=['c7189fb46adad303b97c7a3cee36059bb201e9b7b5fa2cbf3772a4535c841680','6bb877e4ece2819fb063581ff276c305574822a3c55a1dcfa31ed8cc79030227'][stage-1]
 descriptor=W/f'stage{stage}_train_descriptor.json';atomic(descriptor,{'dataset_root':str(root),'splits':['train'],'authority_sha256':sha(authority)})
 index=[{'id':r['id'],'tokens':r['n_tokens'],'split':r['split']} for r in records(root,['train'])];require(len(index)==EXPECTED[stage]['record_cursor'] and sum(r['tokens'] for r in index)==EXPECTED[stage]['cumulative_tokens'],'TRAIN_INDEX_COUNTS');atomic(W/f'stage{stage}_train_index.json',index)
 if stage==1:
  val=[{'id':r['id'],'tokens':r['n_tokens'],'split':r['split']} for r in records(root,['validation'])];atomic(W/'stage1_validation_index.json',val);require(not ({r['id'] for r in index}&{r['id'] for r in val}),'SPLIT_OVERLAP')
 cfg=yaml.safe_load((R/'configs/llama31_8b_stage1_32k_gpu0123_debug_v1.yaml').read_text())
 for k in ['normal_optimizer_step_cap','warmup_optimizer_steps','measured_optimizer_steps','resumed_optimizer_steps','synthetic_data_only','synthetic_data','SYNTHETIC_DATA']:cfg.pop(k,None)
 cfg.update(stage=f'stage{stage}',train_data=str(descriptor),model_base=str(R/'models/Llama-3.1-8B'),record_cursor_final=EXPECTED[stage]['record_cursor'],train_tokens=EXPECTED[stage]['cumulative_tokens'],expected_optimizer_steps=EXPECTED[stage]['update_step'],DEBUG_ONLY=False,PAPER_RESULT=False,FORMAL=True,formal_training=True,config_class='LLAMA31_8B_FORMAL_V1',dataset_file_bindings={str(authority):sha(authority),str(descriptor):sha(descriptor)},save_steps=([25]+list(range(250,4254,250))+[4254]) if stage==1 else [25,75,115],debug_validation_records=0)
 cfg['input_identity']={'worker_sha256':sha(W/f'worker_stage{stage}.py'),'dataset_authority_sha256':sha(authority),'data_manifest_sha256':sha(descriptor),'model_identity_sha256':sha(R/'work/llama31_8b_compat_smoke_v1/model_identity.json'),'cross_data_authority_sha256':sha(R/'provenance/LLAMA31_DERIVED_DATA_AUTHORITY_V1.json'),'llama_port_sha256':bindings[str(R/'packages/llama31_ntp_port_v1/llama31_ntp_port_v1.tar.gz')]}
 if stage==1:cfg.update(SPLIT,validation_data=str(root/'data/validation'),validation_records=185,validation_input_tokens=4850937)
 else:cfg.update(answer_mask_sidecar=str(descriptor),context_group_weight=.1,qa_targets=19397,supervised_answer_tokens=97619,parent_authority_required=str(R/'models/llama31_8b_stage1_500m_32k_final_v1/STAGE1_FINAL_AUTHORITY.json'))
 p=R/'configs'/f'llama31_8b_stage{stage}_{"500m" if stage==1 else "15m"}_32k_gpu0123_formal_v1.yaml';p.write_text(yaml.safe_dump(cfg,sort_keys=False));bindings[str(descriptor)]=sha(descriptor);bindings[str(W/f'stage{stage}_train_index.json')]=sha(W/f'stage{stage}_train_index.json')
 data[f'stage{stage}']={'config':str(p),'data_authority':str(authority),'record_index':str(W/f'stage{stage}_train_index.json'),'run_root':str(R/'runs'/f'llama31_8b_stage{stage}_{"500m" if stage==1 else "15m"}_32k_gpu0123_formal_v1'),'final_root':str(R/'models'/f'llama31_8b_stage{stage}_{"500m" if stage==1 else "15m"}_32k_final_v1'),'expected':EXPECTED[stage]}
c={**data,'base_model':str(R/'models/Llama-3.1-8B'),'model_revision':'d04e592bb4f6aa9cfee91e2e20afa771667e1d4b','port_sha256':'3f1ca6c6cee8f8601fd5bfca396a067fe1876cbba3fbabecb273e75e19fd54bc','bindings':bindings,'physical_gpu_allowlist':[0,1,2,3],'world_size':4,'state_root':str(R/'runs/llama31_8b_stage1_stage2_pipeline_formal_v1'),'downstream_auto_start':False,'final_long_term_model_artifacts':2,'storage_required_bytes':384*(1<<30),'state_inventory':str(R/'work/llama31_8b_compat_smoke_v1/state_contract.json'),'tokenizer_files':['tokenizer.json','tokenizer_config.json','special_tokens_map.json'],'known_memory_warning':'TOO_TIGHT','4GPU_CAPACITY_WARNING_ACCEPTED':True,'CHANGE_CLASS':'ACCOUNTING_CONTRACT_CORRECTION','SCIENTIFIC_DATA_SPLIT_CHANGED':False,'TRAINING_SCIENTIFIC_SEMANTICS_CHANGED':False}
atomic(CONFIG,c)
shells={'launch_llama31_8b_stage1_stage2_pipeline_formal_v1.sh':'launch','status_llama31_8b_stage1_stage2_pipeline_formal_v1.sh':'status','recover_llama31_8b_stage1_stage2_pipeline_formal_v1.sh':'recover','finalize_llama31_8b_stage1_to_stage2_formal_v1.sh':'finalize --stage 1','finalize_llama31_8b_stage2_to_inference_formal_v1.sh':'finalize --stage 2','launch_llama31_8b_stage2_15m_32k_gpu0123_formal_v1.sh':'trainer --stage 2 --execute','prune_llama31_8b_intermediate_checkpoints_after_final_v1.sh':'prune'}
for name,mode in shells.items():
 p=R/'bin'/name;p.write_text('#!/usr/bin/env bash\nset -euo pipefail\nsource /path/to/ttt/bin/activate_ttt.sh\nexport CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 WORLD_SIZE=4\nexec python -B '+str(W/'pipeline.py')+' '+mode+' "$@"\n');p.chmod(0o755)
for name,mode in [('llama31_8b_stage1_stage2_pipeline_formal_v1.py',None),('llama31_8b_stage1_gpu0123_formal_control_v1.py','verify')]:
 p=R/'bin'/name;p.write_text('import sys\nsys.path.insert(0,'+repr(str(W))+')\n'+('sys.argv.insert(1,'+repr(mode)+')\n' if mode else '')+'from pipeline import main\nmain()\n')
print('CONFIGS_AND_WRAPPERS_BUILT_NOT_LAUNCHED')
