#!/usr/bin/env python3
import json, os, shutil, subprocess, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lib.pipeline_core import *

def main():
    root=PIPELINE_ROOT; report={}
    if sys.argv[1:] not in ([],['--formal']): raise SystemExit('usage: preflight.py [--formal]')
    report['preflight_mode']='formal' if sys.argv[1:]==['--formal'] else 'standard'
    report['user_root']=os.geteuid()==0
    report['project_head']=subprocess.check_output(['git','-C',str(PROJECT_ROOT),'rev-parse','HEAD'],text=True).strip()
    report['project_head_valid']=report['project_head']==EXPECTED_HEAD
    report['project_status']=subprocess.check_output(['git','-C',str(PROJECT_ROOT),'status','--short'],text=True).splitlines()
    report['model_local']=(Path(os.path.expandvars('${TTT_MODEL_ROOT}/Qwen3-1.7B-Base/model.safetensors')).is_file())
    report['longtext']=validate_longtext(full_hash=True)
    report['qa10m']=validate_qa(full_hash=True)
    report['eval13k']=validate_benchmark(full_hash=True)
    report['disk_free_bytes']=shutil.disk_usage(os.path.expandvars('${TTT_USER_ROOT}')).free
    report['disk_required_bytes']=46_000_000_000
    report['disk_space_valid']=report['disk_free_bytes'] >= report['disk_required_bytes']
    gpu_query=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True,check=True)
    report['gpu_processes']=[x for x in gpu_query.stdout.splitlines() if x.strip()]
    report['gpu_idle']=not report['gpu_processes']
    import torch
    report['cuda_available']=torch.cuda.is_available()
    report['configs']={p.name:bool(load_yaml(p)) for p in (root/'configs').glob('*.yaml')}
    common=load_yaml(root/'configs/eval_common_13k.yaml'); off=load_yaml(root/'configs/eval_off_13k.yaml'); on=load_yaml(root/'configs/eval_on_13k.yaml')
    runtime=load_yaml(root/'configs/pipeline_runtime.yaml')['auto_shutdown']
    assert on['ttt_enabled'] is True and on['ttt_lr']==1.0 and abs(on['delta_clip_frobenius']-1e-5)<1e-12 and on['batch_size']==1
    assert off['ttt_enabled'] is False and off['online_update_path'] is False and off['batch_size']==2
    assert runtime['enabled'] is True and runtime['on_success'] is True and runtime['on_failure'] is False
    assert runtime['delay_seconds']==300 and runtime['dry_run'] is False
    report['inference_contract']='PASS'
    report['auto_shutdown']={'enabled':True,'on_success':True,'on_failure':False,'delay_seconds':300,'dry_run':False}
    if not report['gpu_idle']: raise RuntimeError('BLOCKED_GPU_BUSY')
    if not all((report['user_root'],report['project_head_valid'],report['model_local'],report['cuda_available'],report['disk_space_valid'])): raise RuntimeError('BLOCKED_PREFLIGHT')
    atomic_json(root/'reports/preflight_report.json',report)
    print(json.dumps(report,indent=2))
    print('PREFLIGHT_MODE='+report['preflight_mode']); print('PREFLIGHT_RESULT=passed')
if __name__=='__main__': main()
