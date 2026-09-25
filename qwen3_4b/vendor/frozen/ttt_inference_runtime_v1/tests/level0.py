#!/usr/bin/env python3
"""Release-tree-only import/help/config/dry-run test; performs no inference."""
import hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from code.runtime import load_config,sha256

EXPECTED={
 "code/inference_model/hf_qwen3/modeling_qwen3.py":"d34faaaf0738f0bd87a188449ee99aff24b8a3d4e5d9adb320a250c00d558c43",
 "code/hf_models/hf_qwen3/ttt_state_core.py":"0328e9fe792c918d6d741f58ade9729161a13bac053b16997f96d7b67064158e",
 "dynamic/formal_dynamic_runner.py":"293886005cb8f78c943f0affe3e3b383d6de238f4bc825a5aea28a6e6dc66f6e",
 "scorers/common.py":"cee3f5cb3cad1709f81abe3dc32b364ca97322e7ebd91747a8b119aecb0ce556"}

def main():
    hashes={name:sha256(ROOT/name) for name in EXPECTED};assert hashes==EXPECTED
    for path in (ROOT/'configs').glob('*.yaml'):load_config(path)
    sys.path.insert(0,str(ROOT/'code'))
    import inference_model.hf_qwen3.configuration_qwen3
    import inference_model.hf_qwen3.modeling_qwen3
    for rel in ['fixed/run_fixed.py','fixed/aggregate_fixed.py','dynamic/run_dynamic.py','dynamic/aggregate_dynamic.py','audits/integrity_audit.py','tests/replay_historical.py']:
        result=subprocess.run([sys.executable,str(ROOT/rel),'--help'],capture_output=True,text=True)
        assert result.returncode==0,(rel,result.stdout,result.stderr)
    for config in (ROOT/'configs').glob('*.yaml'):
        for rel in ['fixed/run_fixed.py','dynamic/run_dynamic.py','run_pipeline.py']:
            result=subprocess.run([sys.executable,str(ROOT/rel),'--config',str(config),'--dry-run'],capture_output=True,text=True)
            assert result.returncode==0,(rel,config,result.stdout,result.stderr)
            assert json.loads(result.stdout)['status']=='DRY_RUN_PASS'
    print(json.dumps({"status":"PASS","INFERENCE_RUNTIME_LEVEL0":"PASS","source_hashes":hashes,"model_weights_loaded":False,"gpu_used":False},sort_keys=True))
if __name__=='__main__':main()
