import json
from pathlib import Path
from dynamic_ttt.eval.runtime import load_config
from dynamic_ttt.mechanisms.runtime import load_config as mechanism_config

def test_windows_environment_paths(monkeypatch):
 root=Path(__file__).resolve().parents[1]
 monkeypatch.setenv('MODEL_ROOT',r'C:\local models')
 cfg=load_config(root/'configs/inference/qwen3_1p7b.json')
 assert cfg['model_path']==r'C:\local models/qwen3_1p7b_stage2'
 cfg=mechanism_config(root/'configs/mechanisms/reverse.json')
 assert cfg['model_path']==r'C:\local models/qwen3_1p7b_stage2'
