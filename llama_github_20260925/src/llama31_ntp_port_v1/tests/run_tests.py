"""Fail-fast test levels. OFF failure stops before any NTP tests."""
import os, subprocess, sys
from pathlib import Path
W=Path(__file__).resolve().parents[1]
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
for script in ['test_level0.py','test_off_parity.py','test_port.py']:
    command=[sys.executable,str(W/'tests'/script)]
    if script=='test_level0.py' and '--extracted' in sys.argv:command.append('--extracted')
    subprocess.run(command,env=env,cwd=W,check=True)
print('LLAMA_PORT_LEVEL0=PASS\nLLAMA_PORT_LEVEL1=PASS\nLLAMA_PORT_LEVEL2=PASS')
