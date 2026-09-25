import sys,json,subprocess,ast
from core import *
import yaml
source_files=['core.py','pipeline.py','finalizer.py','data_adapter.py','entry.py','worker_stage1.py','worker_stage2.py','audit.py','prepare_workers.py','build.py','identity_preflight.py','data_preflight.py','freeze.py']
for name in source_files:ast.parse((W/name).read_text())
guard=R/'bin/llama31_8b_stage1_gpu0123_formal_control_v1.py'
guard.write_text('import sys\nsys.path.insert(0,'+repr(str(W))+')\nfrom pipeline import main\nsys.argv.insert(1,"launch" if "--execute" in sys.argv else "verify")\nmain()\n')
result=subprocess.run([sys.executable,'-B',str(W/'audit.py')],capture_output=True,text=True,check=True);audit=json.loads(result.stdout);atomic(W/'build_independent_audit.json',audit)
c=config();files={str(W/n):sha(W/n) for n in source_files};files[str(W/'build_independent_audit.json')]=sha(W/'build_independent_audit.json')
files[str(CONFIG)]=sha(CONFIG)
for stage in [1,2]:files[c[f'stage{stage}']['config']]=sha(c[f'stage{stage}']['config'])
for p in (R/'bin').glob('*llama31_8b*formal*v1*'):
 if p.is_file():files[str(p)]=sha(p)
for p in (R/'src/training_runtime/code').rglob('*.py'):files[str(p)]=sha(p)
authority={'status':'PASS','CHANGE_CLASS':'ACCOUNTING_CONTRACT_CORRECTION','SCIENTIFIC_DATA_SPLIT_CHANGED':False,'TRAINING_SCIENTIFIC_SEMANTICS_CHANGED':False,'STAGE2_CONTRACT_CHANGED':False,'stage1_train_contract':SPLIT,'stage2_contract':EXPECTED[2],'files':files,'bindings':c['bindings'],'build_audit_sha256':sha(W/'build_independent_audit.json'),'scope':'Build and launch only; stop Codex monitoring after same formal run reaches three healthy steps; no downstream auto-run.'}
atomic(SOURCE,authority);SOURCE.with_suffix('.sha256').write_text(sha(SOURCE)+'  '+SOURCE.name+'\n')
checks=preflight(c,live=True);atomic(W/'prelaunch_gate.json',{'status':'PASS','source_sha256':sha(SOURCE),'checks':checks,'SAFE_TO_START_LLAMA31_FORMAL_PIPELINE':'YES','training_started':False})
print(json.dumps({'status':'PASS','source_sha256':sha(SOURCE),'config_sha256':{str(s):sha(c[f'stage{s}']['config']) for s in [1,2]},'negative_tests':audit['negative_test_n'],'prelaunch':checks},indent=2))
