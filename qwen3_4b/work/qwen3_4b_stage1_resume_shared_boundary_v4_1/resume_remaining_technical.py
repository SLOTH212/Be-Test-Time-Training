from pathlib import Path
import hashlib
original=Path('/path/to/ttt/work/qwen3_4b_stage1_resume_shared_boundary_v4_1/run_branches.py')
assert hashlib.sha256(original.read_bytes()).hexdigest()=='1855adb6c5c0d7d7844ee307428123dd22741f1706e940a80316266800a96d04'
s=original.read_text();old="branches=['A','B1','B2','B3']";assert s.count(old)==1
s=s.replace(old,"branches=['B1','B2','B3']")
exec(compile(s,str(original)+'<technical_resume>', 'exec'),globals())
