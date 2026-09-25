import json,hashlib,time
from pathlib import Path
R=Path('/path/to/ttt');A=R/'audits/qwen3_4b_stage1_stage2_pipeline_formal_v1';p=R/'configs/qwen3_4b_stage1_stage2_pipeline_formal_v1.yaml';c=json.loads(p.read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (Path(c['state_root'])/'EXECUTION_AUTHORIZED.json').exists()
(A/'PIPELINE_CONFIG_BUILD_CANDIDATE_1.json').write_bytes(p.read_bytes())
for stage in (1,2):
 target=c[f'stage{stage}']['guard' if stage==1 else 'controller'];status=R/f'bin/status_qwen3_4b_stage{stage}_gpu45_formal_pipeline_v1.sh'
 status.write_text('#!/usr/bin/env bash\nset -Eeuo pipefail\nsource /path/to/ttt/bin/activate_ttt.sh\nexport CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1\nexec "'+c['python']+'" -B '+target+' status "$@"\n');status.chmod(0o755)
 c[f'stage{stage}']['historical_status_script']=c[f'stage{stage}']['status_script'];c[f'stage{stage}']['status_script']=str(status);c['bindings'][str(status)]=sha(status)
p.write_text(json.dumps(c,indent=2,sort_keys=True)+'\n')
source=R/'provenance/QWEN3_4B_STAGE1_STAGE2_PIPELINE_SOURCE_AUTHORITY_V1.json';source.rename(A/'SOURCE_AUTHORITY_BUILD_CANDIDATE_1.json');source.with_suffix('.sha256').rename(A/'SOURCE_AUTHORITY_BUILD_CANDIDATE_1.sha256')
for name in ['DRY_RUN.log','INDEPENDENT_BUILD_AUDIT.json','INDEPENDENT_BUILD_AUDIT.log']:
 path=A/name
 if path.exists():path.rename(A/(name+'.build_candidate_1'))
print('Preserved first build candidate; final CPU status refinements ready for re-freeze.')
