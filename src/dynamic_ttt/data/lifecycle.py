"""Public data audit and packaging boundaries around preserved builders."""
import hashlib,json
from pathlib import Path

def require_audit(path):
 with Path(path).open(encoding='utf-8') as stream:report=json.load(stream)
 if report.get('status')!='PASS':raise ValueError('INDEPENDENT_DATA_AUDIT_FAILED')
 if report.get('data',{}).get('records',0)<=0:raise ValueError('EMPTY_DATASET_AUDIT')
 return report

def audit_llama(stage):
 from dynamic_ttt.data import llama_audit as auditor
 root=auditor.S1 if stage=='stage1' else auditor.S2
 if not any((root/'data').rglob('*.npz')):raise ValueError('NO_MATERIALIZED_DATA')
 auditor.main(stage)
 return require_audit(root/'audits'/('LLAMA_STAGE1_INDEPENDENT_AUDIT.json' if stage=='stage1' else 'LLAMA_STAGE2_INDEPENDENT_AUDIT.json'))

def finalize_llama():
 from dynamic_ttt.data import derive_llama as builder
 for root,stage in [(builder.S1,'STAGE1'),(builder.S2,'STAGE2')]:
  require_audit(root/'audits'/f'LLAMA_{stage}_INDEPENDENT_AUDIT.json')
 # Record this installed release's source identity, not a missing server manifest.
 modules=['derive_llama.py','llama_audit.py','lifecycle.py']
 source={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in modules}
 builder.SRC=builder.WORK/'release_source_authority'
 builder.atomic_json(builder.SRC/'SOURCE_MANIFEST.json',{'scope':'PUBLIC_RELEASE_BUILDER','files':source})
 builder.finalize()
