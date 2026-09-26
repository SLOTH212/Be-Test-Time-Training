"""Verify and restore frozen records, then CPU-only recompute in a new directory."""
import argparse, gzip, hashlib, importlib.util, json, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
NAMES={'A':'matched_nonzero_control_v1_20260926','B':'target_alignment_control_v1_20260926'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--verify-only',action='store_true');a=p.parse_args()
 dest=a.output.resolve()
 if not a.verify_only and dest.exists():raise SystemExit('Output must be a NEW directory; existing outputs are never overwritten.')
 rows=json.loads((HERE/'FILE_MANIFEST.json').read_text())
 for r in rows:
  data=(HERE/r['stored_path']).read_bytes()
  if r['gzip']:data=gzip.decompress(data)
  assert len(data)==r['size'] and hashlib.sha256(data).hexdigest()==r['sha256'],r['stored_path']
  if not a.verify_only:
   path=dest/r['restored_path'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
 if a.verify_only:print('PASS: all original bytes verified');return
 results={}
 for label,name in NAMES.items():
  root=dest/'runs/formal'/name
  script=root/('summarize_closest.py' if label=='A' else 'summarize.py')
  spec=importlib.util.spec_from_file_location('recompute_'+label,script);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  # Only redirect historical file reads; preserve original config/source bytes and hash checks.
  field='read' if label=='A' else 'load';original=getattr(m,field)
  def relocated(path,_read=original):
   s=str(path);prefix='/home/zonghan/ttt/'
   return _read(dest/s[len(prefix):] if s.startswith(prefix) else path)
  setattr(m,field,relocated)
  expected=json.loads((root/'bootstrap_summary.json').read_text())
  if label=='A':m.summarize(root)
  else:
   sys.argv=[str(script),'--root',str(root)];m.main()
  actual=json.loads((root/'bootstrap_summary.json').read_text())
  assert actual==expected,label+' summary mismatch'
  results[label]={'status':'PASS','exact_bootstrap_summary_match':True}
 (dest/'CPU_RECOMPUTE_CHECK.json').write_text(json.dumps(results,indent=2)+'\n')
 print(json.dumps(results,indent=2))
if __name__=='__main__':main()
