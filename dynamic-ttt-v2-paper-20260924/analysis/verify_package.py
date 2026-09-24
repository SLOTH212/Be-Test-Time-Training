"""Verify all frozen repository payload files using only the Python standard library."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 count=0
 for line in (ROOT/'SHA256SUMS').read_text().splitlines():
  expected,name=line.split('  ',1);p=(ROOT/name).resolve()
  assert p.is_relative_to(ROOT),name
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1<<20),b''):h.update(b)
  assert h.hexdigest()==expected,'CHECKSUM_MISMATCH: '+name
  count+=1
 print(json.dumps(dict(status='PASS',verified_files=count,scope='frozen payload; excludes SHA256SUMS itself, .git and generated outputs')))
if __name__=='__main__':main()
