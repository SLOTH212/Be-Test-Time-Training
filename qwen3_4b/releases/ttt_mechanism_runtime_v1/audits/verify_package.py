#!/usr/bin/env python3
"""Validate exact release inventory, source authorities and frozen dependency."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import runtime as rt

def main():
 manifest=json.loads((ROOT/'provenance/RELEASE_MANIFEST.json').read_text());files=manifest['files'];actual={str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and p!=ROOT/'provenance/RELEASE_MANIFEST.json'}
 if actual!=set(files):raise ValueError('RELEASE_INVENTORY_MISMATCH')
 for p,h in files.items():
  path=ROOT/p
  if path.is_symlink() or not path.resolve().is_relative_to(ROOT) or rt.sha(path)!=h:raise ValueError('RELEASE_HASH_MISMATCH '+p)
 tree=hashlib.sha256(''.join(p+'\0'+h+'\n' for p,h in sorted(files.items())).encode()).hexdigest()
 if tree!=manifest['release_tree_sha256']:raise ValueError('RELEASE_TREE_HASH')
 a=json.loads((ROOT/'provenance/MECHANISM_SOURCE_AUTHORITY.json').read_text())
 for x in a['entries']:
  if rt.sha(ROOT/x['packaged_path'])!=x['sha256']:raise ValueError('FORMAL_SOURCE_HASH')
 for p in (ROOT/'configs').glob('*.yaml'):rt.load_config(p)
 rt.dependency();print(json.dumps({'SOURCE_HASH_VALIDATION':'PASS','INFERENCE_RUNTIME_DEPENDENCY_CHECK':'PASS','CONFIG_PARSING':'PASS','RELEASE_TREE_SHA256':tree,'files':len(files)}))
if __name__=='__main__':main()
