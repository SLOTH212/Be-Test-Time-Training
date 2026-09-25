"""Read-only adapter for frozen Llama NPZ IDs and JSONL metadata."""
import json,hashlib
from pathlib import Path
import numpy as np
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def records(root,splits=('train',)):
 root=Path(root);authority=json.loads((root/'authority/DATASET_AUTHORITY.json').read_text());files=authority['data_file_hashes']
 for split in splits:
  for relative in sorted(p for p in files if p.startswith('data/'+split+'/') and p.endswith('.jsonl')):
   path=root/relative;npz=path.with_suffix('.npz')
   if sha(path)!=files[relative] or sha(npz)!=files[str(npz.relative_to(root))]:raise ValueError('DATA_SHARD_HASH')
   metadata=[json.loads(line) for line in path.read_text().splitlines()]
   with np.load(npz,allow_pickle=False) as arrays:
    offsets=arrays['offsets'];ids=arrays['input_ids']
    if len(offsets)!=len(metadata)+1 or int(offsets[-1])!=len(ids):raise ValueError('OFFSETS')
    for i,row in enumerate(metadata):
     tokens=ids[int(offsets[i]):int(offsets[i+1])];n=row['length']
     if not 0<n<=32768 or len(tokens)<n:raise ValueError('LENGTH')
     real=tokens[:n].astype(np.int64).tolist();sid=relative+':'+str(i)
     if any(t<0 or t>=128256 for t in real):raise ValueError('TOKEN_ID')
     bounds=[(b['start'],b['end']) for b in row.get('document_boundaries',[])]
     answer=[r for u in row.get('units',[]) for r in u['answer_ranges']]
     yield {'sample_id':sid,'id':sid,'input_ids':real,'token_count':n,'n_tokens':n,'document_boundaries_qwen_json':bounds,'answer_ranges':answer,'qa_count':row.get('qa_count',0),'padding':row.get('padding',0),'split':split,'metadata':row}
def windows(path,start,stop,width):
 spec=json.loads(Path(path).read_text());window=[]
 for pos,row in enumerate(records(spec['dataset_root'],spec['splits'])):
  if pos>=stop:break
  if pos>=start:
   window.append((pos,row))
   if len(window)==width:yield window;window=[]
 if window:yield window
def answer_ranges(path):
 spec=json.loads(Path(path).read_text());return {r['id']:r['answer_ranges'] for r in records(spec['dataset_root'],spec['splits'])}
