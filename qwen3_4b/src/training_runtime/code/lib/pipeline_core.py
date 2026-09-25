import hashlib, json, math, os, random, shutil, tempfile, time
from pathlib import Path

PIPELINE_ROOT = Path(os.path.expandvars('${TTT_TRAINING_ROOT}'))
PROJECT_ROOT = Path(os.path.expandvars('${TTT_PROJECT_ROOT}'))
EXPECTED_HEAD = '190580a29780030969cd337763098188bce8c88c'
BENCHMARK_ROOT = Path(os.path.expandvars('${TTT_BENCHMARK_ROOT}/ruler_16k_13task_1000sample_v1'))
BENCHMARK_SHA = 'f921d4bbf2641b1069c925e534beee07e1a3749c4c5f221328ffe4656ab3b977'
LONGTEXT_ROOT = Path(os.path.expandvars('${TTT_DATASET_ROOT}/continual_16k_300m_prolong_v3_full'))
QA_ROOT = Path(os.path.expandvars('${TTT_DATASET_ROOT}/qa_replay_10m_v1'))
QA_MASK_ROOT = Path(os.path.expandvars('${TTT_DATASET_ROOT}/qa_replay_10m_v1_answer_context_masks_v1'))

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def fsync_dir(path):
    fd = os.open(str(path), os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try: os.fsync(fd)
    finally: os.close(fd)

def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(path) + '.partial')
    with open(partial, 'w') as f:
        json.dump(obj, f, indent=2, sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(partial, path); fsync_dir(path.parent)

def state_transition(path, new_state, **fields):
    path = Path(path); old = json.load(open(path)) if path.exists() else {'state':'CREATED'}
    obj = dict(old); obj.update(fields); obj.update({
        'state': new_state, 'previous_state': old.get('state'), 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'hostname': os.uname().nodename, 'pid': os.getpid(),
    })
    atomic_json(path, obj); return obj

def load_yaml(path):
    import yaml
    with open(path) as f: return yaml.safe_load(os.path.expandvars(f.read()))

def separate_group_loss(token_ce, answer_mask, valid_mask, aw=1.0, cw=0.1):
    if len(token_ce) != len(answer_mask) or len(token_ce) != len(valid_mask): raise ValueError('ALIGNMENT_ERROR')
    if any(a and not v for a,v in zip(answer_mask,valid_mask)): raise ValueError('ANSWER_OUTSIDE_VALID')
    context = [v and not a for a,v in zip(answer_mask,valid_mask)]
    if any(a and c for a,c in zip(answer_mask,context)): raise ValueError('MASK_OVERLAP')
    if not all((a or c) == v for a,c,v in zip(answer_mask,context,valid_mask)): raise ValueError('MASK_COVERAGE')
    av=[x for x,a in zip(token_ce,answer_mask) if a]; cv=[x for x,c in zip(token_ce,context) if c]
    if not av: raise ValueError('EMPTY_ANSWER_MASK')
    if not cv: raise ValueError('EMPTY_CONTEXT_MASK')
    al=sum(av)/len(av); cl=sum(cv)/len(cv)
    return al + cw*cl, al, cl, len(av), len(cv)

def clip_delta(values, tau=1e-5):
    if any(not math.isfinite(float(x)) for x in values): raise FloatingPointError('NONFINITE_DELTA')
    norm=math.sqrt(sum(float(x)*float(x) for x in values))
    scale=min(1.0, tau/max(norm, float.fromhex('0x1p-1022')))
    return [float(x)*scale for x in values], norm, norm*scale

def prompt_chunk_plan(n, chunk=1024):
    if n < 0: raise ValueError(n)
    return {'prompt_tokens':n, 'complete_chunks':n//chunk, 'tail_tokens':n%chunk,
            'updates_per_layer':n//chunk, 'generation_updates':0}

def validate_benchmark(full_hash=False):
    manifest=BENCHMARK_ROOT/'benchmark_manifest.json'
    if sha256(manifest) != BENCHMARK_SHA: raise RuntimeError('BLOCKED_EVAL13K_BENCHMARK_HASH_MISMATCH')
    meta=json.load(open(manifest)); ids=[]; task_counts={}
    for task in meta['task_names']:
        path=BENCHMARK_ROOT/'samples'/f'{task}.jsonl'; count=0
        expected=meta['data_files'][task]['sha256']
        if full_hash and sha256(path) != expected: raise RuntimeError(f'benchmark data hash mismatch: {task}')
        with open(path) as f:
            for line in f:
                row=json.loads(line); ids.append(row['sample_id']); count+=1
                if row['task'] != task or row['benchmark_version'] != meta['benchmark_version']: raise RuntimeError('benchmark row identity mismatch')
        task_counts[task]=count
    if len(task_counts)!=13 or set(task_counts.values())!={1000}: raise RuntimeError('benchmark task count mismatch')
    if len(ids)!=13000 or len(set(ids))!=13000: raise RuntimeError('benchmark sample identity mismatch')
    return {'manifest':str(manifest),'sha256':BENCHMARK_SHA,'tasks':13,'samples':13000,'unique_sample_ids':13000,'task_counts':task_counts}

def validate_qa(full_hash=False):
    src=QA_ROOT/'train.jsonl'; masks=QA_MASK_ROOT/'masks.jsonl'
    dm=json.load(open(QA_MASK_ROOT/'derivation_manifest.json'))
    if full_hash and sha256(src) != dm['source_sha256']: raise RuntimeError('QA source hash mismatch')
    if full_hash and sha256(masks) != dm['mask_file_sha256']: raise RuntimeError('QA mask hash mismatch')
    records=seq=answers=0; ids=[]
    with open(src) as f:
        for line in f:
            x=json.loads(line); records+=1; seq+=int(x.get('n_tokens',x.get('num_tokens',0))); ids.append(x.get('id'))
    with open(masks) as f:
        for line in f:
            x=json.loads(line); answers += sum(b-a for a,b in x['answer_ranges'])
    valid=seq-records; context=valid-answers
    expected=(1291,10000000,9998709,13332,9985377)
    actual=(records,seq,valid,answers,context)
    if actual != expected or len(set(ids)) != records: raise RuntimeError(f'BLOCKED_QA10M_INVALID {actual}')
    return dict(zip(('records','sequence_tokens','valid_prediction_tokens','answer_tokens','context_tokens'),actual))

def validate_longtext(full_hash=False):
    meta=json.load(open(LONGTEXT_ROOT/'dataset_manifest.json'))
    counts={p.name:sum(1 for _ in open(p)) for p in (LONGTEXT_ROOT/'train.jsonl',LONGTEXT_ROOT/'validation.jsonl')}
    if counts != {'train.jsonl':21002,'validation.jsonl':220}: raise RuntimeError('longtext record mismatch')
    if (meta['train_tokens'],meta['validation_tokens']) != (297000000,3000000): raise RuntimeError('longtext token mismatch')
    if full_hash:
        if sha256(LONGTEXT_ROOT/'train.jsonl') != meta['train_file_sha256']: raise RuntimeError('train hash mismatch')
        if sha256(LONGTEXT_ROOT/'validation.jsonl') != meta['validation_file_sha256']: raise RuntimeError('validation hash mismatch')
    return {'train_records':21002,'train_tokens':297000000,'validation_records':220,'validation_tokens':3000000}

class RotationStore:
    def __init__(self, root): self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
    def latest(self):
        p=self.root/'latest.json'; return json.load(open(p)) if p.exists() else None
    def save_mock(self, cursor, tokens, payload):
        old=self.latest(); slot='slot_B' if old and old['slot']=='slot_A' else 'slot_A'
        final=self.root/slot; partial=self.root/(slot+'.partial')
        if partial.exists(): shutil.rmtree(partial)
        partial.mkdir(); data=partial/'payload.json'; atomic_json(data,payload)
        manifest={'cursor':cursor,'tokens':tokens,'slot':slot,'files':{'payload.json':sha256(data)},'complete':True}
        atomic_json(partial/'checkpoint_manifest.json',manifest); fsync_dir(partial)
        if final.exists(): shutil.rmtree(final)
        os.replace(partial,final); fsync_dir(self.root)
        atomic_json(self.root/'latest.json',{'slot':slot,'cursor':cursor,'tokens':tokens,'manifest_sha256':sha256(final/'checkpoint_manifest.json')})
        return slot
    def validate_latest(self):
        x=self.latest(); d=self.root/x['slot']; m=json.load(open(d/'checkpoint_manifest.json'))
        return m['complete'] and all(sha256(d/p)==h for p,h in m['files'].items())
