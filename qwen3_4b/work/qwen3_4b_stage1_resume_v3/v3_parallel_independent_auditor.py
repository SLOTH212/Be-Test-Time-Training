import json,hashlib,time
from pathlib import Path
R=Path('/path/to/ttt');W=R/'work/qwen3_4b_stage1_resume_v3';A=R/'audits/v3';D=A/'analysis'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=read(A/'CPU_PARALLEL_PARTITIONS.json');raw=(D/'ALL_TENSOR_PAIR_METRICS.jsonl').read_bytes();rows=[json.loads(x) for x in raw.splitlines()];checks={}
checks['cached_source_unchanged']=sha(W/'v3_metrics_cached_tolerance.py')==p['cached_source_sha256']
checks['wrapper_unchanged']=sha(W/'v3_metrics_parallel_worker.py')==p['worker_sha256']
checks['preserved_prefix_exact']=hashlib.sha256(raw[:p['preserved_prefix_bytes']]).hexdigest()==p['preserved_prefix_sha256']
metadata=read(R/'runs/qwen3_4b_stage1_resume_v3/attempt2/U1/final_rank0.json')['tensors'];keys=sorted(metadata)
checks['ordered_complete_1647_keys']=[x['key'] for x in rows]==keys and len(rows)==1647
bykey={x['key']:x for x in rows};assigned=[]
for i in range(3):
 part=[json.loads(x) for x in (D/('PARALLEL_PART_%d.jsonl'%i)).read_text().splitlines()];assigned+=p['partitions'][i]
 checks['partition_%d_exact_key_coverage'%i]=sorted(x['key'] for x in part)==sorted(p['partitions'][i])
 checks['partition_%d_all_full_records_equal_merge'%i]=all(bykey[x['key']]==x for x in part)
checks['partition_disjoint_remainder']=len(assigned)==len(set(assigned)) and sorted(assigned)==keys[p['preserved_rows']:]
checks['all_workers_succeeded']=all(x['returncode']==0 for x in read(A/'CPU_PARALLEL_RECEIPTS.json') if 'partition' in x)
checks['merged_digest_matches_complete_marker']=sha(D/'ALL_TENSOR_PAIR_METRICS.jsonl')==read(D/'PAIR_METRICS_COMPLETE.json')['metrics_sha256']
assert all(checks.values()),checks
out={'status':'PASS','checks':checks,'check_count':len(checks),'source_sha256':sha(Path(__file__)),'completed_at':time.time(),'scope':'Independent exact key/prefix/record/hash/receipt audit of partitioned CPU assembly. Per-tensor calculations retain the hash-bound validated cached-tolerance source.'}
(A/'CPU_PARALLEL_INDEPENDENT_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print('CPU_PARALLEL_INDEPENDENT_AUDIT_PASS',len(checks))
