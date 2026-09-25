import os,json,hashlib
from pathlib import Path
W=Path('/path/to/ttt/work/qwen3_4b_stage1_resume_v3')
A=W.parents[1]/'audits/v3'
source=(W/'v3_metrics_cached_tolerance.py').read_text()
assert hashlib.sha256(source.encode()).hexdigest()=='3fc96f50ed7edd621046be87c1a093e08f5c0c0cd0348349766da06fff9beca8'
index=int(os.environ['V3_CPU_PARTITION'])
assigned=set(json.loads((A/'CPU_PARALLEL_PARTITIONS.json').read_text())['partitions'][index])
old="target=OUT/'ALL_TENSOR_PAIR_METRICS.jsonl'"
assert source.count(old)==1
source=source.replace(old,"target=OUT/'PARALLEL_PART_%d.jsonl'"%index)
old=' for ki,key in enumerate(keys):'
assert source.count(old)==1
source=source.replace(old,old+'\n  if key not in assigned:continue')
source=source.replace("OUT/'PAIR_METRICS_COMPLETE.json'","OUT/'PARALLEL_PART_%d_COMPLETE.json'"%index)
exec(compile(source,str(W/'v3_metrics_cached_tolerance.py'),'exec'))
