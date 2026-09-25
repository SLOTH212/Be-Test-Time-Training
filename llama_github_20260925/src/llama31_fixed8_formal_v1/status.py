"""Read-only progress snapshot for the isolated Llama Fixed8 experiment."""
import json,time,collections,statistics
from pathlib import Path
R=Path('/path/to/ttt')
C=R/'configs/llama31_8b_ruler32k_fixed8_4k_formal_v1.json'
c=json.loads(C.read_text());root=Path(c['output_root']);state=json.loads((root/'PHASE_STATUS.json').read_text()) if (root/'PHASE_STATUS.json').exists() else {'status':'NOT_STARTED'}
paths=list((root/'samples').glob('*.json'));rows=[json.loads(p.read_text()) for p in paths]
by=collections.Counter(x['sample_id'] for x in rows);recent=sorted(rows,key=lambda x:x['time'])[-200:]
out={'checked_at_unix':time.time(),'state':state,'completed_cells':len(rows),'expected_cells':52000,'completed_samples':sum(v==8 for v in by.values()),'expected_samples':6500,'task_cell_counts':dict(collections.Counter(x['task'] for x in rows)),'worker_progress':{p.name:json.loads(p.read_text()) for p in root.glob('worker*_progress.json')},'worker_done':len(list(root.glob('worker*_done.json')))}
if recent:out.update(recent_mean_cell_seconds=statistics.mean(x['latency_seconds'] for x in recent),estimated_remaining_hours_at_four_workers=(52000-len(rows))*statistics.mean(x['latency_seconds'] for x in recent)/4/3600)
print(json.dumps(out,indent=2))
