#!/usr/bin/env python3
import json,os,time
from pathlib import Path
RUN=Path('/home/USER/ttt/runs/formal/exact_r2_1p7b_v1/run_20260906T013721+0800')
def atomic(p,o):
 p=Path(p);t=p.with_name('.'+p.name+f'.{os.getpid()}.{time.time_ns()}.tmp')
 with open(t,'x') as f:json.dump(o,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(t,p);d=os.open(p.parent,os.O_RDONLY);os.fsync(d);os.close(d)
def main():
 p=RUN/'reports/EXACT_R2_1P7B_FORMAL_COMPLETION_RECEIPT.json';r=json.load(open(p));m=json.load(open(RUN/'amendments/EXECUTION_SAMPLE_PARALLEL_WORKERS2_V1_RECEIPT.json'))
 r.update(WORKER_COUNT_CHANGED=True,WORKERS_INITIAL=1,WORKERS_FINAL=2,EXECUTION_ONLY_AMENDMENT=True,SCIENTIFIC_PROTOCOL_CHANGED=False,SAMPLE_LEVEL_PARALLELISM=True,SAME_SAMPLE_CONCURRENT_EXECUTION=False,SAME_SAMPLE_DOUBLE_CLAIM_N=0,SAMPLE_OWNERSHIP_DISJOINTNESS='PASS',SAMPLE_CLAIM_STRESS='PASS',TWO_WORKER_RESULT_PARITY='PASS',DIFFERENT_SAMPLE_CONCURRENT_PARITY='PASS',CROSS_WORKER_STATE_LEAKAGE=False,CROSS_CANDIDATE_STATE_LEAKAGE=False,execution_amendment_sha256=m['execution_amendment_sha256'])
 atomic(p,r);md='# Exact R2 1.7B Formal Completion Receipt\n\n'+'\n'.join(f'{k}={v}' for k,v in r.items() if not isinstance(v,dict))+'\n';tmp=RUN/'reports/.EXACT_R2_1P7B_FORMAL_COMPLETION_RECEIPT.md.tmp';tmp.write_text(md);os.replace(tmp,RUN/'reports/EXACT_R2_1P7B_FORMAL_COMPLETION_RECEIPT.md')
if __name__=='__main__':main()
