#!/usr/bin/env python3
"""Independent result-set integrity audit; never imposes Dynamic >= Sample Best."""
import argparse,hashlib,json,math,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from code.runtime import fixed_actions,load_config

def rows(path):
    with open(path,encoding='utf-8') as f:return [json.loads(x) for x in f if x.strip()]

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--fixed-jsonl');p.add_argument('--dynamic-dir');p.add_argument('--expected-samples',type=int);a=p.parse_args();cfg=load_config(a.config);report={"status":"PASS","checks":{}}
    if a.fixed_jsonl:
        data=rows(a.fixed_jsonl);actions=fixed_actions(cfg['ttt_layers']);keys=[(x.get('sample_id'),x.get('action',x.get('mode'))) for x in data];counts=Counter(k[1] for k in keys);samples={k[0] for k in keys};expected=a.expected_samples or len(samples);report['checks']['fixed']={"rows":len(data),"expected_cells":expected*len(actions),"duplicates":len(keys)-len(set(keys)),"action_counts":dict(counts),"errors":sum(x.get('status') not in ('SUCCESS','ok') or not isinstance(x.get('score'),(int,float)) or not math.isfinite(x['score']) for x in data)}
        if len(data)!=expected*len(actions) or len(keys)!=len(set(keys)) or any(counts[x]!=expected for x in actions):report['status']='FAIL'
    if a.dynamic_dir:
        data=[]
        for path in Path(a.dynamic_dir).glob('*.json'):
            try:data.append(json.loads(path.read_text()))
            except Exception:data.append({"search_status":"JSON_ERROR"})
        ids=[x.get('sample_id') for x in data];report['checks']['dynamic']={"results":len(data),"duplicates":len(ids)-len(set(ids)),"errors":sum(x.get('search_status')!='SUCCESS' for x in data),"note":"No Dynamic>=SampleBest constraint imposed; actual outcomes are retained."}
        if len(ids)!=len(set(ids)) or any(x.get('search_status')!='SUCCESS' for x in data):report['status']='FAIL'
    print(json.dumps(report,sort_keys=True));raise SystemExit(report['status']!='PASS')
if __name__=='__main__':main()
