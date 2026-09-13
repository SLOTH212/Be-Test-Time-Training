"""Run the source-preserved estimators on complete execution JSONL."""
import argparse,json
from pathlib import Path
from dynamic_ttt.stats.mechanisms import aggregate
def main():
 p=argparse.ArgumentParser()
 p.add_argument('--mechanism',choices=['reverse','reset','kp','deletion','r2'],required=True)
 p.add_argument('--input',required=True);p.add_argument('--seeds',required=True);p.add_argument('--output',required=True)
 x=p.parse_args();rows=[json.loads(s) for s in Path(x.input).read_text(encoding='utf-8').splitlines() if s.strip()]
 seeds=json.loads(Path(x.seeds).read_text(encoding='utf-8'))
 if not isinstance(seeds.get('resamples'),int) or seeds['resamples']<1:raise ValueError('INVALID_RESAMPLES')
 Path(x.output).write_text(json.dumps(aggregate(rows,x.mechanism,seeds),indent=2,allow_nan=False)+'\n',encoding='utf-8')
if __name__=='__main__':main()
