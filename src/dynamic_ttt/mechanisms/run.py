"""Replay formal Qwen mechanisms using local reconstructed benchmark inputs."""
import argparse,json
from pathlib import Path
from dynamic_ttt.mechanisms import runtime as rt
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--cohort',required=True);p.add_argument('--benchmark',required=True);p.add_argument('--fixed',required=True);p.add_argument('--output',required=True);p.add_argument('--plan',action='store_true');x=p.parse_args()
 cfg=rt.load_config(x.config);samples=rt.rows(x.cohort)
 if x.plan:print(json.dumps({'mechanism':cfg['mechanism'],'samples':len(samples),'classification':cfg['classification']}));return
 from dynamic_ttt.mechanisms.execution import Executor
 from dynamic_ttt.eval.runtime import atomic_json
 ex=Executor(cfg,rt.rows(x.benchmark),json.loads(Path(x.fixed).read_text()),x.output)
 identity=rt.canonical(cfg)
 for s in samples:
  path=Path(x.output)/(rt.canonical(s['sample_id'])+'.json')
  if path.exists():
   saved=json.loads(path.read_text())
   if saved['config_identity']!=identity or saved['sample_id']!=s['sample_id']:raise ValueError('RESUME_IDENTITY')
   continue
  result=ex.execute(s);result['config_identity']=identity;atomic_json(path,result)
if __name__=='__main__':main()
