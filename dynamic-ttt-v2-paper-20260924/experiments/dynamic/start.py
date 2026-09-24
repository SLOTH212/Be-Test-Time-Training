import os,sys,argparse,subprocess,json,datetime
from pathlib import Path
os.sched_setaffinity(0,set(range(8,24)))
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--machine',choices=['local','remote'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpus');p.add_argument('--cpu-sets');a=p.parse_args()
a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
gpus=a.gpus or ('0,0' if a.machine=='local' else '0,1,2,3');cpus=a.cpu_sets or ('8-15;16-23' if a.machine=='local' else '8-11;12-15;16-19;20-23')
for phase in ['bridge','run']:
 (a.output/'STARTUP_STATUS.json').write_text(json.dumps(dict(phase=phase,state='RUNNING',at=datetime.datetime.now().astimezone().isoformat())))
 subprocess.run([sys.executable,'-u',str(R/'pipeline.py'),phase,'--machine',a.machine,'--output',str(a.output),'--gpus',gpus,'--cpu-sets',cpus],check=True)
(a.output/'STARTUP_STATUS.json').write_text(json.dumps(dict(state='COMPLETE',at=datetime.datetime.now().astimezone().isoformat())))
