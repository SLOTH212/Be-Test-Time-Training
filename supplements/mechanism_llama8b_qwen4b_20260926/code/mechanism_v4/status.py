"""Read-only snapshot; count each imported/resumed R2 candidate exactly once."""
from pathlib import Path
import json,time,collections
S=Path('/path/to/ttt/work/mechanism_v4_8gpu_20260926')
O=Path('/path/to/ttt/runs/mechanism_v4_8gpu_20260926')
def read(p):return json.loads(p.read_text())
def main():
 j=read(S/'JOB.json');out=dict(time=time.strftime('%Y-%m-%d %H:%M:%S %z'),models={})
 for name in ['PHASE_STATUS','SMOKE_STATUS','LAUNCH']:
  if (O/(name+'.json')).exists():out[name]=read(O/(name+'.json'))
 for label,c in j['models'].items():
  d=O/label;counts=collections.Counter();total=collections.Counter(t['kind'] for t in c['tasks']);sample_done=0;r2={};sham=selective=kv=0
  lat=collections.defaultdict(list)
  for x in c['cohort']:
   sid=x['sample_id'];r2[sid]={};p=O/'imported'/label/sid.replace(':','_')/'R2_PROGRESS.json'
   if p.exists():r2[sid].update({v['full_schedule_hash']:v for v in read(p)['records']})
  for t in c['tasks']:
   p=d/'tasks'/(t['id']+'.json');partial=d/'partial'/(t['id']+'.json')
   if p.exists():
    rec=read(p);assert rec['status']=='COMMITTED' and rec['task']==t
    counts[t['kind']]+=1;v=rec['result']
    if t['kind']=='r2':r2[t['sample_id']].update({z['full_schedule_hash']:z for z in v['records']})
    if t['kind']=='baseline':kv+=int(v.get('kv',{}).get('native_closure')=='PASS')
    if t['kind']=='selective' and v['status']=='PASS':selective+=1;sham+=int(v['sham'] is not None)
   elif partial.exists():
    rec=read(partial);assert rec['task']==t
    r2[t['sample_id']].update({z['full_schedule_hash']:z for z in rec['records']})
  for x in c['cohort']:
   ts=[t for t in c['tasks'] if t['sample_id']==x['sample_id']]
   sample_done+=all((d/'tasks'/(t['id']+'.json')).exists() for t in ts)
  workers=[]
  for p in d.glob('worker*_progress.json'):
   v=read(p);v['age_seconds']=round(time.time()-v['time'],1);workers.append(v)
  out['models'][label]=dict(sample_complete=sample_done,sample_total=len(c['cohort']),tasks_complete=dict(counts),tasks_total=dict(total),r2_complete=sum(len(v) for v in r2.values()),r2_total=sum(56*(len(x['sequence'])-1) for x in c['cohort']),kv_closure_pass=kv,selective_eligible_complete=selective,sham_complete=sham,workers=workers)
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
