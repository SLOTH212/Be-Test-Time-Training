#!/usr/bin/env python3
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lib.pipeline_core import *

def shutdown_gate():
    state=json.load(open(PIPELINE_ROOT/'state/pipeline_state.json'))
    required={
      'state':'PIPELINE_COMPLETE','stage1_final_valid':True,'stage2_final_valid':True,
      'off_success_count':13000,'on_success_count':13000,'pairing_valid':True,
      'final_artifact_count':2,'rotation_checkpoints_remaining':0,'partial_checkpoints_remaining':0,
      'final_artifact_validation':'passed','final_cleanup':'passed'}
    bad={k:(state.get(k),v) for k,v in required.items() if state.get(k)!=v}
    if bad: raise RuntimeError('PIPELINE_FINAL_VALIDATION_FAILED '+repr(bad))
    print('AUTO_SHUTDOWN_FINAL_VALIDATION=passed')

def validate_inference():
    off=PIPELINE_ROOT/'inference/off/predictions.jsonl'; on=PIPELINE_ROOT/'inference/on/predictions.jsonl'
    def records(path): return [json.loads(x) for x in open(path)]
    a,b=records(off),records(on); aid=[x['sample_id'] for x in a]; bid=[x['sample_id'] for x in b]
    assert len(a)==len(b)==13000 and len(set(aid))==len(set(bid))==13000 and aid==bid
    assert len({x['task'] for x in a})==len({x['task'] for x in b})==13
    from collections import Counter
    assert set(Counter(x['task'] for x in a).values())=={1000} and set(Counter(x['task'] for x in b).values())=={1000}
    assert all(x['checkpoint_hash']==y['checkpoint_hash'] and x['benchmark_hash']==y['benchmark_hash'] and
               x['prompt_hash']==y['prompt_hash'] for x,y in zip(a,b))
    assert all(x['ttt_update_count']==0 and x['generation_update_count']==0 and x['conv_forward_count']==0 for x in a)
    assert all(x['ttt_lr']==1 and abs(x['delta_clip_frobenius']-1e-5)<1e-12 and
               x['generation_update_count']==0 and x['conv_forward_count']==0 for x in b)
    print('FINAL_PAIRING_VALIDATION=passed')

def cleanup():
    state_path=PIPELINE_ROOT/'state/pipeline_state.json'; state=json.load(open(state_path))
    if state.get('state')!='FINAL_CLEANUP': raise RuntimeError('cleanup state gate failed')
    for p in (PIPELINE_ROOT/'artifacts/stage1_final_resume',PIPELINE_ROOT/'artifacts/stage2_final_eval'):
        if not p.is_dir() or not (p/'SHA256SUMS.txt').is_file(): raise RuntimeError('final artifact invalid')
    import shutil
    for stage in ('stage1','stage2'):
        for slot in ('slot_A','slot_B'):
            p=PIPELINE_ROOT/'checkpoints'/stage/slot
            if p.exists(): shutil.rmtree(p)
    state_transition(state_path,'PIPELINE_COMPLETE',final_artifact_count=2,rotation_checkpoints_remaining=0,partial_checkpoints_remaining=0,final_cleanup='passed')

def run(_resume):
    # Formal workers are separate processes so OFF and ON never share model state.
    state_path=PIPELINE_ROOT/'state/pipeline_state.json'
    state=json.load(open(state_path)); current=state.get('state','CREATED')
    stages=[
      ('PREFLIGHT_COMPLETE','PREFLIGHT_RUNNING','BLOCKED_PREFLIGHT',['bash',str(PIPELINE_ROOT/'bin/preflight.sh'),'--formal']),
      ('STAGE1_COMPLETE','STAGE1_RUNNING','STAGE1_FAILED',['bash',str(PIPELINE_ROOT/'bin/run_stage1.sh')]),
      ('STAGE2_COMPLETE','STAGE2_RUNNING','STAGE2_FAILED',['bash',str(PIPELINE_ROOT/'bin/run_stage2.sh')]),
      ('EVAL_OFF_COMPLETE','EVAL_OFF_RUNNING','EVAL_OFF_FAILED',['bash',str(PIPELINE_ROOT/'bin/run_eval.sh'),'off']),
      ('EVAL_ON_COMPLETE','EVAL_ON_RUNNING','EVAL_ON_FAILED',['bash',str(PIPELINE_ROOT/'bin/run_eval.sh'),'on'])]
    import subprocess
    order=['CREATED']+[target for target,_,_,_ in stages]+['FINAL_CLEANUP','PIPELINE_COMPLETE']
    running_previous={running:order[index] for index,(_,running,_,_) in enumerate(stages)}
    if current in running_previous: current=running_previous[current]
    if current not in order: raise RuntimeError(f'PIPELINE_RESUME_STATE_UNSUPPORTED={current}')
    current_index=order.index(current)
    for target,running,failed,cmd in stages:
        if order.index(target) <= current_index: continue
        fields={'next_expected_state':target}
        if running=='STAGE1_RUNNING': fields['formal_stage1_started']=True
        if running=='STAGE2_RUNNING': fields['formal_stage2_started']=True
        if running=='EVAL_OFF_RUNNING': fields['formal_off_inference_started']=True
        if running=='EVAL_ON_RUNNING': fields['formal_on_inference_started']=True
        state_transition(state_path,running,**fields)
        try: subprocess.run(cmd,check=True)
        except Exception:
            state_transition(state_path,failed,next_expected_state=None)
            raise
        current=target; state_transition(state_path,target); current_index=order.index(target)
    if current == 'PIPELINE_COMPLETE':
        validate_inference()
    elif current == 'FINAL_CLEANUP':
        subprocess.run(['bash',str(PIPELINE_ROOT/'bin/cleanup_rotation_slots.sh'),'--execute'],check=True)
    else:
        validate_inference(); state_transition(state_path,'FINAL_CLEANUP',pairing_valid=True,off_success_count=13000,on_success_count=13000,stage1_final_valid=True,stage2_final_valid=True,final_artifact_validation='passed')
        subprocess.run(['bash',str(PIPELINE_ROOT/'bin/cleanup_rotation_slots.sh'),'--execute'],check=True)
    env=os.environ.copy(); env.update(PIPELINE_MASTER_PROCESS='1',PIPELINE_EXECUTION_MODE='formal')
    subprocess.run(['bash',str(PIPELINE_ROOT/'bin/auto_shutdown.sh')],check=True,env=env)

def main():
    p=argparse.ArgumentParser(); p.add_argument('cmd',choices=['run','cleanup','shutdown-gate','validate-inference']); p.add_argument('arg',nargs='?'); a=p.parse_args()
    {'run':lambda:run(a.arg),'cleanup':cleanup,'shutdown-gate':shutdown_gate,'validate-inference':validate_inference}[a.cmd]()
if __name__=='__main__': main()
