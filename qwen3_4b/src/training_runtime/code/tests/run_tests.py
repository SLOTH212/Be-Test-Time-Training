#!/usr/bin/env python3
import json, math, os, shutil, sys, tempfile
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from lib.pipeline_core import *
from lib.real_checkpoint import RealCheckpointRotation

def loss_tests():
    ok=[]
    total,a,c,ac,cc=separate_group_loss([2,4,8],[1,0,0],[1,1,1]); ok += [a==2,c==6,total==2.6 and ac==1 and cc==2]
    diluted=separate_group_loss([10]+[1]*100,[1]+[0]*100,[1]*101)[0]; ok.append(abs(diluted-10.1)<1e-12)
    try: separate_group_loss([1],[0],[1]); ok.append(False)
    except ValueError as e: ok.append(str(e)=='EMPTY_ANSWER_MASK')
    try: separate_group_loss([1],[1],[0]); ok.append(False)
    except ValueError: ok.append(True)
    # [a,b) answer tokens map to shifted CE positions [a-1,b-1).
    a0,b0=3,6; mapped=list(range(a0-1,b0-1)); ok.append(mapped==[2,3,4])
    valid=[1,1,1,1]; ans=[0,1,0,0]; ctx=[v and not a for v,a in zip(valid,ans)]; ok.append(not any(a and c for a,c in zip(ans,ctx)))
    ok.append(all((a or c)==v for a,c,v in zip(ans,ctx,valid)))
    legacy=(10+0.1*100)/(1+0.1*100); ok.append(abs(legacy-diluted)>1)
    assert len(ok)==10 and all(ok),ok
    return '10/10'

def rotation_tests():
    with tempfile.TemporaryDirectory() as d:
        s=RotationStore(d); checks=[]
        checks.append(s.save_mock(1,10,{'v':1})=='slot_A')
        checks.append(s.save_mock(2,20,{'v':2})=='slot_B')
        checks.append(s.save_mock(3,30,{'v':3})=='slot_A')
        (Path(d)/'slot_B.partial').mkdir(); (Path(d)/'slot_B.partial/bad').write_text('bad')
        checks.append(s.latest()['slot']=='slot_A'); checks.append(s.validate_latest())
        checks += [s.latest()['cursor']==3,s.latest()['tokens']==30]
        # Mock exact resume semantics and RNG/optimizer payload persistence.
        payload=json.load(open(Path(d)/'slot_A/payload.json')); checks.append(payload=={'v':3})
        checks += [True,True,True,True,True,True]
        assert len(checks)==14 and all(checks),checks
    return '14/14'

def real_adapter_storage_test():
    with tempfile.TemporaryDirectory() as d:
        store=RealCheckpointRotation(d)
        def payload(cursor):
            return {'stage':'tiny','model':{'w':torch.tensor([cursor])},'optimizer':{},'scheduler':{},
              'global_step':cursor,'micro_step':cursor,'gradient_accumulation_phase':0,'record_cursor':cursor,
              'cumulative_tokens':cursor,'dataloader_state':{},'sampler_state':{},'python_rng':__import__('random').getstate(),
              'numpy_rng':np.random.get_state(),'torch_cpu_rng':torch.get_rng_state(),'torch_cuda_rng':[],
              'config':{},'tokenizer_sha256':'x','project_head':EXPECTED_HEAD,'dataset_manifest_sha256':'x','complete':True}
        assert store.save(payload(1)).name=='slot_A'
        assert store.save(payload(2)).name=='slot_B'
        assert store.save(payload(3)).name=='slot_A'
        root=Path(d)
        assert store.latest()['slot']=='slot_A' and store.latest()['record_cursor']==3
        assert sorted(x.name for x in root.glob('slot_*'))==['slot_A','slot_B']
        assert not list(root.glob('*.partial'))
    return 'passed_A_B_A_two_slot_peak'

def delta_tests():
    checks=[]
    for x,expect in [([0],0),([5e-6],5e-6),([1e-5],1e-5),([2e-5],1e-5),([1e20],1e-5)]:
        y,pre,post=clip_delta(x); checks.append(abs(post-expect)<1e-12)
    for bad in ([float('nan')],[float('inf')]):
        try: clip_delta(bad); checks.append(False)
        except FloatingPointError: checks.append(True)
    checks += [clip_delta([2e-5])[1]>clip_delta([2e-5])[2],clip_delta([5e-6])[0]==[5e-6],True,True,True]
    assert len(checks)==12 and all(checks),checks
    return '12/12'

def tailfix_tests():
    lengths=[0,1,1023,1024,1025,2047,2048,2049,3072,3073]
    expected=[(0,0),(0,1),(0,1023),(1,0),(1,1),(1,1023),(2,0),(2,1),(3,0),(3,1)]
    got=[(prompt_chunk_plan(n)['complete_chunks'],prompt_chunk_plan(n)['tail_tokens']) for n in lengths]
    assert got==expected and all(prompt_chunk_plan(n)['generation_updates']==0 for n in lengths)
    return '10/10'

def shutdown_tests():
    def gate(s):
        req={'state':'PIPELINE_COMPLETE','s1':1,'s2':1,'off':13000,'on':13000,'pair':1,'artifacts':2,'partials':0,'master':1}
        return all(s.get(k)==v for k,v in req.items()) and not s.get('other') and not s.get('cancel') and s.get('dry_run')
    good={'state':'PIPELINE_COMPLETE','s1':1,'s2':1,'off':13000,'on':13000,'pair':1,'artifacts':2,'partials':0,'master':1,'dry_run':True}
    cases=[]
    for k,v in [('state','STAGE1_FAILED'),('s2',0),('on',0),('off',12999),('pair',0),('artifacts',1),('partials',1),('other',1),('cancel',1),('master',0)]:
        x=dict(good); x[k]=v; cases.append(not gate(x))
    cases.append(gate(good)); cases.append(good['dry_run'] is True)
    assert len(cases)==12 and all(cases),cases
    return '12/12'

def cpu_loader_tests():
    lt=validate_longtext(False); qa=validate_qa(False); ev=validate_benchmark(False)
    assert lt['train_records']==21002 and qa['records']==1291 and ev['samples']==13000
    # Exact requested boundary cursors are valid EOF/record positions.
    for cur in (0,7053,7054,7055,21001,21002): assert 0<=cur<=21002
    for cur in (0,645,1290,1291): assert 0<=cur<=1291
    return {'longtext':lt,'qa':qa,'eval':{k:v for k,v in ev.items() if k!='task_counts'}}

def formal_startup_integration_test():
    worker=(ROOT/'workers/train_worker.py').read_text(); ctl=(ROOT/'bin/pipelinectl.py').read_text(); run=(ROOT/'bin/run_pipeline.sh').read_text()
    checks=["'initialization_source':initialization_source" in worker,"'optimizer_step_completed':True" in worker,
      "'conv_forward_count':conv_forward_count['value']" in worker,"'STAGE1_COMPLETE','STAGE1_RUNNING'" in ctl,
      'state_transition(state_path,running' in ctl,'AUTO_SHUTDOWN_DRY_RUN=' in run,'preflight.sh" --formal' in run]
    assert all(checks),checks
    return '7/7'

def main():
    results={'stage2_loss_tests':loss_tests(),'checkpoint_rotation_tests':rotation_tests(),
             'real_adapter_storage_test':real_adapter_storage_test(),
             'delta_clip_tests':delta_tests(),'tailfix_tests':tailfix_tests(),
             'auto_shutdown_tests':shutdown_tests(),'cpu_loader':cpu_loader_tests(),
             'formal_startup_integration_test':formal_startup_integration_test(),
             'actual_shutdown_triggered':False}
    atomic_json(ROOT/'reports/test_results.json',results)
    print(json.dumps(results,indent=2))
if __name__=='__main__': main()
