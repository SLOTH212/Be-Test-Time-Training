#!/usr/bin/env python3
import json,os,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from code.runtime import action_layers,best_single,complete_chunks,dynamic_required,exclusive_claim,fixed_actions,load_config,replay_budget,sample_best,validate_layers

def main():
    hist=[0,6,12,18,24];scale=[0,6,12,18,24,30]
    assert fixed_actions(hist)==['OFF','L0','L6','L12','L18','L24','ALL']
    assert fixed_actions(scale)==['OFF','L0','L6','L12','L18','L24','L30','ALL']
    assert action_layers(scale)['ALL']==scale and action_layers(scale)['L30']==[30]
    assert replay_budget(7,15,4,1)==385
    assert replay_budget(8,8,4,1)==260
    assert complete_chunks(32768,4096)==8 and complete_chunks(32767,4096)==7
    assert not dynamic_required(1.0) and dynamic_required(.999)
    scores={a:0.1 for a in fixed_actions(hist)};scores['ALL']=.2
    assert sample_best(scores,hist)==(.2,['ALL'])
    rows=[{f'L{x}':float(x) for x in hist},{f'L{x}':float(x) for x in hist}]
    assert best_single(rows,hist)==('L24',24.0)
    validate_layers(scale,36)
    try:validate_layers(scale,30);raise AssertionError('layer gate failed')
    except ValueError:pass
    for path in (ROOT/'configs').glob('*.yaml'):load_config(path)
    with tempfile.TemporaryDirectory() as tmp:
        claim=Path(tmp)/'x.claim';exclusive_claim(claim,{'sample_id':'x'})
        try:exclusive_claim(claim,{'sample_id':'x'});raise AssertionError('O_EXCL failed')
        except FileExistsError:pass
    print(json.dumps({"status":"PASS","HISTORICAL_ACTION_SET_PARITY":"PASS","SCALEUP_ACTION_SET_TEST":"PASS","DYNAMIC_32K_CONFIG_STATIC_TEST":"PASS","GENERIC_REPLAY_BUDGET_STATUS":"PASS","SAMPLE_BEST_IMPLEMENTATION_STATUS":"PASS","resume_ownership_logic":"PASS"},sort_keys=True))

if __name__=='__main__':main()
