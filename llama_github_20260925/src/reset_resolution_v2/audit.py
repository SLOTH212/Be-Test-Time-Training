"""CPU-only morphology audit; no model/runtime/intervention imports."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import ast
import collections
import hashlib
import itertools
import json
import math
import pathlib
import socket
import getpass
import sys

R = pathlib.Path('/path/to/ttt')
P = R / 'releases/ttt_mechanism_runtime_v1'
A = R / 'audits/reset_resolution_v2'
S = R / 'src/reset_resolution_v2'
S.mkdir(parents=True, exist_ok=True)
A.mkdir(parents=True, exist_ok=True)
EXPECTED = 'ce9ee7c7066d0413e4fcf90b69543cfa605ac1761af8d849ddb2d61ccd6fb1a8'
ACTIONS = ['OFF', 'L0', 'L6', 'L12', 'L18', 'L24', 'ALL']
access = []
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576), b''): h.update(b)
    return h.hexdigest()
def read(p, mode, fields=None):
    access.append({'path': str(p), 'mode': mode, 'fields': fields, 'sha256': sha(p)})
    if mode == 'hash_only': return sha(p)
    if mode == 'project_jsonl':
        return [{k:v for k,v in json.loads(l).items() if k in fields} for l in p.read_text().splitlines() if l.strip()]
    if mode == 'project_json': return {k:v for k,v in json.loads(p.read_text()).items() if k in fields}
    return p.read_text()
def put(name, obj):
    p=A/name; p.write_text(json.dumps(obj, indent=2, sort_keys=True)+'\n'); return p
assert socket.gethostname() == 'amax' and getpass.getuser() == 'USER'
archive=R/'packages/ttt_mechanism_runtime_v1.tar.gz'
assert read(archive,'hash_only') == EXPECTED
manifest=json.loads(read(P/'provenance/RELEASE_MANIFEST.json','protocol_metadata'))
before={str(P/k):read(P/k,'hash_only') for k in manifest['files']}
assert all(before[str(P/k)]==v for k,v in manifest['files'].items())
for p in [P/'provenance/RELEASE_MANIFEST.json', R/'provenance/QWEN3_4B_DOWNSTREAM_PIPELINE_DEBUG_AUTHORITY_V1.json']:
    before[str(p)]=read(p,'hash_only')
put('IMMUTABLE_BASELINE.json',{'files':before,'package_sha256':EXPECTED,'release_manifest_file_count':len(manifest['files']),'mode':'byte hashing only; no outcome decoding'})
source=P/'provenance/original/reset/screening/scripts/screen_state_reset.py'
src=read(source,'source_static')
tree=ast.parse(src)
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='historical_early_steady')
source_authority=P/'provenance/MECHANISM_SOURCE_AUTHORITY.json'
sa=json.loads(read(source_authority,'protocol_metadata'))
entry=next(x for x in sa['entries'] if x['packaged_path']==str(source.relative_to(P)))
assert sha(source) in entry.values()
ns={'Counter':collections.Counter,'ACTIONS':ACTIONS}
exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
v1=ns['historical_early_steady']
chunk_source=R/'vendor/frozen/ttt_inference_runtime_v1/dynamic/formal_dynamic_runner.py'
chunk_src=read(chunk_source,'source_static')
assert 'self.full_chunks=self.n//1024' in chunk_src and 'len(seq)!=self.full_chunks' in chunk_src
contract={'status':'PASS','source':str(source),'source_sha256':sha(source),'source_authority':str(source_authority),'source_authority_sha256':sha(source_authority),'exact_function_source':ast.get_source_segment(src,node),'native_switch_loop':'switches=[i for i in range(1,N) if sequence[i]!=sequence[i-1]]','first_switch_index_convention':'reported position s=switches[0]+1 is 1-based switched-to chunk; internal i is 0-based; earliest s=2','V1_RULE_NATIVE_FORM':'first_norm=(switches[0]+1)/N if switches and N else None; early=first_norm is not None and first_norm<=0.25; eligible=early and steady_fraction>=0.8','sequence_length':'N=len(sequence)=floor(tokenized_prompt_length/chunk_size); historical chunk_size=1024; tail excluded; generation adds no decision chunks','chunk_source':str(chunk_source),'chunk_source_sha256':sha(chunk_source),'late_half':'sequence[N//2:]; 0-based start floor(N/2); 1-based start floor(N/2)+1; length N-floor(N/2)','modal_tie_order':ACTIONS,'OFF_is_action':True,'no_switch':'ineligible','rounding':'no rounding of fractions; Python float division and inclusive comparison; equivalent early integer limit floor(N/4)','modal_computation':'max(ACTIONS,key=lambda a:(counts[a],-ACTIONS.index(a))); count/max(len(tail),1)>=0.8','LATE_MODAL_RULE_CHANGED':False}
put('HISTORICAL_RESET_V1_CONTRACT.json',contract)
parent=read(P/'provenance/IMPROVED_173_PARENT_COHORT.jsonl','project_jsonl',['sample_id','task','action_sequence','chunk_count','completed'])
frozen=read(P/'provenance/original/reset/manifests/STATE_RESET_167_COHORT.jsonl','project_jsonl',['canonical_sample_id','original_dynamic_actions','full_chunk_count','sequence_length','replayable'])
auth={x['canonical_sample_id'] for x in frozen}
old={x['sample_id'] for x in parent if v1(x['action_sequence'])[0]}
assert len(parent)==173 and len(old)==len(auth)==167 and old==auth, 'STOP_V1_REPLAY_FAILURE'
byid={x['sample_id']:x for x in parent}
assert len(byid)==173 and all(len(x['action_sequence'])==x['chunk_count'] for x in parent)
assert all(x['original_dynamic_actions']==byid[x['canonical_sample_id']]['action_sequence'] and x['replayable'] for x in frozen)
put('HISTORICAL_V1_MEMBERSHIP_REPLAY.json',{'status':'PASS','parent_n':173,'eligible_n':167,'ineligible_n':6,'eligible_ids':sorted(old),'authority_eligible_ids':sorted(auth),'membership_diff':[],'frozen_trajectory_parity':'PASS','rows':[dict(x,eligible=v1(x['action_sequence'])[0],details=v1(x['action_sequence'])[1]) for x in parent]})
# The historical gate above precedes any V2 implementation/configuration.
candidate_source='''"""Candidate morphology screening only. No execution of interventions."""
from collections import Counter
def screen(sequence, actions):
    n=len(sequence)
    if not n or any(a not in actions for a in sequence):
        raise ValueError("EMPTY_OR_UNKNOWN_ACTION")
    switches=[i for i in range(1,n) if sequence[i]!=sequence[i-1]]
    s=switches[0]+1 if switches else None
    limit=max(2,n//4)
    tail=sequence[n//2:]
    counts=Counter(tail)
    modal=max(actions,key=lambda a:(counts[a],-actions.index(a)))
    fraction=counts[modal]/max(len(tail),1)
    return {"eligible":s is not None and s<=limit and fraction>=0.8,
            "N":n,"first_switch":s,"first_fraction":s/n if s else None,
            "early_limit":limit,"late_length":len(tail),"late_modal":modal,
            "late_count":counts[modal],"late_fraction":fraction}
def complete_chunks(token_count,chunk_size):
    if token_count<0 or chunk_size<=0: raise ValueError("INVALID_GEOMETRY")
    return token_count//chunk_size
'''
cp=S/'candidate.py'
if cp.exists(): assert cp.read_text()==candidate_source
else: cp.write_text(candidate_source)
cns={};exec(compile(candidate_source,str(cp),'exec'),cns);v2=cns['screen']
config={'STATUS':'CANDIDATE_NOT_FORMAL_4B_FROZEN','CANDIDATE_ONLY':True,'FORMAL_4B_RESET_ELIGIBILITY_V2_FROZEN':False,'human_form':'max(2, floor(0.25*N))','native_form':'s=switches[0]+1; s is not None and s<=max(2,N//4); equivalently i<=max(1,N//4-1)','N':'number of complete decision chunks; partial tail excluded','late_half':'sequence[N//2:]','late_modal_threshold':0.8,'modal_tie_order_historical':ACTIONS,'modal_tie_order_4b':ACTIONS[:-1]+['L30','ALL'],'INTERVENTION_OUTCOME_USED':False,'specified_by':'user request before sensitivity analysis','implementation_path':str(cp),'implementation_sha256':sha(cp)}
configpath=R/'configs/reset_eligibility_resolution_aware_v2_candidate.json'
encoded=json.dumps(config,indent=2,sort_keys=True)+'\n'
if configpath.exists(): assert configpath.read_text()==encoded
else: configpath.write_text(encoded)
new={x['sample_id'] for x in parent if v2(x['action_sequence'],ACTIONS)['eligible']}
changed=sorted(old^new)
retro={'status':'EXACT_PASS' if not changed else 'FAIL','parent_n':173,'V1_ELIGIBLE_N':len(old),'V1_INELIGIBLE_N':173-len(old),'V2_ELIGIBLE_N':len(new),'V2_INELIGIBLE_N':173-len(new),'V1_V2_MEMBERSHIP_DIFF_N':len(changed),'V1_V2_CHANGED_SAMPLE_IDS':changed,'eligible_ids':sorted(new),'changed_details':[dict(sample_id=s,V1=v1(byid[s]['action_sequence']),V2=v2(byid[s]['action_sequence'],ACTIONS)) for s in changed],'historical_N_distribution':dict(collections.Counter(x['chunk_count'] for x in parent)),'result_note':'Historical 1.7B intervention results remain applicable to the same cohort because eligibility membership did not change.' if not changed else 'Membership differs; do not accept candidate.'}
put('HISTORICAL_1P7B_V1_V2_RETROSPECTIVE.json',retro)
minimal=[]
for n in range(2,33):
    p1=[s for s in range(2,n+1) if s/n<=.25];p2=[s for s in range(2,n+1) if s<=max(2,n//4)]
    assert (p1==p2 if p1 else p2==[2])
    minimal.append({'N':n,'V1_limit':n//4,'V2_limit':max(2,n//4),'limit_difference':max(2,n//4)-n//4,'V1_max_legal_position':max(p1) if p1 else None,'V2_max_legal_position':max(p2),'V1_positions':p1,'V2_positions':p2,'added_positions':sorted(set(p2)-set(p1)),'repairs_empty_set':not p1 and bool(p2)})
put('RESOLUTION_AWARE_AMENDMENT_MINIMALITY.json',{'status':'PASS','rows':minimal,'proof':'For N>=8 floor(N/4)>=2 so V2=V1; for 2<=N<=7 V1 has no legal switch and V2 permits exactly s=2. N<2 cannot switch.'})
put('SHORT_GRID_EARLY_FEASIBILITY.json',{'status':'PASS','rows':[dict(x,all_possible_switch_positions=list(range(2,x['N']+1)),V1_legal_count=len(x['V1_positions']),V2_legal_count=len(x['V2_positions'])) for x in minimal if x['N'] in [6,7,8]]})
late=[]
for n in [6,7,8,15]:
    m=n-n//2;k=next(k for k in range(m+1) if k/m>=.8)
    late.append({'N':n,'late_length':m,'required_count':k,'effective_purity':k/m})
put('LATE_MODAL_80_PERCENT_DISCRETIZATION_AUDIT.json',{'status':'PASS','LATE_MODAL_RULE_CHANGED':False,'LATE_MODAL_SHORT_GRID_WARNING':'YES','rows':late})
def shapes(n,seq=(0,)):
    if len(seq)==n:yield seq;return
    for v in range(max(seq)+2):yield from shapes(n,seq+(v,))
combined=[]
for n in [6,7,8]:
    total=passed=labeled=0;example=None
    for seq in shapes(n):
        total+=1
        if v2(seq,list(range(8)))['eligible']:
            passed+=1;k=max(seq)+1;labeled+=math.factorial(8)//math.factorial(8-k)
            if example is None:example=list(seq)
    assert passed>0
    combined.append({'N':n,'total_equality_shapes':total,'eligible_equality_shapes':passed,'eligible_labeled_sequences_8_actions':labeled,'example':example,'feasible':True})
put('COMBINED_SHORT_GRID_FEASIBILITY.json',{'status':'PASS','definition':'A shape is an action equality pattern modulo bijective action relabeling, represented by a restricted-growth string starting 0. Up to 8 distinct actions; exhaustive for N<=8. No score/model. Labeled count weights each k-action shape by 8!/(8-k)!.','rows':combined})
debug=[]
for p in sorted((R/'work/qwen3_4b_downstream_pipeline_debug_v1/dynamic/samples').glob('*.json')):
    x=read(p,'project_json',['sample_id','task','action_sequence','chunk_count','chunk_size','DEBUG_ONLY','PAPER_RESULT','completed'])
    assert x['DEBUG_ONLY'] is True and x['PAPER_RESULT'] is False and x['completed'] and x['chunk_size']==4096 and len(x['action_sequence'])==x['chunk_count']
    actions=ACTIONS[:-1]+['L30','ALL'];d=v2(x['action_sequence'],actions)
    # Historical semantic rule on 4B action domain; no source mutation.
    orig=d['first_switch'] is not None and d['first_fraction']<=.25 and d['late_fraction']>=.8
    debug.append(dict(x,morphology=d,V1_eligible=orig,V2_eligible=d['eligible']))
assert len(debug)==2
put('REAL_4B_DEBUG_MORPHOLOGY.json',{'DEBUG_ONLY':True,'PAPER_RESULT':False,'FORMAL_SCIENTIFIC_RESULT':False,'rows':debug})
sensitivity=[]
for label,t in [('20%',.20),('25%',.25),('30%',.30),('1/3',1/3),('40%',.40),('50%',.50)]:
    selected={x['sample_id'] for x in parent if (d:=v1(x['action_sequence'])[1])['first_switch_position_normalized'] is not None and d['first_switch_position_normalized']<=t and d['steady_tail_fraction']>=.8}
    sensitivity.append({'criterion':label,'eligible_n':len(selected),'membership_diff_n':len(selected^old),'changed_ids':sorted(selected^old)})
sensitivity.append({'criterion':'resolution-aware V2','eligible_n':len(new),'membership_diff_n':len(changed),'changed_ids':changed})
put('THRESHOLD_SENSITIVITY_DESCRIPTIVE_ONLY.json',{'DESCRIPTIVE_ONLY':True,'USED_TO_SELECT_RULE':False,'candidate_config_sha256_before_analysis':sha(configpath),'rows':sensitivity})
put('FIXED_50_PERCENT_COMPARISON.json',{'DESCRIPTIVE_ONLY':True,'membership_diff_n':sensitivity[-2]['membership_diff_n'],'changed_ids':sensitivity[-2]['changed_ids'],'rows':[{'N':n,'V1_max_position':n//4,'fixed_50_max_position':n//2} for n in sorted({x['chunk_count'] for x in parent})],'note':'Fixed 50% expands already feasible historical early regions; V2 only repairs empty early regions.'})
tests=[]
def check(name,condition):
    assert condition,name
    tests.append({'name':name,'status':'PASS'})
for n in [6,7,8,12,13,14,15,16]:
    one=['OFF']+['L0']*(n-1)
    check('earliest_switch_N'+str(n),v2(one,ACTIONS)['eligible'] and v1(one)[0]==(n>=8))
    check('no_switch_N'+str(n),not v2(['OFF']*n,ACTIONS)['eligible'])
    for s in range(2,n+1):
        seq=['OFF']*(s-1)+['L0']*(n-s+1);d=v2(seq,ACTIONS)
        check('index_N%s_s%s'%(n,s),d['first_switch']==s and d['first_fraction']==s/n)
check('exact_25_percent',v1(['OFF']+['L0']*7)[0])
check('just_after_25',not v2(['OFF']*2+['L0']*5,ACTIONS)['eligible'])
seq=['OFF','L0','L12']+['L0']*7
check('multiple_switches_first_not_last',v2(seq,ACTIONS)['first_switch']==2 and v2(seq,ACTIONS)['eligible'])
seq=['OFF']+['L0']*4+['L0']*4+['L12']
check('late_exact_80',v2(seq,ACTIONS)['late_fraction']==.8 and v2(seq,ACTIONS)['eligible'] and v1(seq)[0])
seq=['OFF']+['L0']*4+['L0']*3+['L12']*2
check('late_nearest_below_80',v2(seq,ACTIONS)['late_fraction']==.6 and not v2(seq,ACTIONS)['eligible'] and not v1(seq)[0])
for n in range(2,33):
    check('late_odd_even_'+str(n),v2(['OFF']+['L0']*(n-1),ACTIONS)['late_length']==n-n//2)
    for c in [1024,4096]:
        check('tail_exclusion_%s_%s'%(n,c),all(cns['complete_chunks'](n*c+t,c)==n for t in [0,1,c-1]))
seq=['L0','OFF','L0','OFF']
check('modal_tie_OFF_first',v2(seq,ACTIONS)['late_modal']=='OFF' and v1(seq)[1]['steady_tail_action']=='OFF')
check('empty_rejected',not v1([])[0])
for seq in [[],['UNKNOWN']]:
    try:v2(seq,ACTIONS)
    except ValueError:check('candidate_invalid_'+str(seq),True)
    else:raise AssertionError('invalid input accepted')
check('all_historical_late_metadata_unchanged',all(v2(x['action_sequence'],ACTIONS)['late_fraction']==v1(x['action_sequence'])[1]['steady_tail_fraction'] and v2(x['action_sequence'],ACTIONS)['late_modal']==v1(x['action_sequence'])[1]['steady_tail_action'] for x in parent))
put('RESET_ELIGIBILITY_V2_UNIT_TESTS.json',{'status':'PASS','test_count':len(tests),'tests':tests})
check('immutable_after',all(sha(pathlib.Path(p))==h for p,h in before.items()) and sha(archive)==EXPECTED)
put('MORPHOLOGY_INPUT_PROJECTION.json',{'parent':parent,'authority_eligible_ids':sorted(auth),'debug':debug,'note':'Only projected morphology and membership fields. No intervention outcome files opened semantically. Full release files hashed as opaque bytes solely for immutability.'})
put('INPUT_ACCESS_LOG.json',{'semantic_accesses':[x for x in access if x['mode']!='hash_only'],'hash_only_accesses':[x for x in access if x['mode']=='hash_only'],'discovery_before_script':'Source static read and cohort/debug schema keys were inspected; only morphology/ID values projected. No intervention result values displayed or selected. Package opaque SHA checked.','forbidden_outcome_fields_projected':[]})
put('EXECUTION_ATTESTATION.json',{'HOST':socket.gethostname(),'USER':getpass.getuser(),'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'GPU_USED':False,'MODEL_FORWARD_EXECUTED':False,'RESET_EXECUTED':False,'KP_EXECUTED':False,'DELETION_EXECUTED':False,'DYNAMIC_EXECUTED':False,'R2_EXECUTED':False,'INTERVENTION_OUTCOME_USED_FOR_RULE_SELECTION':False,'loaded_modules':sorted(sys.modules),'execution_scope':'stdlib CPU morphology screening only; no device-query utility used','audit_script':str(S/'audit.py')})
print(json.dumps({'historical':{k:v for k,v in retro.items() if k not in ['eligible_ids','changed_details']},'combined':combined,'sensitivity':sensitivity,'tests':len(tests),'release_files_verified':len(before)},indent=2))

