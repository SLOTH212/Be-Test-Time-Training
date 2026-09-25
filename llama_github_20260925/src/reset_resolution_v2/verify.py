"""Independent CPU read-only verifier. Emits JSON to stdout; never writes files."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import ast
import collections
import hashlib
import itertools
import json
import pathlib
import sys
R=pathlib.Path('/path/to/ttt');P=R/'releases/ttt_mechanism_runtime_v1';A=R/'audits/reset_resolution_v2'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def doc(n):return json.loads((A/n).read_text())
checks={}
def require(name,value):
    checks[name]='PASS' if value else 'FAIL'
    assert value,name
base=doc('IMMUTABLE_BASELINE.json')
require('all_frozen_files_unchanged',all(sha(pathlib.Path(p))==h for p,h in base['files'].items()))
require('package_unchanged',sha(R/'packages/ttt_mechanism_runtime_v1.tar.gz')==base['package_sha256']=='ce9ee7c7066d0413e4fcf90b69543cfa605ac1761af8d849ddb2d61ccd6fb1a8')
# This implementation derives the switch from the first run length, uses integer
# inequalities, and computes modal counts without importing candidate code.
def independent(seq,version):
    n=len(seq)
    first_run=0
    for action in seq:
        if action!=seq[0]:break
        first_run+=1
    s=first_run+1 if first_run<n else None
    tail=seq[n//2:];counts={}
    for action in tail:counts[action]=counts.get(action,0)+1
    k=max(counts.values(),default=0);m=len(tail)
    early=False if s is None else (4*s<=n if version==1 else (s==2 or 4*s<=n))
    return bool(early and m and 5*k>=4*m),s,k,m
parent=[]
for line in (P/'provenance/IMPROVED_173_PARENT_COHORT.jsonl').read_text().splitlines():
    raw=json.loads(line);parent.append({k:raw[k] for k in ['sample_id','action_sequence','chunk_count','task','completed']})
authority={json.loads(l)['canonical_sample_id'] for l in (P/'provenance/original/reset/manifests/STATE_RESET_167_COHORT.jsonl').read_text().splitlines()}
e1={x['sample_id'] for x in parent if independent(x['action_sequence'],1)[0]}
e2={x['sample_id'] for x in parent if independent(x['action_sequence'],2)[0]}
require('historical_exact_V1_membership',len(parent)==173 and len(e1)==167 and e1==authority)
require('historical_exact_V2_membership',e2==e1 and len(e2)==167)
replay=doc('HISTORICAL_V1_MEMBERSHIP_REPLAY.json');retro=doc('HISTORICAL_1P7B_V1_V2_RETROSPECTIVE.json')
require('reported_replays_match',set(replay['eligible_ids'])==e1 and set(retro['eligible_ids'])==e2 and retro['V1_V2_MEMBERSHIP_DIFF_N']==0)
candidate_path=R/'src/reset_resolution_v2/candidate.py';csrc=candidate_path.read_text();ns={}
require('candidate_imports_only_Counter',all(isinstance(n,ast.ImportFrom) and n.module=='collections' and [a.name for a in n.names]==['Counter'] for n in ast.walk(ast.parse(csrc)) if isinstance(n,(ast.Import,ast.ImportFrom))))
exec(compile(csrc,str(candidate_path),'exec'),ns)
actions=['OFF','L0','L6','L12','L18','L24','ALL']
contract=doc('HISTORICAL_RESET_V1_CONTRACT.json')
source=pathlib.Path(contract['source']);node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='historical_early_steady')
native={'Counter':collections.Counter,'ACTIONS':actions}
exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),native)
require('source_contract_exact',ast.get_source_segment(source.read_text(),node)==contract['exact_function_source'] and sha(source)==contract['source_sha256'])
tests=0
for n in range(2,9):
    for seq in itertools.product(actions[:3],repeat=n):
        a=independent(seq,2);b=ns['screen'](seq,actions);c,d=native['historical_early_steady'](seq)
        assert b['eligible']==a[0] and b['first_switch']==a[1] and b['late_count']==a[2] and b['late_length']==a[3]
        assert c==independent(seq,1)[0]
        assert b['late_fraction']==d['steady_tail_fraction'] and b['late_modal']==d['steady_tail_action']
        tests+=1
for x in parent:
    require('historical_native_parity_'+x['sample_id'],ns['screen'](x['action_sequence'],actions)['eligible']==independent(x['action_sequence'],2)[0])
require('exhaustive_candidate_and_unchanged_late_rule',tests==9837)
configpath=R/'configs/reset_eligibility_resolution_aware_v2_candidate.json';config=json.loads(configpath.read_text())
require('candidate_config_exact',config['human_form']=='max(2, floor(0.25*N))' and config['implementation_sha256']==sha(candidate_path) and config['late_modal_threshold']==.8 and config['late_half']=='sequence[N//2:]' and config['FORMAL_4B_RESET_ELIGIBILITY_V2_FROZEN'] is False)
for row in doc('RESOLUTION_AWARE_AMENDMENT_MINIMALITY.json')['rows']:
    n=row['N'];p1=[s for s in range(2,n+1) if 4*s<=n];p2=[s for s in range(2,n+1) if s==2 or 4*s<=n]
    require('minimality_N'+str(n),row['V1_positions']==p1 and row['V2_positions']==p2 and (p1==p2 if p1 else p2==[2]))
for row in doc('LATE_MODAL_80_PERCENT_DISCRETIZATION_AUDIT.json')['rows']:
    n=row['N'];m=(n+1)//2;k=(4*m+4)//5
    require('discrete_late_N'+str(n),row['late_length']==m and row['required_count']==k and row['effective_purity']==k/m)
combined=[]
for row in doc('COMBINED_SHORT_GRID_FEASIBILITY.json')['rows']:
    n=row['N'];prefix=n//2;canonical=set();labeled=0
    # Independent complete enumeration: the late half MUST be constant for
    # these N. Enumerate every labeled prefix and the constant tail action.
    for parts in itertools.product(range(8),repeat=prefix+1):
        seq=parts[:prefix]+(parts[-1],)*(n-prefix)
        if not independent(seq,2)[0]:continue
        labeled+=1;mapping={};shape=[]
        for a in seq:
            if a not in mapping:mapping[a]=len(mapping)
            shape.append(mapping[a])
        canonical.add(tuple(shape))
    require('independent_combined_N'+str(n),len(canonical)==row['eligible_equality_shapes'] and labeled==row['eligible_labeled_sequences_8_actions'] and labeled>0)
    combined.append({'N':n,'shape_count':len(canonical),'labeled_count':labeled})
for row in doc('REAL_4B_DEBUG_MORPHOLOGY.json')['rows']:
    v1=independent(row['action_sequence'],1);v2=independent(row['action_sequence'],2)
    require('debug_'+row['sample_id'],row['V1_eligible']==v1[0] and row['V2_eligible']==v2[0] and row['DEBUG_ONLY'] and not row['PAPER_RESULT'])
fractions={'20%':(1,5),'25%':(1,4),'30%':(3,10),'1/3':(1,3),'40%':(2,5),'50%':(1,2)}
for row in doc('THRESHOLD_SENSITIVITY_DESCRIPTIVE_ONLY.json')['rows']:
    if row['criterion']=='resolution-aware V2':selected=e2
    else:
        num,den=fractions[row['criterion']];selected=set()
        for x in parent:
            _,s,k,m=independent(x['action_sequence'],1)
            if s is not None and s*den<=len(x['action_sequence'])*num and 5*k>=4*m:selected.add(x['sample_id'])
    require('sensitivity_'+row['criterion'],len(selected)==row['eligible_n'] and sorted(selected^e1)==row['changed_ids'])
log=doc('INPUT_ACCESS_LOG.json')
permitted={str(source):None,str(P/'provenance/RELEASE_MANIFEST.json'):None,str(P/'provenance/MECHANISM_SOURCE_AUTHORITY.json'):None,str(R/'vendor/frozen/ttt_inference_runtime_v1/dynamic/formal_dynamic_runner.py'):None,str(P/'provenance/IMPROVED_173_PARENT_COHORT.jsonl'):{'sample_id','task','action_sequence','chunk_count','completed'},str(P/'provenance/original/reset/manifests/STATE_RESET_167_COHORT.jsonl'):{'canonical_sample_id','original_dynamic_actions','full_chunk_count','sequence_length','replayable'}}
for p in (R/'work/qwen3_4b_downstream_pipeline_debug_v1/dynamic/samples').glob('*.json'):
    permitted[str(p)]={'sample_id','task','action_sequence','chunk_count','chunk_size','DEBUG_ONLY','PAPER_RESULT','completed'}
for entry in log['semantic_accesses']:
    require('allowlisted_'+pathlib.Path(entry['path']).name,entry['path'] in permitted and (permitted[entry['path']] is None or set(entry['fields'])==permitted[entry['path']]) and sha(pathlib.Path(entry['path']))==entry['sha256'])
att=doc('EXECUTION_ATTESTATION.json')
for field in ['GPU_USED','MODEL_FORWARD_EXECUTED','RESET_EXECUTED','KP_EXECUTED','DELETION_EXECUTED','DYNAMIC_EXECUTED','R2_EXECUTED','INTERVENTION_OUTCOME_USED_FOR_RULE_SELECTION']:
    require(field+'_false',att[field] is False)
require('cuda_hidden',att['CUDA_VISIBLE_DEVICES']=='' and os.environ['CUDA_VISIBLE_DEVICES']=='')
require('no_model_imports',not any(m.split('.')[0] in {'torch','transformers','tensorflow','jax','cupy'} for m in att['loaded_modules']+list(sys.modules)))
audit_tree=ast.parse((R/'src/reset_resolution_v2/audit.py').read_text())
allowed_imports={'os','ast','collections','hashlib','itertools','json','math','pathlib','socket','getpass','sys'}
imports={alias.name.split('.')[0] for n in ast.walk(audit_tree) if isinstance(n,ast.Import) for alias in n.names}
require('audit_stdlib_only',imports<=allowed_imports and not any(isinstance(n,ast.ImportFrom) and n.module not in allowed_imports for n in ast.walk(audit_tree)))
require('no_intervention_semantic_input',not log['forbidden_outcome_fields_projected'])
require('unit_tests_pass',doc('RESET_ELIGIBILITY_V2_UNIT_TESTS.json')['status']=='PASS')
require('frozen_files_still_unchanged',all(sha(pathlib.Path(p))==h for p,h in base['files'].items()))
print(json.dumps({'status':'PASS','implementation':'Independent first-run length and integer cross-multiplication; independent labeled enumeration for combined geometry; original frozen function invoked only for morphology.','read_only':True,'check_count':len(checks),'exhaustive_sequences':tests,'combined_recomputed':combined,'checks':checks,'verifier_path':str(R/'src/reset_resolution_v2/verify.py'),'verifier_sha256':sha(R/'src/reset_resolution_v2/verify.py'),'GPU_USED':False,'evidence_scope':'Audited scripts, import inventory, CUDA-hidden environment, and allowlisted morphology projections; does not assert machine-wide absence of other users GPU jobs.'},indent=2,sort_keys=True))

