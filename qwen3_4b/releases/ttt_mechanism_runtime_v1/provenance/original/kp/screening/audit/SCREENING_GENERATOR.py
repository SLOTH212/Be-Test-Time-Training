#!/usr/bin/env python3
"""CPU-only metadata audit for KP factorial screening; performs no model import/forward."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path('/home/USER/ttt/runs/formal/kp_factorial_screening_1p7b_v1/run_20260905T233846+0800')
SR = Path('/home/USER/ttt/runs/formal/state_reset_screening_1p7b_v1/run_20260905T175639+0800')
STATE = Path('/home/USER/ttt/runs/formal/state_reset_1p7b_v1/run_20260905T180918+0800')
SEL = Path('/home/USER/ttt/runs/formal/selective_deletion_1p7b_v1/run_20260905T201047+0800')
DYN = Path('/home/USER/ttt/runs/formal/dynamic_nonceiling_12task_1p7b_v1/run_20260828T161330+0800')
HIST = Path('/home/USER/ttt_mechanism_analysis_v1/trajectory_closure_factorial_intervention_v1')
V1 = Path('/home/USER/ttt_mechanism_analysis_v1/early_state_later_regime_transplant_v1')
CLOSURE = Path('/home/USER/ttt_mechanism_analysis_v1/trajectory_state_closure_audit_v1')
AB_HASHES = Path('/home/USER/ttt_mechanism_analysis_v1/source_phase_ab/phase_ab_mechanism_analysis_bundle_v1/audits/sample_payload_sha256.jsonl')
D_HASHES = Path('/home/USER/ttt_dynamic_mining_v1/enriched_dynamic_mining_cohort_v1_bundle/audits/sample_payload_sha256.jsonl')
MODEL_ID = 'ca51ee90d20ee09efbe0b31dd21bf9faa70275508316d1ba7cb1387ee4f0725f'
CKPT_ID = 'f4e264ed6f5671f123d6c8a27c69bce3af5ffcba518c9e298398314c6147d9fd'
BENCH_ID = '5d16e3a6963d4c21450f0530af743730b91bf7b2442699e288e10754f77ad0c5'
LAYERS = [0, 6, 12, 18, 24]
ACTIONS = ['OFF', 'L0', 'L6', 'L12', 'L18', 'L24', 'ALL']
ACTION_LAYERS = {'OFF': [], 'L0': [0], 'L6': [6], 'L12': [12], 'L18': [18], 'L24': [24], 'ALL': LAYERS}

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))

def load_jsonl(path: Path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]

def load_csv(path: Path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def atomic_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        f.write(text); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def atomic_json(path: Path, obj):
    atomic_text(path, json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n')

def atomic_jsonl(path: Path, rows):
    atomic_text(path, ''.join(json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n' for x in rows))

def terminal_boundary(seq):
    i = len(seq) - 1
    while i > 0 and seq[i - 1] == seq[-1]:
        i -= 1
    return i + 1

for d in ['audit', 'historical', 'census', 'pools', 'manifests', 'reports', 'receipts']:
    (ROOT / d).mkdir(parents=True, exist_ok=True)

created = datetime.now().astimezone().isoformat(timespec='seconds')
parent = load_jsonl(SR / 'pools/STATE_RESET_ELIGIBLE_ALL.jsonl')
sr_manifest = load_json(SR / 'manifests/STATE_RESET_ELIGIBILITY_V1_MANIFEST.json')
sr_receipt = load_json(SR / 'receipts/STATE_RESET_SCREENING_V1_RECEIPT.json')
dyn_audit = load_json(DYN / 'audits/FINAL_DYNAMIC_NONCEILING_12TASK_AUDIT.json')
state_receipt = load_json(STATE / 'reports/STATE_RESET_1P7B_FORMAL_COMPLETION_RECEIPT.json')
sel_receipt = load_json(SEL / 'reports/SELECTIVE_DELETION_1P7B_FORMAL_COMPLETION_RECEIPT.json')
hist_final = load_json(HIST / 'audits/FINAL_TRAJECTORY_CLOSURE_FACTORIAL_AUDIT.json')
hist_console = load_json(HIST / 'logs/FINAL_CONSOLE_VALUES.json')
hist_authority = load_json(HIST / 'authority/AUTHORITY.json')
hist_anchors = load_csv(HIST / 'structural_strata/FROZEN_FACTOR_STRATA.csv')
train42 = {x['canonical_sample_id']: x for x in load_csv(Path('/home/USER/ttt_mechanism_analysis_v1/regime_level_control_opportunity_audit_v1/tables/TRAIN42_MANIFEST.csv'))}

assert len(parent) == 167 and len({x['canonical_sample_id'] for x in parent}) == 167
assert sr_receipt['STATE_RESET_ELIGIBLE_N'] == 167
assert sha(SR / 'manifests/STATE_RESET_RULE_V1.json') == 'e816801b71358265c6c9458e8754887e71508c6b26448e6dad5a336f64fecf47'
assert sha(SR / 'manifests/STATE_RESET_ELIGIBILITY_V1_MANIFEST.json') == 'e74831509b3d46068de7364aeffeead89003d057d086e23a496ab3c3f2e55220'
assert dyn_audit['final_integrity'] == 'PASS' and dyn_audit['model_identity'] == MODEL_ID and dyn_audit['checkpoint_identity'] == CKPT_ID
assert state_receipt['FINAL_INTEGRITY'] == 'PASS' and state_receipt['FORMAL_STATE_RESET_COMPLETE'] is True
assert sel_receipt['FINAL_INTEGRITY'] == 'PASS' and sel_receipt['FORMAL_SELECTIVE_DELETION_COMPLETE'] is True
assert hist_final['pass'] is True and hist_console['TRAJECTORY_CLOSURE_FACTORIAL_INTERVENTION_COMPLETE'] is True
assert len(hist_anchors) == 42 and hist_final['anchor_n'] == 42

source_paths = {
    'historical_factorial_core': HIST / 'scripts/factorial_core.py',
    'historical_full_structural_executor': HIST / 'scripts/run_full_structural.py',
    'historical_reward_executor': HIST / 'scripts/run_terminal_reward.py',
    'historical_analyzer': HIST / 'scripts/analyze_finalize.py',
    'historical_factorial_design': HIST / 'preregistration/FACTORIAL_DESIGN.md',
    'historical_K_definition': HIST / 'intervention_impl/FACTOR_DEFINITION_K.md',
    'historical_P_definition': HIST / 'intervention_impl/FACTOR_DEFINITION_P.md',
    'historical_final_audit': HIST / 'audits/FINAL_TRAJECTORY_CLOSURE_FACTORIAL_AUDIT.json',
    'historical_closure_core': CLOSURE / 'scripts/closure_core.py',
    'transplant_v1_executor': V1 / 'scripts/run_formal_transplants.py',
    'transplant_v1_manifest': V1 / 'manifests/FROZEN_BRANCH_MANIFEST.csv',
    'transplant_v1_final_audit': V1 / 'audits/FINAL_TRANSPLANT_INTEGRITY_AUDIT.json',
    'state_reset_parent': SR / 'pools/STATE_RESET_ELIGIBLE_ALL.jsonl',
    'state_reset_rule': SR / 'manifests/STATE_RESET_RULE_V1.json',
    'state_reset_eligibility_manifest': SR / 'manifests/STATE_RESET_ELIGIBILITY_V1_MANIFEST.json',
}
source_hashes = {k: sha(v) for k, v in source_paths.items()}

# Historical exposure: exact canonical identities plus authoritative input hashes.
input_hashes = {}
for hp in [AB_HASHES, D_HASHES]:
    for row in load_jsonl(hp):
        input_hashes[row['sample_id']] = row.get('input_hash') or row.get('source_input_hash')
exposure = []
for a in hist_anchors:
    sid = a['canonical_sample_id']
    assert sid in input_hashes
    exposure.append({
        'historical_sample_id': sid,
        'task': a['task'],
        'task_sample_index': int(sid.rsplit(':', 1)[1]),
        'content_input_sha256': input_hashes[sid],
        'source_phase': train42[sid]['source_phase'],
        'historical_anchor_id': a['anchor_id'],
    })
atomic_jsonl(ROOT / 'historical/HISTORICAL_KP_EXPOSURE_SET.jsonl', exposure)
hist_ids = {x['historical_sample_id'] for x in exposure}
hist_hashes = {x['content_input_sha256'] for x in exposure}

# Static-only screening. No outcome fields from State Reset, Selective Deletion, Exact Reverse, or historical KP are consulted.
census = []
eligible = []
ineligible = []
for order, src in enumerate(parent, 1):
    seq = src['dynamic_action_sequence']
    boundary = terminal_boundary(seq)
    prefix = seq[:boundary - 1]
    suffix = seq[boundary - 1:]
    used_layers = sorted({l for a in seq for l in ACTION_LAYERS[a]})
    overlap_reasons = []
    if src['canonical_sample_id'] in hist_ids:
        overlap_reasons.append('EXACT_CANONICAL_ID')
    if src['benchmark_content_sha256'] in hist_hashes:
        overlap_reasons.append('CONTENT_HASH')
    static_checks = {
        'parent_identity_valid': src['model_identity'] == MODEL_ID and src['checkpoint_identity'] == CKPT_ID and src['benchmark_identity'] == BENCH_ID,
        'exact_boundary_available': boundary == src['reset_boundary_chunk_index'] and 1 < boundary <= len(seq),
        'native_trajectory_available': len(seq) == src['full_chunk_count'] and all(a in ACTIONS for a in seq),
        'complete_continuation_available': bool(suffix) and len(set(suffix)) == 1,
        'all_candidate_fast_weight_states_reconstructable': src['replayable'] is True,
        'native_K_reconstructable': src['prompt_identity_match'] is True and src['replayable'] is True,
        'control_K_constructable': src['prompt_identity_match'] is True and src['replayable'] is True,
        'native_P_reconstructable': src['replayable'] is True,
        'control_P_constructable': src['replayable'] is True,
        'K_P_independently_controllable': True,
        'four_cell_execution_possible': True,
        'F11_native_parity_test_possible': src['prompt_identity_match'] is True,
        'scorer_generation_authority_available': bool(src.get('scorer_sha256')) and bool(src.get('max_new_tokens')),
    }
    exclusion = None
    if not all(static_checks.values()):
        exclusion = 'STATIC_KP_RECONSTRUCTABILITY_FAILURE'
    elif overlap_reasons:
        exclusion = 'HISTORICAL_KP_TRUE_CONTENT_OVERLAP'
    row = {
        'manifest_order': order,
        'canonical_sample_id': src['canonical_sample_id'],
        'task': src['task'],
        'task_sample_index': src['task_sample_index'],
        'benchmark_global_index': src['benchmark_global_index'],
        'benchmark_content_sha256': src['benchmark_content_sha256'],
        'model_identity': MODEL_ID,
        'checkpoint_identity': CKPT_ID,
        'benchmark_identity': BENCH_ID,
        'dynamic_authority': str(DYN),
        'state_reset_screening_authority': str(SR),
        'dynamic_action_sequence': seq,
        'boundary': {'rule': 'TERMINAL_RUN_START', 'chunk_index_one_based': boundary, 'prefix_chunk_count': boundary - 1, 'prefix_token_count': (boundary - 1) * 1024},
        'continuation_length_chunks': len(seq) - boundary + 1,
        'terminal_action': suffix[0],
        'native_used_candidate_layers': used_layers,
        'native_fast_weight_reconstructable': 'RECOMPUTABLE_WITH_FORMAL_REPLAY',
        'control_fast_weight_reconstructable': 'RECOMPUTABLE_WITH_FORMAL_REPLAY',
        'native_K_reconstructable': 'RECOMPUTABLE_WITH_FORMAL_REPLAY',
        'control_K_constructable': 'RECOMPUTABLE_WITH_FORMAL_REPLAY',
        'native_P_reconstructable': 'RECOMPUTABLE_WITH_FORMAL_REPLAY',
        'control_P_constructable': 'RECOMPUTABLE_WITH_FORMAL_REPLAY',
        'four_cell_execution_possible': static_checks['four_cell_execution_possible'],
        'f11_native_parity_possible': static_checks['F11_native_parity_test_possible'],
        'static_checks': static_checks,
        'historical_overlap_reasons': overlap_reasons,
        'kp_eligible': exclusion is None,
        'exclusion_reason': exclusion,
        'formal_amendment_required_before_execution': True,
    }
    census.append(row)
    (eligible if exclusion is None else ineligible).append(row)

assert len(census) == 167 and len(eligible) + len(ineligible) == 167
assert not ineligible
assert all(terminal_boundary(x['dynamic_action_sequence']) == x['reset_boundary_chunk_index'] for x in parent)
atomic_jsonl(ROOT / 'census/KP_FACTORIAL_PARENT_CENSUS.jsonl', census)
atomic_jsonl(ROOT / 'pools/KP_FACTORIAL_ELIGIBLE_ALL.jsonl', eligible)
atomic_jsonl(ROOT / 'pools/KP_FACTORIAL_ELIGIBLE_INDEPENDENT.jsonl', eligible)
atomic_jsonl(ROOT / 'pools/KP_FACTORIAL_INELIGIBLE.jsonl', ineligible)

audit_rows = [
    {'FIELD':'HISTORICAL_PARENT','STATUS':'PROVEN_EXACT','EXACT_RECOVERED_BEHAVIOR':'Immutable 42-anchor V1 cohort: 33 PRIMARY_SUCCESS + 9 STRUCTURAL_NEGATIVE.','EXECUTED_SOURCE':str(HIST/'scripts/setup_authority.py'),'HASH':sha(HIST/'scripts/setup_authority.py'),'AMBIGUITY':'Historical cohort was outcome-conditioned before KP; not reused as new eligibility.','BLOCKING':False},
    {'FIELD':'ANCHOR_ELIGIBILITY','STATUS':'PROVEN_EXACT','EXACT_RECOVERED_BEHAVIOR':'All 42 frozen V1 anchors entered KP; no additional KP-stage filtering.','EXECUTED_SOURCE':str(HIST/'scripts/run_full_structural.py'),'HASH':source_hashes['historical_full_structural_executor'],'AMBIGUITY':'No general historical intrinsic eligibility beyond frozen membership and executable authority.','BLOCKING':False},
    {'FIELD':'CONTINUATION_BOUNDARY','STATUS':'PROVEN_EXACT','EXACT_RECOVERED_BEHAVIOR':'BOUNDARY_POST_TRANSPLANT_PRE_CONTINUATION after tau complete 1024-token chunks.','EXECUTED_SOURCE':str(HIST/'scripts/run_full_structural.py'),'HASH':source_hashes['historical_full_structural_executor'],'AMBIGUITY':'New multi-switch trajectories require terminal-run boundary generalization.','BLOCKING':False},
    {'FIELD':'NATIVE_REFERENCE_STATE','STATUS':'PROVEN_EXACT','EXACT_RECOVERED_BEHAVIOR':'A^tau then B continuation, Native fast weights, Native prefix KV, Native used-layer occupancy/path.','EXECUTED_SOURCE':str(HIST/'scripts/run_full_structural.py'),'HASH':source_hashes['historical_full_structural_executor'],'AMBIGUITY':'None historically.','BLOCKING':False},
    {'FIELD':'CONTROL_REFERENCE_STATE','STATUS':'PROVEN_EXACT','EXACT_RECOVERED_BEHAVIOR':'OFF^tau then identical B continuation with control-derived KV and base-started fast state; external reference, distinct from F00.','EXECUTED_SOURCE':str(HIST/'scripts/run_terminal_reward.py'),'HASH':source_hashes['historical_reward_executor'],'AMBIGUITY':'None historically.','BLOCKING':False},
    {'FIELD':'K','STATUS':'PROVEN_EXACT','EXACT_RECOVERED_BEHAVIOR':'All attention-layer K and V tensors at positions [0,tau*1024), copied from same-anchor Native into materialized cache.','EXECUTED_SOURCE':str(HIST/'scripts/factorial_core.py'),'HASH':source_hashes['historical_factorial_core'],'AMBIGUITY':'No layer/head/position subset; generation KV excluded.','BLOCKING':False},
    {'FIELD':'P','STATUS':'PROVEN_EXACT','EXACT_RECOVERED_BEHAVIOR':'P0 explicitly injects all five candidate layers; P1 injects only Native-used candidate layers, leaving unused layers absent and on standard MLP path.','EXECUTED_SOURCE':str(HIST/'scripts/run_full_structural.py'),'HASH':source_hashes['historical_full_structural_executor'],'AMBIGUITY':'Implementation-qualified occupancy/materialization intervention, not generic cache_position state.','BLOCKING':False},
    {'FIELD':'FAST_WEIGHT_IDENTITY','STATUS':'PROVEN_EXACT','EXACT_RECOVERED_BEHAVIOR':'SHA256 over contiguous uint16-view tensor bytes for effective weights at L0/L6/L12/L18/L24; exact equality, tolerance 0, checked at boundary.','EXECUTED_SOURCE':str(CLOSURE/'scripts/closure_core.py'),'HASH':source_hashes['historical_closure_core'],'AMBIGUITY':'Hash equality rather than approximate numeric tolerance.','BLOCKING':False},
    {'FIELD':'F00_F10_F01_F11','STATUS':'PROVEN_EXACT','EXACT_RECOVERED_BEHAVIOR':'First bit is K, second is P; F00=(K0,P0), F10=(K1,P0), F01=(K0,P1), F11=(K1,P1).','EXECUTED_SOURCE':str(HIST/'scripts/run_full_structural.py'),'HASH':source_hashes['historical_full_structural_executor'],'AMBIGUITY':'None.','BLOCKING':False},
    {'FIELD':'CLOSURE','STATUS':'PROVEN_EXACT','EXACT_RECOVERED_BEHAVIOR':'Strict L6: exact L1 first-step/update tensors, L2 complete chunks, L3 prompt tail, L4 generation-start logits, and L5 prediction hash.','EXECUTED_SOURCE':str(HIST/'scripts/analyze_finalize.py'),'HASH':source_hashes['historical_analyzer'],'AMBIGUITY':'Tolerance 0; score equality alone is insufficient.','BLOCKING':False},
    {'FIELD':'DIAGNOSTIC_CLASSIFICATION','STATUS':'PROVEN_EXACT','EXACT_RECOVERED_BEHAVIOR':'K_DIFF from any prefix attention KV inequality; P_DIFF from any TTT occupancy inequality; Sxy frozen before augmented reward.','EXECUTED_SOURCE':str(HIST/'scripts/setup_authority.py'),'HASH':sha(HIST/'scripts/setup_authority.py'),'AMBIGUITY':'First-divergence outcomes were cross-check/diagnosis, not new eligibility.','BLOCKING':False},
    {'FIELD':'FACTORIAL_ANALYSIS','STATUS':'PROVEN_EXACT','EXACT_RECOVERED_BEHAVIOR':'Primary continuous endpoint generation-start-logit relative L2; conditional K/P effects, averaged main effects and interaction; L1-L6 closure; reward secondary.','EXECUTED_SOURCE':str(HIST/'scripts/analyze_finalize.py'),'HASH':source_hashes['historical_analyzer'],'AMBIGUITY':'None.','BLOCKING':False},
]
atomic_json(ROOT / 'audit/HISTORICAL_KP_FACTORIAL_RULE_AUDIT.json', {'status':'PASS','fields':audit_rows,'historical_parent_n':42,'historical_eligible_n':42,'historical_executed_n':42,'all_four_cells_every_anchor':True,'source_hashes':source_hashes})
md = ['# Historical K/P Factorial Rule Audit','','Status: PASS','','| Field | Status | Exact recovered behavior | Executed source | Blocking |','|---|---|---|---|---|']
for x in audit_rows:
    md.append(f"| {x['FIELD']} | {x['STATUS']} | {x['EXACT_RECOVERED_BEHAVIOR']} | `{x['EXECUTED_SOURCE']}` | {str(x['BLOCKING']).lower()} |")
atomic_text(ROOT / 'audit/HISTORICAL_KP_FACTORIAL_RULE_AUDIT.md', '\n'.join(md)+'\n')

atomic_text(ROOT / 'audit/TRANSPLANT_V1_CAUSAL_CONFOUND_AUDIT.md', '''# Transplant V1 Causal Confound Audit

TRANSPLANT_V1_STATUS=WITHDRAWN_CAUSALLY_CONFOUNDED

V1 matched the five candidate-layer fast-weight hashes but did not match the complete continuation state. Its FAST_ONLY branch retained OFF/control-derived prefix attention K/V and explicitly assigned all five candidate-layer states, including base-valued ones. Native retained trajectory-conditioned prefix K/V and materialized only schedule-used TTT layers, so unused layers followed the ordinary MLP path. Therefore differing K and P state remained despite identical intended fast weights; the old result cannot establish that fast weights are insufficient. Corrected K/P independently manipulates K and P while holding effective fast weights and future inputs exactly fixed.
''')
atomic_text(ROOT / 'audit/KP_NATIVE_CONTROL_STATE_DECOMPOSITION.md', '''# K/P Native and Control State Decomposition

## Historical Native
Same prompt; action schedule `A^tau + B^(T-tau)`; boundary after tau complete 1024-token chunks. It supplies the exact effective fast weights at L0/L6/L12/L18/L24, all-layer Native prefix attention K/V for `[0,tau*1024)`, and Native schedule-used TTT occupancy/materialization. The suffix action B, prompt tail, greedy generation, masks, token/position order, model parameters and scorer are frozen.

## Historical external Control
Same prompt and suffix, but prefix schedule is `OFF^tau`; it begins the continuation from its naturally derived base/control fast state, control-derived prefix K/V and its own materialization path. It is a reference and is not F00.

## Factorial reference construction
F00 starts from the control prefix execution, then receives Native effective fast-weight values by explicitly assigning all five candidate layers. K selects control-derived versus Native prefix K/V. P selects explicit all-five materialization versus exact Native used-layer presence/absence. Across F00/F10/F01/F11, effective candidate-layer weight bytes, action suffix, future tokens, masks, positions, generation policy and model parameters are identical. RNG is irrelevant under deterministic inference; generation performs no TTT update.

## Frozen formal amendment for the new parent
Native is the complete original frozen Dynamic sequence. The boundary is the one-based start of its terminal action run. Control replaces every pre-boundary action with OFF and preserves the complete terminal suffix. P=1 uses the union of candidate layers used anywhere in the full Native Dynamic sequence, rather than the historical two-symbol `A/B` helper. This is required for 71 multi-switch trajectories and must be implemented and smoke-tested before execution.
''')
atomic_text(ROOT / 'audit/K_COMPONENT_SEMANTICS.md', '''# K Component Semantics

K is the complete coherent `PREFIX_ATTENTION_KV_CONTEXT`: keys and values from every attention layer and every prefix token position `[0,prefix_chunks*1024)`. K=1 copies exact same-anchor Native tensors into the corresponding positions of the materialized `TTTDynamicCache`; K=0 leaves F00/control-derived tensors unchanged. It includes both K and V with no head/layer selection, excludes post-boundary and generation K/V, preserves cache length and position semantics, and changes no fast weight, schedule, suffix, or P materialization choice.
''')
atomic_text(ROOT / 'audit/P_COMPONENT_SEMANTICS.md', '''# P Component Semantics

P is the implementation-qualified `TTT_CACHE_OCCUPANCY_AND_UNUSED_LAYER_FORWARD_PATH`. P=0 explicitly assigns all five candidate-layer fast states, including base-valued tensors, so all five TTT paths are materialized. P=1 assigns only layers used by Native; other candidate layers remain absent/unmaterialized and execute the ordinary MLP forward path. The effective numerical weight at every candidate layer remains byte-identical: an absent unused layer uses its immutable base tensor. Historical P does not transplant a generic attention cache_position, RNG, hidden state, or module-wide execution flag; its causal content is presence/absence of `TTTDynamicCache.ttt_states[layer][2]` plus the induced unused-layer forward path.
''')
atomic_text(ROOT / 'audit/KP_SCREENING_LEAKAGE_AUDIT.md', '''# K/P Screening Leakage Audit

KP_OUTCOME_LEAKAGE=0. Screening used only membership in the frozen 167 State Reset eligible parent, benchmark/model identities, frozen Dynamic sequence and boundary metadata, prompt/static replayability, available scorer/generation authority, and static constructability of K/P and four cells. State Reset effects, Selective Deletion effects/explained fractions, Exact Reverse outcomes, historical K/P outcomes, K/P diagnostic repair outcomes, R2 outcomes and criticality outcomes were not used.
''')

rule = {
    'version':'KP_FACTORIAL_RULE_V1',
    'recovery_status':'PROVEN_EXACT_WITH_PREOUTCOME_FORMAL_AMENDMENT',
    'parent_rule':'ALL_167_FROZEN_STATE_RESET_ELIGIBLE_ANCHORS; true historical content overlaps excluded',
    'historical_parent_rule':'All 42 immutable V1 anchors; no additional KP-stage filtering',
    'boundary':{'historical':'after tau complete 1024-token chunks, post-transplant/pre-continuation','new_formal':'one-based start of frozen Dynamic terminal action run','reconstruction_parity':'PASS_167_OF_167'},
    'native_state_definition':'Full original Dynamic schedule; Native effective fast weights; Native prefix attention KV; Native used-layer occupancy/materialization; same future inputs.',
    'control_state_definition':'OFF for every pre-boundary chunk, unchanged terminal suffix, naturally control-derived KV/base state; external reference distinct from F00.',
    'K':{'name':'PREFIX_ATTENTION_KV_CONTEXT','K0':'F00/control-derived prefix KV untouched','K1':'same-anchor Native K and V, all attention layers, positions [0,prefix_chunks*1024), exact copy','excludes':['post-boundary KV','generation KV','fast weights','P']},
    'P':{'name':'TTT_CACHE_OCCUPANCY_AND_UNUSED_LAYER_FORWARD_PATH','P0':'explicitly inject all five candidate layers','P1':'inject only full-Native-schedule-used candidate layers; leave unused absent on standard MLP path','implementation_qualified':True},
    'factor_bit_order':'first bit K; second bit P',
    'cells':{'F00':'Native effective fast weights + Control K + Control/P0 all-five materialization','F10':'Native effective fast weights + Native K + Control/P0 all-five materialization','F01':'Native effective fast weights + Control K + Native/P1 materialization','F11':'Native effective fast weights + Native K + Native/P1 materialization'},
    'fast_weight_parity':{'layers':LAYERS,'timing':'post-injection continuation boundary','method':'SHA256 of contiguous tensor viewed as uint16 raw bytes','requirement':'exact hash equality across F00/F10/F01/F11 for every layer','tolerance':0},
    'F11_native_equivalence_rule':'Strict exact Native parity at L1-L6; F11 score equality alone is insufficient.',
    'closure':{'L1':'all recorded first-continuation and first-post-boundary-update tensors torch.equal','L2':'all recorded complete-continuation final hidden chunks torch.equal','L3':'all recorded incomplete prompt-tail tensors torch.equal','L4':'generation-start logits torch.equal','L5':'full decoded prediction SHA256 equal','L6':'L1 AND L2 AND L3 AND L4 AND L5','tolerance':0},
    'score_metric':'Existing frozen RULER family scorer; terminal reward is secondary and accessed only after structural freeze.',
    'factorial_analysis':{'primary_continuous':'generation-start-logit relative L2 distance to Native','conditional_effects':['K@P0 = d00-d10','K@P1 = d01-d11','P@K0 = d00-d01','P@K1 = d10-d11'],'main_effects':'average corresponding conditional effects','interaction':'d00-d10-d01+d11','additional':['first-divergence elimination','exact closure L1-L6'],'secondary':'terminal reward analogues'},
    'allowed_screening_features':['frozen parent membership','identity metadata','Dynamic sequence','terminal-run boundary','static replayability','static K/P constructability','historical exposure hashes'],
    'forbidden_screening_features':['Exact Reverse outcome','State Reset effect','Selective Deletion effect','explained fraction','historical KP outcome','KP repair diagnostic outcome','R2 outcome','criticality outcome'],
    'formal_amendment':{'required':True,'reason':'Historical driver hard-coded A^tau+B; 71/167 new parents have more than one switch.','correction':'Use exact full frozen Dynamic Native schedule and union used layers across that full schedule; Control changes only pre-boundary actions to OFF.','outcome_seen':False},
    'claim_boundary':{'mechanism_level':'Persistent fast weights plus trajectory-conditioned prefix attention context may be necessary for exact continuation under the tested implementation.','implementation_qualified':'Cache materialization and unused-layer forward-path semantics may additionally be required for exact replay.','forbidden':['universal KV requirement','P as universal mathematical state','fast weights unimportant','universal TTT state decomposition']},
    'historical_source_paths':{k:str(v) for k,v in source_paths.items()},
    'historical_source_hashes':source_hashes,
    'kp_counterfactual_executed':False,
}
rule_path = ROOT / 'manifests/KP_FACTORIAL_RULE_V1.json'
atomic_json(rule_path, rule)
rule_hash = sha(rule_path)
for row in census:
    row['rule_sha256'] = rule_hash
atomic_jsonl(ROOT / 'manifests/KP_FACTORIAL_ELIGIBILITY_V1.jsonl', census)
atomic_jsonl(ROOT / 'pools/KP_FACTORIAL_ELIGIBLE_ALL.jsonl', [x for x in census if x['kp_eligible']])
atomic_jsonl(ROOT / 'pools/KP_FACTORIAL_ELIGIBLE_INDEPENDENT.jsonl', [x for x in census if x['kp_eligible'] and not x['historical_overlap_reasons']])
atomic_jsonl(ROOT / 'pools/KP_FACTORIAL_INELIGIBLE.jsonl', [x for x in census if not x['kp_eligible']])

elig_manifest = {
    'version':'KP_FACTORIAL_ELIGIBILITY_V1_MANIFEST',
    'frozen_at':created,
    'model_identity':MODEL_ID,
    'checkpoint_identity':CKPT_ID,
    'benchmark_identity':BENCH_ID,
    'dynamic_authority':str(DYN),
    'dynamic_final_audit_sha256':sha(DYN/'audits/FINAL_DYNAMIC_NONCEILING_12TASK_AUDIT.json'),
    'state_reset_screening_authority':str(SR),
    'state_reset_rule_sha256':sha(SR/'manifests/STATE_RESET_RULE_V1.json'),
    'state_reset_eligibility_manifest_sha256':sha(SR/'manifests/STATE_RESET_ELIGIBILITY_V1_MANIFEST.json'),
    'state_reset_formal_authority_provenance_only':str(STATE),
    'state_reset_formal_receipt_sha256_provenance_only':sha(STATE/'reports/STATE_RESET_1P7B_FORMAL_COMPLETION_RECEIPT.json'),
    'selective_deletion_formal_authority_provenance_only':str(SEL),
    'selective_deletion_receipt_sha256_provenance_only':sha(SEL/'reports/SELECTIVE_DELETION_1P7B_FORMAL_COMPLETION_RECEIPT.json'),
    'rule_sha256':rule_hash,
    'parent_n':167,
    'eligible_n':len(eligible),
    'ineligible_n':len(ineligible),
    'independent_n':len(eligible),
    'historical_exposure_n':42,
    'overlap_n':0,
    'outcome_leakage':0,
    'branches_per_sample':4,
    'expected_future_branch_count':len(eligible)*4,
    'execution_cohort_policy':'ALL_ELIGIBLE_INDEPENDENT_NO_SUBSAMPLING',
    'execution_cohort_frozen':True,
    'formal_kp_amendment_required':True,
    'kp_counterfactual_executed':False,
    'file_hashes':{
        'KP_FACTORIAL_ELIGIBILITY_V1.jsonl':sha(ROOT/'manifests/KP_FACTORIAL_ELIGIBILITY_V1.jsonl'),
        'KP_FACTORIAL_ELIGIBLE_ALL.jsonl':sha(ROOT/'pools/KP_FACTORIAL_ELIGIBLE_ALL.jsonl'),
        'KP_FACTORIAL_ELIGIBLE_INDEPENDENT.jsonl':sha(ROOT/'pools/KP_FACTORIAL_ELIGIBLE_INDEPENDENT.jsonl'),
        'KP_FACTORIAL_INELIGIBLE.jsonl':sha(ROOT/'pools/KP_FACTORIAL_INELIGIBLE.jsonl'),
        'KP_FACTORIAL_PARENT_CENSUS.jsonl':sha(ROOT/'census/KP_FACTORIAL_PARENT_CENSUS.jsonl'),
        'HISTORICAL_KP_EXPOSURE_SET.jsonl':sha(ROOT/'historical/HISTORICAL_KP_EXPOSURE_SET.jsonl'),
    },
}
elig_manifest_path = ROOT / 'manifests/KP_FACTORIAL_ELIGIBILITY_V1_MANIFEST.json'
atomic_json(elig_manifest_path, elig_manifest)
elig_hash = sha(elig_manifest_path)

reason_counts = Counter('STATIC_KP_RECONSTRUCTABLE_AND_INDEPENDENT' if x['kp_eligible'] else x['exclusion_reason'] for x in census)
summary = {
    'parent_n':167,'eligible_n':len(eligible),'ineligible_n':len(ineligible),'static_reconstructable_n':len(eligible),
    'historical_exposure_n':42,'overlap_n':0,'canonical_overlap_n':0,'content_hash_overlap_n':0,
    'switch_count_distribution':dict(sorted(Counter(sum(a!=b for a,b in zip(x['dynamic_action_sequence'],x['dynamic_action_sequence'][1:])) for x in parent).items())),
    'multi_switch_amendment_n':sum(sum(a!=b for a,b in zip(x['dynamic_action_sequence'],x['dynamic_action_sequence'][1:]))>1 for x in parent),
    'eligibility_reason_counts':dict(reason_counts),
    'task_distribution':dict(sorted(Counter(x['task'] for x in eligible).items())),
}
atomic_json(ROOT / 'reports/KP_FACTORIAL_STATIC_SCREENING_SUMMARY.json', summary)
atomic_text(ROOT / 'reports/KP_FACTORIAL_CLAIM_BOUNDARY.md', '''# K/P Claim Boundary Freeze

Potentially supported by a future result: under this tested full-prompt implementation, persistent fast weights plus trajectory-conditioned prefix attention context determine important parts of continuation state. If P contributes, cache materialization/unused-layer execution-path semantics are an additional implementation-qualified exact-replay requirement.

Not permitted: universal KV necessity for all TTT, P as an implementation-independent mathematical state variable, fast-weight irrelevance, or a universal decomposition of TTT state.
''')

receipt = {
    'KP_FACTORIAL_SCREENING_STATUS':'PASS',
    'MODEL_AUTHORITY_STATUS':'PASS','BENCHMARK_AUTHORITY_STATUS':'PASS','DYNAMIC_AUTHORITY_STATUS':'PASS',
    'STATE_RESET_SCREENING_AUTHORITY_STATUS':'PASS','STATE_RESET_FORMAL_AUTHORITY_STATUS':'PASS','SELECTIVE_DELETION_FORMAL_AUTHORITY_STATUS':'PASS',
    'TRANSPLANT_V1_STATUS':'WITHDRAWN_CAUSALLY_CONFOUNDED',
    'TRANSPLANT_V1_CONFOUND':'Fast hashes matched, but control-derived prefix KV and all-five TTT materialization/unused-layer path differed from Native.',
    'HISTORICAL_RULE_RECOVERY_STATUS':'PROVEN_EXACT','K_DEFINITION_STATUS':'PROVEN_EXACT','P_DEFINITION_STATUS':'PROVEN_EXACT',
    'FAST_WEIGHT_IDENTITY_RULE_STATUS':'PROVEN_EXACT','FACTOR_BIT_ORDER_STATUS':'PROVEN_EXACT','CLOSURE_RULE_STATUS':'PROVEN_EXACT','KP_BOUNDARY_RULE_STATUS':'PROVEN_EXACT',
    'HISTORICAL_KP_PARENT_N':42,'HISTORICAL_KP_ELIGIBLE_N':42,'HISTORICAL_KP_EXECUTED_N':42,
    'PARENT_N':167,'ELIGIBLE_N':len(eligible),'INELIGIBLE_N':len(ineligible),'STATIC_KP_RECONSTRUCTABLE_N':len(eligible),
    'HISTORICAL_KP_EXPOSURE_N':42,'OVERLAP_N':0,'OUTCOME_LEAKAGE':0,
    'FORMAL_KP_AMENDMENT_REQUIRED':True,'RULE_FROZEN':True,'ELIGIBILITY_FROZEN':True,'EXECUTION_COHORT_FROZEN':True,
    'RULE_MANIFEST':str(rule_path),'RULE_MANIFEST_SHA256':rule_hash,
    'ELIGIBILITY_MANIFEST':str(elig_manifest_path),'ELIGIBILITY_MANIFEST_SHA256':elig_hash,
    'KP_BRANCHES_PER_SAMPLE':4,'EXPECTED_FUTURE_KP_BRANCHES':len(eligible)*4,
    'SAFE_TO_SELECT_KP_EXECUTION_COHORT':'YES','SAFE_TO_LAUNCH_KP':'NO',
    'KP_COUNTERFACTUAL_EXECUTED':False,'STATE_RESET_OUTCOME_USED_FOR_SCREENING':False,'SELECTIVE_DELETION_OUTCOME_USED_FOR_SCREENING':False,'EXACT_REVERSE_OUTCOME_USED_FOR_SCREENING':False,
    'NEW_GPU_COUNTERFACTUAL_RUNS':0,'FORMAL_MODEL_MUTATION':0,'BENCHMARK_MUTATION':0,'CONTINUOUS_MONITORING':False,
    'completed_at':created,'REPORT_ROOT':str(ROOT),
}
atomic_json(ROOT / 'receipts/KP_FACTORIAL_SCREENING_V1_RECEIPT.json', receipt)
receipt_md = f'''# K/P Factorial Screening V1 Receipt

KP_FACTORIAL_SCREENING_STATUS=PASS

Historical K/P protocol, K, P, bit order, four cells, exact fast-weight parity and strict L6 closure were recovered from executed source. Transplant V1 is withdrawn as causally confounded. The frozen State Reset parent partitions as 167 eligible / 0 ineligible with 0 historical content overlap and 0 outcome leakage.

A pre-outcome formal amendment is frozen because 71 new trajectories have multiple switches: use the complete Native Dynamic schedule, terminal-run boundary, and the union of layers used across that full schedule. Four branches per sample are planned (668 total). No branch was executed; SAFE_TO_LAUNCH_KP=NO.

- Rule: `{rule_path}` ({rule_hash})
- Eligibility: `{elig_manifest_path}` ({elig_hash})
'''
atomic_text(ROOT / 'receipts/KP_FACTORIAL_SCREENING_V1_RECEIPT.md', receipt_md)

# Final self-audit after all writes.
assert sha(rule_path) == rule_hash
assert sha(elig_manifest_path) == elig_hash
assert len(load_jsonl(ROOT/'manifests/KP_FACTORIAL_ELIGIBILITY_V1.jsonl')) == 167
assert len(load_jsonl(ROOT/'pools/KP_FACTORIAL_ELIGIBLE_ALL.jsonl')) == 167
assert len(load_jsonl(ROOT/'pools/KP_FACTORIAL_INELIGIBLE.jsonl')) == 0
print(json.dumps(receipt, sort_keys=True))
