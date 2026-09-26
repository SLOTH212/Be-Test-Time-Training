"""Independent boundary-sweep interventions on the frozen V2 Engine.

tau means AFTER tau complete 4096-token chunks, before chunk tau+1.
Selective deletion resets one layer's accumulated effective fast weight to its
checkpoint value. It does not remove one update, clear KV, or disable future
updates. Existing Engine.run supplies fresh caches, write-scope checks, parameter
version checks, and first-suffix-chunk/remainder prefill traces.
"""
import hashlib

LAYERS = [0, 6, 12, 18, 24, 30]
SELECTIVE_SOURCE_ROOT = '/path/to/ttt/releases/ttt_mechanism_runtime_v1'
SELECTIVE_SOURCE_HASHES = {
    'runtime.py': '90d5575947567d3804c05c00dfd00644bd1a6af06b88e05c94b671eec957312e',
    'provenance/original/deletion/screening/manifests/SELECTIVE_DELETION_RULE_V1.json': 'c82ea5f5c94b68ee3df5ec238faff2423ff9b25dd530ab8f34cfb92aafb3a800',
    'provenance/original/deletion/rule_sources/build_ablation_manifest.py': '90a2b6c0de856e4199706c105d634ef12cdaa4244886268c4ee5abf109460b91',
    'provenance/original/deletion/scripts/selective_worker.py': '4ad5c8769097a8baffeda9ef0f1baac4ae3b76ef9ab5bc56b7c2b96f4fa1dc10',
}


def boundaries(sequence):
    return list(range(1, min(4, len(sequence) - 1) + 1))


def _hash(tensor, torch):
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8)
                          .cpu().numpy().tobytes()).hexdigest()


def _closure(reference, candidate, name):
    assert reference['suffix_trace'], (name, 'EMPTY_TRACE')
    for key in ('prediction_hash', 'score', 'final_hashes', 'suffix_trace'):
        assert reference[key] == candidate[key], (name, key)


def _prefix_diff(a, b):
    assert a and set(a) == set(b), 'PREFIX_LAYER_COVERAGE'
    different = []
    for layer in sorted(a, key=int):
        for key in ('prefix_tokens', 'key_shape', 'value_shape', 'dtype'):
            assert a[layer][key] == b[layer][key], ('PREFIX_METADATA', layer, key)
        if any(a[layer][key] != b[layer][key] for key in ('key_hash', 'value_hash')):
            different.append(int(layer))
    return different


def supplement_boundary(engine, x, row, tau, run_kv=None, selective_layers=None):
    """Return one JSON-serializable boundary result; caller atomically commits it.

    KV only. Selective uses the separate historical selective_sample protocol.
    run_kv defaults to the original, score-independent eligible cohort.
    """
    seq = list(x['sequence'])
    assert tau in boundaries(seq), ('INVALID_BOUNDARY', tau, len(seq))
    assert row['sample_id'] == x['sample_id']
    eligible = bool(x['reset_screen']['eligible'])
    if run_kv is None:
        run_kv = eligible
    assert not run_kv or eligible, 'KV_COHORT_EXPANSION'
    assert selective_layers is None, 'SELECTIVE_REQUIRES_HISTORICAL_SAMPLE_PROTOCOL'

    native, keep = engine.run(row, seq, tau=tau, capture=True, retain=True)
    assert native['prediction_hash'] == x['prediction_hash'], 'FROZEN_NATIVE_PREDICTION'
    assert abs(native['score'] - x['score']) < 1e-12, 'FROZEN_NATIVE_SCORE'
    expected_layers = {str(i) for i in range(len(engine.model.model.layers))}
    assert set(native['donor']) == expected_layers, 'NATIVE_PREFIX_COVERAGE'
    checkpoint = {l: engine.model.model.layers[l].mlp.down_proj.weight.detach()
                  for l in LAYERS}
    checkpoint_hash = {str(l): _hash(v, engine.torch) for l, v in checkpoint.items()}
    native_weight_hash = {str(l): _hash(v, engine.torch) for l, v in keep['weights'].items()}
    assert native_weight_hash == native['boundary_before_hashes']

    noop, discarded = engine.run(row, seq, tau=tau, inject=keep['weights'],
                                 capture=True, retain=True)
    del discarded
    _closure(native, noop, 'NOOP_INTERVENTION_CLOSURE')
    assert noop['boundary_before_hashes'] == noop['boundary_after_hashes'] == native_weight_hash
    assert not _prefix_diff(native['donor'], noop['donor']), 'NOOP_PREFIX_CHANGED'
    assert not noop['kv_replaced_layers'], 'NOOP_UNEXPECTED_KV_REPLACEMENT'
    result = dict(schema='BOUNDARY_SWEEP_KV_V1', sample_id=x['sample_id'],
                  tau=tau, prefix_tokens=tau * 4096, full_chunk_count=len(seq),
                  native=native, noop=noop, noop_closure='PASS',
                  checkpoint_hashes=checkpoint_hash,
                  kv_eligible=eligible, generation_trace_covered=False)

    if run_kv:
        control_seq = ['OFF'] * tau + seq[tau:]
        control, discarded = engine.run(row, control_seq, tau=tau, capture=True, retain=True)
        del discarded
        k0, discarded = engine.run(row, control_seq, tau=tau, inject=keep['weights'],
                                   capture=True, retain=True)
        del discarded
        k1, discarded = engine.run(row, control_seq, tau=tau, inject=keep['weights'],
                                   donor=keep['donor'], capture=True, retain=True)
        del discarded
        assert not control['kv_replaced_layers'] and not k0['kv_replaced_layers']
        assert k0['boundary_after_hashes'] == k1['boundary_after_hashes'] == native_weight_hash
        assert not _prefix_diff(control['donor'], k0['donor']), 'K0_ORIGINAL_CONTROL_PREFIX_CHANGED'
        assert not _prefix_diff(native['donor'], k1['donor']), 'K1_DONOR_PREFIX_CHANGED'
        _closure(native, k1, 'KV_NATIVE_CLOSURE')
        different = _prefix_diff(native['donor'], k0['donor'])
        result['kv'] = dict(control=control, control_kv=k0, native_kv=k1,
                            native_closure='PASS', suffix_trace_closure='PASS',
                            native_control_prefix_different_layers=different,
                            native_control_prefix_exact_equal=not different,
                            control_trace_equals_native=k0['suffix_trace'] == native['suffix_trace'],
                            score_delta_k1_minus_k0=k1['score'] - k0['score'], P_factor=False)
    else:
        result['kv'] = dict(status='INELIGIBLE' if not eligible else 'NOT_REQUESTED',
                            screen=x['reset_screen'])
    del keep
    return result


def selective_plan(x):
    """Migrated 1.7B deletion_plan, with six layers and frozen 4K reset screen."""
    seq = list(x['sequence'])
    tau = len(seq) - 1
    while tau > 0 and seq[tau - 1] == seq[-1]:
        tau -= 1
    mapping = {'OFF': [], 'ALL': LAYERS, **{f'L{l}': [l] for l in LAYERS}}
    assert all(a in mapping for a in seq)
    prefix = seq[:tau]
    written = sorted({l for a in prefix for l in mapping[a]})
    all_prefix = 'ALL' in prefix
    targets = list(LAYERS) if all_prefix else [mapping[a][0] for a in prefix if mapping[a]][:1]
    parent_eligible = bool(x['reset_screen']['eligible'])
    eligible = parent_eligible and bool(targets)
    post = set(mapping[seq[tau]])
    if all_prefix:
        category = 'ALL_PREFIX'
    elif 'OFF' in prefix or seq[tau] == 'OFF':
        category = 'OFF_INVOLVING'
    elif len(written) == len(post) == 1 and set(written).isdisjoint(post):
        category = 'SINGLE_LAYER_DISJOINT'
    elif set(written) & post:
        category = 'OVERLAPPING_LAYER'
    else:
        category = 'COMPLEX_PREFIX'
    return dict(eligible=eligible, parent_reset_eligible=parent_eligible, tau=tau,
                boundary_chunk_1based=tau + 1, prefix_written_layers=written,
                selected_component_layers=targets if eligible else [], all_prefix=all_prefix,
                sham_layer=next((l for l in LAYERS if l not in written), None) if eligible else None,
                transition_category=category,
                exclusion_reason=None if eligible else ('RESET_INELIGIBLE' if not parent_eligible
                                                       else 'NO_NONZERO_PREFIX_COMPONENT_BEFORE_BOUNDARY'))


def selective_sample(engine, x, row):
    """Historical component selection at the single terminal-run boundary.

    ALL-prefix LOO branches average within sample. SHAM is the lowest unwritten
    layer. FullReset is rerun at the identical boundary for explained fraction.
    The already-approved resolution-aware reset parent replaces 1K screening;
    target/boundary/sham/aggregation rules otherwise preserve the 1.7B protocol.
    """
    plan = selective_plan(x)
    result = dict(schema='HISTORICAL_SELECTIVE_DELETION_4K_V1', sample_id=x['sample_id'],
                  plan=plan, generation_trace_covered=False,
                  source_root=SELECTIVE_SOURCE_ROOT, source_hashes=SELECTIVE_SOURCE_HASHES)
    if not plan['eligible']:
        return dict(result, status='INELIGIBLE')
    seq, tau = list(x['sequence']), plan['tau']
    assert row['sample_id'] == x['sample_id'] and 0 < tau < len(seq)
    native, keep = engine.run(row, seq, tau=tau, capture=True, retain=True)
    assert native['prediction_hash'] == x['prediction_hash'] and abs(native['score'] - x['score']) < 1e-12
    checkpoint = {l: engine.model.model.layers[l].mlp.down_proj.weight.detach() for l in LAYERS}
    checkpoint_hash = {str(l): _hash(v, engine.torch) for l, v in checkpoint.items()}
    weights = {str(l): _hash(v, engine.torch) for l, v in keep['weights'].items()}
    assert weights == native['boundary_before_hashes']
    noop, discarded = engine.run(row, seq, tau=tau, inject=keep['weights'], capture=True, retain=True)
    del discarded
    _closure(native, noop, 'SELECTIVE_NOOP_CLOSURE')
    assert not _prefix_diff(native['donor'], noop['donor'])
    assert noop['boundary_before_hashes'] == noop['boundary_after_hashes'] == weights
    full_reset, discarded = engine.run(row, seq, tau=tau, inject=checkpoint, capture=True, retain=True)
    del discarded
    assert full_reset['boundary_before_hashes'] == weights
    assert full_reset['boundary_after_hashes'] == checkpoint_hash
    assert not _prefix_diff(native['donor'], full_reset['donor'])
    result.update(native=native, noop=noop, noop_closure='PASS', full_reset=full_reset,
                  selective={}, sham=None, checkpoint_hashes=checkpoint_hash)
    for kind, layers in [('SELECTIVE', plan['selected_component_layers']),
                         ('SHAM', [] if plan['sham_layer'] is None else [plan['sham_layer']])]:
        for layer in layers:
            effective = weights[str(layer)] != checkpoint_hash[str(layer)]
            assert effective == (kind == 'SELECTIVE'), ('COMPONENT_NONZERO_PRECONDITION', kind, layer)
            rec, discarded = engine.run(row, seq, tau=tau, inject={layer: checkpoint[layer]},
                                        capture=True, retain=True)
            del discarded
            expected = dict(weights)
            expected[str(layer)] = checkpoint_hash[str(layer)]
            assert rec['boundary_before_hashes'] == weights, 'SELECTIVE_PREFIX_STATE_CHANGED'
            assert rec['boundary_after_hashes'] == expected, 'SELECTIVE_WRONG_LAYER_RESET'
            assert not rec['kv_replaced_layers'], 'SELECTIVE_UNEXPECTED_KV_REPLACEMENT'
            assert not _prefix_diff(native['donor'], rec['donor']), 'SELECTIVE_PREFIX_KV_CHANGED'
            if kind == 'SHAM':
                _closure(native, rec, 'SHAM_NATIVE_CLOSURE')
            preserved_nonzero = sum(weights[str(other)] != checkpoint_hash[str(other)]
                                    for other in LAYERS if other != layer)
            if kind == 'SELECTIVE' and plan['all_prefix']:
                assert preserved_nonzero >= 1, 'ALL_PREFIX_SELECTIVE_COLLAPSED_TO_FULL_RESET'
            branch = dict(intervention_type=kind, target_layer=layer, state_was_changed=effective,
                          deletion='accumulated_fast_weight_to_checkpoint', prefix_kv_unchanged=True,
                          preserved_nonzero_component_count=preserved_nonzero,
                          selective_not_full_reset=rec['boundary_after_hashes'] != checkpoint_hash,
                          effect=native['score'] - rec['score'], result=rec)
            if kind == 'SHAM':
                result['sham'] = branch
            else:
                result['selective'][str(layer)] = branch
    mean_score = sum(z['result']['score'] for z in result['selective'].values()) / len(result['selective'])
    denominator = native['score'] - full_reset['score']
    result.update(status='PASS', selective_mean_score=mean_score,
                  effect=native['score'] - mean_score,
                  explained_fraction=(native['score'] - mean_score) / denominator if denominator > 0 else None)
    del keep
    return result
