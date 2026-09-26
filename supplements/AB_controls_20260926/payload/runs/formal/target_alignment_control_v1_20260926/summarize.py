#!/usr/bin/env python3
"""CPU-only, sample-unit reporting for the frozen target-alignment control."""
import os
os.sched_setaffinity(0, set(range(8, 24)))
import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np

LAYERS = [0, 6, 12, 18, 24]
MASK = {'OFF': [], 'ALL': LAYERS, **{f'L{x}': [x] for x in LAYERS}}
TOL = 1e-12

def load(path):
    return json.loads(Path(path).read_text())

def atomic(path, text):
    path = Path(path)
    tmp = path.with_name('.' + path.name + '.tmp')
    tmp.write_text(text)
    os.replace(tmp, path)

def dump(path, obj):
    atomic(path, json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')

def table(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    tmp = path.with_name('.' + path.name + '.tmp')
    with tmp.open('w', newline='') as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)

def stat(rows, left, right, seed, draws):
    """Exact historical RNG/task/batch convention, paired example differences."""
    a = np.asarray([r[left] for r in rows], dtype=float)
    b = np.asarray([r[right] for r in rows], dtype=float)
    d = a - b
    rng = np.random.default_rng(seed)
    boot = np.zeros(draws)
    for task in sorted({r['task'] for r in rows}):
        v = np.asarray([d[i] for i, r in enumerate(rows) if r['task'] == task])
        n = len(v)
        for j in range(0, draws, 1000):
            m = min(1000, draws - j)
            boot[j:j+m] += v[rng.integers(0, n, (m, n))].sum(1)
    return {'n': len(rows), 'difference_pp': float(d.mean() * 100),
            'CI95': np.quantile(boot / len(rows) * 100, [.025, .975]).tolist()}

def write_key(w):
    return (int(w['chunk_index']), int(w['layer']))

def expected_writes(c):
    return {(j + 1, layer): action for j, action in enumerate(c['sequence']) for layer in MASK[action]}

def calibration_valid(w):
    """Audit scalar-only recorded search without loading model tensors or reward."""
    try:
        c = w['calibration']
        target = w['native_represented_norm']
        raw = w['shuffle_raw_norm']
        candidates = c['evaluated_candidates']
        by_alpha = {q['alpha']: q for q in candidates}
        invalid = c['invalid_alphas']
        if len(by_alpha) != len(candidates) or not candidates:
            return False
        if not all(math.isfinite(a) and a >= 0 for a in [*by_alpha, *invalid]):
            return False
        if c['n_evaluations'] != len(candidates) or c['n_invalid_candidates'] != len(invalid):
            return False
        if c['target_norm'] != target or c['shuffle_raw_norm'] != raw:
            return False
        for q in candidates:
            if not all(math.isfinite(q[k]) and q[k] >= 0 for k in ['raw_norm','clipped_norm','represented_norm','absolute_error']):
                return False
            if q['absolute_error'] != abs(q['represented_norm'] - target):
                return False
        selected = min(candidates, key=lambda q: (q['absolute_error'], q['alpha']))
        if selected['alpha'] != w['scale_alpha'] or selected['represented_norm'] != w['represented_norm']:
            return False
        if c['alpha'] != selected['alpha'] or c['represented_norm'] != w['represented_norm']:
            return False
        if c['relative_error'] != w['relative_norm_error'] or c['matched_within_1pct'] != w['matched_within_1pct']:
            return False
        if c['selection'] != 'closest_evaluated_candidate; ties_smaller_alpha':
            return False
        if c['observedmax'] != max(q['represented_norm'] for q in candidates):
            return False
        if target == 0:
            return c['zero_native'] is True and w['scale_alpha'] == 0 and w['represented_norm'] == 0 and len(candidates) == 1 and c['iterations'] == 0
        if raw == 0:
            return c['zero_shuffle'] is True and w['scale_alpha'] == 0 and len(candidates) == 1 and c['iterations'] == 0
        if c['iterations'] != 12 or len(candidates) > 59:
            return False
        # Reconstruct the exact fixed grid and adaptive neighbor sequence from
        # their scalar errors; invalid finite candidates retain their metadata.
        visited = set()
        attempted = set()
        def evaluate(alpha):
            alpha = float(alpha)
            attempted.add(alpha)
            if alpha in by_alpha:
                visited.add(alpha)
            elif alpha not in invalid:
                raise ValueError('missing prescribed candidate')
        evaluate(0.)
        evaluate(1.)
        alpha_clip = 1e-5/raw
        for exponent in range(-16,17):
            evaluate(math.ldexp(alpha_clip,exponent))
        for _ in range(12):
            best = min(visited,key=lambda a:(by_alpha[a]['absolute_error'],a))
            ordered = sorted(visited)
            i = ordered.index(best)
            proposed = []
            if i:
                lo = ordered[i-1]
                proposed.append(best/2 if lo == 0 else math.exp((math.log(lo)+math.log(best))/2))
            if i+1 < len(ordered):
                hi = ordered[i+1]
                proposed.append(hi/2 if best == 0 else math.exp((math.log(best)+math.log(hi))/2))
            for a in proposed:
                evaluate(a)
        return visited == set(by_alpha) and attempted == set(by_alpha) | set(invalid)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args()
    root = args.root.resolve()
    cfg = load(root / 'config.json')
    manifest = [json.loads(s) for s in (root / 'cohort_manifest.jsonl').read_text().splitlines() if s.strip()]
    authority = load(Path(cfg['historical_root']) / 'COHORT.json')
    formal = {x['sample_id']: x for x in authority}
    all_results = [load(p) for p in sorted((root / 'results').glob('*.json'))]
    results = {r['sample_id']: r for r in all_results}
    n = int(cfg.get('expected_cohort_n', 170))
    seeds = list(cfg['seeds'])
    complete = len(results) == n and set(results) == set(formal)
    if not complete:
        dump(root / 'SUMMARY_PROGRESS.json', {'completed_samples': len(results), 'expected_samples': n,
            'status': 'INCOMPLETE', 'missing_sample_ids': [c['sample_id'] for c in authority if c['sample_id'] not in results]})
        print(f'INCOMPLETE: {len(results)}/{n}; final statistics not emitted')
        return 2
    errors = []
    def require(condition, message):
        if not condition:
            errors.append(message)
        return bool(condition)
    checks = {f'B{i}': True for i in range(1, 15)}
    checks['B1'] = require(len(manifest) == n == len(authority) and
        [c['sample_id'] for c in manifest] == [c['sample_id'] for c in authority] and
        len(all_results) == len(results), 'Cohort/order/unique result mismatch')
    example_rows, seed_rows, write_rows, trajectory_rows = [], [], [], []
    native_write_count = 0
    mode_write_counts = {'B1': 0, 'B2': 0}
    config_hashes = set()
    for c in authority:
        sid = c['sample_id']
        r = results[sid]
        config_hashes.add(r.get('config_sha256'))
        want = expected_writes(c)
        native = {write_key(w): w for w in r['native_writes']}
        native_write_count += len(r['native_writes'])
        tr_ok = r['sequence'] == c['sequence'] and r['task'] == c['task']
        checks['B2'] &= require(tr_ok, f'{sid}: trajectory/task mismatch')
        trajectory_rows.append({'sample_id': sid, 'task': c['task'], 'trajectory_exact': tr_ok,
            'expected_writes': len(want), 'sequence': c['sequence']})
        replay = (r.get('native_replay_pass') is True and r.get('native_repeat_pass') is True and
            abs(r['native_score'] - c['native_score']) <= TOL and
            r['native_prediction_hash'] == c['native_prediction_hash'] and
            abs(r['sample_best'] - c['sample_best']) <= TOL)
        checks['B3'] &= require(replay, f'{sid}: Native replay/reference mismatch')
        checks['B9'] &= require(set(native) == set(want) and len(native) == len(r['native_writes']), f'{sid}: native write mask mismatch')
        for key, w in native.items():
            checks['B9'] &= require(w['action'] == want.get(key), f'{sid}: Native action mismatch')
            require(all(isinstance(w.get(k), str) and len(w[k]) == 64 for k in ['raw_hash', 'clipped_hash', 'before_hash', 'after_hash']), f'{sid}: Native tensor hashes missing')
            require(all(math.isfinite(w[k]) and w[k] >= 0 for k in ['native_raw_norm', 'native_clipped_norm', 'native_represented_norm']), f'{sid}: Native norms invalid')
        require([s['seed'] for s in r['seeds']] == seeds, f'{sid}: frozen seed list/order mismatch')
        for k in checks:
            checks[k] &= require(r.get('checks', {}).get(k) is True, f'{sid}: recorded {k} is not true')
        means = {'B1': [], 'B2': []}
        for s in r['seeds']:
            sr = {'sample_id': sid, 'task': c['task'], 'seed': s['seed'],
                  'sample_best': r['sample_best'], 'native_score': r['native_score']}
            by_mode = {}
            for mode in ['B1', 'B2']:
                q = s[mode]
                require(math.isfinite(q['score']), f'{sid}: nonfinite {mode} score')
                means[mode].append(q['score'])
                sr[mode] = q['score']
                sr[mode + '_prediction_hash'] = q['prediction_hash']
                ws = q['writes']
                by_mode[mode] = {write_key(w): w for w in ws}
                mode_write_counts[mode] += len(ws)
                checks['B9'] &= require(set(by_mode[mode]) == set(want) and len(ws) == len(want), f'{sid}/{s["seed"]}/{mode}: write mask mismatch')
                for w in ws:
                    key = write_key(w)
                    checks['B9'] &= require(w['action'] == want.get(key), f'{sid}/{mode}: action mismatch')
                    checks['B6'] &= require(w['sample_id'] == sid and w['task'] == c['task'] and w['seed'] == s['seed'], f'{sid}/{mode}: sample/seed scope mismatch')
                    checks['B8'] &= require(w['n_targets'] == 1023 and w['fixed_points'] == 0 and bool(w['permutation_hash']), f'{sid}/{mode}: permutation audit failed')
                    if key in native:
                        for field in ['native_raw_norm', 'native_clipped_norm', 'native_represented_norm']:
                            require(w[field] == native[key][field], f'{sid}/{mode}: native norm reference changed: {field}')
                    checks['B4'] &= require(w.get('target_scope') == 'this_sample_this_layer_this_complete_chunk', f'{sid}: target scope invalid')
                    checks['B7'] &= require(w.get('branch_local_targets') is True, f'{sid}: nonlocal branch targets')
                    checks['B10'] &= require(w.get('read_before_update') is True, f'{sid}: update timing assertion missing')
                    checks['B13'] &= require(w.get('reward_used') is False, f'{sid}: reward-blindness assertion missing')
                    if mode == 'B2':
                        checks['B11'] &= require(calibration_valid(w), f'{sid}: frozen scalar search validation failed')
                        target = w['native_represented_norm']
                        actual = w['represented_norm']
                        err = abs(actual - target) / target if target else (0.0 if actual == 0 else float('inf'))
                        checks['B11'] &= require(math.isfinite(err) and abs(err-w['relative_norm_error']) <= 1e-9 and
                            bool(w['matched_within_1pct']) == (err <= .01) and
                            math.isfinite(w['scale_alpha']) and w['scale_alpha'] >= 0,
                            f'{sid}: B2 represented matching metadata invalid')
            seed_rows.append(sr)
            for key in sorted(set(by_mode['B1']) & set(by_mode['B2'])):
                a, b = by_mode['B1'][key], by_mode['B2'][key]
                checks['B8'] &= require(a['permutation_hash'] == b['permutation_hash'], f'{sid}: B1/B2 permutations differ')
                merged = {k: a[k] for k in ['sample_id', 'task', 'chunk_index', 'layer', 'action', 'seed',
                    'native_raw_norm', 'native_clipped_norm', 'native_represented_norm', 'permutation_hash', 'n_targets', 'fixed_points']}
                for mode, w in [('B1', a), ('B2', b)]:
                    for k, value in w.items():
                        if k not in merged and k != 'mode':
                            merged[mode + '_' + k] = value
                    merged['shuffle_raw_norm_' + mode] = w['shuffle_raw_norm']
                write_rows.append(merged)
        b1, b2 = float(np.mean(means['B1'])), float(np.mean(means['B2']))
        gain = r['native_score'] - r['sample_best']
        require(gain > 0, f'{sid}: non-improved sample in cohort')
        example_rows.append({'sample_id': sid, 'task': c['task'], 'sample_best': r['sample_best'],
            'native_score': r['native_score'], 'B1': b1, 'B2': b2,
            'native_minus_B1_pp': (r['native_score']-b1)*100,
            'native_minus_B2_pp': (r['native_score']-b2)*100,
            'B2_minus_sample_best_pp': (b2-r['sample_best'])*100,
            'recovery_B2': (b2-r['sample_best'])/gain if gain else None})
    require(len(config_hashes) == 1 and None not in config_hashes, 'Results do not share one config hash')
    observed_config_hash = hashlib.sha256((root/'config.json').read_bytes()).hexdigest()
    require(config_hashes == {observed_config_hash}, 'Result config hash differs from current frozen config file bytes')
    expected_count = sum(len(expected_writes(c)) for c in authority)
    require(native_write_count == expected_count == 3760, 'Native write count != frozen 3760')
    require(all(v == expected_count * len(seeds) == 18800 for v in mode_write_counts.values()), 'Condition write count != frozen 18800')
    smoke = load(root/'SMOKE_RESULT.json') if (root/'SMOKE_RESULT.json').exists() else {}
    source = load(root/'source_integrity_check.json') if (root/'source_integrity_check.json').exists() else {}
    checks['B12'] &= require(smoke.get('status') == 'PASS' and smoke.get('determinism_pass') is True, 'Smoke determinism gate missing/failing')
    checks['B14'] &= require(smoke.get('identity_exact') is True, 'Identity recovery gate missing/failing')
    checks['B14'] &= require(smoke.get('native_original_step_exact') is True, 'Original native-step numerical parity missing/failing')
    require(source.get('status') == 'PASS', 'Source integrity gate missing/failing')
    bseed = int(cfg.get('bootstrap_seed', 20260921))
    draws = int(cfg.get('bootstrap_replicates', 20000))
    require(bseed == 20260921 and draws == 20000 and len(seeds) == 5, 'Frozen statistics/seed count mismatch')
    summary = {'n': n, 'n_seeds': len(seeds), 'statistical_unit': 'example; average five seeds before bootstrap',
        'bootstrap_seed': bseed, 'bootstrap_replicates': draws, 'numpy_version': np.__version__,
        'bootstrap_convention': 'PCG64; sorted tasks; frozen cohort order; batches 1000; linear 95% percentile',
        'native_minus_B1': stat(example_rows, 'native_score', 'B1', bseed, draws),
        'native_minus_B2': stat(example_rows, 'native_score', 'B2', bseed, draws),
        'B2_minus_sample_best': stat(example_rows, 'B2', 'sample_best', bseed, draws)}
    for key in ['native_score', 'sample_best', 'B1', 'B2']:
        summary[key + '_mean_percent'] = float(np.mean([r[key] for r in example_rows])*100)
    summary['mean_per_example_recovery_B2'] = float(np.mean([r['recovery_B2'] for r in example_rows]))
    summary['aggregate_recovery_B2'] = sum(r['B2']-r['sample_best'] for r in example_rows)/sum(r['native_score']-r['sample_best'] for r in example_rows)
    summary['counts'] = {
        'B2_below_native': sum(r['B2'] < r['native_score']-TOL for r in example_rows),
        'B2_equal_native': sum(abs(r['B2']-r['native_score']) <= TOL for r in example_rows),
        'B2_above_native': sum(r['B2'] > r['native_score']+TOL for r in example_rows),
        'B2_at_or_below_sample_best': sum(r['B2'] <= r['sample_best']+TOL for r in example_rows),
        'B2_above_sample_best': sum(r['B2'] > r['sample_best']+TOL for r in example_rows)}
    errs = np.asarray([w['B2_relative_norm_error'] for w in write_rows])
    mag = {'n_writes': len(errs), 'median_relative_error': float(np.median(errs)),
        'max_relative_error': float(errs.max()), 'within_1pct_rate': float(np.mean(errs <= .01)),
        'outside_1pct_count': int(np.sum(errs > .01)),
        'strict_all_writes_within_1pct': bool(np.all(errs <= .01)),
        'native_zero_norm_count': sum(w['native_represented_norm'] == 0 for w in write_rows),
        'interpretation': 'B11 checks permitted deterministic calibration protocol; strict attainment is reported separately.'}
    summary['magnitude_matching'] = mag
    task_rows = []
    for task in sorted({r['task'] for r in example_rows}):
        rs = [r for r in example_rows if r['task'] == task]
        task_rows.append({'task': task, 'n': len(rs), **{k + '_mean_percent': float(np.mean([r[k] for r in rs])*100) for k in ['sample_best','native_score','B1','B2']},
            'native_minus_B1_pp': float(np.mean([r['native_minus_B1_pp'] for r in rs])),
            'native_minus_B2_pp': float(np.mean([r['native_minus_B2_pp'] for r in rs])),
            'mean_recovery_B2': float(np.mean([r['recovery_B2'] for r in rs]))})
    sys.path.insert(0, str(root/'deps'))
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(write_rows), root/'.write_level_diagnostics.parquet.tmp')
    os.replace(root/'.write_level_diagnostics.parquet.tmp', root/'write_level_diagnostics.parquet')
    table(root/'seed_level_results.csv', seed_rows)
    table(root/'example_level_results.csv', example_rows)
    table(root/'task_level_results.csv', task_rows)
    atomic(root/'trajectory_result_audit.jsonl', ''.join(json.dumps(r, sort_keys=True)+'\n' for r in trajectory_rows))
    status = 'PASS' if not errors and all(checks.values()) else 'FAIL'
    summary.update(status=status, checks=checks, errors=errors, native_write_count=native_write_count,
                   mode_write_counts=mode_write_counts, config_sha256=observed_config_hash)
    dump(root/'bootstrap_summary.json', summary)
    audit = ['# Target alignment execution audit', '', f'Execution status: {status}', '',
             '| Check | Result |', '|---|---|']
    labels = ['COHORT_EXACT','TRAJECTORY_EXACT','NATIVE_REPLAY','NO_CROSS_CHUNK_TARGET',
        'NO_CROSS_LAYER_TARGET','NO_CROSS_SAMPLE_TARGET','NO_FUTURE_NATIVE_CACHE','FIXED_POINT_AUDIT',
        'ACTION_MASK_EXACT','UPDATE_TIMING','B2_MAGNITUDE_MATCH_PROTOCOL','SEED_DETERMINISM',
        'NO_REWARD_LEAKAGE','NATIVE_TARGET_RECOVERY']
    audit += [f'| CHECK_B{i}_{label} | {"PASS" if checks[f"B{i}"] else "FAIL"} |' for i,label in enumerate(labels,1)]
    audit += ['', f'Native writes: {native_write_count}; per-condition writes: {mode_write_counts}.',
        f'Strict 1% attainment: {mag["within_1pct_rate"]:.6%}; exceptions: {mag["outside_1pct_count"]}; max relative error: {mag["max_relative_error"]:.9g}.',
        'B4/B5/B7/B10/B13 combine frozen implementation/source integrity with recorded per-sample assertions; scalar summaries alone cannot establish execution semantics.',
        'B11 PASS denotes the allowed nearest-tested-scalar calibration protocol, not universal 1% attainment or proof of a global optimum.', '', 'Errors:', *[f'- {e}' for e in errors]]
    atomic(root/'audit_report.md', '\n'.join(audit)+'\n')
    contract = {'TARGET_ALIGNMENT_CONTROL_STATUS':status, 'COHORT_N':n,'COHORT_EXACT':checks['B1'],
        'TRAJECTORY_EXACT':checks['B2'],'N_SEEDS':len(seeds),'NATIVE_REPLAY_PASS':checks['B3'],
        'B1_MEAN':summary['B1_mean_percent'],'B2_MEAN':summary['B2_mean_percent'],
        'NATIVE_MEAN':summary['native_score_mean_percent'],'SAMPLE_BEST_MEAN':summary['sample_best_mean_percent'],
        'NATIVE_MINUS_B1_PP':summary['native_minus_B1']['difference_pp'], 'NATIVE_MINUS_B1_CI95':summary['native_minus_B1']['CI95'],
        'NATIVE_MINUS_B2_PP':summary['native_minus_B2']['difference_pp'], 'NATIVE_MINUS_B2_CI95':summary['native_minus_B2']['CI95'],
        'B2_MINUS_SAMPLEBEST_PP':summary['B2_minus_sample_best']['difference_pp'], 'B2_MINUS_SAMPLEBEST_CI95':summary['B2_minus_sample_best']['CI95'],
        'B2_MEAN_GAIN_RECOVERY':summary['mean_per_example_recovery_B2'],'B2_AGG_GAIN_RECOVERY':summary['aggregate_recovery_B2'],
        'B2_MAG_MATCH_MEDIAN_REL_ERROR':mag['median_relative_error'],'B2_MAG_MATCH_MAX_REL_ERROR':mag['max_relative_error'],
        'B2_MATCH_WITHIN_1PCT_RATE':mag['within_1pct_rate'],'B2_STRICT_MAGNITUDE_MATCH':mag['strict_all_writes_within_1pct'],
        'FINAL_INTEGRITY':status}
    lines = [f'{k}={json.dumps(v) if not isinstance(v,str) else v}' for k,v in contract.items()]
    lines += ['', '# Within-Chunk Target-Alignment Control', '',
        'Scores and differences above are percentage points/percent; recovery ratios are unbounded unitless ratios.',
        'All 170 frozen improved examples are retained. No trajectory search or outcome-based selection. Five seeds are averaged within each example before paired task-stratified bootstrap.', '',
        '| Condition | Mean score (%) |','|---|---:|']
    lines += [f'| {k} | {summary[k+"_mean_percent"]:.8f} |' for k in ['sample_best','native_score','B1','B2']]
    lines += ['', '## Magnitude control and interpretation', '',
        f'B2 strict matching rate: {mag["within_1pct_rate"]:.6%}; {mag["outside_1pct_count"]} writes exceed 1%. The scalar search retains the closest tested solution under unchanged clipping and BF16 addition.',
        'If matching exceptions are substantial, these results do not establish an equal-magnitude causal contrast. A significant Native–B2 difference alone cannot remove the residual magnitude confound.' if not mag['strict_all_writes_within_1pct'] else
        'All recorded B2 writes attain the prescribed 1% represented-norm tolerance. Interpret any difference as evidence about within-chunk target alignment/content, without claims about semantic understanding.',
        'No global optimizer guarantee is claimed; finite deterministic search, clipping saturation and BF16 quantization are disclosed.', '',
        '## Provenance and limitations', '', f'Frozen configuration SHA256: `{observed_config_hash}`.',
        f'Historical source: `{cfg["historical_root"]}`; source integrity status: {source.get("status")}.',
        'Identity recovery and seed determinism are validated on the frozen smoke cohort; Native replay and stored prediction/reward agreement are checked for every formal sample.',
        'Raw result JSON retains write hashes and per-seed diagnostics. Parquet contains paired B1/B2 write rows; CSV example data retain unclipped recovery. Task results are descriptive.',
        'Protocol deviations: no unrecorded deviation is inferred by this summarizer. See configuration, smoke record and source integrity evidence for execution provenance; norm-tolerance exceptions are disclosed above.', '',
        'Sample-level comparisons: '+json.dumps(summary['counts'], sort_keys=True), '',
        'Execution errors: '+json.dumps(errors)]
    atomic(root/'final_report.md', '\n'.join(lines)+'\n')
    print(json.dumps({'status':status,'n':n,'magnitude_matching':mag,'errors':errors},sort_keys=True))
    return 0 if status == 'PASS' else 1

if __name__ == '__main__':
    raise SystemExit(main())
