"""CPU-only independent aggregation; never writes historical authorities."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np

TOL = 1e-12

def read(path):
    return json.loads(path.read_text())

def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]

def put(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)

def table(path, rows, fields):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)

def contrast(rows, left, right, seed, draws):
    d = np.array([r[left] - r[right] for r in rows], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sums = np.zeros(draws)
    for task in sorted({r['task'] for r in rows}):
        v = np.array([d[i] for i, r in enumerate(rows) if r['task'] == task])
        n = len(v)
        for start in range(0, draws, 1000):
            size = min(1000, draws - start)
            sums[start:start + size] += v[rng.integers(0, n, (size, n))].sum(1)
    return dict(n=len(rows), mean_pp=float(d.mean() * 100),
                CI95=np.quantile(sums / len(rows) * 100, [.025, .975]).tolist(),
                left_lower=int((d < -TOL).sum()), equal=int((abs(d) <= TOL).sum()),
                left_higher=int((d > TOL).sum()))

def summarize(root):
    cfg = read(root / 'config.json')
    historical = Path(cfg['historical_root'])
    frozen = read(historical / 'COHORT.json')
    expected = [c for c in frozen if c['nontrivial_reset']]
    cohort = lines(root / 'cohort_manifest.jsonl')
    branch_manifest = lines(root / 'branch_manifest.jsonl')
    seeds = cfg['seeds']
    errors = []
    checks = {f'A{i}': True for i in range(1, 13)}
    def test(value, key, message):
        if not value:
            checks[key] = False
            errors.append(message)
    ids = [c['sample_id'] for c in cohort]
    test(len(ids) == len(set(ids)) == cfg['expected_cohort_n'] == 157 and
         ids == [c['sample_id'] for c in expected], 'A1', 'Cohort IDs/order mismatch')
    ec = {c['sample_id']: c for c in expected}
    expected_branches = [(c['sample_id'], l) for c in expected for l in c['deletion_targets']]
    test([(b['sample_id'], b['layer']) for b in branch_manifest] == expected_branches,
         'A3', 'Branch manifest mismatch')
    test(seeds == [20260926, 20260927, 20260928, 20260929, 20260930], 'A11', 'Frozen five-seed list mismatch')
    test(cfg['bootstrap_seed'] == 20260921 and cfg['bootstrap_replicates'] == 20000, 'A1', 'Frozen bootstrap protocol mismatch')
    for c in cohort:
        ref = ec.get(c['sample_id'], {})
        test(all(c.get(k) == ref.get(k) for k in ['tau', 'boundary', 'sequence']), 'A2', 'Boundary/trajectory mismatch: ' + c['sample_id'])
        test(c.get('deletion_targets') == ref.get('deletion_targets'), 'A3', 'Layer rule mismatch: ' + c['sample_id'])
    found = [read(p) for p in sorted((root / 'results').glob('*.json'))]
    by_id = {}
    for r in found:
        sid = r['sample_id']
        test(sid in ec and sid not in by_id, 'A1', 'Unknown/duplicate result: ' + sid)
        by_id[sid] = r
    rows, raw, diagnostics = [], [], []
    for c in expected:
        sid = c['sample_id']
        if sid not in by_id:
            continue
        r = by_id[sid]
        old = read(historical / 'core' / (hashlib.sha256(sid.encode()).hexdigest() + '.json'))
        deletion = {b['layer']: b['result']['score'] for b in old['deletion']}
        test(r['task'] == c['task'], 'A1', 'Task mismatch: ' + sid)
        test(r.get('tau') == c['tau'] and r.get('boundary') == c['boundary'], 'A2', 'Raw result boundary mismatch: ' + sid)
        test(r.get('sequence') == c['sequence'], 'A8', 'Raw result trajectory mismatch: ' + sid)
        test(r.get('config_sha256') == hashlib.sha256((root / 'config.json').read_bytes()).hexdigest(), 'A1', 'Raw result config mismatch: ' + sid)
        test(r.get('native_prediction_hash') == c['native_prediction_hash'], 'A4', 'Native prediction hash mismatch: ' + sid)
        native_hashes = {str(e['layer']): e['before'] for e in old['K']['native']['interventions']}
        test(r.get('native_boundary_hashes') == native_hashes, 'A4', 'Native boundary hash mismatch: ' + sid)
        native_kv = r.get('native_prefix_kv_hashes', {})
        test(set(native_kv) == {str(i) for i in range(28)} and all(v.get('prefix_tokens') == c['tau'] * 1024 for v in native_kv.values()), 'A7', 'Native KV layer coverage mismatch: ' + sid)
        test(r.get('native_replay_pass') is True and r.get('native_repeat_pass') is True and
             abs(r['native_score'] - c['native_score']) <= TOL, 'A4', 'Native replay failure: ' + sid)
        test([b['layer'] for b in r['branches']] == c['deletion_targets'], 'A3', 'Result branch mismatch: ' + sid)
        branch_means = []
        for b in r['branches']:
            layer = b['layer']
            test(b.get('deletion_prediction_hash') == next((d['result']['prediction_hash'] for d in old['deletion'] if d['layer'] == layer), None), 'A5', 'Deletion prediction mismatch: ' + sid)
            test(layer in deletion and abs(b['deletion_score'] - deletion.get(layer, -999)) <= TOL,
                 'A5', 'Deletion branch mismatch: ' + sid)
            test([s['seed'] for s in b['seeds']] == seeds, 'A11', 'Seed list/order mismatch: ' + sid)
            values = []
            for s in b['seeds']:
                values.append(s['score'])
                dg = s['diagnostics']
                test(all(dg.get(k, v) == v for k, v in {'sample_id': sid, 'layer': layer, 'seed': s['seed']}.items()), 'A9', 'Diagnostic identity mismatch: ' + sid)
                events = s.get('interventions', [])
                test(len(events) == 5 and {e.get('layer') for e in events} == {0, 6, 12, 18, 24}, 'A6', 'Intervention layer coverage mismatch: ' + sid)
                for e in events:
                    el = e['layer']
                    test(e.get('chunk_index') == c['tau'] + 1, 'A2', 'Intervention boundary mismatch: ' + sid)
                    test(e.get('before') == native_hashes.get(str(el)), 'A6', 'Preintervention state mismatch: ' + sid)
                    after = s.get('perturbed_weight_hash') if el == layer else native_hashes.get(str(el))
                    test(bool(after) and e.get('after') == after and e.get('injected') == (el == layer) and e.get('reset') is False, 'A6', 'Target-only injection mismatch: ' + sid)
                test(s.get('prefix_kv_hashes') == native_kv and bool(native_kv), 'A7', 'Branch prefix KV mismatch: ' + sid)
                test(np.isfinite(s['score']), 'A4', 'Nonfinite score: ' + sid)
                err = abs(dg['realized_norm'] - dg['target_norm']) / dg['target_norm'] if dg['target_norm'] > 0 else float('inf')
                test(np.isfinite(err) and abs(err - dg['relative_norm_error']) <= 1e-9 and
                     err <= .01 and dg['matched_within_1pct'] is True, 'A9', 'Magnitude mismatch: ' + sid)
                test(dg['realized_norm'] > 0 and dg['target_norm'] > 0, 'A10', 'Zero displacement: ' + sid)
                for k in ['A6', 'A7', 'A8', 'A9', 'A10']:
                    test(s.get('checks', {}).get(k) is True, k, f'{k} branch failure: {sid}')
                item = dict(sample_id=sid, task=c['task'], layer=layer, seed=s['seed'],
                            score=s['score'], prediction_hash=s['prediction_hash'], diagnostics=dg, checks=s['checks'])
                raw.append(item)
                diagnostics.append({**dg, 'sample_id': sid, 'task': c['task'], 'layer': layer, 'seed': s['seed']})
            branch_means.append(float(np.mean(values)))
        deletion_mean = sum(deletion.values()) / len(deletion)
        random_mean = float(np.mean(branch_means))
        test(abs(r['deletion_score'] - deletion_mean) <= TOL, 'A5', 'Deletion average mismatch: ' + sid)
        test(abs(r['random_score'] - random_mean) <= TOL, 'A3', 'Random average mismatch: ' + sid)
        for k in [f'A{i}' for i in range(1, 11)]:
            test(r.get('checks', {}).get(k) is True, k, f'{k} sample failure: {sid}')
        rows.append(dict(sample_id=sid, task=c['task'], native_score=c['native_score'],
                         deletion_score=deletion_mean, random_score=random_mean,
                         native_minus_random_pp=(c['native_score'] - random_mean) * 100,
                         random_minus_deletion_pp=(random_mean - deletion_mean) * 100,
                         n_branches=len(branch_means), n_seeds=len(seeds)))
    smoke_path = root / 'SMOKE_RESULT.json'
    smoke = read(smoke_path) if smoke_path.exists() else {}
    test(smoke.get('status') == 'PASS' and smoke.get('determinism_pass') is True,
         'A11', 'Smoke determinism has not passed')
    # Explicit source-review evidence, written by orchestration before formal run.
    source_path = root / 'source_integrity_check.json'
    source = read(source_path) if source_path.exists() else {}
    integrity = source.get('status') == 'PASS'
    test(source.get('reward_blind_construction') is True or cfg.get('reward_blind_construction_reviewed') is True,
         'A12', 'Reward-blind construction source review not recorded')
    complete = len(rows) == 157 and len(by_id) == 157
    status = 'PASS' if complete and all(checks.values()) and integrity else ('FAIL' if complete else 'BLOCKED')
    historical_a1 = read(historical / 'SUMMARY.json')['results']['deletion']
    summary = dict(status=status, completed_n=len(rows), expected_n=157, checks=checks,
                   errors=errors, source_integrity=integrity, A1_frozen_historical=historical_a1,
                   statistical_unit='example', bootstrap_seed=cfg['bootstrap_seed'],
                   bootstrap_replicates=cfg['bootstrap_replicates'], comparison_tolerance_score=TOL)
    # Partial estimates are explicitly withheld; raw partial outputs remain auditable.
    if complete:
        summary['A2_native_minus_random'] = contrast(rows, 'native_score', 'random_score', cfg['bootstrap_seed'], cfg['bootstrap_replicates'])
        summary['A3_random_minus_deletion'] = contrast(rows, 'random_score', 'deletion_score', cfg['bootstrap_seed'], cfg['bootstrap_replicates'])
        summary['means_percent'] = {k: float(np.mean([r[k] for r in rows]) * 100) for k in ['native_score', 'deletion_score', 'random_score']}
    norm_errors = [d['relative_norm_error'] for d in diagnostics]
    summary['magnitude'] = dict(n=len(norm_errors), median_relative_error=float(np.median(norm_errors)) if norm_errors else None,
        mean_relative_error=float(np.mean(norm_errors)) if norm_errors else None,
        max_relative_error=float(max(norm_errors)) if norm_errors else None,
        matched_within_1pct_fraction=float(np.mean([d['matched_within_1pct'] for d in diagnostics])) if diagnostics else None)
    put(root / 'bootstrap_summary.json', summary)
    (root / 'raw_results.jsonl').write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in raw))
    table(root / 'example_level_results.csv', rows, ['sample_id','task','native_score','deletion_score','random_score','native_minus_random_pp','random_minus_deletion_pp','n_branches','n_seeds'])
    dg_fields = ['sample_id','task','layer','seed'] + sorted({k for d in diagnostics for k in d} - {'sample_id','task','layer','seed'})
    table(root / 'magnitude_match_diagnostics.csv', diagnostics, dg_fields)
    tasks = []
    for task in sorted({r['task'] for r in rows}):
        subset = [r for r in rows if r['task'] == task]
        tasks.append(dict(task=task, n=len(subset), **{k:float(np.mean([r[k] for r in subset]) * (100 if k.endswith('_score') else 1)) for k in ['native_score','deletion_score','random_score','native_minus_random_pp','random_minus_deletion_pp']}))
    table(root / 'task_level_results.csv', tasks, ['task','n','native_score','deletion_score','random_score','native_minus_random_pp','random_minus_deletion_pp'])
    mag = summary['magnitude']; means = summary.get('means_percent', {}); a2 = summary.get('A2_native_minus_random', {}); a3 = summary.get('A3_random_minus_deletion', {})
    contract = dict(MAGNITUDE_MATCHED_CONTROL_STATUS=status, COHORT_N=157, COHORT_EXACT=checks['A1'], N_SEEDS=len(seeds),
        NATIVE_REPLAY_PASS=checks['A4'] and complete, MAGNITUDE_MATCH_MEDIAN_REL_ERROR=mag['median_relative_error'],
        MAGNITUDE_MATCH_MAX_REL_ERROR=mag['max_relative_error'], MATCH_WITHIN_1PCT_RATE=mag['matched_within_1pct_fraction'],
        MEAN_NATIVE=means.get('native_score'), MEAN_DELETION=means.get('deletion_score'), MEAN_MATCHED_RANDOM=means.get('random_score'),
        NATIVE_MINUS_RANDOM_PP=a2.get('mean_pp'), NATIVE_MINUS_RANDOM_CI95=a2.get('CI95'),
        RANDOM_MINUS_DELETION_PP=a3.get('mean_pp'), RANDOM_MINUS_DELETION_CI95=a3.get('CI95'), FINAL_INTEGRITY='PASS' if status == 'PASS' else 'FAIL')
    text = '\n'.join(k+'='+json.dumps(v) if not isinstance(v,str) else k+'='+v for k,v in contract.items())+'\n\n'
    text += f'# Magnitude-matched nonzero control\n\nCompleted {len(rows)}/157 examples. Statistical unit is example; average five seeds within branch, then branches within example. Task tables are descriptive only; score columns in task table are percentages, example table scores are fractions. Equality tolerance is 1e-12 score units.\n\n'
    text += 'A1 is copied verbatim from the historical SUMMARY.json; historical artifacts are read-only. A2/A3 use paired task-stratified 20,000-draw percentile bootstrap with frozen seed and cohort order.\n\n'
    if status == 'PASS':
        lo, hi = a3['CI95']
        if lo > 0:
            text += 'CASE 1: Matched random exceeds deletion; the measured deletion effect cannot be explained solely by displacement magnitude. This does not establish a unique causal mechanism.\n'
        elif hi < 0:
            text += 'CASE 3: Matched random is worse than deletion.\n'
        else:
            text += 'CASE 2: The contrast is inconclusive; this does not establish equivalence or distinguish specific direction effects from generic same-magnitude sensitivity.\n'
        if a3['mean_pp'] < 0:
            text += 'The random-minus-deletion point estimate is negative.\n'
    else:
        text += 'No formal scientific conclusion is issued while completion or audit gates are pending/failed.\n'
    text += '\nAudit checks:\n\n' + '\n'.join(f'- CHECK_{k}: {"PASS" if v else "FAIL/PENDING"}' for k,v in checks.items())+'\n'
    (root / 'final_report.md').write_text(text)
    (root / 'audit_report.md').write_text('# Independent CPU audit\n\n'+json.dumps(dict(status=status, checks=checks, source_integrity=integrity, completed=len(rows), errors=errors), indent=2)+'\n\nNo existing Selective Deletion results were modified. A12 requires explicit reward-blind source-review evidence in config or source integrity check.\n')
    return summary

if __name__ == '__main__':
    os.sched_setaffinity(0, set(range(8, 24)))
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    result = summarize(args.root)
    print(json.dumps(dict(status=result['status'], completed_n=result['completed_n'], checks=result['checks'])))
