"""Offline consistency checks; no GPU, network or model execution."""
from pathlib import Path
import hashlib,json,math,collections
ROOT=Path(__file__).resolve().parents[1]
def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
def close(a,b):assert math.isclose(a,b,abs_tol=1e-10),(a,b)
def mean(xs):return sum(xs)/len(xs)
def main():
    checks=0
    for line in (ROOT/'SHA256SUMS.txt').read_text().splitlines():
        digest,name=line.split('  ',1)
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
        checks+=1
    fixed=read('results/fixed8_v2/FIXED_AGGREGATE.json')
    frows=fixed['per_sample'];assert len(frows)==6500
    fm={r['sample_id']:r for r in frows};assert len(fm)==6500
    assert set(collections.Counter(r['task'] for r in frows).values())=={500}
    for a in fixed['actions']:close(mean([r[a] for r in frows]),fixed['action_means'][a])
    close(mean([max(r[a] for a in fixed['actions']) for r in frows]),fixed['sample_best_mean'])
    dyn=read('results/dynamic_v2/DYNAMIC_AGGREGATE.json');rows=dyn['per_sample']
    assert len(rows)==dyn['sample_count']==6000
    assert len({r['sample_id'] for r in rows})==6000
    assert all(r['task']!='ruler_cwe_16k' for r in rows)
    assert set(collections.Counter(r['task'] for r in rows).values())=={500}
    for r in rows:
        close(r['SampleBest'],fm[r['sample_id']]['sample_best'])
        close(r['delta'],r['Dynamic']-r['SampleBest'])
        assert r['delta']>=-1e-12
        if not r['searched']:close(r['SampleBest'],1);close(r['Dynamic'],1)
    close(mean([r['Dynamic'] for r in rows]),dyn['dynamic_mean'])
    close(mean([r['SampleBest'] for r in rows]),dyn['sample_best_mean'])
    close(mean([r['delta'] for r in rows]),dyn['dynamic_minus_sample_best'])
    searched=[r for r in rows if r['searched']];assert len(searched)==dyn['searched_count']==1278
    assert 6000-len(searched)==dyn['ceiling_count']==4722
    wins=sum(r['delta']>1e-12 for r in searched);assert wins==dyn['improved']==85
    assert sum(abs(r['delta'])<=1e-12 for r in searched)==dyn['equal']==1193
    trajectories=read('results/dynamic_v2/trajectories.json');assert len(trajectories)==1278
    tm={r['sample_id']:r for r in trajectories};assert set(tm)=={r['sample_id'] for r in searched}
    for row in searched:
        t=tm[row['sample_id']];close(t['dynamic_search_score'],row['Dynamic'])
        close(t['fixed_oracle_score'],row['SampleBest'])
        assert len(t['best_sequence'])==t['full_chunk_count']
        assert set(t['best_sequence'])<=set(fixed['actions'])
        assert t['base_weight_unchanged'] and t['generation_update_count']==0
    assert sum(t['cross_action_specific_gain']>1e-12 for t in trajectories)==55
    assert read('results/dynamic_v2/DYNAMIC_FINAL_AUTHORITY.json')['status']=='PASS'
    assert read('results/dynamic_v2/PHASE_STATUS.json')['exit_codes']==[0]*8
    base=read('results/base/AGGREGATE.json');assert len(base['per_sample'])==6500
    assert all('prediction' not in r for r in base['per_sample'])
    close(mean([r['score'] for r in base['per_sample']]),base['mean_score'])
    print(json.dumps({'status':'PASS','files_verified':checks,'fixed_samples':6500,'dynamic_population':6000,'searched':1278,'improved':wins,'dynamic_mean':dyn['dynamic_mean'],'gain_percentage_points':100*dyn['dynamic_minus_sample_best']},indent=2))
if __name__=='__main__':main()
