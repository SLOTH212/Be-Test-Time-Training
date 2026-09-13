from dynamic_ttt.stats.public_results import analyze

def test_public_csv_paired_effect_and_denominator_policy():
 rows=[{'sample_id':str(i),'task':'a' if i<2 else 'b','native':1,'counterfactual':.5,'full_reset':0 if i<2 else 1} for i in range(4)]
 x=analyze(rows,'deletion',{'resamples':50,'paired':1,'stratified':2})
 assert x['paired_ci']==[.5,.5]
 assert x['task_stratified_ci']==[.5,.5]
 assert x['positive_denominator_n']==2
 assert x['nonpositive_denominator_n']==2
 assert x['aggregate_explained_fraction']==.5
