from dynamic_ttt.stats.reverse import ci,stratified_ci
from dynamic_ttt.stats.r2 import bootstrap
def test_paired_and_stratified_degenerate():
 assert ci([.25]*4,42,100)==[.25,.25]
 rows=[{'task':t,'dynamic_minus_reverse':.25} for t in ['a','b','a','b']]
 assert stratified_ci(rows,42,100)==[.25,.25]
def test_r2_ratio_of_sums():
 rows=[{'task':'a','Dynamic_minus_R2':.25,'Dynamic_minus_OFF':1.,'R2_minus_OFF':.75}]*4
 result=bootstrap(rows,42,100,True)
 assert result['mean_dynamic_minus_r2_95ci']==[.25,.25]
 assert result['aggregate_r2_gain_retention_95ci']==[.75,.75]
