#!/usr/bin/env python3
"""Plan or document the fixed -> Sample Best -> Dynamic LB -> audit workflow."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from code.runtime import fixed_actions,load_config,replay_budget
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--dry-run',action='store_true');a=p.parse_args();cfg=load_config(a.config);actions=fixed_actions(cfg['ttt_layers']);plan={"status":"DRY_RUN_PASS","phases":["fixed_evaluation","sample_best_aggregation","non_ceiling_cohort","dynamic_lb","aggregation_bootstrap_integrity"],"actions":actions,"max_unique_candidate_replays_at_context_capacity":replay_budget(len(actions),cfg['context_length']//cfg['ttt_chunk_size'])}
if not a.dry_run:raise SystemExit('orchestration is explicit: run fixed/run_fixed.py, fixed/aggregate_fixed.py, dynamic/run_dynamic.py workers, then dynamic/aggregate_dynamic.py')
print(json.dumps(plan,sort_keys=True))
