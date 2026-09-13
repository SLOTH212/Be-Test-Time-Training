"""Dispatch source-preserved data construction stages."""
import argparse,importlib
def main():
 p=argparse.ArgumentParser();p.add_argument('--family',choices=['qwen','llama'],required=True)
 p.add_argument('--step',required=True,choices=['selection','order','stage1','stage2','subsets','audit-stage1','audit-stage2','finalize']);x=p.parse_args()
 if x.family=='llama':
  from dynamic_ttt.data import derive_llama as d
  from dynamic_ttt.data import lifecycle
  jobs={'subsets':d.prepare_subsets,'stage1':d.materialize_stage1,'stage2':d.materialize_stage2,
        'audit-stage1':lambda:lifecycle.audit_llama('stage1'),
        'audit-stage2':lambda:lifecycle.audit_llama('stage2'),
        'finalize':lifecycle.finalize_llama}
  if x.step not in jobs:p.error('Unsupported Llama data step')
  jobs[x.step]()
 else:
  names={'selection':'stream_select_extension','order':'build_final_content_order','stage1':'materialize_stage1_32k','stage2':'qa_replay','audit-stage2':'qa_audit'}
  if x.step not in names:p.error('Qwen supports selection, order, stage1, stage2')
  importlib.import_module('dynamic_ttt.data.reconstruction.'+names[x.step]).main()
if __name__=='__main__':main()
