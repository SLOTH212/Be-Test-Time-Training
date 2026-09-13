"""Lightweight dispatcher: --help never loads a model."""
import argparse, importlib, json, sys

def main():
 parser=argparse.ArgumentParser(prog='dynamic-ttt',description='NTP and Dynamic Recomputed Lower Bound')
 parser.add_argument('command',choices=['quickstart','fixed','sample-best','dynamic','train','mechanism','prepare-data','analyze'])
 args,rest=parser.parse_known_args()
 if args.command=='quickstart':
  from dynamic_ttt.eval.runtime import fixed_actions,sample_best
  from dynamic_ttt.mechanisms.runtime import reverse,r2_candidates
  layers=[0,6,12,18,24];scores={x:0.2 for x in fixed_actions(layers)};scores['ALL']=0.8
  print(json.dumps({'actions':fixed_actions(layers),'sample_best':sample_best(scores,layers),'reverse':reverse(['L0','OFF','ALL']),'r2_candidates_4_chunks':len(list(r2_candidates(4,layers)))}));return
 modules={'fixed':'eval.fixed','sample-best':'eval.aggregate_fixed','dynamic':'dynamic.run','train':'training.run','mechanism':'mechanisms.run','prepare-data':'data.run','analyze':'stats.analyze'}
 sys.argv=[f'dynamic-ttt {args.command}']+rest
 importlib.import_module('dynamic_ttt.'+modules[args.command]).main()

if __name__=='__main__':main()
