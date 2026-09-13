"""Portable mechanism primitives. Importing this module performs no model work."""
from __future__ import annotations
import ast, copy, hashlib, json, math, os, types
from collections import Counter, defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parent
TOL=1e-12

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def rows(path):
 with Path(path).open() as f:return [json.loads(s) for s in f if s.strip()]
def actions(layers):
 if not layers or any(type(x) is not int or x<0 for x in layers) or sorted(set(layers))!=layers:raise ValueError('INVALID_LAYER_SET')
 return ['OFF',*[f'L{x}' for x in layers],'ALL']
def mapping(layers):return {'OFF':[],**{f'L{x}':[x] for x in layers},'ALL':list(layers)}
def validate_config(c):
 layers=c['configured_ttt_layers'];a=actions(layers)
 if c['actions']!=a:raise ValueError('ACTION_SCHEMA')
 if type(c['chunk_size']) is not int or c['chunk_size']<1 or c['context_length']<c['chunk_size']:raise ValueError('CONTEXT_CHUNK')
 if c['model_family']!='qwen3':raise ValueError('MODEL_FAMILY_NOT_SUPPORTED')
 if c['mechanism'] not in ['reverse','reset','deletion','kp','r2','template']:raise ValueError('UNKNOWN_MECHANISM')
 if c['classification']=='FORMAL_HISTORICAL_REFERENCE' and (layers!=[0,6,12,18,24] or c['chunk_size']!=1024 or c['context_length']!=16384):raise ValueError('HISTORICAL_CONFIG_MUTATION')
 return c
def load_config(path):
 c=json.loads(Path(path).read_text(encoding='utf-8'))
 def expand(value):
  if isinstance(value,str):return os.path.expandvars(value)
  if isinstance(value,list):return [expand(x) for x in value]
  if isinstance(value,dict):return {k:expand(v) for k,v in value.items()}
  return value
 return validate_config(expand(c))


def reset_rules(layers):
 return source_functions('kernels/reset_rules.py',['historical_early_steady','terminal_run_boundary'],{'ACTIONS':actions(layers)})
def deletion_rules(layers):
 return source_functions('kernels/deletion_rules.py',['category','first_real_layer'],{'ACTION_LAYERS':mapping(layers)})
def reverse(sequence):return list(reversed(sequence))
def reset_screen(sequence,layers):
 if not sequence or any(x not in actions(layers) for x in sequence):raise ValueError('INVALID_SEQUENCE')
 rule=reset_rules(layers);passed,details=rule.historical_early_steady(sequence)
 reason=None
 if not passed:
  if details['first_switch_chunk_index'] is None:reason='NO_ACTION_SWITCH'
  elif details['first_switch_position_normalized']>.25:reason='FIRST_SWITCH_AFTER_25_PERCENT'
  else:reason='SECOND_HALF_MODAL_ACTION_FRACTION_BELOW_0P8'
 return {'eligible':passed,'details':details,'boundary':rule.terminal_run_boundary(sequence) if passed else None,'exclusion_reason':reason}
def deletion_plan(sequence,boundary,layers):
 if boundary!=reset_rules(layers).terminal_run_boundary(sequence):raise ValueError('BOUNDARY_MISMATCH')
 prefix=sequence[:boundary-1];m=mapping(layers);rule=deletion_rules(layers)
 written=sorted({l for a in prefix for l in m[a]});all_prefix='ALL' in prefix
 target=list(layers) if all_prefix else ([rule.first_real_layer(prefix)] if written else [])
 return {'eligible':bool(target),'boundary':boundary,'selected_component_layers':target,'all_prefix':all_prefix,'sham_layer':next((l for l in layers if l not in written),None) if target else None,'transition_category':rule.category(prefix,sequence[boundary-1]),'exclusion_reason':None if target else 'NO_NONZERO_PREFIX_COMPONENT_BEFORE_BOUNDARY'}
def kp_plan(sequence,boundary,layers):
 if boundary!=reset_rules(layers).terminal_run_boundary(sequence):raise ValueError('BOUNDARY_MISMATCH')
 used=sorted({l for a in sequence for l in mapping(layers)[a]})
 return {'native_sequence':list(sequence),'control_sequence':['OFF']*(boundary-1)+sequence[boundary-1:],'prefix_chunks':boundary-1,'native_used_layers':used,'cells':{c:{'K_bit':k,'P_bit':p,'W_source':'NATIVE','K_source':'NATIVE' if k else 'CONTROL','injection_layers':used if p else list(layers)} for c,(k,p) in {'F00':(0,0),'F10':(1,0),'F01':(0,1),'F11':(1,1)}.items()}}
def r2_candidates(T,layers):
 if type(T) is not int or T<2:raise ValueError('R2_REQUIRES_TWO_COMPLETE_CHUNKS')
 for A in actions(layers):
  for B in actions(layers):
   if A==B:continue
   for tau in range(1,T):
    seq=[A]*tau+[B]*(T-tau)
    yield {'A':A,'B':B,'tau':tau,'T':T,'sequence':seq,'full_schedule_hash':canonical(seq)}
def r2_winner(records,layers):
 return source_functions('kernels/r2_winner.py',['independent_winner'],{'ACTIONS':actions(layers),'TOL':TOL}).independent_winner(records)
def improved_cohort(fixed,dynamic,layers):
 """Input maps contain complete fixed action scores and one committed Dynamic record per ID."""
 a=actions(layers);out=[]
 for sid,d in sorted(dynamic.items()):
  scores=fixed[sid]
  if set(scores)!=set(a):raise ValueError('FIXED_ACTION_COVERAGE '+sid)
  if any(not math.isfinite(float(v)) for v in scores.values()):raise ValueError('NONFINITE_SCORE')
  sb=max(float(v) for v in scores.values());score=float(d['dynamic_score'])
  if not math.isfinite(score):raise ValueError('NONFINITE_DYNAMIC')
  if abs(sb-d['sample_best'])>TOL:raise ValueError('SAMPLE_BEST_PARITY '+sid)
  if score<sb-TOL:raise ValueError('DYNAMIC_BELOW_SAMPLE_BEST '+sid)
  if d.get('completed') is not True:raise ValueError('UNCOMMITTED_DYNAMIC '+sid)
  seq=d['action_sequence']
  if len(seq)!=d['chunk_count'] or any(x not in a for x in seq):raise ValueError('DYNAMIC_SEQUENCE '+sid)
  if score>sb+TOL:out.append({**d,'sample_id':sid,'fixed_scores':dict(scores),'sample_best':sb})
 return out

def validate_result(r,committed=False):
 required=['schema','model_identity','benchmark_identity','sample_id','task','benchmark_global_index','parent_dynamic_trajectory','parent_dynamic_score','sample_best','intervention','intervention_score','details','errors','timing']
 if any(k not in r for k in required) or r['schema']!='MECHANISM_SAMPLE_V1':raise ValueError('RESULT_SCHEMA')
 if r['intervention'] not in ['reverse','reset','deletion','kp','r2'] or set(r['details'])!={r['intervention']}:raise ValueError('RESULT_NAMESPACE')
 if not r['sample_id'] or not r['task'] or not isinstance(r['parent_dynamic_trajectory'],list):raise ValueError('RESULT_IDENTITY')
 if any(not isinstance(r[k],(int,float)) or not math.isfinite(r[k]) for k in ['parent_dynamic_score','sample_best','intervention_score']):raise ValueError('RESULT_SCORE')
 if r['parent_dynamic_score']<=r['sample_best']+TOL:raise ValueError('RESULT_PARENT_NOT_IMPROVED')
 if r['errors'] or r['timing']['wall_seconds']<0:raise ValueError('RESULT_ERROR_OR_TIMING')
 if committed and (r.get('status')!='COMMITTED' or not r.get('config_authority')):raise ValueError('RESULT_COMMIT')
 return r

def dependency(root=None,archive=None):
 return ROOT.parent

def source_functions(relative,names,env=None,chunk=None,integer_overrides=None):
 """Bind preserved tensor/rule definitions to the configured architecture."""
 p=ROOT/relative
 tree=ast.parse(p.read_text())
 nodes=[copy.deepcopy(x) for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name in names]
 if {x.name for x in nodes}!=set(names):raise ValueError('SOURCE_DEFINITION_SET')
 replacements=dict(integer_overrides or {})
 if chunk is not None:replacements[1024]=chunk
 class Chunk(ast.NodeTransformer):
  def visit_Constant(self,n):return ast.copy_location(ast.Constant(replacements[n.value]),n) if type(n.value) is int and n.value in replacements else n
 nodes=[Chunk().visit(x) for x in nodes]
 ns={'Counter':Counter,'defaultdict':defaultdict,'Path':Path,'hashlib':hashlib,'json':json,'math':math,'os':os,'types':types,**(env or {})}
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(p),'exec'),ns)
 return types.SimpleNamespace(**ns)
