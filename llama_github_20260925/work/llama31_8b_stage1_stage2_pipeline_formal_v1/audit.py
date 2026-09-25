"""Independent read-only CPU audit of actual assignment, contracts and negatives."""
import ast,json,sys,copy,tempfile,subprocess
from pathlib import Path
from core import *
import yaml
sys.path.insert(0,str(R/'src/training_runtime/code'))
from lib.distributed_semantics import BatchPlan
from data_adapter import records
def audit():
 c=config();checks=[];negative=[]
 def reject(label,fn):
  try:fn()
  except Exception:negative.append(label);return
  raise AssertionError('NEGATIVE_ACCEPTED '+label)
 a1=read(c['stage1']['data_authority']);a2=read(c['stage2']['data_authority']);dataset_contract(a1,a2)
 require(sha(c['stage1']['data_authority'])=='c6828d7ea25c4d57bb9db4e690f7907a81746703b5c54b47651a5f9d3ea2cc59','FROZEN_DATA_AUTHORITY_CHANGED')
 plan=BatchPlan(4,1,1)
 for stage in [1,2]:
  cfg=yaml.safe_load(Path(c[f'stage{stage}']['config']).read_text());cfg_contract(cfg,stage);index=read(c[f'stage{stage}']['record_index']);seen=[]
  data_root=Path(c[f'stage{stage}']['data_authority']).parent.parent
  independent=[]
  for p in sorted((data_root/'data/train').glob('*.jsonl')):
   for i,line in enumerate(p.read_text().splitlines()):independent.append({'id':str(p.relative_to(data_root))+':'+str(i),'tokens':json.loads(line)['length'],'split':'train'})
  require(independent==index,'INDEPENDENT_FROZEN_MEMBERSHIP_MISMATCH')
  for rank in range(4):seen.extend(pos for pos,_,_ in plan.rank_positions(rank,0,len(index)))
  require(sorted(seen)==list(range(len(index))) and len(seen)==len(set(seen)),'ASSIGNMENT_COVERAGE')
  require(max(plan.owner(i)[0] for i in seen)+1==EXPECTED[stage]['update_step'],'ACTUAL_UPDATE_COUNT')
  last_start=(EXPECTED[stage]['update_step']-1)*4;lastowners=[plan.owner(i)[2] for i in range(last_start,len(index))];require(lastowners==([0,1,2] if stage==1 else [0]),'FINAL_PARTIAL_GROUP')
  checks.append({'stage':stage,'real_records':len(seen),'updates':EXPECTED[stage]['update_step'],'duplicates':0,'missing':0,'final_real_ranks':lastowners,'final_dummy_ranks':4-len(lastowners)})
  for key,bad in [('record_cursor_final',17200 if stage==1 else 458),('train_tokens',483225589 if stage==1 else 15000000),('expected_optimizer_steps',4300 if stage==1 else 114),('world_size',2),('ttt_chunk_size',1024),('outer_lr',1e-5)]:
   badcfg=copy.deepcopy(cfg);badcfg[key]=bad;reject(f'stage{stage}_{key}',lambda:cfg_contract(badcfg,stage))
  tree=ast.parse((W/f'worker_stage{stage}.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');s=ast.unparse(main)
  require('optimizer.step()' in s and 'if loss is None:' in s and '* 0.0' in s,'DUMMY_COLLECTIVES_MISSING')
  require("cfg.get('stage1_model_only_artifact')" in s and not cfg.get('stage1_model_only_artifact') and cfg['debug_validation_records']==0,'VALIDATION_EXECUTION_SCOPE')
 train=read(c['stage1']['record_index']);val=read(W/'stage1_validation_index.json');tids={x['id'] for x in train};vids={x['id'] for x in val}
 require(len(tids)==17015 and len(vids)==185 and not tids&vids and all(x['split']=='train' for x in train),'TRAIN_VALIDATION_SEPARATION')
 require(sum(x['tokens'] for x in train)==478374652 and sum(x['tokens'] for x in val)==4850937,'SPLIT_TOKEN_TOTALS')
 independent_val=[]
 data_root=Path(c['stage1']['data_authority']).parent.parent
 for p in sorted((data_root/'data/validation').glob('*.jsonl')):
  for i,line in enumerate(p.read_text().splitlines()):independent_val.append({'id':str(p.relative_to(data_root))+':'+str(i),'tokens':json.loads(line)['length'],'split':'validation'})
 require(independent_val==val,'INDEPENDENT_VALIDATION_MEMBERSHIP')
 reject('validation_id_in_training',lambda:require(not ((tids|{val[0]['id']})&vids),'CONTAMINATION'))
 for k,v in [('train_records',17200),('train_input_tokens',483225589),('optimizer_steps',4300),('validation_used_for_training',True)]:
  b={**SPLIT,k:v};reject(k,lambda:split_contract(b))
 from pipeline import policy
 require(policy(True,{},True,False,False)=='ADOPT' and policy(False,{},False,False,False)=='BLOCK' and policy(True,None,False,False,False)=='BLOCK','EXACTLY_ONCE_POLICY')
 reject('duplicate_stage1',lambda:require(policy(True,{},False,False,False)=='LAUNCH','DUPLICATE'))
 reject('duplicate_stage2',lambda:require(policy(False,{},False,False,False)=='LAUNCH','DUPLICATE'))
 with tempfile.TemporaryDirectory(dir=W) as temp:
  p=Path(temp)/'lock'
  with lock(p):reject('duplicate_supervisor',lambda:subprocess.run([sys.executable,'-c','import sys;sys.path.insert(0,'+repr(str(W))+');from core import lock;\nwith lock('+repr(str(p))+'): pass'],check=True,capture_output=True))
  wrong=Path(temp)/'file';wrong.write_text('x')
  for label in ['bad_model_hash','bad_port_hash','wrong_stage1_package_sha','wrong_stage2_package_sha']:reject(label,lambda:hashes({str(wrong):'0'*64}))
  for stage in [1,2]:
   root=Path(temp)/f'final{stage}';root.mkdir();(root/f'STAGE{stage}_FINAL_AUTHORITY.json').write_text(json.dumps({'artifact_class':f'STAGE{stage}_FINAL','DEBUG_ONLY':True,'immutable':False}));reject(f'debug_or_incomplete_final_{stage}',lambda:validate_final(root,stage,c))
   (root/f'STAGE{stage}_FINAL_AUTHORITY.json').write_text(json.dumps({'artifact_class':f'STAGE{stage}_FINAL','DEBUG_ONLY':False,'immutable':True,'final_root':'wrong','integrity':'PASS'}));reject(f'mismatching_existing_final_{stage}',lambda:validate_final(root,stage,c))
 identity_contract(c)
 for k in ['model_revision','port_sha256']:
  b=copy.deepcopy(c);b[k]='wrong';reject('bad_'+k,lambda:identity_contract(b))
 for stage,name in [(1,'llama31_stage1_500m_32k_v1.tar.gz'),(2,'llama31_stage2_15m_32k_v1.tar.gz')]:
  b=copy.deepcopy(c);b['bindings'][str(R/'packages'/name)]='wrong';reject(f'bad_stage{stage}_package_identity',lambda:identity_contract(b))
 require(c['downstream_auto_start'] is False and c['final_long_term_model_artifacts']==2,'NO_DOWNSTREAM_SCOPE')
 return {'status':'PASS','assignment':checks,'negative_tests':negative,'negative_test_n':len(negative),'TRAIN_VALIDATION_SEPARATION':'PASS','DATA_AUTHORITY_IMMUTABLE':'PASS','LLAMA_STAGE1_ACCOUNTING_CORRECTION_AUDIT':'PASS','source_read_only':True,'GPU_tests_run':False}
if __name__=='__main__':print(json.dumps(audit(),indent=2))
