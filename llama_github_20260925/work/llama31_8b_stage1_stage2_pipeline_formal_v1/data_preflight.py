import json,sys,math,subprocess
from pathlib import Path
from data_adapter import records,sha
R=Path('/path/to/ttt');W=R/'work/llama31_8b_stage1_stage2_pipeline_formal_v1';W.mkdir(parents=True,exist_ok=True)
def put(n,x):(W/n).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
put('qwen_before.json',json.loads(subprocess.check_output([str(R/'bin/status_qwen3_4b_stage1_stage2_pipeline_formal_v1.sh')],text=True)))
packages=[('llama31_stage1_500m_32k_v1',671719626,'c7189fb46adad303b97c7a3cee36059bb201e9b7b5fa2cbf3772a4535c841680'),('llama31_stage2_15m_32k_v1',23923710,'6bb877e4ece2819fb063581ff276c305574822a3c55a1dcfa31ed8cc79030227')]
results={}
for stage,(name,size,h) in enumerate(packages,1):
 package=R/'packages'/(name+'.tar.gz');assert package.stat().st_size==size and sha(package)==h
 root=R/'datasets'/name;authority=json.loads((root/'authority/DATASET_AUTHORITY.json').read_text());assert authority['status']=='PASS'
 for p,v in authority['data_file_hashes'].items():assert sha(root/p)==v,p
 result={'package_sha256':h,'package_size':size,'authority_sha256':sha(root/'authority/DATASET_AUTHORITY.json'),'splits':{}}
 for split in (['train','validation'] if stage==1 else ['train']):
  count=tokens=answers=qa=0;seen=set();first=[]
  for row in records(root,[split]):
   assert row['id'] not in seen;seen.add(row['id']);count+=1;tokens+=row['n_tokens'];qa+=row['qa_count']
   if stage==1:
    bounds=row['document_boundaries_qwen_json'];assert bounds[0][0]==0 and bounds[-1][1]==row['n_tokens'];assert all(0<=a<b<=row['n_tokens'] for a,b in bounds)
    for left,right in zip(bounds,bounds[1:]):assert left[1]<=right[0] and (left[1]==right[0] or row['input_ids'][left[1]:right[0]]==[128001])
   else:
    mask=set()
    for a,b in row['answer_ranges']:
     assert 0<=a<b<=row['n_tokens'];mask.update(range(max(1,a)-1,max(1,b)-1))
    assert mask and max(mask)<row['n_tokens']-1;answers+=len(mask)
    assert row['n_tokens']+row['padding']==32768
   if count<=12:first.append({'sample_id':row['id'],'tokens':row['n_tokens'],'split':split,'input_ids_sha256':__import__('hashlib').sha256(bytes(__import__('numpy').array(row['input_ids'],dtype='int64'))).hexdigest()})
  result['splits'][split]={'records':count,'tokens':tokens,'answer_positions':answers,'qa_targets':qa,'first_records':first}
 if stage==1:
  assert result['splits']['train']['records']==17015 and result['splits']['train']['tokens']==478374652
  assert result['splits']['validation']['records']==185 and result['splits']['validation']['tokens']==4850937
 else:assert (count,tokens,answers,qa)==(457,14390540,97619,19397)
 results[str(stage)]=result
put('data_cpu_preflight.json',{'status':'PASS','datasets':results,'stage1_requested_counts_are_train_plus_validation':True,'formal_launch_requires_explicit_split_resolution':True,'training_started':False})
print(json.dumps({'status':'PASS','counts':{s:r['splits'] for s,r in results.items()}},indent=2))
