import json,hashlib,time
from pathlib import Path
import torch
R=Path('/path/to/ttt');L=R/'audits/qwen3_4b_stage1_resume_shared_boundary_v4';RUN=R/'runs/qwen3_4b_stage1_resume_shared_boundary_v4'
torch.set_num_threads(8);assert not torch.cuda.is_initialized()
targets=json.loads((L/'EXACT_V3_TARGETS.json').read_text())['rows'];plan={'parameters':sorted({x['parameter'] for x in targets})};results=[]
for name in [x['branch'] for x in json.loads((L/'RUN_RECEIPTS.json').read_text())]:
 data=[torch.load(RUN/name/f'final_rank{r}.pt',mmap=True,map_location='cpu',weights_only=True) for r in (0,1)];meta=[json.loads((RUN/name/f'final_rank{r}.json').read_text()) for r in (0,1)]
 coeff=meta[0]['clip_observation']['actual_clamped_coefficient'];assert len(coeff['samples'])==1;coefficient=torch.tensor(coeff['samples'][0],dtype=getattr(torch,coeff['dtype'].split('.')[-1]))
 for p in plan['parameters']:
  pre='pre_reduce.gradient.'+p;post='post_reduce.gradient.'+p;pc='pre_clip.gradient.'+p;cl='post_clip.gradient.'+p
  if not all(k in d for d in data for k in [pre,post,pc,cl]):
   results.append({'run':name,'parameter':p,'available':False});continue
  assert all(torch.equal(d[post],d[pc]) for d in data)
  global0=data[0][pre].reshape(-1);global1=data[1][pre].reshape(-1);assert global0.shape==global1.shape
  assert sum(d[post].numel() for d in data)==global0.numel()
  reduction_mismatch=clip_mismatch=0;redmax=clipmax=0.;offset=0
  for rank in (0,1):
   actual=data[rank][post].reshape(-1);before=data[rank][pc].reshape(-1);after=data[rank][cl].reshape(-1)
   for start in range(0,actual.numel(),1<<20):
    stop=min(start+(1<<20),actual.numel());expected=((global0[offset+start:offset+stop].float()+global1[offset+start:offset+stop].float())*.5).to(actual.dtype);a=actual[start:stop]
    diff=(a.float()-expected.float()).abs();reduction_mismatch+=int(torch.count_nonzero(diff));redmax=max(redmax,float(diff.max()) if diff.numel() else 0.)
    expected_clip=before[start:stop]*coefficient;c=after[start:stop];cdiff=(c.float()-expected_clip.float()).abs();clip_mismatch+=int(torch.count_nonzero(cdiff));clipmax=max(clipmax,float(cdiff.max()) if cdiff.numel() else 0.)
   offset+=actual.numel()
  step_before=[float(d['pre_forward.optimizer.'+p+'.step']) for d in data];step_after=[float(d['post_step.optimizer.'+p+'.step']) for d in data]
  assert step_before==[2.,2.] and step_after==[3.,3.]
  results.append({'run':name,'parameter':p,'available':True,'reduction_mismatching_elements':reduction_mismatch,'reduction_max_abs_residual':redmax,'clip_mismatching_elements':clip_mismatch,'clip_max_abs_residual':clipmax,'optimizer_step_before':step_before,'optimizer_step_after':step_after})
  if len(results)%10==0:print('OPERATOR_CHECK',len(results),flush=True)
summary={'DEBUG_ONLY':True,'status':'PASS_AUDIT_COMPLETED','results':results,'all_reductions_match_fp32_two_rank_average':all(x.get('available') and x['reduction_mismatching_elements']==0 for x in results),'all_clips_match_actual_recorded_coefficient':all(x.get('available') and x['clip_mismatching_elements']==0 for x in results),'definition':'For each branch independently, compare actual reduced gradient with FP32 two-rank average cast to actual gradient dtype; compare actual clipped values with pre-clip vector multiplied by the directly observed coefficient. This audits the operators on captured inputs, not a natural-variation baseline or population causal effect.','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'completed_at':time.time()}
(L/'OPERATOR_AUDIT.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='results'},indent=2))
