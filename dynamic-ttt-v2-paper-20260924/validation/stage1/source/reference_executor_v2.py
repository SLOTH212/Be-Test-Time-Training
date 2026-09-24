#!/usr/bin/env python3
import argparse
import csv
import gc
import hashlib
import importlib.util
import itertools
import json
import math
import os
import statistics
import sys
import time
import traceback
import types
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ACTIONS = ['OFF','L0','L6','L12','L18','L24','ALL']
LAYERS = [0,6,12,18,24]
ACTION_LAYERS = {'OFF':[], 'L0':[0], 'L6':[6], 'L12':[12], 'L18':[18], 'L24':[24], 'ALL':LAYERS}
ACTION_MODE = {'OFF':'off','L0':'layer_0','L6':'layer_6','L12':'layer_12','L18':'layer_18','L24':'layer_24','ALL':'all'}
CHECKPOINT_HASH = 'f4e264ed6f5671f123d6c8a27c69bce3af5ffcba518c9e298398314c6147d9fd'
TOL = 1e-12

def now(): return datetime.now().astimezone().isoformat(timespec='seconds')
def canonical(obj): return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def atomic_text(path,text):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name('.'+path.name+'.tmp')
    with open(tmp,'w',encoding='utf-8') as f: f.write(text); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)
def atomic_json(path,obj): atomic_text(path,json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n')
def load_jsonl(path):
    with open(path,encoding='utf-8') as f:return [json.loads(x) for x in f if x.strip()]
def read_csv(path):
    with open(path,newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def write_csv(path,rows):
    rows=list(rows); fields=list(rows[0]) if rows else []; path=Path(path); tmp=path.with_name('.'+path.name+'.tmp')
    with open(tmp,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

class Context:
    def __init__(self, run):
        self.run=Path(run).resolve()
        self.code=self.run/'code'
        self.pipe=Path('/home/zonghan/ttt_phase_c_migrated_v1/ttt_phase_c_migration_20260813/data/dev_authority')
        self.ckpt=Path('/home/zonghan/ttt_phase_c_migrated_v1/ttt_phase_c_migration_20260813/checkpoint')
        self.scorer_path=Path('/home/zonghan/ttt_phase_c_migrated_v1/ttt_phase_c_migration_20260813/local_code_dependencies/opencompass_scorer/common.py')
        for p in [self.pipe/'manifests/dev_manifest.json',self.pipe/'benchmarks/dev/samples',self.ckpt,self.scorer_path]:
            if 'final_test' in str(p).lower(): raise RuntimeError('FINAL_TEST_PATH_REJECTED')
        spec=importlib.util.spec_from_file_location('frozen_ruler_scorer',self.scorer_path)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);self.score_fn=m.per_sample_score
        manifest=json.load(open(self.pipe/'manifests/dev_manifest.json',encoding='utf-8'));self.manifest=manifest
        cache={t:{r['sample_id']:r for r in load_jsonl(self.pipe/'benchmarks/dev/samples'/f'{t}.jsonl')} for t in manifest['tasks']}
        self.rows=[]
        for i,meta in enumerate(manifest['samples']):
            r=dict(cache[meta['task']][meta['sample_id']]);r['_sample_index']=i;self.rows.append(r)
        self.by_id={r['sample_id']:r for r in self.rows}
        self.baseline={}
        for action in ACTIONS:
            self.baseline[action]={r['sample_id']:r for r in load_jsonl(self.pipe/'evaluations/dev_modes/predictions'/ACTION_MODE[action]/'predictions.jsonl')}

def tensor_hash(x):
    import torch
    y=x.detach().contiguous().view(torch.uint16).cpu().numpy()
    return hashlib.sha256(y.tobytes()).hexdigest()

def load_model(ctx):
    import torch
    from transformers import AutoTokenizer
    sys.path.insert(0,str(ctx.code))
    from inference_model.hf_qwen3.configuration_qwen3 import Qwen3Config
    from inference_model.hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM
    cfg=Qwen3Config.from_pretrained(ctx.ckpt,local_files_only=True)
    cfg.ttt_mode=True;cfg.ttt_layers=list(LAYERS);cfg.ttt_chunk=1024;cfg.ttt_lr=1.0
    cfg.ttt_target='hidden_states';cfg.ttt_target_type='gated_next_position_hidden';cfg.ttt_proj=True;cfg.ttt_ntp_gate=True
    cfg.ttt_update_clip_norm=1e-5;cfg.use_cache=True
    tok=AutoTokenizer.from_pretrained(ctx.ckpt,local_files_only=True);tok.padding_side='left'
    model=Qwen3ForCausalLM.from_pretrained(ctx.ckpt,config=cfg,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval()
    for layer in LAYERS:
        if not hasattr(model.model.layers[layer].mlp,'ttt_conv'):raise RuntimeError(f'MISSING_TTT_MODULE layer={layer}')
    install_dynamic_forward(model)
    return model,tok

def install_dynamic_forward(model):
    import torch
    import torch.nn.functional as F
    from hf_models.hf_qwen3.ttt_state_core import TTTState,TTTStats,ttt_chunk_step
    for layer in LAYERS:
        mlp=model.model.layers[layer].mlp
        mlp._dynamic_sequence=None;mlp._branch_id=None;mlp._capture_identity=False;mlp._events=[]
        mlp._generation_forward_calls=0;mlp._captured_final_state=None
        def patched(self,x,t=None,past_w=None,enabled=True):
            h=self.act_fn(self.gate_proj(x))*self.up_proj(x)
            if not hasattr(self,'ttt_conv') or not enabled:raise RuntimeError('V2_CANDIDATE_STANDARD_PATH_FORBIDDEN')
            if x.shape[0]!=1:raise ValueError('TTT-on requires batch_size=1')
            present=self.down_proj.weight if past_w is None else past_w
            if t is None:
                self._generation_forward_calls+=1
                return F.linear(h,present,self.down_proj.bias),present
            full_len=(x.shape[1]//self.ttt_chunk)*self.ttt_chunk;n=full_len//self.ttt_chunk
            if self._dynamic_sequence is None or len(self._dynamic_sequence)!=n:raise RuntimeError(f'ACTION_SEQUENCE_LENGTH layer={self.layer_idx} n={n}')
            outputs=[];state=TTTState(present);stats=[];events=[];version=0
            projection=self.ttt_proj.weight if self.ttt_proj is not None else torch.eye(self.hidden_size,dtype=x.dtype,device=x.device)
            for j,start in enumerate(range(0,full_len,self.ttt_chunk)):
                end=start+self.ttt_chunk;action=self._dynamic_sequence[j];active=int(self.layer_idx) in ACTION_LAYERS[action]
                pre_hash=tensor_hash(state.weight) if self._capture_identity else None
                pre_identity=f'{self._branch_id}:L{self.layer_idx}:v{version}'
                if active:
                    out,proposal,stat=ttt_chunk_step(h[:,start:end],t[:,start:end],self.down_proj.weight,projection,self.ttt_conv.weight,
                        torch.ones((1,self.ttt_chunk),dtype=torch.bool,device=x.device),state,self.ttt_lr,True,
                        self.ttt_update_clip_norm,self.ttt_target_type,getattr(self,'ttt_ntp_gate',None))
                    state=proposal;version+=1
                else:
                    out=F.linear(h[:,start:end],state.weight,self.down_proj.bias)
                    stat=TTTStats(0,self.ttt_chunk-1,0.0,0.0,False)
                post_hash=tensor_hash(state.weight) if self._capture_identity else None
                events.append({'layer':int(self.layer_idx),'chunk_index':j+1,'selected_action':action,'active_update':active,
                    'pre_state_identity':pre_identity,'post_state_identity':f'{self._branch_id}:L{self.layer_idx}:v{version}',
                    'pre_state_hash':pre_hash,'post_state_hash':post_hash,'post_clip_delta_norm':float(stat.post_clip_norm),
                    'subsequent_recomputation':True})
                outputs.append(out);stats.append(stat)
            if full_len<x.shape[1]:outputs.append(F.linear(h[:,full_len:],state.weight,self.down_proj.bias))
            self.last_ttt_stats=stats;self._events=events;self._captured_final_state=state.weight.detach()
            return torch.cat(outputs,dim=1),state.weight
        mlp.forward=types.MethodType(patched,mlp)

def base_hash(model):
    import torch
    h=hashlib.sha256()
    for layer in LAYERS:
        x=model.model.layers[layer].mlp.down_proj.weight.detach().contiguous().view(torch.uint16).cpu().numpy();h.update(x.tobytes())
    return h.hexdigest()

class Evaluator:
    def __init__(self,ctx,model,tok,row,stage,sample_ordinal,total_samples,progress_path):
        import torch
        self.ctx=ctx;self.model=model;self.tok=tok;self.row=row;self.stage=stage
        self.sample_ordinal=sample_ordinal;self.total_samples=total_samples;self.progress_path=progress_path
        self.enc0=tok(row['input'],return_tensors='pt',add_special_tokens=True,truncation=False)
        self.n=int(self.enc0.input_ids.shape[1]);self.full_chunks=self.n//1024
        self.enc={k:v.cuda() for k,v in self.enc0.items()};self.records={};self.started=time.time();self.base_before=base_hash(model)
        self.torch=torch
    def evaluate(self,sequence,capture=False,force=False):
        seq=tuple(sequence);key='|'.join(seq)
        if key in self.records and not force and not capture:return self.records[key]
        if len(seq)!=self.full_chunks:raise RuntimeError('CHUNK_IDENTITY_MISMATCH')
        branch=canonical({'stage':self.stage,'sample_id':self.row['sample_id'],'sequence':seq,'capture':capture,'nonce':len(self.records) if force else 0})[:20]
        for layer in LAYERS:
            m=self.model.model.layers[layer].mlp;m._dynamic_sequence=list(seq);m._branch_id=branch;m._capture_identity=capture;m._events=[];m._generation_forward_calls=0
        # V2: action labels control writes only. All five candidates retain the same functional path.
        assert self.model.model.ttt_mode
        assert all(hasattr(self.model.model.layers[l].mlp,'ttt_conv') for l in LAYERS)
        self.torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        with self.torch.inference_mode():
            out=self.model.generate(**self.enc,max_new_tokens=self.row['max_new_tokens'],do_sample=False,num_beams=1,use_cache=True,pad_token_id=self.tok.eos_token_id)
        self.torch.cuda.synchronize();elapsed=time.perf_counter()-started
        pred=self.tok.decode(out[0,self.n:],skip_special_tokens=True);score=float(self.ctx.score_fn(self.row['family'],pred,self.row['reference']))
        counts={};events=[];gen_calls=0
        for layer in LAYERS:
            m=self.model.model.layers[layer].mlp;ev=list(m._events);events.extend(ev);gen_calls+=m._generation_forward_calls
            counts[str(layer)]=sum(int(x['active_update']) for x in ev)
            if len(ev)!=self.full_chunks:raise RuntimeError(f'EVENT_COUNT layer={layer} got={len(ev)}')
            want=sum(layer in ACTION_LAYERS[a] for a in seq)
            if counts[str(layer)]!=want:raise RuntimeError(f'ACTION_EXECUTION layer={layer} got={counts[str(layer)]} want={want}')
        if any(x['selected_action']!=seq[x['chunk_index']-1] for x in events):raise RuntimeError('CHUNK_ACTION_IDENTITY')
        rec={'sample_id':self.row['sample_id'],'task':self.row['task'],'sequence':list(seq),'sequence_hash':canonical(list(seq)),
            'score':score,'prediction':pred if capture else None,'prediction_hash':hashlib.sha256(pred.encode()).hexdigest(),
            'latency_seconds':elapsed,'prompt_token_count':self.n,'full_chunk_count':self.full_chunks,
            'layer_update_counts':counts,'generation_update_count':0,'generation_forward_calls':gen_calls,
            'gpu_peak_memory_bytes':int(self.torch.cuda.max_memory_allocated()),'branch_id':branch,
            'clean_full_replay':True,'recompute_downstream_updates':True,'stored_delta_stitching':False,
            'chunk_layer_events':events if capture else None,'timestamp':now()}
        if not force:self.records[key]=rec
        c=len(self.records)
        if c and (c==1 or c%10==0):
            best=max(x['score'] for x in self.records.values());elapsed_total=time.time()-self.started;rate=elapsed_total/max(c,1)
            progress={'stage':self.stage,'completed_samples':self.sample_ordinal-1,'total_samples':self.total_samples,
                'current_sample':self.row['sample_id'],'candidate_sequences_evaluated':c,'best_current_score':best,
                'fixed_oracle_score':max(float(self.ctx.baseline[a][self.row['sample_id']]['official_sample_score']) for a in ACTIONS),
                'additional_gain':best-max(float(self.ctx.baseline[a][self.row['sample_id']]['official_sample_score']) for a in ACTIONS),
                'gpu_memory_allocated_bytes':int(self.torch.cuda.memory_allocated()),'runtime_per_candidate_seconds':rate,
                'runtime_current_sample_seconds':elapsed_total,'updated_at':now()}
            atomic_json(self.progress_path,progress)
            if c%50==0:print(json.dumps(progress,ensure_ascii=False),flush=True)
        return rec
    def finish(self):
        after=base_hash(self.model)
        if after!=self.base_before:raise RuntimeError('BASE_WEIGHT_MUTATION')

def hamming(a,b):return sum(x!=y for x,y in zip(a,b))
def diverse_top(records,width):
    unique={tuple(r['sequence']):r for r in records};groups=defaultdict(list)
    for seq,r in unique.items():groups[r['score']].append((seq,r))
    selected=[]
    for score in sorted(groups,reverse=True):
        group=groups[score]
        while group and len(selected)<width:
            if selected:
                group.sort(key=lambda q:(-min(hamming(q[0],s[0]) for s in selected),canonical(list(q[0]))))
            else:group.sort(key=lambda q:canonical(list(q[0])))
            selected.append(group.pop(0))
        if len(selected)>=width:break
    return [r for _,r in selected]

def constants(n):return [tuple([a]*n) for a in ACTIONS]

def beam_search(ev,positions,initial_sequences=None,lookup_only=False):
    seeds=initial_sequences or constants(ev.full_chunks);all_records=[]
    for seq in seeds:all_records.append(ev.evaluate(seq))
    beam=list(all_records)
    for pos in positions:
        expanded=[]
        for rec in beam:
            base=list(rec['sequence'])
            for action in ACTIONS:
                seq=tuple(base[:pos]+[action]+base[pos+1:])
                expanded.append(ev.evaluate(seq))
        all_records.extend(expanded);beam=diverse_top(expanded,4)
    dedup={r['sequence_hash']:r for r in all_records}
    best=max(dedup.values(),key=lambda r:(r['score'],-sum(x!=y for x,y in zip(r['sequence'],r['sequence'][1:])),r['sequence_hash']))
    return best,list(dedup.values())

def baseline_fixed(ctx,sid):
    scores={a:float(ctx.baseline[a][sid]['official_sample_score']) for a in ACTIONS};best=max(scores.values())
    return scores,best,[a for a in ACTIONS if abs(scores[a]-best)<=TOL]

def verify_constants(ctx,ev):
    mismatches=[];recs=[]
    for action,seq in zip(ACTIONS,constants(ev.full_chunks)):
        rec=ev.evaluate(seq);old=ctx.baseline[action][ev.row['sample_id']]
        score_match=abs(rec['score']-float(old['official_sample_score']))<=TOL
        prediction_match=rec['prediction_hash']==hashlib.sha256(old['prediction'].encode()).hexdigest()
        if not(score_match and prediction_match):mismatches.append({'action':action,'score_match':score_match,'prediction_match':prediction_match})
        recs.append((action,rec,old,score_match,prediction_match))
    return recs,mismatches

def run_phase0(ctx,model,tok):
    chosen=[]
    for task in ctx.manifest['tasks']:chosen.extend([r for r in ctx.rows if r['task']==task][:2])
    output=[];progress=ctx.run/'state/progress_phase0.json';mismatch_count=0
    for i,row in enumerate(chosen,1):
        ev=Evaluator(ctx,model,tok,row,'phase0',i,len(chosen),progress);recs,mismatches=verify_constants(ctx,ev);ev.finish()
        for action,rec,old,sm,pm in recs:
            output.append({'task':row['task'],'sample_id':row['sample_id'],'action':action,'full_chunk_count':ev.full_chunks,
                'replay_score':rec['score'],'existing_score':old['official_sample_score'],'score_match':sm,'prediction_match':pm,
                'replay_prediction_hash':rec['prediction_hash'],'existing_prediction_hash':hashlib.sha256(old['prediction'].encode()).hexdigest(),
                'recompute_downstream_updates':True,'stored_delta_stitching':False})
        mismatch_count+=len(mismatches)
        if mismatches:
            write_csv(ctx.run/'tables/constant_replay.csv',output);atomic_json(ctx.run/'state/phase0_gate.json',{'passed':False,'mismatch_count':mismatch_count,'sample_id':row['sample_id'],'mismatches':mismatches});raise RuntimeError('CONSTANT_SEQUENCE_REPLAY_MISMATCH')
    # Cross-branch contamination: A, B, A with captured state identities.
    row=chosen[0];ev=Evaluator(ctx,model,tok,row,'phase0_contamination',1,1,progress)
    a=tuple(['L24']*ev.full_chunks);b=tuple(['ALL']*ev.full_chunks)
    r1=ev.evaluate(a,capture=True,force=True);ev.evaluate(b,capture=True,force=True);r2=ev.evaluate(a,capture=True,force=True);ev.finish()
    contamination=not(r1['prediction_hash']==r2['prediction_hash'] and r1['score']==r2['score'] and
        [x['post_state_hash'] for x in r1['chunk_layer_events']]==[x['post_state_hash'] for x in r2['chunk_layer_events']])
    if contamination:raise RuntimeError('CROSS_BRANCH_CONTAMINATION')
    write_csv(ctx.run/'tables/constant_replay.csv',output)
    atomic_json(ctx.run/'state/phase0_gate.json',{'passed':True,'mismatch_count':0,'sample_count':len(chosen),'replay_count':len(output),'cross_branch_contamination':False})
    atomic_text(ctx.run/'reports/replay_equivalence.md',f'# Constant-sequence replay equivalence\n\n- Samples: {len(chosen)} (two per task).\n- Constant trajectories: 7 per sample.\n- Score/prediction mismatches: 0.\n- Cross-branch A→B→A contamination: 0.\n- Score tolerance: {TOL}. Prediction equality: exact decoded string.\n')
    atomic_text(ctx.run/'audits/action_execution.md','# Action execution audit\n\nPhase 0 passed all per-layer expected update-count checks, per-chunk action identity checks, generation update count=0 checks, and constant prediction/score equivalence checks.\n')
    print(json.dumps({'phase':'phase0','passed':True,'mismatch_count':0,'records':len(output)}),flush=True)

def run_toy(ctx,model,tok):
    if not json.load(open(ctx.run/'state/phase0_gate.json'))['passed']:raise RuntimeError('PHASE0_NOT_PASSED')
    toys=read_csv(ctx.run/'data/toy_manifest.csv');summary=[];table=[];progress=ctx.run/'state/progress_toy.json'
    for i,t in enumerate(toys,1):
        row=ctx.by_id[t['sample_id']];ev=Evaluator(ctx,model,tok,row,'toy',i,len(toys),progress);tail=t['toy_tail_action'];n=ev.full_chunks
        exact_records=[]
        for prefix in itertools.product(ACTIONS,repeat=3):exact_records.append(ev.evaluate(tuple(prefix)+tuple([tail]*(n-3))))
        exact_best=max(r['score'] for r in exact_records)
        toy_seeds=[tuple([a]*3+[tail]*(n-3)) for a in ACTIONS]
        approx,visited=beam_search(ev,range(3),initial_sequences=toy_seeds,lookup_only=True)
        gap=exact_best-approx['score'];hit=abs(gap)<=TOL
        summary.append({'task':row['task'],'sample_id':row['sample_id'],'full_chunk_count':n,'tail_action':tail,
            'exact_candidate_count':343,'approx_unique_visited':len(visited),'exact_oracle_score':exact_best,
            'approx_search_score':approx['score'],'search_gap_to_exact':gap,'exact_optimum_found':hit,
            'exact_optimum_sequence_count':sum(abs(r['score']-exact_best)<=TOL for r in exact_records)})
        for r in exact_records:table.append({'task':row['task'],'sample_id':row['sample_id'],'prefix_sequence':'|'.join(r['sequence'][:3]),'tail_action':tail,'score':r['score'],'is_exact_optimum':abs(r['score']-exact_best)<=TOL})
        ev.finish();atomic_json(ctx.run/'state/results/toy'/f'{canonical(row["sample_id"])}.json',{'summary':summary[-1],'approx_visited':[r['sequence_hash'] for r in visited]})
    recall=sum(x['exact_optimum_found'] for x in summary)/len(summary);median_gap=statistics.median(x['search_gap_to_exact'] for x in summary)
    passed=recall>=6/7-TOL and median_gap<=TOL
    write_csv(ctx.run/'tables/toy_exact_enumeration.csv',table);write_csv(ctx.run/'tables/toy_exactness_summary.csv',summary)
    atomic_json(ctx.run/'state/toy_gate.json',{'passed':passed,'sample_count':len(summary),'exact_optimum_recall':recall,'median_gap':median_gap,'required_recall':6/7})
    atomic_text(ctx.run/'reports/exactness_toy_audit.md',f'# Search exactness toy audit\n\n- Samples: {len(summary)} (one preregistered hash-pilot sample per task).\n- Variable horizon: first 3 chunks; exact candidates/sample: 343.\n- Approximate beam width: 4.\n- Exact optimum score recall: {recall:.12f}.\n- Median exact-minus-approx gap: {median_gap:.12f}.\n- Gate: {"PASS" if passed else "FAIL"} (requires at least 6/7 recall and zero median gap).\n')
    print(json.dumps({'phase':'toy','passed':passed,'recall':recall,'median_gap':median_gap}),flush=True)
    if not passed:raise RuntimeError('TOY_EXACTNESS_GATE_FAILED')

def morphology(seq):
    switches=sum(a!=b for a,b in zip(seq,seq[1:]));distinct=len(set(seq));nonoff=set(seq)-{'OFF'}
    if len(set(seq))==1:category='FIXED_ACTION_ALREADY_OPTIMAL'
    elif 'OFF' in seq and len(nonoff)==1:category='OFF_INTERLEAVING'
    elif 'ALL' in nonoff and len(nonoff)>=2:category='ALL_SINGLE_MIXING'
    elif len({x for x in nonoff if x.startswith('L')})>=2 and 'ALL' not in nonoff:category='CROSS_LAYER_SWITCHING'
    else:category='COMPLEX_PATH'
    return {'action_switch_count':switches,'distinct_action_count':distinct,'off_chunk_fraction':seq.count('OFF')/len(seq),
        'all_chunk_fraction':seq.count('ALL')/len(seq),'sequence_category':category,
        **{f'{a.lower()}_chunk_fraction':seq.count(a)/len(seq) for a in ['L0','L6','L12','L18','L24']}}

def search_sample(ctx,model,tok,row,stage,ordinal,total,progress):
    ev=Evaluator(ctx,model,tok,row,stage,ordinal,total,progress);constant_records,mismatches=verify_constants(ctx,ev)
    if mismatches:raise RuntimeError(f'FORMAL_CONSTANT_REPLAY_MISMATCH {row["sample_id"]} {mismatches}')
    best,records=beam_search(ev,range(ev.full_chunks))
    scores,fixed,best_actions=baseline_fixed(ctx,row['sample_id'])
    if best['score']+TOL<fixed:raise RuntimeError(f'DYNAMIC_BELOW_FIXED {best["score"]} {fixed}')
    tied=[r for r in records if abs(r['score']-best['score'])<=TOL]
    tied.sort(key=lambda r:(sum(a!=b for a,b in zip(r['sequence'],r['sequence'][1:])),len(set(r['sequence'])),r['sequence_hash']))
    chosen=tied[0]
    capture=ev.evaluate(tuple(chosen['sequence']),capture=True,force=True)
    if capture['prediction_hash']!=chosen['prediction_hash'] or abs(capture['score']-chosen['score'])>TOL:raise RuntimeError('BEST_CAPTURE_REPLAY_MISMATCH')
    off_like=[]
    for r in records:
        acts=set(r['sequence']);nonoff=acts-{'OFF'}
        if len(nonoff)<=1:off_like.append(r['score'])
    best_off_like=max(off_like,default=fixed)
    m=morphology(chosen['sequence']);result={'stage':stage,'task':row['task'],'sample_id':row['sample_id'],'sample_index':row['_sample_index'],
        'full_chunk_count':ev.full_chunks,'fixed_scores':scores,'fixed_oracle_score':fixed,'best_fixed_actions':best_actions,
        'dynamic_search_score':chosen['score'],'dynamic_minus_fixed':chosen['score']-fixed,'off_score':scores['OFF'],
        'dynamic_gain_vs_off':chosen['score']-scores['OFF'],'best_sequence':chosen['sequence'],'best_sequence_hash':chosen['sequence_hash'],
        'best_prediction_hash':chosen['prediction_hash'],'candidate_count':len(records),'best_tie_count':len(tied),
        'best_off_interleaving_or_fixed_score':best_off_like,'cross_action_specific_gain':max(0.0,chosen['score']-max(fixed,best_off_like)),
        'fixed_negative':fixed<=scores['OFF']+TOL,'fixed_negative_rescued':fixed<=scores['OFF']+TOL and chosen['score']>scores['OFF']+TOL,
        **m,'selected_chunk_records':[],'base_weight_unchanged':True,'generation_update_count':0,
        'recompute_downstream_updates':True,'stored_delta_stitching':False,'checkpoint_hash':CHECKPOINT_HASH,'timestamp':now()}
    by_chunk=defaultdict(list)
    for event in capture['chunk_layer_events']:by_chunk[event['chunk_index']].append(event)
    for j in range(1,ev.full_chunks+1):
        action=chosen['sequence'][j-1];evs=by_chunk[j]
        result['selected_chunk_records'].append({'sample_id':row['sample_id'],'chunk_index':j,'selected_action':action,
            'active_update_layers':ACTION_LAYERS[action],'previous_chunk_action':chosen['sequence'][j-2] if j>1 else None,
            'current_action':action,'fast_weight_state_identity':{str(e['layer']):e['post_state_hash'] for e in evs},
            'subsequent_recomputation_flag':True})
    result['candidates']=[{k:r[k] for k in ['sequence','sequence_hash','score','prediction_hash','latency_seconds','layer_update_counts','gpu_peak_memory_bytes','clean_full_replay','recompute_downstream_updates','stored_delta_stitching']} for r in records]
    ev.finish();return result

def run_search_stage(ctx,model,tok,stage):
    if not json.load(open(ctx.run/'state/toy_gate.json'))['passed']:raise RuntimeError('TOY_GATE_NOT_PASSED')
    manifest_path=ctx.run/'tables/cohort_manifest.csv' if stage=='phase_a' else ctx.run/'tables/pilot_manifest.csv'
    manifest=read_csv(manifest_path);outdir=ctx.run/'state/results'/stage;progress=ctx.run/f'state/progress_{stage}.json'
    completed=0;started=time.time()
    for i,m in enumerate(manifest,1):
        row=ctx.by_id[m['sample_id']];path=outdir/f'{canonical(row["sample_id"])}.json'
        if path.exists():
            old=json.load(open(path));
            if old['sample_id']!=row['sample_id'] or old['checkpoint_hash']!=CHECKPOINT_HASH:raise RuntimeError('RESUME_IDENTITY_MISMATCH')
            completed+=1;continue
        try:result=search_sample(ctx,model,tok,row,stage,i,len(manifest),progress)
        except Exception as e:
            atomic_json(ctx.run/'state/failure.json',{'stage':stage,'sample_id':row['sample_id'],'error':f'{type(e).__name__}: {e}','traceback':traceback.format_exc(),'timestamp':now()});raise
        atomic_json(path,result);completed+=1
        elapsed=time.time()-started;eta=(elapsed/completed)*(len(manifest)-completed)
        atomic_json(progress,{'stage':stage,'completed_samples':completed,'total_samples':len(manifest),'current_sample':row['sample_id'],
            'candidate_sequences_evaluated':result['candidate_count'],'best_current_score':result['dynamic_search_score'],
            'fixed_oracle_score':result['fixed_oracle_score'],'additional_gain':result['dynamic_minus_fixed'],
            'gpu_memory_allocated_bytes':int(__import__('torch').cuda.memory_allocated()),'runtime_per_sample_seconds':elapsed/completed,
            'eta_seconds':eta,'updated_at':now()})
        print(json.dumps({'stage':stage,'completed':completed,'total':len(manifest),'sample_id':row['sample_id'],
            'candidates':result['candidate_count'],'fixed':result['fixed_oracle_score'],'dynamic':result['dynamic_search_score'],
            'additional':result['dynamic_minus_fixed'],'runtime_seconds':elapsed,'eta_seconds':eta},ensure_ascii=False),flush=True)
    atomic_json(ctx.run/f'state/{stage}_gate.json',{'passed':True,'completed_samples':completed,'total_samples':len(manifest),'completed_at':now()})

def main():
    p=argparse.ArgumentParser();p.add_argument('run_root');p.add_argument('stage',choices=['phase0','toy','phase_a','phase_b']);a=p.parse_args()
    ctx=Context(a.run_root)
    if not (ctx.run/'state/design.json').exists():raise RuntimeError('MISSING_PREREGISTRATION')
    before=sha(ctx.ckpt/'model.safetensors');model,tok=load_model(ctx)
    try:
        {'phase0':run_phase0,'toy':run_toy,'phase_a':lambda c,m,t:run_search_stage(c,m,t,'phase_a'),
         'phase_b':lambda c,m,t:run_search_stage(c,m,t,'phase_b')}[a.stage](ctx,model,tok)
    finally:
        del model;gc.collect();__import__('torch').cuda.empty_cache()
    after=sha(ctx.ckpt/'model.safetensors')
    if before!=after:raise RuntimeError('CHECKPOINT_DISK_MUTATED')
    print(json.dumps({'stage':a.stage,'complete':True,'checkpoint_hash_before':before,'checkpoint_hash_after':after}),flush=True)
if __name__=='__main__':main()
