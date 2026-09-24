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
CHECKPOINT_HASH = 'ca51ee90d20ee09efbe0b31dd21bf9faa70275508316d1ba7cb1387ee4f0725f'
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
    if sha(ctx.ckpt/'model.safetensors') != CHECKPOINT_HASH: raise RuntimeError('CHECKPOINT_IDENTITY_MISMATCH')
    if cfg.ttt_chunk != ctx.training_chunk or list(cfg.ttt_layers) != LAYERS: raise RuntimeError('MODEL_PROTOCOL_MISMATCH')
    if cfg.ttt_lr != 1.0 or cfg.ttt_update_clip_norm != 1e-5: raise RuntimeError('UPDATE_PROTOCOL_MISMATCH')
    cfg.ttt_chunk=ctx.chunk
    cfg.ttt_mode=True;cfg.ttt_layers=list(LAYERS);cfg.ttt_lr=1.0
    cfg.ttt_target='hidden_states';cfg.ttt_target_type='gated_next_position_hidden';cfg.ttt_proj=True;cfg.ttt_ntp_gate=True
    cfg.ttt_update_clip_norm=1e-5;cfg.use_cache=True
    tok=AutoTokenizer.from_pretrained(ctx.ckpt,local_files_only=True);tok.padding_side='left'
    model,info=Qwen3ForCausalLM.from_pretrained(ctx.ckpt,config=cfg,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True,output_loading_info=True)
    if any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']): raise RuntimeError(f'CHECKPOINT_LOADING_MISMATCH {info}')
    model=model.cuda().eval()
    model._loading_info=info
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
                mech=getattr(self,'_mechanism',{})
                if j==mech.get('tau'):
                    self._boundary_before=state.weight.detach().cpu().clone()
                    if int(self.layer_idx) in mech.get('reset_layers',[]):state=TTTState(self.down_proj.weight)
                    if int(self.layer_idx) in mech.get('inject',{}):state=TTTState(mech['inject'][int(self.layer_idx)].to(device=x.device,dtype=x.dtype))
                    self._boundary_after=state.weight.detach().cpu().clone()
                    self._intervention_events.append({'layer':int(self.layer_idx),'chunk_index':j+1,'before':tensor_hash(self._boundary_before),'after':tensor_hash(self._boundary_after),'reset':int(self.layer_idx) in mech.get('reset_layers',[]),'injected':int(self.layer_idx) in mech.get('inject',{})})
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
        self.n=int(self.enc0.input_ids.shape[1]);self.chunk_size=int(model.config.ttt_chunk);self.full_chunks=self.n//self.chunk_size
        self.enc={k:v.cuda() for k,v in self.enc0.items()};self.records={};self.started=time.time();self.base_before=base_hash(model)
        self.torch=torch
    def evaluate(self,sequence,capture=False,force=False):
        seq=tuple(sequence);key='|'.join(seq)
        if key in self.records and not force and not capture:return self.records[key]
        if len(seq)!=self.full_chunks:raise RuntimeError('CHUNK_IDENTITY_MISMATCH')
        if any(a not in ACTIONS for a in seq):raise ValueError('INVALID_ACTION')
        branch=canonical({'stage':self.stage,'sample_id':self.row['sample_id'],'sequence':seq,'capture':capture,'nonce':len(self.records) if force else 0})[:20]
        for layer in LAYERS:
            m=self.model.model.layers[layer].mlp;m._dynamic_sequence=list(seq);m._branch_id=branch;m._capture_identity=capture;m._events=[];m._generation_forward_calls=0;m._captured_final_state=None;m._boundary_before=None;m._boundary_after=None;m._intervention_events=[]
        # V2: action labels control writes only. All five candidates retain the same functional path.
        assert self.model.model.ttt_mode
        assert all(hasattr(self.model.model.layers[l].mlp,'ttt_conv') for l in LAYERS)
        self.torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        with self.torch.inference_mode():
            out=self.model.generate(**self.enc,**({'past_key_values':self.ctx.mechanism_cache} if getattr(self.ctx,'mechanism_cache',None) is not None else {}),max_new_tokens=self.row['max_new_tokens'],do_sample=False,num_beams=1,use_cache=True,pad_token_id=self.tok.eos_token_id)
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
            'latency_seconds':elapsed,'prompt_token_count':self.n,'full_chunk_count':self.full_chunks,'ttt_chunk_size':self.chunk_size,'checkpoint_hash':CHECKPOINT_HASH,
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
                'fixed_oracle_score':max((self.records['|'.join(seq)]['score'] for seq in constants(self.full_chunks) if '|'.join(seq) in self.records),default=0.0),
                'additional_gain':best-max((self.records['|'.join(seq)]['score'] for seq in constants(self.full_chunks) if '|'.join(seq) in self.records),default=0.0),
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
