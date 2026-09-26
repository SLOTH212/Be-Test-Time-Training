# Authority executor function subset; see adapter.py and port provenance.
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

class Evaluator:
    def __init__(self,ctx,model,tok,row,stage,sample_ordinal,total_samples,progress_path):
        import torch
        self.ctx=ctx;self.model=model;self.tok=tok;self.row=row;self.stage=stage
        self.sample_ordinal=sample_ordinal;self.total_samples=total_samples;self.progress_path=progress_path
        self.enc0=tok(row['input'],return_tensors='pt',add_special_tokens=True,truncation=False)
        self.n=int(self.enc0.input_ids.shape[1]);self.full_chunks=self.n//1024
        self.device=next(model.parameters()).device;self.enc={k:v.to(self.device) for k,v in self.enc0.items()};self.records={};self.started=time.time();self.base_before=base_hash(model)
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
        (self.torch.cuda.reset_peak_memory_stats() if self.device.type=='cuda' else 0);started=time.perf_counter()
        with self.torch.inference_mode():
            out=self.model.generate(**self.enc,max_new_tokens=self.row['max_new_tokens'],do_sample=False,num_beams=1,use_cache=True,pad_token_id=self.tok.eos_token_id)
        (self.torch.cuda.synchronize() if self.device.type=='cuda' else 0);elapsed=time.perf_counter()-started
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
            'gpu_peak_memory_bytes':int((self.torch.cuda.max_memory_allocated() if self.device.type=='cuda' else 0)),'branch_id':branch,
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
                'gpu_memory_allocated_bytes':int((self.torch.cuda.memory_allocated() if self.device.type=='cuda' else 0)),'runtime_per_candidate_seconds':rate,
                'runtime_current_sample_seconds':elapsed_total,'updated_at':now()}
            atomic_json(self.progress_path,progress)
            if c%50==0:print(json.dumps(progress,ensure_ascii=False),flush=True)
        return rec
    def finish(self):
        after=base_hash(self.model)
        if after!=self.base_before:raise RuntimeError('BASE_WEIGHT_MUTATION')
