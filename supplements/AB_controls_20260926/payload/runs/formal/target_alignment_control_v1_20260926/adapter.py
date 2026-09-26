"""Independent exact-copy adapter: only active-write call routed to optional controller."""
import types
from executor import LAYERS,ACTION_LAYERS,tensor_hash

def install_adapter(model, controller):
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
                    out,proposal,stat=controller.step(h[:,start:end],t[:,start:end],self.down_proj.weight,projection,self.ttt_conv.weight,
                        torch.ones((1,self.ttt_chunk),dtype=torch.bool,device=x.device),state,self.ttt_lr,True,
                        self.ttt_update_clip_norm,self.ttt_target_type,getattr(self,'ttt_ntp_gate',None),_layer=int(self.layer_idx),_chunk=j+1,_action=action)
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
