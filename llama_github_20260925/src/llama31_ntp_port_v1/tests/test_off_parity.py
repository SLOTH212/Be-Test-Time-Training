"""Hard OFF gate; run before any NTP update tests."""
import copy, importlib.util, json, sys
from pathlib import Path
import torch
W=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(W/'source'))
from hf_models.hf_llama import LlamaConfig, LlamaForCausalLM as Train
from inference_model.hf_llama import LlamaForCausalLM as Infer
from transformers import LlamaForCausalLM as Vanilla

def historical(mode):
    root=W/'reference/inplace_ttt'/('hf_models/hf_llama' if mode=='training' else 'inference_model/hf_llama3')
    name='historical_'+mode
    spec=importlib.util.spec_from_file_location(name,root/'__init__.py',submodule_search_locations=[str(root)])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    spec=importlib.util.spec_from_file_location(name+'.modeling_llama',root/'modeling_llama.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module.LlamaForCausalLM

def config(**kw):
    values=dict(vocab_size=37,hidden_size=16,intermediate_size=24,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=65536,attention_dropout=0.0,pad_token_id=0,bos_token_id=1,eos_token_id=2,ttt_mode=False,ttt_layers=[0],ttt_chunk=1024,ttt_proj=True,ttt_lr=1.0,ttt_target='hidden_states',ttt_target_type='gated_next_position_hidden',ttt_ntp_gate=True,ttt_ntp_gate_init='zero',ttt_ntp_within_chunk_only=True,ttt_ntp_cross_chunk=False,ttt_ntp_cross_record=False,ttt_ntp_cross_document=False,ttt_document_metadata_required=True,ttt_backend='authoritative_document_chunk_loop',ttt_update_clip_norm=None)
    values.update(kw);c=LlamaConfig(**values);c._attn_implementation='sdpa';return c

def compare(a,b,tol=1e-6):
    torch.testing.assert_close(a,b,atol=tol,rtol=tol)
    return float((a.float()-b.float()).detach().abs().max())

def run():
    torch.set_num_threads(2);torch.manual_seed(1729)
    c=config();reference=Vanilla(copy.deepcopy(c)).eval();ids=torch.tensor([[1,4,8,3,11,7,9]])
    result={'atol':1e-6,'rtol':1e-6,'reference':'installed Transformers 4.57.3 vanilla AND exact historical be232 OFF','metrics':{}}
    for variant,rope in [('default',None),('llama31',dict(rope_type='llama3',factor=8.,low_freq_factor=1.,high_freq_factor=4.,original_max_position_embeddings=8192))]:
        c=config(rope_scaling=rope)
        reference=Vanilla(copy.deepcopy(c)).eval()
        for mode,cls in [('training',Train),('inference',Infer)]:
            candidate=cls(copy.deepcopy(c)).eval();hist=historical(mode)(copy.deepcopy(c)).eval()
            candidate.load_state_dict(reference.state_dict(),strict=True);hist.load_state_dict(reference.state_dict(),strict=True)
            with torch.no_grad():
                expected=reference(ids,labels=ids,use_cache=True,output_hidden_states=True)
                for tag,model in [('adapted',candidate),('historical',hist)]:
                    output=model(ids,labels=ids,use_cache=True,output_hidden_states=True)
                    error=compare(output.logits,expected.logits);compare(output.loss,expected.loss)
                    assert len(output.hidden_states)==len(expected.hidden_states)
                    for a,b in zip(output.hidden_states,expected.hidden_states):compare(a,b)
                    for a,b in zip(output.past_key_values.layers,expected.past_key_values.layers):
                        compare(a.keys,b.keys);compare(a.values,b.values)
                    incremental=[];cache=None
                    for i in range(ids.shape[1]):
                        step=model(ids[:,i:i+1],past_key_values=cache,cache_position=torch.tensor([i]),use_cache=True)
                        cache=step.past_key_values;incremental.append(step.logits)
                        assert cache.get_seq_length()==i+1
                    compare(torch.cat(incremental,1),expected.logits)
                    tokens=model.generate(input_ids=ids,max_new_tokens=4,do_sample=False)
                    compare(tokens,reference.generate(input_ids=ids,max_new_tokens=4,do_sample=False),0)
                    result['metrics'][variant+'_'+mode+'_'+tag+'_max_logit_error']=error
    result.update(LLAMA_OFF_MODEL_PARITY='PASS',LLAMA_INFERENCE_OFF_PARITY='PASS',LLAMA_CACHE_COMPATIBILITY_OFF='PASS')
    return result
if __name__=='__main__':
    output=run();(W/'audits/off_parity.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))
