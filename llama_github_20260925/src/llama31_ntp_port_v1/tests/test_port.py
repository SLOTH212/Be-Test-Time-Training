"""Independent tiny CPU oracles and real model interface checks; no formal run."""
import ast, copy, importlib.util, inspect, json, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import torch
import torch.nn.functional as F
W=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(W),str(W/'source'),str(W/'tests')]
from test_off_parity import config, compare
from hf_models.hf_llama import LlamaForCausalLM as Train
from hf_models.hf_llama.modeling_llama import LlamaMLP as TrainMLP
from inference_model.hf_llama import LlamaForCausalLM as Infer
from inference_model.hf_llama.modeling_llama import LlamaMLP as InferMLP, TTTDynamicCache
from ntp_core import ttt_state_core as core
from adapters.model_family import build, configure, decoder_blocks, model_classes
from adapters.inference import action_layers, action_scope, fresh_cache
from ntp_core.validation import validate_config
GATES={};DETAILS={}

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
QCORE=load('frozen_qwen_core',W/'reference/qwen_ntp/hf_models/hf_qwen3/ttt_state_core.py')

def oracle(h,x,w,p,g,valid,lr=1.,clip=None):
    """Deliberately independent scalar pair loop and matrix outer products.

    No adapted state, target-pair, or update helper is used.
    """
    targets=x[:,1:]*g
    projected=targets @ p
    delta=torch.zeros_like(w)
    for i in range(h.shape[1]-1):
        if bool(valid[0,i] and valid[0,i+1]):
            delta=delta+torch.outer(projected[0,i],h[0,i]) * lr
    raw=delta.clone()
    if clip is not None:
        n=delta.float().square().sum().sqrt()
        factor=min(1.,clip/max(float(n.detach()),torch.finfo(torch.float32).tiny))
        delta=delta*torch.tensor(factor,dtype=delta.dtype)
    return F.linear(h,w),w+delta,targets,projected,raw,delta

def nonzero_gate(model):
    with torch.no_grad():
        for layer in model.model.layers:
            if hasattr(layer.mlp,'ttt_ntp_gate'):layer.mlp.ttt_ntp_gate.fill_(0.7)

class PortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        off=json.loads((W/'audits/off_parity.json').read_text())
        assert off['LLAMA_OFF_MODEL_PARITY']=='PASS' and off['LLAMA_INFERENCE_OFF_PARITY']=='PASS','HARD OFF GATE'
        torch.set_num_threads(2)
    def setUp(self):torch.manual_seed(719)

    def test_01_update_oracle(self):
        mlp=TrainMLP(config(ttt_mode=True,ttt_chunk=4),0).double()
        with torch.no_grad():mlp.ttt_ntp_gate.copy_(torch.linspace(-.6,.7,16,dtype=torch.float64))
        x=torch.randn(1,9,16,dtype=torch.float64);valid=torch.tensor([[1,1,0,1]],dtype=torch.bool)
        h=mlp.act_fn(mlp.gate_proj(x))*mlp.up_proj(x)
        w=mlp.down_proj.weight;p=mlp.ttt_proj.weight;g=mlp.ttt_ntp_gate
        for clip in [None,1e-5]:
            expected=oracle(h[:,:4],x[:,:4],w,p,g,valid,clip=clip)
            target,pairs=core.build_chunk_local_ntp_pairs(x[:,:4],valid)
            compare(target*g,expected[2],1e-12);compare((target*g)@p,expected[3],1e-12)
            out,state,stat=core.ttt_chunk_step(h[:,:4],x[:,:4],w,p,mlp.ttt_conv.weight,valid,learning_rate=1.,target_type=core.GATED_NTP_TARGET,target_gate=g,delta_clip_norm=clip)
            compare(out,expected[0],1e-12);compare(state.weight,expected[1],1e-12)
            qout,qstate,qstat=QCORE.ttt_chunk_step(h[:,:4],x[:,:4],w,p,mlp.ttt_conv.weight,valid,learning_rate=1.,target_type=core.GATED_NTP_TARGET,target_gate=g,delta_clip_norm=clip)
            compare(state.weight,qstate.weight,0);self.assertEqual(stat.__dict__,qstat.__dict__)
            self.assertEqual(stat.valid_tokens,1)
        # Exercise adapted MLP, not just a shared helper; all three chunk outputs use exclusive prefix state.
        expected=[];state=w
        for s in range(0,9,4):
            n=min(4,9-s);o,state,*_=oracle(h[:,s:s+n],x[:,s:s+n],state,p,g,torch.ones(1,n,dtype=torch.bool));expected.append(o)
        actual=mlp(x,x,document_boundaries=[(0,9)])
        compare(actual,torch.cat(expected,1),1e-11)
        # Contextual target is post-attention normalized hidden, captured from an actual decoder.
        model=Train(config(ttt_mode=True,ttt_chunk=4)).eval();capture={}
        def hook(module,args,kwargs):capture.update(x=args[0].detach(),target=kwargs['t'].detach())
        handle=model.model.layers[0].mlp.register_forward_pre_hook(hook,with_kwargs=True)
        model(torch.tensor([[1,4,5,6,7,8,9,3,2]]),document_boundaries=[(0,9)],use_cache=False);handle.remove()
        compare(capture['x'],capture['target'],0)
        self.assertFalse(torch.equal(capture['target'],model.model.embed_tokens(torch.tensor([[1,4,5,6,7,8,9,3,2]]))))
        GATES['LLAMA_NTP_UPDATE_ORACLE']='PASS'

    def test_02_apply_and_boundaries(self):
        mlp=TrainMLP(config(ttt_mode=True,ttt_chunk=4),0).double();mlp.ttt_ntp_gate.data.fill_(1)
        x=torch.randn(1,11,16,dtype=torch.float64);target=torch.randn_like(x)
        out=mlp(x,target,document_boundaries=[(0,6),(6,11)])
        h=mlp.act_fn(mlp.gate_proj(x))*mlp.up_proj(x)
        expected=[]
        for start,end in [(0,6),(6,11)]:
            state=mlp.down_proj.weight
            for s in range(start,end,4):
                e=min(s+4,end);v=torch.ones(1,e-s,dtype=torch.bool)
                o,state,*_=oracle(h[:,s:e],target[:,s:e],state,mlp.ttt_proj.weight,mlp.ttt_ntp_gate,v);expected.append(o)
        compare(out,torch.cat(expected,1),1e-11)
        self.assertEqual([[s.valid_tokens for s in d] for d in mlp.last_ttt_stats],[[3,1],[3,0]])
        altered=target.clone();altered[:,2:4]+=100
        changed=mlp(x,altered,document_boundaries=[(0,6),(6,11)])
        compare(out[:,:4],changed[:,:4],0);self.assertGreater(float((out[:,4:6]-changed[:,4:6]).abs().max()),0)
        compare(out[:,6:],changed[:,6:],0)
        # Inference complete chunks only; the partial tail is forward-only.
        infer=InferMLP(config(ttt_mode=True,ttt_chunk=4,ttt_update_clip_norm=1e-5),0).double();infer.load_state_dict(mlp.state_dict())
        y,state=infer(x[:,:6],target[:,:6]);self.assertEqual(len(infer.last_ttt_stats),1)
        ref=oracle(h[:,:4],target[:,:4],mlp.down_proj.weight,mlp.ttt_proj.weight,mlp.ttt_ntp_gate,torch.ones(1,4,dtype=torch.bool),clip=1e-5)
        compare(y[:,:4],ref[0],1e-12);compare(state,ref[1],1e-12);compare(y[:,4:],F.linear(h[:,4:6],state),1e-12)
        GATES.update(APPLY_THEN_UPDATE_PARITY='PASS',LLAMA_BOUNDARY_SEMANTICS='PASS')

    def test_03_gradient_checkpointing_roundtrip(self):
        model,c=build(model_family='llama',ttt_layers=[0],chunk_size=4,config=config(),dtype=torch.float32)
        ids=torch.tensor([[1,3,4,5,6,7,8,9,10]])
        batch=dict(input_ids=ids,labels=ids,attention_mask=torch.ones_like(ids),document_boundaries=[(0,9)],use_cache=False)
        params=list(model.named_parameters(remove_duplicate=False));self.assertEqual(len(params),len({id(p) for _,p in params}))
        result=model(**batch);result.loss.backward();mlp=model.model.layers[0].mlp
        self.assertTrue(torch.equal(mlp.ttt_ntp_gate,torch.zeros_like(mlp.ttt_ntp_gate)))
        self.assertGreater(float(mlp.ttt_ntp_gate.grad.norm()),0)
        self.assertIsNotNone(mlp.ttt_proj.weight.grad);self.assertEqual(float(mlp.ttt_proj.weight.grad.norm()),0)
        self.assertIsNone(mlp.ttt_conv.weight.grad)
        self.assertGreater(float(mlp.down_proj.weight.grad.norm()),0)
        # Zero-initialized gate intentionally blocks projection gradient until gate moves.
        model.zero_grad(set_to_none=True);nonzero_gate(model)
        plain=Train(copy.deepcopy(c));plain.load_state_dict(model.state_dict());plain.train()
        a=model(**batch);b=plain(**batch);compare(a.logits,b.logits,1e-6)
        a.loss.backward();b.loss.backward()
        for (name,p),(other,q) in zip(model.named_parameters(),plain.named_parameters()):
            self.assertEqual(name,other)
            if '.ttt_conv.' in name:self.assertIsNone(p.grad);continue
            self.assertIsNotNone(p.grad,name);self.assertTrue(torch.isfinite(p.grad).all(),name)
            compare(p.grad,q.grad,1e-6)
        self.assertGreater(float(mlp.ttt_proj.weight.grad.norm()),0)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        for block in decoder_blocks(model):
            self.assertTrue(block.gradient_checkpointing)
            self.assertFalse(block._gradient_checkpointing_func.keywords['use_reentrant'])
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'state.pt';torch.save(model.state_dict(),f)
            reloaded=Train(copy.deepcopy(c));keys=reloaded.load_state_dict(torch.load(f,weights_only=True),strict=True)
            self.assertFalse(keys.missing_keys or keys.unexpected_keys)
            model.eval();reloaded.eval();compare(model(**batch).logits,reloaded(**batch).logits,0)
        DETAILS['gradient_flow']={'gate_zero_init_projection_grad':0.,'after_nonzero_gate_projection_grad':float(mlp.ttt_proj.weight.grad.norm()),'conv_grad':None,'all_base_parameters_require_grad':True,'checkpointed_vs_plain_gradients':'matched'}
        GATES.update(LLAMA_TRAINING_GRADIENT_FLOW='PASS',LLAMA_GRADIENT_CHECKPOINTING_STATUS='PASS',LLAMA_STATE_DICT_ROUNDTRIP='PASS')

    def test_04_inference_clip_cache_reset_generation(self):
        model,c=build(model_family='llama',ttt_layers=[0],chunk_size=4,config=config(),inference=True,dtype=torch.float64)
        nonzero_gate(model);model.eval()
        ids=torch.tensor([[1,3,4,5,6,7,8,9,10,11]])
        before={n:p.clone() for n,p in model.state_dict().items()}
        with torch.no_grad():
            full=model(ids,use_cache=False).logits
            cached=model(ids,use_cache=True);compare(full,cached.logits,1e-10)
            # Multi-token incremental prompt blocks are mathematically comparable;
            # single-token calls after position zero are generation in the authority.
            cache=None;pieces=[]
            for s,e in [(0,4),(4,8),(8,10)]:
                o=model(ids[:,s:e],past_key_values=cache,use_cache=True,cache_position=torch.arange(s,e));pieces.append(o.logits);cache=o.past_key_values
            compare(torch.cat(pieces,1),full,1e-9)
            compare(cache.ttt_states[0][2],cached.past_key_values.ttt_states[0][2],1e-10)
            self.assertEqual(cache.get_seq_length(),10)
            weight=cache.ttt_states[0][2].clone();counts=[]
            original=core.ttt_chunk_step
            def counted(*a,**kw):counts.append(1);return original(*a,**kw)
            with patch.object(core,'ttt_chunk_step',counted):
                for pos in range(10,17):
                    o=model(torch.tensor([[3]]),past_key_values=cache,use_cache=True,cache_position=torch.tensor([pos]));cache=o.past_key_values
                    compare(weight,cache.ttt_states[0][2],0)
            self.assertEqual(len(counts),0);self.assertEqual(cache.get_seq_length(),17)
            self.assertIsNone(cache.ttt_states[0][0]);self.assertIsNone(cache.ttt_states[0][1])
            # Existing nonzero fast weight must actually be applied to continuation.
            mlp=model.model.layers[0].mlp;x=torch.randn(1,1,16,dtype=torch.float64)
            applied,w=mlp(x,None,weight);h=mlp.act_fn(mlp.gate_proj(x))*mlp.up_proj(x)
            compare(applied,F.linear(h,weight),0)
            # sample A, reset, B against a fresh reconstructed model.
            b=torch.tensor([[1,12,13,14,15,16]])
            reset=fresh_cache(model);self.assertEqual(reset.get_seq_length(),0)
            self.assertTrue(all(v==(None,None,None) for v in reset.ttt_states))
            after=model(b,past_key_values=reset,use_cache=True)
            fresh=Infer(copy.deepcopy(c)).double().eval();fresh.load_state_dict(model.state_dict())
            expected=fresh(b,use_cache=True);compare(after.logits,expected.logits,0)
            compare(after.past_key_values.ttt_states[0][2],expected.past_key_values.ttt_states[0][2],0)
            first=model.generate(input_ids=b,max_new_tokens=7,do_sample=False)
            model.generate(input_ids=ids,max_new_tokens=2,do_sample=False)
            compare(first,model.generate(input_ids=b,max_new_tokens=7,do_sample=False),0)
            for n,p in model.state_dict().items():compare(p,before[n],0)
        mlp=InferMLP(config(ttt_mode=True,ttt_chunk=4,ttt_update_clip_norm=1e-5),0).double()
        mlp.ttt_ntp_gate.data.fill_(3);x=torch.randn(1,4,16,dtype=torch.float64)*50
        y,state=mlp(x,x)
        stat=mlp.last_ttt_stats[0];self.assertGreater(stat.pre_clip_norm,1e-5);self.assertLessEqual(stat.post_clip_norm,1e-5*(1+1e-6))
        h=mlp.act_fn(mlp.gate_proj(x))*mlp.up_proj(x)
        expected=oracle(h,x,mlp.down_proj.weight,mlp.ttt_proj.weight,mlp.ttt_ntp_gate,torch.ones(1,4,dtype=torch.bool),clip=1e-5)
        compare(state,expected[1],1e-12)
        from inference_model.hf_qwen3.modeling_qwen3 import Qwen3MLP
        delta=expected[4];compare(mlp._clip_ttt_update(delta),Qwen3MLP._clip_ttt_update(mlp,delta),0)
        DETAILS['inference_clip']={'pre_clip_norm':stat.pre_clip_norm,'post_clip_norm':stat.post_clip_norm,'threshold':1e-5,'float32_norm_rounding_tolerance':1e-11}
        GATES.update(LLAMA_CACHE_COMPATIBILITY='PASS',CROSS_SAMPLE_STATE_LEAKAGE_N=0,LLAMA_INFERENCE_CLIP_PARITY='PASS',GENERATION_UPDATE_SEMANTICS_PARITY='PASS')

    def test_05_stage_interfaces(self):
        model,c=build(model_family='llama',ttt_layers=[0],chunk_size=4,config=config(),dtype=torch.bfloat16)
        ids=torch.tensor([[1,3,4,5,6,7,8,9,10]])
        batch=dict(input_ids=ids,labels=ids,attention_mask=torch.ones_like(ids),document_boundaries=[(0,9)])
        loss=model(**batch).loss;self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertEqual(next(model.parameters()).dtype,torch.bfloat16)
        with tempfile.TemporaryDirectory() as d:
            # Model-only HF parent export, strict Stage2 load; no optimizer is serialized.
            model.save_pretrained(d,safe_serialization=True)
            fresh,c2=build(d,model_family='llama',ttt_layers=[0],chunk_size=4,dtype=torch.bfloat16)
            keys=fresh.load_state_dict(model.state_dict(),strict=True);self.assertFalse(keys.missing_keys or keys.unexpected_keys)
            self.assertFalse(any('optimizer' in p.name for p in Path(d).iterdir()))
            # Common runtime owns optimizer; a tiny test fixture merely checks fresh state.
            opt=torch.optim.AdamW(fresh.parameters(),lr=5e-6,betas=(.9,.95),eps=1e-8,weight_decay=.1)
            self.assertFalse(opt.state)
            scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
            self.assertEqual(scheduler.get_last_lr(),[5e-6])
            model.eval();fresh.eval();compare(model(**batch).logits,fresh(**batch).logits,0)
            # Run the exact frozen answer-mask function. Substitute only its CUDA fused
            # kernel with a differentiable CPU reference; no mask/loss logic is changed.
            loss_module=load('frozen_answer_loss',W/'reference/stage2/answer_context_fused_loss.py')
            class CPUFused:
                def __init__(self,ignore_index,reduction,accum_dtype):self.ignore=ignore_index;self.reduction=reduction
                def __call__(self,weight,hidden,labels):return F.cross_entropy(F.linear(hidden,weight).float(),labels,ignore_index=self.ignore,reduction=self.reduction)
            answer=torch.tensor([[0,0,0,0,1,1,0,1,0]],dtype=torch.bool);context=~answer
            with patch.object(loss_module,'LigerFusedLinearCrossEntropyLoss',CPUFused):
                fresh.zero_grad(set_to_none=True)
                total,a,ct,counts=loss_module.single_decoder_fused_answer_context(fresh,batch,answer,context)
                logits=fresh(**batch).logits[:,:-1].float();ce=F.cross_entropy(logits.reshape(-1,37),ids[:,1:].reshape(-1),reduction='none')
                expected=ce[answer[:,1:].flatten()].mean()+.1*ce[context[:,1:].flatten()].mean()
                compare(total,expected,1e-6);total.backward()
                self.assertGreater(float(fresh.lm_head.weight.grad.norm()),0)
                self.assertEqual(counts['decoder_forward_count'],1);self.assertEqual(counts['answer_positions'],3)
            # State-dict naming also supports strict training -> inference reconstruction.
            inf,ic=build(model_family='llama',ttt_layers=[0],chunk_size=4,config=copy.deepcopy(c),inference=True,dtype=torch.bfloat16)
            keys=inf.load_state_dict(model.state_dict(),strict=True);self.assertFalse(keys.missing_keys or keys.unexpected_keys)
        src=(W/'reference/stage2/answer_context_fused_loss.py').read_text();self.assertNotIn('qwen',src.lower());self.assertNotIn('llama',src.lower())
        DETAILS['stage2_cpu_kernel_substitution']='Exact frozen answer-mask function with test-only F.linear/cross_entropy replacement of Liger CUDA kernel; CUDA fused kernel validation deferred.'
        from adapters.pipeline import bind_family_builder
        formal=load('frozen_formal_builder',W/'reference/qwen_ntp/tools/cloud/formal_train_gated_ntp.py')
        original_builder=formal.build
        with bind_family_builder(formal,model_family='llama',ttt_layers=[0],chunk_size=4,device='cpu',dtype=torch.float32):
            built,cfg=formal.build(config=config())
            self.assertIsInstance(built,Train);self.assertEqual(formal.LAYERS,[0]);self.assertFalse(cfg.use_cache)
        self.assertIs(formal.build,original_builder)
        GATES.update(LLAMA_STAGE1_INTERFACE_STATUS='PASS',LLAMA_STAGE2_INTERFACE_STATUS='PASS',STAGE2_LOSS_MODEL_FAMILY_AGNOSTIC='PASS')

    def test_06_generic_validation_and_dynamic_static(self):
        for family in ['llama','qwen3']:
            cfg_cls,cls=model_classes(family)
            c=cfg_cls(**config().to_dict());c.model_type=family
            if family=='qwen3':c.layer_types=['full_attention']*c.num_hidden_layers
            validate_config(c,family)
            with self.assertRaises(ValueError):validate_config(c,'qwen3' if family=='llama' else 'llama')
        for bad in [[2],[-1],[True],[0,0],[1.0]]:
            with self.assertRaises(ValueError):validate_config(config(ttt_layers=bad),'llama')
        with self.assertRaises(ValueError):validate_config(config(architectures=['Qwen3ForCausalLM']),'llama')
        with self.assertRaises(ValueError):validate_config(config(architectures=['LlamaModel']),'llama')
        c=config(num_hidden_layers=32,ttt_layers=[0,6,12,18,24,30],ttt_mode=True,ttt_chunk=4096)
        # Tiny dimensions but real 32 decoder block hierarchy.
        model=Infer(c);self.assertEqual(len(decoder_blocks(model)),32)
        self.assertEqual(len(fresh_cache(model).ttt_states),32)
        expected=['OFF','L0','L6','L12','L18','L24','L30','ALL'];self.assertEqual(list(action_layers(c.ttt_layers)),expected)
        runtime=load('runtime_reference',W/'reference/inference_runtime/runtime.py')
        self.assertEqual(action_layers(c.ttt_layers),runtime.action_layers(c.ttt_layers))
        self.assertEqual(runtime.sample_best(dict.fromkeys(expected,.5),c.ttt_layers),(.5,expected))
        original=set(model.state_dict())
        for a in expected:
            with action_scope(model,a):
                enabled=[i for i in c.ttt_layers if hasattr(model.model.layers[i].mlp,'ttt_conv')]
                self.assertEqual(enabled,action_layers(c.ttt_layers)[a])
                if a=='OFF':self.assertFalse(model.model.ttt_mode)
            self.assertEqual(original,set(model.state_dict()))
        dynamic=load('dynamic_reference',W/'reference/inference_runtime/formal_dynamic_runner.py')
        dynamic.LAYERS=list(c.ttt_layers);dynamic.ACTIONS=expected;dynamic.ACTION_LAYERS=action_layers(c.ttt_layers)
        # Install the unchanged Dynamic MLP wrapper, without model inference or search.
        dynamic.install_dynamic_forward(model)
        for i in c.ttt_layers:
            mlp=model.model.layers[i].mlp
            self.assertEqual(list(inspect.signature(mlp.forward).parameters),['x','t','past_w','enabled'])
            for name in ['ttt_proj','ttt_conv','ttt_ntp_gate','ttt_chunk','ttt_lr','ttt_update_clip_norm','ttt_target_type','_dynamic_sequence','_events']:
                self.assertTrue(hasattr(mlp,name),name)
        GATES.update(MODEL_FAMILY_CONFIG_VALIDATION='PASS',LLAMA_SCALEUP_ACTION_SET_TEST='PASS',LLAMA_DYNAMIC_INTERFACE_STATIC_TEST='PASS',LLAMA_FSDP2_STATIC_COMPATIBILITY='PASS')

    def test_07_actual_32k_chunk_orchestration(self):
        # No quadratic attention allocation: 32768 tokens pass through the actual
        # adapted Llama MLP and document/chunk orchestration, not a metadata mock.
        c=config(hidden_size=4,intermediate_size=8,num_attention_heads=1,num_key_value_heads=1,ttt_mode=True,ttt_chunk=4096)
        x=torch.randn(1,32768,4)*.1
        mlp=TrainMLP(c,0);mlp.ttt_ntp_gate.data.fill_(.1)
        out=mlp(x,x,document_boundaries=[(0,32768)])
        self.assertEqual(tuple(out.shape),(1,32768,4));stats=mlp.last_ttt_stats[0]
        self.assertEqual(len(stats),8);self.assertEqual([s.update_count for s in stats],[1]*8);self.assertEqual([s.valid_tokens for s in stats],[4095]*8)
        again=mlp(x,x,document_boundaries=[(0,32768)]);compare(out,again,0)
        inf=InferMLP(copy.deepcopy(c),0);inf.load_state_dict(mlp.state_dict());inf.ttt_update_clip_norm=1e-5
        y,w=inf(x,x);self.assertEqual(len(inf.last_ttt_stats),8)
        tail=torch.cat([x,torch.ones(1,7,4)],1);z,w2=inf(tail,tail)
        self.assertEqual(len(inf.last_ttt_stats),8);compare(w,w2,0);compare(y,z[:,:32768],1e-7)
        h=inf.act_fn(inf.gate_proj(tail[:,32768:]))*inf.up_proj(tail[:,32768:]);compare(z[:,32768:],F.linear(h,w),1e-7)
        DETAILS['32k']={'tokens_through_actual_mlp':32768,'chunk_size':4096,'updates':8,'valid_pairs_per_chunk':4095,'additional_tail_tokens':7,'tail_updates':0,'attention_32k_run':False}
        GATES['LLAMA_32K_CHUNK4K_STATIC_RUNTIME_TEST']='PASS'

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(PortTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    out=dict(gates=GATES,details=DETAILS,tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),status='PASS' if result.wasSuccessful() else 'FAIL')
    (W/'audits/port_tests.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    sys.exit(not result.wasSuccessful())
