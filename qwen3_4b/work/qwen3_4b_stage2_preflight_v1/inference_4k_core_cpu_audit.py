from pathlib import Path
import importlib.util,sys,json,torch,dataclasses
R=Path('/path/to/ttt');A=R/'audits/qwen3_4b_stage2_30m_32k_gpu23_preflight_v1'
def module(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
old=module('reference_frozen_core',R/'src/inference_runtime/code/hf_models/hf_qwen3/ttt_state_core.py')
new=module('amended_4k_core',R/'src/inference_runtime_4k_stage2_v1/code/hf_models/hf_qwen3/ttt_state_core.py')
torch.set_num_threads(4);torch.manual_seed(42)
h=torch.randn(1,8192,3,dtype=torch.float64);t=torch.randn(1,8192,2,dtype=torch.float64);base=torch.randn(2,3,dtype=torch.float64);proj=torch.randn(2,2,dtype=torch.float64);kernel=torch.zeros(2,1,5,dtype=torch.float64);valid=torch.ones(1,8192,dtype=torch.bool);gate=torch.randn(2,dtype=torch.float64)
before=base.clone();args=(h,t,base,proj,kernel,valid);kw={'learning_rate':1.,'delta_clip_norm':1e-5,'target_type':new.GATED_NTP_TARGET,'target_gate':gate}
a=old.run_document(*args,chunk_size=4096,allow_test_override=True,**kw)
b=new.run_document(*args,**kw)
assert torch.equal(a[0],b[0]) and torch.equal(a[1].weight,b[1].weight)
assert [dataclasses.asdict(x) for x in a[2]]==[dataclasses.asdict(x) for x in b[2]]
assert [x.valid_tokens for x in b[2]]==[4095,4095] and [x.update_count for x in b[2]]==[1,1]
assert torch.equal(b[0][:,:4096],torch.nn.functional.linear(h[:,:4096],base))
assert torch.equal(base,before)
again=new.run_document(*args,**kw);assert torch.equal(again[0],b[0]) and torch.equal(again[1].weight,b[1].weight)
reset,stats=new.run_sequence_with_boundaries(*args,document_boundaries=[(0,4096),(4096,8192)],**kw)
assert torch.equal(reset,torch.nn.functional.linear(h,base)) and [len(x) for x in stats]==[1,1]
try:new.run_document(*args,chunk_size=1024,**kw);raise AssertionError('guard did not reject 1K')
except ValueError as e:assert '4096' in str(e)
v={'status':'PASS','cpu_float64_fixture':True,'tokens':8192,'chunks':2,'chunk_size':4096,'valid_pairs_per_chunk':[4095,4095],'original_formula_exact_at_same_4k_segmentation':True,'reference_used_frozen_test_override_only_for_mathematical_comparison':True,'production_4k_path_used_test_override':False,'apply_then_update':True,'per_document_reset':True,'base_weights_unchanged':True,'one_k_guard_rejection':True,'user_authorization':'推理chunk改为4k','not_claiming_1k_numerical_parity':True}
(A/'INFERENCE_4K_CORE_CPU_AUDIT.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
