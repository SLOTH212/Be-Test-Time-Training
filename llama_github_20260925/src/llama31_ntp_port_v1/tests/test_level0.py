"""Portable static/source/import gate. Original paths checked only if present."""
import ast, hashlib, importlib.util, json, sys
from pathlib import Path
W=Path(__file__).resolve().parents[1];sys.path[:0]=[str(W),str(W/'source')]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def class_source(p,name):
    text=p.read_text();node=next(n for n in ast.parse(text).body if getattr(n,'name',None)==name)
    return ast.dump(node,include_attributes=False)

def run(check_originals=True):
    count=0
    for root in ['source','adapters','tests','provenance']:
        for p in (W/root).rglob('*.py'):ast.parse(p.read_text(),filename=str(p));count+=1
    for manifest,subroot in [('QWEN_NTP_SOURCE_AUTHORITY.json','qwen_ntp'),('INPLACE_TTT_LLAMA_REFERENCE.json','inplace_ttt')]:
        data=json.loads((W/'provenance'/manifest).read_text())
        for item in data['files']:
            p=W/'reference'/subroot/item['relative_path'];assert sha(p)==item['sha256']
            if check_originals:assert sha(Path(item['path']))==item['sha256']
    for item in json.loads((W/'provenance/COMMON_RUNTIME_REFERENCE.json').read_text()):
        assert sha(W/'reference'/item['relative_path'])==item['sha256']
        if check_originals:assert sha(Path(item['path']))==item['sha256']
    # Llama-specific attention, rotary, norm, output/head methods remain historical.
    names=['LlamaAttention','LlamaRotaryEmbedding','LlamaRMSNorm']
    for mode,ref,out in [('training','hf_models/hf_llama','hf_models/hf_llama'),('inference','inference_model/hf_llama3','inference_model/hf_llama')]:
        for name in names:assert class_source(W/'reference/inplace_ttt'/ref/'modeling_llama.py',name)==class_source(W/'source'/out/'modeling_llama.py',name)
    for root in ['hf_models/hf_qwen3','inference_model/hf_qwen3']:
        for p in (W/'source'/root).glob('*.py'):assert sha(p)==sha(W/'reference/qwen_ntp'/root/p.name)
    original=(W/'reference/qwen_ntp/hf_models/hf_qwen3/ttt_state_core.py').read_text()
    expected=original.replace('if chunk_size != 1024 and not allow_test_override:\n        raise ValueError("frozen Static-TTT chunk_size must be 1024")','if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:\n        raise ValueError("chunk_size must be a positive integer")')
    assert (W/'source/ntp_core/ttt_state_core.py').read_text()==expected
    from adapters.model_family import model_classes,configure
    from ntp_core.validation import validate_config
    for family in ['llama','qwen3']:
        for inf in [False,True]:
            cfg_cls,cls=model_classes(family,inf);assert cls.__name__ in ['LlamaForCausalLM','Qwen3ForCausalLM']
            cfg=cfg_cls(num_hidden_layers=2,ttt_layers=[],ttt_mode=False);validate_config(cfg,family)
    for p in (W/'configs').glob('*.yaml'):
        cfg=json.loads(p.read_text());assert cfg['FORMAL_LLAMA_CONFIG_FROZEN'] is False
        assert cfg['classification']=='SCALEUP_CANDIDATE_CONFIG';assert cfg['context_length']//cfg['chunk_size']==8
        cfg_cls,_=model_classes('llama');model_cfg=cfg_cls(num_hidden_layers=32)
        configure(model_cfg,model_family='llama',ttt_layers=cfg['ttt_layers'],chunk_size=cfg['chunk_size'],inference=cfg['stage']=='ruler32k_inference')
        validate_config(model_cfg,'llama')
    return dict(LLAMA_PORT_LEVEL0='PASS',syntax_files=count,authority_hashes='PASS',llama_historical_attention_rope_norm_ast='IDENTICAL',qwen_snapshot_source='BYTE_IDENTICAL',common_core_diff='positive_chunk_size_validation_only',original_paths_checked=check_originals)
if __name__=='__main__':
    out=run('--extracted' not in sys.argv);(W/'audits/level0.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
