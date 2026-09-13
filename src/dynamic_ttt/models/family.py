"""Family selection only; optimizer/data/scheduler/DCP remain common runtime responsibilities.

The caller supplies the model family, layers, and chunk size. The return value
matches formal_train_gated_ntp.build: (model, config).
"""
import importlib
import torch
from dynamic_ttt.models.validation import validate_config

def model_classes(model_family, inference=False):
    if model_family not in ('llama', 'qwen3'):
        raise ValueError('unsupported model family')
    stem='Llama' if model_family=='llama' else 'Qwen3'
    pkg='dynamic_ttt.models.'+('inference' if inference else 'training')+'.hf_'+model_family
    cfg=importlib.import_module(pkg+'.configuration_'+model_family)
    impl=importlib.import_module(pkg+'.modeling_'+model_family)
    return getattr(cfg,stem+'Config'),getattr(impl,stem+'ForCausalLM')

def configure(config, *, model_family, ttt_layers, chunk_size=1024, inference=False, enabled=True):
    config.ttt_layers=list(ttt_layers);config.ttt_chunk=chunk_size;config.ttt_mode=enabled
    settings=dict(ttt_proj=True,ttt_lr=1.0,ttt_target='hidden_states',ttt_target_type='gated_next_position_hidden',ttt_ntp_gate=True,ttt_ntp_gate_init='zero',ttt_ntp_within_chunk_only=True,ttt_ntp_cross_chunk=False,ttt_ntp_cross_record=False,ttt_ntp_cross_document=False,ttt_document_metadata_required=True,ttt_backend='authoritative_document_chunk_loop',optimizer_betas=[0.9,0.95],router_enabled=False,signal_capture_enabled=False,calibration_assignment_enabled=False,ttt_update_clip_norm=1e-5 if inference else None,use_cache=inference)
    for k,v in settings.items():setattr(config,k,v)
    config._attn_implementation='sdpa'
    validate_config(config,model_family)
    # Byte-identical Qwen model retains its frozen 1024 guard.
    if model_family=='qwen3' and enabled and chunk_size!=1024:
        raise ValueError('frozen Qwen runtime requires chunk_size=1024')
    return config

def build(path=None, train_clip_none=True, *, model_family, ttt_layers, chunk_size=1024, config=None, inference=False, enabled=True, device='cpu', dtype=torch.bfloat16, checkpointing=True):
    if not train_clip_none and not inference:
        raise ValueError('training authority always uses unclipped inner updates')
    cfg_cls,cls=model_classes(model_family,inference)
    if config is None:
        if path is None:raise ValueError('model path or tiny config required')
        # AutoConfig verifies actual architecture before family-specific deserialization.
        from transformers import AutoConfig
        actual=AutoConfig.from_pretrained(path,local_files_only=True)
        validate_config(actual,model_family)
        config=cfg_cls.from_pretrained(path,local_files_only=True)
    configure(config,model_family=model_family,ttt_layers=ttt_layers,chunk_size=chunk_size,inference=inference,enabled=enabled)
    if path is None:model=cls(config).to(device=device,dtype=dtype)
    else:model=cls.from_pretrained(path,config=config,dtype=dtype,attn_implementation='sdpa',local_files_only=True).to(device)
    if inference:model.eval()
    else:
        model.train()
        if checkpointing:model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    return model,config

def decoder_blocks(model):
    names=set(model._no_split_modules)
    blocks=[m for m in model.modules() if type(m).__name__ in names]
    if len(blocks)!=model.config.num_hidden_layers or len({id(m) for m in blocks})!=len(blocks):
        raise ValueError('decoder block discovery is not one-to-one')
    if blocks!=list(model.model.layers):raise ValueError('unexpected model root')
    return blocks
