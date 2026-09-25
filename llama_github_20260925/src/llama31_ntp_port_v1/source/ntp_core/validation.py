"""Model-family plumbing and explicit candidate chunk/layer validation."""
def validate_config(config, model_family):
    architectures = {'llama': 'LlamaForCausalLM', 'qwen3': 'Qwen3ForCausalLM'}
    if model_family not in architectures or config.model_type != model_family:
        raise ValueError('model_family does not match config.model_type')
    if config.architectures and any(a != architectures[model_family] for a in config.architectures):
        raise ValueError('wrong causal-LM architecture for model family')
    layers = getattr(config, 'ttt_layers', [])
    if not isinstance(layers, (list, tuple)) or any(type(i) is not int or not 0 <= i < config.num_hidden_layers for i in layers):
        raise ValueError('ttt_layers must contain valid decoder-layer indices')
    if len(layers) != len(set(layers)):
        raise ValueError('duplicate TTT layer')
    chunk = getattr(config, 'ttt_chunk', 1024)
    if type(chunk) is not int or chunk <= 0:
        raise ValueError('ttt_chunk must be a positive integer')
    if getattr(config, 'ttt_mode', False):
        if getattr(config, 'mlp_bias', False):
            raise ValueError('NTP authority uses a bias-free fast-weight path')
        if getattr(config, 'ttt_target_type', None) != 'gated_next_position_hidden':
            raise ValueError('current NTP requires gated_next_position_hidden')
        if getattr(config, 'ttt_target', None) != 'hidden_states' or not getattr(config, 'ttt_proj', False):
            raise ValueError('current NTP requires contextual hidden target and projection')
