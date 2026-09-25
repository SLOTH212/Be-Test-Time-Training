"""Bind V2 execution into target-owned search globals; never load a model here."""
import ast
import hashlib
import types
from pathlib import Path

AUTHORITY_SHA256 = '8b7d8612e889c2a4c868ffa899acac1492c2ef75ab66fc39608e055c0371689c'

def bind_namespace(ns, core_module, chunk_size):
    # Existing search functions resolve Evaluator through this same globals dict.
    layers = list(ns['LAYERS'])
    if not layers or len(layers) != len(set(layers)) or chunk_size <= 0:
        raise ValueError('V2_INVALID_TARGET_CONFIGURATION')
    path = Path(__file__).with_name('v2_functions.py')
    class Target(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if node.module == 'hf_models.hf_qwen3.ttt_state_core':
                node.module = core_module
            return node
        def visit_Constant(self, node):
            if type(node.value) is int and node.value == 1024:
                return ast.copy_location(ast.Constant(chunk_size), node)
            return node
    tree = Target().visit(ast.parse(path.read_text()))
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), ns)
    return ns

def bind_runner(runner, core_module, chunk_size):
    ns = runner.Evaluator.__init__.__globals__
    bind_namespace(ns, core_module, chunk_size)
    runner.Evaluator = ns['Evaluator']
    runner.install_dynamic_forward = ns['install_dynamic_forward']
    return runner

def prepare_fixed(model, layers, core, action, prompt_tokens):
    # Use the same functional application for every configured candidate.
    layers = list(layers)
    mapping = {'OFF': [], 'ALL': layers, **{f'L{l}':[l] for l in layers}}
    if action not in mapping:
        raise ValueError('V2_UNKNOWN_ACTION')
    if not model.model.ttt_mode or list(model.model.ttt_layers) != layers:
        raise ValueError('V2_CANDIDATE_CONFIGURATION_MISMATCH')
    chunks = {model.model.layers[l].mlp.ttt_chunk for l in layers}
    if len(chunks) != 1:
        raise ValueError('V2_CHUNK_CONFIGURATION_MISMATCH')
    def tensor_hash(x):
        import torch
        return hashlib.sha256(x.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
    ns = dict(LAYERS=layers, ACTION_LAYERS=mapping, types=types, tensor_hash=tensor_hash)
    chunk = chunks.pop()
    bind_namespace(ns, core.__name__, chunk)
    ns['install_dynamic_forward'](model)
    for layer in layers:
        m = model.model.layers[layer].mlp
        m._dynamic_sequence = [action] * (prompt_tokens // chunk)
        m._branch_id = 'fixed'
        m.last_ttt_stats = []
