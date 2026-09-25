#!/usr/bin/env python3
import argparse, hashlib, json, math, os, random, statistics, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer
from hf_models.hf_qwen3.configuration_qwen3 import Qwen3Config
from hf_models.hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM

LAYERS = [0, 6, 12, 18, 24]
BASE = Path('/root/autodl-tmp/models/Qwen3-1.7B-Base')
S1 = Path('/root/autodl-tmp/data/continual_16k_100m_prolong_v3_canonical_v1/train.jsonl')
S2 = Path('/root/autodl-tmp/data/qa_replay_10m_v1_veomni_indexed_v1/train.jsonl')
MASK = Path('/root/autodl-tmp/data/qa_replay_10m_v1_answer_context_masks_v1/masks.jsonl')
OLD_LOG = Path('/root/autodl-tmp/logs/static_inplace_ttt_qwen1p7b_100m_qa10m_chunk1024_formal_v1_20260719_161937/stage1/training_log.jsonl')


def atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    os.replace(tmp, path)


def fsha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def flatten_stats(module):
    raw = getattr(module, 'last_ttt_stats', []) or []
    return [item for group in raw for item in (group if isinstance(group, list) else [group])]


def norm(param):
    return float(param.detach().float().norm())


def gradnorm(param):
    return None if param.grad is None else float(param.grad.detach().float().norm())


def optimizer_to(opt, device):
    for state in opt.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_rows(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def build(path, train_clip_none=True):
    cfg = Qwen3Config.from_pretrained(path)
    cfg.ttt_mode = True
    cfg.ttt_layers = LAYERS
    cfg.ttt_proj = True
    cfg.ttt_lr = 1.0
    cfg.ttt_chunk = 1024
    cfg.ttt_target = 'hidden_states'
    cfg.ttt_target_type = 'gated_next_position_hidden'
    cfg.ttt_ntp_gate = True
    cfg.ttt_ntp_gate_init = 'zero'
    cfg.ttt_ntp_within_chunk_only = True
    cfg.ttt_ntp_cross_chunk = False
    cfg.ttt_ntp_cross_record = False
    cfg.ttt_ntp_cross_document = False
    cfg.ttt_document_metadata_required = True
    cfg.ttt_backend = 'authoritative_document_chunk_loop'
    cfg.optimizer_betas = [0.9, 0.95]
    cfg.router_enabled = False
    cfg.signal_capture_enabled = False
    cfg.calibration_assignment_enabled = False
    cfg.ttt_update_clip_norm = None if train_clip_none else 1e-5
    cfg.use_cache = False
    assert cfg.ttt_target_type == 'gated_next_position_hidden'
    assert cfg.ttt_ntp_gate and cfg.ttt_ntp_gate_init == 'zero'
    assert cfg.ttt_ntp_within_chunk_only and not any((cfg.ttt_ntp_cross_chunk, cfg.ttt_ntp_cross_record, cfg.ttt_ntp_cross_document))
    assert cfg.ttt_backend == 'authoritative_document_chunk_loop'
    assert tuple(cfg.optimizer_betas) == (0.9, 0.95)
    model = Qwen3ForCausalLM.from_pretrained(
        path, config=cfg, dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True
    ).cuda().train()
    assert cfg.ttt_lr == 1.0 and model.config.ttt_lr == 1.0
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    return model, cfg


def layer_metrics(model):
    out = {}
    for layer in LAYERS:
        module = model.model.layers[layer].mlp
        stats = flatten_stats(module)
        out[str(layer)] = {
            'updates': sum(item.update_count for item in stats),
            'valid_tokens': sum(item.valid_tokens for item in stats),
            'pre_delta_min': min((item.pre_clip_norm for item in stats), default=0.0),
            'pre_delta_max': max((item.pre_clip_norm for item in stats), default=0.0),
            'conv_norm': norm(module.ttt_conv.weight),
            'target_norm': norm(module.ttt_proj.weight),
            'conv_grad': gradnorm(module.ttt_conv.weight),
            'target_grad': gradnorm(module.ttt_proj.weight),
            'ntp_gate_norm': norm(module.ttt_ntp_gate),
            'ntp_gate_grad': gradnorm(module.ttt_ntp_gate),
            'w_down_grad': gradnorm(module.down_proj.weight),
        }
    return out


def initial_ttt_parameters(model):
    return {
        str(layer): {
            'conv': model.model.layers[layer].mlp.ttt_conv.weight.detach().cpu().clone(),
            'target': model.model.layers[layer].mlp.ttt_proj.weight.detach().cpu().clone(),
            'ntp_gate': model.model.layers[layer].mlp.ttt_ntp_gate.detach().cpu().clone(),
        }
        for layer in LAYERS
    }


def checkpoint_payload(model, opt, scheduler, step, next_sample_id, cfg, stage, lineage, segment_id):
    return {
        'step': step,
        'dataloader_cursor': step,
        'next_sample_id': next_sample_id,
        'model': model.state_dict(),
        'optimizer': opt.state_dict(),
        'scheduler': scheduler.state_dict(),
        'cpu_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state(),
        'python_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'stage': f'STAGE{stage}',
        'lineage': lineage,
        'segment_id': segment_id,
        'ttt_config': {
            'layers': LAYERS,
            'lr': float(cfg.ttt_lr),
            'chunk': int(cfg.ttt_chunk),
            'target': cfg.ttt_target,
            'target_type': cfg.ttt_target_type,
            'within_chunk_only': cfg.ttt_ntp_within_chunk_only,
            'gate': 'per_layer_per_channel_exact_zero_init',
            'operation_order': 'apply_then_update',
            'reset_boundary': 'document',
        },
    }


def save_and_validate_rolling(path, manifest_path, model, opt, scheduler, step, next_sample_id, cfg, stage, lineage, segment_id, validator):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(checkpoint_payload(model, opt, scheduler, step, next_sample_id, cfg, stage, lineage, segment_id), tmp)
    with open(tmp, 'rb') as handle:
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    directory_fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    sha = fsha(path)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    command = [sys.executable, str(validator), '--checkpoint', str(path), '--expected-sha', sha,
               '--expected-step', str(step), '--expected-next-sample', next_sample_id,
               '--output', str(manifest_path)]
    completed = subprocess.run(command, env=env, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(f'CPU checkpoint validator failed: {completed.stderr[-4000:]}')
    result = json.loads(Path(manifest_path).read_text())
    if result.get('status') != 'PASS':
        raise RuntimeError('CPU checkpoint validation did not pass')
    return result


def aggregate_gate(records, model, initial, checkpoint_validation, stage, step):
    eligible = [record for record in records if record['ttt_gradient_eligible']]
    result = {
        'status': 'PENDING', 'stage': stage, 'step': step,
        'eligibility_definition': 'sum_over_documents(max(chunk_count - 1, 0)) > 0',
        'eligible_step_count': len(eligible), 'total_step_count': len(records),
        'first_window_clipping_rate': sum(bool(r['outer_clipped']) for r in records) / len(records),
        'recent20_clipping_rate': sum(bool(r['outer_clipped']) for r in records[-20:]) / len(records[-20:]),
        'loss_median': statistics.median(r['outer_loss'] for r in records),
        'recent20_loss_median': statistics.median(r['outer_loss'] for r in records[-20:]),
        'checkpoint_validation': checkpoint_validation,
        'layers': {}, 'nan_inf': any(r['nan_inf'] for r in records),
        'oom': False, 'reset_failed': False,
    }
    failures = []
    for layer in map(str, LAYERS):
        conv = [r['layers'][layer]['conv_grad'] for r in eligible if r['layers'][layer]['conv_grad'] not in (None, 0.0)]
        target = [r['layers'][layer]['target_grad'] for r in eligible if r['layers'][layer]['target_grad'] not in (None, 0.0)]
        gate = [r['layers'][layer]['ntp_gate_grad'] for r in eligible if r['layers'][layer]['ntp_gate_grad'] not in (None, 0.0)]
        module = model.model.layers[int(layer)].mlp
        conv_changed = not torch.equal(module.ttt_conv.weight.detach().cpu(), initial[layer]['conv'])
        target_changed = not torch.equal(module.ttt_proj.weight.detach().cpu(), initial[layer]['target'])
        gate_changed = not torch.equal(module.ttt_ntp_gate.detach().cpu(), initial[layer]['ntp_gate'])
        layer_result = {
            'eligible_step_count': len(eligible),
            'nonzero_conv_grad_eligible_steps': len(conv),
            'nonzero_target_grad_eligible_steps': len(target),
            'sum_conv_grad_norm_on_eligible_steps': sum(conv),
            'sum_target_grad_norm_on_eligible_steps': sum(target),
            'nonzero_ntp_gate_grad_eligible_steps': len(gate),
            'sum_ntp_gate_grad_norm_on_eligible_steps': sum(gate),
            'conv_parameter_changed_from_initialization': conv_changed,
            'target_parameter_changed_from_initialization': target_changed,
            'ntp_gate_parameter_changed_from_initialization': gate_changed,
        }
        result['layers'][layer] = layer_result
        if conv:
            failures.append(f'layer {layer}: Conv unexpectedly received gradient')
        if conv_changed:
            failures.append(f'layer {layer}: Conv unexpectedly changed')
        if eligible and not gate:
            failures.append(f'layer {layer}: no nonzero NTP gate gradient on eligible steps')
        if eligible and not target:
            failures.append(f'layer {layer}: no nonzero W_target gradient on eligible steps')
        if eligible and not gate_changed:
            failures.append(f'layer {layer}: NTP gate parameter unchanged')
        if eligible and not target_changed:
            failures.append(f'layer {layer}: W_target parameter unchanged')
    if result['nan_inf']:
        failures.append('NaN/Inf observed')
    if checkpoint_validation.get('status') != 'PASS':
        failures.append('checkpoint validation failed')
    if len(eligible) == 0:
        result['status'] = 'INCONCLUSIVE_INSUFFICIENT_MULTI_CHUNK_DATA'
        result['next_gate_step'] = min(200, step + 1)
    elif failures:
        result['status'] = 'FAIL'
        result['failures'] = failures
    else:
        result['status'] = 'PASS'
        result['engineering_stability'] = 'PASS'
        result['optimization_regime'] = 'CLIPPING_SATURATED' if result['first_window_clipping_rate'] == 1.0 else 'NORMAL_OR_PARTIAL_CLIPPING'
    return result


def compare_lr_first100(records, output_path):
    old = list(load_rows(OLD_LOG))[:100]
    new = records[:100]
    def sequence(key):
        return [row.get(key) for row in old] == [row.get(key) for row in new]
    loss_diffs = [abs(a['outer_loss'] - b['outer_loss']) for a, b in zip(old, new)]
    delta_diffs = []
    old_deltas, new_deltas = [], []
    for a, b in zip(old, new):
        for layer in map(str, LAYERS):
            delta_diffs.extend([
                abs(a['layers'][layer]['pre_delta_min'] - b['layers'][layer]['pre_delta_min']),
                abs(a['layers'][layer]['pre_delta_max'] - b['layers'][layer]['pre_delta_max']),
            ])
            old_deltas.extend([a['layers'][layer]['pre_delta_min'], a['layers'][layer]['pre_delta_max']])
            new_deltas.extend([b['layers'][layer]['pre_delta_min'], b['layers'][layer]['pre_delta_max']])
    old_pre = [row['pre_global_grad_norm'] for row in old]
    new_pre = [row['pre_global_grad_norm'] for row in new]
    result = {
        'status': 'PASS' if len(old) == len(new) == 100 else 'FAIL',
        'old_run': str(OLD_LOG), 'compared_steps': min(len(old), len(new)),
        'sample_id_order_identical': sequence('sample_id'),
        'valid_tokens_identical': sequence('valid_tokens'),
        'document_counts_identical': sequence('document_count'),
        'chunks_identical': sequence('chunks_per_document'),
        'update_counts_identical': sequence('update_count_per_layer'),
        'max_abs_loss_difference': max(loss_diffs, default=None),
        'median_abs_loss_difference': statistics.median(loss_diffs) if loss_diffs else None,
        'max_abs_delta_difference': max(delta_diffs, default=None),
        'median_delta_lr3': statistics.median(old_deltas) if old_deltas else None,
        'median_delta_lr0p3': statistics.median(new_deltas) if new_deltas else None,
        'delta_ratio_lr0p3_over_lr3': statistics.median(new_deltas) / statistics.median(old_deltas) if old_deltas and statistics.median(old_deltas) else None,
        'median_preclip_grad_lr3': statistics.median(old_pre),
        'median_preclip_grad_lr0p3': statistics.median(new_pre),
        'grad_ratio_lr0p3_over_lr3': statistics.median(new_pre) / statistics.median(old_pre) if statistics.median(old_pre) else None,
        'eligible_step_count_lr3': sum(bool(row.get('chunks_per_document') and sum(max(c-1, 0) for c in row['chunks_per_document']) > 0) for row in old),
        'eligible_step_count_lr0p3': sum(bool(row.get('ttt_gradient_eligible')) for row in new),
        'clipping_pattern_identical': sequence('outer_clipped'),
        'nan_inf_pattern_identical': sequence('nan_inf'),
        'experiment_type': 'TTT_INNER_LEARNING_RATE_ABLATION',
        'changed_parameter': 'ttt_lr', 'old_value': 3.0, 'new_value': 0.3,
    }
    if not all(result[key] for key in ['sample_id_order_identical', 'valid_tokens_identical', 'document_counts_identical', 'chunks_identical', 'update_counts_identical']):
        result['status'] = 'FAIL'
    atomic(output_path, result)
    return result


def dry_run_parity(record, parent_control, output_path):
    parent = None
    with open(Path(parent_control) / 'stage1/training_log.jsonl') as handle:
        for line in handle:
            candidate = json.loads(line)
            if candidate['step'] == 3001:
                parent = candidate
                break
    if parent is None:
        raise RuntimeError('parent step3001 record not found')
    delta_differences = {}
    gradient_status = {}
    for layer in map(str, LAYERS):
        old, new = parent['layers'][layer], record['layers'][layer]
        pairs = [('pre_delta_min', old['pre_delta_min'], new['pre_delta_min']),
                 ('pre_delta_max', old['pre_delta_max'], new['pre_delta_max'])]
        delta_differences[layer] = {
            name: {'parent': a, 'resumed': b, 'absolute_difference': abs(a-b),
                   'relative_difference': abs(a-b)/max(abs(a), 1e-12)} for name, a, b in pairs
        }
        gradient_status[layer] = {
            'parent_conv_absent': old['conv_grad'] in (None, 0.0),
            'resumed_conv_absent': new['conv_grad'] in (None, 0.0),
            'parent_gate_present': old['ntp_gate_grad'] not in (None, 0.0),
            'resumed_gate_present': new['ntp_gate_grad'] not in (None, 0.0),
            'parent_target_present': old['target_grad'] not in (None, 0.0),
            'resumed_target_present': new['target_grad'] not in (None, 0.0),
        }
    max_delta_relative = max(v['relative_difference'] for layer in delta_differences.values() for v in layer.values())
    checks = {
        'sample_id': record['sample_id'] == parent['sample_id'],
        'valid_tokens': record['valid_tokens'] == parent['valid_tokens'],
        'document_count': record['document_count'] == parent['document_count'],
        'chunks': record['chunks_per_document'] == parent['chunks_per_document'],
        'effective_updates': record['effective_ttt_training_updates'] == parent['effective_ttt_training_updates'],
        'update_counts': record['update_count_per_layer'] == parent['update_count_per_layer'],
        'loss_bf16_tolerance': abs(record['outer_loss'] - parent['outer_loss']) <= 0.02,
        'delta_bf16_tolerance': max_delta_relative <= 0.05,
        'gradient_presence': all(all(v.values()) for v in gradient_status.values()),
        'finite': not record['nan_inf'],
    }
    result = {
        'status': 'PASS' if all(checks.values()) else 'FAIL', 'step': 3001,
        'parent_sample_id': parent['sample_id'], 'resumed_sample_id': record['sample_id'],
        'parent_loss': parent['outer_loss'], 'resumed_loss': record['outer_loss'],
        'loss_absolute_difference': abs(record['outer_loss'] - parent['outer_loss']),
        'parent_pre_grad_norm': parent['pre_global_grad_norm'], 'resumed_pre_grad_norm': record['pre_global_grad_norm'],
        'parent_post_grad_norm': parent['post_global_grad_norm'], 'resumed_post_grad_norm': record['post_global_grad_norm'],
        'delta_differences': delta_differences, 'max_delta_relative_difference': max_delta_relative,
        'gradient_status': gradient_status, 'checks': checks, 'nan_inf': record['nan_inf'],
        'checkpoint_modified': False, 'model_saved': False,
    }
    atomic(output_path, result)
    if result['status'] != 'PASS':
        raise RuntimeError('STATIC_TTT_RESUME_STEP3001_PARITY_FAILED')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int, choices=[1, 2], required=True)
    parser.add_argument('--control', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume-checkpoint')
    parser.add_argument('--resume-step', type=int)
    parser.add_argument('--resume-parent-control')
    parser.add_argument('--resume-log-mode', choices=['new-segment'])
    parser.add_argument('--resume-checkpoint-sha')
    parser.add_argument('--segment-id', default='fresh')
    parser.add_argument('--dry-run-only', action='store_true')
    parser.add_argument('--dry-run-audit')
    args = parser.parse_args()
    control, output, stage = Path(args.control), Path(args.output), args.stage
    stage_dir = control / f'stage{stage}'
    stage_dir.mkdir(parents=True, exist_ok=True)
    explicit_resume = args.resume_checkpoint is not None
    if explicit_resume:
        assert stage == 1
        assert args.resume_step is not None and args.resume_parent_control and args.resume_log_mode == 'new-segment'
        assert args.resume_checkpoint_sha
        log = stage_dir / 'training_log_resume_segment.jsonl'
    else:
        log = stage_dir / 'training_log.jsonl'
    rolling = output / f'stage{stage}_rolling' / 'resume.pt'
    init = BASE if stage == 1 else output / 'stage1_final/hf_ckpt'
    if stage == 1 and not explicit_resume:
        assert init == BASE and not rolling.exists(), 'Fresh Stage 1 must initialize from Base without a rolling checkpoint'
    if explicit_resume:
        assert init == BASE and not rolling.exists(), 'Continuation writes only to a new OUTPUT'
    expected = 7054 if stage == 1 else 1291
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.reset_peak_memory_stats()
    base_before = fsha(BASE / 'model.safetensors')
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    model, cfg = build(init)
    initial = initial_ttt_parameters(model)
    opt = torch.optim.AdamW(
        model.parameters(), lr=5e-6, betas=(0.9, 0.95), eps=1e-8,
        weight_decay=0.1, fused=True,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    start = 0
    lineage = {'parent_run_id': None, 'resume_checkpoint_step': None, 'resume_checkpoint_sha256': None}
    if explicit_resume:
        resume_path = Path(args.resume_checkpoint)
        assert resume_path.is_file()
        actual_sha = fsha(resume_path)
        assert actual_sha == args.resume_checkpoint_sha, (actual_sha, args.resume_checkpoint_sha)
        checkpoint = torch.load(resume_path, map_location='cpu', weights_only=False)
        assert checkpoint['step'] == args.resume_step == 3000
        assert checkpoint['dataloader_cursor'] == 3000
        assert checkpoint['next_sample_id'] == 'prolong-v3:book-65536:train:000835'
        assert checkpoint['ttt_config']['lr'] == 0.3
        assert checkpoint['ttt_config']['chunk'] == 1024
        assert checkpoint['ttt_config']['layers'] == LAYERS
        model.load_state_dict(checkpoint['model'], strict=True)
        opt.load_state_dict(checkpoint['optimizer'])
        optimizer_to(opt, 'cuda')
        scheduler.load_state_dict(checkpoint['scheduler'])
        torch.set_rng_state(checkpoint['cpu_rng'])
        torch.cuda.set_rng_state(checkpoint['cuda_rng'])
        start = checkpoint['step']
        lineage = {'parent_run_id': Path(args.resume_parent_control).name,
                   'resume_checkpoint_step': start, 'resume_checkpoint_sha256': actual_sha}
        del checkpoint
    elif rolling.exists():
        checkpoint = torch.load(rolling, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=True)
        opt.load_state_dict(checkpoint['optimizer'])
        optimizer_to(opt, 'cuda')
        scheduler.load_state_dict(checkpoint['scheduler'])
        torch.set_rng_state(checkpoint['cpu_rng'])
        torch.cuda.set_rng_state(checkpoint['cuda_rng'])
        if 'python_rng' in checkpoint: random.setstate(checkpoint['python_rng'])
        if 'numpy_rng' in checkpoint: np.random.set_state(checkpoint['numpy_rng'])
        start = checkpoint['step']
        lineage = checkpoint.get('lineage', lineage)
        del checkpoint
    if stage == 2:
        import sys
        sys.path.insert(0, '/root/opencompass_clean/tools/autodl/v2_full_epoch_ruler_bs2')
        from answer_context_data import process_answer_context_example
        from answer_context_fused_loss import single_decoder_fused_answer_context
        os.environ['ANSWER_CONTEXT_MASK_SIDECAR'] = str(MASK)
    rows = load_rows(S1 if stage == 1 else S2)
    for _ in range(start):
        next(rows)
    records = list(load_rows(log)) if log.exists() else []
    losses = [r['outer_loss'] for r in records]
    explosive = 0
    t0all = time.time()
    step = start
    for step, row in enumerate(rows, start + 1):
        if step > expected:
            break
        if explicit_resume and step == 3001:
            assert row.get('id', row.get('sample_id')) == 'prolong-v3:book-65536:train:000835'
        opt.zero_grad(set_to_none=True)
        ts = time.time()
        torch.cuda.reset_peak_memory_stats()
        if stage == 1:
            ids = tokenizer(row['text'], add_special_tokens=False, truncation=True, max_length=16384, return_tensors='pt').input_ids.cuda()
            valid = ids.shape[1]
            attention = torch.ones_like(ids)
            metadata = row.get('metadata', {})
            raw = metadata.get('document_boundaries_qwen_json', f'[[0,{valid}]]')
            bounds = json.loads(raw) if isinstance(raw, str) else raw
            bounds = [(max(0, int(a)), min(valid, int(b))) for a, b in bounds if int(a) < valid and int(b) > 0]
            output_obj = model(input_ids=ids, attention_mask=attention, labels=ids, use_cache=False, document_boundaries=bounds)
            loss, answer_loss, context_loss, counts = output_obj.loss, None, None, {}
        else:
            import torch.nn.functional as F
            ex = process_answer_context_example(row, tokenizer, 16384)[0]
            valid = len(ex['input_ids'])
            pad = 16384 - valid
            ids = F.pad(ex['input_ids'], (0, pad), value=tokenizer.eos_token_id).unsqueeze(0).cuda()
            attention = F.pad(ex['attention_mask'], (0, pad), value=0).unsqueeze(0).cuda()
            labels = F.pad(ex['labels'], (0, pad), value=-100).unsqueeze(0).cuda()
            answer_mask = F.pad(ex['answer_mask'], (0, pad), value=False).unsqueeze(0).cuda()
            context_mask = F.pad(ex['context_mask'], (0, pad), value=False).unsqueeze(0).cuda()
            bounds = [(0, valid)]
            loss, answer_loss, context_loss, counts = single_decoder_fused_answer_context(
                model, {'input_ids': ids, 'attention_mask': attention, 'labels': labels, 'document_boundaries': bounds}, answer_mask, context_mask
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f'nonfinite loss step {step}')
        loss.backward()
        pre = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        post = min(float(pre), 1.0)
        metrics = layer_metrics(model)
        if not math.isfinite(float(pre)):
            raise FloatingPointError(f'nonfinite grad step {step}')
        chunks = [(b - a + 1023) // 1024 for a, b in bounds]
        total_updates = sum(chunks)
        effective_updates = sum(max(count - 1, 0) for count in chunks)
        eligible = effective_updates > 0
        if any(value['updates'] != total_updates for value in metrics.values()):
            raise RuntimeError(f'update count mismatch step {step}')
        opt.step()
        scheduler.step()
        torch.cuda.synchronize()
        loss_value = float(loss.detach())
        losses.append(loss_value)
        median = statistics.median(losses[-50:-1]) if len(losses) > 1 else loss_value
        explosive = explosive + 1 if median > 0 and loss_value > 20 * median else 0
        if explosive >= 10:
            raise RuntimeError('ten consecutive explosive losses')
        record = {
            'stage': stage, 'step': step, 'sample_id': row.get('id', row.get('sample_id')),
            'valid_tokens': valid, 'document_count': len(bounds), 'boundaries': bounds,
            'chunks_per_document': chunks, 'total_update_count': total_updates,
            'loss_effective_update_count': effective_updates,
            'effective_ttt_training_updates': effective_updates,
            'ttt_gradient_eligible': eligible,
            'conv_gradient_expectation': 'ALWAYS_NONE_IN_GATED_NTP',
            'ntp_gate_gradient_expectation': 'REQUIRED_ON_ELIGIBLE_STEP' if eligible else 'EXPECTED_NO_FUTURE_CHUNK',
            'target_gradient_expectation': 'REQUIRED_AFTER_GATE_OPENS' if eligible else 'EXPECTED_NO_FUTURE_CHUNK',
            'update_count_per_layer': {key: value['updates'] for key, value in metrics.items()},
            'outer_loss': loss_value,
            'answer_loss': None if answer_loss is None else float(answer_loss.detach()),
            'context_loss': None if context_loss is None else float(context_loss.detach()),
            **counts, 'pre_global_grad_norm': float(pre), 'post_global_grad_norm': post,
            'outer_clipped': float(pre) > 1.0, 'layers': metrics,
            'peak_allocated': torch.cuda.max_memory_allocated(), 'peak_reserved': torch.cuda.max_memory_reserved(),
            'step_seconds': time.time() - ts, 'tokens_per_second': valid / max(time.time() - ts, 1e-9),
            'nan_inf': False, 'oom': False, 'reset_failed': False,
            'parent_checkpoint_sha256': lineage.get('resume_checkpoint_sha256'),
            'resume_segment_id': args.segment_id,
        }
        if args.dry_run_only:
            assert step == 3001 and args.dry_run_audit
            dry_run_parity(record, args.resume_parent_control, args.dry_run_audit)
            return
        records.append(record)
        with open(log, 'a') as f:
            f.write(json.dumps(record) + '\n')
        atomic(control / 'HEARTBEAT.json', {
            'status': 'ACTIVE', 'stage': f'STAGE{stage}', 'step': step, 'total_steps': expected,
            'worker_pid': os.getpid(), 'updated_epoch': time.time(), 'ttt_lr': float(model.config.ttt_lr),
        })
        detailed = [1, 2, 5, 10, 20, 50, 100] if stage == 1 else [1, 2, 5, 10, 20, 50]
        if step in detailed:
            atomic(stage_dir / f'detailed_step_{step}.json', record)
        gate_step = 100 if stage == 1 else 50
        save_steps = ([100] + list(range(1000, expected + 1, 1000))) if stage == 1 else [50, 500, 1000]
        checkpoint_validation = None
        if step in save_steps:
            next_sample_id = f'index:{step}' if step >= expected else None
            if step < expected:
                # Deterministic order: the next record is index `step` in the JSONL.
                with open(S1 if stage == 1 else S2) as source:
                    for index, next_row in enumerate(source):
                        if index == step:
                            next_sample = json.loads(next_row)
                            next_sample_id = next_sample.get('id', next_sample.get('sample_id'))
                            break
            checkpoint_validation = save_and_validate_rolling(
                rolling, stage_dir / f'checkpoint_validation_step_{step}.json', model, opt, scheduler,
                step, next_sample_id, cfg, stage, lineage, args.segment_id, control / 'checkpoint_validator_cpu.py'
            )
        if step == gate_step:
            gate_result = aggregate_gate(records[:gate_step], model, initial, checkpoint_validation, stage, step)
            if gate_result['status'] == 'PASS':
                atomic(stage_dir / f'STAGE{stage}_STABILITY_GATE_PASS.json', gate_result)
            elif gate_result['status'].startswith('INCONCLUSIVE'):
                atomic(stage_dir / f'STAGE{stage}_STABILITY_GATE_INCONCLUSIVE.json', gate_result)
            else:
                atomic(stage_dir / f'STAGE{stage}_STABILITY_GATE_FAILED.json', gate_result)
                raise RuntimeError('aggregated eligible-step stability gate failed: ' + '; '.join(gate_result.get('failures', [])))
    if step != expected:
        raise RuntimeError(f'dataset steps {step} != {expected}')
    final = output / ('stage1_final/hf_ckpt' if stage == 1 else 'final_gated_ntp/hf_ckpt')
    final.mkdir(parents=True, exist_ok=True)
    model.config.ttt_update_clip_norm = 1e-5
    model.save_pretrained(final, safe_serialization=True)
    tokenizer.save_pretrained(final)
    summary = {
        'status': 'PASS', 'stage': stage, 'steps': step, 'records': expected,
        'wall_seconds': time.time() - t0all, 'final_hf': str(final),
        'model_sha256': fsha(final / 'model.safetensors'),
        'base_sha_unchanged': fsha(BASE / 'model.safetensors') == base_before,
        'peak_allocated': torch.cuda.max_memory_allocated(), 'peak_reserved': torch.cuda.max_memory_reserved(),
    }
    atomic(stage_dir / f'stage{stage}_training_summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
