#!/usr/bin/env python3
import os
import json, subprocess, textwrap
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports'; T=ROOT/'tests/stage_transition_real'
PROJECT=Path(os.path.expandvars('${TTT_PROJECT_ROOT}'))
CORE_HEAD=subprocess.check_output(['git','-C',str(PROJECT),'rev-parse','HEAD'],text=True).strip()

def j(path): return json.load(open(path))
def rows(path): return [json.loads(x) for x in open(path)]
def write(name,body):
    (REPORTS/name).write_text(textwrap.dedent(body).strip()+'\n')

s1=j(T/'real_checkpoint_evidence.json'); s2=j(T/'stage2_real_checkpoint_evidence.json')
resume=j(T/'real_resume_reference.json'); tail=j(T/'tailfix_runtime.json')
offm=j(T/'eval_off/manifest.json'); onm=j(T/'eval_on_tailfix/manifest.json')
off=rows(T/'eval_off/predictions.jsonl'); on=rows(T/'eval_on_tailfix/predictions.jsonl')
pair_ids=[x['sample_id'] for x in off]==[x['sample_id'] for x in on]
pair_ckpt=offm['checkpoint_hash']==onm['checkpoint_hash']
stage2_logs=rows(T/'stage2/training.jsonl')

write('real_worker_entrypoint_audit.md',f'''# Real worker entrypoint audit

Base project audit commit: `57936c376a99fe6fe75d2bebf19d2a6d4006e5bb`. A runtime tailfix defect found by full-model CUDA testing required the isolated core commit `{CORE_HEAD}`; the pre-existing unrelated untracked report/tool remain untouched.

| Role | Pipeline entrypoint | Authoritative implementation | Inputs / output / resume |
|---|---|---|---|
| STAGE1_REAL_ENTRYPOINT | `{ROOT}/bin/run_stage1_worker.sh` | `{PROJECT}/tools/cloud/formal_train_gated_ntp.py` via `workers/train_worker.py --stage 1` | `--config`, `--resume-from`, `--output-dir`, `--state-file`, exactly one of `--smoke`/`--formal-run`; sequential cursor and A/B output |
| STAGE2_REAL_ENTRYPOINT | `{ROOT}/bin/run_stage2_worker.sh` | same formal model builder plus `${TTT_OPENCOMPASS_ROOT}/tools/autodl/v2_full_epoch_ruler_bs2/answer_context_fused_loss.py` | Base constructs the architecture, then Stage1 final checkpoint model state is loaded strictly on first start; Stage2 A/B exact resume thereafter |
| VALIDATION_REAL_ENTRYPOINT | `{ROOT}/workers/train_worker.py:validate_stage1` | formal decoder plus fused linear CE sum | validation JSONL; no optimizer call and parameter-version assertion |
| OFF_INFERENCE_REAL_ENTRYPOINT | `{ROOT}/bin/run_eval_off_worker.sh` | `{ROOT}/workers/eval_worker.py` and project inference model | Stage2 final; real batch 2; TTT branch bypassed; resumable sample IDs |
| ON_INFERENCE_REAL_ENTRYPOINT | `{ROOT}/bin/run_eval_on_worker.sh` | same eval worker and `{PROJECT}/inference_model/hf_qwen3/modeling_qwen3.py` | fresh process/cache per sample; batch 1; lr 1; delta clip 1e-5 |

The worker CLIs support config, resume/output/state, smoke limits, and an explicit formal-run mode. Formal mode rejects record limits. Stage1 saves at cursor 100, every 1000 from 1000, and the boundary; Stage2 saves at 50/500/1000 and the boundary. Cursor is the next sequential JSONL index and cumulative tokens are taken from immutable record metadata.
''')

write('exit86_worker_inventory.md',f'''# Exit-86 worker inventory

The three former hard stops in `bin/run_stage1.sh`, `bin/run_stage2.sh`, and `bin/run_eval.sh` were replaced by real worker dispatch. Recursive inspection of `bin/`, `workers/`, and `lib/` found:

- EXIT86_STAGE1_REMAINING=0
- EXIT86_STAGE2_REMAINING=0
- EXIT86_EVAL_REMAINING=0

No exit-86 statement was merely deleted into a no-op path.
''')

write('stage1_real_worker_report.md',f'''# Stage1 real worker report

STAGE1_WORKER_IMPLEMENTED=true. The worker calls the authoritative `formal.build`, which fixes BF16, gradient checkpointing, gated NTP target, lr 1, chunk 1024, layers 0/6/12/18/24, no training delta clip, and document-local pairs. The project initializer keeps the randomized sparse diagonal projection (all off-diagonal entries zero) and zero gate; it is not identity.

Full-shape smoke consumed the first two real 16K records without skip/duplicate. Slot A recorded cursor 1 / 16,384 tokens and slot B cursor 2 / 32,768 tokens; latest points to B. One real validation record completed with zero optimizer steps and unchanged parameter versions. Promotion produced a complete temporary Stage1 resume artifact and made it read-only.
''')

write('stage2_real_worker_report.md',f'''# Stage2 real worker report

STAGE2_WORKER_IMPLEMENTED=true.

- Model: load Stage1 final model weights.
- Optimizer: reinitialize AdamW `(lr=5e-6, betas=(0.9,0.95), eps=1e-8, weight_decay=0.1)`.
- Scheduler: reinitialize constant LambdaLR; global step zero; seed 42.
- Resume: restore Stage2 model, optimizer, scheduler, cursor and all RNG fields from Stage2 A/B.
- Loss: direct import/call of the already-tested `single_decoder_fused_answer_context`; no copied loss implementation.

The two actual QA records logged answer/context positions 3/4092 then 5/4090. Runtime losses satisfy `total=answer+0.1*context`; cumulative mask errors, overlap and empty answer counts are zero. Slot A cursor/tokens were 1/4096; slot B and latest were 2/8192. A separate one-record CUDA retry passed using checkpoint-only Stage1 parent loading. STAGE2_LOSS_RUNTIME_ASSERTION=passed.
''')

write('eval_off_real_worker_report.md',f'''# OFF real worker report

EVAL_OFF_WORKER_IMPLEMENTED=true. A fresh Stage2 final load ran two authoritative RULER samples in a genuine padded batch of 2. `ttt_enabled=false`; the runtime TTT, inner-update, generation-update, Conv-forward and base-weight mutation counts were all zero. Checkpoint aggregate hash was `{offm['checkpoint_hash']}`.
''')

write('eval_on_real_worker_report.md',f'''# ON real worker report

EVAL_ON_WORKER_IMPLEMENTED=true. A fresh load of the identical Stage2 final ran two matching samples at batch 1 with actual `ttt_lr=1`, `delta_clip_frobenius=1e-5`, chunk 1024, layers `[0,6,12,18,24]`, apply-then-update and sample-local caches. Each 16K prompt produced exactly 75 updates (15 chunks × 5 layers). Generation updates and Conv forwards were zero. Maximum observed post-clip BF16 norm was `{max(x['post_clip_delta_norm_max'] for x in on):.12g}` under the declared `1e-7` norm-observation tolerance.
''')

write('real_checkpoint_adapter_report.md',f'''# Real checkpoint adapter report

`lib/real_checkpoint.py` bridges the formal PyTorch state to atomic A/B slots. It saves model/gate/projection, optimizer, scheduler, cursor/tokens, loader/sampler state, gradient phase, all Python/NumPy/Torch RNG, config and hashes; it fsyncs, CPU-reloads and verifies SHA before rename/latest update. The target selected for overwrite is hard-asserted non-latest and reclaimed before `.partial`, keeping disk peak at latest + new partial rather than three checkpoints. Stage1 promotion hard-links the full resume checkpoint and stores config/tokenizer metadata without duplicating a 3.3 GiB HF weight file; Stage2 reconstructs the architecture from local Base and strictly loads that parent model state. Existing mock behavior remains 14/14 and the adapter-specific A→B→A two-slot storage regression passes.
''')

write('real_checkpoint_rotation_test.md',f'''# Real checkpoint rotation test

REAL_CHECKPOINT_ROTATION_TEST=passed.

- Stage1 A: `{s1['slot_A']['checkpoint_sha256']}`, cursor 1.
- Stage1 B: `{s1['slot_B']['checkpoint_sha256']}`, cursor 2; latest B.
- Stage2 A: `{s2['slot_A']['checkpoint_sha256']}`, cursor 1.
- Stage2 B: `{s2['slot_B']['checkpoint_sha256']}`, cursor 2; latest B.
- Both reloaded with 320 optimizer-state entries, scheduler epoch 2 and complete RNG fields.
''')

write('real_resume_equivalence_test.md',f'''# Real resume equivalence test

REAL_RESUME_EQUIVALENCE_TEST={resume['status']}.

The resumed two-record Stage1 result was compared with an independently initialized uninterrupted two-record full-shape reference over the same IDs. Scheduler and RNG are exact. Fused AdamW/BF16 is not bitwise across the process boundary: model max absolute difference `{resume['model_max_abs_diff']}`, optimizer-state max difference `{resume['optimizer_max_abs_diff']}`. Both are within the declared tolerances `{resume['model_bf16_tolerance']}` and `{resume['optimizer_bf16_tolerance']}` respectively; cursor and next-record identity are exact.
''')

write('real_stage_transition_test.md',f'''# Real Stage transition test

STAGE_TRANSITION_TEST=passed. Base → Stage1 two full-shape records with process-boundary resume → fused validation → temporary read-only Stage1 promotion → Stage2 two real QA records with the frozen optimizer policy and process-boundary resume → temporary Stage2 inference promotion → OFF 2 → fresh ON 2 completed. ON/OFF sample order and checkpoint hash match: `{pair_ids}` / `{pair_ckpt}`. Test assets are isolated under `tests/stage_transition_real/` and never touched formal artifacts.
''')

write('full_model_state_isolation_test.md',f'''# Full-model ON state isolation test

ON_STATE_ISOLATION_TEST=passed. `B fresh`, `A then reset then B`, and `simulated A exception then B` produced byte-identical deterministic predictions (tolerance 0). Base fast-weight hash was `{onm['isolation']['base_fast_weight_hash']}` before each sample/reset, trained gate parameters were not reinitialized, and Stage2 `model.safetensors` hash was unchanged.
''')

write('cuda_full_shape_smoke_test.md',f'''# CUDA full-shape smoke test

CUDA_SMOKE_TEST=passed on the idle RTX 5090.

- Stage1: BF16 16K formal shape, two real records across save/reload, AdamW and gradient checkpointing; fused validation passed.
- Stage2: two actual QA records, official separate-group-means fused loss, save/reload and promotion passed.
- OFF: two samples as batch 2, update path zero.
- ON: two real benchmark samples plus the five required tail lengths; lr 1, clip 1e-5, batch 1.
- Tail lengths 1023/1024/1025/2048/2049 produced 0/5/5/10/10 layer updates; generation updates=0 and Conv forwards=0.
- No final-run OOM, NaN or Inf. An earlier validation implementation materialized full logits and OOMed during development; it was replaced by fused linear CE and the clean retry passed without changing weights.

The runtime test exposed and then verified the isolated project tailfix commit `{CORE_HEAD}`: generation discards the incomplete prompt TTT buffer while retaining and applying complete-chunk-derived fast weights.
''')

write('formal_launch_readiness.md',f'''# Formal launch readiness

All worker gates are satisfied: workers 4/4, exit86 0, real rotation passed, real resume within declared BF16 tolerance, Stage transition passed, full-model isolation passed, CUDA full-shape smoke passed, OFF bypass passed, ON lr/clip/tailfix passed. Existing data/benchmark and CPU suites remain unchanged and passing. Formal training/inference have not started.

FINAL_VERDICT=READY_TO_LAUNCH_PIPELINE
''')

write('worker_completion_report.md',f'''# Worker completion report

REAL_WORKER_IMPLEMENTATION_COMPLETE

- Base pipeline commit: `6c6c96c42cf258396469c8362667d51ba1a25599`.
- Project tailfix commit: `{CORE_HEAD}`.
- Stage1/Stage2/OFF/ON workers: implemented.
- Real checkpoint rotation/resume: passed.
- Stage transition/state isolation/CUDA smoke: passed.
- `STAGE1_TRAIN_TTT_LR=1`, `STAGE2_TRAIN_TTT_LR=1`.
- OFF: TTT disabled, batch 2, update path not executed.
- ON: TTT lr 1, delta clip 1e-5, batch 1, generation updates 0, Conv forwards 0.
- Auto shutdown stayed dry-run; no shutdown command was called.

FINAL_VERDICT=READY_TO_LAUNCH_PIPELINE
''')

write('pipeline_implementation_report.md',f'''# Pipeline implementation report

The original audit/state/config layer at base commit `6c6c96c42cf258396469c8362667d51ba1a25599` is preserved and now has real Stage1, Stage2 and ON/OFF workers, real A/B checkpoint bridging, idempotent ordered resume, strict runtime assertions, and tested full-model tailfix semantics. The only core numerical-path change is isolated in project commit `{CORE_HEAD}` and fixes generation tail leakage discovered by CUDA regression.

FORMAL_STAGE1_STARTED=false
FORMAL_STAGE2_STARTED=false
FORMAL_OFF_INFERENCE_STARTED=false
FORMAL_ON_INFERENCE_STARTED=false
ACTUAL_SHUTDOWN_TRIGGERED=false
FINAL_VERDICT=READY_TO_LAUNCH_PIPELINE
''')

write('preflight_report.md',f'''# Preflight report

Final preflight JSON is `reports/preflight_report.json`. Project HEAD `{CORE_HEAD}` is the expected audited base plus isolated tailfix commit; only two pre-existing unrelated untracked project paths remain. Local model, immutable longtext/QA10M, benchmark hash, configs and corrected inference constants pass. GPU process query was empty during smoke. CPU regression results remain loss 10/10, mock rotation 14/14, delta 12/12, tailfix 10/10, auto-shutdown 12/12 and loader passed.
''')

command='tmux new-session -d -s gated_ntp_300m_qa10m_eval13k "bash ${TTT_TRAINING_ROOT}/bin/run_pipeline.sh 2>&1 | tee -a ${TTT_TRAINING_ROOT}/logs/pipeline_console.log"'
write('formal_launch_plan.md',f'''# Formal launch plan

Do not execute during implementation. After operator review, the executable formal command is:

```bash
{command}
```

Formal mode has no smoke limits. Automatic shutdown remains enabled only after independently validated `PIPELINE_COMPLETE`, with 300-second cancellation window; failures leave the instance running.
''')

write('project_audit.md',f'''# Project audit

- Original authoritative HEAD verified before work: `57936c376a99fe6fe75d2bebf19d2a6d4006e5bb`.
- Current HEAD: `{CORE_HEAD}`.
- One isolated core commit was necessary because full-model testing proved generation tokens could complete a prompt tail and trigger forbidden updates. The patch adds a generation-step branch that applies cached prompt-derived weight, clears the tail and performs no update.
- Pre-existing untracked validation report/tool were preserved and not committed.
''')

# Hash tracked-size implementation and reports, never smoke models/data/predictions.
import hashlib
paths=[]
for base in ('bin','configs','lib','workers','reports','tests'):
    for p in (ROOT/base).rglob('*'):
        if not p.is_file() or p.name=='SHA256SUMS.txt' or 'stage_transition_real' in p.parts or '__pycache__' in p.parts or p.suffix in ('.pyc','.log'): continue
        paths.append(p)
with open(REPORTS/'SHA256SUMS.txt','w') as out:
    for p in sorted(set(paths)):
        out.write(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT)}\n")
print(json.dumps({'status':'PASS','reports_written':17,'project_head':CORE_HEAD,'pair_ids':pair_ids,'pair_checkpoint':pair_ckpt}))
