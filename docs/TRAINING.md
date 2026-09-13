# Training

`configs/training/` contains source-derived Qwen3-4B and Llama-3.1-8B Stage1/Stage2 YAMLs. Inspect world size and every dataset/checkpoint hash before launch. Paths expand through MODEL_ROOT, DATA_ROOT, and OUTPUT_ROOT. GPU selection inherits CUDA_VISIBLE_DEVICES; no hardware UUID is supplied.

```sh
torchrun --standalone --nproc_per_node=2 -m dynamic_ttt.training.run --family qwen --stage 1 --config configs/training/qwen3_4b_stage1.yaml
torchrun --standalone --nproc_per_node=2 -m dynamic_ttt.training.run --family qwen --stage 2 --config configs/training/qwen3_4b_stage2.yaml
```

Use the actual configured world size; the command above illustrates a two-rank launch. For Llama change the family, config, and rank count together. The worker validates frozen runtime and data authorities. FSDP2, BF16, activation checkpointing, AdamW, gradient clipping, distributed accounting, and stage-specific DCP resume adapters are retained. CUDA, NCCL and Liger/Triton are required for the full worker; CPU tests do not certify GPU compatibility.

Stage2 predicts labels[1:] from hidden[:-1]. Answer and context masks exclude padding. Loss is sum(answer CE)/A + 0.1*sum(context CE)/C, where A and C count the complete distributed optimizer window; source world-size scaling compensates distributed gradient averaging. The original odd-final/dummy-rank and checkpoint semantics remain. Zero-answer records reject; do not assume an added zero-context policy. CPU fixtures check the actual masks and window counts.

No 0.6B or 1.7B training recipe is invented from a larger model's settings. The 1.7B historical inference/result authority is included independently. A new training run is not automatically the same scientific artifact as the historical checkpoint.
