# Inference

Run `dynamic-ttt fixed --config CONFIG` followed by `dynamic-ttt dynamic --config CONFIG`. Both expose `--dry-run` for validating plans and available artifacts. The historical JSON is `configs/inference/qwen3_1p7b.json`; the Qwen4B JSON is a separate scale-up setting. Llama's inference adapter is available as Python code; engineering mechanisms are not certified through the historical Qwen mechanism CLI.

Set MODEL_ROOT, DATA_ROOT, OUTPUT_ROOT to local directories before parsing configs. The model directory needs Transformers config/tokenizer assets and the expected `model.safetensors`. The benchmark directory needs `benchmark_manifest.json`, `sample_manifest.jsonl`, and per-task `samples/<task>.jsonl`. Source readers require each sample's ID, task, sample index, input, answers, and explicit generation budget; use the exact frozen schema rather than renaming unrelated data files. No benchmark text ships here.

Fixed emits one committed JSON per sample/action under its configured run root. Aggregate complete long-form records with `dynamic-ttt sample-best --config CONFIG --input fixed.jsonl --output sample_best.json`. Do not aggregate partial action coverage. Dynamic uses the same fixed commits and validates resume identity. Greedy generation uses the tokenizer EOS and per-sample token budget, without generation-time updates.
