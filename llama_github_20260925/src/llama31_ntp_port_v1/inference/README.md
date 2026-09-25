Separate inference source: `source/inference_model/hf_llama/`. Action and fresh
sample-cache adapters: `adapters/inference.py`. Model parameters are immutable
during fast-weight inference; prompt updates live in the per-sample cache.
