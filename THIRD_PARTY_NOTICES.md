# Third-party notices

LICENSE_STATUS=NEEDS_USER_DECISION

No repository-wide license is assigned. Original project code remains without a newly inserted license. Existing copyright and license headers are retained, including upstream names required for attribution.

The vendored Qwen and Llama configuration/model implementations derive from Hugging Face Transformers and their Qwen / Llama modeling lineage. Existing headers identify Hugging Face, Qwen, EleutherAI, and ByteDance where applicable; those exact headers govern attribution. These files state Apache License 2.0; its text is in `licenses/Apache-2.0.txt`. Release changes relocate imports, separate training/inference adapters and parameterize infrastructure paths; the model adaptations were already present in the source snapshots. The retained In-Place-TTT reference identifies commit `be2324829b0e91c8fd10a74d4b43714fde6676e1` and includes Apache-2.0. That reference does not establish a license for the entire project.

## Files with preserved Apache headers
- `src/dynamic_ttt/models/inference/hf_llama/configuration_llama.py`
- `src/dynamic_ttt/models/inference/hf_llama/modeling_llama.py`
- `src/dynamic_ttt/models/inference/hf_qwen3/configuration_qwen3.py`
- `src/dynamic_ttt/models/inference/hf_qwen3/modeling_qwen3.py`
- `src/dynamic_ttt/models/inference/hf_qwen3_scaleup/configuration_qwen3.py`
- `src/dynamic_ttt/models/inference/hf_qwen3_scaleup/modeling_qwen3.py`
- `src/dynamic_ttt/models/training/hf_llama/configuration_llama.py`
- `src/dynamic_ttt/models/training/hf_llama/modeling_llama.py`
- `src/dynamic_ttt/models/training/hf_qwen3/configuration_qwen3.py`
- `src/dynamic_ttt/models/training/hf_qwen3/modeling_qwen3.py`

## Dependencies and excluded assets
PyTorch, Transformers, NumPy, tokenizers, safetensors, einops, opt_einsum, PyYAML, filelock, pytest, jsonschema and optional Liger are installed as dependencies, not relicensed by this release. Refer to their installed distributions for license texts. Model weights and raw training/benchmark data are excluded and retain their own access and license terms. The local build evidence records exact source paths and hashes without exposing private infrastructure in this public notice.
