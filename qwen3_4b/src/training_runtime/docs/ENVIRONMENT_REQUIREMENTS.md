# Environment requirements

Local Levels 0–2 used Python 3.11.15, torch 2.12.0+cu130, CUDA-build 13.0, cuDNN 9.2, transformers 4.57.3, tokenizers 0.22.2, safetensors 0.7.0, NumPy 2.4.6, PyYAML 6.0.3, and liger-kernel 0.8.0.

HIT must provide an NCCL-capable PyTorch build with compatible composable FSDP2 and DCP APIs. These versions are expectations, not a claim of HIT parity. Legacy evaluation additionally relies on its external OpenCompass and answer/context fused-loss environment. No dependency is bundled.
