# Future HIT deployment

No transfer has been executed. Transfer this archive and the separately frozen
inference runtime archive through your approved deployment process. Required
inference archive SHA256:
65a0d7bd9c8801c92fc30477ab8662d3eb52261612eff2acef0bbeb3edf6ba51

Extract each into its own directory. Set TTT_INFERENCE_RUNTIME_ROOT to the extracted
inference directory and TTT_INFERENCE_RUNTIME_ARCHIVE to its unchanged archive.
Use an existing compatible Python environment with NumPy, PyTorch and Transformers.
Run audits/verify_package.py and tests/level0.py plus tests/level1.py with
CUDA_VISIBLE_DEVICES='' before any experiment. Configure model and benchmark paths
already available on HIT; neither asset is duplicated in this package.

Freeze model-specific Fixed and Dynamic results first. Replace template paths and
real identities, select the mechanism and statistical procedure, and use a new
output directory. The model's configured TTT layer set must include every candidate
layer. Static success is not a claim of GPU numerical parity on a future model;
perform model-specific native replay and factor-isolation gates during execution.
The execution adapter enforces native score/prediction parity before interventions.
