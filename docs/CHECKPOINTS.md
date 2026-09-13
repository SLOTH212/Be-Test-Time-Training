# Checkpoints

No model weights, optimizer state, DCP shards, or tokenizer bundles are included. Obtain the base model and tokenizer through the model provider's access terms. The adapted Stage1 FINAL and Stage2 FINAL artifacts must be distributed separately with model config, tokenizer, weight hashes and training lineage.

| Artifact | Public location |
|---|---|
| Stage1 FINAL | Pending user publication; no URL assigned |
| Stage2 FINAL | Pending user publication; no URL assigned |

Inference expects the source loader's Transformers directory with `model.safetensors`. Training uses its stage-specific DCP adapters. A DCP checkpoint is not interchangeable with an inference directory by renaming it. Match historical model/checkpoint identities from the inference config before claiming formal reproduction.
