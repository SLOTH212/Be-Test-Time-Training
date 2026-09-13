# Data construction and outstanding inputs

Raw datasets and benchmark text are excluded. Obtain sources under their own access and redistribution terms. The source builders preserve source reading, identity/content deduplication, deterministic extension selection, tokenizer checks, packing, quota stops, and QA answer masks. They require frozen authority metadata as well as source text.

## Commands and prerequisites
Qwen: `dynamic-ttt prepare-data --family qwen --step selection`, then `--step order`, then `--step stage1`. These Stage1 replay stages use `DATA_ROOT` and the expected v2 anchor / v3 streaming workspace layout in the source. Some original builders use Windows `msvcrt` file locks. They have not been validated as portable Linux data-build commands.

Stage2: set `QA_BUILD_ROOT` and run `dynamic-ttt prepare-data --family qwen --step stage2`. The recovered-authority directory must contain historical `data/qa_replay_10m_v1/train.jsonl`, the sibling `data/qa_replay_10m_v1_answer_context_masks_v1/masks.jsonl`, tokenizer manifests, and the exact MRQA source files required by the builder. The new extension selection uses deterministic SHA256 ordering; packing is sequential at 32,768 tokens. Source/config hash gates remain intact.

Llama: run `dynamic-ttt prepare-data --family llama --step subsets`, then `--step stage1` or `--step stage2` with the parent frozen packages and tokenizer assets specified in `derive_llama.py`. The included configs describe the supported derived package; parent package identity is essential.

After both Llama stages, run `--step audit-stage1` and `--step audit-stage2`, then `--step finalize`. Finalization requires nonempty passing independent audits and records the installed release builder hashes before packaging. Package file keys use portable relative paths and exclude self-referential authority/package-verification receipts. For Qwen Stage2, `--step audit-stage2` runs the independent answer-mask/package auditor. These commands do not supply the missing historical inputs.

## Exact reconstruction limitation
The historical Stage2 selector, packer, and RNG are explicitly recorded as lost in the authoritative builder. The current recipe replays 1,291 historical physical records (10M tokens, 2,807 QA targets), then adds a deterministic 20M extension and packs a 30M dataset. It does not reconstruct the lost original generation process.

The exact historical train file SHA256 is `95bfb1242b7ccfb93dcbc5c82deee9250a105558d39deea0a458f7225bc65440`; masks SHA256 is `a58a4fec89fe149a544de3102a604cd5c3d84f9e9eb8c6961c598cc15452ec68`. Neither file is distributed here. Stage1 additionally needs a historical 300M raw-anchor manifest with SHA256 `017a74186b419b2c3f566f568e6c2d120723e3ddafb69024138eb6b321c90ae8` and large ordering/selection manifests.

Consequently a fresh public-source-only reproduction of exact formal training data is **not yet available**. Supplying hash-verifiable, lawfully accessible frozen anchors, or authorizing a separately named new dataset protocol, is necessary to close this gap. No new protocol was substituted during release assembly.

The five MRQA inputs are NaturalQuestionsShort, SQuAD, NewsQA, TriviaQA-web, and HotpotQA. Exact compressed-source hashes are retained in `qa_replay.EXPECTED_SOURCE_SHA`; SearchQA and RULER are not construction inputs.
