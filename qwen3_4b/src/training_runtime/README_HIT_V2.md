# scaleup_multigpu_32k_hit_v2

VERSION: scaleup_multigpu_32k_hit_v2

Lineage: original single-GPU 6665596be4cccb51387b7cf6a4f15b338f7d103a; pre-HIT multi-GPU dfe16e97cd6fc37a42d350fd826fe1f9005b67f6; real HIT validated tree 982b3d4dd574f5d867edcb21cff2d3556c7ab5b46d97ad093e56377c60aeb02a; final NTP builder 960465d4c0850345fb998e81b6bf19f2517ecea104aed2946a8b3e31bb416cf3.

Real HIT validation passed on 2 x NVIDIA RTX A6000, physical GPUs 2 and 3, Qwen3-4B-Base, 32768 context, TTT chunk 4096, layers [0,6,12,18,24,30]. Peak allocated memory was 20,690,839,040 bytes per GPU. The measured synthetic-fixture step was about 30.8886 seconds and 2121.69 tokens/second; this is not formal throughput.

The config under configs is SCALEUP_DEBUG_CONFIGURATION. FORMAL_CAPACITY_CONFIG_FROZEN=false. Generic Stage1 and Stage2 configs were not overwritten.

code/bin contains launchers; code/lib and code/workers contain the pipeline, FSDP2, DCP, Stage1 and Stage2 runtime; code/vendor contains the exact NTP source; code/tests contains regressions; audits/hit_l3_l4_v1 contains inherited real 2-GPU evidence.

Next: separately authorized real Stage1-data integration smoke. FORMAL TRAINING IS NOT YET AUTHORIZED.
