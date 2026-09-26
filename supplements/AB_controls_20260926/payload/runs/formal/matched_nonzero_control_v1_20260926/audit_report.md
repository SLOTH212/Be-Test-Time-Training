# Independent CPU audit

{
  "status": "PASS",
  "checks": {
    "A1": true,
    "A2": true,
    "A3": true,
    "A4": true,
    "A5": true,
    "A6": true,
    "A7": true,
    "A8": true,
    "A9": true,
    "A10": true,
    "A11": true,
    "A12": true
  },
  "source_integrity": true,
  "final_integrity": true,
  "protocol_closest_fallback_validated": true,
  "strict_within_1pct": false,
  "magnitude_notes": [
    "Magnitude mismatch: ruler_qa_squad_16k:0161"
  ],
  "closest_exceptions": [
    {
      "sample_id": "ruler_qa_squad_16k:0161",
      "layer": 18,
      "seed": 20260930,
      "relative_norm_error": 0.01173097268340272,
      "closest_fallback_validated": true
    }
  ],
  "completed": 157,
  "errors": []
}

No existing Selective Deletion results were modified. A12 requires explicit reward-blind source-review evidence in config or source integrity check.

## 完整审计说明

| 检查 | 结果与证据 |
|---|---|
| CHECK_A1_COHORT_EXACT | 157 IDs及顺序与正式nontrivial-prefix cohort一致 |
| CHECK_A2_BOUNDARY_EXACT | 所有样本tau/boundary及注入事件chunk编号一致 |
| CHECK_A3_LAYER_RULE_EXACT | 109单分支+48五分支=349分支；种子后分支的sample内平均 |
| CHECK_A4_NATIVE_REPLAY | 157个样本的Native得分、预测hash、边界权重hash复现；每个样本结束再次复现Native |
| CHECK_A5_DELETION_REFERENCE | 读取349个历史分支映射及分数；smoke另有13次Deletion重放一致；A1正式值直接引用 |
| CHECK_A6_NO_OTHER_LAYER_CHANGE | 1745次干预五层before与Native一致；非目标after不变，目标after等于校准权重hash |
| CHECK_A7_PREFIX_KV_NATIVE | 每次干预28个attention层prefix K/V hash与本样本Native exact相同 |
| CHECK_A8_SUFFIX_ACTIONS_NATIVE | 完整动作数组和suffix保持Native；clean replay重算后续状态 |
| CHECK_A9_MAGNITUDE_MATCH | 按允许的最接近解协议PASS；严格1%为1744/1745，1次误差1.173097% |
| CHECK_A10_RANDOM_NONZERO | 1745/1745实际represented norm>0 |
| CHECK_A11_SEED_DETERMINISM | 5个smoke样本13个layer分支：首个seed重复，权重、预测hash、得分exact；所有分支固定5个seed |
| CHECK_A12_NO_REWARD_IN_CONSTRUCTION | 校准函数只接收权重和sample/layer/seed；无reward/reference/scorer；局部RNG与固定搜索预算 |

历史源码、协议、cohort、core和winner来源hash均保持不变。修订前72份结果hash逐一保持不变。源码完整性与最终数据完整性PASS。
