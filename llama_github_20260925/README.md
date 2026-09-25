# Llama-3.1-8B：训练与 RULER 32K 实验快照

本包汇总 Llama 的 Stage1/Stage2 训练代码、Fixed8 V2、Dynamic V2（不含 CWE）及 Base 对照结果。导出日期：2026-09-25。未包含模型权重、检查点、原始训练数据、RULER 提示/答案、生成文本、环境二进制或 Qwen 实验结果。

## 结果

| 范围 | Base | Stage2 OFF | Sample Best | Dynamic |
|---|---:|---:|---:|---:|
| 13 类 / 6500 样本，含 CWE | 89.8482% | 91.5818% | 92.0615% | 未运行此完整范围 |
| 12 类 / 6000 样本，不含 CWE | 见逐样本结果 | 92.3953% | 92.8033% | 93.1883% |

Dynamic 实际搜索 570 个未达满分的样本，5430 个满分样本跳过。搜索子集相对 Sample Best：39 个提升、531 个持平、0 个下降，均分提升 4.0526 个百分点；摊到 6000 个样本提升 0.3850 个百分点。

Sample Best 和 Dynamic 都使用参考答案评分选择结果，属于 oracle 搜索；不能视作已学会的部署策略。Dynamic 包含固定动作候选，因此不下降部分由搜索设计保证。这里的提升只支持当前 RULER 实验范围，不能据此证明通用能力完全没有损失。任务名中的 `_16k` 为历史命名，本实验实际上下文配置为 32K。

## 文件

- `src/`：Llama 适配、训练共享运行时、数据构造、Fixed8/Base/Dynamic 和 V2 executor。
- `work/`：实际 Stage1/Stage2 流水线实现，Stage2 最终使用 GPU0–4 五卡版本。
- `configs/`、`bin/`：服务器历史配置与运行脚本。
- `results/base/`、`results/fixed8_v1/`、`results/fixed8_v2/`、`results/dynamic_v2_no_cwe/`：结果、逐样本分数、完成状态和历史 authority。
- `results/dynamic_v2_no_cwe/trajectories.json`：570 个搜索样本的最优动作序列、分数、计数及原始收据 hash；省略全部候选和逐 chunk 状态。
- `provenance/`：Stage2 FINAL、benchmark 和源文件 SHA256 溯源。
- `validation/`：已有 GPU acceptance 记录与本次导出校验。
- `environment-observed.json`：导出时实际环境版本，供参考，不保证与所有历史运行时版本完全相同。
- `scripts/check_results.py`：只依赖 Python 标准库的结果核验程序。

少量目录、模块仍含 Qwen 名称，这是 Llama 使用的共享代码和历史来源，不表示打包了 Qwen 的实验结果。机制 runtime 中的 debug/smoke 记录不等于正式机制实验结论。

## 核验结果

在本目录运行：

```bash
python scripts/check_results.py
```

## 重跑前适配

这是归档研究快照，不是一键可迁移的发布产品。所有服务器根路径替换为 `/path/to/ttt`，个人账户/主机路径已脱敏。先准备合法取得的 Llama-3.1-8B、Stage2 FINAL 权重和冻结 benchmark，再按新环境配置路径、GPU 数量和依赖；不要直接运行历史启动、恢复或 finalizer 脚本。

历史配置使用严格 source/model/data hash 绑定。脱敏、代码复制和后续共享运行时修改意味着历史配置不应直接作为新的启动授权。重跑前必须审阅依赖、重新生成绑定并跑对应 CPU/GPU preflight。Base/V1 runner 的共享依赖后来升级过，本包当前源树不能被宣称为逐字节恢复所有历史运行的代码。

`SOURCE_MANIFEST.json` 保存服务器原始文件的 SHA256；authority 内的 hash 指向原始记录。脱敏后的文件、裁剪后的结果与其原始 hash 不相等是预期行为。`SHA256SUMS.txt` 才是本导出包内容的校验值。动态轨迹为裁剪导出，保留的 payload hash 仅作原始收据溯源。

Stage1 实际约 478.37M 输入 token；Stage2 约 14.39M token。冻结的 Stage1 validation 数据存在，但没有正式 validation-loss 评估记录，不能将其描述为验证集表现已通过。

## 许可与第三方来源

请阅读 `THIRD_PARTY_NOTICES.md`。本包不附加新的开源许可证，也不重新许可 Llama、Transformers 或其他上游代码。
