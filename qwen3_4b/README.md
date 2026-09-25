# Qwen3-4B：Stage1 → Stage2 → RULER 32K

本目录是 Qwen3-4B 的训练、推理及评测研究快照，整理于 2026-09-25。可作为 `SLOTH212/Be-Test-Time-Training` 仓库中的独立 `qwen3_4b/` 子目录，与 Llama 内容并存。

## 最终结果

Dynamic V2 最终 authority 为 **PASS**，1,278 个搜索样本全部完成。完整评测口径为 12 个任务、6,000 个样本，不含 CWE。

| 方法 | 12 任务平均得分 |
|---|---:|
| OFF | 85.6264% |
| 最佳统一固定动作 L0 | 85.6419% |
| Fixed oracle / SampleBest（逐样本选最佳固定动作） | 86.3922% |
| Dynamic | **87.2406%** |

Dynamic 相对 Fixed oracle **+0.8483 个百分点**，task-stratified paired bootstrap 95% 区间 **[+0.6439, +1.0633] 个百分点**（10,000 次）。相对 OFF +1.6142 个百分点。

4,722 个 Fixed 满分样本跳过搜索并沿用满分；1,278 个搜索样本中，85 个改善、1,193 个持平、0 个下降。改善样本平均增加 59.8824 个百分点；55 个样本超过已搜索的 OFF 与单一非 OFF 动作组合的最佳值。这里的对照受相同搜索预算限制，不是穷举所有组合。

Fixed8 和 Base 的 13 任务 / 6,500 样本结果也保留，包含 CWE，与上述 12 任务结果不能直接混比。13 任务 Base 为 82.3915%，Stage2 OFF 为 82.2905%，L0 为 82.3187%，Fixed oracle 为 83.2344%。

**科学解释：** SampleBest 和 Dynamic 使用参考答案评分选择动作，属于 oracle 搜索，并非训练出的可部署路由器。Dynamic 包含固定动作候选，不下降部分由搜索设计保证。收益只能支持此 benchmark 和搜索协议，不能据此宣称可泛化策略已经有效。

## 训练与搜索设置

- Stage1：1B 为数据逻辑预算；实际训练 989,996,971 输入 token、19,168 次更新。
- Stage2：30,000,000 输入 token、463 次更新。
- 上下文 32,768；训练与正式推理 chunk 均为 4,096。
- TTT 层：0、6、12、18、24、30；动作：OFF、L0、L6、L12、L18、L24、L30、ALL。
- Dynamic：beam=4，1 sweep，forward position order，逐样本 reset，重算下游更新，不拼接旧 delta；无 tail/generation update。
- 正式训练使用 GPU4–5；Dynamic 先用六卡，后扩至八卡，并在服务器重启后完成续跑。资源扩展保留原科学配置及 store identity。
- 任务名中的 `_16k` 是历史名称，本次实际使用 32K 上下文配置。

## 内容

- `src/`、`releases/`、`vendor/`：模型、训练、推理、评分及共享运行时；其中历史、debug、mechanism 工具不代表本次正式执行结论。
- `work/`、`bin/`：训练流水线、预检/恢复实现和正式资源调度脚本。
- `configs/`：科学参数及历史资源配置。
- `data_preparation/`：原数据准备脚本与配置；依赖外部冻结输入，不附原始文本。
- `results/training/`：两个阶段的标量训练曲线 CSV、完成记录与流水线收据。
- `results/base/`、`results/fixed8_v2/`：完整逐样本分数及汇总。
- `results/dynamic_v2/`：6,000 行分数、1,278 条裁剪后的最优动作轨迹、完成及最终校验记录。
- `model_metadata/`：模型配置、参数审计和 authority；不包含权重或 tokenizer 资产。
- `provenance/`：源文件、数据、模型和实验参数的历史绑定及已有预检证据。
- `SOURCE_MANIFEST.json`：导出来源文件的原始 SHA256；`SHA256SUMS.txt`：本包实际文件校验值。
- `scripts/check_results.py`：离线验证文件完整性、逐样本聚合及动作轨迹一致性。

## 离线核验

从本目录执行（仅需 Python 3.9+ 标准库，无需 GPU）：

```bash
python scripts/check_results.py
```

这只核验导出内容与结果统计，不会启动训练，也不重新证明历史训练/模型数值等价。

## 复现与发布

见 [REPRODUCING.md](REPRODUCING.md) 和 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。本包是研究归档，并非解压即运行的软件发行版。服务器根路径已改为 `/path/to/ttt`，个人路径已脱敏；历史硬编码路径、GPU UUID、源文件 hash 和数据绑定需要按新环境重建。

包内不含模型权重、检查点、原始训练数据、RULER 提示/答案、实际生成文本、完整候选回放、缓存、环境二进制或账户凭据。源码中的字段名、评分规则和测试用例属于实现本身。

脱敏和结果裁剪改变了文件字节；历史 authority 中的 hash 仍指向服务器原始记录，不应用于校验修改后的导出文件。请使用本包的 SHA256SUMS。共享源码包含历史兼容逻辑，不保证该当前源树逐字节重建每次早期实验。
