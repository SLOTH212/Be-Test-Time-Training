# 1.7B 等预算随机搜索：补充运行记录

本目录补充 Qwen3-1.7B（旧 1K chunk、Stage2）单次等预算均匀随机搜索实验的运行记录。实验对应服务器目录 `runs/formal/random_equal_budget_single_seed_20260923`，于 2026-09-24 07:40 完成。

实验的协议、代码、计划、逐轨迹结果和汇总**已经收在** `dynamic-ttt-v2-paper-20260924/` 中：

- `experiments/random/`：协议、代码（`prepare.py`、`run.py`、`sampler.py`、`finalize.py`）、运行时和冻结记录；
- `data/random/`：随机计划、全部 62,772 条轨迹结果（gzip）、cohort、逐样本汇总、预算审计、bootstrap 输入；
- `results/random/`：最终报告、汇总、bootstrap 审计、结构分析、逐任务汇总。

本目录只补充上面没有的部分：

- `samples/`：170 个样本的逐样本结果，包括随机最优、Dynamic、Sample Best、获胜轨迹及其结构。
- `parity/`：170 个样本的 Dynamic 获胜轨迹核验记录。这些核验不计入随机预算。
- `run_records/`：两个执行台账（gzip）、运行日志、状态与进程记录、权威哈希清单 `AUTHORITY_HASHES.json`，以及服务器运行目录的原始 `SHA256SUMS`。
- `FILE_MANIFEST.json`：每个文件的原始路径、大小和 SHA256；gzip 文件记录的是解压后的值。

两个 worker 的原始结果 `WORKER_RESULTS_0/1.jsonl` 与已上传的合并结果 `RANDOM_RESULTS.jsonl` 是同一批记录；`RANDOM_PLAN.tsv` 与 `RANDOM_PLAN.jsonl` 内容相同。这几个文件都没有重复收入。

## 结果

摘自 `dynamic-ttt-v2-paper-20260924/results/random/RANDOM_FINAL_REPORT.md`。分数为逐样本 scorer 值的均值 ×100，差值单位是百分点。

| 项目 | 数值 |
|---|---:|
| 样本数（Dynamic 提升样本，来自 1,433 个搜索样本） | 170 |
| Sample Best | 40.41 |
| Dynamic | 93.53 |
| 等预算随机搜索 | 89.96 |
| 随机搜索恢复的 Dynamic 增益 | 93.28% |
| Dynamic − 随机（配对、按任务分层的 95% CI） | **+3.57** [1.45, 6.04] |
| 随机 vs Dynamic：追平 / 低于 / 高于 | 159 / 11 / 0 |
| 随机 vs Sample Best：高于 / 持平 | 159 / 11 |

- **预算**：两边都是 62,772 条互不重复、新评估的非恒定轨迹，完全相等。不计入 7 条恒定轨迹和 Dynamic 获胜轨迹的重复核验。
- **随机化**：主种子 20260923，每个样本的种子由 SHA256 派生；无放回均匀抽样，排除恒定轨迹，但不排除与 Dynamic 重合的轨迹；执行前冻结计划（`RANDOM_PLAN_SHA256=fbcc4194…`）。
- **平局**：共享的 Fixed 轨迹优先，否则取冻结计划中最早的随机轨迹。
- **完整性**：计划和实际评估都是 62,772 条，没有缺失、错误、重复执行或未授权重试；checkpoint 与正式 Dynamic 产物均未改变。

## 解释与限制

- 结果只针对以结果为条件选出的 Dynamic 提升 cohort，不能推广到全部搜索样本，也不支持普遍优越性的结论。
- 这是**单次**随机实现。CI 只反映样本不确定性，不反映随机种子的不确定性，也不是随机搜索的期望表现。
- Dynamic 和随机搜索都用参考答案评分选出最优轨迹，属于 oracle 搜索。
- 逐任务结果只作描述。

## 说明

文件保持服务器上的原始字节，与论文包的做法一致，没有做路径脱敏，因此可以用 `run_records/SHA256SUMS` 直接对照。两个执行台账做了 gzip 压缩，解压后字节不变。本目录不含模型权重、benchmark 提示或答案，也不含生成文本；记录中只有预测 hash。
