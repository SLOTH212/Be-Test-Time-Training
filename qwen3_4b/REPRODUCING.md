# 复现与仓库整合

## 放进现有仓库

将本目录内容放入 `Be-Test-Time-Training/qwen3_4b/`，保留 README 和校验清单；无需把 ZIP 本身提交到 Git。然后从仓库根目录审查并提交：

```bash
git add qwen3_4b
git diff --cached --stat
git commit -m "Add Qwen3-4B training and final RULER Dynamic results"
```

本次交付没有执行 GitHub 推送。

## 重跑需要的外部资产

1. 从合法来源准备 Qwen3-4B Base、训练所需的冻结 ProLong/MRQA 派生数据与 RULER 32K benchmark。原数据/tokenizer/权重不在包内。历史身份见 provenance 和 model_metadata。
2. 如只重跑评测，还需要对应 Stage2 FINAL 权重；只取得原始 Base 权重不能复现 Stage2 结果。
3. `data_preparation` 是冻结输入驱动的历史构造实现，Stage1 脚本含 Windows msvcrt 依赖。外部 authority、缓存和历史 QA bundle 需要另外准备；不能凭此包从零自动恢复全部历史输入。
4. 参考 `provenance/requirements_frozen.txt` 和 `INSTALLED_ENVIRONMENT.json` 构建兼容环境。历史 pip freeze 包含平台专属 CUDA 包和本地 file URL，不能未经调整直接当作通用 requirements 安装。
5. 在独立工作目录映射路径、GPU UUID 和可用设备；重新生成当前代码/data/model hash 绑定并运行 CPU/GPU preflight。保留本归档作为证据，不原地篡改历史记录来伪造匹配。
6. 训练入口由 `work/qwen3_4b_stage1_stage2_pipeline_formal_v1/pipeline_supervisor.py` 调度；核心实际实现位于 `src/training_runtime/code/workers/`。评测由 `src/qwen3_4b_downstream_pipeline_v1/formal_control.py` / `formal_worker.py` 调用 V2 executor。
7. 八卡 Dynamic 的 `work/qwen3_4b_dynamic_gpu07_v2/entry.py` 是历史资源覆盖层。原六卡 config/store identity 保持不变，八卡分片由独立 resume plan 记录；这不是配置错误。历史计划中的已完成项和 PID 不能作为新任务启动依据。

包中的启动、恢复、清理和 finalizer 脚本均为研究源代码，勿未经适配直接执行。没有提供声称已测试通过的一键跨机器复现入口。
