import json,hashlib,datetime,subprocess
from pathlib import Path
R=Path('/path/to/ttt');W=R/'work/qwen3_4b_stage2_preflight_v1';A=R/'audits/qwen3_4b_stage2_30m_32k_gpu23_preflight_v1'
def load(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
auditpath=A/'INDEPENDENT_FINAL_AUDIT.json';audit=load(auditpath)
assert audit['STAGE2_PRELAUNCH_AUDIT']=='PASS' and all(audit['checks'].values())
inf=load(A/'STAGE2_INFERENCE_HANDOFF_AUDIT.json');assert inf['status']=='PASS' and inf['inference_chunk']==4096
assert not (R/'runs/qwen3_4b_stage2_formal_current.json').exists()
perf=load(A/'STAGE2_THROUGHPUT_AUDIT.json');resume=load(A/'STAGE2_RESUME_COMPARISON.json');density=load(A/'STAGE2_SUPERVISION_DENSITY_AUDIT.json')
cfg=R/'configs/qwen3_4b_stage2_30m_32k_gpu23_formal_v1.yaml';infercfg=R/'configs/qwen3_4b_stage2_inference_4k_v1.json';parent=A/'DEBUG_STAGE1_PARENT_AUTHORITY.json'
peak={}
for name in ['uninterrupted_A','resume_B','final_record_edge']:
 rows=list(map(json.loads,(R/'runs/qwen3_4b_stage2_preflight_v1'/name/'training.jsonl').read_text().splitlines()))
 peak[name]={str(g):max(v['peak_reserved_bytes'] for x in rows for v in x['rank_runtime'] if v['physical_gpu']==g) for g in [2,3]}
gates='TRAINING_RUNTIME_AUTHORITY STAGE2_LOSS_SEMANTICS_AUTHORITY CORRECTED_STAGE2_LOSS_LINEAGE STAGE2_DATASET_RUNTIME_AUTHORITY INDEPENDENT_STAGE2_ANSWER_MASK_AUDIT STAGE2_SHIFT_ALIGNMENT_TEST STAGE2_LOSS_NUMERATOR_DENOMINATOR_AUDIT DISTRIBUTED_STAGE2_MASKED_LOSS STAGE2_SUPERVISION_DENSITY_AUDIT STAGE1_TO_STAGE2_HANDOFF REAL_STAGE2_DATALOADER_PREFLIGHT REAL_STAGE2_2GPU_FORWARD REAL_STAGE2_2GPU_TRAIN_STEP STAGE2_GRADIENT_COVERAGE STAGE2_ZERO_MASK_SAFETY STAGE2_UNEQUAL_MASK_GPU_TEST STAGE2_DCP_SAVE STAGE2_DCP_STATE_ROUNDTRIP STAGE2_RESUME_STATE_INTEGRITY STAGE2_STOPPING_SEMANTICS STAGE2_FINAL_RECORD_EDGE STAGE2_DEBUG_HF_EXPORT STAGE2_HF_STATE_DICT_AUDIT STAGE2_TO_INFERENCE_HANDOFF STAGE2_TO_TTT_INFERENCE_HANDOFF FORMAL_STAGE1_PARENT_CONTRACT FORMAL_STAGE2_DATA_CONTRACT STAGE2_SCIENTIFIC_CONTRACT STAGE2_PRELAUNCH_AUDIT'.split()
v={k:'PASS' for k in gates}
v.update({
'QWEN3_4B_STAGE2_PREFLIGHT_STATUS':'PASS_WITH_WARNING','HOST':'amax','USER':'USER',
'TRAINING_RUNTIME_SHA256':'79d89c95908ea787c1358a651c9003ca14eced1957853154b43041db845c162b',
'TRAINING_RUNTIME_SCOPE':'Frozen package and core unchanged; separately audited user-authorized Stage2 data interface and lazy optimizer restore adapter.',
'STAGE2_LOSS_SOURCE':str(W/'recovered_0p6b/source/loss/answer_context_fused_loss.py'),
'STAGE2_LOSS_SOURCE_SHA256':'25c686f51693fe881ff9fa1c7f6feaeb9a51fcd6f05a05554c900d8a895adf31','STAGE2_LOSS_MODIFIED':False,
'STAGE2_SUPERVISED_POSITION_DEFINITION':'All valid noninitial tokens in each packed record, partitioned into answer spans and complement-context. Question/context tokens contribute at group weight 0.1; answer group weight 1.0. This is not answer-only loss.',
'STAGE2_MASK_SHIFT_RULE':'hidden[:,:-1] predicts labels[:,1:]; input answer interval [start,stop) maps to prediction interval [max(1,start)-1,max(1,stop)-1).',
'STAGE2_LOSS_DENOMINATOR':'Separate global answer-position count and global context-position count within each optimizer window: answer_sum/A + 0.1*context_sum/C.',
'STAGE2_DISTRIBUTED_REDUCTION':'Counts summed across ranks; local loss is world_size*(local_answer_sum/A + 0.1*local_context_sum/C), compensating FSDP averaged gradients. Unequal count GPU loss and gradient fixture passed.',
'STAGE2_PADDING_TREATMENT':'Actual worker tokenizes unpadded text and requires length == n_tokens. Physical storage padding contributes neither training input count nor labels/masks. Frozen helper label -100 and shifted valid masks also verified.',
'STAGE2_ZERO_MASK_BEHAVIOR':'Frozen helper rejects ZERO_ANSWER_DENOMINATOR or ZERO_CONTEXT_DENOMINATOR. GPU fixture coordinates rejected update with zero optimizer steps. Current masks_for rejects a zero-answer record; no arbitrary all-context-zero continuation is claimed. All 925 bound real records have positive answer/context groups. Missing final rank slot uses two zero-weight dummy forwards and no token accounting.',
'STAGE2_ZERO_MASK_SAFETY_SCOPE':'PASS for source-defined rejection plus valid frozen dataset and real empty-rank final edge; not a new zero-loss continuation policy.',
'STAGE2_DATASET_ROOT':str(R/'shared/datasets'),'STAGE2_LOGICAL_TOKENS':30000000,'STAGE2_PHYSICAL_TOKENS':30310400,'STAGE2_PADDING_TOKENS':310400,'STAGE2_RECORD_N':925,'STAGE2_QA_TARGET_N':69321,
'STAGE2_ANSWER_POSITIONS':388806,'STAGE2_CONTEXT_POSITIONS':29610269,
'STAGE2_DATASET_AUTHORITY_SHA256':'e423c322602c5124acc839470079cfe23559b43c0a29eb9042cf1a3a4ee6ddd0',
'STAGE2_PACKAGE_SHA256':'60a54b694982a4a1c21708ba7c177b25101a5bc76a771f542f90e745611ce230',
'DEBUG_STAGE1_PARENT_PATH':str(W/'stage1_debug_parent'),'DEBUG_STAGE1_PARENT_AUTHORITY':str(parent),'PARENT_CLASS':'DEBUG_ONLY',
'PHYSICAL_GPU_ALLOWLIST':[2,3],'FORMAL_WORLD_SIZE':2,'UNAUTHORIZED_GPU_USED':False,'GPU_SHARED_USE_AUTHORIZED':True,
'STAGE2_RESUME_NUMERICAL_PARITY':resume['STAGE2_RESUME_NUMERICAL_PARITY'],
'STAGE2_STEP_TIME_MEAN':perf['step_mean_seconds'],'STAGE2_STEP_TIME_P50':perf['step_p50_seconds'],'STAGE2_STEP_TIME_P95':perf['step_p95_seconds'],
'STAGE2_INPUT_TOKENS_PER_SEC':perf['input_tokens_per_second'],'STAGE2_SUPERVISED_TOKENS_PER_SEC':perf['supervised_all_groups_tokens_per_second'],'STAGE2_ANSWER_TOKENS_PER_SEC':perf['answer_tokens_per_second'],'STAGE2_DATALOADER_WAIT_FRACTION':perf['dataloader_wait_fraction'],
'GPU2_PEAK_MEMORY':max(x['2'] for x in peak.values()),'GPU3_PEAK_MEMORY':max(x['3'] for x in peak.values()),'GPU_PEAK_MEMORY_UNIT':'torch.cuda.max_memory_reserved bytes; process allocator peak, not whole GPU usage','GPU_PEAK_MEMORY_BY_PHASE':peak,
'FORMAL_STAGE2_CONFIG_PATH':str(cfg),'FORMAL_STAGE2_CONFIG_SHA256':sha(cfg),
'FORMAL_STAGE2_LAUNCH_SCRIPT':str(R/'bin/launch_qwen3_4b_stage2_30m_32k_gpu23_formal_v1.sh'),
'FORMAL_STAGE2_STATUS_SCRIPT':str(R/'bin/status_qwen3_4b_stage2_30m_32k_gpu23_formal_v1.sh'),
'FORMAL_STAGE2_TO_INFERENCE_SCRIPT':str(R/'bin/finalize_qwen3_4b_stage2_to_inference_formal_v1.sh'),
'FORMAL_PARENT_SCHEMA':str(R/'configs/qwen3_4b_stage2_parent_authority_schema_v1.json'),
'INFERENCE_CHUNK':4096,'INFERENCE_RUNTIME_AUTHORITY':'PASS_USER_AUTHORIZED_4K_AMENDMENT',
'INFERENCE_RUNTIME_ROOT':str(R/'src/inference_runtime_4k_stage2_v1/code'),
'INFERENCE_RUNTIME_BASE_PACKAGE_SHA256':'65a0d7bd9c8801c92fc30477ab8662d3eb52261612eff2acef0bbeb3edf6ba51',
'INFERENCE_CONFIG_PATH':str(infercfg),'INFERENCE_CONFIG_SHA256':sha(infercfg),
'INFERENCE_USER_AMENDMENT':'推理chunk改为4k',
'DEBUG_STAGE2_HF_PATH':inf['checkpoint'],'DEBUG_INFERENCE_PARENT_AUTHORITY_SHA256':sha(Path(inf['checkpoint'])/'INFERENCE_PARENT_AUTHORITY.json'),
'DEBUG_ONLY':True,'PAPER_RESULT':False,'FORMAL_STAGE2_TRAINING_STARTED':False,
'SAFE_TO_RUN_FORMAL_STAGE2_AFTER_VALID_STAGE1_PARENT':'YES',
'FORMAL_LAUNCH_CONDITION':'A future valid complete Stage1 FINAL authority must pass the parser and hash checks; GPU2/3 must each have >=48000 MiB free. Current DEBUG parent is rejected for formal launch.',
'BLOCKING_PHASE':[],'BLOCKING_REASON':[],
'WARNINGS':[
'User explicitly changed inference chunk from frozen 1024 to 4096. Active runtime is an independently hashed adaptation; original frozen source/package remains unchanged. No claim of numerical equivalence to 1K inference.',
'Exact DCP loaded state passed, but future resumed trajectory is not bitwise equal. Model max_abs=2.288818359375e-05; optimizer max_abs=0.00341796875; not all tensors meet reporting allclose thresholds. No deterministic mode enabled.',
'569.1333 input tokens/s is a shared-GPU first26-record DEBUG measurement with concurrent CPU export, not an isolated full30M ETA or paper benchmark.',
'QA density is 8.2319x and answer-token density is 9.7211x the historical10M dataset. Frozen separate group means retained; no resampling/reweighting.',
'Zero-mask safety means authority-defined rejection and tested empty-rank handling; it does not mean arbitrary zero-mask data can continue training.',
'Only a DEBUG Stage1 parent was used. No formal Stage1 FINAL or Stage2 30M result is asserted.'
],
'INDEPENDENT_AUDIT_PATH':str(auditpath),'INDEPENDENT_AUDIT_CHECK_N':len(audit['checks']),
'supervision_density':density,'resume_numerical_details':resume,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
'phase':'COMPLETED_DEBUG_PREFLIGHT_4K_INFERENCE_USER_AMENDMENT'
})
bound=dict(audit['file_sha256']);bound[str(auditpath)]=sha(auditpath);bound[str(Path(__file__))]=sha(Path(__file__));v['file_sha256']=bound
authority=R/'provenance/QWEN3_4B_STAGE2_30M_32K_GPU23_DEBUG_PREFLIGHT_V1.json';v['DEBUG_PREFLIGHT_AUTHORITY_PATH']=str(authority)
authority.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');ah=sha(authority);authority.with_suffix('.sha256').write_text(ah+'  '+authority.name+'\n')
answers=[
'是。历史0.6B已完成运行的 source_manifest、实际 train_worker::stage2_input、原 loss 文件及完成状态形成可核验谱系；当前 loss 函数 AST 和缩放表达式未改，真实 Liger 数值与梯度对照通过。',
'每条 packed record 的第1个输入 token 以后，所有有效非 padding token 都参与；answer span 属于权重1.0的独立均值，余下 question/context 属于权重0.1的独立均值。不是 answer-only。',
'是。目标位置 t 由 hidden[t-1] 预测；8条独立内容解析样本、全925条区间审计和真实 GPU shifted CE 测试均通过。',
'是。在 GPU2/3 的 answer=1/3、context=6/4 不等计数 fixture 上，独立全局 loss 与梯度对照通过；真实 FSDP2 双卡训练也完成。',
'是。冻结包、authority、顺序清单、packed manifest、配置及原始 train/masks 哈希通过；925条、30M非padding、69,321 QA，全量 tokenizer/边界审计无异常。',
'当前 QA/M=2310.7，历史=280.7，密度8.2319倍；当前 answer positions/M=12960.2，历史=1333.2，密度9.7211倍。没有据此改 loss 或重采样。',
'是，通过标准 DCP model-only consolidation 和同一父权威解析器自动加载，无手工权重键转换。Stage2 AdamW/scheduler/RNG重新初始化；本次父模型是 DEBUG，正式入口必须收到有效Stage1 FINAL权威。',
'是。完成真实 forward-only、13个连续 optimizer steps、恢复后3步及最后一条记录的双卡单步测试；410个参与参数梯度有限，6个未使用conv不要求梯度。',
'是。实际加载时417个模型state-dict项、410个optimizer状态及两卡scheduler/RNG/cursor精确恢复；续跑记录顺序和状态计数正确。未来数值轨迹有已披露差异。',
'是。4K版本导出无权重变化，加载无缺失/多余/形状错误，OFF与L0均通过。按用户最新授权使用独立4K推理适配版；原冻结1K运行时保留，不把4K结果称作1K复现。',
'是。配置、父schema、启动、只读状态和最终导出脚本齐备并通过审计；DEBUG父、错误数据哈希和伪造完成权威均会拒绝。实际正式启动仍需未来Stage1 FINAL父权威和显存检查。',
'是。完整30M正式Stage2训练刻意未启动，所有本次训练/导出都是DEBUG_ONLY，不是paper结果。'
]
lines=[
'# Qwen3-4B Stage2 30M/32K 预检完整报告（推理4K定稿）','',
'最终状态：**PASS_WITH_WARNING**。训练及推理 chunk 均为4096；GPU验证和独立只读审计已完成。正式30M训练没有启动。','',
'适用主机：amax；账户：USER。全部新增文件位于 /path/to/ttt。用户最后指令“推理chunk改为4k”覆盖原冻结推理1024限制；本报告以独立4K推理版本为准。','',
'## 1. 范围与权威','',
'训练基于 scaleup_multigpu_32k_hit_v2 冻结包；原训练worker和checkpoint库保留。新增Stage2候选worker修正数据接口、计数和空rank参与同步，独立恢复适配器处理PyTorch惰性optimizer初始化。训练loss函数及其全局缩放表达式未改。','',
'历史 corrected loss 源自 ssh REMOTE_HOST（实际主机REMOTE_HOST）的 /home/USER/ttt/qwen3_0p6b_replication_v1。已完成的执行入口使用 train_worker.py::stage2_input；缺失的旧builder answer_context_data.py不在该完成运行的实际数据路径中。stage2/state.json记载STAGE2_COMPLETE、cursor1291、10M输入。源码/manifest/完成状态SHA见最终权威与谱系审计。','',
'## 2. 实际冻结训练loss','',
'公式：L = sum(answer CE)/A + 0.1 × sum(context CE)/C。A、C分别是一个optimizer window内跨rank求和的answer/context有效预测位置数，不能换成统一加权分母。每rank本地项乘world_size=2，以抵消FSDP梯度平均。','',
'hidden[:, :-1] 与 labels[:, 1:] 对齐；输入answer区间[a,b)映射为预测下标[max(1,a)-1,max(1,b)-1)。每条记录的输入位置0不被预测。后续有效BOS/EOS若出现，按其mask所属组参与；没有按特殊token ID统一排除。实际样本EOS未出现，合成fixture单独验证EOS作为context目标的语义。','',
'数据存储的padding不进入实际worker的未padding文本tokenization，也不计输入token。历史helper路径验证了padding labels=-100和shifted mask共同忽略padding。','',
'零分母沿用冻结helper的拒绝语义。单rank零answer、另一rank有效的GPU fixture以0次optimizer更新协调退出，无NaN/除零/死锁。该结果不代表允许任意坏mask继续训练。全925条冻结记录两组均非零；真正无数据的末尾rank由两个零权重dummy forward参与FSDP，不增加样本/token计数。','',
'## 3. 数据、mask及密度','',
'数据集：QA_REPLAY_30M_32K_QWEN3_STAGE2_V2。925条记录；30,000,000逻辑/非padding token；30,310,400物理token；310,400 padding；69,321 QA targets。shift后answer=388,806、context=29,610,269，合计29,999,075=30M−925。','',
'全量925条真实tokenization、长度、unit边界、answer区间和packed sidecar对应关系通过，malformed/overlength/OOB/mask mismatch均为0。另对0、1、2、100、400、800、923、924号记录，从实际Question/Answer渲染文本独立解析答案并重建token区间，覆盖首尾、full/partial、multiple-QA与边界。','',
'| 数据集 | Logical tokens | QA targets | QA targets / M | Shifted answer positions / M |',
'|---|---:|---:|---:|---:|',
'| 历史Stage2 10M | 10,000,000 | 2,807 | 280.7 | 1,333.2 |',
'| 当前Stage2 30M | 30,000,000 | 69,321 | 2,310.7 | 12,960.2 |','',
'QA密度约8.232倍，answer预测位置密度约9.721倍。独立分组归一化未改；密度差不等同于按同样倍数改变loss权重。','',
'## 4. 父模型、训练配置与真实GPU测试','',
'模型4,061,881,856个唯一参数、36层、context32768、训练chunk4096、TTT层[0,6,12,18,24,30]。state_dict有417项，embedding/lm_head tied alias导致按项求和大于唯一参数数。','',
'DEBUG Stage1父来自已验证5步checkpoint（cursor10、input286544）；仅标准consolidation得到model-only artifact，不加载Stage1 optimizer。Stage2重新初始化AdamW、constant scheduler、计数和每rank RNG。','',
'AdamW：LR5e-6、betas(0.9,0.95)、eps1e-8、weight_decay0.1、outer_grad_clip1.0；BF16、非重入gradient checkpointing、TTT LR1.0、训练inner clip=None。world_size2、每卡microbatch1、GA1、global batch2，仅物理GPU2/3。','',
'真实预检：forward-only；连续13步/26条/849,150输入token；从第2步checkpoint恢复后续跑3步到cursor10；最后第925条记录单独训练1步。所有410个实际参与参数梯度有限，6个frozen NTP路径未使用的ttt_conv参数无梯度符合源码。','',
'## 5. DCP与恢复','',
'初次严格state审计发现PyTorch在load初始化optimizer时，为6个从未参与梯度的conv额外创建了零moment、step1状态。新增Stage2专用适配器只移除“checkpoint未保存且名字为ttt_conv且moment精确全零且step为0或1”的初始化条目；任何非零/异常项均拒绝。原checkpoint库、已有模型/optimizer状态及loss未改。','',
'修复后真实load审计：417模型项、410optimizer状态精确一致；两rank的完整scheduler、游标和Python/NumPy/CPU/CUDA RNG精确一致。恢复后样本序列、cursor、累计token/answer/context和scheduler/RNG继续与未中断分支一致。','',
'未切换严格确定性模式。续跑后模型最大绝对差2.288818359375e-05，optimizer最大绝对差0.00341796875；417个模型tensor中98个精确相同、309个满足报告阈值；1230个optimizer tensor中410个精确相同、960个满足报告阈值。三个后续loss差为0、0.0035340189933776855、0.0008798092603683472。该数值偏离单独披露，不冒充未来轨迹精确复现。','',
'## 6. 停止语义与最后记录','',
'正式训练按冻结完整数据cursor925结束，对应恰好30M非padding输入；global batch2共463次optimizer更新，最后一个window仅1条真实记录。没有重复/补齐数据集，也不把物理padding计入token预算。','',
'真实最后记录edge：从cursor924开始，GPU2处理24,870输入token（answer620、context24,249），GPU3样本列表为空、计数0；1次optimizer更新后cursor925、DCP保存、两rank barrier与shutdown通过。这个DEBUG分支从0累计输入，因此24,870不能冒充完成30M。','',
'## 7. HF导出与用户授权4K推理','',
'最终HF路径：'+inf['checkpoint']+'。同一正式导出脚本以--debug-test执行DCP→HF；文件只读、拒绝覆盖，含config/tokenizer/state_dict_audit/INFERENCE_PARENT_AUTHORITY。权重哈希与之前导出相同，无手工key转换或参数修改。','',
'原冻结ttt_inference_runtime_v1强制chunk1024，初次4096配置会被拒绝；1024对照已通过并保存。用户随后明确要求“推理chunk改为4k”。因此建立 '+str(R/'src/inference_runtime_4k_stage2_v1/code')+'，仅ttt_state_core.py的chunk默认/校验/错误提示改变；其余推理源码及原冻结包保持原样。','',
'当前推理chunk4096、TTT LR1.0、delta Frobenius clip1e-5、APPLY_THEN_UPDATE、每样本重置、完整prompt chunk更新、尾部和generation不更新。4K改变了chunk边界和更新频率，不能声称与旧1K数值等价；没有因此修改训练科学配置。','',
'CPU同4K分块对照验证原数学公式精确一致，两chunk各4095有效配对，先应用后更新及每文档重置通过。GPU3实际加载缺失/多余/形状不符均为0；OFF用7-token prompt生成1token；L0用4097-token prompt生成1token，记录恰好1次更新及4095有效配对。没有RULER、Fixed sweep、Dynamic或paper benchmark。','',
'Transformers对导出路径发出通用Mistral regex警告；已验证导出与冻结tokenizer内部结构、special token映射一致，代表性真实记录ID一致，未应用会改变冻结tokenization的regex修补。','',
'## 8. 实测吞吐与容量','',
f"连续分支3步warmup+10步measured：mean={perf['step_mean_seconds']:.6f}s，p50={perf['step_p50_seconds']:.6f}s，p95={perf['step_p95_seconds']:.6f}s；input={perf['input_tokens_per_second']:.6f} token/s；所有监督组={perf['supervised_all_groups_tokens_per_second']:.6f}/s；answer-only={perf['answer_tokens_per_second']:.6f}/s；dataloader wait fraction={perf['dataloader_wait_fraction']:.9g}。",'',
f"连续训练每卡torch reserved峰值={peak['uninterrupted_A']['2']} bytes（{peak['uninterrupted_A']['2']/2**30:.4f} GiB）；带完整state恢复审计分支每卡峰值={peak['resume_B']['2']} bytes（{peak['resume_B']['2']/2**30:.4f} GiB）。完整预检GPU2/3峰值字段采用后者。正式脚本要求启动时每卡至少48,000 MiB空闲。",'',
'保存开销约174.6/200.7/205.6秒（连续分支），恢复分支最终保存约211.8秒，final-edge约167.9秒。吞吐测量存在用户授权的共享GPU负载与并行CPU导出，且前26条answer密度偏低；不据此给出干净独占卡的30M ETA，不作为科学性能结论。','',
'## 9. 正式接口与使用边界','',
'正式启动仅接受显式Stage1 FINAL authority，拒绝当前DEBUG父；验证实际model-only manifest、配置/tokenizer/权重哈希、模型层结构、Stage1完成cursor38335和989,996,971输入token。启动还验证冻结Stage2数据、原始文件、候选worker、恢复适配器、最终预检绑定文件和GPU显存。','',
'正式配置保存更新点[25,250,463]，来自历史记录cursor50/500/1000按global batch2换算并在dataset925处最终保存。status脚本只读，报告进程、parent、cursor、输入/answer token、step、loss、throughput、DCP、GPU、elapsed/ETA/errors。最终导出拒绝DEBUG、未完成和失败权威，并核对实际DCP的925/30M/388806/29610269计数。','',
'以下仅是未来使用接口，本次未执行launch：','',
'```bash',
'bash '+str(R/'bin/status_qwen3_4b_stage2_30m_32k_gpu23_formal_v1.sh'),
'bash '+str(R/'bin/launch_qwen3_4b_stage2_30m_32k_gpu23_formal_v1.sh')+' --parent-authority <有效Stage1_FINAL权威JSON>',
'bash '+str(R/'bin/finalize_qwen3_4b_stage2_to_inference_formal_v1.sh')+' --stage2-authority <有效Stage2_FINAL权威JSON> --output <R下尚不存在的HF目录>',
'```','',
'## 10. 最终权威与完整字段','',
'权威文件：'+str(authority),'权威SHA256：'+ah,'',
'下列file_sha256映射由独立审计保存于权威JSON；为便于阅读，本段只展开最终字段，不重复几十个源文件的哈希表。','',
'```json',json.dumps({k:x for k,x in v.items() if k not in ['file_sha256','resume_numerical_details','supervision_density']},ensure_ascii=False,indent=2),'```','',
'## 11. 十二项最终回答','']
lines += [f'{i}. {answer}\n' for i,answer in enumerate(answers,1)]
lines += ['## 12. 证据文件索引','','独立审计检查数：'+str(len(audit['checks']))+'；审计状态PASS。源文件、配置、数据包和收据SHA绑定见权威JSON。原先失败收据及1K对照保留用于追踪，不是当前最终状态。','']
for p in sorted(A.glob('*.json')):
 if any(k in p.name for k in ['INITIAL','BEFORE','NEGATIVE_FAKE','BLOCKER']):continue
 lines.append('- '+str(p))
lines += ['','报告仅证明DEBUG预检和接口准备完成。SAFE_TO_RUN_FORMAL_STAGE2_AFTER_VALID_STAGE1_PARENT=YES是附带有效未来Stage1 FINAL父权威及资源检查的条件性结论；当前DEBUG父不能用于正式启动。']
report=R/'reports/qwen3_4b_stage2_30m_32k_gpu23_preflight_final_v1.md';report.write_text('\n'.join(lines)+'\n');rh=sha(report);report.with_suffix('.sha256').write_text(rh+'  '+report.name+'\n')
print(json.dumps({'status':v['QWEN3_4B_STAGE2_PREFLIGHT_STATUS'],'authority':str(authority),'authority_sha256':ah,'report':str(report),'report_sha256':rh,'config_sha256':sha(cfg),'inference_config_sha256':sha(infercfg),'independent_checks':len(audit['checks']),'formal_started':False},ensure_ascii=False,indent=2))
