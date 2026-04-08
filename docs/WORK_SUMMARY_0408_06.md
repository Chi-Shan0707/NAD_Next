# WORK SUMMARY 0408_06 (UTC)

生成时间：`2026-04-08 06:00 UTC`（按小时精度记录）

## 1. 本轮完成的工作

### 1.1 DeepSets 主线扩展
- 新增最小 DeepSets 通用核心：`nad/core/selectors/deepsets_core.py`
- 新增三条实现：
  - `nad/core/selectors/gpqa_deepsets_impl.py`
  - `nad/core/selectors/code_deepsets_impl.py`
  - `nad/core/selectors/math_deepsets_impl.py`
- 对应新增 plugin 与 runner 脚本，完成可复跑实验流程。

### 1.2 Best-of-N 提交补丁导出
- 新增脚本：`scripts/patch_bestofn_submission_with_math_deepsets_round1.py`
- 在当前 promoted stack（`code_v2 + science_hybrid_round3`）基础上，仅替换 math slice，导出新 Best-of-N 文件。

### 1.3 EarlyStop 导出线
- 完成 `earlystop_svd_lowrank_lr_v1` 导出链路。
- 完成 SVM bridge（同一模型复用导出 BestofN + EarlyStop）。

## 2. 方法与任务类型

### 2.1 方法
- 采用最小 DeepSets（小 MLP + group pooling），严格使用 run-level structured features。
- 不使用 attention / Set Transformer / raw neuron rows。

### 2.2 任务类型
- Science：`GPQA`
- Coding：`LCB v5`
- Math：`aime24 / aime25 / brumo25 / hmmt25`
- Submission：`cache_test` blind 导出（Best-of-N + EarlyStop）

## 3. 各线成效（含 validate 口径）

### 3.1 GPQA DeepSets（离线 validate：是）
- 最佳候选（文档锁定）：`gpqa_deepsets_round1_max_pairaux0p25`
- 指标：`AUROC 65.38%`，`Hit@1 66.67%`，`Pairwise 59.60%`，`SelAcc@10 72.32%`
- 结论：`NO-PROMOTE`（Hit@1 未超过当前 science_hybrid_round3 的 guardrail）

### 3.2 Coding DeepSets（离线 validate：是）
- 最佳候选：`code_deepsets_round1_max`
- 指标：`AUROC 52.47%`，`Hit@1 62.28%`，`Pairwise 50.29%`，`SelAcc@10 55.75%`
- 相对 `code_v2`：Hit@1 小升，但 SelAcc@10 明显回落
- 结论：`NO-PROMOTE`

### 3.3 Math DeepSets（离线 validate：是）
- 最佳候选：`math_deepsets_round1_max_pairaux0p25`
- 指标：`AUROC 86.26%`，`Hit@1 77.50%`，`Pairwise 74.42%`，`SelAcc@10 98.70%`
- patched system proxy（sample-weighted）相对 current：
  - `Hit@1 +1.03pp`
  - `SelAcc@10 +6.22pp`
- 结论：`PROMOTE`（math gate 与 system gate 均通过）

### 3.4 Submission validate 结果（结构校验）
- `submission/BestofN/extreme12/patches/extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank__code_v2_lcb__science_hybrid_round3_gpqa__math_deepsets_round1_patch.json`
  - payload validate：`cache_keys=12, problems=970, samples=62080`
- `submission/BestofN/extreme12/patches/extreme12_svm_bridge_bestofn_v1.json`
  - payload validate：`cache_keys=12, problems=970, samples=62080`
- `submission/EarlyStop/earlystop_svd_lowrank_lr_v1.json`
  - blind submission 导出覆盖 `12` cache keys
- `submission/EarlyStop/earlystop_from_bestofn_svm_bridge_v1.json`
  - blind submission 导出覆盖 `12` cache keys

> 注：submission validate 属于结构/覆盖校验，不代表 leaderboard 在线分数验证。

## 4. 提交状态
- 已完成相关 commit 并 push 到 `origin/main`。
- 关键提交：
  - `9868d31` selectors: add code math deepsets and export bestofn patch
  - `1446977` submission: add bestofn svm bridge and earlystop reuse
