# EARLYSTOP PREFIX10 SVD ROUND1B PLAN (2026-04-09)

## 开头确认

- repo 已经有 `earlystop_svd_lowrank_lr_v1` 的导出链路
- 这次任务只做 Early-Stop
- 目标是提高公开/综合 Early-Stop 分数，不是扩平台
- 本轮核心假设：prefix 前 10% 已包含足够强的早停信号，值得单独训练一个 prefix-10 专用 low-rank 模型

## round1b 目标

- 修正 round1 忽略 `cache_train` 的协议问题。
- 保持方法不变：仍然只做 `low-rank SVD + LogisticRegression`。
- 扩大 labeled train pool，但要给 non-coding 留出一块真正的 holdout。

## 数据协议

- `main root`：`MUI_HUB/cache`
- `extra root`：`/home/jovyan/public-ro/MUI_HUB/cache_train`
- non-coding（math + science）按 `dataset + problem_id` 做确定性 holdout split。
- 同一题如果同时出现在 `cache` 与 `cache_train`，必须落在同一侧，避免跨 root 泄露。
- coding 继续使用 `MUI_HUB/cache` 的 `lcb_v5`，因为 `cache_train` 没有 coding cache。

## 训练与筛选

- 继续只保留 prefix-safe 特征，继续删除 `self_similarity`。
- 继续跑小搜索：
  - family：`token_only` / `token_plus_traj` / `all`
  - rep：`raw` / `rank` / `raw+rank`
  - svd dim：`2,4,6,8,12,16`
  - LR `C`：`0.05,0.1,0.2,0.5,1.0`
  - class weight：`None` / `balanced`
  - whiten：`off` / `on`
- 继续比较：
  - `global_anchor4`
  - `noncoding_anchor4_coding_v1`
  - 基线：`tok_conf_prefix_mean_v1` / `earlystop_from_bestofn_svm_bridge_v1` / `earlystop_svd_lowrank_lr_v1`

## 评测口径

- `train-side direct-eval`：只作为诊断，观察训练侧是否过拟合。
- `holdout self-test`：作为本轮主决策口径。
- checkpoint 对照继续报告 `5% / 10% / 15% / 20%`，不要只报最优点。
- 主要指标继续报告：
  - AUC of SelAcc
  - Earliest > 0.6
  - AUROC@100%
  - Stop Acc@100%

## 产出

- `scripts/run_earlystop_prefix10_svd_round1b.py`
- `results/scans/earlystop/earlystop_prefix10_svd_round1b_summary.json`
- `results/scans/earlystop/earlystop_prefix10_svd_round1b_eval.json`
- `docs/EARLYSTOP_PREFIX10_SVD_ROUND1B_RESULTS_20260409.md`
