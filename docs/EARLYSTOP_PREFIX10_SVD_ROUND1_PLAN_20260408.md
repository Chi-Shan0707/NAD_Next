# EARLYSTOP PREFIX10 / ANCHOR4 SVD ROUND1 PLAN (2026-04-08)

## 开头确认

- repo 已经有 `earlystop_svd_lowrank_lr_v1` 的导出链路
- 这次任务只做 Early-Stop
- 目标是提高公开/综合 Early-Stop 分数，不是扩平台
- 本轮核心假设：prefix 前 10% 已包含足够强的早停信号，值得单独训练一个 prefix-10 专用 low-rank 模型

## 目标

- 只沿用现有 `low-rank SVD + LogisticRegression` 范式。
- 所有新特征都必须严格从可见 prefix 定义，不混入后验信息。
- 主比较对象锁定为 Early-Stop：`tok_conf_prefix_mean_v1`、`earlystop_svd_lowrank_lr_v1`、`earlystop_from_bestofn_svm_bridge_v1`。

## 方法

- prefix-safe 特征家族：
  - `token_only`
  - `token_plus_traj`
  - `all`
- 搜索空间严格限制为：
  - `representation ∈ {raw, rank, raw+rank}`
  - `svd dims ∈ {2,4,6,8,12,16}`
  - `logistic C ∈ {0.05,0.1,0.2,0.5,1.0}`
  - `class weight ∈ {none, balanced}`
  - `whiten ∈ {false, true}`
- `self_similarity` 视为不满足 prefix-safe，明确删除。

## 实验设计

- checkpoint 对照：
  - 跑 `5% / 10% / 15% / 20%` 单 checkpoint 模型，验证 10% 是否是最强早段信号。
- 正式 bundle：
  - 训练四个 anchor：`10% / 40% / 70% / 100%`
  - 十槽映射：`10/20/30→10`，`40/50/60→40`，`70/80/90→70`，`100→100`
- 域拆分：
  - `global_anchor4`：全域统一训练
  - `noncoding_anchor4_coding_v1`：math+science 统一训练，coding 复用 `earlystop_svd_lowrank_lr_v1`

## 评测口径

- 本地对齐 leaderboard 风格指标：
  - `AUC of AUROC`
  - `AUC of SelAcc`
  - `Earliest > 0.6`
  - `AUROC@100%`
  - `Stop Acc@100%`
- 聚合方式使用 `6` 个 labeled cache 等权平均。

## 导出规则

- 只有当新候选相对 `earlystop_svd_lowrank_lr_v1` 明确胜出时才导出：
  - `submission/EarlyStop/earlystop_prefix10_svd_round1.json`
- 若未满足严格胜出条件，则保留实验结果与复跑链路，不导出新 submission。
