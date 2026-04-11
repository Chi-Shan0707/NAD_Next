# SVDomain

本目录用于新的 SVD 工作面，不再继续扩散到旧的 `earlystop_prefix10_*` 命名。

## Current Canonical IDs

- `es_svd_math_rr_r1`
- `es_svd_science_rr_r1`
- `es_svd_ms_rr_r1`
- `es_svd_coding_rr_r1`

## Naming

- `es_svd`：EarlyStop SVD family
- `math` / `science` / `ms`：覆盖域
- `coding`：coding 单域模型
- `rr`：`raw+rank`
- `r1`：当前分域重训第一版

说明：

- 名字里故意去掉 `prefix` / `p10`
- 但训练协议仍然使用 4 anchors：`10 / 40 / 70 / 100`
- `20 / 30 -> 10`，`50 / 60 -> 40`，`80 / 90 -> 70`

## Current Training Script

- `SVDomain/train_es_svd_ms_rr_r1.py`
- `SVDomain/train_es_svd_coding_rr_r1.py`

## Current Protocol

- 数据：`MUI_HUB/cache` + `MUI_HUB/cache_train`
- 切分：`85 / 15` holdout，按 `dataset + problem_id` 跨 root 一致切分
- 域：`math` / `science` 分开训练
- 表示：只用 `raw+rank`
- 特征：固定为旧 `earlystop_prefix10_svd_round1` noncoding 主特征组
- 路由：不允许 baseline / single-feature route 进入最终模型
- coding：本轮故意不纳入最终 bundle，避免把旧 baseline routing 带进来

## Coding Workflow

- `es_svd_coding_rr_r1`：只在 coding 域训练，仍然使用 `10 / 40 / 70 / 100` 四个 anchors。
- 训练特征与 `es_svd_ms_rr_r1` 完全一致：固定 `raw+rank` + `token_plus_traj` 特征组。
- 当前可用监督数据里，`cache_train` 没有 coding cache，因此 coding 训练实际来自 `MUI_HUB/cache/DS-R1/lcb_v5`，再做 `85 / 15` holdout。
- blind 导出不重跑 math/science：以 `submission/EarlyStop/es_svd_ms_rr_r1__coding_from_round1c.json` 为 base，只替换 `DS-R1/lcb_v5` 与 `Qwen3-4B/lcb_v5` 两个 coding cache。
- merged submission 命名：`es_svd_ms_rr_r1__coding_rr_r1.json`。

## Current Coding Artifacts

- model：`models/ml_selectors/es_svd_coding_rr_r1.pkl`
- doc：`docs/ES_SVD_CODING_RR_R1.md`
- summary：`results/scans/earlystop/es_svd_coding_rr_r1_summary.json`
- eval：`results/scans/earlystop/es_svd_coding_rr_r1_eval.json`
- blind coding scores：`results/scans/earlystop/es_svd_coding_rr_r1_blind_coding_scores.json`
- merged submission：`submission/EarlyStop/es_svd_ms_rr_r1__coding_rr_r1.json`

## Current Coding Result

- 这版 `es_svd_coding_rr_r1` 在 coding holdout 上没有超过 `tok_conf_prefix_mean_v1`，也没有超过旧 `earlystop_prefix10_svd_round1`。
- 因此该 merged submission 已生成，但从当前 holdout 证据看，不应默认替代 `es_svd_ms_rr_r1__coding_from_round1c` 作为更优主提交。
- 线上提交 `#147 / es_svd_ms_rr_r1__coding_rr_r1` 已验证：`primary score` 仍为 `3.8125`，但 `auc_of_auroc` 从 `0.7428` 降到 `0.7427`，`auc_of_selacc` 从 `0.8317` 降到 `0.8276`，结论不变。

## Current RL Checkpoint Ranking

- `submission`：`submission/CheckpointRanking/es_svd_math_rr_r1/es_svd_math_rr_r1__math5000rl_slot100_meanconf.json`
- `local eval`：`results/scans/checkpoint_ranking/es_svd_math_rr_r1/es_svd_math_rr_r1__math5000rl_slot100_meanconf_eval.json`
- `doc`：`docs/ES_SVD_MATH_RL_CHECKPOINT_RANKING.md`
- `leaderboard receipt`：`submission/resultofleaderboard/Checkpoint Ranking — es_svd_math_rr_r1__math5000rl_slot100_meanconf.txt`
- 线上结果：`drfit / rank 3`，`Spearman ρ=0.7364`，`Pearson r=0.8398`，`Kendall τ=0.6000`，`Top-1=0`，`Top-3=0`。
