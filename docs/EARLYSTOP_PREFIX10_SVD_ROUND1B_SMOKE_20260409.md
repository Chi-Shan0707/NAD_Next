# EARLYSTOP PREFIX10 / ANCHOR4 SVD ROUND1B RESULTS (2026-04-09)

## 开头确认

- repo 已经有 `earlystop_svd_lowrank_lr_v1` 的导出链路
- 这次任务只做 Early-Stop
- 目标是提高公开/综合 Early-Stop 分数，不是扩平台
- 本轮核心假设：prefix 前 10% 已包含足够强的早停信号，值得单独训练一个 prefix-10 专用 low-rank 模型

## 1. 我确认的当前 repo 状态

- repo 已有 `earlystop_svd_lowrank_lr_v1` 的完整导出链路；本轮仍只做 Early-Stop。
- round1 的离线提升来自 `MUI_HUB/cache` 训练侧 / 同源 direct-eval，没有把 `cache_train` 纳入正式协议。
- round1b 改成：扩大训练集，并对 non-coding 按 `dataset + problem_id` 做确定性 holdout，再继续筛选与训练。

## 2. 训练 / 自测 / holdout 口径

- `主 labeled root`：`/home/jovyan/public-ro/MUI_HUB/cache`。
- `额外 labeled root`：`/home/jovyan/public-ro/MUI_HUB/cache_train`。
- `holdout 规则`：对 non-coding（math + science）按 `dataset + problem_id` 做确定性 `20 / 80 split`；同一题若同时出现在 `cache` 与 `cache_train`，会被分到同一侧，避免跨 root 泄露。
- `coding`：`cache_train` 没有 `lcb_v5`，因此 coding 仍只来自 `MUI_HUB/cache`，并继续保留 `earlystop_svd_lowrank_lr_v1` fallback。
- `train-side direct-eval`：在 split-train 上打分，只用于看训练侧增益；本轮 train slice 共 `10` 个 problem-slices / `640` 个 samples。
- `holdout self-test`：在 split-holdout 上打分，这是本轮主选择口径；holdout slice 共 `0` 个 problem-slices / `0` 个 samples。

### non-coding split 摘要

| Dataset | Unique Problems | Train | Holdout |
|---|---:|---:|---:|
| aime24 | 1 | 1 | 0 |
| aime25 | 1 | 1 | 0 |
| brumo25 | 1 | 1 | 0 |
| gpqa | 1 | 1 | 0 |
| hmmt25 | 1 | 1 | 0 |

## 3. prefix-10 特征定义

### 保留特征

- `token_only`: `tok_conf_*`、`tok_gini_*`、`tok_neg_entropy_*`、`tok_selfcert_*`、`tok_logprob_*` 与对应 token availability flags。
- `token_plus_traj`: 上述 token 特征 + `traj_continuity`、`traj_reflection_count`、`traj_novelty`、`traj_max_reflection`、`traj_late_convergence` + `has_rows_bank`。
- `all`: `token_plus_traj` + `nc_mean`、`nc_slope` + 全部 availability flags。

### 删除特征

- `self_similarity`：会把更后段信息泄露进 prefix 视角；round1b 继续删除。

## 4. 5/10/15/20 checkpoint 对照（holdout）

### 全域统一单 checkpoint / holdout

| Checkpoint | AUROC | SelAcc@10 | StopAcc |
|---|---:|---:|---:|
| 5% | N/A | N/A | N/A |
| 10% | N/A | N/A | N/A |
| 15% | N/A | N/A | N/A |
| 20% | N/A | N/A | N/A |

### 非 coding 单 checkpoint / holdout

| Checkpoint | AUROC | SelAcc@10 | StopAcc |
|---|---:|---:|---:|
| 5% | N/A | N/A | N/A |
| 10% | N/A | N/A | N/A |
| 15% | N/A | N/A | N/A |
| 20% | N/A | N/A | N/A |

- 本轮决策以 holdout 为准，不再用 train-side direct-eval 决定是否推进。
- holdout control 结果里，全域最好是 `5%`，非 coding 最好是 `5%`。

## 5. 与 `earlystop_svd_lowrank_lr_v1` 的对比（holdout）

### holdout 总体对比

| Method | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| tok_conf_prefix_mean_v1 | N/A | N/A | N/A | N/A | N/A |
| earlystop_from_bestofn_svm_bridge_v1 | N/A | N/A | N/A | N/A | N/A |
| earlystop_svd_lowrank_lr_v1 | N/A | N/A | N/A | N/A | N/A |
| global_anchor4 | N/A | N/A | N/A | N/A | N/A |
| noncoding_anchor4_coding_v1 | N/A | N/A | N/A | N/A | N/A |

### holdout best candidate per-cache

| Cache | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|

## 6. 训练侧 direct-eval（只作诊断，不作决策）

### train-side 总体对比

| Method | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| tok_conf_prefix_mean_v1 | N/A | 93.86% | N/A | N/A | 90.00% |
| earlystop_from_bestofn_svm_bridge_v1 | N/A | 94.14% | N/A | N/A | 100.00% |
| earlystop_svd_lowrank_lr_v1 | N/A | 94.00% | N/A | N/A | 100.00% |
| global_anchor4 | N/A | 91.86% | N/A | N/A | 90.00% |
| global_anchor4 | N/A | 91.86% | N/A | N/A | 90.00% |

- holdout winner 在 train-side 的方法名仍是 `global_anchor4`。
- 这里的提升仅用于判断是否存在明显过拟合；最终是否推进，只看 holdout 表。

## 7. 是否导出新 submission

- `holdout winner`：`global_anchor4`。
- `holdout 是否严格胜出 v1`：`NO`。
- `本轮是否导出`：`NO`。
- 判定理由：未满足严格胜出条件。

## 8. 如果没有胜出，失败原因是什么

- 未满足严格胜出条件.

## 9. 改了哪些文件

- `docs/EARLYSTOP_PREFIX10_SVD_ROUND1B_PLAN_20260409.md`
- `docs/EARLYSTOP_PREFIX10_SVD_ROUND1B_RESULTS_20260409.md`
- `scripts/run_earlystop_prefix10_svd_round1b.py`
- `models/ml_selectors/earlystop_prefix10_svd_round1b_smoke.pkl`

## 10. 如何复跑

```bash
bash cookbook/00_setup/verify.sh
python3 scripts/run_earlystop_prefix10_svd_round1b.py \
  --main-cache-root MUI_HUB/cache \
  --extra-cache-root /home/jovyan/public-ro/MUI_HUB/cache_train
```

### full-fit candidate

- 保存路径：`models/ml_selectors/earlystop_prefix10_svd_round1b_smoke.pkl`。
- full-fit 采用的 winner family：`global_anchor4`。

