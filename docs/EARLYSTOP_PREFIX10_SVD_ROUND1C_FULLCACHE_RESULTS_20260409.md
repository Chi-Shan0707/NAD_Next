# EARLYSTOP PREFIX10 / ANCHOR4 SVD ROUND1C FULLCACHE RESULTS (2026-04-09)

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
- `holdout 规则`：对 non-coding（math + science）按 `dataset + problem_id` 做确定性 `80 / 20 train/holdout split`；同一题若同时出现在 `cache` 与 `cache_train`，会被分到同一侧，避免跨 root 泄露。
- `coding`：`cache_train` 没有 `lcb_v5`，因此 coding 仍只来自 `MUI_HUB/cache`，并继续保留 `earlystop_svd_lowrank_lr_v1` fallback。
- `train-side direct-eval`：在 split-train 上打分，只用于看训练侧增益；本轮 train slice 共 `651` 个 problem-slices / `41664` 个 samples。
- `holdout self-test`：在 split-holdout 上打分，这是本轮主选择口径；holdout slice 共 `122` 个 problem-slices / `7808` 个 samples。

### non-coding split 摘要

| Dataset | Unique Problems | Train | Holdout |
|---|---:|---:|---:|
| aime24 | 30 | 24 | 6 |
| aime25 | 30 | 24 | 6 |
| brumo25 | 30 | 24 | 6 |
| gpqa | 198 | 158 | 40 |
| hmmt25 | 30 | 24 | 6 |

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
| 5% | 76.08% | 92.61% | 69.54% |
| 10% | 79.24% | 98.08% | 71.94% |
| 15% | 82.74% | 98.91% | 64.72% |
| 20% | 83.65% | 98.82% | 65.83% |

### 非 coding 单 checkpoint / holdout

| Checkpoint | AUROC | SelAcc@10 | StopAcc |
|---|---:|---:|---:|
| 5% | 79.33% | 95.41% | 68.98% |
| 10% | 82.79% | 98.54% | 69.26% |
| 15% | 83.51% | 99.48% | 67.13% |
| 20% | 84.19% | 99.41% | 67.69% |

- 本轮决策以 holdout 为准，不再用 train-side direct-eval 决定是否推进。
- holdout control 结果里，全域最好是 `20%`，非 coding 最好是 `20%`。

## 5. 与 `earlystop_svd_lowrank_lr_v1` 的对比（holdout）

### holdout 总体对比

| Method | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| tok_conf_prefix_mean_v1 | 70.47% | 92.78% | 10% | 72.25% | 68.24% |
| earlystop_from_bestofn_svm_bridge_v1 | 73.43% | 91.63% | 10% | 84.75% | 68.98% |
| earlystop_svd_lowrank_lr_v1 | 75.30% | 95.06% | 10% | 85.45% | 66.11% |
| global_anchor4 | 86.53% | 98.86% | 10% | 91.15% | 72.41% |
| noncoding_anchor4_coding_v1 | 87.76% | 99.30% | 10% | 91.52% | 75.93% |

### holdout best candidate per-cache

| Cache | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| cache/DS-R1/aime24 | 94.11% | 100.00% | 10% | 95.01% | 66.67% |
| cache/DS-R1/aime25 | 79.75% | 100.00% | 10% | 92.79% | 83.33% |
| cache/DS-R1/brumo25 | 96.14% | 100.00% | 10% | 97.62% | 83.33% |
| cache/DS-R1/gpqa | 74.63% | 97.11% | 10% | 77.68% | 75.00% |
| cache/DS-R1/hmmt25 | 92.91% | 99.49% | 10% | 94.00% | 83.33% |
| cache_train/DS-R1/aime25 | 87.41% | 98.72% | 10% | 93.11% | 66.67% |
| cache_train/DS-R1/brumo25 | 90.71% | 100.00% | 10% | 94.46% | 83.33% |
| cache_train/DS-R1/gpqa | 79.50% | 98.67% | 10% | 82.90% | 75.00% |
| cache_train/DS-R1/hmmt25 | 94.66% | 99.74% | 10% | 96.13% | 66.67% |

## 6. 训练侧 direct-eval（只作诊断，不作决策）

### train-side 总体对比

| Method | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| tok_conf_prefix_mean_v1 | 69.04% | 89.36% | 10% | 70.39% | 67.55% |
| earlystop_from_bestofn_svm_bridge_v1 | 74.72% | 91.92% | 10% | 82.24% | 72.01% |
| earlystop_svd_lowrank_lr_v1 | 76.55% | 93.72% | 10% | 82.58% | 71.85% |
| global_anchor4 | 81.74% | 93.48% | 10% | 85.00% | 71.52% |
| noncoding_anchor4_coding_v1 | 83.29% | 94.40% | 10% | 86.33% | 71.91% |

- holdout winner 在 train-side 的方法名仍是 `noncoding_anchor4_coding_v1`。
- 这里的提升仅用于判断是否存在明显过拟合；最终是否推进，只看 holdout 表。

## 7. 是否导出新 submission

- `holdout winner`：`noncoding_anchor4_coding_v1`。
- `holdout 是否严格胜出 v1`：`YES`。
- `本轮是否导出`：`NO`。
- 判定理由：holdout split strict dominance over earlystop_svd_lowrank_lr_v1；本轮先不自动导出，保留为下一次 blind 提交候选。

## 8. 如果没有胜出，失败原因是什么

- 不适用：holdout 已严格胜出 v1。

## 9. 改了哪些文件

- `docs/EARLYSTOP_PREFIX10_SVD_ROUND1B_PLAN_20260409.md`
- `docs/EARLYSTOP_PREFIX10_SVD_ROUND1C_FULLCACHE_RESULTS_20260409.md`
- `scripts/run_earlystop_prefix10_svd_round1b.py`
- `models/ml_selectors/earlystop_prefix10_svd_round1c_fullcache.pkl`

## 10. 如何复跑

```bash
bash cookbook/00_setup/verify.sh
python3 scripts/run_earlystop_prefix10_svd_round1b.py \
  --main-cache-root MUI_HUB/cache \
  --extra-cache-root /home/jovyan/public-ro/MUI_HUB/cache_train
```

### full-fit candidate

- 保存路径：`models/ml_selectors/earlystop_prefix10_svd_round1c_fullcache.pkl`。
- full-fit 采用的 winner family：`noncoding_anchor4_coding_v1`。
