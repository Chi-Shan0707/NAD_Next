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
- `holdout 规则`：对 non-coding（math + science）按 `dataset + problem_id` 做确定性 `80 / 20 train/holdout split`；同一题若同时出现在 `cache` 与 `cache_train`，会被分到同一侧，避免跨 root 泄露。
- `coding`：`cache_train` 没有 `lcb_v5`，因此 coding 仍只来自 `MUI_HUB/cache`，并继续保留 `earlystop_svd_lowrank_lr_v1` fallback。
- `train-side direct-eval`：在 split-train 上打分，只用于看训练侧增益；本轮 train slice 共 `11` 个 problem-slices / `704` 个 samples。
- `holdout self-test`：在 split-holdout 上打分，这是本轮主选择口径；holdout slice 共 `9` 个 problem-slices / `576` 个 samples。

- `capped screening`：本轮每个 cache 最多使用 `2` 个 problems，用于先把正确协议下的筛选跑通；不是 full-data 终版。
- `blind export`：后续若要做 blind 导出，只允许用 `cache_test` 无标签推理；本文件中的所有选型都不看 `cache_test` 标签。

### 本轮训练侧 / 自测侧提升摘要

- `训练于`：`cache` + `cache_train` 的 non-coding 标注样本，并对 non-coding 做题目级 `80/20` split；coding 仍只来自 `cache/lcb_v5`，并保留旧 `v1` fallback。
- `训练侧（split-train）相对 v1`：`noncoding_anchor4_coding_v1` 的 `AUC of SelAcc` 与 v1 同为 `95.24%`，没有形成明确优势。
- `自测侧（split-holdout）相对 v1`：`noncoding_anchor4_coding_v1` 的 `AUC of SelAcc` 从 `82.70%` 降到 `81.90%`，因此 cap2 只证明协议可跑通，不足以推进导出。
- `选择逻辑`：本轮只按 holdout self-test 决定是否推进，不按训练侧最好点推进，也不使用 `cache_test` 做任何选型。

### non-coding split 摘要

| Dataset | Unique Problems | Train | Holdout |
|---|---:|---:|---:|
| aime24 | 2 | 1 | 1 |
| aime25 | 2 | 1 | 1 |
| brumo25 | 2 | 1 | 1 |
| gpqa | 2 | 1 | 1 |
| hmmt25 | 2 | 1 | 1 |

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
| 5% | N/A | 80.95% | 77.78% |
| 10% | N/A | 82.54% | 77.78% |
| 15% | N/A | 77.78% | 77.78% |
| 20% | N/A | 77.78% | 77.78% |

### 非 coding 单 checkpoint / holdout

| Checkpoint | AUROC | SelAcc@10 | StopAcc |
|---|---:|---:|---:|
| 5% | N/A | 80.95% | 77.78% |
| 10% | N/A | 82.54% | 88.89% |
| 15% | N/A | 79.37% | 77.78% |
| 20% | N/A | 85.71% | 88.89% |

- 本轮决策以 holdout 为准，不再用 train-side direct-eval 决定是否推进。
- holdout control 结果里，全域最好是 `5%`，非 coding 最好是 `5%`。

## 5. 与 `earlystop_svd_lowrank_lr_v1` 的对比（holdout）

### holdout 总体对比

| Method | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| tok_conf_prefix_mean_v1 | N/A | 82.54% | N/A | N/A | 100.00% |
| earlystop_from_bestofn_svm_bridge_v1 | N/A | 80.95% | N/A | N/A | 88.89% |
| earlystop_svd_lowrank_lr_v1 | N/A | 82.70% | N/A | N/A | 77.78% |
| global_anchor4 | N/A | 79.84% | N/A | N/A | 77.78% |
| noncoding_anchor4_coding_v1 | N/A | 81.90% | N/A | N/A | 77.78% |

### holdout best candidate per-cache

| Cache | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| cache/DS-R1/aime24 | 72.40% | 91.43% | 10% | 58.66% | 100.00% |
| cache/DS-R1/aime25 | 76.35% | 100.00% | 10% | 44.44% | 100.00% |
| cache/DS-R1/brumo25 | N/A | 100.00% | N/A | N/A | 100.00% |
| cache/DS-R1/gpqa | 47.13% | 38.57% | N/A | 52.31% | 0.00% |
| cache/DS-R1/hmmt25 | N/A | 100.00% | N/A | N/A | 100.00% |
| cache_train/DS-R1/aime25 | N/A | 100.00% | N/A | N/A | 100.00% |
| cache_train/DS-R1/brumo25 | N/A | 100.00% | N/A | N/A | 100.00% |
| cache_train/DS-R1/gpqa | 31.75% | 7.14% | N/A | 28.32% | 0.00% |
| cache_train/DS-R1/hmmt25 | N/A | 100.00% | N/A | N/A | 100.00% |

## 6. 训练侧 direct-eval（只作诊断，不作决策）

### train-side 总体对比

| Method | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| tok_conf_prefix_mean_v1 | N/A | 93.48% | N/A | N/A | 95.00% |
| earlystop_from_bestofn_svm_bridge_v1 | N/A | 94.88% | N/A | N/A | 100.00% |
| earlystop_svd_lowrank_lr_v1 | N/A | 95.24% | N/A | N/A | 100.00% |
| global_anchor4 | N/A | 94.69% | N/A | N/A | 95.00% |
| noncoding_anchor4_coding_v1 | N/A | 95.24% | N/A | N/A | 100.00% |

- holdout winner 在 train-side 的方法名仍是 `noncoding_anchor4_coding_v1`。
- 这里的提升仅用于判断是否存在明显过拟合；最终是否推进，只看 holdout 表。

## 7. 是否导出新 submission

- `holdout winner`：`noncoding_anchor4_coding_v1`。
- `holdout 是否严格胜出 v1`：`NO`。
- `本轮是否导出`：`NO`。
- 判定理由：AUC of SelAcc 未超过 v1。

## 8. 如果没有胜出，失败原因是什么

- AUC of SelAcc 未超过 v1.

## 9. 改了哪些文件

- `docs/EARLYSTOP_PREFIX10_SVD_ROUND1B_PLAN_20260409.md`
- `docs/EARLYSTOP_PREFIX10_SVD_ROUND1B_CAP2_RESULTS_20260409.md`
- `scripts/run_earlystop_prefix10_svd_round1b.py`
- `models/ml_selectors/earlystop_prefix10_svd_round1b_cap2.pkl`

## 10. 如何复跑

```bash
bash cookbook/00_setup/verify.sh
python3 scripts/run_earlystop_prefix10_svd_round1b.py \
  --main-cache-root MUI_HUB/cache \
  --extra-cache-root /home/jovyan/public-ro/MUI_HUB/cache_train
```

### full-fit candidate

- 保存路径：`models/ml_selectors/earlystop_prefix10_svd_round1b_cap2.pkl`。
- full-fit 采用的 winner family：`noncoding_anchor4_coding_v1`。
