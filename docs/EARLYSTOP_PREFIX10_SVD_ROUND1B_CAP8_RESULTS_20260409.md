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
- `train-side direct-eval`：在 split-train 上打分，只用于看训练侧增益；本轮 train slice 共 `62` 个 problem-slices / `3968` 个 samples。
- `holdout self-test`：在 split-holdout 上打分，这是本轮主选择口径；holdout slice 共 `18` 个 problem-slices / `1152` 个 samples。

- `capped screening`：本轮每个 cache 最多使用 `8` 个 problems，用于先把正确协议下的筛选跑通；不是 full-data 终版。
- `blind export`：后续 Best-of-N 导出只使用 `cache_test` 做无标签推理；模型选择与超参筛选均不看 `cache_test` 标签。

### 本轮训练侧 / 自测侧提升摘要

- `训练于`：`cache` + `cache_train` 的 non-coding 标注样本，并对 non-coding 做题目级 `80/20` split；coding 仍只来自 `cache/lcb_v5`，并保留旧 `v1` fallback。
- `训练侧（split-train）相对 v1`：`global_anchor4` 的 `AUC of AUROC` 从 `78.35%` 提到 `82.14%`，`AUROC@100%` 从 `81.69%` 提到 `85.19%`；但 `AUC of SelAcc` 从 `99.04%` 小降到 `98.42%`。
- `自测侧（split-holdout）相对 v1`：`global_anchor4` 的 `AUC of SelAcc` 从 `86.50%` 提到 `93.85%`，这是本轮最重要的 holdout 增益；`Stop Acc@100%` 与 v1 持平，均为 `72.22%`。
- `选择逻辑`：本轮只按 holdout self-test 选模型，不按训练侧最好点推进，也不使用 `cache_test` 做任何选型。

### non-coding split 摘要

| Dataset | Unique Problems | Train | Holdout |
|---|---:|---:|---:|
| aime24 | 8 | 6 | 2 |
| aime25 | 8 | 6 | 2 |
| brumo25 | 8 | 6 | 2 |
| gpqa | 8 | 6 | 2 |
| hmmt25 | 8 | 6 | 2 |

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
| 5% | N/A | 94.87% | 83.33% |
| 10% | N/A | 95.73% | 77.78% |
| 15% | N/A | 93.16% | 72.22% |
| 20% | N/A | 91.45% | 72.22% |

### 非 coding 单 checkpoint / holdout

| Checkpoint | AUROC | SelAcc@10 | StopAcc |
|---|---:|---:|---:|
| 5% | N/A | 96.58% | 77.78% |
| 10% | N/A | 93.16% | 72.22% |
| 15% | N/A | 94.02% | 72.22% |
| 20% | N/A | 91.45% | 72.22% |

- 本轮决策以 holdout 为准，不再用 train-side direct-eval 决定是否推进。
- holdout control 结果里，全域最好是 `5%`，非 coding 最好是 `5%`。

## 5. 与 `earlystop_svd_lowrank_lr_v1` 的对比（holdout）

### holdout 总体对比

| Method | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| tok_conf_prefix_mean_v1 | N/A | 94.44% | N/A | N/A | 77.78% |
| earlystop_from_bestofn_svm_bridge_v1 | N/A | 85.90% | N/A | N/A | 72.22% |
| earlystop_svd_lowrank_lr_v1 | N/A | 86.50% | N/A | N/A | 72.22% |
| global_anchor4 | N/A | 93.85% | N/A | N/A | 72.22% |
| noncoding_anchor4_coding_v1 | N/A | 92.56% | N/A | N/A | 72.22% |

### holdout best candidate per-cache

| Cache | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| cache/DS-R1/aime24 | 99.93% | 100.00% | 10% | 100.00% | 50.00% |
| cache/DS-R1/aime25 | 98.45% | 100.00% | 10% | 98.40% | 100.00% |
| cache/DS-R1/brumo25 | N/A | 100.00% | N/A | N/A | 100.00% |
| cache/DS-R1/gpqa | 58.16% | 45.38% | 10% | 60.32% | 50.00% |
| cache/DS-R1/hmmt25 | 97.57% | 100.00% | 10% | 98.58% | 50.00% |
| cache_train/DS-R1/aime25 | 98.61% | 100.00% | 10% | 100.00% | 100.00% |
| cache_train/DS-R1/brumo25 | N/A | 100.00% | N/A | N/A | 100.00% |
| cache_train/DS-R1/gpqa | 91.17% | 99.23% | 10% | 93.98% | 50.00% |
| cache_train/DS-R1/hmmt25 | 99.01% | 100.00% | 10% | 99.12% | 50.00% |

## 6. 训练侧 direct-eval（只作诊断，不作决策）

### train-side 总体对比

| Method | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| tok_conf_prefix_mean_v1 | 76.90% | 98.18% | 10% | 79.52% | 78.33% |
| earlystop_from_bestofn_svm_bridge_v1 | 76.82% | 99.03% | 10% | 81.71% | 79.17% |
| earlystop_svd_lowrank_lr_v1 | 78.35% | 99.04% | 10% | 81.69% | 75.83% |
| global_anchor4 | 82.14% | 98.42% | 10% | 85.19% | 79.17% |
| noncoding_anchor4_coding_v1 | 84.22% | 99.28% | 10% | 89.61% | 79.17% |

- holdout winner 在 train-side 的方法名仍是 `global_anchor4`。
- 这里的提升仅用于判断是否存在明显过拟合；最终是否推进，只看 holdout 表。

## 7. 是否导出新 submission

- `holdout winner`：`global_anchor4`。
- `holdout 是否严格胜出 v1`：`NO`。
- `本轮是否导出`：`NO`。
- 判定理由：未满足严格胜出条件。
- 但在“继续训练并验证后，用当前 best validated model 生成 Best-of-N blind submission”这个后续任务里，仍以 `global_anchor4` 对应的 full-fit bundle `models/ml_selectors/earlystop_prefix10_svd_round1b_cap8.pkl` 作为 blind 推理模型，因为它是当前 round1b 中 holdout 最强的已验证候选。

## 8. 如果没有胜出，失败原因是什么

- 未满足严格胜出条件.

## 9. 改了哪些文件

- `docs/EARLYSTOP_PREFIX10_SVD_ROUND1B_PLAN_20260409.md`
- `docs/EARLYSTOP_PREFIX10_SVD_ROUND1B_CAP8_RESULTS_20260409.md`
- `scripts/run_earlystop_prefix10_svd_round1b.py`
- `models/ml_selectors/earlystop_prefix10_svd_round1b_cap8.pkl`

## 10. 如何复跑

```bash
bash cookbook/00_setup/verify.sh
python3 scripts/run_earlystop_prefix10_svd_round1b.py \
  --main-cache-root MUI_HUB/cache \
  --extra-cache-root /home/jovyan/public-ro/MUI_HUB/cache_train
```

### full-fit candidate

- 保存路径：`models/ml_selectors/earlystop_prefix10_svd_round1b_cap8.pkl`。
- full-fit 采用的 winner family：`global_anchor4`。

## 11. Best-of-N blind 导出（使用当前 best validated model）

- 导出时间：`2026-04-09`。
- 使用模型：`models/ml_selectors/earlystop_prefix10_svd_round1b_cap8.pkl`（slot `100%`）。
- 导出脚本：`scripts/export_bestofn_from_earlystop_svd_model.py`。
- 导出文件：`submission/BestofN/extreme12/patches/extreme12_earlystop_prefix10_svd_round1b_cap8_slot100__svm_bridge_lcb.json`。
- payload 校验：`cache_keys=12`、`problems=970`、`samples=62080`。
- override 策略：`DS-R1/lcb_v5`、`Qwen3-4B/lcb_v5` 直接沿用 `extreme12_svm_bridge_bestofn_v1.json`，避免 coding 侧形态差异拖低本轮 non-coding 主线。

## 12. Submission #123 线上评测结果（Best-of-N）

- submission：`#123`
- method：`extreme12_earlystop_prefix10_svd_round1b_cap8_slot100__svm_bridge_lcb`
- status：`Not best`
- primary score（Average Rank，越低越好）：`3.0000`
- All Metrics：`0.8403`
- AUROC：`0.7406`
- Hit@1：`0.8123`
- Hit@3：`0.9263`
- SelAcc@10%：`0.7252`
- Pairwise Acc：`0.7252`
- 样本规模：`62080` samples，`970` problems

### 为什么 Best-of-N 方法名里带 `earlystop`

- 这个 Best-of-N 分数不是直接训练一个新的 Best-of-N 模型得到的，而是从 `earlystop_prefix10_svd_round1b_cap8` 模型的 `100%` slot 直接提取得分后导出。
- 因此方法名保留了 `earlystop_prefix10_svd_round1b_cap8_slot100`，用于明确“分数来源模型”和“来源 checkpoint”。
- 后缀 `__svm_bridge_lcb` 表示 `lcb_v5` 两个 cache 仍使用 `svm_bridge` override。
