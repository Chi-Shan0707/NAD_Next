# EARLYSTOP SVD PROBLEM-CENTERED ROUND1 RESULTS (2026-04-09)

## 开头确认结论

- round1b 正确协议是：non-coding 使用 `cache + cache_train`，并按 `dataset + problem_id` 做跨 root 一致的 `80/20` holdout；coding 继续 fallback 到旧 v1。
- strong-feature 线已经把 family 收窄到 `tok_conf_prefix / tok_conf_recency / traj_reflection_count` 及少量 tail / recovery 信号。
- 当前 holdout winner `strongfeat_noncoding_anchor4_coding_v1_ref020` 的四个 anchors（10/40/70/100）全部使用 `raw+rank`。
- 因此，本轮任务不是“直接去掉 rank”，而是检验更偏题内排序的表示是否能超过当前 `raw+rank`。

## 1. 你确认的当前 repo 状态

- `main root`：`/home/jovyan/public-ro/MUI_HUB/cache`。
- `extra root`：`/home/jovyan/public-ro/MUI_HUB/cache_train`。
- `holdout split`：`80/20` 按 `dataset + problem_id`。
- `max_problems_per_cache`：`8`。

## 2. 为什么本轮不是“删除 rank”

- 目标是检验题内排序表示是否有增益，而不是先验假设 rank 一定无效。
- 因此本轮做并行对照：`raw / rank / raw+rank / centered_raw / centered_raw+rank`（可选再加 `zscore_within_problem_raw`）。

## 3. 你新增了哪些表示

- 本轮实际表示集合：`raw`, `rank`, `raw+rank`, `centered_raw`, `centered_raw+rank`, `zscore_within_problem_raw`。
- `centered_raw` 定义：每题（64 runs）每个 slot 下按特征减去该题均值。

## 4. EarlyStop 结果

### holdout baselines

| Method | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| tok_conf_prefix_mean_v1 | 86.31% | 94.44% | 10% | 88.62% | 77.78% |
| earlystop_from_bestofn_svm_bridge_v1 | 73.71% | 85.90% | 10% | 82.71% | 72.22% |
| earlystop_svd_lowrank_lr_v1 | 75.14% | 86.50% | 10% | 84.28% | 72.22% |
| earlystop_prefix10_svd_round1b_cap8 | 94.08% | 95.04% | 10% | 93.81% | 72.22% |
| earlystop_strongfeat_round1_cap8 | 95.15% | 97.35% | 10% | 95.23% | 72.22% |

### holdout candidates (all representation x threshold)

| Representation | Threshold | Method | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---|---:|---:|---:|---:|---:|
| raw+rank | 0.20 | problem_centered_rawplusrank_noncoding_anchor4_coding_v1_ref020 | 94.84% | 95.73% | 10% | 95.39% | 83.33% |
| raw+rank | 0.30 | problem_centered_rawplusrank_noncoding_anchor4_coding_v1_ref030 | 91.33% | 95.47% | 10% | 94.68% | 66.67% |
| centered_raw | 0.20 | problem_centered_centered_raw_noncoding_anchor4_coding_v1_ref020 | 93.81% | 95.13% | 10% | 94.46% | 72.22% |
| centered_raw+rank | 0.20 | problem_centered_centered_rawplusrank_noncoding_anchor4_coding_v1_ref020 | 93.81% | 95.13% | 10% | 94.46% | 72.22% |
| rank | 0.20 | problem_centered_rank_noncoding_anchor4_coding_v1_ref020 | 93.81% | 95.13% | 10% | 94.46% | 72.22% |
| zscore_within_problem_raw | 0.20 | problem_centered_zscore_wp_raw_noncoding_anchor4_coding_v1_ref020 | 93.81% | 95.13% | 10% | 94.46% | 72.22% |
| raw | 0.20 | problem_centered_raw_noncoding_anchor4_coding_v1_ref020 | 94.02% | 95.04% | 10% | 93.48% | 72.22% |
| raw | 0.30 | problem_centered_raw_noncoding_anchor4_coding_v1_ref030 | 89.85% | 95.04% | 10% | 91.85% | 66.67% |
| centered_raw | 0.30 | problem_centered_centered_raw_noncoding_anchor4_coding_v1_ref030 | 85.11% | 91.45% | 10% | 83.44% | 77.78% |
| centered_raw+rank | 0.30 | problem_centered_centered_rawplusrank_noncoding_anchor4_coding_v1_ref030 | 85.11% | 91.45% | 10% | 83.44% | 77.78% |
| rank | 0.30 | problem_centered_rank_noncoding_anchor4_coding_v1_ref030 | 85.11% | 91.45% | 10% | 83.44% | 77.78% |
| zscore_within_problem_raw | 0.30 | problem_centered_zscore_wp_raw_noncoding_anchor4_coding_v1_ref030 | 85.11% | 91.45% | 10% | 83.44% | 77.78% |

### holdout best-by-representation

| Representation | Threshold | Method | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---|---:|---:|---:|---:|---:|
| centered_raw | 0.20 | problem_centered_centered_raw_noncoding_anchor4_coding_v1_ref020 | 93.81% | 95.13% | 10% | 94.46% | 72.22% |
| centered_raw+rank | 0.20 | problem_centered_centered_rawplusrank_noncoding_anchor4_coding_v1_ref020 | 93.81% | 95.13% | 10% | 94.46% | 72.22% |
| rank | 0.20 | problem_centered_rank_noncoding_anchor4_coding_v1_ref020 | 93.81% | 95.13% | 10% | 94.46% | 72.22% |
| raw | 0.20 | problem_centered_raw_noncoding_anchor4_coding_v1_ref020 | 94.02% | 95.04% | 10% | 93.48% | 72.22% |
| raw+rank | 0.20 | problem_centered_rawplusrank_noncoding_anchor4_coding_v1_ref020 | 94.84% | 95.73% | 10% | 95.39% | 83.33% |
| zscore_within_problem_raw | 0.20 | problem_centered_zscore_wp_raw_noncoding_anchor4_coding_v1_ref020 | 93.81% | 95.13% | 10% | 94.46% | 72.22% |

### holdout winner per-cache

| Cache | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| cache/DS-R1/aime24 | 99.98% | 100.00% | 10% | 100.00% | 50.00% |
| cache/DS-R1/aime25 | 99.33% | 100.00% | 10% | 100.00% | 100.00% |
| cache/DS-R1/brumo25 | N/A | 100.00% | N/A | N/A | 100.00% |
| cache/DS-R1/gpqa | 73.58% | 61.54% | 10% | 71.63% | 50.00% |
| cache/DS-R1/hmmt25 | 97.40% | 100.00% | 10% | 99.39% | 100.00% |
| cache_train/DS-R1/aime25 | 97.46% | 100.00% | 10% | 100.00% | 100.00% |
| cache_train/DS-R1/brumo25 | N/A | 100.00% | N/A | N/A | 100.00% |
| cache_train/DS-R1/gpqa | 96.92% | 100.00% | 10% | 96.68% | 50.00% |
| cache_train/DS-R1/hmmt25 | 99.18% | 100.00% | 10% | 100.00% | 100.00% |

## 5. Slot100 -> BestofN bridge 结果

### slot100 bridge baseline

| Method | Hit@1 | Hit@3 | SelAcc@10 | Pairwise |
|---|---:|---:|---:|---:|
| slot100_strongfeat_current | 73.08% | 84.62% | 95.27% | 70.95% |

### slot100 bridge candidates (all representation x threshold)

| Method | Hit@1 | Hit@3 | SelAcc@10 | Pairwise |
|---|---:|---:|---:|---:|
| problem_centered_rawplusrank_noncoding_anchor4_coding_v1_ref020 | 80.77% | 84.62% | 95.27% | 71.12% |
| problem_centered_centered_raw_noncoding_anchor4_coding_v1_ref030 | 76.92% | 92.31% | 89.35% | 66.72% |
| problem_centered_centered_rawplusrank_noncoding_anchor4_coding_v1_ref030 | 76.92% | 92.31% | 89.35% | 66.72% |
| problem_centered_rank_noncoding_anchor4_coding_v1_ref030 | 76.92% | 92.31% | 89.35% | 66.72% |
| problem_centered_zscore_wp_raw_noncoding_anchor4_coding_v1_ref030 | 76.92% | 92.31% | 89.35% | 66.72% |
| problem_centered_raw_noncoding_anchor4_coding_v1_ref020 | 73.08% | 80.77% | 93.49% | 68.89% |
| problem_centered_centered_raw_noncoding_anchor4_coding_v1_ref020 | 73.08% | 80.77% | 95.27% | 60.28% |
| problem_centered_centered_rawplusrank_noncoding_anchor4_coding_v1_ref020 | 73.08% | 80.77% | 95.27% | 60.28% |
| problem_centered_rank_noncoding_anchor4_coding_v1_ref020 | 73.08% | 80.77% | 95.27% | 60.28% |
| problem_centered_zscore_wp_raw_noncoding_anchor4_coding_v1_ref020 | 73.08% | 80.77% | 95.27% | 60.28% |
| problem_centered_rawplusrank_noncoding_anchor4_coding_v1_ref030 | 69.23% | 80.77% | 93.49% | 70.06% |
| problem_centered_raw_noncoding_anchor4_coding_v1_ref030 | 69.23% | 73.08% | 93.49% | 69.84% |

### slot100 bridge best-by-representation

| Method | Hit@1 | Hit@3 | SelAcc@10 | Pairwise |
|---|---:|---:|---:|---:|
| problem_centered_centered_raw_noncoding_anchor4_coding_v1_ref030 | 76.92% | 92.31% | 89.35% | 66.72% |
| problem_centered_centered_rawplusrank_noncoding_anchor4_coding_v1_ref030 | 76.92% | 92.31% | 89.35% | 66.72% |
| problem_centered_rank_noncoding_anchor4_coding_v1_ref030 | 76.92% | 92.31% | 89.35% | 66.72% |
| problem_centered_raw_noncoding_anchor4_coding_v1_ref020 | 73.08% | 80.77% | 93.49% | 68.89% |
| problem_centered_rawplusrank_noncoding_anchor4_coding_v1_ref020 | 80.77% | 84.62% | 95.27% | 71.12% |
| problem_centered_zscore_wp_raw_noncoding_anchor4_coding_v1_ref030 | 76.92% | 92.31% | 89.35% | 66.72% |

## 6. 最终结论：最值得保留的表示是什么

- holdout winner：`problem_centered_rawplusrank_noncoding_anchor4_coding_v1_ref020`（representation=`raw+rank`，thr=`0.20`）。
- 对应 bridge（sample-weighted）：Hit@1=80.77%，Hit@3=84.62%，SelAcc@10=95.27%，Pairwise=71.12%。

## 7. 如果没有胜过当前 strongest winner，失败原因是什么

- 是否在 EarlyStop holdout 上超过 `earlystop_strongfeat_round1_cap8`：`NO`。
- 是否在 slot100 bridge 上超过当前 baseline：`YES`。
- 判定说明：EarlyStop holdout 未稳定超过 strongfeat 当前 winner

## 8. 必答问题（Q1~Q4）

1. 直接去掉 rank 是否有帮助？**NO，去掉 rank 未带来稳定收益。**
2. `problem-centered raw` 是否比 `raw+rank` 更好？**NO，`centered_raw` 未超过 `raw+rank`。**
3. strongest setting 更偏 global correctness 还是 within-problem ranking？**当前 strongest setting 仍更偏 global correctness，within-problem 显式编码增益有限。**
4. 新表示是否改善 `slot100` bridge 的 Best-of-N 表现？**NO，slot100 bridge 未稳定超过当前 baseline。**

## 9. 改了哪些文件

- `nad/ops/earlystop_svd.py`
- `scripts/run_earlystop_svd_problem_centered_round1.py`
- `docs/EARLYSTOP_SVD_PROBLEM_CENTERED_ROUND1_PLAN_20260409.md`
- `docs/EARLYSTOP_SVD_PROBLEM_CENTERED_ROUND1_RESULTS_20260409.md`
- `results/scans/earlystop/earlystop_svd_problem_centered_round1_cap8_summary.json`
- `results/scans/earlystop/earlystop_svd_problem_centered_round1_cap8_eval.json`
- `models/ml_selectors/earlystop_svd_problem_centered_round1_cap8.pkl`

## 10. 如何复跑

```bash
bash cookbook/00_setup/verify.sh
python3 scripts/run_earlystop_svd_problem_centered_round1.py \
  --main-cache-root MUI_HUB/cache \
  --extra-cache-root /home/jovyan/public-ro/MUI_HUB/cache_train \
  --max-problems-per-cache 8
```

```bash
python3 scripts/run_earlystop_svd_problem_centered_round1.py \
  --main-cache-root MUI_HUB/cache \
  --extra-cache-root /home/jovyan/public-ro/MUI_HUB/cache_train \
  --max-problems-per-cache 0 \
  --run-full-pass
```

