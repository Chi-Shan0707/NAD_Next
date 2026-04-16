# EARLYSTOP STRONG FEATURES ROUND1 (2026-04-09)

## 开头确认

- repo 已经有 `earlystop_svd_lowrank_lr_v1` 的导出链路
- 本轮继续只做 Early-Stop，不切换模型家族
- 当前 strongest-feature 证据来自 `docs/reference/FEATURES.md`、`results/selector_comparison/selector_comparison.md` 与 `README.md`
- 本轮目标是把 EarlyStop 特征范围收窄到强单特征及其邻近 tail / recovery 信号，再用 `10/40/70/100` 训练

## 1. 特征收窄原则

- 参考 `docs/reference/FEATURES.md`、`results/selector_comparison/selector_comparison.md` 与 `README.md` 中仓库已记录的 strongest-feature 证据。
- 本轮不再沿用宽特征 `all` 搜索，而是只保留 `tok_conf_prefix` / `tok_conf_recency` / `traj_reflection_count` 及少量 tail / recovery 信号。
- 保留的强特征族：`strong_core3`、`strong_tail5`、`strong_stable6`、`strong_event7`、`strong_recovery8`。
- 反思阈值小搜索：`0.20`, `0.30`。

## 2. 训练 / holdout 协议

- `main root`：`/home/jovyan/public-ro/MUI_HUB/cache`。
- `extra root`：`/home/jovyan/public-ro/MUI_HUB/cache_train`。
- `holdout split`：`80/20`，按 `dataset + problem_id` 跨 root 一致切分。
- `max_problems_per_cache`：`8`。
- `train(non-coding)`：`54` problem-slices / `3456` samples。
- `holdout(non-coding)`：`18` problem-slices / `1152` samples。

## 3. holdout 对比

### holdout baselines

| Method | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| tok_conf_prefix_mean_v1 | 86.31% | 94.44% | 10% | 88.62% | 77.78% |
| earlystop_from_bestofn_svm_bridge_v1 | 73.71% | 85.90% | 10% | 82.71% | 72.22% |
| earlystop_svd_lowrank_lr_v1 | 75.14% | 86.50% | 10% | 84.28% | 72.22% |
| earlystop_prefix10_svd_round1b_cap8 | 92.08% | 94.10% | 10% | 90.27% | 66.67% |

### holdout strong-feature candidates

| Method | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| strongfeat_global_anchor4_ref020 | 94.19% | 95.21% | 10% | 94.86% | 72.22% |
| strongfeat_noncoding_anchor4_coding_v1_ref020 | 94.19% | 95.21% | 10% | 94.86% | 72.22% |
| strongfeat_global_anchor4_ref030 | 91.13% | 95.90% | 10% | 92.03% | 66.67% |
| strongfeat_noncoding_anchor4_coding_v1_ref030 | 91.13% | 95.90% | 10% | 92.03% | 66.67% |

### holdout winner per-cache

| Cache | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| cache/DS-R1/aime24 | 98.54% | 100.00% | 10% | 100.00% | 50.00% |
| cache/DS-R1/aime25 | 95.23% | 100.00% | 10% | 100.00% | 100.00% |
| cache/DS-R1/brumo25 | N/A | 100.00% | N/A | N/A | 100.00% |
| cache/DS-R1/gpqa | 70.32% | 64.62% | 10% | 65.92% | 0.00% |
| cache/DS-R1/hmmt25 | 96.90% | 100.00% | 10% | 99.53% | 50.00% |
| cache_train/DS-R1/aime25 | 84.13% | 100.00% | 10% | 100.00% | 100.00% |
| cache_train/DS-R1/brumo25 | N/A | 100.00% | N/A | N/A | 100.00% |
| cache_train/DS-R1/gpqa | 93.90% | 98.46% | 10% | 78.80% | 50.00% |
| cache_train/DS-R1/hmmt25 | 98.88% | 100.00% | 10% | 99.95% | 50.00% |

## 4. train-side 诊断（non-coding only）

### train baselines

| Method | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| tok_conf_prefix_mean_v1 | 78.10% | 98.75% | 10% | 81.06% | 81.48% |
| earlystop_from_bestofn_svm_bridge_v1 | 77.66% | 99.09% | 10% | 82.94% | 79.63% |
| earlystop_svd_lowrank_lr_v1 | 79.52% | 99.43% | 10% | 82.91% | 75.93% |
| earlystop_prefix10_svd_round1b_cap8 | 85.34% | 99.46% | 10% | 90.33% | 81.48% |

### train strong-feature candidates

| Method | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| strongfeat_global_anchor4_ref020 | 86.80% | 99.66% | 10% | 89.60% | 83.33% |
| strongfeat_noncoding_anchor4_coding_v1_ref020 | 86.80% | 99.66% | 10% | 89.60% | 83.33% |
| strongfeat_global_anchor4_ref030 | 81.81% | 99.09% | 10% | 86.57% | 87.04% |
| strongfeat_noncoding_anchor4_coding_v1_ref030 | 81.81% | 99.09% | 10% | 86.57% | 87.04% |

## 5. winner route summary

- `winner`：`strongfeat_noncoding_anchor4_coding_v1_ref030`。
- `bundle type`：`noncoding_anchor4_coding_v1`。
- `reflection threshold`：`0.30`。

### split-fit winner anchors

| Anchor | Route | Detail |
|---|---|---|
| 10% | svd | strong_event7 / raw+rank / rank=12 / C=0.1 / whiten=False / thr=0.30 |
| 40% | svd | strong_recovery8 / raw+rank / rank=16 / C=0.1 / whiten=False / thr=0.30 |
| 70% | svd | strong_recovery8 / raw+rank / rank=16 / C=1.0 / whiten=False / thr=0.30 |
| 100% | svd | strong_recovery8 / raw+rank / rank=16 / C=0.05 / whiten=True / thr=0.30 |

### full-fit winner anchors

| Anchor | Route | Detail |
|---|---|---|
| 10% | svd | strong_event7 / raw+rank / rank=12 / C=1.0 / whiten=True / thr=0.30 |
| 40% | svd | strong_event7 / raw+rank / rank=12 / C=0.05 / whiten=False / thr=0.30 |
| 70% | svd | strong_recovery8 / raw+rank / rank=16 / C=0.05 / whiten=False / thr=0.30 |
| 100% | svd | strong_recovery8 / raw+rank / rank=16 / C=1.0 / whiten=True / thr=0.30 |

## 6. 结论

- `holdout 是否超过 old SVD v1`：`NO`。
- `holdout 是否超过 round1b cap8`：`YES`。
- `holdout 是否超过 tok_conf baseline`：`YES`。
- `是否建议导出 blind submission`：`NO`。
- `理由`：未稳定超过 `earlystop_svd_lowrank_lr_v1`；Stop Acc@100% 未通过保守 guardrail。

## 7. 改了哪些文件

- `nad/ops/earlystop_svd.py`
- `scripts/run_earlystop_prefix10_svd_round1.py`
- `scripts/run_earlystop_strongfeat_round1.py`
- `docs/EARLYSTOP_SSL_BASELINE_STRONGFEAT_CAP8_REPRO_20260416.md`

## 8. 如何复跑

```bash
bash cookbook/00_setup/verify.sh
python3 scripts/run_earlystop_strongfeat_round1.py \
  --main-cache-root MUI_HUB/cache \
  --extra-cache-root /home/jovyan/public-ro/MUI_HUB/cache_train \
  --max-problems-per-cache 8
```

```bash
python3 scripts/run_earlystop_strongfeat_round1.py \
  --main-cache-root MUI_HUB/cache \
  --extra-cache-root /home/jovyan/public-ro/MUI_HUB/cache_train \
  --max-problems-per-cache 0
```

### full-fit candidate

- 保存路径：`models/ml_selectors/earlystop_ssl_baselines/earlystop_strongfeat_round1_cap8_repro.pkl`。
- 采用方法：`strongfeat_noncoding_anchor4_coding_v1_ref030`。

