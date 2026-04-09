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
- `max_problems_per_cache`：`1`。
- `train(non-coding)`：`9` problem-slices / `576` samples。
- `holdout(non-coding)`：`0` problem-slices / `0` samples。

## 3. holdout 对比

### holdout baselines

| Method | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| tok_conf_prefix_mean_v1 | N/A | N/A | N/A | N/A | N/A |
| earlystop_from_bestofn_svm_bridge_v1 | N/A | N/A | N/A | N/A | N/A |
| earlystop_svd_lowrank_lr_v1 | N/A | N/A | N/A | N/A | N/A |
| earlystop_prefix10_svd_round1b_cap8 | N/A | N/A | N/A | N/A | N/A |

### holdout strong-feature candidates

| Method | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| strongfeat_global_anchor4_ref020 | N/A | N/A | N/A | N/A | N/A |
| strongfeat_noncoding_anchor4_coding_v1_ref020 | N/A | N/A | N/A | N/A | N/A |
| strongfeat_global_anchor4_ref030 | N/A | N/A | N/A | N/A | N/A |
| strongfeat_noncoding_anchor4_coding_v1_ref030 | N/A | N/A | N/A | N/A | N/A |

### holdout winner per-cache

| Cache | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|

## 4. train-side 诊断（non-coding only）

### train baselines

| Method | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| tok_conf_prefix_mean_v1 | N/A | 99.68% | N/A | N/A | 100.00% |
| earlystop_from_bestofn_svm_bridge_v1 | N/A | 99.52% | N/A | N/A | 100.00% |
| earlystop_svd_lowrank_lr_v1 | N/A | 99.84% | N/A | N/A | 100.00% |
| earlystop_prefix10_svd_round1b_cap8 | N/A | 100.00% | N/A | N/A | 100.00% |

### train strong-feature candidates

| Method | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| strongfeat_global_anchor4_ref020 | N/A | 100.00% | N/A | N/A | 100.00% |
| strongfeat_noncoding_anchor4_coding_v1_ref020 | N/A | 100.00% | N/A | N/A | 100.00% |
| strongfeat_global_anchor4_ref030 | N/A | 99.21% | N/A | N/A | 100.00% |
| strongfeat_noncoding_anchor4_coding_v1_ref030 | N/A | 99.21% | N/A | N/A | 100.00% |

## 5. winner route summary

- `winner`：`strongfeat_global_anchor4_ref020`。
- `bundle type`：`global_anchor4`。
- `reflection threshold`：`0.20`。

### split-fit winner anchors

| Anchor | Route | Detail |
|---|---|---|
| 10% | svd | strong_tail5 / raw+rank / rank=8 / C=1.0 / whiten=True / thr=0.20 |
| 40% | svd | strong_stable6 / raw / rank=4 / C=0.05 / whiten=True / thr=0.20 |
| 70% | svd | strong_tail5 / raw / rank=4 / C=0.05 / whiten=True / thr=0.20 |
| 100% | svd | strong_core3 / raw / rank=2 / C=0.05 / whiten=True / thr=0.20 |

### full-fit winner anchors

| Anchor | Route | Detail |
|---|---|---|
| 10% | svd | strong_tail5 / raw+rank / rank=8 / C=1.0 / whiten=True / thr=0.20 |
| 40% | svd | strong_stable6 / raw / rank=4 / C=0.05 / whiten=True / thr=0.20 |
| 70% | svd | strong_tail5 / raw / rank=4 / C=0.05 / whiten=True / thr=0.20 |
| 100% | svd | strong_core3 / raw / rank=2 / C=0.05 / whiten=True / thr=0.20 |

## 6. 结论

- `holdout 是否超过 old SVD v1`：`NO`。
- `holdout 是否超过 round1b cap8`：`NO`。
- `holdout 是否超过 tok_conf baseline`：`NO`。
- `是否建议导出 blind submission`：`NO`。
- `理由`：未稳定超过 `earlystop_svd_lowrank_lr_v1`；AUC of SelAcc 未超过 `earlystop_prefix10_svd_round1b_cap8`；AUC of SelAcc 未超过 `tok_conf_prefix_mean_v1`；Stop Acc@100% 未通过保守 guardrail。

## 7. 改了哪些文件

- `nad/ops/earlystop_svd.py`
- `scripts/run_earlystop_prefix10_svd_round1.py`
- `scripts/run_earlystop_strongfeat_round1.py`
- `docs/EARLYSTOP_STRONG_FEATURES_ROUND1_SMOKE_20260409.md`

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

- 保存路径：`models/ml_selectors/earlystop_strongfeat_round1_smoke.pkl`。
- 采用方法：`strongfeat_global_anchor4_ref020`。

