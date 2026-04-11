# Comparison Tables

这份文档把论文最需要的对比结论压缩成一份简洁摘要。

---

## 1. Holdout 主结论

### Combined noncoding

| Method | AUC of AUROC | AUC of SelAcc | AUROC@100% | StopAcc@100% |
|---|---:|---:|---:|---:|
| tok_conf_prefix_mean_v1 | 71.27% | 94.69% | 72.21% | 73.89% |
| earlystop_svd_lowrank_lr_v1 | 77.58% | 95.82% | 88.01% | 72.59% |
| earlystop_prefix10_svd_round1 | 89.88% | 99.51% | 92.64% | **77.41%** |
| es_svd_ms_rr_r1 | **92.26%** | **99.52%** | **95.05%** | 72.78% |

### Math

| Method | AUC of AUROC | AUC of SelAcc |
|---|---:|---:|
| earlystop_prefix10_svd_round1 | 93.48% | **99.95%** |
| es_svd_math_rr_r1 | **95.81%** | 99.73% |

### Science

| Method | AUC of AUROC | AUC of SelAcc |
|---|---:|---:|
| tok_conf_prefix_mean_v1 | **80.89%** | 96.30% |
| es_svd_science_rr_r1 | 79.85% | **98.80%** |

### Coding

| Method | AUC of AUROC | AUC of SelAcc |
|---|---:|---:|
| tok_conf_prefix_mean_v1 | **50.62%** | **45.81%** |
| es_svd_coding_rr_r1 | 43.42% | 24.00% |

结论：

- canonical `r1` 的 strongest evidence 在 `math` 和 `combined noncoding`
- `science` 更像结构更干净、解释更强，但绝对收益较温和
- `coding` 是明确负结果

---

## 2. Blind leaderboard：EarlyStop 主提交

| Method | Submission ID | Primary Score | AUC-AUROC | AUC-SelAcc | AUROC@100% | StopAcc@100% | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| earlystop_prefix10_svd_round1 | #115 | 4.0000 | 0.7379 | 0.8311 | **0.8492** | **0.7504** | previous best |
| es_svd_ms_rr_r1__coding_from_round1c | #144 | **3.8125** | **0.7428** | **0.8317** | 0.8468 | 0.7299 | CURRENT BEST |
| es_svd_ms_rr_r1__coding_rr_r1 | #147 | **3.8125** | 0.7427 | 0.8276 | 0.8475 | 0.7279 | Not best |

结论：

- `#144` 是论文最值得强调的在线主结果
- `#147` 说明 coding merge 不值得替换主线

---

## 3. Best-of-N：slot100 直抽 vs 历史最佳

| Method | Avg Rank | Hit@1 | Hit@3 | SelAcc@10% | Pairwise Acc | Status |
|---|---:|---:|---:|---:|---:|---|
| extreme12_earlystop_prefix10_svd_round1_slot100__svm_bridge_lcb | **2.4000** | **0.7504** | **0.8049** | 0.9286 | **0.7510** | historical best |
| es_svd_ms_rr_r1__coding_from_round1c__slot100 | 3.6000 | 0.7299 | 0.8010 | **0.9313** | 0.7396 | Not best |

结论：

- 纯 `slot100` 提取不是最优 best-of-n 解
- 它更像桥接 baseline，而不是最终结论

---

## 4. Checkpoint ranking side task

| Method | Rank | Spearman ρ | Pearson r | Kendall τ | Top-1 | Top-3 |
|---|---:|---:|---:|---:|---:|---:|
| es_svd_math_rr_r1__math5000rl_slot100_meanconf | 3 | 0.7364 | 0.8398 | 0.6000 | 0 | 0 |

结论：

- 这是有价值的 side task 证据
- 但不建议抢主 early-stop 结果的正文空间

---

## 5. Interpretability sanity

| Model | Problems | Problem×Anchor | Runs | Max Abs Error | Mean Abs Error |
|---|---:|---:|---:|---:|---:|
| es_svd_math_rr_r1 | 7 | 28 | 1792 | 3.64e-14 | 1.29e-14 |
| es_svd_science_rr_r1 | 2 | 8 | 512 | 9.77e-15 | 2.26e-15 |
| es_svd_ms_rr_r1 | 9 | 36 | 2304 | 3.64e-14 | 1.06e-14 |

结论：

- 解释性分数重构在数值上是闭合的
- 可以放心把 contribution 分析写进 paper
