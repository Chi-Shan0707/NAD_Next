# SVD Slot100 Coding Domain R1

## Summary

- 只覆盖 `coding` 域，只训练 `100% slot`。
- 主模型家族仍然是 `raw+rank + low-rank linear`。
- 允许 `code_v2` 核心特征、reflection/dynamic 辅助、terminal 辅助、slice 导数特征混合搜索。

## Key Findings

- 本轮最有信息量的真实结果来自 `cap10` 运行：`results/scans/bestofn_bridge/slot100_svd_code_domain_r1_cap10_summary.json`。
- 没有出现“单头全面胜出”的候选；当前更合理的结论是保留 `dual_head` 建议，而不是宣称单一 winner。
- `Pairwise` 方向上，`svd_code_deriv__pairwise__rank24__w1__C0p10__squared_hinge` 在 capped holdout 上把 `Pairwise` 从当前 baseline 的 `53.75%` 提到 `56.75%`，但 `Hit@1` 从 `50.00%` 降到 `0.00%`。
- `Hit@1` 方向上，`svd_code_dyn__pairwise__rank16__w1__C3p00__hinge` 在 capped holdout 上只与当前 baseline 打平 `50.00%`，没有形成明确优势；但在 full-system proxy 里给出了正的 `sample_weighted ΔHit@1`。
- 负结果不能忽略：当前 `current_svd_slot100` baseline 仍然在 capped holdout 的 `AUROC / SelAcc@10 / AvgRank proxy` 上更稳。

## Implementation Fixes

- 修复了一个性能根因：`slot100` 训练脚本原先仍在抽取全部 early-stop positions，现在只抽取 `1.0` 位置，和任务范围一致。
- 修复了一个比较公平性 bug：`current_svd_slot100` baseline 使用的 `tok_gini_prefix` 已被显式加入 feature extraction 依赖，避免 baseline 因缺特征被低估。
- 保留了 `cap4` 端到端运行产物作为 pipeline sanity check，但当前推荐以 `cap10` 结果为主阅读。

## Protocol

- `main cache root`：`/home/jovyan/public-ro/MUI_HUB/cache`。
- `extra cache root`：`/home/jovyan/public-ro/MUI_HUB/cache_train`。
- `blind cache root`：`/home/jovyan/public-ro/MUI_HUB/cache_test`。
- `holdout split`：`0.2`。
- `split seed`：`42`。
- `cv folds`：`4`。
- `max_problems_per_cache`：`10`。

## Holdout Baselines

| Method | AUROC | Hit@1 | Hit@3 | SelAcc@10 | Pairwise | AvgRank proxy |
|---|---:|---:|---:|---:|---:|---:|
| code_v2 | 51.54% | 0.00% | 0.00% | 15.38% | 50.50% | 4.000 |
| current_svd_slot100 | 68.75% | 50.00% | 50.00% | 46.15% | 53.75% | 6.500 |

## Holdout Candidates

| Candidate | Head | Family | AUROC | Hit@1 | Hit@3 | SelAcc@10 | Pairwise | AvgRank proxy |
|---|---|---|---:|---:|---:|---:|---:|---:|
| svd_code_core__pairwise__rank16__w1__C3p00__hinge | pairwise | svd_code_core | 25.61% | 0.00% | 0.00% | 7.69% | 45.50% | 8.500 |
| svd_code_core__pointwise__rank4__w0__C0p10__cwnone | pointwise | svd_code_core | 70.14% | 0.00% | 0.00% | 30.77% | 45.67% | 9.000 |
| svd_code_deriv__pairwise__rank24__w1__C0p10__squared_hinge | pairwise | svd_code_deriv | 44.96% | 0.00% | 50.00% | 15.38% | 56.75% | 4.500 |
| svd_code_deriv__pointwise__rank8__w1__C0p10__cwbalanced | pointwise | svd_code_deriv | 69.96% | 50.00% | 50.00% | 23.08% | 43.25% | 13.500 |
| svd_code_dyn__pairwise__rank16__w1__C3p00__hinge | pairwise | svd_code_dyn | 26.36% | 50.00% | 50.00% | 0.00% | 46.92% | 21.500 |
| svd_code_dyn__pointwise__rank12__w0__C10p00__cwnone | pointwise | svd_code_dyn | 69.57% | 50.00% | 50.00% | 38.46% | 45.17% | 11.000 |
| svd_code_messy_all__pairwise__rank16__w1__C10p00__squared_hinge | pairwise | svd_code_messy_all | 42.64% | 0.00% | 0.00% | 7.69% | 42.58% | 15.000 |
| svd_code_messy_all__pointwise__rank24__w0__C0p10__cwnone | pointwise | svd_code_messy_all | 72.04% | 0.00% | 50.00% | 30.77% | 48.08% | 20.000 |

## Recommendation

- `deploy_mode`：`dual_head`。
- `hit1 route`：`svd_code_dyn__pairwise__rank16__w1__C3p00__hinge`。
- `pairwise route`：`svd_code_deriv__pairwise__rank24__w1__C0p10__squared_hinge`。

## Interpretation

- `hit1 head` 的意义是：在不继续依赖旧 fallback 路线的前提下，先保留一条真正的 `coding SVD` 候选，它目前更像“继续放大验证”的路线，而不是已经稳定超过 baseline 的路线。
- `pairwise head` 的意义更明确：导数增强的 `pairwise SVD` 已经在 capped holdout 上给出 `Pairwise` 改善，这是当前最像 `BestOfN bridge` 主力候选的 coding SVD 线。
- 当前推荐不是“这两条都已经赢了 baseline”，而是“`Hit@1` 与 `Pairwise` 最优解开始分叉，因此 coding 更适合双头方案，而不是再强迫一个头同时负责所有指标”。
- 这份文档里的 blind patch 结果主要用于验证导出链路；因为本轮使用了 `--blind-max-problems-per-cache 1`，blind 部分不是最终泛化结论。

## Full-System Proxy

### hit1

- `sample_weighted ΔHit@1`：`2.91%`。
- `sample_weighted ΔSelAcc@10`：`1.99%`。
- `equal_cache ΔHit@1`：`1.39%`。
- `equal_cache ΔSelAcc@10`：`0.32%`。

### pairwise

- `sample_weighted ΔHit@1`：`2.60%`。
- `sample_weighted ΔSelAcc@10`：`2.71%`。
- `equal_cache ΔHit@1`：`-0.28%`。
- `equal_cache ΔSelAcc@10`：`4.23%`。

## Artifacts

- `summary json`：`results/scans/bestofn_bridge/slot100_svd_code_domain_r1_cap10_summary.json`。
- `eval json`：`results/scans/bestofn_bridge/slot100_svd_code_domain_r1_cap10_eval.json`。
- `candidate json`：`results/scans/bestofn_bridge/slot100_svd_code_domain_r1_cap10_candidates.json`。
- `doc`：`docs/SVD_SLOT100_DOMAIN_R1_RESULTS_20260411.md`。
- `hit1 model`：`models/ml_selectors/slot100_svd_code_domain_r1_cap10__hit1.pkl`。
- `pairwise model`：`models/ml_selectors/slot100_svd_code_domain_r1_cap10__pairwise.pkl`。
- `hit1 submission`：`submission/BestofN/extreme12/patches/slot100_svd_code_domain_r1_cap10__hit1.json`。
- `pairwise submission`：`submission/BestofN/extreme12/patches/slot100_svd_code_domain_r1_cap10__pairwise.json`。
