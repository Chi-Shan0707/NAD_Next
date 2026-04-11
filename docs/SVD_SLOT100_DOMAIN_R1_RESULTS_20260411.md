# SVD Slot100 Coding Domain R1

## Summary

- 只覆盖 `coding` 域，只训练 `100% slot`。
- 主模型家族仍然是 `raw+rank + low-rank linear`。
- 允许 `code_v2` 核心特征、reflection/dynamic 辅助、terminal 辅助、slice 导数特征混合搜索。

## Key Findings

- 本轮最新、最有信息量的结果来自 `focus20`：`results/scans/bestofn_bridge/slot100_svd_code_domain_r1_focus20_summary.json`。
- 在这个更大的 focused holdout 上，`svd_code_deriv__pointwise__rank4__w0__C0p03__cwbalanced` 同时把 `Hit@1` 从当前 baseline 的 `75.00%` 提到 `100.00%`，并把 `Pairwise` 从 `49.37%` 提到 `60.09%`。
- `svd_code_dyn__pointwise__rank8__w0__C0p03__cwnone` 也给出正结果：`Hit@1=100.00%`，`Pairwise=59.26%`。
- 这说明一个新的经验已经开始稳定成形：`coding` 上真正最值得继续放大的不是“dynamic pairwise 双头分叉”，而是 **低秩 pointwise + derivative features**。
- 负结果也要保留：新 winner 仍然没有赢下 `AUROC` 与 `SelAcc@10`，所以它更像“BestOfN/top-rank head 变强”，而不是全面替代当前 baseline。

## Caveats

- 这次 `focus20` 为了加快搜索，使用了 `--skip-system-proxy` 和 `--skip-blind-export`，所以这里的结论主要针对 `coding holdout`，不是最终全系统发布结论。
- 当前文档已经被更新为 `focus20` 结果；之前的 `cap10` 结果仍保留在 `results/scans/bestofn_bridge/slot100_svd_code_domain_r1_cap10_*` 中，可作为上一个阶段的参考。

## Protocol

- `main cache root`：`/home/jovyan/public-ro/MUI_HUB/cache`。
- `extra cache root`：`/home/jovyan/public-ro/MUI_HUB/cache_train`。
- `blind cache root`：`/home/jovyan/public-ro/MUI_HUB/cache_test`。
- `holdout split`：`0.2`。
- `split seed`：`42`。
- `cv folds`：`5`。
- `max_problems_per_cache`：`20`。

## Holdout Baselines

| Method | AUROC | Hit@1 | Hit@3 | SelAcc@10 | Pairwise | AvgRank proxy |
|---|---:|---:|---:|---:|---:|---:|
| code_v2 | 47.19% | 75.00% | 100.00% | 73.08% | 46.31% | 1.500 |
| current_svd_slot100 | 65.96% | 75.00% | 100.00% | 92.31% | 49.37% | 1.500 |

## Holdout Candidates

| Candidate | Head | Family | AUROC | Hit@1 | Hit@3 | SelAcc@10 | Pairwise | AvgRank proxy |
|---|---|---|---:|---:|---:|---:|---:|---:|
| svd_code_deriv__pairwise__rank8__w1__C1p00__hinge | pairwise | svd_code_deriv | 53.01% | 75.00% | 100.00% | 80.77% | 57.58% | 1.500 |
| svd_code_deriv__pointwise__rank4__w0__C0p03__cwbalanced | pointwise | svd_code_deriv | 61.28% | 100.00% | 100.00% | 80.77% | 60.09% | 1.000 |
| svd_code_dyn__pairwise__rank16__w1__C1p00__hinge | pairwise | svd_code_dyn | 48.69% | 75.00% | 100.00% | 92.31% | 44.56% | 1.250 |
| svd_code_dyn__pointwise__rank8__w0__C0p03__cwnone | pointwise | svd_code_dyn | 58.69% | 100.00% | 100.00% | 88.46% | 59.26% | 1.000 |

## Recommendation

- `deploy_mode`：`single`。
- `hit1 route`：`svd_code_deriv__pointwise__rank4__w0__C0p03__cwbalanced`。
- `pairwise route`：`svd_code_deriv__pointwise__rank4__w0__C0p03__cwbalanced`。

## Interpretation

- 这一轮 focused 搜索的核心收获是：`derivative` 特征不只是对 `pairwise` 有帮助，它和一个很小的 `rank4 pointwise` 头结合后，也能直接改善 `Hit@1`。
- `dynamic` 仍然有价值，但目前更像第二候选，而不是新的主头。
- 因此下一步最合理的动作不再是继续横向乱扫 family，而是围绕 `deriv pointwise rank4/8 + 小 C + 轻正则` 做更大样本验证，然后再补一次完整 blind/export。

## Full-System Proxy

## Artifacts

- `summary json`：`results/scans/bestofn_bridge/slot100_svd_code_domain_r1_focus20_summary.json`。
- `eval json`：`results/scans/bestofn_bridge/slot100_svd_code_domain_r1_focus20_eval.json`。
- `candidate json`：`results/scans/bestofn_bridge/slot100_svd_code_domain_r1_focus20_candidates.json`。
- `doc`：`docs/SVD_SLOT100_DOMAIN_R1_RESULTS_20260411.md`。
- `hit1 model`：`models/ml_selectors/slot100_svd_code_domain_r1_focus20__hit1.pkl`。
- `pairwise model`：`models/ml_selectors/slot100_svd_code_domain_r1_focus20__pairwise.pkl`。
