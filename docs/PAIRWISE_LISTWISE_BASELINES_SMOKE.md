# Pairwise / Listwise Ranking Baselines

## Protocol

- Repository: `work/NAD_Next` only.
- Grouping unit for ranking: within-problem runs only; no cross-problem pairs.
- Informative-group filter: drop within-problem groups that do not contain both positive and non-positive labels.
- Holdout unit: `dataset + problem_id`, matched across cache roots, `holdout_split=0.15`, `split_seed=42`.
- Optimization target for model selection: grouped CV `pairwise_acc`, tie-broken by `hit@1`, `ndcg@3`, then `mrr`.
- Reported metrics: holdout `pairwise_acc`, `hit@1`, `hit@3`, `mrr`, `ndcg@3`, `auroc`, `selacc@10%`; for EarlyStop single-anchor eval, `stop_acc = hit@1`.
- Full machine-readable summary: `/home/jovyan/work/NAD_Next/results/scans/pairwise_listwise_rankers/summary.smoke.json`.

## Aggregate Holdout Comparison

| Slice | Pointwise `pairwise_acc` | Ranking-family `pairwise_acc` | Pointwise `hit@1` | Ranking-family `hit@1` |
| --- | ---: | ---: | ---: | ---: |
| Best-of-N math | nan | nan | nan | nan |
| EarlyStop math | nan | nan | nan | nan |
| EarlyStop science | nan | nan | nan | nan |
| EarlyStop coding (late anchors) | nan | nan | nan | nan |

## Explicit Answers

### 1) Are pairwise objectives better than pointwise objectives?

They are effectively tied on this split. The mean holdout `pairwise_acc` across the four summary slices is `0.0000` for pointwise logistic versus `0.0000` for the ranking-family baselines.

### 2) Does ranking loss help coding more than math/science?

Not clearly; coding and math/science gain by similar amounts. Coding changes by `+0.0000` `pairwise_acc` versus the pointwise control, while the mean noncoding EarlyStop delta is `+0.0000`.

### 3) Does pairwise-on-frozen-z beat pointwise-on-frozen-z?

They are effectively tied. Across EarlyStop slices, frozen-z pointwise logistic reaches `nan` holdout `pairwise_acc`, while frozen-z pairwise logistic reaches `nan`.

### 4) Should the paper treat coding as a “ranking-head mismatch” rather than a total representation failure?

Not yet; the current results do not support a strong ranking-head-mismatch interpretation.

The concrete evidence used here is the same-representation comparison on coding late anchors: raw-feature pointwise logistic versus pairwise/listwise objectives, plus the frozen-z pointwise versus frozen-z pairwise comparison.

## Notes

- Best-of-N math uses the existing augmented math feature family `all_aug`.
- EarlyStop math/science/coding use the shared feature family `token_plus_traj_fixed` with representation `raw+rank`.
- The CSV contains one row per task/domain/anchor/model candidate after final refit on the train split and evaluation on the grouped holdout split.
