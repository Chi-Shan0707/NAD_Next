# Pairwise / Listwise Ranking Baselines

## Protocol

- Repository: `work/NAD_Next` only.
- Grouping unit for ranking: within-problem runs only; no cross-problem pairs.
- Informative-group filter: drop within-problem groups that do not contain both positive and non-positive labels.
- Holdout unit: `dataset + problem_id`, matched across cache roots, `holdout_split=0.15`, `split_seed=42`.
- Optimization target for model selection: grouped CV `pairwise_acc`, tie-broken by `hit@1`, `ndcg@3`, then `mrr`.
- For the pointwise-vs-ranking comparison below, the ranking-family row uses the strongest raw ranking baseline per task slice after CV selection among `pairwise_logistic`, `pairwise_linear_svm`, and `xgboost_rank_ndcg`.
- Reported metrics: holdout `pairwise_acc`, `hit@1`, `hit@3`, `mrr`, `ndcg@3`, `auroc`, `selacc@10%`; for EarlyStop single-anchor eval, `stop_acc = hit@1`.
- Full machine-readable summary: `results/scans/pairwise_listwise_rankers/summary.json`.

## Aggregate Holdout Comparison

| Slice | Pointwise `pairwise_acc` | Ranking-family `pairwise_acc` | Pointwise `hit@1` | Ranking-family `hit@1` |
| --- | ---: | ---: | ---: | ---: |
| Best-of-N math | 0.7915 | 0.7916 | 0.8333 | 0.8333 |
| EarlyStop math | 0.8318 | 0.8221 | 0.8333 | 0.8333 |
| EarlyStop science | 0.5849 | 0.5813 | 0.6214 | 0.6286 |
| EarlyStop coding (late anchors) | 0.4661 | 0.4855 | 0.4688 | 0.5781 |

## Explicit Answers

### 1) Are pairwise objectives better than pointwise objectives?

Slightly yes. The mean holdout `pairwise_acc` across the four summary slices is `0.6686` for pointwise logistic versus `0.6701` for the best CV-selected ranking-family baseline per slice.

### 2) Does ranking loss help coding more than math/science?

Yes; coding gains more from ranking loss than math/science in this run. Coding changes by `+0.0194` `pairwise_acc` versus the pointwise control, while the mean noncoding EarlyStop delta is `-0.0067`.

### 3) Does pairwise-on-frozen-z beat pointwise-on-frozen-z?

No. Across EarlyStop slices, frozen-z pointwise logistic reaches `0.6260` holdout `pairwise_acc`, while frozen-z pairwise logistic reaches `0.6234`.

### 4) Should the paper treat coding as a “ranking-head mismatch” rather than a total representation failure?

Partially; the raw representation still contains ranking signal, but frozen-z does not fully confirm it.

The concrete evidence used here is the same-representation comparison on coding late anchors: raw-feature pointwise logistic versus pairwise/listwise objectives, plus the frozen-z pointwise versus frozen-z pairwise comparison.

## Notes

- Best-of-N math uses the existing augmented math feature family `all_aug`.
- EarlyStop math/science/coding use the shared feature family `token_plus_traj_fixed` with representation `raw+rank`.
- The CSV contains one row per task/domain/anchor/model family winner after grouped-CV selection, final refit on the train split, and grouped holdout evaluation.
