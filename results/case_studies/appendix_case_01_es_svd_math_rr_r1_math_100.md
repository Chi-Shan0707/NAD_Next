# Case 1: es_svd_math_rr_r1 math correct @ 100%

- `method_id`: `es_svd_math_rr_r1`
- `domain`: `math`
- `cache_key`: `cache/DS-R1/aime25`
- `problem_id`: `5`
- `anchor_pct`: `100`
- `case_type`: `correct`

## Decision

- `top1`: run `43` / sample `362` / correct=`True` / score=`14.7530`
- `top2_or_best_correct`: run `34` / sample `353` / correct=`True` / score=`12.6648`
- `margin_or_gap`: `2.0882`

## Top Family Deltas

- `trajectory`: Δ=`1.4950`
- `confidence`: Δ=`1.3897`
- `uncertainty`: Δ=`-0.3962`
- `self_cert_logprob`: Δ=`-0.2241`
- `availability_meta`: Δ=`-0.1762`

## Top Feature Deltas

- `tok_conf_recency`: Δ=`0.7138`
- `tok_conf_prefix`: Δ=`0.6759`
- `traj_continuity`: Δ=`0.3730`
- `traj_novelty`: Δ=`0.3720`
- `traj_reflection_count`: Δ=`0.3108`
- `traj_late_convergence`: Δ=`0.2814`
- `traj_max_reflection`: Δ=`0.1578`
- `tok_gini_prefix`: Δ=`-0.1545`

## Deletion Sanity

- `top_family` `confidence`: score_drop=`22.2012`, margin_drop=`22.2145`, flip=`True`, new_correct=`True`
- `low_family` `availability_meta`: score_drop=`-0.8221`, margin_drop=`-0.8417`, flip=`False`, new_correct=`True`
- `top_feature` `traj_continuity`: score_drop=`12.8070`, margin_drop=`12.7838`, flip=`True`, new_correct=`True`
- `low_feature` `tok_logprob_recency`: score_drop=`-0.0464`, margin_drop=`-0.0464`, flip=`False`, new_correct=`True`

## Paper Note

- 该 correct case 展示了 top1-vs-top2 的局部因果相关性：删除 top family / feature 后，选中 run 的 margin 明显下降。
