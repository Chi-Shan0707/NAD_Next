# Case 4: es_svd_math_rr_r1 math wrong @ 10%

- `method_id`: `es_svd_math_rr_r1`
- `domain`: `math`
- `cache_key`: `cache/DS-R1/brumo25`
- `problem_id`: `brumo25-12`
- `anchor_pct`: `10`
- `case_type`: `wrong`
- `archetype`: `Trajectory over-bias`

## Decision

- `top1`: run `20` / sample `787` / correct=`False` / score=`0.0049`
- `top2_or_best_correct`: run `29` / sample `796` / correct=`True` / score=`-3.3830`
- `margin_or_gap`: `3.3879`

## Top Family Deltas

- `trajectory`: Δ=`3.0650`
- `self_cert_logprob`: Δ=`1.7017`

## Top Feature Deltas

- `traj_reflection_count`: Δ=`4.6866`
- `tok_logprob_prefix`: Δ=`1.2749`
- `tok_logprob_recency`: Δ=`1.2403`
- `tok_gini_slope`: Δ=`0.3390`
- `traj_max_reflection`: Δ=`0.1648`
- `traj_continuity`: Δ=`0.1437`
- `traj_late_convergence`: Δ=`0.0634`
- `tok_gini_tail`: Δ=`0.0434`

## Deletion Sanity

- `top_family` `confidence`: score_drop=`6.9689`, margin_drop=`6.9689`, flip=`True`, new_correct=`False`
- `low_family` `availability_meta`: score_drop=`0.0671`, margin_drop=`0.0671`, flip=`False`, new_correct=`False`
- `top_feature` `traj_continuity`: score_drop=`5.4566`, margin_drop=`5.4566`, flip=`True`, new_correct=`False`
- `low_feature` `has_tok_selfcert`: score_drop=`0.0112`, margin_drop=`0.0112`, flip=`False`, new_correct=`False`

## Paper Note

- 该 wrong-top1 case 属于 `Trajectory over-bias`：错误 top1 相对最佳正确 run 的领先主要来自 `trajectory`。
