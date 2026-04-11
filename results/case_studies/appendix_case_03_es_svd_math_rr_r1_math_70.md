# Case 3: es_svd_math_rr_r1 math wrong @ 70%

- `method_id`: `es_svd_math_rr_r1`
- `domain`: `math`
- `cache_key`: `cache_train/DS-R1/brumo25`
- `problem_id`: `brumo25-26`
- `anchor_pct`: `70`
- `case_type`: `wrong`
- `archetype`: `Mixed-signal conflict`

## Decision

- `top1`: run `47` / sample `1710` / correct=`False` / score=`5.7757`
- `top2_or_best_correct`: run `57` / sample `1720` / correct=`True` / score=`1.0570`
- `margin_or_gap`: `4.7187`

## Top Family Deltas

- `trajectory`: Δ=`4.8700`
- `availability_meta`: Δ=`0.0246`

## Top Feature Deltas

- `traj_novelty`: Δ=`2.7741`
- `traj_continuity`: Δ=`1.7522`
- `traj_max_reflection`: Δ=`1.1787`
- `tok_gini_tail`: Δ=`0.0344`
- `traj_late_convergence`: Δ=`0.0196`
- `tok_logprob_recency`: Δ=`0.0119`
- `tok_conf_prefix`: Δ=`0.0051`
- `has_tok_logprob`: Δ=`0.0041`

## Deletion Sanity

- `top_family` `confidence`: score_drop=`10.4245`, margin_drop=`10.4287`, flip=`True`, new_correct=`False`
- `low_family` `availability_meta`: score_drop=`-0.1130`, margin_drop=`-0.1130`, flip=`False`, new_correct=`False`
- `top_feature` `traj_max_reflection`: score_drop=`6.6219`, margin_drop=`6.6219`, flip=`True`, new_correct=`False`
- `low_feature` `tok_logprob_prefix`: score_drop=`-0.0136`, margin_drop=`-0.0136`, flip=`False`, new_correct=`False`

## Paper Note

- 该 wrong-top1 case 属于 `Mixed-signal conflict`：错误 top1 相对最佳正确 run 的领先主要来自 `trajectory`。
