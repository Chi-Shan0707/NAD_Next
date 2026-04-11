# Case 2: es_svd_science_rr_r1 science correct @ 100%

- `method_id`: `es_svd_science_rr_r1`
- `domain`: `science`
- `cache_key`: `cache/DS-R1/gpqa`
- `problem_id`: `gpqa-141`
- `anchor_pct`: `100`
- `case_type`: `correct`

## Decision

- `top1`: run `10` / sample `9033` / correct=`True` / score=`5.5765`
- `top2_or_best_correct`: run `48` / sample `9071` / correct=`True` / score=`5.1435`
- `margin_or_gap`: `0.4330`

## Top Family Deltas

- `confidence`: Δ=`0.4919`
- `uncertainty`: Δ=`0.3059`
- `trajectory`: Δ=`-0.2924`
- `availability_meta`: Δ=`-0.0803`
- `self_cert_logprob`: Δ=`0.0079`

## Top Feature Deltas

- `traj_continuity`: Δ=`-0.3855`
- `tok_gini_tail`: Δ=`0.2617`
- `tok_conf_recency`: Δ=`0.2497`
- `tok_conf_prefix`: Δ=`0.2423`
- `traj_novelty`: Δ=`0.0927`
- `traj_late_convergence`: Δ=`0.0787`
- `traj_reflection_count`: Δ=`-0.0443`
- `tok_gini_slope`: Δ=`0.0395`

## Deletion Sanity

- `top_family` `confidence`: score_drop=`12.9801`, margin_drop=`12.9767`, flip=`True`, new_correct=`True`
- `low_family` `availability_meta`: score_drop=`0.0190`, margin_drop=`0.0190`, flip=`False`, new_correct=`True`
- `top_feature` `tok_gini_tail`: score_drop=`6.8183`, margin_drop=`6.8090`, flip=`True`, new_correct=`True`
- `low_feature` `traj_novelty`: score_drop=`-0.0031`, margin_drop=`0.0013`, flip=`False`, new_correct=`True`

## Paper Note

- 该 correct case 展示了 top1-vs-top2 的局部因果相关性：删除 top family / feature 后，选中 run 的 margin 明显下降。
