# Rule and Family Baselines

Interpretability-first baselines that isolate uncertainty-only, logprob-only, and trajectory-only signals under the grouped Early-Stop holdout protocol.

## Protocol

- Repository: `NAD_Next` only.
- Holdout: `85/15` grouped split by `dataset + problem_id`, seed `42`.
- Anchors: `10, 20, 30, 40, 50, 60, 70, 80, 90, 100`.
- Main cache root: `MUI_HUB/cache`.
- Extra cache root: `MUI_HUB/cache_train`.
- Reflection threshold: `0.30`.

## Data Coverage

| Domain | Train Problems | Train Samples | Holdout Problems | Holdout Samples |
|---|---|---|---|---|
| math | 182 | 11648 | 28 | 1792 |
| science | 336 | 21504 | 60 | 3840 |
| ms | 518 | 33152 | 88 | 5632 |
| coding | 142 | 9088 | 25 | 1600 |

## Domain Summary

| Domain | Best Single Feature | Best Simple Rule | Best Family LR | Reference SVDomain |
|---|---|---|---|---|
| math | traj_reflection_count (89.28%) | traj_reflection_count (89.28%) | trajectory_lr (97.03%) | es_svd_math_rr_r2 (96.74%) |
| science | tok_conf_recency (81.03%) | tok_conf_recency (81.03%) | uncertainty_lr (81.00%) | es_svd_science_rr_r2 (82.86%) |
| ms | traj_reflection_count (82.16%) | traj_reflection_count (82.16%) | trajectory_lr (90.34%) | es_svd_ms_rr_r2 (93.66%) |
| coding | traj_novelty (55.93%) | traj_novelty (55.93%) | trajectory_lr (54.24%) | es_svd_coding_rr_r1 (43.42%) |

## Answers

- How strong are the best simple rules? `math`: `traj_reflection_count` at `89.28%`; `science`: `tok_conf_recency` at `81.03%`; `ms`: `traj_reflection_count` at `82.16%`; `coding`: `traj_novelty` at `55.93%`
- Is trajectory-only enough in math? Yes, with a caveat: trajectory-only reaches SVDomain-level math performance once you allow the family LR. `trajectory_lr` gets `97.03%`, `+0.28pp` versus `es_svd_math_rr_r2`; the best fixed trajectory rule is `traj_reflection_count` at `89.28%`.
- Is uncertainty-only stronger in science? Yes: uncertainty-only wins science with `tok_conf_recency` at `81.03%`.
- Is coding weak because every family is weak, or because only the combined route fails? Both signals show up: every family is fairly weak in absolute terms, but the combined route fails even harder. Coding tops out at `traj_novelty` / `55.93%`; the best uncertainty/logprob/trajectory baselines are `51.31%` / `50.24%` / `55.93%`; `es_svd_coding_rr_r1` is lower at `43.42%`.
- Which simple baseline is the most serious competitor to SVDomain? The biggest anomaly is `traj_novelty` on `coding` at `55.93%` (`+12.51pp` versus `es_svd_coding_rr_r1`), but on the main noncoding benchmark the strongest competitor is `trajectory_lr` on `math` at `97.03%` (`+0.28pp` versus `es_svd_math_rr_r2`).

## Notes

- Rule baselines are fixed, transparent formulas only; there is no learned weighting in `single_rule` or `combo_rule` rows.
- Family LRs use only same-family features with `raw+rank`, standard scaling, and no SVD.
- All files are generated under `NAD_Next` outputs: `scripts/baselines/`, `results/tables/`, and `docs/`.
