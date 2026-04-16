# Science Dense Slot Search (2026-04-16)

## Goal

- Keep the user-selected aggressive coding routes intact.
- Re-search `science/gpqa` on the user-targeted slots instead of reusing the frozen science patch.
- Search over `source anchor × feature family × model family`, then patch only the targeted blind science slots back into the aggressive submission.

## Why this run differs

- Labeled science features are rebuilt from current code, not the old `r2` prebuilt cache.
- The rebuilt bank uses the current `47`-dimensional EarlyStop features, so the search can test post-`r2` window / instability / tail-delta signals.
- The candidate grid includes science front-half families centered on `tok_conf_*`, `prefix_best_window_quality`, `tail_q10`, `traj_continuity`, `nc_mean`, and `last_block_instability`.
- Direct linear search stays focused on `raw+rank` linear heads over the strongest front-half families, keeping the search targeted without falling back to the older frozen route.
- Selection is dense: each target slot can choose a different training source anchor.
- Untouched science slots inherit the base aggressive submission values.

## Protocol

- `holdout split`: `85/15`, grouped by `dataset + problem_id`, `split_seed=42`.
- `reflection threshold`: `0.30`.
- `source anchors`: `10, 20, 30, 40, 50`.
- `target anchors`: `10, 20, 30, 40, 50`.
- `candidate families`: `fixed22, science_front11, science_front13, wide46`.
- `transfer margin`: `0.0010` AUROC.
- `tree margin`: `0.0025` AUROC.

## Holdout Readout

| Slice | AUC of AUROC | AUC of SelAcc | AUROC@50% | Stop Acc@50% |
|---|---:|---:|---:|---:|
| `science r2` | 78.47% | 68.33% | 82.21% | 66.67% |
| `dense task-specific` | 82.07% | 70.83% | 85.99% | 66.67% |
| `dense final` | 82.67% | 69.17% | 85.99% | 66.67% |

## Selected Routes

| Target | Source | Route | Family | Rep | Holdout AUROC | SelAcc | Note |
|---|---:|---|---|---|---:|---:|---|
| 10% | 10% | lr_l1 | fixed22 | raw+rank | 0.8017 | 0.7333 | best_holdout |
| 20% | 30% | lr_l2 | fixed22 | raw+rank | 0.7995 | 0.7333 | best_holdout |
| 30% | 50% | lr_l2 | science_front11 | raw+rank | 0.8272 | 0.6667 | best_holdout |
| 40% | 50% | lr_l1 | wide46 | raw+rank | 0.8494 | 0.6667 | best_holdout |
| 50% | 50% | lr_l2 | wide46 | raw+rank | 0.8599 | 0.6667 | best_holdout |

## Front-Half Readout

- `10%`: source `10%`, `lr_l1` on `fixed22` (`raw+rank`), holdout AUROC `0.8017`.
- `20%`: source `30%`, `lr_l2` on `fixed22` (`raw+rank`), holdout AUROC `0.7995`.
- `30%`: source `50%`, `lr_l2` on `science_front11` (`raw+rank`), holdout AUROC `0.8272`.
- `40%`: source `50%`, `lr_l1` on `wide46` (`raw+rank`), holdout AUROC `0.8494`.
- `50%`: source `50%`, `lr_l2` on `wide46` (`raw+rank`), holdout AUROC `0.8599`.

## Export

- `patched submission`: `submission/EarlyStop/es_svd_ms_rr_r1__coding_rotation_adapter_aggressive_tokenonly_late70100__science_front50_holdout_search_20260416.json`
- `candidate csv`: `results/tables/science_front50_holdout_search_candidates_20260416.csv`
- `selected csv`: `results/tables/science_front50_holdout_search_selected_20260416.csv`
- `eval json`: `results/scans/earlystop/science_front50_holdout_search_20260416_eval.json`

## Reading

- If `dense final` beats `dense task-specific`, the gain comes from cross-anchor transfer and/or wider feature families, not from simply densifying the old route.
- If early slots choose `science_front11`, `science_front13`, or `wide46`, that is direct evidence that the newer front-half science features matter.
- The patched JSON is the file to score externally; all other artifacts are there to justify how it was chosen.
