# Science Front-Half Feature Scan (2026-04-16)

## Goal

- Inspect `science/gpqa` early-slot signal directly on the rebuilt `47`-dimensional EarlyStop feature bank.
- Identify which front-half features are strong enough to justify a denser search family than the older `fixed22` / `all24` / `dynamic17` menu.
- Use the findings to steer the `science`-only dense 10-slot search toward earlier usable signal instead of only late-slot refinement.

## Data

- `feature cache`: `results/cache/science_dense_slot_search/labeled/cache_all_9469fe42440e1ab4.pkl`
- `feature cache_train`: `results/cache/science_dense_slot_search/labeled/cache_train_all_db41b4559ac7bf2f.pkl`
- `domain`: `science`
- `dataset`: `gpqa`
- `positions`: `10, 20, 30, 40, 50, 60, 70, 80, 90, 100`

The scan below uses the rebuilt current-code feature bank, not the older `r2` 30-d cache.

## Main readout

### Slot 10

- `0.6445`: `prefix_best_window_quality`
- `0.6396`: `tok_conf_recency`
- `0.6390`: `tok_conf_prefix`
- `0.6244`: `traj_reflection_count`
- `0.6179`: `tok_selfcert_recency`
- `0.6179`: `tok_neg_entropy_recency`
- `0.6171`: `tail_q10`

### Slot 20

- `0.6579`: `tok_conf_recency`
- `0.6568`: `tok_conf_prefix`
- `0.6485`: `prefix_best_window_quality`
- `0.6462`: `tail_q10`
- `0.6436`: `nc_mean` (negative direction)
- `0.6291`: `tok_selfcert_recency`
- `0.6194`: `traj_reflection_count`
- `0.5949`: `last_block_instability`

### Slot 30

- `0.6747`: `tok_conf_recency`
- `0.6731`: `nc_mean` (negative direction)
- `0.6730`: `tok_conf_prefix`
- `0.6657`: `tail_q10`
- `0.6466`: `prefix_best_window_quality`
- `0.6384`: `traj_continuity`
- `0.6269`: `last_block_instability`

### Slot 40

- `0.6915`: `nc_mean` (negative direction)
- `0.6842`: `tok_conf_recency`
- `0.6825`: `tok_conf_prefix`
- `0.6642`: `tail_q10`
- `0.6606`: `traj_continuity`
- `0.6598`: `last_block_instability`
- `0.6411`: `prefix_best_window_quality`

### Slot 50

- `0.7039`: `nc_mean` (negative direction)
- `0.6884`: `tok_conf_recency`
- `0.6871`: `tok_conf_prefix`
- `0.6849`: `last_block_instability`
- `0.6740`: `traj_continuity`
- `0.6599`: `tail_q10`
- `0.6386`: `prefix_best_window_quality`
- `0.6347`: `conf_abs_d1_tail_mean`

## Takeaways

- Front-half science signal is **not** dominated by a single old `fixed22` family; it is spread across confidence, window-quality, trajectory, and instability features.
- `prefix_best_window_quality` is already a top feature at `10%`, which means the new post-`r2` window features matter immediately rather than only late.
- `tail_q10` stays strong across the whole `10–50%` band, so low-tail confidence is a stable early warning signal for science.
- `nc_mean` becomes very strong by `20–50%`, mostly in the negative direction, which supports including prefix-length / normalization metadata in front-half science families.
- `traj_continuity` and `last_block_instability` rise in importance from `30%` onward, suggesting that mid-prefix trajectory shape and local instability are useful for the science early-stop regime.

## Search changes motivated by this scan

- Add `science_front7`, `science_front9`, `science_front11`, and `science_front13` families to the dense science slot search.
- Keep `wide46` in the search so the run can still win with a broad feature bank if regularization supports it.
- Add `elasticnet_lr` and widen the linear `C` grid so front-half subsets are not forced into only `l1` or `l2`.

## Reading

- If the final dense search selects one of the new `science_front*` families for `10–50%`, that is direct evidence that science front-half performance benefits from the rebuilt `47`-feature bank rather than the older frozen route.
- If the final search still prefers `wide46`, the scan still matters: it tells us which substructure inside `wide46` likely carries the gain.
