# Scripts Layout

This folder keeps executable entrypoints at the top level so existing commands continue to work.

## Groups
- `train_*`: model training and selector fitting.
- `run_*`: experiment runners that produce timestamped outputs.
- `export_*`: submission/export helpers.
- `scan_*` / `analyze_*`: one-off analysis or signal scans.
- `plot_*`: visualization and figure generation.
- `rank_*` / `reorder_*` / `rescale_*`: post-processing utilities.
- `test_*.sh` / `compare_*.sh`: compatibility or comparison helpers.

## Recommended reading order
- Start with `run_*` for experiment orchestration.
- Use `train_*` when regenerating models.
- Use `export_*` after a run is frozen.
- Use `rank_*` and `plot_*` for summaries.

## Note
- New local-only experiments should prefer subfolders conceptually, but stable user-facing entry scripts remain in `scripts/` for backwards compatibility.
