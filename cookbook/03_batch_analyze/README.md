# Chapter 03 — Batch Analysis (Parallel)

## Overview

Runs NAD analysis in parallel across all model/dataset caches under `MUI_HUB/cache`,
producing per-selector accuracy results (medoid, knn-medoid, dbscan-medoid, etc.)
for every task.

The output directory can be passed directly to `step2.3_selector_ranking.sh`
for cross-model selector ranking.

---

## Prerequisites

- Chapter 00 completed — all 12 checks green (`bash cookbook/00_setup/verify.sh`)
- The `MUI_HUB` symlink created (Chapter 00, Step 3)
- `GNU parallel` installed (optional — falls back to `xargs` automatically)

---

## Quick start

```bash
# Dry run: show task list without executing
bash cookbook/03_batch_analyze/batch_analyze.sh --dry-run

# Quick test: first 3 caches, 4 parallel jobs
bash cookbook/03_batch_analyze/batch_analyze.sh --limit 3 -j 4 -y

# Full batch analysis (default: full mode, 4 parallel jobs)
bash cookbook/03_batch_analyze/batch_analyze.sh -y

# Full + position-window analysis, 8 parallel jobs
bash cookbook/03_batch_analyze/batch_analyze.sh --mode all -j 8 -y
```

---

## All options

```
batch_analyze.sh [options...]

  --mode MODE       Analysis mode (default: full)
                      full          Full-sequence analysis only
                      positions     Fixed windows (0-1, 0-2, 0-8)
                      all           full + positions
                      all-positions Fine-grained analysis for every position
  --pos-max N       Cap maximum position in all-positions mode
  --threads N       Distance-computation threads per job (default: 8)
  --backend BACKEND Jaccard backend: roaring/numpy/auto (default: roaring)
  -j N              Number of parallel jobs (default: 4)
  --limit N         Limit number of caches to process (0 = all)
  --output-dir DIR  Output directory (default: ./result/all_model_TIMESTAMP)
  --dry-run         Show task list only, do not execute
  -y, --auto-yes    Skip confirmation prompt
  -h, --help        Show help
```

---

## Output structure

```
./result/all_model_TIMESTAMP/
  {model}/{dataset}/{cache}/
    full_sequence_result.json       # Selection results
    accuracy_full_sequence.json     # Accuracy report
    window_0-1_result.json          # (positions/all mode only)
    accuracy_window_0-1.json
    window_0-2_result.json
    accuracy_window_0-2.json
    window_0-8_result.json
    accuracy_window_0-8.json
  summary.json                      # Aggregated summary
  .logs/                            # Per-task log files
```

---

## Next steps

Pass the output directory to the selector ranking script:

```bash
./step2.3_selector_ranking.sh --results-dir ./result/all_model_TIMESTAMP
```

Or use `quick` mode for a fast ranking overview:

```bash
./step2.3_selector_ranking.sh quick --results-dir ./result/all_model_TIMESTAMP
```
