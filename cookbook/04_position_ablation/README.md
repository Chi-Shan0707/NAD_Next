# Chapter 04 — Position Window Ablation

## Overview

Runs ablation experiments across different position windows (e.g. 0-1, 0-2, 0-8, 0-32, 0-128, 0-512)
for all caches under `MUI_HUB/cache`, measuring how selector accuracy changes
as more tokens are included.

---

## Prerequisites

- Chapter 00 completed — all 12 checks green (`bash cookbook/00_setup/verify.sh`)
- The `MUI_HUB` symlink created (Chapter 00, Step 3)

---

## Quick start

```bash
# Dry run: show task list without executing
bash cookbook/04_position_ablation/position_ablation.sh --dry-run

# Quick mode: 3 windows (0-1, 0-8, 0-128), all caches
bash cookbook/04_position_ablation/position_ablation.sh --quick

# Full ablation: 6 windows (0-1, 0-2, 0-8, 0-32, 0-128, 0-512)
bash cookbook/04_position_ablation/position_ablation.sh

# Latest cache only per dataset, quick mode
bash cookbook/04_position_ablation/position_ablation.sh --quick --latest-only

# Specific datasets
bash cookbook/04_position_ablation/position_ablation.sh --datasets aime24,aime25
```

---

## All options

```
position_ablation.sh [options...]

  --windows LIST    Comma-separated position windows (default: 0-1,0-2,0-8,0-32,0-128,0-512)
  --threads N       Distance-computation threads (default: 64)
  --backend BACKEND Jaccard backend: roaring/numpy/auto (default: roaring)
  --output-dir DIR  Output directory (default: ./result/position_ablation_TIMESTAMP)
  --dry-run         Show tasks without executing
  --quick           Quick mode: windows 0-1, 0-8, 0-128 only
  -j N              Parallel jobs (default: 4)
  --model MODEL     Model name (default: DeepSeek-R1-0528-Qwen3-8B)
  --datasets LIST   Comma-separated dataset names (default: all)
  --latest-only     Use only the latest cache per dataset
  --help            Show help
```

---

## Output structure

```
./result/position_ablation_TIMESTAMP/
  {dataset}/{cache}/
    window_0-1_result.json
    accuracy_window_0-1.json
    window_0-2_result.json
    accuracy_window_0-2.json
    ...
  summary.json
```
