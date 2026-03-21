# Chapter 05 — DeepConf Selector Analysis

## Overview

Runs specialised analysis for the DeepConf-style selector, which uses
token-level confidence metrics (tok_conf, tok_neg_entropy)
to pick the most reliable sample from each problem group.

Supports multiple configurations and optional comparison against standard selectors.

---

## Prerequisites

- Chapter 00 completed — all 12 checks green (`bash cookbook/00_setup/verify.sh`)
- The `MUI_HUB` symlink created (Chapter 00, Step 3)
- A cache with `token_data/` present (type-1 caches: `cache_neuron_output_1_*`)

---

## Quick start

```bash
# Quick test (default config only: tok_conf + min_group + window=20)
bash cookbook/05_deepconf_analysis/deepconf_analysis.sh quick

# Standard analysis (6 config variants compared)
bash cookbook/05_deepconf_analysis/deepconf_analysis.sh standard

# Full analysis (standard + position windows 0-1, 0-2, 0-8)
bash cookbook/05_deepconf_analysis/deepconf_analysis.sh full

# Compare DeepConf against all standard selectors
bash cookbook/05_deepconf_analysis/deepconf_analysis.sh standard --compare-all

# Custom config
bash cookbook/05_deepconf_analysis/deepconf_analysis.sh custom \
  --metric tok_neg_entropy --reduction mean --group-size 10

# Point at a specific cache
bash cookbook/05_deepconf_analysis/deepconf_analysis.sh \
  MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/aime24/cache_neuron_output_1_act_no_rms_20250902_025610 \
  standard
```

---

## Modes

| Mode | Description |
|------|-------------|
| `quick` | Default config only — fastest |
| `standard` | 4 config variants (metric × reduction combinations) |
| `full` | standard + position windows (0-1, 0-2, 0-8) |
| `custom` | User-specified metric, reduction, group-size |

---

## All options

```
deepconf_analysis.sh [cache_name] [mode] [options...]

  --metric METRIC       tok_conf (default) | tok_neg_entropy
  --reduction METHOD    min_group (default) | mean
  --group-size N        Moving-average window size (default: 20)
  --compare-all         Compare with all standard selectors
  --compare-baseline    Compare with baseline selectors only
  --threads N           Distance-computation threads (default: 64)
  --backend BACKEND     Jaccard backend: roaring/numpy/auto (default: roaring)
  --enable-profiling    Enable performance profiling
  --log-level LEVEL     Logging level (default: INFO)
```

---

## Token metrics

| Metric | Formula | Better direction |
|--------|---------|-----------------|
| `tok_conf` | −mean(log p_topk) | lower → more confident |
| `tok_neg_entropy` | Σ p log p | closer to 0 → more certain |

---

## Output structure

```
./results_deepconf_{cache_name}/
  deepconf_default.json               # Default config result
  deepconf_default_accuracy.json      # Accuracy report
  deepconf_conf_mingroup_20.json      # Config variants (standard/full)
  ...
  comparison_all.json                 # Comparison with other selectors
```
