# NAD Next — Cookbook

Step-by-step tutorials for reproducing NAD experiments.

## Chapters

| Chapter | Topic | Description |
|---------|-------|-------------|
| [00 — Setup](./00_setup/README.md) | Environment | Install packages, link caches, sanity check |
| [01 — Cache Browser](./01_cache_browser/README.md) | Web UI | Browse available caches, inspect metadata, JSON API |
| [02 — Visualization Server](./02_visualization/README.md) | Web UI | Interactive exploration of a single cache with token decoding |
| [03 — 批量分析](./03_batch_analyze/README.md) | Analysis | 并行批量分析所有模型/数据集缓存，输出选择器准确率 |
| [04 — Position Window Ablation](./04_position_ablation/README.md) | Analysis | Ablation across position windows (0-1 … 0-512) to study early-token selector behaviour |
| [05 — DeepConf Analysis](./05_deepconf_analysis/README.md) | Analysis | Token-confidence selector analysis (tok_conf / tok_neg_entropy) |
---

> **Tip:** Start with Chapter 00 before running anything else.
