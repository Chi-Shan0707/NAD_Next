# NAD Next — Neuron Activation Distribution

A framework for analyzing neural network activations via binary CSR caches, selector algorithms, and a cookbook of reproducible experiments. NAD Next processes raw NPZ activation shards into efficient memory-mapped caches (CSR format with Roaring Bitmap indexing), applies 10 selection algorithms to pick the most representative sample per problem, and evaluates selector accuracy across models and datasets.

---

## Quick Start

```bash
bash cookbook/00_setup/install.sh                          # install dependencies
bash cookbook/00_setup/verify.sh                            # all checks should be green
bash cookbook/01_cache_browser/cache_browser.sh --background  # browse caches at :5003
bash cookbook/02_visualization/visualization.sh \
  MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/aime24/cache_neuron_output_1_act_no_rms_20250902_025610 \
  --background                                             # explore one cache at :5002
bash cookbook/03_batch_analyze/batch_analyze.sh -y          # analyze all caches
bash cookbook/05_deepconf_analysis/deepconf_analysis.sh standard --compare-all  # deepconf
```

See each cookbook chapter below for full options.

---

## Cookbook

All scripts live in [`cookbook/`](cookbook/). Run from the **repo root**.

### 00 — Setup

Install Python dependencies and verify the environment.

```bash
bash cookbook/00_setup/install.sh    # numpy, pyroaring, flask, plotly, ...
bash cookbook/00_setup/verify.sh     # 3 groups: Python version, packages, MUI_HUB symlink
```

The verification script checks Python 3.9+, all required/optional packages, and that the `MUI_HUB` symlink points to accessible cache storage. Re-run `install.sh` if any check fails.

> [Chapter README](cookbook/00_setup/README.md) for full details.

### 01 — Cache Browser

Web UI to browse all available caches at **port 5003**. Shows dataset name, sample count, model accuracy, temperature, and build date for every cache under `MUI_HUB/cache/`.

```bash
bash cookbook/01_cache_browser/cache_browser.sh --background   # start
bash cookbook/01_cache_browser/cache_browser.sh --kill          # stop
```

Also exposes a JSON API at `/api/caches` for scripting.

> [Chapter README](cookbook/01_cache_browser/README.md) for all options and API docs.

### 02 — Visualization

Interactive single-cache explorer at **port 5002** (Flask + Plotly). Browse neuron activation patterns per problem, inspect token-level entropy/confidence, view selector performance, and decode token IDs.

```bash
bash cookbook/02_visualization/visualization.sh \
  MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/aime24/cache_neuron_output_1_act_no_rms_20250902_025610 \
  --background
bash cookbook/02_visualization/visualization.sh --kill
```

Always pass cache paths relative to the repo root via the `MUI_HUB` symlink.

> [Chapter README](cookbook/02_visualization/README.md) for tokenizer setup and all options.

### 03 — Batch Analysis

Parallel analysis across all caches under `MUI_HUB/cache/`. Produces per-selector accuracy results for every model/dataset combination.

```bash
bash cookbook/03_batch_analyze/batch_analyze.sh --dry-run       # preview task list
bash cookbook/03_batch_analyze/batch_analyze.sh --limit 3 -y    # quick test (3 caches)
bash cookbook/03_batch_analyze/batch_analyze.sh -y              # full run
```

Results land in `./result/all_model_TIMESTAMP/`. Pass that directory to `scripts/rank_selectors.py` for cross-task ranking.

> [Chapter README](cookbook/03_batch_analyze/README.md) for modes, parallelism, and output structure.

### 04 — Position Ablation

Ablation across 6 position windows (0-1, 0-2, 0-8, 0-32, 0-128, 0-512) to measure how selector accuracy changes as more tokens are included.

```bash
bash cookbook/04_position_ablation/position_ablation.sh --quick   # 3 windows: 0-1, 0-8, 0-128
bash cookbook/04_position_ablation/position_ablation.sh           # all 6 windows
bash cookbook/04_position_ablation/position_ablation.sh --datasets aime24,aime25  # subset
```

> [Chapter README](cookbook/04_position_ablation/README.md) for all options and output structure.

### 05 — DeepConf Analysis

Token-confidence selector using `tok_conf` and `tok_neg_entropy` metrics from `token_data/`. Four modes: quick (1 config), standard (4 variants), full (standard + position windows), custom.

```bash
bash cookbook/05_deepconf_analysis/deepconf_analysis.sh quick
bash cookbook/05_deepconf_analysis/deepconf_analysis.sh standard --compare-all
bash cookbook/05_deepconf_analysis/deepconf_analysis.sh full
bash cookbook/05_deepconf_analysis/deepconf_analysis.sh custom \
  --metric tok_neg_entropy --reduction mean --group-size 10
```

> [Chapter README](cookbook/05_deepconf_analysis/README.md) for metric semantics and all options.

---

## Architecture

### Project Layout

```
NAD_Next/
  nad/                         # Core Python package
    api.py                     # High-level API: open_cache(), load_correctness_map()
    cli/                       # CLI: python3 -m nad.cli {analyze, accuracy, cache-build, cache-build-fast}
    core/
      adapters/                # NPZ shard -> CSR conversion (shard_reader, batch_processor)
      distance/                # Jaccard distance engine (roaring / numpy, auto-switch at 4096)
      schema/                  # Cache manifest (v4.0+)
      selectors/               # 10 selector algorithms + plugin loader
      storage/                 # Binary cache I/O (mmap)
      views/                   # CacheReader (lazy mmap access)
    io/                        # NadNextLoader (256 MB LRU), index, viz catalog
    ops/                       # Accuracy scoring, selector ranking (RNS / Copeland), uniques
    pipeline/                  # Analysis orchestration, 2-stage Map-Reduce cache builder
  cookbook/                     # 6 experiment chapters (00-05)
  scripts/                     # rank_selectors.py, plot_*.py
  plugins/                     # kink_selector.py (reference custom selector)
  tools/                       # Ground truth generation, cache browser server
  minimal_visualization_next/  # Flask + Plotly visualization server
```

### Data Pipeline

```
NPZ Shards --> cache-build-fast --> Binary Cache (CSR + mmap)
                                         |
                                    nad.cli analyze --> Selection JSON
                                         |
                                    nad.cli accuracy --> Accuracy Report
                                         |
                                    rank_selectors.py --> Cross-task Ranking
```

**Stage 1 — Build**: Raw NPZ activation shards are converted into a binary cache via a two-stage Map-Reduce process (`cache-build-fast`). The output is a set of memory-mapped CSR files with sorted indices and optional token-level statistics.

**Stage 2 — Analyze**: `nad.cli analyze` reads a cache, groups samples by problem, computes pairwise Jaccard distances, and runs each selector algorithm to pick one representative per problem.

**Stage 3 — Accuracy**: `nad.cli accuracy` evaluates each selector's pick against ground truth from `evaluation_report_compact.json`.

**Stage 4 — Rank**: `scripts/rank_selectors.py` aggregates accuracy across multiple tasks and produces cross-task rankings using RNS, regret, and Copeland scoring with optional Bootstrap CI.

### Cache Format

Caches are stored under `MUI_HUB/cache/{model}/{dataset}/cache_neuron_output_1_*`.

```
cache_neuron_output_1_*/
  meta.json                      # Sample -> problem mapping (auto-parsed by analysis)
  manifest.json                  # Schema version + checksums
  evaluation_report_compact.json # Ground truth, 99% compressed (0.8 MB vs 82 MB)
  base/                          # CSR neuron activations: row_ptr, keys, w_max, w_sum
  index/                         # Sorted indices for fast lookups
  token_data/                    # Per-token LM stats: tok_conf, tok_neg_entropy
  rows/                          # Row-CSR Bank for position-window queries
```

**Two independent data dimensions share the same row structure:**
- **Neuron data** (`base/`, `rows/`): which neurons fired at each token position
- **Token data** (`token_data/`): LM output distribution statistics per token

Cache type `cache_neuron_output_1_*` = 1 token/row. Type `cache_neuron_output_2_*` = ~32 tokens/row.

---

## Selectors

Ten built-in selector algorithms pick the most representative sample from each problem group.

| Selector | Description |
|----------|-------------|
| `min-activation` | Fewest total neuron activations |
| `max-activation` | Most total neuron activations |
| `medoid` | Geometric centre of the group (Jaccard distance) |
| `knn-medoid` | Medoid restricted to K nearest neighbours |
| `dbscan-medoid` | Medoid of the densest DBSCAN cluster |
| `consensus-min` | Minimum of consensus voting scores |
| `consensus-max` | Maximum of consensus voting scores |
| `deepconf` | Token-confidence based (requires `token_data/`) |
| `con64@` | Consensus of top-64 runs (full sequence only) |
| `avg64@` | Average score of top-64 runs (full sequence only) |

`con64@` and `avg64@` require full-sequence context and are not available in position-window mode.

<details>
<summary><strong>Custom Selectors</strong></summary>

Inherit from `Selector`, implement `bind(context)` and `select(D, run_stats) -> int`:

```python
from nad.core.selectors.base import Selector, SelectorContext
from nad.ops.uniques import extract_tokenwise_counts

class MySelector(Selector):
    def bind(self, context: SelectorContext):
        self._context = context

    def select(self, D, run_stats) -> int:
        # D: pairwise distance matrix
        # self._context.cache: CacheReader
        return selected_index
```

Load with `file:path/to/selector.py:ClassName`. Always filter `inf` values before `argmax`/`argmin` (occurs when a denominator is zero). See [`plugins/kink_selector.py`](plugins/kink_selector.py) for a complete example.

</details>

---

## CLI Reference

### analyze — Run selector algorithms on a cache

```bash
python3 -m nad.cli analyze \
  --cache-root MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/aime24/cache_neuron_output_1_act_no_rms_20250902_025610 \
  --distance ja --selectors all \
  --group-topk-policy legacy-min \
  --distance-threads 16 \
  --out result.json
```

### analyze — Position window mode

```bash
python3 -m nad.cli analyze \
  --cache-root <cache_path> \
  --pos-window 0-8 --pos-size 32 \
  --out window_0-8_result.json
```

### accuracy — Evaluate selector picks against ground truth

```bash
python3 -m nad.cli accuracy \
  --selection result.json \
  --cache-root <cache_path> \
  --out accuracy.json
```

### rank — Cross-task selector ranking

```bash
python3 scripts/rank_selectors.py \
  --results-dir ./result/all_model_TIMESTAMP \
  --no-bootstrap --csv --json
```

### Custom selector plugin

```bash
python3 -m nad.cli analyze \
  --cache-root <cache_path> \
  --selectors 'all,file:./plugins/kink_selector.py:KinkSelector'
```

### Jaccard Backend

| Backend | When faster |
|---------|-------------|
| `roaring` | Set size >= 4096 elements (default) |
| `numpy` | Small sets |
| `auto` | Switches at the 4096 threshold |

### DeepConf Metric Semantics

| Metric | Formula | Quality direction |
|--------|---------|-------------------|
| `tok_conf` | -mean(log p_topk) | lower -> more confident -> `quality = -tok_conf` |
| `tok_neg_entropy` | sum(p log p) <= 0 | closer to 0 -> more certain -> `quality = tok_neg_entropy` (identity) |

Sliding window uses strict trailing windows with no padding. Sequences shorter than `window_size` return their global mean.

---

## Configuration

### Dependencies

```bash
pip install .             # recommended: installs from pyproject.toml
# or install core deps directly:
pip install numpy>=1.20.0 pyroaring>=0.4.5
```

### Environment Variables

```bash
export NAD_JA_BACKEND=roaring   # Jaccard backend: roaring / numpy / auto
export OMP_NUM_THREADS=1        # prevent NumPy thread oversubscription
export MKL_NUM_THREADS=1
```

### Thread Limits

This machine has **16 cores**. All scripts must satisfy `THREADS x PARALLEL_JOBS <= 16`.

| PARALLEL_JOBS | THREADS |
|---------------|---------|
| 1 | 16 |
| 4 | 4 |
| 8 | 2 |

Current cookbook defaults:
- `03_batch_analyze`: THREADS=4, PARALLEL_JOBS=4
- `04_position_ablation`: THREADS=4, PARALLEL_JOBS=4
- `05_deepconf_analysis`: THREADS=16, PARALLEL_JOBS=1

### nad_config.json

`nad_config.json` at repo root specifies `model_search_dirs` for tokenizer lookup (used by the visualization server).

```json
{
  "model_search_dirs": [
    "/home/jovyan/public-ro/model"
  ]
}
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `meta.json` not found (fallback n=100) | Ensure `meta.json` is present in each cache |
| Slow distance computation | Use `roaring` backend for large sets |
| Missing Row-CSR Bank | Rebuild cache with `--row-bank` |
| DeepConf metric not found (ValueError) | Check `token_data/` exists; use `--metric` matching available keys |
| Port 5002 already in use | `bash cookbook/02_visualization/visualization.sh --kill` |
| Port 5003 already in use | `bash cookbook/01_cache_browser/cache_browser.sh --kill` |

---

## License

Internal use only.
