# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NAD Next (Neuron Activation Distribution) is a framework for analyzing neural network activations, selecting representative samples per problem group, and evaluating selector accuracy. It processes raw NPZ activation shards into efficient binary caches (CSR format with Roaring Bitmap indexing) and provides 10 selection algorithms.

## Commands

```bash
# Setup
bash cookbook/00_setup/install.sh          # install all dependencies
bash cookbook/00_setup/verify.sh            # verify environment (Python, packages, MUI_HUB)

# Core CLI (always run from repo root)
python3 -m nad.cli analyze --cache-root <path> --distance ja --selectors all --distance-threads 16 --out result.json
python3 -m nad.cli analyze --cache-root <path> --pos-window 0-8 --pos-size 32 --out window.json
python3 -m nad.cli accuracy --selection result.json --cache-root <path> --out accuracy.json
python3 scripts/rank_selectors.py --results-dir ./result/all_model_TIMESTAMP --no-bootstrap --csv --json

# Cookbook scripts (always run from repo root)
bash cookbook/01_cache_browser/cache_browser.sh --background    # port 5003
bash cookbook/02_visualization/visualization.sh MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/aime24/cache_neuron_output_1_act_no_rms_20250902_025610 --background  # port 5002
bash cookbook/03_batch_analyze/batch_analyze.sh --dry-run       # preview
bash cookbook/03_batch_analyze/batch_analyze.sh -y              # run all
bash cookbook/04_position_ablation/position_ablation.sh --quick
bash cookbook/05_deepconf_analysis/deepconf_analysis.sh standard --compare-all

# Validation (Python REPL)
from nad.core.views.reader import CacheReader; r = CacheReader('./cache_path'); print(r.num_runs)
```

No formal test suite exists. Validate changes by running `verify.sh` and testing against a real cache with the CLI.

## Thread / Parallelism Constraint

**This machine has 16 cores.** All scripts must satisfy `THREADS × PARALLEL_JOBS ≤ 16`.

| PARALLEL_JOBS | THREADS |
|---------------|---------|
| 1 | 16 |
| 4 | 4 |
| 8 | 2 |

Cookbook defaults: `03_batch_analyze` THREADS=4/PARALLEL_JOBS=4, `04_position_ablation` THREADS=4/PARALLEL_JOBS=4, `05_deepconf_analysis` THREADS=16/PARALLEL_JOBS=1.

## Architecture

```
nad/
  api.py              # Public API: open_cache(), load_correctness_map(), extract_tokenwise_counts()
  cli/__main__.py     # CLI entry: {analyze, accuracy, cache-build, cache-build-fast}
  pipeline/
    analysis.py       # Main orchestrator: meta.json → group by problem → distance matrix → selectors → JSON
    build_cache_fast.py  # Two-stage Map-Reduce NPZ→CSR builder
    window_cache.py   # Position-window cache management
    profiler.py       # Optional perf monitoring
  core/
    distance/engine.py   # DistanceEngine: Jaccard (roaring ≥4096 or numpy), weighted Jaccard
    selectors/
      base.py         # Selector + SelectorContext (stable plugin API)
      impl.py         # 10 selector implementations
      registry.py     # build_selector(), expand_selector_all(), external plugin loader (file: / py:)
    storage/binary_io.py  # mmap_from_file() for all binary arrays
    views/reader.py   # CacheReader: lazy mmap access, RunView(keys, weights), TokenView
  io/loader.py        # NadNextLoader: 256 MB LRU cache, entropy/confidence aggregates
  ops/
    accuracy.py       # compute_accuracy_report() against ground truth
    selector_ranking.py  # RNS / regret / Copeland scoring with Bootstrap CI
    uniques.py        # extract_tokenwise_counts() for custom selectors
cookbook/              # 6 chapters (00-05), each with a README.md + shell script
scripts/              # rank_selectors.py, plot_*.py
plugins/kink_selector.py  # Reference custom selector implementation
tools/                # cache_browser.py, auto_generate_ground_truth.py
minimal_visualization_next/  # Flask + Plotly visualization server (app.py + templates/)
```

### Key Data Flow

`NPZ Shards → cache-build-fast → Binary Cache (CSR + mmap) → nad.cli analyze → Selection JSON → nad.cli accuracy → Accuracy Report → rank_selectors.py → Cross-task Ranking`

### Critical Call Chain (analyze)

`cli/__main__.py` → `pipeline/analysis.py:analyze()` → reads `meta.json`, groups samples by `problem_id` → creates `CacheReader` → builds `RunView` per sample via `ViewSpec(agg, cut, order)` → `DistanceEngine.compute_matrix()` (threaded) → each `Selector.select(D, run_stats)` returns group-internal index (0 to n-1) → mapped to global `run_id` → output JSON.

## Cache Layout

```
cache_neuron_output_1_*/
  meta.json                      # Sample→problem mapping (auto-parsed)
  manifest.json                  # Schema v4.0+ checksums
  evaluation_report_compact.json # Ground truth (0.8 MB compressed)
  base/                          # CSR neuron activations: row_ptr, keys, w_max, w_sum
  index/                         # Sorted indices for lookups
  token_data/                    # Per-token LM stats: tok_conf, tok_neg_entropy, tok_gini, tok_selfcert, tok_logprob
  rows/                          # Row-CSR Bank for position-window queries (v4.1+)
```

**Neuron data** (`base/`, `rows/`) and **token data** (`token_data/`) are independent dimensions sharing the same row structure. Type `cache_neuron_output_1_*` = 1 token/row; type `cache_neuron_output_2_*` = ~32 tokens/row.

The `MUI_HUB` symlink at repo root points to shared cache storage. **Always use relative `MUI_HUB/...` paths** — scripts prepend `../` internally, so absolute paths break.

## Key Implementation Details

**Backend selection**: `roaring` backend is 3–5× faster when `max(|A|, |B|) ≥ 4096`; `numpy` is faster for small sets. Default `auto` switches at this threshold. Controlled by `NAD_JA_BACKEND` env var or `DistanceSpec.ja_backend`.

**DeepConf quality direction** (easy to get wrong):
- `tok_conf`: lower → more confident → `quality = -tok_conf`
- `tok_neg_entropy`: closer to 0 → more certain → `quality = tok_neg_entropy` (identity, NOT negated)

**Sliding window** (DeepConf): strict trailing window, no padding. Sequences shorter than `window_size` return global mean.

**Selector contract**: `select(D, run_stats)` must return **group-internal indices** (0 to n-1), NOT global run_ids. The pipeline maps indices to run_ids. Filter `inf` values before `argmax`/`argmin` (denominator-zero case).

**Entropy semantics**: `tok_neg_entropy = Σ p log p ≤ 0`. `entropy_sum` from `loader.get_slice_entropy_sum_for_sample` = `−Σ tok_neg_entropy` = positive Shannon entropy per token.

**External selector loading**: `file:path/to/module.py:ClassName` or `py:module.path:ClassName`. Duck-typing: must implement `select()`; optionally `bind()`.

**All binary I/O** goes through `nad.core.storage.binary_io.mmap_from_file()` — arrays are memory-mapped, not loaded into RAM.

## Environment

```bash
pip install .   # or: pip install numpy>=1.20.0 pyroaring>=0.4.5
export NAD_JA_BACKEND=roaring   # or numpy, auto
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

`nad_config.json` at repo root specifies `model_search_dirs` for tokenizer lookup (visualization server only).
