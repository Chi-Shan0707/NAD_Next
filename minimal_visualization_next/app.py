#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NAD Next Streaming Visualization Server v4.1

A Flask-based web server for interactive visualization of neural activation distributions.
Provides on-demand streaming of neuron activation data with LRU caching for efficiency.

Features:
- Streaming data loading via NadNextLoader
- Interactive Plotly visualizations
- Token-level entropy and metadata display
- Problem-based run grouping (correct/incorrect)
- Multiple API endpoints for data access

Author: NAD Next Team
License: MIT
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
from flask import Flask, jsonify, render_template, request

from nad.io import build_problem_catalog, detect_nad_next_cache, load_nad_next_index, NadNextLoader

# =============================================================================
# Configuration Constants
# =============================================================================

DEFAULT_LRU_CACHE_MB = 256
DEFAULT_MAX_RUNS = 50
DEFAULT_PARETO_TOPK = 0.20
TOKEN_ENTROPY_PERCENTILE = 80.0
DEFAULT_PORT = 5001
DEFAULT_HOST = "0.0.0.0"

# =============================================================================
# Logging Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s in %(module)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Optional Tokenizer Backends
# =============================================================================

try:  # pragma: no cover - optional dependency
    from transformers import AutoTokenizer, PreTrainedTokenizerFast  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    AutoTokenizer = None  # type: ignore
    PreTrainedTokenizerFast = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from tokenizers import Tokenizer as HFTokenizer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    HFTokenizer = None  # type: ignore

# =============================================================================
# Flask Application Setup
# =============================================================================

TEMPLATE_DIR = Path(__file__).parent / "templates"
app = Flask(__name__, template_folder=str(TEMPLATE_DIR))
app.config["JSON_SORT_KEYS"] = False

# Support JupyterHub reverse proxy: strips X-Forwarded-Prefix so Flask
# routes resolve correctly under /user/<name>/proxy/<port>/
from werkzeug.middleware.proxy_fix import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Track server start time for uptime monitoring
SERVER_START_TIME = time.time()


class _TokWrapper:
    """
    Lightweight wrapper that normalizes the tokenizer API.

    Supports multiple tokenizer backends:
    - hf-auto: transformers.AutoTokenizer
    - hf-fast: transformers.PreTrainedTokenizerFast
    - tokenizers: tokenizers.Tokenizer

    Args:
        kind: Backend type identifier
        obj: Tokenizer object instance
    """

    def __init__(self, kind: str, obj):
        self.kind = kind
        self._obj = obj

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text string."""
        if self._obj is None:
            return ""
        if self.kind == "hf-auto":
            return self._obj.decode(ids, skip_special_tokens=False)
        if self.kind in {"hf-fast", "tokenizers"}:
            return self._obj.decode(ids)
        return ""


def _load_tokenizer(model_path: str) -> Optional[_TokWrapper]:
    """
    Load tokenizer with graceful degradation through multiple backends.

    Attempts to load in order:
    1. transformers.AutoTokenizer (best compatibility)
    2. transformers.PreTrainedTokenizerFast (fast, requires tokenizer.json)
    3. tokenizers.Tokenizer (minimal, requires tokenizer.json)

    Args:
        model_path: Path to model directory containing tokenizer files

    Returns:
        _TokWrapper instance if successful, None otherwise

    Example:
        >>> tokenizer = _load_tokenizer("/path/to/model")
        >>> text = tokenizer.decode([101, 2023, 2003])  # [CLS] this is
    """
    last_err: Optional[Exception] = None
    logger.info(f"Attempting to load tokenizer from: {model_path}")

    # Strategy 1: AutoTokenizer (most compatible)
    if AutoTokenizer is not None:
        try:
            tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            logger.info(f"✓ Loaded tokenizer via AutoTokenizer from {model_path}")
            return _TokWrapper("hf-auto", tok)
        except Exception as exc:  # pragma: no cover
            logger.debug(f"AutoTokenizer failed: {exc}")
            last_err = exc

    # Strategy 2: PreTrainedTokenizerFast (requires tokenizer.json)
    if PreTrainedTokenizerFast is not None:
        try:
            tj = os.path.join(model_path, "tokenizer.json")
            if os.path.exists(tj):
                tok = PreTrainedTokenizerFast(tokenizer_file=tj)
                logger.info(f"✓ Loaded tokenizer via PreTrainedTokenizerFast from {tj}")
                return _TokWrapper("hf-fast", tok)
            else:
                logger.debug(f"tokenizer.json not found at {tj}")
        except Exception as exc:  # pragma: no cover
            logger.debug(f"PreTrainedTokenizerFast failed: {exc}")
            last_err = exc

    # Strategy 3: HF Tokenizers library (minimal)
    if HFTokenizer is not None:
        try:
            tj = os.path.join(model_path, "tokenizer.json")
            if os.path.exists(tj):
                tok = HFTokenizer.from_file(tj)
                logger.info(f"✓ Loaded tokenizer via tokenizers.Tokenizer from {tj}")
                return _TokWrapper("tokenizers", tok)
        except Exception as exc:  # pragma: no cover
            logger.debug(f"HF Tokenizer failed: {exc}")
            last_err = exc

    logger.warning(f"⚠️  All tokenizer backends failed. Last error: {last_err}")
    logger.warning(f"⚠️  Token decoding will not be available. Check model_path in meta.json")
    return None


# =============================================================================
# Model Path Adaptive Search
# =============================================================================

# Fallback directories for model search (in order of priority)
# Format: direct paths where model_name folder might exist
MODEL_SEARCH_DIRS = []

# Load extra search dirs from nad_config.json (repo root or parent of this file)
def _load_config_search_dirs() -> list:
    for candidate in [
        Path(__file__).parent.parent / "nad_config.json",  # NAD_Next/nad_config.json
        Path(__file__).parent / "nad_config.json",
    ]:
        if candidate.exists():
            try:
                cfg = json.loads(candidate.read_text())
                dirs = cfg.get("model_search_dirs", [])
                if dirs:
                    logger.debug(f"Loaded model_search_dirs from {candidate}: {dirs}")
                return dirs
            except Exception as e:
                logger.warning(f"Failed to read {candidate}: {e}")
    return []

MODEL_SEARCH_DIRS = _load_config_search_dirs() + MODEL_SEARCH_DIRS


def _find_model_path(original_path: Optional[str]) -> Optional[str]:
    """
    Find model path with adaptive fallback search.

    If the original path exists, return it directly.
    If not, extract the model name and search in predefined directories.

    Args:
        original_path: Original model path from meta.json

    Returns:
        Valid model path if found, None otherwise

    Example:
        >>> # Original path doesn't exist
        >>> path = _find_model_path("/data/chenkang/models/DeepSeek-R1-0528-Qwen3-8B")
        >>> # Searches and finds: /datacenter/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
    """
    if not original_path:
        return None

    # Check if original path exists
    if os.path.exists(original_path):
        return original_path

    # Extract model name from path
    model_name = os.path.basename(original_path.rstrip('/'))
    if not model_name:
        logger.warning(f"⚠️  Cannot extract model name from: {original_path}")
        return None

    logger.info(f"🔍 Model path not found: {original_path}")
    logger.info(f"🔍 Searching for model '{model_name}' in fallback directories...")

    # Simple direct search in each directory
    for search_dir in MODEL_SEARCH_DIRS:
        candidate = os.path.join(search_dir, model_name)
        if os.path.isdir(candidate):
            logger.info(f"✓ Found model at: {candidate}")
            return candidate

    logger.warning(f"⚠️  Model '{model_name}' not found in any fallback directory")
    return None


# =============================================================================
# Global State Management
# =============================================================================

GLOBAL_STATE: Dict[str, Any] = {
    "loader": None,                          # NadNextLoader instance
    "viz_index": None,                       # Lightweight index from load_nad_next_index()
    "problems_data": None,                   # Problem catalog from build_problem_catalog()
    "problem_ids": [],                       # Sorted list of problem IDs
    "precompute_status": None,               # {"entropy": bool, "cumcnt": bool}
    "tokenizer": None,                       # _TokWrapper instance
    "neuron_meta": None,                     # Loaded meta.json content
    "data_loaded": False,                    # True when initialization complete
    "loading_status": "Not started",         # User-facing status string
    "data_format": "nad_next_streaming",     # Backend identifier
    "has_evaluation_report": False,          # True if evaluation_report found
    "warnings": [],                          # List of warning messages
    # Multi-cache mode fields
    "multi_cache_mode": False,               # True when --cache-root is used
    "cache_tree": {},                        # {model: {dataset: [cache_name, ...]}}
    "current_selection": {"model": None, "dataset": None, "cache": None},
    "cache_root_path": None,                 # Resolved Path to the models base dir
    "max_cache_mb_setting": 256,             # Preserved for re-loading
}

GLOBAL_LOCK = threading.Lock()


def _update_state(**updates) -> None:
    """Thread-safe update of GLOBAL_STATE dictionary."""
    with GLOBAL_LOCK:
        GLOBAL_STATE.update(updates)


def _scan_cache_tree(root_path: str) -> tuple:
    """
    Scan a cache root directory for model/dataset/cache hierarchy.

    Expects structure: root_path/[cache/]model/dataset/cache_dir/manifest.json
    Automatically detects whether a 'cache/' subdirectory exists.

    Args:
        root_path: Path to the root directory (e.g., ./MUI_HUB)

    Returns:
        (resolved_base_path, tree_dict) where tree_dict is
        {model_name: {dataset_name: [cache_name, ...]}}
    """
    root = Path(root_path).resolve()
    # Auto-detect cache/ subdirectory
    if (root / "cache").is_dir():
        base = root / "cache"
    else:
        base = root

    tree: Dict[str, Dict[str, List[str]]] = {}
    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith('.'):
            continue
        datasets: Dict[str, List[str]] = {}
        for dataset_dir in sorted(model_dir.iterdir()):
            if not dataset_dir.is_dir() or dataset_dir.name.startswith('.'):
                continue
            caches: List[str] = []
            for cache_dir in sorted(dataset_dir.iterdir()):
                if not cache_dir.is_dir() or cache_dir.name.startswith('.'):
                    continue
                # Validate by checking manifest.json existence (fast, no parse)
                if (cache_dir / "manifest.json").exists():
                    caches.append(cache_dir.name)
            if caches:
                datasets[dataset_dir.name] = caches
        if datasets:
            tree[model_dir.name] = datasets

    total = sum(len(c) for d in tree.values() for c in d.values())
    logger.info(f"Scanned cache tree: {len(tree)} models, {total} caches total")
    return base, tree


def _switch_cache(model: str, dataset: str, cache_name: str, max_cache_mb: int) -> None:
    """
    Switch to a different cache in multi-cache mode.

    Builds the cache path from the stored root, resets loader state,
    and launches load_data_background.

    Args:
        model: Model directory name
        dataset: Dataset directory name
        cache_name: Cache directory name
        max_cache_mb: LRU cache memory limit in MB
    """
    cache_root_path = GLOBAL_STATE.get("cache_root_path")
    if cache_root_path is None:
        logger.error("_switch_cache called but cache_root_path is None")
        return

    cache_dir = Path(cache_root_path) / model / dataset / cache_name
    logger.info(f"Switching cache to: {cache_dir}")

    # Reset data state
    _update_state(
        loader=None,
        viz_index=None,
        problems_data=None,
        problem_ids=[],
        precompute_status=None,
        tokenizer=None,
        neuron_meta=None,
        data_loaded=False,
        loading_status=f"Loading {model}/{dataset}/{cache_name}...",
        has_evaluation_report=False,
        warnings=[],
        current_selection={"model": model, "dataset": dataset, "cache": cache_name},
    )

    # Reuse existing loading logic
    load_data_background(str(cache_dir), max_cache_mb)


def load_data_background(cache_root: str, max_cache_mb: int = 256) -> None:
    """
    Initialize data loader and populate GLOBAL_STATE.

    Runs in background thread to avoid blocking Flask startup.
    Performs comprehensive validation of cache structure and loads all necessary components.

    Args:
        cache_root: Path to NAD_NEXT cache directory
        max_cache_mb: LRU cache memory limit in MB

    Side Effects:
        Updates GLOBAL_STATE with loader, index, problems, tokenizer, etc.

    Validation Steps:
        1. Detect NAD_NEXT cache format (manifest.json exists)
        2. Verify required directories (base/, rows/, index/, token_data/)
        3. Initialize NadNextLoader
        4. Load meta.json and model metadata
        5. Build lightweight viz index
        6. Load evaluation report (if available)
        7. Build problem catalog
        8. Pre-load tokenizer (if model_path available)
    """
    warnings_list: List[str] = []

    try:
        _update_state(loading_status="Detecting NAD_NEXT cache...", data_loaded=False)
        logger.info("🔄 Detecting NAD_NEXT cache...")

        # Step 1: Detect cache format
        cache_path = Path(cache_root)
        if not detect_nad_next_cache(cache_root):
            msg = f"NAD_NEXT cache not detected at {cache_root}"
            logger.error(f"❌ {msg}")
            logger.error(f"   Expected manifest.json and base/ directory")
            _update_state(loading_status=msg)
            return

        # Step 2: Validate required directories
        logger.info("Validating cache structure...")
        required_dirs = ["base", "rows", "index"]
        optional_dirs = ["token_data", "window_cache"]

        for dirname in required_dirs:
            dir_path = cache_path / dirname
            if not dir_path.exists():
                msg = f"Missing required directory: {dirname}/"
                logger.error(f"❌ {msg}")
                _update_state(loading_status=f"Error: {msg}")
                return
            logger.debug(f"  ✓ Found {dirname}/")

        for dirname in optional_dirs:
            dir_path = cache_path / dirname
            if dir_path.exists():
                logger.debug(f"  ✓ Found {dirname}/")
            else:
                logger.debug(f"  - Optional {dirname}/ not found")

        logger.info("✓ Cache structure valid")
        logger.info("🚀 Initializing NAD_NEXT streaming loader...")
        _update_state(loading_status="Initializing NAD_NEXT loader...")

        # Step 3: Initialize loader
        loader = NadNextLoader(cache_root, max_cache_mb=max_cache_mb, enable_progress=True)
        logger.info(f"✓ Loader initialized (LRU cache limit: {max_cache_mb} MB)")

        # Step 4: Load meta.json and model metadata
        logger.info("Loading meta.json...")
        meta_path = cache_path / "meta.json"
        neuron_meta = None
        model_path = None

        if meta_path.exists():
            try:
                neuron_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                model_path = neuron_meta.get("model_path")
                if model_path:
                    logger.info(f"📁 Model path: {model_path}")
                else:
                    logger.warning("⚠️  meta.json missing 'model_path' field")
                    warnings_list.append("meta.json missing 'model_path' - token decoding unavailable")
            except Exception as exc:
                logger.error(f"Failed to parse meta.json: {exc}")
                warnings_list.append(f"meta.json parse error: {exc}")
        else:
            logger.warning(f"⚠️  meta.json not found at {meta_path}")
            warnings_list.append("meta.json not found")

        # Step 5: Build lightweight index
        logger.info("Building lightweight visualization index...")
        _update_state(loading_status="Building visualization index...")
        viz_index = load_nad_next_index(loader)
        num_samples = len(viz_index['sample_ids'])
        logger.info(f"✅ Loaded NAD_NEXT index with {num_samples} samples")

        # Step 6: Load evaluation report (correctness labels)
        logger.info("Loading evaluation report...")
        eval_report_path = cache_path / "evaluation_report_compact.json"
        correctness_map = {}
        has_evaluation_report = False

        if eval_report_path.exists():
            try:
                logger.info("📊 Loading evaluation_report_compact.json...")
                report = json.loads(eval_report_path.read_text(encoding="utf-8"))
                for problem in report.get("results", []):
                    # Keep problem_id as string to support formats like "gpqa-0" or "123"
                    pid = str(problem["problem_id"])
                    for run in problem.get("runs", []):
                        correctness_map[(pid, int(run.get("run_index", 0)))] = bool(run.get("is_correct", False))
                logger.info(f"   ✓ Loaded correctness data for {len(correctness_map)} runs")
                has_evaluation_report = True
            except Exception as exc:
                logger.error(f"Failed to parse evaluation report: {exc}")
                warnings_list.append(f"evaluation report parse error: {exc}")
        else:
            logger.warning("⚠️  evaluation_report_compact.json not found")
            logger.warning("   → All runs will be categorized as 'correct'")
            warnings_list.append("No evaluation report - all runs marked 'correct'")

        # Step 7: Build problem catalog
        logger.info("Building problem catalog...")
        _update_state(loading_status="Grouping runs by problem...")
        problems_data = build_problem_catalog(viz_index, correctness_map)
        problem_ids = sorted(list(problems_data.keys()))
        logger.info(f"✅ Cataloged {len(problem_ids)} problems")

        # Step 8: Pre-load tokenizer (avoids lazy loading race conditions)
        # Use adaptive model path search
        tokenizer = None
        resolved_model_path = _find_model_path(model_path)

        if resolved_model_path:
            if resolved_model_path != model_path:
                logger.info(f"📁 Resolved model path: {model_path} -> {resolved_model_path}")
            logger.info("Pre-loading tokenizer...")
            _update_state(loading_status="Loading tokenizer...")
            tokenizer = _load_tokenizer(resolved_model_path)
            if tokenizer:
                logger.info("✓ Tokenizer pre-loaded successfully")
            else:
                logger.warning("⚠️  Tokenizer pre-load failed")
                warnings_list.append("Tokenizer unavailable - token decoding disabled")
        else:
            if model_path:
                logger.warning(f"⚠️  Model path not found after search: {model_path}")
                warnings_list.append(f"Model path not found: {model_path}")

        # Determine operational mode
        precompute_status = loader.precompute_status
        mode = "optimal" if all(precompute_status.values()) else "fallback (LRU)"
        status = f"Ready! {len(problem_ids)} problems (streaming, {mode})"

        # Update global state with all loaded components
        _update_state(
            loader=loader,
            viz_index=viz_index,
            problems_data=problems_data,
            problem_ids=problem_ids,
            precompute_status=precompute_status,
            tokenizer=tokenizer,
            neuron_meta=neuron_meta,
            data_loaded=True,
            loading_status=status,
            has_evaluation_report=has_evaluation_report,
            warnings=warnings_list,
        )

        # Print summary
        logger.info("=" * 70)
        logger.info(f"✅ NAD_NEXT streaming visualization ready!")
        logger.info(f"   Problems: {len(problem_ids)}")
        logger.info(f"   Samples: {num_samples}")
        logger.info(f"   Mode: {mode}")
        logger.info(f"   LRU cache: {max_cache_mb} MB")
        logger.info(f"   Tokenizer: {'✓ Loaded' if tokenizer else '✗ Not available'}")
        logger.info(f"   Eval report: {'✓ Loaded' if has_evaluation_report else '✗ Missing'}")
        if warnings_list:
            logger.info(f"   Warnings: {len(warnings_list)}")
            for warning in warnings_list:
                logger.warning(f"     - {warning}")
        logger.info("=" * 70)

        if not all(precompute_status.values()):
            logger.warning("\n⚠️  PERFORMANCE NOTICE:")
            logger.warning("   First-time queries will be slower (on-demand aggregation)")
            logger.warning("   Consider running cache build with --precompute flags")

    except Exception as exc:  # pragma: no cover
        import traceback
        error_msg = f"Initialization failed: {exc}"
        logger.error("=" * 70)
        logger.error(f"❌ {error_msg}")
        logger.error("=" * 70)
        traceback.print_exc()
        _update_state(loading_status=error_msg, warnings=warnings_list)


# =============================================================================
# Helper Functions
# =============================================================================

def _get_loader() -> NadNextLoader:
    """
    Get the global NadNextLoader instance.

    Returns:
        NadNextLoader instance

    Raises:
        RuntimeError: If loader not yet initialized
    """
    loader = GLOBAL_STATE.get("loader")
    if loader is None:
        raise RuntimeError("Loader not initialized yet - data still loading")
    return loader  # type: ignore[return-value]


# =============================================================================
# HMM Segmentation (verbatim from reuse_hmm_no_balance/code/icml_mini_analysis.py)
# =============================================================================

from hmmlearn.hmm import GaussianHMM as _GaussianHMM
from sklearn.mixture import GaussianMixture

_HMM_EPS = 1e-6


def _enforce_min_run_bool(labels: np.ndarray, *, min_run: int = 2) -> np.ndarray:
    """
    Merge short runs in a boolean label sequence into neighboring segments.

    This is a lightweight alternative to HSMM duration modeling. It avoids the
    E/U/E/U single-step chattering typical of naive state assignment.
    """
    x = np.asarray(labels, dtype=bool).copy()
    n = int(x.size)
    if n == 0 or min_run <= 1:
        return x

    i = 0
    while i < n:
        j = i + 1
        while j < n and bool(x[j]) == bool(x[i]):
            j += 1
        run_len = int(j - i)
        if run_len < int(min_run):
            prev_val = bool(x[i - 1]) if i > 0 else None
            next_val = bool(x[j]) if j < n else None
            if prev_val is None and next_val is None:
                pass
            else:
                new_val = prev_val if prev_val is not None else next_val
                x[i:j] = bool(new_val)
        i = j
    return x


def _hmmlearn_decode(
    X: np.ndarray,
    *,
    means: np.ndarray,
    covars: np.ndarray,
    transmat: np.ndarray,
    startprob: np.ndarray,
) -> np.ndarray:
    """Viterbi decode via hmmlearn GaussianHMM with pre-set parameters.

    Parameters
    ----------
    X : (T, D) observation matrix
    means : (K, D) emission means
    covars : (K, D) diagonal emission variances
    transmat : (K, K) transition matrix
    startprob : (K,) initial state probabilities
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)

    means = np.asarray(means, dtype=np.float64)
    covars = np.asarray(covars, dtype=np.float64)
    if means.ndim == 1:
        means = means.reshape(-1, 1)
    if covars.ndim == 1:
        covars = covars.reshape(-1, 1)

    n_components = int(means.shape[0])
    model = _GaussianHMM(
        n_components=n_components,
        covariance_type="diag",
        init_params="",
        params="",
    )
    model.startprob_ = np.asarray(startprob, dtype=np.float64)
    model.transmat_ = np.asarray(transmat, dtype=np.float64)
    model.means_ = means
    model.covars_ = covars

    _, states = model.decode(X, algorithm="viterbi")
    return states.astype(np.int64, copy=False)


def infer_explore_mask_hmm_from_slopes(
    slopes: np.ndarray, *, min_run: int = 2, detrend: bool = True,
) -> np.ndarray:
    """
    Infer an Explore/Exploit state sequence from novelty slopes using a
    **sticky 2-state Gaussian HMM**.

    Observation
    ----------
    We use a de-trended log-slope as observation:
      z_t = log(1 + slope_t)
      r_t = z_t - (a + b log(1+t))   (power-law saturation removal)

    Then we standardize r_t by (median, MAD) and fit a 2-component Gaussian
    mixture to initialize emission parameters, followed by Viterbi decoding
    under a fixed sticky transition matrix.

    This avoids a global fixed top-q fraction (e.g., 25%) and adapts per run.

    Parameters
    ----------
    detrend : bool
        If True (default), remove power-law trend via log(1+t) regression.
        Set to False for signals without structural decay (e.g. entropy),
        where only the median is subtracted before robust standardization.
    """
    slopes_arr = np.asarray(slopes, dtype=np.float32)
    if slopes_arr.size < 3:
        return np.zeros((int(slopes_arr.size),), dtype=bool)

    z = np.log1p(np.clip(slopes_arr.astype(np.float64), 0.0, None))

    if detrend:
        # De-trend: z_t ≈ a + b log(1+t)  (power-law saturation removal)
        t = np.arange(int(z.size), dtype=np.float64)
        x1 = np.log1p(t)
        X = np.stack([np.ones_like(x1), x1], axis=1)
        try:
            coef, *_ = np.linalg.lstsq(X, z, rcond=None)
            z_hat = X @ coef
            r = z - z_hat
        except Exception:
            r = z - float(np.median(z))
    else:
        # No structural trend to remove; subtract median only.
        r = z - float(np.median(z))

    # Robust standardization (median/MAD).
    med = float(np.median(r))
    mad = float(np.median(np.abs(r - med)))
    if (not np.isfinite(mad)) or mad <= 0.0:
        # Fall back: above-median residual means explore.
        explore = r > med
        return _enforce_min_run_bool(explore, min_run=min_run)
    r = (r - med) / (mad + _HMM_EPS)

    # Fit 2 Gaussians (init for HMM emissions).
    try:
        gmm = GaussianMixture(
            n_components=2,
            covariance_type="diag",
            random_state=0,
            max_iter=50,
            reg_covar=1e-6,
        )
        gmm.fit(r.reshape(-1, 1))
        means = gmm.means_.reshape(-1)
        vars_ = gmm.covariances_.reshape(-1)
        vars_ = np.clip(vars_, 1e-6, None)
    except Exception:
        explore = r > 0.0
        return _enforce_min_run_bool(explore, min_run=min_run)

    rr = r.reshape(-1, 1)

    # Sticky transition (fixed, not tuned).
    p = 0.95
    states = _hmmlearn_decode(
        rr,
        means=means.reshape(-1, 1),
        covars=vars_.reshape(-1, 1),
        transmat=np.array([[p, 1.0 - p], [1.0 - p, p]], dtype=np.float64),
        startprob=np.array([0.5, 0.5], dtype=np.float64),
    )
    explore_state = int(np.argmax(means))
    explore = states == explore_state
    return _enforce_min_run_bool(explore, min_run=min_run)


# =============================================================================
# Web Routes
# =============================================================================

@app.route("/")
def index():
    """Serve the main interactive visualization page."""
    from flask import request
    base_url = request.script_root or ""
    return render_template("index.html", base_url=base_url)


# =============================================================================
# API Endpoints - Server Status
# =============================================================================

@app.route("/api/health")
def api_health():
    """
    Health check endpoint for monitoring.

    Returns:
        JSON with server health status, uptime, and basic metrics.

    Example:
        GET /api/health
        → {"status": "healthy", "uptime_seconds": 123.45, ...}
    """
    uptime = time.time() - SERVER_START_TIME
    return jsonify({
        "status": "healthy",
        "uptime_seconds": round(uptime, 2),
        "cache_loaded": GLOBAL_STATE.get("data_loaded", False),
        "version": "4.1",
        "backend": "nad_next_streaming",
    })


@app.route("/api/status")
def api_status():
    """
    Get detailed server status and configuration.

    Returns:
        JSON with:
        - loaded: bool - whether data initialization complete
        - status: str - current status message
        - problem_count: int - number of problems available
        - problem_ids: list - first 100 problem IDs
        - backend: str - data format identifier
        - precompute_status: dict - optimization status (entropy, cumcnt)
        - has_evaluation_report: bool - whether eval report loaded
        - warnings: list - any initialization warnings

    Example:
        GET /api/status
        → {"loaded": true, "problem_count": 30, ...}
    """
    problem_ids = GLOBAL_STATE.get("problem_ids") or []
    precompute_status = GLOBAL_STATE.get("precompute_status") or {}
    warnings = GLOBAL_STATE.get("warnings") or []

    return jsonify({
        "loaded": GLOBAL_STATE.get("data_loaded", False),
        "status": GLOBAL_STATE.get("loading_status", "Not started"),
        "problem_count": len(problem_ids),
        "problem_ids": problem_ids[:100],
        "backend": GLOBAL_STATE.get("data_format", "nad_next_streaming"),
        "precompute_status": precompute_status,
        "has_evaluation_report": GLOBAL_STATE.get("has_evaluation_report", False),
        "warnings": warnings,
        "multi_cache_mode": GLOBAL_STATE.get("multi_cache_mode", False),
        "current_selection": GLOBAL_STATE.get("current_selection", {}),
    })


# =============================================================================
# API Endpoints - Performance & Diagnostics
# =============================================================================

@app.route("/api/v1/precompute_status")
def api_precompute_status():
    """
    Get precomputation status and LRU cache statistics.

    Returns detailed information about optimization status and cache performance.

    Returns:
        JSON with:
        - entropy_precomputed: bool
        - cumcnt_precomputed: bool
        - warnings: list of performance warnings
        - lru_stats: cache hit/miss statistics

    Example:
        GET /api/v1/precompute_status
        → {"entropy_precomputed": true, "lru_stats": {"hit_rate": 0.85}, ...}
    """
    try:
        loader = _get_loader()
    except RuntimeError as exc:
        return jsonify({"error": str(exc), "source": "app.py:api_precompute_status"}), 404

    precompute = loader.precompute_status
    warnings: List[str] = []
    if not precompute.get("entropy", False):
        warnings.append("Missing precomputed entropy sums - first-time queries will be slower")
    if not precompute.get("cumcnt", False):
        warnings.append("Missing precomputed neuron cumcnt - on-demand computation required")

    return jsonify({
        "entropy_precomputed": precompute.get("entropy", False),
        "cumcnt_precomputed": precompute.get("cumcnt", False),
        "warnings": warnings,
        "lru_stats": loader.get_lru_stats(),
    })


# =============================================================================
# API Endpoints - Data Queries
# =============================================================================

@app.route("/api/v1/slice_metrics")
def api_slice_metrics():
    """
    Query slice-level metrics (entropy, neuron counts) for a sample.

    Query Parameters:
        sample_id (required): int - sample ID (0-based)
        start (optional): int - start slice index
        end (optional): int - end slice index (exclusive)
        with_cumcnt (optional): "0"|"1" - include neuron cumulative counts

    Returns:
        JSON with:
        - success: bool
        - slice_ids: list[int] - slice IDs
        - entropy_sum: list[float] - entropy per slice
        - neuron_cumcnt: list[int] - neuron counts (if requested)
        - row_start, row_end: int - sample row range

    Example:
        GET /api/v1/slice_metrics?sample_id=0&start=0&end=10&with_cumcnt=1
        → {"success": true, "slice_ids": [...], ...}
    """
    try:
        loader = _get_loader()
        sample_id = int(request.args["sample_id"])
    except (KeyError, ValueError) as exc:
        return jsonify({
            "success": False,
            "error": f"Invalid parameters: {exc}",
            "hint": "Required: sample_id (int)",
            "source": "app.py:api_slice_metrics"
        }), 400
    except RuntimeError as exc:
        return jsonify({"success": False, "error": str(exc)}), 404

    # Parse window parameters with validation
    start = request.args.get("start")
    end = request.args.get("end")
    with_cumcnt = request.args.get("with_cumcnt", "0") == "1"

    try:
        start_i = int(start) if start is not None else None
        end_i = int(end) if end is not None else None

        # Validate bounds
        if start_i is not None and start_i < 0:
            return jsonify({
                "success": False,
                "error": f"start must be >= 0, got {start_i}",
                "source": "app.py:api_slice_metrics"
            }), 400

        if end_i is not None and start_i is not None and end_i <= start_i:
            return jsonify({
                "success": False,
                "error": f"end ({end_i}) must be > start ({start_i})",
                "source": "app.py:api_slice_metrics"
            }), 400

    except ValueError as exc:
        return jsonify({
            "success": False,
            "error": f"Invalid start/end values: {exc}",
            "source": "app.py:api_slice_metrics"
        }), 400

    try:
        slice_ids = loader.get_slice_ids_for_sample(sample_id, start_i, end_i).tolist()
        entropy_sum = loader.get_slice_entropy_sum_for_sample(sample_id, start_i, end_i).tolist()

        response = {
            "success": True,
            "slice_ids": slice_ids,
            "entropy_sum": entropy_sum,
        }

        if with_cumcnt:
            response["neuron_cumcnt"] = loader.get_neuron_cumcnt_for_sample(sample_id, start_i, end_i).tolist()

        row_lo, row_hi = loader.get_row_range_for_sample(sample_id)
        response.update({"row_start": int(row_lo), "row_end": int(row_hi)})
        return jsonify(response)

    except Exception as exc:
        logger.error(f"Error in api_slice_metrics: {exc}")
        return jsonify({
            "success": False,
            "error": str(exc),
            "source": "app.py:api_slice_metrics"
        }), 500


@app.route("/api/v1/slice_tokens", methods=["POST"])
def api_slice_tokens():
    try:
        loader = _get_loader()
    except RuntimeError as exc:
        return jsonify({"success": False, "error": str(exc)}), 404

    payload = request.get_json(force=True) or {}
    sample_id = int(payload.get("sample_id", -1))
    slice_indices = [int(idx) for idx in payload.get("slice_indices", [])]

    try:
        tokens_by_slice = loader.get_batch_token_data(sample_id, slice_indices)
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500

    return jsonify({"success": True, "sample_id": sample_id, "tokens_by_slice": tokens_by_slice})


def _serialise_runs(runs: List[Dict[str, object]], max_runs: int, loader: NadNextLoader,
                    pareto_topk: float) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for run in runs[:max_runs]:
        sample_id = int(run["sample_id"])
        tokens = loader.get_slice_ids_for_sample(sample_id).tolist()
        neuron_counts = loader.get_neuron_cumcnt_for_sample(sample_id).tolist()
        entropy_sums = loader.get_slice_entropy_sum_for_sample(sample_id).tolist()

        slice_indices = list(range(len(tokens)))
        token_data = loader.get_batch_token_data(sample_id, slice_indices)

        token_ids = [token_data.get(i, {}).get("token_ids", []) for i in slice_indices]
        token_entropies = [token_data.get(i, {}).get("token_entropies", []) for i in slice_indices]

        flat_ents = [float(e) for arr in token_entropies for e in arr if e is not None]
        token_any_threshold = float(np.percentile(flat_ents, 80.0)) if flat_ents else None
        has_high_any_token = (
            [any((e is not None and float(e) >= token_any_threshold) for e in arr) for arr in token_entropies]
            if token_any_threshold is not None else [False] * len(tokens)
        )

        is_topk_entropy_slice = [False] * len(tokens)
        chunk_threshold = None
        if entropy_sums and pareto_topk > 0.0:
            k = max(1, int(np.ceil(pareto_topk * len(entropy_sums))))
            order = np.argsort(np.asarray(entropy_sums, dtype=np.float32))
            topk_idx = set(int(i) for i in order[-k:])
            for idx in topk_idx:
                is_topk_entropy_slice[idx] = True
            chunk_threshold = min(entropy_sums[i] for i in topk_idx) if topk_idx else None

        payload = {
            "sample_id": sample_id,
            "tokens": tokens,
            "neuron_counts": neuron_counts,
            "token_ids": token_ids,
            "token_entropies": token_entropies,
            "slice_entropy_sums": entropy_sums,
            "pareto_threshold_token": token_any_threshold,
            "pareto_threshold_chunk": chunk_threshold,
            "has_high_entropy_any_token": has_high_any_token,
            "is_topk_entropy_slice": is_topk_entropy_slice,
            "max_entropy_slice_index": int(np.argmax(entropy_sums)) if entropy_sums else None,
            "viz_row_start": int(run.get("viz_row_start", 0)),
            "viz_row_end": int(run.get("viz_row_end", 0)),
        }

        # HMM segmentation for neuron and entropy curves
        try:
            nc_arr = np.array(neuron_counts, dtype=np.float64)
            neuron_slopes = np.diff(nc_arr, prepend=0.0)
            payload["hmm_neuron_explore"] = infer_explore_mask_hmm_from_slopes(neuron_slopes).tolist()
            payload["hmm_entropy_explore"] = infer_explore_mask_hmm_from_slopes(
                np.array(entropy_sums, dtype=np.float32),
            ).tolist()
        except Exception:
            payload["hmm_neuron_explore"] = [False] * len(neuron_counts)
            payload["hmm_entropy_explore"] = [False] * len(entropy_sums)

        output.append(payload)
    return output


@app.route("/api/plotly_data/<path:problem_id>")
def api_plotly_data(problem_id: str):
    if not GLOBAL_STATE.get("data_loaded"):
        return jsonify({"success": False, "error": "Data still loading"}), 404

    max_runs = request.args.get("max_runs", 50, type=int)
    pareto_topk = max(0.0, min(1.0, request.args.get("pareto_topk", 0.20, type=float)))

    problem_ids: List[int] = GLOBAL_STATE.get("problem_ids") or []
    if problem_ids and isinstance(problem_ids[0], int):
        try:
            actual_problem_id = int(problem_id)
        except ValueError:
            return jsonify({"success": False, "error": f"Invalid problem ID: {problem_id}"}), 404
    else:
        actual_problem_id = problem_id

    problems_data = GLOBAL_STATE.get("problems_data") or {}
    if actual_problem_id not in problems_data:
        return jsonify({"success": False, "error": f"Problem {problem_id} not found"}), 404

    loader = _get_loader()
    problem_runs = problems_data[actual_problem_id]

    correct_runs = _serialise_runs(problem_runs["correct_runs"], max_runs, loader, pareto_topk)
    incorrect_runs = _serialise_runs(problem_runs["incorrect_runs"], max_runs, loader, pareto_topk)

    return jsonify({
        "success": True,
        "problem_id": actual_problem_id,
        "correct_runs": correct_runs,
        "incorrect_runs": incorrect_runs,
        "total_correct": len(problem_runs["correct_runs"]),
        "total_incorrect": len(problem_runs["incorrect_runs"]),
        "data_format": "nad_next_streaming",
    })


@app.route("/api/run_slice_tokens", methods=["POST"])
def api_run_slice_tokens():
    return api_slice_tokens()


@app.route("/api/problem_stats/<path:problem_id>")
def api_problem_stats(problem_id: str):
    problems_data = GLOBAL_STATE.get("problems_data") or {}

    # Try string first, then integer
    if problem_id in problems_data:
        pid = problem_id
    else:
        try:
            int_pid = int(problem_id)
            if int_pid in problems_data:
                pid = int_pid
            else:
                return jsonify({"error": "Problem not found"}), 404
        except ValueError:
            return jsonify({"error": "Problem not found"}), 404

    info = problems_data[pid]
    correct_count = len(info["correct_runs"])
    incorrect_count = len(info["incorrect_runs"])
    total = correct_count + incorrect_count

    return jsonify({
        "problem_id": pid,
        "correct_runs": correct_count,
        "incorrect_runs": incorrect_count,
        "total_runs": total,
        "accuracy": (correct_count / total) if total > 0 else 0,
    })


@app.route("/api/problem_text/<path:problem_id>")
def api_problem_text(problem_id: str):
    meta = GLOBAL_STATE.get("neuron_meta") or {}
    samples = meta.get("samples") or []
    pid_str = str(problem_id)

    def _extract_text(candidate) -> Optional[str]:
        if not isinstance(candidate, (dict, str)):
            return None
        if isinstance(candidate, str):
            return candidate.strip() or None
        keys = [
            "question",
            "prompt",
            "input",
            "instruction",
            "query",
            "problem",
            "text",
            "problem_text",
            "question_text",
            "original_question",
            "task",
            "content",
        ]
        for key in keys:
            val = candidate.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return None

    text = None
    source = None
    for sample in samples:
        if str(sample.get("problem_id")) == pid_str:
            text = _extract_text(sample)
            source = "samples"
            if text:
                break

    if not text:
        problems = meta.get("problems") or {}
        if pid_str in problems:
            text = _extract_text(problems[pid_str])
            source = "problems"

    return jsonify({
        "success": bool(text),
        "problem_id": problem_id,
        "text": text or "",
        "source": source,
    })


@app.route("/api/decode_tokens", methods=["POST"])
def api_decode_tokens():
    """
    Decode token IDs to human-readable text.

    Uses pre-loaded tokenizer from initialization (thread-safe).

    POST Body (JSON):
        - token_ids: list[int] - token IDs to decode
        - needed_offsets: list[int] | null - specific indices needing offsets (optimization)
        - skip_offsets: bool - skip character offset calculation (default: false)

    Returns:
        JSON with:
        - success: bool
        - tokens: list[str] - decoded text per token
        - text: str - full concatenated text
        - char_offsets: list[int] - character position of each token end
        - token_ids: list[int] - echoed input

    Example:
        POST /api/decode_tokens
        Body: {"token_ids": [101, 2023, 2003]}
        → {"success": true, "text": "[CLS] this is", ...}
    """
    payload = request.get_json(force=True) or {}
    token_ids = [int(tid) for tid in payload.get("token_ids", [])]
    needed_offsets = payload.get("needed_offsets")
    skip_offsets = bool(payload.get("skip_offsets", False))

    logger.debug(f"decode_tokens: {len(token_ids)} tokens, needed_offsets={needed_offsets}")

    # Use pre-loaded tokenizer (populated during initialization, thread-safe)
    tokenizer: Optional[_TokWrapper] = GLOBAL_STATE.get("tokenizer")  # type: ignore[assignment]

    if not tokenizer:
        return jsonify({
            "success": False,
            "error": "Tokenizer not available",
            "hint": "Tokenizer must be pre-loaded during initialization. Check meta.json model_path.",
            "source": "app.py:api_decode_tokens"
        }), 500

    full_text = tokenizer.decode(token_ids)
    if skip_offsets:
        return jsonify({"success": True, "text": full_text, "token_ids": token_ids})

    tokens_text = [tokenizer.decode([tid]) for tid in token_ids]

    char_offsets: List[Optional[int]]
    if needed_offsets and isinstance(needed_offsets, list):
        char_offsets = [None] * len(token_ids)
        for idx in needed_offsets:
            if 0 <= idx < len(token_ids):
                char_offsets[idx] = len(tokenizer.decode(token_ids[: idx + 1]))
    else:
        cumulative = 0
        char_offsets = []
        for token in tokens_text:
            cumulative += len(token)
            char_offsets.append(cumulative)

        if token_ids and abs(cumulative - len(full_text)) > 2:
            print("[DEBUG decode_tokens] Falling back to precise offsets")
            char_offsets = []
            for i in range(len(token_ids)):
                char_offsets.append(len(tokenizer.decode(token_ids[: i + 1])))

    return jsonify({
        "success": True,
        "tokens": tokens_text,
        "text": full_text,
        "char_offsets": char_offsets,
        "token_ids": token_ids,
    })


@app.route("/api/cache_tree")
def api_cache_tree():
    """
    Return the full cache tree for dropdown population.

    Returns:
        JSON with:
        - multi_cache_mode: bool
        - tree: {model: {dataset: [cache_name, ...]}}
        - current: {model, dataset, cache} or empty
    """
    return jsonify({
        "multi_cache_mode": GLOBAL_STATE.get("multi_cache_mode", False),
        "tree": GLOBAL_STATE.get("cache_tree", {}),
        "current": GLOBAL_STATE.get("current_selection", {}),
    })


@app.route("/api/switch_cache", methods=["POST"])
def api_switch_cache():
    """
    Switch to a different cache in multi-cache mode.

    POST Body (JSON):
        - model: str - model directory name
        - dataset: str - dataset directory name
        - cache: str - cache directory name

    Returns:
        JSON with status of the switch operation.
    """
    if not GLOBAL_STATE.get("multi_cache_mode"):
        return jsonify({"success": False, "error": "Not in multi-cache mode"}), 400

    payload = request.get_json(force=True) or {}
    model = payload.get("model", "").strip()
    dataset = payload.get("dataset", "").strip()
    cache_name = payload.get("cache", "").strip()

    if not model or not dataset or not cache_name:
        return jsonify({"success": False, "error": "model, dataset, and cache are required"}), 400

    # Validate against tree
    tree = GLOBAL_STATE.get("cache_tree", {})
    if model not in tree:
        return jsonify({"success": False, "error": f"Unknown model: {model}"}), 400
    if dataset not in tree[model]:
        return jsonify({"success": False, "error": f"Unknown dataset: {dataset}"}), 400
    if cache_name not in tree[model][dataset]:
        return jsonify({"success": False, "error": f"Unknown cache: {cache_name}"}), 400

    # Check if already loading
    current_status = GLOBAL_STATE.get("loading_status", "")
    if not GLOBAL_STATE.get("data_loaded") and "Loading" in current_status:
        return jsonify({"success": False, "error": "A cache is currently loading"}), 409

    # Check if already loaded
    current = GLOBAL_STATE.get("current_selection", {})
    if (GLOBAL_STATE.get("data_loaded") and
            current.get("model") == model and
            current.get("dataset") == dataset and
            current.get("cache") == cache_name):
        return jsonify({"success": True, "status": "already_loaded"})

    # Spawn loading thread
    max_cache_mb = GLOBAL_STATE.get("max_cache_mb_setting", 256)
    switch_thread = threading.Thread(
        target=_switch_cache,
        args=(model, dataset, cache_name, max_cache_mb),
        daemon=True,
    )
    switch_thread.start()

    return jsonify({"success": True, "status": "loading"})


@app.route("/api/search")
def api_search():
    query = request.args.get("q", "").strip().lower()
    if not query:
        return jsonify([])

    problem_ids = GLOBAL_STATE.get("problem_ids") or []
    matches = [pid for pid in problem_ids if query in str(pid).lower()][:20]
    return jsonify(matches)


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive Plotly token activation visualization")
    parser.add_argument("--data-dir", required=False, default=None,
                        help="NAD_NEXT cache directory (single-cache mode)")
    parser.add_argument("--cache-root", required=False, default=None,
                        help="Root directory with model/dataset/cache hierarchy (multi-cache mode)")
    parser.add_argument("--port", type=int, default=5001, help="Port for web server")
    parser.add_argument("--host", default="0.0.0.0", help="Host for web server")
    parser.add_argument("--max-cache-mb", type=int, default=256, help="LRU cache memory limit (MB)")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = create_arg_parser()
    args = parser.parse_args(argv)

    if not args.data_dir and not args.cache_root:
        parser.error("Either --data-dir or --cache-root is required")

    if args.data_dir and args.cache_root:
        parser.error("Cannot use both --data-dir and --cache-root simultaneously")

    if args.cache_root:
        # Multi-cache mode
        base_path, tree = _scan_cache_tree(args.cache_root)
        if not tree:
            parser.error(f"No valid caches found under {args.cache_root}")

        _update_state(
            multi_cache_mode=True,
            cache_tree=tree,
            cache_root_path=str(base_path),
            max_cache_mb_setting=args.max_cache_mb,
            data_loaded=False,
            loading_status="Select a cache from the dropdowns",
        )
        total = sum(len(c) for d in tree.values() for c in d.values())
        print(f"Multi-cache mode: {len(tree)} models, {total} caches")
        print("No cache loaded yet - use the web UI dropdowns to select one")
    else:
        # Single-cache mode (original behavior)
        loader_thread = threading.Thread(
            target=load_data_background,
            args=(args.data_dir, args.max_cache_mb),
            daemon=True,
        )
        loader_thread.start()

    print(f"Starting web server on http://{args.host}:{args.port}")
    print("Open the URL in your browser to use the interactive visualization")
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover
    main()
