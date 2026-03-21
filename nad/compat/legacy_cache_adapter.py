"""Compatibility layer for legacy NAD_tmp cache readers.

This module exposes the legacy-style APIs ``load_efficient_cache`` and
``load_neuron_data_parallel`` but serves the data from a NAD_Next cache.

It allows old scripts that expect the efficient cache utilities to work
transparently with the new binary cache format.  The resolver tries to find a
NAD_Next cache next to the original neuron_output directory; alternatively the
environment variable ``NAD_NEXT_CACHE_ROOT`` can point to the cache.

Only the minimal subset that the old analysis scripts rely on is implemented:

* ``load_efficient_cache`` returns correctness labels, metadata and the path to
  the evaluation report.  Top-K caches are not available in NAD_Next, therefore
  an empty stub is returned.
* ``load_neuron_data_parallel`` reconstructs the per-sample / per-slice neuron
  activations from the Row-CSR bank contained in the NAD_Next cache.  The
  structure matches the output of the legacy reader:

      neuron_data[sample_id][slice_id] = [(layer, neuron, score), ...]

Token metrics are currently not reconstructed and the function returns ``None``
for that part, matching the legacy behaviour when such data was absent.

Usage
-----

    from nad.compat.legacy_cache_adapter import (
        load_efficient_cache, load_neuron_data_parallel
    )

    correctness_map, meta, eval_file, _ = load_efficient_cache(legacy_dir)
    neuron_data, _ = load_neuron_data_parallel(legacy_dir)

"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import numpy as np

from nad.core.views.reader import CacheReader, Agg


def _resolve_cache_dir(legacy_dir: str) -> Path:
    """Resolve the NAD_Next cache directory for a legacy neuron output path."""

    # Explicit override via environment variable
    override = os.getenv("NAD_NEXT_CACHE_ROOT")
    if override:
        override_path = Path(override).expanduser().resolve()
        if (override_path / "manifest.json").is_file():
            return override_path
        raise FileNotFoundError(f"NAD_NEXT_CACHE_ROOT={override_path} is not a valid cache directory")

    legacy_path = Path(legacy_dir).expanduser().resolve()

    # If caller already points at a NAD_Next cache, use it directly
    if (legacy_path / "manifest.json").is_file():
        return legacy_path

    parent = legacy_path.parent
    if not parent.exists():
        raise FileNotFoundError(f"Legacy directory not found: {legacy_path}")

    dataset_name = parent.name

    # Heuristic 1: sibling named cache_<dataset>
    candidate = parent / f"cache_{dataset_name}"
    if (candidate / "manifest.json").is_file():
        return candidate

    # Heuristic 2: any sibling containing a manifest.json
    for sibling in parent.iterdir():
        if sibling.is_dir() and (sibling / "manifest.json").is_file():
            return sibling

    raise FileNotFoundError(
        f"Unable to locate NAD_Next cache near {legacy_path}. "
        "Set NAD_NEXT_CACHE_ROOT to the new cache directory."
    )


def _load_correctness_map(cache_dir: Path) -> Tuple[Dict[int, bool], Dict[str, Any], str]:
    meta_path = cache_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.json not found in {cache_dir}")

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    # evaluation report (compact preferred)
    eval_compact = cache_dir / "evaluation_report_compact.json"
    eval_full = cache_dir / "evaluation_report.json"
    if eval_compact.is_file():
        eval_path = eval_compact
    elif eval_full.is_file():
        eval_path = eval_full
    else:
        report_path = meta.get("report_path")
        if report_path and Path(report_path).is_file():
            eval_path = Path(report_path)
        else:
            raise FileNotFoundError(
                "evaluation_report_compact.json/evaluation_report.json not found."
            )

    with eval_path.open("r", encoding="utf-8") as f:
        eval_report = json.load(f)

    sample_index: Dict[Tuple[int, int], int] = {}
    for sid, sample in enumerate(meta.get("samples", [])):
        sample_index[(sample["problem_id"], sample["run_index"])] = sid

    correctness_map: Dict[int, bool] = {}
    for result in eval_report.get("results", []):
        pid = result["problem_id"]
        for run in result.get("runs", []):
            key = (pid, run["run_index"])
            if key in sample_index:
                correctness_map[sample_index[key]] = bool(run["is_correct"])

    return correctness_map, meta, str(eval_path)


def load_efficient_cache(
    data_dir: str,
    cache_subdir: str = "efficient_cache",
    required_topk: Optional[int] = None,
    cache_root_override: Optional[str] = None,
) -> Tuple[Dict[int, bool], Dict[str, Any], str, Dict[str, Any]]:
    """Legacy compatible wrapper for :func:`efficient_cache_utils.load_efficient_cache`."""

    if cache_root_override:
        os.environ.setdefault("NAD_NEXT_CACHE_ROOT", cache_root_override)

    cache_dir = _resolve_cache_dir(data_dir)
    correctness_map, meta, eval_path = _load_correctness_map(cache_dir)

    # Top-K caches are not materialised in NAD_Next.  Return an explicit stub to
    # keep downstream code functional.
    topk_stub = {
        "metadata": {
            "note": "Generated by legacy_cache_adapter; Top-K data is unavailable in NAD_Next"
        },
        "method1": {},
        "method2": {}
    }
    return correctness_map, meta, eval_path, topk_stub


def _decode_key(key: int) -> Tuple[int, int]:
    layer = int((key >> 16) & 0xFFFF)
    neuron = int(key & 0xFFFF)
    return layer, neuron


def load_neuron_data_parallel(
    backup_dir: str,
    num_shards: int = None,
    use_preload_cache: bool = False,
    process_workers: int = None,
    include_token_metrics: bool = False,
    cache_root_override: Optional[str] = None,
) -> Tuple[Dict[int, Dict[int, list]], Optional[Dict]]:
    """Legacy compatible wrapper for ``fast_neuron_reader.load_neuron_data_parallel``.

    Parameters other than ``backup_dir`` are accepted for API compatibility but
    ignored—the NAD_Next cache already contains the fully materialised data.
    """

    if cache_root_override:
        os.environ.setdefault("NAD_NEXT_CACHE_ROOT", cache_root_override)

    cache_dir = _resolve_cache_dir(backup_dir)
    reader = CacheReader(str(cache_dir))

    rows_srp = reader.rows_sample_row_ptr
    rows_rp = reader.rows_row_ptr
    rows_keys = reader.rows_keys
    rows_w = reader.rows_weights_for(Agg.MAX)
    rows_slice_ids = reader.rows_slice_ids

    if any(x is None for x in (rows_srp, rows_rp, rows_keys, rows_w, rows_slice_ids)):
        raise RuntimeError(
            "Row-CSR bank not available in the NAD_Next cache. Re-run cache-build-fast with --row-bank."
        )

    neuron_data: Dict[int, Dict[int, list]] = {}

    total_runs = reader.num_runs()
    for sample_id in range(total_runs):
        rs = int(rows_srp[sample_id])
        re = int(rows_srp[sample_id + 1])
        if re <= rs:
            continue

        slice_ids = np.asarray(rows_slice_ids[rs:re], dtype=np.int64)
        order = np.argsort(slice_ids, kind="mergesort")

        sample_dict: Dict[int, list] = {}
        for pos, rel_idx in enumerate(order):
            row_idx = rs + rel_idx
            slice_id = int(slice_ids[rel_idx])

            k0 = int(rows_rp[row_idx])
            k1 = int(rows_rp[row_idx + 1])
            if k1 <= k0:
                continue

            keys = np.asarray(rows_keys[k0:k1], dtype=np.uint32)
            weights = np.asarray(rows_w[k0:k1], dtype=np.float32)

            entries = [(*_decode_key(int(key)), float(weight)) for key, weight in zip(keys, weights)]
            sample_dict[slice_id] = entries

        if sample_dict:
            neuron_data[sample_id] = sample_dict

    # Token metrics reconstruction is not yet supported; return None to match
    # the legacy API when such data was missing.
    token_metrics = None
    if include_token_metrics:
        token_metrics = {}

    return neuron_data, token_metrics


__all__ = [
    "load_efficient_cache",
    "load_neuron_data_parallel",
]

