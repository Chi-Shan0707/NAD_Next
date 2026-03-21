#!/usr/bin/env python3
"""Lightweight NAD Next index helpers reused by visualization tools."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import numpy as np

from .loader import NadNextLoader

CacheLike = Union[str, Path, NadNextLoader]


def ensure_loader(cache_root_or_loader: CacheLike) -> NadNextLoader:
    """Return a ``NadNextLoader`` for the given cache root or loader instance."""
    if isinstance(cache_root_or_loader, NadNextLoader):
        return cache_root_or_loader
    return NadNextLoader(cache_root_or_loader)


def load_nad_next_index(cache_root_or_loader: CacheLike) -> Dict[str, np.ndarray]:
    """Build a lightweight index dictionary used by visualization front-ends."""
    loader = ensure_loader(cache_root_or_loader)

    sample_ids = loader.sample_ids()
    samples_row_ptr = loader.rows_sample_row_ptr()

    index: Dict[str, np.ndarray] = {
        "sample_ids": sample_ids,
        "samples_row_ptr": samples_row_ptr,
    }

    try:
        problem_ids = loader.problem_ids()
        if problem_ids is not None:
            index["problem_ids"] = problem_ids
    except Exception:
        pass

    try:
        run_indices = loader.run_indices()
        if run_indices is not None:
            index["run_indices"] = run_indices
    except Exception:
        pass

    # Derived counts for convenience (small ints, no extra I/O)
    try:
        index["num_samples"] = np.asarray([sample_ids.shape[0]], dtype=np.int64)
    except Exception:
        index["num_samples"] = np.asarray([0], dtype=np.int64)

    try:
        total_rows = int(samples_row_ptr[-1]) if samples_row_ptr.size else 0
        index["num_rows"] = np.asarray([total_rows], dtype=np.int64)
    except Exception:
        index["num_rows"] = np.asarray([0], dtype=np.int64)

    return index

