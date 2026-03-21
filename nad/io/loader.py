#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generalized NAD Next cache loader compatible with the minimal_visualization tools.

Goals:
- Provide a stable, reusable API for reading NAD v4.x caches
- Keep backward compatibility with minimal_visualization.nad_next_loader.NadNextLoader
- Compute common row-level aggregates on demand (entropy sum per row, cumulative unique neuron count)
- Avoid tight coupling with front-end; only expose data access + lightweight compute "operators"

This module builds on nad.core.storage.CachePaths and nad.core.views.reader.CacheReader.
"""

from __future__ import annotations
from dataclasses import dataclass
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import json
import numpy as np

from ..core.storage.cache_paths import CachePaths
from ..core.storage.binary_io import mmap_from_file
from ..core.views.reader import CacheReader

# ----------------------------
# Small LRU for ndarray values
# ----------------------------

@dataclass
class _LruArrays:
    max_bytes: int = 256 * 1024 * 1024  # 256 MB

    def __post_init__(self):
        self.store: "OrderedDict[Tuple[str,int], np.ndarray]" = OrderedDict()
        self.curr_bytes: int = 0
        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0

    @staticmethod
    def _nbytes(arr: np.ndarray) -> int:
        try:
            return int(arr.nbytes)
        except Exception:
            return int(arr.size * arr.dtype.itemsize)

    def get(self, key: Tuple[str,int]) -> Optional[np.ndarray]:
        if key in self.store:
            self.store.move_to_end(key)  # LRU bump
            self.hits += 1
            return self.store[key]
        self.misses += 1
        return None

    def put(self, key: Tuple[str,int], arr: np.ndarray) -> None:
        size = self._nbytes(arr)
        if size >= self.max_bytes:
            return  # too large to cache
        old = self.store.pop(key, None)
        if old is not None:
            self.curr_bytes -= self._nbytes(old)
        while self.curr_bytes + size > self.max_bytes and self.store:
            k, v = self.store.popitem(last=False)
            self.curr_bytes -= self._nbytes(v)
            self.evictions += 1
        self.store[key] = arr
        self.curr_bytes += size

    def stats(self) -> Dict[str, float]:
        max_mb = float(self.max_bytes) / (1024 * 1024)
        curr_mb = float(self.curr_bytes) / (1024 * 1024)
        hit_count = int(self.hits)
        miss_count = int(self.misses)
        total = hit_count + miss_count
        hit_rate = (hit_count / total) if total > 0 else 0.0
        return {
            "current_mb": curr_mb,
            "max_mb": max_mb,
            "num_entries": len(self.store),
            "hit_count": hit_count,
            "miss_count": miss_count,
            "hit_rate": hit_rate,
            "evictions": int(self.evictions),
        }

# ----------------------------
# Cache loader (public class)
# ----------------------------

class NadNextLoader:
    """
    Back-end loader for NAD Next caches.

    This loader intentionally exposes only *data access* and *lightweight compute*
    operators needed by back-end and visualization, without any Plotly/Flask code.

    Compatible API surface (subset) extracted from minimal_visualization.nad_next_loader:
      - sample_ids(), problem_ids(), run_indices()
      - rows_sample_row_ptr(), rows_row_ptr(), rows_keys(), rows_token_row_ptr()
      - token_ids(), tok_neg_entropy(), rows_slice_ids()
      - get_row_range_for_sample(), get_slice_ids_for_sample()
      - get_slice_entropy_sum_for_sample(), get_neuron_cumcnt_for_sample()
      - get_batch_token_data(), get_max_entropy_slice_index_for_sample()

    Internally it leans on nad.core.views.reader.CacheReader to remain future-proof
    with respect to on-disk layout changes.
    """

    def __init__(self, cache_root: str | Path,
                 lru_max_bytes: int = None,
                 max_cache_mb: int = None,
                 enable_progress: bool = True):  # Backward compat: ignored but accepted
        """
        Initialize NAD Next cache loader.

        Args:
            cache_root: Path to cache directory
            lru_max_bytes: LRU cache size in bytes (default: 256MB)
            max_cache_mb: Backward compat - LRU cache size in MB (overrides lru_max_bytes if set)
            enable_progress: Backward compat - ignored but accepted for compatibility
        """
        self.root = Path(cache_root)
        if not self.root.exists():
            raise FileNotFoundError(f"Cache root not found: {self.root}")
        self.paths = CachePaths(str(self.root))
        self.reader = CacheReader(str(self.root))

        # Handle backward compatibility for cache size parameter
        if max_cache_mb is not None:
            # Old parameter name (in MB)
            cache_bytes = int(max_cache_mb * 1024 * 1024)
        elif lru_max_bytes is not None:
            # New parameter name (in bytes)
            cache_bytes = int(lru_max_bytes)
        else:
            # Default: 256 MB
            cache_bytes = 256 * 1024 * 1024

        self._lru = _LruArrays(max_bytes=cache_bytes)
        self._meta: Optional[Dict] = None
        # Lazy-mapped arrays (open on first use)
        self._rows_sample_row_ptr = None
        self._rows_row_ptr = None
        self._rows_keys = None
        self._rows_token_row_ptr = None
        self._token_ids = None
        self._tok_neg_entropy = None
        self._rows_slice_ids = None

        # Backward compatibility: precompute_status for visualization
        # In streaming mode, these are computed on-demand with LRU cache (not precomputed)
        self.precompute_status = {
            'entropy': False,  # On-demand with LRU cache
            'cumcnt': False    # On-demand with LRU cache
        }

    def __getitem__(self, key: str):
        """
        Backward compatibility: make loader subscriptable for old visualization code.
        Maps dict-style access to method calls.
        """
        if key == 'sample_ids':
            return self.sample_ids()
        elif key == 'samples_row_ptr':
            return self.rows_sample_row_ptr()
        elif key == 'problem_ids':
            return self.problem_ids()
        elif key == 'run_indices':
            return self.run_indices()
        else:
            raise KeyError(f"NadNextLoader has no key '{key}'")

    def get(self, key: str, default=None):
        """
        Backward compatibility: dict-like get() method with default value.
        """
        try:
            return self[key]
        except KeyError:
            return default

    # ----------------------------
    # Basic metadata
    # ----------------------------

    def _load_meta(self) -> Dict:
        if self._meta is None:
            meta_path = self.root / "meta.json"
            if not meta_path.exists():
                # Try manifest.json (older caches)
                manifest = self.root / "manifest.json"
                if manifest.exists():
                    self._meta = json.loads(Path(manifest).read_text(encoding="utf-8"))
                else:
                    self._meta = {}
            else:
                self._meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return self._meta

    def sample_ids(self) -> np.ndarray:
        """Global sample_id per run (int32)"""
        # Try to load from file first
        sample_ids_path = Path(self.paths.sample_ids)
        if sample_ids_path.exists():
            return mmap_from_file(self.paths.sample_ids, np.int32)

        # Fallback: Derive from meta.json (implicit 0-indexed)
        meta = self._load_meta()
        if meta and "samples" in meta:
            num_samples = len(meta["samples"])
        elif meta and "total_samples" in meta:
            num_samples = int(meta["total_samples"])
        else:
            # Last resort: infer from rows_sample_row_ptr length
            try:
                srp = self.rows_sample_row_ptr()
                num_samples = len(srp) - 1
            except Exception:
                num_samples = 0

        return np.arange(num_samples, dtype=np.int32)

    def problem_ids(self) -> np.ndarray:
        """Vector of problem_id per sample_id (object/str). Derived from meta.json if present.

        Returns string array to support both numeric (e.g., "123") and non-numeric (e.g., "gpqa-0") IDs.
        """
        meta = self._load_meta()
        if not meta or "samples" not in meta:
            # Best-effort fallback: use contiguous ids [0..num_samples) as strings
            num_samples = int(self.sample_ids().max()) + 1 if self.paths.sample_ids else 0
            return np.array([str(i) for i in range(num_samples)], dtype=object)

        # Keep problem_ids as strings to support formats like "gpqa-0" or "123"
        arr = np.array([str(s["problem_id"]) for s in meta["samples"]], dtype=object)
        return arr

    def run_indices(self) -> np.ndarray:
        """run_index per sample_id (int32). Derived from meta.json when available."""
        meta = self._load_meta()
        if not meta or "samples" not in meta:
            # Fall back to zeros
            n = int(self.sample_ids().max()) + 1
            return np.zeros((n,), dtype=np.int32)
        arr = np.fromiter(
            (int(s.get("run_index", 0)) for s in meta["samples"]),
            dtype=np.int32, count=len(meta["samples"])
        )
        return arr

    # ----------------------------
    # Low-level arrays (mmap)
    # ----------------------------

    def rows_sample_row_ptr(self) -> np.ndarray:
        if self._rows_sample_row_ptr is None:
            self._rows_sample_row_ptr = mmap_from_file(self.paths.rows_sample_row_ptr, np.int64)
        return self._rows_sample_row_ptr

    def rows_row_ptr(self) -> np.ndarray:
        if self._rows_row_ptr is None:
            self._rows_row_ptr = mmap_from_file(self.paths.rows_row_ptr, np.int64)
        return self._rows_row_ptr

    def rows_keys(self) -> np.ndarray:
        if self._rows_keys is None:
            self._rows_keys = mmap_from_file(self.paths.rows_keys, np.uint32)
        return self._rows_keys

    def rows_token_row_ptr(self) -> np.ndarray:
        if self._rows_token_row_ptr is None:
            self._rows_token_row_ptr = mmap_from_file(self.paths.rows_token_row_ptr, np.int64)
        return self._rows_token_row_ptr

    def token_ids(self) -> np.ndarray:
        if self._token_ids is None:
            self._token_ids = mmap_from_file(self.paths.token_ids, np.int32)
        return self._token_ids

    def tok_neg_entropy(self) -> np.ndarray:
        """
        Load token negative entropy array (shape: [total_tokens]).

        NAD v4.0+ caches store NEGATIVE entropy values:
        - Values are negative (e.g., -2.3 to 0.0)
        - Higher values (closer to 0) indicate higher certainty
        - This representation allows direct sorting for high-confidence tokens

        Returns:
            np.ndarray: Negative entropy array (dtype=float32)

        Raises:
            FileNotFoundError: If tok_neg_entropy.float32 file is missing
        """
        if self._tok_neg_entropy is None:
            self._tok_neg_entropy = mmap_from_file(self.paths.tok_neg_entropy, np.float32)
        return self._tok_neg_entropy

    def tok_entropy(self) -> np.ndarray:
        """
        Get token entropy array (standard positive entropy, shape: [total_tokens]).

        This is a convenience wrapper that returns the standard positive entropy
        representation by negating the internally stored negative entropy.

        Entropy interpretation:
        - Values are positive (e.g., 0.0 to 2.3)
        - Higher values indicate higher uncertainty
        - Lower values indicate higher confidence
        - This is the standard entropy definition used in information theory

        Note: This operation is zero-copy (returns numpy view, not a copy).

        Returns:
            np.ndarray: Positive entropy array (dtype=float32)

        Raises:
            FileNotFoundError: If tok_neg_entropy.float32 file is missing

        Example:
            >>> loader = NadNextLoader("cache_aime24")
            >>> entropy = loader.tok_entropy()  # Standard positive entropy
            >>> print(entropy[:5])  # [1.44, 0.82, 1.18, 0.71, 1.09]
        """
        return -self.tok_neg_entropy()

    def rows_slice_ids(self) -> np.ndarray:
        if self._rows_slice_ids is None:
            self._rows_slice_ids = mmap_from_file(self.paths.rows_slice_ids, np.int32)
        return self._rows_slice_ids

    # ----------------------------
    # Helpers
    # ----------------------------

    def num_runs(self) -> int:
        ri = self.run_indices()
        return int(ri.max()) + 1 if ri.size else 0

    def get_row_range_for_sample(self, sample_id: int) -> Tuple[int,int]:
        """Return [row_lo, row_hi) global row indices for the given sample_id."""
        srp = self.rows_sample_row_ptr()
        if sample_id < 0 or sample_id + 1 >= srp.shape[0]:
            raise IndexError(f"sample_id {sample_id} out of range [0,{srp.shape[0]-2}]")
        start = int(srp[sample_id])
        end = int(srp[sample_id + 1])
        return start, end

    @staticmethod
    def _normalize_window(length: int, start: Optional[int], end: Optional[int]) -> Tuple[int, int]:
        if length < 0:
            raise ValueError("length must be non-negative")
        if start is None:
            start_idx = 0
        else:
            start_idx = max(0, min(length, int(start)))
        if end is None:
            end_idx = length
        else:
            end_idx = max(start_idx, min(length, int(end)))
        return start_idx, end_idx

    def get_slice_ids_for_sample(self, sample_id: int, start: Optional[int] = None,
                                  end: Optional[int] = None) -> np.ndarray:
        """Return slice_id for the given sample; optionally window the result."""
        row_lo, row_hi = self.get_row_range_for_sample(sample_id)
        arr = self.rows_slice_ids()[row_lo:row_hi]
        if start is None and end is None:
            return arr
        s, e = self._normalize_window(arr.shape[0], start, end)
        return arr[s:e]

    def _row_entropy_sum_range(self, row_lo: int, row_hi: int) -> np.ndarray:
        """
        Compute per-row sum of token negative entropy for rows [row_lo,row_hi).
        Vectorized via cumsum + indexed differencing (no Python loop).
        """
        num_rows = row_hi - row_lo
        if num_rows <= 0:
            return np.zeros(0, dtype=np.float32)
        trp = self.rows_token_row_ptr()
        negH = self.tok_neg_entropy()
        trp_slice = trp[row_lo:row_hi + 1]
        base = int(trp_slice[0])
        end = int(trp_slice[-1])
        if end <= base:
            return np.zeros(num_rows, dtype=np.float32)
        chunk = negH[base:end]
        # Cumulative sum with leading zero for indexed differencing
        cumsum = np.empty(chunk.size + 1, dtype=np.float64)
        cumsum[0] = 0.0
        np.cumsum(chunk, out=cumsum[1:])
        # Per-row sum = cumsum[end] - cumsum[start]
        offsets = (trp_slice - base).astype(np.intp)
        out = (cumsum[offsets[1:]] - cumsum[offsets[:-1]]).astype(np.float32)
        return out

    def get_slice_entropy_sum_for_sample(self, sample_id: int,
                                         start: Optional[int] = None,
                                         end: Optional[int] = None) -> np.ndarray:
        """Per-row entropy sum for the sample's rows (float32).

        Returns POSITIVE entropy values (standard definition for visualization).
        Internally stored negative entropy is converted to positive.
        """
        row_lo, row_hi = self.get_row_range_for_sample(sample_id)
        key = ("rows_entropy_sum", sample_id)
        cached = self._lru.get(key)
        if cached is not None:
            arr = cached
        else:
            # Get negative entropy sum and convert to positive
            neg_entropy_arr = self._row_entropy_sum_range(row_lo, row_hi)
            arr = -neg_entropy_arr  # Convert to positive entropy
            self._lru.put(key, arr)
        if start is None and end is None:
            return arr
        s, e = self._normalize_window(arr.shape[0], start, end)
        return arr[s:e]

    def _neuron_cumcnt_range(self, row_lo: int, row_hi: int) -> np.ndarray:
        """
        Compute cumulative count of unique neuron keys across rows [row_lo,row_hi).
        Vectorized: np.unique to find first occurrences, searchsorted + bincount + cumsum.
        """
        num_rows = row_hi - row_lo
        if num_rows <= 0:
            return np.zeros(0, dtype=np.int32)
        rrp = self.rows_row_ptr()
        keys = self.rows_keys()
        rrp_slice = rrp[row_lo:row_hi + 1]
        k_base = int(rrp_slice[0])
        k_end = int(rrp_slice[-1])
        if k_end <= k_base:
            return np.zeros(num_rows, dtype=np.int32)
        all_keys = keys[k_base:k_end]
        # Find first occurrence index of each unique key
        _, first_idx = np.unique(all_keys, return_index=True)
        # Map each first-occurrence position to its row index
        row_offsets = (rrp_slice - k_base).astype(np.intp)
        row_of_first = np.searchsorted(row_offsets, first_idx, side='right') - 1
        # Count new unique keys per row, then cumulative sum
        delta = np.bincount(row_of_first, minlength=num_rows)
        return np.cumsum(delta).astype(np.int32)

    def get_neuron_cumcnt_for_sample(self, sample_id: int,
                                     start: Optional[int] = None,
                                     end: Optional[int] = None) -> np.ndarray:
        """Cumulative unique neuron count curve for sample's rows (int32)."""
        row_lo, row_hi = self.get_row_range_for_sample(sample_id)
        key = ("rows_neuron_cumcnt", sample_id)
        cached = self._lru.get(key)
        if cached is not None:
            arr = cached
        else:
            arr = self._neuron_cumcnt_range(row_lo, row_hi)
            self._lru.put(key, arr)
        if start is None and end is None:
            return arr
        s, e = self._normalize_window(arr.shape[0], start, end)
        return arr[s:e]

    def get_batch_token_data(self, sample_id: int, token_lo_or_slices, token_hi: int = None):
        """
        Return token-level data for the given sample.

        Supports two calling modes for backward compatibility:

        1. New API (token range):
           get_batch_token_data(sample_id, token_lo, token_hi)
           Returns dict with 'token_ids' and 'tok_neg_entropy' arrays

        2. Old API (slice indices - for visualization compatibility):
           get_batch_token_data(sample_id, slice_indices)
           Returns dict mapping slice index -> {'token_ids': list, 'token_entropies': list}
        """
        # Detect which API mode based on second parameter type
        if isinstance(token_lo_or_slices, (list, tuple, range)):
            # Old API: slice-based access for visualization
            return self._get_batch_token_data_by_slices(sample_id, token_lo_or_slices)
        else:
            # New API: token range access
            if token_hi is None:
                raise ValueError("token_hi required when using token range API")
            return self._get_batch_token_data_by_range(sample_id, token_lo_or_slices, token_hi)

    def get_lru_stats(self) -> Dict[str, float]:
        """Return current LRU cache statistics."""
        return self._lru.stats()

    def _get_batch_token_data_by_range(self, sample_id: int, token_lo: int, token_hi: int) -> Dict[str, np.ndarray]:
        """
        New API: Return a batch of token-level arrays for [token_lo, token_hi) within the given sample.
        Token indices are *relative to the sample* (0-based).

        Returns dict with:
        - token_ids: token ID array
        - tok_neg_entropy: NEGATIVE entropy array (stored internally)
        """
        # Convert to global token indices for the sample
        # We approximate base offset by using rows_token_row_ptr at sample's first row
        row_lo, _ = self.get_row_range_for_sample(sample_id)
        trp = self.rows_token_row_ptr()
        base = int(trp[row_lo])
        g_lo = base + int(token_lo)
        g_hi = base + int(token_hi)
        token_ids = self.token_ids()[g_lo:g_hi]
        negH = self.tok_neg_entropy()[g_lo:g_hi]
        return {"token_ids": token_ids, "tok_neg_entropy": negH}

    def _get_batch_token_data_by_slices(self, sample_id: int, slice_indices: Iterable[int]) -> Dict[int, Dict[str, list]]:
        """
        Old API: Return token data organized by slice index (for visualization compatibility).

        Args:
            sample_id: Sample ID
            slice_indices: List of slice indices (0-based within sample)

        Returns:
            Dict mapping slice_index -> {'token_ids': list, 'token_entropies': list}
            Note: 'token_entropies' now contains POSITIVE entropy values (standard definition)
        """
        row_lo, row_hi = self.get_row_range_for_sample(sample_id)
        trp = self.rows_token_row_ptr()
        token_ids_arr = self.token_ids()
        entropy_arr = self.tok_entropy()  # Use positive entropy for visualization

        result = {}
        for slice_idx in slice_indices:
            abs_row_idx = row_lo + int(slice_idx)
            if abs_row_idx >= row_hi:
                # Slice index out of range for this sample
                continue

            # Get token range for this slice
            tok_start = int(trp[abs_row_idx])
            tok_end = int(trp[abs_row_idx + 1])

            # Extract token data for this slice
            slice_token_ids = token_ids_arr[tok_start:tok_end].tolist()
            slice_entropy = entropy_arr[tok_start:tok_end].tolist()

            result[slice_idx] = {
                'token_ids': slice_token_ids,
                'token_entropies': slice_entropy  # Positive entropy (standard definition)
            }

        return result

    def get_max_entropy_slice_index_for_sample(self, sample_id: int, smooth: int = 1) -> int:
        """
        Return the slice index with the maximum (optionally smoothed) entropy sum for the sample.

        Now uses argmax since get_slice_entropy_sum_for_sample returns positive entropy.
        """
        arr = self.get_slice_entropy_sum_for_sample(sample_id)
        if smooth and smooth > 1 and arr.size > smooth:
            # Simple moving average
            kernel = np.ones((smooth,), dtype=np.float32) / smooth
            arr = np.convolve(arr, kernel, mode="same")
        # Use argmax to find maximum positive entropy
        return int(np.argmax(arr))

# ----------------------------
# Compatibility helpers
# ----------------------------

def detect_nad_next_cache(path: str | Path) -> Path:
    """
    Detect a valid NAD Next cache root given a file/dir path.
    """
    p = Path(path)
    if p.is_file():
        p = p.parent
    # Heuristic: look for meta.json or manifest.json
    for cand in [p, p.parent]:
        if (cand / "meta.json").exists() or (cand / "manifest.json").exists():
            return cand
    raise FileNotFoundError(f"未能在该路径发现 NAD Next 缓存: {path}")

def load_nad_next_index(cache_root_or_loader):
    """Backward-compatible import for ``nad.io.index.load_nad_next_index``."""
    from .index import load_nad_next_index as _load_index

    return _load_index(cache_root_or_loader)
