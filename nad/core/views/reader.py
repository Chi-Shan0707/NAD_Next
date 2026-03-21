
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple, Optional, Tuple, List
import numpy as np
import os
from ..storage.cache_paths import CachePaths
from ..storage.binary_io import mmap_from_file

class Agg(str, Enum):
    MAX = "max"
    SUM = "sum"

class CutType(str, Enum):
    TOPK = "topk"
    MASS = "mass"

class Order(str, Enum):
    BY_KEY = "by_key"
    BY_WEIGHT = "by_weight"

@dataclass(frozen=True)
class CutSpec:
    type: CutType
    value: float  # k (int) or mass [0,1]; we store as float and cast when needed

@dataclass(frozen=True)
class ViewSpec:
    agg: Agg
    cut: CutSpec
    order: Order

class RunView(NamedTuple):
    keys: np.ndarray        # uint32; a small gathered array
    weights: Optional[np.ndarray]  # float16 or float32 if normalized

class TokenView(NamedTuple):
    """Token metadata for a single run (sample)"""
    token_ids: Optional[np.ndarray]         # int32 token IDs
    tok_conf: Optional[np.ndarray]          # float32 confidence scores
    tok_neg_entropy: Optional[np.ndarray]   # float32 negative entropy
    tok_gini: Optional[np.ndarray]          # float32 Gini coefficients
    tok_selfcert: Optional[np.ndarray]      # float32 self-certainty
    tok_logprob: Optional[np.ndarray]       # float32 log probabilities

class CacheReader:
    def __init__(self, cache_root: str):
        self.paths = CachePaths(cache_root)
        # Lazy open - neuron data
        self._row_ptr = None
        self._keys = None
        self._w_max = None
        self._w_sum = None
        self._perm_max = None
        self._perm_sum = None
        self._prefix_max = None
        self._prefix_sum = None
        # Lazy open - token metadata (optional, v4.0+)
        self._token_ids = None
        self._token_row_ptr = None
        self._tok_logprob = None
        self._tok_conf = None
        self._tok_neg_entropy = None
        self._tok_entropy_legacy = None  # Backward compatibility
        self._tok_gini = None
        self._tok_selfcert = None
        # Lazy open - rows/ bank (optional, v4.1+)
        self._rows_sample_row_ptr = None
        self._rows_row_ptr = None
        self._rows_keys = None
        self._rows_w_max = None
        self._rows_w_sum = None
        self._rows_slice_ids = None
        self._rows_token_row_ptr = None

    # -- Lazy arrays
    @property
    def row_ptr(self):
        if self._row_ptr is None:
            self._row_ptr = mmap_from_file(self.paths.row_ptr, np.int64)
        return self._row_ptr
    @property
    def keys(self):
        if self._keys is None:
            self._keys = mmap_from_file(self.paths.keys, np.uint32)
        return self._keys
    def weights_for(self, agg: Agg):
        if agg == Agg.MAX:
            if self._w_max is None:
                self._w_max = mmap_from_file(self.paths.w_max, np.float16)
            return self._w_max
        else:
            if self._w_sum is None:
                self._w_sum = mmap_from_file(self.paths.w_sum, np.float16)
            return self._w_sum
    def perm_for(self, agg: Agg):
        if agg == Agg.MAX:
            if self._perm_max is None:
                self._perm_max = mmap_from_file(self.paths.perm_max, np.int32)
            return self._perm_max
        else:
            if self._perm_sum is None:
                self._perm_sum = mmap_from_file(self.paths.perm_sum, np.int32)
            return self._perm_sum
    def prefix_for(self, agg: Agg):
        if agg == Agg.MAX:
            if self._prefix_max is None:
                self._prefix_max = mmap_from_file(self.paths.prefix_max, np.float16)
            return self._prefix_max
        else:
            if self._prefix_sum is None:
                self._prefix_sum = mmap_from_file(self.paths.prefix_sum, np.float16)
            return self._prefix_sum

    def num_runs(self) -> int:
        return len(self.row_ptr) - 1

    def get_run_view(self, run_id: int, spec: ViewSpec, normalize_l1: bool=False) -> RunView:
        rp = self.row_ptr
        start, end = rp[run_id], rp[run_id+1]
        if end - start == 0:
            return RunView(keys=np.empty((0,), dtype=np.uint32),
                           weights=np.empty((0,), dtype=np.float16))

        perm = self.perm_for(spec.agg)[start:end]
        prefix = self.prefix_for(spec.agg)[start:end].astype(np.float32, copy=False) # for search
        k = end - start

        # Decide r by topk/mass
        if spec.cut.type == CutType.TOPK:
            r = int(min(int(spec.cut.value), k))
        else:
            # mass in [0, 1]; find first idx where prefix >= mass
            mass = float(spec.cut.value)
            if mass <= 0: r = 0
            elif mass >= 1: r = k
            else:
                r = int(np.searchsorted(prefix, mass, side="left") + 1)  # prefix is monotonic
                r = min(r, k)

        # Take top-r by weight (perm already is descending)
        top_abs_idx = perm[:r].astype(np.int64, copy=False)

        # Order output
        if spec.order == Order.BY_KEY:
            # sort absolute indices (because base keys are ascending by index)
            top_abs_idx = np.sort(top_abs_idx, kind="mergesort")

        base_keys = self.keys
        base_w = self.weights_for(spec.agg)

        keys = base_keys[top_abs_idx]
        weights = base_w[top_abs_idx].astype(np.float32, copy=False)  # work in fp32 if normalize

        if normalize_l1 and weights.size > 0:
            s = float(weights.sum())
            if s > 0:
                weights = (weights / s).astype(np.float32, copy=False)

        # Ensure C-contiguous arrays with fixed dtypes for optimized distance computation
        keys = np.ascontiguousarray(keys.astype(np.int32, copy=False))
        weights = np.ascontiguousarray(weights.astype(np.float32, copy=False))
        return RunView(keys=keys, weights=weights)

    # -- Optional token metadata (v4.0+)
    @property
    def token_ids(self) -> Optional[np.ndarray]:
        """Token IDs. Returns None if not available."""
        if hasattr(self.paths, 'token_ids') and os.path.exists(self.paths.token_ids):
            if self._token_ids is None:
                self._token_ids = mmap_from_file(self.paths.token_ids, np.int32)
            return self._token_ids
        return None

    @property
    def token_row_ptr(self) -> Optional[np.ndarray]:
        """Token CSR row pointer. Returns None if not available."""
        if hasattr(self.paths, 'token_row_ptr') and os.path.exists(self.paths.token_row_ptr):
            if self._token_row_ptr is None:
                self._token_row_ptr = mmap_from_file(self.paths.token_row_ptr, np.int64)
            return self._token_row_ptr
        return None

    @property
    def tok_logprob(self) -> Optional[np.ndarray]:
        """Token log probabilities. Returns None if not available."""
        if hasattr(self.paths, 'tok_logprob') and os.path.exists(self.paths.tok_logprob):
            if self._tok_logprob is None:
                self._tok_logprob = mmap_from_file(self.paths.tok_logprob, np.float32)
            return self._tok_logprob
        return None

    @property
    def tok_conf(self) -> Optional[np.ndarray]:
        """Token confidence scores. Returns None if not available."""
        if hasattr(self.paths, 'tok_conf') and os.path.exists(self.paths.tok_conf):
            if self._tok_conf is None:
                self._tok_conf = mmap_from_file(self.paths.tok_conf, np.float32)
            return self._tok_conf
        return None

    @property
    def tok_neg_entropy(self) -> Optional[np.ndarray]:
        """Token negative entropy (primary, format.md). Falls back to legacy tok_entropy if not available."""
        # Try primary path first (tok_neg_entropy)
        if hasattr(self.paths, 'tok_neg_entropy') and os.path.exists(self.paths.tok_neg_entropy):
            if self._tok_neg_entropy is None:
                self._tok_neg_entropy = mmap_from_file(self.paths.tok_neg_entropy, np.float32)
            return self._tok_neg_entropy
        # Fallback to legacy tok_entropy
        if hasattr(self.paths, 'tok_entropy') and os.path.exists(self.paths.tok_entropy):
            if self._tok_entropy_legacy is None:
                self._tok_entropy_legacy = mmap_from_file(self.paths.tok_entropy, np.float32)
            return self._tok_entropy_legacy
        return None

    @property
    def tok_gini(self) -> Optional[np.ndarray]:
        """Token Gini coefficient. Returns None if not available."""
        if hasattr(self.paths, 'tok_gini') and os.path.exists(self.paths.tok_gini):
            if self._tok_gini is None:
                self._tok_gini = mmap_from_file(self.paths.tok_gini, np.float32)
            return self._tok_gini
        return None

    @property
    def tok_selfcert(self) -> Optional[np.ndarray]:
        """Token self-certainty. Returns None if not available."""
        if hasattr(self.paths, 'tok_selfcert') and os.path.exists(self.paths.tok_selfcert):
            if self._tok_selfcert is None:
                self._tok_selfcert = mmap_from_file(self.paths.tok_selfcert, np.float32)
            return self._tok_selfcert
        return None

    def get_token_view(self, run_id: int) -> TokenView:
        """
        Get token metadata for a specific run (sample).

        Args:
            run_id: The run/sample ID to retrieve token data for

        Returns:
            TokenView with sliced arrays for this run. Fields are None if not available.
        """
        rptr = self.token_row_ptr
        if rptr is None:
            # No token data available
            return TokenView(None, None, None, None, None, None)

        if run_id < 0 or run_id >= len(rptr) - 1:
            raise ValueError(f"run_id {run_id} out of range [0, {len(rptr)-2}]")

        a, b = int(rptr[run_id]), int(rptr[run_id + 1])

        def maybe_slice(arr):
            """Helper to slice array or return None if unavailable"""
            if arr is None:
                return None
            if a >= b:
                # Empty range, return empty array with correct dtype
                return np.empty((0,), dtype=arr.dtype)
            return arr[a:b]

        return TokenView(
            token_ids=maybe_slice(self.token_ids),
            tok_conf=maybe_slice(self.tok_conf),
            tok_neg_entropy=maybe_slice(self.tok_neg_entropy),
            tok_gini=maybe_slice(self.tok_gini),
            tok_selfcert=maybe_slice(self.tok_selfcert),
            tok_logprob=maybe_slice(self.tok_logprob)
        )

    # -- Row-CSR Bank properties (optional, v4.1+)
    @property
    def rows_sample_row_ptr(self) -> Optional[np.ndarray]:
        """Sample-level row pointer for rows/ bank. Returns None if not available."""
        if hasattr(self.paths, 'rows_sample_row_ptr') and os.path.exists(self.paths.rows_sample_row_ptr):
            if self._rows_sample_row_ptr is None:
                self._rows_sample_row_ptr = mmap_from_file(self.paths.rows_sample_row_ptr, np.int64)
            return self._rows_sample_row_ptr
        return None

    @property
    def rows_row_ptr(self) -> Optional[np.ndarray]:
        """Row-level CSR pointer for rows/ bank. Returns None if not available."""
        if hasattr(self.paths, 'rows_row_ptr') and os.path.exists(self.paths.rows_row_ptr):
            if self._rows_row_ptr is None:
                self._rows_row_ptr = mmap_from_file(self.paths.rows_row_ptr, np.int64)
            return self._rows_row_ptr
        return None

    @property
    def rows_keys(self) -> Optional[np.ndarray]:
        """Row-level keys for rows/ bank. Returns None if not available."""
        if hasattr(self.paths, 'rows_keys') and os.path.exists(self.paths.rows_keys):
            if self._rows_keys is None:
                self._rows_keys = mmap_from_file(self.paths.rows_keys, np.uint32)
            return self._rows_keys
        return None

    def rows_weights_for(self, agg: Agg) -> Optional[np.ndarray]:
        """Row-level weights for specified aggregation. Returns None if not available."""
        if agg == Agg.MAX:
            if hasattr(self.paths, 'rows_w_max') and os.path.exists(self.paths.rows_w_max):
                if self._rows_w_max is None:
                    self._rows_w_max = mmap_from_file(self.paths.rows_w_max, np.float16)
                return self._rows_w_max
        else:
            if hasattr(self.paths, 'rows_w_sum') and os.path.exists(self.paths.rows_w_sum):
                if self._rows_w_sum is None:
                    self._rows_w_sum = mmap_from_file(self.paths.rows_w_sum, np.float16)
                return self._rows_w_sum
        return None

    @property
    def rows_slice_ids(self) -> Optional[np.ndarray]:
        """Slice IDs for each row in rows/ bank. Returns None if not available."""
        if hasattr(self.paths, 'rows_slice_ids') and os.path.exists(self.paths.rows_slice_ids):
            if self._rows_slice_ids is None:
                self._rows_slice_ids = mmap_from_file(self.paths.rows_slice_ids, np.int32)
            return self._rows_slice_ids
        return None

    @property
    def rows_token_row_ptr(self) -> Optional[np.ndarray]:
        """Per-sample local token row pointer for rows/ bank. Returns None if not available."""
        if hasattr(self.paths, 'rows_token_row_ptr') and os.path.exists(self.paths.rows_token_row_ptr):
            if self._rows_token_row_ptr is None:
                self._rows_token_row_ptr = mmap_from_file(self.paths.rows_token_row_ptr, np.int64)
            return self._rows_token_row_ptr
        return None

    def get_window_view(
        self,
        run_id: int,
        pos_lo: int,
        pos_hi: int,
        pos_size: int,
        spec: ViewSpec,
        normalize_l1: bool = False
    ) -> RunView:
        """
        Get a window view for a specific run and position range.

        This method supports position-based window queries using the rows/ bank.
        If rows/ bank is not available, it falls back to full-query.

        Args:
            run_id: The run/sample ID
            pos_lo: Lower position bound (inclusive, in position units)
            pos_hi: Upper position bound (exclusive, in position units)
            pos_size: Position size (tokens per position, default 32)
            spec: ViewSpec defining aggregation, cut, and order
            normalize_l1: Whether to L1-normalize the weights

        Returns:
            RunView with keys and weights for the specified window
        """
        # Check if rows/ bank exists
        rows_srp = self.rows_sample_row_ptr
        rows_rp = self.rows_row_ptr
        rows_keys_arr = self.rows_keys
        rows_w_arr = self.rows_weights_for(spec.agg)
        rows_trp = self.rows_token_row_ptr

        if rows_srp is None or rows_rp is None or rows_keys_arr is None or rows_w_arr is None or rows_trp is None:
            # No rows/ bank, fall back to full query
            return self.get_run_view(run_id, spec, normalize_l1)

        # Convert position range to token range
        tok_lo = pos_lo * pos_size
        tok_hi = pos_hi * pos_size

        # Get row range for this sample
        row_start = int(rows_srp[run_id])
        row_end = int(rows_srp[run_id + 1])
        # Base (global cumsum) at the start of this sample
        base = int(rows_trp[row_start])

        if row_end == row_start:
            # No rows for this sample
            return RunView(keys=np.empty((0,), dtype=np.uint32),
                          weights=np.empty((0,), dtype=np.float16))

        # Find rows that overlap with the token window
        # Each row has a local token range within the sample
        # rows_trp[row_i] gives the cumulative token count up to row i (local, resets at sample boundary)

        # Collect all keys and weights from rows that overlap [tok_lo, tok_hi)
        all_keys = []
        all_weights = []

        for row_idx in range(row_start, row_end):
            # Local token range for this row (within the sample)
            row_tok_start = int(rows_trp[row_idx]) - base
            row_tok_end = int(rows_trp[row_idx + 1]) - base

            # Check if this row overlaps with [tok_lo, tok_hi)
            if row_tok_end <= tok_lo or row_tok_start >= tok_hi:
                continue  # No overlap

            # This row overlaps, extract its keys and weights
            key_start = int(rows_rp[row_idx])
            key_end = int(rows_rp[row_idx + 1])

            if key_end > key_start:
                all_keys.append(rows_keys_arr[key_start:key_end])
                all_weights.append(rows_w_arr[key_start:key_end])

        if not all_keys:
            # No data in window
            return RunView(keys=np.empty((0,), dtype=np.uint32),
                          weights=np.empty((0,), dtype=np.float16))

        # Concatenate all keys and weights
        keys_concat = np.concatenate(all_keys)
        weights_concat = np.concatenate(all_weights).astype(np.float32, copy=False)

        # Aggregate by key (merge duplicate keys)
        # Sort by key
        order = np.argsort(keys_concat, kind='mergesort')
        keys_sorted = keys_concat[order]
        weights_sorted = weights_concat[order]

        # Find unique keys and aggregate weights
        diff = np.empty_like(keys_sorted, dtype=bool)
        diff[0] = True
        diff[1:] = keys_sorted[1:] != keys_sorted[:-1]
        idx = np.nonzero(diff)[0]

        keys_unique = keys_sorted[idx]

        if spec.agg == Agg.MAX:
            weights_unique = np.maximum.reduceat(weights_sorted, idx).astype(np.float32, copy=False)
        else:  # SUM
            weights_unique = np.add.reduceat(weights_sorted, idx).astype(np.float32, copy=False)

        # Now apply cut (topk or mass)
        k_total = len(keys_unique)

        # Sort by weight (descending) to apply cut
        perm = np.argsort(-weights_unique, kind='mergesort')
        keys_by_weight = keys_unique[perm]
        weights_by_weight = weights_unique[perm]

        # Compute prefix (cumulative sum normalized)
        total_w = float(weights_by_weight.sum())
        if total_w > 0:
            prefix = np.cumsum(weights_by_weight) / total_w
        else:
            prefix = np.zeros_like(weights_by_weight)

        # Decide r by topk/mass
        if spec.cut.type == CutType.TOPK:
            r = int(min(int(spec.cut.value), k_total))
        else:
            # mass in [0, 1]
            mass = float(spec.cut.value)
            if mass <= 0:
                r = 0
            elif mass >= 1:
                r = k_total
            else:
                r = int(np.searchsorted(prefix, mass, side="left") + 1)
                r = min(r, k_total)

        # Take top-r
        keys_cut = keys_by_weight[:r]
        weights_cut = weights_by_weight[:r]

        # Order output
        if spec.order == Order.BY_KEY:
            # Sort by key (ascending)
            order_final = np.argsort(keys_cut, kind='mergesort')
            keys_final = keys_cut[order_final]
            weights_final = weights_cut[order_final]
        else:
            # Already sorted by weight (descending)
            keys_final = keys_cut
            weights_final = weights_cut

        if normalize_l1 and weights_final.size > 0:
            s = float(weights_final.sum())
            if s > 0:
                weights_final = (weights_final / s).astype(np.float32, copy=False)

        # Ensure C-contiguous arrays with fixed dtypes for optimized distance computation
        keys_final = np.ascontiguousarray(keys_final.astype(np.int32, copy=False))
        weights_final = np.ascontiguousarray(weights_final.astype(np.float32, copy=False))
        return RunView(keys=keys_final, weights=weights_final)
