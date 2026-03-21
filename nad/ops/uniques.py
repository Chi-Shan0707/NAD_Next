#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic cumulative-unique counter utilities operating on NAD row-bank (CSR-like) arrays.

This is a generic building block (not tied to any specific algorithm).
It computes the cumulative number of unique 'keys' seen as we sweep rows in order.
Optionally, if a per-row 'slice_id' ordering is provided, it will sort rows by that
order before counting.

Inputs (row-bank):
- rows_srp: sample row pointer (length n_samples+1), mapping run_id -> [row_start,row_end)
- rows_rp : row pointer into keys array (length n_rows+1)
- rows_keys: keys array (uint32), concatenated per row
- rows_slice_ids: optional per-row ids (e.g., token positions) to define a custom sweep order
- rows_trp: optional token row pointer; if token_axis='tokens', used to emit cumulative token counts

Returns:
- (tokens, counts): both 1-D int arrays of length n_rows for the given run.
"""

from __future__ import annotations
from typing import Optional, Tuple, Literal
import numpy as np

def extract_tokenwise_counts(
    run_id: int,
    rows_srp: np.ndarray,
    rows_rp: np.ndarray,
    rows_keys: np.ndarray,
    rows_slice_ids: Optional[np.ndarray],
    rows_trp: Optional[np.ndarray],
    *,
    token_axis: Literal['row', 'tokens'] = 'row',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized cumulative-unique counting with zero Python loops.

    Strategy:
      - Work on the single run's contiguous slice of rows/keys without extra copies.
      - If rows_slice_ids is provided, compute each key's (row_position) under the
        stable sort by slice_id, then lexsort by (key, pos), take first occurrence,
        bincount over pos -> cumsum.
      - If no reorder is needed, take the faster path using np.unique(..., return_index=True)
        to get first global index and map back to row positions via searchsorted.

    Returns:
      tokens : x-axis aligned with counts (row mode: sorted slice_ids or original row order)
      counts : cumulative unique count after each row
    """
    rs = int(rows_srp[run_id])
    re = int(rows_srp[run_id + 1])
    if re <= rs:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    base = int(rows_rp[rs])
    k0, k1 = base, int(rows_rp[re])
    if k1 <= k0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    # All keys for this sample as a contiguous view
    keys_slice = np.asarray(rows_keys[k0:k1], dtype=np.uint32, order='C')

    # Row-local row_ptr (0-based within this sample)
    row_ptr_local = (rows_rp[rs:re+1] - rows_rp[rs]).astype(np.int64, copy=False)
    n_rows = row_ptr_local.size - 1
    if n_rows <= 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    # Optional reorder by slice_id
    order = None
    slice_ids_sorted = None
    if rows_slice_ids is not None:
        slice_ids = np.asarray(rows_slice_ids[rs:re], dtype=np.int64)
        order = np.argsort(slice_ids, kind='mergesort')
        slice_ids_sorted = slice_ids[order]

    if order is None:
        # Fast path: original row order
        _, first_idx = np.unique(keys_slice, return_index=True)
        first_idx = first_idx.astype(np.int64, copy=False)
        # Map first-index to row index via row_ptr_local
        first_rows = np.searchsorted(row_ptr_local[1:], first_idx, side='right')
        add_unique = np.bincount(first_rows, minlength=n_rows).astype(np.int32, copy=False)
        counts = np.cumsum(add_unique, dtype=np.int32)

        # X-axis
        if token_axis == 'tokens' and rows_trp is not None:
            base_tok = int(rows_trp[rs])
            tokens = (rows_trp[rs+1:re+1] - base_tok).astype(np.int64, copy=False)
        else:
            tokens = np.arange(n_rows, dtype=np.int32)
        return tokens, counts

    # Reorder needed: compute each row's position under sorted slice_ids
    # Row-local row starts/ends
    row_starts = row_ptr_local[:-1]
    row_ends   = row_ptr_local[1:]

    # Positions for each row under the sorted order
    row_pos = np.empty(n_rows, dtype=np.int32)
    row_pos[order] = np.arange(n_rows, dtype=np.int32)

    # For each key occurrence, compute its row_pos
    seg_ids = np.repeat(np.arange(n_rows, dtype=np.int32), row_ends - row_starts)
    pos = row_pos[seg_ids]

    # Sort by (key, pos), take first in each key group
    keys_sorted_idx = np.lexsort((pos, keys_slice))
    pos_sorted = pos[keys_sorted_idx]
    keys_sorted = keys_slice[keys_sorted_idx]

    first_mask = np.ones_like(pos_sorted, dtype=bool)
    if first_mask.size > 1:
        first_mask[1:] = keys_sorted[1:] != keys_sorted[:-1]
    pos_first = pos_sorted[first_mask]
    add_unique = np.bincount(pos_first, minlength=n_rows).astype(np.int32, copy=False)
    counts = np.cumsum(add_unique, dtype=np.int32)

    # X-axis
    if token_axis == 'tokens' and rows_trp is not None:
        base_tok = int(rows_trp[rs])
        tokens = (rows_trp[rs+1:re+1] - base_tok).astype(np.int64, copy=False)
    else:
        tokens = slice_ids_sorted.astype(np.int32, copy=False)

    return tokens, counts
