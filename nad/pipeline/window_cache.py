"""
Window Cache Management for NAD v4.1

This module implements lazy window cache construction. Window caches are built
on-demand from the rows/ bank and persisted for future use.

Window semantics: position-based, where 1 position = pos_size tokens (default 32)
"""

from __future__ import annotations
import os
import numpy as np
from typing import Tuple
from ..core.storage.cache_paths import CachePaths
from ..core.storage.binary_io import create_memmap, write_array_atomic, mmap_from_file


def ensure_window_cache(
    cache_root: str,
    pos_lo: int,
    pos_hi: int,
    pos_size: int = 32
) -> bool:
    """
    Ensure that a window cache exists for the specified position range.

    If the cache doesn't exist, builds it from the rows/ bank.
    If rows/ bank doesn't exist, returns False (caller should use fallback).

    Args:
        cache_root: Root directory of the cache
        pos_lo: Lower position bound (inclusive)
        pos_hi: Upper position bound (exclusive)
        pos_size: Position size (tokens per position, default 32)

    Returns:
        True if window cache was created/exists, False if rows/ bank is not available

    Raises:
        RuntimeError: If rows/ bank exists but window cache construction fails
    """
    # [DISABLED] Window cache 写入已屏蔽
    # 原因：CacheReader.get_window_view 未实现读取 window_cache/ 的逻辑，
    # 每次查询仍从 rows/ bank 硬算，保存的缓存从未被使用（死代码）。
    # 屏蔽写入以避免无意义的磁盘 IO 和空间占用。
    # TODO: 实现 CacheReader 读取 window cache 后，删除此 return 以启用缓存
    return True  # 假装缓存已就绪，调用方继续使用 rows/ 硬算

    # --- 以下为原有写入逻辑（保留但不执行）---
    paths = CachePaths(cache_root)

    # Check if window cache already exists
    window_dir = paths.window_cache_dir(pos_lo, pos_hi)
    if os.path.exists(window_dir):
        # Check if all required files exist
        required_files = [
            paths.window_row_ptr(pos_lo, pos_hi),
            paths.window_keys(pos_lo, pos_hi),
            paths.window_w(pos_lo, pos_hi, "max"),
            paths.window_w(pos_lo, pos_hi, "sum")
        ]
        if all(os.path.exists(f) for f in required_files):
            return True  # Cache exists and is complete

    # Check if rows/ bank exists
    if not os.path.exists(paths.rows_dir):
        return False  # No rows/ bank, caller should use fallback

    # Load rows/ bank data
    try:
        rows_srp = mmap_from_file(paths.rows_sample_row_ptr, np.int64)
        rows_rp = mmap_from_file(paths.rows_row_ptr, np.int64)
        rows_keys = mmap_from_file(paths.rows_keys, np.uint32)
        rows_w_max = mmap_from_file(paths.rows_w_max, np.float16)
        rows_w_sum = mmap_from_file(paths.rows_w_sum, np.float16)
        rows_trp = mmap_from_file(paths.rows_token_row_ptr, np.int64)
    except Exception as e:
        raise RuntimeError(f"Failed to load rows/ bank: {e}")

    # Convert position range to token range
    tok_lo = pos_lo * pos_size
    tok_hi = pos_hi * pos_size

    num_samples = len(rows_srp) - 1

    # Build window cache: for each sample, extract keys/weights in [tok_lo, tok_hi)
    sample_row_ptrs = np.zeros((num_samples + 1,), dtype=np.int64)
    all_keys = []
    all_w_max = []
    all_w_sum = []

    for sample_id in range(num_samples):
        row_start = int(rows_srp[sample_id])
        row_end = int(rows_srp[sample_id + 1])
        # Base (global cumsum) at the start of this sample
        base = int(rows_trp[row_start])

        if row_end == row_start:
            # No rows for this sample
            sample_row_ptrs[sample_id + 1] = sample_row_ptrs[sample_id]
            continue

        # Collect keys/weights from rows that overlap [tok_lo, tok_hi)
        sample_keys = []
        sample_w_max = []
        sample_w_sum = []

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
                sample_keys.append(rows_keys[key_start:key_end])
                sample_w_max.append(rows_w_max[key_start:key_end])
                sample_w_sum.append(rows_w_sum[key_start:key_end])

        if not sample_keys:
            # No data in window for this sample
            sample_row_ptrs[sample_id + 1] = sample_row_ptrs[sample_id]
            continue

        # Concatenate all keys and weights for this sample
        keys_concat = np.concatenate(sample_keys)
        w_max_concat = np.concatenate(sample_w_max).astype(np.float32, copy=False)
        w_sum_concat = np.concatenate(sample_w_sum).astype(np.float32, copy=False)

        # Aggregate by key (merge duplicate keys)
        # Sort by key
        order = np.argsort(keys_concat, kind='mergesort')
        keys_sorted = keys_concat[order]
        w_max_sorted = w_max_concat[order]
        w_sum_sorted = w_sum_concat[order]

        # Find unique keys and aggregate weights
        diff = np.empty_like(keys_sorted, dtype=bool)
        diff[0] = True
        diff[1:] = keys_sorted[1:] != keys_sorted[:-1]
        idx = np.nonzero(diff)[0]

        keys_unique = keys_sorted[idx]
        w_max_unique = np.maximum.reduceat(w_max_sorted, idx).astype(np.float32, copy=False)
        w_sum_unique = np.add.reduceat(w_sum_sorted, idx).astype(np.float32, copy=False)

        # Append to all_* lists
        all_keys.append(keys_unique)
        all_w_max.append(w_max_unique)
        all_w_sum.append(w_sum_unique)

        sample_row_ptrs[sample_id + 1] = sample_row_ptrs[sample_id] + len(keys_unique)

    # Create window cache directory
    os.makedirs(window_dir, exist_ok=True)

    # Write row_ptr
    write_array_atomic(paths.window_row_ptr(pos_lo, pos_hi), sample_row_ptrs)

    # Write keys and weights
    total_keys = int(sample_row_ptrs[-1])
    if total_keys > 0:
        keys_all = np.concatenate(all_keys).astype(np.uint32, copy=False)
        w_max_all = np.concatenate(all_w_max).astype(np.float16, copy=False)
        w_sum_all = np.concatenate(all_w_sum).astype(np.float16, copy=False)

        # Use create_memmap + write for atomic writes
        keys_mm = create_memmap(paths.window_keys(pos_lo, pos_hi), np.uint32, shape=(total_keys,))
        keys_mm[:] = keys_all
        del keys_mm

        w_max_mm = create_memmap(paths.window_w(pos_lo, pos_hi, "max"), np.float16, shape=(total_keys,))
        w_max_mm[:] = w_max_all
        del w_max_mm

        w_sum_mm = create_memmap(paths.window_w(pos_lo, pos_hi, "sum"), np.float16, shape=(total_keys,))
        w_sum_mm[:] = w_sum_all
        del w_sum_mm
    else:
        # Empty cache (no data in window)
        create_memmap(paths.window_keys(pos_lo, pos_hi), np.uint32, shape=(0,))
        create_memmap(paths.window_w(pos_lo, pos_hi, "max"), np.float16, shape=(0,))
        create_memmap(paths.window_w(pos_lo, pos_hi, "sum"), np.float16, shape=(0,))

    return True
