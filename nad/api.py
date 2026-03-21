#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-level convenience API for NAD.

This module re-exports common building blocks for users:
- open_cache(cache_root) -> CacheReader
- load_correctness_map(cache_root) -> Dict[int, bool]
- extract_tokenwise_counts(...) -> (tokens, counts)
"""
from __future__ import annotations
from typing import Dict
from .core.views.reader import CacheReader
from .ops.accuracy import load_correctness_map
from .ops.uniques import extract_tokenwise_counts

__all__ = ['open_cache', 'load_correctness_map', 'extract_tokenwise_counts']

def open_cache(cache_root: str) -> CacheReader:
    """
    Open a NAD cache for reading.

    Args:
        cache_root: Path to cache directory

    Returns:
        CacheReader instance with lazy-loaded arrays
    """
    return CacheReader(cache_root)
