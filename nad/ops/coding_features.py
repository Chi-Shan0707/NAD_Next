"""Code-structural and sequence-level features for the coding domain (LCBv5).

Tier 1 features are derived from token IDs and token confidence arrays alone
(always available at train + test time, no tokenizer required).

All functions take a CacheReader + run_id and return dicts of float scalars.
"""
from __future__ import annotations

from typing import Optional
import numpy as np

from nad.core.views.reader import CacheReader, TokenView

# ── constants ─────────────────────────────────────────────────────────────────

#: Token budget at which generation is considered truncated.
MAX_TOKEN_THRESHOLD = 32000

TIER1_FEATURE_NAMES = [
    "response_token_count",
    "at_max_tokens",
    "trigram_repetition_rate",
    "unique_token_fraction",
    "tok_conf_early_vs_late_gap",
    "tok_conf_section_early",
    "tok_conf_section_mid",
    "tok_conf_section_late",
]

# ── helpers ───────────────────────────────────────────────────────────────────

def _section_mean(arr: np.ndarray, lo: float, hi: float) -> float:
    """Mean of arr[lo*n : hi*n]."""
    n = len(arr)
    if n == 0:
        return 0.0
    start = int(lo * n)
    end = max(start + 1, int(hi * n))
    end = min(end, n)
    return float(np.mean(arr[start:end]))


def _trigram_repetition_rate(token_ids: np.ndarray) -> float:
    """Fraction of distinct trigrams (by token ID) appearing ≥2 times."""
    n = len(token_ids)
    if n < 3:
        return 0.0
    counts: dict[tuple, int] = {}
    for i in range(n - 2):
        tri = (int(token_ids[i]), int(token_ids[i + 1]), int(token_ids[i + 2]))
        counts[tri] = counts.get(tri, 0) + 1
    if not counts:
        return 0.0
    repeated = sum(1 for v in counts.values() if v >= 2)
    return float(repeated) / float(len(counts))


# ── public API ────────────────────────────────────────────────────────────────

def extract_tier1_features(
    reader: CacheReader,
    run_id: int,
    max_token_threshold: int = MAX_TOKEN_THRESHOLD,
) -> dict[str, float]:
    """Extract Tier-1 code-structural features for one run.

    Returns a dict with keys matching TIER1_FEATURE_NAMES. All values are
    finite floats (defaulting to 0.0 if data unavailable).
    """
    tv: TokenView = reader.get_token_view(int(run_id))

    # Token count and ID features
    if tv.token_ids is not None and len(tv.token_ids) > 0:
        token_ids = np.asarray(tv.token_ids, dtype=np.int32)
        n = int(len(token_ids))
        at_max = 1.0 if n >= int(max_token_threshold) else 0.0
        unique_frac = float(len(np.unique(token_ids))) / float(n)
        trigram_rep = _trigram_repetition_rate(token_ids)
    else:
        n = 0
        at_max = 0.0
        unique_frac = 0.0
        trigram_rep = 0.0

    # Confidence-section features (derived from tok_conf)
    if tv.tok_conf is not None and len(tv.tok_conf) > 0:
        conf = np.asarray(tv.tok_conf, dtype=np.float64)
        early_vs_late = _section_mean(conf, 0.0, 0.10) - _section_mean(conf, 0.90, 1.0)
        sec_early = _section_mean(conf, 0.0, 1.0 / 3.0)
        sec_mid = _section_mean(conf, 1.0 / 3.0, 2.0 / 3.0)
        sec_late = _section_mean(conf, 2.0 / 3.0, 1.0)
    else:
        early_vs_late = 0.0
        sec_early = 0.0
        sec_mid = 0.0
        sec_late = 0.0

    return {
        "response_token_count": float(n),
        "at_max_tokens": at_max,
        "trigram_repetition_rate": trigram_rep,
        "unique_token_fraction": unique_frac,
        "tok_conf_early_vs_late_gap": early_vs_late,
        "tok_conf_section_early": sec_early,
        "tok_conf_section_mid": sec_mid,
        "tok_conf_section_late": sec_late,
    }


def extract_tier1_feature_matrix(
    reader: CacheReader,
    sample_ids: np.ndarray,
    max_token_threshold: int = MAX_TOKEN_THRESHOLD,
    verbose: bool = False,
) -> np.ndarray:
    """Extract Tier-1 features for an array of sample IDs.

    Returns ndarray of shape (n_samples, len(TIER1_FEATURE_NAMES)).
    """
    n = int(len(sample_ids))
    out = np.zeros((n, len(TIER1_FEATURE_NAMES)), dtype=np.float64)
    for i, sid in enumerate(sample_ids):
        feat = extract_tier1_features(reader, int(sid), max_token_threshold=max_token_threshold)
        for j, name in enumerate(TIER1_FEATURE_NAMES):
            out[i, j] = feat[name]
        if verbose and i > 0 and i % 1000 == 0:
            print(f"  extract_tier1_feature_matrix: {i}/{n}", flush=True)
    return out
