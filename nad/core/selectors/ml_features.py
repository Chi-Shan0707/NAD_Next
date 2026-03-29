"""
Feature extraction for ML-based selectors.

All features are group-normalized (z-score or rank within the problem group)
so the model generalises across problems of different difficulty.

Feature vector (12 dimensions, N_FEATURES):
  0  mean_dist_z    z-score of mean pairwise distance (lower dist → more central)
  1  mean_dist_r    rank [0,1] of mean distance   (1 = closest/most central)
  2  knn3_z         z-score of KNN-3 similarity
  3  knn3_r         rank [0,1] of KNN-3 similarity  (1 = highest)
  4  length_z       z-score of activation length
  5  length_r       rank [0,1] of activation length (1 = longest)
  6  dc_z           z-score of DeepConf quality score
  7  dc_r           rank [0,1] of DeepConf quality  (1 = most confident)
  8  copeland_z     z-score of Copeland win count
  9  copeland_r     rank [0,1] of Copeland wins     (1 = most wins)
 10  log_n          log(group_size), same for all runs in a group
 11  log_length     log(activation_length)
"""
from __future__ import annotations
import numpy as np
from typing import Optional

FEATURE_NAMES = [
    "mean_dist_z", "mean_dist_r",
    "knn3_z",      "knn3_r",
    "length_z",    "length_r",
    "dc_z",        "dc_r",
    "copeland_z",  "copeland_r",
    "log_n",       "log_length",
]
N_FEATURES = len(FEATURE_NAMES)


# ── internal helpers ──────────────────────────────────────────────────────────

def _zscore(x: np.ndarray) -> np.ndarray:
    std = float(x.std())
    if std < 1e-10:
        return np.zeros_like(x, dtype=np.float64)
    return (x - x.mean()) / std


def _rank01(x: np.ndarray) -> np.ndarray:
    """Ascending rank normalised to [0, 1].  Ties share average rank."""
    n = len(x)
    if n <= 1:
        return np.zeros(n, dtype=np.float64)
    order = x.argsort()
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    return ranks / max(n - 1, 1)


def _knn_scores(D: np.ndarray, k: int = 3) -> np.ndarray:
    n = D.shape[0]
    S = 1.0 - D
    scores = np.zeros(n)
    for i in range(n):
        sims = np.concatenate([S[i, :i], S[i, i+1:]])
        kk = min(k, len(sims))
        scores[i] = float(np.partition(sims, -kk)[-kk:].mean())
    return scores


def _copeland_wins(D: np.ndarray) -> np.ndarray:
    """
    Vectorised Copeland: for each pair (i,j), compare how many third-party
    runs k (k≠i, k≠j) are closer to i vs j.  Winner gets +1, tie +0.5.

    Uses O(n³) numpy broadcasting – fast for n ≤ 64.
    """
    n = D.shape[0]
    # diff[i,j,k] = D[i,k] - D[j,k]  →  <0 means i closer to k than j
    diff = D[:, np.newaxis, :] - D[np.newaxis, :, :]      # (n, n, n)

    i_beats = (diff < 0).astype(np.float32)               # 1 where i closer to k than j
    j_beats = (diff > 0).astype(np.float32)

    # Exclude k == i or k == j from the vote
    k_idx = np.arange(n)
    row_eq_k = (k_idx[:, np.newaxis, np.newaxis] ==
                k_idx[np.newaxis, np.newaxis, :])          # (n,1,n) broadcast
    col_eq_k = (k_idx[np.newaxis, :, np.newaxis] ==
                k_idx[np.newaxis, np.newaxis, :])          # (1,n,n) broadcast
    exclude = row_eq_k | col_eq_k                         # (n,n,n)

    i_beats[exclude] = 0.0
    j_beats[exclude] = 0.0

    count_i = i_beats.sum(axis=2)   # (n,n)  sum over k
    count_j = j_beats.sum(axis=2)   # (n,n)

    wins = np.zeros(n)
    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = count_i[i, j], count_j[i, j]
            if ci > cj:
                wins[i] += 1.0
            elif cj > ci:
                wins[j] += 1.0
            else:
                wins[i] += 0.5
                wins[j] += 0.5
    return wins


def _deepconf_quality(context) -> Optional[np.ndarray]:
    """Return DeepConf quality scores for all runs in context, or None."""
    if context is None:
        return None
    try:
        from .impl import DeepConfSelector
        cache = context.cache
        run_ids = context.run_ids
        n = len(run_ids)
        dc = DeepConfSelector(metric="tok_conf")
        dc.bind(context)
        q = np.full(n, np.nan, dtype=np.float64)
        for i, rid in enumerate(run_ids):
            tv = cache.get_token_view(int(rid))
            if tv is None:
                continue
            try:
                series = dc._get_token_metric(tv)
                if series.size > 0:
                    q[i] = float(dc._quality_from_conf(dc._aggregate(series)))
            except Exception:
                pass
        return q
    except Exception:
        return None


# ── public API ────────────────────────────────────────────────────────────────

def extract_run_features(
    D: np.ndarray,
    run_stats: dict,
    context=None,
    k_nn: int = 3,
) -> np.ndarray:
    """
    Extract a group-normalised feature matrix for one problem group.

    Parameters
    ----------
    D         : (n, n) pairwise Jaccard distance matrix
    run_stats : dict with key "lengths" (np.ndarray, shape (n,))
    context   : SelectorContext (optional, provides DeepConf access)
    k_nn      : number of nearest neighbours for KNN feature

    Returns
    -------
    features : (n, N_FEATURES) float64 array
    """
    n = D.shape[0]
    lengths = np.asarray(run_stats["lengths"], dtype=np.float64)

    # 1–2. Mean-distance (medoid-like score)
    diag = np.eye(n, dtype=bool)
    mean_d = np.where(diag, np.nan, D)
    mean_d = np.nanmean(mean_d, axis=1) if n > 1 else np.zeros(n)
    mean_dist_z = _zscore(mean_d)
    mean_dist_r = _rank01(-mean_d)            # lower distance → higher rank

    # 3–4. KNN similarity
    knn = _knn_scores(D, k_nn)
    knn_z = _zscore(knn)
    knn_r = _rank01(knn)

    # 5–6. Activation length
    len_z = _zscore(lengths)
    len_r = _rank01(lengths)

    # 7–8. DeepConf quality
    dc_raw = _deepconf_quality(context)
    if dc_raw is not None and np.isfinite(dc_raw).any():
        finite_mean = float(np.nanmean(dc_raw))
        dc_raw = np.where(np.isfinite(dc_raw), dc_raw, finite_mean)
        dc_z = _zscore(dc_raw)
        dc_r = _rank01(dc_raw)
    else:
        dc_z = np.zeros(n)
        dc_r = np.full(n, 0.5)

    # 9–10. Copeland wins
    cop = _copeland_wins(D)
    cop_z = _zscore(cop)
    cop_r = _rank01(cop)

    # 11–12. Context / absolute features
    log_n = np.full(n, np.log(max(n, 1)), dtype=np.float64)
    log_length = np.log(np.maximum(lengths, 1.0))

    return np.column_stack([
        mean_dist_z, mean_dist_r,
        knn_z,       knn_r,
        len_z,       len_r,
        dc_z,        dc_r,
        cop_z,       cop_r,
        log_n,       log_length,
    ]).astype(np.float64)
