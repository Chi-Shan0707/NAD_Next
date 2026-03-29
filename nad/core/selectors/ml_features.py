"""
ML 选择器特征提取 | Feature extraction for ML-based selectors.

所有特征均在组内归一化（z-score 或 rank），确保模型在不同难度的题目间具备泛化能力。
All features are group-normalized (z-score or rank within the problem group)
so the model generalises across problems of different difficulty.

特征向量（12 维） | Feature vector (12 dimensions, N_FEATURES):
  0  mean_dist_z    组内平均距离的 z-score（距离越小 → 越居中）
                    z-score of mean pairwise distance (lower dist → more central)
  1  mean_dist_r    平均距离的组内 rank [0,1]（1 = 最居中）
                    rank [0,1] of mean distance (1 = closest/most central)
  2  knn3_z         KNN-3 相似度的 z-score
                    z-score of KNN-3 similarity
  3  knn3_r         KNN-3 相似度的 rank [0,1]（1 = 最高）
                    rank [0,1] of KNN-3 similarity (1 = highest)
  4  length_z       激活长度的 z-score
                    z-score of activation length
  5  length_r       激活长度的 rank [0,1]（1 = 最长）
                    rank [0,1] of activation length (1 = longest)
  6  dc_z           DeepConf quality 分数的 z-score
                    z-score of DeepConf quality score
  7  dc_r           DeepConf quality 的 rank [0,1]（1 = 最自信）
                    rank [0,1] of DeepConf quality (1 = most confident)
  8  copeland_z     Copeland 胜场数的 z-score
                    z-score of Copeland win count
  9  copeland_r     Copeland 胜场数的 rank [0,1]（1 = 最多胜场）
                    rank [0,1] of Copeland wins (1 = most wins)
 10  log_n          log(组大小)，组内所有 run 相同
                    log(group_size), same for all runs in a group
 11  log_length     log(激活长度)，绝对尺度特征
                    log(activation_length), absolute scale feature
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


# ── 内部工具函数 | internal helpers ───────────────────────────────────────────

def _zscore(x: np.ndarray) -> np.ndarray:
    std = float(x.std())
    if std < 1e-10:
        return np.zeros_like(x, dtype=np.float64)
    return (x - x.mean()) / std


def _rank01(x: np.ndarray) -> np.ndarray:
    """升序 rank 归一化到 [0, 1]，并列者取平均 rank。
    Ascending rank normalised to [0, 1].  Ties share average rank."""
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
    向量化 Copeland 投票：对每一对 (i,j)，统计有多少第三方 k（k≠i,k≠j）离 i 更近 vs 离 j 更近。
    胜者 +1，平局各 +0.5。使用 numpy broadcasting，O(n³) 复杂度，n≤64 时速度足够快。

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
    """返回 context 中所有 run 的 DeepConf quality 分数，若无法获取则返回 None。
    Return DeepConf quality scores for all runs in context, or None."""
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


# ── 公开接口 | public API ─────────────────────────────────────────────────────

def extract_run_features(
    D: np.ndarray,
    run_stats: dict,
    context=None,
    k_nn: int = 3,
) -> np.ndarray:
    """
    为一个题目组内的所有 run 提取组内归一化特征矩阵。
    Extract a group-normalised feature matrix for one problem group.

    参数 | Parameters
    ----------
    D         : (n, n) 两两 Jaccard 距离矩阵 | pairwise Jaccard distance matrix
    run_stats : 包含 "lengths" 键的字典（np.ndarray, shape (n,)）
                dict with key "lengths" (np.ndarray, shape (n,))
    context   : SelectorContext（可选，用于读取 DeepConf 数据）
                SelectorContext (optional, provides DeepConf access)
    k_nn      : KNN 特征使用的近邻数 | number of nearest neighbours for KNN feature

    返回 | Returns
    -------
    features : (n, N_FEATURES) float64 数组 | float64 array
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
