
from __future__ import annotations
import numpy as np
from .base import Selector
from typing import List, Tuple

class MinActivationSelector(Selector):
    def select(self, D: np.ndarray, run_stats):
        lengths = run_stats["lengths"]
        return int(np.argmin(lengths))

class MaxActivationSelector(Selector):
    def select(self, D: np.ndarray, run_stats):
        lengths = run_stats["lengths"]
        return int(np.argmax(lengths))

class MedoidSelector(Selector):
    """
    Medoid selector: choose run with minimum average distance to others
    (Legacy-compatible: uses mean instead of sum for consistency with old NAD)
    """
    def select(self, D: np.ndarray, run_stats):
        if D.shape[0] <= 1:
            return 0
        meanD = D.mean(axis=1)
        return int(np.argmin(meanD))

class KNNMedoidSelector(Selector):
    """
    KNN-Medoid selector: choose run with highest average similarity to k nearest neighbors
    (Legacy-compatible: uses similarity matrix and mean aggregation for consistency with old NAD)
    """
    def __init__(self, k: int = 3):
        self.k = k

    def select(self, D: np.ndarray, run_stats):
        """Select run using similarity-based k-nearest neighbors approach"""
        R = D.shape[0]
        if R <= 1:
            return 0

        k = max(1, min(self.k, R-1))

        # Convert distance matrix to similarity matrix (1 - D)
        S = 1.0 - D

        scores = np.zeros(R, dtype=np.float32)

        for i in range(R):
            # Remove self-similarity (exclude diagonal)
            sims = np.delete(S[i], i)

            # Get top-k highest similarities
            topk = np.partition(sims, -k)[-k:]

            # Compute mean similarity (not sum)
            scores[i] = float(topk.mean())

        # Select run with highest score
        best = int(np.argmax(scores))

        # Tie-breaking: use medoid among tied candidates
        ties = np.where(np.isclose(scores, scores[best]))[0]
        if ties.size > 1:
            meanD = D.mean(axis=1)
            best = int(ties[np.argmin(meanD[ties])])

        return best

class DBSCANMedoidSelector(Selector):
    """
    简化 DBSCAN：
      - 半径阈值 eps（距离矩阵 D 里 D<=eps 视为邻居）
      - min_samples：核心点阈值
      - 取"最大簇"的 medoid（行和最小的点）
      - 如无簇（全噪声），回退 Medoid
    """
    def __init__(self, eps = None, min_samples: int = 3):
        # eps: None or 'auto' -> adaptive per problem (30% quantile of off-diagonal D)
        if eps is None or (isinstance(eps, str) and str(eps).lower() == 'auto'):
            self.eps = None
        else:
            self.eps = float(eps)
        self.min_samples = int(min_samples)
    def select(self, D: np.ndarray, run_stats):
        n = D.shape[0]
        # 决定实际 eps（支持自适应）
        if self.eps is None:
            if n <= 1:
                eps_val = 0.0
            else:
                tri = D[np.triu_indices(n, k=1)]
                eps_val = float(np.quantile(tri, 0.30)) if tri.size else 0.0
        else:
            eps_val = float(self.eps)
        # 邻接矩阵（含自身）
        nbr = (D <= eps_val)
        # core points
        core = np.asarray(nbr.sum(axis=1) >= self.min_samples)
        visited = np.zeros(n, dtype=bool)
        clusters: List[np.ndarray] = []
        for i in range(n):
            if visited[i] or not core[i]:
                continue
            # BFS 从核心点扩展（core 点 + 其邻居）
            queue = [i]
            comp = []
            visited[i] = True
            while queue:
                u = queue.pop()
                comp.append(u)
                # 只把核心点的邻居继续扩展
                if core[u]:
                    for v in np.where(nbr[u])[0]:
                        if not visited[v]:
                            visited[v] = True
                            queue.append(int(v))
            if comp:
                clusters.append(np.asarray(comp, dtype=int))
        if not clusters:
            # fallback
            return MedoidSelector().select(D, run_stats)
        # 选择最大簇 (tie-breaking: 簇大小相同时选择包含最小索引的簇)
        clusters.sort(key=lambda c: (-c.size, c.min()))
        c = clusters[0]

        # 计算簇内每个点的距离和
        sums = D[c][:, c].sum(axis=1)
        min_sum = sums.min()

        # 找出所有最小距离和的候选
        candidates_in_cluster = np.where(np.abs(sums - min_sum) < 1e-10)[0]

        # Tie-breaking: 选择全局索引最小的
        global_candidates = c[candidates_in_cluster]
        return int(global_candidates[0])

class ConsensusMinSelector(Selector):
    """
    取 (knn-medoid, medoid, dbscan-medoid) 三者的候选集合，
    在候选中选长度最小的 run（min-activation）。
    """
    def __init__(self, k:int=3, eps:float=0.3, min_samples:int=3):
        self.k = k; self.eps = eps; self.min_samples = min_samples
    def select(self, D: np.ndarray, run_stats):
        cands = [
            KNNMedoidSelector(self.k).select(D, run_stats),
            MedoidSelector().select(D, run_stats),
            DBSCANMedoidSelector(self.eps, self.min_samples).select(D, run_stats),
        ]
        cands = np.unique(np.asarray(cands, dtype=int))
        lengths = run_stats["lengths"][cands]

        # Find minimum length
        min_length = lengths.min()

        # Find all candidates with minimum length (tie-breaking)
        min_indices = np.where(np.abs(lengths - min_length) < 1e-10)[0]

        # Tie-breaking: select the candidate with smallest run index
        # This ensures deterministic behavior
        return int(cands[min_indices[0]])

class ConsensusMaxSelector(Selector):
    """
    同上，但在候选里选长度最大的 run（max-activation）。
    """
    def __init__(self, k:int=3, eps:float=0.3, min_samples:int=3):
        self.k = k; self.eps = eps; self.min_samples = min_samples
    def select(self, D: np.ndarray, run_stats):
        cands = [
            KNNMedoidSelector(self.k).select(D, run_stats),
            MedoidSelector().select(D, run_stats),
            DBSCANMedoidSelector(self.eps, self.min_samples).select(D, run_stats),
        ]
        cands = np.unique(np.asarray(cands, dtype=int))
        lengths = run_stats["lengths"][cands]

        # Find maximum length
        max_length = lengths.max()

        # Find all candidates with maximum length (tie-breaking)
        max_indices = np.where(np.abs(lengths - max_length) < 1e-10)[0]

        # Tie-breaking: select the candidate with smallest run index
        # This ensures deterministic behavior
        return int(cands[max_indices[0]])


class BaselineSelector(Selector):
    """
    Baseline selector: Virtual selector for statistical baselines (avgN@, conN@)
    Always returns index 0 (first run), but actual accuracy is computed differently
    in accuracy.py based on the selector name pattern.
    """
    def select(self, D: np.ndarray, run_stats):
        # Return first run index (will be ignored in accuracy calculation)
        return 0


class DeepConfSelector(Selector):
    """
    DeepConf-style selector using token-level confidence cached in NAD v4.x.

    It reads per-run token arrays from CacheReader.get_token_view(run_id)
    and aggregates them into a single score per run, then picks the run
    with the highest *quality* score.

    Parameters
    ----------
    metric : str
        Which token metric to use. One of:
            - "tok_conf"        : DeepConf token confidence ( -mean(log p_topk) )  -> lower is better
            - "tok_selfcert"    : Self-certainty KL(p||U)                           -> higher is better
            - "tok_neg_entropy" : sum p log p                                       -> closer to 0 is better
        Default: "tok_conf".
    reduction : str
        How to aggregate token series for a run. One of:
            - "min_group" : min of a moving-average with window = group_size
            - "mean"      : simple mean across all response tokens
        Default: "min_group".
    group_size : int
        Window size for moving-average when reduction == "min_group". Default: 20.

    Notes
    -----
    - Missing token arrays fall back to empty; such runs get score = -inf and will never be selected.
    - Ties are broken by selecting the medoid among tied candidates (smallest row-sum in D).
    - Uses get_token_view() to maintain compatibility with NAD framework.
    - All computations use float64 for precision.
    """

    def __init__(self, metric: str = "tok_conf", reduction: str = "min_group", group_size: int = 20):
        self.metric = str(metric).lower()
        self.reduction = str(reduction).lower()
        self.group_size = int(group_size)
        self._context = None  # bound via bind()

    def bind(self, context):
        """Bind context from NAD pipeline (provides cache access)"""
        self._context = context

    @staticmethod
    def _least_grouped_strict(x: np.ndarray, w: int) -> float:
        """
        严格尾随滑窗（与 deepconf.utils.compute_least_grouped 对齐）：
          - 若 len(x) < w ：返回 mean(x)（单个标量）
          - 否则：按窗口 w、步长 1、无 padding 计算所有窗口均值，
                  返回这些均值的最小值
        """
        x = np.asarray(x, dtype=np.float64)
        n = x.size
        if n == 0:
            return float("inf")
        if w is None or w <= 1:
            return float(np.mean(x))
        if n < w:
            return float(np.mean(x))
        # 尾随窗口均值：cumsum 差分快速实现
        c = np.cumsum(x)
        sums = c[w-1:] - np.concatenate(([0.0], c[:-w]))
        means = sums / float(w)
        return float(np.min(means))

    def _aggregate(self, arr: np.ndarray) -> np.float64:
        """
        Aggregate token-level metrics into a single score.
        Returns inf/-inf for empty arrays (handled later in quality mapping).
        """
        if arr.size == 0:
            # Return special values that will be mapped to -inf quality later
            if self.metric == "tok_conf":
                return np.float64(np.inf)  # Higher conf value = lower quality
            else:
                return np.float64(-np.inf)  # Lower value = lower quality

        # 简单平均（与原版不冲突）：直接句级平均
        if self.reduction == "mean" or self.group_size is None or self.group_size <= 0:
            return np.float64(np.mean(arr))

        # 严格 least-grouped：与 deepconf 原版一致
        val = self._least_grouped_strict(arr, int(self.group_size))
        return np.float64(val)

    def _quality_from_conf(self, conf_val: np.float64) -> np.float64:
        """
        Map an aggregated 'conf' value to 'quality score' where higher is better.

        For DeepConf token confidence ( -mean(log p_topk) ), lower is better -> negate.
        For self-certainty KL(p||U), higher is better -> identity.
        For negative entropy sum p log p (<=0), closer to 0 is better -> identity (maximize).
        """
        if self.metric == "tok_conf":
            # Lower conf value means better quality -> negate
            return -np.float64(conf_val)
        # tok_selfcert and tok_neg_entropy: larger is better (closer to 0 for neg-entropy)
        return np.float64(conf_val)

    def _get_token_metric(self, token_view) -> np.ndarray:
        """
        Extract the appropriate token metric from token view.
        Strictly requires the specified metric to exist - no fallback.

        Raises
        ------
        ValueError
            If token_view is None, no metrics are available, or the requested metric is not found.
        """
        # Check if token_view is valid
        if token_view is None:
            raise ValueError(
                "Token view is None. Cache may not contain token-level data. "
                "Ensure cache was built with NAD v4+ and includes token collection."
            )

        # Detect available metrics
        available = []
        if token_view.tok_conf is not None:
            available.append("tok_conf")
        if token_view.tok_selfcert is not None:
            available.append("tok_selfcert")
        if token_view.tok_neg_entropy is not None:
            available.append("tok_neg_entropy")

        # Check if any metrics are available
        if not available:
            raise ValueError(
                "No token metrics available in cache. "
                "Ensure cache was built with NAD v4+ and includes token data."
            )

        # Strictly match the requested metric (no fallback)
        if self.metric == "tok_conf":
            if token_view.tok_conf is not None:
                return token_view.tok_conf
        elif self.metric == "tok_selfcert":
            if token_view.tok_selfcert is not None:
                return token_view.tok_selfcert
        elif self.metric == "tok_neg_entropy":
            if token_view.tok_neg_entropy is not None:
                return token_view.tok_neg_entropy

        # If we reach here, the requested metric is not available
        raise ValueError(
            f"Requested metric '{self.metric}' not found in cache. "
            f"Available metrics: {', '.join(available)}. "
            f"Please specify one of the available metrics or rebuild cache with the required metric."
        )

    def select(self, D: np.ndarray, run_stats):
        """
        Select the best run based on DeepConf-style token confidence scores.

        Returns group-internal index (0 to n-1), not global run_id.

        Raises
        ------
        ValueError
            If token data is missing or requested metric is not available.
        """
        assert self._context is not None, "DeepConfSelector.bind(context) was not called by pipeline"

        cache = self._context.cache
        run_ids = self._context.run_ids
        n_runs = len(run_ids)

        # Pre-allocate score array with -inf (worst possible score)
        scores = np.full(n_runs, -np.inf, dtype=np.float64)

        # Process each run to compute DeepConf score
        for i, rid in enumerate(run_ids):
            # Get token view using NAD framework method
            token_view = cache.get_token_view(int(rid))

            # Extract the appropriate metric series (will raise ValueError if not available)
            series = self._get_token_metric(token_view)

            # Check if series is empty
            if series.size == 0:
                raise ValueError(
                    f"Token metric '{self.metric}' is empty for run {rid}. "
                    f"Cache may be corrupted or incomplete."
                )

            # Aggregate token-level metrics into a single value
            conf_val = self._aggregate(series)

            # Convert to quality score (higher is better)
            scores[i] = self._quality_from_conf(conf_val)

        # Sanity check: ensure we have at least one valid score
        # (This should never happen due to earlier validations, but kept as safeguard)
        if not np.isfinite(scores).any():
            raise RuntimeError(
                "Internal error: All scores are -inf after validation. "
                "This indicates a bug in DeepConfSelector. Please report this issue."
            )

        # Find the best score
        best_score = np.float64(np.max(scores))

        # Handle ties: find all runs with the best score
        ties = np.where(np.isclose(scores, best_score))[0]

        if ties.size == 1:
            # No tie, return the best
            return int(ties[0])
        else:
            # Tie-breaking: use medoid among tied candidates
            # Compute distance sums only for tied candidates
            tie_distance_sums = D[ties][:, ties].sum(axis=1)
            best_tie_idx = int(np.argmin(tie_distance_sums))
            return int(ties[best_tie_idx])
