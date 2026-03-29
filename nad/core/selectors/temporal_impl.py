"""
时序折扣切片选择器 | Temporal Discount Slice Selector.

将 token 序列按固定大小（slice_size）分段，从末尾（最近的 slice）往前
以指数折扣 γ^(2k) 加权，截断低于阈值 T 的远端 slice，
选出加权质量分数最高的 run。

Splits the token sequence into fixed-size slices, applies exponential
discounting γ^(2k) from the last slice backwards (k=0 = last),
drops slices whose weight falls below threshold T,
and selects the run with the highest weighted quality score.
"""
from __future__ import annotations

import math
import numpy as np
from .base import Selector


class TemporalSliceSelector(Selector):
    """
    时序折扣切片选择器。
    Temporal discount slice selector.

    对 run r 的 token 序列，按 slice_size 分成 S 段（最后一段可能不足 slice_size）。
    Splits the token sequence of run r into S slices of slice_size tokens each
    (the last slice may be shorter).

    质量分数公式 | Quality score formula:
        score(r) = Σ_{k=0}^{K-1} γ^(2k) · quality(r, slice_{S-1-k})

    其中 k=0 对应最后一段（最近），K 为满足 γ^(2k) ≥ T 的最大 k+1。
    where k=0 is the last (most recent) slice, and K is the number of slices
    satisfying γ^(2k) ≥ T  (i.e., k ≤ floor(log T / log γ²)).

    quality 方向 | Quality direction:
        - "tok_conf"        : quality = -mean（低 tok_conf → 更自信 → 高分）
                               quality = -mean (lower tok_conf = more confident = higher score)
        - "tok_neg_entropy" : quality = mean（更接近 0 → 更确定 → 高分）
                               quality = mean (closer to 0 = more certain = higher score)
        - "tok_selfcert"    : quality = mean（越大越好）
                               quality = mean (higher = better)

    参数 | Parameters
    ----------
    metric : str
        使用的 token 级别指标，默认 "tok_conf"。
        Token-level metric to use.  Default: "tok_conf".
    gamma : float
        折扣因子（0 < gamma ≤ 1）；gamma=1 时等权所有 slice，默认 0.9。
        Discount factor (0 < gamma ≤ 1); gamma=1 weights all slices equally.
        Default: 0.9.
    threshold : float
        权重截断阈值；γ^(2k) < threshold 的 slice 被丢弃，默认 0.01。
        Weight cutoff; slices with γ^(2k) < threshold are discarded.
        Default: 0.01.
    slice_size : int
        每段 token 数，默认 32。
        Number of tokens per slice.  Default: 32.
    """

    def __init__(
        self,
        metric: str = "tok_conf",
        gamma: float = 0.9,
        threshold: float = 0.01,
        slice_size: int = 32,
    ):
        self.metric     = str(metric).lower()
        self.gamma      = float(gamma)
        self.threshold  = float(threshold)
        self.slice_size = int(slice_size)
        self._context   = None

    # ------------------------------------------------------------------
    # bind / pipeline hook
    # ------------------------------------------------------------------

    def bind(self, context) -> None:
        """由 NAD pipeline 在调用 select() 前注入上下文。
        Called by NAD pipeline before select() to inject context."""
        self._context = context

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _get_metric_array(self, token_view) -> np.ndarray | None:
        """从 TokenView 提取对应 metric 的一维数组。
        Extract the 1-D metric array from a TokenView."""
        if token_view is None:
            return None
        if self.metric == "tok_conf":
            arr = token_view.tok_conf
        elif self.metric == "tok_neg_entropy":
            arr = token_view.tok_neg_entropy
        elif self.metric == "tok_selfcert":
            arr = token_view.tok_selfcert
        else:
            return None
        if arr is None or len(arr) == 0:
            return None
        return np.asarray(arr, dtype=np.float64)

    @staticmethod
    def _compute_k_limit(gamma: float, threshold: float, S: int) -> int:
        """计算满足 γ^(2k) ≥ threshold 的最大 slice 数 K。
        Compute K = max number of slices (from the end) with γ^(2k) ≥ threshold.

        γ^(2k) ≥ T  ⟺  k ≤ log(T) / log(γ²)  (since log(γ²) < 0 for γ < 1)
        """
        if gamma >= 1.0 or threshold <= 0.0:
            return S   # 全部包含 | include all
        if gamma <= 0.0:
            return 1   # 只有最后一段 | only last slice
        log_g2 = math.log(gamma ** 2)
        if log_g2 >= 0:
            return S
        max_k = int(math.floor(math.log(threshold) / log_g2))
        K = max_k + 1          # k goes 0..max_k → K slices
        return max(1, min(K, S))

    def _weighted_score(self, arr: np.ndarray) -> float:
        """对单个 run 的 metric 数组计算时序折扣质量分。
        Compute the temporally-discounted quality score for one run's metric array."""
        n = len(arr)
        # 分成 S 段 | split into S slices
        S = max(1, (n + self.slice_size - 1) // self.slice_size)

        # 每段均值 | per-slice mean
        slice_means = np.array([
            float(np.mean(arr[s * self.slice_size : (s + 1) * self.slice_size]))
            for s in range(S)
        ], dtype=np.float64)

        # 质量方向 | quality direction
        if self.metric == "tok_conf":
            quality = -slice_means   # 低 tok_conf → 更自信 | lower conf = more confident
        else:
            quality = slice_means    # tok_neg_entropy / tok_selfcert: 越高越好 | higher is better

        # 有效 slice 数 K | number of valid (above-threshold) slices
        K = self._compute_k_limit(self.gamma, self.threshold, S)

        # 加权求和 (k=0 = 最后一段) | weighted sum (k=0 = last slice)
        score = 0.0
        for k in range(K):
            w_k = self.gamma ** (2 * k)
            score += w_k * quality[S - 1 - k]
        return score

    # ------------------------------------------------------------------
    # Selector interface
    # ------------------------------------------------------------------

    def select(self, D: np.ndarray, run_stats: dict) -> int:
        """
        选出加权质量分最高的 run 的组内索引。
        Return the group-internal index of the run with the highest weighted quality score.

        参数 D（距离矩阵）不被本选择器使用；token 数据直接从 context.cache 读取。
        The distance matrix D is not used; token data is read from context.cache.
        """
        assert self._context is not None, (
            "TemporalSliceSelector.bind(context) was not called by pipeline"
        )

        cache   = self._context.cache
        run_ids = self._context.run_ids
        n       = len(run_ids)

        scores = np.full(n, -np.inf, dtype=np.float64)

        for i, rid in enumerate(run_ids):
            token_view = cache.get_token_view(int(rid))
            arr = self._get_metric_array(token_view)
            if arr is None:
                continue   # 保持 -inf | keep -inf score
            scores[i] = self._weighted_score(arr)

        # 全部无效时回退到第 0 个 | fallback to index 0 if all scores are -inf
        if not np.isfinite(scores).any():
            return 0

        return int(np.argmax(scores))
