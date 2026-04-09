"""
轨迹分析选择器 | Trajectory-based reasoning path analysis selectors.

基于神经元激活模式的时序轨迹来预测推理正确性。利用 rows/ 位置感知 CSR 库（v4.1+）
提供的逐切片（32 token）激活数据，分析激活模式随 token 位置的演变规律。

Predict reasoning correctness by analysing the temporal trajectory of neuron
activation patterns.  Uses the rows/ position-aware CSR bank (v4.1+) which
provides per-slice (32-token) activation data.

实验 7 — TrajectorySelector       轨迹结构选择器（连续性 / 反思 / 新颖度）
实验 8 — LayerStratifiedSelector  分层激活选择器（深层 vs 浅层激活分布）
实验 9 — TrajectoryFusionSelector 轨迹 + 置信度融合选择器（ML 融合）

Exp 7 — TrajectorySelector        trajectory structure (continuity / reflection / novelty)
Exp 8 — LayerStratifiedSelector   layer-wise activation distribution (deep vs shallow)
Exp 9 — TrajectoryFusionSelector  ML fusion of trajectory + layer + existing features
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

import numpy as np

from .base import Selector, SelectorContext


# ── 默认模型路径 | default model path ─────────────────────────────────────────
_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models" / "ml_selectors"
DEFAULT_REFLECTION_THRESHOLD = 0.30


# ============================================================================
#  辅助函数 | Helper functions
# ============================================================================

def _extract_slice_keysets(cache, run_id: int) -> List[np.ndarray]:
    """
    从 rows/ bank 中提取每个 32-token 切片的激活 key 集合。
    Extract sorted key arrays for each 32-token slice of a run from the rows/ bank.

    返回 | Returns
    -------
    list[np.ndarray]  每个元素是一个切片的 uint32 keys（已排序）；
                      若 rows/ bank 不可用，返回空列表。
                      Each element is a sorted uint32 key array for one slice;
                      returns empty list if rows/ bank is unavailable.
    """
    rows_srp = cache.rows_sample_row_ptr
    rows_rp = cache.rows_row_ptr
    rows_keys_arr = cache.rows_keys

    if rows_srp is None or rows_rp is None or rows_keys_arr is None:
        return []

    if run_id < 0 or run_id >= len(rows_srp) - 1:
        return []

    row_start = int(rows_srp[run_id])
    row_end = int(rows_srp[run_id + 1])

    slices = []
    for row_idx in range(row_start, row_end):
        key_start = int(rows_rp[row_idx])
        key_end = int(rows_rp[row_idx + 1])
        if key_end > key_start:
            keys = np.asarray(rows_keys_arr[key_start:key_end], dtype=np.uint32)
            if keys.size > 1 and np.any(keys[1:] < keys[:-1]):
                keys = np.sort(keys)
            slices.append(keys)
        else:
            slices.append(np.empty(0, dtype=np.uint32))
    return slices


def _jaccard_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    计算两个已排序 key 集合的 Jaccard 相似度。
    Jaccard similarity between two sorted uint32 key arrays.

    J(A,B) = |A ∩ B| / |A ∪ B|.  空集返回 0.0。
    Returns 0.0 for empty sets.
    """
    if a.size == 0 or b.size == 0:
        return 0.0
    # numpy intersect1d / union1d on sorted arrays
    inter = np.intersect1d(a, b, assume_unique=True).size
    union = a.size + b.size - inter
    return inter / union if union > 0 else 0.0


def _compute_trajectory_arrays(slice_keysets: List[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-slice continuity / novelty / reflection arrays."""
    S = len(slice_keysets)
    if S <= 1:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )

    continuity = []
    novelty = []
    reflection = []

    for t in range(1, S):
        c_t = _jaccard_sim(slice_keysets[t], slice_keysets[t - 1])
        continuity.append(c_t)

        max_sim_any = c_t
        max_sim_nonadj = 0.0
        for s in range(0, t - 1):
            sim = _jaccard_sim(slice_keysets[t], slice_keysets[s])
            if sim > max_sim_any:
                max_sim_any = sim
            if sim > max_sim_nonadj:
                max_sim_nonadj = sim

        novelty.append(1.0 - max_sim_any)
        reflection.append(max_sim_nonadj)

    return (
        np.asarray(continuity, dtype=np.float64),
        np.asarray(novelty, dtype=np.float64),
        np.asarray(reflection, dtype=np.float64),
    )


def _compute_trajectory_scores(
    slice_keysets: List[np.ndarray],
    reflection_threshold: float = DEFAULT_REFLECTION_THRESHOLD,
) -> dict:
    """
    根据逐切片 key 集合序列计算轨迹特征。
    Compute trajectory features from a sequence of per-slice key sets.
    """
    S = len(slice_keysets)
    if S <= 1:
        return {
            "mean_continuity": 0.0,
            "mean_novelty": 1.0,
            "max_reflection": 0.0,
            "reflection_count": 0,
            "late_convergence": 0.0,
        }

    cont_arr, nov_arr, refl_arr = _compute_trajectory_arrays(slice_keysets)

    split = max(1, int(len(cont_arr) * 0.75))
    early_cont = float(cont_arr[:split].mean()) if split > 0 else 0.0
    late_cont = float(cont_arr[split:].mean()) if split < len(cont_arr) else 0.0
    late_conv = 1.0 if late_cont > early_cont else 0.0

    return {
        "mean_continuity": float(cont_arr.mean()) if cont_arr.size > 0 else 0.0,
        "mean_novelty": float(nov_arr.mean()) if nov_arr.size > 0 else 1.0,
        "max_reflection": float(refl_arr.max()) if refl_arr.size > 0 else 0.0,
        "reflection_count": int((refl_arr > float(reflection_threshold)).sum()),
        "late_convergence": late_conv,
    }


def _compute_trajectory_scores_for_prefix_counts(
    slice_keysets: List[np.ndarray],
    prefix_counts: List[int],
    reflection_threshold: float = DEFAULT_REFLECTION_THRESHOLD,
) -> dict[int, dict[str, float]]:
    """Compute exact trajectory scores for multiple prefix lengths in one pass."""
    if not prefix_counts:
        return {}

    total_slices = len(slice_keysets)
    if total_slices <= 1:
        default_scores = {
            "mean_continuity": 0.0,
            "mean_novelty": 1.0,
            "max_reflection": 0.0,
            "reflection_count": 0,
            "late_convergence": 0.0,
        }
        return {
            int(k): dict(default_scores)
            for k in sorted({max(1, min(int(k), max(1, total_slices))) for k in prefix_counts})
        }

    cont_arr, nov_arr, refl_arr = _compute_trajectory_arrays(slice_keysets)
    cont_csum = np.concatenate(([0.0], np.cumsum(cont_arr, dtype=np.float64)))
    nov_csum = np.concatenate(([0.0], np.cumsum(nov_arr, dtype=np.float64)))
    refl_count_csum = np.concatenate(([0], np.cumsum(refl_arr > float(reflection_threshold), dtype=np.int32)))
    refl_prefix_max = np.maximum.accumulate(refl_arr) if refl_arr.size > 0 else np.zeros(0, dtype=np.float64)

    out: dict[int, dict[str, float]] = {}
    for prefix_count in sorted({max(1, min(int(k), total_slices)) for k in prefix_counts}):
        if prefix_count <= 1:
            out[int(prefix_count)] = {
                "mean_continuity": 0.0,
                "mean_novelty": 1.0,
                "max_reflection": 0.0,
                "reflection_count": 0,
                "late_convergence": 0.0,
            }
            continue

        n_steps = int(prefix_count - 1)
        split = max(1, int(n_steps * 0.75))
        early_sum = float(cont_csum[split] - cont_csum[0])
        late_sum = float(cont_csum[n_steps] - cont_csum[split])
        early_cont = early_sum / float(split)
        late_count = n_steps - split
        late_cont = late_sum / float(late_count) if late_count > 0 else 0.0

        out[int(prefix_count)] = {
            "mean_continuity": float(cont_csum[n_steps] / float(n_steps)),
            "mean_novelty": float(nov_csum[n_steps] / float(n_steps)),
            "max_reflection": float(refl_prefix_max[n_steps - 1]) if refl_prefix_max.size > 0 else 0.0,
            "reflection_count": int(refl_count_csum[n_steps]),
            "late_convergence": 1.0 if late_cont > early_cont else 0.0,
        }

    return out


def _slice_metric_means(cache, run_id: int) -> dict[str, np.ndarray]:
    """Mean token metrics for each raw slice row of a run."""
    empty = {name: np.zeros(0, dtype=np.float64) for name in ("entropy", "conf", "gini")}

    rows_srp = cache.rows_sample_row_ptr
    rows_trp = cache.rows_token_row_ptr
    if rows_srp is None or rows_trp is None:
        return empty

    token_view = cache.get_token_view(int(run_id))
    if token_view.token_ids is None:
        return empty

    row_start = int(rows_srp[run_id])
    row_end = int(rows_srp[run_id + 1])
    if row_end <= row_start:
        return empty

    offsets = np.asarray(rows_trp[row_start:row_end + 1], dtype=np.int64)
    base = int(offsets[0])
    offsets = offsets - base
    n_tokens = len(token_view.token_ids)
    offsets = np.clip(offsets, 0, n_tokens)

    metric_arrays = {
        "entropy": None if token_view.tok_neg_entropy is None else -np.asarray(token_view.tok_neg_entropy, dtype=np.float64),
        "conf": None if token_view.tok_conf is None else np.asarray(token_view.tok_conf, dtype=np.float64),
        "gini": None if token_view.tok_gini is None else np.asarray(token_view.tok_gini, dtype=np.float64),
    }

    out = {name: [] for name in metric_arrays}
    for i in range(len(offsets) - 1):
        lo = int(offsets[i])
        hi = int(offsets[i + 1])
        for name, arr in metric_arrays.items():
            if arr is None or hi <= lo or lo >= len(arr):
                out[name].append(np.nan)
                continue
            seg = arr[lo:min(hi, len(arr))]
            if seg.size == 0 or np.all(np.isnan(seg)):
                out[name].append(np.nan)
            else:
                out[name].append(float(np.nanmean(seg)))

    return {name: np.asarray(values, dtype=np.float64) for name, values in out.items()}


def _discrete_derivatives(values: np.ndarray) -> dict[str, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    d1 = np.full(arr.shape, np.nan, dtype=np.float64)
    d2 = np.full(arr.shape, np.nan, dtype=np.float64)

    if arr.size >= 2:
        d1[1:] = arr[1:] - arr[:-1]
    if arr.size >= 3:
        d2[2:] = d1[2:] - d1[1:-1]

    return {
        "avg": arr,
        "d1": d1,
        "d2": d2,
    }


def extract_run_dynamics(
    cache,
    run_id: int,
    reflection_threshold: float = DEFAULT_REFLECTION_THRESHOLD,
) -> dict:
    """Export raw slice-level trajectory and token-metric dynamics for one run."""
    slice_keysets = _extract_slice_keysets(cache, int(run_id))
    cont_arr, nov_arr, refl_arr = _compute_trajectory_arrays(slice_keysets)
    scores = _compute_trajectory_scores(
        slice_keysets,
        reflection_threshold=reflection_threshold,
    )
    slice_metrics = _slice_metric_means(cache, int(run_id))
    derivatives = {name: _discrete_derivatives(values) for name, values in slice_metrics.items()}

    return {
        "num_slices": int(len(slice_keysets)),
        "reflection_threshold": float(reflection_threshold),
        "continuity_scores": cont_arr,
        "novelty_scores": nov_arr,
        "reflection_scores": refl_arr,
        "trajectory_scores": scores,
        "slice_metrics": slice_metrics,
        "derivatives": derivatives,
    }


def _extract_layer_features(keys: np.ndarray) -> dict:
    """
    从 neuron key 编码（layer<<16 | neuron_id）中提取分层激活特征。
    Extract layer-stratified features from neuron keys encoded as layer<<16|neuron_id.

    特征 | Features
    -------
    deep_shallow_ratio  深层（top 25%）vs 浅层（bottom 25%）激活比
                        ratio of deep-layer (top 25%) to shallow-layer (bottom 25%) activations
    layer_entropy       层激活分布的 Shannon 熵（归一化）
                        Shannon entropy of layer-wise activation distribution (normalised)
    layer_gini          层激活计数的 Gini 系数
                        Gini coefficient of layer-wise activation counts
    n_active_layers     激活层数
                        number of distinct active layers
    deep_frac           深层激活占总激活的比例
                        fraction of activations in deepest 25% of layers
    """
    if keys.size == 0:
        return {
            "deep_shallow_ratio": 1.0,
            "layer_entropy": 0.0,
            "layer_gini": 0.0,
            "n_active_layers": 0,
            "deep_frac": 0.0,
        }

    # Decode layer IDs: layer = key >> 16
    layers = np.right_shift(keys.astype(np.uint32), 16)
    unique_layers, counts = np.unique(layers, return_counts=True)
    n_layers = len(unique_layers)

    if n_layers == 0:
        return {
            "deep_shallow_ratio": 1.0,
            "layer_entropy": 0.0,
            "layer_gini": 0.0,
            "n_active_layers": 0,
            "deep_frac": 0.0,
        }

    # Deep vs shallow: define by quartile of active layer IDs
    layer_min = int(unique_layers.min())
    layer_max = int(unique_layers.max())
    layer_range = layer_max - layer_min + 1

    if layer_range <= 1:
        deep_shallow_ratio = 1.0
        deep_frac = 1.0
    else:
        q25 = layer_min + layer_range * 0.25
        q75 = layer_min + layer_range * 0.75
        shallow_count = int(counts[unique_layers < q25].sum()) if (unique_layers < q25).any() else 0
        deep_count = int(counts[unique_layers >= q75].sum()) if (unique_layers >= q75).any() else 0
        deep_shallow_ratio = (deep_count / shallow_count) if shallow_count > 0 else float(deep_count + 1)
        deep_frac = deep_count / keys.size

    # Layer entropy (normalised by log(n_layers))
    probs = counts.astype(np.float64) / counts.sum()
    raw_entropy = -float((probs * np.log(probs + 1e-30)).sum())
    max_entropy = math.log(n_layers) if n_layers > 1 else 1.0
    layer_entropy = raw_entropy / max_entropy

    # Gini coefficient of layer counts
    sorted_counts = np.sort(counts).astype(np.float64)
    n = len(sorted_counts)
    index = np.arange(1, n + 1, dtype=np.float64)
    gini = float((2 * (index * sorted_counts).sum() / sorted_counts.sum() - (n + 1)) / n) if n > 1 else 0.0
    gini = max(0.0, min(1.0, gini))

    return {
        "deep_shallow_ratio": deep_shallow_ratio,
        "layer_entropy": layer_entropy,
        "layer_gini": gini,
        "n_active_layers": n_layers,
        "deep_frac": deep_frac,
    }


# ── 轨迹特征提取（公开 API）| Trajectory feature extraction (public API) ─────

TRAJECTORY_FEATURE_NAMES = [
    # 轨迹结构 (Exp 7) | trajectory structure
    "mean_continuity",    # 平均连续性
    "mean_novelty",       # 平均新颖度
    "max_reflection",     # 最大反思回溯
    "reflection_count_r", # 反思次数 rank [0,1]
    "late_convergence",   # 末尾收敛
    # 分层激活 (Exp 8) | layer stratified
    "deep_shallow_ratio_z",  # 深浅比 z-score
    "layer_entropy",         # 层熵
    "layer_gini",            # 层 Gini
    "deep_frac_z",           # 深层占比 z-score
    "n_active_layers_z",     # 活跃层数 z-score
]
N_TRAJECTORY_FEATURES = len(TRAJECTORY_FEATURE_NAMES)


def _zscore(x: np.ndarray) -> np.ndarray:
    std = float(x.std())
    if std < 1e-10:
        return np.zeros_like(x, dtype=np.float64)
    return (x - x.mean()) / std


def _rank01(x: np.ndarray) -> np.ndarray:
    n = len(x)
    if n <= 1:
        return np.zeros(n, dtype=np.float64)
    order = x.argsort()
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    return ranks / max(n - 1, 1)


def extract_trajectory_features(
    context: SelectorContext,
    reflection_threshold: float = DEFAULT_REFLECTION_THRESHOLD,
) -> np.ndarray:
    """
    为一个题目组内的所有 run 提取轨迹 + 层特征矩阵。
    Extract trajectory + layer feature matrix for all runs in a problem group.

    参数 | Parameters
    ----------
    context : SelectorContext (must have cache, run_ids, views)

    返回 | Returns
    -------
    features : (n, N_TRAJECTORY_FEATURES) float64 数组 | float64 array
    """
    cache = context.cache
    run_ids = context.run_ids
    n = len(run_ids)

    # Raw per-run features
    raw_cont = np.zeros(n, dtype=np.float64)
    raw_nov = np.zeros(n, dtype=np.float64)
    raw_refl = np.zeros(n, dtype=np.float64)
    raw_refl_count = np.zeros(n, dtype=np.float64)
    raw_late_conv = np.zeros(n, dtype=np.float64)
    raw_ds_ratio = np.ones(n, dtype=np.float64)
    raw_lent = np.zeros(n, dtype=np.float64)
    raw_lgini = np.zeros(n, dtype=np.float64)
    raw_dfrac = np.zeros(n, dtype=np.float64)
    raw_nlayers = np.zeros(n, dtype=np.float64)

    for i, rid in enumerate(run_ids):
        # Trajectory features (from rows/ bank)
        slices = _extract_slice_keysets(cache, int(rid))
        if slices:
            traj = _compute_trajectory_scores(
                slices,
                reflection_threshold=reflection_threshold,
            )
            raw_cont[i] = traj["mean_continuity"]
            raw_nov[i] = traj["mean_novelty"]
            raw_refl[i] = traj["max_reflection"]
            raw_refl_count[i] = traj["reflection_count"]
            raw_late_conv[i] = traj["late_convergence"]

        # Layer features (from main keys)
        view = context.views[i] if i < len(context.views) else None
        if view is not None and hasattr(view, 'keys') and view.keys.size > 0:
            lf = _extract_layer_features(view.keys)
            raw_ds_ratio[i] = lf["deep_shallow_ratio"]
            raw_lent[i] = lf["layer_entropy"]
            raw_lgini[i] = lf["layer_gini"]
            raw_dfrac[i] = lf["deep_frac"]
            raw_nlayers[i] = lf["n_active_layers"]

    # Group-normalise
    return np.column_stack([
        raw_cont,                     # mean_continuity (already 0-1)
        raw_nov,                      # mean_novelty (already 0-1)
        raw_refl,                     # max_reflection (already 0-1)
        _rank01(raw_refl_count),      # reflection_count rank [0,1]
        raw_late_conv,                # late_convergence (0 or 1)
        _zscore(raw_ds_ratio),        # deep_shallow_ratio z-score
        raw_lent,                     # layer_entropy (already 0-1)
        raw_lgini,                    # layer_gini (already 0-1)
        _zscore(raw_dfrac),           # deep_frac z-score
        _zscore(raw_nlayers),         # n_active_layers z-score
    ]).astype(np.float64)


# ============================================================================
#  实验 7: TrajectorySelector 轨迹结构选择器
# ============================================================================

class TrajectorySelector(Selector):
    """
    实验 7: 基于轨迹结构的选择器。
    Exp 7: Select based on trajectory structure (continuity / reflection / novelty).

    假设 | Hypothesis:
    正确推理表现出连贯的"骨干"（高连续性）、适度的反思（bounded reflection）、
    和有限的探索（bounded novelty）。错误推理则过度漂移或反复循环。

    Correct reasoning exhibits a coherent backbone (high continuity), moderate
    reflection, and bounded exploration.  Incorrect reasoning drifts excessively
    or gets stuck in repetitive loops.

    评分 | Scoring:
    backbone_score = α·mean_continuity - β·mean_novelty + γ·late_convergence
                   + δ·bounded_reflection

    参数 | Parameters
    ----------
    alpha   : float (default 1.0)  连续性权重 | continuity weight
    beta    : float (default 0.5)  新颖度惩罚 | novelty penalty
    gamma   : float (default 0.3)  末尾收敛奖励 | late convergence bonus
    delta   : float (default 0.2)  适度反思奖励 | bounded reflection bonus
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.3,
        delta: float = 0.2,
        reflection_threshold: float = DEFAULT_REFLECTION_THRESHOLD,
    ):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.delta = float(delta)
        self.reflection_threshold = float(reflection_threshold)
        self._context: Optional[SelectorContext] = None

    def bind(self, context: SelectorContext) -> None:
        self._context = context

    def select(self, D: np.ndarray, run_stats: dict) -> int:
        n = D.shape[0]
        if n <= 1:
            return 0

        assert self._context is not None, "TrajectorySelector requires bind(context)"
        cache = self._context.cache
        run_ids = self._context.run_ids

        scores = np.full(n, -np.inf, dtype=np.float64)

        for i, rid in enumerate(run_ids):
            slices = _extract_slice_keysets(cache, int(rid))
            if not slices or len(slices) <= 1:
                scores[i] = 0.0
                continue

            traj = _compute_trajectory_scores(
                slices,
                reflection_threshold=self.reflection_threshold,
            )
            cont = traj["mean_continuity"]
            nov = traj["mean_novelty"]
            late = traj["late_convergence"]
            refl = traj["max_reflection"]

            # Bounded reflection: reward moderate reflection (0.1–0.5), penalise extremes
            bounded_refl = max(0.0, min(refl, 0.5) - max(refl - 0.5, 0.0))

            scores[i] = (
                self.alpha * cont
                - self.beta * nov
                + self.gamma * late
                + self.delta * bounded_refl
            )

        if not np.isfinite(scores).any():
            return 0

        # Tie-breaking with medoid (smallest row-sum in D)
        best = float(np.max(scores))
        ties = np.where(np.isclose(scores, best))[0]
        if ties.size == 1:
            return int(ties[0])
        tie_dsums = D[ties][:, ties].sum(axis=1)
        return int(ties[np.argmin(tie_dsums)])


# ============================================================================
#  实验 8: LayerStratifiedSelector 分层激活选择器
# ============================================================================

class LayerStratifiedSelector(Selector):
    """
    实验 8: 基于分层激活分布的选择器。
    Exp 8: Select based on layer-wise activation distribution.

    假设 | Hypothesis:
    正确推理更一致地激活深层（transformer 后层），表现为更高的 deep-to-shallow
    比率和更均匀的层分布（高熵、低 Gini）。

    Correct reasoning activates deeper layers more consistently, showing higher
    deep-to-shallow ratio and more uniform layer spread (high entropy, low Gini).

    评分 | Scoring:
    layer_score = α·deep_frac_z + β·layer_entropy - γ·layer_gini

    参数 | Parameters
    ----------
    alpha : float (default 1.0)  深层激活权重
    beta  : float (default 0.5)  层熵奖励
    gamma : float (default 0.3)  层集中度惩罚
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.3,
    ):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self._context: Optional[SelectorContext] = None

    def bind(self, context: SelectorContext) -> None:
        self._context = context

    def select(self, D: np.ndarray, run_stats: dict) -> int:
        n = D.shape[0]
        if n <= 1:
            return 0

        assert self._context is not None, "LayerStratifiedSelector requires bind(context)"
        views = self._context.views

        # Compute raw features for all runs
        raw_dfrac = np.zeros(n, dtype=np.float64)
        raw_lent = np.zeros(n, dtype=np.float64)
        raw_lgini = np.zeros(n, dtype=np.float64)

        for i in range(n):
            view = views[i] if i < len(views) else None
            if view is not None and hasattr(view, 'keys') and view.keys.size > 0:
                lf = _extract_layer_features(view.keys)
                raw_dfrac[i] = lf["deep_frac"]
                raw_lent[i] = lf["layer_entropy"]
                raw_lgini[i] = lf["layer_gini"]

        # Z-score normalise deep_frac within group
        dfrac_z = _zscore(raw_dfrac)

        scores = (
            self.alpha * dfrac_z
            + self.beta * raw_lent
            - self.gamma * raw_lgini
        )

        if not np.isfinite(scores).any():
            return 0

        # Tie-breaking with medoid
        best = float(np.max(scores))
        ties = np.where(np.isclose(scores, best))[0]
        if ties.size == 1:
            return int(ties[0])
        tie_dsums = D[ties][:, ties].sum(axis=1)
        return int(ties[np.argmin(tie_dsums)])


# ============================================================================
#  实验 9: TrajectoryFusionSelector 轨迹-置信度融合选择器
# ============================================================================

def _load_model(path: Path):
    """加载 sklearn 模型 | Load sklearn model from pickle."""
    import joblib
    return joblib.load(path)


class TrajectoryFusionSelector(Selector):
    """
    实验 9: 轨迹 + 层 + 现有特征的 ML 融合选择器。
    Exp 9: ML fusion of trajectory + layer + existing features.

    假设 | Hypothesis:
    轨迹结构（激活动态）与 token 级置信度（DeepConf）捕获互补信号，
    融合后能超越任何单一信号源。

    Trajectory structure (activation dynamics) and token confidence (DeepConf)
    capture complementary signals; fusing them should outperform any single source.

    方法 | Method:
    拼接轨迹特征（10 维）+ 现有 ML 特征（12 维）= 22 维，
    用预训练的 LogisticRegression 预测 P(correct)。

    Concatenate trajectory features (10-D) + existing ML features (12-D) = 22-D,
    predict P(correct) with pre-trained LogisticRegression.

    参数 | Parameters
    ----------
    model_path : str | None  模型文件路径；默认 models/ml_selectors/trajectory_fusion.pkl
    """

    def __init__(
        self,
        model_path: str | None = None,
        reflection_threshold: float = DEFAULT_REFLECTION_THRESHOLD,
    ):
        self._model_path = Path(model_path) if model_path else _DEFAULT_MODEL_DIR / "trajectory_fusion.pkl"
        self.reflection_threshold = float(reflection_threshold)
        self._model = None
        self._context: Optional[SelectorContext] = None

    def bind(self, context: SelectorContext) -> None:
        self._context = context

    def _get_model(self):
        if self._model is None:
            self._model = _load_model(self._model_path)
        return self._model

    def select(self, D: np.ndarray, run_stats: dict) -> int:
        n = D.shape[0]
        if n <= 1:
            return 0

        assert self._context is not None, "TrajectoryFusionSelector requires bind(context)"

        # Extract existing 12-D features
        from .ml_features import extract_run_features
        base_feat = extract_run_features(D, run_stats, context=self._context)  # (n, 12)

        # Extract trajectory 10-D features
        traj_feat = extract_trajectory_features(
            self._context,
            reflection_threshold=self.reflection_threshold,
        )  # (n, 10)

        # Concatenate: (n, 22)
        feat = np.hstack([base_feat, traj_feat])

        # Predict
        model = self._get_model()
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(feat)[:, 1]
        else:
            scores = model.predict(feat)

        return int(np.argmax(scores))
