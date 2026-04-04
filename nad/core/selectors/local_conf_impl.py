"""
局部置信度特征提取 | Local confidence feature extraction.

从 tok_conf 时序中提取局部聚合特征，补充 DeepConf 的全局均值信号。
DeepConf 论文的增益来自局部聚合算子（tail / lowest-group / bottom-q）；
本模块实现这些算子作为 Extreme9 的输入特征。

Extracts local aggregation features from tok_conf time series to supplement
the global-mean signal of dc_r/dc_z.  Implements tail / LGC / bottom-q operators
from the DeepConf paper as Extreme9 input features.

特征质量方向 | Quality direction:
  所有 tok_conf 特征：越低 = 越自信 = 越好。
  All tok_conf features: lower = more confident = better.
  Exception: head_tail_gap = mean(head) - mean(tail); positive = tail more confident = better.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .base import Selector, SelectorContext
from .trajectory_impl import (
    DEFAULT_REFLECTION_THRESHOLD,
    _compute_trajectory_arrays,
    _extract_slice_keysets,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _least_grouped_strict(x: np.ndarray, w: int) -> float:
    """
    Sliding-window minimum group mean (mirrors DeepConf LGC).

    - If len(x) < w: returns mean(x).
    - Otherwise: computes all window means (stride 1, no padding), returns minimum.
    Returns inf for empty arrays.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return float("inf")
    if w is None or w <= 1:
        return float(np.mean(x))
    if n < w:
        return float(np.mean(x))
    c = np.cumsum(x)
    sums = c[w - 1:] - np.concatenate(([0.0], c[:-w]))
    means = sums / float(w)
    return float(np.min(means))


def _get_slice_token_boundaries(cache, run_id: int) -> Optional[np.ndarray]:
    """
    Return local (zero-based) token-boundary array of shape (n_slices+1,) for run_id.

    boundaries[s]   = local start index of slice s
    boundaries[s+1] = local end index of slice s (= start of slice s+1)
    boundaries[-1]  = total tokens for this run

    Returns None if rows_token_row_ptr is unavailable or run_id is out of range.
    """
    rows_srp = cache.rows_sample_row_ptr
    rows_trp = cache.rows_token_row_ptr
    if rows_srp is None or rows_trp is None:
        return None
    if run_id < 0 or run_id >= len(rows_srp) - 1:
        return None
    row_start = int(rows_srp[run_id])
    row_end = int(rows_srp[run_id + 1])
    if row_end <= row_start:
        return None
    offsets = np.asarray(rows_trp[row_start:row_end + 1], dtype=np.int64)
    base = int(offsets[0])
    return offsets - base


# ─────────────────────────────────────────────────────────────────────────────
#  Public feature extraction
# ─────────────────────────────────────────────────────────────────────────────

LOCAL_CONF_FEATURE_NAMES = [
    "tail_2k",
    "tail_q10",
    "lgc_512",
    "lgc_2k",
    "bottom_q10",
    "head_tail_gap",
    "last_event_tail_conf",
    "event_nonevent_gap",
]


def extract_local_conf_raw(
    context: SelectorContext,
    reflection_threshold: float = DEFAULT_REFLECTION_THRESHOLD,
) -> dict[str, np.ndarray]:
    """
    Compute per-run local confidence features for all runs in context.

    Parameters
    ----------
    context : SelectorContext
        Bound selector context with cache and run_ids.
    reflection_threshold : float
        Jaccard similarity threshold to classify a slice as a reflection event.

    Returns
    -------
    dict[str, np.ndarray]
        Keys: tail_2k, tail_q10, lgc_512, lgc_2k, bottom_q10,
              head_tail_gap, last_event_tail_conf, event_nonevent_gap.
        Each array has shape (n_runs,) and dtype float64.
        NaN entries indicate the feature could not be computed (imputed later).

    Quality direction
    -----------------
    tok_conf features: lower = more confident = better (use _rank01(-arr) in builder).
    head_tail_gap: higher = tail more confident = better (use _rank01(arr) in builder).
    event_nonevent_gap: lower = events more confident relative to non-events.
    last_event_tail_conf: lower = more confident after last reflection event.
    """
    n = len(context.run_ids)
    cache = context.cache

    results: dict[str, np.ndarray] = {
        name: np.full(n, np.nan, dtype=np.float64)
        for name in LOCAL_CONF_FEATURE_NAMES
    }

    for i, rid in enumerate(context.run_ids):
        try:
            tv = cache.get_token_view(int(rid))
        except Exception:
            continue
        if tv is None or tv.tok_conf is None:
            continue
        arr = np.asarray(tv.tok_conf, dtype=np.float64)
        T = arr.size
        if T == 0:
            continue

        # ── 1. tail_2k: mean of last min(2000, T) tokens ───────────────────
        tail_len = min(2000, T)
        results["tail_2k"][i] = float(np.mean(arr[-tail_len:]))

        # ── 2. tail_q10: mean of last 10% tokens ───────────────────────────
        q10_len = max(1, T // 10)
        results["tail_q10"][i] = float(np.mean(arr[-q10_len:]))

        # ── 3–4. lgc_512 / lgc_2k: least-grouped confidence ────────────────
        results["lgc_512"][i] = _least_grouped_strict(arr, 512)
        results["lgc_2k"][i] = _least_grouped_strict(arr, 2000)

        # ── 5. bottom_q10: 10th percentile ────────────────────────────────
        results["bottom_q10"][i] = float(np.percentile(arr, 10))

        # ── 6. head_tail_gap: mean(front 10%) − mean(tail 10%) ─────────────
        #   Positive value → tail is more confident (lower conf) → better.
        head_len = max(1, T // 10)
        head_mean = float(np.mean(arr[:head_len]))
        tail_mean = float(np.mean(arr[-head_len:]))
        results["head_tail_gap"][i] = head_mean - tail_mean

        # ── 7–8. Slice-based features (require rows_token_row_ptr) ──────────
        slice_boundaries = _get_slice_token_boundaries(cache, int(rid))
        slice_keysets = _extract_slice_keysets(cache, int(rid))

        if slice_boundaries is None or len(slice_keysets) < 2:
            continue

        _, _, refl_arr = _compute_trajectory_arrays(slice_keysets)
        n_slices = len(slice_keysets)
        n_bounds = len(slice_boundaries)  # = n_slices + 1

        # ── 8. event_nonevent_gap ──────────────────────────────────────────
        #   refl_arr[t] = reflection score for slice t+1 (0-indexed t).
        #   Slice s_idx is an "event" if s_idx >= 1 and refl_arr[s_idx-1] > threshold.
        event_parts: List[np.ndarray] = []
        nonevent_parts: List[np.ndarray] = []
        for s_idx in range(n_slices):
            if s_idx + 1 >= n_bounds:
                break
            lo = int(slice_boundaries[s_idx])
            hi = int(slice_boundaries[s_idx + 1])
            hi = min(hi, T)
            lo = min(lo, T)
            if hi <= lo:
                continue
            slice_tokens = arr[lo:hi]
            is_event = (
                s_idx >= 1
                and refl_arr.size >= s_idx
                and float(refl_arr[s_idx - 1]) > float(reflection_threshold)
            )
            if is_event:
                event_parts.append(slice_tokens)
            else:
                nonevent_parts.append(slice_tokens)

        if event_parts and nonevent_parts:
            event_mean = float(np.mean(np.concatenate(event_parts)))
            nonevent_mean = float(np.mean(np.concatenate(nonevent_parts)))
            results["event_nonevent_gap"][i] = event_mean - nonevent_mean

        # ── 7. last_event_tail_conf ────────────────────────────────────────
        #   Find last slice s_idx (>= 1) where refl_arr[s_idx-1] > threshold.
        #   Compute mean tok_conf for all tokens AFTER that slice.
        last_event_slice = -1
        for s_idx in range(n_slices - 1, 0, -1):
            if refl_arr.size >= s_idx and float(refl_arr[s_idx - 1]) > float(reflection_threshold):
                last_event_slice = s_idx
                break

        if last_event_slice >= 0 and (last_event_slice + 1) < n_bounds:
            post_start = int(slice_boundaries[last_event_slice + 1])
            post_start = min(post_start, T)
            if post_start < T:
                results["last_event_tail_conf"][i] = float(np.mean(arr[post_start:]))

    return results


ERROR_MASS_FEATURE_NAMES = [
    "instability_mass",
    "tail_variance",
    "event_pre_post_delta",
]


def extract_error_mass_raw(
    context: SelectorContext,
    reflection_threshold: float = DEFAULT_REFLECTION_THRESHOLD,
) -> dict[str, np.ndarray]:
    """
    Compute per-run error-mass and late-stage confidence features.

    Features
    --------
    instability_mass : float
        Fraction of tokens where tok_conf > mean + 0.5 * std.
        Lower = more stable = better.
    tail_variance : float
        np.var of the last max(1, T//10) tokens.
        Lower = more stable = better.
    event_pre_post_delta : float
        mean(2 slices before last event) - mean(after last event).
        Positive = confidence recovered after last reflection.
        Higher = better.

    Quality direction
    -----------------
    instability_mass  : lower = better → use _rank01(-arr) in feature builder
    tail_variance     : lower = better → use _rank01(-arr)
    event_pre_post_delta : higher = better → use _rank01(arr)
    """
    n = len(context.run_ids)
    cache = context.cache

    results: dict[str, np.ndarray] = {
        name: np.full(n, np.nan, dtype=np.float64)
        for name in ERROR_MASS_FEATURE_NAMES
    }
    # sensible defaults for runs where feature cannot be computed
    results["instability_mass"][:] = 0.5
    results["event_pre_post_delta"][:] = 0.0

    for i, rid in enumerate(context.run_ids):
        try:
            tv = cache.get_token_view(int(rid))
        except Exception:
            continue
        if tv is None or tv.tok_conf is None:
            continue
        arr = np.asarray(tv.tok_conf, dtype=np.float64)
        T = arr.size
        if T == 0:
            continue

        # ── instability_mass ──────────────────────────────────────────────
        mu = float(arr.mean())
        sigma = float(arr.std())
        if sigma > 0.0:
            results["instability_mass"][i] = float(np.mean(arr > mu + 0.5 * sigma))
        else:
            results["instability_mass"][i] = 0.0

        # ── tail_variance ─────────────────────────────────────────────────
        tail_len = max(1, T // 10)
        results["tail_variance"][i] = float(np.var(arr[-tail_len:]))

        # ── event_pre_post_delta ──────────────────────────────────────────
        slice_boundaries = _get_slice_token_boundaries(cache, int(rid))
        slice_keysets = _extract_slice_keysets(cache, int(rid))

        if slice_boundaries is None or len(slice_keysets) < 2:
            continue

        _, _, refl_arr = _compute_trajectory_arrays(slice_keysets)
        n_slices = len(slice_keysets)
        n_bounds = len(slice_boundaries)

        # find last reflection event slice (same logic as last_event_tail_conf)
        last_event_slice = -1
        for s_idx in range(n_slices - 1, 0, -1):
            if refl_arr.size >= s_idx and float(refl_arr[s_idx - 1]) > float(reflection_threshold):
                last_event_slice = s_idx
                break

        if last_event_slice < 0:
            continue

        # pre_window: 2 slices before last event
        pre_start_slice = max(0, last_event_slice - 2)
        pre_lo = int(slice_boundaries[pre_start_slice])
        pre_hi = int(slice_boundaries[last_event_slice])
        pre_lo = min(pre_lo, T)
        pre_hi = min(pre_hi, T)

        # post_window: tokens after last event slice
        if (last_event_slice + 1) >= n_bounds:
            continue
        post_start = int(slice_boundaries[last_event_slice + 1])
        post_start = min(post_start, T)

        if pre_hi <= pre_lo or post_start >= T:
            continue

        pre_mean = float(np.mean(arr[pre_lo:pre_hi]))
        post_mean = float(np.mean(arr[post_start:]))
        results["event_pre_post_delta"][i] = pre_mean - post_mean

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Zero-training baseline selector
# ─────────────────────────────────────────────────────────────────────────────

class LocalConfTailSelector(Selector):
    """
    Zero-training baseline: selects the run with the minimum tail_2k value.

    tail_2k = mean tok_conf of the last min(2000, T) tokens.
    Lower tok_conf = more confident = better.

    Purpose: validate that local confidence features carry additional signal
    beyond the global mean (dc_r).  If local_conf_tail > dc_r on blind eval,
    the local feature direction is confirmed.
    """

    name = "local_conf_tail"

    def __init__(
        self,
        reflection_threshold: float = DEFAULT_REFLECTION_THRESHOLD,
    ) -> None:
        self.reflection_threshold = float(reflection_threshold)
        self._context: Optional[SelectorContext] = None
        self._raw: Optional[dict[str, np.ndarray]] = None

    def bind(self, context: SelectorContext) -> None:
        self._context = context
        self._raw = extract_local_conf_raw(
            context,
            reflection_threshold=self.reflection_threshold,
        )

    def select(self, D: np.ndarray, run_stats: dict) -> int:
        assert self._raw is not None, "LocalConfTailSelector requires bind(context)"
        tail_2k = self._raw["tail_2k"]
        finite_mask = np.isfinite(tail_2k)
        if not finite_mask.any():
            return 0
        scores = np.where(finite_mask, tail_2k, np.inf)
        return int(np.argmin(scores))
