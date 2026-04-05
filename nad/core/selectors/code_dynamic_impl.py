from __future__ import annotations

import math
from typing import Any, Iterable, Optional

import numpy as np

from .base import SelectorContext
from .ml_features import _rank01
from .trajectory_impl import (
    DEFAULT_REFLECTION_THRESHOLD,
    _extract_slice_keysets,
    _jaccard_sim,
)

CODE_DYNAMIC_FEATURE_NAMES = [
    "prefix_best_window_quality_r",
    "head_tail_gap_r",
    "reflection_density_r",
    "tail_variance_r",
    "post_reflection_recovery_r",
]

DEFAULT_CODE_DYNAMIC_WEIGHTS = {
    "prefix_best_window_quality": 0.32,
    "head_tail_gap": 0.16,
    "reflection_density": 0.10,
    "tail_variance": 0.16,
    "post_reflection_recovery": 0.26,
}

DEFAULT_CODE_DYNAMIC_REFLECTION_THRESHOLD = DEFAULT_REFLECTION_THRESHOLD
DEFAULT_CODE_DYNAMIC_REFLECTION_LOOKBACK = 16
CODE_DYNAMIC_WEIGHT_NAMES = list(DEFAULT_CODE_DYNAMIC_WEIGHTS.keys())


def _get_slice_token_boundaries(cache, run_id: int) -> Optional[np.ndarray]:
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


def prepare_code_dynamic_run_state(
    cache,
    run_id: int,
    token_view=None,
) -> dict[str, Any]:
    if token_view is None:
        token_view = cache.get_token_view(int(run_id))
    if token_view is not None and token_view.tok_conf is not None:
        tok_conf = np.asarray(token_view.tok_conf, dtype=np.float64)
    else:
        tok_conf = np.zeros(0, dtype=np.float64)
    return {
        "run_id": int(run_id),
        "tok_conf": tok_conf,
        "slice_keysets": _extract_slice_keysets(cache, int(run_id)),
        "boundaries": _get_slice_token_boundaries(cache, int(run_id)),
    }


def _sliding_window_min_mean(arr: np.ndarray, window_tokens: int) -> float:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    win = max(1, min(int(window_tokens), int(arr.size)))
    if arr.size <= win:
        return float(np.mean(arr))
    cumsum = np.cumsum(arr, dtype=np.float64)
    cumsum = np.insert(cumsum, 0, 0.0)
    window_means = (cumsum[win:] - cumsum[:-win]) / float(win)
    return float(np.min(window_means))


def _safe_feature_array(values: np.ndarray, *, lower_is_better: bool, fill: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(arr)
    if not valid.any():
        return np.full(arr.shape, 0.5, dtype=np.float64)
    if valid.sum() == 1:
        return np.full(arr.shape, 0.5, dtype=np.float64)
    median_val = float(np.median(arr[valid]))
    filled = np.where(valid, arr, median_val)
    ranked = _rank01(-filled if lower_is_better else filled)
    return np.asarray(ranked, dtype=np.float64)


def resolve_code_dynamic_weights(
    weights: dict[str, float] | None = None,
    *,
    disabled_features: Iterable[str] | None = None,
) -> dict[str, float]:
    use_weights = dict(DEFAULT_CODE_DYNAMIC_WEIGHTS)
    if weights:
        use_weights.update({str(k): float(v) for k, v in weights.items()})
    for feat_name in list(disabled_features or []):
        if feat_name not in use_weights:
            raise KeyError(f"Unknown code-dynamic feature name: {feat_name}")
        use_weights[feat_name] = 0.0
    return use_weights


def _compute_bounded_reflection_stats(
    slice_keysets: list[np.ndarray],
    *,
    reflection_threshold: float,
    lookback_slices: int,
) -> tuple[float, int]:
    """
    Compute reflection_count and last_event_slice with bounded non-adjacent history.

    This keeps the code-oriented selector practical on long coding traces while
    preserving the intended local "revisiting earlier structure" signal.
    """
    n_slices = len(slice_keysets)
    if n_slices <= 1:
        return 0.0, -1

    lookback = max(2, int(lookback_slices))
    reflection_count = 0.0
    last_event_slice = -1

    threshold = float(reflection_threshold)

    for cur_idx in range(1, n_slices):
        start_idx = max(0, cur_idx - lookback)
        if cur_idx - start_idx <= 1:
            continue

        cur_keys = slice_keysets[cur_idx]
        is_reflection = False
        for prev_idx in range(cur_idx - 2, start_idx - 1, -1):
            sim = _jaccard_sim(cur_keys, slice_keysets[prev_idx])
            if sim > threshold:
                is_reflection = True
                break

        if is_reflection:
            reflection_count += 1.0
            last_event_slice = cur_idx

    return reflection_count, last_event_slice


def extract_code_dynamic_raw_from_state(
    run_state: dict[str, Any],
    *,
    token_limit: int | None = None,
    slice_limit: int | None = None,
    reflection_threshold: float = DEFAULT_CODE_DYNAMIC_REFLECTION_THRESHOLD,
    reflection_lookback_slices: int = DEFAULT_CODE_DYNAMIC_REFLECTION_LOOKBACK,
    prefix_fraction: float = 0.20,
    prefix_window_tokens: int = 128,
) -> dict[str, float]:
    arr = np.asarray(run_state.get("tok_conf", np.zeros(0, dtype=np.float64)), dtype=np.float64)
    if token_limit is not None:
        arr = arr[: max(0, int(token_limit))]

    prefix_best_window_quality = float("nan")
    head_tail_gap = float("nan")
    tail_variance = float("nan")

    if arr.size > 0:
        prefix_end = max(1, min(arr.size, max(prefix_window_tokens, int(math.ceil(arr.size * float(prefix_fraction))))))
        prefix_arr = arr[:prefix_end]
        prefix_best_window_quality = _sliding_window_min_mean(prefix_arr, prefix_window_tokens)

        seg_len = max(1, arr.size // 10)
        head_mean = float(np.mean(arr[:seg_len]))
        tail_mean = float(np.mean(arr[-seg_len:]))
        head_tail_gap = head_mean - tail_mean
        tail_variance = float(np.var(arr[-seg_len:]))

    raw_slice_keysets = list(run_state.get("slice_keysets", []) or [])
    slice_keysets = raw_slice_keysets[: max(0, int(slice_limit))] if slice_limit is not None else raw_slice_keysets
    boundaries = run_state.get("boundaries")
    if boundaries is not None:
        boundaries = np.asarray(boundaries, dtype=np.int64)
    if boundaries is not None and slice_limit is not None:
        end = min(len(boundaries), max(1, int(slice_limit)) + 1)
        boundaries = np.asarray(boundaries[:end], dtype=np.int64)

    reflection_density = 0.0
    post_reflection_recovery = 0.0
    reflection_count = 0.0
    n_slices = len(slice_keysets)

    if n_slices > 1:
        reflection_count, last_event_slice = _compute_bounded_reflection_stats(
            slice_keysets,
            reflection_threshold=reflection_threshold,
            lookback_slices=reflection_lookback_slices,
        )
        reflection_density = reflection_count / math.log(max(n_slices, 2))

        if arr.size > 0 and boundaries is not None and len(boundaries) >= 2:
            if last_event_slice >= 1 and (last_event_slice + 1) < len(boundaries):
                pre_start_slice = max(0, last_event_slice - 1)
                pre_lo = int(boundaries[pre_start_slice])
                pre_hi = int(boundaries[last_event_slice + 1])
                post_lo = int(boundaries[last_event_slice + 1])
                pre_hi = min(pre_hi, arr.size)
                post_lo = min(post_lo, arr.size)
                if pre_hi > pre_lo and post_lo < arr.size:
                    pre_mean = float(np.mean(arr[pre_lo:pre_hi]))
                    post_mean = float(np.mean(arr[post_lo:]))
                    post_reflection_recovery = pre_mean - post_mean

    return {
        "prefix_best_window_quality": float(prefix_best_window_quality),
        "head_tail_gap": float(head_tail_gap),
        "reflection_density": float(reflection_density),
        "tail_variance": float(tail_variance),
        "post_reflection_recovery": float(post_reflection_recovery),
        "reflection_count": float(reflection_count),
        "num_slices": float(n_slices),
        "num_tokens": float(arr.size),
    }


def extract_code_dynamic_raw(
    cache,
    run_id: int,
    token_view=None,
    *,
    token_limit: int | None = None,
    slice_limit: int | None = None,
    reflection_threshold: float = DEFAULT_CODE_DYNAMIC_REFLECTION_THRESHOLD,
    reflection_lookback_slices: int = DEFAULT_CODE_DYNAMIC_REFLECTION_LOOKBACK,
    prefix_fraction: float = 0.20,
    prefix_window_tokens: int = 128,
) -> dict[str, float]:
    """
    Extract code-oriented dynamic structure features for one run or one observed prefix.

    Quality directions:
      - prefix_best_window_quality: lower is better
      - head_tail_gap: higher is better
      - reflection_density: lower is better
      - tail_variance: lower is better
      - post_reflection_recovery: higher is better
    """
    run_state = prepare_code_dynamic_run_state(
        cache,
        int(run_id),
        token_view=token_view,
    )
    return extract_code_dynamic_raw_from_state(
        run_state,
        token_limit=token_limit,
        slice_limit=slice_limit,
        reflection_threshold=reflection_threshold,
        reflection_lookback_slices=reflection_lookback_slices,
        prefix_fraction=prefix_fraction,
        prefix_window_tokens=prefix_window_tokens,
    )


def extract_code_dynamic_raw_matrix(
    context: SelectorContext,
    *,
    reflection_threshold: float = DEFAULT_CODE_DYNAMIC_REFLECTION_THRESHOLD,
    reflection_lookback_slices: int = DEFAULT_CODE_DYNAMIC_REFLECTION_LOOKBACK,
    prefix_fraction: float = 0.20,
    prefix_window_tokens: int = 128,
) -> dict[str, np.ndarray]:
    run_ids = context.run_ids
    n = len(run_ids)
    out = {
        "prefix_best_window_quality": np.full(n, np.nan, dtype=np.float64),
        "head_tail_gap": np.full(n, np.nan, dtype=np.float64),
        "reflection_density": np.full(n, np.nan, dtype=np.float64),
        "tail_variance": np.full(n, np.nan, dtype=np.float64),
        "post_reflection_recovery": np.full(n, np.nan, dtype=np.float64),
    }
    for i, run_id in enumerate(run_ids):
        tv = context.cache.get_token_view(int(run_id))
        raw = extract_code_dynamic_raw(
            context.cache,
            int(run_id),
            token_view=tv,
            reflection_threshold=reflection_threshold,
            reflection_lookback_slices=reflection_lookback_slices,
            prefix_fraction=prefix_fraction,
            prefix_window_tokens=prefix_window_tokens,
        )
        for key in out:
            out[key][i] = float(raw.get(key, np.nan))
    return out


def build_code_dynamic_rank_features(
    context: SelectorContext,
    *,
    reflection_threshold: float = DEFAULT_CODE_DYNAMIC_REFLECTION_THRESHOLD,
    reflection_lookback_slices: int = DEFAULT_CODE_DYNAMIC_REFLECTION_LOOKBACK,
    prefix_fraction: float = 0.20,
    prefix_window_tokens: int = 128,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    raw = extract_code_dynamic_raw_matrix(
        context,
        reflection_threshold=reflection_threshold,
        reflection_lookback_slices=reflection_lookback_slices,
        prefix_fraction=prefix_fraction,
        prefix_window_tokens=prefix_window_tokens,
    )
    cols = [
        _safe_feature_array(raw["prefix_best_window_quality"], lower_is_better=True, fill=0.0),
        _safe_feature_array(raw["head_tail_gap"], lower_is_better=False, fill=0.0),
        _safe_feature_array(raw["reflection_density"], lower_is_better=True, fill=0.0),
        _safe_feature_array(raw["tail_variance"], lower_is_better=True, fill=0.0),
        _safe_feature_array(raw["post_reflection_recovery"], lower_is_better=False, fill=0.0),
    ]
    return np.column_stack(cols).astype(np.float64), raw


def build_code_dynamic_rank_features_from_raw(
    raw: dict[str, np.ndarray],
) -> np.ndarray:
    cols = [
        _safe_feature_array(np.asarray(raw["prefix_best_window_quality"], dtype=np.float64), lower_is_better=True, fill=0.0),
        _safe_feature_array(np.asarray(raw["head_tail_gap"], dtype=np.float64), lower_is_better=False, fill=0.0),
        _safe_feature_array(np.asarray(raw["reflection_density"], dtype=np.float64), lower_is_better=True, fill=0.0),
        _safe_feature_array(np.asarray(raw["tail_variance"], dtype=np.float64), lower_is_better=True, fill=0.0),
        _safe_feature_array(np.asarray(raw["post_reflection_recovery"], dtype=np.float64), lower_is_better=False, fill=0.0),
    ]
    return np.column_stack(cols).astype(np.float64)


def compute_code_dynamic_primary_scores_from_raw(
    raw: dict[str, np.ndarray],
    *,
    weights: dict[str, float] | None = None,
    disabled_features: Iterable[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    feat = build_code_dynamic_rank_features_from_raw(raw)
    use_weights = resolve_code_dynamic_weights(
        weights,
        disabled_features=disabled_features,
    )
    scores = (
        use_weights["prefix_best_window_quality"] * feat[:, 0]
        + use_weights["head_tail_gap"] * feat[:, 1]
        + use_weights["reflection_density"] * feat[:, 2]
        + use_weights["tail_variance"] * feat[:, 3]
        + use_weights["post_reflection_recovery"] * feat[:, 4]
    )
    return np.asarray(scores, dtype=np.float64), feat


def compute_code_dynamic_primary_scores(
    context: SelectorContext,
    *,
    weights: dict[str, float] | None = None,
    disabled_features: Iterable[str] | None = None,
    reflection_threshold: float = DEFAULT_CODE_DYNAMIC_REFLECTION_THRESHOLD,
    reflection_lookback_slices: int = DEFAULT_CODE_DYNAMIC_REFLECTION_LOOKBACK,
    prefix_fraction: float = 0.20,
    prefix_window_tokens: int = 128,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    feat, raw = build_code_dynamic_rank_features(
        context,
        reflection_threshold=reflection_threshold,
        reflection_lookback_slices=reflection_lookback_slices,
        prefix_fraction=prefix_fraction,
        prefix_window_tokens=prefix_window_tokens,
    )
    scores, feat = compute_code_dynamic_primary_scores_from_raw(
        raw,
        weights=weights,
        disabled_features=disabled_features,
    )
    return np.asarray(scores, dtype=np.float64), feat, raw


def order_code_dynamic_group_indices(
    scores: np.ndarray,
    D: np.ndarray,
    *,
    run_ids: Iterable[int] | None = None,
    atol: float = 1e-9,
) -> np.ndarray:
    scores_arr = np.asarray(scores, dtype=np.float64)
    n = int(scores_arr.size)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if run_ids is None:
        run_id_arr = np.arange(n, dtype=np.int64)
    else:
        run_id_arr = np.asarray(list(run_ids), dtype=np.int64)
        if run_id_arr.size != n:
            raise ValueError("run_ids size must match score count")

    base_order = np.argsort(-scores_arr, kind="mergesort")
    final_order: list[int] = []
    start = 0
    while start < n:
        anchor_idx = int(base_order[start])
        anchor_score = float(scores_arr[anchor_idx])
        end = start + 1
        while end < n and np.isclose(
            float(scores_arr[int(base_order[end])]),
            anchor_score,
            atol=atol,
            rtol=0.0,
        ):
            end += 1

        block = np.asarray(base_order[start:end], dtype=np.int64)
        if block.size > 1:
            tie_dsums = np.asarray(D[block][:, block].sum(axis=1), dtype=np.float64)
            block_order = np.lexsort((run_id_arr[block], tie_dsums))
            block = block[block_order]
        final_order.extend(int(idx) for idx in block.tolist())
        start = end

    return np.asarray(final_order, dtype=np.int64)


def select_code_dynamic_best_index(
    scores: np.ndarray,
    D: np.ndarray,
    *,
    run_ids: Iterable[int] | None = None,
    atol: float = 1e-9,
) -> int:
    order = order_code_dynamic_group_indices(
        scores,
        D,
        run_ids=run_ids,
        atol=atol,
    )
    if order.size == 0:
        return 0
    return int(order[0])


def compute_code_dynamic_score_series(
    cache,
    run_id: int,
    token_view=None,
    *,
    positions: Iterable[float],
    reflection_threshold: float = DEFAULT_CODE_DYNAMIC_REFLECTION_THRESHOLD,
    reflection_lookback_slices: int = DEFAULT_CODE_DYNAMIC_REFLECTION_LOOKBACK,
    prefix_window_tokens: int = 128,
    weights: dict[str, float] | None = None,
) -> list[float]:
    if token_view is None:
        token_view = cache.get_token_view(int(run_id))
    if token_view is not None and token_view.tok_conf is not None:
        tok_conf = np.asarray(token_view.tok_conf, dtype=np.float64)
        n_tokens = int(tok_conf.size)
    else:
        tok_conf = np.zeros(0, dtype=np.float64)
        n_tokens = 0
    slice_keysets = _extract_slice_keysets(cache, int(run_id))
    n_slices = len(slice_keysets)

    use_weights = dict(DEFAULT_CODE_DYNAMIC_WEIGHTS)
    if weights:
        use_weights.update(weights)

    scores: list[float] = []
    for p in positions:
        token_limit = max(1, int(float(p) * n_tokens)) if n_tokens > 0 else 0
        slice_limit = max(1, int(float(p) * n_slices)) if n_slices > 0 else 0
        raw = extract_code_dynamic_raw(
            cache,
            int(run_id),
            token_view=token_view,
            token_limit=token_limit,
            slice_limit=slice_limit,
            reflection_threshold=reflection_threshold,
            reflection_lookback_slices=reflection_lookback_slices,
            prefix_fraction=1.0,
            prefix_window_tokens=prefix_window_tokens,
        )
        score = (
            -use_weights["prefix_best_window_quality"] * float(raw["prefix_best_window_quality"])
            + use_weights["head_tail_gap"] * float(raw["head_tail_gap"])
            - use_weights["reflection_density"] * float(raw["reflection_density"])
            - use_weights["tail_variance"] * float(raw["tail_variance"])
            + use_weights["post_reflection_recovery"] * float(raw["post_reflection_recovery"])
        )
        scores.append(float(np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)))
    return scores
