from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from .base import SelectorContext
from .code_dynamic_impl import (
    DEFAULT_CODE_DYNAMIC_REFLECTION_LOOKBACK,
    DEFAULT_CODE_DYNAMIC_REFLECTION_THRESHOLD,
    extract_code_dynamic_raw_from_state,
    order_code_dynamic_group_indices,
    prepare_code_dynamic_run_state,
)
from .ml_features import _rank01

CODE_V2_FEATURE_NAMES = [
    "prefix_best_window_quality_r",
    "head_tail_gap_r",
    "tail_variance_r",
    "post_reflection_recovery_r",
    "last_block_instability_r",
]

DEFAULT_CODE_V2_WEIGHTS = {
    "prefix_best_window_quality": 0.42,
    "head_tail_gap": 0.06,
    "tail_variance": 0.08,
    "post_reflection_recovery": 0.28,
    "last_block_instability": 0.16,
}

CODE_V2_WEIGHT_NAMES = list(DEFAULT_CODE_V2_WEIGHTS.keys())


def _safe_feature_array(values: np.ndarray, *, lower_is_better: bool) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(arr)
    if not valid.any() or valid.sum() == 1:
        return np.full(arr.shape, 0.5, dtype=np.float64)
    median_val = float(np.median(arr[valid]))
    filled = np.where(valid, arr, median_val)
    ranked = _rank01(-filled if lower_is_better else filled)
    return np.asarray(ranked, dtype=np.float64)


def resolve_code_v2_weights(
    weights: dict[str, float] | None = None,
    *,
    disabled_features: Iterable[str] | None = None,
) -> dict[str, float]:
    use_weights = dict(DEFAULT_CODE_V2_WEIGHTS)
    if weights:
        use_weights.update({str(k): float(v) for k, v in weights.items()})
    for feat_name in list(disabled_features or []):
        if feat_name not in use_weights:
            raise KeyError(f"Unknown code-v2 feature name: {feat_name}")
        use_weights[feat_name] = 0.0
    return use_weights


def _last_block_instability(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=np.float64)
    n = int(arr.size)
    if n <= 0:
        return 0.0
    tail_len = max(1, int(math.ceil(0.20 * n)))
    tail = arr[-tail_len:]
    tail_mean = float(np.mean(tail))
    tail_worst = float(np.min(tail))
    return float(np.var(tail) + max(0.0, tail_mean - tail_worst))


def extract_code_v2_raw_from_state(
    run_state: dict[str, Any],
    *,
    reflection_threshold: float = DEFAULT_CODE_DYNAMIC_REFLECTION_THRESHOLD,
    reflection_lookback_slices: int = DEFAULT_CODE_DYNAMIC_REFLECTION_LOOKBACK,
    prefix_fraction: float = 0.30,
    prefix_window_tokens: int = 128,
) -> dict[str, float]:
    base = extract_code_dynamic_raw_from_state(
        run_state,
        reflection_threshold=reflection_threshold,
        reflection_lookback_slices=reflection_lookback_slices,
        prefix_fraction=prefix_fraction,
        prefix_window_tokens=prefix_window_tokens,
    )
    arr = np.asarray(run_state.get("tok_conf", np.zeros(0, dtype=np.float64)), dtype=np.float64)
    return {
        "prefix_best_window_quality": float(base["prefix_best_window_quality"]),
        "head_tail_gap": float(base["head_tail_gap"]),
        "tail_variance": float(base["tail_variance"]),
        "post_reflection_recovery": float(base["post_reflection_recovery"]),
        "last_block_instability": float(_last_block_instability(arr)),
    }


def extract_code_v2_raw(
    cache,
    run_id: int,
    token_view=None,
    *,
    reflection_threshold: float = DEFAULT_CODE_DYNAMIC_REFLECTION_THRESHOLD,
    reflection_lookback_slices: int = DEFAULT_CODE_DYNAMIC_REFLECTION_LOOKBACK,
    prefix_fraction: float = 0.30,
    prefix_window_tokens: int = 128,
) -> dict[str, float]:
    run_state = prepare_code_dynamic_run_state(
        cache,
        int(run_id),
        token_view=token_view,
    )
    return extract_code_v2_raw_from_state(
        run_state,
        reflection_threshold=reflection_threshold,
        reflection_lookback_slices=reflection_lookback_slices,
        prefix_fraction=prefix_fraction,
        prefix_window_tokens=prefix_window_tokens,
    )


def extract_code_v2_raw_matrix(
    context: SelectorContext,
    *,
    reflection_threshold: float = DEFAULT_CODE_DYNAMIC_REFLECTION_THRESHOLD,
    reflection_lookback_slices: int = DEFAULT_CODE_DYNAMIC_REFLECTION_LOOKBACK,
    prefix_fraction: float = 0.30,
    prefix_window_tokens: int = 128,
) -> dict[str, np.ndarray]:
    n = len(context.run_ids)
    out = {
        "prefix_best_window_quality": np.full(n, np.nan, dtype=np.float64),
        "head_tail_gap": np.full(n, np.nan, dtype=np.float64),
        "tail_variance": np.full(n, np.nan, dtype=np.float64),
        "post_reflection_recovery": np.full(n, np.nan, dtype=np.float64),
        "last_block_instability": np.full(n, np.nan, dtype=np.float64),
    }
    for idx, run_id in enumerate(context.run_ids):
        tv = context.cache.get_token_view(int(run_id))
        raw = extract_code_v2_raw(
            context.cache,
            int(run_id),
            token_view=tv,
            reflection_threshold=reflection_threshold,
            reflection_lookback_slices=reflection_lookback_slices,
            prefix_fraction=prefix_fraction,
            prefix_window_tokens=prefix_window_tokens,
        )
        for key in out:
            out[key][idx] = float(raw[key])
    return out


def build_code_v2_rank_features_from_raw(
    raw: dict[str, np.ndarray],
) -> np.ndarray:
    cols = [
        _safe_feature_array(np.asarray(raw["prefix_best_window_quality"], dtype=np.float64), lower_is_better=True),
        _safe_feature_array(np.asarray(raw["head_tail_gap"], dtype=np.float64), lower_is_better=False),
        _safe_feature_array(np.asarray(raw["tail_variance"], dtype=np.float64), lower_is_better=True),
        _safe_feature_array(np.asarray(raw["post_reflection_recovery"], dtype=np.float64), lower_is_better=False),
        _safe_feature_array(np.asarray(raw["last_block_instability"], dtype=np.float64), lower_is_better=True),
    ]
    return np.column_stack(cols).astype(np.float64)


def compute_code_v2_primary_scores_from_raw(
    raw: dict[str, np.ndarray],
    *,
    weights: dict[str, float] | None = None,
    disabled_features: Iterable[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    feat = build_code_v2_rank_features_from_raw(raw)
    use_weights = resolve_code_v2_weights(
        weights,
        disabled_features=disabled_features,
    )
    scores = (
        use_weights["prefix_best_window_quality"] * feat[:, 0]
        + use_weights["head_tail_gap"] * feat[:, 1]
        + use_weights["tail_variance"] * feat[:, 2]
        + use_weights["post_reflection_recovery"] * feat[:, 3]
        + use_weights["last_block_instability"] * feat[:, 4]
    )
    return np.asarray(scores, dtype=np.float64), feat


def compute_code_v2_primary_scores(
    context: SelectorContext,
    *,
    weights: dict[str, float] | None = None,
    disabled_features: Iterable[str] | None = None,
    reflection_threshold: float = DEFAULT_CODE_DYNAMIC_REFLECTION_THRESHOLD,
    reflection_lookback_slices: int = DEFAULT_CODE_DYNAMIC_REFLECTION_LOOKBACK,
    prefix_fraction: float = 0.30,
    prefix_window_tokens: int = 128,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    raw = extract_code_v2_raw_matrix(
        context,
        reflection_threshold=reflection_threshold,
        reflection_lookback_slices=reflection_lookback_slices,
        prefix_fraction=prefix_fraction,
        prefix_window_tokens=prefix_window_tokens,
    )
    scores, feat = compute_code_v2_primary_scores_from_raw(
        raw,
        weights=weights,
        disabled_features=disabled_features,
    )
    return np.asarray(scores, dtype=np.float64), feat, raw


def select_code_v2_best_index(
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
