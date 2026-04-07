from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from .base import SelectorContext
from .ml_features import _rank01

SCIENCE_DYNAMIC_FEATURE_NAMES = [
    "prefix_conf_mean_r",
    "recency_conf_mean_r",
    "late_worst_window_r",
    "late_recovery_r",
]

DEFAULT_SCIENCE_DYNAMIC_WEIGHTS = {
    "prefix_conf_mean": 0.0,
    "recency_conf_mean": 1.0,
    "late_worst_window": 0.0,
    "late_recovery": 0.0,
}

SCIENCE_DYNAMIC_CANDIDATE_WEIGHTS = {
    "prefix_conf_mean": 0.16,
    "recency_conf_mean": 0.56,
    "late_worst_window": 0.16,
    "late_recovery": 0.12,
}

SCIENCE_DYNAMIC_WEIGHT_NAMES = list(DEFAULT_SCIENCE_DYNAMIC_WEIGHTS.keys())

DEFAULT_SCIENCE_PREFIX_FRACTION = 0.40
DEFAULT_SCIENCE_TAIL_FRACTION = 0.25
DEFAULT_SCIENCE_RECENCY_EXP = 0.30
DEFAULT_SCIENCE_WINDOW_TOKENS = 128


def prepare_science_dynamic_run_state(
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
    }


def _safe_feature_array(values: np.ndarray, *, lower_is_better: bool) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(arr)
    if not valid.any() or valid.sum() == 1:
        return np.full(arr.shape, 0.5, dtype=np.float64)
    median_val = float(np.median(arr[valid]))
    filled = np.where(valid, arr, median_val)
    ranked = _rank01(-filled if lower_is_better else filled)
    return np.asarray(ranked, dtype=np.float64)


def _exp_weighted_mean(arr: np.ndarray, strength: float) -> float:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    weights = np.exp(float(strength) * np.arange(arr.size, dtype=np.float64) / max(int(arr.size), 1))
    return float(np.average(arr, weights=weights))


def _sliding_window_means(arr: np.ndarray, window_tokens: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return np.zeros(0, dtype=np.float64)
    win = max(1, min(int(window_tokens), int(arr.size)))
    if arr.size <= win:
        return np.asarray([float(np.mean(arr))], dtype=np.float64)
    cumsum = np.cumsum(arr, dtype=np.float64)
    cumsum = np.insert(cumsum, 0, 0.0)
    return np.asarray((cumsum[win:] - cumsum[:-win]) / float(win), dtype=np.float64)


def resolve_science_dynamic_weights(
    weights: dict[str, float] | None = None,
    *,
    disabled_features: Iterable[str] | None = None,
) -> dict[str, float]:
    use_weights = dict(DEFAULT_SCIENCE_DYNAMIC_WEIGHTS)
    if weights:
        use_weights.update({str(k): float(v) for k, v in weights.items()})
    for feat_name in list(disabled_features or []):
        if feat_name not in use_weights:
            raise KeyError(f"Unknown science-dynamic feature name: {feat_name}")
        use_weights[feat_name] = 0.0
    return use_weights


def extract_science_dynamic_raw_from_state(
    run_state: dict[str, Any],
    *,
    prefix_fraction: float = DEFAULT_SCIENCE_PREFIX_FRACTION,
    tail_fraction: float = DEFAULT_SCIENCE_TAIL_FRACTION,
    recency_exp: float = DEFAULT_SCIENCE_RECENCY_EXP,
    window_tokens: int = DEFAULT_SCIENCE_WINDOW_TOKENS,
) -> dict[str, float]:
    arr = np.asarray(run_state.get("tok_conf", np.zeros(0, dtype=np.float64)), dtype=np.float64)
    n = int(arr.size)
    if n <= 0:
        return {
            "prefix_conf_mean": 0.0,
            "recency_conf_mean": 0.0,
            "late_worst_window": 0.0,
            "late_recovery": 0.0,
        }

    prefix_n = max(1, min(n, int(math.ceil(float(prefix_fraction) * n))))
    tail_n = max(1, min(n, int(math.ceil(float(tail_fraction) * n))))
    prefix = arr[:prefix_n]
    tail = arr[-tail_n:]
    tail_window_means = _sliding_window_means(tail, int(window_tokens))
    tail_win = max(1, min(int(window_tokens), int(tail.size)))
    final_window_mean = float(np.mean(tail[-tail_win:]))
    late_worst_window = float(np.min(tail_window_means)) if tail_window_means.size > 0 else final_window_mean
    late_recovery = float(final_window_mean - late_worst_window)

    return {
        "prefix_conf_mean": float(np.mean(prefix)),
        "recency_conf_mean": float(_exp_weighted_mean(arr, float(recency_exp))),
        "late_worst_window": float(late_worst_window),
        "late_recovery": float(late_recovery),
    }


def extract_science_dynamic_raw(
    cache,
    run_id: int,
    token_view=None,
    *,
    prefix_fraction: float = DEFAULT_SCIENCE_PREFIX_FRACTION,
    tail_fraction: float = DEFAULT_SCIENCE_TAIL_FRACTION,
    recency_exp: float = DEFAULT_SCIENCE_RECENCY_EXP,
    window_tokens: int = DEFAULT_SCIENCE_WINDOW_TOKENS,
) -> dict[str, float]:
    run_state = prepare_science_dynamic_run_state(
        cache,
        int(run_id),
        token_view=token_view,
    )
    return extract_science_dynamic_raw_from_state(
        run_state,
        prefix_fraction=prefix_fraction,
        tail_fraction=tail_fraction,
        recency_exp=recency_exp,
        window_tokens=window_tokens,
    )


def extract_science_dynamic_raw_matrix(
    context: SelectorContext,
    *,
    prefix_fraction: float = DEFAULT_SCIENCE_PREFIX_FRACTION,
    tail_fraction: float = DEFAULT_SCIENCE_TAIL_FRACTION,
    recency_exp: float = DEFAULT_SCIENCE_RECENCY_EXP,
    window_tokens: int = DEFAULT_SCIENCE_WINDOW_TOKENS,
) -> dict[str, np.ndarray]:
    n = len(context.run_ids)
    out = {
        "prefix_conf_mean": np.full(n, np.nan, dtype=np.float64),
        "recency_conf_mean": np.full(n, np.nan, dtype=np.float64),
        "late_worst_window": np.full(n, np.nan, dtype=np.float64),
        "late_recovery": np.full(n, np.nan, dtype=np.float64),
    }
    for idx, run_id in enumerate(context.run_ids):
        tv = context.cache.get_token_view(int(run_id))
        raw = extract_science_dynamic_raw(
            context.cache,
            int(run_id),
            token_view=tv,
            prefix_fraction=prefix_fraction,
            tail_fraction=tail_fraction,
            recency_exp=recency_exp,
            window_tokens=window_tokens,
        )
        for key in out:
            out[key][idx] = float(raw[key])
    return out


def build_science_dynamic_rank_features_from_raw(
    raw: dict[str, np.ndarray],
) -> np.ndarray:
    cols = [
        _safe_feature_array(np.asarray(raw["prefix_conf_mean"], dtype=np.float64), lower_is_better=False),
        _safe_feature_array(np.asarray(raw["recency_conf_mean"], dtype=np.float64), lower_is_better=False),
        _safe_feature_array(np.asarray(raw["late_worst_window"], dtype=np.float64), lower_is_better=False),
        _safe_feature_array(np.asarray(raw["late_recovery"], dtype=np.float64), lower_is_better=False),
    ]
    return np.column_stack(cols).astype(np.float64)


def compute_science_dynamic_primary_scores_from_raw(
    raw: dict[str, np.ndarray],
    *,
    weights: dict[str, float] | None = None,
    disabled_features: Iterable[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    feat = build_science_dynamic_rank_features_from_raw(raw)
    use_weights = resolve_science_dynamic_weights(
        weights,
        disabled_features=disabled_features,
    )
    scores = (
        use_weights["prefix_conf_mean"] * feat[:, 0]
        + use_weights["recency_conf_mean"] * feat[:, 1]
        + use_weights["late_worst_window"] * feat[:, 2]
        + use_weights["late_recovery"] * feat[:, 3]
    )
    return np.asarray(scores, dtype=np.float64), feat


def compute_science_dynamic_primary_scores(
    context: SelectorContext,
    *,
    weights: dict[str, float] | None = None,
    disabled_features: Iterable[str] | None = None,
    prefix_fraction: float = DEFAULT_SCIENCE_PREFIX_FRACTION,
    tail_fraction: float = DEFAULT_SCIENCE_TAIL_FRACTION,
    recency_exp: float = DEFAULT_SCIENCE_RECENCY_EXP,
    window_tokens: int = DEFAULT_SCIENCE_WINDOW_TOKENS,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    raw = extract_science_dynamic_raw_matrix(
        context,
        prefix_fraction=prefix_fraction,
        tail_fraction=tail_fraction,
        recency_exp=recency_exp,
        window_tokens=window_tokens,
    )
    scores, feat = compute_science_dynamic_primary_scores_from_raw(
        raw,
        weights=weights,
        disabled_features=disabled_features,
    )
    return np.asarray(scores, dtype=np.float64), feat, raw
