"""
Extreme9 selector — 11-dimensional local-confidence feature expansion.

Inherits the 3-feature Extreme8 core (dc_z, dc_r, reflection_count_r) and
augments with 8 local tok_conf aggregation features derived from the
DeepConf paper's tail / LGC / bottom-q operators.

Feature list (EXTREME9_FEATURE_NAMES, 11 dims):
  0  dc_z                  DeepConf quality z-score (same as Extreme8)
  1  dc_r                  DeepConf quality rank    (same as Extreme8)
  2  reflection_count_r    reflection count rank    (same as Extreme8)
  3  tail_2k_r             rank of mean tok_conf in last min(2000,T) tokens (↑ = more confident)
  4  tail_q10_r            rank of mean tok_conf in last 10% tokens
  5  lgc_512_r             rank of least-grouped-confidence, window=512
  6  lgc_2k_r              rank of LGC window=2000
  7  bottom_q10_r          rank of 10th-percentile tok_conf
  8  head_tail_gap_r       rank of (mean head 10%) − (mean tail 10%) (↑ = tail more confident)
  9  last_event_tail_conf_r rank of mean tok_conf after last reflection event slice
 10  event_nonevent_gap_r  rank of (event mean) − (non-event mean) tok_conf

Quality direction: for all tok_conf features, lower = more confident = better.
                   Rank features are computed as _rank01(-arr) so higher rank = better.
                   Exception: head_tail_gap_r uses _rank01(arr) (positive = tail confident).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .base import Selector, SelectorContext
from .extreme8_impl import (
    DEFAULT_BAND_SIZES,
    LinearRankModel,
    ZeroRankModel,
    _impute_finite,
    _load_model as _load_model_e8,
    accumulate_extreme8_scores,
    build_extreme8_features,
    extract_extreme8_raw_values,
    normalize_weight_direction,
    sample_tuple_indices,
    tuple_band_counts,
    compute_band_reward,
    band_reward_bounds,
    normalize_band_reward,
    compute_normalized_band_reward,
)
from .local_conf_impl import extract_local_conf_raw
from .ml_features import _rank01
from .trajectory_impl import DEFAULT_REFLECTION_THRESHOLD

_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models" / "ml_selectors"

EXTREME9_FEATURE_NAMES = [
    "dc_z",
    "dc_r",
    "reflection_count_r",
    "tail_2k_r",
    "tail_q10_r",
    "lgc_512_r",
    "lgc_2k_r",
    "bottom_q10_r",
    "head_tail_gap_r",
    "last_event_tail_conf_r",
    "event_nonevent_gap_r",
]
N_EXTREME9_FEATURES = len(EXTREME9_FEATURE_NAMES)


def _load_model(path: Path):
    try:
        import joblib
        return joblib.load(path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load Extreme9 selector model from {path}.\n"
            f"Run: python scripts/train_extreme9_selectors.py\nOriginal error: {exc}"
        )


@dataclass
class LinearRankModel9:
    """11-dimensional linear ranking model for Extreme9."""
    weights: np.ndarray
    bias: float = 0.0

    def __post_init__(self) -> None:
        arr = np.asarray(self.weights, dtype=np.float64).reshape(-1)
        if arr.size != N_EXTREME9_FEATURES:
            raise ValueError(
                f"LinearRankModel9 expects {N_EXTREME9_FEATURES} weights, got {arr.size}"
            )
        self.weights = arr
        self.bias = float(self.bias)

    def predict(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != self.weights.size:
            raise ValueError(
                f"Expected X shape (n, {self.weights.size}), got {arr.shape}"
            )
        return arr @ self.weights + self.bias


def extract_extreme9_raw_values(
    context: SelectorContext,
    reflection_threshold: float = DEFAULT_REFLECTION_THRESHOLD,
) -> dict[str, np.ndarray]:
    """
    Compute the full set of raw per-run values for the Extreme9 model.

    Merges Extreme8 raw values (dc_raw, reflection_count) with local
    confidence features (tail_2k, tail_q10, lgc_512, lgc_2k, bottom_q10,
    head_tail_gap, last_event_tail_conf, event_nonevent_gap).
    """
    e8_raw = extract_extreme8_raw_values(context, reflection_threshold=reflection_threshold)
    lc_raw = extract_local_conf_raw(context, reflection_threshold=reflection_threshold)
    return {**e8_raw, **lc_raw}


def build_extreme9_features(
    raw_values: dict[str, np.ndarray],
    indices: np.ndarray | list[int] | None = None,
) -> np.ndarray:
    """
    Build the 11-D Extreme9 feature matrix for the selected tuple indices.

    The first 3 columns are the Extreme8 features (dc_z, dc_r, reflection_count_r).
    The next 8 columns are rank-normalised local confidence features.

    For tok_conf features: lower = more confident = better → _rank01(-arr) gives
    higher rank to more confident runs.
    For head_tail_gap: positive = tail more confident → _rank01(arr).
    For event_nonevent_gap: lower event-minus-nonevent conf suggests events are
    more confident; _rank01(-arr) for consistency (direction learned by training).
    """
    # ── Extreme8 base features (3 dims) ─────────────────────────────────────
    e8_feat = build_extreme8_features(raw_values, indices)  # (n_sub, 3)

    n_total = len(raw_values["dc_raw"])
    if indices is None:
        idx = np.arange(n_total, dtype=np.int64)
    else:
        idx = np.asarray(indices, dtype=np.int64)

    # ── Local confidence rank features (8 dims) ──────────────────────────────
    # Features where lower tok_conf = better → _rank01(-arr) (higher rank = better)
    conf_lower_better = [
        "tail_2k", "tail_q10", "lgc_512", "lgc_2k",
        "bottom_q10", "last_event_tail_conf",
    ]
    # Features where higher value = better
    conf_higher_better = [
        "head_tail_gap", "event_nonevent_gap",
    ]

    rank_cols = []
    for name in conf_lower_better:
        raw = _impute_finite(raw_values.get(name, None), fill=0.0)
        if raw.size != n_total:
            raw = np.zeros(n_total, dtype=np.float64)
        sub = raw[idx]
        rank_cols.append(_rank01(-sub))  # negate: lower conf → higher rank

    for name in conf_higher_better:
        raw = _impute_finite(raw_values.get(name, None), fill=0.0)
        if raw.size != n_total:
            raw = np.zeros(n_total, dtype=np.float64)
        sub = raw[idx]
        rank_cols.append(_rank01(sub))

    extra_feat = np.column_stack(rank_cols) if rank_cols else np.empty((len(idx), 0))

    return np.column_stack([e8_feat, extra_feat]).astype(np.float64)


def accumulate_extreme9_scores(
    best_model,
    worst_model,
    raw_values: dict[str, np.ndarray],
    tuple_size: int,
    num_tuples: int,
    seed: int,
    labels: Optional[np.ndarray] = None,
    require_mixed: bool = False,
) -> dict[str, np.ndarray]:
    """Aggregate tuple-level predictions into per-run best/worst/mixed scores (Extreme9)."""
    n_runs = int(len(raw_values["dc_raw"]))
    rng = np.random.RandomState(int(seed))
    tuples = sample_tuple_indices(
        n_runs=n_runs,
        tuple_size=tuple_size,
        num_tuples=num_tuples,
        rng=rng,
        labels=labels,
        require_mixed=require_mixed,
    )

    score_best = np.zeros(n_runs, dtype=np.float64)
    score_worst = np.zeros(n_runs, dtype=np.float64)
    counts = np.zeros(n_runs, dtype=np.float64)

    for idx in tuples:
        feat = build_extreme9_features(raw_values, idx)
        if best_model is not None:
            if hasattr(best_model, "predict_proba"):
                best_probs = best_model.predict_proba(feat)[:, 1]
            else:
                best_probs = np.asarray(best_model.predict(feat), dtype=np.float64)
            score_best[idx] += np.asarray(best_probs, dtype=np.float64)
        if worst_model is not None:
            if hasattr(worst_model, "predict_proba"):
                worst_probs = worst_model.predict_proba(feat)[:, 1]
            else:
                worst_probs = np.asarray(worst_model.predict(feat), dtype=np.float64)
            score_worst[idx] += np.asarray(worst_probs, dtype=np.float64)
        counts[idx] += 1.0

    valid = counts > 0
    if valid.any():
        score_best[valid] /= counts[valid]
        score_worst[valid] /= counts[valid]

    score_mix = score_best - score_worst
    return {
        "score_best": score_best,
        "score_worst": score_worst,
        "score_mix": score_mix,
        "counts": counts,
        "num_tuples": np.array([len(tuples)], dtype=np.int64),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Selector classes
# ─────────────────────────────────────────────────────────────────────────────

class _Extreme9BaseSelector(Selector):
    def __init__(
        self,
        tuple_size: int = 8,
        num_tuples: int = 1024,
        seed: int = 42,
        reflection_threshold: float = DEFAULT_REFLECTION_THRESHOLD,
    ):
        self.tuple_size = int(tuple_size)
        self.num_tuples = int(num_tuples)
        self.seed = int(seed)
        self.reflection_threshold = float(reflection_threshold)
        self._context: Optional[SelectorContext] = None
        self._raw_values: Optional[dict[str, np.ndarray]] = None

    def bind(self, context: SelectorContext) -> None:
        self._context = context
        self._raw_values = extract_extreme9_raw_values(
            context,
            reflection_threshold=self.reflection_threshold,
        )

    def _score_payload(self, best_model, worst_model) -> dict[str, np.ndarray]:
        assert self._context is not None, "Extreme9 selector requires bind(context)"
        assert self._raw_values is not None, "Extreme9 selector missing cached raw values"
        return accumulate_extreme9_scores(
            best_model=best_model,
            worst_model=worst_model,
            raw_values=self._raw_values,
            tuple_size=self.tuple_size,
            num_tuples=self.num_tuples,
            seed=self.seed,
            labels=None,
            require_mixed=False,
        )


class Extreme9BestSelector(_Extreme9BaseSelector):
    def __init__(
        self,
        model_path: str | None = None,
        tuple_size: int = 8,
        num_tuples: int = 1024,
        seed: int = 42,
        reflection_threshold: float = DEFAULT_REFLECTION_THRESHOLD,
    ):
        super().__init__(
            tuple_size=tuple_size,
            num_tuples=num_tuples,
            seed=seed,
            reflection_threshold=reflection_threshold,
        )
        self._model_path = Path(model_path) if model_path else _DEFAULT_MODEL_DIR / "extreme9_best.pkl"
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = _load_model(self._model_path)
        return self._model

    def select(self, D: np.ndarray, run_stats: dict) -> int:
        payload = self._score_payload(best_model=self._get_model(), worst_model=None)
        return int(np.argmax(payload["score_best"]))


class Extreme9WorstSelector(_Extreme9BaseSelector):
    def __init__(
        self,
        model_path: str | None = None,
        tuple_size: int = 8,
        num_tuples: int = 1024,
        seed: int = 42,
        reflection_threshold: float = DEFAULT_REFLECTION_THRESHOLD,
    ):
        super().__init__(
            tuple_size=tuple_size,
            num_tuples=num_tuples,
            seed=seed,
            reflection_threshold=reflection_threshold,
        )
        self._model_path = Path(model_path) if model_path else _DEFAULT_MODEL_DIR / "extreme9_worst.pkl"
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = _load_model(self._model_path)
        return self._model

    def select(self, D: np.ndarray, run_stats: dict) -> int:
        payload = self._score_payload(best_model=None, worst_model=self._get_model())
        return int(np.argmax(payload["score_worst"]))


class Extreme9MixedSelector(_Extreme9BaseSelector):
    def __init__(
        self,
        best_model_path: str | None = None,
        worst_model_path: str | None = None,
        tuple_size: int = 8,
        num_tuples: int = 1024,
        seed: int = 42,
        reflection_threshold: float = DEFAULT_REFLECTION_THRESHOLD,
    ):
        super().__init__(
            tuple_size=tuple_size,
            num_tuples=num_tuples,
            seed=seed,
            reflection_threshold=reflection_threshold,
        )
        self._best_model_path = (
            Path(best_model_path) if best_model_path else _DEFAULT_MODEL_DIR / "extreme9_best.pkl"
        )
        self._worst_model_path = (
            Path(worst_model_path) if worst_model_path else _DEFAULT_MODEL_DIR / "extreme9_worst.pkl"
        )
        self._best_model = None
        self._worst_model = None

    def _get_best_model(self):
        if self._best_model is None:
            self._best_model = _load_model(self._best_model_path)
        return self._best_model

    def _get_worst_model(self):
        if self._worst_model is None:
            self._worst_model = _load_model(self._worst_model_path)
        return self._worst_model

    def select(self, D: np.ndarray, run_stats: dict) -> int:
        payload = self._score_payload(
            best_model=self._get_best_model(),
            worst_model=self._get_worst_model(),
        )
        return int(np.argmax(payload["score_mix"]))
