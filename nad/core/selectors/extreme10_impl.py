"""
Extreme10 selector — 17-dimensional graph-topology + error-mass feature expansion.

Inherits the 11-feature Extreme9 core (dc_z, dc_r, reflection_count_r + 8 local
tok_conf features) and augments with:
  - 3 graph topology features derived from the 64-run Jaccard distance matrix D
    (lazy-computed in select() because D is not available in bind())
  - 3 error-mass / late-stage features from tok_conf time-series (bound in bind())

Feature list (EXTREME10_FEATURE_NAMES, 17 dims):
   0  dc_z                   DeepConf quality z-score
   1  dc_r                   DeepConf quality rank
   2  reflection_count_r     reflection count rank
   3  tail_2k_r              mean tok_conf in last min(2000,T) tokens rank
   4  tail_q10_r             mean tok_conf in last 10% tokens rank
   5  lgc_512_r              LGC window=512 rank
   6  lgc_2k_r               LGC window=2000 rank
   7  bottom_q10_r           10th-percentile tok_conf rank
   8  head_tail_gap_r        (mean head 10%) − (mean tail 10%) rank
   9  last_event_tail_conf_r mean tok_conf after last reflection event rank
  10  event_nonevent_gap_r   (event mean) − (non-event mean) tok_conf rank
  11  local_cc_r             local clustering coefficient rank (from D)
  12  norm_degree_r          normalised in-graph degree rank (from D)
  13  cluster_size_r         DBSCAN cluster size fraction rank (from D)
  14  instability_mass_r     fraction of unstable tokens rank (lower = better)
  15  tail_variance_r        variance of last 10% tokens rank (lower = better)
  16  event_pre_post_delta_r pre-event minus post-event confidence delta rank

Quality direction:
  Graph features (11-13): higher = more connected / larger cluster = better → _rank01(sub)
  instability_mass, tail_variance (14-15): lower = better → _rank01(-sub)
  event_pre_post_delta (16): higher = better → _rank01(sub)
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
from .extreme9_impl import (
    build_extreme9_features,
    extract_extreme9_raw_values,
)
from .graph_topo_impl import extract_graph_topo_raw
from .local_conf_impl import extract_error_mass_raw
from .ml_features import _rank01
from .trajectory_impl import DEFAULT_REFLECTION_THRESHOLD

_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models" / "ml_selectors"

EXTREME10_FEATURE_NAMES = [
    # Extreme9 base (11)
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
    # Graph topology — from D in select() (3)
    "local_cc_r",
    "norm_degree_r",
    "cluster_size_r",
    # Error-mass + late-stage — from bind() (3)
    "instability_mass_r",
    "tail_variance_r",
    "event_pre_post_delta_r",
]
N_EXTREME10_FEATURES = len(EXTREME10_FEATURE_NAMES)


def _load_model(path: Path):
    try:
        import joblib
        return joblib.load(path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load Extreme10 selector model from {path}.\n"
            f"Run: python scripts/train_extreme10_selectors.py\nOriginal error: {exc}"
        )


@dataclass
class LinearRankModel10:
    """17-dimensional linear ranking model for Extreme10."""
    weights: np.ndarray
    bias: float = 0.0

    def __post_init__(self) -> None:
        arr = np.asarray(self.weights, dtype=np.float64).reshape(-1)
        if arr.size != N_EXTREME10_FEATURES:
            raise ValueError(
                f"LinearRankModel10 expects {N_EXTREME10_FEATURES} weights, got {arr.size}"
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


def extract_extreme10_raw_values(
    context: SelectorContext,
    reflection_threshold: float = DEFAULT_REFLECTION_THRESHOLD,
) -> dict[str, np.ndarray]:
    """
    Compute the full set of raw per-run values for the Extreme10 model.

    Merges Extreme9 raw values (dc_raw, reflection_count, local conf features)
    with error-mass features (instability_mass, tail_variance,
    event_pre_post_delta).  Graph topology features are NOT included here
    because they require D (computed in select()).
    """
    e9_raw = extract_extreme9_raw_values(context, reflection_threshold=reflection_threshold)
    em_raw = extract_error_mass_raw(context, reflection_threshold=reflection_threshold)
    return {**e9_raw, **em_raw}


def build_extreme10_features(
    raw_values: dict[str, np.ndarray],
    graph_raw: dict[str, np.ndarray],
    indices: np.ndarray | list[int] | None = None,
) -> np.ndarray:
    """
    Build the 17-D Extreme10 feature matrix for the selected tuple indices.

    Parameters
    ----------
    raw_values : dict
        Output of extract_extreme10_raw_values() — 14 keys (no graph).
    graph_raw : dict
        Output of extract_graph_topo_raw() — 3 keys (local_cc, norm_degree,
        cluster_size_frac).  Must have n_runs entries.
    indices : array-like or None
        Subset of run indices.  None = all runs.

    Returns
    -------
    np.ndarray, shape (n_sub, 17)
    """
    # ── Extreme9 base features (11 dims) ────────────────────────────────────
    e9_feat = build_extreme9_features(raw_values, indices)  # (n_sub, 11)

    n_total = len(raw_values["dc_raw"])
    if indices is None:
        idx = np.arange(n_total, dtype=np.int64)
    else:
        idx = np.asarray(indices, dtype=np.int64)

    # ── Graph topology rank features (3 dims) ────────────────────────────────
    # All three: higher = better → _rank01(sub)
    graph_cols = []
    for name in ("local_cc", "norm_degree", "cluster_size_frac"):
        raw = graph_raw.get(name, None)
        if raw is None or len(raw) == 0:
            raw = np.zeros(n_total, dtype=np.float64)
        raw = np.asarray(raw, dtype=np.float64)
        if raw.size != n_total:
            raw = np.zeros(n_total, dtype=np.float64)
        sub = raw[idx]
        graph_cols.append(_rank01(sub))

    # ── Error-mass rank features (3 dims) ────────────────────────────────────
    # instability_mass, tail_variance: lower = better → _rank01(-sub)
    # event_pre_post_delta: higher = better → _rank01(sub)
    em_cols = []
    for name in ("instability_mass", "tail_variance"):
        raw = _impute_finite(raw_values.get(name, None), fill=0.5)
        if raw.size != n_total:
            raw = np.full(n_total, 0.5, dtype=np.float64)
        sub = raw[idx]
        em_cols.append(_rank01(-sub))  # negate: lower = better

    raw_epd = _impute_finite(raw_values.get("event_pre_post_delta", None), fill=0.0)
    if raw_epd.size != n_total:
        raw_epd = np.zeros(n_total, dtype=np.float64)
    sub_epd = raw_epd[idx]
    em_cols.append(_rank01(sub_epd))  # higher = better

    graph_feat = np.column_stack(graph_cols) if graph_cols else np.empty((len(idx), 0))
    em_feat = np.column_stack(em_cols) if em_cols else np.empty((len(idx), 0))

    return np.column_stack([e9_feat, graph_feat, em_feat]).astype(np.float64)


def accumulate_extreme10_scores(
    best_model,
    worst_model,
    raw_values: dict[str, np.ndarray],
    tuple_size: int,
    num_tuples: int,
    seed: int,
    graph_raw: dict[str, np.ndarray] | None = None,
    D: np.ndarray | None = None,
    labels: Optional[np.ndarray] = None,
    require_mixed: bool = False,
) -> dict[str, np.ndarray]:
    """
    Aggregate tuple-level predictions into per-run best/worst/mixed scores (Extreme10).

    Either graph_raw or D must be provided (or both — graph_raw takes priority).
    If only D is provided, graph_raw is computed from D.
    """
    if graph_raw is None:
        if D is None:
            raise ValueError("accumulate_extreme10_scores: must provide graph_raw or D")
        graph_raw = extract_graph_topo_raw(D)

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
        feat = build_extreme10_features(raw_values, graph_raw, idx)
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

class _Extreme10BaseSelector(Selector):
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
        self._graph_raw: Optional[dict[str, np.ndarray]] = None  # lazy per-group cache

    def bind(self, context: SelectorContext) -> None:
        self._context = context
        self._raw_values = extract_extreme10_raw_values(
            context,
            reflection_threshold=self.reflection_threshold,
        )
        self._graph_raw = None  # reset for new group (bind() is called per group)

    def _score_payload(self, D: np.ndarray, best_model, worst_model) -> dict[str, np.ndarray]:
        assert self._context is not None, "Extreme10 selector requires bind(context)"
        assert self._raw_values is not None, "Extreme10 selector missing cached raw values"
        # Lazy-compute graph topology (once per group)
        if self._graph_raw is None:
            self._graph_raw = extract_graph_topo_raw(D)
        return accumulate_extreme10_scores(
            best_model=best_model,
            worst_model=worst_model,
            raw_values=self._raw_values,
            tuple_size=self.tuple_size,
            num_tuples=self.num_tuples,
            seed=self.seed,
            graph_raw=self._graph_raw,
            labels=None,
            require_mixed=False,
        )


class Extreme10BestSelector(_Extreme10BaseSelector):
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
        self._model_path = Path(model_path) if model_path else _DEFAULT_MODEL_DIR / "extreme10_best.pkl"
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = _load_model(self._model_path)
        return self._model

    def select(self, D: np.ndarray, run_stats: dict) -> int:
        payload = self._score_payload(D, best_model=self._get_model(), worst_model=None)
        return int(np.argmax(payload["score_best"]))


class Extreme10WorstSelector(_Extreme10BaseSelector):
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
        self._model_path = Path(model_path) if model_path else _DEFAULT_MODEL_DIR / "extreme10_worst.pkl"
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = _load_model(self._model_path)
        return self._model

    def select(self, D: np.ndarray, run_stats: dict) -> int:
        payload = self._score_payload(D, best_model=None, worst_model=self._get_model())
        return int(np.argmax(payload["score_worst"]))


class Extreme10MixedSelector(_Extreme10BaseSelector):
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
            Path(best_model_path) if best_model_path else _DEFAULT_MODEL_DIR / "extreme10_best.pkl"
        )
        self._worst_model_path = (
            Path(worst_model_path) if worst_model_path else _DEFAULT_MODEL_DIR / "extreme10_worst.pkl"
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
            D,
            best_model=self._get_best_model(),
            worst_model=self._get_worst_model(),
        )
        return int(np.argmax(payload["score_mix"]))
