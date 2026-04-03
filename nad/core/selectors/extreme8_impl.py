from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .base import Selector, SelectorContext
from .ml_features import _deepconf_quality, _rank01, _zscore
from .trajectory_impl import (
    DEFAULT_REFLECTION_THRESHOLD,
    _compute_trajectory_scores,
    _extract_slice_keysets,
)

_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models" / "ml_selectors"

EXTREME8_FEATURE_NAMES = [
    "dc_z",
    "dc_r",
    "reflection_count_r",
]
N_EXTREME8_FEATURES = len(EXTREME8_FEATURE_NAMES)


def _load_model(path: Path):
    try:
        import joblib
        return joblib.load(path)
    except Exception as exc:  # pragma: no cover - error path
        raise RuntimeError(
            f"Failed to load Extreme8 selector model from {path}.\n"
            f"Run: python scripts/train_extreme8_selectors.py\nOriginal error: {exc}"
        )


def _impute_finite(values: Optional[np.ndarray], fill: float = 0.0) -> np.ndarray:
    if values is None:
        return np.zeros(0, dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    finite = np.isfinite(arr)
    if finite.any():
        fill = float(np.nanmean(arr[finite]))
    return np.where(finite, arr, fill).astype(np.float64, copy=False)


def extract_extreme8_raw_values(
    context: SelectorContext,
    reflection_threshold: float = DEFAULT_REFLECTION_THRESHOLD,
) -> dict[str, np.ndarray]:
    """Precompute the raw per-run values needed by the Extreme8 models."""
    n = len(context.run_ids)

    dc_raw = _deepconf_quality(context)
    if dc_raw is None:
        dc_raw = np.zeros(n, dtype=np.float64)
    dc_raw = _impute_finite(dc_raw)

    reflection_count = np.zeros(n, dtype=np.float64)
    for i, rid in enumerate(context.run_ids):
        slices = _extract_slice_keysets(context.cache, int(rid))
        if not slices:
            continue
        traj = _compute_trajectory_scores(
            slices,
            reflection_threshold=float(reflection_threshold),
        )
        reflection_count[i] = float(traj["reflection_count"])

    return {
        "dc_raw": dc_raw,
        "reflection_count": reflection_count.astype(np.float64, copy=False),
    }


def build_extreme8_features(
    raw_values: dict[str, np.ndarray],
    indices: np.ndarray | list[int] | None = None,
) -> np.ndarray:
    """Build the 3-D Extreme8 feature matrix for the selected tuple indices."""
    dc_raw = np.asarray(raw_values["dc_raw"], dtype=np.float64)
    reflection_count = np.asarray(raw_values["reflection_count"], dtype=np.float64)

    if indices is None:
        idx = np.arange(len(dc_raw), dtype=np.int64)
    else:
        idx = np.asarray(indices, dtype=np.int64)

    sub_dc = dc_raw[idx]
    sub_reflection = reflection_count[idx]

    return np.column_stack([
        _zscore(sub_dc),
        _rank01(sub_dc),
        _rank01(sub_reflection),
    ]).astype(np.float64)


def sample_tuple_indices(
    n_runs: int,
    tuple_size: int,
    num_tuples: int,
    rng: np.random.RandomState,
    labels: Optional[np.ndarray] = None,
    require_mixed: bool = False,
) -> list[np.ndarray]:
    """Sample tuple indices with optional correct/incorrect mixing."""
    if n_runs <= 0:
        return []

    k = min(int(tuple_size), n_runs)
    num_tuples = max(1, int(num_tuples))

    if labels is not None:
        labels = np.asarray(labels, dtype=np.int32)

    if not require_mixed or labels is None:
        if n_runs <= k:
            return [np.arange(n_runs, dtype=np.int64)]
        return [
            np.sort(rng.choice(n_runs, size=k, replace=False).astype(np.int64, copy=False))
            for _ in range(num_tuples)
        ]

    pos = np.flatnonzero(labels > 0)
    neg = np.flatnonzero(labels <= 0)
    if pos.size == 0 or neg.size == 0:
        if n_runs <= k:
            return [np.arange(n_runs, dtype=np.int64)]
        return [
            np.sort(rng.choice(n_runs, size=k, replace=False).astype(np.int64, copy=False))
            for _ in range(num_tuples)
        ]

    tuples: list[np.ndarray] = []
    attempts = 0
    max_attempts = max(100, num_tuples * 20)
    while len(tuples) < num_tuples and attempts < max_attempts:
        idx = np.sort(rng.choice(n_runs, size=k, replace=False).astype(np.int64, copy=False))
        sub_labels = labels[idx]
        if np.any(sub_labels > 0) and np.any(sub_labels <= 0):
            tuples.append(idx)
        attempts += 1

    while len(tuples) < num_tuples:
        chosen = {int(rng.choice(pos)), int(rng.choice(neg))}
        need = max(0, k - len(chosen))
        if need > 0:
            pool = np.setdiff1d(np.arange(n_runs, dtype=np.int64), np.fromiter(chosen, dtype=np.int64), assume_unique=False)
            extra = rng.choice(pool, size=need, replace=False).astype(np.int64, copy=False)
            idx = np.sort(np.concatenate([np.fromiter(chosen, dtype=np.int64), extra]))
        else:
            idx = np.sort(np.fromiter(chosen, dtype=np.int64))
        tuples.append(idx)

    return tuples


def accumulate_extreme8_scores(
    best_model,
    worst_model,
    raw_values: dict[str, np.ndarray],
    tuple_size: int,
    num_tuples: int,
    seed: int,
    labels: Optional[np.ndarray] = None,
    require_mixed: bool = False,
) -> dict[str, np.ndarray]:
    """Aggregate tuple-level predictions into per-run best/worst/mixed scores."""
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
        feat = build_extreme8_features(raw_values, idx)
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


class _Extreme8BaseSelector(Selector):
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
        self._raw_values = extract_extreme8_raw_values(
            context,
            reflection_threshold=self.reflection_threshold,
        )

    def _score_payload(self, best_model, worst_model) -> dict[str, np.ndarray]:
        assert self._context is not None, "Extreme8 selector requires bind(context)"
        assert self._raw_values is not None, "Extreme8 selector missing cached raw values"
        return accumulate_extreme8_scores(
            best_model=best_model,
            worst_model=worst_model,
            raw_values=self._raw_values,
            tuple_size=self.tuple_size,
            num_tuples=self.num_tuples,
            seed=self.seed,
            labels=None,
            require_mixed=False,
        )


class Extreme8BestSelector(_Extreme8BaseSelector):
    def __init__(
        self,
        model_path: str | None = None,
        tuple_size: int = 8,
        num_tuples: int = 1024,
        seed: int = 42,
        reflection_threshold: float = DEFAULT_REFLECTION_THRESHOLD,
    ):
        super().__init__(tuple_size=tuple_size, num_tuples=num_tuples, seed=seed, reflection_threshold=reflection_threshold)
        self._model_path = Path(model_path) if model_path else _DEFAULT_MODEL_DIR / "extreme8_best.pkl"
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = _load_model(self._model_path)
        return self._model

    def select(self, D: np.ndarray, run_stats: dict) -> int:
        payload = self._score_payload(best_model=self._get_model(), worst_model=None)
        return int(np.argmax(payload["score_best"]))


class Extreme8WorstSelector(_Extreme8BaseSelector):
    def __init__(
        self,
        model_path: str | None = None,
        tuple_size: int = 8,
        num_tuples: int = 1024,
        seed: int = 42,
        reflection_threshold: float = DEFAULT_REFLECTION_THRESHOLD,
    ):
        super().__init__(tuple_size=tuple_size, num_tuples=num_tuples, seed=seed, reflection_threshold=reflection_threshold)
        self._model_path = Path(model_path) if model_path else _DEFAULT_MODEL_DIR / "extreme8_worst.pkl"
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = _load_model(self._model_path)
        return self._model

    def select(self, D: np.ndarray, run_stats: dict) -> int:
        payload = self._score_payload(best_model=None, worst_model=self._get_model())
        return int(np.argmax(payload["score_worst"]))


class Extreme8MixedSelector(_Extreme8BaseSelector):
    def __init__(
        self,
        best_model_path: str | None = None,
        worst_model_path: str | None = None,
        tuple_size: int = 8,
        num_tuples: int = 1024,
        seed: int = 42,
        reflection_threshold: float = DEFAULT_REFLECTION_THRESHOLD,
    ):
        super().__init__(tuple_size=tuple_size, num_tuples=num_tuples, seed=seed, reflection_threshold=reflection_threshold)
        self._best_model_path = Path(best_model_path) if best_model_path else _DEFAULT_MODEL_DIR / "extreme8_best.pkl"
        self._worst_model_path = Path(worst_model_path) if worst_model_path else _DEFAULT_MODEL_DIR / "extreme8_worst.pkl"
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
