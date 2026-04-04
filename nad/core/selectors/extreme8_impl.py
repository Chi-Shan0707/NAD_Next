from __future__ import annotations

from dataclasses import dataclass
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
DEFAULT_BAND_SIZES = (4, 4, 4)


def _load_model(path: Path):
    try:
        import joblib
        return joblib.load(path)
    except Exception as exc:  # pragma: no cover - error path
        raise RuntimeError(
            f"Failed to load Extreme8 selector model from {path}.\n"
            f"Run: python scripts/train_extreme8_selectors.py\nOriginal error: {exc}"
        )


@dataclass
class LinearRankModel:
    weights: np.ndarray
    bias: float = 0.0

    def __post_init__(self) -> None:
        arr = np.asarray(self.weights, dtype=np.float64).reshape(-1)
        if arr.size != N_EXTREME8_FEATURES:
            raise ValueError(
                f"LinearRankModel expects {N_EXTREME8_FEATURES} weights, got {arr.size}"
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


@dataclass
class ZeroRankModel:
    def predict(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2-D X, got {arr.shape}")
        return np.zeros(arr.shape[0], dtype=np.float64)


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


def normalize_weight_direction(weights: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= eps:
        if arr.size == 0:
            return arr
        out = np.zeros_like(arr)
        out[0] = 1.0
        return out
    return arr / norm


def tuple_band_counts(
    sorted_labels: np.ndarray,
    band_sizes: tuple[int, ...] = DEFAULT_BAND_SIZES,
) -> tuple[int, ...]:
    labels = np.asarray(sorted_labels, dtype=np.int32).reshape(-1)
    if labels.size != int(sum(band_sizes)):
        raise ValueError(
            f"sorted_labels has size {labels.size}, expected {sum(band_sizes)} for band_sizes={band_sizes}"
        )

    counts: list[int] = []
    start = 0
    for size in band_sizes:
        stop = start + int(size)
        counts.append(int(labels[start:stop].sum()))
        start = stop
    return tuple(counts)


def compute_band_reward(
    sorted_labels: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    band_sizes: tuple[int, ...] = DEFAULT_BAND_SIZES,
) -> float:
    counts = tuple_band_counts(sorted_labels, band_sizes=band_sizes)
    weights = np.asarray([alpha, beta, gamma], dtype=np.float64)
    if len(counts) != len(weights):
        raise ValueError(
            f"Band count length {len(counts)} does not match weight length {len(weights)}"
        )
    return float(np.dot(np.asarray(counts, dtype=np.float64), weights))


def band_reward_bounds(
    n_correct: int,
    alpha: float,
    beta: float,
    gamma: float,
    band_sizes: tuple[int, ...] = DEFAULT_BAND_SIZES,
) -> tuple[float, float]:
    total = int(sum(band_sizes))
    correct = int(max(0, min(total, int(n_correct))))
    weights = np.asarray([alpha, beta, gamma], dtype=np.float64)
    sizes = np.asarray(band_sizes, dtype=np.int32)
    if weights.size != sizes.size:
        raise ValueError(
            f"Weight length {weights.size} does not match band_sizes length {sizes.size}"
        )

    def _fill(order: np.ndarray) -> float:
        remain = correct
        reward = 0.0
        for idx in order:
            take = min(remain, int(sizes[idx]))
            reward += float(take) * float(weights[idx])
            remain -= take
            if remain <= 0:
                break
        return float(reward)

    best = _fill(np.argsort(-weights, kind="mergesort"))
    worst = _fill(np.argsort(weights, kind="mergesort"))
    return best, worst


def normalize_band_reward(
    reward: float,
    n_correct: int,
    alpha: float,
    beta: float,
    gamma: float,
    band_sizes: tuple[int, ...] = DEFAULT_BAND_SIZES,
    eps: float = 1e-8,
) -> float:
    best, worst = band_reward_bounds(
        n_correct=n_correct,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        band_sizes=band_sizes,
    )
    denom = max(float(best - worst), float(eps))
    return float((float(reward) - float(worst)) / denom)


def compute_normalized_band_reward(
    scores: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    band_sizes: tuple[int, ...] = DEFAULT_BAND_SIZES,
) -> float:
    score_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    label_arr = np.asarray(labels, dtype=np.int32).reshape(-1)
    if score_arr.size != label_arr.size:
        raise ValueError(
            f"scores size {score_arr.size} does not match labels size {label_arr.size}"
        )
    order = np.argsort(-score_arr, kind="mergesort")
    sorted_labels = label_arr[order]
    raw_reward = compute_band_reward(
        sorted_labels=sorted_labels,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        band_sizes=band_sizes,
    )
    return normalize_band_reward(
        reward=raw_reward,
        n_correct=int(label_arr.sum()),
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        band_sizes=band_sizes,
    )


def sample_tuple_indices(
    n_runs: int,
    tuple_size: int,
    num_tuples: int,
    rng: np.random.RandomState,
    labels: Optional[np.ndarray] = None,
    require_mixed: bool = False,
    min_correct: Optional[int] = None,
    max_correct: Optional[int] = None,
) -> list[np.ndarray]:
    """Sample tuple indices with optional label-count constraints."""
    if n_runs <= 0:
        return []

    k = min(int(tuple_size), n_runs)
    num_tuples = max(1, int(num_tuples))

    if labels is not None:
        labels = np.asarray(labels, dtype=np.int32)

    def _valid_count(n_correct: int) -> bool:
        if require_mixed and not (0 < int(n_correct) < k):
            return False
        if min_correct is not None and int(n_correct) < int(min_correct):
            return False
        if max_correct is not None and int(n_correct) > int(max_correct):
            return False
        return True

    def _accept(idx: np.ndarray) -> bool:
        if labels is None:
            return True
        return _valid_count(int(labels[idx].sum()))

    if labels is None:
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

    min_pos = max(
        0,
        int(k - neg.size),
        int(min_correct) if min_correct is not None else 0,
        1 if require_mixed else 0,
    )
    max_pos = min(
        int(k),
        int(pos.size),
        int(max_correct) if max_correct is not None else int(k),
        int(k - 1) if require_mixed else int(k),
    )
    if min_pos > max_pos:
        raise ValueError(
            "No feasible tuple satisfies the requested correct-count constraints: "
            f"n_runs={n_runs}, tuple_size={tuple_size}, pos={pos.size}, neg={neg.size}, "
            f"require_mixed={require_mixed}, min_correct={min_correct}, max_correct={max_correct}"
        )

    tuples: list[np.ndarray] = []
    attempts = 0
    max_attempts = max(100, num_tuples * 20)
    while len(tuples) < num_tuples and attempts < max_attempts:
        idx = np.sort(rng.choice(n_runs, size=k, replace=False).astype(np.int64, copy=False))
        if _accept(idx):
            tuples.append(idx)
        attempts += 1

    while len(tuples) < num_tuples:
        n_pos = int(rng.randint(min_pos, max_pos + 1))
        n_neg = int(k - n_pos)
        pos_pick = rng.choice(pos, size=n_pos, replace=False).astype(np.int64, copy=False)
        neg_pick = rng.choice(neg, size=n_neg, replace=False).astype(np.int64, copy=False)
        idx = np.sort(np.concatenate([pos_pick, neg_pick]))
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
