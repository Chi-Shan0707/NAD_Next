#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import ElasticNet, HuberRegressor, LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, LinearSVR


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURE_STORE = (
    REPO_ROOT
    / "results"
    / "cache"
    / "export_rl_checkpoint_ranking_from_svd_models"
    / "feature_store_all_ref030_05edbff25fbfa65b.pkl"
)
DEFAULT_OUTPUTS_DIR = REPO_ROOT / "outputs"

OFFICIAL_CHECKPOINTS = (
    "base",
    "step-100",
    "step-200",
    "step-300",
    "step-400",
    "step-500",
    "step-600",
    "step-700",
    "step-800",
    "step-900",
    "step-1000",
)
CHECKPOINT_ORDER = {name: idx for idx, name in enumerate(OFFICIAL_CHECKPOINTS)}

warnings.filterwarnings("ignore", category=ConvergenceWarning)


@dataclass(frozen=True)
class ScenarioCheckpointGroup:
    scenario_id: str
    checkpoint_idx: int
    checkpoint_name: str
    row_indices: np.ndarray
    true_accuracy: float


@dataclass(frozen=True)
class DatasetSubset:
    x: np.ndarray
    y: np.ndarray
    scenario_ids: np.ndarray
    checkpoint_idx: np.ndarray
    checkpoint_names: np.ndarray
    row_to_group: np.ndarray
    group_sizes: np.ndarray
    group_true_accuracy: np.ndarray
    groups: tuple[ScenarioCheckpointGroup, ...]
    trajectory_group_indices: tuple[np.ndarray, ...]
    checkpoint_group_indices: dict[int, np.ndarray]
    scenarios: tuple[str, ...]


@dataclass(frozen=True)
class HeadSpec:
    name: str
    family: str
    configs: tuple[dict[str, Any], ...]
    builder: Callable[[dict[str, Any], dict[str, Any]], "BaseHead"]


def _official_checkpoint_name(cache_key: str) -> str:
    model_dir_name = str(cache_key).split("/", 1)[0]
    if model_dir_name == "Qwen3-4B-Base_base":
        return "base"
    if "_math7500-step-" in model_dir_name:
        step = int(model_dir_name.rsplit("-step-", 1)[1])
        name = f"step-{step}"
        if name in CHECKPOINT_ORDER:
            return name
    raise ValueError(f"Unrecognized RL checkpoint cache key: {cache_key!r}")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def _safe_statistic(result: Any) -> float | None:
    value = getattr(result, "statistic", result)
    if isinstance(value, tuple):
        value = value[0]
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def _safe_mean(values: Sequence[float]) -> float | None:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return None
    return float(np.mean(arr))


def _score_tuple(metrics: dict[str, Any]) -> tuple[float, float, float, float, float]:
    def _coerce(value: Any, default: float = float("-inf")) -> float:
        if value is None:
            return default
        try:
            numeric = float(value)
        except Exception:
            return default
        if not math.isfinite(numeric):
            return default
        return numeric

    return (
        _coerce(metrics.get("checkpoint_spearman")),
        _coerce(metrics.get("checkpoint_kendall")),
        _coerce(metrics.get("top1_hit")),
        _coerce(metrics.get("top3_hit")),
        -_coerce(metrics.get("scenario_rmse"), default=float("inf")),
    )


def _sigmoid_clip(values: np.ndarray) -> np.ndarray:
    return expit(np.asarray(values, dtype=np.float64))


def _clip01(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)


def load_feature_store(path: Path) -> DatasetSubset:
    payload = pickle.load(path.open("rb"))
    feature_store = payload["feature_store"]
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    scenario_parts: list[np.ndarray] = []
    checkpoint_idx_parts: list[np.ndarray] = []
    checkpoint_name_parts: list[np.ndarray] = []
    groups: list[ScenarioCheckpointGroup] = []
    row_cursor = 0

    ordered_entries = sorted(
        feature_store,
        key=lambda item: int(CHECKPOINT_ORDER[_official_checkpoint_name(item["cache_key"])]),
    )
    for entry in ordered_entries:
        checkpoint_name = _official_checkpoint_name(str(entry["cache_key"]))
        checkpoint_idx = int(CHECKPOINT_ORDER[checkpoint_name])
        tensor = np.asarray(entry["tensor"], dtype=np.float64)
        labels = np.asarray(entry["labels"], dtype=np.int32)
        problem_ids = [str(value) for value in entry["problem_ids"]]
        offsets = [int(value) for value in entry["problem_offsets"]]
        if tensor.ndim != 3 or tensor.shape[1] != 1:
            raise ValueError(f"Expected shape [n, 1, d], got {tensor.shape} for {entry['cache_key']}")
        x_full = tensor[:, 0, :]
        for problem_idx, scenario_id in enumerate(problem_ids):
            start = offsets[problem_idx]
            end = offsets[problem_idx + 1]
            if end <= start:
                continue
            x_group = np.asarray(x_full[start:end], dtype=np.float64)
            y_group = np.asarray(labels[start:end], dtype=np.int32)
            n_rows = int(end - start)
            x_parts.append(x_group)
            y_parts.append(y_group)
            scenario_parts.append(np.asarray([scenario_id] * n_rows, dtype=object))
            checkpoint_idx_parts.append(np.full(n_rows, checkpoint_idx, dtype=np.int32))
            checkpoint_name_parts.append(np.asarray([checkpoint_name] * n_rows, dtype=object))
            row_indices = np.arange(row_cursor, row_cursor + n_rows, dtype=np.int64)
            row_cursor += n_rows
            groups.append(
                ScenarioCheckpointGroup(
                    scenario_id=str(scenario_id),
                    checkpoint_idx=int(checkpoint_idx),
                    checkpoint_name=str(checkpoint_name),
                    row_indices=row_indices,
                    true_accuracy=float(np.mean(y_group)),
                )
            )

    x = np.vstack(x_parts).astype(np.float64, copy=False)
    y = np.concatenate(y_parts).astype(np.int32, copy=False)
    scenario_ids = np.concatenate(scenario_parts).astype(object, copy=False)
    checkpoint_idx = np.concatenate(checkpoint_idx_parts).astype(np.int32, copy=False)
    checkpoint_names = np.concatenate(checkpoint_name_parts).astype(object, copy=False)
    scenarios = tuple(sorted({str(value) for value in scenario_ids.tolist()}))

    row_to_group = np.zeros(y.shape[0], dtype=np.int32)
    group_sizes = np.zeros(len(groups), dtype=np.int32)
    group_true_accuracy = np.zeros(len(groups), dtype=np.float64)
    by_scenario: dict[str, list[int]] = defaultdict(list)
    checkpoint_group_indices: dict[int, list[int]] = defaultdict(list)
    for group_idx, group in enumerate(groups):
        row_to_group[group.row_indices] = int(group_idx)
        group_sizes[group_idx] = int(group.row_indices.shape[0])
        group_true_accuracy[group_idx] = float(group.true_accuracy)
        by_scenario[str(group.scenario_id)].append(int(group_idx))
        checkpoint_group_indices[int(group.checkpoint_idx)].append(int(group_idx))

    trajectory_group_indices = []
    for scenario_id in scenarios:
        idxs = sorted(by_scenario[scenario_id], key=lambda idx: groups[idx].checkpoint_idx)
        trajectory_group_indices.append(np.asarray(idxs, dtype=np.int32))

    return DatasetSubset(
        x=x,
        y=y,
        scenario_ids=scenario_ids,
        checkpoint_idx=checkpoint_idx,
        checkpoint_names=checkpoint_names,
        row_to_group=row_to_group,
        group_sizes=group_sizes,
        group_true_accuracy=group_true_accuracy,
        groups=tuple(groups),
        trajectory_group_indices=tuple(trajectory_group_indices),
        checkpoint_group_indices={key: np.asarray(value, dtype=np.int32) for key, value in checkpoint_group_indices.items()},
        scenarios=scenarios,
    )


def subset_by_scenarios(dataset: DatasetSubset, scenario_ids: Sequence[str]) -> DatasetSubset:
    scenario_set = {str(value) for value in scenario_ids}
    row_mask = np.asarray([str(value) in scenario_set for value in dataset.scenario_ids.tolist()], dtype=bool)
    row_indices_full = np.flatnonzero(row_mask).astype(np.int64)
    x = np.asarray(dataset.x[row_indices_full], dtype=np.float64)
    y = np.asarray(dataset.y[row_indices_full], dtype=np.int32)
    scenario_arr = np.asarray(dataset.scenario_ids[row_indices_full], dtype=object)
    checkpoint_idx = np.asarray(dataset.checkpoint_idx[row_indices_full], dtype=np.int32)
    checkpoint_names = np.asarray(dataset.checkpoint_names[row_indices_full], dtype=object)

    full_to_local = {int(full_idx): int(local_idx) for local_idx, full_idx in enumerate(row_indices_full.tolist())}
    groups: list[ScenarioCheckpointGroup] = []
    row_to_group = np.zeros(y.shape[0], dtype=np.int32)
    group_sizes: list[int] = []
    group_true_accuracy: list[float] = []
    by_scenario: dict[str, list[int]] = defaultdict(list)
    checkpoint_group_indices: dict[int, list[int]] = defaultdict(list)

    for group in dataset.groups:
        if str(group.scenario_id) not in scenario_set:
            continue
        local_rows = np.asarray([full_to_local[int(idx)] for idx in group.row_indices.tolist()], dtype=np.int64)
        group_idx = len(groups)
        groups.append(
            ScenarioCheckpointGroup(
                scenario_id=str(group.scenario_id),
                checkpoint_idx=int(group.checkpoint_idx),
                checkpoint_name=str(group.checkpoint_name),
                row_indices=local_rows,
                true_accuracy=float(group.true_accuracy),
            )
        )
        row_to_group[local_rows] = int(group_idx)
        group_sizes.append(int(local_rows.shape[0]))
        group_true_accuracy.append(float(group.true_accuracy))
        by_scenario[str(group.scenario_id)].append(int(group_idx))
        checkpoint_group_indices[int(group.checkpoint_idx)].append(int(group_idx))

    subset_scenarios = tuple(sorted(scenario_set))
    trajectory_group_indices = []
    for scenario_id in subset_scenarios:
        idxs = sorted(by_scenario[scenario_id], key=lambda idx: groups[idx].checkpoint_idx)
        trajectory_group_indices.append(np.asarray(idxs, dtype=np.int32))

    return DatasetSubset(
        x=x,
        y=y,
        scenario_ids=scenario_arr,
        checkpoint_idx=checkpoint_idx,
        checkpoint_names=checkpoint_names,
        row_to_group=row_to_group,
        group_sizes=np.asarray(group_sizes, dtype=np.int32),
        group_true_accuracy=np.asarray(group_true_accuracy, dtype=np.float64),
        groups=tuple(groups),
        trajectory_group_indices=tuple(trajectory_group_indices),
        checkpoint_group_indices={key: np.asarray(value, dtype=np.int32) for key, value in checkpoint_group_indices.items()},
        scenarios=subset_scenarios,
    )


def build_group_splits(scenarios: Sequence[str], n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    scenarios_arr = np.asarray(list(scenarios), dtype=object)
    if scenarios_arr.shape[0] < 2:
        return []
    splits = min(int(n_splits), int(scenarios_arr.shape[0]))
    if splits < 2:
        return []
    dummy = np.zeros((scenarios_arr.shape[0], 1), dtype=np.float64)
    gkf = GroupKFold(n_splits=splits)
    return list(gkf.split(dummy, groups=scenarios_arr))


def aggregate_group_scores(subset: DatasetSubset, run_values: np.ndarray) -> np.ndarray:
    values = np.asarray(run_values, dtype=np.float64).reshape(-1)
    if values.shape[0] != subset.y.shape[0]:
        raise ValueError(f"run_values length mismatch: {values.shape[0]} vs {subset.y.shape[0]}")
    sums = np.bincount(subset.row_to_group, weights=values, minlength=len(subset.groups)).astype(np.float64)
    return sums / np.maximum(1.0, subset.group_sizes.astype(np.float64))


def checkpoint_scores_from_group_scores(subset: DatasetSubset, group_scores: np.ndarray) -> dict[str, float]:
    scores = np.asarray(group_scores, dtype=np.float64)
    out: dict[str, float] = {}
    for checkpoint_idx, name in enumerate(OFFICIAL_CHECKPOINTS):
        idxs = subset.checkpoint_group_indices.get(checkpoint_idx)
        if idxs is None or idxs.size <= 0:
            out[name] = float("nan")
            continue
        out[name] = float(np.mean(scores[idxs]))
    return out


def reliability_curve(pred: np.ndarray, true: np.ndarray, n_bins: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_arr = _clip01(np.asarray(pred, dtype=np.float64).reshape(-1))
    true_arr = _clip01(np.asarray(true, dtype=np.float64).reshape(-1))
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1, dtype=np.float64)
    x_vals: list[float] = []
    y_vals: list[float] = []
    counts: list[float] = []
    for low, high in zip(bins[:-1], bins[1:]):
        if high >= 1.0:
            mask = (pred_arr >= low) & (pred_arr <= high)
        else:
            mask = (pred_arr >= low) & (pred_arr < high)
        if not np.any(mask):
            continue
        x_vals.append(float(np.mean(pred_arr[mask])))
        y_vals.append(float(np.mean(true_arr[mask])))
        counts.append(float(np.sum(mask)))
    return (
        np.asarray(x_vals, dtype=np.float64),
        np.asarray(y_vals, dtype=np.float64),
        np.asarray(counts, dtype=np.float64),
    )


def expected_calibration_error(pred: np.ndarray, true: np.ndarray, n_bins: int = 10) -> float:
    pred_arr = _clip01(np.asarray(pred, dtype=np.float64).reshape(-1))
    true_arr = _clip01(np.asarray(true, dtype=np.float64).reshape(-1))
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1, dtype=np.float64)
    total = float(pred_arr.shape[0])
    if total <= 0:
        return float("nan")
    ece = 0.0
    for low, high in zip(bins[:-1], bins[1:]):
        if high >= 1.0:
            mask = (pred_arr >= low) & (pred_arr <= high)
        else:
            mask = (pred_arr >= low) & (pred_arr < high)
        if not np.any(mask):
            continue
        weight = float(np.sum(mask)) / total
        ece += weight * abs(float(np.mean(pred_arr[mask])) - float(np.mean(true_arr[mask])))
    return float(ece)


def build_group_lookup(subset: DatasetSubset) -> dict[tuple[str, int], int]:
    return {
        (str(group.scenario_id), int(group.checkpoint_idx)): int(idx)
        for idx, group in enumerate(subset.groups)
    }


def evaluate_group_predictions(subset: DatasetSubset, group_scores: np.ndarray) -> dict[str, Any]:
    pred = _clip01(np.asarray(group_scores, dtype=np.float64).reshape(-1))
    true = _clip01(np.asarray(subset.group_true_accuracy, dtype=np.float64).reshape(-1))
    if pred.shape[0] != true.shape[0]:
        raise ValueError(f"group score length mismatch: {pred.shape[0]} vs {true.shape[0]}")

    checkpoint_pred = checkpoint_scores_from_group_scores(subset, pred)
    checkpoint_true = checkpoint_scores_from_group_scores(subset, true)
    pred_vec = np.asarray([checkpoint_pred[name] for name in OFFICIAL_CHECKPOINTS], dtype=np.float64)
    true_vec = np.asarray([checkpoint_true[name] for name in OFFICIAL_CHECKPOINTS], dtype=np.float64)
    pred_order = sorted(OFFICIAL_CHECKPOINTS, key=lambda name: (-checkpoint_pred[name], CHECKPOINT_ORDER[name]))
    true_order = sorted(OFFICIAL_CHECKPOINTS, key=lambda name: (-checkpoint_true[name], CHECKPOINT_ORDER[name]))
    top3_hit = len(set(pred_order[:3]) & set(true_order[:3]))
    mse = float(np.mean((pred - true) ** 2))

    return {
        "scenario_rmse": float(math.sqrt(mse)),
        "scenario_brier": mse,
        "scenario_ece": expected_calibration_error(pred, true),
        "checkpoint_spearman": _safe_statistic(spearmanr(pred_vec, true_vec)),
        "checkpoint_kendall": _safe_statistic(kendalltau(pred_vec, true_vec)),
        "checkpoint_pearson": _safe_statistic(pearsonr(pred_vec, true_vec)),
        "top1_hit": int(pred_order[0] == true_order[0]),
        "top3_hit": int(top3_hit),
        "checkpoint_pred_scores": checkpoint_pred,
        "checkpoint_true_scores": checkpoint_true,
        "scenario_pred": pred,
        "scenario_true": true,
        "predicted_rank_order": pred_order,
        "true_rank_order": true_order,
        "n_groups": int(pred.shape[0]),
        "n_scenarios": int(len(subset.scenarios)),
    }


def _sample_pair_indices(
    pos_idx: np.ndarray,
    neg_idx: np.ndarray,
    max_pairs: int,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    total = int(pos_idx.shape[0] * neg_idx.shape[0])
    if total <= max_pairs:
        left = np.repeat(pos_idx, neg_idx.shape[0])
        right = np.tile(neg_idx, pos_idx.shape[0])
        return left.astype(np.int64), right.astype(np.int64)
    flat = rng.choice(total, size=int(max_pairs), replace=False)
    left = pos_idx[flat // neg_idx.shape[0]]
    right = neg_idx[flat % neg_idx.shape[0]]
    return left.astype(np.int64), right.astype(np.int64)


def build_pairwise_examples(
    subset: DatasetSubset,
    *,
    max_pairs_per_group: int,
    random_state: int,
    binary_targets: bool,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(int(random_state))
    pair_x_parts: list[np.ndarray] = []
    pair_y_parts: list[np.ndarray] = []
    n_feat = int(subset.x.shape[1])
    for group in subset.groups:
        rows = group.row_indices
        y_group = subset.y[rows]
        pos_idx_local = np.flatnonzero(y_group > 0)
        neg_idx_local = np.flatnonzero(y_group <= 0)
        if pos_idx_local.size <= 0 or neg_idx_local.size <= 0:
            continue
        left_local, right_local = _sample_pair_indices(
            pos_idx_local,
            neg_idx_local,
            int(max_pairs_per_group),
            rng,
        )
        pos_rows = rows[left_local]
        neg_rows = rows[right_local]
        diffs_pos = subset.x[pos_rows] - subset.x[neg_rows]
        diffs_neg = subset.x[neg_rows] - subset.x[pos_rows]
        pair_x_parts.append(np.concatenate([diffs_pos, diffs_neg], axis=0))
        if binary_targets:
            pair_y_parts.append(
                np.concatenate(
                    [
                        np.ones(diffs_pos.shape[0], dtype=np.int32),
                        np.zeros(diffs_neg.shape[0], dtype=np.int32),
                    ],
                    axis=0,
                )
            )
        else:
            pair_y_parts.append(
                np.concatenate(
                    [
                        np.ones(diffs_pos.shape[0], dtype=np.int32),
                        -np.ones(diffs_neg.shape[0], dtype=np.int32),
                    ],
                    axis=0,
                )
            )
    if not pair_x_parts:
        return np.zeros((0, n_feat), dtype=np.float64), np.zeros((0,), dtype=np.int32)
    return (
        np.concatenate(pair_x_parts, axis=0).astype(np.float64, copy=False),
        np.concatenate(pair_y_parts, axis=0).astype(np.int32, copy=False),
    )


class BaseHead:
    def fit(self, train: DatasetSubset) -> None:
        raise NotImplementedError

    def predict_group_scores(self, subset: DatasetSubset) -> np.ndarray:
        raise NotImplementedError


class PointwiseEstimatorHead(BaseHead):
    def __init__(self, estimator, *, output_mode: str) -> None:
        self.output_mode = str(output_mode)
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )

    def fit(self, train: DatasetSubset) -> None:
        self.pipeline.fit(train.x, train.y)

    def _predict_run_values(self, x: np.ndarray) -> np.ndarray:
        if self.output_mode == "proba":
            return np.asarray(self.pipeline.predict_proba(x)[:, 1], dtype=np.float64)
        if self.output_mode == "predict":
            return _clip01(np.asarray(self.pipeline.predict(x), dtype=np.float64))
        if self.output_mode == "decision_sigmoid":
            raw = np.asarray(self.pipeline.decision_function(x), dtype=np.float64)
            return _sigmoid_clip(raw)
        raise ValueError(f"Unsupported output_mode: {self.output_mode}")

    def predict_group_scores(self, subset: DatasetSubset) -> np.ndarray:
        return aggregate_group_scores(subset, self._predict_run_values(subset.x))


class PairwiseUtilityHead(BaseHead):
    def __init__(self, estimator, *, binary_targets: bool, max_pairs_per_group: int, random_state: int) -> None:
        self.binary_targets = bool(binary_targets)
        self.max_pairs_per_group = int(max_pairs_per_group)
        self.random_state = int(random_state)
        self.scaler = StandardScaler()
        self.estimator = estimator

    def fit(self, train: DatasetSubset) -> None:
        x_scaled = self.scaler.fit_transform(train.x)
        scaled_train = DatasetSubset(
            x=x_scaled,
            y=train.y,
            scenario_ids=train.scenario_ids,
            checkpoint_idx=train.checkpoint_idx,
            checkpoint_names=train.checkpoint_names,
            row_to_group=train.row_to_group,
            group_sizes=train.group_sizes,
            group_true_accuracy=train.group_true_accuracy,
            groups=train.groups,
            trajectory_group_indices=train.trajectory_group_indices,
            checkpoint_group_indices=train.checkpoint_group_indices,
            scenarios=train.scenarios,
        )
        pair_x, pair_y = build_pairwise_examples(
            scaled_train,
            max_pairs_per_group=self.max_pairs_per_group,
            random_state=self.random_state,
            binary_targets=self.binary_targets,
        )
        if pair_x.shape[0] <= 0:
            raise ValueError("No valid pairwise examples found")
        self.estimator.fit(pair_x, pair_y)

    def _run_utility(self, x: np.ndarray) -> np.ndarray:
        x_scaled = self.scaler.transform(x)
        if hasattr(self.estimator, "decision_function"):
            raw = np.asarray(self.estimator.decision_function(x_scaled), dtype=np.float64)
        elif hasattr(self.estimator, "coef_"):
            raw = np.asarray(x_scaled @ np.asarray(self.estimator.coef_).reshape(-1), dtype=np.float64)
        else:
            raw = np.asarray(self.estimator.predict(x_scaled), dtype=np.float64)
        return _sigmoid_clip(raw.reshape(-1))

    def predict_group_scores(self, subset: DatasetSubset) -> np.ndarray:
        return aggregate_group_scores(subset, self._run_utility(subset.x))


class LogisticGroupBase:
    def __init__(self, config: dict[str, Any]) -> None:
        class_weight = config.get("class_weight")
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=float(config["C"]),
                        class_weight=None if class_weight in (None, "none") else str(class_weight),
                        max_iter=4000,
                        solver="lbfgs",
                        random_state=int(config.get("random_state", 42)),
                    ),
                ),
            ]
        )

    def fit(self, train: DatasetSubset) -> None:
        self.pipeline.fit(train.x, train.y)

    def predict_run_proba(self, subset: DatasetSubset) -> np.ndarray:
        return np.asarray(self.pipeline.predict_proba(subset.x)[:, 1], dtype=np.float64)

    def predict_group_raw(self, subset: DatasetSubset) -> np.ndarray:
        return aggregate_group_scores(subset, self.predict_run_proba(subset))


class OrdinalLogit1D:
    def __init__(self, n_bins: int) -> None:
        self.n_bins = int(n_bins)
        self.params_: np.ndarray | None = None
        self.bin_edges_: np.ndarray | None = None
        self.bin_centers_: np.ndarray | None = None

    def _encode(self, y: np.ndarray) -> np.ndarray:
        if self.bin_edges_ is None:
            raise RuntimeError("OrdinalLogit1D.fit() must be called before encode")
        labels = np.digitize(y, self.bin_edges_[1:-1], right=False).astype(np.int32)
        return np.clip(labels, 0, self.n_bins - 1)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
        y_arr = _clip01(np.asarray(y, dtype=np.float64).reshape(-1))
        quantiles = np.linspace(0.0, 1.0, self.n_bins + 1, dtype=np.float64)
        edges = np.quantile(y_arr, quantiles)
        edges[0] = 0.0
        edges[-1] = 1.0
        for idx in range(1, edges.shape[0]):
            if edges[idx] <= edges[idx - 1]:
                edges[idx] = min(1.0, edges[idx - 1] + 1e-3)
        self.bin_edges_ = np.clip(edges, 0.0, 1.0)
        self.bin_centers_ = 0.5 * (self.bin_edges_[:-1] + self.bin_edges_[1:])
        y_cls = self._encode(y_arr)

        def unpack(params: np.ndarray) -> tuple[float, np.ndarray]:
            beta = float(params[0])
            raw = np.asarray(params[1:], dtype=np.float64)
            theta = np.zeros(self.n_bins - 1, dtype=np.float64)
            if raw.size > 0:
                theta[0] = raw[0]
                for j in range(1, raw.size):
                    theta[j] = theta[j - 1] + math.log1p(math.exp(raw[j]))
            return beta, theta

        def objective(params: np.ndarray) -> float:
            beta, theta = unpack(params)
            eta = beta * x_arr
            cdf = np.zeros((x_arr.shape[0], self.n_bins + 1), dtype=np.float64)
            cdf[:, 0] = 0.0
            cdf[:, -1] = 1.0
            for j in range(self.n_bins - 1):
                cdf[:, j + 1] = expit(theta[j] - eta)
            probs = np.clip(cdf[:, 1:] - cdf[:, :-1], 1e-8, 1.0)
            ll = -np.mean(np.log(probs[np.arange(y_cls.shape[0]), y_cls]))
            ll += 1e-3 * beta * beta
            return float(ll)

        init = np.zeros(self.n_bins, dtype=np.float64)
        res = minimize(objective, init, method="L-BFGS-B", options={"maxiter": 500})
        self.params_ = np.asarray(res.x, dtype=np.float64)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.params_ is None or self.bin_centers_ is None:
            raise RuntimeError("OrdinalLogit1D.fit() must be called before predict")
        x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
        beta = float(self.params_[0])
        raw = np.asarray(self.params_[1:], dtype=np.float64)
        theta = np.zeros(self.n_bins - 1, dtype=np.float64)
        if raw.size > 0:
            theta[0] = raw[0]
            for j in range(1, raw.size):
                theta[j] = theta[j - 1] + math.log1p(math.exp(raw[j]))
        eta = beta * x_arr
        cdf = np.zeros((x_arr.shape[0], self.n_bins + 1), dtype=np.float64)
        cdf[:, 0] = 0.0
        cdf[:, -1] = 1.0
        for j in range(self.n_bins - 1):
            cdf[:, j + 1] = expit(theta[j] - eta)
        probs = np.clip(cdf[:, 1:] - cdf[:, :-1], 1e-8, 1.0)
        return np.asarray(probs @ self.bin_centers_, dtype=np.float64)


class CalibratedLogisticHead(BaseHead):
    def __init__(self, logistic_config: dict[str, Any], calibrator_kind: str, config: dict[str, Any]) -> None:
        self.base = LogisticGroupBase(logistic_config)
        self.calibrator_kind = str(calibrator_kind)
        self.config = dict(config)
        self.affine_: tuple[float, float] | None = None
        self.calibrator: Any = None

    def fit(self, train: DatasetSubset) -> None:
        self.base.fit(train)
        raw = self.base.predict_group_raw(train)
        target = np.asarray(train.group_true_accuracy, dtype=np.float64)
        if self.calibrator_kind == "ordinal_logit":
            calibrator = OrdinalLogit1D(n_bins=int(self.config["n_bins"]))
            calibrator.fit(raw, target)
            self.calibrator = calibrator
            return
        if self.calibrator_kind == "isotonic":
            calibrator = IsotonicRegression(increasing=True, out_of_bounds="clip")
            calibrator.fit(raw, target)
            self.calibrator = calibrator
            return
        if self.calibrator_kind == "affine_isotonic":
            x_design = np.column_stack([raw, np.ones(raw.shape[0], dtype=np.float64)])
            reg = float(self.config.get("ridge_alpha", 1e-6))
            gram = x_design.T @ x_design + reg * np.eye(2, dtype=np.float64)
            rhs = x_design.T @ target
            coeff = np.linalg.solve(gram, rhs)
            self.affine_ = (float(coeff[0]), float(coeff[1]))
            affine_raw = coeff[0] * raw + coeff[1]
            calibrator = IsotonicRegression(increasing=True, out_of_bounds="clip")
            calibrator.fit(affine_raw, target)
            self.calibrator = calibrator
            return
        raise ValueError(f"Unsupported calibrator kind: {self.calibrator_kind}")

    def predict_group_scores(self, subset: DatasetSubset) -> np.ndarray:
        raw = self.base.predict_group_raw(subset)
        if self.calibrator_kind == "affine_isotonic":
            if self.affine_ is None:
                raise RuntimeError("Affine calibrator missing")
            raw = self.affine_[0] * raw + self.affine_[1]
        return _clip01(np.asarray(self.calibrator.predict(raw), dtype=np.float64))


class CompositeLinearHead(BaseHead):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = dict(config)
        self.scaler = StandardScaler()
        self.params_: np.ndarray | None = None

    def fit(self, train: DatasetSubset) -> None:
        x = self.scaler.fit_transform(train.x)
        y = np.asarray(train.y, dtype=np.float64)
        row_to_group = np.asarray(train.row_to_group, dtype=np.int32)
        group_sizes = np.asarray(train.group_sizes, dtype=np.float64)
        group_true = np.asarray(train.group_true_accuracy, dtype=np.float64)
        n_groups = int(group_sizes.shape[0])
        n_feat = int(x.shape[1])

        pair_left: list[int] = []
        pair_right: list[int] = []
        pair_sign: list[float] = []
        for trajectory in train.trajectory_group_indices:
            for i in range(int(trajectory.shape[0])):
                for j in range(i + 1, int(trajectory.shape[0])):
                    left = int(trajectory[i])
                    right = int(trajectory[j])
                    diff = float(group_true[left] - group_true[right])
                    if abs(diff) < 1e-12:
                        continue
                    if diff > 0.0:
                        pair_left.append(left)
                        pair_right.append(right)
                        pair_sign.append(1.0)
                    else:
                        pair_left.append(right)
                        pair_right.append(left)
                        pair_sign.append(1.0)
        pair_left_arr = np.asarray(pair_left, dtype=np.int32)
        pair_right_arr = np.asarray(pair_right, dtype=np.int32)
        pair_sign_arr = np.asarray(pair_sign, dtype=np.float64)

        trajectories = tuple(np.asarray(value, dtype=np.int32) for value in train.trajectory_group_indices)
        reg_lambda = float(self.config.get("reg_lambda", 1e-3))
        w_run = float(self.config.get("w_run", 0.0))
        w_rank = float(self.config.get("w_rank", 0.0))
        w_cal = float(self.config.get("w_cal", 0.0))
        w_smooth = float(self.config.get("w_smooth", 0.0))
        smooth_kind = str(self.config.get("smooth_kind", "none"))
        tv_eps = float(self.config.get("tv_eps", 1e-4))

        def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
            w = np.asarray(params[:-1], dtype=np.float64)
            bias = float(params[-1])
            logits = x @ w + bias
            prob = expit(logits)
            grad_logits = np.zeros_like(prob)
            loss = 0.0

            group_sum = np.bincount(row_to_group, weights=prob, minlength=n_groups).astype(np.float64)
            group_pred = group_sum / np.maximum(1.0, group_sizes)
            grad_group = np.zeros(n_groups, dtype=np.float64)

            if w_run > 0.0:
                run_loss = np.mean(np.logaddexp(0.0, logits) - y * logits)
                loss += w_run * float(run_loss)
                grad_logits += w_run * (prob - y) / max(1.0, float(y.shape[0]))

            if w_cal > 0.0:
                diff = group_pred - group_true
                cal_loss = np.mean(diff * diff)
                loss += w_cal * float(cal_loss)
                grad_group += w_cal * (2.0 * diff / max(1.0, float(n_groups)))

            if w_rank > 0.0 and pair_left_arr.size > 0:
                pair_margin = group_pred[pair_left_arr] - group_pred[pair_right_arr]
                rank_loss = np.mean(np.logaddexp(0.0, -pair_sign_arr * pair_margin))
                loss += w_rank * float(rank_loss)
                grad_margin = w_rank * (-pair_sign_arr * expit(-pair_sign_arr * pair_margin)) / max(
                    1.0, float(pair_margin.shape[0])
                )
                np.add.at(grad_group, pair_left_arr, grad_margin)
                np.add.at(grad_group, pair_right_arr, -grad_margin)

            if w_smooth > 0.0 and smooth_kind != "none":
                diffs_total = 0
                for trajectory in trajectories:
                    if trajectory.shape[0] <= 1:
                        continue
                    left = trajectory[:-1]
                    right = trajectory[1:]
                    diff = group_pred[left] - group_pred[right]
                    diffs_total += int(diff.shape[0])
                    if smooth_kind == "smooth_l2":
                        loss += w_smooth * float(np.mean(diff * diff))
                        grad_diff = w_smooth * (2.0 * diff / max(1.0, float(diff.shape[0])))
                    elif smooth_kind == "tv":
                        denom = np.sqrt(diff * diff + tv_eps)
                        loss += w_smooth * float(np.mean(denom))
                        grad_diff = w_smooth * (diff / np.maximum(1e-8, denom)) / max(1.0, float(diff.shape[0]))
                    elif smooth_kind == "weak_monotone":
                        positive = diff > 0.0
                        penalty = np.where(positive, diff * diff, 0.0)
                        loss += w_smooth * float(np.mean(penalty))
                        grad_diff = w_smooth * np.where(positive, 2.0 * diff, 0.0) / max(
                            1.0, float(diff.shape[0])
                        )
                    else:
                        raise ValueError(f"Unsupported smooth kind: {smooth_kind}")
                    np.add.at(grad_group, left, grad_diff)
                    np.add.at(grad_group, right, -grad_diff)

            if np.any(grad_group):
                grad_prob = grad_group[row_to_group] / np.maximum(1.0, group_sizes[row_to_group])
                grad_logits += grad_prob * prob * (1.0 - prob)

            loss += 0.5 * reg_lambda * float(np.dot(w, w))
            grad_w = x.T @ grad_logits + reg_lambda * w
            grad_b = float(np.sum(grad_logits))
            grad = np.concatenate([grad_w, np.asarray([grad_b], dtype=np.float64)], axis=0)
            return float(loss), grad

        init = np.zeros(n_feat + 1, dtype=np.float64)
        res = minimize(
            lambda params: objective(params)[0],
            init,
            jac=lambda params: objective(params)[1],
            method="L-BFGS-B",
            options={"maxiter": int(self.config.get("max_iter", 300)), "maxls": 50},
        )
        self.params_ = np.asarray(res.x, dtype=np.float64)

    def predict_group_scores(self, subset: DatasetSubset) -> np.ndarray:
        if self.params_ is None:
            raise RuntimeError("CompositeLinearHead.fit() must be called before predict_group_scores()")
        x = self.scaler.transform(subset.x)
        w = np.asarray(self.params_[:-1], dtype=np.float64)
        bias = float(self.params_[-1])
        prob = expit(x @ w + bias)
        return aggregate_group_scores(subset, prob)


def build_logistic_head(config: dict[str, Any], context: dict[str, Any]) -> BaseHead:
    class_weight = config.get("class_weight")
    estimator = LogisticRegression(
        C=float(config["C"]),
        class_weight=None if class_weight in (None, "none") else str(class_weight),
        max_iter=4000,
        solver="lbfgs",
        random_state=int(context["random_state"]),
    )
    return PointwiseEstimatorHead(estimator, output_mode="proba")


def build_ridge_head(config: dict[str, Any], context: dict[str, Any]) -> BaseHead:
    estimator = Ridge(alpha=float(config["alpha"]))
    return PointwiseEstimatorHead(estimator, output_mode="predict")


def build_huber_head(config: dict[str, Any], context: dict[str, Any]) -> BaseHead:
    estimator = HuberRegressor(
        alpha=float(config["alpha"]),
        epsilon=float(config["epsilon"]),
        max_iter=200,
    )
    return PointwiseEstimatorHead(estimator, output_mode="predict")


def build_elasticnet_head(config: dict[str, Any], context: dict[str, Any]) -> BaseHead:
    estimator = ElasticNet(
        alpha=float(config["alpha"]),
        l1_ratio=float(config["l1_ratio"]),
        max_iter=1500,
        tol=1e-3,
        random_state=int(context["random_state"]),
    )
    return PointwiseEstimatorHead(estimator, output_mode="predict")


def build_linear_svr_head(config: dict[str, Any], context: dict[str, Any]) -> BaseHead:
    estimator = LinearSVR(
        C=float(config["C"]),
        epsilon=float(config["epsilon"]),
        max_iter=3000,
        random_state=int(context["random_state"]),
    )
    return PointwiseEstimatorHead(estimator, output_mode="predict")


def build_bradley_terry_head(config: dict[str, Any], context: dict[str, Any]) -> BaseHead:
    estimator = LogisticRegression(
        C=float(config["C"]),
        fit_intercept=False,
        max_iter=4000,
        solver="lbfgs",
        random_state=int(context["random_state"]),
    )
    return PairwiseUtilityHead(
        estimator,
        binary_targets=True,
        max_pairs_per_group=int(config["max_pairs_per_group"]),
        random_state=int(context["random_state"]),
    )


def build_ranksvm_head(config: dict[str, Any], context: dict[str, Any]) -> BaseHead:
    estimator = LinearSVC(
        C=float(config["C"]),
        loss="hinge",
        dual=True,
        fit_intercept=False,
        max_iter=20000,
        random_state=int(context["random_state"]),
    )
    return PairwiseUtilityHead(
        estimator,
        binary_targets=False,
        max_pairs_per_group=int(config["max_pairs_per_group"]),
        random_state=int(context["random_state"]),
    )


def build_pairwise_hinge_head(config: dict[str, Any], context: dict[str, Any]) -> BaseHead:
    estimator = LinearSVC(
        C=float(config["C"]),
        loss="squared_hinge",
        dual="auto",
        fit_intercept=False,
        max_iter=20000,
        random_state=int(context["random_state"]),
    )
    return PairwiseUtilityHead(
        estimator,
        binary_targets=False,
        max_pairs_per_group=int(config["max_pairs_per_group"]),
        random_state=int(context["random_state"]),
    )


def build_pairwise_checkpoint_bce_head(config: dict[str, Any], context: dict[str, Any]) -> BaseHead:
    merged = dict(config)
    merged.setdefault("w_run", 0.25)
    merged.setdefault("w_rank", 1.0)
    merged.setdefault("w_cal", 0.0)
    merged.setdefault("w_smooth", 0.0)
    merged.setdefault("smooth_kind", "none")
    return CompositeLinearHead(merged)


def build_cumulative_link_head(config: dict[str, Any], context: dict[str, Any]) -> BaseHead:
    return CalibratedLogisticHead(context["best_logistic_config"], "ordinal_logit", config)


def build_isotonic_head(config: dict[str, Any], context: dict[str, Any]) -> BaseHead:
    return CalibratedLogisticHead(context["best_logistic_config"], "isotonic", config)


def build_affine_isotonic_head(config: dict[str, Any], context: dict[str, Any]) -> BaseHead:
    return CalibratedLogisticHead(context["best_logistic_config"], "affine_isotonic", config)


def build_smoothness_head(config: dict[str, Any], context: dict[str, Any]) -> BaseHead:
    merged = dict(config)
    merged.setdefault("w_run", 1.0)
    merged.setdefault("w_rank", 0.0)
    merged.setdefault("w_cal", 0.5)
    merged["smooth_kind"] = "smooth_l2"
    return CompositeLinearHead(merged)


def build_tv_head(config: dict[str, Any], context: dict[str, Any]) -> BaseHead:
    merged = dict(config)
    merged.setdefault("w_run", 1.0)
    merged.setdefault("w_rank", 0.0)
    merged.setdefault("w_cal", 0.5)
    merged["smooth_kind"] = "tv"
    return CompositeLinearHead(merged)


def build_weak_monotone_head(config: dict[str, Any], context: dict[str, Any]) -> BaseHead:
    merged = dict(config)
    merged.setdefault("w_run", 1.0)
    merged.setdefault("w_rank", 0.0)
    merged.setdefault("w_cal", 0.5)
    merged["smooth_kind"] = "weak_monotone"
    return CompositeLinearHead(merged)


def build_multi_objective_head(config: dict[str, Any], context: dict[str, Any]) -> BaseHead:
    merged = dict(config)
    merged.setdefault("smooth_kind", "smooth_l2")
    return CompositeLinearHead(merged)


def get_head_specs() -> tuple[HeadSpec, ...]:
    pair_cfgs = ({"C": 1.0, "max_pairs_per_group": 96},)
    return (
        HeadSpec(
            name="logistic_regression",
            family="pointwise",
            configs=tuple(
                {"C": c_value, "class_weight": class_weight}
                for c_value in (0.25, 1.0)
                for class_weight in ("none", "balanced")
            ),
            builder=build_logistic_head,
        ),
        HeadSpec(
            name="ridge_regression",
            family="pointwise",
            configs=({"alpha": 1.0},),
            builder=build_ridge_head,
        ),
        HeadSpec(
            name="huber_regression",
            family="pointwise",
            configs=({"alpha": 1e-4, "epsilon": 1.35},),
            builder=build_huber_head,
        ),
        HeadSpec(
            name="elastic_net",
            family="pointwise",
            configs=(
                {"alpha": 1e-4, "l1_ratio": 0.5},
                {"alpha": 1e-3, "l1_ratio": 0.8},
            ),
            builder=build_elasticnet_head,
        ),
        HeadSpec(
            name="linear_svr",
            family="pointwise",
            configs=({"C": 1.0, "epsilon": 0.05},),
            builder=build_linear_svr_head,
        ),
        HeadSpec(
            name="bradley_terry_pairwise_logistic",
            family="pairwise",
            configs=pair_cfgs,
            builder=build_bradley_terry_head,
        ),
        HeadSpec(
            name="ranksvm",
            family="pairwise",
            configs=pair_cfgs,
            builder=build_ranksvm_head,
        ),
        HeadSpec(
            name="pairwise_hinge",
            family="pairwise",
            configs=pair_cfgs,
            builder=build_pairwise_hinge_head,
        ),
        HeadSpec(
            name="pairwise_bce_checkpoint_pairs",
            family="pairwise",
            configs=(
                {"w_run": 0.25, "w_rank": 1.0, "w_cal": 0.0, "reg_lambda": 1e-3},
                {"w_run": 0.25, "w_rank": 2.0, "w_cal": 0.5, "reg_lambda": 1e-3},
            ),
            builder=build_pairwise_checkpoint_bce_head,
        ),
        HeadSpec(
            name="cumulative_link_ordinal_logit",
            family="ordinal",
            configs=({"n_bins": 5},),
            builder=build_cumulative_link_head,
        ),
        HeadSpec(
            name="isotonic_calibrator",
            family="ordinal",
            configs=({"kind": "isotonic"},),
            builder=build_isotonic_head,
        ),
        HeadSpec(
            name="affine_plus_isotonic",
            family="ordinal",
            configs=({"ridge_alpha": 1e-6}, {"ridge_alpha": 1e-2}),
            builder=build_affine_isotonic_head,
        ),
        HeadSpec(
            name="smoothness_regularized_linear",
            family="trajectory",
            configs=({"w_smooth": 0.1, "reg_lambda": 1e-3}, {"w_smooth": 0.5, "reg_lambda": 1e-3}),
            builder=build_smoothness_head,
        ),
        HeadSpec(
            name="fused_lasso_tv_linear",
            family="trajectory",
            configs=(
                {"w_smooth": 0.02, "reg_lambda": 1e-3, "tv_eps": 1e-4},
                {"w_smooth": 0.05, "reg_lambda": 1e-3, "tv_eps": 1e-4},
            ),
            builder=build_tv_head,
        ),
        HeadSpec(
            name="weak_monotone_linear",
            family="trajectory",
            configs=({"w_smooth": 0.1, "reg_lambda": 1e-3}, {"w_smooth": 0.5, "reg_lambda": 1e-3}),
            builder=build_weak_monotone_head,
        ),
        HeadSpec(
            name="multi_objective_linear",
            family="multi_objective",
            configs=(
                {"w_run": 1.0, "w_rank": 0.5, "w_cal": 0.5, "w_smooth": 0.1, "reg_lambda": 1e-3},
                {"w_run": 1.0, "w_rank": 2.0, "w_cal": 1.0, "w_smooth": 0.25, "reg_lambda": 1e-2},
            ),
            builder=build_multi_objective_head,
        ),
    )


def fit_predict_head(
    head_spec: HeadSpec,
    config: dict[str, Any],
    train: DatasetSubset,
    test: DatasetSubset,
    context: dict[str, Any],
) -> np.ndarray:
    model = head_spec.builder(config, context)
    model.fit(train)
    return model.predict_group_scores(test)


def select_best_config(
    head_spec: HeadSpec,
    train: DatasetSubset,
    context: dict[str, Any],
    *,
    inner_splits: int,
    random_state: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    splits = build_group_splits(train.scenarios, int(inner_splits))
    if not splits or len(head_spec.configs) == 1:
        cfg = dict(head_spec.configs[0])
        predictions = fit_predict_head(head_spec, cfg, train, train, context)
        metrics = evaluate_group_predictions(train, predictions)
        return cfg, metrics
    split_train_idx, split_val_idx = splits[0]
    split_train_scenarios = [train.scenarios[int(idx)] for idx in split_train_idx.tolist()]
    split_val_scenarios = [train.scenarios[int(idx)] for idx in split_val_idx.tolist()]
    split_train = subset_by_scenarios(train, split_train_scenarios)
    split_val = subset_by_scenarios(train, split_val_scenarios)

    best_config: dict[str, Any] | None = None
    best_metrics: dict[str, Any] | None = None
    for cfg in head_spec.configs:
        inner_context = dict(context)
        inner_context["random_state"] = int(random_state)
        try:
            preds = fit_predict_head(head_spec, dict(cfg), split_train, split_val, inner_context)
        except Exception:
            continue
        metrics = evaluate_group_predictions(split_val, preds)
        if best_metrics is None or _score_tuple(metrics) > _score_tuple(best_metrics):
            best_config = dict(cfg)
            best_metrics = metrics
    if best_config is None or best_metrics is None:
        best_config = dict(head_spec.configs[0])
        predictions = fit_predict_head(head_spec, best_config, train, train, context)
        best_metrics = evaluate_group_predictions(train, predictions)
    return best_config, best_metrics


def select_best_logistic_config(
    train: DatasetSubset,
    *,
    inner_splits: int,
    random_state: int,
) -> dict[str, Any]:
    logistic_spec = next(spec for spec in get_head_specs() if spec.name == "logistic_regression")
    best_config, _ = select_best_config(
        logistic_spec,
        train,
        {"random_state": int(random_state), "best_logistic_config": {"C": 1.0, "class_weight": "balanced"}},
        inner_splits=inner_splits,
        random_state=random_state,
    )
    return best_config


def run_nested_oof(
    dataset: DatasetSubset,
    *,
    outer_splits: int,
    inner_splits: int,
    random_state: int,
) -> dict[str, Any]:
    head_specs = get_head_specs()
    full_lookup = build_group_lookup(dataset)
    results: dict[str, dict[str, Any]] = {
        spec.name: {
            "family": spec.family,
            "group_oof": np.full(len(dataset.groups), np.nan, dtype=np.float64),
            "fold_records": [],
            "selected_configs": [],
        }
        for spec in head_specs
    }

    outer_splits_list = build_group_splits(dataset.scenarios, int(outer_splits))
    for fold_idx, (train_s_idx, test_s_idx) in enumerate(outer_splits_list):
        train_scenarios = [dataset.scenarios[int(idx)] for idx in train_s_idx.tolist()]
        test_scenarios = [dataset.scenarios[int(idx)] for idx in test_s_idx.tolist()]
        print(
            f"[outer {fold_idx + 1}/{len(outer_splits_list)}] train={len(train_scenarios)} test={len(test_scenarios)}",
            flush=True,
        )
        train = subset_by_scenarios(dataset, train_scenarios)
        test = subset_by_scenarios(dataset, test_scenarios)
        best_logistic_config = select_best_logistic_config(
            train,
            inner_splits=inner_splits,
            random_state=int(random_state + fold_idx * 100),
        )
        base_context = {
            "random_state": int(random_state + fold_idx),
            "best_logistic_config": dict(best_logistic_config),
        }
        for head_spec in head_specs:
            print(f"  - {head_spec.name}", flush=True)
            best_cfg, inner_metrics = select_best_config(
                head_spec,
                train,
                base_context,
                inner_splits=inner_splits,
                random_state=int(random_state + fold_idx * 1000),
            )
            preds = fit_predict_head(head_spec, best_cfg, train, test, base_context)
            fold_metrics = evaluate_group_predictions(test, preds)
            results[head_spec.name]["selected_configs"].append(
                {
                    "fold": int(fold_idx),
                    "config": dict(best_cfg),
                    "inner_checkpoint_spearman": inner_metrics["checkpoint_spearman"],
                    "inner_checkpoint_kendall": inner_metrics["checkpoint_kendall"],
                    "inner_scenario_rmse": inner_metrics["scenario_rmse"],
                }
            )
            results[head_spec.name]["fold_records"].append(
                {
                    "fold": int(fold_idx),
                    "metrics": fold_metrics,
                    "config": dict(best_cfg),
                    "n_scenarios": int(len(test.scenarios)),
                }
            )
            for local_group_idx, group in enumerate(test.groups):
                global_group_idx = full_lookup[(str(group.scenario_id), int(group.checkpoint_idx))]
                if global_group_idx is None:
                    raise RuntimeError("Failed to map test group back to full dataset")
                results[head_spec.name]["group_oof"][global_group_idx] = float(preds[local_group_idx])

    for head_spec in head_specs:
        oof = results[head_spec.name]["group_oof"]
        if np.any(~np.isfinite(oof)):
            raise RuntimeError(f"Incomplete OOF predictions for head={head_spec.name}")
        overall_metrics = evaluate_group_predictions(dataset, oof)
        results[head_spec.name]["overall_metrics"] = overall_metrics

    return results


def write_oof_csv(path: Path, dataset: DatasetSubset, results: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "head_name",
                "family",
                "row_type",
                "fold",
                "checkpoint",
                "checkpoint_idx",
                "oof_checkpoint_score",
                "true_checkpoint_accuracy",
                "checkpoint_spearman",
                "checkpoint_kendall",
                "checkpoint_pearson",
                "top1_hit",
                "top3_hit",
                "scenario_rmse",
                "scenario_brier",
                "scenario_ece",
                "selected_config",
            ],
        )
        writer.writeheader()
        for head_name, payload in results.items():
            metrics = payload["overall_metrics"]
            checkpoint_pred = metrics["checkpoint_pred_scores"]
            checkpoint_true = metrics["checkpoint_true_scores"]
            selected = payload["selected_configs"]
            for checkpoint_name in OFFICIAL_CHECKPOINTS:
                writer.writerow(
                    {
                        "head_name": head_name,
                        "family": payload["family"],
                        "row_type": "overall_checkpoint",
                        "fold": "overall",
                        "checkpoint": checkpoint_name,
                        "checkpoint_idx": int(CHECKPOINT_ORDER[checkpoint_name]),
                        "oof_checkpoint_score": float(checkpoint_pred[checkpoint_name]),
                        "true_checkpoint_accuracy": float(checkpoint_true[checkpoint_name]),
                        "checkpoint_spearman": metrics["checkpoint_spearman"],
                        "checkpoint_kendall": metrics["checkpoint_kendall"],
                        "checkpoint_pearson": metrics["checkpoint_pearson"],
                        "top1_hit": metrics["top1_hit"],
                        "top3_hit": metrics["top3_hit"],
                        "scenario_rmse": metrics["scenario_rmse"],
                        "scenario_brier": metrics["scenario_brier"],
                        "scenario_ece": metrics["scenario_ece"],
                        "selected_config": json.dumps(selected, ensure_ascii=False, default=_json_default),
                    }
                )
            for fold_record in payload["fold_records"]:
                fold_metrics = fold_record["metrics"]
                for checkpoint_name in OFFICIAL_CHECKPOINTS:
                    writer.writerow(
                        {
                            "head_name": head_name,
                            "family": payload["family"],
                            "row_type": "fold_checkpoint",
                            "fold": int(fold_record["fold"]),
                            "checkpoint": checkpoint_name,
                            "checkpoint_idx": int(CHECKPOINT_ORDER[checkpoint_name]),
                            "oof_checkpoint_score": float(fold_metrics["checkpoint_pred_scores"][checkpoint_name]),
                            "true_checkpoint_accuracy": float(fold_metrics["checkpoint_true_scores"][checkpoint_name]),
                            "checkpoint_spearman": fold_metrics["checkpoint_spearman"],
                            "checkpoint_kendall": fold_metrics["checkpoint_kendall"],
                            "checkpoint_pearson": fold_metrics["checkpoint_pearson"],
                            "top1_hit": fold_metrics["top1_hit"],
                            "top3_hit": fold_metrics["top3_hit"],
                            "scenario_rmse": fold_metrics["scenario_rmse"],
                            "scenario_brier": fold_metrics["scenario_brier"],
                            "scenario_ece": fold_metrics["scenario_ece"],
                            "selected_config": json.dumps(fold_record["config"], ensure_ascii=False, default=_json_default),
                        }
                    )


def plot_calibration_curves(path: Path, results: dict[str, Any]) -> None:
    ordered_heads = sorted(
        results.keys(),
        key=lambda name: (
            -float(results[name]["overall_metrics"]["checkpoint_spearman"] or float("-inf")),
            -float(results[name]["overall_metrics"]["checkpoint_kendall"] or float("-inf")),
            name,
        ),
    )
    n_heads = len(ordered_heads)
    n_cols = 4
    n_rows = int(math.ceil(n_heads / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4.2 * n_rows), squeeze=False)
    for ax in axes.ravel():
        ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="lightgray", linewidth=1.0)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Predicted scenario-checkpoint accuracy")
        ax.set_ylabel("Observed scenario-checkpoint accuracy")
    for ax, head_name in zip(axes.ravel(), ordered_heads):
        metrics = results[head_name]["overall_metrics"]
        x_vals, y_vals, counts = reliability_curve(metrics["scenario_pred"], metrics["scenario_true"], n_bins=8)
        if x_vals.size > 0:
            ax.plot(x_vals, y_vals, marker="o", linewidth=1.5)
            ax.scatter(x_vals, y_vals, s=20 + 1.2 * counts, alpha=0.7)
        ax.set_title(
            f"{head_name}\n"
            f"ρ={metrics['checkpoint_spearman']:.3f}  "
            f"τ={metrics['checkpoint_kendall']:.3f}  "
            f"RMSE={metrics['scenario_rmse']:.3f}"
        )
    for ax in axes.ravel()[n_heads:]:
        ax.axis("off")
    fig.suptitle("Checkpoint-ranking linear heads: OOF calibration curves", fontsize=15)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_summary_table(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for head_name, payload in results.items():
        metrics = payload["overall_metrics"]
        config_counter = Counter(json.dumps(item["config"], sort_keys=True, default=_json_default) for item in payload["selected_configs"])
        most_common_config = config_counter.most_common(1)[0][0] if config_counter else "{}"
        rows.append(
            {
                "head_name": head_name,
                "family": payload["family"],
                "checkpoint_spearman": metrics["checkpoint_spearman"],
                "checkpoint_kendall": metrics["checkpoint_kendall"],
                "checkpoint_pearson": metrics["checkpoint_pearson"],
                "top1_hit": metrics["top1_hit"],
                "top3_hit": metrics["top3_hit"],
                "scenario_rmse": metrics["scenario_rmse"],
                "scenario_brier": metrics["scenario_brier"],
                "scenario_ece": metrics["scenario_ece"],
                "predicted_rank_order": " > ".join(metrics["predicted_rank_order"]),
                "true_rank_order": " > ".join(metrics["true_rank_order"]),
                "representative_config": most_common_config,
            }
        )
    rows.sort(
        key=lambda item: (
            float(item["checkpoint_spearman"] if item["checkpoint_spearman"] is not None else float("-inf")),
            float(item["checkpoint_kendall"] if item["checkpoint_kendall"] is not None else float("-inf")),
            float(item["top1_hit"]),
            float(item["top3_hit"]),
            -float(item["scenario_rmse"]),
        ),
        reverse=True,
    )
    return rows


def write_report(path: Path, dataset: DatasetSubset, results: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary_rows = build_summary_table(results)
    winner = summary_rows[0]
    checkpoint_true = checkpoint_scores_from_group_scores(dataset, dataset.group_true_accuracy)
    lines: list[str] = []
    lines.append("# Linear Heads for Checkpoint Ranking\n")
    lines.append("## Summary\n")
    lines.append(
        f"- Data: `{DEFAULT_FEATURE_STORE.relative_to(REPO_ROOT)}` with "
        f"{len(dataset.scenarios)} scenarios × {len(OFFICIAL_CHECKPOINTS)} checkpoints × "
        f"{int(dataset.y.shape[0] / max(1, len(dataset.scenarios) * len(OFFICIAL_CHECKPOINTS)))} runs.\n"
    )
    lines.append(
        "- OOF protocol: 5-fold GroupKFold by scenario_id; hyperparameters chosen with a grouped inner validation split on the training scenarios.\n"
    )
    lines.append("- Primary decision rule: checkpoint ranking first, calibration second.\n")
    lines.append(
        f"- Recommended ensemble head: `{winner['head_name']}` "
        f"(ρ={winner['checkpoint_spearman']:.3f}, τ={winner['checkpoint_kendall']:.3f}, "
        f"RMSE={winner['scenario_rmse']:.3f}).\n"
    )

    lines.append("\n## OOF Metrics\n")
    lines.append(
        "| Head | Family | Spearman | Kendall | Pearson | Top1 | Top3 | RMSE | Brier | ECE |\n"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for row in summary_rows:
        lines.append(
            f"| `{row['head_name']}` | {row['family']} | "
            f"{row['checkpoint_spearman']:.3f} | {row['checkpoint_kendall']:.3f} | {row['checkpoint_pearson']:.3f} | "
            f"{int(row['top1_hit'])} | {int(row['top3_hit'])} | "
            f"{row['scenario_rmse']:.3f} | {row['scenario_brier']:.3f} | {row['scenario_ece']:.3f} |\n"
        )

    lines.append("\n## Calibration / Ranking Notes\n")
    for row in summary_rows[:6]:
        lines.append(
            f"- `{row['head_name']}`: predicted order `{row['predicted_rank_order']}`; "
            f"representative config `{row['representative_config']}`.\n"
        )

    lines.append("\n## True Checkpoint Accuracy\n")
    lines.append("| Checkpoint | True accuracy |\n")
    lines.append("| --- | ---: |\n")
    for checkpoint_name in OFFICIAL_CHECKPOINTS:
        lines.append(f"| `{checkpoint_name}` | {checkpoint_true[checkpoint_name]:.4f} |\n")

    lines.append("\n## Why this winner\n")
    lines.append(
        f"- `{winner['head_name']}` ranks checkpoints best under the primary OOF criterion and stays competitive on calibration.\n"
    )
    lines.append(
        "- It remains lightweight, linear or near-linear, and keeps the feature pipeline frozen.\n"
    )
    lines.append(
        "- It is therefore the safest candidate to add into the final ensemble as the checkpoint-accuracy specialist head.\n"
    )
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare lightweight linear heads for RL checkpoint ranking")
    ap.add_argument("--feature-store", default=str(DEFAULT_FEATURE_STORE))
    ap.add_argument("--outputs-dir", default=str(DEFAULT_OUTPUTS_DIR))
    ap.add_argument("--outer-splits", type=int, default=5)
    ap.add_argument("--inner-splits", type=int, default=2)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    feature_store_path = Path(args.feature_store).resolve()
    outputs_dir = Path(args.outputs_dir).resolve()
    dataset = load_feature_store(feature_store_path)
    results = run_nested_oof(
        dataset,
        outer_splits=int(args.outer_splits),
        inner_splits=int(args.inner_splits),
        random_state=int(args.random_state),
    )
    write_oof_csv(outputs_dir / "linear_heads_oof.csv", dataset, results)
    plot_calibration_curves(outputs_dir / "linear_heads_calibration.png", results)
    write_report(outputs_dir / "linear_heads_report.md", dataset, results)
    summary = build_summary_table(results)
    print(json.dumps({"winner": summary[0], "n_heads": len(summary)}, indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
