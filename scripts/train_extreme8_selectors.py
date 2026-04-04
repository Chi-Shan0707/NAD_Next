#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.core.selectors.base import SelectorContext
from nad.core.selectors.extreme8_impl import (
    DEFAULT_BAND_SIZES,
    EXTREME8_FEATURE_NAMES,
    LinearRankModel,
    ZeroRankModel,
    band_reward_bounds,
    build_extreme8_features,
    extract_extreme8_raw_values,
    normalize_weight_direction,
    sample_tuple_indices,
    accumulate_extreme8_scores,
)
from nad.core.selectors.trajectory_impl import DEFAULT_REFLECTION_THRESHOLD
from nad.core.views.reader import CacheReader
from nad.ops.accuracy import _load_ground_truth
from nad.ops.bestofn_extreme8 import (
    CacheEntry,
    CacheScores,
    discover_cache_entries,
    evaluate_cache_scores,
    mean_metric_bundle,
)

DEFAULT_DATASETS = (
    "aime24",
    "aime25",
    "brumo25",
    "gpqa",
    "hmmt25",
    "livecodebench_v5",
)
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
OBJECTIVE_CHOICES = (
    "pointwise",
    "band_reward",
    "aggregated_selacc10",
)
DEFAULT_TUNE_SPLIT = 0.20
DEFAULT_NUM_TUPLES_TUNE = 256
DEFAULT_HIT1_GUARDRAIL_DROP = 0.01
DEFAULT_PAIRWISE_GUARDRAIL_DROP = 0.005


def _set_single_thread_env() -> None:
    for key in THREAD_ENV_VARS:
        os.environ.setdefault(key, "1")


_set_single_thread_env()


def _fit_logistic(X: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", class_weight="balanced")),
    ])
    pipe.fit(X, y)
    return pipe


def _payload_problem_block_key(payload: dict[str, Any]) -> tuple[str, str]:
    return str(payload["dataset_name"]), str(payload["problem_id"])


def _summarize_payload_partition(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    problem_blocks = {_payload_problem_block_key(payload) for payload in payloads}
    per_dataset_problem_blocks: dict[str, set[tuple[str, str]]] = {}
    per_dataset_payloads: dict[str, int] = {}
    for payload in payloads:
        dataset_name = str(payload["dataset_name"])
        key = _payload_problem_block_key(payload)
        per_dataset_problem_blocks.setdefault(dataset_name, set()).add(key)
        per_dataset_payloads[dataset_name] = int(per_dataset_payloads.get(dataset_name, 0) + 1)
    return {
        "payloads": int(len(payloads)),
        "problem_blocks": int(len(problem_blocks)),
        "per_dataset_problem_blocks": {
            dataset_name: int(len(keys))
            for dataset_name, keys in sorted(per_dataset_problem_blocks.items())
        },
        "per_dataset_payloads": {
            dataset_name: int(count)
            for dataset_name, count in sorted(per_dataset_payloads.items())
        },
    }


def _split_payloads_for_tuning(
    payloads: list[dict[str, Any]],
    tune_split: float,
    tune_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not payloads:
        summary = {
            "tune_split": float(tune_split),
            "tune_seed": int(tune_seed),
            "degenerate": True,
            "fit": _summarize_payload_partition([]),
            "tune": _summarize_payload_partition([]),
        }
        return [], [], summary

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for payload in payloads:
        dataset_name = str(payload["dataset_name"])
        problem_id = str(payload["problem_id"])
        grouped.setdefault(dataset_name, {}).setdefault(problem_id, []).append(payload)

    fit_payloads: list[dict[str, Any]] = []
    tune_payloads: list[dict[str, Any]] = []
    for dataset_name, by_problem in sorted(grouped.items()):
        problem_ids = sorted(by_problem.keys(), key=_problem_sort_key)
        tune_problem_ids: set[str] = set()
        if len(problem_ids) > 1:
            rng = np.random.RandomState(int(tune_seed) + _stable_hash(dataset_name))
            order = rng.permutation(len(problem_ids))
            n_tune = int(round(len(problem_ids) * float(tune_split)))
            n_tune = max(1, n_tune)
            n_tune = min(len(problem_ids) - 1, n_tune)
            tune_problem_ids = {str(problem_ids[int(idx)]) for idx in order[:n_tune]}

        for problem_id in problem_ids:
            target = tune_payloads if str(problem_id) in tune_problem_ids else fit_payloads
            target.extend(by_problem[problem_id])

    degenerate = not fit_payloads or not tune_payloads
    if degenerate:
        fit_payloads = list(payloads)
        tune_payloads = list(payloads)

    summary = {
        "tune_split": float(tune_split),
        "tune_seed": int(tune_seed),
        "degenerate": bool(degenerate),
        "fit": _summarize_payload_partition(fit_payloads),
        "tune": _summarize_payload_partition(tune_payloads),
    }
    return fit_payloads, tune_payloads, summary


def _selector_accuracy(model, groups: list[tuple[np.ndarray, np.ndarray]]) -> float:
    correct = total = 0
    for X_g, y_g in groups:
        probs = model.predict_proba(X_g)[:, 1]
        chosen = int(np.argmax(probs))
        correct += int(y_g[chosen])
        total += 1
    return correct / total if total else 0.0


def _build_direction_candidates(
    feature_dim: int,
    seed: int,
    seed_direction: np.ndarray | None,
) -> list[np.ndarray]:
    rng = np.random.RandomState(int(seed))
    candidates: list[np.ndarray] = []
    if seed_direction is not None:
        candidates.append(normalize_weight_direction(seed_direction))
    candidates.append(normalize_weight_direction(np.ones(feature_dim, dtype=np.float64)))
    for dim in range(feature_dim):
        basis = np.zeros(feature_dim, dtype=np.float64)
        basis[dim] = 1.0
        candidates.append(basis.copy())
        candidates.append(-basis.copy())
    for _ in range(32):
        candidates.append(normalize_weight_direction(rng.normal(size=feature_dim)))
    return candidates


def _resolve_threshold(raw: str) -> float:
    if raw.lower() != "auto":
        return float(raw)
    summary_path = REPO_ROOT / "results" / "reflection_dynamics" / "threshold_sweep_summary.json"
    if not summary_path.exists():
        print(f"[WARN] {summary_path} not found; fallback to {DEFAULT_REFLECTION_THRESHOLD:.2f}")
        return float(DEFAULT_REFLECTION_THRESHOLD)
    data = json.loads(summary_path.read_text())
    value = data.get("best_threshold_loo")
    if value is None:
        print(f"[WARN] best_threshold_loo missing in {summary_path}; fallback to {DEFAULT_REFLECTION_THRESHOLD:.2f}")
        return float(DEFAULT_REFLECTION_THRESHOLD)
    return float(value)


def _problem_sort_key(problem_id: str) -> tuple[int, Any]:
    try:
        return (0, int(problem_id))
    except (TypeError, ValueError):
        return (1, str(problem_id))


def _build_groups(meta: dict) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for sid, sample in enumerate(meta["samples"]):
        pid = str(sample["problem_id"])
        groups.setdefault(pid, []).append(int(sid))
    return groups


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _resolve_path(raw: str | None) -> Path | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _parse_datasets(raw: str) -> list[str]:
    if not str(raw).strip():
        return list(DEFAULT_DATASETS)
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _stable_hash(text: str) -> int:
    total = 0
    for idx, ch in enumerate(text):
        total = (total + (idx + 1) * ord(ch)) % 1_000_003
    return int(total)


def _discover_entries(base_root: Path | None, requested: set[str]) -> list[Any]:
    if base_root is None:
        return []
    entries = discover_cache_entries(base_root)
    if requested:
        entries = [entry for entry in entries if entry.dataset_name in requested]
    return sorted(entries, key=lambda entry: (entry.model_name, entry.dataset_name, str(entry.cache_root)))


def _build_problem_records(
    entry,
    min_accuracy: float,
    max_accuracy: float,
) -> list[dict[str, Any]]:
    meta = json.loads((entry.cache_root / "meta.json").read_text())
    correctness = _load_ground_truth(entry.cache_root)
    groups = _build_groups(meta)
    records: list[dict[str, Any]] = []
    for problem_id, sample_ids in sorted(groups.items(), key=lambda kv: _problem_sort_key(kv[0])):
        labels = np.asarray([int(bool(correctness.get(int(sample_id), False))) for sample_id in sample_ids], dtype=np.int32)
        accuracy = float(labels.mean()) if labels.size else 0.0
        eligible = bool(
            labels.size > 0
            and accuracy >= float(min_accuracy)
            and accuracy <= float(max_accuracy)
            and int(labels.sum()) not in (0, int(labels.size))
        )
        records.append({
            "cache_root": str(entry.cache_root),
            "cache_key": str(entry.cache_key),
            "model_name": str(entry.model_name),
            "dataset_name": str(entry.dataset_name),
            "problem_id": str(problem_id),
            "sample_ids": [int(sample_id) for sample_id in sample_ids],
            "labels": labels.tolist(),
            "accuracy": float(accuracy),
            "eligible": bool(eligible),
        })
    return records


def _split_records(records: list[dict[str, Any]], val_split: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not records:
        return [], []
    if len(records) == 1:
        return [], [records[0]]
    rng = np.random.RandomState(int(seed))
    order = rng.permutation(len(records))
    n_val = int(round(len(records) * float(val_split)))
    n_val = max(1, n_val)
    n_val = min(len(records) - 1, n_val)
    val_indices = set(int(idx) for idx in order[:n_val])
    train_records = [record for idx, record in enumerate(records) if idx not in val_indices]
    val_records = [record for idx, record in enumerate(records) if idx in val_indices]
    return train_records, val_records


def _limit_records(records: list[dict[str, Any]], max_problems: int | None) -> list[dict[str, Any]]:
    if max_problems is None:
        return list(records)
    return list(records[: max(0, int(max_problems))])


def _collect_specs(args) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    requested = set(_parse_datasets(args.datasets))
    main_entries = _discover_entries(_resolve_path(args.cache_root), requested)
    extra_entries = _discover_entries(_resolve_path(args.train_extra_cache_root), requested)
    split_entries = _discover_entries(_resolve_path(args.val_cache_root), requested)

    train_specs: list[dict[str, Any]] = []
    val_all_specs: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "train_full": {},
        "train_split": {},
        "val_eligible": {},
        "val_all": {},
    }

    for entry in main_entries + extra_entries:
        records = _build_problem_records(entry, args.min_accuracy, args.max_accuracy)
        eligible_records = _limit_records([record for record in records if record["eligible"]], args.max_problems)
        train_specs.extend(eligible_records)
        summary["train_full"][entry.dataset_name] = int(len(eligible_records))

    for entry in split_entries:
        records = _build_problem_records(entry, args.min_accuracy, args.max_accuracy)
        train_records, val_records = _split_records(
            records,
            val_split=float(args.val_split),
            seed=int(args.split_seed) + _stable_hash(entry.dataset_name),
        )
        train_eligible = _limit_records([record for record in train_records if record["eligible"]], args.max_problems)
        val_all = _limit_records(val_records, args.max_problems)
        val_eligible = [record for record in val_all if record["eligible"]]
        train_specs.extend(train_eligible)
        val_all_specs.extend(val_all)
        summary["train_split"][entry.dataset_name] = int(len(train_eligible))
        summary["val_eligible"][entry.dataset_name] = int(len(val_eligible))
        summary["val_all"][entry.dataset_name] = int(len(val_all))

    train_specs = sorted(train_specs, key=lambda item: (item["dataset_name"], _problem_sort_key(item["problem_id"])))
    val_all_specs = sorted(val_all_specs, key=lambda item: (item["dataset_name"], _problem_sort_key(item["problem_id"])))
    return train_specs, val_all_specs, summary


def _bucketize_specs(specs: list[dict[str, Any]], workers: int) -> list[list[dict[str, Any]]]:
    if not specs:
        return []
    n_workers = max(1, min(int(workers), len(specs)))
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(n_workers)]
    for spec in specs:
        idx = min(range(n_workers), key=lambda pos: len(buckets[pos]))
        buckets[idx].append(spec)
    return buckets


def _extract_problem_payloads(batch: list[dict[str, Any]], reflection_threshold: float) -> list[dict[str, Any]]:
    _set_single_thread_env()
    readers: dict[str, CacheReader] = {}
    payloads: list[dict[str, Any]] = []
    for spec in batch:
        cache_root = str(spec["cache_root"])
        reader = readers.get(cache_root)
        if reader is None:
            reader = CacheReader(cache_root)
            readers[cache_root] = reader
        ctx = SelectorContext(
            cache=reader,
            problem_id=str(spec["problem_id"]),
            run_ids=[int(sample_id) for sample_id in spec["sample_ids"]],
            views=[],
        )
        raw_values = extract_extreme8_raw_values(
            ctx,
            reflection_threshold=float(reflection_threshold),
        )
        payload = dict(spec)
        payload["raw_values"] = {
            "dc_raw": np.asarray(raw_values["dc_raw"], dtype=np.float64),
            "reflection_count": np.asarray(raw_values["reflection_count"], dtype=np.float64),
        }
        payloads.append(payload)
    return payloads


def _load_problem_payloads(
    specs: list[dict[str, Any]],
    reflection_threshold: float,
    workers: int,
) -> tuple[list[dict[str, Any]], list[int]]:
    if not specs:
        return [], []
    buckets = _bucketize_specs(specs, workers)
    bucket_sizes = [len(bucket) for bucket in buckets]
    if len(buckets) == 1:
        payloads = _extract_problem_payloads(buckets[0], reflection_threshold=float(reflection_threshold))
    else:
        payloads = []
        with ProcessPoolExecutor(max_workers=len(buckets)) as ex:
            futures = [ex.submit(_extract_problem_payloads, bucket, float(reflection_threshold)) for bucket in buckets]
            for future in tqdm(futures, desc="extract", total=len(futures)):
                payloads.extend(future.result())
    payloads = sorted(payloads, key=lambda item: (item["dataset_name"], _problem_sort_key(item["problem_id"])))
    return payloads, bucket_sizes


def _iter_sampled_problem_tuples(
    payloads: list[dict[str, Any]],
    tuple_size: int,
    num_tuples: int,
    seed: int,
    tuple_min_correct: int | None,
    tuple_max_correct: int | None,
    require_mixed: bool,
):
    for payload_index, payload in enumerate(payloads):
        labels = np.asarray(payload["labels"], dtype=np.int32)
        tuples = sample_tuple_indices(
            n_runs=len(payload["sample_ids"]),
            tuple_size=int(tuple_size),
            num_tuples=int(num_tuples),
            rng=np.random.RandomState(int(seed) + payload_index),
            labels=labels,
            require_mixed=bool(require_mixed),
            min_correct=tuple_min_correct,
            max_correct=tuple_max_correct,
        )
        yield payload, labels, tuples


def _build_pointwise_training_data(
    payloads: list[dict[str, Any]],
    tuple_size: int,
    num_tuples: int,
    seed: int,
    tuple_min_correct: int | None,
    tuple_max_correct: int | None,
    require_mixed: bool,
):
    X_best_rows: list[np.ndarray] = []
    y_best_rows: list[np.ndarray] = []
    X_worst_rows: list[np.ndarray] = []
    y_worst_rows: list[np.ndarray] = []
    best_groups: list[tuple[np.ndarray, np.ndarray]] = []
    worst_groups: list[tuple[np.ndarray, np.ndarray]] = []
    tuple_total = 0

    for payload, labels, tuples in _iter_sampled_problem_tuples(
        payloads,
        tuple_size=tuple_size,
        num_tuples=num_tuples,
        seed=seed,
        tuple_min_correct=tuple_min_correct,
        tuple_max_correct=tuple_max_correct,
        require_mixed=require_mixed,
    ):
        raw_values = payload["raw_values"]
        for idx in tuples:
            feat = build_extreme8_features(raw_values, idx)
            y_best = labels[idx].astype(np.int32, copy=False)
            y_worst = (1 - y_best).astype(np.int32, copy=False)
            X_best_rows.append(feat)
            y_best_rows.append(y_best)
            X_worst_rows.append(feat)
            y_worst_rows.append(y_worst)
            best_groups.append((feat, y_best))
            worst_groups.append((feat, y_worst))
            tuple_total += 1

    if not X_best_rows:
        raise SystemExit("No eligible tuples were collected for pointwise training.")

    return {
        "X_best": np.vstack(X_best_rows),
        "y_best": np.concatenate(y_best_rows),
        "X_worst": np.vstack(X_worst_rows),
        "y_worst": np.concatenate(y_worst_rows),
        "best_groups": best_groups,
        "worst_groups": worst_groups,
        "n_tuples": int(tuple_total),
    }


def _build_band_training_data(
    payloads: list[dict[str, Any]],
    tuple_size: int,
    num_tuples: int,
    seed: int,
    tuple_min_correct: int | None,
    tuple_max_correct: int | None,
    require_mixed: bool,
):
    tuple_features: list[np.ndarray] = []
    tuple_labels: list[np.ndarray] = []
    seed_X_rows: list[np.ndarray] = []
    seed_y_rows: list[np.ndarray] = []

    for payload, labels, tuples in _iter_sampled_problem_tuples(
        payloads,
        tuple_size=tuple_size,
        num_tuples=num_tuples,
        seed=seed,
        tuple_min_correct=tuple_min_correct,
        tuple_max_correct=tuple_max_correct,
        require_mixed=require_mixed,
    ):
        raw_values = payload["raw_values"]
        for idx in tuples:
            feat = build_extreme8_features(raw_values, idx)
            sub_labels = labels[idx].astype(np.int32, copy=False)
            tuple_features.append(np.asarray(feat, dtype=np.float64))
            tuple_labels.append(np.asarray(sub_labels, dtype=np.int32))
            seed_X_rows.append(np.asarray(feat, dtype=np.float64))
            seed_y_rows.append(np.asarray(sub_labels, dtype=np.int32))

    if not tuple_features:
        raise SystemExit("No eligible tuples were collected for band-reward training.")

    return {
        "tuple_features": np.stack(tuple_features, axis=0),
        "tuple_labels": np.stack(tuple_labels, axis=0),
        "seed_X": np.vstack(seed_X_rows),
        "seed_y": np.concatenate(seed_y_rows),
        "n_tuples": int(len(tuple_features)),
    }


def _extract_direction_from_logistic(model) -> np.ndarray:
    scaler = model.named_steps["sc"]
    lr = model.named_steps["lr"]
    coef = np.asarray(lr.coef_, dtype=np.float64).reshape(-1)
    scale = np.asarray(getattr(scaler, "scale_", np.ones_like(coef)), dtype=np.float64).reshape(-1)
    scale = np.where(scale == 0.0, 1.0, scale)
    return coef / scale


def _reward_lookup(alpha: float, beta: float, gamma: float, tuple_size: int) -> tuple[np.ndarray, np.ndarray]:
    best = np.zeros(tuple_size + 1, dtype=np.float64)
    worst = np.zeros(tuple_size + 1, dtype=np.float64)
    for n_correct in range(tuple_size + 1):
        best[n_correct], worst[n_correct] = band_reward_bounds(
            n_correct=n_correct,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            band_sizes=DEFAULT_BAND_SIZES,
        )
    return best, worst


def _mean_normalized_band_reward(
    tuple_features: np.ndarray,
    tuple_labels: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    reward_best: np.ndarray,
    reward_worst: np.ndarray,
) -> float:
    score_mat = np.einsum("tnf,f->tn", tuple_features, weights)
    order = np.argsort(-score_mat, axis=1, kind="mergesort")
    sorted_labels = np.take_along_axis(tuple_labels, order, axis=1)

    start = 0
    raw_reward = np.zeros(tuple_labels.shape[0], dtype=np.float64)
    for band_weight, band_size in zip((alpha, beta, gamma), DEFAULT_BAND_SIZES):
        stop = start + int(band_size)
        raw_reward += float(band_weight) * sorted_labels[:, start:stop].sum(axis=1)
        start = stop

    n_correct = tuple_labels.sum(axis=1).astype(np.int64, copy=False)
    denom = np.maximum(reward_best[n_correct] - reward_worst[n_correct], 1e-8)
    norm_reward = (raw_reward - reward_worst[n_correct]) / denom
    return float(np.mean(norm_reward))


def _fit_band_reward_model(
    tuple_features: np.ndarray,
    tuple_labels: np.ndarray,
    seed_direction: np.ndarray | None,
    alpha: float,
    beta: float,
    gamma: float,
    seed: int,
):
    rng = np.random.RandomState(int(seed))
    reward_best, reward_worst = _reward_lookup(alpha, beta, gamma, tuple_size=int(tuple_features.shape[1]))
    feature_dim = int(tuple_features.shape[2])
    candidates = _build_direction_candidates(
        feature_dim=feature_dim,
        seed=int(seed),
        seed_direction=seed_direction,
    )

    seen: set[tuple[float, ...]] = set()
    best_score = -np.inf
    best_weights = None
    evaluated = 0

    def _evaluate(candidate: np.ndarray) -> None:
        nonlocal best_score, best_weights, evaluated
        weights = normalize_weight_direction(candidate)
        key = tuple(np.round(weights, 8).tolist())
        if key in seen:
            return
        seen.add(key)
        score = _mean_normalized_band_reward(
            tuple_features=tuple_features,
            tuple_labels=tuple_labels,
            weights=weights,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            reward_best=reward_best,
            reward_worst=reward_worst,
        )
        evaluated += 1
        if score > best_score:
            best_score = float(score)
            best_weights = weights

    for candidate in candidates:
        _evaluate(candidate)

    assert best_weights is not None
    center = np.asarray(best_weights, dtype=np.float64)
    for sigma, n_samples in ((0.25, 32), (0.10, 24), (0.05, 24)):
        _evaluate(center)
        for _ in range(n_samples):
            _evaluate(center + float(sigma) * rng.normal(size=feature_dim))
        center = np.asarray(best_weights, dtype=np.float64)

    model = LinearRankModel(weights=np.asarray(best_weights, dtype=np.float64))
    return model, {
        "training_mean_band_reward": float(best_score),
        "search_candidates_evaluated": int(evaluated),
        "linear_weights": np.asarray(best_weights, dtype=np.float64).tolist(),
    }


def _best_only_mean_metrics(validation: dict[str, Any]) -> dict[str, float | None]:
    metrics = validation.get("mean", {}).get("best_only", {})
    return {
        key: (None if value is None else float(value))
        for key, value in metrics.items()
    }


def _metric_rank_key(metrics: dict[str, float | None]) -> tuple[float, float, float, float]:
    def _value(name: str) -> float:
        value = metrics.get(name)
        if value is None:
            return float("-inf")
        value = float(value)
        if not np.isfinite(value):
            return float("-inf")
        return value

    return (
        _value("selective_acc_at_10pct"),
        _value("hit_at_3"),
        _value("hit_at_1"),
        _value("pairwise_accuracy"),
    )


def _passes_guardrails(
    metrics: dict[str, float | None],
    baseline_metrics: dict[str, float | None],
    hit1_drop: float = DEFAULT_HIT1_GUARDRAIL_DROP,
    pairwise_drop: float = DEFAULT_PAIRWISE_GUARDRAIL_DROP,
) -> bool:
    hit1 = metrics.get("hit_at_1")
    baseline_hit1 = baseline_metrics.get("hit_at_1")
    if baseline_hit1 is not None:
        if hit1 is None:
            return False
        if float(hit1) < float(baseline_hit1) - float(hit1_drop):
            return False

    pairwise = metrics.get("pairwise_accuracy")
    baseline_pairwise = baseline_metrics.get("pairwise_accuracy")
    if baseline_pairwise is not None:
        if pairwise is None:
            return False
        if float(pairwise) < float(baseline_pairwise) - float(pairwise_drop):
            return False

    return True


def _fit_aggregated_selacc10_model(
    payloads: list[dict[str, Any]],
    tuple_size: int,
    num_tuples: int,
    num_tuples_tune: int,
    seed: int,
    tuple_min_correct: int | None,
    tuple_max_correct: int | None,
    require_mixed: bool,
    tune_split: float,
    tune_seed: int,
    hit1_guardrail_drop: float = DEFAULT_HIT1_GUARDRAIL_DROP,
    pairwise_guardrail_drop: float = DEFAULT_PAIRWISE_GUARDRAIL_DROP,
):
    fit_payloads, tune_payloads, tune_summary = _split_payloads_for_tuning(
        payloads,
        tune_split=float(tune_split),
        tune_seed=int(tune_seed),
    )

    fit_data = _build_pointwise_training_data(
        fit_payloads,
        tuple_size=int(tuple_size),
        num_tuples=int(num_tuples),
        seed=int(seed),
        tuple_min_correct=tuple_min_correct,
        tuple_max_correct=tuple_max_correct,
        require_mixed=require_mixed,
    )
    best_seed_model = _fit_logistic(fit_data["X_best"], fit_data["y_best"])
    worst_baseline_model = _fit_logistic(fit_data["X_worst"], fit_data["y_worst"])
    seed_direction = _extract_direction_from_logistic(best_seed_model)

    baseline_validation = _evaluate_payloads(
        tune_payloads,
        best_model=best_seed_model,
        worst_model=worst_baseline_model,
        tuple_size=int(tuple_size),
        num_tuples=int(num_tuples_tune),
        seed=int(seed) + 5_000_000,
    )
    baseline_metrics = _best_only_mean_metrics(baseline_validation)

    feature_dim = int(len(seed_direction))
    candidates = _build_direction_candidates(
        feature_dim=feature_dim,
        seed=int(seed),
        seed_direction=seed_direction,
    )
    rng = np.random.RandomState(int(seed))
    seen: set[tuple[float, ...]] = set()
    search_trace: list[dict[str, Any]] = []
    best_pass = None
    best_any = None
    selected_weights = None

    def _update_best(current_best, candidate_row: dict[str, Any]):
        if current_best is None:
            return candidate_row
        if _metric_rank_key(candidate_row["metrics"]) > _metric_rank_key(current_best["metrics"]):
            return candidate_row
        return current_best

    def _evaluate(candidate: np.ndarray) -> None:
        nonlocal best_pass, best_any, selected_weights
        weights = normalize_weight_direction(candidate)
        key = tuple(np.round(weights, 8).tolist())
        if key in seen:
            return
        seen.add(key)
        best_model = LinearRankModel(weights=weights)
        validation = _evaluate_payloads(
            tune_payloads,
            best_model=best_model,
            worst_model=ZeroRankModel(),
            tuple_size=int(tuple_size),
            num_tuples=int(num_tuples_tune),
            seed=int(seed) + 6_000_000,
        )
        metrics = _best_only_mean_metrics(validation)
        passes_guardrails = _passes_guardrails(
            metrics,
            baseline_metrics=baseline_metrics,
            hit1_drop=float(hit1_guardrail_drop),
            pairwise_drop=float(pairwise_guardrail_drop),
        )
        candidate_row = {
            "weights": np.asarray(weights, dtype=np.float64).tolist(),
            "metrics": metrics,
            "passes_guardrails": bool(passes_guardrails),
        }
        search_trace.append(candidate_row)
        best_any = _update_best(best_any, candidate_row)
        if passes_guardrails:
            best_pass = _update_best(best_pass, candidate_row)

    for candidate in candidates:
        _evaluate(candidate)

    center = None if best_any is None else np.asarray(best_any["weights"], dtype=np.float64)
    if center is not None:
        for sigma, n_samples in ((0.25, 32), (0.10, 24), (0.05, 24)):
            _evaluate(center)
            for _ in range(n_samples):
                _evaluate(center + float(sigma) * rng.normal(size=feature_dim))
            chosen = best_pass if best_pass is not None else best_any
            center = np.asarray(chosen["weights"], dtype=np.float64)

    selected = best_pass if best_pass is not None else best_any
    assert selected is not None
    selected_weights = np.asarray(selected["weights"], dtype=np.float64)

    best_model = LinearRankModel(weights=selected_weights)
    worst_model = ZeroRankModel()
    return best_model, worst_model, {
        "selection_objective": "aggregated_selacc10",
        "seed_logistic_direction": normalize_weight_direction(seed_direction).tolist(),
        "n_seed_training_tuples": int(fit_data["n_tuples"]),
        "search_candidates_evaluated": int(len(search_trace)),
        "linear_weights": selected_weights.tolist(),
        "tune_summary": tune_summary,
        "num_tuples_tune": int(num_tuples_tune),
        "baseline_inner_tune_metrics": baseline_metrics,
        "selected_candidate": selected,
        "guardrails": {
            "hit1_drop_max": float(hit1_guardrail_drop),
            "pairwise_drop_max": float(pairwise_guardrail_drop),
            "n_passed": int(sum(1 for row in search_trace if bool(row["passes_guardrails"]))),
        },
        "search_trace": search_trace,
    }


def _score_result_to_dict(result) -> dict[str, Any]:
    return {
        "auroc": None if result.auroc is None else float(result.auroc),
        "hit_at_1": float(result.hit_at_1),
        "hit_at_3": float(result.hit_at_3),
        "selective_acc_at_10pct": float(result.selective_acc_at_10pct),
        "pairwise_accuracy": None if result.pairwise_accuracy is None else float(result.pairwise_accuracy),
        "n_problems": int(result.n_problems),
        "n_samples": int(result.n_samples),
        "n_positive": int(result.n_positive),
        "n_negative": int(result.n_negative),
    }


def _evaluate_payloads(
    payloads: list[dict[str, Any]],
    best_model,
    worst_model,
    tuple_size: int,
    num_tuples: int,
    seed: int,
) -> dict[str, Any]:
    if not payloads:
        return {
            "mean": {},
            "per_dataset": {},
            "n_bundles": 0,
            "n_problems": 0,
        }

    grouped: dict[str, dict[str, Any]] = {}
    for payload_index, payload in enumerate(payloads):
        cache_root = str(payload["cache_root"])
        bucket = grouped.setdefault(cache_root, {
            "entry": payload,
            "problem_scores": {},
        })
        result = bucket["problem_scores"]
        raw_values = payload["raw_values"]
        score_payload = accumulate_extreme8_scores(
            best_model=best_model,
            worst_model=worst_model,
            raw_values=raw_values,
            tuple_size=int(tuple_size),
            num_tuples=int(num_tuples),
            seed=int(seed) + payload_index,
            labels=None,
            require_mixed=False,
        )
        sample_ids = [int(sample_id) for sample_id in payload["sample_ids"]]
        result[str(payload["problem_id"])] = {
            "sample_ids": sample_ids,
            "run_indices": sample_ids,
            "score_best": np.asarray(score_payload["score_best"], dtype=np.float64).tolist(),
            "score_worst": np.asarray(score_payload["score_worst"], dtype=np.float64).tolist(),
            "score_mix": np.asarray(score_payload["score_mix"], dtype=np.float64).tolist(),
            "counts": np.asarray(score_payload["counts"], dtype=np.float64).tolist(),
            "num_tuples": int(np.asarray(score_payload["num_tuples"], dtype=np.int64)[0]),
        }

    bundles = []
    per_dataset: dict[str, dict[str, Any]] = {}
    for cache_root, bucket in sorted(grouped.items(), key=lambda kv: kv[0]):
        entry_payload = bucket["entry"]
        cache_scores = CacheScores(
            entry=CacheEntry(
                cache_key=entry_payload["cache_key"],
                cache_root=Path(cache_root),
                model_name=entry_payload["model_name"],
                dataset_name=entry_payload["dataset_name"],
            ),
            problem_scores=bucket["problem_scores"],
        )
        correctness = _load_ground_truth(Path(cache_root))
        bundle = evaluate_cache_scores(cache_scores, correctness)
        bundles.append(bundle)
        per_dataset[entry_payload["dataset_name"]] = {
            score_name: _score_result_to_dict(result)
            for score_name, result in bundle.by_score.items()
        }

    mean_bundle = mean_metric_bundle(bundles)
    return {
        "mean": {
            score_name: {
                metric_name: (None if metric_value is None else float(metric_value))
                for metric_name, metric_value in metrics.items()
            }
            for score_name, metrics in mean_bundle.by_score.items()
        },
        "per_dataset": per_dataset,
        "n_bundles": int(len(bundles)),
        "n_problems": int(sum(len(bucket["problem_scores"]) for bucket in grouped.values())),
    }


def train_selector_artifacts(
    *,
    train_payloads: list[dict[str, Any]],
    val_all_payloads: list[dict[str, Any]],
    split_summary: dict[str, Any],
    train_bucket_sizes: list[int],
    val_bucket_sizes: list[int],
    reflection_threshold: float,
    out_dir: Path,
    artifact_prefix: str,
    objective: str,
    tuple_size: int,
    num_tuples: int,
    num_tuples_val: int,
    tuple_min_correct: int | None,
    tuple_max_correct: int | None,
    datasets: list[str],
    split_seed: int,
    val_split: float,
    alpha: float,
    beta: float,
    gamma: float,
    workers: int,
    seed: int,
    tune_split: float = DEFAULT_TUNE_SPLIT,
    tune_seed: int = 42,
    num_tuples_tune: int = DEFAULT_NUM_TUPLES_TUNE,
) -> dict[str, Any]:
    if not train_payloads:
        raise SystemExit("No eligible training problems were found.")

    val_eligible_payloads = [payload for payload in val_all_payloads if bool(payload.get("eligible", False))]
    require_mixed = bool(objective == "pointwise" or tuple_min_correct is not None or tuple_max_correct is not None)

    if objective == "band_reward" and int(tuple_size) != int(sum(DEFAULT_BAND_SIZES)):
        raise SystemExit(f"band_reward requires tuple_size={sum(DEFAULT_BAND_SIZES)}")

    if objective == "pointwise":
        train_data = _build_pointwise_training_data(
            train_payloads,
            tuple_size=int(tuple_size),
            num_tuples=int(num_tuples),
            seed=int(seed),
            tuple_min_correct=tuple_min_correct,
            tuple_max_correct=tuple_max_correct,
            require_mixed=require_mixed,
        )
        best_model = _fit_logistic(train_data["X_best"], train_data["y_best"])
        worst_model = _fit_logistic(train_data["X_worst"], train_data["y_worst"])
        objective_stats = {
            "best_training_tuple_accuracy": float(_selector_accuracy(best_model, train_data["best_groups"])),
            "worst_training_error_hit": float(_selector_accuracy(worst_model, train_data["worst_groups"])),
            "n_training_tuples": int(train_data["n_tuples"]),
            "best_positive_rate": float(train_data["y_best"].mean()),
            "worst_positive_rate": float(train_data["y_worst"].mean()),
        }
    elif objective == "band_reward":
        train_data = _build_band_training_data(
            train_payloads,
            tuple_size=int(tuple_size),
            num_tuples=int(num_tuples),
            seed=int(seed),
            tuple_min_correct=tuple_min_correct,
            tuple_max_correct=tuple_max_correct,
            require_mixed=require_mixed,
        )
        seed_model = _fit_logistic(train_data["seed_X"], train_data["seed_y"])
        seed_direction = _extract_direction_from_logistic(seed_model)
        best_model, band_stats = _fit_band_reward_model(
            tuple_features=train_data["tuple_features"],
            tuple_labels=train_data["tuple_labels"],
            seed_direction=seed_direction,
            alpha=float(alpha),
            beta=float(beta),
            gamma=float(gamma),
            seed=int(seed),
        )
        worst_model = ZeroRankModel()
        objective_stats = {
            "seed_logistic_direction": normalize_weight_direction(seed_direction).tolist(),
            "n_training_tuples": int(train_data["n_tuples"]),
            **band_stats,
        }
    elif objective == "aggregated_selacc10":
        best_model, worst_model, objective_stats = _fit_aggregated_selacc10_model(
            payloads=train_payloads,
            tuple_size=int(tuple_size),
            num_tuples=int(num_tuples),
            num_tuples_tune=int(num_tuples_tune),
            seed=int(seed),
            tuple_min_correct=tuple_min_correct,
            tuple_max_correct=tuple_max_correct,
            require_mixed=require_mixed,
            tune_split=float(tune_split),
            tune_seed=int(tune_seed),
        )
    else:
        raise KeyError(f"Unknown objective: {objective}")

    from joblib import dump

    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / f"{artifact_prefix}_best.pkl"
    worst_path = out_dir / f"{artifact_prefix}_worst.pkl"
    dump(best_model, best_path)
    dump(worst_model, worst_path)

    validation_eligible = _evaluate_payloads(
        val_eligible_payloads,
        best_model=best_model,
        worst_model=worst_model,
        tuple_size=int(tuple_size),
        num_tuples=int(num_tuples_val),
        seed=int(seed) + 7_000_000,
    )
    validation_all = _evaluate_payloads(
        val_all_payloads,
        best_model=best_model,
        worst_model=worst_model,
        tuple_size=int(tuple_size),
        num_tuples=int(num_tuples_val),
        seed=int(seed) + 8_000_000,
    )

    stats = {
        "feature_names": EXTREME8_FEATURE_NAMES,
        "objective": str(objective),
        "artifact_prefix": str(artifact_prefix),
        "reflection_threshold": float(reflection_threshold),
        "tuple_size": int(tuple_size),
        "num_tuples_per_problem": int(num_tuples),
        "num_tuples_validation": int(num_tuples_val),
        "num_tuples_tune": int(num_tuples_tune),
        "tuple_min_correct": None if tuple_min_correct is None else int(tuple_min_correct),
        "tuple_max_correct": None if tuple_max_correct is None else int(tuple_max_correct),
        "seed": int(seed),
        "workers": int(workers),
        "datasets": list(datasets),
        "split_seed": int(split_seed),
        "val_split": float(val_split),
        "tune_split": float(tune_split),
        "tune_seed": int(tune_seed),
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
        "problem_counts": {
            "train": split_summary,
            "train_total": int(len(train_payloads)),
            "val_total": int(len(val_all_payloads)),
            "val_eligible_total": int(len(val_eligible_payloads)),
        },
        "worker_buckets": {
            "train": [int(x) for x in train_bucket_sizes],
            "validation": [int(x) for x in val_bucket_sizes],
        },
        "objective_stats": objective_stats,
        "validation": {
            "eligible": validation_eligible,
            "all": validation_all,
        },
        "model_paths": {
            "best": _display_path(best_path),
            "worst": _display_path(worst_path),
        },
    }
    stats_path = out_dir / f"{artifact_prefix}_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    return {
        "best_model": best_model,
        "worst_model": worst_model,
        "best_path": best_path,
        "worst_path": worst_path,
        "stats": stats,
        "stats_path": stats_path,
    }


def main():
    ap = argparse.ArgumentParser(description="Train Extreme8 / Extreme12 selectors with optional ranking-search objectives")
    ap.add_argument("--datasets", default=",".join(DEFAULT_DATASETS), help="Comma-separated datasets")
    ap.add_argument("--cache-root", default="MUI_HUB/cache", help="Fully included training cache root")
    ap.add_argument("--train-extra-cache-root", default="", help="Optional extra training cache root without splitting")
    ap.add_argument("--val-cache-root", default="", help="Optional cache root to split into train/validate")
    ap.add_argument("--val-split", type=float, default=0.20, help="Validation fraction for --val-cache-root")
    ap.add_argument("--split-seed", type=int, default=42, help="Seed for validation split")
    ap.add_argument("--out", default="models/ml_selectors", help="Output directory")
    ap.add_argument("--artifact-prefix", default="extreme8", help="Artifact filename prefix")
    ap.add_argument("--objective", choices=OBJECTIVE_CHOICES, default="pointwise", help="Training objective")
    ap.add_argument("--tuple-size", type=int, default=8, help="Tuple size")
    ap.add_argument("--num-tuples", type=int, default=256, help="Training tuples per eligible problem")
    ap.add_argument("--num-tuples-val", type=int, default=1024, help="Blind validation tuples per problem")
    ap.add_argument("--num-tuples-tune", type=int, default=DEFAULT_NUM_TUPLES_TUNE, help="Blind tuning tuples per problem for aggregated_selacc10")
    ap.add_argument("--tuple-min-correct", type=int, default=None, help="Optional min correct count for training tuples")
    ap.add_argument("--tuple-max-correct", type=int, default=None, help="Optional max correct count for training tuples")
    ap.add_argument("--min-accuracy", type=float, default=0.10, help="Minimum problem accuracy to keep")
    ap.add_argument("--max-accuracy", type=float, default=0.90, help="Maximum problem accuracy to keep")
    ap.add_argument("--reflection-threshold", default=f"{DEFAULT_REFLECTION_THRESHOLD:.2f}", help="Float threshold or 'auto'")
    ap.add_argument("--alpha", type=float, default=1.0, help="Top-band reward weight")
    ap.add_argument("--beta", type=float, default=0.35, help="Middle-band reward weight")
    ap.add_argument("--gamma", type=float, default=-0.50, help="Bottom-band reward weight")
    ap.add_argument("--workers", type=int, default=4, help="Parallel workers for raw-value extraction")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--tune-split", type=float, default=DEFAULT_TUNE_SPLIT, help="Inner tuning fraction for aggregated_selacc10")
    ap.add_argument("--tune-seed", type=int, default=42, help="Inner tuning split seed for aggregated_selacc10")
    ap.add_argument("--max-problems", type=int, default=None, help="Optional cap per dataset/split for smoke tests")
    args = ap.parse_args()

    reflection_threshold = _resolve_threshold(str(args.reflection_threshold))
    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(REPO_ROOT)

    train_specs, val_all_specs, split_summary = _collect_specs(args)
    if not train_specs:
        raise SystemExit("No eligible training problems were found.")

    print(f"Collecting raw values for {len(train_specs)} training problems")
    train_payloads, train_bucket_sizes = _load_problem_payloads(
        train_specs,
        reflection_threshold=reflection_threshold,
        workers=int(args.workers),
    )
    print(f"Collecting raw values for {len(val_all_specs)} validation problems")
    val_all_payloads, val_bucket_sizes = _load_problem_payloads(
        val_all_specs,
        reflection_threshold=reflection_threshold,
        workers=int(args.workers),
    )
    result = train_selector_artifacts(
        train_payloads=train_payloads,
        val_all_payloads=val_all_payloads,
        split_summary=split_summary,
        train_bucket_sizes=train_bucket_sizes,
        val_bucket_sizes=val_bucket_sizes,
        reflection_threshold=float(reflection_threshold),
        out_dir=out_dir,
        artifact_prefix=str(args.artifact_prefix),
        objective=str(args.objective),
        tuple_size=int(args.tuple_size),
        num_tuples=int(args.num_tuples),
        num_tuples_val=int(args.num_tuples_val),
        tuple_min_correct=args.tuple_min_correct,
        tuple_max_correct=args.tuple_max_correct,
        datasets=_parse_datasets(args.datasets),
        split_seed=int(args.split_seed),
        val_split=float(args.val_split),
        alpha=float(args.alpha),
        beta=float(args.beta),
        gamma=float(args.gamma),
        workers=int(args.workers),
        seed=int(args.seed),
        tune_split=float(args.tune_split),
        tune_seed=int(args.tune_seed),
        num_tuples_tune=int(args.num_tuples_tune),
    )

    print(f"Saved best model to {result['best_path']}")
    print(f"Saved worst model to {result['worst_path']}")
    print(f"Saved stats to {result['stats_path']}")
    validation_eligible = result["stats"].get("validation", {}).get("eligible", {})
    if validation_eligible.get("mean"):
        mean_best = validation_eligible["mean"].get("best_only", {})
        if mean_best:
            print(
                "Validation eligible best_only: "
                f"Hit@1={mean_best.get('hit_at_1', 0.0):.4f} "
                f"Hit@3={mean_best.get('hit_at_3', 0.0):.4f} "
                f"Pairwise={mean_best.get('pairwise_accuracy', 0.0) or 0.0:.4f} "
                f"SelAcc@10={mean_best.get('selective_acc_at_10pct', 0.0):.4f}"
            )


if __name__ == "__main__":
    main()
