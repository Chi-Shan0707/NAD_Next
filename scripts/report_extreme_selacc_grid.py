#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.core.distance.engine import DistanceEngine, DistanceSpec
from nad.core.selectors.base import SelectorContext
from nad.core.selectors.code_dynamic_impl import (
    DEFAULT_CODE_DYNAMIC_REFLECTION_LOOKBACK,
    DEFAULT_CODE_DYNAMIC_REFLECTION_THRESHOLD,
    compute_code_dynamic_primary_scores_from_raw,
    extract_code_dynamic_raw_from_state,
    order_code_dynamic_group_indices,
    prepare_code_dynamic_run_state,
)
from nad.core.selectors.extreme8_impl import accumulate_extreme8_scores, extract_extreme8_raw_values
from nad.core.selectors.science_dynamic_impl import (
    DEFAULT_SCIENCE_DYNAMIC_WEIGHTS,
    DEFAULT_SCIENCE_PREFIX_FRACTION,
    DEFAULT_SCIENCE_RECENCY_EXP,
    DEFAULT_SCIENCE_TAIL_FRACTION,
    DEFAULT_SCIENCE_WINDOW_TOKENS,
    compute_science_dynamic_primary_scores_from_raw,
    extract_science_dynamic_raw_from_state,
    prepare_science_dynamic_run_state,
)
from nad.core.views.reader import CacheReader
from nad.ops.accuracy import _load_ground_truth
from nad.ops.bestofn_extreme8 import (
    build_problem_groups,
    discover_cache_entries,
    load_model,
    score_cache_entry,
)
from scripts.run_code_baseline_v1_phase2 import DEFAULT_VIEW

DOMAIN_ORDER = ("math", "science", "coding")
DOMAIN_LABELS = {
    "math": "Math",
    "science": "Science",
    "coding": "Coding",
}
PERCENT_GRID_DEFAULT = tuple(range(5, 101, 5))
MATH_DATASETS = {"aime25", "brumo25", "hmmt25"}
SCIENCE_DATASETS = {"gpqa"}
CODING_DATASETS = {"lcb_v5", "livecodebench_v5"}
GENERIC_METHOD = "baseline12_pointwise"
SCIENCE_METHOD = "science_baseline_v1"
CODE_METHOD = "code_baseline_v1"
_MODEL_CACHE: dict[str, Any] = {}
_READER_CACHE: dict[str, CacheReader] = {}
_CORRECTNESS_CACHE: dict[str, dict[int, bool]] = {}
_ENGINE_CACHE: dict[int, DistanceEngine] = {}


def _problem_sort_key(problem_id: str) -> tuple[int, str]:
    try:
        return (0, f"{int(str(problem_id)):09d}")
    except Exception:
        return (1, str(problem_id))


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _get_cached_model(path: str | Path):
    key = str(path)
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = load_model(Path(path))
        _MODEL_CACHE[key] = model
    return model


def _get_cached_reader(cache_root: str | Path) -> CacheReader:
    key = str(cache_root)
    reader = _READER_CACHE.get(key)
    if reader is None:
        reader = CacheReader(key)
        _READER_CACHE[key] = reader
    return reader


def _get_cached_correctness(cache_root: str | Path) -> dict[int, bool]:
    key = str(cache_root)
    correctness = _CORRECTNESS_CACHE.get(key)
    if correctness is None:
        correctness = _load_ground_truth(Path(cache_root))
        _CORRECTNESS_CACHE[key] = correctness
    return correctness


def _get_cached_engine(distance_threads: int) -> DistanceEngine:
    key = int(distance_threads)
    engine = _ENGINE_CACHE.get(key)
    if engine is None:
        engine = DistanceEngine(DistanceSpec("ja", num_threads=key))
        _ENGINE_CACHE[key] = engine
    return engine


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.2f}%"


def _parse_percent_grid(raw: str) -> list[int]:
    values = [int(item.strip()) for item in str(raw).split(",") if item.strip()]
    if not values:
        raise SystemExit("Percent grid cannot be empty")
    out: list[int] = []
    for value in values:
        if value <= 0 or value > 100:
            raise SystemExit(f"Invalid percent value: {value}")
        if value not in out:
            out.append(value)
    return out


def _load_entry_map(base_root: str | Path) -> dict[str, Any]:
    return {entry.cache_key: entry for entry in discover_cache_entries(base_root)}


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stable_score_order(scores: np.ndarray, sample_ids: np.ndarray) -> np.ndarray:
    score_arr = np.asarray(scores, dtype=np.float64)
    sample_arr = np.asarray(sample_ids, dtype=np.int64)
    return np.lexsort((sample_arr, -score_arr)).astype(np.int64)


def _compute_selacc_curve_from_labels(labels: np.ndarray, percents: list[int]) -> dict[str, float]:
    arr = np.asarray(labels, dtype=np.int32).reshape(-1)
    n = int(arr.size)
    out: dict[str, float] = {}
    if n <= 0:
        return {str(p): 0.0 for p in percents}
    for percent in percents:
        topk = max(1, int(math.ceil(float(percent) * n / 100.0)))
        out[str(percent)] = float(arr[:topk].mean())
    return out


def _compute_global_selacc_curve(records: list[dict[str, Any]], percents: list[int]) -> dict[str, float]:
    if not records:
        return {str(p): 0.0 for p in percents}
    ordered = sorted(
        records,
        key=lambda row: (
            -float(row["score"]),
            str(row["cache_key"]),
            _problem_sort_key(str(row["problem_id"])),
            int(row["sample_id"]),
        ),
    )
    labels = np.asarray([int(row["label"]) for row in ordered], dtype=np.int32)
    return _compute_selacc_curve_from_labels(labels, percents)


def _compute_local_selacc_curve(problems: dict[str, dict[str, Any]], percents: list[int]) -> dict[str, float]:
    per_problem_values: dict[str, list[float]] = {str(p): [] for p in percents}
    for _, payload in sorted(problems.items(), key=lambda kv: _problem_sort_key(kv[0])):
        labels = np.asarray(payload["labels"], dtype=np.int32)
        order = np.asarray(payload["order"], dtype=np.int64)
        if labels.size <= 0 or order.size <= 0:
            continue
        ordered_labels = labels[order]
        curve = _compute_selacc_curve_from_labels(ordered_labels, percents)
        for percent in percents:
            per_problem_values[str(percent)].append(float(curve[str(percent)]))
    out: dict[str, float] = {}
    for percent in percents:
        vals = per_problem_values[str(percent)]
        out[str(percent)] = float(np.mean(vals)) if vals else 0.0
    return out


def _build_cache_metric_payload(
    *,
    method_name: str,
    domain: str,
    cache_key: str,
    cache_root: Path,
    dataset_name: str,
    records: list[dict[str, Any]],
    problems: dict[str, dict[str, Any]],
    percents: list[int],
    source: str,
) -> dict[str, Any]:
    return {
        "method_name": str(method_name),
        "domain": str(domain),
        "cache_key": str(cache_key),
        "cache_root": str(cache_root),
        "dataset_name": str(dataset_name),
        "source": str(source),
        "n_samples": int(len(records)),
        "n_problems": int(len(problems)),
        "global_selacc": _compute_global_selacc_curve(records, percents),
        "local_selacc": _compute_local_selacc_curve(problems, percents),
    }


def _aggregate_domain_payload(
    *,
    method_name: str,
    domain: str,
    cache_payloads: list[dict[str, Any]],
    records: list[dict[str, Any]],
    problems: dict[str, dict[str, Any]],
    percents: list[int],
    note: str,
) -> dict[str, Any]:
    return {
        "method_name": str(method_name),
        "domain": str(domain),
        "note": str(note),
        "n_caches": int(len(cache_payloads)),
        "n_samples": int(len(records)),
        "n_problems": int(len(problems)),
        "global_selacc": _compute_global_selacc_curve(records, percents),
        "local_selacc": _compute_local_selacc_curve(problems, percents),
        "caches": {payload["cache_key"]: payload for payload in cache_payloads},
    }


def _problem_result_payload(
    *,
    cache_key: str,
    cache_root: str | Path,
    dataset_name: str,
    model_name: str,
    problem_id: str,
    sample_ids: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    order: np.ndarray,
) -> dict[str, Any]:
    return {
        "cache_key": str(cache_key),
        "cache_root": str(cache_root),
        "dataset_name": str(dataset_name),
        "model_name": str(model_name),
        "problem_id": str(problem_id),
        "sample_ids": np.asarray(sample_ids, dtype=np.int64).tolist(),
        "scores": np.asarray(scores, dtype=np.float64).tolist(),
        "labels": np.asarray(labels, dtype=np.int32).tolist(),
        "order": np.asarray(order, dtype=np.int64).tolist(),
    }


def _generic_problem_worker(
    *,
    cache_root: str,
    cache_key: str,
    dataset_name: str,
    model_name: str,
    problem_id: str,
    sample_ids: list[int],
    best_model_path: str,
    tuple_size: int,
    num_tuples: int,
    reflection_threshold: float,
    seed: int,
) -> dict[str, Any]:
    reader = _get_cached_reader(cache_root)
    best_model = _get_cached_model(best_model_path)
    correctness = _get_cached_correctness(cache_root)
    run_ids = list(map(int, sample_ids))
    ctx = SelectorContext(cache=reader, problem_id=str(problem_id), run_ids=run_ids, views=[])
    raw_values = extract_extreme8_raw_values(ctx, reflection_threshold=float(reflection_threshold))
    payload = accumulate_extreme8_scores(
        best_model=best_model,
        worst_model=None,
        raw_values=raw_values,
        tuple_size=int(tuple_size),
        num_tuples=int(num_tuples),
        seed=int(seed),
        labels=None,
        require_mixed=False,
    )
    sample_arr = np.asarray(run_ids, dtype=np.int64)
    score_arr = np.asarray(payload["score_best"], dtype=np.float64)
    label_arr = np.asarray([int(bool(correctness[int(sample_id)])) for sample_id in sample_arr], dtype=np.int32)
    order = _stable_score_order(score_arr, sample_arr)
    return _problem_result_payload(
        cache_key=cache_key,
        cache_root=cache_root,
        dataset_name=dataset_name,
        model_name=model_name,
        problem_id=str(problem_id),
        sample_ids=sample_arr,
        scores=score_arr,
        labels=label_arr,
        order=order,
    )


def _science_problem_worker(
    *,
    cache_root: str,
    cache_key: str,
    dataset_name: str,
    model_name: str,
    problem_id: str,
    sample_ids: list[int],
    distance_threads: int,
) -> dict[str, Any]:
    reader = _get_cached_reader(cache_root)
    correctness = _get_cached_correctness(cache_root)
    engine = _get_cached_engine(int(distance_threads))
    run_ids = list(map(int, sample_ids))
    raw = {
        "prefix_conf_mean": np.full(len(run_ids), np.nan, dtype=np.float64),
        "recency_conf_mean": np.full(len(run_ids), np.nan, dtype=np.float64),
        "late_worst_window": np.full(len(run_ids), np.nan, dtype=np.float64),
        "late_recovery": np.full(len(run_ids), np.nan, dtype=np.float64),
    }
    run_views = []
    for idx, run_id in enumerate(run_ids):
        token_view = reader.get_token_view(int(run_id))
        run_state = prepare_science_dynamic_run_state(reader, int(run_id), token_view=token_view)
        row = extract_science_dynamic_raw_from_state(
            run_state,
            prefix_fraction=DEFAULT_SCIENCE_PREFIX_FRACTION,
            tail_fraction=DEFAULT_SCIENCE_TAIL_FRACTION,
            recency_exp=DEFAULT_SCIENCE_RECENCY_EXP,
            window_tokens=DEFAULT_SCIENCE_WINDOW_TOKENS,
        )
        for key in raw:
            raw[key][idx] = float(row[key])
        run_views.append(reader.get_run_view(int(run_id), DEFAULT_VIEW))
    D = engine.dense_matrix(run_views)
    scores, _ = compute_science_dynamic_primary_scores_from_raw(raw, weights=DEFAULT_SCIENCE_DYNAMIC_WEIGHTS)
    sample_arr = np.asarray(run_ids, dtype=np.int64)
    label_arr = np.asarray([int(bool(correctness[int(sample_id)])) for sample_id in sample_arr], dtype=np.int32)
    order = order_code_dynamic_group_indices(scores, D, run_ids=run_ids)
    return _problem_result_payload(
        cache_key=cache_key,
        cache_root=cache_root,
        dataset_name=dataset_name,
        model_name=model_name,
        problem_id=str(problem_id),
        sample_ids=sample_arr,
        scores=np.asarray(scores, dtype=np.float64),
        labels=label_arr,
        order=np.asarray(order, dtype=np.int64),
    )


def _code_problem_worker(
    *,
    cache_root: str,
    cache_key: str,
    dataset_name: str,
    model_name: str,
    problem_id: str,
    sample_ids: list[int],
    distance_threads: int,
    prefix_fraction: float,
    prefix_window_tokens: int,
) -> dict[str, Any]:
    reader = _get_cached_reader(cache_root)
    correctness = _get_cached_correctness(cache_root)
    engine = _get_cached_engine(int(distance_threads))
    run_ids = list(map(int, sample_ids))
    raw = {
        "prefix_best_window_quality": np.full(len(run_ids), np.nan, dtype=np.float64),
        "head_tail_gap": np.full(len(run_ids), np.nan, dtype=np.float64),
        "reflection_density": np.full(len(run_ids), np.nan, dtype=np.float64),
        "tail_variance": np.full(len(run_ids), np.nan, dtype=np.float64),
        "post_reflection_recovery": np.full(len(run_ids), np.nan, dtype=np.float64),
    }
    run_views = []
    for idx, run_id in enumerate(run_ids):
        token_view = reader.get_token_view(int(run_id))
        run_state = prepare_code_dynamic_run_state(reader, int(run_id), token_view=token_view)
        row = extract_code_dynamic_raw_from_state(
            run_state,
            reflection_threshold=float(DEFAULT_CODE_DYNAMIC_REFLECTION_THRESHOLD),
            reflection_lookback_slices=int(DEFAULT_CODE_DYNAMIC_REFLECTION_LOOKBACK),
            prefix_fraction=float(prefix_fraction),
            prefix_window_tokens=int(prefix_window_tokens),
        )
        for key in raw:
            raw[key][idx] = float(row[key])
        run_views.append(reader.get_run_view(int(run_id), DEFAULT_VIEW))
    D = engine.dense_matrix(run_views)
    scores, _ = compute_code_dynamic_primary_scores_from_raw(raw)
    sample_arr = np.asarray(run_ids, dtype=np.int64)
    label_arr = np.asarray([int(bool(correctness[int(sample_id)])) for sample_id in sample_arr], dtype=np.int32)
    order = order_code_dynamic_group_indices(scores, D, run_ids=run_ids)
    return _problem_result_payload(
        cache_key=cache_key,
        cache_root=cache_root,
        dataset_name=dataset_name,
        model_name=model_name,
        problem_id=str(problem_id),
        sample_ids=sample_arr,
        scores=np.asarray(scores, dtype=np.float64),
        labels=label_arr,
        order=np.asarray(order, dtype=np.int64),
    )


def _empty_method_bundle(method_name: str, domain: str, note: str) -> dict[str, Any]:
    return {
        "method_name": str(method_name),
        "domain": str(domain),
        "note": str(note),
        "n_caches": 0,
        "n_samples": 0,
        "n_problems": 0,
        "global_selacc": {},
        "local_selacc": {},
        "caches": {},
    }


def _accumulate_problem_result(
    cache_results: dict[str, dict[str, Any]],
    result: dict[str, Any],
) -> None:
    cache_key = str(result["cache_key"])
    bucket = cache_results.setdefault(
        cache_key,
        {
            "entry": type(
                "EntryPayload",
                (),
                {
                    "cache_key": cache_key,
                    "cache_root": Path(result["cache_root"]),
                    "dataset_name": str(result["dataset_name"]),
                    "model_name": str(result["model_name"]),
                },
            )(),
            "records": [],
            "problems": {},
        },
    )
    sample_ids = np.asarray(result["sample_ids"], dtype=np.int64)
    scores = np.asarray(result["scores"], dtype=np.float64)
    labels = np.asarray(result["labels"], dtype=np.int32)
    order = np.asarray(result["order"], dtype=np.int64)
    problem_id = str(result["problem_id"])
    bucket["problems"][problem_id] = {
        "problem_id": problem_id,
        "sample_ids": sample_ids.tolist(),
        "scores": scores.tolist(),
        "labels": labels.tolist(),
        "order": order.tolist(),
    }
    for idx, sample_id in enumerate(sample_ids.tolist()):
        bucket["records"].append(
            {
                "cache_key": cache_key,
                "problem_id": problem_id,
                "sample_id": int(sample_id),
                "score": float(scores[idx]),
                "label": int(labels[idx]),
            }
        )


def _evaluate_generic_cache(
    *,
    entry,
    best_model,
    tuple_size: int,
    num_tuples: int,
    reflection_threshold: float,
    seed: int,
    max_problems: int | None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    cache_scores = score_cache_entry(
        entry=entry,
        best_model=best_model,
        worst_model=None,
        reflection_threshold=float(reflection_threshold),
        num_tuples=int(num_tuples),
        tuple_size=int(tuple_size),
        seed=int(seed),
        max_problems=max_problems,
    )
    correctness = _load_ground_truth(entry.cache_root)
    all_records: list[dict[str, Any]] = []
    problems: dict[str, dict[str, Any]] = {}
    for problem_id, payload in sorted(cache_scores.problem_scores.items(), key=lambda kv: _problem_sort_key(kv[0])):
        sample_ids = np.asarray(payload["sample_ids"], dtype=np.int64)
        scores = np.asarray(payload["score_best"], dtype=np.float64)
        labels = np.asarray([int(bool(correctness[int(sample_id)])) for sample_id in sample_ids], dtype=np.int32)
        order = _stable_score_order(scores, sample_ids)
        problems[str(problem_id)] = {
            "problem_id": str(problem_id),
            "sample_ids": sample_ids.tolist(),
            "scores": scores.tolist(),
            "labels": labels.tolist(),
            "order": order.tolist(),
        }
        for idx, sample_id in enumerate(sample_ids.tolist()):
            all_records.append(
                {
                    "cache_key": str(entry.cache_key),
                    "problem_id": str(problem_id),
                    "sample_id": int(sample_id),
                    "score": float(scores[idx]),
                    "label": int(labels[idx]),
                }
            )
    return all_records, problems


def _evaluate_generic_cache_worker(
    *,
    cache_root: str,
    cache_key: str,
    best_model_path: str,
    tuple_size: int,
    num_tuples: int,
    reflection_threshold: float,
    seed: int,
    max_problems: int | None,
) -> dict[str, Any]:
    entry_map = _load_entry_map(cache_root)
    if cache_key not in entry_map:
        raise KeyError(f"Cache key not found under {cache_root}: {cache_key}")
    entry = entry_map[cache_key]
    best_model = load_model(Path(best_model_path))
    records, problems = _evaluate_generic_cache(
        entry=entry,
        best_model=best_model,
        tuple_size=tuple_size,
        num_tuples=num_tuples,
        reflection_threshold=reflection_threshold,
        seed=seed,
        max_problems=max_problems,
    )
    return {
        "cache_key": str(entry.cache_key),
        "cache_root": str(entry.cache_root),
        "dataset_name": str(entry.dataset_name),
        "model_name": str(entry.model_name),
        "records": records,
        "problems": problems,
    }


def _evaluate_science_cache(
    *,
    entry,
    distance_threads: int,
    max_problems: int | None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    meta = json.loads((entry.cache_root / "meta.json").read_text(encoding="utf-8"))
    groups = build_problem_groups(meta)
    correctness = _load_ground_truth(entry.cache_root)
    reader = CacheReader(str(entry.cache_root))
    engine = DistanceEngine(DistanceSpec("ja", num_threads=int(distance_threads)))
    all_records: list[dict[str, Any]] = []
    problems: dict[str, dict[str, Any]] = {}

    for problem_index, (problem_id, run_ids) in enumerate(sorted(groups.items(), key=lambda kv: _problem_sort_key(kv[0]))):
        if max_problems is not None and problem_index >= int(max_problems):
            break
        run_ids = list(map(int, run_ids))
        raw = {
            "prefix_conf_mean": np.full(len(run_ids), np.nan, dtype=np.float64),
            "recency_conf_mean": np.full(len(run_ids), np.nan, dtype=np.float64),
            "late_worst_window": np.full(len(run_ids), np.nan, dtype=np.float64),
            "late_recovery": np.full(len(run_ids), np.nan, dtype=np.float64),
        }
        run_views = []
        for idx, run_id in enumerate(run_ids):
            token_view = reader.get_token_view(int(run_id))
            run_state = prepare_science_dynamic_run_state(reader, int(run_id), token_view=token_view)
            row = extract_science_dynamic_raw_from_state(
                run_state,
                prefix_fraction=DEFAULT_SCIENCE_PREFIX_FRACTION,
                tail_fraction=DEFAULT_SCIENCE_TAIL_FRACTION,
                recency_exp=DEFAULT_SCIENCE_RECENCY_EXP,
                window_tokens=DEFAULT_SCIENCE_WINDOW_TOKENS,
            )
            for key in raw:
                raw[key][idx] = float(row[key])
            run_views.append(reader.get_run_view(int(run_id), DEFAULT_VIEW))

        D = engine.dense_matrix(run_views)
        scores, _ = compute_science_dynamic_primary_scores_from_raw(
            raw,
            weights=DEFAULT_SCIENCE_DYNAMIC_WEIGHTS,
        )
        labels = np.asarray([int(bool(correctness.get(int(run_id), False))) for run_id in run_ids], dtype=np.int32)
        order = order_code_dynamic_group_indices(scores, D, run_ids=run_ids)
        problems[str(problem_id)] = {
            "problem_id": str(problem_id),
            "sample_ids": list(map(int, run_ids)),
            "scores": np.asarray(scores, dtype=np.float64).tolist(),
            "labels": labels.tolist(),
            "order": np.asarray(order, dtype=np.int64).tolist(),
        }
        for idx, run_id in enumerate(run_ids):
            all_records.append(
                {
                    "cache_key": str(entry.cache_key),
                    "problem_id": str(problem_id),
                    "sample_id": int(run_id),
                    "score": float(scores[idx]),
                    "label": int(labels[idx]),
                }
            )
    return all_records, problems


def _evaluate_science_cache_worker(
    *,
    cache_root: str,
    cache_key: str,
    distance_threads: int,
    max_problems: int | None,
) -> dict[str, Any]:
    entry_map = _load_entry_map(cache_root)
    if cache_key not in entry_map:
        raise KeyError(f"Cache key not found under {cache_root}: {cache_key}")
    entry = entry_map[cache_key]
    records, problems = _evaluate_science_cache(
        entry=entry,
        distance_threads=distance_threads,
        max_problems=max_problems,
    )
    return {
        "cache_key": str(entry.cache_key),
        "cache_root": str(entry.cache_root),
        "dataset_name": str(entry.dataset_name),
        "model_name": str(entry.model_name),
        "records": records,
        "problems": problems,
    }


def _evaluate_code_cache(
    *,
    entry,
    distance_threads: int,
    prefix_fraction: float,
    prefix_window_tokens: int,
    max_problems: int | None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    meta = json.loads((entry.cache_root / "meta.json").read_text(encoding="utf-8"))
    groups = build_problem_groups(meta)
    correctness = _load_ground_truth(entry.cache_root)
    reader = CacheReader(str(entry.cache_root))
    engine = DistanceEngine(DistanceSpec("ja", num_threads=int(distance_threads)))
    all_records: list[dict[str, Any]] = []
    problems: dict[str, dict[str, Any]] = {}

    for problem_index, (problem_id, run_ids) in enumerate(sorted(groups.items(), key=lambda kv: _problem_sort_key(kv[0]))):
        if max_problems is not None and problem_index >= int(max_problems):
            break
        run_ids = list(map(int, run_ids))
        raw = {
            "prefix_best_window_quality": np.full(len(run_ids), np.nan, dtype=np.float64),
            "head_tail_gap": np.full(len(run_ids), np.nan, dtype=np.float64),
            "reflection_density": np.full(len(run_ids), np.nan, dtype=np.float64),
            "tail_variance": np.full(len(run_ids), np.nan, dtype=np.float64),
            "post_reflection_recovery": np.full(len(run_ids), np.nan, dtype=np.float64),
        }
        run_views = []
        for idx, run_id in enumerate(run_ids):
            token_view = reader.get_token_view(int(run_id))
            run_state = prepare_code_dynamic_run_state(reader, int(run_id), token_view=token_view)
            row = extract_code_dynamic_raw_from_state(
                run_state,
                reflection_threshold=float(DEFAULT_CODE_DYNAMIC_REFLECTION_THRESHOLD),
                reflection_lookback_slices=int(DEFAULT_CODE_DYNAMIC_REFLECTION_LOOKBACK),
                prefix_fraction=float(prefix_fraction),
                prefix_window_tokens=int(prefix_window_tokens),
            )
            for key in raw:
                raw[key][idx] = float(row[key])
            run_views.append(reader.get_run_view(int(run_id), DEFAULT_VIEW))

        D = engine.dense_matrix(run_views)
        scores, _ = compute_code_dynamic_primary_scores_from_raw(raw)
        labels = np.asarray([int(bool(correctness.get(int(run_id), False))) for run_id in run_ids], dtype=np.int32)
        order = order_code_dynamic_group_indices(scores, D, run_ids=run_ids)
        problems[str(problem_id)] = {
            "problem_id": str(problem_id),
            "sample_ids": list(map(int, run_ids)),
            "scores": np.asarray(scores, dtype=np.float64).tolist(),
            "labels": labels.tolist(),
            "order": np.asarray(order, dtype=np.int64).tolist(),
        }
        for idx, run_id in enumerate(run_ids):
            all_records.append(
                {
                    "cache_key": str(entry.cache_key),
                    "problem_id": str(problem_id),
                    "sample_id": int(run_id),
                    "score": float(scores[idx]),
                    "label": int(labels[idx]),
                }
            )
    return all_records, problems


def _evaluate_code_cache_worker(
    *,
    cache_root: str,
    cache_key: str,
    distance_threads: int,
    prefix_fraction: float,
    prefix_window_tokens: int,
    max_problems: int | None,
) -> dict[str, Any]:
    entry_map = _load_entry_map(cache_root)
    if cache_key not in entry_map:
        raise KeyError(f"Cache key not found under {cache_root}: {cache_key}")
    entry = entry_map[cache_key]
    records, problems = _evaluate_code_cache(
        entry=entry,
        distance_threads=distance_threads,
        prefix_fraction=prefix_fraction,
        prefix_window_tokens=prefix_window_tokens,
        max_problems=max_problems,
    )
    return {
        "cache_key": str(entry.cache_key),
        "cache_root": str(entry.cache_root),
        "dataset_name": str(entry.dataset_name),
        "model_name": str(entry.model_name),
        "records": records,
        "problems": problems,
    }


def _merge_problem_maps(
    existing: dict[str, dict[str, Any]],
    incoming: dict[str, dict[str, Any]],
    *,
    cache_key: str,
) -> None:
    for problem_id, payload in incoming.items():
        merged_key = f"{cache_key}::{problem_id}"
        if merged_key in existing:
            raise ValueError(f"Duplicate merged problem key: {merged_key}")
        existing[merged_key] = dict(payload)


def _method_order(method_names: list[str]) -> list[str]:
    preferred = [GENERIC_METHOD, SCIENCE_METHOD, CODE_METHOD]
    ordered = [name for name in preferred if name in method_names]
    ordered.extend([name for name in method_names if name not in ordered])
    return ordered


def _domain_note(domain: str) -> str:
    if domain == "math":
        return "Math uses cache_train caches: aime25, brumo25, hmmt25."
    if domain == "science":
        return "Science uses cache_train gpqa and compares generic extreme against science_baseline_v1."
    if domain == "coding":
        return "Coding uses MUI_HUB/cache DS-R1/lcb_v5 because cache_train has no coding cache."
    raise KeyError(domain)


def _domain_note_cn(domain: str) -> str:
    if domain == "math":
        return "数学域使用 cache_train 中的 aime25、brumo25、hmmt25。"
    if domain == "science":
        return "科学域使用 cache_train 中的 gpqa，并比较通用 extreme 与 science_baseline_v1。"
    if domain == "coding":
        return "编程域使用 MUI_HUB/cache 下的 DS-R1/lcb_v5，因为 cache_train 不包含 coding cache。"
    raise KeyError(domain)


def _observation_lines(domain_payload: dict[str, Any]) -> list[str]:
    methods = domain_payload["methods"]
    method_names = _method_order(list(methods.keys()))
    if not method_names:
        return ["- No methods were evaluated for this domain."]
    lines: list[str] = []
    if len(method_names) == 1:
        only = methods[method_names[0]]
        lines.append(
            f"- `{method_names[0]}`: Global SelAcc@10={_fmt_pct(only['global_selacc'].get('10'))}, "
            f"Local SelAcc@10={_fmt_pct(only['local_selacc'].get('10'))}."
        )
        return lines

    global_best = max(
        method_names,
        key=lambda name: float(methods[name]["global_selacc"].get("10", 0.0)),
    )
    local_best = max(
        method_names,
        key=lambda name: float(methods[name]["local_selacc"].get("10", 0.0)),
    )
    lines.append(
        f"- Global SelAcc@10 winner: `{global_best}` = {_fmt_pct(methods[global_best]['global_selacc'].get('10'))}."
    )
    lines.append(
        f"- Local SelAcc@10 winner: `{local_best}` = {_fmt_pct(methods[local_best]['local_selacc'].get('10'))}."
    )
    if GENERIC_METHOD in methods:
        generic = methods[GENERIC_METHOD]
        for specialized in (SCIENCE_METHOD, CODE_METHOD):
            if specialized in methods:
                lines.append(
                    f"- `{specialized}` vs `{GENERIC_METHOD}` at 10%: "
                    f"global Δ={_fmt_pct(float(methods[specialized]['global_selacc'].get('10', 0.0)) - float(generic['global_selacc'].get('10', 0.0)))}, "
                    f"local Δ={_fmt_pct(float(methods[specialized]['local_selacc'].get('10', 0.0)) - float(generic['local_selacc'].get('10', 0.0)))}."
                )
    return lines


def _observation_lines_cn(domain_payload: dict[str, Any]) -> list[str]:
    methods = domain_payload["methods"]
    method_names = _method_order(list(methods.keys()))
    if not method_names:
        return ["- 该领域没有可用方法结果。"]
    lines: list[str] = []
    if len(method_names) == 1:
        only = methods[method_names[0]]
        lines.append(
            f"- `{method_names[0]}`：Global SelAcc@10={_fmt_pct(only['global_selacc'].get('10'))}，"
            f"Local SelAcc@10={_fmt_pct(only['local_selacc'].get('10'))}。"
        )
        return lines

    global_best = max(
        method_names,
        key=lambda name: float(methods[name]["global_selacc"].get("10", 0.0)),
    )
    local_best = max(
        method_names,
        key=lambda name: float(methods[name]["local_selacc"].get("10", 0.0)),
    )
    lines.append(
        f"- Global SelAcc@10 最优：`{global_best}` = {_fmt_pct(methods[global_best]['global_selacc'].get('10'))}。"
    )
    lines.append(
        f"- Local SelAcc@10 最优：`{local_best}` = {_fmt_pct(methods[local_best]['local_selacc'].get('10'))}。"
    )
    if GENERIC_METHOD in methods:
        generic = methods[GENERIC_METHOD]
        for specialized in (SCIENCE_METHOD, CODE_METHOD):
            if specialized in methods:
                lines.append(
                    f"- `{specialized}` 相对 `{GENERIC_METHOD}` 在 10% 处："
                    f"global Δ={_fmt_pct(float(methods[specialized]['global_selacc'].get('10', 0.0)) - float(generic['global_selacc'].get('10', 0.0)))}，"
                    f"local Δ={_fmt_pct(float(methods[specialized]['local_selacc'].get('10', 0.0)) - float(generic['local_selacc'].get('10', 0.0)))}。"
                )
    return lines


def _build_curve_table(
    *,
    title: str,
    metric_key: str,
    bundles: dict[str, dict[str, Any]],
    percents: list[int],
) -> list[str]:
    method_names = _method_order(list(bundles.keys()))
    lines = [f"### {title}", ""]
    header = "| Top-p | " + " | ".join(method_names) + " |"
    sep = "| ---: | " + " | ".join(["---:"] * len(method_names)) + " |"
    lines.extend([header, sep])
    for percent in percents:
        row = [f"{percent}%"]
        for method_name in method_names:
            value = bundles[method_name].get(metric_key, {}).get(str(percent))
            row.append(_fmt_pct(value))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def _write_outputs(
    *,
    results_payload: dict[str, Any],
    out_dir: Path,
    doc_out: Path,
    percents: list[int],
) -> None:
    metrics_path = out_dir / "metrics.json"
    summary_path = out_dir / "summary.md"
    metrics_path.write_text(json.dumps(results_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Extreme SelAcc Grid Evaluation",
        "",
        f"- Generated: {results_payload['generated_at_utc']}",
        f"- Metrics JSON: `{_display_path(metrics_path)}`",
        f"- Cache-train root: `{results_payload['config']['cache_train_root']}`",
        f"- Coding root: `{results_payload['config']['coding_root']}`",
        f"- Percent grid: `{','.join(str(v) for v in percents)}`",
        "- Generic extreme line uses the frozen `baseline12_pointwise` manifest and evaluates the recommended `best_only` score branch.",
        "- Coding is evaluated on `MUI_HUB/cache` because `cache_train` has no `lcb_v5` cache.",
        "",
        "## Metric Definitions",
        "",
        "- Global SelAcc@p: flatten all samples in a method/domain, rank by final score, then measure top-p% accuracy.",
        "- Local SelAcc@p: within each problem, rank by final score/order, measure top-p% accuracy, then average across problems.",
        "",
    ]

    for domain in DOMAIN_ORDER:
        domain_payload = results_payload["domains"].get(domain)
        if not domain_payload:
            continue
        lines.extend(
            [
                f"## {DOMAIN_LABELS[domain]}",
                "",
                f"- Note: {domain_payload['note']}",
                f"- Caches: {', '.join(f'`{cache_key}`' for cache_key in domain_payload['cache_keys'])}",
                f"- Methods: {', '.join(f'`{name}`' for name in _method_order(list(domain_payload['methods'].keys())))}",
                f"- Samples: `{domain_payload['n_samples']}`; problems: `{domain_payload['n_problems']}`.",
                "",
            ]
        )
        lines.extend(_observation_lines(domain_payload))
        lines.append("")
        lines.extend(
            _build_curve_table(
                title="Global SelAcc@p",
                metric_key="global_selacc",
                bundles=domain_payload["methods"],
                percents=percents,
            )
        )
        lines.extend(
            _build_curve_table(
                title="Local Mean SelAcc@p",
                metric_key="local_selacc",
                bundles=domain_payload["methods"],
                percents=percents,
            )
        )
        lines.append("### Cache Details")
        lines.append("")
        for cache_key in domain_payload["cache_keys"]:
            method_bundles = {
                method_name: method_payload["caches"][cache_key]
                for method_name, method_payload in domain_payload["methods"].items()
                if cache_key in method_payload["caches"]
            }
            if not method_bundles:
                continue
            sample_payload = next(iter(method_bundles.values()))
            lines.extend(
                [
                    f"#### `{cache_key}`",
                    "",
                    f"- Cache root: `{sample_payload['cache_root']}`",
                    f"- Dataset: `{sample_payload['dataset_name']}`",
                    f"- Samples: `{sample_payload['n_samples']}`; problems: `{sample_payload['n_problems']}`.",
                    "",
                ]
            )
            lines.extend(
                _build_curve_table(
                    title="Global SelAcc@p",
                    metric_key="global_selacc",
                    bundles=method_bundles,
                    percents=percents,
                )
            )
            lines.extend(
                _build_curve_table(
                    title="Local Mean SelAcc@p",
                    metric_key="local_selacc",
                    bundles=method_bundles,
                    percents=percents,
                )
            )

    lines.extend(
        [
            "",
            "---",
            "",
            "# Extreme SelAcc 网格评估",
            "",
            f"- 生成时间：{results_payload['generated_at_utc']}",
            f"- 指标 JSON：`{_display_path(metrics_path)}`",
            f"- Cache-train 根目录：`{results_payload['config']['cache_train_root']}`",
            f"- Coding 根目录：`{results_payload['config']['coding_root']}`",
            f"- 分位网格：`{','.join(str(v) for v in percents)}`",
            "- 通用 extreme 主线使用冻结的 `baseline12_pointwise` manifest，并评估推荐的 `best_only` 分支。",
            "- 由于 `cache_train` 没有 `lcb_v5`，coding 结果来自 `MUI_HUB/cache`。",
            "",
            "## 指标定义",
            "",
            "- Global SelAcc@p：把一个方法/领域内的全部样本拉平，按最终分数全局排序，再计算 top-p% 的正确率。",
            "- Local SelAcc@p：每道题内部按最终分数/顺序排序，计算 top-p% 的正确率，再对题目做平均。",
            "",
        ]
    )

    for domain in DOMAIN_ORDER:
        domain_payload = results_payload["domains"].get(domain)
        if not domain_payload:
            continue
        lines.extend(
            [
                f"## {DOMAIN_LABELS[domain]}（中文）",
                "",
                f"- 说明：{_domain_note_cn(domain)}",
                f"- Cache：{', '.join(f'`{cache_key}`' for cache_key in domain_payload['cache_keys'])}",
                f"- 方法：{', '.join(f'`{name}`' for name in _method_order(list(domain_payload['methods'].keys())))}",
                f"- 样本数：`{domain_payload['n_samples']}`；题目数：`{domain_payload['n_problems']}`。",
                "",
            ]
        )
        lines.extend(_observation_lines_cn(domain_payload))
        lines.append("")
        lines.extend(
            _build_curve_table(
                title="全局 SelAcc@p",
                metric_key="global_selacc",
                bundles=domain_payload["methods"],
                percents=percents,
            )
        )
        lines.extend(
            _build_curve_table(
                title="按题平均 SelAcc@p",
                metric_key="local_selacc",
                bundles=domain_payload["methods"],
                percents=percents,
            )
        )
        lines.append("### Cache 明细")
        lines.append("")
        for cache_key in domain_payload["cache_keys"]:
            method_bundles = {
                method_name: method_payload["caches"][cache_key]
                for method_name, method_payload in domain_payload["methods"].items()
                if cache_key in method_payload["caches"]
            }
            if not method_bundles:
                continue
            sample_payload = next(iter(method_bundles.values()))
            lines.extend(
                [
                    f"#### `{cache_key}`",
                    "",
                    f"- Cache 路径：`{sample_payload['cache_root']}`",
                    f"- 数据集：`{sample_payload['dataset_name']}`",
                    f"- 样本数：`{sample_payload['n_samples']}`；题目数：`{sample_payload['n_problems']}`。",
                    "",
                ]
            )
            lines.extend(
                _build_curve_table(
                    title="全局 SelAcc@p",
                    metric_key="global_selacc",
                    bundles=method_bundles,
                    percents=percents,
                )
            )
            lines.extend(
                _build_curve_table(
                    title="按题平均 SelAcc@p",
                    metric_key="local_selacc",
                    bundles=method_bundles,
                    percents=percents,
                )
            )

    summary_text = "\n".join(lines).rstrip() + "\n"
    summary_path.write_text(summary_text, encoding="utf-8")
    doc_out.parent.mkdir(parents=True, exist_ok=True)
    doc_out.write_text(summary_text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate frozen extreme lines on math/science/coding SelAcc@p grids")
    ap.add_argument(
        "--manifest",
        default="submission/BestofN/extreme12/manifests/extreme12_baseline12_pointwise_export_manifest.json",
        help="Manifest for the frozen baseline12_pointwise export",
    )
    ap.add_argument(
        "--cache-train-root",
        default="/home/jovyan/public-ro/MUI_HUB/cache_train",
        help="cache_train root used for math + science",
    )
    ap.add_argument(
        "--coding-root",
        default="MUI_HUB/cache",
        help="Coding root used for lcb_v5 because cache_train has no coding cache",
    )
    ap.add_argument(
        "--coding-cache-key",
        default="DS-R1/lcb_v5",
        help="Coding cache key to evaluate under coding root",
    )
    ap.add_argument(
        "--out-dir",
        default="",
        help="Optional output directory; defaults to result/extreme_selacc_grid_<timestamp>",
    )
    ap.add_argument(
        "--doc-out",
        default="docs/EXTREME_SELACC_GRID_20260406.md",
        help="Markdown report output path",
    )
    ap.add_argument(
        "--percents",
        default=",".join(str(v) for v in PERCENT_GRID_DEFAULT),
        help="Comma-separated percent grid, e.g. 5,10,15,...,100",
    )
    ap.add_argument("--distance-threads", type=int, default=8, help="Distance threads for science/code tie-break matrices")
    ap.add_argument("--workers", type=int, default=4, help="Worker count for generic cache evaluation")
    ap.add_argument("--seed", type=int, default=42, help="Base seed for generic baseline tuple sampling")
    ap.add_argument("--prefix-window-tokens", type=int, default=128, help="Prefix window for code baseline")
    ap.add_argument("--prefix-fraction", type=float, default=0.20, help="Prefix fraction for code baseline")
    ap.add_argument("--max-problems", type=int, default=None, help="Optional max problems per cache for smoke tests")
    args = ap.parse_args()

    os_out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "result" / f"extreme_selacc_grid_{_now_tag()}"
    out_dir = os_out_dir if os_out_dir.is_absolute() else (REPO_ROOT / os_out_dir)
    doc_out = Path(args.doc_out)
    if not doc_out.is_absolute():
        doc_out = REPO_ROOT / doc_out
    out_dir.mkdir(parents=True, exist_ok=True)

    percents = _parse_percent_grid(args.percents)
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    manifest = _load_manifest(manifest_path)
    best_model_path = REPO_ROOT / manifest["model_paths"]["best"]
    tuple_size = int(manifest["export_config"]["tuple_size"])
    num_tuples = int(manifest["export_config"]["num_tuples"])
    reflection_threshold = float(manifest["export_config"]["reflection_threshold"])

    cache_train_map = _load_entry_map(args.cache_train_root)
    coding_map = _load_entry_map(args.coding_root)
    target_generic_keys = [
        cache_key
        for cache_key, entry in sorted(cache_train_map.items())
        if entry.dataset_name in (MATH_DATASETS | SCIENCE_DATASETS)
    ]
    if args.coding_cache_key not in coding_map:
        raise SystemExit(f"Coding cache key not found under {args.coding_root}: {args.coding_cache_key}")

    results_payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "manifest": _display_path(manifest_path),
            "cache_train_root": str(args.cache_train_root),
            "coding_root": str(args.coding_root),
            "coding_cache_key": str(args.coding_cache_key),
            "percents": percents,
            "distance_threads": int(args.distance_threads),
            "seed": int(args.seed),
            "max_problems": None if args.max_problems is None else int(args.max_problems),
        },
        "methods": {
            GENERIC_METHOD: {
                "type": "generic_extreme",
                "manifest": _display_path(manifest_path),
                "best_model_path": _display_path(best_model_path),
                "tuple_size": tuple_size,
                "num_tuples": num_tuples,
                "reflection_threshold": reflection_threshold,
                "score_branch": "best_only",
            },
            SCIENCE_METHOD: {
                "type": "science_specialized",
                "weights": DEFAULT_SCIENCE_DYNAMIC_WEIGHTS,
            },
            CODE_METHOD: {
                "type": "coding_specialized",
                "reflection_threshold": float(DEFAULT_CODE_DYNAMIC_REFLECTION_THRESHOLD),
                "reflection_lookback_slices": int(DEFAULT_CODE_DYNAMIC_REFLECTION_LOOKBACK),
                "prefix_fraction": float(args.prefix_fraction),
                "prefix_window_tokens": int(args.prefix_window_tokens),
            },
        },
        "domains": {},
    }

    coding_entry = coding_map[args.coding_cache_key]
    science_entry = next((entry for _, entry in cache_train_map.items() if entry.dataset_name in SCIENCE_DATASETS), None)
    if science_entry is None:
        raise SystemExit("No science cache found in cache_train root")

    def _problem_specs_for_entry(entry, *, base_seed: int) -> list[dict[str, Any]]:
        meta = json.loads((entry.cache_root / "meta.json").read_text(encoding="utf-8"))
        groups = sorted(build_problem_groups(meta).items(), key=lambda kv: _problem_sort_key(kv[0]))
        if args.max_problems is not None:
            groups = groups[: max(0, int(args.max_problems))]
        specs: list[dict[str, Any]] = []
        for problem_index, (problem_id, sample_ids) in enumerate(groups):
            specs.append(
                {
                    "cache_key": str(entry.cache_key),
                    "cache_root": str(entry.cache_root),
                    "dataset_name": str(entry.dataset_name),
                    "model_name": str(entry.model_name),
                    "problem_id": str(problem_id),
                    "sample_ids": list(map(int, sample_ids)),
                    "seed": int(base_seed) + problem_index,
                }
            )
        return specs

    generic_problem_specs: list[dict[str, Any]] = []
    for idx, cache_key in enumerate(target_generic_keys, start=1):
        generic_problem_specs.extend(_problem_specs_for_entry(cache_train_map[cache_key], base_seed=int(args.seed) + idx * 100_000))
    generic_problem_specs.extend(_problem_specs_for_entry(coding_entry, base_seed=int(args.seed) + 9_999_999))

    generic_cache_results: dict[str, dict[str, Any]] = {}
    generic_total = len(generic_problem_specs)
    print(f"[generic] scheduling {generic_total} problems with workers={int(args.workers)}", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        future_map = {
            pool.submit(
                _generic_problem_worker,
                cache_root=spec["cache_root"],
                cache_key=spec["cache_key"],
                dataset_name=spec["dataset_name"],
                model_name=spec["model_name"],
                problem_id=spec["problem_id"],
                sample_ids=spec["sample_ids"],
                best_model_path=str(best_model_path),
                tuple_size=tuple_size,
                num_tuples=num_tuples,
                reflection_threshold=reflection_threshold,
                seed=spec["seed"],
            ): spec
            for spec in generic_problem_specs
        }
        for done_count, future in enumerate(as_completed(future_map), start=1):
            result = future.result()
            _accumulate_problem_result(generic_cache_results, result)
            if done_count % 25 == 0 or done_count == generic_total:
                print(f"[generic] {done_count}/{generic_total} problems done", flush=True)

    def _quick_domain_bundle(cache_results: dict[str, dict[str, Any]], domain_filter: set[str]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        all_records: list[dict[str, Any]] = []
        all_problems: dict[str, dict[str, Any]] = {}
        for cache_key, payload in cache_results.items():
            if payload["entry"].dataset_name not in domain_filter:
                continue
            all_records.extend(payload["records"])
            _merge_problem_maps(all_problems, payload["problems"], cache_key=cache_key)
        return all_records, all_problems

    generic_math_records, generic_math_problems = _quick_domain_bundle(generic_cache_results, MATH_DATASETS)
    generic_science_records, generic_science_problems = _quick_domain_bundle(generic_cache_results, SCIENCE_DATASETS)
    generic_coding_records, generic_coding_problems = _quick_domain_bundle(generic_cache_results, CODING_DATASETS)
    print(
        "[generic summary] "
        f"math G10={_fmt_pct(_compute_global_selacc_curve(generic_math_records, percents).get('10'))}, "
        f"science G10={_fmt_pct(_compute_global_selacc_curve(generic_science_records, percents).get('10'))}, "
        f"coding G10={_fmt_pct(_compute_global_selacc_curve(generic_coding_records, percents).get('10'))}",
        flush=True,
    )

    science_specs = _problem_specs_for_entry(science_entry, base_seed=0)
    science_cache_results: dict[str, dict[str, Any]] = {}
    science_total = len(science_specs)
    print(f"[science specialized] scheduling {science_total} problems", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        future_map = {
            pool.submit(
                _science_problem_worker,
                cache_root=spec["cache_root"],
                cache_key=spec["cache_key"],
                dataset_name=spec["dataset_name"],
                model_name=spec["model_name"],
                problem_id=spec["problem_id"],
                sample_ids=spec["sample_ids"],
                distance_threads=int(args.distance_threads),
            ): spec
            for spec in science_specs
        }
        for done_count, future in enumerate(as_completed(future_map), start=1):
            result = future.result()
            _accumulate_problem_result(science_cache_results, result)
            if done_count % 25 == 0 or done_count == science_total:
                print(f"[science specialized] {done_count}/{science_total} problems done", flush=True)
    science_records, science_problems = _quick_domain_bundle(science_cache_results, SCIENCE_DATASETS)
    print(
        "[science specialized summary] "
        f"G10={_fmt_pct(_compute_global_selacc_curve(science_records, percents).get('10'))}, "
        f"L10={_fmt_pct(_compute_local_selacc_curve(science_problems, percents).get('10'))}",
        flush=True,
    )

    code_specs = _problem_specs_for_entry(coding_entry, base_seed=0)
    code_cache_results: dict[str, dict[str, Any]] = {}
    code_total = len(code_specs)
    print(f"[coding specialized] scheduling {code_total} problems", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        future_map = {
            pool.submit(
                _code_problem_worker,
                cache_root=spec["cache_root"],
                cache_key=spec["cache_key"],
                dataset_name=spec["dataset_name"],
                model_name=spec["model_name"],
                problem_id=spec["problem_id"],
                sample_ids=spec["sample_ids"],
                distance_threads=int(args.distance_threads),
                prefix_fraction=float(args.prefix_fraction),
                prefix_window_tokens=int(args.prefix_window_tokens),
            ): spec
            for spec in code_specs
        }
        for done_count, future in enumerate(as_completed(future_map), start=1):
            result = future.result()
            _accumulate_problem_result(code_cache_results, result)
            if done_count % 25 == 0 or done_count == code_total:
                print(f"[coding specialized] {done_count}/{code_total} problems done", flush=True)
    code_records, code_problems = _quick_domain_bundle(code_cache_results, CODING_DATASETS)
    print(
        "[coding specialized summary] "
        f"G10={_fmt_pct(_compute_global_selacc_curve(code_records, percents).get('10'))}, "
        f"L10={_fmt_pct(_compute_local_selacc_curve(code_problems, percents).get('10'))}",
        flush=True,
    )

    for domain in DOMAIN_ORDER:
        results_payload["domains"][domain] = {
            "label": DOMAIN_LABELS[domain],
            "note": _domain_note(domain),
            "cache_keys": [],
            "n_samples": 0,
            "n_problems": 0,
            "methods": {},
        }

    generic_domain_records: dict[str, list[dict[str, Any]]] = {domain: [] for domain in DOMAIN_ORDER}
    generic_domain_problems: dict[str, dict[str, dict[str, Any]]] = {domain: {} for domain in DOMAIN_ORDER}
    generic_domain_caches: dict[str, list[dict[str, Any]]] = {domain: [] for domain in DOMAIN_ORDER}

    for cache_key, payload in generic_cache_results.items():
        entry = payload["entry"]
        if entry.dataset_name in MATH_DATASETS:
            domain = "math"
        elif entry.dataset_name in SCIENCE_DATASETS:
            domain = "science"
        elif entry.dataset_name in CODING_DATASETS:
            domain = "coding"
        else:
            continue
        cache_metric_payload = _build_cache_metric_payload(
            method_name=GENERIC_METHOD,
            domain=domain,
            cache_key=entry.cache_key,
            cache_root=entry.cache_root,
            dataset_name=entry.dataset_name,
            records=payload["records"],
            problems=payload["problems"],
            percents=percents,
            source="frozen baseline12_pointwise manifest",
        )
        generic_domain_caches[domain].append(cache_metric_payload)
        generic_domain_records[domain].extend(payload["records"])
        _merge_problem_maps(generic_domain_problems[domain], payload["problems"], cache_key=entry.cache_key)

    for domain in DOMAIN_ORDER:
        if generic_domain_caches[domain]:
            generic_bundle = _aggregate_domain_payload(
                method_name=GENERIC_METHOD,
                domain=domain,
                cache_payloads=generic_domain_caches[domain],
                records=generic_domain_records[domain],
                problems=generic_domain_problems[domain],
                percents=percents,
                note="Frozen generic extreme baseline.",
            )
        else:
            generic_bundle = _empty_method_bundle(GENERIC_METHOD, domain, "Frozen generic extreme baseline.")
        results_payload["domains"][domain]["methods"][GENERIC_METHOD] = generic_bundle

    science_cache_payload = _build_cache_metric_payload(
        method_name=SCIENCE_METHOD,
        domain="science",
        cache_key=science_entry.cache_key,
        cache_root=science_entry.cache_root,
        dataset_name=science_entry.dataset_name,
        records=science_records,
        problems=science_problems,
        percents=percents,
        source="science dynamic specialized baseline",
    )
    results_payload["domains"]["science"]["methods"][SCIENCE_METHOD] = _aggregate_domain_payload(
        method_name=SCIENCE_METHOD,
        domain="science",
        cache_payloads=[science_cache_payload],
        records=science_records,
        problems={f"{science_entry.cache_key}::{problem_id}": payload for problem_id, payload in science_problems.items()},
        percents=percents,
        note="Science-specialized dynamic baseline.",
    )

    code_cache_payload = _build_cache_metric_payload(
        method_name=CODE_METHOD,
        domain="coding",
        cache_key=coding_entry.cache_key,
        cache_root=coding_entry.cache_root,
        dataset_name=coding_entry.dataset_name,
        records=code_records,
        problems=code_problems,
        percents=percents,
        source="code dynamic specialized baseline",
    )
    results_payload["domains"]["coding"]["methods"][CODE_METHOD] = _aggregate_domain_payload(
        method_name=CODE_METHOD,
        domain="coding",
        cache_payloads=[code_cache_payload],
        records=code_records,
        problems={f"{coding_entry.cache_key}::{problem_id}": payload for problem_id, payload in code_problems.items()},
        percents=percents,
        note="Coding-specialized PrefixSaturation / code_dynamic baseline.",
    )

    for domain in DOMAIN_ORDER:
        cache_keys: set[str] = set()
        n_samples = 0
        n_problems = 0
        for method_payload in results_payload["domains"][domain]["methods"].values():
            cache_keys.update(method_payload["caches"].keys())
            n_samples = max(n_samples, int(method_payload["n_samples"]))
            n_problems = max(n_problems, int(method_payload["n_problems"]))
        results_payload["domains"][domain]["cache_keys"] = sorted(cache_keys)
        results_payload["domains"][domain]["n_samples"] = int(n_samples)
        results_payload["domains"][domain]["n_problems"] = int(n_problems)

    _write_outputs(
        results_payload=results_payload,
        out_dir=out_dir,
        doc_out=doc_out,
        percents=percents,
    )

    print(f"Saved metrics to {_display_path(out_dir / 'metrics.json')}")
    print(f"Saved summary to {_display_path(out_dir / 'summary.md')}")
    print(f"Saved doc to {_display_path(doc_out)}")


if __name__ == "__main__":
    main()
