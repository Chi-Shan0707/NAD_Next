from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from nad.core.selectors.base import SelectorContext
from nad.core.selectors.extreme8_impl import accumulate_extreme8_scores, extract_extreme8_raw_values
from nad.core.views.reader import CacheReader
from nad.ops.accuracy import _load_ground_truth

MODEL_SUBMISSION_NAMES = {
    "DeepSeek-R1-0528-Qwen3-8B": "DS-R1",
    "Qwen3-4B-Thinking-2507": "Qwen3-4B",
}

DATASET_SUBMISSION_NAMES = {
    "livecodebench_v5": "lcb_v5",
}


@dataclass(frozen=True)
class CacheEntry:
    cache_key: str
    cache_root: Path
    model_name: str
    dataset_name: str


@dataclass(frozen=True)
class CacheScores:
    entry: CacheEntry
    problem_scores: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class ScoreMetricResult:
    auroc: float | None
    hit_at_1: float
    hit_at_3: float
    selective_acc_at_10pct: float
    pairwise_accuracy: float | None
    n_problems: int
    n_samples: int
    n_positive: int
    n_negative: int


@dataclass(frozen=True)
class CacheMetricBundle:
    entry: CacheEntry
    by_score: dict[str, ScoreMetricResult]


@dataclass(frozen=True)
class MeanMetricBundle:
    by_score: dict[str, dict[str, float | None]]


SCORE_NAME_MAP = {
    "best_only": "score_best",
    "mix": "score_mix",
}


def _problem_sort_key(problem_id: str) -> tuple[int, Any]:
    try:
        return (0, int(problem_id))
    except (TypeError, ValueError):
        return (1, str(problem_id))


def submission_model_name(model_name: str) -> str:
    return MODEL_SUBMISSION_NAMES.get(model_name, model_name)


def submission_dataset_name(dataset_name: str) -> str:
    return DATASET_SUBMISSION_NAMES.get(dataset_name, dataset_name)


def submission_cache_key(model_name: str, dataset_name: str) -> str:
    return f"{submission_model_name(model_name)}/{submission_dataset_name(dataset_name)}"


def discover_cache_entries(base_root: str | Path) -> list[CacheEntry]:
    root = Path(base_root)
    if not root.exists():
        raise FileNotFoundError(f"Cache root not found: {root}")

    entries: list[CacheEntry] = []
    for model_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for dataset_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            cache_dirs = sorted(path for path in dataset_dir.iterdir() if path.is_dir())
            if not cache_dirs:
                continue
            cache_root = cache_dirs[-1]
            entries.append(CacheEntry(
                cache_key=submission_cache_key(model_dir.name, dataset_dir.name),
                cache_root=cache_root,
                model_name=model_dir.name,
                dataset_name=dataset_dir.name,
            ))
    return entries


def build_problem_groups(meta: dict[str, Any]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for sample_id, sample in enumerate(meta.get("samples", [])):
        problem_id = str(sample["problem_id"])
        groups.setdefault(problem_id, []).append(int(sample_id))
    return groups


def _sample_run_indices(meta: dict[str, Any], sample_ids: list[int]) -> list[int]:
    samples = meta.get("samples", [])
    return [int(samples[sample_id].get("run_index", 0)) for sample_id in sample_ids]


def score_cache_entry(
    entry: CacheEntry,
    best_model,
    worst_model,
    reflection_threshold: float,
    num_tuples: int,
    tuple_size: int = 8,
    seed: int = 42,
    max_problems: int | None = None,
) -> CacheScores:
    meta = json.loads((entry.cache_root / "meta.json").read_text(encoding="utf-8"))
    groups = build_problem_groups(meta)
    reader = CacheReader(str(entry.cache_root))

    problem_scores: dict[str, dict[str, Any]] = {}
    for problem_index, (problem_id, sample_ids) in enumerate(
        sorted(groups.items(), key=lambda kv: _problem_sort_key(kv[0]))
    ):
        if max_problems is not None and problem_index >= max_problems:
            break
        ctx = SelectorContext(cache=reader, problem_id=problem_id, run_ids=list(map(int, sample_ids)), views=[])
        raw_values = extract_extreme8_raw_values(ctx, reflection_threshold=reflection_threshold)
        payload = accumulate_extreme8_scores(
            best_model=best_model,
            worst_model=worst_model,
            raw_values=raw_values,
            tuple_size=tuple_size,
            num_tuples=num_tuples,
            seed=int(seed) + problem_index,
            labels=None,
            require_mixed=False,
        )
        problem_scores[str(problem_id)] = {
            "sample_ids": list(map(int, sample_ids)),
            "run_indices": _sample_run_indices(meta, sample_ids),
            "score_best": np.asarray(payload["score_best"], dtype=np.float64).tolist(),
            "score_worst": np.asarray(payload["score_worst"], dtype=np.float64).tolist(),
            "score_mix": np.asarray(payload["score_mix"], dtype=np.float64).tolist(),
            "counts": np.asarray(payload["counts"], dtype=np.float64).tolist(),
            "num_tuples": int(np.asarray(payload["num_tuples"], dtype=np.int64)[0]),
        }
    return CacheScores(entry=entry, problem_scores=problem_scores)


def build_submission_scores(cache_scores: CacheScores, score_name: str) -> dict[str, dict[str, float]]:
    score_key = SCORE_NAME_MAP[score_name]
    payload: dict[str, dict[str, float]] = {}
    for problem_id, item in sorted(cache_scores.problem_scores.items(), key=lambda kv: _problem_sort_key(kv[0])):
        scores = np.asarray(item[score_key], dtype=np.float64)
        sample_ids = item["sample_ids"]
        payload[str(problem_id)] = {
            str(sample_id): float(score)
            for sample_id, score in zip(sample_ids, scores.tolist())
        }
    return payload


def load_correctness(cache_root: str | Path) -> dict[int, bool]:
    return _load_ground_truth(Path(cache_root))


def _safe_mean(values: list[float | None]) -> float | None:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not finite:
        return None
    return float(np.mean(finite))


def _score_metric(problem_scores: dict[str, dict[str, Any]], correctness: dict[int, bool], score_key: str) -> ScoreMetricResult:
    all_scores: list[float] = []
    all_labels: list[int] = []
    hit1_total = 0
    hit3_total = 0
    pair_num = 0.0
    pair_den = 0.0
    n_problems = 0

    for problem_id, item in sorted(problem_scores.items(), key=lambda kv: _problem_sort_key(kv[0])):
        sample_ids = np.asarray(item["sample_ids"], dtype=np.int64)
        scores = np.asarray(item[score_key], dtype=np.float64)
        labels = np.asarray([int(bool(correctness[int(sample_id)])) for sample_id in sample_ids], dtype=np.int32)
        if sample_ids.size == 0:
            continue

        order = np.argsort(-scores, kind="mergesort")
        hit1_total += int(labels[order[0]] > 0)
        topk = order[: min(3, len(order))]
        hit3_total += int(np.any(labels[topk] > 0))

        pos_scores = scores[labels > 0]
        neg_scores = scores[labels <= 0]
        if pos_scores.size > 0 and neg_scores.size > 0:
            diff = pos_scores[:, None] - neg_scores[None, :]
            pair_num += float((diff > 0).sum()) + 0.5 * float((diff == 0).sum())
            pair_den += float(diff.size)

        all_scores.extend(scores.tolist())
        all_labels.extend(labels.tolist())
        n_problems += 1

    labels_arr = np.asarray(all_labels, dtype=np.int32)
    scores_arr = np.asarray(all_scores, dtype=np.float64)
    auroc = None
    if labels_arr.size > 0 and np.unique(labels_arr).size >= 2:
        auroc = float(roc_auc_score(labels_arr, scores_arr))

    selective_acc = 0.0
    if labels_arr.size > 0:
        keep = max(1, int(math.ceil(0.10 * labels_arr.size)))
        order = np.argsort(-scores_arr, kind="mergesort")[:keep]
        selective_acc = float(labels_arr[order].mean())

    pairwise_accuracy = None if pair_den <= 0 else float(pair_num / pair_den)

    return ScoreMetricResult(
        auroc=auroc,
        hit_at_1=float(hit1_total / n_problems) if n_problems else 0.0,
        hit_at_3=float(hit3_total / n_problems) if n_problems else 0.0,
        selective_acc_at_10pct=selective_acc,
        pairwise_accuracy=pairwise_accuracy,
        n_problems=int(n_problems),
        n_samples=int(labels_arr.size),
        n_positive=int(labels_arr.sum()),
        n_negative=int((labels_arr == 0).sum()),
    )


def evaluate_cache_scores(cache_scores: CacheScores, correctness: dict[int, bool]) -> CacheMetricBundle:
    by_score = {
        score_name: _score_metric(cache_scores.problem_scores, correctness, score_key)
        for score_name, score_key in SCORE_NAME_MAP.items()
    }
    return CacheMetricBundle(entry=cache_scores.entry, by_score=by_score)


def mean_metric_bundle(bundles: list[CacheMetricBundle]) -> MeanMetricBundle:
    out: dict[str, dict[str, float | None]] = {}
    for score_name in SCORE_NAME_MAP:
        out[score_name] = {
            "auroc": _safe_mean([bundle.by_score[score_name].auroc for bundle in bundles]),
            "hit_at_1": _safe_mean([bundle.by_score[score_name].hit_at_1 for bundle in bundles]),
            "hit_at_3": _safe_mean([bundle.by_score[score_name].hit_at_3 for bundle in bundles]),
            "selective_acc_at_10pct": _safe_mean([bundle.by_score[score_name].selective_acc_at_10pct for bundle in bundles]),
            "pairwise_accuracy": _safe_mean([bundle.by_score[score_name].pairwise_accuracy for bundle in bundles]),
        }
    return MeanMetricBundle(by_score=out)


def load_model(path: str | Path):
    import joblib

    return joblib.load(Path(path))


def format_threshold_tag(reflection_threshold: float) -> str:
    return f"ref{int(round(float(reflection_threshold) * 100.0)):03d}"


def default_submission_filename(score_name: str, reflection_threshold: float, num_tuples: int) -> str:
    tag = format_threshold_tag(reflection_threshold)
    if score_name == "best_only":
        return f"best_only_{tag}_t{int(num_tuples)}.json"
    if score_name == "mix":
        return f"mix_{tag}_t{int(num_tuples)}.json"
    raise KeyError(f"Unknown score_name: {score_name}")


def default_method_name(score_name: str, reflection_threshold: float, num_tuples: int) -> str:
    tag = format_threshold_tag(reflection_threshold)
    if score_name == "best_only":
        return f"extreme8_best_only_{tag}_t{int(num_tuples)}"
    if score_name == "mix":
        return f"extreme8_mix_{tag}_t{int(num_tuples)}"
    raise KeyError(f"Unknown score_name: {score_name}")


def validate_cache_scores(cache_scores: CacheScores, expected_samples_per_problem: int | None = 64) -> None:
    if not cache_scores.problem_scores:
        raise ValueError(f"{cache_scores.entry.cache_key}: no problem scores were produced")

    for problem_id, item in sorted(cache_scores.problem_scores.items(), key=lambda kv: _problem_sort_key(kv[0])):
        sample_ids = [int(sample_id) for sample_id in item.get("sample_ids", [])]
        run_indices = [int(run_index) for run_index in item.get("run_indices", [])]
        score_best = list(item.get("score_best", []))
        score_worst = list(item.get("score_worst", []))
        score_mix = list(item.get("score_mix", []))
        counts = list(item.get("counts", []))
        num_tuples = int(item.get("num_tuples", 0))

        if not sample_ids:
            raise ValueError(f"{cache_scores.entry.cache_key}/{problem_id}: empty sample_ids")
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError(f"{cache_scores.entry.cache_key}/{problem_id}: duplicate sample_ids")
        lengths = {
            len(sample_ids),
            len(run_indices),
            len(score_best),
            len(score_worst),
            len(score_mix),
            len(counts),
        }
        if len(lengths) != 1:
            raise ValueError(
                f"{cache_scores.entry.cache_key}/{problem_id}: inconsistent vector lengths {sorted(lengths)}"
            )
        if expected_samples_per_problem is not None and len(sample_ids) != int(expected_samples_per_problem):
            raise ValueError(
                f"{cache_scores.entry.cache_key}/{problem_id}: expected {expected_samples_per_problem} "
                f"samples, got {len(sample_ids)}"
            )
        if num_tuples <= 0:
            raise ValueError(f"{cache_scores.entry.cache_key}/{problem_id}: num_tuples must be positive")

        for score_name, values in {
            "score_best": score_best,
            "score_worst": score_worst,
            "score_mix": score_mix,
            "counts": counts,
        }.items():
            arr = np.asarray(values, dtype=np.float64)
            if not np.isfinite(arr).all():
                raise ValueError(f"{cache_scores.entry.cache_key}/{problem_id}: non-finite values in {score_name}")


def summarize_cache_scores(cache_scores: CacheScores) -> dict[str, int]:
    n_problems = len(cache_scores.problem_scores)
    n_samples = sum(len(item.get("sample_ids", [])) for item in cache_scores.problem_scores.values())
    return {
        "n_problems": int(n_problems),
        "n_samples": int(n_samples),
    }


def build_submission_payload(
    cache_scores_list: list[CacheScores],
    score_name: str,
    method_name: str,
) -> dict[str, Any]:
    if score_name not in SCORE_NAME_MAP:
        raise KeyError(f"Unknown score_name: {score_name}")

    scores: dict[str, dict[str, dict[str, float]]] = {}
    for cache_scores in cache_scores_list:
        cache_key = cache_scores.entry.cache_key
        if cache_key in scores:
            raise ValueError(f"Duplicate cache_key in submission payload: {cache_key}")
        scores[cache_key] = build_submission_scores(cache_scores, score_name)

    return {
        "task": "best_of_n",
        "method_name": str(method_name),
        "scores": scores,
    }


def validate_submission_payload(
    payload: dict[str, Any],
    expected_cache_keys: list[str] | None = None,
) -> dict[str, int]:
    if payload.get("task") != "best_of_n":
        raise ValueError(f"submission task must be 'best_of_n', got {payload.get('task')!r}")

    method_name = payload.get("method_name")
    if not isinstance(method_name, str) or not method_name.strip():
        raise ValueError("submission method_name must be a non-empty string")

    scores = payload.get("scores")
    if not isinstance(scores, dict) or not scores:
        raise ValueError("submission scores must be a non-empty mapping")

    if expected_cache_keys is not None:
        expected = set(expected_cache_keys)
        actual = set(scores.keys())
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(f"submission cache keys mismatch; missing={missing}, extra={extra}")

    total_problems = 0
    total_samples = 0
    for cache_key, problem_map in sorted(scores.items()):
        if not isinstance(problem_map, dict) or not problem_map:
            raise ValueError(f"{cache_key}: problem map must be a non-empty mapping")
        total_problems += len(problem_map)

        for problem_id, sample_map in sorted(problem_map.items(), key=lambda kv: _problem_sort_key(kv[0])):
            if not isinstance(sample_map, dict) or not sample_map:
                raise ValueError(f"{cache_key}/{problem_id}: sample score map must be non-empty")
            total_samples += len(sample_map)

            for sample_id, score in sample_map.items():
                try:
                    int(sample_id)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{cache_key}/{problem_id}: invalid sample_id {sample_id!r}") from exc
                if not np.isfinite(float(score)):
                    raise ValueError(f"{cache_key}/{problem_id}/{sample_id}: score must be finite")

    return {
        "cache_keys": int(len(scores)),
        "problems": int(total_problems),
        "samples": int(total_samples),
    }


def write_submission_payload(payload: dict[str, Any], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path
