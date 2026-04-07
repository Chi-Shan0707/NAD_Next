from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from nad.core.views.reader import CacheReader

MODEL_SUBMISSION_NAMES = {
    "DeepSeek-R1-0528-Qwen3-8B": "DS-R1",
    "Qwen3-4B-Thinking-2507": "Qwen3-4B",
}

DATASET_SUBMISSION_NAMES = {
    "livecodebench_v5": "lcb_v5",
}

EARLY_STOP_POSITIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_POSITIONS = len(EARLY_STOP_POSITIONS)  # = 10


@dataclass(frozen=True)
class CacheEntry:
    cache_key: str
    cache_root: Path
    model_name: str
    dataset_name: str


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


def compute_earlystop_scores_for_sample(tok_conf: np.ndarray) -> list[float]:
    """
    Return 10 scores for positions [10%, 20%, ..., 100%] of generation.
    Score = mean(tok_conf[:T_p]).  Higher = more likely correct (for math).
    """
    T = len(tok_conf)
    if T == 0:
        return [0.0] * N_POSITIONS
    return [
        float(np.mean(tok_conf[:max(1, int(p * T))]))
        for p in EARLY_STOP_POSITIONS
    ]


def score_cache_entry_earlystop(
    entry: CacheEntry,
    max_problems: int | None = None,
) -> dict[str, dict[str, list[float]]]:
    """
    Returns problem_scores: {problem_id: {sample_id: [10 floats]}}
    """
    meta = json.loads((entry.cache_root / "meta.json").read_text(encoding="utf-8"))
    groups = build_problem_groups(meta)
    reader = CacheReader(str(entry.cache_root))

    problem_scores: dict[str, dict[str, list[float]]] = {}
    for problem_index, (problem_id, sample_ids) in enumerate(
        sorted(groups.items(), key=lambda kv: _problem_sort_key(kv[0]))
    ):
        if max_problems is not None and problem_index >= max_problems:
            break
        run_scores: dict[str, list[float]] = {}
        for sample_id in sample_ids:
            tv = reader.get_token_view(int(sample_id))
            if tv is None or tv.tok_conf is None:
                scores_10 = [0.0] * N_POSITIONS
            else:
                arr = np.asarray(tv.tok_conf, dtype=np.float64)
                scores_10 = compute_earlystop_scores_for_sample(arr)
            run_scores[str(sample_id)] = scores_10
        problem_scores[str(problem_id)] = run_scores
    return problem_scores


def build_earlystop_payload(
    cache_scores_list: list[tuple[str, dict]],
    method_name: str,
) -> dict:
    return {
        "task": "early_stop",
        "method_name": method_name,
        "scores": {cache_key: ps for cache_key, ps in cache_scores_list},
    }


def validate_earlystop_payload(payload: dict) -> dict:
    """
    Validates task="early_stop", all entries present,
    each sample_id maps to a list of exactly 10 finite floats.
    """
    assert payload["task"] == "early_stop", f"Expected task='early_stop', got {payload['task']!r}"
    scores = payload["scores"]
    total_problems = 0
    total_samples = 0
    for cache_key, problem_map in scores.items():
        for problem_id, sample_map in problem_map.items():
            total_problems += 1
            for sample_id, score_list in sample_map.items():
                assert isinstance(score_list, list) and len(score_list) == N_POSITIONS, (
                    f"{cache_key}/{problem_id}/{sample_id}: expected list of {N_POSITIONS}, "
                    f"got {type(score_list)} len={len(score_list) if isinstance(score_list, list) else '?'}"
                )
                assert all(np.isfinite(s) for s in score_list), (
                    f"{cache_key}/{problem_id}/{sample_id}: non-finite scores"
                )
                total_samples += 1
    return {"total_problems": total_problems, "total_samples": total_samples}


def write_earlystop_payload(payload: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
