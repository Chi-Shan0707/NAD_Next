"""
Early Stop v3 — domain-aware with neuron-structural signals for coding.

Domain routing:
  math    (aime24, aime25, brumo25, hmmt25) → tok_conf prefix mean (v1/v2 baseline, AUROC 0.68-0.76)
  science (gpqa)                            → tok_conf prefix mean (AUROC 0.659 — check scan for blend)
  coding  (livecodebench_v5)               → traj_continuity OR traj_reflection_count from scan results
                                              (falls back to tok_gini if neuron data unavailable)

Routing selection: update CODING_SIGNAL after reviewing scan_earlystop_signals.py output.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from nad.core.selectors.trajectory_impl import (
    DEFAULT_REFLECTION_THRESHOLD,
    _compute_trajectory_scores,
    _extract_slice_keysets,
)
from nad.core.views.reader import CacheReader
from nad.ops.earlystop import (
    CacheEntry,
    EARLY_STOP_POSITIONS,
    N_POSITIONS,
    build_problem_groups,
    discover_cache_entries,
    validate_earlystop_payload,
    write_earlystop_payload,
    _problem_sort_key,
)

# Datasets by domain
MATH_DATASETS = {"aime24", "aime25", "brumo25", "hmmt25"}
SCIENCE_DATASETS = {"gpqa"}
CODING_DATASETS = {"livecodebench_v5", "lcb_v5"}

# Coding signal selection — update after reviewing scan results.
# Options: "traj_continuity", "traj_reflection_count", "tok_gini", "nc_mean"
# "traj_continuity": mean Jaccard between consecutive slices — neuron stability = correct coding
# "traj_reflection_count": fewer reflections → more direct code
# Direction: traj_continuity higher=better, traj_reflection_count lower=better (negate)
CODING_SIGNAL = "traj_continuity"  # UPDATE after scan

# Optional: blend token signal into coding score (set to 0.0 to disable)
CODING_TOKEN_BLEND = 0.0   # weight for tok_gini in blend: score = (1-w)*traj + w*tok_gini
CODING_TOKEN_FIELD = "tok_gini"

# Optional: gpqa blend — set GPQA_TRAJ_BLEND > 0 to mix traj_continuity with tok_conf
GPQA_TRAJ_BLEND = 0.0  # weight for traj feature in gpqa: score = (1-w)*tok_conf + w*traj


def get_domain(dataset_name: str) -> str:
    if dataset_name in MATH_DATASETS:
        return "math"
    if dataset_name in SCIENCE_DATASETS:
        return "science"
    if dataset_name in CODING_DATASETS:
        return "coding"
    return "math"  # safe default


def _prefix_mean(arr: np.ndarray, p: float) -> float:
    T = len(arr)
    if T == 0:
        return 0.0
    return float(np.mean(arr[:max(1, int(p * T))]))


def compute_scores_math_science(tv) -> list[float]:
    """tok_conf prefix mean — higher = more correct (math + science)."""
    arr = np.asarray(tv.tok_conf, dtype=np.float64)
    return [_prefix_mean(arr, p) for p in EARLY_STOP_POSITIONS]


def _traj_feature(slices: list, k: int, signal: str) -> float:
    """Compute one trajectory feature using the first k slices."""
    if k == 0 or not slices:
        return 0.0
    traj = _compute_trajectory_scores(slices[:k], reflection_threshold=DEFAULT_REFLECTION_THRESHOLD)
    if signal == "traj_continuity":
        return float(traj["mean_continuity"])
    elif signal == "traj_reflection_count":
        # Negate: fewer reflections = higher score = more correct
        return -float(traj["reflection_count"])
    elif signal == "traj_novelty":
        return float(traj["mean_novelty"])
    elif signal == "traj_max_reflection":
        return -float(traj["max_reflection"])
    elif signal == "nc_mean":
        return 0.0  # placeholder; nc_mean computed separately
    else:
        return 0.0


def compute_scores_coding(reader: CacheReader, run_id: int, tv) -> list[float]:
    """
    Compute coding scores using structural neuron signals.
    Falls back to tok_gini if rows/ bank unavailable.
    """
    slices = _extract_slice_keysets(reader, int(run_id))
    n_slices = len(slices)

    scores = []
    for p in EARLY_STOP_POSITIONS:
        k = max(1, int(p * n_slices)) if n_slices > 0 else 0

        if n_slices > 0:
            traj_score = _traj_feature(slices, k, CODING_SIGNAL)
        else:
            # Fallback to tok_gini if no rows/ bank
            traj_score = None

        if traj_score is None:
            # Pure tok_gini fallback
            if tv is not None and tv.tok_gini is not None:
                arr = np.asarray(tv.tok_gini, dtype=np.float64)
                scores.append(_prefix_mean(arr, p))
            else:
                scores.append(0.0)
            continue

        if CODING_TOKEN_BLEND > 0.0 and tv is not None:
            gini_arr = getattr(tv, CODING_TOKEN_FIELD, None)
            if gini_arr is not None and len(gini_arr) > 0:
                tok_score = _prefix_mean(np.asarray(gini_arr, dtype=np.float64), p)
                traj_score = (1.0 - CODING_TOKEN_BLEND) * traj_score + CODING_TOKEN_BLEND * tok_score

        scores.append(float(traj_score))

    return scores


def compute_scores_science_blend(reader: CacheReader, run_id: int, tv) -> list[float]:
    """
    gpqa: tok_conf baseline, optionally blended with traj feature.
    """
    tok_conf_scores = compute_scores_math_science(tv)
    if GPQA_TRAJ_BLEND <= 0.0:
        return tok_conf_scores

    slices = _extract_slice_keysets(reader, int(run_id))
    n_slices = len(slices)

    scores = []
    for i, p in enumerate(EARLY_STOP_POSITIONS):
        k = max(1, int(p * n_slices)) if n_slices > 0 else 0
        tc = tok_conf_scores[i]
        if n_slices > 0:
            traj = _traj_feature(slices, k, "traj_continuity")
            blended = (1.0 - GPQA_TRAJ_BLEND) * tc + GPQA_TRAJ_BLEND * traj
        else:
            blended = tc
        scores.append(float(blended))
    return scores


def compute_earlystop_scores_v3(
    reader: CacheReader,
    run_id: int,
    tv,
    domain: str,
) -> list[float]:
    """Route to domain scorer. Returns 10 floats for positions 10%..100%."""
    if tv is None:
        return [0.0] * N_POSITIONS

    if domain == "coding":
        return compute_scores_coding(reader, run_id, tv)
    elif domain == "science":
        if tv.tok_conf is None:
            return [0.0] * N_POSITIONS
        return compute_scores_science_blend(reader, run_id, tv)
    else:  # math
        if tv.tok_conf is None:
            return [0.0] * N_POSITIONS
        return compute_scores_math_science(tv)


def score_cache_entry_earlystop_v3(
    entry: CacheEntry,
    max_problems: int | None = None,
) -> dict[str, dict[str, list[float]]]:
    """
    Domain-aware v3 scoring with neuron-structural signals.
    Returns problem_scores: {problem_id: {sample_id: [10 floats]}}
    """
    domain = get_domain(entry.dataset_name)
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
            scores_10 = compute_earlystop_scores_v3(reader, int(sample_id), tv, domain)
            run_scores[str(sample_id)] = scores_10
        problem_scores[str(problem_id)] = run_scores
    return problem_scores
