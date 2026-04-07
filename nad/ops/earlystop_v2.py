"""
Early Stop v2 — domain-aware scoring.

Domain routing (by dataset_name):
  math    (aime24, aime25, brumo25, hmmt25) → tok_conf prefix mean (higher = more correct)
  science (gpqa)                            → tok_conf prefix mean (tok_neg_entropy tested but WORSE: 0.631 vs 0.659)
  coding  (livecodebench_v5)               → tok_gini prefix mean (best available: rho≈+0.05; AUROC≈0.52 — Phase 3 work)

Validated on MUI_HUB DS-R1 labelled data 2026-04-04:
  math   AUROC @10%: 0.68–0.76 ★
  gpqa   AUROC @10%: 0.659 (tok_conf), 0.631 (tok_neg_entropy) — tok_conf wins
  coding AUROC @10%: 0.509 (tok_conf/tok_gini both ~random)
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from nad.core.views.reader import CacheReader
from nad.ops.earlystop import (
    CacheEntry,
    EARLY_STOP_POSITIONS,
    N_POSITIONS,
    build_earlystop_payload,
    build_problem_groups,
    discover_cache_entries,
    validate_earlystop_payload,
    write_earlystop_payload,
    _problem_sort_key,
)

# Datasets by domain (use submission names / raw dataset dir names both)
MATH_DATASETS = {"aime24", "aime25", "brumo25", "hmmt25"}
SCIENCE_DATASETS = {"gpqa"}
CODING_DATASETS = {"livecodebench_v5", "lcb_v5"}


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


def compute_scores_math(tv) -> list[float]:
    """tok_conf prefix mean — higher = more correct."""
    arr = np.asarray(tv.tok_conf, dtype=np.float64)
    return [_prefix_mean(arr, p) for p in EARLY_STOP_POSITIONS]


def compute_scores_coding(tv) -> list[float]:
    """tok_gini prefix mean — best available coding signal (rho≈+0.05, AUROC≈0.52).
    Higher gini = more concentrated token distribution = more decisive generation.
    Replace when Phase 3 coding experiment identifies stronger signal."""
    arr = np.asarray(tv.tok_gini, dtype=np.float64)
    return [_prefix_mean(arr, p) for p in EARLY_STOP_POSITIONS]


def compute_earlystop_scores_v2(tv, domain: str) -> list[float]:
    """Route to domain-specific scorer. Returns 10 floats for positions 10%..100%.

    Science uses tok_conf (same as math) — tok_neg_entropy was tested and is worse.
    Coding uses tok_gini — slightly better than tok_conf (rho +0.05 vs +0.015).
    """
    if tv is None:
        return [0.0] * N_POSITIONS
    if domain == "coding":
        if tv.tok_gini is None:
            return [0.0] * N_POSITIONS
        return compute_scores_coding(tv)
    else:  # math + science: tok_conf is best
        if tv.tok_conf is None:
            return [0.0] * N_POSITIONS
        return compute_scores_math(tv)


def score_cache_entry_earlystop_v2(
    entry: CacheEntry,
    max_problems: int | None = None,
) -> dict[str, dict[str, list[float]]]:
    """
    Domain-aware scoring.
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
            scores_10 = compute_earlystop_scores_v2(tv, domain)
            run_scores[str(sample_id)] = scores_10
        problem_scores[str(problem_id)] = run_scores
    return problem_scores
