from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover - optional dependency in some environments
    roc_auc_score = None


def _sorted_problem_ids(problem_groups: Mapping[str, Sequence[int]]) -> list[str]:
    return sorted((str(pid) for pid in problem_groups.keys()), key=lambda pid: (0, pid))


def _local_order(scores: np.ndarray, sample_ids: np.ndarray) -> np.ndarray:
    return np.asarray(
        sorted(range(len(sample_ids)), key=lambda idx: (-float(scores[idx]), int(sample_ids[idx]))),
        dtype=np.int64,
    )


def _pairwise_auc_local(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = np.asarray(scores[labels > 0], dtype=np.float64)
    neg = np.asarray(scores[labels <= 0], dtype=np.float64)
    if pos.size <= 0 or neg.size <= 0:
        return float("nan")
    wins = float((pos[:, None] > neg[None, :]).sum())
    ties = float((pos[:, None] == neg[None, :]).sum())
    total = float(pos.size * neg.size)
    return (wins + 0.5 * ties) / total if total > 0 else float("nan")


def difficulty_bucket_summary(
    problem_records: Sequence[Mapping[str, Any]],
    *,
    difficulty_threshold: float = 0.5,
) -> dict[str, dict[str, float | int | None]]:
    buckets = {
        "hard": [],
        "easy": [],
    }
    for record in problem_records:
        oracle_rate = float(record.get("oracle_positive_rate", 0.0) or 0.0)
        key = "hard" if oracle_rate < float(difficulty_threshold) else "easy"
        buckets[key].append(record)

    out: dict[str, dict[str, float | int | None]] = {}
    for key, rows in buckets.items():
        if not rows:
            out[key] = {
                "n_problems": 0,
                "top1_accuracy": None,
                "baseline_top1_accuracy": None,
                "pass@1_uplift_abs": None,
            }
            continue
        top1 = np.array([int(bool(row.get("selected_is_correct", False))) for row in rows], dtype=np.float64)
        baseline_vals = [
            row.get("baseline_selected_is_correct")
            for row in rows
            if row.get("baseline_selected_is_correct") is not None
        ]
        baseline_top1 = float(np.mean(baseline_vals)) if baseline_vals else None
        uplift = None
        if baseline_top1 is not None:
            uplift = float(np.mean(top1) - baseline_top1)
        out[key] = {
            "n_problems": int(len(rows)),
            "top1_accuracy": float(np.mean(top1)),
            "baseline_top1_accuracy": baseline_top1,
            "pass@1_uplift_abs": uplift,
        }
    return out


def evaluate_grouped_scores(
    problem_groups: Mapping[str, Sequence[int]],
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    sample_ids: np.ndarray | None = None,
    baseline_scores: np.ndarray | None = None,
    top_fraction: float = 0.10,
) -> dict[str, Any]:
    labels_arr = np.asarray(labels, dtype=np.int32).reshape(-1)
    scores_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels_arr.shape[0] != scores_arr.shape[0]:
        raise ValueError(f"labels/scores length mismatch: {labels_arr.shape[0]} vs {scores_arr.shape[0]}")
    sample_id_arr = (
        np.arange(labels_arr.shape[0], dtype=np.int64)
        if sample_ids is None
        else np.asarray(sample_ids, dtype=np.int64).reshape(-1)
    )
    if sample_id_arr.shape[0] != labels_arr.shape[0]:
        raise ValueError(f"sample_ids length mismatch: {sample_id_arr.shape[0]} vs {labels_arr.shape[0]}")
    baseline_arr = None if baseline_scores is None else np.asarray(baseline_scores, dtype=np.float64).reshape(-1)
    if baseline_arr is not None and baseline_arr.shape[0] != labels_arr.shape[0]:
        raise ValueError(f"baseline_scores length mismatch: {baseline_arr.shape[0]} vs {labels_arr.shape[0]}")

    problem_records: list[dict[str, Any]] = []
    pairwise_vals: list[float] = []
    selected_hits: list[float] = []
    baseline_hits: list[float] = []
    head_to_head_wins = 0
    head_to_head_losses = 0
    head_to_head_ties = 0
    local_selacc_vals: list[float] = []

    for problem_id in _sorted_problem_ids(problem_groups):
        idx = np.asarray(problem_groups[problem_id], dtype=np.int64)
        if idx.size <= 0:
            continue
        local_scores = scores_arr[idx]
        local_labels = labels_arr[idx]
        local_sample_ids = sample_id_arr[idx]
        order = _local_order(local_scores, local_sample_ids)
        best_local = int(order[0]) if order.size else 0
        best_idx = int(idx[best_local]) if idx.size else 0
        topk = max(1, int(math.ceil(float(top_fraction) * idx.size)))
        topk_local = order[:topk]
        pairwise_local = _pairwise_auc_local(local_scores, local_labels)
        if np.isfinite(pairwise_local):
            pairwise_vals.append(float(pairwise_local))
        local_selacc = float(local_labels[topk_local].mean()) if topk_local.size > 0 else 0.0
        local_selacc_vals.append(local_selacc)

        baseline_best_idx = None
        baseline_is_correct = None
        if baseline_arr is not None:
            local_baseline = baseline_arr[idx]
            baseline_order = _local_order(local_baseline, local_sample_ids)
            baseline_best_idx = int(idx[int(baseline_order[0])]) if baseline_order.size else int(idx[0])
            baseline_is_correct = bool(labels_arr[baseline_best_idx] > 0)

        selected_is_correct = bool(labels_arr[best_idx] > 0)
        selected_hits.append(float(selected_is_correct))
        if baseline_is_correct is not None:
            baseline_hits.append(float(baseline_is_correct))
            if selected_is_correct and not baseline_is_correct:
                head_to_head_wins += 1
            elif baseline_is_correct and not selected_is_correct:
                head_to_head_losses += 1
            else:
                head_to_head_ties += 1

        oracle_positive_count = int(local_labels.sum())
        best_positive_rank = None
        if oracle_positive_count > 0:
            positive_positions = np.where(local_labels[order] > 0)[0]
            if positive_positions.size > 0:
                best_positive_rank = int(positive_positions[0]) + 1

        problem_records.append(
            {
                "problem_id": str(problem_id),
                "n_candidates": int(idx.size),
                "oracle_positive_count": oracle_positive_count,
                "oracle_positive_rate": float(local_labels.mean()),
                "selected_local_index": int(best_local),
                "selected_sample_id": int(sample_id_arr[best_idx]),
                "selected_score": float(scores_arr[best_idx]),
                "selected_is_correct": selected_is_correct,
                "best_positive_rank": best_positive_rank,
                "local_pairwise_auc": float(pairwise_local) if np.isfinite(pairwise_local) else None,
                "local_selacc@10%": float(local_selacc),
                "baseline_selected_sample_id": None if baseline_best_idx is None else int(sample_id_arr[baseline_best_idx]),
                "baseline_selected_is_correct": baseline_is_correct,
            }
        )

    pooled_auroc = None
    if roc_auc_score is not None and labels_arr.size > 0 and np.unique(labels_arr).size >= 2:
        pooled_auroc = float(roc_auc_score(labels_arr, scores_arr))

    baseline_top1 = float(np.mean(baseline_hits)) if baseline_hits else None
    top1_accuracy = float(np.mean(selected_hits)) if selected_hits else 0.0
    head_to_head_non_ties = head_to_head_wins + head_to_head_losses
    pass_uplift_abs = None if baseline_top1 is None else float(top1_accuracy - baseline_top1)

    return {
        "top1_accuracy": top1_accuracy,
        "pass@1": top1_accuracy,
        "baseline_top1_accuracy": baseline_top1,
        "pass@1_uplift_abs": pass_uplift_abs,
        "pairwise_auc": float(np.mean(pairwise_vals)) if pairwise_vals else None,
        "pairwise_win_rate": float(np.mean(pairwise_vals)) if pairwise_vals else None,
        "local_selacc@10%": float(np.mean(local_selacc_vals)) if local_selacc_vals else 0.0,
        "pooled_pointwise_auroc": pooled_auroc,
        "n_problems": int(len(problem_records)),
        "n_samples": int(labels_arr.size),
        "head_to_head": {
            "wins": int(head_to_head_wins),
            "losses": int(head_to_head_losses),
            "ties": int(head_to_head_ties),
            "win_rate_non_ties": (
                float(head_to_head_wins / head_to_head_non_ties) if head_to_head_non_ties > 0 else None
            ),
        },
        "difficulty_summary": difficulty_bucket_summary(problem_records),
        "problem_records": problem_records,
    }


def write_problem_records_jsonl(problem_records: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for record in problem_records:
            fh.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
