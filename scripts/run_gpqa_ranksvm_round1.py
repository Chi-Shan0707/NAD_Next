#!/usr/bin/env python3
"""
GPQA RankSVM Round 1 Evaluation

Runs LOPO (leave-one-problem-out) evaluation of a narrow GPQA-only RankSVM
baseline against:
  - science_baseline_v1
  - tournament-copeland
  - existing GPQA metrics/gates used by gpqa_pairwise_round1

This experiment intentionally changes only:
  Bradley-Terry logistic pairwise objective -> linear pairwise hinge objective

and uses direct per-run utility inference:
  u_i = w · x_i

Final model is saved to:
  models/ml_selectors/gpqa_ranksvm_round1.pkl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.core.selectors.gpqa_ranksvm_impl import (
    GPQARankSVMScorer,
    build_pairwise_hinge_training_examples,
)
from nad.core.selectors.science_dynamic_impl import (
    DEFAULT_SCIENCE_DYNAMIC_WEIGHTS,
    compute_science_dynamic_primary_scores_from_raw,
)
from scripts.run_code_baseline_v1_phase2 import (
    MetricAccumulator,
    _compute_copeland_scores,
)
from scripts.run_gpqa_pairwise_round1 import (
    _DEFAULT_DISTANCE_THREADS,
    _DEFAULT_WORKERS,
    _ProblemData,
    _check_gate,
    _extract_all_problems,
    _summarize_table,
)

_DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "ml_selectors" / "gpqa_ranksvm_round1.pkl"


def _run_lopo(
    all_problems: list[_ProblemData],
    acc: MetricAccumulator,
) -> None:
    """Leave-one-problem-out training + evaluation of GPQARankSVMScorer."""
    n_total = len(all_problems)

    for held_out_idx, held_out in enumerate(all_problems):
        if held_out_idx % 20 == 0:
            print(f"[gpqa_ranksvm_round1] fold {held_out_idx + 1}/{n_total}", flush=True)

        pair_X_list: list[np.ndarray] = []
        pair_y_list: list[np.ndarray] = []
        for idx, prob in enumerate(all_problems):
            if idx == held_out_idx:
                continue
            X_pairs, y_pairs = build_pairwise_hinge_training_examples(prob.X, prob.labels)
            if X_pairs.shape[0] > 0:
                pair_X_list.append(X_pairs)
                pair_y_list.append(y_pairs)

        if not pair_X_list:
            scores = np.zeros(len(held_out.run_ids), dtype=np.float64)
        else:
            scorer = GPQARankSVMScorer(C=1.0)
            scorer.fit(
                np.concatenate(pair_X_list, axis=0),
                np.concatenate(pair_y_list, axis=0),
            )
            scores = scorer.score_group(held_out.X)

        acc.add_problem(
            held_out.problem_id,
            held_out.run_ids,
            scores,
            held_out.labels,
            held_out.D,
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="GPQA RankSVM round-1 LOPO evaluation"
    )
    ap.add_argument(
        "--cache-root",
        required=True,
        help="Path to GPQA cache directory (contains meta.json + evaluation_report*.json)",
    )
    ap.add_argument(
        "--out",
        default="",
        help="Optional path to write JSON results (default: print only)",
    )
    ap.add_argument(
        "--model-out",
        default="",
        help=f"Path to save final model (default: {_DEFAULT_MODEL_PATH})",
    )
    ap.add_argument(
        "--distance-threads",
        type=int,
        default=_DEFAULT_DISTANCE_THREADS,
        help="Jaccard threads per worker (workers × distance-threads ≤ 16)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=_DEFAULT_WORKERS,
        help="Parallel extraction workers (workers × distance-threads ≤ 16)",
    )
    args = ap.parse_args()

    cache_root = Path(args.cache_root)
    model_out = Path(args.model_out) if args.model_out else _DEFAULT_MODEL_PATH

    print("[run_gpqa_ranksvm_round1] Extracting features for all problems …", flush=True)
    all_problems = _extract_all_problems(
        cache_root,
        distance_threads=args.distance_threads,
        workers=args.workers,
    )
    print(f"[run_gpqa_ranksvm_round1] {len(all_problems)} problems extracted.", flush=True)

    acc_baseline = MetricAccumulator("science_baseline_v1", use_code_tiebreak=True)
    acc_copeland = MetricAccumulator("tournament-copeland", use_code_tiebreak=True)

    for prob in all_problems:
        baseline_scores, _ = compute_science_dynamic_primary_scores_from_raw(
            prob.sci_raw,
            weights=DEFAULT_SCIENCE_DYNAMIC_WEIGHTS,
        )
        copeland_scores = _compute_copeland_scores(prob.D)

        acc_baseline.add_problem(
            prob.problem_id,
            prob.run_ids,
            baseline_scores,
            prob.labels,
            prob.D,
        )
        acc_copeland.add_problem(
            prob.problem_id,
            prob.run_ids,
            copeland_scores,
            prob.labels,
            prob.D,
        )

    print("[run_gpqa_ranksvm_round1] Running LOPO evaluation …", flush=True)
    acc_ranksvm = MetricAccumulator("gpqa_ranksvm_round1", use_code_tiebreak=True)
    _run_lopo(all_problems, acc_ranksvm)

    metrics_baseline = acc_baseline.finalize()
    metrics_copeland = acc_copeland.finalize()
    metrics_ranksvm = acc_ranksvm.finalize()

    all_metrics = {
        "science_baseline_v1": metrics_baseline,
        "tournament-copeland": metrics_copeland,
        "gpqa_ranksvm_round1": metrics_ranksvm,
    }

    rows = [
        {"name": name, **{k: m[k] for k in ("auroc", "hit@1", "pairwise", "selacc@10%")}}
        for name, m in all_metrics.items()
    ]
    print()
    print(_summarize_table(rows))
    print()

    passed, failed = _check_gate(metrics_ranksvm)
    if passed:
        print("PROMOTE: gpqa_ranksvm_round1 passes all thresholds")
        print("NOTE: implementation keeps this as an offline baseline; no defaults are changed")
    else:
        print("NO-PROMOTE: gpqa_ranksvm_round1 failed the following thresholds:")
        for msg in failed:
            print(f"  - {msg}")
    print()

    payload: dict[str, Any] = {
        "metrics": all_metrics,
        "gate": {
            "passed": passed,
            "failed_thresholds": failed,
        },
        "n_problems": len(all_problems),
        "model_path": str(model_out),
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Results written to: {out_path}", flush=True)

    print("[run_gpqa_ranksvm_round1] Training final model on all problems …", flush=True)
    all_pair_X: list[np.ndarray] = []
    all_pair_y: list[np.ndarray] = []
    for prob in all_problems:
        X_pairs, y_pairs = build_pairwise_hinge_training_examples(prob.X, prob.labels)
        if X_pairs.shape[0] > 0:
            all_pair_X.append(X_pairs)
            all_pair_y.append(y_pairs)

    if all_pair_X:
        final_scorer = GPQARankSVMScorer(C=1.0)
        final_scorer.fit(
            np.concatenate(all_pair_X, axis=0),
            np.concatenate(all_pair_y, axis=0),
        )
        final_scorer.save(model_out)
        print(f"Final model saved to: {model_out}", flush=True)
    else:
        print("WARNING: no training pairs found — final model not saved.", flush=True)

    print(json.dumps({
        "promote": passed,
        "failed_thresholds": failed,
        "selacc10_ranksvm": metrics_ranksvm.get("selacc@10%"),
        "selacc10_baseline": metrics_baseline.get("selacc@10%"),
        "auroc_ranksvm": metrics_ranksvm.get("auroc"),
        "auroc_baseline": metrics_baseline.get("auroc"),
        "model_path": str(model_out),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
