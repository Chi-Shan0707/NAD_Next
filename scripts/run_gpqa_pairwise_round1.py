#!/usr/bin/env python3
"""
GPQA Group Pairwise Scorer — Round 1 Evaluation

Runs LOPO (leave-one-problem-out) evaluation of the GPQAPairwiseScorer
against science_baseline_v1 and tournament-copeland on the GPQA cache.

After LOPO evaluation, trains a final model on all GPQA problems and saves
it to models/ml_selectors/gpqa_pairwise_round1.pkl.

Usage:
    python scripts/run_gpqa_pairwise_round1.py \\
        --cache-root MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/gpqa/<cache> \\
        --out result/gpqa_pairwise_round1_YYYYMMDD.json

The script prints a summary table and a promote/no-promote decision.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.core.distance.engine import DistanceEngine, DistanceSpec
from nad.core.selectors.base import SelectorContext
from nad.core.selectors.gpqa_pairwise_impl import (
    GPQAPairwiseScorer,
    build_gpqa_pairwise_features,
    build_pairwise_training_pairs,
    extract_gpqa_pairwise_raw,
)
from nad.core.selectors.science_dynamic_impl import (
    DEFAULT_SCIENCE_DYNAMIC_WEIGHTS,
    compute_science_dynamic_primary_scores_from_raw,
)
from nad.core.views.reader import CacheReader
from nad.ops.accuracy import _load_ground_truth
from nad.ops.bestofn_extreme8 import build_problem_groups
from scripts.run_code_baseline_v1_phase2 import (
    DEFAULT_VIEW,
    MetricAccumulator,
    _compute_copeland_scores,
    _fmt_pct,
    _problem_sort_key,
)

# ── promote/no-promote thresholds (vs science_baseline_v1 on GPQA) ───────────
BASELINE_SELACC10 = 0.6435
BASELINE_AUROC    = 0.5386
BASELINE_HIT1     = 0.6616
BASELINE_PAIRWISE = 0.5871

GATE_SELACC10  = BASELINE_SELACC10            # must exceed
GATE_AUROC     = BASELINE_AUROC              # must exceed
GATE_HIT1      = BASELINE_HIT1 - 0.01       # guardrail: ≥ 65.16%
GATE_PAIRWISE  = BASELINE_PAIRWISE - 0.005  # guardrail: ≥ 58.21%

_DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "ml_selectors" / "gpqa_pairwise_round1.pkl"
_DEFAULT_DISTANCE_THREADS = 4
_DEFAULT_WORKERS = 4


@dataclass
class _ProblemData:
    """Precomputed data for one problem group."""
    problem_id: str
    run_ids: list[int]
    labels: np.ndarray          # (N,) int32
    X: np.ndarray               # (N, 6) float64 pairwise features
    D: np.ndarray               # (N, N) float64 distance matrix
    sci_raw: dict[str, np.ndarray]  # full science raw for baseline eval


# ── data extraction ───────────────────────────────────────────────────────────

# ── per-process state for multiprocessing workers ─────────────────────────────
# Each worker process initialises these once (avoids per-task CacheReader cost).
_worker_reader: CacheReader | None = None
_worker_engine: DistanceEngine | None = None
_worker_correctness: dict[int, bool] = {}


def _worker_init(cache_root_str: str, distance_threads: int, correctness: dict[int, bool]) -> None:
    global _worker_reader, _worker_engine, _worker_correctness
    _worker_reader = CacheReader(cache_root_str)
    _worker_engine = DistanceEngine(DistanceSpec("ja", num_threads=distance_threads))
    _worker_correctness = correctness


def _extract_one_problem_worker(args: tuple[str, list[int]]) -> _ProblemData:
    """Worker function — uses process-local reader/engine initialised by _worker_init."""
    problem_id, run_ids = args
    reader = _worker_reader
    engine = _worker_engine
    correctness = _worker_correctness

    labels = np.asarray(
        [int(bool(correctness.get(rid, False))) for rid in run_ids],
        dtype=np.int32,
    )
    run_views = [reader.get_run_view(rid, DEFAULT_VIEW) for rid in run_ids]
    D = engine.dense_matrix(run_views)

    ctx = SelectorContext(
        cache=reader,
        problem_id=str(problem_id),
        run_ids=run_ids,
        views=run_views,
        pos_window=None,
    )
    raw = extract_gpqa_pairwise_raw(ctx)
    X = build_gpqa_pairwise_features(raw)

    sci_raw: dict[str, np.ndarray] = {
        "prefix_conf_mean":  raw["prefix_conf_mean"],
        "recency_conf_mean": raw["recency_conf_mean"],
        "late_worst_window": raw["late_worst_window"],
        "late_recovery":     raw["late_recovery"],
    }
    return _ProblemData(
        problem_id=str(problem_id),
        run_ids=run_ids,
        labels=labels,
        X=X,
        D=D,
        sci_raw=sci_raw,
    )


def _extract_all_problems(
    cache_root: Path,
    *,
    distance_threads: int,
    workers: int,
    max_problems: int = 0,
) -> list[_ProblemData]:
    """Parallel extraction using multiprocessing.Pool (bypasses the GIL).

    Each worker process initialises ONE CacheReader + ONE DistanceEngine and
    handles a slice of problems serially.  No per-task initialisation overhead.

    Process budget: workers processes × distance_threads OMP threads each.
    Keep workers × distance_threads ≤ 16.
    """
    meta = json.loads((cache_root / "meta.json").read_text(encoding="utf-8"))
    groups = build_problem_groups(meta)
    correctness = _load_ground_truth(cache_root)

    sorted_problems = sorted(groups.items(), key=lambda kv: _problem_sort_key(kv[0]))
    if int(max_problems) > 0:
        sorted_problems = sorted_problems[: int(max_problems)]
    n_total = len(sorted_problems)
    tasks = [(str(pid), list(map(int, rids))) for pid, rids in sorted_problems]

    print(
        f"[extract] {n_total} problems — {workers} processes × {distance_threads} "
        f"dist-threads = {workers * distance_threads} total threads",
        flush=True,
    )

    init_args = (str(cache_root), distance_threads, correctness)
    chunksize = max(1, n_total // (workers * 4))  # keep workers busy without large chunks

    results_by_pid: dict[str, _ProblemData] = {}
    completed = 0

    with Pool(
        processes=workers,
        initializer=_worker_init,
        initargs=init_args,
    ) as pool:
        for data in pool.imap_unordered(
            _extract_one_problem_worker, tasks, chunksize=chunksize
        ):
            results_by_pid[data.problem_id] = data
            completed += 1
            if completed % 20 == 0 or completed == n_total:
                print(f"[extract] {completed}/{n_total} done", flush=True)

    # Restore original sort order
    return [results_by_pid[str(pid)] for pid, _ in sorted_problems]


# ── LOPO evaluation ───────────────────────────────────────────────────────────

def _run_lopo(
    all_problems: list[_ProblemData],
    acc: MetricAccumulator,
) -> None:
    """Leave-one-problem-out training + evaluation of GPQAPairwiseScorer."""
    n_total = len(all_problems)

    for k, held_out in enumerate(all_problems):
        if k % 20 == 0:
            print(f"[lopo] fold {k + 1}/{n_total}", flush=True)

        # Collect training pairs from all problems except held-out
        pair_X_list: list[np.ndarray] = []
        pair_y_list: list[np.ndarray] = []
        for i, prob in enumerate(all_problems):
            if i == k:
                continue
            Xp, yp = build_pairwise_training_pairs(prob.X, prob.labels)
            if Xp.shape[0] > 0:
                pair_X_list.append(Xp)
                pair_y_list.append(yp)

        if not pair_X_list:
            # No training pairs available — assign uniform scores
            n = len(held_out.run_ids)
            scores = np.zeros(n, dtype=np.float64)
            acc.add_problem(
                held_out.problem_id, held_out.run_ids,
                scores, held_out.labels, held_out.D,
            )
            continue

        X_train = np.concatenate(pair_X_list, axis=0)
        y_train = np.concatenate(pair_y_list, axis=0)

        scorer = GPQAPairwiseScorer()
        scorer.fit(X_train, y_train)

        scores = scorer.score_group(held_out.X)
        acc.add_problem(
            held_out.problem_id, held_out.run_ids,
            scores, held_out.labels, held_out.D,
        )


# ── output formatting ─────────────────────────────────────────────────────────

def _summarize_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Selector              | AUROC   | Hit@1   | Pairwise | SelAcc@10 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {name:<21} | {auroc} | {hit1} | {pairwise} | {selacc} |".format(
                name=row["name"],
                auroc=_fmt_pct(row.get("auroc")),
                hit1=_fmt_pct(row.get("hit@1")),
                pairwise=_fmt_pct(row.get("pairwise")),
                selacc=_fmt_pct(row.get("selacc@10%")),
            )
        )
    return "\n".join(lines)


def _check_gate(metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    """Return (passed, list_of_failed_metrics) vs promote thresholds."""
    failed: list[str] = []

    selacc = metrics.get("selacc@10%") or 0.0
    auroc  = metrics.get("auroc")      or 0.0
    hit1   = metrics.get("hit@1")      or 0.0
    pw     = metrics.get("pairwise")   or 0.0

    if selacc <= GATE_SELACC10:
        failed.append(f"SelAcc@10 {_fmt_pct(selacc)} ≤ threshold {_fmt_pct(GATE_SELACC10)}")
    if auroc <= GATE_AUROC:
        failed.append(f"AUROC {_fmt_pct(auroc)} ≤ threshold {_fmt_pct(GATE_AUROC)}")
    if hit1 < GATE_HIT1:
        failed.append(f"Hit@1 {_fmt_pct(hit1)} < guardrail {_fmt_pct(GATE_HIT1)}")
    if pw < GATE_PAIRWISE:
        failed.append(f"Pairwise {_fmt_pct(pw)} < guardrail {_fmt_pct(GATE_PAIRWISE)}")

    return (len(failed) == 0, failed)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="GPQA group pairwise model — round-1 LOPO evaluation"
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

    # ── step 1: extract all features + labels in one pass ────────────────────
    print("[run_gpqa_pairwise_round1] Extracting features for all problems …", flush=True)
    all_problems = _extract_all_problems(
        cache_root,
        distance_threads=args.distance_threads,
        workers=args.workers,
    )
    print(f"[run_gpqa_pairwise_round1] {len(all_problems)} problems extracted.", flush=True)

    # ── step 2: baseline accumulators (one pass, no LOPO needed) ─────────────
    acc_baseline = MetricAccumulator("science_baseline_v1",  use_code_tiebreak=True)
    acc_copeland = MetricAccumulator("tournament-copeland",   use_code_tiebreak=True)

    for prob in all_problems:
        baseline_scores, _ = compute_science_dynamic_primary_scores_from_raw(
            prob.sci_raw,
            weights=DEFAULT_SCIENCE_DYNAMIC_WEIGHTS,
        )
        copeland_scores = _compute_copeland_scores(prob.D)

        acc_baseline.add_problem(
            prob.problem_id, prob.run_ids, baseline_scores, prob.labels, prob.D
        )
        acc_copeland.add_problem(
            prob.problem_id, prob.run_ids, copeland_scores, prob.labels, prob.D
        )

    # ── step 3: LOPO evaluation of gpqa_pairwise_round1 ──────────────────────
    print("[run_gpqa_pairwise_round1] Running LOPO evaluation …", flush=True)
    acc_pairwise = MetricAccumulator("gpqa_pairwise_round1", use_code_tiebreak=True)
    _run_lopo(all_problems, acc_pairwise)

    # ── step 4: finalise metrics ──────────────────────────────────────────────
    metrics_baseline = acc_baseline.finalize()
    metrics_copeland = acc_copeland.finalize()
    metrics_pairwise = acc_pairwise.finalize()

    all_metrics = {
        "science_baseline_v1":  metrics_baseline,
        "tournament-copeland":  metrics_copeland,
        "gpqa_pairwise_round1": metrics_pairwise,
    }

    # ── step 5: summary table ─────────────────────────────────────────────────
    rows = [
        {"name": name, **{k: m[k] for k in ("auroc", "hit@1", "pairwise", "selacc@10%")}}
        for name, m in all_metrics.items()
    ]
    print()
    print(_summarize_table(rows))
    print()

    # ── step 6: promote/no-promote gate ──────────────────────────────────────
    passed, failed = _check_gate(metrics_pairwise)
    if passed:
        print("PROMOTE: gpqa_pairwise_round1 passes all thresholds → promote to gpqa_pairwise_v1")
    else:
        print("NO-PROMOTE: gpqa_pairwise_round1 failed the following thresholds:")
        for msg in failed:
            print(f"  - {msg}")
    print()

    # ── step 7: write JSON results ────────────────────────────────────────────
    payload: dict[str, Any] = {
        "metrics": all_metrics,
        "gate": {
            "passed": passed,
            "failed_thresholds": failed,
            "thresholds": {
                "selacc@10%": GATE_SELACC10,
                "auroc":      GATE_AUROC,
                "hit@1":      GATE_HIT1,
                "pairwise":   GATE_PAIRWISE,
            },
            "baseline": {
                "selacc@10%": BASELINE_SELACC10,
                "auroc":      BASELINE_AUROC,
                "hit@1":      BASELINE_HIT1,
                "pairwise":   BASELINE_PAIRWISE,
            },
        },
        "n_problems": len(all_problems),
        "model_path": str(model_out),
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Results written to: {out_path}", flush=True)

    # ── step 8: train final model on ALL problems ─────────────────────────────
    print("[run_gpqa_pairwise_round1] Training final model on all problems …", flush=True)
    all_pair_X: list[np.ndarray] = []
    all_pair_y: list[np.ndarray] = []
    for prob in all_problems:
        Xp, yp = build_pairwise_training_pairs(prob.X, prob.labels)
        if Xp.shape[0] > 0:
            all_pair_X.append(Xp)
            all_pair_y.append(yp)

    if all_pair_X:
        X_final = np.concatenate(all_pair_X, axis=0)
        y_final = np.concatenate(all_pair_y, axis=0)
        final_scorer = GPQAPairwiseScorer()
        final_scorer.fit(X_final, y_final)
        final_scorer.save(model_out)
        print(f"Final model saved to: {model_out}", flush=True)
    else:
        print("WARNING: no training pairs found — final model not saved.", flush=True)

    print(json.dumps({
        "promote": passed,
        "failed_thresholds": failed,
        "selacc10_pairwise": metrics_pairwise.get("selacc@10%"),
        "selacc10_baseline": metrics_baseline.get("selacc@10%"),
        "auroc_pairwise": metrics_pairwise.get("auroc"),
        "auroc_baseline": metrics_baseline.get("auroc"),
        "model_path": str(model_out),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
