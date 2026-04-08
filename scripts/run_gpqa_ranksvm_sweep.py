#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.core.selectors.gpqa_ranksvm_impl import (
    GPQARankSVMScorer,
    build_pairwise_hinge_training_examples,
)
from scripts.run_code_baseline_v1_phase2 import MetricAccumulator
from scripts.run_gpqa_pairwise_round1 import (
    _DEFAULT_DISTANCE_THREADS,
    _DEFAULT_WORKERS,
    _ProblemData,
    _check_gate,
    _extract_all_problems,
    _summarize_table,
)
from scripts.run_gpqa_pairwise_round2 import _baseline_rows, _variant_feature_matrix

DEFAULT_OUT_DIR = REPO_ROOT / "result" / f"gpqa_ranksvm_sweep_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _variant_configs() -> list[dict[str, Any]]:
    feature_cfgs = (
        ("base", False, False),
        ("margin", True, False),
        ("dominance", False, True),
        ("margin_dominance", True, True),
    )
    model_cfgs = (
        {"loss": "hinge", "C": 0.03, "fit_intercept": False, "backend": "utility"},
        {"loss": "hinge", "C": 0.10, "fit_intercept": False, "backend": "utility"},
        {"loss": "hinge", "C": 0.30, "fit_intercept": False, "backend": "utility"},
        {"loss": "hinge", "C": 1.00, "fit_intercept": False, "backend": "utility"},
        {"loss": "hinge", "C": 3.00, "fit_intercept": False, "backend": "utility"},
        {"loss": "squared_hinge", "C": 0.03, "fit_intercept": False, "backend": "utility"},
        {"loss": "squared_hinge", "C": 0.10, "fit_intercept": False, "backend": "utility"},
        {"loss": "squared_hinge", "C": 0.30, "fit_intercept": False, "backend": "utility"},
        {"loss": "squared_hinge", "C": 1.00, "fit_intercept": False, "backend": "utility"},
        {"loss": "squared_hinge", "C": 3.00, "fit_intercept": False, "backend": "utility"},
        {"loss": "hinge", "C": 0.10, "fit_intercept": True, "backend": "mean_margin"},
        {"loss": "hinge", "C": 0.30, "fit_intercept": True, "backend": "mean_margin"},
        {"loss": "hinge", "C": 1.00, "fit_intercept": True, "backend": "mean_margin"},
        {"loss": "hinge", "C": 0.10, "fit_intercept": True, "backend": "win_count"},
        {"loss": "hinge", "C": 0.30, "fit_intercept": True, "backend": "win_count"},
        {"loss": "hinge", "C": 1.00, "fit_intercept": True, "backend": "win_count"},
        {"loss": "squared_hinge", "C": 0.10, "fit_intercept": True, "backend": "mean_margin"},
        {"loss": "squared_hinge", "C": 0.30, "fit_intercept": True, "backend": "mean_margin"},
        {"loss": "squared_hinge", "C": 1.00, "fit_intercept": True, "backend": "mean_margin"},
        {"loss": "squared_hinge", "C": 0.10, "fit_intercept": True, "backend": "win_count"},
        {"loss": "squared_hinge", "C": 0.30, "fit_intercept": True, "backend": "win_count"},
        {"loss": "squared_hinge", "C": 1.00, "fit_intercept": True, "backend": "win_count"},
    )

    variants: list[dict[str, Any]] = []
    for feature_name, include_margin, include_dominance in feature_cfgs:
        for model_cfg in model_cfgs:
            loss = str(model_cfg["loss"])
            backend = str(model_cfg["backend"])
            fit_intercept = bool(model_cfg["fit_intercept"])
            c_value = float(model_cfg["C"])
            variants.append({
                "name": (
                    f"{feature_name}__{loss}"
                    f"__C{c_value:.2f}"
                    f"__{'bias' if fit_intercept else 'nobias'}"
                    f"__{backend}"
                ).replace(".", "p"),
                "feature_name": feature_name,
                "include_margin": bool(include_margin),
                "include_dominance": bool(include_dominance),
                "loss": loss,
                "C": c_value,
                "fit_intercept": fit_intercept,
                "backend": backend,
                "dual": "auto",
                "max_iter": 100000,
                "tol": 1e-4,
            })
    return variants


def _prepare_variant_data(
    all_problems: list[_ProblemData],
    cfg: dict[str, Any],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    variant_X: list[np.ndarray] = []
    pair_X_by_problem: list[np.ndarray] = []
    pair_y_by_problem: list[np.ndarray] = []

    for prob in all_problems:
        X_prob = _variant_feature_matrix(prob, cfg)
        variant_X.append(X_prob)
        X_pairs, y_pairs = build_pairwise_hinge_training_examples(X_prob, prob.labels)
        pair_X_by_problem.append(X_pairs)
        pair_y_by_problem.append(y_pairs)

    return variant_X, pair_X_by_problem, pair_y_by_problem


def _evaluate_variant(
    all_problems: list[_ProblemData],
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    variant_X, pair_X_by_problem, pair_y_by_problem = _prepare_variant_data(all_problems, cfg)
    acc = MetricAccumulator(str(cfg["name"]), use_code_tiebreak=True)
    convergence_flags: list[bool] = []

    for held_out_idx, held_out in enumerate(all_problems):
        if held_out_idx % 20 == 0:
            print(f"[gpqa-ranksvm-sweep:{cfg['name']}] fold {held_out_idx + 1}/{len(all_problems)}", flush=True)

        train_X = [
            pair_X_by_problem[idx]
            for idx in range(len(all_problems))
            if idx != held_out_idx and pair_X_by_problem[idx].shape[0] > 0
        ]
        train_y = [
            pair_y_by_problem[idx]
            for idx in range(len(all_problems))
            if idx != held_out_idx and pair_y_by_problem[idx].shape[0] > 0
        ]

        if not train_X:
            scores = np.zeros(len(held_out.run_ids), dtype=np.float64)
            convergence_flags.append(True)
        else:
            scorer = GPQARankSVMScorer(
                C=float(cfg["C"]),
                loss=str(cfg["loss"]),
                fit_intercept=bool(cfg["fit_intercept"]),
                backend=str(cfg["backend"]),
                dual=cfg.get("dual", "auto"),
                max_iter=int(cfg.get("max_iter", 100000)),
                tol=float(cfg.get("tol", 1e-4)),
            )
            scorer.fit(
                np.concatenate(train_X, axis=0),
                np.concatenate(train_y, axis=0),
            )
            scores = scorer.score_group(variant_X[held_out_idx])
            convergence_flags.append(bool(getattr(scorer, "converged_", True)))

        acc.add_problem(
            held_out.problem_id,
            held_out.run_ids,
            scores,
            held_out.labels,
            held_out.D,
        )

    metrics = acc.finalize()
    passed, failed = _check_gate(metrics)
    aux = {
        "gate_passed": bool(passed),
        "failed_thresholds": failed,
        "converged_folds": int(sum(bool(x) for x in convergence_flags)),
        "total_folds": int(len(convergence_flags)),
    }
    return metrics, aux


def _rank_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    metrics = row["metrics"]
    return (
        1.0 if row["aux"]["gate_passed"] else 0.0,
        float(metrics.get("hit@1") or 0.0),
        float(metrics.get("selacc@10%") or 0.0),
        float(metrics.get("pairwise") or 0.0),
        float(metrics.get("auroc") or 0.0),
    )


def _write_outputs(
    out_dir: Path,
    baseline_metrics: dict[str, Any],
    variant_results: list[dict[str, Any]],
    *,
    n_problems: int,
    final: bool,
) -> None:
    if not variant_results:
        return

    ranked = sorted(variant_results, key=_rank_key, reverse=True)
    best = ranked[0]

    rows = [
        {"name": "science_baseline_v1", **{k: baseline_metrics["science_baseline_v1"][k] for k in ("auroc", "hit@1", "pairwise", "selacc@10%")}},
        {"name": "tournament-copeland", **{k: baseline_metrics["tournament-copeland"][k] for k in ("auroc", "hit@1", "pairwise", "selacc@10%")}},
    ]
    rows.extend(
        {
            "name": row["name"],
            **{k: row["metrics"][k] for k in ("auroc", "hit@1", "pairwise", "selacc@10%")},
        }
        for row in ranked[:10]
    )

    summary_lines = [
        _summarize_table(rows),
        "",
        f"Best variant: {best['name']}",
        json.dumps({
            "config": best["config"],
            "metrics": best["metrics"],
            "aux": best["aux"],
        }, indent=2, ensure_ascii=False),
    ]
    summary_text = "\n".join(summary_lines)

    payload = {
        "baseline_metrics": baseline_metrics,
        "best_variant": best,
        "variants": ranked,
        "n_variants": len(ranked),
        "n_problems": n_problems,
    }

    suffix = "" if final else ".partial"
    (out_dir / f"gpqa_ranksvm_sweep{suffix}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / f"summary{suffix}.txt").write_text(summary_text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="GPQA RankSVM sweep over loss/C/backend/feature variants")
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--distance-threads", type=int, default=_DEFAULT_DISTANCE_THREADS)
    ap.add_argument("--workers", type=int, default=_DEFAULT_WORKERS)
    ap.add_argument("--limit", type=int, default=0, help="Optional cap on number of variants to evaluate")
    ap.add_argument("--resume", action="store_true", help="Resume from partial outputs when available")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "result" / f"gpqa_ranksvm_sweep_{_now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_problems = _extract_all_problems(
        Path(args.cache_root),
        distance_threads=args.distance_threads,
        workers=args.workers,
    )
    baseline_metrics = _baseline_rows(all_problems)

    cfgs = _variant_configs()
    if int(args.limit) > 0:
        cfgs = cfgs[: int(args.limit)]

    partial_path = out_dir / "gpqa_ranksvm_sweep.partial.json"
    completed_names: set[str] = set()
    variant_results: list[dict[str, Any]] = []
    if args.resume and partial_path.exists():
        partial_payload = json.loads(partial_path.read_text(encoding="utf-8"))
        for row in partial_payload.get("variants", []):
            name = str(row["name"])
            if name in completed_names:
                continue
            completed_names.add(name)
            variant_results.append(row)

    for cfg in cfgs:
        if str(cfg["name"]) in completed_names:
            continue
        metrics, aux = _evaluate_variant(all_problems, cfg)
        variant_results.append({
            "name": str(cfg["name"]),
            "config": cfg,
            "metrics": metrics,
            "aux": aux,
        })
        _write_outputs(
            out_dir,
            baseline_metrics,
            variant_results,
            n_problems=len(all_problems),
            final=False,
        )

    _write_outputs(
        out_dir,
        baseline_metrics,
        variant_results,
        n_problems=len(all_problems),
        final=True,
    )
    print((out_dir / "summary.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
