#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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
from nad.core.selectors.science_dynamic_impl import (
    DEFAULT_SCIENCE_DYNAMIC_WEIGHTS,
    compute_science_dynamic_primary_scores_from_raw,
)
from nad.core.selectors.science_hybrid_impl import (
    ScienceHybridConfig,
    compute_science_hybrid_decision,
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
from scripts.run_science_hybrid_round3 import _tau_grid_from_baseline_gaps


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _rank_configs(
    *,
    feature_names: list[str] | None = None,
    losses: list[str] | None = None,
    c_values: list[float] | None = None,
    matrix_modes: list[str] | None = None,
    antisymmetrize_options: list[bool] | None = None,
    sigmoid_temps: list[float] | None = None,
) -> list[dict[str, Any]]:
    feature_cfgs = {
        "base": {"include_margin": False, "include_dominance": False},
        "margin": {"include_margin": True, "include_dominance": False},
        "dominance": {"include_margin": False, "include_dominance": True},
        "margin_dominance": {"include_margin": True, "include_dominance": True},
    }
    losses = losses or ["hinge", "squared_hinge"]
    c_values = c_values or [0.10, 0.30, 1.00]
    matrix_modes = matrix_modes or ["hard_sign", "sigmoid"]
    antisymmetrize_options = antisymmetrize_options or [False, True]
    sigmoid_temps = sigmoid_temps or [1.0]
    feature_names = feature_names or list(feature_cfgs.keys())

    configs: list[dict[str, Any]] = []
    for feature_name in feature_names:
        feat_cfg = feature_cfgs[feature_name]
        for loss in losses:
            for c_value in c_values:
                for antisym in antisymmetrize_options:
                    for matrix_mode in matrix_modes:
                        temps = [None] if matrix_mode == "hard_sign" else list(sigmoid_temps)
                        for temp in temps:
                            name = (
                                f"{feature_name}__{loss}__C{c_value:.2f}"
                                f"__{matrix_mode}"
                                f"{'' if temp is None else f'__t{temp:.2f}'}"
                                f"__{'antisym' if antisym else 'raw'}"
                            ).replace(".", "p")
                            configs.append({
                                "name": name,
                                "feature_name": feature_name,
                                "include_margin": bool(feat_cfg["include_margin"]),
                                "include_dominance": bool(feat_cfg["include_dominance"]),
                                "loss": loss,
                                "C": float(c_value),
                                "matrix_mode": matrix_mode,
                                "matrix_temperature": None if temp is None else float(temp),
                                "antisymmetrize": bool(antisym),
                                "fit_intercept": False,
                                "dual": "auto",
                                "max_iter": 100000,
                                "tol": 1e-4,
                            })
    return configs


def _hybrid_configs(
    *,
    hybrid_families: list[str] | None,
    tau_grid: list[float],
    alphas: list[float] | None,
    shortlist_ks: list[int] | None,
    override_ms: list[float] | None,
    backends: list[str] | None,
    temperatures: list[float] | None,
) -> list[ScienceHybridConfig]:
    families = hybrid_families or ["shortlist_blend", "margin_fallback", "hard_override"]
    alphas = alphas or [0.10, 0.25, 0.50, 0.75]
    shortlist_ks = shortlist_ks or [2, 3, 4, 5]
    override_ms = override_ms or [0.00, 0.02, 0.05]
    backends = backends or ["mean", "win_count", "copeland_margin", "softmax_mean"]
    temperatures = temperatures or [0.50, 0.75, 1.00]

    configs: list[ScienceHybridConfig] = []
    for backend in backends:
        temperature_grid = temperatures if backend == "softmax_mean" else [0.75]
        for temperature in temperature_grid:
            if "margin_fallback" in families:
                for tau in tau_grid:
                    for k in shortlist_ks:
                        configs.append(
                            ScienceHybridConfig(
                                family="margin_fallback",
                                backend=backend,
                                tau=float(tau),
                                k=int(k),
                                alpha=0.50,
                                m=0.0,
                                temperature=float(temperature),
                            )
                        )
            if "shortlist_blend" in families:
                for k in shortlist_ks:
                    for alpha in alphas:
                        configs.append(
                            ScienceHybridConfig(
                                family="shortlist_blend",
                                backend=backend,
                                tau=0.0,
                                k=int(k),
                                alpha=float(alpha),
                                m=0.0,
                                temperature=float(temperature),
                            )
                        )
            if "hard_override" in families:
                for tau in tau_grid:
                    for margin in override_ms:
                        configs.append(
                            ScienceHybridConfig(
                                family="hard_override",
                                backend=backend,
                                tau=float(tau),
                                k=3,
                                alpha=0.50,
                                m=float(margin),
                                temperature=float(temperature),
                            )
                        )
    return configs


def _margin_matrix_to_prob_matrix(
    margins: np.ndarray,
    *,
    matrix_mode: str,
    matrix_temperature: float | None,
    antisymmetrize: bool,
) -> np.ndarray:
    margin_arr = np.asarray(margins, dtype=np.float64)
    if bool(antisymmetrize):
        margin_arr = 0.5 * (margin_arr - margin_arr.T)
    n = int(margin_arr.shape[0])
    probs = np.full((n, n), 0.5, dtype=np.float64)
    mask = ~np.eye(n, dtype=bool)
    off_diag = margin_arr[mask]
    if matrix_mode == "hard_sign":
        off_prob = np.where(off_diag > 0.0, 1.0, np.where(off_diag < 0.0, 0.0, 0.5))
    elif matrix_mode == "sigmoid":
        temp = max(float(matrix_temperature or 1.0), 1e-6)
        clipped = np.clip(off_diag / temp, -40.0, 40.0)
        off_prob = 1.0 / (1.0 + np.exp(-clipped))
    else:
        raise ValueError(f"Unknown matrix mode: {matrix_mode}")
    probs[mask] = np.asarray(off_prob, dtype=np.float64)
    np.fill_diagonal(probs, 0.5)
    return probs


def _prepare_rank_features(
    all_problems: list[_ProblemData],
    rank_cfg: dict[str, Any],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    variant_X: list[np.ndarray] = []
    pair_X_by_problem: list[np.ndarray] = []
    pair_y_by_problem: list[np.ndarray] = []
    for prob in all_problems:
        X_prob = _variant_feature_matrix(prob, rank_cfg)
        variant_X.append(X_prob)
        X_pairs, y_pairs = build_pairwise_hinge_training_examples(X_prob, prob.labels)
        pair_X_by_problem.append(X_pairs)
        pair_y_by_problem.append(y_pairs)
    return variant_X, pair_X_by_problem, pair_y_by_problem


def _baseline_maps(all_problems: list[_ProblemData]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    baseline_scores_by_problem: dict[str, np.ndarray] = {}
    baseline_gate_scores_by_problem: dict[str, np.ndarray] = {}
    for prob in all_problems:
        baseline_scores, _ = compute_science_dynamic_primary_scores_from_raw(
            prob.sci_raw,
            weights=DEFAULT_SCIENCE_DYNAMIC_WEIGHTS,
        )
        baseline_scores_by_problem[str(prob.problem_id)] = np.asarray(baseline_scores, dtype=np.float64)
        baseline_gate_scores_by_problem[str(prob.problem_id)] = np.asarray(
            prob.sci_raw["recency_conf_mean"],
            dtype=np.float64,
        )
    return baseline_scores_by_problem, baseline_gate_scores_by_problem


def _variant_name(rank_cfg: dict[str, Any], hybrid_cfg: ScienceHybridConfig) -> str:
    hybrid_name = hybrid_cfg.family
    if hybrid_cfg.family == "shortlist_blend":
        hybrid_name = f"familyB__k{int(hybrid_cfg.k)}__a{hybrid_cfg.alpha:.2f}".replace(".", "p")
    elif hybrid_cfg.family == "margin_fallback":
        hybrid_name = (
            f"familyA__k{int(hybrid_cfg.k)}__tau{hybrid_cfg.tau:.5f}__{hybrid_cfg.backend}"
        ).replace(".", "p")
    elif hybrid_cfg.family == "hard_override":
        hybrid_name = (
            f"familyC__tau{hybrid_cfg.tau:.5f}__m{hybrid_cfg.m:.2f}__{hybrid_cfg.backend}"
        ).replace(".", "p")
    if hybrid_cfg.backend == "softmax_mean":
        hybrid_name = (
            f"{hybrid_name}__temp{hybrid_cfg.temperature:.2f}"
        ).replace(".", "p")
    return f"{rank_cfg['name']}__{hybrid_name}"


def _evaluate_rank_config(
    all_problems: list[_ProblemData],
    rank_cfg: dict[str, Any],
) -> list[np.ndarray]:
    variant_X, pair_X_by_problem, pair_y_by_problem = _prepare_rank_features(all_problems, rank_cfg)
    prob_matrices: list[np.ndarray] = []
    for held_out_idx, held_out in enumerate(all_problems):
        if held_out_idx % 20 == 0:
            print(
                f"[gpqa-ranksvm-hybrid:{rank_cfg['name']}] fold {held_out_idx + 1}/{len(all_problems)}",
                flush=True,
            )
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
            probs = np.full((len(held_out.run_ids), len(held_out.run_ids)), 0.5, dtype=np.float64)
        else:
            scorer = GPQARankSVMScorer(
                C=float(rank_cfg["C"]),
                loss=str(rank_cfg["loss"]),
                fit_intercept=bool(rank_cfg["fit_intercept"]),
                backend="utility",
                dual=rank_cfg.get("dual", "auto"),
                max_iter=int(rank_cfg.get("max_iter", 100000)),
                tol=float(rank_cfg.get("tol", 1e-4)),
            )
            scorer.fit(np.concatenate(train_X, axis=0), np.concatenate(train_y, axis=0))
            margins = scorer.pairwise_margin_matrix(variant_X[held_out_idx])
            probs = _margin_matrix_to_prob_matrix(
                margins,
                matrix_mode=str(rank_cfg["matrix_mode"]),
                matrix_temperature=rank_cfg.get("matrix_temperature"),
                antisymmetrize=bool(rank_cfg.get("antisymmetrize", False)),
            )
        prob_matrices.append(np.asarray(probs, dtype=np.float64))
    return prob_matrices


def _evaluate_hybrid_variants(
    all_problems: list[_ProblemData],
    rank_cfg: dict[str, Any],
    hybrid_cfgs: list[ScienceHybridConfig],
    prob_matrices: list[np.ndarray],
    baseline_scores_by_problem: dict[str, np.ndarray],
    baseline_gate_scores_by_problem: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    accs: dict[str, MetricAccumulator] = {}
    for hybrid_cfg in hybrid_cfgs:
        accs[_variant_name(rank_cfg, hybrid_cfg)] = MetricAccumulator(
            _variant_name(rank_cfg, hybrid_cfg),
            use_code_tiebreak=True,
        )

    for prob, prob_matrix in zip(all_problems, prob_matrices):
        baseline_scores = baseline_scores_by_problem[str(prob.problem_id)]
        baseline_gate_scores = baseline_gate_scores_by_problem[str(prob.problem_id)]
        for hybrid_cfg in hybrid_cfgs:
            name = _variant_name(rank_cfg, hybrid_cfg)
            decision = compute_science_hybrid_decision(
                baseline_scores,
                prob_matrix,
                prob.D,
                run_ids=prob.run_ids,
                baseline_gate_scores=baseline_gate_scores,
                config=hybrid_cfg,
            )
            accs[name].add_problem(
                prob.problem_id,
                prob.run_ids,
                decision.hybrid_scores,
                prob.labels,
                prob.D,
            )

    rows: list[dict[str, Any]] = []
    for hybrid_cfg in hybrid_cfgs:
        name = _variant_name(rank_cfg, hybrid_cfg)
        metrics = accs[name].finalize()
        passed, failed = _check_gate(metrics)
        rows.append({
            "name": name,
            "rank_config": rank_cfg,
            "hybrid_config": hybrid_cfg.as_dict(),
            "metrics": metrics,
            "aux": {
                "gate_passed": bool(passed),
                "failed_thresholds": failed,
            },
        })
    return rows


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
    summary_text = "\n".join([
        _summarize_table(rows),
        "",
        f"Best variant: {best['name']}",
        json.dumps(
            {
                "rank_config": best["rank_config"],
                "hybrid_config": best["hybrid_config"],
                "metrics": best["metrics"],
                "aux": best["aux"],
            },
            indent=2,
            ensure_ascii=False,
        ),
    ])
    payload = {
        "baseline_metrics": baseline_metrics,
        "best_variant": best,
        "variants": ranked,
        "n_variants": len(ranked),
        "n_problems": n_problems,
    }
    suffix = "" if final else ".partial"
    (out_dir / f"gpqa_ranksvm_hybrid_sweep{suffix}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / f"summary{suffix}.txt").write_text(summary_text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="GPQA RankSVM-backed science hybrid sweep")
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--distance-threads", type=int, default=_DEFAULT_DISTANCE_THREADS)
    ap.add_argument("--workers", type=int, default=_DEFAULT_WORKERS)
    ap.add_argument("--limit-rank", type=int, default=0)
    ap.add_argument("--limit-hybrid", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--rank-features", default="", help="Comma-separated feature variants")
    ap.add_argument("--rank-losses", default="", help="Comma-separated losses")
    ap.add_argument("--rank-c-values", default="", help="Comma-separated C values")
    ap.add_argument("--matrix-modes", default="", help="Comma-separated matrix modes: hard_sign,sigmoid")
    ap.add_argument("--antisymmetrize", default="", help="Comma-separated booleans: true,false")
    ap.add_argument("--sigmoid-temps", default="", help="Comma-separated sigmoid temperatures")
    ap.add_argument("--hybrid-families", default="", help="Comma-separated hybrid families")
    ap.add_argument("--hybrid-backends", default="", help="Comma-separated hybrid backends")
    ap.add_argument("--alphas", default="", help="Comma-separated shortlist_blend alphas")
    ap.add_argument("--shortlist-ks", default="", help="Comma-separated shortlist ks")
    ap.add_argument("--override-ms", default="", help="Comma-separated hard_override margins")
    ap.add_argument("--hybrid-temperatures", default="", help="Comma-separated softmax temperatures")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "result" / f"gpqa_ranksvm_hybrid_sweep_{_now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    precomputed_path = out_dir / "precomputed.pkl"
    if args.resume and precomputed_path.exists():
        import joblib

        precomputed = joblib.load(precomputed_path)
        all_problems = precomputed["all_problems"]
        baseline_metrics = precomputed["baseline_metrics"]
        baseline_scores_by_problem = precomputed["baseline_scores_by_problem"]
        baseline_gate_scores_by_problem = precomputed["baseline_gate_scores_by_problem"]
        tau_grid = precomputed["tau_grid"]
    else:
        all_problems = _extract_all_problems(
            Path(args.cache_root),
            distance_threads=args.distance_threads,
            workers=args.workers,
        )
        baseline_metrics = _baseline_rows(all_problems)
        baseline_scores_by_problem, baseline_gate_scores_by_problem = _baseline_maps(all_problems)
        tau_grid = _tau_grid_from_baseline_gaps(
            all_problems,
            baseline_scores_by_problem,
            baseline_gate_scores_by_problem,
        )
        import joblib

        joblib.dump(
            {
                "all_problems": all_problems,
                "baseline_metrics": baseline_metrics,
                "baseline_scores_by_problem": baseline_scores_by_problem,
                "baseline_gate_scores_by_problem": baseline_gate_scores_by_problem,
                "tau_grid": tau_grid,
            },
            precomputed_path,
        )

    rank_features = [x.strip() for x in args.rank_features.split(",") if x.strip()]
    rank_losses = [x.strip() for x in args.rank_losses.split(",") if x.strip()]
    rank_c_values = [float(x.strip()) for x in args.rank_c_values.split(",") if x.strip()]
    matrix_modes = [x.strip() for x in args.matrix_modes.split(",") if x.strip()]
    antisymmetrize_values = []
    for item in [x.strip().lower() for x in args.antisymmetrize.split(",") if x.strip()]:
        if item in ("true", "1", "yes", "y"):
            antisymmetrize_values.append(True)
        elif item in ("false", "0", "no", "n"):
            antisymmetrize_values.append(False)
        else:
            raise ValueError(f"Invalid antisymmetrize value: {item}")
    sigmoid_temps = [float(x.strip()) for x in args.sigmoid_temps.split(",") if x.strip()]
    hybrid_families = [x.strip() for x in args.hybrid_families.split(",") if x.strip()]
    hybrid_backends = [x.strip() for x in args.hybrid_backends.split(",") if x.strip()]
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    shortlist_ks = [int(x.strip()) for x in args.shortlist_ks.split(",") if x.strip()]
    override_ms = [float(x.strip()) for x in args.override_ms.split(",") if x.strip()]
    hybrid_temperatures = [float(x.strip()) for x in args.hybrid_temperatures.split(",") if x.strip()]

    rank_cfgs = _rank_configs(
        feature_names=rank_features or None,
        losses=rank_losses or None,
        c_values=rank_c_values or None,
        matrix_modes=matrix_modes or None,
        antisymmetrize_options=antisymmetrize_values or None,
        sigmoid_temps=sigmoid_temps or None,
    )
    if int(args.limit_rank) > 0:
        rank_cfgs = rank_cfgs[: int(args.limit_rank)]
    hybrid_cfgs = _hybrid_configs(
        hybrid_families=hybrid_families or None,
        tau_grid=tau_grid,
        alphas=alphas or None,
        shortlist_ks=shortlist_ks or None,
        override_ms=override_ms or None,
        backends=hybrid_backends or None,
        temperatures=hybrid_temperatures or None,
    )
    if int(args.limit_hybrid) > 0:
        hybrid_cfgs = hybrid_cfgs[: int(args.limit_hybrid)]

    partial_path = out_dir / "gpqa_ranksvm_hybrid_sweep.partial.json"
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

    for rank_cfg in rank_cfgs:
        prob_matrices = _evaluate_rank_config(all_problems, rank_cfg)
        hybrid_rows = _evaluate_hybrid_variants(
            all_problems,
            rank_cfg,
            hybrid_cfgs,
            prob_matrices,
            baseline_scores_by_problem,
            baseline_gate_scores_by_problem,
        )
        for row in hybrid_rows:
            if row["name"] in completed_names:
                continue
            completed_names.add(row["name"])
            variant_results.append(row)
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
