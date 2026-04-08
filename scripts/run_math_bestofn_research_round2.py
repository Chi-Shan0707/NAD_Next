#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.core.selectors.deepsets_core import DeepSetsConfig
from nad.core.selectors.lambda_svm_core import build_weighted_pairwise_training_examples
from nad.core.selectors.math_deepsets_impl import MathDeepSetsScorer
from nad.core.selectors.math_lambdasvm_impl import MathLambdaSVMScorer
from nad.core.selectors.math_set_transformer_lite_impl import MathSetTransformerLiteScorer
from nad.core.selectors.math_svm_impl import select_math_feature_family
from nad.core.selectors.set_transformer_lite_core import SetTransformerLiteConfig
from scripts.run_bestofn_score_recovery_20260408 import (
    DEFAULT_SCIENCE_JSON_GLOB,
    DS_MATH_KEYS,
    _display_path,
    _load_current_system_bundle,
)
from scripts.run_code_baseline_v1_phase2 import MetricAccumulator
from scripts.run_math_deepsets_round1 import _math_gate
from scripts.run_math_svm_sweep import _MathProblemData, _baseline_metrics, _extract_all_problems
from scripts.run_science_hybrid_round3 import (
    CODE_V2_EXHAUSTIVE_JSON,
    ProblemScoreRecord,
    _combine_cache_metric_proxy,
    _evaluate_problem_records,
    _system_delta,
)

MODEL_DIR = REPO_ROOT / "models" / "ml_selectors"
DEFAULT_OUT_DIR = REPO_ROOT / "result" / "math_bestofn_research_round2"
DEFAULT_LAMBDASVM_MODEL_OUT = MODEL_DIR / "math_lambdasvm_round1.pkl"
DEFAULT_DEEPSETS_MODEL_OUT = MODEL_DIR / "math_deepsets_round2.pkl"
DEFAULT_SETTF_MODEL_OUT = MODEL_DIR / "math_set_transformer_lite_round1.pkl"


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def _latest_current_science_json() -> Path:
    matches = sorted(glob.glob(str(REPO_ROOT / DEFAULT_SCIENCE_JSON_GLOB)))
    if not matches:
        raise FileNotFoundError("No science_hybrid_round3.json payload found under result/")
    return Path(matches[-1])


def _reference_scores(X: np.ndarray) -> np.ndarray:
    return np.asarray(np.mean(np.asarray(X, dtype=np.float64), axis=1), dtype=np.float64)


def _candidate_cache_metrics(records: list[ProblemScoreRecord]) -> dict[str, dict[str, Any]]:
    out: dict[str, list[ProblemScoreRecord]] = {}
    for rec in records:
        out.setdefault(str(rec.cache_key), []).append(rec)
    return {cache_key: _evaluate_problem_records(recs) for cache_key, recs in sorted(out.items())}


def _build_proxy_bundle(
    *,
    current_bundle: dict[str, Any],
    current_cache_metrics: dict[str, dict[str, Any]],
    candidate_cache_metrics: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    full_cache_metrics = {cache_key: dict(metrics) for cache_key, metrics in current_cache_metrics.items()}
    for cache_key, metrics in candidate_cache_metrics.items():
        full_cache_metrics[cache_key] = dict(metrics)
    full_bundle = _combine_cache_metric_proxy(full_cache_metrics)
    system_delta = _system_delta(full_bundle, current_bundle)

    math_only_cache_metrics = {
        cache_key: dict(candidate_cache_metrics.get(cache_key, current_cache_metrics[cache_key]))
        for cache_key in DS_MATH_KEYS
    }
    math_only_bundle = _combine_cache_metric_proxy(math_only_cache_metrics)
    return full_bundle, math_only_bundle, system_delta


def _row_rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        bool(row["math_gate_passed"]),
        float(row["system_delta"]["sample_weighted"]["hit@1"]),
        float(row["system_delta"]["sample_weighted"]["selacc@10%"]),
        float(row["metrics"].get("hit@1") or 0.0),
        float(row["metrics"].get("selacc@10%") or 0.0),
        float(row["metrics"].get("pairwise") or 0.0),
        str(row["name"]),
    )


def _evaluate_lambdasvm_candidate(
    problems: list[_MathProblemData],
    *,
    feature_family: str,
    loss: str,
    c_value: float,
    pair_weight_mode: str,
    current_bundle: dict[str, Any],
    current_cache_metrics: dict[str, dict[str, Any]],
    knn_metrics: dict[str, Any],
) -> dict[str, Any]:
    variant_name = (
        f"math_lambdasvm_round1__{feature_family}__{loss}__C{c_value:.2f}__{pair_weight_mode}"
    ).replace(".", "p")
    X_by_problem = [select_math_feature_family(prob.X_all, feature_family) for prob in problems]
    acc = MetricAccumulator(variant_name, use_code_tiebreak=False)
    records: list[ProblemScoreRecord] = []
    convergence_flags: list[bool] = []

    for held_out_idx, held_out in enumerate(problems):
        if held_out_idx % 20 == 0:
            print(f"[math-research:lambdasvm:{variant_name}] fold {held_out_idx + 1}/{len(problems)}", flush=True)
        pair_X_list: list[np.ndarray] = []
        pair_y_list: list[np.ndarray] = []
        pair_w_list: list[np.ndarray] = []
        for idx, prob in enumerate(problems):
            if idx == held_out_idx:
                continue
            X_pairs, y_pairs, w_pairs = build_weighted_pairwise_training_examples(
                X_by_problem[idx],
                prob.labels,
                reference_scores=_reference_scores(X_by_problem[idx]),
                pair_weight_mode=pair_weight_mode,
            )
            if X_pairs.shape[0] <= 0:
                continue
            pair_X_list.append(X_pairs)
            pair_y_list.append(y_pairs)
            pair_w_list.append(w_pairs)

        if not pair_X_list:
            scores = np.zeros(len(held_out.run_ids), dtype=np.float64)
            convergence_flags.append(True)
        else:
            scorer = MathLambdaSVMScorer(
                feature_family=feature_family,
                C=float(c_value),
                loss=str(loss),
                fit_intercept=False,
                backend="utility",
                dual="auto",
                max_iter=100000,
                tol=1e-4,
            )
            scorer.fit(
                np.concatenate(pair_X_list, axis=0),
                np.concatenate(pair_y_list, axis=0),
                sample_weight=np.concatenate(pair_w_list, axis=0),
            )
            scores = scorer.score_group(X_by_problem[held_out_idx])
            convergence_flags.append(bool(getattr(scorer, "converged_", True)))

        acc.add_problem(held_out.problem_id, held_out.run_ids, scores, held_out.labels, held_out.D)
        records.append(
            ProblemScoreRecord(
                cache_key=f"DS-R1/{held_out.dataset}",
                problem_id=str(held_out.problem_id),
                sample_ids=list(map(int, held_out.run_ids)),
                labels=np.asarray(held_out.labels, dtype=np.int32),
                scores=np.asarray(scores, dtype=np.float64),
            )
        )

    metrics = acc.finalize()
    math_gate_passed, math_gate_failed = _math_gate(metrics, knn_metrics)
    candidate_cache_metrics = _candidate_cache_metrics(records)
    full_bundle, math_only_bundle, system_delta = _build_proxy_bundle(
        current_bundle=current_bundle,
        current_cache_metrics=current_cache_metrics,
        candidate_cache_metrics=candidate_cache_metrics,
    )
    return {
        "name": variant_name,
        "family": "lambdasvm",
        "config": {
            "feature_family": feature_family,
            "loss": loss,
            "C": float(c_value),
            "pair_weight_mode": pair_weight_mode,
            "backend": "utility",
        },
        "metrics": metrics,
        "math_gate_passed": bool(math_gate_passed),
        "math_gate_failed": math_gate_failed,
        "candidate_cache_metrics": candidate_cache_metrics,
        "full_system_proxy": full_bundle,
        "math_only_proxy": math_only_bundle,
        "system_delta": system_delta,
        "converged_folds": int(sum(bool(x) for x in convergence_flags)),
        "total_folds": int(len(convergence_flags)),
    }


def _evaluate_deepsets_candidate(
    problems: list[_MathProblemData],
    *,
    config: DeepSetsConfig,
    torch_threads: int,
    current_bundle: dict[str, Any],
    current_cache_metrics: dict[str, dict[str, Any]],
    knn_metrics: dict[str, Any],
) -> dict[str, Any]:
    variant_name = (
        f"math_deepsets_round2_{config.pooling}"
        f"__h{int(config.hidden_dim)}__e{int(config.embed_dim)}"
        f"__pairaux{float(config.pairwise_aux_weight):.2f}"
    ).replace(".", "p")
    acc = MetricAccumulator(variant_name, use_code_tiebreak=False)
    X_all = np.stack([np.asarray(prob.X_all, dtype=np.float32) for prob in problems], axis=0)
    y_all = np.stack([np.asarray(prob.labels, dtype=np.float32) for prob in problems], axis=0)
    records: list[ProblemScoreRecord] = []
    fold_losses: list[float] = []

    for held_out_idx, held_out in enumerate(problems):
        if held_out_idx % 20 == 0:
            print(f"[math-research:deepsets:{variant_name}] fold {held_out_idx + 1}/{len(problems)}", flush=True)
        train_mask = [idx for idx in range(len(problems)) if idx != held_out_idx]
        scorer = MathDeepSetsScorer(config=config)
        scorer.fit_problem_batches(X_all[train_mask], y_all[train_mask], torch_threads=torch_threads)
        fold_losses.append(float(scorer.training_summary.get("train_loss", 0.0)))
        scores = scorer.score_group(X_all[held_out_idx])
        acc.add_problem(held_out.problem_id, held_out.run_ids, scores, held_out.labels, held_out.D)
        records.append(
            ProblemScoreRecord(
                cache_key=f"DS-R1/{held_out.dataset}",
                problem_id=str(held_out.problem_id),
                sample_ids=list(map(int, held_out.run_ids)),
                labels=np.asarray(held_out.labels, dtype=np.int32),
                scores=np.asarray(scores, dtype=np.float64),
            )
        )

    metrics = acc.finalize()
    metrics["mean_train_loss"] = float(np.mean(fold_losses)) if fold_losses else None
    math_gate_passed, math_gate_failed = _math_gate(metrics, knn_metrics)
    candidate_cache_metrics = _candidate_cache_metrics(records)
    full_bundle, math_only_bundle, system_delta = _build_proxy_bundle(
        current_bundle=current_bundle,
        current_cache_metrics=current_cache_metrics,
        candidate_cache_metrics=candidate_cache_metrics,
    )
    return {
        "name": variant_name,
        "family": "deepsets",
        "config": config.as_dict(),
        "metrics": metrics,
        "math_gate_passed": bool(math_gate_passed),
        "math_gate_failed": math_gate_failed,
        "candidate_cache_metrics": candidate_cache_metrics,
        "full_system_proxy": full_bundle,
        "math_only_proxy": math_only_bundle,
        "system_delta": system_delta,
    }


def _evaluate_set_transformer_candidate(
    problems: list[_MathProblemData],
    *,
    config: SetTransformerLiteConfig,
    torch_threads: int,
    current_bundle: dict[str, Any],
    current_cache_metrics: dict[str, dict[str, Any]],
    knn_metrics: dict[str, Any],
) -> dict[str, Any]:
    variant_name = (
        f"math_set_transformer_lite_round1_{config.pooling}"
        f"__d{int(config.model_dim)}__h{int(config.num_heads)}"
        f"__pairaux{float(config.pairwise_aux_weight):.2f}"
    ).replace(".", "p")
    acc = MetricAccumulator(variant_name, use_code_tiebreak=False)
    X_all = np.stack([np.asarray(prob.X_all, dtype=np.float32) for prob in problems], axis=0)
    y_all = np.stack([np.asarray(prob.labels, dtype=np.float32) for prob in problems], axis=0)
    records: list[ProblemScoreRecord] = []
    fold_losses: list[float] = []

    for held_out_idx, held_out in enumerate(problems):
        if held_out_idx % 20 == 0:
            print(f"[math-research:settf:{variant_name}] fold {held_out_idx + 1}/{len(problems)}", flush=True)
        train_mask = [idx for idx in range(len(problems)) if idx != held_out_idx]
        scorer = MathSetTransformerLiteScorer(config=config)
        scorer.fit_problem_batches(X_all[train_mask], y_all[train_mask], torch_threads=torch_threads)
        fold_losses.append(float(scorer.training_summary.get("train_loss", 0.0)))
        scores = scorer.score_group(X_all[held_out_idx])
        acc.add_problem(held_out.problem_id, held_out.run_ids, scores, held_out.labels, held_out.D)
        records.append(
            ProblemScoreRecord(
                cache_key=f"DS-R1/{held_out.dataset}",
                problem_id=str(held_out.problem_id),
                sample_ids=list(map(int, held_out.run_ids)),
                labels=np.asarray(held_out.labels, dtype=np.int32),
                scores=np.asarray(scores, dtype=np.float64),
            )
        )

    metrics = acc.finalize()
    metrics["mean_train_loss"] = float(np.mean(fold_losses)) if fold_losses else None
    math_gate_passed, math_gate_failed = _math_gate(metrics, knn_metrics)
    candidate_cache_metrics = _candidate_cache_metrics(records)
    full_bundle, math_only_bundle, system_delta = _build_proxy_bundle(
        current_bundle=current_bundle,
        current_cache_metrics=current_cache_metrics,
        candidate_cache_metrics=candidate_cache_metrics,
    )
    return {
        "name": variant_name,
        "family": "set_transformer_lite",
        "config": config.as_dict(),
        "metrics": metrics,
        "math_gate_passed": bool(math_gate_passed),
        "math_gate_failed": math_gate_failed,
        "candidate_cache_metrics": candidate_cache_metrics,
        "full_system_proxy": full_bundle,
        "math_only_proxy": math_only_bundle,
        "system_delta": system_delta,
    }


def _train_and_save_family_best(
    family_best: dict[str, Any],
    problems: list[_MathProblemData],
    *,
    torch_threads: int,
) -> str:
    X_groups = np.stack([np.asarray(prob.X_all, dtype=np.float32) for prob in problems], axis=0)
    y_groups = np.stack([np.asarray(prob.labels, dtype=np.float32) for prob in problems], axis=0)
    family = str(family_best["family"])

    if family == "lambdasvm":
        cfg = dict(family_best["config"])
        feature_family = str(cfg["feature_family"])
        X_by_problem = [select_math_feature_family(prob.X_all, feature_family) for prob in problems]
        pair_X_list: list[np.ndarray] = []
        pair_y_list: list[np.ndarray] = []
        pair_w_list: list[np.ndarray] = []
        for prob, X_prob in zip(problems, X_by_problem):
            X_pairs, y_pairs, w_pairs = build_weighted_pairwise_training_examples(
                X_prob,
                prob.labels,
                reference_scores=_reference_scores(X_prob),
                pair_weight_mode=str(cfg["pair_weight_mode"]),
            )
            if X_pairs.shape[0] <= 0:
                continue
            pair_X_list.append(X_pairs)
            pair_y_list.append(y_pairs)
            pair_w_list.append(w_pairs)
        if not pair_X_list:
            raise RuntimeError("No math LambdaSVM training pairs found for final model.")
        scorer = MathLambdaSVMScorer(
            feature_family=feature_family,
            C=float(cfg["C"]),
            loss=str(cfg["loss"]),
            fit_intercept=False,
            backend="utility",
        )
        scorer.fit(
            np.concatenate(pair_X_list, axis=0),
            np.concatenate(pair_y_list, axis=0),
            sample_weight=np.concatenate(pair_w_list, axis=0),
        )
        scorer.save(DEFAULT_LAMBDASVM_MODEL_OUT)
        return _display_path(DEFAULT_LAMBDASVM_MODEL_OUT)

    if family == "deepsets":
        scorer = MathDeepSetsScorer(config=DeepSetsConfig(**family_best["config"]).validate())
        scorer.fit_problem_batches(X_groups, y_groups, torch_threads=torch_threads)
        scorer.save(DEFAULT_DEEPSETS_MODEL_OUT)
        return _display_path(DEFAULT_DEEPSETS_MODEL_OUT)

    if family == "set_transformer_lite":
        scorer = MathSetTransformerLiteScorer(config=SetTransformerLiteConfig(**family_best["config"]).validate())
        scorer.fit_problem_batches(X_groups, y_groups, torch_threads=torch_threads)
        scorer.save(DEFAULT_SETTF_MODEL_OUT)
        return _display_path(DEFAULT_SETTF_MODEL_OUT)

    raise ValueError(f"Unsupported family: {family}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Continue Best-of-N math research with LambdaSVM / DeepSets / SetTransformerLite")
    ap.add_argument("--profile", default="main", choices=("train", "main", "test"))
    ap.add_argument("--ds-cache-root", default="MUI_HUB/cache")
    ap.add_argument("--science-json", default="")
    ap.add_argument("--code-v2-json", default=str(CODE_V2_EXHAUSTIVE_JSON))
    ap.add_argument("--distance-threads", type=int, default=8)
    ap.add_argument("--torch-threads", type=int, default=8)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--max-problems", type=int, default=0)
    ap.add_argument("--skip-save-model", action="store_true")
    args = ap.parse_args()

    ds_cache_root = Path(args.ds_cache_root)
    science_json = Path(args.science_json) if args.science_json else _latest_current_science_json()
    code_v2_json = Path(args.code_v2_json)
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_DIR / _now_tag()
    out_dir.mkdir(parents=True, exist_ok=True)

    current_bundle, current_cache_metrics, _ = _load_current_system_bundle(
        ds_cache_root=ds_cache_root,
        science_json=science_json,
        code_v2_json=code_v2_json,
    )

    print("[math-research] extracting math problems …", flush=True)
    problems = _extract_all_problems(
        str(args.profile),
        distance_threads=int(args.distance_threads),
        max_problems=int(args.max_problems),
    )
    baseline_metrics = _baseline_metrics(problems)
    knn_metrics = dict(baseline_metrics["knn-medoid"]["metrics"])
    torch_available = _torch_available()

    lambdasvm_rows: list[dict[str, Any]] = []
    for feature_family in ("all_aug", "consensus_aug"):
        for loss in ("hinge", "squared_hinge"):
            for c_value in (0.1, 1.0):
                for pair_weight_mode in ("uniform", "dcg_delta"):
                    lambdasvm_rows.append(
                        _evaluate_lambdasvm_candidate(
                            problems,
                            feature_family=feature_family,
                            loss=loss,
                            c_value=float(c_value),
                            pair_weight_mode=pair_weight_mode,
                            current_bundle=current_bundle,
                            current_cache_metrics=current_cache_metrics,
                            knn_metrics=knn_metrics,
                        )
                    )

    deepsets_rows: list[dict[str, Any]] = []
    deepsets_status: dict[str, Any] = {"executed": True, "reason": ""}
    if not torch_available:
        deepsets_status = {
            "executed": False,
            "reason": "Skipped because torch is not installed in the current environment.",
        }
    else:
        for pooling in ("mean", "max"):
            for hidden_dim, embed_dim in ((16, 8), (24, 12)):
                for pairwise_aux_weight in (0.25, 0.50):
                    deepsets_rows.append(
                        _evaluate_deepsets_candidate(
                            problems,
                            config=DeepSetsConfig(
                                pooling=pooling,
                                hidden_dim=int(hidden_dim),
                                embed_dim=int(embed_dim),
                                head_hidden_dim=max(8, int(embed_dim)),
                                epochs=120,
                                lr=2e-3,
                                weight_decay=1e-4,
                                pairwise_aux_weight=float(pairwise_aux_weight),
                                seed=42,
                            ).validate(),
                            torch_threads=int(args.torch_threads),
                            current_bundle=current_bundle,
                            current_cache_metrics=current_cache_metrics,
                            knn_metrics=knn_metrics,
                        )
                    )

    prior_gate_pass = any(bool(row["math_gate_passed"]) for row in lambdasvm_rows + deepsets_rows)
    set_transformer_rows: list[dict[str, Any]] = []
    set_transformer_status: dict[str, Any] = {"executed": True, "reason": ""}
    if not torch_available:
        set_transformer_status = {
            "executed": False,
            "reason": "Skipped because torch is not installed in the current environment.",
        }
    elif prior_gate_pass:
        set_transformer_status = {
            "executed": False,
            "reason": "Skipped because LambdaSVM or DeepSets already produced a math-gate-passing candidate.",
        }
    else:
        for num_heads in (2, 4):
            for pairwise_aux_weight in (0.0, 0.25):
                set_transformer_rows.append(
                    _evaluate_set_transformer_candidate(
                        problems,
                        config=SetTransformerLiteConfig(
                            pooling="mean",
                            model_dim=16,
                            num_heads=int(num_heads),
                            ff_hidden_dim=32,
                            head_hidden_dim=8,
                            epochs=120,
                            lr=2e-3,
                            weight_decay=1e-4,
                            pairwise_aux_weight=float(pairwise_aux_weight),
                            seed=42,
                        ).validate(),
                        torch_threads=int(args.torch_threads),
                        current_bundle=current_bundle,
                        current_cache_metrics=current_cache_metrics,
                        knn_metrics=knn_metrics,
                    )
                )

    family_best = {"lambdasvm": max(lambdasvm_rows, key=_row_rank_key)}
    if deepsets_rows:
        family_best["deepsets"] = max(deepsets_rows, key=_row_rank_key)
    if set_transformer_rows:
        family_best["set_transformer_lite"] = max(set_transformer_rows, key=_row_rank_key)
    overall_best = max(family_best.values(), key=_row_rank_key)

    model_paths: dict[str, str] = {}
    if not args.skip_save_model:
        for family_name, best_row in family_best.items():
            model_paths[family_name] = _train_and_save_family_best(
                best_row,
                problems,
                torch_threads=int(args.torch_threads),
            )

    payload = {
        "status_summary": {
            "submission_line_kept_frozen_from_score_recovery": True,
            "math_full_patch_remains_reverted_default": True,
            "new_research_families": ["lambdasvm", "deepsets_round2", "set_transformer_lite"],
        },
        "inputs": {
            "profile": str(args.profile),
            "ds_cache_root": _display_path(ds_cache_root),
            "science_json": _display_path(science_json),
            "code_v2_json": _display_path(code_v2_json),
            "distance_threads": int(args.distance_threads),
            "torch_threads": int(args.torch_threads),
            "max_problems": int(args.max_problems),
        },
        "baseline_metrics": baseline_metrics,
        "family_runs": {
            "lambdasvm": lambdasvm_rows,
            "deepsets_round2": deepsets_rows,
            "deepsets_status": deepsets_status,
            "set_transformer_lite": set_transformer_rows,
            "set_transformer_status": set_transformer_status,
        },
        "family_best": family_best,
        "overall_best": overall_best,
        "saved_models": model_paths,
    }

    out_path = out_dir / "math_bestofn_research_round2.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {_display_path(out_path)}")
    for family_name, model_path in model_paths.items():
        print(f"[model] {family_name}: {model_path}")


if __name__ == "__main__":
    main()
