#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.core.distance.engine import DistanceEngine, DistanceSpec
from nad.core.selectors.base import SelectorContext
from nad.core.selectors.math_svm_impl import (
    MATH_SVM_FEATURE_FAMILIES,
    MathLinearSVMScorer,
    MathRankSVMScorer,
    augment_math_svm_features,
    build_math_pairwise_hinge_training_examples,
    select_math_feature_family,
)
from nad.core.selectors.ml_features import extract_run_features
from nad.core.views.reader import Agg, CacheReader, CutSpec, CutType, Order, ViewSpec
from nad.ops.accuracy import _load_ground_truth
from scripts.run_code_baseline_v1_phase2 import MetricAccumulator
from scripts.run_gpqa_pairwise_round1 import _summarize_table

_AGG = Agg("max")
_CS = CutSpec(CutType.MASS, 0.98)
_VSPEC = ViewSpec(agg=_AGG, cut=_CS, order=Order.BY_KEY)

MATH_CACHE_PROFILES: dict[str, list[tuple[str, str]]] = {
    "train": [
        ("aime25", "MUI_HUB/cache_train/DeepSeek-R1-0528-Qwen3-8B/aime25/cache_neuron_output_1_act_no_rms_20251201_013209"),
        ("brumo25", "MUI_HUB/cache_train/DeepSeek-R1-0528-Qwen3-8B/brumo25/cache_neuron_output_1_act_no_rms_20251201_120226"),
        ("hmmt25", "MUI_HUB/cache_train/DeepSeek-R1-0528-Qwen3-8B/hmmt25/cache_neuron_output_1_act_no_rms_20251123_142339"),
    ],
    "main": [
        ("aime24", "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/aime24/cache_neuron_output_1_act_no_rms_20250902_025610"),
        ("aime25", "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/aime25/cache_neuron_output_1_act_no_rms_20251126_114548"),
        ("brumo25", "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/brumo25/cache_neuron_output_1_act_no_rms_20251126_071142"),
        ("hmmt25", "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/hmmt25/cache_neuron_output_1_act_no_rms_20251126_223151"),
    ],
    "test": [
        ("aime24", "MUI_HUB/cache_test/DeepSeek-R1-0528-Qwen3-8B/aime24/cache_neuron_output_1_act_no_rms_20251125_223716"),
        ("aime25", "MUI_HUB/cache_test/DeepSeek-R1-0528-Qwen3-8B/aime25/cache_neuron_output_1_act_no_rms_20250902_115633"),
        ("brumo25", "MUI_HUB/cache_test/DeepSeek-R1-0528-Qwen3-8B/brumo25/cache_neuron_output_1_act_no_rms_20251123_102708"),
        ("hmmt25", "MUI_HUB/cache_test/DeepSeek-R1-0528-Qwen3-8B/hmmt25/cache_neuron_output_1_act_no_rms_20251201_073112"),
    ],
}


@dataclass
class _MathProblemData:
    dataset: str
    problem_id: str
    run_ids: list[int]
    labels: np.ndarray
    X_all: np.ndarray
    D: np.ndarray


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _problem_sort_key(problem_id: str) -> tuple[int, str]:
    try:
        return (0, f"{int(str(problem_id).split('-')[-1]):09d}")
    except Exception:
        return (1, str(problem_id))


def _resolve_profile_entries(profile: str) -> list[tuple[str, Path]]:
    if profile not in MATH_CACHE_PROFILES:
        raise ValueError(f"Unknown math cache profile: {profile}")
    out: list[tuple[str, Path]] = []
    for dataset, rel_path in MATH_CACHE_PROFILES[profile]:
        out.append((dataset, REPO_ROOT / rel_path))
    return out


def _extract_cache_problems(
    dataset: str,
    cache_root: Path,
    *,
    distance_threads: int,
) -> list[_MathProblemData]:
    reader = CacheReader(str(cache_root))
    correctness = _load_ground_truth(str(cache_root))
    meta = json.loads((cache_root / "meta.json").read_text(encoding="utf-8"))
    groups: dict[str, list[int]] = {}
    for sid, sample in enumerate(meta["samples"]):
        pid = str(sample["problem_id"])
        groups.setdefault(pid, []).append(int(sid))

    engine = DistanceEngine(
        DistanceSpec(
            name="ja",
            normalize=True,
            num_threads=int(distance_threads),
            assume_unique=True,
        )
    )
    out: list[_MathProblemData] = []
    for pid in sorted(groups.keys(), key=_problem_sort_key):
        run_ids = list(groups[pid])
        if len(run_ids) < 2:
            continue
        views = [reader.get_run_view(rid, _VSPEC, normalize_l1=True) for rid in run_ids]
        lengths = np.asarray([len(v.keys) for v in views], dtype=np.int32)
        D = engine.dense_matrix(views)
        ctx = SelectorContext(
            cache=reader,
            problem_id=str(pid),
            run_ids=run_ids,
            views=views,
            pos_window=None,
        )
        X_base = extract_run_features(
            D,
            {"lengths": lengths, "views": views},
            context=ctx,
        )
        X_all = augment_math_svm_features(X_base)
        labels = np.asarray([int(bool(correctness.get(rid, False))) for rid in run_ids], dtype=np.int32)
        out.append(
            _MathProblemData(
                dataset=str(dataset),
                problem_id=f"{dataset}:{pid}",
                run_ids=run_ids,
                labels=labels,
                X_all=np.asarray(X_all, dtype=np.float64),
                D=np.asarray(D, dtype=np.float64),
            )
        )
    return out


def _extract_all_problems(profile: str, *, distance_threads: int) -> list[_MathProblemData]:
    all_problems: list[_MathProblemData] = []
    for dataset, cache_root in _resolve_profile_entries(profile):
        print(f"[math-svm] extracting {dataset} from {cache_root}", flush=True)
        all_problems.extend(_extract_cache_problems(dataset, cache_root, distance_threads=distance_threads))
    return all_problems


def _baseline_metrics(all_problems: list[_MathProblemData]) -> dict[str, dict[str, Any]]:
    baseline_columns = {
        "medoid": 1,
        "knn-medoid": 3,
        "deepconf": 7,
        "tournament-copeland": 9,
    }
    out: dict[str, dict[str, Any]] = {}
    for selector_name, col_idx in baseline_columns.items():
        acc = MetricAccumulator(selector_name, use_code_tiebreak=False)
        per_dataset = {
            prob.dataset: MetricAccumulator(f"{selector_name}:{prob.dataset}", use_code_tiebreak=False)
            for prob in all_problems
        }
        for prob in all_problems:
            scores = np.asarray(prob.X_all[:, int(col_idx)], dtype=np.float64)
            acc.add_problem(prob.problem_id, prob.run_ids, scores, prob.labels, prob.D)
            per_dataset[prob.dataset].add_problem(prob.problem_id, prob.run_ids, scores, prob.labels, prob.D)
        out[selector_name] = {
            "metrics": acc.finalize(),
            "by_dataset": {ds: per_dataset[ds].finalize() for ds in sorted(per_dataset.keys())},
        }
    return out


def _variant_configs(
    *,
    feature_families: list[str] | None = None,
    model_families: list[str] | None = None,
    losses: list[str] | None = None,
    c_grid: list[float] | None = None,
) -> list[dict[str, Any]]:
    feature_families = list(feature_families) if feature_families else list(MATH_SVM_FEATURE_FAMILIES)
    c_grid = list(c_grid) if c_grid else [0.03, 0.10, 0.30, 1.00, 3.00]
    losses = list(losses) if losses else ["hinge", "squared_hinge"]
    model_families_set = set(model_families or ["runwise", "ranksvm"])
    variants: list[dict[str, Any]] = []

    if "runwise" in model_families_set:
        for feature_family in feature_families:
            for loss in losses:
                for c_value in c_grid:
                    for fit_intercept in (False, True):
                        for class_weight in (None, "balanced"):
                            variants.append({
                                "name": (
                                    f"runwise__{feature_family}__{loss}"
                                    f"__C{c_value:.2f}"
                                    f"__{'bias' if fit_intercept else 'nobias'}"
                                    f"__{'balanced' if class_weight else 'plain'}"
                                ).replace(".", "p"),
                                "model_family": "runwise",
                                "feature_family": feature_family,
                                "loss": loss,
                                "C": float(c_value),
                                "fit_intercept": bool(fit_intercept),
                                "class_weight": class_weight,
                                "dual": "auto",
                                "max_iter": 100000,
                                "tol": 1e-4,
                            })

    if "ranksvm" in model_families_set:
        for feature_family in feature_families:
            for loss in losses:
                for c_value in c_grid:
                    variants.append({
                        "name": (
                            f"ranksvm__{feature_family}__{loss}"
                            f"__C{c_value:.2f}__nobias__utility"
                        ).replace(".", "p"),
                        "model_family": "ranksvm",
                        "feature_family": feature_family,
                        "loss": loss,
                        "C": float(c_value),
                        "fit_intercept": False,
                        "backend": "utility",
                        "dual": "auto",
                        "max_iter": 100000,
                        "tol": 1e-4,
                    })
                    for backend in ("mean_margin", "win_count"):
                        variants.append({
                            "name": (
                                f"ranksvm__{feature_family}__{loss}"
                                f"__C{c_value:.2f}__bias__{backend}"
                            ).replace(".", "p"),
                            "model_family": "ranksvm",
                            "feature_family": feature_family,
                            "loss": loss,
                            "C": float(c_value),
                            "fit_intercept": True,
                            "backend": backend,
                            "dual": "auto",
                            "max_iter": 100000,
                            "tol": 1e-4,
                        })

    return variants


def _prepare_family_views(
    all_problems: list[_MathProblemData],
) -> dict[str, dict[str, list[Any]]]:
    out: dict[str, dict[str, list[Any]]] = {}
    for family in MATH_SVM_FEATURE_FAMILIES:
        family_X = [select_math_feature_family(prob.X_all, family) for prob in all_problems]
        pair_X = []
        pair_y = []
        for prob, X_prob in zip(all_problems, family_X):
            X_pairs, y_pairs = build_math_pairwise_hinge_training_examples(X_prob, prob.labels)
            pair_X.append(X_pairs)
            pair_y.append(y_pairs)
        out[family] = {
            "X": family_X,
            "pair_X": pair_X,
            "pair_y": pair_y,
        }
    return out


def _bundle_metrics(
    overall_acc: MetricAccumulator,
    dataset_accs: dict[str, MetricAccumulator],
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics = overall_acc.finalize()
    by_dataset = {ds: dataset_accs[ds].finalize() for ds in sorted(dataset_accs.keys())}
    dataset_hit1 = [
        float(by_dataset[ds].get("hit@1") or 0.0)
        for ds in sorted(by_dataset.keys())
    ]
    aux = {
        "dataset_mean_hit@1": float(np.mean(dataset_hit1)) if dataset_hit1 else 0.0,
        "dataset_min_hit@1": float(np.min(dataset_hit1)) if dataset_hit1 else 0.0,
    }
    return metrics, {"by_dataset": by_dataset, **aux}


def _evaluate_variant(
    all_problems: list[_MathProblemData],
    family_views: dict[str, dict[str, list[Any]]],
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    family = str(cfg["feature_family"])
    X_by_problem = family_views[family]["X"]
    pair_X_by_problem = family_views[family]["pair_X"]
    pair_y_by_problem = family_views[family]["pair_y"]

    overall_acc = MetricAccumulator(str(cfg["name"]), use_code_tiebreak=False)
    dataset_accs = {
        prob.dataset: MetricAccumulator(f"{cfg['name']}:{prob.dataset}", use_code_tiebreak=False)
        for prob in all_problems
    }
    convergence_flags: list[bool] = []

    for held_out_idx, held_out in enumerate(all_problems):
        if held_out_idx % 20 == 0:
            print(
                f"[math-svm:{cfg['name']}] fold {held_out_idx + 1}/{len(all_problems)}",
                flush=True,
            )

        if cfg["model_family"] == "runwise":
            train_X = [X_by_problem[idx] for idx in range(len(all_problems)) if idx != held_out_idx]
            train_y = [all_problems[idx].labels for idx in range(len(all_problems)) if idx != held_out_idx]
            scorer = MathLinearSVMScorer(
                C=float(cfg["C"]),
                loss=str(cfg["loss"]),
                fit_intercept=bool(cfg["fit_intercept"]),
                class_weight=cfg.get("class_weight"),
                dual=cfg.get("dual", "auto"),
                max_iter=int(cfg.get("max_iter", 100000)),
                tol=float(cfg.get("tol", 1e-4)),
            )
            scorer.fit(np.concatenate(train_X, axis=0), np.concatenate(train_y, axis=0))
            scores = scorer.score_group(X_by_problem[held_out_idx])
            convergence_flags.append(bool(getattr(scorer, "converged_", True)))
        elif cfg["model_family"] == "ranksvm":
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
                scorer = MathRankSVMScorer(
                    C=float(cfg["C"]),
                    loss=str(cfg["loss"]),
                    fit_intercept=bool(cfg["fit_intercept"]),
                    backend=str(cfg["backend"]),
                    dual=cfg.get("dual", "auto"),
                    max_iter=int(cfg.get("max_iter", 100000)),
                    tol=float(cfg.get("tol", 1e-4)),
                )
                scorer.fit(np.concatenate(train_X, axis=0), np.concatenate(train_y, axis=0))
                scores = scorer.score_group(X_by_problem[held_out_idx])
                convergence_flags.append(bool(getattr(scorer, "converged_", True)))
        else:
            raise ValueError(f"Unknown model family: {cfg['model_family']}")

        overall_acc.add_problem(held_out.problem_id, held_out.run_ids, scores, held_out.labels, held_out.D)
        dataset_accs[held_out.dataset].add_problem(
            held_out.problem_id,
            held_out.run_ids,
            scores,
            held_out.labels,
            held_out.D,
        )

    metrics, aux = _bundle_metrics(overall_acc, dataset_accs)
    aux["converged_folds"] = int(sum(bool(x) for x in convergence_flags))
    aux["total_folds"] = int(len(convergence_flags))
    return metrics, aux


def _rank_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    metrics = row["metrics"]
    aux = row["aux"]
    return (
        float(metrics.get("hit@1") or 0.0),
        float(aux.get("dataset_mean_hit@1") or 0.0),
        float(metrics.get("selacc@10%") or 0.0),
        float(metrics.get("pairwise") or 0.0),
        float(metrics.get("auroc") or 0.0),
    )


def _write_outputs(
    out_dir: Path,
    baseline_metrics: dict[str, dict[str, Any]],
    variant_results: list[dict[str, Any]],
    *,
    profile: str,
    n_problems: int,
    final_model_bundle: dict[str, Any] | None,
    final: bool,
) -> None:
    if not variant_results:
        return

    ranked = sorted(variant_results, key=_rank_key, reverse=True)
    best = ranked[0]
    rows = [
        {"name": name, **{k: baseline_metrics[name]["metrics"][k] for k in ("auroc", "hit@1", "pairwise", "selacc@10%")}}
        for name in ("medoid", "knn-medoid", "deepconf", "tournament-copeland")
        if name in baseline_metrics
    ]
    rows.extend(
        {
            "name": row["name"],
            **{k: row["metrics"][k] for k in ("auroc", "hit@1", "pairwise", "selacc@10%")},
        }
        for row in ranked[:10]
    )

    summary_lines = [
        f"Math SVM profile: {profile}",
        _summarize_table(rows),
        "",
        f"Best variant: {best['name']}",
        json.dumps(
            {
                "config": best["config"],
                "metrics": best["metrics"],
                "aux": best["aux"],
            },
            indent=2,
            ensure_ascii=False,
        ),
    ]
    summary_text = "\n".join(summary_lines)
    payload = {
        "profile": profile,
        "baseline_metrics": baseline_metrics,
        "best_variant": best,
        "variants": ranked,
        "n_variants": len(ranked),
        "n_problems": n_problems,
        "final_model_bundle": final_model_bundle,
    }
    suffix = "" if final else ".partial"
    (out_dir / f"math_svm_sweep{suffix}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / f"summary{suffix}.txt").write_text(summary_text, encoding="utf-8")


def _train_final_model_bundle(
    all_problems: list[_MathProblemData],
    family_views: dict[str, dict[str, list[Any]]],
    cfg: dict[str, Any],
    model_out: Path,
) -> dict[str, Any]:
    family = str(cfg["feature_family"])
    X_by_problem = family_views[family]["X"]
    if cfg["model_family"] == "runwise":
        scorer = MathLinearSVMScorer(
            C=float(cfg["C"]),
            loss=str(cfg["loss"]),
            fit_intercept=bool(cfg["fit_intercept"]),
            class_weight=cfg.get("class_weight"),
            dual=cfg.get("dual", "auto"),
            max_iter=int(cfg.get("max_iter", 100000)),
            tol=float(cfg.get("tol", 1e-4)),
        )
        scorer.fit(
            np.concatenate(X_by_problem, axis=0),
            np.concatenate([prob.labels for prob in all_problems], axis=0),
        )
    elif cfg["model_family"] == "ranksvm":
        pair_X = family_views[family]["pair_X"]
        pair_y = family_views[family]["pair_y"]
        scorer = MathRankSVMScorer(
            C=float(cfg["C"]),
            loss=str(cfg["loss"]),
            fit_intercept=bool(cfg["fit_intercept"]),
            backend=str(cfg["backend"]),
            dual=cfg.get("dual", "auto"),
            max_iter=int(cfg.get("max_iter", 100000)),
            tol=float(cfg.get("tol", 1e-4)),
        )
        scorer.fit(
            np.concatenate([arr for arr in pair_X if arr.shape[0] > 0], axis=0),
            np.concatenate([arr for arr in pair_y if arr.shape[0] > 0], axis=0),
        )
    else:
        raise ValueError(f"Unknown model family: {cfg['model_family']}")

    import joblib

    model_out.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "config": cfg,
        "scorer": scorer,
    }
    joblib.dump(bundle, model_out)
    return {
        "model_path": str(model_out),
        "config": cfg,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Math-domain SVM sweep over runwise LinearSVC and pairwise RankSVM")
    ap.add_argument("--profile", default="train", choices=sorted(MATH_CACHE_PROFILES.keys()))
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--distance-threads", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--feature-families",
        default="",
        help="Optional comma-separated feature families to evaluate",
    )
    ap.add_argument(
        "--model-families",
        default="",
        help="Optional comma-separated model families to evaluate: runwise,ranksvm",
    )
    ap.add_argument(
        "--losses",
        default="",
        help="Optional comma-separated losses to evaluate: hinge,squared_hinge",
    )
    ap.add_argument(
        "--c-values",
        default="",
        help="Optional comma-separated C values, e.g. 0.03,0.1,0.3,1,3",
    )
    ap.add_argument(
        "--model-out",
        default="",
        help="Optional path for the best full-data model bundle",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "result" / f"math_svm_sweep_{args.profile}_{_now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_problems = _extract_all_problems(args.profile, distance_threads=int(args.distance_threads))
    baseline_metrics = _baseline_metrics(all_problems)
    family_views = _prepare_family_views(all_problems)

    feature_families = [x.strip() for x in args.feature_families.split(",") if x.strip()]
    model_families = [x.strip() for x in args.model_families.split(",") if x.strip()]
    losses = [x.strip() for x in args.losses.split(",") if x.strip()]
    c_values = [float(x.strip()) for x in args.c_values.split(",") if x.strip()]

    cfgs = _variant_configs(
        feature_families=feature_families or None,
        model_families=model_families or None,
        losses=losses or None,
        c_grid=c_values or None,
    )
    if int(args.limit) > 0:
        cfgs = cfgs[: int(args.limit)]

    partial_path = out_dir / "math_svm_sweep.partial.json"
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
        metrics, aux = _evaluate_variant(all_problems, family_views, cfg)
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
            profile=str(args.profile),
            n_problems=len(all_problems),
            final_model_bundle=None,
            final=False,
        )

    ranked = sorted(variant_results, key=_rank_key, reverse=True)
    best_cfg = ranked[0]["config"] if ranked else None
    final_model_bundle = None
    if best_cfg is not None:
        model_out = (
            Path(args.model_out)
            if args.model_out
            else out_dir / "best_model.pkl"
        )
        final_model_bundle = _train_final_model_bundle(all_problems, family_views, best_cfg, model_out)

    _write_outputs(
        out_dir,
        baseline_metrics,
        variant_results,
        profile=str(args.profile),
        n_problems=len(all_problems),
        final_model_bundle=final_model_bundle,
        final=True,
    )
    print((out_dir / "summary.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
