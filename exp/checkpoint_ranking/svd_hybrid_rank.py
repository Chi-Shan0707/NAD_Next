#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import linear_heads as lh
import weight_spectral as ws
from nad.ops.earlystop_svd import _build_representation, _rank_transform_matrix, load_earlystop_svd_bundle

DEFAULT_FEATURE_STORE = lh.DEFAULT_FEATURE_STORE
DEFAULT_BUNDLE_PATH = REPO_ROOT / "models" / "ml_selectors" / "es_svd_math_rr_r1.pkl"
DEFAULT_WEIGHT_FEATURE_FRAME = REPO_ROOT / "outputs" / "weight_spectral_feature_frame.csv"
DEFAULT_OUTPUTS_DIR = REPO_ROOT / "outputs"
DEFAULT_MEANCONF_EVAL = (
    REPO_ROOT
    / "results"
    / "scans"
    / "checkpoint_ranking"
    / "es_svd_math_rr_r1"
    / "es_svd_math_rr_r1__math5000rl_slot100_meanconf_eval.json"
)


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: str
    base_feature_kind: str
    use_weight_prior: bool
    smooth_kind: str
    configs: tuple[dict[str, Any], ...]
    description: str


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _score_tuple(metrics: dict[str, Any]) -> tuple[float, float, float, float, float]:
    def _coerce(value: Any, default: float = float("-inf")) -> float:
        if value is None:
            return default
        try:
            numeric = float(value)
        except Exception:
            return default
        if not math.isfinite(numeric):
            return default
        return numeric

    return (
        _coerce(metrics.get("checkpoint_spearman")),
        _coerce(metrics.get("checkpoint_kendall")),
        _coerce(metrics.get("top1_hit")),
        _coerce(metrics.get("top3_hit")),
        -_coerce(metrics.get("scenario_rmse"), default=float("inf")),
    )


def _make_dataset_with_x(subset: lh.DatasetSubset, x: np.ndarray) -> lh.DatasetSubset:
    return lh.DatasetSubset(
        x=np.asarray(x, dtype=np.float64),
        y=subset.y,
        scenario_ids=subset.scenario_ids,
        checkpoint_idx=subset.checkpoint_idx,
        checkpoint_names=subset.checkpoint_names,
        row_to_group=subset.row_to_group,
        group_sizes=subset.group_sizes,
        group_true_accuracy=subset.group_true_accuracy,
        groups=subset.groups,
        trajectory_group_indices=subset.trajectory_group_indices,
        checkpoint_group_indices=subset.checkpoint_group_indices,
        scenarios=subset.scenarios,
    )


def _load_or_build_weight_feature_frame(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)

    args = argparse.Namespace(
        model_root=ws.DEFAULT_MODEL_ROOT,
        truth_json=ws.DEFAULT_TRUTH_JSON,
        output_dir=DEFAULT_OUTPUTS_DIR,
        scenario_name="math5000rl_qwen3_4b",
        sketch_rows=192,
        sketch_cols=192,
        svd_rank=6,
        svd_oversamples=4,
        svd_iters=2,
        probe_count=4,
        exact_drift=False,
        drift_sample_rows=96,
        drift_sample_cols=96,
        drift_sample_vector=8192,
        full_load_numel=30_000_000,
        chunk_rows=512,
        feature_cap=96,
        ridge_alpha=2.0,
        logreg_c=0.35,
        smooth_lambda=0.35,
        abs_sparsity_eps=1e-6,
        rel_sparsity_eps=1e-3,
        sign_eps=1e-7,
        torch_threads=4,
        device="cpu",
    )
    truth_map = ws._load_truth_map(args.truth_json)
    checkpoints = ws._discover_checkpoints(args.model_root, truth_map)
    tie_word_embeddings = bool(ws._load_config_flag(checkpoints[0].model_dir, "tie_word_embeddings", True))
    feature_df = ws._extract_feature_frame(
        checkpoints,
        args,
        scenario_name=str(args.scenario_name),
        tie_word_embeddings=tie_word_embeddings,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(path, index=False)
    return feature_df


def _build_route_latent_features(
    dataset: lh.DatasetSubset,
    bundle_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    bundle = load_earlystop_svd_bundle(bundle_path)
    route = bundle["domains"]["math"]["routes"][9]
    feature_indices = [int(value) for value in route["feature_indices"]]
    scaler = route["model"]["scaler"]
    svd = route["model"]["svd"]
    lr = route["model"]["lr"]
    whiten = bool(route["model"].get("whiten", False))

    latent_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    for group in dataset.groups:
        x_raw = dataset.x[group.row_indices]
        x_rank = _rank_transform_matrix(x_raw)
        x_rep = _build_representation(
            x_raw=x_raw,
            x_rank=x_rank,
            feature_indices=feature_indices,
            representation=str(route["representation"]),
        )
        z = svd.transform(scaler.transform(x_rep))
        if whiten:
            singular_values = np.asarray(svd.singular_values_, dtype=np.float64)
            singular_values = np.where(np.abs(singular_values) < 1e-8, 1.0, singular_values)
            z = z / singular_values
        score = np.asarray(lr.decision_function(z), dtype=np.float64).reshape(-1, 1)
        latent_parts.append(np.asarray(z, dtype=np.float64))
        score_parts.append(score)
    return (
        np.vstack(score_parts).astype(np.float64, copy=False),
        np.vstack(latent_parts).astype(np.float64, copy=False),
    )


def _build_base_datasets(
    dataset: lh.DatasetSubset,
    *,
    bundle_path: Path,
) -> dict[str, lh.DatasetSubset]:
    route_score, route_latent = _build_route_latent_features(dataset, bundle_path)
    return {
        "raw": dataset,
        "raw_route_latent": _make_dataset_with_x(
            dataset,
            np.hstack([dataset.x, route_score, route_latent]),
        ),
    }


def _checkpoint_target_map(subset: lh.DatasetSubset) -> dict[str, float]:
    out = lh.checkpoint_scores_from_group_scores(subset, subset.group_true_accuracy)
    return {str(key): float(value) for key, value in out.items()}


def _prior_cache_key(target_map: dict[str, float]) -> tuple[float, ...]:
    return tuple(round(float(target_map[name]), 10) for name in lh.OFFICIAL_CHECKPOINTS)


def _fit_weight_prior(
    feature_frame: pd.DataFrame,
    target_map: dict[str, float],
    *,
    feature_cap: int,
    ridge_alpha: float,
    logreg_c: float,
    smooth_lambda: float,
    cache: dict[tuple[float, ...], tuple[pd.DataFrame, dict[str, Any]]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    key = _prior_cache_key(target_map)
    cached = cache.get(key)
    if cached is not None:
        return cached

    df = feature_frame.copy()
    df["true_accuracy"] = df["checkpoint_name"].map(target_map).astype(float)
    y = np.asarray(df["true_accuracy"], dtype=np.float64)
    feature_cols, feature_rank_df = ws._select_features(df, y, feature_cap=int(feature_cap))
    fit_df, fit_payload = ws._fit_full_scores(
        df,
        feature_cols,
        ridge_alpha=float(ridge_alpha),
        logreg_c=float(logreg_c),
        smooth_lambda=float(smooth_lambda),
    )
    meta = {
        "selected_feature_count": int(len(feature_cols)),
        "selected_features": list(feature_cols),
        "blend_alpha": float(fit_payload["alpha"]),
        "feature_rank_df": feature_rank_df,
    }
    cache[key] = (fit_df.copy(), meta)
    return cache[key]


def _augment_with_weight_prior(
    subset: lh.DatasetSubset,
    prior_df: pd.DataFrame,
) -> lh.DatasetSubset:
    prior_lookup = prior_df.set_index("checkpoint_name")
    prior_cols = (
        "fit_pointwise",
        "fit_pairwise",
        "fit_combined_raw",
        "fit_combined_smoothed",
    )
    prior_parts = []
    for col in prior_cols:
        prior_parts.append(
            np.asarray(
                [float(prior_lookup.loc[str(name), col]) for name in subset.checkpoint_names.tolist()],
                dtype=np.float64,
            ).reshape(-1, 1)
        )
    position = np.asarray(
        [float(lh.CHECKPOINT_ORDER[str(name)]) / max(1, len(lh.OFFICIAL_CHECKPOINTS) - 1) for name in subset.checkpoint_names.tolist()],
        dtype=np.float64,
    ).reshape(-1, 1)
    return _make_dataset_with_x(subset, np.hstack([subset.x, *prior_parts, position]))


def _build_model(candidate: CandidateSpec, config: dict[str, Any]) -> lh.BaseHead:
    merged = dict(config)
    merged.setdefault("reg_lambda", 1e-3)
    merged.setdefault("max_iter", 300)
    merged["smooth_kind"] = candidate.smooth_kind
    return lh.CompositeLinearHead(merged)


def get_candidate_specs() -> tuple[CandidateSpec, ...]:
    baseline_cfgs = (
        {"w_run": 1.0, "w_rank": 0.0, "w_cal": 0.5, "w_smooth": 0.1, "reg_lambda": 1e-3},
        {"w_run": 1.0, "w_rank": 0.0, "w_cal": 0.5, "w_smooth": 0.5, "reg_lambda": 1e-3},
    )
    prior_smooth_cfgs = (
        {"w_run": 0.50, "w_rank": 0.0, "w_cal": 1.0, "w_smooth": 0.25, "reg_lambda": 1e-3},
        {"w_run": 0.25, "w_rank": 0.5, "w_cal": 1.0, "w_smooth": 0.50, "reg_lambda": 1e-3},
    )
    prior_weak_cfgs = (
        {"w_run": 0.50, "w_rank": 0.0, "w_cal": 1.0, "w_smooth": 0.10, "reg_lambda": 1e-3},
        {"w_run": 0.25, "w_rank": 0.5, "w_cal": 1.0, "w_smooth": 0.25, "reg_lambda": 1e-3},
    )
    prior_multi_cfgs = (
        {"w_run": 0.50, "w_rank": 0.5, "w_cal": 1.0, "w_smooth": 0.25, "reg_lambda": 1e-3},
        {"w_run": 0.25, "w_rank": 1.0, "w_cal": 1.0, "w_smooth": 0.50, "reg_lambda": 1e-3},
    )
    return (
        CandidateSpec(
            name="smooth_raw_baseline",
            family="baseline",
            base_feature_kind="raw",
            use_weight_prior=False,
            smooth_kind="smooth_l2",
            configs=baseline_cfgs,
            description="Raw response features with trajectory L2 smoothing only.",
        ),
        CandidateSpec(
            name="smooth_route_latent",
            family="activation_svd",
            base_feature_kind="raw_route_latent",
            use_weight_prior=False,
            smooth_kind="smooth_l2",
            configs=baseline_cfgs,
            description="Raw response features plus slot-100 route score and route latent coordinates.",
        ),
        CandidateSpec(
            name="svd_weight_prior_smooth",
            family="hybrid",
            base_feature_kind="raw",
            use_weight_prior=True,
            smooth_kind="smooth_l2",
            configs=prior_smooth_cfgs,
            description="Raw response features plus checkpoint-level SVD weight prior.",
        ),
        CandidateSpec(
            name="svd_weight_prior_route_latent_smooth",
            family="hybrid",
            base_feature_kind="raw_route_latent",
            use_weight_prior=True,
            smooth_kind="smooth_l2",
            configs=prior_smooth_cfgs,
            description="Activation-SVD route latent plus weight-SVD checkpoint prior with smoothness regularization.",
        ),
        CandidateSpec(
            name="svd_weight_prior_route_latent_weak_monotone",
            family="hybrid",
            base_feature_kind="raw_route_latent",
            use_weight_prior=True,
            smooth_kind="weak_monotone",
            configs=prior_weak_cfgs,
            description="Activation-SVD route latent plus weight-SVD checkpoint prior under weak-monotone regularization.",
        ),
        CandidateSpec(
            name="svd_weight_prior_route_latent_multi_objective",
            family="hybrid",
            base_feature_kind="raw_route_latent",
            use_weight_prior=True,
            smooth_kind="smooth_l2",
            configs=prior_multi_cfgs,
            description="Activation-SVD route latent plus weight-SVD checkpoint prior with run/rank/calibration multitask loss.",
        ),
    )


def _prepare_candidate_subset(
    base_subset: lh.DatasetSubset,
    candidate: CandidateSpec,
    *,
    feature_frame: pd.DataFrame,
    target_map: dict[str, float] | None,
    prior_cache: dict[tuple[float, ...], tuple[pd.DataFrame, dict[str, Any]]],
    prior_feature_cap: int,
    prior_ridge_alpha: float,
    prior_logreg_c: float,
    prior_smooth_lambda: float,
) -> tuple[lh.DatasetSubset, dict[str, Any]]:
    if not candidate.use_weight_prior:
        return base_subset, {}
    if target_map is None:
        raise ValueError("target_map is required when candidate.use_weight_prior=True")
    prior_df, prior_meta = _fit_weight_prior(
        feature_frame,
        target_map,
        feature_cap=prior_feature_cap,
        ridge_alpha=prior_ridge_alpha,
        logreg_c=prior_logreg_c,
        smooth_lambda=prior_smooth_lambda,
        cache=prior_cache,
    )
    return _augment_with_weight_prior(base_subset, prior_df), prior_meta


def _select_best_config_for_candidate(
    candidate: CandidateSpec,
    train_base: lh.DatasetSubset,
    *,
    feature_frame: pd.DataFrame,
    inner_splits: int,
    prior_cache: dict[tuple[float, ...], tuple[pd.DataFrame, dict[str, Any]]],
    prior_feature_cap: int,
    prior_ridge_alpha: float,
    prior_logreg_c: float,
    prior_smooth_lambda: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    inner_split_list = lh.build_group_splits(train_base.scenarios, int(inner_splits))
    if not inner_split_list or len(candidate.configs) == 1:
        cfg = dict(candidate.configs[0])
        train_target_map = _checkpoint_target_map(train_base)
        train_subset, prior_meta = _prepare_candidate_subset(
            train_base,
            candidate,
            feature_frame=feature_frame,
            target_map=train_target_map,
            prior_cache=prior_cache,
            prior_feature_cap=prior_feature_cap,
            prior_ridge_alpha=prior_ridge_alpha,
            prior_logreg_c=prior_logreg_c,
            prior_smooth_lambda=prior_smooth_lambda,
        )
        model = _build_model(candidate, cfg)
        model.fit(train_subset)
        metrics = lh.evaluate_group_predictions(train_subset, model.predict_group_scores(train_subset))
        metrics["prior_meta"] = prior_meta
        return cfg, metrics

    split_train_idx, split_val_idx = inner_split_list[0]
    split_train = lh.subset_by_scenarios(train_base, [train_base.scenarios[int(idx)] for idx in split_train_idx.tolist()])
    split_val = lh.subset_by_scenarios(train_base, [train_base.scenarios[int(idx)] for idx in split_val_idx.tolist()])
    split_train_target_map = _checkpoint_target_map(split_train)
    split_train_subset, prior_meta = _prepare_candidate_subset(
        split_train,
        candidate,
        feature_frame=feature_frame,
        target_map=split_train_target_map,
        prior_cache=prior_cache,
        prior_feature_cap=prior_feature_cap,
        prior_ridge_alpha=prior_ridge_alpha,
        prior_logreg_c=prior_logreg_c,
        prior_smooth_lambda=prior_smooth_lambda,
    )
    split_val_subset, _ = _prepare_candidate_subset(
        split_val,
        candidate,
        feature_frame=feature_frame,
        target_map=split_train_target_map,
        prior_cache=prior_cache,
        prior_feature_cap=prior_feature_cap,
        prior_ridge_alpha=prior_ridge_alpha,
        prior_logreg_c=prior_logreg_c,
        prior_smooth_lambda=prior_smooth_lambda,
    )

    best_config: dict[str, Any] | None = None
    best_metrics: dict[str, Any] | None = None
    for config in candidate.configs:
        model = _build_model(candidate, dict(config))
        try:
            model.fit(split_train_subset)
            preds = model.predict_group_scores(split_val_subset)
            metrics = lh.evaluate_group_predictions(split_val_subset, preds)
        except Exception:
            continue
        if best_metrics is None or _score_tuple(metrics) > _score_tuple(best_metrics):
            best_config = dict(config)
            best_metrics = dict(metrics)
    if best_config is None or best_metrics is None:
        best_config = dict(candidate.configs[0])
        best_metrics = {}
    best_metrics["prior_meta"] = prior_meta
    return best_config, best_metrics


def run_nested_oof(
    dataset: lh.DatasetSubset,
    *,
    base_datasets: dict[str, lh.DatasetSubset],
    feature_frame: pd.DataFrame,
    outer_splits: int,
    inner_splits: int,
    prior_feature_cap: int,
    prior_ridge_alpha: float,
    prior_logreg_c: float,
    prior_smooth_lambda: float,
) -> dict[str, dict[str, Any]]:
    candidates = get_candidate_specs()
    full_lookup = lh.build_group_lookup(dataset)
    prior_cache: dict[tuple[float, ...], tuple[pd.DataFrame, dict[str, Any]]] = {}
    results: dict[str, dict[str, Any]] = {
        candidate.name: {
            "candidate": candidate,
            "family": candidate.family,
            "group_oof": np.full(len(dataset.groups), np.nan, dtype=np.float64),
            "fold_records": [],
            "selected_configs": [],
        }
        for candidate in candidates
    }

    outer_split_list = lh.build_group_splits(dataset.scenarios, int(outer_splits))
    for fold_idx, (train_s_idx, test_s_idx) in enumerate(outer_split_list):
        train_scenarios = [dataset.scenarios[int(idx)] for idx in train_s_idx.tolist()]
        test_scenarios = [dataset.scenarios[int(idx)] for idx in test_s_idx.tolist()]
        print(
            f"[outer {fold_idx + 1}/{len(outer_split_list)}] train={len(train_scenarios)} test={len(test_scenarios)}",
            flush=True,
        )
        for candidate in candidates:
            print(f"  - {candidate.name}", flush=True)
            train_base = lh.subset_by_scenarios(base_datasets[candidate.base_feature_kind], train_scenarios)
            test_base = lh.subset_by_scenarios(base_datasets[candidate.base_feature_kind], test_scenarios)
            best_config, inner_metrics = _select_best_config_for_candidate(
                candidate,
                train_base,
                feature_frame=feature_frame,
                inner_splits=inner_splits,
                prior_cache=prior_cache,
                prior_feature_cap=prior_feature_cap,
                prior_ridge_alpha=prior_ridge_alpha,
                prior_logreg_c=prior_logreg_c,
                prior_smooth_lambda=prior_smooth_lambda,
            )
            train_target_map = _checkpoint_target_map(train_base)
            train_subset, prior_meta = _prepare_candidate_subset(
                train_base,
                candidate,
                feature_frame=feature_frame,
                target_map=train_target_map,
                prior_cache=prior_cache,
                prior_feature_cap=prior_feature_cap,
                prior_ridge_alpha=prior_ridge_alpha,
                prior_logreg_c=prior_logreg_c,
                prior_smooth_lambda=prior_smooth_lambda,
            )
            test_subset, _ = _prepare_candidate_subset(
                test_base,
                candidate,
                feature_frame=feature_frame,
                target_map=train_target_map,
                prior_cache=prior_cache,
                prior_feature_cap=prior_feature_cap,
                prior_ridge_alpha=prior_ridge_alpha,
                prior_logreg_c=prior_logreg_c,
                prior_smooth_lambda=prior_smooth_lambda,
            )

            model = _build_model(candidate, best_config)
            model.fit(train_subset)
            preds = model.predict_group_scores(test_subset)
            fold_metrics = lh.evaluate_group_predictions(test_subset, preds)
            results[candidate.name]["selected_configs"].append(
                {
                    "fold": int(fold_idx),
                    "config": dict(best_config),
                    "inner_checkpoint_spearman": inner_metrics.get("checkpoint_spearman"),
                    "inner_checkpoint_kendall": inner_metrics.get("checkpoint_kendall"),
                    "inner_scenario_rmse": inner_metrics.get("scenario_rmse"),
                    "prior_selected_feature_count": int(prior_meta.get("selected_feature_count", 0)),
                    "prior_blend_alpha": prior_meta.get("blend_alpha"),
                }
            )
            results[candidate.name]["fold_records"].append(
                {
                    "fold": int(fold_idx),
                    "metrics": fold_metrics,
                    "config": dict(best_config),
                    "n_scenarios": int(len(test_subset.scenarios)),
                    "prior_selected_feature_count": int(prior_meta.get("selected_feature_count", 0)),
                    "prior_blend_alpha": prior_meta.get("blend_alpha"),
                }
            )
            for local_group_idx, group in enumerate(test_subset.groups):
                global_group_idx = full_lookup[(str(group.scenario_id), int(group.checkpoint_idx))]
                results[candidate.name]["group_oof"][global_group_idx] = float(preds[local_group_idx])

    for candidate in candidates:
        oof = results[candidate.name]["group_oof"]
        if np.any(~np.isfinite(oof)):
            raise RuntimeError(f"Incomplete OOF predictions for candidate={candidate.name}")
        results[candidate.name]["overall_metrics"] = lh.evaluate_group_predictions(dataset, oof)
    return results


def build_summary_table(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for name, payload in results.items():
        metrics = payload["overall_metrics"]
        config_counter: dict[str, int] = {}
        for item in payload["selected_configs"]:
            key = json.dumps(item["config"], sort_keys=True, default=_json_default)
            config_counter[key] = config_counter.get(key, 0) + 1
        representative_config = max(config_counter.items(), key=lambda item: item[1])[0] if config_counter else "{}"
        prior_feature_counts = [int(item.get("prior_selected_feature_count", 0)) for item in payload["selected_configs"]]
        prior_blends = [float(item["prior_blend_alpha"]) for item in payload["selected_configs"] if item.get("prior_blend_alpha") is not None]
        candidate = payload["candidate"]
        rows.append(
            {
                "candidate_name": name,
                "family": payload["family"],
                "base_feature_kind": candidate.base_feature_kind,
                "use_weight_prior": int(candidate.use_weight_prior),
                "smooth_kind": candidate.smooth_kind,
                "checkpoint_spearman": metrics["checkpoint_spearman"],
                "checkpoint_kendall": metrics["checkpoint_kendall"],
                "checkpoint_pearson": metrics["checkpoint_pearson"],
                "top1_hit": metrics["top1_hit"],
                "top3_hit": metrics["top3_hit"],
                "scenario_rmse": metrics["scenario_rmse"],
                "scenario_brier": metrics["scenario_brier"],
                "scenario_ece": metrics["scenario_ece"],
                "predicted_rank_order": " > ".join(metrics["predicted_rank_order"]),
                "representative_config": representative_config,
                "prior_selected_feature_count_mean": float(np.mean(prior_feature_counts)) if prior_feature_counts else 0.0,
                "prior_blend_alpha_mean": float(np.mean(prior_blends)) if prior_blends else None,
                "description": candidate.description,
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["checkpoint_spearman"] if row["checkpoint_spearman"] is not None else float("-inf")),
            float(row["checkpoint_kendall"] if row["checkpoint_kendall"] is not None else float("-inf")),
            float(row["top1_hit"]),
            float(row["top3_hit"]),
            -float(row["scenario_rmse"]),
        ),
        reverse=True,
    )
    return rows


def write_oof_csv(path: Path, dataset: lh.DatasetSubset, results: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "candidate_name",
                "family",
                "base_feature_kind",
                "use_weight_prior",
                "row_type",
                "fold",
                "checkpoint",
                "checkpoint_idx",
                "oof_checkpoint_score",
                "true_checkpoint_accuracy",
                "checkpoint_spearman",
                "checkpoint_kendall",
                "checkpoint_pearson",
                "top1_hit",
                "top3_hit",
                "scenario_rmse",
                "scenario_brier",
                "scenario_ece",
                "selected_config",
            ],
        )
        writer.writeheader()
        for candidate_name, payload in results.items():
            candidate = payload["candidate"]
            metrics = payload["overall_metrics"]
            checkpoint_pred = metrics["checkpoint_pred_scores"]
            checkpoint_true = metrics["checkpoint_true_scores"]
            writerows = []
            for checkpoint_name in lh.OFFICIAL_CHECKPOINTS:
                writerows.append(
                    {
                        "candidate_name": candidate_name,
                        "family": payload["family"],
                        "base_feature_kind": candidate.base_feature_kind,
                        "use_weight_prior": int(candidate.use_weight_prior),
                        "row_type": "overall_checkpoint",
                        "fold": "overall",
                        "checkpoint": checkpoint_name,
                        "checkpoint_idx": int(lh.CHECKPOINT_ORDER[checkpoint_name]),
                        "oof_checkpoint_score": float(checkpoint_pred[checkpoint_name]),
                        "true_checkpoint_accuracy": float(checkpoint_true[checkpoint_name]),
                        "checkpoint_spearman": metrics["checkpoint_spearman"],
                        "checkpoint_kendall": metrics["checkpoint_kendall"],
                        "checkpoint_pearson": metrics["checkpoint_pearson"],
                        "top1_hit": metrics["top1_hit"],
                        "top3_hit": metrics["top3_hit"],
                        "scenario_rmse": metrics["scenario_rmse"],
                        "scenario_brier": metrics["scenario_brier"],
                        "scenario_ece": metrics["scenario_ece"],
                        "selected_config": json.dumps(payload["selected_configs"], ensure_ascii=False, default=_json_default),
                    }
                )
            for fold_record in payload["fold_records"]:
                fold_metrics = fold_record["metrics"]
                for checkpoint_name in lh.OFFICIAL_CHECKPOINTS:
                    writerows.append(
                        {
                            "candidate_name": candidate_name,
                            "family": payload["family"],
                            "base_feature_kind": candidate.base_feature_kind,
                            "use_weight_prior": int(candidate.use_weight_prior),
                            "row_type": "fold_checkpoint",
                            "fold": int(fold_record["fold"]),
                            "checkpoint": checkpoint_name,
                            "checkpoint_idx": int(lh.CHECKPOINT_ORDER[checkpoint_name]),
                            "oof_checkpoint_score": float(fold_metrics["checkpoint_pred_scores"][checkpoint_name]),
                            "true_checkpoint_accuracy": float(fold_metrics["checkpoint_true_scores"][checkpoint_name]),
                            "checkpoint_spearman": fold_metrics["checkpoint_spearman"],
                            "checkpoint_kendall": fold_metrics["checkpoint_kendall"],
                            "checkpoint_pearson": fold_metrics["checkpoint_pearson"],
                            "top1_hit": fold_metrics["top1_hit"],
                            "top3_hit": fold_metrics["top3_hit"],
                            "scenario_rmse": fold_metrics["scenario_rmse"],
                            "scenario_brier": fold_metrics["scenario_brier"],
                            "scenario_ece": fold_metrics["scenario_ece"],
                            "selected_config": json.dumps(fold_record["config"], ensure_ascii=False, default=_json_default),
                        }
                    )
            writer.writerows(writerows)


def plot_calibration_curves(path: Path, results: dict[str, dict[str, Any]]) -> None:
    ordered = build_summary_table(results)
    n_items = len(ordered)
    n_cols = 3
    n_rows = int(math.ceil(n_items / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.3 * n_rows), squeeze=False)
    for ax in axes.ravel():
        ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="lightgray", linewidth=1.0)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Predicted scenario-checkpoint accuracy")
        ax.set_ylabel("Observed scenario-checkpoint accuracy")
    for ax, row in zip(axes.ravel(), ordered):
        metrics = results[row["candidate_name"]]["overall_metrics"]
        x_vals, y_vals, counts = lh.reliability_curve(metrics["scenario_pred"], metrics["scenario_true"], n_bins=8)
        if x_vals.size > 0:
            ax.plot(x_vals, y_vals, marker="o", linewidth=1.5)
            ax.scatter(x_vals, y_vals, s=20 + 1.1 * counts, alpha=0.7)
        ax.set_title(
            f"{row['candidate_name']}\n"
            f"ρ={metrics['checkpoint_spearman']:.3f}  "
            f"τ={metrics['checkpoint_kendall']:.3f}  "
            f"RMSE={metrics['scenario_rmse']:.3f}"
        )
    for ax in axes.ravel()[n_items:]:
        ax.axis("off")
    fig.suptitle("SVD hybrid checkpoint-rank heads: OOF calibration curves", fontsize=15)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _load_meanconf_baseline(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return None
    return metrics


def write_report(
    path: Path,
    dataset: lh.DatasetSubset,
    results: dict[str, dict[str, Any]],
    *,
    feature_store_path: Path,
    weight_feature_frame_path: Path,
    bundle_path: Path,
    meanconf_baseline: dict[str, Any] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary_rows = build_summary_table(results)
    winner = summary_rows[0]
    checkpoint_true = lh.checkpoint_scores_from_group_scores(dataset, dataset.group_true_accuracy)
    lines: list[str] = []
    lines.append("# SVD Hybrid RL Checkpoint Ranking\n")
    lines.append("\n## Summary\n")
    lines.append(
        f"- Data: `{_display_path(feature_store_path)}` with {len(dataset.scenarios)} scenarios × {len(lh.OFFICIAL_CHECKPOINTS)} checkpoints × "
        f"{int(dataset.y.shape[0] / max(1, len(dataset.scenarios) * len(lh.OFFICIAL_CHECKPOINTS)))} runs.\n"
    )
    lines.append(f"- Activation SVD bundle: `{_display_path(bundle_path)}` (slot-100 math route latent appended when requested).\n")
    lines.append(f"- Weight prior frame: `{_display_path(weight_feature_frame_path)}`.\n")
    lines.append("- OOF protocol: 5-fold GroupKFold by scenario, with one grouped inner split for config selection.\n")
    if meanconf_baseline is not None:
        lines.append(
            f"- Local slot-100 mean-confidence baseline: ρ={float(meanconf_baseline.get('spearman_rho', 0.0)):.3f}, "
            f"τ={float(meanconf_baseline.get('kendall_tau', 0.0)):.3f}.\n"
        )
    lines.append(
        f"- Best SVD head: `{winner['candidate_name']}` "
        f"(ρ={winner['checkpoint_spearman']:.3f}, τ={winner['checkpoint_kendall']:.3f}, "
        f"RMSE={winner['scenario_rmse']:.3f}).\n"
    )

    lines.append("\n## OOF Metrics\n")
    lines.append("| Candidate | Family | Base | Prior | Spearman | Kendall | Pearson | Top1 | Top3 | RMSE | ECE |\n")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for row in summary_rows:
        lines.append(
            f"| `{row['candidate_name']}` | {row['family']} | `{row['base_feature_kind']}` | {int(row['use_weight_prior'])} | "
            f"{row['checkpoint_spearman']:.3f} | {row['checkpoint_kendall']:.3f} | {row['checkpoint_pearson']:.3f} | "
            f"{int(row['top1_hit'])} | {int(row['top3_hit'])} | {row['scenario_rmse']:.3f} | {row['scenario_ece']:.3f} |\n"
        )

    lines.append("\n## Candidate Notes\n")
    for row in summary_rows:
        blend_note = ""
        if row["prior_blend_alpha_mean"] is not None:
            blend_note = f"; mean prior blend alpha `{float(row['prior_blend_alpha_mean']):.3f}`"
        lines.append(
            f"- `{row['candidate_name']}`: {row['description']} "
            f"Representative config `{row['representative_config']}`; "
            f"predicted order `{row['predicted_rank_order']}`; "
            f"mean selected prior features `{row['prior_selected_feature_count_mean']:.1f}`{blend_note}.\n"
        )

    lines.append("\n## True Checkpoint Accuracy\n")
    lines.append("| Checkpoint | True accuracy |\n")
    lines.append("| --- | ---: |\n")
    for checkpoint_name in lh.OFFICIAL_CHECKPOINTS:
        lines.append(f"| `{checkpoint_name}` | {checkpoint_true[checkpoint_name]:.4f} |\n")

    lines.append("\n## Recommendation\n")
    lines.append(
        f"- Prefer `{winner['candidate_name']}` for the final ensemble: it gives the strongest checkpoint-order OOF among the tested SVD heads.\n"
    )
    if winner["use_weight_prior"]:
        lines.append(
            "- The gain comes from combining two complementary SVD views: run-level activation route latent and checkpoint-level weight spectral prior.\n"
        )
    else:
        lines.append(
            "- The gain comes from a simpler activation-side SVD view without adding a checkpoint prior.\n"
        )
    lines.append(
        "- It stays lightweight, linear, and interpretable: every added signal is either a route latent coordinate or a checkpoint-global linear prior from weight drift spectra.\n"
    )
    path.write_text("".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Hybrid SVD heads for RL checkpoint ranking")
    ap.add_argument("--feature-store", type=Path, default=DEFAULT_FEATURE_STORE)
    ap.add_argument("--bundle-path", type=Path, default=DEFAULT_BUNDLE_PATH)
    ap.add_argument("--weight-feature-frame", type=Path, default=DEFAULT_WEIGHT_FEATURE_FRAME)
    ap.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR)
    ap.add_argument("--outer-splits", type=int, default=5)
    ap.add_argument("--inner-splits", type=int, default=2)
    ap.add_argument("--prior-feature-cap", type=int, default=96)
    ap.add_argument("--prior-ridge-alpha", type=float, default=2.0)
    ap.add_argument("--prior-logreg-c", type=float, default=0.35)
    ap.add_argument("--prior-smooth-lambda", type=float, default=0.35)
    ap.add_argument("--meanconf-eval", type=Path, default=DEFAULT_MEANCONF_EVAL)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    dataset = lh.load_feature_store(Path(args.feature_store).resolve())
    weight_feature_frame_path = Path(args.weight_feature_frame).resolve()
    feature_frame = _load_or_build_weight_feature_frame(weight_feature_frame_path)
    base_datasets = _build_base_datasets(dataset, bundle_path=Path(args.bundle_path).resolve())
    results = run_nested_oof(
        dataset,
        base_datasets=base_datasets,
        feature_frame=feature_frame,
        outer_splits=int(args.outer_splits),
        inner_splits=int(args.inner_splits),
        prior_feature_cap=int(args.prior_feature_cap),
        prior_ridge_alpha=float(args.prior_ridge_alpha),
        prior_logreg_c=float(args.prior_logreg_c),
        prior_smooth_lambda=float(args.prior_smooth_lambda),
    )

    outputs_dir = Path(args.outputs_dir).resolve()
    outputs_dir.mkdir(parents=True, exist_ok=True)
    oof_csv = outputs_dir / "svd_hybrid_rank_oof.csv"
    calibration_png = outputs_dir / "svd_hybrid_rank_calibration.png"
    report_md = outputs_dir / "svd_hybrid_rank_report.md"
    write_oof_csv(oof_csv, dataset, results)
    plot_calibration_curves(calibration_png, results)
    write_report(
        report_md,
        dataset,
        results,
        feature_store_path=Path(args.feature_store).resolve(),
        weight_feature_frame_path=weight_feature_frame_path,
        bundle_path=Path(args.bundle_path).resolve(),
        meanconf_baseline=_load_meanconf_baseline(Path(args.meanconf_eval).resolve()),
    )
    summary = build_summary_table(results)
    print(
        json.dumps(
            {
                "winner": summary[0],
                "n_candidates": len(summary),
                "oof_csv": str(oof_csv),
                "report": str(report_md),
            },
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        )
    )


if __name__ == "__main__":
    main()
