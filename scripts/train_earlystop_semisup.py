#!/usr/bin/env python3
"""Label-efficiency / semi-supervised follow-up for EarlyStop SSL bases."""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Callable

for _env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env, "1")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nad.ops.earlystop_ssl import (
    AnchorTable,
    augment_with_pseudo_labels,
    build_anchor_tables,
    collect_problem_keys,
    extract_route_models,
    fit_linear_head,
    frozen_svd_transform,
    linear_head_scores,
    mask_features,
    subset_store_by_problem_keys,
    transform_pair_bundle,
    transform_shared_pair_single_input,
    transform_single_bundle,
)
from nad.ops.earlystop_svd import _rank_transform_matrix, load_earlystop_svd_bundle
from scripts.run_earlystop_prefix10_svd_round1 import (
    EARLY_STOP_POSITIONS,
    SEARCH_C_VALUES,
    evaluate_method_from_feature_store,
)
from scripts.train_earlystop_ssl_basis import (
    ANCHOR_POS_INDICES,
    BASIS_CONFIGS,
    FIXED_FEATURE_INDICES,
    POS_TO_ANCHOR_IDX,
    PROTOCOL_TEXT,
    TOKEN_FIXED_INDICES,
    _aggregate_row,
    _fit_basis,
    _fit_heads_from_tables,
    _load_prebuilt_stores,
    _make_score_fn,
    _resolve_path,
    _write_csv,
)
from SVDomain.train_es_svd_ms_rr_r1 import _build_holdout_problem_map, _split_feature_store


DEFAULT_CONFIGS = ["adjacent_cca", "tokenpair_rrr", "denoise_full"]
DEFAULT_LABEL_FRACTIONS = [0.05, 0.10, 0.25, 0.50, 1.0]
DEFAULT_PSEUDO_THRESHOLDS = [0.90, 0.95]
DEFAULT_SUPERVISED_BUNDLE = "models/ml_selectors/es_svd_ms_rr_r1.pkl"


def _quiet_eval(
    *,
    method_name: str,
    holdout_store: list[dict[str, Any]],
    score_fn: Callable[[str, int, np.ndarray], np.ndarray],
) -> dict[str, Any]:
    capture = io.StringIO()
    with redirect_stdout(capture):
        return evaluate_method_from_feature_store(
            method_name=method_name,
            feature_store=holdout_store,
            position_values=tuple(float(v) for v in EARLY_STOP_POSITIONS),
            score_fn=score_fn,
        )


def _budget_keys(
    payloads: list[dict[str, Any]],
    *,
    domain: str,
    fraction: float,
    seed: int,
) -> set[str]:
    problem_keys = collect_problem_keys(payloads, domain=domain)
    if not problem_keys:
        return set()
    rng = np.random.RandomState(int(seed))
    order = rng.permutation(len(problem_keys))
    n_select = max(1, int(round(float(fraction) * len(problem_keys))))
    n_select = min(n_select, len(problem_keys))
    return {problem_keys[int(idx)] for idx in order[:n_select].tolist()}


def _table_features_with_bundle(
    table: AnchorTable,
    config_name: str,
    bundle: Any,
) -> np.ndarray:
    if config_name == "adjacent_cca":
        return transform_shared_pair_single_input(bundle, table.x_full)
    if config_name == "adjacent_rrr":
        return transform_shared_pair_single_input(bundle, table.x_full)
    if config_name == "denoise_full":
        return transform_single_bundle(bundle, table.x_full)
    if config_name == "tokenpair_rrr":
        return transform_pair_bundle(bundle, table.x_token, table.x_full)
    if config_name == "tokenpair_cca":
        return transform_pair_bundle(bundle, table.x_token, table.x_full)
    if config_name == "raw_rank_rrr":
        return transform_pair_bundle(bundle, table.x_raw, table.x_rank)
    if config_name == "raw_rank_cca":
        return transform_pair_bundle(bundle, table.x_raw, table.x_rank)
    raise ValueError(f"Unsupported config_name: {config_name}")


def _perturbed_table_features(
    table: AnchorTable,
    config_name: str,
    bundle: Any,
    *,
    seed: int,
    mask_rate: float,
) -> np.ndarray:
    if config_name in {"adjacent_cca", "adjacent_rrr"}:
        return transform_shared_pair_single_input(
            bundle,
            mask_features(table.x_full, mask_rate=mask_rate, seed=seed),
        )
    if config_name == "denoise_full":
        return transform_single_bundle(
            bundle,
            mask_features(table.x_full, mask_rate=mask_rate, seed=seed),
        )
    if config_name in {"tokenpair_rrr", "tokenpair_cca"}:
        return transform_pair_bundle(
            bundle,
            mask_features(table.x_token, mask_rate=mask_rate, seed=seed),
            mask_features(table.x_full, mask_rate=mask_rate, seed=seed + 17),
        )
    if config_name in {"raw_rank_rrr", "raw_rank_cca"}:
        return transform_pair_bundle(
            bundle,
            mask_features(table.x_raw, mask_rate=mask_rate, seed=seed),
            mask_features(table.x_rank, mask_rate=mask_rate, seed=seed + 17),
        )
    raise ValueError(f"Unsupported config_name: {config_name}")


def _make_frozen_svd_score_fn(
    *,
    domain: str,
    route_models: dict[int, dict[str, Any]],
    heads: dict[int, Any],
) -> Callable[[str, int, np.ndarray], np.ndarray]:
    def _score(dom: str, pos_idx: int, x_raw_all: np.ndarray) -> np.ndarray:
        n_rows = int(x_raw_all.shape[0])
        if dom != domain:
            return np.zeros(n_rows, dtype=np.float64)
        anchor_idx = int(POS_TO_ANCHOR_IDX.get(pos_idx, 0))
        head = heads.get(anchor_idx)
        route_model = route_models.get(anchor_idx)
        if head is None or route_model is None:
            return np.zeros(n_rows, dtype=np.float64)
        x_fixed = np.asarray(x_raw_all[:, FIXED_FEATURE_INDICES], dtype=np.float64)
        x_full = np.concatenate([x_fixed, _rank_transform_matrix(x_fixed)], axis=1)
        feats = frozen_svd_transform(route_model, x_full)
        return linear_head_scores(head, feats)

    return _score


def _write_label_summary(rows: list[dict[str, Any]], out_path: Path) -> None:
    def _norm_key_value(value: Any) -> Any:
        try:
            if isinstance(value, float) and not np.isfinite(value):
                return "nan"
            value_f = float(value)
            if not np.isfinite(value_f):
                return "nan"
            return value_f
        except Exception:
            return value

    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["stage"],
            row["protocol"],
            row["domain"],
            row["condition"],
            row["basis_scope"],
            row["view_name"],
            row["method"],
            row["positive_mode"],
            row["ssl_rank"],
            _norm_key_value(row["label_fraction"]),
            _norm_key_value(row["pseudo_threshold"]),
        )
        buckets.setdefault(key, []).append(row)

    def _mean_std(key_name: str, bucket: list[dict[str, Any]]) -> tuple[float, float]:
        vals = [float(r[key_name]) for r in bucket if np.isfinite(float(r[key_name]))]
        if not vals:
            return float("nan"), float("nan")
        arr = np.asarray(vals, dtype=np.float64)
        return float(np.mean(arr)), float(np.std(arr))

    summary_rows: list[dict[str, Any]] = []
    for key, bucket in sorted(buckets.items()):
        auc_mean, auc_std = _mean_std("auc_of_auroc", bucket)
        sel_mean, sel_std = _mean_std("auc_of_selacc", bucket)
        earliest_mean, earliest_std = _mean_std("earliest_gt_0_6", bucket)
        au100_mean, au100_std = _mean_std("auroc_at_100", bucket)
        stop_mean, stop_std = _mean_std("stop_acc_at_100", bucket)
        pseudo_n_mean, pseudo_n_std = _mean_std("n_pseudo", bucket)
        pseudo_p_mean, pseudo_p_std = _mean_std("pseudo_precision", bucket)
        n_lab_mean, n_lab_std = _mean_std("n_labeled_problems", bucket)
        summary_rows.append(
            {
                "stage": key[0],
                "protocol": key[1],
                "domain": key[2],
                "condition": key[3],
                "basis_scope": key[4],
                "view_name": key[5],
                "method": key[6],
                "positive_mode": key[7],
                "ssl_rank": key[8],
                "label_fraction": key[9],
                "pseudo_threshold": key[10],
                "n_seeds": len(bucket),
                "n_labeled_problems_mean": n_lab_mean,
                "n_labeled_problems_std": n_lab_std,
                "auc_of_auroc_mean": auc_mean,
                "auc_of_auroc_std": auc_std,
                "auc_of_selacc_mean": sel_mean,
                "auc_of_selacc_std": sel_std,
                "earliest_gt_0_6_mean": earliest_mean,
                "earliest_gt_0_6_std": earliest_std,
                "auroc_at_100_mean": au100_mean,
                "auroc_at_100_std": au100_std,
                "stop_acc_at_100_mean": stop_mean,
                "stop_acc_at_100_std": stop_std,
                "n_pseudo_mean": pseudo_n_mean,
                "n_pseudo_std": pseudo_n_std,
                "pseudo_precision_mean": pseudo_p_mean,
                "pseudo_precision_std": pseudo_p_std,
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(summary_rows, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EarlyStop SSL semi-supervised label-efficiency study")
    parser.add_argument("--prebuilt-cache-dir", default="results/cache/es_svd_ms_rr_r1")
    parser.add_argument("--supervised-bundle", default=DEFAULT_SUPERVISED_BUNDLE)
    parser.add_argument("--holdout-split", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--domains", nargs="+", default=["math", "science"])
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--ssl-ranks", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--label-fractions", nargs="+", type=float, default=DEFAULT_LABEL_FRACTIONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--pseudo-thresholds", nargs="+", type=float, default=DEFAULT_PSEUDO_THRESHOLDS)
    parser.add_argument("--pseudo-max-per-group", type=int, default=8)
    parser.add_argument("--cca-reg", type=float, default=0.10)
    parser.add_argument("--mask-rate", type=float, default=0.15)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out-csv", default="results/tables/earlystop_ssl_label_efficiency.csv")
    parser.add_argument("--out-summary-csv", default="results/tables/earlystop_ssl_label_efficiency_summary.csv")
    parser.add_argument("--out-detail-json", default="results/scans/earlystop_ssl/ssl_semisup_detail.json")
    args = parser.parse_args()

    config_map = {cfg.name: cfg for cfg in BASIS_CONFIGS}
    active_configs = [config_map[name] for name in args.configs if name in config_map]
    if not active_configs:
        raise SystemExit("No active SSL configs selected")

    label_fractions = [float(v) for v in args.label_fractions]
    seeds = [int(v) for v in args.seeds]
    ssl_ranks = [int(v) for v in args.ssl_ranks]
    thresholds = [float(v) for v in args.pseudo_thresholds]
    if args.smoke:
        label_fractions = label_fractions[:2]
        seeds = seeds[:1]
        ssl_ranks = ssl_ranks[:1]
        thresholds = thresholds[:1]

    prebuilt_cache_dir = _resolve_path(args.prebuilt_cache_dir)
    feature_store = _load_prebuilt_stores(prebuilt_cache_dir)
    holdout_problem_map, split_summary = _build_holdout_problem_map(
        feature_store,
        holdout_split=float(args.holdout_split),
        split_seed=int(args.split_seed),
    )
    train_payloads, holdout_payloads, _ = _split_feature_store(
        feature_store,
        holdout_problem_map=holdout_problem_map,
    )
    holdout_by_domain = {
        domain: [payload for payload in holdout_payloads if str(payload["domain"]) == domain]
        for domain in args.domains
    }
    full_train_tables = {
        domain: build_anchor_tables(
            train_payloads,
            fixed_feature_indices=FIXED_FEATURE_INDICES,
            token_feature_indices=TOKEN_FIXED_INDICES,
            anchor_position_indices=ANCHOR_POS_INDICES,
            domain=domain,
        )
        for domain in args.domains
    }

    bundle_path = _resolve_path(args.supervised_bundle)
    supervised_bundle = load_earlystop_svd_bundle(bundle_path)
    supervised_route_models = {
        domain: extract_route_models(
            supervised_bundle,
            domain=domain,
            early_stop_positions=tuple(float(v) for v in EARLY_STOP_POSITIONS),
            pos_to_anchor_idx=POS_TO_ANCHOR_IDX,
        )
        for domain in args.domains
    }

    shared_basis_cache: dict[tuple[str, str, int, int], Any] = {}
    for domain in args.domains:
        tables = full_train_tables[domain]
        for config in active_configs:
            for ssl_rank in ssl_ranks:
                for seed in seeds:
                    key = (domain, config.name, ssl_rank, seed)
                    shared_basis_cache[key] = _fit_basis(
                        config,
                        tables=tables,
                        rank=ssl_rank,
                        seed=seed,
                        cca_reg=float(args.cca_reg),
                        mask_rate=float(args.mask_rate),
                        basis_scope="shared",
                        positive_mode="same_run",
                    )

    rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []

    for domain in args.domains:
        holdout_store = holdout_by_domain.get(domain, [])
        if not holdout_store:
            print(f"[skip] domain={domain}: no holdout rows")
            continue

        route_models = supervised_route_models.get(domain, {})
        print(f"\n[domain] {domain}")

        for seed in seeds:
            for fraction in label_fractions:
                labeled_keys = _budget_keys(
                    train_payloads,
                    domain=domain,
                    fraction=fraction,
                    seed=seed,
                )
                all_keys = set(collect_problem_keys(train_payloads, domain=domain))
                unlabeled_keys = set(all_keys) - set(labeled_keys)
                labeled_store = subset_store_by_problem_keys(
                    train_payloads,
                    domain=domain,
                    problem_keys=labeled_keys,
                )
                unlabeled_store = subset_store_by_problem_keys(
                    train_payloads,
                    domain=domain,
                    problem_keys=unlabeled_keys,
                )
                labeled_tables = build_anchor_tables(
                    labeled_store,
                    fixed_feature_indices=FIXED_FEATURE_INDICES,
                    token_feature_indices=TOKEN_FIXED_INDICES,
                    anchor_position_indices=ANCHOR_POS_INDICES,
                    domain=domain,
                )
                unlabeled_tables = build_anchor_tables(
                    unlabeled_store,
                    fixed_feature_indices=FIXED_FEATURE_INDICES,
                    token_feature_indices=TOKEN_FIXED_INDICES,
                    anchor_position_indices=ANCHOR_POS_INDICES,
                    domain=domain,
                )
                n_labeled = len(labeled_keys)
                print(f"[budget] domain={domain} seed={seed} frac={fraction:.0%} n_labeled={n_labeled}")

                no_basis_heads = _fit_heads_from_tables(
                    labeled_tables,
                    feature_fn=lambda _anchor_idx, table: table.x_full,
                    seed=seed,
                )
                no_basis_score = _make_score_fn(
                    domain=domain,
                    config=None,
                    bundle_spec=None,
                    heads=no_basis_heads,
                    baseline_mode="raw+rank",
                )
                eval_result = _quiet_eval(
                    method_name=f"no_basis/{domain}/f{fraction:.2f}/s{seed}",
                    holdout_store=holdout_store,
                    score_fn=no_basis_score,
                )
                row = {
                    **_aggregate_row(
                        stage="ssl_semisup",
                        domain=domain,
                        seed=seed,
                        condition="no_basis_lr",
                        basis_scope="none",
                        view_name="raw+rank",
                        method="lr",
                        positive_mode="n/a",
                        ssl_rank=0,
                        metrics=eval_result["aggregate"],
                    ),
                    "label_fraction": float(fraction),
                    "n_labeled_problems": int(n_labeled),
                    "pseudo_threshold": float("nan"),
                    "n_pseudo": 0,
                    "pseudo_precision": float("nan"),
                }
                rows.append(row)
                detail_rows.append({**row, "by_cache": eval_result["by_cache"]})

                frozen_svd_heads = _fit_heads_from_tables(
                    labeled_tables,
                    feature_fn=lambda anchor_idx, table, routes=route_models: frozen_svd_transform(
                        routes[anchor_idx],
                        table.x_full,
                    ) if anchor_idx in routes else np.zeros((table.x_full.shape[0], 1), dtype=np.float64),
                    seed=seed,
                )
                frozen_svd_score = _make_frozen_svd_score_fn(
                    domain=domain,
                    route_models=route_models,
                    heads=frozen_svd_heads,
                )
                eval_result = _quiet_eval(
                    method_name=f"frozen_svd/{domain}/f{fraction:.2f}/s{seed}",
                    holdout_store=holdout_store,
                    score_fn=frozen_svd_score,
                )
                row = {
                    **_aggregate_row(
                        stage="ssl_semisup",
                        domain=domain,
                        seed=seed,
                        condition="frozen_svd",
                        basis_scope="shared",
                        view_name="canonical_svd",
                        method="svd_lr",
                        positive_mode="n/a",
                        ssl_rank=0,
                        metrics=eval_result["aggregate"],
                    ),
                    "label_fraction": float(fraction),
                    "n_labeled_problems": int(n_labeled),
                    "pseudo_threshold": float("nan"),
                    "n_pseudo": 0,
                    "pseudo_precision": float("nan"),
                }
                rows.append(row)
                detail_rows.append({**row, "by_cache": eval_result["by_cache"]})

                for config in active_configs:
                    for ssl_rank in ssl_ranks:
                        shared_bundle = shared_basis_cache[(domain, config.name, ssl_rank, seed)]

                        ssl_heads = _fit_heads_from_tables(
                            labeled_tables,
                            feature_fn=lambda anchor_idx, table, cfg_name=config.name, bundle=shared_bundle: _table_features_with_bundle(
                                table,
                                cfg_name,
                                bundle if not isinstance(bundle, dict) else bundle[anchor_idx],
                            ),
                            seed=seed,
                        )
                        ssl_score = _make_score_fn(
                            domain=domain,
                            config=config,
                            bundle_spec=shared_bundle,
                            heads=ssl_heads,
                            baseline_mode="ssl",
                        )
                        eval_result = _quiet_eval(
                            method_name=f"frozen_ssl/{config.name}/{domain}/r{ssl_rank}/f{fraction:.2f}/s{seed}",
                            holdout_store=holdout_store,
                            score_fn=ssl_score,
                        )
                        base_row = {
                            **_aggregate_row(
                                stage="ssl_semisup",
                                domain=domain,
                                seed=seed,
                                condition="frozen_ssl_sup",
                                basis_scope="shared",
                                view_name=config.view,
                                method=config.method,
                                positive_mode="same_run",
                                ssl_rank=ssl_rank,
                                metrics=eval_result["aggregate"],
                            ),
                            "label_fraction": float(fraction),
                            "n_labeled_problems": int(n_labeled),
                            "pseudo_threshold": float("nan"),
                            "n_pseudo": 0,
                            "pseudo_precision": float("nan"),
                        }
                        rows.append(base_row)
                        detail_rows.append({**base_row, "config_name": config.name, "by_cache": eval_result["by_cache"]})

                        if config.supports_task_specific:
                            try:
                                task_bundle = _fit_basis(
                                    config,
                                    tables=labeled_tables,
                                    rank=ssl_rank,
                                    seed=seed,
                                    cca_reg=float(args.cca_reg),
                                    mask_rate=float(args.mask_rate),
                                    basis_scope="task_specific",
                                    positive_mode="same_run",
                                )
                            except Exception:
                                task_bundle = None
                            if isinstance(task_bundle, dict) and task_bundle:
                                task_heads = _fit_heads_from_tables(
                                    labeled_tables,
                                    feature_fn=lambda anchor_idx, table, cfg_name=config.name, bundle=task_bundle: _table_features_with_bundle(
                                        table,
                                        cfg_name,
                                        bundle[anchor_idx],
                                    ),
                                    seed=seed,
                                )
                                task_score = _make_score_fn(
                                    domain=domain,
                                    config=config,
                                    bundle_spec=task_bundle,
                                    heads=task_heads,
                                    baseline_mode="ssl",
                                )
                                eval_result = _quiet_eval(
                                    method_name=f"task_ssl/{config.name}/{domain}/r{ssl_rank}/f{fraction:.2f}/s{seed}",
                                    holdout_store=holdout_store,
                                    score_fn=task_score,
                                )
                                row = {
                                    **_aggregate_row(
                                        stage="ssl_semisup",
                                        domain=domain,
                                        seed=seed,
                                        condition="task_ssl_sup",
                                        basis_scope="task_specific",
                                        view_name=config.view,
                                        method=config.method,
                                        positive_mode="same_run",
                                        ssl_rank=ssl_rank,
                                        metrics=eval_result["aggregate"],
                                    ),
                                    "label_fraction": float(fraction),
                                    "n_labeled_problems": int(n_labeled),
                                    "pseudo_threshold": float("nan"),
                                    "n_pseudo": 0,
                                    "pseudo_precision": float("nan"),
                                }
                                rows.append(row)
                                detail_rows.append({**row, "config_name": config.name, "by_cache": eval_result["by_cache"]})

                        for threshold in thresholds:
                            pseudo_heads: dict[int, Any] = {}
                            agreement_heads: dict[int, Any] = {}
                            pseudo_count = 0
                            pseudo_correct = 0
                            agreement_count = 0
                            agreement_correct = 0

                            for anchor_idx, table_lab in labeled_tables.items():
                                if anchor_idx not in ssl_heads:
                                    continue
                                table_un = unlabeled_tables.get(anchor_idx)
                                if table_un is None or table_un.x_full.shape[0] <= 0:
                                    agreement_heads[anchor_idx] = ssl_heads[anchor_idx]
                                    pseudo_heads[anchor_idx] = ssl_heads[anchor_idx]
                                    continue

                                bundle = shared_bundle if not isinstance(shared_bundle, dict) else shared_bundle[anchor_idx]
                                x_lab = _table_features_with_bundle(table_lab, config.name, bundle)
                                x_un = _table_features_with_bundle(table_un, config.name, bundle)
                                x_un_alt = _perturbed_table_features(
                                    table_un,
                                    config.name,
                                    bundle,
                                    seed=seed + anchor_idx + 97,
                                    mask_rate=float(args.mask_rate),
                                )
                                teacher_head = ssl_heads[anchor_idx]

                                aug = augment_with_pseudo_labels(
                                    x_lab,
                                    table_lab.y,
                                    table_lab.groups,
                                    x_un,
                                    table_un.groups,
                                    teacher_head,
                                    threshold=float(threshold),
                                    max_per_group=int(args.pseudo_max_per_group),
                                )
                                if int(aug["selected_idx"].shape[0]) > 0:
                                    pseudo_count += int(aug["selected_idx"].shape[0])
                                    pseudo_correct += int(np.sum(aug["pseudo_y"] == table_un.y[aug["selected_idx"]]))
                                head = fit_linear_head(
                                    aug["x"],
                                    aug["y"],
                                    aug["groups"],
                                    c_values=tuple(float(v) for v in SEARCH_C_VALUES),
                                    random_state=seed,
                                )
                                if head is not None:
                                    pseudo_heads[anchor_idx] = head

                                aug_agreement = augment_with_pseudo_labels(
                                    x_lab,
                                    table_lab.y,
                                    table_lab.groups,
                                    x_un,
                                    table_un.groups,
                                    teacher_head,
                                    threshold=float(threshold),
                                    alt_unlabeled=x_un_alt,
                                    max_per_group=int(args.pseudo_max_per_group),
                                )
                                if int(aug_agreement["selected_idx"].shape[0]) > 0:
                                    agreement_count += int(aug_agreement["selected_idx"].shape[0])
                                    agreement_correct += int(
                                        np.sum(aug_agreement["pseudo_y"] == table_un.y[aug_agreement["selected_idx"]])
                                    )
                                head = fit_linear_head(
                                    aug_agreement["x"],
                                    aug_agreement["y"],
                                    aug_agreement["groups"],
                                    c_values=tuple(float(v) for v in SEARCH_C_VALUES),
                                    random_state=seed,
                                )
                                if head is not None:
                                    agreement_heads[anchor_idx] = head

                            pseudo_score = _make_score_fn(
                                domain=domain,
                                config=config,
                                bundle_spec=shared_bundle,
                                heads=pseudo_heads,
                                baseline_mode="ssl",
                            )
                            eval_result = _quiet_eval(
                                method_name=f"pseudo/{config.name}/{domain}/r{ssl_rank}/f{fraction:.2f}/t{threshold:.2f}/s{seed}",
                                holdout_store=holdout_store,
                                score_fn=pseudo_score,
                            )
                            row = {
                                **_aggregate_row(
                                    stage="ssl_semisup",
                                    domain=domain,
                                    seed=seed,
                                    condition="frozen_ssl_pseudo",
                                    basis_scope="shared",
                                    view_name=config.view,
                                    method=config.method,
                                    positive_mode="same_run",
                                    ssl_rank=ssl_rank,
                                    metrics=eval_result["aggregate"],
                                ),
                                "label_fraction": float(fraction),
                                "n_labeled_problems": int(n_labeled),
                                "pseudo_threshold": float(threshold),
                                "n_pseudo": int(pseudo_count),
                                "pseudo_precision": float(pseudo_correct / pseudo_count) if pseudo_count > 0 else float("nan"),
                            }
                            rows.append(row)
                            detail_rows.append({**row, "config_name": config.name, "by_cache": eval_result["by_cache"]})

                            agreement_score = _make_score_fn(
                                domain=domain,
                                config=config,
                                bundle_spec=shared_bundle,
                                heads=agreement_heads,
                                baseline_mode="ssl",
                            )
                            eval_result = _quiet_eval(
                                method_name=f"agreement/{config.name}/{domain}/r{ssl_rank}/f{fraction:.2f}/t{threshold:.2f}/s{seed}",
                                holdout_store=holdout_store,
                                score_fn=agreement_score,
                            )
                            row = {
                                **_aggregate_row(
                                    stage="ssl_semisup",
                                    domain=domain,
                                    seed=seed,
                                    condition="frozen_ssl_agreement",
                                    basis_scope="shared",
                                    view_name=config.view,
                                    method=config.method,
                                    positive_mode="same_run",
                                    ssl_rank=ssl_rank,
                                    metrics=eval_result["aggregate"],
                                ),
                                "label_fraction": float(fraction),
                                "n_labeled_problems": int(n_labeled),
                                "pseudo_threshold": float(threshold),
                                "n_pseudo": int(agreement_count),
                                "pseudo_precision": float(agreement_correct / agreement_count) if agreement_count > 0 else float("nan"),
                            }
                            rows.append(row)
                            detail_rows.append({**row, "config_name": config.name, "by_cache": eval_result["by_cache"]})

    out_csv = _resolve_path(args.out_csv)
    out_summary_csv = _resolve_path(args.out_summary_csv)
    out_detail_json = _resolve_path(args.out_detail_json)
    _write_csv(rows, out_csv)
    _write_label_summary(rows, out_summary_csv)
    out_detail_json.parent.mkdir(parents=True, exist_ok=True)
    out_detail_json.write_text(
        json.dumps(
            {
                "protocol": {
                    "text": PROTOCOL_TEXT,
                    "prebuilt_cache_dir": str(prebuilt_cache_dir),
                    "holdout_split": float(args.holdout_split),
                    "split_seed": int(args.split_seed),
                    "split_summary": split_summary,
                    "supervised_bundle": str(bundle_path),
                },
                "rows": detail_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[out] csv -> {out_csv}")
    print(f"[out] summary -> {out_summary_csv}")
    print(f"[out] detail -> {out_detail_json}")


if __name__ == "__main__":
    main()
