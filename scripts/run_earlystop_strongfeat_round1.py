#!/usr/bin/env python3
"""Run EarlyStop strong-feature SVD round1 with cache + cache_train holdout."""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.core.selectors.trajectory_impl import DEFAULT_REFLECTION_THRESHOLD
from nad.ops.earlystop import EARLY_STOP_POSITIONS
from nad.ops.earlystop_svd import (
    FULL_FEATURE_NAMES,
    _auroc,
    _build_representation,
    _cv_auroc_baseline,
    _fit_svd_lr_model,
    _group_folds,
    _rank_transform_matrix,
    load_earlystop_svd_bundle,
    save_earlystop_svd_bundle,
)
from nad.ops.earlystop_svm import load_earlystop_svm_bundle
from scripts.run_earlystop_prefix10_svd_round1 import (
    ANCHOR_POSITIONS,
    DEFAULT_FEATURE_CHUNK_PROBLEMS,
    EXTRACTION_POSITIONS,
    OFFICIAL_SLOT_TO_ANCHOR,
    _display_path,
    _fmt_earliest,
    _fmt_pct,
    _now_utc,
    _pct_label,
    build_anchor_bundle,
    build_feature_store,
    choose_best_candidate,
    collect_required_features_for_eval,
    evaluate_method_from_feature_store,
    _metric_ge,
    make_bridge_score_fn,
    make_svd_bundle_score_fn,
    make_tok_conf_score_fn,
    summarise_route,
)
from scripts.run_earlystop_prefix10_svd_round1b import (
    _build_holdout_problem_map,
    _build_scope_training_tables_with_dual_groups,
    _render_cache_table,
    _render_method_table,
    _split_feature_store,
    _summarise_feature_store,
)


STRONG_FEATURE_FAMILY_MAP = {
    "strong_core3": [
        "tok_conf_prefix",
        "tok_conf_recency",
        "traj_reflection_count",
    ],
    "strong_tail5": [
        "tok_conf_prefix",
        "tok_conf_recency",
        "traj_reflection_count",
        "tail_q10",
        "head_tail_gap",
    ],
    "strong_stable6": [
        "tok_conf_prefix",
        "tok_conf_recency",
        "traj_reflection_count",
        "tail_q10",
        "head_tail_gap",
        "tail_variance",
    ],
    "strong_event7": [
        "tok_conf_prefix",
        "tok_conf_recency",
        "traj_reflection_count",
        "tail_q10",
        "head_tail_gap",
        "tail_variance",
        "last_event_tail_conf",
    ],
    "strong_recovery8": [
        "tok_conf_prefix",
        "tok_conf_recency",
        "traj_reflection_count",
        "tail_q10",
        "head_tail_gap",
        "tail_variance",
        "last_event_tail_conf",
        "event_pre_post_delta",
    ],
}

STRONG_BASELINE_SIGNAL_NAMES = (
    "tok_conf_prefix",
    "tok_conf_recency",
    "traj_reflection_count",
    "tail_q10",
    "head_tail_gap",
    "tail_variance",
    "last_event_tail_conf",
    "event_pre_post_delta",
)

SEARCH_REPRESENTATIONS = ("raw", "rank", "raw+rank")
SEARCH_RANKS = (2, 4, 6, 8, 12, 16)
SEARCH_C_VALUES = (0.05, 0.10, 0.20, 0.50, 1.00)
SEARCH_WHITEN = (False, True)
SEARCH_CLASS_WEIGHT = ("none", "balanced")
SEARCH_REFLECTION_THRESHOLDS = (0.20, 0.30)

REPO_STATUS_LINES = [
    "repo 已经有 `earlystop_svd_lowrank_lr_v1` 的导出链路",
    "本轮继续只做 Early-Stop，不切换模型家族",
    "当前 strongest-feature 证据来自 `docs/reference/FEATURES.md`、`results/selector_comparison/selector_comparison.md` 与 `README.md`",
    "本轮目标是把 EarlyStop 特征范围收窄到强单特征及其邻近 tail / recovery 信号，再用 `10/40/70/100` 训练",
]


def _qualify_feature_store(feature_store: list[dict[str, Any]], source_name: str) -> list[dict[str, Any]]:
    qualified: list[dict[str, Any]] = []
    for payload in feature_store:
        item = dict(payload)
        item["source_name"] = str(source_name)
        item["base_cache_key"] = str(payload["cache_key"])
        item["cache_key"] = f"{source_name}/{payload['cache_key']}"
        qualified.append(item)
    return qualified


def _feature_cache_key(
    *,
    source_name: str,
    cache_root: str,
    positions: tuple[float, ...],
    required_feature_names: set[str],
    max_problems_per_cache: Optional[int],
    reflection_threshold: float,
) -> str:
    payload = {
        "version": 1,
        "source_name": str(source_name),
        "cache_root": str(cache_root),
        "positions": [float(p) for p in positions],
        "required_feature_names": sorted(str(v) for v in required_feature_names),
        "max_problems_per_cache": None if max_problems_per_cache is None else int(max_problems_per_cache),
        "reflection_threshold": float(reflection_threshold),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


def _feature_cache_path(
    *,
    cache_dir: Path,
    source_name: str,
    cache_root: str,
    positions: tuple[float, ...],
    required_feature_names: set[str],
    max_problems_per_cache: Optional[int],
    reflection_threshold: float,
) -> Path:
    key = _feature_cache_key(
        source_name=source_name,
        cache_root=cache_root,
        positions=positions,
        required_feature_names=required_feature_names,
        max_problems_per_cache=max_problems_per_cache,
        reflection_threshold=reflection_threshold,
    )
    suffix = "all" if max_problems_per_cache is None else f"cap{int(max_problems_per_cache)}"
    thr_tag = f"ref{int(round(float(reflection_threshold) * 100.0)):03d}"
    return cache_dir / f"{source_name}_{suffix}_{thr_tag}_{key}.pkl"


def _load_or_build_qualified_feature_store(
    *,
    source_name: str,
    cache_root: str,
    positions: tuple[float, ...],
    required_feature_names: set[str],
    max_problems_per_cache: Optional[int],
    reflection_threshold: float,
    max_workers: int,
    chunk_problems: int,
    feature_cache_dir: Optional[Path],
    refresh_feature_cache: bool,
) -> tuple[list[dict[str, Any]], Optional[Path], str]:
    cache_path: Optional[Path] = None
    if feature_cache_dir is not None:
        cache_path = _feature_cache_path(
            cache_dir=feature_cache_dir,
            source_name=source_name,
            cache_root=cache_root,
            positions=positions,
            required_feature_names=required_feature_names,
            max_problems_per_cache=max_problems_per_cache,
            reflection_threshold=reflection_threshold,
        )
        if cache_path.exists() and not refresh_feature_cache:
            print(f"[strongfeat] loading feature cache source={source_name} thr={reflection_threshold:.2f} path={cache_path}")
            with cache_path.open("rb") as handle:
                payload = pickle.load(handle)
            return list(payload["feature_store"]), cache_path, "loaded"

    print(f"[strongfeat] building feature store source={source_name} thr={reflection_threshold:.2f} root={cache_root}")
    store = _qualify_feature_store(
        build_feature_store(
            cache_root=cache_root,
            positions=positions,
            required_feature_names=required_feature_names,
            max_problems_per_cache=max_problems_per_cache,
            max_workers=max(1, int(max_workers)),
            chunk_problems=max(1, int(chunk_problems)),
            reflection_threshold=float(reflection_threshold),
        ),
        source_name=source_name,
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with tmp_path.open("wb") as handle:
            pickle.dump(
                {
                    "source_name": str(source_name),
                    "cache_root": str(cache_root),
                    "positions": [float(p) for p in positions],
                    "max_problems_per_cache": None if max_problems_per_cache is None else int(max_problems_per_cache),
                    "reflection_threshold": float(reflection_threshold),
                    "feature_store": store,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        tmp_path.replace(cache_path)
        print(f"[strongfeat] saved feature cache source={source_name} thr={reflection_threshold:.2f} path={cache_path}")
    return store, cache_path, "built"


def _filter_noncoding(feature_store: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [payload for payload in feature_store if payload["domain"] in {"math", "science"}]


def _candidate_method_name(bundle_type: str, reflection_threshold: float) -> str:
    thr_tag = f"ref{int(round(float(reflection_threshold) * 100.0)):03d}"
    return f"strongfeat_{bundle_type}_{thr_tag}"


def _fit_route_for_table_strong(
    *,
    x_raw: np.ndarray,
    x_rank: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    position: float,
    training_scope: str,
    reflection_threshold: float,
    n_splits: int,
    random_state: int,
) -> dict[str, Any]:
    if x_raw.shape[0] == 0 or np.unique(y).shape[0] < 2:
        return {
            "route_type": "baseline",
            "signal_name": "tok_conf_prefix",
            "cv_auroc": float("nan"),
            "n_valid_folds": 0,
            "training_position": float(position),
            "training_scope": str(training_scope),
            "reflection_threshold": float(reflection_threshold),
            "note": "insufficient labeled data",
        }

    baseline_candidates = [name for name in STRONG_BASELINE_SIGNAL_NAMES if name in FULL_FEATURE_NAMES]
    best_baseline = {
        "signal_name": "tok_conf_prefix",
        "cv_auroc": float("-inf"),
        "n_valid_folds": 0,
    }
    feature_to_idx = {name: idx for idx, name in enumerate(FULL_FEATURE_NAMES)}
    for signal_name in baseline_candidates:
        score_col = feature_to_idx[signal_name]
        score_vec = x_raw[:, score_col]
        cv_auc, n_folds = _cv_auroc_baseline(
            scores=score_vec,
            y=y,
            groups=groups,
            n_splits=n_splits,
        )
        if np.isfinite(cv_auc) and cv_auc > float(best_baseline["cv_auroc"]):
            best_baseline = {
                "signal_name": str(signal_name),
                "cv_auroc": float(cv_auc),
                "n_valid_folds": int(n_folds),
            }

    best_svd: dict[str, Any] = {"cv_auroc": float("-inf")}
    folds = _group_folds(groups, n_splits=n_splits)
    for family_name, family_features in STRONG_FEATURE_FAMILY_MAP.items():
        feature_indices = [feature_to_idx[name] for name in family_features]
        for representation in SEARCH_REPRESENTATIONS:
            x_rep = _build_representation(
                x_raw=x_raw,
                x_rank=x_rank,
                feature_indices=feature_indices,
                representation=representation,
            )
            if not folds:
                continue

            candidate_scores: dict[tuple[int, float, bool, str], list[float]] = {}
            for train_idx, test_idx in folds:
                y_train = y[train_idx]
                y_test = y[test_idx]
                if np.unique(y_train).shape[0] < 2 or np.unique(y_test).shape[0] < 2:
                    continue

                scaler = StandardScaler(with_mean=True, with_std=True)
                x_train_scaled = scaler.fit_transform(x_rep[train_idx])
                x_test_scaled = scaler.transform(x_rep[test_idx])

                max_rank = min(int(max(SEARCH_RANKS)), int(x_train_scaled.shape[1]), int(x_train_scaled.shape[0] - 1))
                if max_rank < 1:
                    continue

                svd = TruncatedSVD(n_components=max_rank, random_state=int(random_state))
                z_train_full = svd.fit_transform(x_train_scaled)
                z_test_full = svd.transform(x_test_scaled)
                singular_values = np.asarray(svd.singular_values_, dtype=np.float64)
                singular_values = np.where(np.abs(singular_values) < 1e-8, 1.0, singular_values)

                valid_ranks = [int(rank) for rank in SEARCH_RANKS if int(rank) <= max_rank]
                for rank in valid_ranks:
                    z_train = z_train_full[:, :rank]
                    z_test = z_test_full[:, :rank]
                    for whiten in SEARCH_WHITEN:
                        if whiten:
                            scale = singular_values[:rank]
                            z_train_use = z_train / scale
                            z_test_use = z_test / scale
                        else:
                            z_train_use = z_train
                            z_test_use = z_test
                        for c_value in SEARCH_C_VALUES:
                            for class_weight in SEARCH_CLASS_WEIGHT:
                                clf = LogisticRegression(
                                    C=float(c_value),
                                    class_weight=None if class_weight == "none" else "balanced",
                                    max_iter=2000,
                                    random_state=int(random_state),
                                )
                                try:
                                    clf.fit(z_train_use, y_train)
                                    scores = np.asarray(clf.decision_function(z_test_use), dtype=np.float64)
                                except Exception:
                                    continue
                                fold_auc = _auroc(scores, y_test)
                                if np.isfinite(fold_auc):
                                    candidate_scores.setdefault(
                                        (int(rank), float(c_value), bool(whiten), str(class_weight)),
                                        [],
                                    ).append(float(fold_auc))

            for (rank, c_value, whiten, class_weight), values in candidate_scores.items():
                if not values:
                    continue
                cv_auc = float(np.mean(values))
                n_folds = int(len(values))
                if cv_auc > float(best_svd["cv_auroc"]):
                    best_svd = {
                        "cv_auroc": float(cv_auc),
                        "n_valid_folds": int(n_folds),
                        "family_name": str(family_name),
                        "representation": str(representation),
                        "rank": int(rank),
                        "c_value": float(c_value),
                        "whiten": bool(whiten),
                        "class_weight": str(class_weight),
                        "feature_names": list(family_features),
                        "feature_indices": list(feature_indices),
                    }

    baseline_auc = float(best_baseline["cv_auroc"]) if np.isfinite(best_baseline["cv_auroc"]) else float("-inf")
    svd_auc = float(best_svd["cv_auroc"]) if np.isfinite(best_svd["cv_auroc"]) else float("-inf")

    if svd_auc > baseline_auc and np.isfinite(svd_auc):
        x_rep_full = _build_representation(
            x_raw=x_raw,
            x_rank=x_rank,
            feature_indices=best_svd["feature_indices"],
            representation=best_svd["representation"],
        )
        model = _fit_svd_lr_model(
            x=x_rep_full,
            y=y,
            rank=best_svd["rank"],
            c_value=best_svd["c_value"],
            whiten=best_svd["whiten"],
            class_weight_name=best_svd["class_weight"],
            random_state=random_state,
        )
        if model is not None:
            return {
                "route_type": "svd",
                "cv_auroc": float(best_svd["cv_auroc"]),
                "n_valid_folds": int(best_svd["n_valid_folds"]),
                "family_name": str(best_svd["family_name"]),
                "representation": str(best_svd["representation"]),
                "rank": int(best_svd["rank"]),
                "c_value": float(best_svd["c_value"]),
                "whiten": bool(best_svd["whiten"]),
                "class_weight": str(best_svd["class_weight"]),
                "feature_names": list(best_svd["feature_names"]),
                "feature_indices": list(best_svd["feature_indices"]),
                "baseline_signal_name": best_baseline["signal_name"],
                "baseline_cv_auroc": float(best_baseline["cv_auroc"]),
                "training_position": float(position),
                "training_scope": str(training_scope),
                "reflection_threshold": float(reflection_threshold),
                "model": model,
            }

    return {
        "route_type": "baseline",
        "signal_name": best_baseline["signal_name"],
        "cv_auroc": float(best_baseline["cv_auroc"]),
        "n_valid_folds": int(best_baseline["n_valid_folds"]),
        "svd_best_cv_auroc": None if not np.isfinite(svd_auc) else float(svd_auc),
        "training_position": float(position),
        "training_scope": str(training_scope),
        "reflection_threshold": float(reflection_threshold),
    }


def _train_routes_for_scope_strong(
    *,
    tables: list[dict[str, np.ndarray]],
    positions: tuple[float, ...],
    scope_name: str,
    reflection_threshold: float,
    n_splits: int,
    random_state: int,
    max_workers: int,
) -> dict[float, dict[str, Any]]:
    routes: dict[float, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
        future_map = {}
        for pos_idx, position in enumerate(positions):
            future = executor.submit(
                _fit_route_for_table_strong,
                x_raw=tables[pos_idx]["x_raw"],
                x_rank=tables[pos_idx]["x_rank"],
                y=tables[pos_idx]["y"],
                groups=tables[pos_idx]["groups"],
                position=float(position),
                training_scope=str(scope_name),
                reflection_threshold=float(reflection_threshold),
                n_splits=int(n_splits),
                random_state=int(random_state),
            )
            future_map[future] = float(position)

        for future in as_completed(future_map):
            position = future_map[future]
            route = future.result()
            routes[float(position)] = route
            if route["route_type"] == "svd":
                print(
                    f"[strongfeat] scope={scope_name:<18s} pos={_pct_label(position):>4s} "
                    f"thr={reflection_threshold:.2f} route=svd auc={route['cv_auroc']:.4f} "
                    f"family={route['family_name']} rep={route['representation']}"
                )
            else:
                print(
                    f"[strongfeat] scope={scope_name:<18s} pos={_pct_label(position):>4s} "
                    f"thr={reflection_threshold:.2f} route=baseline auc={route['cv_auroc']:.4f} "
                    f"signal={route['signal_name']}"
                )
    return routes


def _annotate_bundle_thresholds(
    bundle: dict[str, Any],
    *,
    noncoding_threshold: float,
    coding_threshold: Optional[float] = None,
) -> dict[str, Any]:
    out = copy.deepcopy(bundle)
    for domain_name, domain_bundle in out["domains"].items():
        if domain_name in {"math", "science"}:
            threshold = float(noncoding_threshold)
        elif coding_threshold is not None:
            threshold = float(coding_threshold)
        else:
            threshold = float(noncoding_threshold)
        for route in domain_bundle["routes"]:
            route["reflection_threshold"] = float(threshold)
    out["reflection_threshold_by_domain"] = {
        "math": float(noncoding_threshold),
        "science": float(noncoding_threshold),
        "coding": float(coding_threshold if coding_threshold is not None else noncoding_threshold),
    }
    return out


def _decision(
    *,
    best_new: dict[str, Any],
    baseline_v1: dict[str, Any],
    baseline_cap8: dict[str, Any],
    baseline_tok_conf: dict[str, Any],
    allow_export: bool,
) -> dict[str, Any]:
    cand = best_new["aggregate"]
    v1 = baseline_v1["aggregate"]
    cap8 = baseline_cap8["aggregate"]
    tok_conf = baseline_tok_conf["aggregate"]

    beats_v1 = (
        float(cand["auc_of_selacc"]) > float(v1["auc_of_selacc"])
        and _metric_ge(float(cand["auroc@100%"]), float(v1["auroc@100%"]))
        and float(cand["stop_acc@100%"]) >= float(v1["stop_acc@100%"])
    )
    beats_cap8 = float(cand["auc_of_selacc"]) > float(cap8["auc_of_selacc"])
    beats_tok_conf = float(cand["auc_of_selacc"]) > float(tok_conf["auc_of_selacc"])
    stop_guard = float(cand["stop_acc@100%"]) >= max(float(cap8["stop_acc@100%"]), float(tok_conf["stop_acc@100%"]))
    export_recommended = bool(beats_v1 and beats_cap8 and beats_tok_conf and stop_guard and allow_export)

    failures: list[str] = []
    if not beats_v1:
        failures.append("未稳定超过 `earlystop_svd_lowrank_lr_v1`")
    if not beats_cap8:
        failures.append("AUC of SelAcc 未超过 `earlystop_prefix10_svd_round1b_cap8`")
    if not beats_tok_conf:
        failures.append("AUC of SelAcc 未超过 `tok_conf_prefix_mean_v1`")
    if not stop_guard:
        failures.append("Stop Acc@100% 未通过保守 guardrail")
    reason = "；".join(failures) if failures else "满足强特征线保守导出 gate"
    if not allow_export and not failures:
        reason += "；本轮未开启自动导出"

    return {
        "winner_method_name": str(best_new["method_name"]),
        "holdout_beats_svd_v1": bool(beats_v1),
        "holdout_beats_round1b_cap8": bool(beats_cap8),
        "holdout_beats_tok_conf": bool(beats_tok_conf),
        "export_recommended": bool(export_recommended),
        "reason": str(reason),
    }


def _route_summary_lines(title: str, route_map: dict[float, dict[str, Any]]) -> list[str]:
    lines = [f"### {title}", "", "| Anchor | Route | Detail |", "|---|---|---|"]
    for position in ANCHOR_POSITIONS:
        route = route_map[float(position)]
        if route["route_type"] == "svd":
            detail = (
                f"{route['family_name']} / {route['representation']} / rank={route['rank']} / "
                f"C={route['c_value']} / whiten={route['whiten']} / thr={route['reflection_threshold']:.2f}"
            )
        else:
            detail = f"{route['signal_name']} / thr={route['reflection_threshold']:.2f}"
        lines.append(f"| {_pct_label(position)} | {route['route_type']} | {detail} |")
    lines.append("")
    return lines


def _write_results_doc(path: Path, summary: dict[str, Any], changed_files: list[str]) -> None:
    holdout_rows = list(summary["holdout_eval"]["candidates"].values())
    train_rows = list(summary["train_eval"]["candidates"].values())
    winner_name = summary["decision"]["winner_method_name"]
    winner_meta = summary["winner_meta"]
    winner_holdout = summary["holdout_eval"]["candidates"][winner_name]

    lines: list[str] = [
        "# EARLYSTOP STRONG FEATURES ROUND1 (2026-04-09)",
        "",
        "## 开头确认",
        "",
    ]
    for line in REPO_STATUS_LINES:
        lines.append(f"- {line}")

    lines.extend([
        "",
        "## 1. 特征收窄原则",
        "",
        "- 参考 `docs/reference/FEATURES.md`、`results/selector_comparison/selector_comparison.md` 与 `README.md` 中仓库已记录的 strongest-feature 证据。",
        "- 本轮不再沿用宽特征 `all` 搜索，而是只保留 `tok_conf_prefix` / `tok_conf_recency` / `traj_reflection_count` 及少量 tail / recovery 信号。",
        "- 保留的强特征族：`strong_core3`、`strong_tail5`、`strong_stable6`、`strong_event7`、`strong_recovery8`。",
        f"- 反思阈值小搜索：{', '.join(f'`{thr:.2f}`' for thr in summary['config']['reflection_thresholds'])}。",
        "",
        "## 2. 训练 / holdout 协议",
        "",
        f"- `main root`：`{summary['protocol']['main_cache_root']}`。",
        f"- `extra root`：`{summary['protocol']['extra_cache_root']}`。",
        f"- `holdout split`：`{int(round((1.0 - float(summary['protocol']['holdout_split'])) * 100))}/{int(round(float(summary['protocol']['holdout_split']) * 100))}`，按 `dataset + problem_id` 跨 root 一致切分。",
        f"- `max_problems_per_cache`：`{summary['config']['max_problems_per_cache']}`。",
        f"- `train(non-coding)`：`{summary['protocol']['train_store']['total_problems']}` problem-slices / `{summary['protocol']['train_store']['total_samples']}` samples。",
        f"- `holdout(non-coding)`：`{summary['protocol']['holdout_store']['total_problems']}` problem-slices / `{summary['protocol']['holdout_store']['total_samples']}` samples。",
        "",
        "## 3. holdout 对比",
        "",
    ])

    holdout_baselines = [
        summary["holdout_eval"]["baselines"]["tok_conf_prefix_mean_v1"],
        summary["holdout_eval"]["baselines"]["earlystop_from_bestofn_svm_bridge_v1"],
        summary["holdout_eval"]["baselines"]["earlystop_svd_lowrank_lr_v1"],
        summary["holdout_eval"]["baselines"]["earlystop_prefix10_svd_round1b_cap8"],
    ]
    lines.extend(_render_method_table("holdout baselines", holdout_baselines))
    lines.extend(_render_method_table("holdout strong-feature candidates", holdout_rows))
    lines.extend(_render_cache_table("holdout winner per-cache", winner_holdout))

    lines.extend([
        "## 4. train-side 诊断（non-coding only）",
        "",
    ])
    train_baselines = [
        summary["train_eval"]["baselines"]["tok_conf_prefix_mean_v1"],
        summary["train_eval"]["baselines"]["earlystop_from_bestofn_svm_bridge_v1"],
        summary["train_eval"]["baselines"]["earlystop_svd_lowrank_lr_v1"],
        summary["train_eval"]["baselines"]["earlystop_prefix10_svd_round1b_cap8"],
    ]
    lines.extend(_render_method_table("train baselines", train_baselines))
    lines.extend(_render_method_table("train strong-feature candidates", train_rows))

    lines.extend([
        "## 5. winner route summary",
        "",
        f"- `winner`：`{winner_name}`。",
        f"- `bundle type`：`{winner_meta['bundle_type']}`。",
        f"- `reflection threshold`：`{winner_meta['reflection_threshold']:.2f}`。",
        "",
    ])
    lines.extend(_route_summary_lines("split-fit winner anchors", summary["splitfit_route_summary"][winner_name]))
    lines.extend(_route_summary_lines("full-fit winner anchors", summary["fullfit_route_summary"][winner_name]))

    lines.extend([
        "## 6. 结论",
        "",
        f"- `holdout 是否超过 old SVD v1`：`{'YES' if summary['decision']['holdout_beats_svd_v1'] else 'NO'}`。",
        f"- `holdout 是否超过 round1b cap8`：`{'YES' if summary['decision']['holdout_beats_round1b_cap8'] else 'NO'}`。",
        f"- `holdout 是否超过 tok_conf baseline`：`{'YES' if summary['decision']['holdout_beats_tok_conf'] else 'NO'}`。",
        f"- `是否建议导出 blind submission`：`{'YES' if summary['decision']['export_recommended'] else 'NO'}`。",
        f"- `理由`：{summary['decision']['reason']}。",
        "",
        "## 7. 改了哪些文件",
        "",
    ])
    for file_path in changed_files:
        lines.append(f"- `{file_path}`")

    lines.extend([
        "",
        "## 8. 如何复跑",
        "",
        "```bash",
        "bash cookbook/00_setup/verify.sh",
        "python3 scripts/run_earlystop_strongfeat_round1.py \\",
        "  --main-cache-root MUI_HUB/cache \\",
        "  --extra-cache-root /home/jovyan/public-ro/MUI_HUB/cache_train \\",
        "  --max-problems-per-cache 8",
        "```",
        "",
        "```bash",
        "python3 scripts/run_earlystop_strongfeat_round1.py \\",
        "  --main-cache-root MUI_HUB/cache \\",
        "  --extra-cache-root /home/jovyan/public-ro/MUI_HUB/cache_train \\",
        "  --max-problems-per-cache 0",
        "```",
        "",
        "### full-fit candidate",
        "",
        f"- 保存路径：`{summary['fullfit']['saved_model']}`。",
        f"- 采用方法：`{summary['fullfit']['winner_method_name']}`。",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run EarlyStop strong-feature SVD round1")
    ap.add_argument("--main-cache-root", default="MUI_HUB/cache")
    ap.add_argument("--extra-cache-root", default="/home/jovyan/public-ro/MUI_HUB/cache_train")
    ap.add_argument("--out-model", default="models/ml_selectors/earlystop_strongfeat_round1.pkl")
    ap.add_argument("--out-summary", default="results/scans/earlystop/earlystop_strongfeat_round1_summary.json")
    ap.add_argument("--out-eval", default="results/scans/earlystop/earlystop_strongfeat_round1_eval.json")
    ap.add_argument("--out-doc", default="docs/EARLYSTOP_STRONG_FEATURES_ROUND1_20260409.md")
    ap.add_argument("--holdout-split", type=float, default=0.20)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--feature-chunk-problems", type=int, default=DEFAULT_FEATURE_CHUNK_PROBLEMS)
    ap.add_argument("--feature-cache-dir", default="results/cache/earlystop_strongfeat_round1")
    ap.add_argument("--refresh-feature-cache", action="store_true")
    ap.add_argument("--max-problems-per-cache", type=int, default=8, help="0 means all problems")
    ap.add_argument("--export-if-holdout-win", action="store_true")
    ap.add_argument("--export-cache-root", default="/home/jovyan/public-ro/MUI_HUB/cache_test")
    ap.add_argument("--export-method-name", default="earlystop_strongfeat_round1")
    ap.add_argument("--export-filename", default="earlystop_strongfeat_round1.json")
    args = ap.parse_args()

    main_cache_root = str((REPO_ROOT / args.main_cache_root).resolve()) if not Path(args.main_cache_root).is_absolute() else str(Path(args.main_cache_root).resolve())
    extra_cache_root = str(Path(args.extra_cache_root).resolve())
    feature_cache_dir = None if str(args.feature_cache_dir).strip().lower() in {"", "none", "off"} else (REPO_ROOT / str(args.feature_cache_dir)).resolve()
    max_problems_per_cache = None if int(args.max_problems_per_cache) <= 0 else int(args.max_problems_per_cache)

    v1_bundle = load_earlystop_svd_bundle(REPO_ROOT / "models/ml_selectors/earlystop_svd_lowrank_lr_v1.pkl")
    cap8_bundle = load_earlystop_svd_bundle(REPO_ROOT / "models/ml_selectors/earlystop_prefix10_svd_round1b_cap8.pkl")
    bridge_bundle = load_earlystop_svm_bundle(REPO_ROOT / "models/ml_selectors/bestofn_svm_bridge_v1.pkl")

    required_features = collect_required_features_for_eval(
        v1_bundle=v1_bundle,
        bridge_bundle=bridge_bundle,
    )
    for features in STRONG_FEATURE_FAMILY_MAP.values():
        required_features.update(features)

    stores_by_threshold: dict[float, dict[str, Any]] = {}
    holdout_problem_map: Optional[dict[str, set[str]]] = None
    holdout_problem_summary: dict[str, Any] = {}

    for reflection_threshold in SEARCH_REFLECTION_THRESHOLDS:
        main_store, main_cache_path, main_cache_status = _load_or_build_qualified_feature_store(
            source_name="cache",
            cache_root=main_cache_root,
            positions=EXTRACTION_POSITIONS,
            required_feature_names=required_features,
            max_problems_per_cache=max_problems_per_cache,
            reflection_threshold=float(reflection_threshold),
            max_workers=int(args.workers),
            chunk_problems=int(args.feature_chunk_problems),
            feature_cache_dir=feature_cache_dir,
            refresh_feature_cache=bool(args.refresh_feature_cache),
        )
        extra_store, extra_cache_path, extra_cache_status = _load_or_build_qualified_feature_store(
            source_name="cache_train",
            cache_root=extra_cache_root,
            positions=EXTRACTION_POSITIONS,
            required_feature_names=required_features,
            max_problems_per_cache=max_problems_per_cache,
            reflection_threshold=float(reflection_threshold),
            max_workers=int(args.workers),
            chunk_problems=int(args.feature_chunk_problems),
            feature_cache_dir=feature_cache_dir,
            refresh_feature_cache=bool(args.refresh_feature_cache),
        )
        combined_store = main_store + extra_store
        if holdout_problem_map is None:
            holdout_problem_map, holdout_problem_summary = _build_holdout_problem_map(
                combined_store,
                holdout_split=float(args.holdout_split),
                split_seed=int(args.split_seed),
            )
        train_store, holdout_store, full_store = _split_feature_store(
            combined_store,
            holdout_problem_map=holdout_problem_map,
        )
        stores_by_threshold[float(reflection_threshold)] = {
            "train_store": _filter_noncoding(train_store),
            "holdout_store": holdout_store,
            "full_store": _filter_noncoding(full_store),
            "main_cache_status": str(main_cache_status),
            "extra_cache_status": str(extra_cache_status),
            "main_cache_path": None if main_cache_path is None else str(main_cache_path),
            "extra_cache_path": None if extra_cache_path is None else str(extra_cache_path),
        }

    baseline_threshold = 0.30
    baseline_store = stores_by_threshold[baseline_threshold]

    holdout_baselines = {
        "tok_conf_prefix_mean_v1": evaluate_method_from_feature_store(
            method_name="tok_conf_prefix_mean_v1",
            feature_store=baseline_store["holdout_store"],
            position_values=tuple(float(p) for p in EARLY_STOP_POSITIONS),
            score_fn=make_tok_conf_score_fn(),
        ),
        "earlystop_from_bestofn_svm_bridge_v1": evaluate_method_from_feature_store(
            method_name="earlystop_from_bestofn_svm_bridge_v1",
            feature_store=baseline_store["holdout_store"],
            position_values=tuple(float(p) for p in EARLY_STOP_POSITIONS),
            score_fn=make_bridge_score_fn(bridge_bundle),
        ),
        "earlystop_svd_lowrank_lr_v1": evaluate_method_from_feature_store(
            method_name="earlystop_svd_lowrank_lr_v1",
            feature_store=baseline_store["holdout_store"],
            position_values=tuple(float(p) for p in EARLY_STOP_POSITIONS),
            score_fn=make_svd_bundle_score_fn(v1_bundle),
        ),
        "earlystop_prefix10_svd_round1b_cap8": evaluate_method_from_feature_store(
            method_name="earlystop_prefix10_svd_round1b_cap8",
            feature_store=baseline_store["holdout_store"],
            position_values=tuple(float(p) for p in EARLY_STOP_POSITIONS),
            score_fn=make_svd_bundle_score_fn(cap8_bundle),
        ),
    }

    train_baselines = {
        "tok_conf_prefix_mean_v1": evaluate_method_from_feature_store(
            method_name="tok_conf_prefix_mean_v1",
            feature_store=baseline_store["train_store"],
            position_values=tuple(float(p) for p in EARLY_STOP_POSITIONS),
            score_fn=make_tok_conf_score_fn(),
        ),
        "earlystop_from_bestofn_svm_bridge_v1": evaluate_method_from_feature_store(
            method_name="earlystop_from_bestofn_svm_bridge_v1",
            feature_store=baseline_store["train_store"],
            position_values=tuple(float(p) for p in EARLY_STOP_POSITIONS),
            score_fn=make_bridge_score_fn(bridge_bundle),
        ),
        "earlystop_svd_lowrank_lr_v1": evaluate_method_from_feature_store(
            method_name="earlystop_svd_lowrank_lr_v1",
            feature_store=baseline_store["train_store"],
            position_values=tuple(float(p) for p in EARLY_STOP_POSITIONS),
            score_fn=make_svd_bundle_score_fn(v1_bundle),
        ),
        "earlystop_prefix10_svd_round1b_cap8": evaluate_method_from_feature_store(
            method_name="earlystop_prefix10_svd_round1b_cap8",
            feature_store=baseline_store["train_store"],
            position_values=tuple(float(p) for p in EARLY_STOP_POSITIONS),
            score_fn=make_svd_bundle_score_fn(cap8_bundle),
        ),
    }

    holdout_candidates: dict[str, dict[str, Any]] = {}
    train_candidates: dict[str, dict[str, Any]] = {}
    candidate_meta: dict[str, dict[str, Any]] = {}
    splitfit_route_summary: dict[str, dict[float, dict[str, Any]]] = {}

    for reflection_threshold in SEARCH_REFLECTION_THRESHOLDS:
        store_pack = stores_by_threshold[float(reflection_threshold)]
        train_tables = _build_scope_training_tables_with_dual_groups(
            store_pack["train_store"],
            positions=ANCHOR_POSITIONS,
        )

        splitfit_global_routes = _train_routes_for_scope_strong(
            tables=[train_tables["global"][idx] for idx in range(len(ANCHOR_POSITIONS))],
            positions=ANCHOR_POSITIONS,
            scope_name=f"global_ref{int(round(reflection_threshold * 100)):03d}",
            reflection_threshold=float(reflection_threshold),
            n_splits=int(args.n_splits),
            random_state=int(args.random_state),
            max_workers=int(args.workers),
        )
        splitfit_noncoding_routes = _train_routes_for_scope_strong(
            tables=[train_tables["noncoding"][idx] for idx in range(len(ANCHOR_POSITIONS))],
            positions=ANCHOR_POSITIONS,
            scope_name=f"noncoding_ref{int(round(reflection_threshold * 100)):03d}",
            reflection_threshold=float(reflection_threshold),
            n_splits=int(args.n_splits),
            random_state=int(args.random_state),
            max_workers=int(args.workers),
        )

        global_bundle = _annotate_bundle_thresholds(
            build_anchor_bundle(
                bundle_version=f"earlystop_strongfeat_round1_global_ref{int(round(reflection_threshold * 100)):03d}_splitfit",
                anchor_routes=splitfit_global_routes,
            ),
            noncoding_threshold=float(reflection_threshold),
        )
        noncoding_bundle = _annotate_bundle_thresholds(
            build_anchor_bundle(
                bundle_version=f"earlystop_strongfeat_round1_noncoding_ref{int(round(reflection_threshold * 100)):03d}_splitfit",
                anchor_routes=splitfit_noncoding_routes,
                coding_routes=copy.deepcopy(v1_bundle["domains"]["coding"]["routes"]),
            ),
            noncoding_threshold=float(reflection_threshold),
            coding_threshold=float(DEFAULT_REFLECTION_THRESHOLD),
        )

        for bundle_type, bundle, route_map in (
            ("global_anchor4", global_bundle, splitfit_global_routes),
            ("noncoding_anchor4_coding_v1", noncoding_bundle, splitfit_noncoding_routes),
        ):
            method_name = _candidate_method_name(bundle_type, float(reflection_threshold))
            holdout_eval = evaluate_method_from_feature_store(
                method_name=method_name,
                feature_store=store_pack["holdout_store"],
                position_values=tuple(float(p) for p in EARLY_STOP_POSITIONS),
                score_fn=make_svd_bundle_score_fn(bundle),
            )
            train_eval = evaluate_method_from_feature_store(
                method_name=method_name,
                feature_store=store_pack["train_store"],
                position_values=tuple(float(p) for p in EARLY_STOP_POSITIONS),
                score_fn=make_svd_bundle_score_fn(bundle),
            )
            holdout_candidates[method_name] = holdout_eval
            train_candidates[method_name] = train_eval
            candidate_meta[method_name] = {
                "bundle_type": str(bundle_type),
                "reflection_threshold": float(reflection_threshold),
                "splitfit_bundle": bundle,
            }
            splitfit_route_summary[method_name] = {
                float(position): summarise_route(route)
                for position, route in route_map.items()
            }

    holdout_best = choose_best_candidate(list(holdout_candidates.values()))
    decision = _decision(
        best_new=holdout_best,
        baseline_v1=holdout_baselines["earlystop_svd_lowrank_lr_v1"],
        baseline_cap8=holdout_baselines["earlystop_prefix10_svd_round1b_cap8"],
        baseline_tok_conf=holdout_baselines["tok_conf_prefix_mean_v1"],
        allow_export=bool(args.export_if_holdout_win),
    )

    winner_name = str(decision["winner_method_name"])
    winner_threshold = float(candidate_meta[winner_name]["reflection_threshold"])
    winner_bundle_type = str(candidate_meta[winner_name]["bundle_type"])
    winner_store = stores_by_threshold[winner_threshold]
    full_tables = _build_scope_training_tables_with_dual_groups(
        winner_store["full_store"],
        positions=ANCHOR_POSITIONS,
    )

    fullfit_global_routes = _train_routes_for_scope_strong(
        tables=[full_tables["global"][idx] for idx in range(len(ANCHOR_POSITIONS))],
        positions=ANCHOR_POSITIONS,
        scope_name=f"global_fullfit_ref{int(round(winner_threshold * 100)):03d}",
        reflection_threshold=winner_threshold,
        n_splits=int(args.n_splits),
        random_state=int(args.random_state),
        max_workers=int(args.workers),
    )
    fullfit_noncoding_routes = _train_routes_for_scope_strong(
        tables=[full_tables["noncoding"][idx] for idx in range(len(ANCHOR_POSITIONS))],
        positions=ANCHOR_POSITIONS,
        scope_name=f"noncoding_fullfit_ref{int(round(winner_threshold * 100)):03d}",
        reflection_threshold=winner_threshold,
        n_splits=int(args.n_splits),
        random_state=int(args.random_state),
        max_workers=int(args.workers),
    )

    fullfit_global_bundle = _annotate_bundle_thresholds(
        build_anchor_bundle(
            bundle_version=f"earlystop_strongfeat_round1_global_ref{int(round(winner_threshold * 100)):03d}_fullfit",
            anchor_routes=fullfit_global_routes,
        ),
        noncoding_threshold=winner_threshold,
    )
    fullfit_noncoding_bundle = _annotate_bundle_thresholds(
        build_anchor_bundle(
            bundle_version=f"earlystop_strongfeat_round1_noncoding_ref{int(round(winner_threshold * 100)):03d}_fullfit",
            anchor_routes=fullfit_noncoding_routes,
            coding_routes=copy.deepcopy(v1_bundle["domains"]["coding"]["routes"]),
        ),
        noncoding_threshold=winner_threshold,
        coding_threshold=float(DEFAULT_REFLECTION_THRESHOLD),
    )

    best_bundle = fullfit_global_bundle if winner_bundle_type == "global_anchor4" else fullfit_noncoding_bundle
    out_model = REPO_ROOT / args.out_model
    save_earlystop_svd_bundle(best_bundle, out_model)

    fullfit_route_summary = {
        _candidate_method_name("global_anchor4", winner_threshold): {
            float(position): summarise_route(route)
            for position, route in fullfit_global_routes.items()
        },
        _candidate_method_name("noncoding_anchor4_coding_v1", winner_threshold): {
            float(position): summarise_route(route)
            for position, route in fullfit_noncoding_routes.items()
        },
    }

    summary = {
        "created_at_utc": _now_utc(),
        "repo_status": REPO_STATUS_LINES,
        "protocol": {
            "main_cache_root": str(main_cache_root),
            "extra_cache_root": str(extra_cache_root),
            "feature_cache_dir": None if feature_cache_dir is None else str(feature_cache_dir),
            "holdout_split": float(args.holdout_split),
            "split_seed": int(args.split_seed),
            "train_store": _summarise_feature_store(baseline_store["train_store"]),
            "holdout_store": _summarise_feature_store(baseline_store["holdout_store"]),
            "winner_full_store": _summarise_feature_store(winner_store["full_store"]),
            "holdout_problem_summary": holdout_problem_summary,
            "feature_cache_status_by_threshold": {
                f"{thr:.2f}": {
                    "cache": stores_by_threshold[thr]["main_cache_status"],
                    "cache_train": stores_by_threshold[thr]["extra_cache_status"],
                    "cache_path": stores_by_threshold[thr]["main_cache_path"],
                    "cache_train_path": stores_by_threshold[thr]["extra_cache_path"],
                }
                for thr in SEARCH_REFLECTION_THRESHOLDS
            },
        },
        "config": {
            "anchor_positions": list(ANCHOR_POSITIONS),
            "official_slot_to_anchor": {str(k): float(v) for k, v in OFFICIAL_SLOT_TO_ANCHOR.items()},
            "reflection_thresholds": list(SEARCH_REFLECTION_THRESHOLDS),
            "family_map": STRONG_FEATURE_FAMILY_MAP,
            "baseline_signal_names": list(STRONG_BASELINE_SIGNAL_NAMES),
            "representations": list(SEARCH_REPRESENTATIONS),
            "svd_dims": list(SEARCH_RANKS),
            "c_values": list(SEARCH_C_VALUES),
            "class_weight": list(SEARCH_CLASS_WEIGHT),
            "whiten": [bool(v) for v in SEARCH_WHITEN],
            "workers": int(args.workers),
            "feature_chunk_problems": int(args.feature_chunk_problems),
            "max_problems_per_cache": 0 if max_problems_per_cache is None else int(max_problems_per_cache),
        },
        "train_eval": {
            "baselines": train_baselines,
            "candidates": train_candidates,
        },
        "holdout_eval": {
            "baselines": holdout_baselines,
            "candidates": holdout_candidates,
        },
        "splitfit_route_summary": splitfit_route_summary,
        "fullfit_route_summary": fullfit_route_summary,
        "winner_meta": {
            "bundle_type": winner_bundle_type,
            "reflection_threshold": winner_threshold,
        },
        "fullfit": {
            "winner_method_name": winner_name,
            "saved_model": _display_path(out_model),
            "bundle_version": str(best_bundle["bundle_version"]),
        },
        "decision": decision,
    }

    out_summary = REPO_ROOT / args.out_summary
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    out_eval = REPO_ROOT / args.out_eval
    out_eval.parent.mkdir(parents=True, exist_ok=True)
    out_eval.write_text(
        json.dumps(
            {
                "train_eval": summary["train_eval"],
                "holdout_eval": summary["holdout_eval"],
                "winner_meta": summary["winner_meta"],
                "decision": summary["decision"],
                "fullfit": summary["fullfit"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    changed_files = [
        "nad/ops/earlystop_svd.py",
        "scripts/run_earlystop_prefix10_svd_round1.py",
        "scripts/run_earlystop_strongfeat_round1.py",
        str(args.out_doc),
    ]
    _write_results_doc(REPO_ROOT / args.out_doc, summary, changed_files)

    if summary["decision"]["export_recommended"]:
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/export_earlystop_svd_submission.py"),
                "--cache-root",
                str(Path(args.export_cache_root).resolve()),
                "--model-path",
                str(out_model),
                "--method-name",
                str(args.export_method_name),
                "--filename",
                str(args.export_filename),
            ],
            check=True,
        )

    print("\nStrong-feature training finished")
    print(f"  Winner     : {winner_name}")
    print(f"  Model      : {out_model}")
    print(f"  Summary    : {out_summary}")
    print(f"  Eval       : {out_eval}")
    print(f"  Doc        : {REPO_ROOT / args.out_doc}")


if __name__ == "__main__":
    main()
