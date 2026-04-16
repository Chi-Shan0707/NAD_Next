#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

for env_name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(env_name, "1")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.baselines import run_science_dense_slot_search as mod
from SVDomain.train_es_svd_ms_rr_r2 import (
    _build_domain_training_tables,
    _build_holdout_problem_map,
    _split_feature_store,
)
from nad.ops.earlystop_svd import load_earlystop_svd_bundle


FAMILY_NAMES = ("fixed22", "science_front11", "science_front13", "wide46")
LINEAR_MODEL_NAMES = ("lr_l2", "lr_l1")
LINEAR_C_VALUES = (0.01, 0.10, 1.0)
LINEAR_CLASS_WEIGHT = ("none", "balanced")
ELASTICNET_L1_RATIOS = (0.2, 0.5, 0.8)
SEARCH_SEEDS = (42,)
FULL_ENSEMBLE_SEEDS = (42, 101, 202)


def _candidate_id(candidate: dict[str, Any]) -> str:
    parts = [
        f"src{int(candidate['source_anchor_pct']):03d}",
        str(candidate["route_family"]),
        str(candidate["model_name"]),
        str(candidate["family_name"]),
        str(candidate["representation"]).replace("+", "p"),
    ]
    if "c_value" in candidate:
        parts.append(f"c{str(candidate['c_value']).replace('.', 'p')}")
    if "class_weight" in candidate:
        parts.append(str(candidate["class_weight"]))
    if candidate.get("l1_ratio") is not None:
        parts.append(f"l1r{str(candidate['l1_ratio']).replace('.', 'p')}")
    if "config_tag" in candidate:
        parts.append(str(candidate["config_tag"]))
    return "__".join(parts)


def _full_seed_values(candidate: dict[str, Any]) -> list[int]:
    if str(candidate["route_family"]) == "xgboost":
        return [int(v) for v in FULL_ENSEMBLE_SEEDS]
    if str(candidate["model_name"]) in {"lr_l1", "elasticnet_lr"}:
        return [int(v) for v in FULL_ENSEMBLE_SEEDS]
    return [42]


def _linear_l1_ratios(model_name: str) -> tuple[Optional[float], ...]:
    if str(model_name) == "elasticnet_lr":
        return tuple(float(v) for v in ELASTICNET_L1_RATIOS)
    return (None,)


def _evaluate_candidate_rows(
    *,
    candidate: dict[str, Any],
    route: dict[str, Any],
    holdout_tables: list[dict[str, np.ndarray]],
    target_anchor_pcts: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target_anchor_pct in target_anchor_pcts:
        target_idx = mod.POSITION_TO_INDEX[float(int(target_anchor_pct)) / 100.0]
        table = holdout_tables[target_idx]
        x_raw = np.asarray(table["x_raw"], dtype=np.float64)
        y = np.asarray(table["y"], dtype=np.int32)
        groups = np.asarray(table["groups"], dtype=object)
        scores = mod._score_route(route, x_raw)
        rows.append(
            {
                "candidate_id": str(candidate["candidate_id"]),
                "target_anchor_pct": int(target_anchor_pct),
                "source_anchor_pct": int(candidate["source_anchor_pct"]),
                "route_family": str(candidate["route_family"]),
                "model_name": str(candidate["model_name"]),
                "family_name": str(candidate["family_name"]),
                "representation": str(candidate["representation"]),
                "feature_count": int(candidate["feature_count"]),
                "cv_mean_auroc": float("nan"),
                "holdout_auroc": float(mod._auroc(scores, y)),
                "holdout_group_top1": float(mod._group_top1(scores, y, groups)),
                "is_task_specific": bool(int(candidate["source_anchor_pct"]) == int(target_anchor_pct)),
            }
        )
    return rows


def _aggregate_curve(xs: list[int], ys: list[float]) -> float:
    if not xs or not ys:
        return float("nan")
    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    if x_arr.size == 1:
        return float(y_arr[0])
    width = float(x_arr[-1] - x_arr[0])
    if width <= 0:
        return float(np.mean(y_arr))
    area = np.sum((x_arr[1:] - x_arr[:-1]) * (y_arr[1:] + y_arr[:-1]) * 0.5, dtype=np.float64)
    return float(area / width)


def _evaluate_score_fn_on_holdout(
    *,
    score_fn,
    holdout_tables: list[dict[str, np.ndarray]],
    target_anchor_pcts: tuple[int, ...],
) -> dict[str, Any]:
    position_rows: list[dict[str, Any]] = []
    for target_anchor_pct in target_anchor_pcts:
        target_idx = mod.POSITION_TO_INDEX[float(int(target_anchor_pct)) / 100.0]
        table = holdout_tables[target_idx]
        x_raw = np.asarray(table["x_raw"], dtype=np.float64)
        y = np.asarray(table["y"], dtype=np.int32)
        groups = np.asarray(table["groups"], dtype=object)
        scores = score_fn(mod.SCIENCE_DOMAIN, target_idx, x_raw)
        position_rows.append(
            {
                "target_anchor_pct": int(target_anchor_pct),
                "auroc": float(mod._auroc(scores, y)),
                "selacc": float(mod._group_top1(scores, y, groups)),
            }
        )
    anchor_pcts = [int(row["target_anchor_pct"]) for row in position_rows]
    aurocs = [float(row["auroc"]) for row in position_rows]
    selaccs = [float(row["selacc"]) for row in position_rows]
    final_anchor_pct = int(anchor_pcts[-1])
    return {
        "positions": position_rows,
        "aggregate": {
            "auc_of_auroc": _aggregate_curve(anchor_pcts, aurocs),
            "auc_of_selacc": _aggregate_curve(anchor_pcts, selaccs),
            f"auroc@{final_anchor_pct}%": float(aurocs[-1]),
            f"stop_acc@{final_anchor_pct}%": float(selaccs[-1]),
        },
    }


def _search_source_direct(
    *,
    source_anchor_pct: int,
    train_tables: list[dict[str, np.ndarray]],
    holdout_tables: list[dict[str, np.ndarray]],
    target_anchor_pcts: tuple[int, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_idx = mod.POSITION_TO_INDEX[float(int(source_anchor_pct)) / 100.0]
    source_table = train_tables[source_idx]
    candidates: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []

    for family_name in FAMILY_NAMES:
        feature_names = list(mod.SEARCH_FAMILY_MAP[str(family_name)])
        feature_count = len(feature_names)
        for model_name in LINEAR_MODEL_NAMES:
            for c_value in LINEAR_C_VALUES:
                for class_weight in LINEAR_CLASS_WEIGHT:
                    for l1_ratio in _linear_l1_ratios(model_name):
                        candidate = {
                            "route_family": "linear",
                            "model_name": str(model_name),
                            "representation": "raw+rank",
                            "source_anchor_pct": int(source_anchor_pct),
                            "family_name": str(family_name),
                            "feature_count": int(feature_count),
                            "feature_names": list(feature_names),
                            "c_value": float(c_value),
                            "class_weight": str(class_weight),
                            "seed_values": [int(v) for v in SEARCH_SEEDS],
                        }
                        if l1_ratio is not None:
                            candidate["l1_ratio"] = float(l1_ratio)
                        candidate["candidate_id"] = _candidate_id(candidate)
                        route = mod._fit_candidate_route(candidate=candidate, table=source_table)
                        candidates.append(candidate)
                        holdout_rows.extend(
                            _evaluate_candidate_rows(
                                candidate=candidate,
                                route=route,
                                holdout_tables=holdout_tables,
                                target_anchor_pcts=target_anchor_pcts,
                            )
                        )

    return candidates, holdout_rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Fast front-half science holdout search with partial blind patching")
    ap.add_argument("--main-cache-root", default="MUI_HUB/cache")
    ap.add_argument("--extra-cache-root", default="MUI_HUB/cache_train")
    ap.add_argument("--blind-cache-root", default="/home/jovyan/public-ro/MUI_HUB/cache_test")
    ap.add_argument("--feature-cache-dir", default="results/cache/science_dense_slot_search/labeled")
    ap.add_argument("--blind-feature-cache-dir", default="results/cache/science_dense_slot_search/blind")
    ap.add_argument("--refresh-feature-cache", action="store_true")
    ap.add_argument("--refresh-blind-feature-cache", action="store_true")
    ap.add_argument("--feature-workers", type=int, default=8)
    ap.add_argument("--search-workers", type=int, default=5)
    ap.add_argument("--feature-chunk-problems", type=int, default=mod.DEFAULT_FEATURE_CHUNK_PROBLEMS)
    ap.add_argument("--max-problems-per-cache", type=int, default=0)
    ap.add_argument("--holdout-split", type=float, default=0.15)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--source-anchors", default="10,20,30,40,50")
    ap.add_argument("--target-anchors", default="10,20,30,40,50")
    ap.add_argument("--transfer-margin", type=float, default=0.0010)
    ap.add_argument("--tree-margin", type=float, default=0.0025)
    ap.add_argument("--base-json", default=str(mod.BASE_AGGRESSIVE_JSON))
    ap.add_argument(
        "--out-json",
        default="submission/EarlyStop/es_svd_ms_rr_r1__coding_rotation_adapter_aggressive_tokenonly_late70100__science_front50_holdout_search_20260416.json",
    )
    ap.add_argument(
        "--method-name",
        default="es_svd_ms_rr_r1__coding_rotation_adapter_aggressive_tokenonly_late70100__science_front50_holdout_search_20260416",
    )
    ap.add_argument(
        "--candidate-csv",
        default="results/tables/science_front50_holdout_search_candidates_20260416.csv",
    )
    ap.add_argument(
        "--selected-csv",
        default="results/tables/science_front50_holdout_search_selected_20260416.csv",
    )
    ap.add_argument(
        "--eval-json",
        default="results/scans/earlystop/science_front50_holdout_search_20260416_eval.json",
    )
    ap.add_argument(
        "--manifest-json",
        default="results/tables/science_front50_holdout_search_manifest_20260416.json",
    )
    ap.add_argument(
        "--doc-out",
        default="docs/SCIENCE_FRONT50_HOLDOUT_SEARCH_20260416.md",
    )
    args = ap.parse_args()

    source_anchor_pcts = mod._parse_anchor_csv(args.source_anchors)
    target_anchor_pcts = mod._parse_anchor_csv(args.target_anchors)
    eval_positions = tuple(float(int(v)) / 100.0 for v in target_anchor_pcts)
    target_position_indices = {
        mod.POSITION_TO_INDEX[float(int(v)) / 100.0]
        for v in target_anchor_pcts
    }
    required_feature_names = mod._collect_required_features()
    max_problems_per_cache = None if int(args.max_problems_per_cache) <= 0 else int(args.max_problems_per_cache)

    labeled_feature_cache_dir = None
    if str(args.feature_cache_dir).strip().lower() not in {"", "none", "off"}:
        labeled_feature_cache_dir = (mod.REPO_ROOT / str(args.feature_cache_dir)).resolve()

    blind_feature_cache_dir = None
    if str(args.blind_feature_cache_dir).strip().lower() not in {"", "none", "off"}:
        blind_feature_cache_dir = (mod.REPO_ROOT / str(args.blind_feature_cache_dir)).resolve()

    labeled_store, labeled_cache_state = mod._load_labeled_science_store(
        main_cache_root=mod._resolve_path(str(args.main_cache_root)),
        extra_cache_root=mod._resolve_path(str(args.extra_cache_root)),
        feature_cache_dir=labeled_feature_cache_dir,
        refresh_feature_cache=bool(args.refresh_feature_cache),
        feature_workers=int(args.feature_workers),
        feature_chunk_problems=int(args.feature_chunk_problems),
        required_feature_names=required_feature_names,
        max_problems_per_cache=max_problems_per_cache,
    )
    holdout_problem_map, holdout_problem_summary = _build_holdout_problem_map(
        labeled_store,
        holdout_split=float(args.holdout_split),
        split_seed=int(args.split_seed),
    )
    train_store, holdout_store, full_store = _split_feature_store(
        labeled_store,
        holdout_problem_map=holdout_problem_map,
    )

    train_tables = _build_domain_training_tables(train_store, mod.POSITIONS)
    holdout_tables = _build_domain_training_tables(holdout_store, mod.POSITIONS)
    full_tables = _build_domain_training_tables(full_store, mod.POSITIONS)

    print(
        f"[data] labeled science payloads={len(labeled_store)} train_payloads={len(train_store)} holdout_payloads={len(holdout_store)}",
        flush=True,
    )
    print(
        f"[data] train_groups={len(np.unique(train_tables[0]['groups']))} holdout_groups={len(np.unique(holdout_tables[0]['groups']))}",
        flush=True,
    )

    candidate_by_id: dict[str, dict[str, Any]] = {}
    holdout_rows: list[dict[str, Any]] = []
    worker_count = max(1, min(int(args.search_workers), len(source_anchor_pcts), 8))
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(
                _search_source_direct,
                source_anchor_pct=int(source_anchor_pct),
                train_tables=train_tables,
                holdout_tables=holdout_tables,
                target_anchor_pcts=target_anchor_pcts,
            ): int(source_anchor_pct)
            for source_anchor_pct in source_anchor_pcts
        }
        for future in as_completed(future_map):
            source_anchor_pct = future_map[future]
            candidates, rows = future.result()
            for candidate in candidates:
                candidate_by_id[str(candidate["candidate_id"])] = candidate
            holdout_rows.extend(rows)
            print(
                f"[search] source={int(source_anchor_pct)}% candidates={len(candidates)} rows={len(rows)}",
                flush=True,
            )

    task_specific_rows, selected_rows = mod._select_rows(
        holdout_rows=holdout_rows,
        target_anchor_pcts=target_anchor_pcts,
        transfer_margin=float(args.transfer_margin),
        tree_margin=float(args.tree_margin),
    )

    candidate_csv = mod.REPO_ROOT / str(args.candidate_csv)
    selected_csv = mod.REPO_ROOT / str(args.selected_csv)
    mod._write_csv(candidate_csv, holdout_rows)
    mod._write_csv(selected_csv, selected_rows)

    train_route_ids = {
        str(row["candidate_id"])
        for row in task_specific_rows + selected_rows
    }
    train_routes_by_id: dict[str, dict[str, Any]] = {}
    for candidate_id in sorted(train_route_ids):
        candidate = dict(candidate_by_id[candidate_id])
        source_idx = mod.POSITION_TO_INDEX[float(int(candidate["source_anchor_pct"])) / 100.0]
        train_routes_by_id[candidate_id] = mod._fit_candidate_route(
            candidate=candidate,
            table=train_tables[source_idx],
        )

    task_selected_routes = []
    for row in task_specific_rows:
        route = dict(train_routes_by_id[str(row["candidate_id"])])
        route["target_anchor_pct"] = int(row["target_anchor_pct"])
        task_selected_routes.append(route)
    dense_selected_routes = []
    for row in selected_rows:
        route = dict(train_routes_by_id[str(row["candidate_id"])])
        route["target_anchor_pct"] = int(row["target_anchor_pct"])
        dense_selected_routes.append(route)

    task_score_fn = mod._build_selected_score_fn(task_selected_routes)
    dense_score_fn = mod._build_selected_score_fn(dense_selected_routes)
    reference_bundle = load_earlystop_svd_bundle(mod.CURRENT_SCIENCE_BUNDLE)
    reference_score_fn = mod.make_svd_bundle_score_fn(reference_bundle)
    reference_r2_eval = _evaluate_score_fn_on_holdout(
        score_fn=reference_score_fn,
        holdout_tables=holdout_tables,
        target_anchor_pcts=target_anchor_pcts,
    )
    task_eval = _evaluate_score_fn_on_holdout(
        score_fn=task_score_fn,
        holdout_tables=holdout_tables,
        target_anchor_pcts=target_anchor_pcts,
    )
    dense_eval = _evaluate_score_fn_on_holdout(
        score_fn=dense_score_fn,
        holdout_tables=holdout_tables,
        target_anchor_pcts=target_anchor_pcts,
    )

    fullfit_routes_by_target: list[dict[str, Any]] = []
    for row in selected_rows:
        selected_candidate = dict(candidate_by_id[str(row["candidate_id"])])
        selected_candidate["seed_values"] = _full_seed_values(selected_candidate)
        source_idx = mod.POSITION_TO_INDEX[float(int(selected_candidate["source_anchor_pct"])) / 100.0]
        full_route = mod._fit_candidate_route(
            candidate=selected_candidate,
            table=full_tables[source_idx],
        )
        full_route["target_anchor_pct"] = int(row["target_anchor_pct"])
        fullfit_routes_by_target.append(full_route)
    blind_score_fn = mod._build_selected_score_fn(fullfit_routes_by_target)

    blind_feature_store, blind_cache_path, blind_cache_status = mod._load_blind_science_feature_store(
        blind_cache_root=str(args.blind_cache_root),
        required_feature_names=required_feature_names,
        feature_workers=int(args.feature_workers),
        feature_chunk_problems=int(args.feature_chunk_problems),
        feature_cache_dir=blind_feature_cache_dir,
        refresh_feature_cache=bool(args.refresh_blind_feature_cache),
        max_problems=max_problems_per_cache,
    )
    base_payload = mod._load_json(Path(args.base_json))
    base_scores = base_payload["scores"]
    science_score_map: dict[str, dict[str, dict[str, list[float]]]] = {}
    for payload in blind_feature_store:
        cache_key = str(payload.get("cache_key"))
        problem_scores = mod._problem_scores_from_payload_partial(
            payload=payload,
            score_fn=blind_score_fn,
            base_problem_scores=base_scores[cache_key],
            target_position_indices=target_position_indices,
        )
        science_score_map[cache_key] = problem_scores
        print(
            f"[blind] cache={cache_key} problems={len(problem_scores)} samples={sum(len(v) for v in problem_scores.values())}",
            flush=True,
        )

    patched_out = mod.REPO_ROOT / str(args.out_json)
    patch_result = mod._patch_base_submission(
        base_json=Path(args.base_json),
        science_score_map=science_score_map,
        out_path=patched_out,
        method_name=str(args.method_name),
    )
    delta_summary = mod._compare_science_delta(
        base_json=Path(args.base_json),
        science_score_map=science_score_map,
    )

    eval_json = mod.REPO_ROOT / str(args.eval_json)
    eval_payload = {
        "created_at_utc": mod._now_utc(),
        "method_name": str(args.method_name),
        "holdout_problem_summary": holdout_problem_summary,
        "reference_r2": reference_r2_eval,
        "task_specific": task_eval,
        "dense_final": dense_eval,
        "selected_rows": selected_rows,
    }
    mod._write_json(eval_json, eval_payload)

    manifest = {
        "created_at_utc": mod._now_utc(),
        "method_name": str(args.method_name),
        "protocol": {
            "selection_protocol": "direct_holdout",
            "holdout_split": float(args.holdout_split),
            "split_seed": int(args.split_seed),
            "reflection_threshold": float(mod.DEFAULT_REFLECTION_THRESHOLD),
            "source_anchor_pcts": [int(v) for v in source_anchor_pcts],
            "target_anchor_pcts": [int(v) for v in target_anchor_pcts],
            "family_names": list(FAMILY_NAMES),
            "transfer_margin": float(args.transfer_margin),
            "tree_margin": float(args.tree_margin),
            "linear_models": list(LINEAR_MODEL_NAMES),
        },
        "cache_state": {
            "labeled": labeled_cache_state,
            "blind": {
                "status": str(blind_cache_status),
                "path": None if blind_cache_path is None else mod._display_path(blind_cache_path),
                "num_payloads": int(len(blind_feature_store)),
            },
        },
        "artifacts": {
            "patched_submission": mod._display_path(patched_out),
            "candidate_csv": mod._display_path(candidate_csv),
            "selected_csv": mod._display_path(selected_csv),
            "eval_json": mod._display_path(eval_json),
        },
        "holdout": {
            "reference_r2": reference_r2_eval,
            "task_specific": task_eval,
            "dense_final": dense_eval,
        },
        "selected_rows": selected_rows,
        "science_delta_vs_base": delta_summary,
        "patch_validation": patch_result["validation"],
    }
    manifest_json = mod.REPO_ROOT / str(args.manifest_json)
    mod._write_json(manifest_json, manifest)

    doc_out = mod.REPO_ROOT / str(args.doc_out)
    mod._render_doc(path=doc_out, manifest=manifest)

    print(
        "[holdout] dense_final auc_of_auroc={dense:.4f} vs task_specific={task:.4f} vs r2={r2:.4f}".format(
            dense=float(dense_eval["aggregate"]["auc_of_auroc"]),
            task=float(task_eval["aggregate"]["auc_of_auroc"]),
            r2=float(reference_r2_eval["aggregate"]["auc_of_auroc"]),
        ),
        flush=True,
    )
    print(f"[artifact] patched_submission={mod._display_path(patched_out)}", flush=True)
    print(f"[artifact] candidate_csv={mod._display_path(candidate_csv)}", flush=True)
    print(f"[artifact] selected_csv={mod._display_path(selected_csv)}", flush=True)
    print(f"[artifact] eval_json={mod._display_path(eval_json)}", flush=True)
    print(f"[artifact] manifest_json={mod._display_path(manifest_json)}", flush=True)
    print(f"[artifact] doc={mod._display_path(doc_out)}", flush=True)
    print(f"[validate] {patch_result['validation']}", flush=True)


if __name__ == "__main__":
    main()
