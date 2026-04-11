#!/usr/bin/env python3
"""Train coding-domain EarlyStop SVD model with raw+rank and export merged test submission."""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.ops.earlystop import EARLY_STOP_POSITIONS, discover_cache_entries, validate_earlystop_payload, write_earlystop_payload
from nad.ops.earlystop_svd import DEFAULT_REFLECTION_THRESHOLD, get_domain, load_earlystop_svd_bundle, save_earlystop_svd_bundle
from scripts.export_earlystop_svd_submission import (
    _bundle_reflection_thresholds,
    _collect_required_features as _collect_export_required_features,
    _load_or_build_feature_store,
    _problem_scores_from_payload,
)
from scripts.run_earlystop_prefix10_svd_round1 import (
    ANCHOR_POSITIONS,
    EXTRACTION_POSITIONS,
    OFFICIAL_SLOT_TO_ANCHOR,
    _display_path,
    _now_utc,
    _pct_label,
    evaluate_method_from_feature_store,
    make_svd_bundle_score_fn,
    make_tok_conf_score_fn,
    summarise_route,
    _render_cache_table,
    _render_method_table,
)
from SVDomain.train_es_svd_ms_rr_r1 import (
    DEFAULT_FEATURE_CHUNK_PROBLEMS,
    FIXED_EXCLUDED_FEATURES,
    FIXED_FAMILY_NAME,
    FIXED_FEATURE_NAMES,
    FIXED_REPRESENTATION,
    _baseline_bundle_result,
    _build_domain_bundle,
    _build_domain_training_tables,
    _build_holdout_problem_map,
    _collect_required_features,
    _qualify_feature_store,
    _resolve_path,
    _route_summary,
    _split_feature_store,
    _summarise_feature_store,
    _train_domain_anchor_routes,
)

METHOD_ID = "es_svd_coding_rr_r1"
MERGED_SUBMISSION_METHOD = "es_svd_ms_rr_r1__coding_rr_r1"
DOMAIN_NAME = "coding"


def _parse_csv(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(raw).split(",") if item.strip())


def _render_route_table(title: str, route_summary: dict[str, Any]) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| Anchor | CV AUROC | Baseline | Baseline CV | Rank | C | Whiten | Class Weight |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for anchor_label, route in route_summary.items():
        lines.append(
            "| {anchor} | {cv_auc:.4f} | {baseline} | {baseline_cv:.4f} | {rank} | {c_value:.2f} | {whiten} | {class_weight} |".format(
                anchor=anchor_label,
                cv_auc=float(route["cv_auroc"]),
                baseline=str(route["baseline_signal_name"]),
                baseline_cv=float(route["baseline_cv_auroc"]),
                rank=int(route["rank"]),
                c_value=float(route["c_value"]),
                whiten="yes" if bool(route["whiten"]) else "no",
                class_weight=str(route["class_weight"]),
            )
        )
    lines.append("")
    return lines


def _load_feature_store_from_pickle(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    feature_store = payload["feature_store"] if isinstance(payload, dict) and "feature_store" in payload else payload
    return list(feature_store)


def _discover_domain_cache_keys(cache_root: str, domain_name: str) -> tuple[str, ...]:
    keys = []
    for entry in discover_cache_entries(cache_root):
        if get_domain(entry.dataset_name) == domain_name:
            keys.append(str(entry.cache_key))
    return tuple(sorted(keys))


def _load_training_store(
    *,
    source_name: str,
    cache_root: str,
    include_cache_keys: tuple[str, ...],
    positions: tuple[float, ...],
    required_feature_names: set[str],
    max_problems_per_cache: int | None,
    feature_workers: int,
    chunk_problems: int,
    feature_cache_dir: Path | None,
    refresh_feature_cache: bool,
    feature_store_pkl: Path | None,
) -> tuple[list[dict[str, Any]], Path | None, str]:
    if feature_store_pkl is not None:
        return _load_feature_store_from_pickle(feature_store_pkl), feature_store_pkl, "loaded_from_pickle"
    if not include_cache_keys:
        return [], None, "skipped_empty_domain"
    raw_store, cache_path, cache_status = _load_or_build_feature_store(
        cache_root=cache_root,
        positions=positions,
        required_feature_names=required_feature_names,
        max_problems=max_problems_per_cache,
        reflection_threshold=float(DEFAULT_REFLECTION_THRESHOLD),
        workers=int(feature_workers),
        feature_chunk_problems=int(chunk_problems),
        feature_cache_dir=feature_cache_dir,
        refresh_feature_cache=bool(refresh_feature_cache),
        include_cache_keys=set(include_cache_keys),
    )
    return _qualify_feature_store(raw_store, source_name), cache_path, cache_status


def _write_registry(path: Path, summary: dict[str, Any]) -> None:
    registry: dict[str, Any]
    if path.exists():
        registry = json.loads(path.read_text(encoding="utf-8"))
    else:
        registry = {"family": "es_svd", "methods": []}

    registry["family"] = "es_svd"
    registry["updated_at_utc"] = _now_utc()
    existing_methods = {
        str(item.get("method_id")): dict(item)
        for item in registry.get("methods", [])
        if isinstance(item, dict) and item.get("method_id")
    }
    coding_entry = {
        "method_id": METHOD_ID,
        "kind": "single_domain_bundle",
        "domain": DOMAIN_NAME,
        "representation": FIXED_REPRESENTATION,
        "anchors": [int(round(float(v) * 100.0)) for v in ANCHOR_POSITIONS],
        "model_path": summary["artifacts"]["model"],
        "summary_path": summary["artifacts"]["summary_json"],
        "eval_path": summary["artifacts"]["eval_json"],
        "doc_path": summary["artifacts"]["doc_md"],
    }
    if summary["artifacts"].get("submission_json"):
        coding_entry["submission_path"] = summary["artifacts"]["submission_json"]
    if summary["artifacts"].get("blind_coding_scores_json"):
        coding_entry["blind_coding_scores_path"] = summary["artifacts"]["blind_coding_scores_json"]
    existing_methods[METHOD_ID] = coding_entry

    order = {
        "es_svd_math_rr_r1": 10,
        "es_svd_science_rr_r1": 20,
        "es_svd_ms_rr_r1": 30,
        METHOD_ID: 40,
    }
    registry["methods"] = sorted(
        existing_methods.values(),
        key=lambda item: (order.get(str(item.get("method_id")), 999), str(item.get("method_id"))),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_results_doc(path: Path, summary: dict[str, Any]) -> None:
    validate_rows = [
        summary["validate"]["baselines"]["tok_conf_prefix_mean_v1"],
        summary["validate"]["baselines"]["earlystop_svd_lowrank_lr_v1"],
        summary["validate"]["baselines"]["earlystop_prefix10_svd_round1"],
        summary["validate"]["candidate"],
    ]
    lines = [
        "# ES SVD CODING RR R1",
        "",
        "## Naming",
        "",
        "- `es_svd`：EarlyStop SVD family。",
        "- `coding`：只覆盖 coding 域。",
        "- `rr`：只用 `raw+rank` 表示。",
        "- `r1`：当前 coding 分域重训第一版。",
        "- 名称里故意去掉 `prefix` / `p10`，但训练协议仍是四个 anchors：`10/40/70/100`。",
        "",
        "## Feature Spec",
        "",
        f"- `representation`：`{summary['feature_spec']['representation']}`。",
        f"- `feature family`：`{summary['feature_spec']['family_name']}`。",
        f"- `included features`：`{', '.join(summary['feature_spec']['feature_names'])}`。",
        f"- `excluded features`：`{', '.join(summary['feature_spec']['excluded_feature_names'])}`。",
        "- `row feature`：不使用任何数值 row 特征；只保留 `has_rows_bank` 作为 availability flag。",
        "",
        "## Protocol",
        "",
        f"- `main cache root`：`{summary['protocol']['main_cache_root']}`。",
        f"- `extra cache root`：`{summary['protocol']['extra_cache_root']}`。",
        f"- `holdout split`：`{summary['protocol']['train_ratio']}/{summary['protocol']['holdout_ratio']}`。",
        f"- `holdout unit`：按 `dataset + problem_id` 做跨 root 一致切分，`split_seed={summary['protocol']['split_seed']}`。",
        f"- `anchors`：`{', '.join(str(v) for v in summary['protocol']['anchor_positions_pct'])}`。",
        "- `routing policy`：训练与最终 bundle 均不允许 baseline / single-feature route；baseline 只做对照。",
        f"- `extra cache usage`：`{summary['protocol']['extra_cache_usage']}`。",
        "",
        "## Artifacts",
        "",
        f"- `coding model`：`{summary['artifacts']['model']}`。",
        f"- `summary json`：`{summary['artifacts']['summary_json']}`。",
        f"- `eval json`：`{summary['artifacts']['eval_json']}`。",
        f"- `blind coding scores`：`{summary['artifacts']['blind_coding_scores_json']}`。",
        f"- `merged submission`：`{summary['artifacts']['submission_json']}`。",
        "",
        "## Validate",
        "",
    ]
    lines.extend(_render_method_table("coding holdout", validate_rows))
    lines.extend(_render_cache_table(summary["validate"]["candidate"]))
    lines.extend(_render_route_table("coding full-fit anchor routes", summary["fullfit"]["route_summary"]))
    lines.extend(
        [
            "## Blind Export",
            "",
            f"- `base submission`：`{summary['test_export']['base_submission']}`。",
            f"- `merged method_name`：`{summary['test_export']['merged_method_name']}`。",
            f"- `coding cache keys replaced`：`{', '.join(summary['test_export']['coding_cache_keys'])}`。",
            f"- `blind feature cache status`：`{summary['test_export']['feature_cache_status']}`。",
            f"- `blind feature cache path`：`{summary['test_export']['feature_cache_path']}`。",
            f"- `submission validation`：`{summary['test_export']['validation_stats']}`。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _export_blind_merge(
    *,
    bundle: dict[str, Any],
    cache_root: str,
    base_submission_path: Path,
    coding_cache_keys: tuple[str, ...],
    merged_method_name: str,
    out_submission_path: Path,
    out_blind_scores_path: Path,
    workers: int,
    feature_chunk_problems: int,
    feature_cache_dir: Path | None,
    refresh_feature_cache: bool,
    blind_feature_store_pkl: Path | None,
) -> dict[str, Any]:
    base_payload = json.loads(base_submission_path.read_text(encoding="utf-8"))
    validate_earlystop_payload(base_payload)

    threshold_values = sorted(_bundle_reflection_thresholds(bundle))
    if len(threshold_values) != 1:
        raise ValueError(f"{METHOD_ID} expected one reflection threshold, got {threshold_values}")
    reflection_threshold = float(threshold_values[0])

    required_features = _collect_export_required_features(bundle)
    if blind_feature_store_pkl is not None:
        feature_store = _load_feature_store_from_pickle(blind_feature_store_pkl)
        feature_cache_path = blind_feature_store_pkl
        feature_cache_status = "loaded_from_pickle"
    else:
        feature_store, feature_cache_path, feature_cache_status = _load_or_build_feature_store(
            cache_root=cache_root,
            positions=tuple(float(v) for v in EARLY_STOP_POSITIONS),
            required_feature_names=required_features,
            max_problems=None,
            reflection_threshold=reflection_threshold,
            workers=int(workers),
            feature_chunk_problems=int(feature_chunk_problems),
            feature_cache_dir=feature_cache_dir,
            refresh_feature_cache=bool(refresh_feature_cache),
            include_cache_keys=set(coding_cache_keys),
        )

    score_fn = make_svd_bundle_score_fn(bundle)
    score_map = {}
    for payload in feature_store:
        cache_key = str(payload["cache_key"])
        score_map[cache_key] = _problem_scores_from_payload(payload, score_fn)

    missing = [cache_key for cache_key in coding_cache_keys if cache_key not in score_map]
    if missing:
        raise ValueError(f"Missing blind coding scores for cache keys: {missing}")

    out_blind_scores_path.parent.mkdir(parents=True, exist_ok=True)
    out_blind_scores_path.write_text(
        json.dumps(
            {
                "method_id": METHOD_ID,
                "created_at_utc": _now_utc(),
                "cache_root": str(cache_root),
                "coding_cache_keys": list(coding_cache_keys),
                "scores": score_map,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    merged_payload = copy.deepcopy(base_payload)
    merged_payload["method_name"] = str(merged_method_name)
    merged_scores = dict(base_payload["scores"])
    for cache_key in coding_cache_keys:
        merged_scores[str(cache_key)] = score_map[str(cache_key)]
    merged_payload["scores"] = merged_scores
    validation_stats = validate_earlystop_payload(merged_payload)

    out_submission_path.parent.mkdir(parents=True, exist_ok=True)
    write_earlystop_payload(merged_payload, out_submission_path)
    return {
        "base_submission": _display_path(base_submission_path),
        "merged_method_name": str(merged_method_name),
        "coding_cache_keys": list(coding_cache_keys),
        "feature_cache_status": str(feature_cache_status),
        "feature_cache_path": None if feature_cache_path is None else _display_path(feature_cache_path),
        "blind_scores_path": _display_path(out_blind_scores_path),
        "submission_path": _display_path(out_submission_path),
        "validation_stats": dict(validation_stats),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train coding-domain EarlyStop SVD and export merged blind submission")
    ap.add_argument("--main-cache-root", default="MUI_HUB/cache", help="Primary labeled cache root")
    ap.add_argument("--extra-cache-root", default="MUI_HUB/cache_train", help="Extra labeled cache root")
    ap.add_argument("--test-cache-root", default="/home/jovyan/public-ro/MUI_HUB/cache_test", help="Blind test cache root")
    ap.add_argument("--holdout-split", type=float, default=0.15, help="Holdout ratio; 0.15 means 85/15")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--feature-workers", type=int, default=8)
    ap.add_argument("--fit-workers", type=int, default=8)
    ap.add_argument("--feature-chunk-problems", type=int, default=DEFAULT_FEATURE_CHUNK_PROBLEMS)
    ap.add_argument("--feature-cache-dir", default="results/cache/es_svd_coding_rr_r1")
    ap.add_argument("--blind-feature-cache-dir", default="results/cache/export_earlystop_svd_submission_es_svd_coding_rr_r1")
    ap.add_argument("--main-feature-store-pkl", default="none", help="Optional prebuilt labeled feature_store pickle")
    ap.add_argument("--extra-feature-store-pkl", default="none", help="Optional prebuilt extra labeled feature_store pickle")
    ap.add_argument("--blind-feature-store-pkl", default="none", help="Optional prebuilt blind coding feature_store pickle")
    ap.add_argument("--refresh-feature-cache", action="store_true")
    ap.add_argument("--refresh-blind-feature-cache", action="store_true")
    ap.add_argument("--max-problems-per-cache", type=int, default=0, help="0 means all problems")
    ap.add_argument("--out-model", default="models/ml_selectors/es_svd_coding_rr_r1.pkl")
    ap.add_argument("--out-summary", default="results/scans/earlystop/es_svd_coding_rr_r1_summary.json")
    ap.add_argument("--out-eval", default="results/scans/earlystop/es_svd_coding_rr_r1_eval.json")
    ap.add_argument("--out-doc", default="docs/ES_SVD_CODING_RR_R1.md")
    ap.add_argument("--registry-path", default="SVDomain/registry.json")
    ap.add_argument("--base-submission", default="submission/EarlyStop/es_svd_ms_rr_r1__coding_from_round1c.json")
    ap.add_argument("--coding-cache-keys", default="DS-R1/lcb_v5,Qwen3-4B/lcb_v5")
    ap.add_argument("--merged-method-name", default=MERGED_SUBMISSION_METHOD)
    ap.add_argument("--out-submission", default="submission/EarlyStop/es_svd_ms_rr_r1__coding_rr_r1.json")
    ap.add_argument("--out-blind-coding-scores", default="results/scans/earlystop/es_svd_coding_rr_r1_blind_coding_scores.json")
    args = ap.parse_args()

    main_cache_root = _resolve_path(str(args.main_cache_root))
    extra_cache_root = _resolve_path(str(args.extra_cache_root))
    test_cache_root = _resolve_path(str(args.test_cache_root))
    max_problems_per_cache = None if int(args.max_problems_per_cache) <= 0 else int(args.max_problems_per_cache)
    feature_cache_dir = None if str(args.feature_cache_dir).strip().lower() in {"", "none", "off"} else (REPO_ROOT / str(args.feature_cache_dir)).resolve()
    blind_feature_cache_dir = None if str(args.blind_feature_cache_dir).strip().lower() in {"", "none", "off"} else (REPO_ROOT / str(args.blind_feature_cache_dir)).resolve()
    main_feature_store_pkl = None if str(args.main_feature_store_pkl).strip().lower() in {"", "none", "off"} else (REPO_ROOT / str(args.main_feature_store_pkl)).resolve()
    extra_feature_store_pkl = None if str(args.extra_feature_store_pkl).strip().lower() in {"", "none", "off"} else (REPO_ROOT / str(args.extra_feature_store_pkl)).resolve()
    blind_feature_store_pkl = None if str(args.blind_feature_store_pkl).strip().lower() in {"", "none", "off"} else (REPO_ROOT / str(args.blind_feature_store_pkl)).resolve()
    coding_cache_keys = _parse_csv(args.coding_cache_keys)

    v1_bundle = load_earlystop_svd_bundle(REPO_ROOT / "models/ml_selectors/earlystop_svd_lowrank_lr_v1.pkl")
    legacy_prefix10_bundle = load_earlystop_svd_bundle(REPO_ROOT / "models/ml_selectors/earlystop_prefix10_svd_round1.pkl")
    required_features = _collect_required_features(v1_bundle, legacy_prefix10_bundle)
    main_domain_cache_keys = _discover_domain_cache_keys(main_cache_root, DOMAIN_NAME)
    extra_domain_cache_keys = _discover_domain_cache_keys(extra_cache_root, DOMAIN_NAME)

    main_store, main_cache_path, main_cache_status = _load_training_store(
        source_name="cache",
        cache_root=main_cache_root,
        include_cache_keys=main_domain_cache_keys,
        positions=tuple(float(v) for v in EXTRACTION_POSITIONS),
        required_feature_names=required_features,
        max_problems_per_cache=max_problems_per_cache,
        feature_workers=int(args.feature_workers),
        chunk_problems=int(args.feature_chunk_problems),
        feature_cache_dir=feature_cache_dir,
        refresh_feature_cache=bool(args.refresh_feature_cache),
        feature_store_pkl=main_feature_store_pkl,
    )
    extra_store, extra_cache_path, extra_cache_status = _load_training_store(
        source_name="cache_train",
        cache_root=extra_cache_root,
        include_cache_keys=extra_domain_cache_keys,
        positions=tuple(float(v) for v in EXTRACTION_POSITIONS),
        required_feature_names=required_features,
        max_problems_per_cache=max_problems_per_cache,
        feature_workers=int(args.feature_workers),
        chunk_problems=int(args.feature_chunk_problems),
        feature_cache_dir=feature_cache_dir,
        refresh_feature_cache=bool(args.refresh_feature_cache),
        feature_store_pkl=extra_feature_store_pkl,
    )

    domain_main_store = [payload for payload in main_store if payload["domain"] == DOMAIN_NAME]
    domain_extra_store = [payload for payload in extra_store if payload["domain"] == DOMAIN_NAME]
    full_store = list(domain_main_store) + list(domain_extra_store)
    if not full_store:
        raise ValueError(f"No {DOMAIN_NAME} payloads found under {main_cache_root} and {extra_cache_root}")

    holdout_problem_map, holdout_problem_summary = _build_holdout_problem_map(
        full_store,
        holdout_split=float(args.holdout_split),
        split_seed=int(args.split_seed),
    )
    train_store, holdout_store, domain_full_store = _split_feature_store(
        full_store,
        holdout_problem_map=holdout_problem_map,
    )
    if not train_store:
        raise ValueError(f"{DOMAIN_NAME} train_store is empty after holdout split")
    if not holdout_store:
        raise ValueError(f"{DOMAIN_NAME} holdout_store is empty after holdout split")

    protocol_core = {
        "main_cache_root": str(main_cache_root),
        "extra_cache_root": str(extra_cache_root),
        "test_cache_root": str(test_cache_root),
        "train_ratio": "85",
        "holdout_ratio": "15",
        "holdout_split": float(args.holdout_split),
        "split_seed": int(args.split_seed),
        "n_splits": int(args.n_splits),
        "random_state": int(args.random_state),
        "feature_workers": int(args.feature_workers),
        "fit_workers": int(args.fit_workers),
        "feature_chunk_problems": int(args.feature_chunk_problems),
        "max_problems_per_cache": None if max_problems_per_cache is None else int(max_problems_per_cache),
        "anchor_positions": [float(v) for v in ANCHOR_POSITIONS],
        "anchor_positions_pct": [int(round(float(v) * 100.0)) for v in ANCHOR_POSITIONS],
        "official_slot_to_anchor": {str(int(round(float(k) * 100.0))): int(round(float(v) * 100.0)) for k, v in OFFICIAL_SLOT_TO_ANCHOR.items()},
        "coding_included": True,
        "extra_cache_usage": "scanned but no coding caches found" if not domain_extra_store else "coding caches found and used",
    }

    train_tables = _build_domain_training_tables(train_store, ANCHOR_POSITIONS)
    splitfit_routes = _train_domain_anchor_routes(
        domain_name=f"{DOMAIN_NAME}_splitfit",
        tables=train_tables,
        positions=ANCHOR_POSITIONS,
        n_splits=int(args.n_splits),
        random_state=int(args.random_state),
        fit_workers=int(args.fit_workers),
    )
    splitfit_bundle = _build_domain_bundle(
        method_id=f"{METHOD_ID}_splitfit",
        domain_name=DOMAIN_NAME,
        routes=splitfit_routes,
        protocol=protocol_core,
    )

    full_tables = _build_domain_training_tables(domain_full_store, ANCHOR_POSITIONS)
    fullfit_routes = _train_domain_anchor_routes(
        domain_name=f"{DOMAIN_NAME}_fullfit",
        tables=full_tables,
        positions=ANCHOR_POSITIONS,
        n_splits=int(args.n_splits),
        random_state=int(args.random_state),
        fit_workers=int(args.fit_workers),
    )
    fullfit_bundle = _build_domain_bundle(
        method_id=METHOD_ID,
        domain_name=DOMAIN_NAME,
        routes=fullfit_routes,
        protocol=protocol_core,
    )

    validate_block = {
        "candidate": evaluate_method_from_feature_store(
            method_name=METHOD_ID,
            feature_store=holdout_store,
            position_values=tuple(float(v) for v in EARLY_STOP_POSITIONS),
            score_fn=make_svd_bundle_score_fn(splitfit_bundle),
        ),
        "baselines": {
            "tok_conf_prefix_mean_v1": evaluate_method_from_feature_store(
                method_name="tok_conf_prefix_mean_v1",
                feature_store=holdout_store,
                position_values=tuple(float(v) for v in EARLY_STOP_POSITIONS),
                score_fn=make_tok_conf_score_fn(),
            ),
            "earlystop_svd_lowrank_lr_v1": _baseline_bundle_result(
                method_name="earlystop_svd_lowrank_lr_v1",
                feature_store=holdout_store,
                bundle=v1_bundle,
            ),
            "earlystop_prefix10_svd_round1": _baseline_bundle_result(
                method_name="earlystop_prefix10_svd_round1",
                feature_store=holdout_store,
                bundle=legacy_prefix10_bundle,
            ),
        },
    }

    out_model = REPO_ROOT / str(args.out_model)
    out_summary = REPO_ROOT / str(args.out_summary)
    out_eval = REPO_ROOT / str(args.out_eval)
    out_doc = REPO_ROOT / str(args.out_doc)
    registry_path = REPO_ROOT / str(args.registry_path)
    base_submission_path = REPO_ROOT / str(args.base_submission)
    out_submission = REPO_ROOT / str(args.out_submission)
    out_blind_coding_scores = REPO_ROOT / str(args.out_blind_coding_scores)

    save_earlystop_svd_bundle(fullfit_bundle, out_model)
    test_export = _export_blind_merge(
        bundle=fullfit_bundle,
        cache_root=test_cache_root,
        base_submission_path=base_submission_path,
        coding_cache_keys=coding_cache_keys,
        merged_method_name=str(args.merged_method_name),
        out_submission_path=out_submission,
        out_blind_scores_path=out_blind_coding_scores,
        workers=int(args.feature_workers),
        feature_chunk_problems=int(args.feature_chunk_problems),
        feature_cache_dir=blind_feature_cache_dir,
        refresh_feature_cache=bool(args.refresh_blind_feature_cache),
        blind_feature_store_pkl=blind_feature_store_pkl,
    )

    summary = {
        "method_family": "es_svd",
        "method_id": METHOD_ID,
        "created_at_utc": _now_utc(),
        "naming": {
            "removed_prefix_from_name": True,
            "rr_means": FIXED_REPRESENTATION,
            "anchor_protocol_omitted_from_name": [int(round(float(v) * 100.0)) for v in ANCHOR_POSITIONS],
        },
        "protocol": dict(protocol_core),
        "feature_spec": {
            "family_name": FIXED_FAMILY_NAME,
            "representation": FIXED_REPRESENTATION,
            "feature_names": list(FIXED_FEATURE_NAMES),
            "excluded_feature_names": list(FIXED_EXCLUDED_FEATURES),
            "uses_numeric_row_features": False,
            "uses_has_rows_bank_flag": True,
            "reflection_threshold": float(DEFAULT_REFLECTION_THRESHOLD),
        },
        "data": {
            "feature_cache_status": {
                "cache": str(main_cache_status),
                "cache_train": str(extra_cache_status),
            },
            "feature_cache_paths": {
                "cache": None if main_cache_path is None else str(main_cache_path),
                "cache_train": None if extra_cache_path is None else str(extra_cache_path),
            },
            "prebuilt_feature_store_paths": {
                "cache": None if main_feature_store_pkl is None else _display_path(main_feature_store_pkl),
                "cache_train": None if extra_feature_store_pkl is None else _display_path(extra_feature_store_pkl),
                "blind_coding": None if blind_feature_store_pkl is None else _display_path(blind_feature_store_pkl),
            },
            "store_summary": {
                "coding": {
                    "cache": _summarise_feature_store(domain_main_store),
                    "cache_train": _summarise_feature_store(domain_extra_store),
                    "train": _summarise_feature_store(train_store),
                    "holdout": _summarise_feature_store(holdout_store),
                    "full": _summarise_feature_store(domain_full_store),
                    "holdout_problem_summary": holdout_problem_summary,
                },
            },
        },
        "validate": validate_block,
        "fullfit": {
            "saved_model": _display_path(out_model),
            "route_summary": _route_summary(fullfit_routes),
        },
        "test_export": dict(test_export),
        "artifacts": {
            "model": _display_path(out_model),
            "summary_json": _display_path(out_summary),
            "eval_json": _display_path(out_eval),
            "doc_md": _display_path(out_doc),
            "registry_json": _display_path(registry_path),
            "blind_coding_scores_json": _display_path(out_blind_coding_scores),
            "submission_json": _display_path(out_submission),
        },
    }
    eval_payload = {
        "protocol": summary["protocol"],
        "feature_spec": summary["feature_spec"],
        "validate": summary["validate"],
        "test_export": summary["test_export"],
        "artifacts": summary["artifacts"],
    }

    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_eval.parent.mkdir(parents=True, exist_ok=True)
    out_eval.write_text(json.dumps(eval_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_results_doc(out_doc, summary)
    _write_registry(registry_path, summary)

    print("[done] artifacts", flush=True)
    print(f"  coding model        : {_display_path(out_model)}", flush=True)
    print(f"  merged submission   : {_display_path(out_submission)}", flush=True)
    print(f"  blind coding scores : {_display_path(out_blind_coding_scores)}", flush=True)
    print(f"  summary json        : {_display_path(out_summary)}", flush=True)
    print(f"  eval json           : {_display_path(out_eval)}", flush=True)
    print(f"  doc                 : {_display_path(out_doc)}", flush=True)
    print(f"  registry            : {_display_path(registry_path)}", flush=True)


if __name__ == "__main__":
    main()
