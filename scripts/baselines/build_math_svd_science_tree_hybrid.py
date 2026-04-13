#!/usr/bin/env python3
"""Build a mixed EarlyStop bundle: math from SVD, science/coding from tree fulltrain.

This script does three things in one place:

1. Refit the frozen best `science` tree route on the grouped 85/15 train split and
   evaluate it on the holdout.
2. Compose a noncoding holdout proxy by combining:
   - frozen official `math` SVD holdout caches from `es_svd_ms_rr_r2_20260412_eval.json`
   - freshly rerun `science` tree holdout caches
3. Build a full-train blind-export bundle that uses:
   - `math` routes from `es_svd_math_rr_r2_20260412`
   - `science` routes from `tree_baselines_fulltrain_v1`
   - `coding` routes from `tree_baselines_fulltrain_v1`
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import pickle
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from nad.ops.earlystop import (
    EARLY_STOP_POSITIONS,
    build_earlystop_payload,
    discover_cache_entries,
    validate_earlystop_payload,
    write_earlystop_payload,
)
from nad.ops.earlystop_mixed import collect_required_features, make_mixed_bundle_score_fn
from nad.ops.earlystop_svd import load_earlystop_svd_bundle
from scripts.export_earlystop_svd_submission import _problem_scores_from_payload
from scripts.run_earlystop_prefix10_svd_round1 import (
    _display_path,
    aggregate_cache_metrics,
    evaluate_method_from_feature_store,
)
from scripts.baselines.export_tree_baselines_submission import (
    _load_blind_feature_store,
    _load_labeled_feature_store,
)
from scripts.baselines.train_tree_baselines import (
    _build_tree_bundle,
    _fit_domain_tree_routes,
    _required_features_from_bundles,
    make_tree_bundle_score_fn,
)
from SVDomain.train_es_svd_ms_rr_r2 import (
    _build_domain_training_tables,
    _build_holdout_problem_map,
    _resolve_path,
    _split_feature_store,
)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_csv(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(raw).split(",") if part.strip())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _save_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _load_best_tree_row(csv_path: Path, domain: str) -> dict[str, Any]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    candidates = [
        row
        for row in rows
        if str(row.get("domain")) == str(domain)
        and str(row.get("protocol")) == "id_grouped_85_15"
    ]
    if not candidates:
        raise ValueError(f"No tree baseline row found for domain={domain} in {csv_path}")
    return max(candidates, key=lambda row: float(row["auc_of_auroc"]))


def _load_math_eval(eval_path: Path) -> dict[str, Any]:
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    return payload["validate"]["math"]["candidate"]


def _load_svd_combined_eval(eval_path: Path) -> dict[str, Any]:
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    return payload["validate"]["combined_noncoding"]["candidate"]


def _build_hybrid_bundle(
    *,
    method_id: str,
    math_bundle: dict[str, Any],
    tree_bundle: dict[str, Any],
    include_coding: bool,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    math_feature_names = list(math_bundle["feature_names"])
    tree_feature_names = list(tree_bundle["feature_names"])
    if math_feature_names != tree_feature_names:
        raise ValueError("math SVD bundle and tree bundle expose different feature banks")

    domains: dict[str, dict[str, Any]] = {
        "math": {"routes": list(math_bundle["domains"]["math"]["routes"])},
        "science": {"routes": list(tree_bundle["domains"]["science"]["routes"])},
    }
    if include_coding:
        domains["coding"] = {"routes": list(tree_bundle["domains"]["coding"]["routes"])}

    return {
        "bundle_version": str(method_id),
        "created_at_utc": _now_utc(),
        "feature_names": math_feature_names,
        "positions": list(EARLY_STOP_POSITIONS),
        "domains": domains,
        "protocol": dict(protocol),
    }


def _render_doc(
    *,
    path: Path,
    method_name: str,
    math_bundle_path: Path,
    tree_bundle_path: Path,
    science_row: dict[str, Any],
    math_eval: dict[str, Any],
    science_eval: dict[str, Any],
    hybrid_holdout: dict[str, Any],
    svd_combined_eval: dict[str, Any],
    bundle_path: Path,
    submission_path: Path,
    manifest_path: Path,
) -> None:
    math_agg = math_eval["aggregate"]
    science_agg = science_eval["aggregate"]
    hybrid_agg = hybrid_holdout["aggregate"]
    svd_agg = svd_combined_eval["aggregate"]
    lines = [
        "# Math-SVD + Science-Tree Hybrid",
        "",
        "## Goal",
        "",
        "- Keep the strongest existing `math` EarlyStop line unchanged: `es_svd_math_rr_r2_20260412`.",
        "- Swap the weaker `science` SVD line for the stronger grouped-holdout tree baseline.",
        "- Export a full-train blind EarlyStop payload without retraining the already-frozen math SVD bundle.",
        "",
        "## Source Bundles",
        "",
        f"- `math SVD`: `{_display_path(math_bundle_path)}`",
        f"- `science/coding tree`: `{_display_path(tree_bundle_path)}`",
        "",
        "## Frozen Science Tree Selection",
        "",
        "- `domain`: `science`",
        f"- `model_family`: `{science_row['model_family']}`",
        f"- `feature_variant`: `{science_row['feature_variant']}`",
        f"- `holdout AUC of AUROC`: `{float(science_row['auc_of_auroc']):.2%}`",
        f"- `delta vs science SVD`: `{100.0 * float(science_row['delta_auc_of_auroc_vs_svd']):+.2f}` pts",
        f"- `search config`: `{science_row['search_config_json']}`",
        "",
        "## Holdout Readout",
        "",
        "| Slice | AUC of AUROC | AUC of SelAcc | AUROC@100% | Stop Acc@100% |",
        "|---|---:|---:|---:|---:|",
        "| `math` frozen SVD | {auc:.2%} | {sel:.2%} | {a100:.2%} | {s100:.2%} |".format(
            auc=float(math_agg["auc_of_auroc"]),
            sel=float(math_agg["auc_of_selacc"]),
            a100=float(math_agg["auroc@100%"]),
            s100=float(math_agg["stop_acc@100%"]),
        ),
        "| `science` rerun tree | {auc:.2%} | {sel:.2%} | {a100:.2%} | {s100:.2%} |".format(
            auc=float(science_agg["auc_of_auroc"]),
            sel=float(science_agg["auc_of_selacc"]),
            a100=float(science_agg["auroc@100%"]),
            s100=float(science_agg["stop_acc@100%"]),
        ),
        "| `noncoding` frozen SVD | {auc:.2%} | {sel:.2%} | {a100:.2%} | {s100:.2%} |".format(
            auc=float(svd_agg["auc_of_auroc"]),
            sel=float(svd_agg["auc_of_selacc"]),
            a100=float(svd_agg["auroc@100%"]),
            s100=float(svd_agg["stop_acc@100%"]),
        ),
        "| `noncoding` hybrid | {auc:.2%} | {sel:.2%} | {a100:.2%} | {s100:.2%} |".format(
            auc=float(hybrid_agg["auc_of_auroc"]),
            sel=float(hybrid_agg["auc_of_selacc"]),
            a100=float(hybrid_agg["auroc@100%"]),
            s100=float(hybrid_agg["stop_acc@100%"]),
        ),
        "",
        "## Noncoding Delta vs Frozen SVD",
        "",
        "- `AUC of AUROC`: `{delta:+.2f}` pts".format(
            delta=100.0 * (float(hybrid_agg["auc_of_auroc"]) - float(svd_agg["auc_of_auroc"]))
        ),
        "- `AUC of SelAcc`: `{delta:+.2f}` pts".format(
            delta=100.0 * (float(hybrid_agg["auc_of_selacc"]) - float(svd_agg["auc_of_selacc"]))
        ),
        "- `AUROC@100%`: `{delta:+.2f}` pts".format(
            delta=100.0 * (float(hybrid_agg["auroc@100%"]) - float(svd_agg["auroc@100%"]))
        ),
        "- `Stop Acc@100%`: `{delta:+.2f}` pts".format(
            delta=100.0 * (float(hybrid_agg["stop_acc@100%"]) - float(svd_agg["stop_acc@100%"]))
        ),
        "",
        "## Full-Train Export",
        "",
        f"- `method_name`: `{method_name}`",
        f"- `bundle`: `{_display_path(bundle_path)}`",
        f"- `submission`: `{_display_path(submission_path)}`",
        f"- `manifest`: `{_display_path(manifest_path)}`",
        "",
        "## Reading",
        "",
        "- The gain comes from a targeted domain swap rather than a new global model family.",
        "- `math` remains an early-strong linear/SVD domain, so the hybrid leaves it untouched.",
        "- `science` benefits from the stronger non-linear tree route while preserving the existing feature bank and holdout protocol.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build math-SVD + science-tree hybrid EarlyStop bundle")
    ap.add_argument("--selection-csv", default="results/tables/tree_baselines.csv")
    ap.add_argument("--math-bundle", default="models/ml_selectors/es_svd_math_rr_r2_20260412.pkl")
    ap.add_argument("--science-ref-bundle", default="models/ml_selectors/es_svd_science_rr_r2_20260412.pkl")
    ap.add_argument("--tree-fulltrain-bundle", default="results/models/tree_baselines_fulltrain_v1.pkl")
    ap.add_argument("--math-eval-json", default="results/scans/earlystop/es_svd_ms_rr_r2_20260412_eval.json")
    ap.add_argument("--main-cache-root", default="MUI_HUB/cache")
    ap.add_argument("--extra-cache-root", default="MUI_HUB/cache_train")
    ap.add_argument("--blind-cache-root", default="/home/jovyan/public-ro/MUI_HUB/cache_test")
    ap.add_argument("--feature-cache-dir", default="results/cache/tree_baselines_fulltrain")
    ap.add_argument("--blind-feature-cache-dir", default="results/cache/export_tree_baselines_submission")
    ap.add_argument("--refresh-feature-cache", action="store_true")
    ap.add_argument("--refresh-blind-feature-cache", action="store_true")
    ap.add_argument("--holdout-split", type=float, default=0.15)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--feature-workers", type=int, default=8)
    ap.add_argument("--feature-chunk-problems", type=int, default=8)
    ap.add_argument("--tree-threads", type=int, default=8)
    ap.add_argument("--seeds", default="42,101,202")
    ap.add_argument("--include-coding", action="store_true", help="Also carry tree fulltrain coding routes into export")
    ap.add_argument("--method-name", default="earlystop_math_svd_science_tree_hybrid_fulltrain_v1")
    ap.add_argument("--bundle-out", default="results/models/math_svd_science_tree_hybrid_fulltrain_v1.pkl")
    ap.add_argument("--submission-out", default="submission/EarlyStop/earlystop_math_svd_science_tree_hybrid_fulltrain_v1.json")
    ap.add_argument("--manifest-out", default="results/tables/math_svd_science_tree_hybrid_manifest.json")
    ap.add_argument("--holdout-out", default="results/scans/earlystop/math_svd_science_tree_hybrid_holdout_eval.json")
    ap.add_argument("--doc-out", default="docs/MATH_SVD_SCIENCE_TREE_HYBRID.md")
    args = ap.parse_args()

    selection_csv = REPO_ROOT / str(args.selection_csv)
    math_bundle_path = REPO_ROOT / str(args.math_bundle)
    science_ref_bundle_path = REPO_ROOT / str(args.science_ref_bundle)
    tree_fulltrain_path = REPO_ROOT / str(args.tree_fulltrain_bundle)
    math_eval_path = REPO_ROOT / str(args.math_eval_json)
    bundle_out = REPO_ROOT / str(args.bundle_out)
    submission_out = REPO_ROOT / str(args.submission_out)
    manifest_out = REPO_ROOT / str(args.manifest_out)
    holdout_out = REPO_ROOT / str(args.holdout_out)
    doc_out = REPO_ROOT / str(args.doc_out)

    seeds = tuple(int(v) for v in _parse_csv(args.seeds))
    if not seeds:
        seeds = (42, 101, 202)

    math_bundle = load_earlystop_svd_bundle(math_bundle_path)
    science_ref_bundle = load_earlystop_svd_bundle(science_ref_bundle_path)
    tree_fulltrain_bundle = load_earlystop_svd_bundle(tree_fulltrain_path)

    science_row = _load_best_tree_row(selection_csv, "science")
    search_summary = {
        "cv_auroc": float(science_row["search_cv_auroc"]),
        "search_positions": [
            int(v) for v in str(science_row["search_positions_pct"]).split(",") if str(v).strip()
        ],
        "config": json.loads(str(science_row["search_config_json"])),
    }

    required_train_features = _required_features_from_bundles(science_ref_bundle)
    feature_cache_dir = None
    if str(args.feature_cache_dir).strip().lower() not in {"", "none", "off"}:
        feature_cache_dir = (REPO_ROOT / str(args.feature_cache_dir)).resolve()

    labeled_store, labeled_cache_state = _load_labeled_feature_store(
        main_cache_root=_resolve_path(str(args.main_cache_root)),
        extra_cache_root=_resolve_path(str(args.extra_cache_root)),
        feature_cache_dir=feature_cache_dir,
        refresh_feature_cache=bool(args.refresh_feature_cache),
        feature_workers=int(args.feature_workers),
        feature_chunk_problems=int(args.feature_chunk_problems),
        required_feature_names=required_train_features,
    )

    science_store = [payload for payload in labeled_store if str(payload.get("domain")) == "science"]
    holdout_problem_map, holdout_problem_summary = _build_holdout_problem_map(
        science_store,
        holdout_split=float(args.holdout_split),
        split_seed=int(args.split_seed),
    )
    science_train_store, science_holdout_store, _science_full_store = _split_feature_store(
        science_store,
        holdout_problem_map=holdout_problem_map,
    )
    science_train_tables = _build_domain_training_tables(
        science_train_store,
        tuple(float(v) for v in EARLY_STOP_POSITIONS),
    )
    science_routes = _fit_domain_tree_routes(
        domain="science",
        train_tables=science_train_tables,
        reference_bundle=science_ref_bundle,
        model_family=str(science_row["model_family"]),
        feature_variant=str(science_row["feature_variant"]),
        best_config=search_summary,
        seeds=seeds,
        n_jobs=int(args.tree_threads),
    )
    science_splitfit_bundle = _build_tree_bundle(
        method_id="science_tree_splitfit_hybrid_proxy",
        domain_routes={"science": science_routes},
        protocol={
            "stage": "holdout_proxy",
            "holdout_split": float(args.holdout_split),
            "split_seed": int(args.split_seed),
            "source": "tree_baselines.csv:science_best",
            "seed_values": [int(v) for v in seeds],
        },
    )
    science_eval = evaluate_method_from_feature_store(
        method_name="science_tree_splitfit_hybrid_proxy",
        feature_store=science_holdout_store,
        position_values=tuple(float(v) for v in EARLY_STOP_POSITIONS),
        score_fn=make_tree_bundle_score_fn(science_splitfit_bundle),
    )

    math_eval = _load_math_eval(math_eval_path)
    svd_combined_eval = _load_svd_combined_eval(math_eval_path)
    hybrid_holdout = {
        "method_name": "math_svd_science_tree_hybrid_holdout_proxy",
        "aggregate": aggregate_cache_metrics(
            list(math_eval["by_cache"]) + list(science_eval["by_cache"])
        ),
        "by_cache": list(math_eval["by_cache"]) + list(science_eval["by_cache"]),
    }

    holdout_payload = {
        "created_at_utc": _now_utc(),
        "method_name": "math_svd_science_tree_hybrid_holdout_proxy",
        "math_eval_json": _display_path(math_eval_path),
        "tree_selection_csv": _display_path(selection_csv),
        "science_tree_selection": {
            "model_family": str(science_row["model_family"]),
            "feature_variant": str(science_row["feature_variant"]),
            "search_cv_auroc": float(science_row["search_cv_auroc"]),
            "search_positions_pct": search_summary["search_positions"],
            "search_config": search_summary["config"],
        },
        "science_holdout_problem_summary": holdout_problem_summary,
        "math_svd": math_eval,
        "science_tree": science_eval,
        "svd_combined_noncoding": svd_combined_eval,
        "hybrid_noncoding": hybrid_holdout,
    }
    _write_json(holdout_out, holdout_payload)

    hybrid_bundle = _build_hybrid_bundle(
        method_id=str(args.method_name),
        math_bundle=math_bundle,
        tree_bundle=tree_fulltrain_bundle,
        include_coding=bool(args.include_coding),
        protocol={
            "stage": "fulltrain_blind_export",
            "created_at_utc": _now_utc(),
            "math_source_bundle": _display_path(math_bundle_path),
            "science_source_bundle": _display_path(tree_fulltrain_path),
            "coding_source_bundle": _display_path(tree_fulltrain_path) if bool(args.include_coding) else None,
            "holdout_proxy_path": _display_path(holdout_out),
            "selection_csv": _display_path(selection_csv),
            "science_selection": {
                "model_family": str(science_row["model_family"]),
                "feature_variant": str(science_row["feature_variant"]),
                "holdout_auc_of_auroc": float(science_row["auc_of_auroc"]),
                "delta_auc_of_auroc_vs_svd": float(science_row["delta_auc_of_auroc_vs_svd"]),
                "search_config": search_summary["config"],
            },
            "seed_values": [int(v) for v in seeds],
        },
    )
    _save_pickle(bundle_out, hybrid_bundle)

    required_blind_features = collect_required_features(hybrid_bundle)
    blind_feature_cache_dir = None
    if str(args.blind_feature_cache_dir).strip().lower() not in {"", "none", "off"}:
        blind_feature_cache_dir = (REPO_ROOT / str(args.blind_feature_cache_dir)).resolve()

    blind_feature_store, blind_cache_path, blind_cache_status = _load_blind_feature_store(
        blind_cache_root=str(args.blind_cache_root),
        positions=tuple(float(v) for v in EARLY_STOP_POSITIONS),
        required_feature_names=required_blind_features,
        feature_workers=int(args.feature_workers),
        feature_chunk_problems=int(args.feature_chunk_problems),
        feature_cache_dir=blind_feature_cache_dir,
        refresh_feature_cache=bool(args.refresh_blind_feature_cache),
    )
    score_fn = make_mixed_bundle_score_fn(hybrid_bundle)
    score_map: dict[str, dict[str, list[float]]] = {}
    for payload in blind_feature_store:
        cache_key = str(payload.get("base_cache_key") or payload.get("cache_key"))
        score_map[cache_key] = _problem_scores_from_payload(payload, score_fn)

    entries = discover_cache_entries(str(args.blind_cache_root))
    expected_cache_keys = [str(entry.cache_key) for entry in entries]
    missing = [cache_key for cache_key in expected_cache_keys if cache_key not in score_map]
    if missing:
        raise ValueError(f"Missing blind scores for cache keys: {missing}")

    ordered_scores = [(cache_key, score_map[cache_key]) for cache_key in expected_cache_keys]
    submission_payload = build_earlystop_payload(ordered_scores, method_name=str(args.method_name))
    validation = validate_earlystop_payload(submission_payload)
    submission_out.parent.mkdir(parents=True, exist_ok=True)
    write_earlystop_payload(submission_payload, submission_out)

    manifest = {
        "created_at_utc": _now_utc(),
        "method_name": str(args.method_name),
        "math_bundle": _display_path(math_bundle_path),
        "tree_fulltrain_bundle": _display_path(tree_fulltrain_path),
        "selection_csv": _display_path(selection_csv),
        "science_selection_row": {
            "model_family": str(science_row["model_family"]),
            "feature_variant": str(science_row["feature_variant"]),
            "holdout_auc_of_auroc": float(science_row["auc_of_auroc"]),
            "delta_auc_of_auroc_vs_svd": float(science_row["delta_auc_of_auroc_vs_svd"]),
            "search_cv_auroc": float(science_row["search_cv_auroc"]),
            "search_positions_pct": search_summary["search_positions"],
            "search_config": search_summary["config"],
        },
        "holdout_proxy_path": _display_path(holdout_out),
        "holdout_proxy_delta_vs_svd": {
            "auc_of_auroc": float(hybrid_holdout["aggregate"]["auc_of_auroc"]) - float(svd_combined_eval["aggregate"]["auc_of_auroc"]),
            "auc_of_selacc": float(hybrid_holdout["aggregate"]["auc_of_selacc"]) - float(svd_combined_eval["aggregate"]["auc_of_selacc"]),
            "auroc@100%": float(hybrid_holdout["aggregate"]["auroc@100%"]) - float(svd_combined_eval["aggregate"]["auroc@100%"]),
            "stop_acc@100%": float(hybrid_holdout["aggregate"]["stop_acc@100%"]) - float(svd_combined_eval["aggregate"]["stop_acc@100%"]),
        },
        "train_cache_state": labeled_cache_state,
        "blind_cache_root": str(args.blind_cache_root),
        "blind_feature_cache_status": str(blind_cache_status),
        "blind_feature_cache_path": None if blind_cache_path is None else _display_path(blind_cache_path),
        "bundle_path": _display_path(bundle_out),
        "submission_path": _display_path(submission_out),
        "validation": validation,
        "include_coding": bool(args.include_coding),
        "seed_values": [int(v) for v in seeds],
    }
    _write_json(manifest_out, manifest)

    _render_doc(
        path=doc_out,
        method_name=str(args.method_name),
        math_bundle_path=math_bundle_path,
        tree_bundle_path=tree_fulltrain_path,
        science_row=science_row,
        math_eval=math_eval,
        science_eval=science_eval,
        hybrid_holdout=hybrid_holdout,
        svd_combined_eval=svd_combined_eval,
        bundle_path=bundle_out,
        submission_path=submission_out,
        manifest_path=manifest_out,
    )

    print(f"[holdout] {holdout_out}", flush=True)
    print(
        "[delta] noncoding hybrid vs svd "
        f"auc_of_auroc={100.0 * (float(hybrid_holdout['aggregate']['auc_of_auroc']) - float(svd_combined_eval['aggregate']['auc_of_auroc'])):+.2f} pts",
        flush=True,
    )
    print(f"[bundle] {bundle_out}", flush=True)
    print(f"[submission] {submission_out}", flush=True)
    print(f"[manifest] {manifest_out}", flush=True)
    print(f"[doc] {doc_out}", flush=True)
    print(f"[validate] {validation}", flush=True)


if __name__ == "__main__":
    main()
