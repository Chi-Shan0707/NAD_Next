#!/usr/bin/env python3
"""Refit selected tree baselines on all labeled data and export blind EarlyStop submission.

This script is intentionally separate from `train_tree_baselines.py`.

Workflow
--------
1. Read the documented holdout study in `results/tables/tree_baselines.csv`.
2. Freeze the best ID tree variant per domain (`math`, `science`, `coding`).
3. Refit those routes on all labeled data (`cache + cache_train`) with no holdout.
4. Score `cache_test` and write a standard EarlyStop submission JSON.

The motivation is practical rather than scientific: once the holdout study has
already recorded model selection, the blind export should use as much labeled
training data as possible.
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
from nad.ops.earlystop_svd import DEFAULT_REFLECTION_THRESHOLD, load_earlystop_svd_bundle
from scripts.export_earlystop_svd_submission import (
    _load_or_build_feature_store as _load_or_build_blind_feature_store,
    _problem_scores_from_payload,
)
from scripts.run_earlystop_prefix10_svd_round1 import _display_path
from scripts.run_structured_ood_suite import (
    _ensure_training_payload,
    _filter_feature_store,
    _load_prebuilt_feature_store,
)
from scripts.baselines.train_tree_baselines import (
    ALL_DOMAINS,
    DEFAULT_SEEDS,
    _build_tree_bundle,
    _current_bundle_paths,
    _fit_domain_tree_routes,
    _parse_csv,
    _required_features_from_bundles,
    make_tree_bundle_score_fn,
)
from SVDomain.train_es_svd_ms_rr_r1 import (
    _build_domain_training_tables,
    _load_or_build_qualified_feature_store,
    _resolve_path,
)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_best_selection_rows(
    *,
    csv_path: Path,
    domains: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    selected: dict[str, dict[str, Any]] = {}
    for domain in domains:
        candidates = [
            row
            for row in rows
            if str(row.get("domain")) == str(domain) and str(row.get("protocol")) == "id_grouped_85_15"
        ]
        if not candidates:
            raise ValueError(f"No ID rows found for domain={domain} in {csv_path}")
        selected[domain] = max(candidates, key=lambda row: float(row["auc_of_auroc"]))
    return selected


def _load_labeled_feature_store(
    *,
    main_cache_root: str,
    extra_cache_root: str,
    feature_cache_dir: Path | None,
    refresh_feature_cache: bool,
    feature_workers: int,
    feature_chunk_problems: int,
    required_feature_names: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stores: list[dict[str, Any]] = []
    state: dict[str, Any] = {}
    for source_name, cache_root in (("cache", main_cache_root), ("cache_train", extra_cache_root)):
        if not refresh_feature_cache:
            prebuilt_store, prebuilt_paths = _load_prebuilt_feature_store(str(source_name))
            if prebuilt_store:
                stores.extend(list(prebuilt_store))
                state[source_name] = {
                    "status": "loaded_prebuilt",
                    "paths": list(prebuilt_paths),
                    "cache_root": str(cache_root),
                }
                continue
        store, cache_path, cache_status = _load_or_build_qualified_feature_store(
            source_name=str(source_name),
            cache_root=str(cache_root),
            positions=tuple(float(v) for v in EARLY_STOP_POSITIONS),
            required_feature_names=required_feature_names,
            max_problems_per_cache=None,
            feature_workers=int(feature_workers),
            chunk_problems=int(feature_chunk_problems),
            feature_cache_dir=feature_cache_dir,
            refresh_feature_cache=bool(refresh_feature_cache),
        )
        fixed_store = [_ensure_training_payload(payload, source_name=str(source_name)) for payload in list(store)]
        stores.extend(fixed_store)
        state[source_name] = {
            "status": str(cache_status),
            "path": None if cache_path is None else _display_path(cache_path),
            "cache_root": str(cache_root),
        }
    return stores, state


def _load_blind_feature_store(
    *,
    blind_cache_root: str,
    positions: tuple[float, ...],
    required_feature_names: set[str],
    feature_workers: int,
    feature_chunk_problems: int,
    feature_cache_dir: Path | None,
    refresh_feature_cache: bool,
) -> tuple[list[dict[str, Any]], Path | None, str]:
    if not refresh_feature_cache:
        prebuilt_store, prebuilt_paths = _load_prebuilt_feature_store("cache_test")
        if prebuilt_store:
            return list(prebuilt_store), None, f"loaded_prebuilt:{','.join(prebuilt_paths)}"
    return _load_or_build_blind_feature_store(
        cache_root=str(blind_cache_root),
        positions=positions,
        required_feature_names=required_feature_names,
        max_problems=None,
        reflection_threshold=float(DEFAULT_REFLECTION_THRESHOLD),
        workers=int(feature_workers),
        feature_chunk_problems=int(feature_chunk_problems),
        feature_cache_dir=feature_cache_dir,
        refresh_feature_cache=bool(refresh_feature_cache),
    )


def _save_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_doc(
    *,
    path: Path,
    method_name: str,
    selection_csv: Path,
    bundle_path: Path,
    submission_path: Path,
    manifest_path: Path,
    selected_rows: dict[str, dict[str, Any]],
    seeds: tuple[int, ...],
    blind_cache_root: str,
) -> None:
    lines = [
        "# Tree Baselines Full-Train Blind Export",
        "",
        "## Purpose",
        "",
        "- This note records the blind-export stage that follows the documented holdout study in `docs/TREE_BASELINES.md`.",
        "- Model selection stays frozen from `results/tables/tree_baselines.csv`.",
        "- Final training uses all labeled data from `cache + cache_train` with **no holdout**.",
        "- The resulting bundle is exported on `cache_test` as a standard EarlyStop submission JSON.",
        "",
        "## Frozen holdout selections",
        "",
    ]
    for domain in ("math", "science", "coding"):
        row = selected_rows.get(domain)
        if row is None:
            continue
        lines.append(
            "- `{domain}`: `{model}` + `{variant}` | holdout `AUC of AUROC={auc:.2%}` | vs SVD `{svd:.2%}` | Δ `{delta:+.2f}` pts | config `{config}`".format(
                domain=domain,
                model=row["model_family"],
                variant=row["feature_variant"],
                auc=float(row["auc_of_auroc"]),
                svd=float(row["svd_auc_of_auroc"]),
                delta=100.0 * float(row["delta_auc_of_auroc_vs_svd"]),
                config=row["search_config_json"],
            )
        )
    lines.extend(
        [
            "",
            "## Full-train protocol",
            "",
            "- Labeled training sources: `cache` and `cache_train`.",
            "- No grouped holdout is carved out at this stage.",
            f"- Official positions: `{', '.join(str(int(round(100.0 * float(p)))) + '%' for p in EARLY_STOP_POSITIONS)}`.",
            f"- Seed ensemble: `{', '.join(str(v) for v in seeds)}`.",
            "- Feature bank stays identical to the paper-facing SVD route because every tree route reuses the reference route feature subset.",
            f"- Blind export target: `{blind_cache_root}`.",
            "",
            "## Artifacts",
            "",
            f"- Selection table: `{_display_path(selection_csv)}`",
            f"- Full-train bundle: `{_display_path(bundle_path)}`",
            f"- Submission JSON: `{_display_path(submission_path)}`",
            f"- Manifest JSON: `{_display_path(manifest_path)}`",
            "",
            "## Submission note",
            "",
            f"- `method_name`: `{method_name}`",
            "- This export is the right object to submit manually and use for external feedback, because it does not waste labeled training data on a local holdout.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Refit selected tree baselines on all labeled data and export blind EarlyStop submission")
    ap.add_argument("--selection-csv", default="results/tables/tree_baselines.csv")
    ap.add_argument("--main-cache-root", default="MUI_HUB/cache")
    ap.add_argument("--extra-cache-root", default="MUI_HUB/cache_train")
    ap.add_argument("--blind-cache-root", default="/home/jovyan/public-ro/MUI_HUB/cache_test")
    ap.add_argument("--domains", default="math,science,coding")
    ap.add_argument("--feature-cache-dir", default="results/cache/tree_baselines_fulltrain")
    ap.add_argument("--blind-feature-cache-dir", default="results/cache/export_tree_baselines_submission")
    ap.add_argument("--refresh-feature-cache", action="store_true")
    ap.add_argument("--refresh-blind-feature-cache", action="store_true")
    ap.add_argument("--feature-workers", type=int, default=8)
    ap.add_argument("--feature-chunk-problems", type=int, default=8)
    ap.add_argument("--tree-threads", type=int, default=8)
    ap.add_argument("--seeds", default="42,101,202")
    ap.add_argument("--method-name", default="earlystop_tree_baselines_fulltrain_v1")
    ap.add_argument("--filename", default="earlystop_tree_baselines_fulltrain_v1.json")
    ap.add_argument("--out-dir", default="submission/EarlyStop")
    ap.add_argument("--bundle-out", default="results/models/tree_baselines_fulltrain_v1.pkl")
    ap.add_argument("--manifest-out", default="results/tables/tree_baselines_fulltrain_submission_manifest.json")
    ap.add_argument("--doc-out", default="docs/TREE_BASELINES_FULLTRAIN_SUBMISSION.md")
    args = ap.parse_args()

    selection_csv = REPO_ROOT / str(args.selection_csv)
    requested_domains = tuple(v for v in _parse_csv(args.domains) if v in ALL_DOMAINS)
    if not requested_domains:
        requested_domains = ("math", "science", "coding")
    seeds = tuple(int(v) for v in _parse_csv(args.seeds)) or DEFAULT_SEEDS

    reference_bundles = {
        name: load_earlystop_svd_bundle(path)
        for name, path in _current_bundle_paths().items()
    }
    required_train_features = _required_features_from_bundles(
        reference_bundles["math"],
        reference_bundles["science"],
        reference_bundles["coding"],
    )

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

    selected_rows = _load_best_selection_rows(csv_path=selection_csv, domains=requested_domains)
    domain_routes: dict[str, dict[float, dict[str, Any]]] = {}
    selection_manifest: dict[str, Any] = {}

    for domain in requested_domains:
        row = selected_rows[domain]
        domain_store = _filter_feature_store(labeled_store, domain=domain)
        if not domain_store:
            raise ValueError(f"No labeled feature-store payloads found for domain={domain}")
        train_tables = _build_domain_training_tables(
            domain_store,
            tuple(float(v) for v in EARLY_STOP_POSITIONS),
        )
        search_summary = {
            "cv_auroc": float(row["search_cv_auroc"]),
            "search_positions": [int(v) for v in str(row["search_positions_pct"]).split(",") if str(v).strip()],
            "config": json.loads(str(row["search_config_json"])),
        }
        domain_routes[domain] = _fit_domain_tree_routes(
            domain=domain,
            train_tables=train_tables,
            reference_bundle=reference_bundles[domain],
            model_family=str(row["model_family"]),
            feature_variant=str(row["feature_variant"]),
            best_config=search_summary,
            seeds=seeds,
            n_jobs=int(args.tree_threads),
        )
        selection_manifest[domain] = {
            "model_family": str(row["model_family"]),
            "feature_variant": str(row["feature_variant"]),
            "holdout_auc_of_auroc": float(row["auc_of_auroc"]),
            "holdout_delta_auc_of_auroc_vs_svd": float(row["delta_auc_of_auroc_vs_svd"]),
            "search_cv_auroc": float(row["search_cv_auroc"]),
            "search_positions_pct": [int(v) for v in str(row["search_positions_pct"]).split(",") if str(v).strip()],
            "search_config": json.loads(str(row["search_config_json"])),
        }
        print(
            f"[select] {domain}: {row['model_family']} + {row['feature_variant']} "
            f"| holdout_auc={float(row['auc_of_auroc']):.4f}",
            flush=True,
        )

    bundle = _build_tree_bundle(
        method_id=str(args.method_name),
        domain_routes=domain_routes,
        protocol={
            "stage": "fulltrain_blind_export",
            "created_at_utc": _now_utc(),
            "selection_csv": _display_path(selection_csv),
            "selection_mode": "best_id_row_per_domain",
            "train_sources": ["cache", "cache_train"],
            "holdout_used_for_final_fit": False,
            "seed_values": [int(v) for v in seeds],
        },
    )

    bundle_path = REPO_ROOT / str(args.bundle_out)
    _save_pickle(bundle_path, bundle)
    print(f"[artifact] bundle={_display_path(bundle_path)}", flush=True)

    required_blind_features = _required_features_from_bundles(bundle)
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
    print(
        f"[blind-cache] status={blind_cache_status} path={None if blind_cache_path is None else _display_path(blind_cache_path)}",
        flush=True,
    )

    score_fn = make_tree_bundle_score_fn(bundle)
    score_map: dict[str, dict[str, list[float]]] = {}
    for payload in blind_feature_store:
        problem_scores = _problem_scores_from_payload(payload, score_fn)
        cache_key = str(payload.get("base_cache_key") or payload.get("cache_key"))
        score_map[cache_key] = problem_scores
        print(
            f"[blind] {cache_key} problems={len(problem_scores)} "
            f"samples={sum(len(v) for v in problem_scores.values())}",
            flush=True,
        )

    entries = discover_cache_entries(str(args.blind_cache_root))
    expected_cache_keys = [str(entry.cache_key) for entry in entries]
    missing = [cache_key for cache_key in expected_cache_keys if cache_key not in score_map]
    if missing:
        raise ValueError(f"Missing blind scores for cache keys: {missing}")

    ordered_scores = [(cache_key, score_map[cache_key]) for cache_key in expected_cache_keys]
    submission_payload = build_earlystop_payload(ordered_scores, method_name=str(args.method_name))
    validation = validate_earlystop_payload(submission_payload)

    submission_path = REPO_ROOT / str(args.out_dir) / str(args.filename)
    write_earlystop_payload(submission_payload, submission_path)
    print(f"[artifact] submission={_display_path(submission_path)}", flush=True)
    print(f"[validate] {validation}", flush=True)

    manifest_path = REPO_ROOT / str(args.manifest_out)
    manifest = {
        "created_at_utc": _now_utc(),
        "method_name": str(args.method_name),
        "selection_csv": _display_path(selection_csv),
        "selection_rows": selection_manifest,
        "train_cache_state": labeled_cache_state,
        "blind_cache_root": str(args.blind_cache_root),
        "blind_feature_cache_status": str(blind_cache_status),
        "blind_feature_cache_path": None if blind_cache_path is None else _display_path(blind_cache_path),
        "bundle_path": _display_path(bundle_path),
        "submission_path": _display_path(submission_path),
        "validation": validation,
        "seed_values": [int(v) for v in seeds],
    }
    _write_json(manifest_path, manifest)
    print(f"[artifact] manifest={_display_path(manifest_path)}", flush=True)

    doc_path = REPO_ROOT / str(args.doc_out)
    _write_doc(
        path=doc_path,
        method_name=str(args.method_name),
        selection_csv=selection_csv,
        bundle_path=bundle_path,
        submission_path=submission_path,
        manifest_path=manifest_path,
        selected_rows=selected_rows,
        seeds=seeds,
        blind_cache_root=str(args.blind_cache_root),
    )
    print(f"[artifact] doc={_display_path(doc_path)}", flush=True)


if __name__ == "__main__":
    main()
