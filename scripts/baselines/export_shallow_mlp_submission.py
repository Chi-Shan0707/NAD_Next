#!/usr/bin/env python3
"""Train full-fit shallow MLP domain models and export a cache_test submission."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from nad.ops.earlystop import (  # noqa: E402
    EARLY_STOP_POSITIONS,
    build_earlystop_payload,
    discover_cache_entries,
    validate_earlystop_payload,
    write_earlystop_payload,
)
from nad.ops.earlystop_svd import FULL_FEATURE_NAMES, get_domain  # noqa: E402
from scripts.baselines.train_shallow_mlp_baselines import (  # noqa: E402
    ARTIFACT_FEATURE_NAMES,
    DEFAULT_EXTRA_CACHE_ROOT,
    DEFAULT_FEATURE_CACHE_DIR,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_MIN_DELTA,
    DEFAULT_PATIENCE,
    DEFAULT_SPLIT_SEED,
    DEFAULT_VAL_SPLIT,
    _build_domain_training_tables,
    _display_path,
    _filter_feature_store,
    _load_or_build_qualified_feature_store,
    _make_mlp_score_fn,
    _search_position_config,
    _summarise_feature_store,
    _train_scope_family_seed_bundle,
)
from scripts.export_earlystop_svd_submission import (  # noqa: E402
    _load_or_build_feature_store,
    _problem_scores_from_payload,
)
from scripts.run_earlystop_prefix10_svd_round1 import EXTRACTION_POSITIONS  # noqa: E402


DEFAULT_MAIN_CACHE_ROOT = "MUI_HUB/cache"
DEFAULT_TEST_CACHE_ROOT = "/home/jovyan/public-ro/MUI_HUB/cache_test"
DEFAULT_SELECTION_CSV = "results/tables/shallow_mlp_baselines.csv"
DEFAULT_OUT_MODEL = "models/ml_selectors/shallow_mlp_submission_20260413.pkl"
DEFAULT_OUT_JSON = "submission/EarlyStop/shallow_mlp_submission_20260413.json"
DEFAULT_OUT_SUMMARY = "results/scans/earlystop/shallow_mlp_submission_20260413_summary.json"
DEFAULT_BLIND_FEATURE_CACHE_DIR = "results/cache/export_shallow_mlp_submission"
DEFAULT_METHOD_NAME = "shallow_mlp_submission_20260413"
DEFAULT_DOMAINS = ("math", "science", "coding")
DEFAULT_FALLBACK_FAMILY = "mlp_2h"
DEFAULT_FALLBACK_SEED = 42
DEFAULT_REPRESENTATION = "raw+rank"
DEFAULT_WORKERS = 4
DEFAULT_FEATURE_CHUNK_PROBLEMS = 24
DEFAULT_BLIND_FEATURE_CHUNK_PROBLEMS = 8
DEFAULT_BLIND_REFLECTION_THRESHOLD = 0.30
PREBUILT_BLIND_FEATURE_CACHE_DIRS = (
    REPO_ROOT / "results" / "cache" / "export_earlystop_svd_submission_round1c_override",
    REPO_ROOT / "results" / "cache" / "export_earlystop_svd_submission_es_svd_ms_rr_r1",
    REPO_ROOT / "results" / "cache" / "export_earlystop_svd_submission_strongfeat_20260410",
)


def _parse_domain_assignments(raw: str) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for item in [part.strip() for part in str(raw).split(",") if part.strip()]:
        if "=" not in item:
            raise ValueError(f"Expected domain=value entry, got: {item}")
        domain, value = item.split("=", 1)
        domain = str(domain).strip()
        value = str(value).strip()
        if not domain or not value:
            raise ValueError(f"Invalid domain=value entry: {item}")
        assignments[domain] = value
    return assignments


def _load_selection_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _select_domain_families_and_seeds(
    *,
    selection_csv: Path,
    family_overrides: dict[str, str],
    seed_overrides: dict[str, int],
) -> tuple[dict[str, str], dict[str, int]]:
    rows = _load_selection_rows(selection_csv)
    chosen_families: dict[str, str] = {}
    chosen_seeds: dict[str, int] = {}

    for domain in DEFAULT_DOMAINS:
        if domain in family_overrides:
            chosen_families[domain] = str(family_overrides[domain])
        else:
            best_rows = [
                row for row in rows
                if str(row.get("row_kind")) == "mlp_best_mean" and str(row.get("domain")) == domain
            ]
            if best_rows:
                chosen_families[domain] = str(best_rows[0]["family_name"])
            else:
                chosen_families[domain] = DEFAULT_FALLBACK_FAMILY

        if domain in seed_overrides:
            chosen_seeds[domain] = int(seed_overrides[domain])
            continue

        seed_rows = [
            row for row in rows
            if str(row.get("row_kind")) in {"mlp_seed_best_family", "mlp_seed"}
            and str(row.get("domain")) == domain
            and str(row.get("family_name")) == chosen_families[domain]
        ]
        best_row: dict[str, Any] | None = None
        for row in seed_rows:
            try:
                auc = float(row["auc_of_auroc"])
            except Exception:
                continue
            if best_row is None or auc > float(best_row["auc_of_auroc"]):
                best_row = row
        chosen_seeds[domain] = int(best_row["seed"]) if best_row is not None else DEFAULT_FALLBACK_SEED

    return chosen_families, chosen_seeds


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _ensure_cv_group_keys(feature_store: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for payload in feature_store:
        if "cv_group_keys" in payload:
            out.append(payload)
            continue
        offsets = [int(v) for v in payload["problem_offsets"]]
        problem_ids = [str(v) for v in payload["problem_ids"]]
        parts: list[np.ndarray] = []
        for problem_idx, problem_id in enumerate(problem_ids):
            width = max(0, offsets[problem_idx + 1] - offsets[problem_idx])
            if width <= 0:
                continue
            parts.append(np.asarray([f"{payload['dataset_name']}::{problem_id}"] * width, dtype=object))
        item = dict(payload)
        item["cv_group_keys"] = np.concatenate(parts).astype(object, copy=False) if parts else np.asarray([], dtype=object)
        out.append(item)
    return out


def _positions_match(lhs: list[float] | tuple[float, ...], rhs: list[float] | tuple[float, ...]) -> bool:
    lhs_arr = np.asarray(lhs, dtype=np.float64)
    rhs_arr = np.asarray(rhs, dtype=np.float64)
    return lhs_arr.shape == rhs_arr.shape and bool(np.allclose(lhs_arr, rhs_arr, atol=1e-12, rtol=0.0))


def _load_prebuilt_blind_feature_store(
    *,
    cache_root: str,
    positions: tuple[float, ...],
    min_feature_dim: int,
) -> tuple[list[dict[str, Any]] | None, list[Path]]:
    merged_by_cache: dict[str, dict[str, Any]] = {}
    used_paths: list[Path] = []
    for cache_dir in PREBUILT_BLIND_FEATURE_CACHE_DIRS:
        if not cache_dir.exists():
            continue
        for candidate in sorted(cache_dir.glob("feature_store_all_ref030_*.pkl")):
            try:
                with candidate.open("rb") as handle:
                    payload = pickle.load(handle)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            if str(payload.get("cache_root", "")) != str(cache_root):
                continue
            if not _positions_match(payload.get("positions", ()), positions):
                continue
            feature_store = payload.get("feature_store")
            if not isinstance(feature_store, list) or not feature_store:
                continue
            tensor = np.asarray(feature_store[0].get("tensor"))
            if tensor.ndim != 3 or int(tensor.shape[-1]) < int(min_feature_dim):
                continue
            used_paths.append(candidate)
            for feature_payload in feature_store:
                merged_by_cache.setdefault(str(feature_payload["cache_key"]), feature_payload)
    if not merged_by_cache:
        return None, []
    merged_store = [merged_by_cache[key] for key in sorted(merged_by_cache.keys())]
    return merged_store, used_paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Train full-fit shallow MLP routes and export a cache_test submission")
    ap.add_argument("--cache-root", default=DEFAULT_MAIN_CACHE_ROOT)
    ap.add_argument("--extra-cache-root", default=DEFAULT_EXTRA_CACHE_ROOT)
    ap.add_argument("--test-cache-root", default=DEFAULT_TEST_CACHE_ROOT)
    ap.add_argument("--selection-csv", default=DEFAULT_SELECTION_CSV)
    ap.add_argument("--domain-families", default="", help="Optional overrides like math=mlp_2h,science=mlp_2h,coding=mlp_2h")
    ap.add_argument("--domain-seeds", default="", help="Optional overrides like math=42,science=42,coding=42")
    ap.add_argument("--feature-cache-dir", default=DEFAULT_FEATURE_CACHE_DIR)
    ap.add_argument("--blind-feature-cache-dir", default=DEFAULT_BLIND_FEATURE_CACHE_DIR)
    ap.add_argument("--refresh-feature-cache", action="store_true")
    ap.add_argument("--refresh-blind-feature-cache", action="store_true")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--feature-chunk-problems", type=int, default=DEFAULT_FEATURE_CHUNK_PROBLEMS)
    ap.add_argument("--blind-feature-chunk-problems", type=int, default=DEFAULT_BLIND_FEATURE_CHUNK_PROBLEMS)
    ap.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    ap.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    ap.add_argument("--val-split", type=float, default=DEFAULT_VAL_SPLIT)
    ap.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA)
    ap.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    ap.add_argument("--blind-reflection-threshold", type=float, default=DEFAULT_BLIND_REFLECTION_THRESHOLD)
    ap.add_argument("--out-model", default=DEFAULT_OUT_MODEL)
    ap.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    ap.add_argument("--out-summary", default=DEFAULT_OUT_SUMMARY)
    ap.add_argument("--method-name", default=DEFAULT_METHOD_NAME)
    ap.add_argument("--existing-model", default="", help="Optional pre-trained bundle to reuse; skips full-fit training when set")
    args = ap.parse_args()

    main_cache_root = str((REPO_ROOT / args.cache_root).resolve()) if not Path(args.cache_root).is_absolute() else str(Path(args.cache_root).resolve())
    extra_cache_root = str((REPO_ROOT / args.extra_cache_root).resolve()) if not Path(args.extra_cache_root).is_absolute() else str(Path(args.extra_cache_root).resolve())
    test_cache_root = str((REPO_ROOT / args.test_cache_root).resolve()) if not Path(args.test_cache_root).is_absolute() else str(Path(args.test_cache_root).resolve())

    selection_csv = Path(args.selection_csv)
    if not selection_csv.is_absolute():
        selection_csv = REPO_ROOT / selection_csv
    if not selection_csv.exists():
        raise SystemExit(f"Selection CSV not found: {selection_csv}")

    feature_cache_dir = Path(args.feature_cache_dir)
    if not feature_cache_dir.is_absolute():
        feature_cache_dir = REPO_ROOT / feature_cache_dir
    blind_feature_cache_dir = Path(args.blind_feature_cache_dir)
    if not blind_feature_cache_dir.is_absolute():
        blind_feature_cache_dir = REPO_ROOT / blind_feature_cache_dir
    existing_model_raw = str(args.existing_model).strip()
    existing_model = None if existing_model_raw.lower() in {"", "none", "off"} else Path(existing_model_raw)
    if existing_model is not None and not existing_model.is_absolute():
        existing_model = REPO_ROOT / existing_model

    family_overrides = _parse_domain_assignments(args.domain_families) if str(args.domain_families).strip() else {}
    seed_overrides_raw = _parse_domain_assignments(args.domain_seeds) if str(args.domain_seeds).strip() else {}
    seed_overrides = {domain: int(value) for domain, value in seed_overrides_raw.items()}
    chosen_families: dict[str, str]
    chosen_seeds: dict[str, int]
    train_positions = tuple(float(v) for v in EXTRACTION_POSITIONS)
    blind_positions = tuple(float(v) for v in EARLY_STOP_POSITIONS)
    main_cache_path: Path | None = None
    extra_cache_path: Path | None = None
    main_cache_status = "skipped_existing_bundle"
    extra_cache_status = "skipped_existing_bundle"
    domain_route_summaries: dict[str, Any] = {}
    domain_training_summaries: dict[str, Any] = {}

    out_model = Path(args.out_model)
    if not out_model.is_absolute():
        out_model = REPO_ROOT / out_model

    if existing_model is not None:
        if not existing_model.exists():
            raise SystemExit(f"Existing model not found: {existing_model}")
        with existing_model.open("rb") as handle:
            combined_bundle = pickle.load(handle)
        chosen_families = {domain: str(combined_bundle.get("domain_families", {}).get(domain, DEFAULT_FALLBACK_FAMILY)) for domain in DEFAULT_DOMAINS}
        chosen_seeds = {domain: int(combined_bundle.get("domain_seeds", {}).get(domain, DEFAULT_FALLBACK_SEED)) for domain in DEFAULT_DOMAINS}
        out_model = existing_model
        print(f"[shallow-mlp-submit] reusing bundle={_display_path(out_model)}")
    else:
        chosen_families, chosen_seeds = _select_domain_families_and_seeds(
            selection_csv=selection_csv,
            family_overrides=family_overrides,
            seed_overrides=seed_overrides,
        )
        print("[shallow-mlp-submit] selected families/seeds")
        for domain in DEFAULT_DOMAINS:
            print(f"  domain={domain} family={chosen_families[domain]} seed={chosen_seeds[domain]}")

        required_feature_names = tuple(str(name) for name in ARTIFACT_FEATURE_NAMES)
        main_store, main_cache_path, main_cache_status = _load_or_build_qualified_feature_store(
            source_name="cache",
            cache_root=main_cache_root,
            positions=train_positions,
            required_feature_names=required_feature_names,
            max_problems_per_cache=None,
            max_workers=int(args.workers),
            chunk_problems=int(args.feature_chunk_problems),
            feature_cache_dir=feature_cache_dir,
            refresh_feature_cache=bool(args.refresh_feature_cache),
        )
        extra_store, extra_cache_path, extra_cache_status = _load_or_build_qualified_feature_store(
            source_name="cache_train",
            cache_root=extra_cache_root,
            positions=train_positions,
            required_feature_names=required_feature_names,
            max_problems_per_cache=None,
            max_workers=int(args.workers),
            chunk_problems=int(args.feature_chunk_problems),
            feature_cache_dir=feature_cache_dir,
            refresh_feature_cache=bool(args.refresh_feature_cache),
        )
        full_store = _ensure_cv_group_keys(list(main_store) + list(extra_store))

        domain_bundles: dict[str, dict[str, Any]] = {}
        for domain in DEFAULT_DOMAINS:
            scope_store = _filter_feature_store(full_store, {domain})
            if not scope_store:
                raise SystemExit(f"No labeled feature store found for domain={domain}")
            domain_training_summaries[domain] = _summarise_feature_store(scope_store)
            position_tables = _build_domain_training_tables(scope_store, tuple(float(v) for v in EARLY_STOP_POSITIONS))
            tuning_routes: list[dict[str, Any]] = []
            for pos_idx, position in enumerate(EARLY_STOP_POSITIONS):
                table = position_tables[pos_idx]
                route = _search_position_config(
                    x_raw=table["x_raw"],
                    x_rank=table["x_rank"],
                    y=table["y"],
                    groups=table["groups"],
                    representation=DEFAULT_REPRESENTATION,
                    family_name=chosen_families[domain],
                    tuning_seed=chosen_seeds[domain],
                    max_epochs=int(args.max_epochs),
                    patience=int(args.patience),
                    min_delta=float(args.min_delta),
                    val_fraction=float(args.val_split),
                )
                tuning_routes.append(route)
                print(
                    f"[shallow-mlp-submit] tuned domain={domain} pos={int(round(position * 100.0))}% "
                    f"route={route['route_type']} val_auc={float(route.get('val_auroc', float('nan'))):.4f}"
                )

            bundle = _train_scope_family_seed_bundle(
                scope_name=domain,
                family_name=chosen_families[domain],
                representation=DEFAULT_REPRESENTATION,
                seed=chosen_seeds[domain],
                tuning_routes=tuning_routes,
                position_tables=position_tables,
                max_epochs=int(args.max_epochs),
                patience=int(args.patience),
                min_delta=float(args.min_delta),
                val_fraction=float(args.val_split),
            )
            domain_bundles[domain] = bundle
            domain_route_summaries[domain] = {
                f"{int(round(float(position) * 100.0))}%": {
                    "route_type": str(route["route_type"]),
                    "hidden_layers": list(route.get("hidden_layers", [])),
                    "alpha": route.get("alpha", ""),
                    "batch_size": route.get("batch_size", ""),
                    "best_epoch": int(route.get("best_epoch", 0)),
                    "val_auroc": float(route.get("val_auroc", float("nan"))),
                    "fallback_signal": route.get("signal_name", ""),
                }
                for position, route in zip(EARLY_STOP_POSITIONS, bundle["domains"][domain]["routes"])
            }

        combined_bundle = {
            "bundle_version": "shallow_mlp_submission_v1",
            "created_at_utc": _now_utc(),
            "method_name": str(args.method_name),
            "positions": [float(v) for v in EARLY_STOP_POSITIONS],
            "feature_names": list(FULL_FEATURE_NAMES),
            "artifact_feature_names": list(ARTIFACT_FEATURE_NAMES),
            "representation": DEFAULT_REPRESENTATION,
            "domain_families": {domain: str(chosen_families[domain]) for domain in DEFAULT_DOMAINS},
            "domain_seeds": {domain: int(chosen_seeds[domain]) for domain in DEFAULT_DOMAINS},
            "domains": {domain: domain_bundles[domain]["domains"][domain] for domain in DEFAULT_DOMAINS},
        }

        out_model.parent.mkdir(parents=True, exist_ok=True)
        with out_model.open("wb") as handle:
            pickle.dump(combined_bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[shallow-mlp-submit] saved bundle={_display_path(out_model)}")

    score_fn = _make_mlp_score_fn(combined_bundle)
    blind_feature_store: list[dict[str, Any]]
    blind_cache_path: Path | None
    blind_cache_status: str
    blind_supplemental_paths: list[str] = []
    if not bool(args.refresh_blind_feature_cache):
        blind_feature_store, blind_prebuilt_paths = _load_prebuilt_blind_feature_store(
            cache_root=test_cache_root,
            positions=blind_positions,
            min_feature_dim=len(ARTIFACT_FEATURE_NAMES),
        )
        if blind_feature_store is not None:
            blind_cache_status = "prebuilt"
            blind_cache_path = blind_prebuilt_paths[0] if blind_prebuilt_paths else None
            blind_supplemental_paths.extend(_display_path(path) for path in blind_prebuilt_paths[1:])
        else:
            blind_feature_store = []
            blind_cache_status = ""
            blind_cache_path = None
    else:
        blind_feature_store = []
        blind_cache_status = ""
        blind_cache_path = None

    if not blind_feature_store:
        blind_feature_store, blind_cache_path, blind_cache_status = _load_or_build_feature_store(
            cache_root=test_cache_root,
            positions=blind_positions,
            required_feature_names=set(str(name) for name in ARTIFACT_FEATURE_NAMES),
            max_problems=None,
            reflection_threshold=float(args.blind_reflection_threshold),
            workers=int(args.workers),
            feature_chunk_problems=int(args.blind_feature_chunk_problems),
            feature_cache_dir=blind_feature_cache_dir,
            refresh_feature_cache=bool(args.refresh_blind_feature_cache),
        )
    print(
        f"[shallow-mlp-submit] blind features status={blind_cache_status} "
        f"path={None if blind_cache_path is None else _display_path(blind_cache_path)}"
    )

    score_map: dict[str, dict[str, dict[str, list[float]]]] = {}
    for payload in blind_feature_store:
        cache_key = str(payload["cache_key"])
        problem_scores = _problem_scores_from_payload(payload, score_fn)
        score_map[cache_key] = problem_scores
        print(
            f"[shallow-mlp-submit] scored cache={cache_key} domain={payload['domain']} "
            f"problems={len(problem_scores)} samples={sum(len(v) for v in problem_scores.values())}"
        )

    entries = discover_cache_entries(test_cache_root)
    expected_cache_keys = [str(entry.cache_key) for entry in entries]
    missing_cache_keys = [cache_key for cache_key in expected_cache_keys if cache_key not in score_map]
    if missing_cache_keys:
        supplemental_store, supplemental_cache_path, supplemental_status = _load_or_build_feature_store(
            cache_root=test_cache_root,
            positions=blind_positions,
            required_feature_names=set(str(name) for name in ARTIFACT_FEATURE_NAMES),
            max_problems=None,
            reflection_threshold=float(args.blind_reflection_threshold),
            workers=int(args.workers),
            feature_chunk_problems=int(args.blind_feature_chunk_problems),
            feature_cache_dir=blind_feature_cache_dir,
            refresh_feature_cache=bool(args.refresh_blind_feature_cache),
            include_cache_keys=set(missing_cache_keys),
        )
        blind_cache_status = f"{blind_cache_status}+supplemental_{supplemental_status}" if blind_cache_status else f"supplemental_{supplemental_status}"
        if supplemental_cache_path is not None:
            blind_supplemental_paths.append(_display_path(supplemental_cache_path))
        for payload in supplemental_store:
            cache_key = str(payload["cache_key"])
            problem_scores = _problem_scores_from_payload(payload, score_fn)
            score_map[cache_key] = problem_scores
            print(
                f"[shallow-mlp-submit] scored supplemental cache={cache_key} domain={payload['domain']} "
                f"problems={len(problem_scores)} samples={sum(len(v) for v in problem_scores.values())}"
            )
        missing_cache_keys = [cache_key for cache_key in expected_cache_keys if cache_key not in score_map]
        if missing_cache_keys:
            raise SystemExit(f"Missing blind cache scores for: {missing_cache_keys}")

    submission_payload = build_earlystop_payload(
        [(cache_key, score_map[cache_key]) for cache_key in expected_cache_keys],
        method_name=str(args.method_name),
    )
    submission_stats = validate_earlystop_payload(submission_payload)

    out_json = Path(args.out_json)
    if not out_json.is_absolute():
        out_json = REPO_ROOT / out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    write_earlystop_payload(submission_payload, out_json)
    print(f"[shallow-mlp-submit] wrote submission={_display_path(out_json)}")
    print(f"[shallow-mlp-submit] submission stats={submission_stats}")

    out_summary = Path(args.out_summary)
    if not out_summary.is_absolute():
        out_summary = REPO_ROOT / out_summary
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "created_at_utc": _now_utc(),
        "selection_csv": _display_path(selection_csv),
        "cache_roots": {
            "cache": main_cache_root,
            "cache_train": extra_cache_root,
            "cache_test": test_cache_root,
        },
        "feature_cache": {
            "labeled_main_status": str(main_cache_status),
            "labeled_main_path": None if main_cache_path is None else _display_path(main_cache_path),
            "labeled_extra_status": str(extra_cache_status),
            "labeled_extra_path": None if extra_cache_path is None else _display_path(extra_cache_path),
            "blind_status": str(blind_cache_status),
            "blind_path": None if blind_cache_path is None else _display_path(blind_cache_path),
            "blind_supplemental_paths": blind_supplemental_paths,
        },
        "selection": {
            "families": {domain: str(chosen_families[domain]) for domain in DEFAULT_DOMAINS},
            "seeds": {domain: int(chosen_seeds[domain]) for domain in DEFAULT_DOMAINS},
            "representation": DEFAULT_REPRESENTATION,
            "train_positions": list(train_positions),
            "blind_positions": list(blind_positions),
            "training_skipped_via_existing_model": bool(existing_model is not None),
        },
        "training": {
            "store_summary_by_domain": domain_training_summaries,
            "route_summary_by_domain": domain_route_summaries,
            "max_epochs": int(args.max_epochs),
            "patience": int(args.patience),
            "val_split": float(args.val_split),
            "min_delta": float(args.min_delta),
        },
        "artifacts": {
            "bundle_path": _display_path(out_model),
            "submission_path": _display_path(out_json),
            "summary_path": _display_path(out_summary),
        },
        "submission_validation": submission_stats,
    }
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[shallow-mlp-submit] wrote summary={_display_path(out_summary)}")


if __name__ == "__main__":
    main()
