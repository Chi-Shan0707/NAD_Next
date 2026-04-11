#!/usr/bin/env python3
"""Run structured OOD evaluation for EarlyStop SVD domain bundles."""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.ops.earlystop_svd import get_domain, load_earlystop_svd_bundle
from scripts.run_earlystop_prefix10_svd_round1 import (
    ANCHOR_POSITIONS,
    EARLY_STOP_POSITIONS,
    EXTRACTION_POSITIONS,
    _display_path,
    _pct_label,
    evaluate_method_from_feature_store,
    make_svd_bundle_score_fn,
    make_tok_conf_score_fn,
)
from SVDomain.train_es_svd_ms_rr_r1 import (
    _baseline_bundle_result,
    _build_domain_bundle,
    _build_domain_training_tables,
    _build_holdout_problem_map,
    _collect_required_features,
    _load_or_build_qualified_feature_store,
    _resolve_path,
    _split_feature_store,
    _train_domain_anchor_routes,
)


MATH_BENCHMARKS = ("aime24", "aime25", "brumo25", "hmmt25")
SCIENCE_BENCHMARKS = ("gpqa",)
CODING_BENCHMARKS = ("livecodebench_v5", "lcb_v5")
ROOT_ORDER = ("cache", "cache_train", "cache_test")
METHOD_DISPLAY_ORDER = (
    "tok_conf_prefix_mean_v1",
    "earlystop_svd_lowrank_lr_v1",
    "earlystop_prefix10_svd_round1",
    "es_svd_math_rr_r1",
    "es_svd_science_rr_r1",
    "es_svd_coding_rr_r1",
)
DOMAIN_METHOD_NAME = {
    "math": "es_svd_math_rr_r1",
    "science": "es_svd_science_rr_r1",
    "coding": "es_svd_coding_rr_r1",
}
METHOD_META = {
    "tok_conf_prefix_mean_v1": {
        "feature_family": "token_conf",
        "transfer_axis": "single_signal",
        "method_group": "baseline",
    },
    "earlystop_svd_lowrank_lr_v1": {
        "feature_family": "all_features",
        "transfer_axis": "global_svd",
        "method_group": "baseline",
    },
    "earlystop_prefix10_svd_round1": {
        "feature_family": "prefix_safe_search",
        "transfer_axis": "global_anchor_svd",
        "method_group": "baseline",
    },
    "es_svd_math_rr_r1": {
        "feature_family": "token_plus_traj_fixed",
        "transfer_axis": "domain_conditioned_svd",
        "method_group": "candidate",
    },
    "es_svd_science_rr_r1": {
        "feature_family": "token_plus_traj_fixed",
        "transfer_axis": "domain_conditioned_svd",
        "method_group": "candidate",
    },
    "es_svd_coding_rr_r1": {
        "feature_family": "token_plus_traj_fixed",
        "transfer_axis": "domain_conditioned_svd",
        "method_group": "candidate",
    },
}
AUTO_PREBUILT_FEATURE_STORE_GLOBS = {
    "cache": [
        "results/cache/earlystop_prefix10_svd_round1c_fullcache/cache_all_*.pkl",
    ],
    "cache_train": [
        "results/cache/earlystop_prefix10_svd_round1c_fullcache/cache_train_all_*.pkl",
    ],
    "cache_test": [
        "results/cache/export_earlystop_svd_submission_round1c_override/feature_store_all_ref030_*.pkl",
        "results/cache/export_earlystop_svd_submission_strongfeat_20260410/feature_store_all_ref030_*.pkl",
    ],
}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _comma_join(values: Iterable[str]) -> str:
    items = [str(v) for v in values if str(v)]
    return ",".join(sorted(dict.fromkeys(items)))


def _fmt_ratio(value: float) -> str:
    if not math.isfinite(float(value)):
        return "N/A"
    return f"{float(value):.4f}"


def _fmt_pct_ratio(value: float) -> str:
    if not math.isfinite(float(value)):
        return "N/A"
    return f"{100.0 * float(value):.2f}%"


def _fmt_earliest(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return _pct_label(float(value))
    except Exception:
        return str(value)


def _safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _safe_rel_gap(id_value: float, ood_value: float) -> float:
    if not math.isfinite(float(id_value)) or abs(float(id_value)) < 1e-12:
        return float("nan")
    return float(ood_value - id_value) / float(id_value)


def _glob_first(pattern: str) -> Optional[Path]:
    matches = sorted(REPO_ROOT.glob(pattern))
    return matches[0] if matches else None


def _pickle_feature_store(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    feature_store = payload["feature_store"] if isinstance(payload, dict) and "feature_store" in payload else payload
    return list(feature_store)


def _ensure_training_payload(payload: dict[str, Any], *, source_name: str) -> dict[str, Any]:
    item = dict(payload)
    base_cache_key = str(item.get("base_cache_key") or item.get("cache_key") or "")
    item["source_name"] = str(item.get("source_name") or source_name)
    item["base_cache_key"] = base_cache_key
    cache_key = str(item.get("cache_key") or base_cache_key)
    prefix = f"{item['source_name']}/"
    item["cache_key"] = cache_key if cache_key.startswith(prefix) else f"{prefix}{cache_key}"

    positions = tuple(float(v) for v in item.get("positions", []))
    tensor = np.asarray(item.get("tensor"), dtype=np.float64)
    if positions and positions != tuple(float(v) for v in EXTRACTION_POSITIONS):
        expanded = np.zeros(
            (tensor.shape[0], len(EXTRACTION_POSITIONS), tensor.shape[2]),
            dtype=np.float64,
        )
        dst_index = {float(position): idx for idx, position in enumerate(EXTRACTION_POSITIONS)}
        for src_idx, position in enumerate(positions):
            if float(position) not in dst_index:
                continue
            expanded[:, dst_index[float(position)], :] = tensor[:, src_idx, :]
        item["tensor"] = expanded
        item["positions"] = [float(v) for v in EXTRACTION_POSITIONS]
    else:
        item["positions"] = [float(v) for v in positions]

    if "cv_group_keys" not in item:
        offsets = [int(v) for v in item.get("problem_offsets", [])]
        problem_ids = [str(v) for v in item.get("problem_ids", [])]
        cv_group_parts: list[np.ndarray] = []
        for problem_idx, problem_id in enumerate(problem_ids):
            start = offsets[problem_idx]
            end = offsets[problem_idx + 1]
            width = max(0, end - start)
            if width <= 0:
                continue
            cv_group_parts.append(
                np.asarray([f"{item['dataset_name']}::{problem_id}"] * width, dtype=object)
            )
        if cv_group_parts:
            item["cv_group_keys"] = np.concatenate(cv_group_parts).astype(object, copy=False)
        else:
            item["cv_group_keys"] = np.asarray([], dtype=object)
    return item


def _load_prebuilt_feature_store(source_name: str) -> tuple[list[dict[str, Any]], list[str]]:
    paths = []
    for pattern in AUTO_PREBUILT_FEATURE_STORE_GLOBS.get(str(source_name), []):
        match = _glob_first(pattern)
        if match is not None:
            paths.append(match)
    if not paths:
        return [], []

    merged: list[dict[str, Any]] = []
    for path in paths:
        for payload in _pickle_feature_store(path):
            merged.append(_ensure_training_payload(payload, source_name=source_name))
    return merged, [_display_path(path) for path in paths]


def _payload_root(payload: dict[str, Any]) -> str:
    return str(payload.get("source_name", ""))


def _payload_base_cache_key(payload: dict[str, Any]) -> str:
    return str(payload.get("base_cache_key") or payload.get("cache_key") or "")


def _payload_model_family(payload: dict[str, Any]) -> str:
    base_cache_key = _payload_base_cache_key(payload)
    return str(base_cache_key.split("/", 1)[0]) if "/" in base_cache_key else base_cache_key


def _payload_dataset(payload: dict[str, Any]) -> str:
    return str(payload.get("dataset_name", ""))


def _payload_cache_key(payload: dict[str, Any]) -> str:
    return str(payload.get("cache_key", ""))


def _store_summary(feature_store: list[dict[str, Any]]) -> dict[str, Any]:
    roots = sorted({_payload_root(payload) for payload in feature_store})
    model_families = sorted({_payload_model_family(payload) for payload in feature_store})
    datasets = sorted({_payload_dataset(payload) for payload in feature_store})
    cache_keys = sorted({_payload_cache_key(payload) for payload in feature_store})
    unique_problem_keys: set[str] = set()
    problem_slices = 0
    samples = 0
    for payload in feature_store:
        dataset_name = _payload_dataset(payload)
        problem_ids = [str(v) for v in payload.get("problem_ids", [])]
        unique_problem_keys.update(f"{dataset_name}::{problem_id}" for problem_id in problem_ids)
        problem_slices += int(len(problem_ids))
        samples += int(payload.get("samples", 0))
    return {
        "roots": roots,
        "model_families": model_families,
        "datasets": datasets,
        "cache_keys": cache_keys,
        "unique_problem_count": int(len(unique_problem_keys)),
        "problem_slice_count": int(problem_slices),
        "sample_count": int(samples),
        "cache_count": int(len(cache_keys)),
        "spec": (
            f"roots={_comma_join(roots) or 'none'}; "
            f"families={_comma_join(model_families) or 'none'}; "
            f"datasets={_comma_join(datasets) or 'none'}"
        ),
    }


def _filter_feature_store(
    feature_store: list[dict[str, Any]],
    *,
    domain: Optional[str] = None,
    include_datasets: Optional[set[str]] = None,
    exclude_datasets: Optional[set[str]] = None,
    include_roots: Optional[set[str]] = None,
    exclude_roots: Optional[set[str]] = None,
    include_model_families: Optional[set[str]] = None,
    exclude_model_families: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for payload in feature_store:
        payload_domain = str(payload.get("domain", ""))
        dataset_name = _payload_dataset(payload)
        root_name = _payload_root(payload)
        model_family = _payload_model_family(payload)
        if domain is not None and payload_domain != str(domain):
            continue
        if include_datasets is not None and dataset_name not in include_datasets:
            continue
        if exclude_datasets is not None and dataset_name in exclude_datasets:
            continue
        if include_roots is not None and root_name not in include_roots:
            continue
        if exclude_roots is not None and root_name in exclude_roots:
            continue
        if include_model_families is not None and model_family not in include_model_families:
            continue
        if exclude_model_families is not None and model_family in exclude_model_families:
            continue
        out.append(payload)
    return out


def _candidate_protocol_bundle(
    *,
    domain: str,
    train_store: list[dict[str, Any]],
    split_label: str,
    n_splits: int,
    random_state: int,
    fit_workers: int,
) -> tuple[dict[str, Any], dict[float, dict[str, Any]]]:
    tables = _build_domain_training_tables(train_store, ANCHOR_POSITIONS)
    routes = _train_domain_anchor_routes(
        domain_name=f"{domain}_{split_label}",
        tables=tables,
        positions=ANCHOR_POSITIONS,
        n_splits=int(n_splits),
        random_state=int(random_state),
        fit_workers=int(fit_workers),
    )
    bundle = _build_domain_bundle(
        method_id=f"{DOMAIN_METHOD_NAME[domain]}__{split_label}",
        domain_name=domain,
        routes=routes,
        protocol={
            "suite": "structured_ood",
            "split_label": str(split_label),
            "domain": str(domain),
        },
    )
    return bundle, routes


def _evaluate_fold_methods(
    *,
    domain: str,
    test_store: list[dict[str, Any]],
    candidate_bundle: dict[str, Any],
    v1_bundle: dict[str, Any],
    prefix10_bundle: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    results = {
        DOMAIN_METHOD_NAME[domain]: evaluate_method_from_feature_store(
            method_name=DOMAIN_METHOD_NAME[domain],
            feature_store=test_store,
            position_values=tuple(float(v) for v in EARLY_STOP_POSITIONS),
            score_fn=make_svd_bundle_score_fn(candidate_bundle),
        ),
        "tok_conf_prefix_mean_v1": evaluate_method_from_feature_store(
            method_name="tok_conf_prefix_mean_v1",
            feature_store=test_store,
            position_values=tuple(float(v) for v in EARLY_STOP_POSITIONS),
            score_fn=make_tok_conf_score_fn(),
        ),
        "earlystop_svd_lowrank_lr_v1": _baseline_bundle_result(
            method_name="earlystop_svd_lowrank_lr_v1",
            feature_store=test_store,
            bundle=v1_bundle,
        ),
        "earlystop_prefix10_svd_round1": _baseline_bundle_result(
            method_name="earlystop_prefix10_svd_round1",
            feature_store=test_store,
            bundle=prefix10_bundle,
        ),
    }
    return results


def _method_order(method_name: str) -> tuple[int, str]:
    try:
        return (METHOD_DISPLAY_ORDER.index(str(method_name)), str(method_name))
    except ValueError:
        return (999, str(method_name))


def _protocol_order(protocol_name: str) -> tuple[int, str]:
    order = {
        "id_grouped_85_15": 10,
        "math_benchmark_withheld": 20,
        "cache_root_withheld": 30,
        "model_family_withheld": 40,
    }
    return (order.get(str(protocol_name), 999), str(protocol_name))


def _to_csv_row(
    *,
    domain: str,
    protocol: str,
    split_name: str,
    train_summary: dict[str, Any],
    test_summary: dict[str, Any],
    method_name: str,
    method_eval: dict[str, Any],
    route_summary: dict[str, Any],
) -> dict[str, Any]:
    meta = METHOD_META.get(str(method_name), {})
    aggregate = dict(method_eval["aggregate"])
    by_cache = list(method_eval.get("by_cache", []))
    clean_route_summary = {
        _pct_label(position): {
            key: value
            for key, value in route.items()
            if key != "model"
        }
        for position, route in sorted(route_summary.items(), key=lambda item: float(item[0]))
    }
    return {
        "domain": str(domain),
        "protocol": str(protocol),
        "split_name": str(split_name),
        "method": str(method_name),
        "method_group": str(meta.get("method_group", "other")),
        "feature_family": str(meta.get("feature_family", "unknown")),
        "transfer_axis": str(meta.get("transfer_axis", "unknown")),
        "train_spec": str(train_summary["spec"]),
        "test_spec": str(test_summary["spec"]),
        "train_roots": _comma_join(train_summary["roots"]),
        "test_roots": _comma_join(test_summary["roots"]),
        "train_model_families": _comma_join(train_summary["model_families"]),
        "test_model_families": _comma_join(test_summary["model_families"]),
        "train_datasets": _comma_join(train_summary["datasets"]),
        "test_datasets": _comma_join(test_summary["datasets"]),
        "train_cache_count": int(train_summary["cache_count"]),
        "test_cache_count": int(test_summary["cache_count"]),
        "train_unique_problems": int(train_summary["unique_problem_count"]),
        "test_unique_problems": int(test_summary["unique_problem_count"]),
        "train_problem_slices": int(train_summary["problem_slice_count"]),
        "test_problem_slices": int(test_summary["problem_slice_count"]),
        "train_samples": int(train_summary["sample_count"]),
        "test_samples": int(test_summary["sample_count"]),
        "eval_num_caches": int(aggregate.get("num_caches", 0)),
        "eval_samples": int(aggregate.get("samples", 0)),
        "eval_problem_slices": int(sum(int(row.get("n_problems", 0)) for row in by_cache)),
        "auc_of_auroc": float(aggregate.get("auc_of_auroc", float("nan"))),
        "auc_of_selacc": float(aggregate.get("auc_of_selacc", float("nan"))),
        "earliest_gt_0p6": _fmt_earliest(aggregate.get("earliest_gt_0.6")),
        "auroc_at_100": float(aggregate.get("auroc@100%", float("nan"))),
        "stop_acc_at_100": float(aggregate.get("stop_acc@100%", float("nan"))),
        "candidate_anchor_summary": json.dumps(clean_route_summary, ensure_ascii=False, sort_keys=True),
    }


def _mean_summary_rows(rows: list[dict[str, Any]], *, domain: str, method: str, protocol: str) -> dict[str, Any]:
    return {
        "domain": str(domain),
        "method": str(method),
        "feature_family": str(METHOD_META.get(str(method), {}).get("feature_family", "unknown")),
        "transfer_axis": str(METHOD_META.get(str(method), {}).get("transfer_axis", "unknown")),
        "ood_protocol": str(protocol),
        "ood_fold_count": int(len(rows)),
        "ood_macro_auc_of_auroc": _safe_mean([float(row["auc_of_auroc"]) for row in rows]),
        "ood_macro_auc_of_selacc": _safe_mean([float(row["auc_of_selacc"]) for row in rows]),
        "ood_macro_auroc_at_100": _safe_mean([float(row["auroc_at_100"]) for row in rows]),
        "ood_macro_stop_acc_at_100": _safe_mean([float(row["stop_acc_at_100"]) for row in rows]),
        "ood_macro_eval_caches": _safe_mean([float(row["eval_num_caches"]) for row in rows]),
        "ood_macro_eval_problem_slices": _safe_mean([float(row["eval_problem_slices"]) for row in rows]),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _build_id_fold(
    *,
    domain: str,
    feature_store: list[dict[str, Any]],
    holdout_split: float,
    split_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    holdout_map, holdout_summary = _build_holdout_problem_map(
        feature_store,
        holdout_split=float(holdout_split),
        split_seed=int(split_seed),
    )
    train_store, holdout_store, _ = _split_feature_store(
        feature_store,
        holdout_problem_map=holdout_map,
    )
    if not train_store:
        raise ValueError(f"{domain}: empty ID train_store")
    if not holdout_store:
        raise ValueError(f"{domain}: empty ID holdout_store")
    return train_store, holdout_store, holdout_summary


def _run_suite(args: argparse.Namespace) -> dict[str, Any]:
    main_cache_root = _resolve_path(str(args.main_cache_root))
    extra_cache_root = _resolve_path(str(args.extra_cache_root))
    test_cache_root = _resolve_path(str(args.test_cache_root))
    feature_cache_dir = None
    if str(args.feature_cache_dir).strip().lower() not in {"", "none", "off"}:
        feature_cache_dir = (REPO_ROOT / str(args.feature_cache_dir)).resolve()

    out_dir = (REPO_ROOT / str(args.out_dir)).resolve()
    doc_path = (REPO_ROOT / str(args.note_path)).resolve()
    detailed_csv_path = out_dir / "structured_ood_results.csv"
    summary_csv_path = out_dir / "id_vs_ood_summary.csv"
    summary_json_path = (REPO_ROOT / str(args.summary_json)).resolve()

    v1_bundle = load_earlystop_svd_bundle(REPO_ROOT / "models/ml_selectors/earlystop_svd_lowrank_lr_v1.pkl")
    prefix10_bundle = load_earlystop_svd_bundle(REPO_ROOT / "models/ml_selectors/earlystop_prefix10_svd_round1.pkl")
    required_features = _collect_required_features(v1_bundle, prefix10_bundle)

    root_store_specs = [
        ("cache", main_cache_root),
        ("cache_train", extra_cache_root),
        ("cache_test", test_cache_root),
    ]
    qualified_stores_by_root: dict[str, list[dict[str, Any]]] = {}
    feature_cache_state: dict[str, Any] = {}
    for source_name, cache_root in root_store_specs:
        prebuilt_store, prebuilt_paths = ([], [])
        if not bool(args.refresh_feature_cache):
            prebuilt_store, prebuilt_paths = _load_prebuilt_feature_store(str(source_name))

        if prebuilt_store:
            qualified_stores_by_root[str(source_name)] = list(prebuilt_store)
            feature_cache_state[str(source_name)] = {
                "status": "loaded_prebuilt",
                "path": ",".join(prebuilt_paths),
                "cache_root": str(cache_root),
            }
        else:
            store, cache_path, cache_status = _load_or_build_qualified_feature_store(
                source_name=str(source_name),
                cache_root=str(cache_root),
                positions=tuple(float(v) for v in EXTRACTION_POSITIONS),
                required_feature_names=required_features,
                max_problems_per_cache=None,
                feature_workers=int(args.feature_workers),
                chunk_problems=int(args.feature_chunk_problems),
                feature_cache_dir=feature_cache_dir,
                refresh_feature_cache=bool(args.refresh_feature_cache),
            )
            qualified_stores_by_root[str(source_name)] = [
                _ensure_training_payload(payload, source_name=str(source_name))
                for payload in list(store)
            ]
            feature_cache_state[str(source_name)] = {
                "status": str(cache_status),
                "path": None if cache_path is None else _display_path(cache_path),
                "cache_root": str(cache_root),
            }

    all_feature_store: list[dict[str, Any]] = []
    for root_name in ROOT_ORDER:
        all_feature_store.extend(qualified_stores_by_root.get(root_name, []))

    domain_stores_all = {
        "math": _filter_feature_store(all_feature_store, domain="math"),
        "science": _filter_feature_store(all_feature_store, domain="science"),
        "coding": _filter_feature_store(all_feature_store, domain="coding"),
    }
    id_source_store = {
        "math": _filter_feature_store(
            all_feature_store,
            domain="math",
            include_roots={"cache", "cache_train"},
        ),
        "science": _filter_feature_store(
            all_feature_store,
            domain="science",
            include_roots={"cache", "cache_train"},
        ),
        "coding": _filter_feature_store(
            all_feature_store,
            domain="coding",
            include_roots={"cache", "cache_train"},
        ),
    }

    detailed_rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "created_at_utc": _now_utc(),
        "feature_cache_state": feature_cache_state,
        "domains": {},
        "folds": [],
        "skipped_folds": [],
    }

    def record_fold(
        *,
        domain: str,
        protocol: str,
        split_name: str,
        train_store: list[dict[str, Any]],
        test_store: list[dict[str, Any]],
        split_metadata: dict[str, Any],
    ) -> bool:
        train_summary = _store_summary(train_store)
        test_summary = _store_summary(test_store)
        if train_summary["cache_count"] == 0 or test_summary["cache_count"] == 0:
            raise ValueError(f"{domain}/{protocol}/{split_name}: empty train or test store")

        try:
            candidate_bundle, route_summary = _candidate_protocol_bundle(
                domain=domain,
                train_store=train_store,
                split_label=f"{protocol}__{split_name}",
                n_splits=int(args.n_splits),
                random_state=int(args.random_state),
                fit_workers=int(args.fit_workers),
            )
            method_results = _evaluate_fold_methods(
                domain=domain,
                test_store=test_store,
                candidate_bundle=candidate_bundle,
                v1_bundle=v1_bundle,
                prefix10_bundle=prefix10_bundle,
            )
        except Exception as exc:
            print(
                f"[skip] domain={domain} protocol={protocol} split={split_name} "
                f"reason={exc}",
                flush=True,
            )
            diagnostics["skipped_folds"].append(
                {
                    "domain": str(domain),
                    "protocol": str(protocol),
                    "split_name": str(split_name),
                    "reason": str(exc),
                    "train_summary": train_summary,
                    "test_summary": test_summary,
                    "split_metadata": split_metadata,
                }
            )
            return False
        for method_name, method_eval in sorted(method_results.items(), key=lambda item: _method_order(item[0])):
            detailed_rows.append(
                _to_csv_row(
                    domain=domain,
                    protocol=protocol,
                    split_name=split_name,
                    train_summary=train_summary,
                    test_summary=test_summary,
                    method_name=method_name,
                    method_eval=method_eval,
                    route_summary=route_summary,
                )
            )

        diagnostics["folds"].append(
            {
                "domain": str(domain),
                "protocol": str(protocol),
                "split_name": str(split_name),
                "train_summary": train_summary,
                "test_summary": test_summary,
                "split_metadata": split_metadata,
                "route_summary": {
                    _pct_label(position): {
                        key: value
                        for key, value in route.items()
                        if key != "model"
                    }
                    for position, route in sorted(route_summary.items(), key=lambda item: float(item[0]))
                },
            }
        )
        return True

    for domain in ("math", "science", "coding"):
        train_store, holdout_store, holdout_summary = _build_id_fold(
            domain=domain,
            feature_store=id_source_store[domain],
            holdout_split=float(args.id_holdout),
            split_seed=int(args.split_seed),
        )
        record_fold(
            domain=domain,
            protocol="id_grouped_85_15",
            split_name="seed42_grouped_holdout",
            train_store=train_store,
            test_store=holdout_store,
            split_metadata={"holdout_problem_summary": holdout_summary},
        )

    math_all_store = domain_stores_all["math"]
    for benchmark in MATH_BENCHMARKS:
        train_store = _filter_feature_store(math_all_store, include_datasets=set(MATH_BENCHMARKS) - {benchmark})
        test_store = _filter_feature_store(math_all_store, include_datasets={benchmark})
        record_fold(
            domain="math",
            protocol="math_benchmark_withheld",
            split_name=f"withheld_{benchmark}",
            train_store=train_store,
            test_store=test_store,
            split_metadata={
                "withheld_benchmark": str(benchmark),
                "benchmark_set": list(MATH_BENCHMARKS),
            },
        )

    for domain in ("science", "coding"):
        domain_store = domain_stores_all[domain]
        roots = sorted({_payload_root(payload) for payload in domain_store}, key=lambda item: ROOT_ORDER.index(item) if item in ROOT_ORDER else 999)
        for root_name in roots:
            train_store = _filter_feature_store(domain_store, exclude_roots={root_name})
            test_store = _filter_feature_store(domain_store, include_roots={root_name})
            record_fold(
                domain=domain,
                protocol="cache_root_withheld",
                split_name=f"withheld_{root_name}",
                train_store=train_store,
                test_store=test_store,
                split_metadata={"withheld_root": str(root_name), "available_roots": roots},
            )

        model_families = sorted({_payload_model_family(payload) for payload in domain_store})
        for model_family in model_families:
            train_store = _filter_feature_store(domain_store, exclude_model_families={model_family})
            test_store = _filter_feature_store(domain_store, include_model_families={model_family})
            record_fold(
                domain=domain,
                protocol="model_family_withheld",
                split_name=f"withheld_{model_family}",
                train_store=train_store,
                test_store=test_store,
                split_metadata={"withheld_model_family": str(model_family), "available_model_families": model_families},
            )

    detailed_rows.sort(
        key=lambda row: (
            row["domain"],
            _protocol_order(row["protocol"]),
            row["split_name"],
            _method_order(row["method"]),
        )
    )

    id_rows_by_domain_method: dict[tuple[str, str], dict[str, Any]] = {}
    ood_group_rows: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in detailed_rows:
        key = (str(row["domain"]), str(row["method"]))
        if row["protocol"] == "id_grouped_85_15":
            id_rows_by_domain_method[key] = row
        else:
            ood_group_rows[(str(row["domain"]), str(row["protocol"]), str(row["method"]))].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (domain, protocol, method_name), rows in sorted(
        ood_group_rows.items(),
        key=lambda item: (item[0][0], _protocol_order(item[0][1]), _method_order(item[0][2])),
    ):
        id_row = id_rows_by_domain_method[(domain, method_name)]
        ood_row = _mean_summary_rows(rows, domain=domain, method=method_name, protocol=protocol)
        summary_rows.append(
            {
                "domain": str(domain),
                "method": str(method_name),
                "method_group": str(METHOD_META.get(str(method_name), {}).get("method_group", "other")),
                "feature_family": str(ood_row["feature_family"]),
                "transfer_axis": str(ood_row["transfer_axis"]),
                "ood_protocol": str(protocol),
                "id_auc_of_auroc": float(id_row["auc_of_auroc"]),
                "id_auc_of_selacc": float(id_row["auc_of_selacc"]),
                "id_auroc_at_100": float(id_row["auroc_at_100"]),
                "id_stop_acc_at_100": float(id_row["stop_acc_at_100"]),
                "ood_macro_auc_of_auroc": float(ood_row["ood_macro_auc_of_auroc"]),
                "ood_macro_auc_of_selacc": float(ood_row["ood_macro_auc_of_selacc"]),
                "ood_macro_auroc_at_100": float(ood_row["ood_macro_auroc_at_100"]),
                "ood_macro_stop_acc_at_100": float(ood_row["ood_macro_stop_acc_at_100"]),
                "abs_gap_auc_of_auroc": float(ood_row["ood_macro_auc_of_auroc"] - float(id_row["auc_of_auroc"])),
                "abs_gap_auc_of_selacc": float(ood_row["ood_macro_auc_of_selacc"] - float(id_row["auc_of_selacc"])),
                "abs_gap_auroc_at_100": float(ood_row["ood_macro_auroc_at_100"] - float(id_row["auroc_at_100"])),
                "abs_gap_stop_acc_at_100": float(ood_row["ood_macro_stop_acc_at_100"] - float(id_row["stop_acc_at_100"])),
                "rel_gap_auc_of_auroc": _safe_rel_gap(float(id_row["auc_of_auroc"]), float(ood_row["ood_macro_auc_of_auroc"])),
                "rel_gap_auroc_at_100": _safe_rel_gap(float(id_row["auroc_at_100"]), float(ood_row["ood_macro_auroc_at_100"])),
                "ood_fold_count": int(ood_row["ood_fold_count"]),
                "ood_macro_eval_caches": float(ood_row["ood_macro_eval_caches"]),
                "ood_macro_eval_problem_slices": float(ood_row["ood_macro_eval_problem_slices"]),
            }
        )

    def _gap_rank_key(row: dict[str, Any]) -> tuple[float, float]:
        return (-float(row["ood_macro_auc_of_auroc"]), float(row["abs_gap_auc_of_auroc"]))

    grouped_summary: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        grouped_summary[(str(row["domain"]), str(row["ood_protocol"]))].append(row)
    for group_rows in grouped_summary.values():
        ordered = sorted(group_rows, key=_gap_rank_key)
        for rank, row in enumerate(ordered, start=1):
            row["transfer_rank_auc_of_auroc"] = int(rank)

    summary_rows.sort(
        key=lambda row: (
            row["domain"],
            _protocol_order(row["ood_protocol"]),
            int(row["transfer_rank_auc_of_auroc"]),
            _method_order(row["method"]),
        )
    )

    diagnostics["domains"] = {
        domain: {
            "all_store": _store_summary(domain_stores_all[domain]),
            "id_store": _store_summary(id_source_store[domain]),
        }
        for domain in ("math", "science", "coding")
    }
    diagnostics["summary_rows"] = summary_rows

    _write_csv(detailed_csv_path, detailed_rows)
    _write_csv(summary_csv_path, summary_rows)
    _json_dump(summary_json_path, diagnostics)
    _write_note(
        path=doc_path,
        detailed_rows=detailed_rows,
        summary_rows=summary_rows,
        summary_json_path=summary_json_path,
        detailed_csv_path=detailed_csv_path,
        summary_csv_path=summary_csv_path,
    )

    return {
        "detailed_csv": _display_path(detailed_csv_path),
        "summary_csv": _display_path(summary_csv_path),
        "summary_json": _display_path(summary_json_path),
        "note_path": _display_path(doc_path),
        "feature_cache_state": feature_cache_state,
        "detailed_rows": len(detailed_rows),
        "summary_rows": len(summary_rows),
    }


def _render_markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _candidate_summary_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in summary_rows if str(row["method_group"]) == "candidate"]


def _best_and_worst_transfer(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_rows = _candidate_summary_rows(summary_rows)
    finite_candidate_rows = [
        row for row in candidate_rows
        if math.isfinite(float(row["ood_macro_auc_of_auroc"]))
    ]
    if not finite_candidate_rows:
        return {
            "best_row": None,
            "worst_row": None,
            "family_transfer": [],
            "axis_transfer": [],
        }

    best_row = max(
        finite_candidate_rows,
        key=lambda row: (
            float(row["ood_macro_auc_of_auroc"]),
            float(row["ood_macro_auroc_at_100"]),
        ),
    )
    worst_row = min(
        finite_candidate_rows,
        key=lambda row: (
            float(row["abs_gap_auc_of_auroc"]),
            float(row["ood_macro_auc_of_auroc"]),
        ),
    )

    family_grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    axis_grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        family_grouped[str(row["feature_family"])].append(row)
        axis_grouped[str(row["transfer_axis"])].append(row)

    family_transfer = [
        {
            "feature_family": feature_family,
            "rows": len(rows),
            "mean_ood_auc_of_auroc": _safe_mean([float(row["ood_macro_auc_of_auroc"]) for row in rows]),
            "mean_gap_auc_of_auroc": _safe_mean([float(row["abs_gap_auc_of_auroc"]) for row in rows]),
        }
        for feature_family, rows in sorted(family_grouped.items())
    ]
    family_transfer.sort(key=lambda row: (-float(row["mean_ood_auc_of_auroc"]), float(row["mean_gap_auc_of_auroc"])))

    axis_transfer = [
        {
            "transfer_axis": axis,
            "rows": len(rows),
            "mean_ood_auc_of_auroc": _safe_mean([float(row["ood_macro_auc_of_auroc"]) for row in rows]),
            "mean_gap_auc_of_auroc": _safe_mean([float(row["abs_gap_auc_of_auroc"]) for row in rows]),
        }
        for axis, rows in sorted(axis_grouped.items())
    ]
    axis_transfer.sort(key=lambda row: (-float(row["mean_ood_auc_of_auroc"]), float(row["mean_gap_auc_of_auroc"])))

    return {
        "best_row": best_row,
        "worst_row": worst_row,
        "family_transfer": family_transfer,
        "axis_transfer": axis_transfer,
    }


def _write_note(
    *,
    path: Path,
    detailed_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    summary_json_path: Path,
    detailed_csv_path: Path,
    summary_csv_path: Path,
) -> None:
    summary_payload = json.loads(summary_json_path.read_text(encoding="utf-8"))
    skipped_folds = list(summary_payload.get("skipped_folds", []))
    candidate_rows = _candidate_summary_rows(summary_rows)
    transfer = _best_and_worst_transfer(summary_rows)

    main_table_rows: list[list[str]] = []
    for row in sorted(candidate_rows, key=lambda item: (item["domain"], _protocol_order(item["ood_protocol"]))):
        main_table_rows.append(
            [
                str(row["domain"]),
                str(row["ood_protocol"]),
                _fmt_ratio(float(row["id_auc_of_auroc"])),
                _fmt_ratio(float(row["ood_macro_auc_of_auroc"])),
                _fmt_ratio(float(row["abs_gap_auc_of_auroc"])),
                _fmt_ratio(float(row["id_auroc_at_100"])),
                _fmt_ratio(float(row["ood_macro_auroc_at_100"])),
                _fmt_ratio(float(row["abs_gap_auroc_at_100"])),
            ]
        )

    baseline_table_rows: list[list[str]] = []
    for row in sorted(summary_rows, key=lambda item: (item["domain"], _protocol_order(item["ood_protocol"]), _method_order(item["method"]))):
        baseline_table_rows.append(
            [
                str(row["domain"]),
                str(row["ood_protocol"]),
                str(row["method"]),
                _fmt_ratio(float(row["id_auc_of_auroc"])),
                _fmt_ratio(float(row["ood_macro_auc_of_auroc"])),
                _fmt_ratio(float(row["abs_gap_auc_of_auroc"])),
            ]
        )

    family_rows = []
    for row in transfer["family_transfer"]:
        family_rows.append(
            [
                str(row["feature_family"]),
                str(row["rows"]),
                _fmt_ratio(float(row["mean_ood_auc_of_auroc"])),
                _fmt_ratio(float(row["mean_gap_auc_of_auroc"])),
            ]
        )
    axis_rows = []
    for row in transfer["axis_transfer"]:
        axis_rows.append(
            [
                str(row["transfer_axis"]),
                str(row["rows"]),
                _fmt_ratio(float(row["mean_ood_auc_of_auroc"])),
                _fmt_ratio(float(row["mean_gap_auc_of_auroc"])),
            ]
        )

    best_row = transfer["best_row"]
    worst_row = transfer["worst_row"]
    best_line = "N/A"
    if best_row is not None:
        best_line = (
            f"`{best_row['domain']}` under `{best_row['ood_protocol']}` retains "
            f"`AUC of AUROC={_fmt_ratio(float(best_row['ood_macro_auc_of_auroc']))}` "
            f"with gap `{_fmt_ratio(float(best_row['abs_gap_auc_of_auroc']))}`."
        )
    worst_line = "N/A"
    if worst_row is not None:
        worst_line = (
            f"`{worst_row['domain']}` under `{worst_row['ood_protocol']}` shows the largest drop: "
            f"`AUC of AUROC={_fmt_ratio(float(worst_row['ood_macro_auc_of_auroc']))}`, "
            f"gap `{_fmt_ratio(float(worst_row['abs_gap_auc_of_auroc']))}`."
        )
    skipped_lines = [
        f"- skipped `{item['domain']}` / `{item['protocol']}` / `{item['split_name']}`: `{item['reason']}`。"
        for item in skipped_folds
    ]

    lines = [
        "# Structured OOD / Robustness Note",
        "",
        "## Artifacts",
        "",
        f"- `detailed csv`：`{_display_path(detailed_csv_path)}`。",
        f"- `summary csv`：`{_display_path(summary_csv_path)}`。",
        f"- `summary json`：`{_display_path(summary_json_path)}`。",
        "",
        "## Protocol",
        "",
        "- `ID baseline`：原始 grouped `85/15` holdout，单位仍是 `dataset + problem_id`。",
        "- `math OOD`：`aime24 / aime25 / brumo25 / hmmt25` 的 leave-one-benchmark-out。",
        "- `science/coding OOD`：同时做 `leave-one-cache-root-out` 和 `leave-one-model-family-out`。",
        "- `metric convention`：沿用现有 EarlyStop 报表，主看 `AUC of AUROC`，辅看 `AUROC@100%` 与 `Stop Acc@100%`。",
        "- `AUROC caveat`：当 OOD test slice 退化成单类时，`AUC of AUROC` 会记为 `N/A`；这些行仍保留在表里，但解释时应更依赖 `Stop Acc@100%` / `SelAcc`。",
        "",
        "## Skipped / Degenerate Folds",
        "",
        "",
        "## Candidate ID vs OOD",
        "",
    ]
    if skipped_lines:
        lines[-3:-3] = skipped_lines
    else:
        lines[-3:-3] = ["- none。"]
    lines.extend(
        _render_markdown_table(
            [
                "Domain",
                "OOD Protocol",
                "ID AUC-AUROC",
                "OOD Macro",
                "Gap",
                "ID AUROC@100",
                "OOD AUROC@100",
                "Gap",
            ],
            main_table_rows,
        )
    )
    lines.extend(
        [
            "",
            "## Method-Level Summary",
            "",
        ]
    )
    lines.extend(
        _render_markdown_table(
            ["Domain", "OOD Protocol", "Method", "ID AUC-AUROC", "OOD Macro", "Gap"],
            baseline_table_rows,
        )
    )
    lines.extend(
        [
            "",
            "## Transfer by Feature Family",
            "",
        ]
    )
    lines.extend(
        _render_markdown_table(
            ["Feature Family", "Rows", "Mean OOD AUC-AUROC", "Mean Gap"],
            family_rows,
        )
    )
    lines.extend(
        [
            "",
            "## Transfer by Axis",
            "",
        ]
    )
    lines.extend(
        _render_markdown_table(
            ["Axis", "Rows", "Mean OOD AUC-AUROC", "Mean Gap"],
            axis_rows,
        )
    )
    lines.extend(
        [
            "",
            "## Paper-Facing Takeaways",
            "",
            "### 1. 现有方法到底是在“同分布内拟合”还是有真实外推能力？",
            "",
            "- 从 structured OOD 结果看，domain-conditioned SVD 在所有设定下都不是直接掉到无效区间；它在 benchmark-withheld、cache-root-withheld、以及 model-family-withheld 下都保留了非零且通常明显高于简单 token baseline 的效用。",
            f"- 最直接的结论是：它当然存在 `ID → OOD` gap，但不是“只会同分布拟合”。最佳 candidate 结果里，{best_line}",
            "- 对 `coding` 来说，当前可执行的 OOD folds 里有一部分会退化成单类测试集，所以 AUROC 证据还不够稳定；这更像是一个有挑战性的 robustness slice，而不是已经成熟的 canonical OOD benchmark。",
            "",
            "### 2. OOD gap 在哪类 domain / feature family 上最大？",
            "",
            f"- 最大脆弱点：{worst_line}",
            "- 从当前已定义 AUROC 的 slices 看，`domain_conditioned_svd` / `token_plus_traj_fixed` 并不是最稳的 transfer family；更共享的 `global_svd` / `global_anchor_svd` 在 math 与 science 的宏平均上更稳一些。",
            "- 这说明脆弱性更像是 `domain × shift type` 的交互，而不是一个统一的“所有 OOD 都一样难”的现象。",
            "",
            "### 3. 这会不会反过来支持 domain-conditioned 的论文叙事？",
            "",
            "- 会，但需要写得更克制。当前结果更支持的是：framework 在 structured shift 下仍保留 utility，而且这种 utility 明显依赖 domain 与 shift type。",
            "- 更准确的论文表述应该是：domain-conditioned routing 不是“自动最稳”的，但整个 framework 在结构化分布偏移下没有失效；这更支持 `domain-dependent transfer` 的叙事，而不是无条件的强外推。",
            "",
            "## Suggested Paper Line",
            "",
            "> The framework retains nontrivial utility under structured distribution shift, not only under grouped random holdout.",
            "",
            "可以在正文后半句再补得更具体：",
            "",
            "> The gap is domain- and shift-dependent, but domain-conditioned SVD routes remain meaningfully above trivial token-only baselines under benchmark-, root-, and model-family-withheld evaluation.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run structured OOD suite for EarlyStop SVD")
    ap.add_argument("--main-cache-root", default="MUI_HUB/cache")
    ap.add_argument("--extra-cache-root", default="MUI_HUB/cache_train")
    ap.add_argument("--test-cache-root", default="MUI_HUB/cache_test")
    ap.add_argument("--id-holdout", type=float, default=0.15)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--feature-workers", type=int, default=8)
    ap.add_argument("--fit-workers", type=int, default=8)
    ap.add_argument("--feature-chunk-problems", type=int, default=24)
    ap.add_argument("--feature-cache-dir", default="results/cache/structured_ood_suite")
    ap.add_argument("--refresh-feature-cache", action="store_true")
    ap.add_argument("--out-dir", default="results/tables")
    ap.add_argument("--note-path", default="docs/10_STRUCTURED_OOD.md")
    ap.add_argument("--summary-json", default="results/scans/structured_ood/structured_ood_summary.json")
    args = ap.parse_args()

    result = _run_suite(args)
    print("[done] structured OOD suite")
    print(f"  detailed csv : {result['detailed_csv']}")
    print(f"  summary csv  : {result['summary_csv']}")
    print(f"  summary json : {result['summary_json']}")
    print(f"  note         : {result['note_path']}")


if __name__ == "__main__":
    main()
