#!/usr/bin/env python3
"""Export EarlyStop SVD submission from cache_test."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.ops.earlystop import (
    build_earlystop_payload,
    discover_cache_entries,
    EARLY_STOP_POSITIONS,
    validate_earlystop_payload,
    write_earlystop_payload,
)
from nad.ops.earlystop_svd import (
    DEFAULT_REFLECTION_THRESHOLD,
    get_domain,
    load_earlystop_svd_bundle,
    score_cache_entry_earlystop_svd,
)
from scripts.run_earlystop_prefix10_svd_round1 import (
    build_feature_store,
    make_svd_bundle_score_fn,
)


def _collect_required_features(bundle: dict) -> set[str]:
    required: set[str] = set()
    for domain_bundle in bundle["domains"].values():
        for route in domain_bundle["routes"]:
            if route["route_type"] == "baseline":
                required.add(str(route["signal_name"]))
            else:
                required.update(str(name) for name in route["feature_names"])
    return required


def _feature_cache_key(
    *,
    cache_root: str,
    positions: tuple[float, ...],
    required_feature_names: set[str],
    max_problems: int | None,
    reflection_threshold: float,
    include_cache_keys: set[str] | None = None,
    exclude_cache_keys: set[str] | None = None,
) -> str:
    payload = {
        "version": 1,
        "cache_root": str(cache_root),
        "positions": [float(p) for p in positions],
        "required_feature_names": sorted(str(v) for v in required_feature_names),
        "max_problems": None if max_problems is None else int(max_problems),
        "reflection_threshold": float(reflection_threshold),
        "include_cache_keys": None if include_cache_keys is None else sorted(str(v) for v in include_cache_keys),
        "exclude_cache_keys": None if exclude_cache_keys is None else sorted(str(v) for v in exclude_cache_keys),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


def _feature_cache_path(
    *,
    cache_dir: Path,
    cache_root: str,
    positions: tuple[float, ...],
    required_feature_names: set[str],
    max_problems: int | None,
    reflection_threshold: float,
    include_cache_keys: set[str] | None = None,
    exclude_cache_keys: set[str] | None = None,
) -> Path:
    suffix = "all" if max_problems is None else f"cap{int(max_problems)}"
    key = _feature_cache_key(
        cache_root=cache_root,
        positions=positions,
        required_feature_names=required_feature_names,
        max_problems=max_problems,
        reflection_threshold=reflection_threshold,
        include_cache_keys=include_cache_keys,
        exclude_cache_keys=exclude_cache_keys,
    )
    thr_tag = f"ref{int(round(float(reflection_threshold) * 100.0)):03d}"
    return cache_dir / f"feature_store_{suffix}_{thr_tag}_{key}.pkl"


def _load_or_build_feature_store(
    *,
    cache_root: str,
    positions: tuple[float, ...],
    required_feature_names: set[str],
    max_problems: int | None,
    reflection_threshold: float,
    workers: int,
    feature_chunk_problems: int,
    feature_cache_dir: Path | None,
    refresh_feature_cache: bool,
    include_cache_keys: set[str] | None = None,
    exclude_cache_keys: set[str] | None = None,
) -> tuple[list[dict], Path | None, str]:
    cache_path: Path | None = None
    if feature_cache_dir is not None:
        cache_path = _feature_cache_path(
            cache_dir=feature_cache_dir,
            cache_root=cache_root,
            positions=positions,
            required_feature_names=required_feature_names,
            max_problems=max_problems,
            reflection_threshold=reflection_threshold,
            include_cache_keys=include_cache_keys,
            exclude_cache_keys=exclude_cache_keys,
        )
        if cache_path.exists() and not refresh_feature_cache:
            print(f"Loading feature cache from {cache_path}")
            with cache_path.open("rb") as handle:
                payload = pickle.load(handle)
            return list(payload["feature_store"]), cache_path, "loaded"

    feature_store = build_feature_store(
        cache_root=cache_root,
        positions=positions,
        required_feature_names=required_feature_names,
        max_problems_per_cache=max_problems,
        max_workers=int(workers),
        chunk_problems=int(feature_chunk_problems),
        include_cache_keys=include_cache_keys,
        exclude_cache_keys=exclude_cache_keys,
        reflection_threshold=float(reflection_threshold),
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with tmp_path.open("wb") as handle:
            pickle.dump(
                {
                    "cache_root": str(cache_root),
                    "positions": [float(p) for p in positions],
                    "max_problems": None if max_problems is None else int(max_problems),
                    "reflection_threshold": float(reflection_threshold),
                    "include_cache_keys": None if include_cache_keys is None else sorted(str(v) for v in include_cache_keys),
                    "exclude_cache_keys": None if exclude_cache_keys is None else sorted(str(v) for v in exclude_cache_keys),
                    "feature_store": feature_store,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        tmp_path.replace(cache_path)
        print(f"Saved feature cache to {cache_path}")
    return feature_store, cache_path, "built"


def _problem_scores_from_payload(
    payload: dict,
    score_fn,
) -> dict[str, dict[str, list[float]]]:
    problem_scores: dict[str, dict[str, list[float]]] = {}
    tensor = payload["tensor"]
    sample_ids_all = payload["sample_ids"]
    problem_ids = payload["problem_ids"]
    problem_offsets = payload["problem_offsets"]
    n_positions = len(EARLY_STOP_POSITIONS)

    for problem_idx, problem_id in enumerate(problem_ids):
        start = int(problem_offsets[problem_idx])
        end = int(problem_offsets[problem_idx + 1])
        problem_tensor = tensor[start:end]
        sample_ids = sample_ids_all[start:end]
        run_scores = {str(sample_id): [0.0] * n_positions for sample_id in sample_ids.tolist()}

        for pos_idx in range(n_positions):
            x_raw = problem_tensor[:, pos_idx, :]
            scores = score_fn(payload["domain"], pos_idx, x_raw)
            for row_idx, sample_id in enumerate(sample_ids.tolist()):
                run_scores[str(sample_id)][pos_idx] = float(scores[row_idx])

        problem_scores[str(problem_id)] = run_scores
    return problem_scores


def _bundle_reflection_thresholds(bundle: dict[str, Any]) -> set[float]:
    thresholds: set[float] = set()
    default_threshold = float(bundle.get("reflection_threshold", DEFAULT_REFLECTION_THRESHOLD))
    for domain_bundle in bundle["domains"].values():
        for route in domain_bundle["routes"]:
            thresholds.add(float(route.get("reflection_threshold", default_threshold)))
    return thresholds


def _domain_route_thresholds(bundle: dict[str, Any], domain: str) -> set[float]:
    default_threshold = float(bundle.get("reflection_threshold", DEFAULT_REFLECTION_THRESHOLD))
    domain_bundle = bundle["domains"][str(domain)]
    return {
        float(route.get("reflection_threshold", default_threshold))
        for route in domain_bundle["routes"]
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Export EarlyStop SVD submission")
    ap.add_argument("--cache-root", default="/home/jovyan/public-ro/MUI_HUB/cache_test")
    ap.add_argument("--model-path", default="models/ml_selectors/earlystop_svd_lowrank_lr_v1.pkl")
    ap.add_argument("--out-dir", default="submission/EarlyStop")
    ap.add_argument("--method-name", default="earlystop_svd_lowrank_lr_v1")
    ap.add_argument("--filename", default="earlystop_svd_lowrank_lr_v1.json")
    ap.add_argument("--max-problems", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--feature-chunk-problems", type=int, default=8)
    ap.add_argument("--feature-cache-dir", default="results/cache/export_earlystop_svd_submission", help="Directory for cached blind feature stores; use 'none' to disable")
    ap.add_argument("--refresh-feature-cache", action="store_true", help="Ignore existing cached blind feature stores")
    args = ap.parse_args()

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = REPO_ROOT / model_path
    bundle = load_earlystop_svd_bundle(model_path)
    required_features = _collect_required_features(bundle)
    score_fn = make_svd_bundle_score_fn(bundle)
    feature_cache_dir = None if str(args.feature_cache_dir).strip().lower() in {"", "none", "off"} else (REPO_ROOT / str(args.feature_cache_dir)).resolve()
    positions = tuple(float(p) for p in EARLY_STOP_POSITIONS)
    bundle_thresholds = _bundle_reflection_thresholds(bundle)

    entries = discover_cache_entries(args.cache_root)
    print(f"Loaded bundle from {model_path}")
    print(f"Found {len(entries)} cache entries\n")

    for entry in entries:
        domain = get_domain(entry.dataset_name)
        print(f"  [{entry.cache_key}] domain={domain} queued")

    threshold_groups: dict[float, set[str]] = {}
    direct_entries: list[Any] = []
    for entry in entries:
        domain = get_domain(entry.dataset_name)
        route_thresholds = _domain_route_thresholds(bundle, domain)
        if len(route_thresholds) == 1:
            threshold = next(iter(route_thresholds))
            threshold_groups.setdefault(float(threshold), set()).add(str(entry.cache_key))
        else:
            direct_entries.append(entry)

    all_scores: list[tuple[str, dict]] = []
    if len(bundle_thresholds) == 1 or threshold_groups:
        if len(bundle_thresholds) == 1:
            print(f"Single route threshold detected: {sorted(bundle_thresholds)}")
        else:
            print(f"Mixed route thresholds detected: {sorted(bundle_thresholds)}")
            print("Grouping cache entries by per-domain threshold before export.\n")

        for reflection_threshold, include_cache_keys in sorted(threshold_groups.items()):
            feature_store, feature_cache_path, feature_cache_status = _load_or_build_feature_store(
                cache_root=args.cache_root,
                positions=positions,
                required_feature_names=required_features,
                max_problems=args.max_problems,
                reflection_threshold=float(reflection_threshold),
                workers=int(args.workers),
                feature_chunk_problems=int(args.feature_chunk_problems),
                feature_cache_dir=feature_cache_dir,
                refresh_feature_cache=bool(args.refresh_feature_cache),
                include_cache_keys=include_cache_keys,
            )
            print(
                f"Feature cache status: {feature_cache_status} | threshold={reflection_threshold:.2f} "
                f"| path={feature_cache_path} | caches={sorted(include_cache_keys)}\n"
            )

            for payload in feature_store:
                ps = _problem_scores_from_payload(payload, score_fn)
                all_scores.append((payload["cache_key"], ps))
                print(f"  [{payload['cache_key']}] domain={payload['domain']}")
                print(f"    problems : {len(ps)}")
                print(f"    samples  : {sum(len(v) for v in ps.values())}\n")

    if direct_entries:
        print(f"Direct fallback still needed for {len(direct_entries)} cache entries.\n")
        for entry in direct_entries:
            ps = score_cache_entry_earlystop_svd(entry, bundle, max_problems=args.max_problems)
            all_scores.append((entry.cache_key, ps))
            print(f"  [{entry.cache_key}] domain={get_domain(entry.dataset_name)}")
            print(f"    problems : {len(ps)}")
            print(f"    samples  : {sum(len(v) for v in ps.values())}\n")

    payload = build_earlystop_payload(all_scores, method_name=args.method_name)
    stats = validate_earlystop_payload(payload)
    print(f"Validation passed: {stats}\n")

    out_path = REPO_ROOT / args.out_dir / args.filename
    write_earlystop_payload(payload, out_path)
    print(f"Written to {out_path}\n")


if __name__ == "__main__":
    main()
