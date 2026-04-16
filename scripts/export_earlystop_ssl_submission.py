#!/usr/bin/env python3
"""Export experimental EarlyStop submission from Round1 SSL findings."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for _env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env, "1")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nad.ops.earlystop import (
    EARLY_STOP_POSITIONS,
    build_earlystop_payload,
    discover_cache_entries,
    validate_earlystop_payload,
    write_earlystop_payload,
)
from nad.ops.earlystop_svd import get_domain
from scripts.export_earlystop_svd_submission import (
    _load_json,
    _load_or_build_feature_store,
    _problem_scores_from_payload,
)
from scripts.train_earlystop_ssl_basis import (
    ANCHOR_POS_INDICES,
    BASIS_CONFIGS,
    FIXED_FEATURE_NAMES,
    FIXED_FEATURE_INDICES,
    POS_TO_ANCHOR_IDX,
    TOKEN_FIXED_INDICES,
    _fit_basis,
    _fit_heads_from_tables,
    _load_prebuilt_stores,
    _make_score_fn,
    _resolve_path,
)
from nad.ops.earlystop_ssl import build_anchor_tables


DEFAULT_BASE_SUBMISSION = "submission/EarlyStop/es_svd_ms_rr_r1__coding_from_round1c.json"
DEFAULT_CACHE_ROOT = "/home/jovyan/public-ro/MUI_HUB/cache_test"
DEFAULT_FEATURE_CACHE_DIR = "results/cache/export_earlystop_ssl_submission"
DEFAULT_OUT_MODEL = "models/ml_selectors/earlystop_ssl/earlystop_ssl_round1_shared_ms.pkl"
DEFAULT_METHOD = "earlystop_ssl_round1_shared_ms_experimental"
DEFAULT_FILENAME = "earlystop_ssl_round1_shared_ms_experimental.json"


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _parse_domain_specs(raw_specs: list[str]) -> dict[str, dict[str, Any]]:
    config_map = {cfg.name: cfg for cfg in BASIS_CONFIGS}
    out: dict[str, dict[str, Any]] = {}
    for raw in raw_specs:
        text = str(raw).strip()
        if not text:
            continue
        if "=" not in text or ":" not in text:
            raise ValueError(f"Bad --domain-spec value: {raw}")
        domain_name, rhs = text.split("=", 1)
        config_name, rank_raw = rhs.split(":", 1)
        domain_name = domain_name.strip()
        config_name = config_name.strip()
        rank = int(rank_raw.strip())
        if config_name not in config_map:
            raise ValueError(f"Unknown config name: {config_name}")
        out[domain_name] = {
            "config_name": config_name,
            "config": config_map[config_name],
            "rank": int(rank),
        }
    if not out:
        raise ValueError("No domain specs provided")
    return out


def _fit_full_models(
    *,
    feature_store: list[dict[str, Any]],
    domain_specs: dict[str, dict[str, Any]],
    seed: int,
    cca_reg: float,
    mask_rate: float,
) -> dict[str, dict[str, Any]]:
    domain_models: dict[str, dict[str, Any]] = {}
    for domain_name, spec in domain_specs.items():
        config = spec["config"]
        rank = int(spec["rank"])
        tables = build_anchor_tables(
            feature_store,
            fixed_feature_indices=FIXED_FEATURE_INDICES,
            token_feature_indices=TOKEN_FIXED_INDICES,
            anchor_position_indices=ANCHOR_POS_INDICES,
            domain=domain_name,
        )
        bundle = _fit_basis(
            config,
            tables=tables,
            rank=rank,
            seed=seed,
            cca_reg=float(cca_reg),
            mask_rate=float(mask_rate),
            basis_scope="shared",
            positive_mode="same_run",
        )
        heads = _fit_heads_from_tables(
            tables,
            feature_fn=lambda anchor_idx, table, cfg=config, local_bundle=bundle: (
                _make_features_for_table(table, cfg.name, local_bundle if not isinstance(local_bundle, dict) else local_bundle[anchor_idx])
            ),
            seed=seed,
        )
        domain_models[domain_name] = {
            "config_name": str(config.name),
            "view_name": str(config.view),
            "method": str(config.method),
            "rank": rank,
            "bundle": bundle,
            "heads": heads,
        }
    return domain_models


def _make_features_for_table(table, config_name: str, bundle: Any) -> np.ndarray:
    from scripts.train_earlystop_semisup import _table_features_with_bundle

    return _table_features_with_bundle(table, config_name, bundle)


def _target_cache_keys(entries: list[Any], domain_specs: dict[str, dict[str, Any]]) -> set[str]:
    targets: set[str] = set()
    for entry in entries:
        if get_domain(entry.dataset_name) in domain_specs:
            targets.add(str(entry.cache_key))
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Export experimental EarlyStop SSL submission")
    parser.add_argument("--prebuilt-cache-dir", default="results/cache/es_svd_ms_rr_r1")
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--domain-spec", nargs="+", default=["math=denoise_full:16", "science=tokenpair_rrr:16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cca-reg", type=float, default=0.10)
    parser.add_argument("--mask-rate", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--feature-chunk-problems", type=int, default=8)
    parser.add_argument("--feature-cache-dir", default=DEFAULT_FEATURE_CACHE_DIR)
    parser.add_argument("--blind-feature-store-pkl", default="none")
    parser.add_argument("--refresh-feature-cache", action="store_true")
    parser.add_argument("--base-submission", default=DEFAULT_BASE_SUBMISSION)
    parser.add_argument("--method-name", default=DEFAULT_METHOD)
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument("--out-dir", default="submission/EarlyStop")
    parser.add_argument("--out-model", default=DEFAULT_OUT_MODEL)
    args = parser.parse_args()

    prebuilt_cache_dir = _resolve_path(args.prebuilt_cache_dir)
    cache_root = str(args.cache_root)
    out_dir = _resolve_path(args.out_dir)
    out_model = _resolve_path(args.out_model)
    base_submission_path = _resolve_path(args.base_submission)
    feature_cache_dir = _resolve_path(args.feature_cache_dir)
    blind_feature_store_pkl = str(args.blind_feature_store_pkl).strip()
    domain_specs = _parse_domain_specs(list(args.domain_spec))

    feature_store = _load_prebuilt_stores(prebuilt_cache_dir)
    domain_models = _fit_full_models(
        feature_store=feature_store,
        domain_specs=domain_specs,
        seed=int(args.seed),
        cca_reg=float(args.cca_reg),
        mask_rate=float(args.mask_rate),
    )

    model_bundle = {
        "bundle_version": 1,
        "created_at_utc": _now_utc(),
        "family": "earlystop_ssl_round1",
        "feature_names": list(FIXED_FEATURE_NAMES),
        "positions": [float(v) for v in EARLY_STOP_POSITIONS],
        "seed": int(args.seed),
        "domain_specs": {
            domain_name: {
                "config_name": str(spec["config_name"]),
                "rank": int(spec["rank"]),
            }
            for domain_name, spec in domain_specs.items()
        },
        "domains": domain_models,
    }
    out_model.parent.mkdir(parents=True, exist_ok=True)
    with out_model.open("wb") as handle:
        pickle.dump(model_bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)

    entries = discover_cache_entries(cache_root)
    expected_cache_keys = [str(entry.cache_key) for entry in entries]
    target_cache_keys = _target_cache_keys(entries, domain_specs)
    required_features = set(FIXED_FEATURE_NAMES)

    if blind_feature_store_pkl.lower() not in {"", "none", "off"}:
        blind_path = _resolve_path(blind_feature_store_pkl)
        with blind_path.open("rb") as handle:
            blind_obj = pickle.load(handle)
        feature_store_blind = list(blind_obj["feature_store"])
        feature_cache_path = blind_path
        feature_cache_status = "prebuilt"
        if target_cache_keys:
            feature_store_blind = [
                payload for payload in feature_store_blind
                if str(payload["cache_key"]) in target_cache_keys
            ]
    else:
        feature_store_blind, feature_cache_path, feature_cache_status = _load_or_build_feature_store(
            cache_root=cache_root,
            positions=tuple(float(v) for v in EARLY_STOP_POSITIONS),
            required_feature_names=required_features,
            max_problems=None,
            reflection_threshold=0.20,
            workers=int(args.workers),
            feature_chunk_problems=int(args.feature_chunk_problems),
            feature_cache_dir=feature_cache_dir,
            refresh_feature_cache=bool(args.refresh_feature_cache),
            include_cache_keys=target_cache_keys,
        )
    print(f"[blind] feature cache status={feature_cache_status} path={feature_cache_path}")

    score_map: dict[str, dict[str, Any]] = {}
    for domain_name, spec in domain_specs.items():
        config = spec["config"]
        model_info = domain_models[domain_name]
        score_fn = _make_score_fn(
            domain=domain_name,
            config=config,
            bundle_spec=model_info["bundle"],
            heads=model_info["heads"],
            baseline_mode="ssl",
        )
        for payload in feature_store_blind:
            if str(payload["domain"]) != domain_name:
                continue
            score_map[str(payload["cache_key"])] = _problem_scores_from_payload(payload, score_fn)

    base_submission = _load_json(base_submission_path)
    validate_earlystop_payload(base_submission)
    base_scores = dict(base_submission["scores"])
    for cache_key, score_obj in base_scores.items():
        if cache_key not in target_cache_keys:
            score_map[str(cache_key)] = score_obj

    missing = [cache_key for cache_key in expected_cache_keys if cache_key not in score_map]
    if missing:
        raise ValueError(f"Missing cache scores: {missing}")

    ordered_scores = [(cache_key, score_map[cache_key]) for cache_key in expected_cache_keys]
    payload = build_earlystop_payload(ordered_scores, method_name=str(args.method_name))
    stats = validate_earlystop_payload(payload)
    out_path = out_dir / str(args.filename)
    write_earlystop_payload(payload, out_path)

    manifest = {
        "generated_at_utc": _now_utc(),
        "method_name": str(args.method_name),
        "submission_path": str(out_path),
        "model_path": str(out_model),
        "base_submission": str(base_submission_path),
        "feature_cache_status": str(feature_cache_status),
        "feature_cache_path": None if feature_cache_path is None else str(feature_cache_path),
        "target_domains": sorted(domain_specs.keys()),
        "target_cache_keys": sorted(target_cache_keys),
        "validation": stats,
        "domain_specs": {
            domain_name: {
                "config_name": str(spec["config_name"]),
                "rank": int(spec["rank"]),
            }
            for domain_name, spec in domain_specs.items()
        },
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[out] model -> {out_model}")
    print(f"[out] submission -> {out_path}")
    print(f"[out] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
