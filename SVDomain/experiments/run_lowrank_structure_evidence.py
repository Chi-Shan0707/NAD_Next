#!/usr/bin/env python3
"""Measure low-rank structure in canonical SVDomain feature views."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.explain.svd_explain import get_anchor_route
from nad.ops.earlystop_svd import (
    FULL_FEATURE_NAMES,
    LEGACY_FULL_FEATURE_NAMES,
    _build_representation,
    _rank_transform_matrix,
    load_earlystop_svd_bundle,
)
from scripts.run_earlystop_prefix10_svd_round1 import (
    ANCHOR_POSITIONS,
    DEFAULT_FEATURE_CHUNK_PROBLEMS,
    EXTRACTION_POSITION_INDEX,
)
from SVDomain.train_es_svd_ms_rr_r1 import (
    FIXED_FEATURE_NAMES,
    _load_or_build_qualified_feature_store,
)


LEADING_SINGULAR_COUNT = 5
DEFAULT_NULL_REPS = 8
DEFAULT_BOOTSTRAP_REPS = 400
NULL_TYPE_ORDER = (
    "column_permutation",
    "within_group_permutation",
    "gaussian_covariance_matched",
)
NULL_LABELS = {
    "column_permutation": "Column permutation",
    "within_group_permutation": "Within-group permutation",
    "gaussian_covariance_matched": "Gaussian covariance-matched",
}
DOMAIN_ORDER = ("math", "science", "ms", "coding")
ANCHOR_ORDER = tuple(int(round(float(v) * 100.0)) for v in ANCHOR_POSITIONS)


@dataclass(frozen=True)
class DomainSpec:
    domain: str
    bundle_id: str
    bundle_path: str
    source_domains: tuple[str, ...]
    route_by_payload_domain: bool = False


@dataclass
class MatrixPack:
    domain: str
    bundle_id: str
    anchor_pct: int
    base_feature_count: int
    rep_feature_count: int
    n_rows: int
    n_route_groups: int
    n_problem_groups: int
    eval_blocks: list[np.ndarray]
    x_full: np.ndarray
    problem_group_grams: np.ndarray
    problem_group_rows: np.ndarray


DOMAIN_SPECS: dict[str, DomainSpec] = {
    "math": DomainSpec(
        domain="math",
        bundle_id="es_svd_math_rr_r1",
        bundle_path="models/ml_selectors/es_svd_math_rr_r1.pkl",
        source_domains=("math",),
        route_by_payload_domain=False,
    ),
    "science": DomainSpec(
        domain="science",
        bundle_id="es_svd_science_rr_r1",
        bundle_path="models/ml_selectors/es_svd_science_rr_r1.pkl",
        source_domains=("science",),
        route_by_payload_domain=False,
    ),
    "ms": DomainSpec(
        domain="ms",
        bundle_id="es_svd_ms_rr_r1",
        bundle_path="models/ml_selectors/es_svd_ms_rr_r1.pkl",
        source_domains=("math", "science"),
        route_by_payload_domain=True,
    ),
    "coding": DomainSpec(
        domain="coding",
        bundle_id="es_svd_coding_rr_r1",
        bundle_path="models/ml_selectors/es_svd_coding_rr_r1.pkl",
        source_domains=("coding",),
        route_by_payload_domain=False,
    ),
}


def _stable_seed(*parts: Any) -> int:
    raw = "::".join(str(part) for part in parts)
    return int(hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8], 16)


def _parse_csv(raw: str) -> list[str]:
    values = [token.strip() for token in str(raw).split(",") if token.strip()]
    if not values:
        raise ValueError("Need at least one token")
    return values


def _parse_anchor_csv(raw: str) -> list[int]:
    values: list[int] = []
    for token in _parse_csv(raw):
        value = float(token)
        if value <= 1.0:
            value = value * 100.0
        anchor_pct = int(round(value))
        if anchor_pct not in ANCHOR_ORDER:
            raise ValueError(f"Unsupported anchor: {token}")
        values.append(anchor_pct)
    return list(dict.fromkeys(values))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _json_array(values: np.ndarray | list[float], digits: int = 8) -> str:
    arr = np.asarray(values, dtype=np.float64)
    rounded = [round(float(v), digits) for v in arr.tolist()]
    return json.dumps(rounded, ensure_ascii=False, separators=(",", ":"))


def _infer_feature_axis_names(feature_dim: int) -> tuple[str, ...]:
    if int(feature_dim) == len(LEGACY_FULL_FEATURE_NAMES):
        return tuple(str(v) for v in LEGACY_FULL_FEATURE_NAMES)
    if int(feature_dim) == len(FULL_FEATURE_NAMES):
        return tuple(str(v) for v in FULL_FEATURE_NAMES)
    if int(feature_dim) <= len(FULL_FEATURE_NAMES):
        return tuple(str(v) for v in FULL_FEATURE_NAMES[: int(feature_dim)])
    raise ValueError(f"Unsupported feature dimension: {feature_dim}")


def _resolve_route_feature_indices(route: dict[str, Any], feature_axis_names: tuple[str, ...]) -> list[int]:
    indices = [int(v) for v in route.get("feature_indices", [])]
    if indices and max(indices) < len(feature_axis_names):
        return indices
    axis_index = {name: idx for idx, name in enumerate(feature_axis_names)}
    return [int(axis_index[str(name)]) for name in route.get("feature_names", [])]


def _safe_symmetric(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    return 0.5 * (arr + arr.T)


def _spectral_summary_from_gram(gram: np.ndarray) -> dict[str, Any]:
    gram_sym = _safe_symmetric(gram)
    eigvals = np.linalg.eigvalsh(gram_sym)
    eigvals = np.clip(eigvals, 0.0, None)[::-1]
    singular_values = np.sqrt(eigvals)
    total_energy = float(np.sum(eigvals))
    if total_energy > 0.0:
        explained = eigvals / total_energy
        cumulative = np.cumsum(explained)
        probs = explained[explained > 0.0]
        effective_rank = float(math.exp(-float(np.sum(probs * np.log(probs))))) if probs.size else 0.0
        sq_energy = float(np.sum(np.square(eigvals)))
        participation_ratio = float((total_energy * total_energy) / sq_energy) if sq_energy > 0.0 else 0.0
        stable_rank = float(total_energy / eigvals[0]) if eigvals[0] > 0.0 else 0.0
    else:
        explained = np.zeros_like(eigvals)
        cumulative = np.zeros_like(eigvals)
        effective_rank = 0.0
        participation_ratio = 0.0
        stable_rank = 0.0
    return {
        "singular_values": singular_values,
        "eigenvalues": eigvals,
        "explained_variance": explained,
        "cumulative_variance": cumulative,
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "stable_rank": stable_rank,
        "top1_variance_share": float(explained[0]) if explained.size else 0.0,
    }


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "ci_lo": float(np.percentile(arr, 2.5)),
        "ci_hi": float(np.percentile(arr, 97.5)),
    }


def _array_band(values: np.ndarray) -> dict[str, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D array for band summary")
    return {
        "median": np.median(arr, axis=0),
        "p05": np.percentile(arr, 5.0, axis=0),
        "p95": np.percentile(arr, 95.0, axis=0),
    }


def _load_feature_store_sources(
    *,
    main_cache_root: str,
    extra_cache_root: str,
    feature_cache_dir: Optional[Path],
    refresh_feature_cache: bool,
    feature_workers: int,
    chunk_problems: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    required_feature_names = set(str(name) for name in FIXED_FEATURE_NAMES)
    positions = tuple(float(v) for v in sorted(EXTRACTION_POSITION_INDEX.keys()))
    payloads: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}

    for source_name, cache_root in (("cache", main_cache_root), ("cache_train", extra_cache_root)):
        direct_loaded = False
        if feature_cache_dir is not None:
            direct_candidates = sorted(feature_cache_dir.glob(f"{source_name}_all_*.pkl"))
            if len(direct_candidates) == 1 and direct_candidates[0].exists():
                with direct_candidates[0].open("rb") as handle:
                    payload = pickle.load(handle)
                store = list(payload["feature_store"])
                payloads.extend(store)
                metadata[source_name] = {
                    "cache_root": str(cache_root),
                    "cache_path": str(direct_candidates[0]),
                    "cache_status": "loaded_direct_existing",
                    "n_payloads": int(len(store)),
                    "domains": sorted({str(item.get("domain")) for item in store}),
                }
                direct_loaded = True
        if direct_loaded:
            continue

        store, cache_path, cache_status = _load_or_build_qualified_feature_store(
            source_name=source_name,
            cache_root=cache_root,
            positions=positions,
            required_feature_names=required_feature_names,
            max_problems_per_cache=None,
            feature_workers=int(feature_workers),
            chunk_problems=int(chunk_problems),
            feature_cache_dir=feature_cache_dir,
            refresh_feature_cache=bool(refresh_feature_cache),
        )
        payloads.extend(store)
        metadata[source_name] = {
            "cache_root": str(cache_root),
            "cache_path": None if cache_path is None else str(cache_path),
            "cache_status": str(cache_status),
            "n_payloads": int(len(store)),
            "domains": sorted({str(payload.get("domain")) for payload in store}),
        }
    return payloads, metadata


def _build_matrix_pack(
    *,
    spec: DomainSpec,
    bundle: dict[str, Any],
    anchor_pct: int,
    feature_store: list[dict[str, Any]],
) -> MatrixPack:
    anchor_value = float(anchor_pct) / 100.0
    position_idx = EXTRACTION_POSITION_INDEX[anchor_value]
    eval_blocks: list[np.ndarray] = []
    split_group_to_blocks: dict[str, list[np.ndarray]] = {}
    rep_feature_count: Optional[int] = None
    base_feature_count: Optional[int] = None

    for payload in feature_store:
        payload_domain = str(payload["domain"])
        if payload_domain not in spec.source_domains:
            continue

        route_domain = payload_domain if spec.route_by_payload_domain else spec.domain
        route = dict(get_anchor_route(bundle, route_domain, anchor_value))
        tensor = np.asarray(payload["tensor"], dtype=np.float64)
        feature_axis_names = _infer_feature_axis_names(int(tensor.shape[2]))
        feature_indices = _resolve_route_feature_indices(route, feature_axis_names)
        representation = str(route.get("representation", "raw"))
        scaler = route["model"]["scaler"]
        problem_ids = [str(v) for v in payload["problem_ids"]]
        offsets = [int(v) for v in payload["problem_offsets"]]
        payload_x_raw = np.asarray(tensor[:, position_idx, :], dtype=np.float64)

        for problem_idx, problem_id in enumerate(problem_ids):
            start = offsets[problem_idx]
            end = offsets[problem_idx + 1]
            if end <= start:
                continue
            x_raw = np.asarray(payload_x_raw[start:end], dtype=np.float64)
            x_rank = _rank_transform_matrix(x_raw)
            x_rep = _build_representation(
                x_raw=x_raw,
                x_rank=x_rank,
                feature_indices=feature_indices,
                representation=representation,
            )
            x_std = np.asarray(scaler.transform(x_rep), dtype=np.float64)
            eval_blocks.append(x_std)
            split_group_key = f"{payload['dataset_name']}::{problem_id}"
            split_group_to_blocks.setdefault(split_group_key, []).append(x_std)

            if rep_feature_count is None:
                rep_feature_count = int(x_std.shape[1])
            if base_feature_count is None:
                base_feature_count = int(len(route.get("feature_names", [])))

    if not eval_blocks:
        raise RuntimeError(f"No rows found for domain={spec.domain} anchor={anchor_pct}")

    x_full = np.concatenate(eval_blocks, axis=0).astype(np.float64, copy=False)
    group_grams: list[np.ndarray] = []
    group_rows: list[int] = []
    for split_group_key in sorted(split_group_to_blocks.keys()):
        block_parts = split_group_to_blocks[split_group_key]
        block = block_parts[0] if len(block_parts) == 1 else np.concatenate(block_parts, axis=0)
        group_grams.append(np.asarray(block.T @ block, dtype=np.float64))
        group_rows.append(int(block.shape[0]))

    return MatrixPack(
        domain=spec.domain,
        bundle_id=spec.bundle_id,
        anchor_pct=int(anchor_pct),
        base_feature_count=int(base_feature_count or 0),
        rep_feature_count=int(rep_feature_count or x_full.shape[1]),
        n_rows=int(x_full.shape[0]),
        n_route_groups=int(len(eval_blocks)),
        n_problem_groups=int(len(group_rows)),
        eval_blocks=eval_blocks,
        x_full=x_full,
        problem_group_grams=np.stack(group_grams, axis=0).astype(np.float64, copy=False),
        problem_group_rows=np.asarray(group_rows, dtype=np.int64),
    )


def _bootstrap_summary(pack: MatrixPack, bootstrap_reps: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    n_groups = int(pack.problem_group_rows.shape[0])
    if n_groups <= 0:
        raise RuntimeError("Need at least one bootstrap group")

    singular_stack: list[np.ndarray] = []
    cumulative_stack: list[np.ndarray] = []
    effective_ranks: list[float] = []
    participation_ratios: list[float] = []
    stable_ranks: list[float] = []

    for _ in range(int(bootstrap_reps)):
        sampled = rng.integers(0, n_groups, size=n_groups)
        counts = np.bincount(sampled, minlength=n_groups).astype(np.float64)
        gram = np.tensordot(counts, pack.problem_group_grams, axes=(0, 0))
        summary = _spectral_summary_from_gram(gram)
        singular_stack.append(np.asarray(summary["singular_values"], dtype=np.float64))
        cumulative_stack.append(np.asarray(summary["cumulative_variance"], dtype=np.float64))
        effective_ranks.append(float(summary["effective_rank"]))
        participation_ratios.append(float(summary["participation_ratio"]))
        stable_ranks.append(float(summary["stable_rank"]))

    singular_arr = np.asarray(singular_stack, dtype=np.float64)
    cumulative_arr = np.asarray(cumulative_stack, dtype=np.float64)
    return {
        "bootstrap_reps": int(bootstrap_reps),
        "singular_values": singular_arr,
        "cumulative_variance": cumulative_arr,
        "effective_rank": np.asarray(effective_ranks, dtype=np.float64),
        "participation_ratio": np.asarray(participation_ratios, dtype=np.float64),
        "stable_rank": np.asarray(stable_ranks, dtype=np.float64),
    }


def _column_permutation_gram(x_full: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    permuted = np.empty_like(x_full)
    n_rows = int(x_full.shape[0])
    for feature_idx in range(int(x_full.shape[1])):
        permuted[:, feature_idx] = x_full[rng.permutation(n_rows), feature_idx]
    return np.asarray(permuted.T @ permuted, dtype=np.float64)


def _within_group_permutation_gram(eval_blocks: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    if not eval_blocks:
        raise RuntimeError("Need at least one evaluation block")
    rep_dim = int(eval_blocks[0].shape[1])
    gram = np.zeros((rep_dim, rep_dim), dtype=np.float64)
    for block in eval_blocks:
        if block.shape[0] <= 1:
            gram += np.asarray(block.T @ block, dtype=np.float64)
            continue
        permuted = np.empty_like(block)
        for feature_idx in range(int(block.shape[1])):
            permuted[:, feature_idx] = block[rng.permutation(block.shape[0]), feature_idx]
        gram += np.asarray(permuted.T @ permuted, dtype=np.float64)
    return gram


def _gaussian_sampler(x_full: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    mu = np.mean(x_full, axis=0)
    centered = x_full - mu
    denom = float(max(1, x_full.shape[0]))
    cov = np.asarray((centered.T @ centered) / denom, dtype=np.float64)
    cov = _safe_symmetric(cov)
    eigvals, eigvecs = np.linalg.eigh(cov)
    ridge = 0.0
    min_eig = float(np.min(eigvals))
    if min_eig < 0.0:
        ridge = float(-min_eig + 1e-10)
        eigvals = eigvals + ridge
    eigvals = np.clip(eigvals, 0.0, None)
    transform = eigvecs @ np.diag(np.sqrt(eigvals))
    return mu.astype(np.float64), transform.astype(np.float64), float(ridge)


def _gaussian_covariance_matched_gram(
    *,
    n_rows: int,
    mu: np.ndarray,
    transform: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    z = rng.standard_normal((int(n_rows), int(mu.shape[0])))
    samples = np.asarray(z @ transform.T + mu, dtype=np.float64)
    return np.asarray(samples.T @ samples, dtype=np.float64)


def _null_summaries(pack: MatrixPack, null_reps: int, seed: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    gaussian_mu, gaussian_transform, gaussian_ridge = _gaussian_sampler(pack.x_full)

    for null_type in NULL_TYPE_ORDER:
        rng = np.random.default_rng(int(_stable_seed(seed, pack.domain, pack.anchor_pct, null_type)))
        singular_stack: list[np.ndarray] = []
        cumulative_stack: list[np.ndarray] = []
        effective_ranks: list[float] = []
        participation_ratios: list[float] = []
        stable_ranks: list[float] = []
        top1_shares: list[float] = []

        for _ in range(int(null_reps)):
            if null_type == "column_permutation":
                gram = _column_permutation_gram(pack.x_full, rng)
            elif null_type == "within_group_permutation":
                gram = _within_group_permutation_gram(pack.eval_blocks, rng)
            elif null_type == "gaussian_covariance_matched":
                gram = _gaussian_covariance_matched_gram(
                    n_rows=pack.n_rows,
                    mu=gaussian_mu,
                    transform=gaussian_transform,
                    rng=rng,
                )
            else:
                raise ValueError(f"Unsupported null type: {null_type}")

            summary = _spectral_summary_from_gram(gram)
            singular_stack.append(np.asarray(summary["singular_values"], dtype=np.float64))
            cumulative_stack.append(np.asarray(summary["cumulative_variance"], dtype=np.float64))
            effective_ranks.append(float(summary["effective_rank"]))
            participation_ratios.append(float(summary["participation_ratio"]))
            stable_ranks.append(float(summary["stable_rank"]))
            top1_shares.append(float(summary["top1_variance_share"]))

        singular_arr = np.asarray(singular_stack, dtype=np.float64)
        cumulative_arr = np.asarray(cumulative_stack, dtype=np.float64)
        out[null_type] = {
            "null_type": str(null_type),
            "n_reps": int(null_reps),
            "singular_values": singular_arr,
            "cumulative_variance": cumulative_arr,
            "effective_rank": np.asarray(effective_ranks, dtype=np.float64),
            "participation_ratio": np.asarray(participation_ratios, dtype=np.float64),
            "stable_rank": np.asarray(stable_ranks, dtype=np.float64),
            "top1_variance_share": np.asarray(top1_shares, dtype=np.float64),
            "ridge_eps": float(gaussian_ridge if null_type == "gaussian_covariance_matched" else 0.0),
        }
    return out


def _spectra_row(
    *,
    pack: MatrixPack,
    real_summary: dict[str, Any],
    bootstrap_summary: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "domain": str(pack.domain),
        "bundle_id": str(pack.bundle_id),
        "anchor_pct": int(pack.anchor_pct),
        "n_rows": int(pack.n_rows),
        "n_route_groups": int(pack.n_route_groups),
        "n_problem_groups": int(pack.n_problem_groups),
        "base_feature_count": int(pack.base_feature_count),
        "rep_feature_count": int(pack.rep_feature_count),
        "rank_max": int(min(pack.n_rows, pack.rep_feature_count)),
        "singular_values_json": _json_array(real_summary["singular_values"]),
        "explained_variance_json": _json_array(real_summary["explained_variance"]),
        "cumulative_variance_json": _json_array(real_summary["cumulative_variance"]),
        "effective_rank": round(float(real_summary["effective_rank"]), 8),
        "participation_ratio": round(float(real_summary["participation_ratio"]), 8),
        "stable_rank": round(float(real_summary["stable_rank"]), 8),
        "top1_variance_share": round(float(real_summary["top1_variance_share"]), 8),
        "bootstrap_reps": int(bootstrap_summary["bootstrap_reps"]),
    }

    boot_er = _summary_stats(bootstrap_summary["effective_rank"])
    boot_pr = _summary_stats(bootstrap_summary["participation_ratio"])
    boot_sr = _summary_stats(bootstrap_summary["stable_rank"])
    row.update(
        {
            "bootstrap_effective_rank_median": round(float(boot_er["median"]), 8),
            "bootstrap_effective_rank_ci_lo": round(float(boot_er["ci_lo"]), 8),
            "bootstrap_effective_rank_ci_hi": round(float(boot_er["ci_hi"]), 8),
            "bootstrap_participation_ratio_median": round(float(boot_pr["median"]), 8),
            "bootstrap_participation_ratio_ci_lo": round(float(boot_pr["ci_lo"]), 8),
            "bootstrap_participation_ratio_ci_hi": round(float(boot_pr["ci_hi"]), 8),
            "bootstrap_stable_rank_median": round(float(boot_sr["median"]), 8),
            "bootstrap_stable_rank_ci_lo": round(float(boot_sr["ci_lo"]), 8),
            "bootstrap_stable_rank_ci_hi": round(float(boot_sr["ci_hi"]), 8),
            "bootstrap_singular_values_median_json": _json_array(_array_band(bootstrap_summary["singular_values"])["median"]),
            "bootstrap_singular_values_p05_json": _json_array(_array_band(bootstrap_summary["singular_values"])["p05"]),
            "bootstrap_singular_values_p95_json": _json_array(_array_band(bootstrap_summary["singular_values"])["p95"]),
        }
    )

    leading_count = min(LEADING_SINGULAR_COUNT, int(np.asarray(real_summary["singular_values"]).shape[0]))
    boot_sv = np.asarray(bootstrap_summary["singular_values"], dtype=np.float64)
    for idx in range(leading_count):
        stats = _summary_stats(boot_sv[:, idx])
        row[f"s{idx + 1}"] = round(float(real_summary["singular_values"][idx]), 8)
        row[f"bootstrap_s{idx + 1}_median"] = round(float(stats["median"]), 8)
        row[f"bootstrap_s{idx + 1}_ci_lo"] = round(float(stats["ci_lo"]), 8)
        row[f"bootstrap_s{idx + 1}_ci_hi"] = round(float(stats["ci_hi"]), 8)
    return row


def _null_row(
    *,
    pack: MatrixPack,
    null_type: str,
    null_summary: dict[str, Any],
) -> dict[str, Any]:
    singular_band = _array_band(null_summary["singular_values"])
    cumulative_band = _array_band(null_summary["cumulative_variance"])
    eff = _summary_stats(null_summary["effective_rank"])
    part = _summary_stats(null_summary["participation_ratio"])
    stable = _summary_stats(null_summary["stable_rank"])
    top1 = _summary_stats(null_summary["top1_variance_share"])
    row: dict[str, Any] = {
        "domain": str(pack.domain),
        "bundle_id": str(pack.bundle_id),
        "anchor_pct": int(pack.anchor_pct),
        "null_type": str(null_type),
        "null_label": str(NULL_LABELS[null_type]),
        "n_reps": int(null_summary["n_reps"]),
        "n_rows": int(pack.n_rows),
        "n_route_groups": int(pack.n_route_groups),
        "n_problem_groups": int(pack.n_problem_groups),
        "rep_feature_count": int(pack.rep_feature_count),
        "rank_max": int(min(pack.n_rows, pack.rep_feature_count)),
        "ridge_eps": round(float(null_summary["ridge_eps"]), 12),
        "singular_values_median_json": _json_array(singular_band["median"]),
        "singular_values_p05_json": _json_array(singular_band["p05"]),
        "singular_values_p95_json": _json_array(singular_band["p95"]),
        "cumulative_variance_median_json": _json_array(cumulative_band["median"]),
        "cumulative_variance_p05_json": _json_array(cumulative_band["p05"]),
        "cumulative_variance_p95_json": _json_array(cumulative_band["p95"]),
        "effective_rank_mean": round(float(eff["mean"]), 8),
        "effective_rank_median": round(float(eff["median"]), 8),
        "effective_rank_ci_lo": round(float(eff["ci_lo"]), 8),
        "effective_rank_ci_hi": round(float(eff["ci_hi"]), 8),
        "participation_ratio_mean": round(float(part["mean"]), 8),
        "participation_ratio_median": round(float(part["median"]), 8),
        "participation_ratio_ci_lo": round(float(part["ci_lo"]), 8),
        "participation_ratio_ci_hi": round(float(part["ci_hi"]), 8),
        "stable_rank_mean": round(float(stable["mean"]), 8),
        "stable_rank_median": round(float(stable["median"]), 8),
        "stable_rank_ci_lo": round(float(stable["ci_lo"]), 8),
        "stable_rank_ci_hi": round(float(stable["ci_hi"]), 8),
        "top1_variance_share_median": round(float(top1["median"]), 8),
        "top1_variance_share_ci_lo": round(float(top1["ci_lo"]), 8),
        "top1_variance_share_ci_hi": round(float(top1["ci_hi"]), 8),
    }
    leading_count = min(LEADING_SINGULAR_COUNT, int(np.asarray(singular_band["median"]).shape[0]))
    singular_values = np.asarray(null_summary["singular_values"], dtype=np.float64)
    for idx in range(leading_count):
        stats = _summary_stats(singular_values[:, idx])
        row[f"s{idx + 1}_median"] = round(float(stats["median"]), 8)
        row[f"s{idx + 1}_ci_lo"] = round(float(stats["ci_lo"]), 8)
        row[f"s{idx + 1}_ci_hi"] = round(float(stats["ci_hi"]), 8)
    return row


def _plot_scree(domain: str, anchor_payloads: dict[int, dict[str, Any]], out_path: Path) -> None:
    anchors = sorted(anchor_payloads.keys())
    n_cols = 2
    n_rows = int(math.ceil(len(anchors) / float(n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8), squeeze=False)
    color_map = {
        "column_permutation": "#1f77b4",
        "within_group_permutation": "#ff7f0e",
        "gaussian_covariance_matched": "#2ca02c",
    }

    for ax, anchor_pct in zip(axes.flat, anchors):
        payload = anchor_payloads[anchor_pct]
        real_summary = payload["real_summary"]
        null_summaries = payload["null_summaries"]
        components = np.arange(1, int(len(real_summary["singular_values"])) + 1)
        real_sv = np.maximum(np.asarray(real_summary["singular_values"], dtype=np.float64), 1e-8)
        ax.plot(
            components,
            real_sv,
            color="black",
            linewidth=2.0,
            label="Real",
        )
        for null_type in NULL_TYPE_ORDER:
            null_band = _array_band(null_summaries[null_type]["singular_values"])
            band_median = np.maximum(np.asarray(null_band["median"], dtype=np.float64), 1e-8)
            band_p05 = np.maximum(np.asarray(null_band["p05"], dtype=np.float64), 1e-8)
            band_p95 = np.maximum(np.asarray(null_band["p95"], dtype=np.float64), 1e-8)
            ax.plot(
                components,
                band_median,
                color=color_map[null_type],
                linewidth=1.6,
                label=NULL_LABELS[null_type],
            )
            ax.fill_between(
                components,
                band_p05,
                band_p95,
                color=color_map[null_type],
                alpha=0.12,
            )
        ax.set_title(f"{domain} @ {anchor_pct}%")
        ax.set_xlabel("Component")
        ax.set_ylabel("Singular value")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)

    for ax in axes.flat[len(anchors):]:
        ax.axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(f"Scree plots — {domain}", y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_cumulative_variance(domain: str, anchor_payloads: dict[int, dict[str, Any]], out_path: Path) -> None:
    anchors = sorted(anchor_payloads.keys())
    n_cols = 2
    n_rows = int(math.ceil(len(anchors) / float(n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8), squeeze=False)
    color_map = {
        "column_permutation": "#1f77b4",
        "within_group_permutation": "#ff7f0e",
        "gaussian_covariance_matched": "#2ca02c",
    }

    for ax, anchor_pct in zip(axes.flat, anchors):
        payload = anchor_payloads[anchor_pct]
        real_summary = payload["real_summary"]
        null_summaries = payload["null_summaries"]
        components = np.arange(1, int(len(real_summary["cumulative_variance"])) + 1)
        ax.plot(
            components,
            np.asarray(real_summary["cumulative_variance"], dtype=np.float64),
            color="black",
            linewidth=2.0,
            label="Real",
        )
        for null_type in NULL_TYPE_ORDER:
            null_band = _array_band(null_summaries[null_type]["cumulative_variance"])
            ax.plot(
                components,
                null_band["median"],
                color=color_map[null_type],
                linewidth=1.6,
                label=NULL_LABELS[null_type],
            )
            ax.fill_between(
                components,
                null_band["p05"],
                null_band["p95"],
                color=color_map[null_type],
                alpha=0.12,
            )
        ax.set_ylim(0.0, 1.02)
        ax.set_title(f"{domain} @ {anchor_pct}%")
        ax.set_xlabel("Component")
        ax.set_ylabel("Cumulative variance")
        ax.grid(True, alpha=0.25)

    for ax in axes.flat[len(anchors):]:
        ax.axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(f"Cumulative variance — {domain}", y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_effective_rank_comparison(results_by_domain: dict[str, dict[int, dict[str, Any]]], out_path: Path) -> None:
    domains = [domain for domain in DOMAIN_ORDER if domain in results_by_domain]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), squeeze=False)
    colors = {
        "real": "#111111",
        "column_permutation": "#1f77b4",
        "within_group_permutation": "#ff7f0e",
        "gaussian_covariance_matched": "#2ca02c",
    }

    for ax, domain in zip(axes.flat, domains):
        anchor_payloads = results_by_domain[domain]
        anchors = sorted(anchor_payloads.keys())
        x = np.arange(len(anchors), dtype=np.float64)
        width = 0.18

        real_values = [float(anchor_payloads[a]["real_summary"]["effective_rank"]) for a in anchors]
        real_boot = [anchor_payloads[a]["bootstrap_summary"]["effective_rank"] for a in anchors]
        real_ci_lo = [
            float(np.percentile(np.asarray(values, dtype=np.float64), 2.5))
            for values in real_boot
        ]
        real_ci_hi = [
            float(np.percentile(np.asarray(values, dtype=np.float64), 97.5))
            for values in real_boot
        ]
        yerr = np.vstack([
            np.asarray(real_values, dtype=np.float64) - np.asarray(real_ci_lo, dtype=np.float64),
            np.asarray(real_ci_hi, dtype=np.float64) - np.asarray(real_values, dtype=np.float64),
        ])
        ax.bar(x - 1.5 * width, real_values, width=width, color=colors["real"], label="Real")
        ax.errorbar(x - 1.5 * width, real_values, yerr=yerr, fmt="none", ecolor=colors["real"], capsize=3)

        for offset_idx, null_type in enumerate(NULL_TYPE_ORDER, start=1):
            medians = [
                float(_summary_stats(anchor_payloads[a]["null_summaries"][null_type]["effective_rank"])["median"])
                for a in anchors
            ]
            ci_lo = [
                float(_summary_stats(anchor_payloads[a]["null_summaries"][null_type]["effective_rank"])["ci_lo"])
                for a in anchors
            ]
            ci_hi = [
                float(_summary_stats(anchor_payloads[a]["null_summaries"][null_type]["effective_rank"])["ci_hi"])
                for a in anchors
            ]
            null_yerr = np.vstack([
                np.asarray(medians, dtype=np.float64) - np.asarray(ci_lo, dtype=np.float64),
                np.asarray(ci_hi, dtype=np.float64) - np.asarray(medians, dtype=np.float64),
            ])
            xpos = x + (offset_idx - 1.5) * width
            ax.bar(xpos, medians, width=width, color=colors[null_type], label=NULL_LABELS[null_type])
            ax.errorbar(xpos, medians, yerr=null_yerr, fmt="none", ecolor=colors[null_type], capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{anchor}%" for anchor in anchors])
        ax.set_ylabel("Effective rank")
        ax.set_title(domain)
        ax.grid(True, axis="y", alpha=0.25)

    for ax in axes.flat[len(domains):]:
        ax.axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Effective-rank comparison", y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _answer_sharper_than_nulls(results_by_domain: dict[str, dict[int, dict[str, Any]]]) -> tuple[str, list[str]]:
    total = 0
    permutation_hits = 0
    gaussian_hits = 0
    lines: list[str] = []
    for domain in DOMAIN_ORDER:
        if domain not in results_by_domain:
            continue
        domain_perm_hits = 0
        domain_gauss_hits = 0
        domain_total = 0
        for anchor_pct, payload in sorted(results_by_domain[domain].items()):
            total += 1
            domain_total += 1
            real_eff = float(payload["real_summary"]["effective_rank"])
            real_top1 = float(payload["real_summary"]["top1_variance_share"])
            column_eff = float(_summary_stats(payload["null_summaries"]["column_permutation"]["effective_rank"])["median"])
            column_top1 = float(_summary_stats(payload["null_summaries"]["column_permutation"]["top1_variance_share"])["median"])
            within_eff = float(_summary_stats(payload["null_summaries"]["within_group_permutation"]["effective_rank"])["median"])
            within_top1 = float(_summary_stats(payload["null_summaries"]["within_group_permutation"]["top1_variance_share"])["median"])
            gaussian_eff = float(_summary_stats(payload["null_summaries"]["gaussian_covariance_matched"]["effective_rank"])["median"])
            gaussian_top1 = float(_summary_stats(payload["null_summaries"]["gaussian_covariance_matched"]["top1_variance_share"])["median"])

            perm_hit = real_eff < column_eff and real_top1 > column_top1 and real_eff < within_eff and real_top1 > within_top1
            gauss_hit = real_eff < gaussian_eff and real_top1 > gaussian_top1
            if perm_hit:
                permutation_hits += 1
                domain_perm_hits += 1
            if gauss_hit:
                gaussian_hits += 1
                domain_gauss_hits += 1
        if domain_total > 0:
            lines.append(
                f"- `{domain}`: permutation-null separation at {domain_perm_hits}/{domain_total} anchors; "
                f"covariance-matched Gaussian separation at {domain_gauss_hits}/{domain_total} anchors."
            )

    if total == 0:
        return "No spectral comparisons were produced.", lines
    if permutation_hits == total and gaussian_hits <= total // 2:
        summary = (
            f"Real matrices are clearly sharper than both permutation nulls in all {permutation_hits}/{total} "
            f"domain-anchor settings. Against the covariance-matched Gaussian null, the spectra are usually similar "
            f"rather than cleanly separated ({gaussian_hits}/{total} settings beat the Gaussian median on both metrics), "
            f"which suggests the low-rank signature is primarily a covariance-level effect."
        )
    elif permutation_hits == total:
        summary = (
            f"Real matrices are sharper than the permutation nulls in all {permutation_hits}/{total} settings, "
            f"but the covariance-matched Gaussian null often reproduces a similar spectrum "
            f"({gaussian_hits}/{total} settings beat the Gaussian median on both metrics)."
        )
    else:
        summary = (
            f"Real matrices show stronger spectral decay than the permutation nulls in {permutation_hits}/{total} settings, "
            f"while Gaussian covariance matching reduces the separation further ({gaussian_hits}/{total} settings beat the Gaussian median on both metrics)."
        )
    return summary, lines


def _answer_coding_flatness(results_by_domain: dict[str, dict[int, dict[str, Any]]]) -> tuple[str, list[str]]:
    if "coding" not in results_by_domain:
        return "Coding was not included in this run.", []

    coding_payload = results_by_domain["coding"]
    lines: list[str] = []
    comparisons: dict[str, int] = {}
    for other_domain in ("math", "science", "ms"):
        if other_domain not in results_by_domain:
            continue
        wins = 0
        common_anchors = sorted(set(coding_payload.keys()) & set(results_by_domain[other_domain].keys()))
        for anchor_pct in common_anchors:
            coding_eff = float(coding_payload[anchor_pct]["real_summary"]["effective_rank"])
            coding_top1 = float(coding_payload[anchor_pct]["real_summary"]["top1_variance_share"])
            other_eff = float(results_by_domain[other_domain][anchor_pct]["real_summary"]["effective_rank"])
            other_top1 = float(results_by_domain[other_domain][anchor_pct]["real_summary"]["top1_variance_share"])
            if coding_eff > other_eff and coding_top1 < other_top1:
                wins += 1
        comparisons[other_domain] = wins
        lines.append(f"- Versus `{other_domain}`: coding is flatter at {wins}/{len(common_anchors)} matched anchors.")

    if comparisons.get("math", 0) == 4 and comparisons.get("science", 0) == 0 and comparisons.get("ms", 0) == 0:
        summary = (
            "No clear coding-is-flatter story emerges. Coding is consistently a bit flatter than math, "
            "but it is still sharper than science and the pooled `ms` view."
        )
    elif max(comparisons.values(), default=0) == 0:
        summary = "Coding does not look flatter than the comparison domains in this run."
    else:
        summary = "Coding differs from the other domains, but the flattening pattern is mixed rather than uniform."
    return summary, lines


def _answer_stability(results_by_domain: dict[str, dict[int, dict[str, Any]]]) -> tuple[str, list[str]]:
    lines: list[str] = []
    stable_domains = 0
    total_domains = 0
    for domain in DOMAIN_ORDER:
        if domain not in results_by_domain:
            continue
        total_domains += 1
        anchors = sorted(results_by_domain[domain].keys())
        effranks = np.asarray(
            [float(results_by_domain[domain][anchor]["real_summary"]["effective_rank"]) for anchor in anchors],
            dtype=np.float64,
        )
        top1 = np.asarray(
            [float(results_by_domain[domain][anchor]["real_summary"]["top1_variance_share"]) for anchor in anchors],
            dtype=np.float64,
        )
        rel_range = float((np.max(effranks) - np.min(effranks)) / max(np.mean(effranks), 1e-12))
        lines.append(
            f"- `{domain}`: effective-rank range {np.min(effranks):.2f}–{np.max(effranks):.2f}, "
            f"top-1 variance share {np.min(top1):.3f}–{np.max(top1):.3f}."
        )
        if rel_range <= 0.25:
            stable_domains += 1

    if total_domains == 0:
        return "No domain-level stability summary was available.", lines
    if stable_domains == total_domains:
        summary = "The low-rank signatures are fairly stable across anchors within each analyzed domain."
    elif stable_domains >= max(1, total_domains - 1):
        summary = "The low-rank signatures are mostly stable across anchors, with one domain showing noticeably larger spread."
    else:
        summary = "Anchor-to-anchor stability is mixed, so any cross-anchor claim should stay modest."
    return summary, lines


def _write_report(
    *,
    out_path: Path,
    results_by_domain: dict[str, dict[int, dict[str, Any]]],
    spectra_rows: list[dict[str, Any]],
    null_rows: list[dict[str, Any]],
    figure_dir: Path,
    bootstrap_reps: int,
    null_reps: int,
    feature_store_metadata: dict[str, Any],
) -> None:
    sharper_summary, sharper_lines = _answer_sharper_than_nulls(results_by_domain)
    coding_summary, coding_lines = _answer_coding_flatness(results_by_domain)
    stability_summary, stability_lines = _answer_stability(results_by_domain)

    row_map = {(str(row["domain"]), int(row["anchor_pct"])): row for row in spectra_rows}
    lines = [
        "# Low-Rank Structure Evidence",
        "",
        "## Setup",
        "",
        "- Uses the saved canonical `r1` bundles: `es_svd_math_rr_r1`, `es_svd_science_rr_r1`, `es_svd_ms_rr_r1`, and `es_svd_coding_rr_r1`.",
        "- Reconstructs the exact canonical `raw+rank` route inputs and then applies the frozen route `StandardScaler`; spectra are measured on this standardized pre-SVD matrix.",
        "- Pools rows from the canonical feature stores under the same data sources as the paper route (`MUI_HUB/cache` and `MUI_HUB/cache_train` where available).",
        "- Uses bootstrap units keyed by `dataset + problem_id` across roots, matching the canonical grouped split unit.",
        f"- Bootstrap: `{bootstrap_reps}` grouped replicates. Null controls: `{null_reps}` replicates per control.",
        "",
        "## Main Answers",
        "",
        f"### 1. Do real feature views show sharper spectral decay than nulls?",
        "",
        sharper_summary,
        "",
    ]
    lines.extend(sharper_lines)
    lines.extend(
        [
            "",
            f"### 2. Is coding meaningfully flatter than math/science?",
            "",
            coding_summary,
            "",
        ]
    )
    lines.extend(coding_lines)
    lines.extend(
        [
            "",
            f"### 3. Are the low-rank signatures stable across anchors and domains?",
            "",
            stability_summary,
            "",
        ]
    )
    lines.extend(stability_lines)
    lines.extend(
        [
            "",
            "## Selected Numbers",
            "",
            "| Domain | Anchor | Eff. rank | Participation ratio | Stable rank | Top-1 variance share |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for domain in DOMAIN_ORDER:
        if domain not in results_by_domain:
            continue
        for anchor_pct in sorted(results_by_domain[domain].keys()):
            row = row_map[(domain, anchor_pct)]
            lines.append(
                f"| {domain} | {anchor_pct}% | {float(row['effective_rank']):.2f} | "
                f"{float(row['participation_ratio']):.2f} | {float(row['stable_rank']):.2f} | "
                f"{float(row['top1_variance_share']):.3f} |"
            )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `results/tables/lowrank_structure_spectra.csv`",
            "- `results/tables/lowrank_structure_nulls.csv`",
            "- `results/figures/lowrank_structure/*.png`",
            "",
            "Figure files:",
        ]
    )
    for domain in DOMAIN_ORDER:
        if domain not in results_by_domain:
            continue
        lines.append(f"- `results/figures/lowrank_structure/scree_{domain}.png`")
        lines.append(f"- `results/figures/lowrank_structure/cumulative_variance_{domain}.png`")
    lines.append("- `results/figures/lowrank_structure/effective_rank_comparison.png`")
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- This is descriptive evidence about feature organization, not a new predictive evaluation.",
            "- Coding should still be treated as a boundary case: even if some low-rank signature appears, it does not imply the current coding route is strong downstream.",
            "- The spectra are measured after the frozen route scaler, so they reflect the paper-facing representation actually seen by the trained SVD head.",
            "- Within-group permutation preserves cache-local problem groups (`cache_key::problem_id`), while bootstrap grouping follows the split unit (`dataset + problem_id` across roots).",
            "",
            "## Cache Notes",
            "",
            f"- `cache` feature-store status: `{feature_store_metadata['cache']['cache_status']}`",
            f"- `cache_train` feature-store status: `{feature_store_metadata['cache_train']['cache_status']}`",
            f"- `cache` feature-store path: `{feature_store_metadata['cache']['cache_path']}`",
            f"- `cache_train` feature-store path: `{feature_store_metadata['cache_train']['cache_path']}`",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Low-rank structure evidence for canonical SVDomain feature views")
    ap.add_argument("--domains", default="math,science,ms,coding")
    ap.add_argument("--anchors", default="10,40,70,100")
    ap.add_argument("--main-cache-root", default="MUI_HUB/cache")
    ap.add_argument("--extra-cache-root", default="MUI_HUB/cache_train")
    ap.add_argument("--feature-cache-dir", default="results/cache/es_svd_ms_rr_r1")
    ap.add_argument("--refresh-feature-cache", action="store_true")
    ap.add_argument("--feature-workers", type=int, default=8)
    ap.add_argument("--feature-chunk-problems", type=int, default=DEFAULT_FEATURE_CHUNK_PROBLEMS)
    ap.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    ap.add_argument("--null-reps", type=int, default=DEFAULT_NULL_REPS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-spectra", default="results/tables/lowrank_structure_spectra.csv")
    ap.add_argument("--out-nulls", default="results/tables/lowrank_structure_nulls.csv")
    ap.add_argument("--figure-dir", default="results/figures/lowrank_structure")
    ap.add_argument("--out-report", default="docs/LOWRANK_STRUCTURE_EVIDENCE.md")
    args = ap.parse_args()

    selected_domains = [domain for domain in _parse_csv(args.domains) if domain in DOMAIN_SPECS]
    if not selected_domains:
        raise ValueError("No valid domains selected")
    selected_anchors = _parse_anchor_csv(args.anchors)

    feature_cache_dir = None if str(args.feature_cache_dir).strip().lower() in {"", "none", "off"} else (REPO_ROOT / str(args.feature_cache_dir)).resolve()
    out_spectra = REPO_ROOT / str(args.out_spectra)
    out_nulls = REPO_ROOT / str(args.out_nulls)
    figure_dir = REPO_ROOT / str(args.figure_dir)
    out_report = REPO_ROOT / str(args.out_report)

    start_time = time.time()
    feature_store, feature_store_metadata = _load_feature_store_sources(
        main_cache_root=str(args.main_cache_root),
        extra_cache_root=str(args.extra_cache_root),
        feature_cache_dir=feature_cache_dir,
        refresh_feature_cache=bool(args.refresh_feature_cache),
        feature_workers=int(args.feature_workers),
        chunk_problems=int(args.feature_chunk_problems),
    )

    bundles = {
        domain: load_earlystop_svd_bundle(REPO_ROOT / DOMAIN_SPECS[domain].bundle_path)
        for domain in selected_domains
    }

    spectra_rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    results_by_domain: dict[str, dict[int, dict[str, Any]]] = {}

    for domain in selected_domains:
        spec = DOMAIN_SPECS[domain]
        bundle = bundles[domain]
        results_by_domain.setdefault(domain, {})
        for anchor_pct in selected_anchors:
            print(f"[analyze] domain={domain:<7s} anchor={anchor_pct:>3d}%", flush=True)
            pack = _build_matrix_pack(
                spec=spec,
                bundle=bundle,
                anchor_pct=int(anchor_pct),
                feature_store=feature_store,
            )
            real_summary = _spectral_summary_from_gram(pack.x_full.T @ pack.x_full)
            bootstrap_summary = _bootstrap_summary(
                pack=pack,
                bootstrap_reps=int(args.bootstrap_reps),
                seed=int(_stable_seed(args.seed, domain, anchor_pct, "bootstrap")),
            )
            null_summaries = _null_summaries(
                pack=pack,
                null_reps=int(args.null_reps),
                seed=int(args.seed),
            )

            spectra_rows.append(
                _spectra_row(
                    pack=pack,
                    real_summary=real_summary,
                    bootstrap_summary=bootstrap_summary,
                )
            )
            for null_type in NULL_TYPE_ORDER:
                null_rows.append(
                    _null_row(
                        pack=pack,
                        null_type=null_type,
                        null_summary=null_summaries[null_type],
                    )
                )

            results_by_domain[domain][int(anchor_pct)] = {
                "pack": pack,
                "real_summary": real_summary,
                "bootstrap_summary": bootstrap_summary,
                "null_summaries": null_summaries,
            }

    _write_csv(out_spectra, spectra_rows)
    _write_csv(out_nulls, null_rows)

    for domain in selected_domains:
        if domain not in results_by_domain or not results_by_domain[domain]:
            continue
        _plot_scree(
            domain=domain,
            anchor_payloads=results_by_domain[domain],
            out_path=figure_dir / f"scree_{domain}.png",
        )
        _plot_cumulative_variance(
            domain=domain,
            anchor_payloads=results_by_domain[domain],
            out_path=figure_dir / f"cumulative_variance_{domain}.png",
        )

    _plot_effective_rank_comparison(results_by_domain=results_by_domain, out_path=figure_dir / "effective_rank_comparison.png")
    _write_report(
        out_path=out_report,
        results_by_domain=results_by_domain,
        spectra_rows=spectra_rows,
        null_rows=null_rows,
        figure_dir=figure_dir,
        bootstrap_reps=int(args.bootstrap_reps),
        null_reps=int(args.null_reps),
        feature_store_metadata=feature_store_metadata,
    )

    elapsed = time.time() - start_time
    print(
        f"[done] domains={','.join(selected_domains)} anchors={','.join(str(v) for v in selected_anchors)} "
        f"bootstrap={args.bootstrap_reps} nulls={args.null_reps} elapsed_sec={elapsed:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
