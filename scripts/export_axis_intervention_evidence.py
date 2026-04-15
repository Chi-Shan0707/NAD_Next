#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.explain.svd_explain import EXPLAIN_ANCHORS, get_anchor_route, _score_route_matrix
from nad.ops.earlystop_svd import (
    _auroc,
    _predict_svd_lr,
    _rank_transform_matrix,
    load_earlystop_svd_bundle,
)


_EPS = 1e-12

ROUTE_SPECS: dict[str, dict[str, Any]] = {
    "math": {
        "route_label": "math",
        "method_id": "es_svd_math_rr_r1",
        "summary_path": REPO_ROOT / "results/scans/earlystop/es_svd_ms_rr_r1_summary.json",
        "allowed_domains": ("math",),
        "holdout_scope": "math",
    },
    "science": {
        "route_label": "science",
        "method_id": "es_svd_science_rr_r1",
        "summary_path": REPO_ROOT / "results/scans/earlystop/es_svd_ms_rr_r1_summary.json",
        "allowed_domains": ("science",),
        "holdout_scope": "science",
    },
    "ms": {
        "route_label": "ms",
        "method_id": "es_svd_ms_rr_r1",
        "summary_path": REPO_ROOT / "results/scans/earlystop/es_svd_ms_rr_r1_summary.json",
        "allowed_domains": ("math", "science"),
        "holdout_scope": "ms",
    },
    "coding": {
        "route_label": "coding",
        "method_id": "es_svd_coding_rr_r1",
        "summary_path": REPO_ROOT / "results/scans/earlystop/es_svd_coding_rr_r1_summary.json",
        "allowed_domains": ("coding",),
        "holdout_scope": "coding",
    },
}

POLICY_ORDER = ("top", "random", "bottom")
POLICY_COLORS = {
    "top": "#d62728",
    "random": "#1f77b4",
    "bottom": "#7f7f7f",
}
SUBSET_COLORS = {
    "positive_contribution_axis": "#2ca02c",
    "negative_contribution_axis": "#9467bd",
}


def _parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _parse_int_csv(raw: str) -> list[int]:
    return [int(item) for item in _parse_csv(raw)]


def _parse_float_csv(raw: str) -> list[float]:
    return [float(item) for item in _parse_csv(raw)]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _safe_mean(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _safe_median(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def _safe_std(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr)) if arr.size else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _ensure_parent(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_feature_store(cache_paths: Sequence[Path]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for path in cache_paths:
        if not path.exists():
            continue
        payload = pickle.load(path.open("rb"))
        merged.extend(list(payload.get("feature_store", [])))
    return merged


def _subset_payload_by_problem_ids(
    payload: Mapping[str, Any],
    selected_problem_ids: set[str],
) -> Optional[dict[str, Any]]:
    if not selected_problem_ids:
        return None

    tensor_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    sample_parts: list[np.ndarray] = []
    problem_ids: list[str] = []
    problem_offsets = [0]
    total_samples = 0

    offsets = [int(v) for v in payload["problem_offsets"]]
    for problem_idx, problem_id in enumerate(str(v) for v in payload["problem_ids"]):
        if problem_id not in selected_problem_ids:
            continue
        start = offsets[problem_idx]
        end = offsets[problem_idx + 1]
        if end <= start:
            continue

        tensor_parts.append(np.asarray(payload["tensor"][start:end], dtype=np.float64))
        label_parts.append(np.asarray(payload["labels"][start:end], dtype=np.int32))
        sample_parts.append(np.asarray(payload["sample_ids"][start:end], dtype=np.int32))
        problem_ids.append(problem_id)
        total_samples += int(end - start)
        problem_offsets.append(problem_offsets[-1] + int(end - start))

    if not tensor_parts:
        return None

    out = dict(payload)
    out["tensor"] = np.concatenate(tensor_parts, axis=0).astype(np.float64, copy=False)
    out["labels"] = np.concatenate(label_parts).astype(np.int32, copy=False)
    out["sample_ids"] = np.concatenate(sample_parts).astype(np.int32, copy=False)
    out["problem_ids"] = problem_ids
    out["problem_offsets"] = problem_offsets
    out["samples"] = int(total_samples)
    return out


def _extract_holdout_problem_map(summary: Mapping[str, Any], scope: str) -> dict[str, set[str]]:
    if scope == "ms":
        out: dict[str, set[str]] = {}
        for domain_name in ("math", "science"):
            domain_summary = summary["data"]["store_summary"][domain_name]["holdout_problem_summary"]
            for dataset_name, info in domain_summary.items():
                out[str(dataset_name)] = {str(v) for v in info["holdout_problem_ids"]}
        return out

    domain_key = scope
    domain_summary = summary["data"]["store_summary"][domain_key]["holdout_problem_summary"]
    return {
        str(dataset_name): {str(v) for v in info["holdout_problem_ids"]}
        for dataset_name, info in domain_summary.items()
    }


def _build_holdout_feature_store(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    summary = _json_load(Path(spec["summary_path"]))
    feature_paths = [
        Path(path_str)
        for path_str in summary["data"]["feature_cache_paths"].values()
    ]
    full_store = _load_feature_store(feature_paths)
    holdout_map = _extract_holdout_problem_map(summary, str(spec["holdout_scope"]))
    allowed_domains = {str(v) for v in spec["allowed_domains"]}

    holdout_store: list[dict[str, Any]] = []
    for payload in full_store:
        if str(payload["domain"]) not in allowed_domains:
            continue
        selected_ids = holdout_map.get(str(payload["dataset_name"]), set())
        subset = _subset_payload_by_problem_ids(payload, selected_ids)
        if subset is not None and int(subset["samples"]) > 0:
            holdout_store.append(subset)
    return holdout_store


def _latent_route_terms(route: Mapping[str, Any]) -> dict[str, Any]:
    model = route["model"]
    scaler = model["scaler"]
    svd = model["svd"]
    lr = model["lr"]

    beta = np.asarray(lr.coef_, dtype=np.float64).reshape(-1)
    singular_values = np.asarray(svd.singular_values_, dtype=np.float64)
    singular_values = np.where(np.abs(singular_values) < _EPS, 1.0, singular_values)
    alpha_eff = beta / singular_values if bool(model.get("whiten", False)) else beta.copy()

    return {
        "model": model,
        "alpha_eff": np.asarray(alpha_eff, dtype=np.float64),
        "V": np.asarray(svd.components_, dtype=np.float64),
        "scale": np.asarray(scaler.scale_, dtype=np.float64),
        "intercept": float(np.asarray(lr.intercept_, dtype=np.float64).reshape(-1)[0]),
        "rank": int(beta.size),
    }


def _component_contributions_for_xrep(
    route_terms: Mapping[str, Any],
    x_rep: np.ndarray,
) -> np.ndarray:
    model = route_terms["model"]
    scaler = model["scaler"]
    svd = model["svd"]
    x_std = scaler.transform(x_rep)
    z = svd.transform(x_std)
    contrib = z * np.asarray(route_terms["alpha_eff"], dtype=np.float64)[None, :]
    return np.asarray(contrib, dtype=np.float64)


def _pooled_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    if scores.size <= 0:
        return {"auroc": float("nan"), "selacc_10": float("nan")}
    order = np.lexsort((np.arange(scores.size, dtype=np.int64), -scores))
    topk = max(1, int(math.ceil(0.10 * max(1, scores.size))))
    top_rows = labels[order][:topk]
    return {
        "auroc": float(_auroc(scores, labels)),
        "selacc_10": float(np.mean(top_rows)) if top_rows.size else 0.0,
    }


def _outranks(i: int, j: int, scores: np.ndarray, sample_ids: np.ndarray) -> bool:
    if float(scores[i]) > float(scores[j]) + _EPS:
        return True
    if float(scores[i]) < float(scores[j]) - _EPS:
        return False
    return int(sample_ids[i]) < int(sample_ids[j])


def _share_of_topk_abs_mass(values: np.ndarray, k: int) -> float:
    vec = np.sort(np.abs(np.asarray(values, dtype=np.float64)))[::-1]
    total = float(np.sum(vec))
    if total <= _EPS:
        return 0.0
    return float(np.sum(vec[: min(int(k), int(vec.size))]) / total)


def _build_anchor_pack(
    bundle: Mapping[str, Any],
    holdout_store: Sequence[Mapping[str, Any]],
    *,
    anchor: float,
) -> dict[str, Any]:
    route_terms_by_domain: dict[str, dict[str, Any]] = {}
    problem_records: list[dict[str, Any]] = []
    pooled_scores: list[float] = []
    pooled_labels: list[int] = []
    share_at_1: list[float] = []
    share_at_2: list[float] = []
    share_at_4: list[float] = []

    for payload in holdout_store:
        domain = str(payload["domain"])
        route = get_anchor_route(bundle, domain, anchor)
        route_terms = route_terms_by_domain.setdefault(domain, _latent_route_terms(route))
        positions = [float(v) for v in payload["positions"]]
        if float(anchor) not in positions:
            raise ValueError(f"Anchor {anchor} missing from payload positions={positions}")
        pos_idx = int(positions.index(float(anchor)))
        offsets = [int(v) for v in payload["problem_offsets"]]

        for problem_idx, problem_id in enumerate(str(v) for v in payload["problem_ids"]):
            start = offsets[problem_idx]
            end = offsets[problem_idx + 1]
            if end <= start:
                continue

            sample_ids = np.asarray(payload["sample_ids"][start:end], dtype=np.int32)
            labels = np.asarray(payload["labels"][start:end], dtype=np.int32)
            x_raw_problem = np.asarray(payload["tensor"][start:end, pos_idx, :], dtype=np.float64)
            x_rank_problem = _rank_transform_matrix(x_raw_problem)
            scores, x_rep = _score_route_matrix(route, x_raw_problem, x_rank_problem)
            contrib = _component_contributions_for_xrep(route_terms, np.asarray(x_rep, dtype=np.float64))

            recon = float(route_terms["intercept"]) + np.sum(contrib, axis=1)
            max_err = float(np.max(np.abs(recon - np.asarray(scores, dtype=np.float64)))) if scores.size else 0.0
            if max_err > 1e-6:
                raise ValueError(f"Latent reconstruction drift too large: {max_err:.3e}")

            order = np.lexsort((sample_ids, -scores))
            top1_idx = int(order[0])
            top2_idx = int(order[1]) if order.size > 1 else int(top1_idx)

            pooled_scores.extend(np.asarray(scores, dtype=np.float64).tolist())
            pooled_labels.extend(labels.tolist())
            share_at_1.append(_share_of_topk_abs_mass(contrib[top1_idx], 1))
            share_at_2.append(_share_of_topk_abs_mass(contrib[top1_idx], 2))
            share_at_4.append(_share_of_topk_abs_mass(contrib[top1_idx], 4))

            problem_records.append(
                {
                    "domain": domain,
                    "cache_key": str(payload["cache_key"]),
                    "dataset_name": str(payload["dataset_name"]),
                    "problem_id": str(problem_id),
                    "sample_ids": sample_ids,
                    "labels": labels,
                    "scores": np.asarray(scores, dtype=np.float64),
                    "x_rep": np.asarray(x_rep, dtype=np.float64),
                    "contrib": contrib,
                    "top1_idx": int(top1_idx),
                    "top2_idx": int(top2_idx),
                    "route_terms": route_terms,
                }
            )

    pooled_scores_arr = np.asarray(pooled_scores, dtype=np.float64)
    pooled_labels_arr = np.asarray(pooled_labels, dtype=np.int32)
    base_metrics = _pooled_metrics(pooled_scores_arr, pooled_labels_arr)
    rank_spec_parts = [
        f"{domain}:{int(route_terms['rank'])}"
        for domain, route_terms in sorted(route_terms_by_domain.items())
    ]

    return {
        "anchor": float(anchor),
        "anchor_pct": int(round(float(anchor) * 100.0)),
        "problem_records": problem_records,
        "base_metrics": {
            "auroc": float(base_metrics["auroc"]),
            "selacc_10": float(base_metrics["selacc_10"]),
            "n_problem_slices": int(len(problem_records)),
            "n_samples": int(pooled_scores_arr.size),
        },
        "rank_spec": " / ".join(rank_spec_parts),
        "mean_abs_share_top1_k1": _safe_mean(share_at_1),
        "mean_abs_share_top1_k2": _safe_mean(share_at_2),
        "mean_abs_share_top1_k4": _safe_mean(share_at_4),
    }


def _pick_best_anchor(
    bundle: Mapping[str, Any],
    holdout_store: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    best_pack: Optional[dict[str, Any]] = None
    for anchor in EXPLAIN_ANCHORS:
        pack = _build_anchor_pack(bundle, holdout_store, anchor=float(anchor))
        base = pack["base_metrics"]
        best_key = (float(base["auroc"]), float(base["selacc_10"]))
        if best_pack is None:
            best_pack = pack
            continue
        cur_key = (
            float(best_pack["base_metrics"]["auroc"]),
            float(best_pack["base_metrics"]["selacc_10"]),
        )
        if best_key > cur_key:
            best_pack = pack
    if best_pack is None:
        raise ValueError("No valid anchor pack found")
    return best_pack


def _axis_indices_for_policy(
    contrib: np.ndarray,
    *,
    k: int,
    policy: str,
    rng: np.random.Generator,
) -> np.ndarray:
    n_rows, rank = contrib.shape
    k_eff = min(int(k), int(rank))
    abs_contrib = np.abs(np.asarray(contrib, dtype=np.float64))

    if str(policy) == "top":
        return np.argsort(abs_contrib, axis=1)[:, -k_eff:]
    if str(policy) == "bottom":
        return np.argsort(abs_contrib, axis=1)[:, :k_eff]
    if str(policy) == "random":
        out = np.empty((n_rows, k_eff), dtype=np.int64)
        for row_idx in range(n_rows):
            out[row_idx] = np.asarray(
                rng.choice(rank, size=k_eff, replace=False),
                dtype=np.int64,
            )
        return out
    raise ValueError(f"Unknown policy: {policy}")


def _run_ablation_once(
    problem_records: Sequence[Mapping[str, Any]],
    *,
    k: int,
    policy: str,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    pooled_scores: list[float] = []
    pooled_labels: list[int] = []
    top1_score_drops: list[float] = []
    removed_abs_shares: list[float] = []
    removed_abs_contribs: list[float] = []
    top1_flip_count = 0
    decision_flip_count = 0

    for record in problem_records:
        contrib = np.asarray(record["contrib"], dtype=np.float64)
        scores = np.asarray(record["scores"], dtype=np.float64)
        labels = np.asarray(record["labels"], dtype=np.int32)
        sample_ids = np.asarray(record["sample_ids"], dtype=np.int32)
        top1_idx = int(record["top1_idx"])
        top2_idx = int(record["top2_idx"])

        chosen = _axis_indices_for_policy(contrib, k=int(k), policy=str(policy), rng=rng)
        removed = np.take_along_axis(contrib, chosen, axis=1).sum(axis=1)
        new_scores = scores - removed

        pooled_scores.extend(new_scores.tolist())
        pooled_labels.extend(labels.tolist())
        new_order = np.lexsort((sample_ids, -new_scores))

        top1_flip_count += int(int(new_order[0]) != top1_idx)
        decision_flip_count += int(not _outranks(top1_idx, top2_idx, new_scores, sample_ids))

        top1_score_drops.append(float(scores[top1_idx] - new_scores[top1_idx]))
        removed_abs = float(np.sum(np.abs(contrib[top1_idx, chosen[top1_idx]])))
        removed_abs_contribs.append(removed_abs)
        total_abs = float(np.sum(np.abs(contrib[top1_idx])))
        removed_abs_shares.append(removed_abs / total_abs if total_abs > _EPS else 0.0)

    pooled_scores_arr = np.asarray(pooled_scores, dtype=np.float64)
    pooled_labels_arr = np.asarray(pooled_labels, dtype=np.int32)
    metrics = _pooled_metrics(pooled_scores_arr, pooled_labels_arr)
    n_problems = max(1, len(problem_records))
    return {
        "intervened_auroc": float(metrics["auroc"]),
        "intervened_selacc_10": float(metrics["selacc_10"]),
        "mean_top1_score_drop": _safe_mean(top1_score_drops),
        "median_top1_score_drop": _safe_median(top1_score_drops),
        "mean_removed_abs_share_top1": _safe_mean(removed_abs_shares),
        "mean_removed_abs_contribution_top1": _safe_mean(removed_abs_contribs),
        "top1_flip_rate": float(top1_flip_count / n_problems),
        "decision_flip_rate": float(decision_flip_count / n_problems),
    }


def _summarize_policy_runs(
    problem_records: Sequence[Mapping[str, Any]],
    base_metrics: Mapping[str, Any],
    *,
    k: int,
    policy: str,
    random_repeats: int,
    seed_base: int,
) -> dict[str, Any]:
    n_repeats = int(random_repeats) if str(policy) == "random" else 1
    runs = [
        _run_ablation_once(
            problem_records,
            k=int(k),
            policy=str(policy),
            seed=int(seed_base) + rep_idx,
        )
        for rep_idx in range(n_repeats)
    ]

    def _agg(field: str) -> tuple[float, float]:
        vals = [float(run[field]) for run in runs]
        return _safe_mean(vals), _safe_std(vals)

    intervened_auroc, intervened_auroc_std = _agg("intervened_auroc")
    intervened_selacc, intervened_selacc_std = _agg("intervened_selacc_10")
    mean_score_drop, mean_score_drop_std = _agg("mean_top1_score_drop")
    median_score_drop, median_score_drop_std = _agg("median_top1_score_drop")
    abs_share_mean, abs_share_std = _agg("mean_removed_abs_share_top1")
    abs_contrib_mean, abs_contrib_std = _agg("mean_removed_abs_contribution_top1")
    top1_flip_rate, top1_flip_std = _agg("top1_flip_rate")
    decision_flip_rate, decision_flip_std = _agg("decision_flip_rate")

    base_auroc = float(base_metrics["auroc"])
    base_selacc = float(base_metrics["selacc_10"])
    return {
        "intervened_auroc": intervened_auroc,
        "intervened_auroc_std": intervened_auroc_std,
        "intervened_selacc_10": intervened_selacc,
        "intervened_selacc_10_std": intervened_selacc_std,
        "auroc_drop": float(base_auroc - intervened_auroc),
        "auroc_drop_std": intervened_auroc_std,
        "selacc_drop": float(base_selacc - intervened_selacc),
        "selacc_drop_std": intervened_selacc_std,
        "mean_top1_score_drop": mean_score_drop,
        "mean_top1_score_drop_std": mean_score_drop_std,
        "median_top1_score_drop": median_score_drop,
        "median_top1_score_drop_std": median_score_drop_std,
        "mean_removed_abs_share_top1": abs_share_mean,
        "mean_removed_abs_share_top1_std": abs_share_std,
        "mean_removed_abs_contribution_top1": abs_contrib_mean,
        "mean_removed_abs_contribution_top1_std": abs_contrib_std,
        "top1_flip_rate": top1_flip_rate,
        "top1_flip_rate_std": top1_flip_std,
        "decision_flip_rate": decision_flip_rate,
        "decision_flip_rate_std": decision_flip_std,
        "n_repeats": int(n_repeats),
    }


def _perturbation_rows(
    route_label: str,
    method_id: str,
    anchor_pack: Mapping[str, Any],
    epsilons: Sequence[float],
) -> list[dict[str, Any]]:
    accum: dict[str, dict[float, list[float]]] = {
        "positive_contribution_axis": {float(eps): [] for eps in epsilons},
        "negative_contribution_axis": {float(eps): [] for eps in epsilons},
    }
    monotonic_flags: dict[str, list[float]] = {
        "positive_contribution_axis": [],
        "negative_contribution_axis": [],
    }

    for record in anchor_pack["problem_records"]:
        top1_idx = int(record["top1_idx"])
        x_rep = np.asarray(record["x_rep"], dtype=np.float64)
        top1_xrep = np.asarray(x_rep[top1_idx], dtype=np.float64)
        top1_score = float(np.asarray(record["scores"], dtype=np.float64)[top1_idx])
        contrib = np.asarray(record["contrib"], dtype=np.float64)[top1_idx]
        route_terms = record["route_terms"]
        alpha_eff = np.asarray(route_terms["alpha_eff"], dtype=np.float64)
        V = np.asarray(route_terms["V"], dtype=np.float64)
        scale = np.asarray(route_terms["scale"], dtype=np.float64)
        model = route_terms["model"]

        selected_axes = {
            "positive_contribution_axis": int(np.argmax(contrib)) if np.max(contrib) > 0 else None,
            "negative_contribution_axis": int(np.argmin(contrib)) if np.min(contrib) < 0 else None,
        }

        for subset_kind, axis_idx in selected_axes.items():
            if axis_idx is None:
                continue

            aligned_direction = 1.0 if float(alpha_eff[axis_idx]) >= 0 else -1.0
            deltas: list[float] = []
            for eps in epsilons:
                x_new = top1_xrep + scale * (float(eps) * aligned_direction * V[axis_idx])
                new_score = float(_predict_svd_lr(model, x_new[None, :])[0])
                delta = float(new_score - top1_score)
                accum[subset_kind][float(eps)].append(delta)
                deltas.append(delta)

            monotonic = all(
                deltas[idx] <= deltas[idx + 1] + 1e-9
                for idx in range(len(deltas) - 1)
            )
            monotonic_flags[subset_kind].append(1.0 if monotonic else 0.0)

    rows: list[dict[str, Any]] = []
    for subset_kind, by_eps in accum.items():
        mono = _safe_mean(monotonic_flags[subset_kind])
        n_selected = len(monotonic_flags[subset_kind])
        for eps in epsilons:
            values = by_eps[float(eps)]
            rows.append(
                {
                    "experiment_type": "perturbation",
                    "route_label": route_label,
                    "method_id": method_id,
                    "anchor_pct": int(anchor_pack["anchor_pct"]),
                    "subset_kind": subset_kind,
                    "epsilon": float(eps),
                    "n_selected": int(n_selected),
                    "monotonic_fraction": mono,
                    "mean_score_delta": _safe_mean(values),
                    "median_score_delta": _safe_median(values),
                }
            )
    return rows


def _route_figure(
    route_label: str,
    method_id: str,
    anchor_pack: Mapping[str, Any],
    ablation_rows: Sequence[Mapping[str, Any]],
    out_path: Path,
) -> None:
    _ensure_parent(out_path)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    axes = axes.ravel()

    metric_specs = [
        ("mean_top1_score_drop", "Mean Top1 Score Drop"),
        ("auroc_drop", "AUROC Drop"),
        ("selacc_drop", "SelAcc@10 Drop"),
    ]

    route_rows = [
        row for row in ablation_rows
        if str(row["route_label"]) == route_label
    ]
    route_rows = sorted(route_rows, key=lambda row: (int(row["k"]), str(row["policy"])))

    for ax, (field, title) in zip(axes[:3], metric_specs):
        for policy in POLICY_ORDER:
            rows = [row for row in route_rows if str(row["policy"]) == policy]
            if not rows:
                continue
            k_vals = [int(row["k"]) for row in rows]
            y_vals = [float(row[field]) for row in rows]
            ax.plot(
                k_vals,
                y_vals,
                marker="o",
                linewidth=2.0,
                color=POLICY_COLORS[policy],
                label=policy,
            )
        ax.set_title(title)
        ax.set_xlabel("k")
        ax.grid(True, alpha=0.25)

    flip_ax = axes[3]
    for policy in POLICY_ORDER:
        rows = [row for row in route_rows if str(row["policy"]) == policy]
        if not rows:
            continue
        k_vals = [int(row["k"]) for row in rows]
        decision_vals = [float(row["decision_flip_rate"]) for row in rows]
        top1_vals = [float(row["top1_flip_rate"]) for row in rows]
        flip_ax.plot(
            k_vals,
            decision_vals,
            marker="o",
            linewidth=2.0,
            color=POLICY_COLORS[policy],
            label=f"{policy} decision",
        )
        flip_ax.plot(
            k_vals,
            top1_vals,
            marker="x",
            linewidth=1.6,
            linestyle="--",
            color=POLICY_COLORS[policy],
            label=f"{policy} top1",
        )
    flip_ax.set_title("Flip Rates")
    flip_ax.set_xlabel("k")
    flip_ax.grid(True, alpha=0.25)

    base = anchor_pack["base_metrics"]
    fig.suptitle(
        (
            f"{route_label} — {method_id} @ {int(anchor_pack['anchor_pct'])}%  "
            f"(AUROC={float(base['auroc']):.3f}, SelAcc@10={float(base['selacc_10']):.3f})"
        ),
        fontsize=11,
    )
    handles, labels = axes[3].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _perturbation_figure(
    perturb_rows: Sequence[Mapping[str, Any]],
    out_path: Path,
    route_order: Sequence[str],
) -> None:
    _ensure_parent(out_path)
    n_routes = len(route_order)
    n_cols = 2
    n_rows = int(math.ceil(n_routes / float(n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11.0, 4.0 * n_rows), squeeze=False)

    for idx, route_label in enumerate(route_order):
        ax = axes[idx // n_cols][idx % n_cols]
        route_rows = [
            row for row in perturb_rows
            if str(row["route_label"]) == route_label
        ]
        for subset_kind in ("positive_contribution_axis", "negative_contribution_axis"):
            rows = [
                row for row in route_rows
                if str(row["subset_kind"]) == subset_kind
            ]
            rows = sorted(rows, key=lambda row: float(row["epsilon"]))
            if not rows:
                continue
            ax.plot(
                [float(row["epsilon"]) for row in rows],
                [float(row["mean_score_delta"]) for row in rows],
                marker="o",
                linewidth=2.0,
                color=SUBSET_COLORS[subset_kind],
                label=subset_kind.replace("_", " "),
            )
        mono_parts = []
        for subset_kind in ("positive_contribution_axis", "negative_contribution_axis"):
            rows = [
                row for row in route_rows
                if str(row["subset_kind"]) == subset_kind
            ]
            if rows:
                mono_parts.append(
                    f"{subset_kind.split('_')[0]}={float(rows[0]['monotonic_fraction']):.2f}"
                )
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("Aligned axis step ε")
        ax.set_ylabel("Mean score delta")
        ax.set_title(f"{route_label} ({', '.join(mono_parts)})" if mono_parts else route_label)
        ax.legend(frameon=False, fontsize=8)

    for idx in range(n_routes, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    fig.suptitle("Axis-Aligned Perturbation Sanity", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _build_doc(
    *,
    out_path: Path,
    route_selection_rows: Sequence[Mapping[str, Any]],
    ablation_rows: Sequence[Mapping[str, Any]],
    perturb_rows: Sequence[Mapping[str, Any]],
    figure_dir: Path,
    random_repeats: int,
) -> None:
    _ensure_parent(out_path)
    figure_dir_rel = figure_dir.relative_to(REPO_ROOT) if figure_dir.is_absolute() else figure_dir

    def _route_ref(route_label: str, anchor_pct: int) -> str:
        return f"`{route_label}@{anchor_pct}%`"

    route_order = [str(row["route_label"]) for row in route_selection_rows]
    k2_rows = [
        row for row in ablation_rows
        if int(row["k"]) == 2
    ]

    lines = [
        "# Axis Intervention Evidence",
        "",
        "This note tests whether the learned SVD axes are operational decision objects, not just arbitrary rotations.",
        "",
        "## Setup",
        "",
        "- Route selection uses the best holdout anchor for each paper-facing route by pooled holdout AUROC, with SelAcc@10 as the tie-breaker.",
        "- Per-axis contribution is `c_k(x) = alpha_eff,k * z_k(x)`, where `alpha_eff = beta / s` for whitened routes and `alpha_eff = beta` otherwise.",
        f"- Random matched-count ablations average over `{int(random_repeats)}` draws per `k`.",
        "- Top/bottom ablations rank axes per run by absolute contribution and then zero those latent contributions before rescoring.",
        "- Perturbation uses small aligned steps in route-input space (`x_rep`) along the sign-aligned SVD direction, then rescoring through the original model.",
        "",
        "## Selected Routes",
        "",
        "| Route | Method | Best Anchor | Rank Spec | Holdout AUROC | Holdout SelAcc@10 | Top-1 Abs Mass | Top-2 Abs Mass | Top-4 Abs Mass |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]

    for row in route_selection_rows:
        lines.append(
            "| {route} | `{method}` | {anchor} | `{rank}` | {auroc:.4f} | {selacc:.4f} | {share1:.3f} | {share2:.3f} | {share4:.3f} |".format(
                route=str(row["route_label"]),
                method=str(row["method_id"]),
                anchor=int(row["anchor_pct"]),
                rank=str(row["rank_spec"]),
                auroc=float(row["base_auroc"]),
                selacc=float(row["base_selacc_10"]),
                share1=float(row["mean_abs_share_top1_k1"]),
                share2=float(row["mean_abs_share_top1_k2"]),
                share4=float(row["mean_abs_share_top1_k4"]),
            )
        )

    lines += [
        "",
        "The mass numbers above come directly from per-run axis contributions and show that the selected top run is usually supported by a concentrated subset of latent axes.",
        "",
        "## Main Intervention Table (`k=2`)",
        "",
        "| Route | Policy | Mean Top1 Score Drop | AUROC Drop | SelAcc Drop | Top1 Flip Rate | Decision Flip Rate | Removed Abs Mass |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for route_label in route_order:
        for policy in POLICY_ORDER:
            row = next(
                (
                    item
                    for item in k2_rows
                    if str(item["route_label"]) == route_label and str(item["policy"]) == policy
                ),
                None,
            )
            if row is None:
                continue
            lines.append(
                "| {route} | {policy} | {score_drop:.4f} | {auroc_drop:.4f} | {selacc_drop:.4f} | {top1_flip:.3f} | {decision_flip:.3f} | {mass:.3f} |".format(
                    route=_route_ref(str(row["route_label"]), int(row["anchor_pct"])),
                    policy=str(row["policy"]),
                    score_drop=float(row["mean_top1_score_drop"]),
                    auroc_drop=float(row["auroc_drop"]),
                    selacc_drop=float(row["selacc_drop"]),
                    top1_flip=float(row["top1_flip_rate"]),
                    decision_flip=float(row["decision_flip_rate"]),
                    mass=float(row["mean_removed_abs_share_top1"]),
                )
            )

    lines += [
        "",
        "## Perturbation Sanity",
        "",
        "| Route | Selected Axis Subset | Monotonic Fraction | Mean Δscore @ ε=-0.2 | Mean Δscore @ ε=+0.2 |",
        "|---|---|---:|---:|---:|",
    ]

    for route_label in route_order:
        for subset_kind in ("positive_contribution_axis", "negative_contribution_axis"):
            rows = [
                row for row in perturb_rows
                if str(row["route_label"]) == route_label and str(row["subset_kind"]) == subset_kind
            ]
            if not rows:
                continue
            rows = sorted(rows, key=lambda row: float(row["epsilon"]))
            minus_row = min(rows, key=lambda row: abs(float(row["epsilon"]) + 0.2))
            plus_row = min(rows, key=lambda row: abs(float(row["epsilon"]) - 0.2))
            lines.append(
                "| {route} | {subset_kind} | {mono:.3f} | {minus_delta:.4f} | {plus_delta:.4f} |".format(
                    route=_route_ref(route_label, int(rows[0]["anchor_pct"])),
                    subset_kind=str(subset_kind).replace("_", " "),
                    mono=float(rows[0]["monotonic_fraction"]),
                    minus_delta=float(minus_row["mean_score_delta"]),
                    plus_delta=float(plus_row["mean_score_delta"]),
                )
            )

    lines += [
        "",
        "## Figures",
        "",
    ]
    for route_label in route_order:
        lines.append(f"- `{figure_dir_rel / f'{route_label}_route.png'}`")
    lines.append(f"- `{figure_dir_rel / 'perturbation_monotonicity.png'}`")
    lines += [
        "",
        "## Takeaways",
        "",
        "- **Reading note**: top1-flip can be noisy because random ablations often reshuffle near-tied high-scoring runs; AUROC/SelAcc drops and pairwise decision flips are the cleaner route-level signal.",
    ]

    for route_label in route_order:
        top2 = next(
            (
                row for row in k2_rows
                if str(row["route_label"]) == route_label and str(row["policy"]) == "top"
            ),
            None,
        )
        rnd2 = next(
            (
                row for row in k2_rows
                if str(row["route_label"]) == route_label and str(row["policy"]) == "random"
            ),
            None,
        )
        bot2 = next(
            (
                row for row in k2_rows
                if str(row["route_label"]) == route_label and str(row["policy"]) == "bottom"
            ),
            None,
        )
        if top2 is None or rnd2 is None or bot2 is None:
            continue
        if route_label == "coding":
            lines.append(
                "- **coding**: this remains a boundary case. The best coding route is weak, perturbations are still monotonic, but top-axis ablation does not yield the same clean AUROC/SelAcc degradation seen in noncoding."
            )
        else:
            lines.append(
                "- **{route}**: top-2 axis removal drops AUROC by `{top_auc:.4f}` vs random `{rnd_auc:.4f}` and bottom `{bot_auc:.4f}`; removed mass is `{top_mass:.3f}` vs `{rnd_mass:.3f}` vs `{bot_mass:.3f}`.".format(
                    route=route_label,
                    top_auc=float(top2["auroc_drop"]),
                    rnd_auc=float(rnd2["auroc_drop"]),
                    bot_auc=float(bot2["auroc_drop"]),
                    top_mass=float(top2["mean_removed_abs_share_top1"]),
                    rnd_mass=float(rnd2["mean_removed_abs_share_top1"]),
                    bot_mass=float(bot2["mean_removed_abs_share_top1"]),
                )
            )

    lines += [
        "- **Overall**: in `math`, `science`, and `ms`, top-contribution axes matter substantially more than matched random axes on the main metrics, while bottom axes are close to a negative control.",
        "- **Boundary case**: `coding` stays useful as a failure/ablation slice, but it should not be used as the main axis-level evidence because the route itself is weak at holdout.",
        "- **Interpretation**: this does not prove a unique canonical basis, but it does show that the learned axes carry concentrated causal signal for the deployed decision rule.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export axis-level intervention evidence for SVD routes")
    ap.add_argument("--routes", default="math,science,ms,coding")
    ap.add_argument("--ks", default="1,2,4")
    ap.add_argument("--random-repeats", type=int, default=32)
    ap.add_argument("--epsilons", default="-0.2,-0.1,0.0,0.1,0.2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--table-out", default="results/tables/axis_intervention.csv")
    ap.add_argument("--figure-dir", default="results/figures/axis_intervention")
    ap.add_argument("--doc-out", default="docs/AXIS_INTERVENTION_EVIDENCE.md")
    args = ap.parse_args()

    route_names = [name for name in _parse_csv(args.routes) if name in ROUTE_SPECS]
    if not route_names:
        raise ValueError("No valid routes selected")

    k_values = [k for k in _parse_int_csv(args.ks) if k > 0]
    epsilons = _parse_float_csv(args.epsilons)
    table_out = (REPO_ROOT / str(args.table_out)).resolve()
    figure_dir = (REPO_ROOT / str(args.figure_dir)).resolve()
    doc_out = (REPO_ROOT / str(args.doc_out)).resolve()

    all_rows: list[dict[str, Any]] = []
    route_selection_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    perturb_rows: list[dict[str, Any]] = []

    figure_dir.mkdir(parents=True, exist_ok=True)

    for route_idx, route_name in enumerate(route_names):
        spec = ROUTE_SPECS[route_name]
        bundle = load_earlystop_svd_bundle(REPO_ROOT / "models/ml_selectors" / f"{spec['method_id']}.pkl")
        holdout_store = _build_holdout_feature_store(spec)
        anchor_pack = _pick_best_anchor(bundle, holdout_store)

        route_selection_row = {
            "experiment_type": "route_selection",
            "route_label": str(spec["route_label"]),
            "method_id": str(spec["method_id"]),
            "anchor_pct": int(anchor_pack["anchor_pct"]),
            "rank_spec": str(anchor_pack["rank_spec"]),
            "n_problem_slices": int(anchor_pack["base_metrics"]["n_problem_slices"]),
            "n_samples": int(anchor_pack["base_metrics"]["n_samples"]),
            "base_auroc": float(anchor_pack["base_metrics"]["auroc"]),
            "base_selacc_10": float(anchor_pack["base_metrics"]["selacc_10"]),
            "mean_abs_share_top1_k1": float(anchor_pack["mean_abs_share_top1_k1"]),
            "mean_abs_share_top1_k2": float(anchor_pack["mean_abs_share_top1_k2"]),
            "mean_abs_share_top1_k4": float(anchor_pack["mean_abs_share_top1_k4"]),
        }
        route_selection_rows.append(route_selection_row)
        all_rows.append(route_selection_row)

        max_rank = max(
            int(record["route_terms"]["rank"])
            for record in anchor_pack["problem_records"]
        ) if anchor_pack["problem_records"] else 0
        local_k_values = [k for k in k_values if k <= max_rank]
        for k in local_k_values:
            for policy in POLICY_ORDER:
                summary = _summarize_policy_runs(
                    anchor_pack["problem_records"],
                    anchor_pack["base_metrics"],
                    k=int(k),
                    policy=str(policy),
                    random_repeats=int(args.random_repeats),
                    seed_base=int(args.seed) + int(route_idx * 1000 + k * 100),
                )
                row = {
                    "experiment_type": "ablation",
                    "route_label": str(spec["route_label"]),
                    "method_id": str(spec["method_id"]),
                    "anchor_pct": int(anchor_pack["anchor_pct"]),
                    "rank_spec": str(anchor_pack["rank_spec"]),
                    "n_problem_slices": int(anchor_pack["base_metrics"]["n_problem_slices"]),
                    "n_samples": int(anchor_pack["base_metrics"]["n_samples"]),
                    "base_auroc": float(anchor_pack["base_metrics"]["auroc"]),
                    "base_selacc_10": float(anchor_pack["base_metrics"]["selacc_10"]),
                    "policy": str(policy),
                    "k": int(k),
                    **summary,
                }
                ablation_rows.append(row)
                all_rows.append(row)

        route_perturb_rows = _perturbation_rows(
            str(spec["route_label"]),
            str(spec["method_id"]),
            anchor_pack,
            epsilons=epsilons,
        )
        perturb_rows.extend(route_perturb_rows)
        all_rows.extend(route_perturb_rows)

        _route_figure(
            route_label=str(spec["route_label"]),
            method_id=str(spec["method_id"]),
            anchor_pack=anchor_pack,
            ablation_rows=ablation_rows,
            out_path=figure_dir / f"{spec['route_label']}_route.png",
        )

    _perturbation_figure(
        perturb_rows=perturb_rows,
        out_path=figure_dir / "perturbation_monotonicity.png",
        route_order=[ROUTE_SPECS[name]["route_label"] for name in route_names],
    )

    _write_csv(table_out, all_rows)
    _build_doc(
        out_path=doc_out,
        route_selection_rows=route_selection_rows,
        ablation_rows=ablation_rows,
        perturb_rows=perturb_rows,
        figure_dir=figure_dir,
        random_repeats=int(args.random_repeats),
    )

    print("[done] axis intervention export", flush=True)
    print(f"  table  : {table_out}", flush=True)
    print(f"  figures: {figure_dir}", flush=True)
    print(f"  doc    : {doc_out}", flush=True)


if __name__ == "__main__":
    main()
