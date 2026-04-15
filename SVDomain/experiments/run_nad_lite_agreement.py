#!/usr/bin/env python3
"""NAD-Lite Neuron Agreement Features: do set-based neuron agreement signals add predictive value?

Tests whether within-problem pairwise Jaccard similarity between runs improves the supervised
SVDomain verifier beyond the canonical 22-feature token+trajectory bank.

Conditions tested across math / science / coding:
  C0  canonical                  — TOKEN[0-10] + TRAJ[11-15] + AVAIL[19-24] (22 feat)
  C1  canonical_plus_nc          — C0 + nc_mean[16] + nc_slope[17] (24 feat)
  C2  canonical_plus_nad_scalar  — C0 + 4 NAD scalar features (26 feat)
  C3  canonical_plus_nad_agree   — C0 + 5 NAD agreement features (27 feat)
  C4  canonical_plus_all_nad     — C0 + 4 scalar + 5 agreement (31 feat)

Modeling variants per condition: no_svd (ScalerLR) and svd_r12 (Scaler+TruncSVD+LR).
Representation: always raw+rank.

NAD features use ViewSpec(agg=MAX, cut=MASS(0.95), order=BY_KEY) for neuron extraction.
All agreement features operate at full-sequence position (1.0 = base CSR) only.

Outputs:
  results/cache/nad_lite_features.pkl
  results/tables/nad_lite_feature_ablation.csv
  results/tables/nad_lite_agreement_summary.csv
  results/tables/nad_lite_group_sensitivity.csv
  results/tables/nad_lite_coding_gain_summary.csv
  docs/17_NAD_LITE_NEURON_AGREEMENT.md
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path
from typing import Any

import warnings

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

# Suppress the "roaring backend not available" warning (expected in this environment)
warnings.filterwarnings("ignore", message=".*[Rr]oaring.*", category=RuntimeWarning)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.core.distance.engine import DistanceEngine, DistanceSpec
from nad.core.views.reader import Agg, CacheReader, CutSpec, CutType, Order, RunView, ViewSpec
from nad.ops.earlystop import discover_cache_entries
from nad.ops.earlystop_svd import (
    AVAILABILITY_FEATURES,
    LEGACY_FULL_FEATURE_NAMES,
    TRAJ_FEATURES,
    TOKEN_FEATURES,
    _auroc,
    _build_representation,
    _group_folds,
    _rank_transform_matrix,
)

# ─── constants ─────────────────────────────────────────────────────────────────

DEFAULT_STORE_PATH   = "results/cache/es_svd_ms_rr_r1/cache_all_547b9060debe139e.pkl"
DEFAULT_CACHE_ROOT   = "MUI_HUB/cache"
DEFAULT_NAD_CACHE    = "results/cache/nad_lite_features.pkl"
DEFAULT_OUT_DIR      = "results/tables"
DEFAULT_DOC_PATH     = "docs/17_NAD_LITE_NEURON_AGREEMENT.md"

POS_INDEX = 11       # position 1.0 in the 12-position schema
DOMAINS   = ["math", "science", "coding"]

# NAD scalar feature names and their offset in the expanded matrix
NAD_SCALAR_NAMES = [
    "activated_neuron_count",
    "neuron_density",
    "topk10_neuron_count",
    "neuron_weight_entropy",
]
NAD_AGREE_NAMES = [
    "mean_jaccard_sim",
    "max_jaccard_sim",
    "knn_agree_score",
    "medoid_distance",
    "minact_rank",
]

# Indices in the expanded [n, 39] feature matrix
NAD_SCALAR_OFFSET = len(LEGACY_FULL_FEATURE_NAMES)   # 30
NAD_AGREE_OFFSET  = NAD_SCALAR_OFFSET + len(NAD_SCALAR_NAMES)  # 34

N_LEGACY = len(LEGACY_FULL_FEATURE_NAMES)  # 30
N_NAD    = len(NAD_SCALAR_NAMES) + len(NAD_AGREE_NAMES)  # 9
N_TOTAL  = N_LEGACY + N_NAD  # 39


# ─── feature index helpers ─────────────────────────────────────────────────────

def _name_to_idx() -> dict[str, int]:
    return {n: i for i, n in enumerate(LEGACY_FULL_FEATURE_NAMES)}


def _get_conditions() -> dict[str, list[int]]:
    """Return {condition_name: feature_indices} for 5 ablation conditions (C0-C4)."""
    n2i = _name_to_idx()
    token_idx = [n2i[n] for n in TOKEN_FEATURES]         # 0-10
    traj_idx  = [n2i[n] for n in TRAJ_FEATURES]          # 11-15
    avail_idx = [n2i[n] for n in AVAILABILITY_FEATURES]  # 19-24
    nc_idx    = [n2i["nc_mean"], n2i["nc_slope"]]         # 16, 17

    canonical  = sorted(token_idx + traj_idx + avail_idx)   # 22 feat
    nad_scalar = list(range(NAD_SCALAR_OFFSET, NAD_SCALAR_OFFSET + len(NAD_SCALAR_NAMES)))  # 30-33
    nad_agree  = list(range(NAD_AGREE_OFFSET,  NAD_AGREE_OFFSET  + len(NAD_AGREE_NAMES)))   # 34-38

    return {
        "canonical":                  canonical,
        "canonical_plus_nc":          sorted(canonical + nc_idx),
        "canonical_plus_nad_scalar":  sorted(canonical + nad_scalar),
        "canonical_plus_nad_agree":   sorted(canonical + nad_agree),
        "canonical_plus_all_nad":     sorted(canonical + nad_scalar + nad_agree),
    }


# ─── NAD feature computation ──────────────────────────────────────────────────

def _nad_scalar_features(
    keys_list: list[np.ndarray],
    weights_list: list[np.ndarray],
) -> np.ndarray:
    """Compute per-run NAD scalar features for one problem group.

    Returns: [n, 4] float64 — (activated_neuron_count, neuron_density,
                                topk10_neuron_count, neuron_weight_entropy)
    """
    n = len(keys_list)
    out = np.zeros((n, 4), dtype=np.float64)

    counts = np.array([len(k) for k in keys_list], dtype=np.float64)
    max_count = float(np.max(counts)) if n > 0 else 1.0

    for i, (keys, weights) in enumerate(zip(keys_list, weights_list)):
        nc = float(len(keys))
        out[i, 0] = nc

        # neuron_density: normalized count within group
        out[i, 1] = nc / max(max_count, 1.0)

        # topk10_neuron_count: neurons with weight >= 90th percentile of this run
        if weights.size > 0:
            p90 = float(np.percentile(weights, 90))
            out[i, 2] = float(np.sum(weights >= p90))
        else:
            out[i, 2] = 0.0

        # neuron_weight_entropy: Shannon entropy of normalized weight distribution
        if weights.size > 0:
            w = weights.astype(np.float64, copy=False)
            w_sum = float(w.sum())
            if w_sum > 0:
                p = w / w_sum
                p = np.where(p > 0, p, 1e-12)
                out[i, 3] = float(-np.sum(p * np.log(p)))
            else:
                out[i, 3] = 0.0
        else:
            out[i, 3] = 0.0

    return out


def _nad_agreement_features(
    D: np.ndarray,
    counts: np.ndarray,
    knn_k: int,
) -> np.ndarray:
    """Compute per-run NAD agreement features from pairwise Jaccard distance matrix.

    Args:
        D:       [n, n] float32 Jaccard distance matrix (0=identical, 1=disjoint)
        counts:  [n] activated_neuron_count per run (for minact_rank)
        knn_k:   k for kNN agreement score

    Returns: [n, 5] float64 — (mean_jaccard_sim, max_jaccard_sim,
                                knn_agree_score, medoid_distance, minact_rank)
              NaN for all features if n < 2.
    """
    n = D.shape[0]
    out = np.full((n, 5), fill_value=np.nan, dtype=np.float64)

    if n < 2:
        return out

    S = 1.0 - D.astype(np.float64)  # similarity [n, n]
    np.fill_diagonal(S, -np.inf)     # exclude self for max/knn

    # mean_jaccard_sim: mean over j≠i
    # We zero-out diagonal in D for mean computation
    D_no_diag = D.astype(np.float64).copy()
    np.fill_diagonal(D_no_diag, np.nan)
    sim_no_diag = 1.0 - D_no_diag
    out[:, 0] = np.nanmean(sim_no_diag, axis=1)   # mean_jaccard_sim
    out[:, 1] = np.nanmax(sim_no_diag, axis=1)    # max_jaccard_sim

    # knn_agree_score: mean of top-k similarities (excluding self)
    k = min(knn_k, n - 1)
    for i in range(n):
        row = np.copy(sim_no_diag[i])
        row_valid = row[~np.isnan(row)]
        if row_valid.size == 0:
            out[i, 2] = np.nan
        else:
            topk_idx = np.argpartition(row_valid, -k)[-k:]
            out[i, 2] = float(np.mean(row_valid[topk_idx]))

    # medoid_distance: mean(D[i,:]) including self (self=0, so mean is slightly lower)
    # Per plan: lower = more central
    out[:, 3] = np.nanmean(D_no_diag, axis=1)  # mean distance to all peers

    # minact_rank: rank of activated_neuron_count in [0=smallest, 1=largest]
    rank_order = np.argsort(counts, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[rank_order] = np.arange(n, dtype=np.float64)
    out[:, 4] = ranks / float(n - 1)

    return out


def _extract_problem_nad_features(
    reader: CacheReader,
    run_ids: np.ndarray,
    view_spec: ViewSpec,
    knn_k: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """For one problem group (n runs), extract NAD features.

    Returns:
        scalar_features:    [n, 4] float64
        agreement_features: [n, 5] float64  (NaN if n < 2)
        agreement_valid:    bool  (True if n >= 2)
    """
    n = len(run_ids)
    views: list[RunView] = []
    keys_list: list[np.ndarray] = []
    weights_list: list[np.ndarray] = []

    for rid in run_ids:
        rv = reader.get_run_view(int(rid), view_spec)
        views.append(rv)
        keys_list.append(np.asarray(rv.keys, dtype=np.float64))
        weights_list.append(np.asarray(rv.weights, dtype=np.float64))

    scalar_feat = _nad_scalar_features(keys_list, weights_list)

    agreement_valid = (n >= 2)
    if agreement_valid:
        engine = DistanceEngine(DistanceSpec(name="ja", num_threads=4))
        D = engine.dense_matrix(views)
        counts = scalar_feat[:, 0]  # activated_neuron_count
        agree_feat = _nad_agreement_features(D, counts, knn_k)
    else:
        agree_feat = np.full((n, 5), fill_value=np.nan, dtype=np.float64)

    return scalar_feat, agree_feat, agreement_valid


# ─── cache discovery & NAD feature extraction ─────────────────────────────────

def _build_cache_root_map(cache_root: str | Path) -> dict[str, Path]:
    """Map base_cache_key -> actual cache root directory."""
    entries = discover_cache_entries(cache_root)
    return {e.cache_key: e.cache_root for e in entries}


def _extract_nad_lite_features(
    store_data: dict,
    cache_root: str | Path,
    mass_cut: float,
    knn_k: int,
) -> dict[str, dict]:
    """Extract NAD-lite scalar + agreement features from live caches.

    Returns: dict[base_cache_key] -> {
        "scalars":    [n, 4] float64
        "agreement":  [n, 5] float64
        "agreement_valid": [n] bool
    }
    """
    cache_root_map = _build_cache_root_map(cache_root)
    view_spec = ViewSpec(
        agg=Agg.MAX,
        cut=CutSpec(CutType.MASS, mass_cut),
        order=Order.BY_KEY,
    )
    result: dict[str, dict] = {}

    for item in store_data["feature_store"]:
        base_key   = item["base_cache_key"]  # e.g. "DS-R1/aime24"
        sample_ids = np.asarray(item["sample_ids"], dtype=np.int64)
        prob_offs  = item["problem_offsets"]
        prob_ids   = item["problem_ids"]
        n_samples  = len(sample_ids)
        n_problems = len(prob_ids)

        if base_key not in cache_root_map:
            print(f"  [WARN] cache_key={base_key!r} not found in {cache_root} — skipping")
            continue

        cache_path = cache_root_map[base_key]
        print(f"  Opening cache: {cache_path.name}  ({n_samples} samples, {n_problems} problems)")

        scalars   = np.zeros((n_samples, len(NAD_SCALAR_NAMES)), dtype=np.float64)
        agreement = np.full((n_samples, len(NAD_AGREE_NAMES)), fill_value=np.nan, dtype=np.float64)
        agree_valid = np.zeros(n_samples, dtype=bool)

        reader = CacheReader(str(cache_path))
        n_degenerate = 0

        for j in range(n_problems):
            start = int(prob_offs[j])
            end   = int(prob_offs[j + 1]) if j + 1 < len(prob_offs) else n_samples
            run_ids = sample_ids[start:end]

            sc, ag, av = _extract_problem_nad_features(reader, run_ids, view_spec, knn_k)
            scalars[start:end]     = sc
            agreement[start:end]   = ag
            agree_valid[start:end] = av
            if not av:
                n_degenerate += 1

        if n_degenerate > 0:
            print(f"    {n_degenerate}/{n_problems} degenerate problems (n_runs < 2)")

        result[base_key] = {
            "scalars":       scalars,
            "agreement":     agreement,
            "agreement_valid": agree_valid,
        }

    return result


# ─── data assembly ────────────────────────────────────────────────────────────

def _assemble_domain_data(
    store_data: dict,
    nad_features: dict[str, dict],
    domain: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build expanded X [n, 39], y [n], groups [n] for one domain.

    X[:,0:30] = legacy features at pos 1.0
    X[:,30:34] = NAD scalar features
    X[:,34:39] = NAD agreement features
    """
    items = [it for it in store_data["feature_store"] if it["domain"] == domain]
    X_parts, y_parts, g_parts = [], [], []

    for item in items:
        base_key = item["base_cache_key"]
        tensor   = np.asarray(item["tensor"], dtype=np.float64)  # [n, 12, 30]
        labels   = np.asarray(item["labels"], dtype=np.int32)
        gkeys    = np.asarray(item["group_keys"], dtype=object)
        n        = tensor.shape[0]

        X_legacy = tensor[:, POS_INDEX, :]  # [n, 30]

        if base_key in nad_features:
            nad = nad_features[base_key]
            X_scalar = nad["scalars"].astype(np.float64)    # [n, 4]
            X_agree  = nad["agreement"].astype(np.float64)  # [n, 5]
        else:
            print(f"  [WARN] NAD features missing for {base_key!r} — filling with zeros/nan")
            X_scalar = np.zeros((n, len(NAD_SCALAR_NAMES)), dtype=np.float64)
            X_agree  = np.full((n, len(NAD_AGREE_NAMES)), fill_value=np.nan, dtype=np.float64)

        X_full = np.concatenate([X_legacy, X_scalar, X_agree], axis=1)  # [n, 39]
        X_parts.append(X_full)
        y_parts.append(labels)
        g_parts.append(gkeys)

    if not X_parts:
        return (
            np.zeros((0, N_TOTAL), dtype=np.float64),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=object),
        )

    return (
        np.concatenate(X_parts, axis=0),
        np.concatenate(y_parts),
        np.concatenate(g_parts),
    )


# ─── metric helpers ───────────────────────────────────────────────────────────

def _stop_acc(scores: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    unique_g = np.unique(groups)
    hits: list[int] = []
    for g in unique_g:
        mask = groups == g
        if mask.sum() == 0:
            continue
        hits.append(int(y[mask][np.argmax(scores[mask])]))
    return float(np.mean(hits)) if hits else float("nan")


def _balanced_acc_from_scores(scores: np.ndarray, y: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    preds = (scores > 0).astype(int)
    return float(balanced_accuracy_score(y, preds))


def _fmt(v: float) -> str:
    return f"{v:.4f}" if np.isfinite(v) else "nan"


# ─── NaN imputation ───────────────────────────────────────────────────────────

def _impute_nans_global(X: np.ndarray) -> tuple[np.ndarray, int]:
    """Fill NaN values with column means. Returns (X_clean, n_nan_rows)."""
    has_nan = np.any(np.isnan(X), axis=1)
    n_nan_rows = int(has_nan.sum())
    if n_nan_rows == 0:
        return X, 0
    X = X.copy()
    for col in range(X.shape[1]):
        mask = np.isnan(X[:, col])
        if mask.any():
            col_mean = float(np.nanmean(X[:, col]))
            X[mask, col] = 0.0 if np.isnan(col_mean) else col_mean
    return X, n_nan_rows


# ─── core CV harness ──────────────────────────────────────────────────────────

def _cv_one_condition(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_indices: list[int],
    rank: int,
    n_splits: int,
    random_state: int,
) -> dict[str, Any]:
    """GroupKFold CV; rank=0 → no_svd (plain LR). Returns aggregate stats.

    NaN values in X are imputed with column means before entering the fold loop.
    """
    # Global NaN imputation (agreement features are NaN for singleton groups)
    X_clean, n_imputed = _impute_nans_global(X)

    x_rank = _rank_transform_matrix(X_clean)
    X_rep  = _build_representation(X_clean, x_rank, feature_indices, "raw+rank")
    folds  = _group_folds(groups, n_splits)

    auroc_vals, bac_vals, stop_vals = [], [], []
    n_degen = 0

    for train_idx, test_idx in folds:
        X_tr, X_te = X_rep[train_idx], X_rep[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        g_orig_te  = groups[test_idx]

        single = np.unique(y_te).shape[0] < 2 or np.unique(y_tr).shape[0] < 2
        if single:
            n_degen += 1
            continue

        try:
            sc     = StandardScaler()
            Xtr_sc = sc.fit_transform(X_tr)
            Xte_sc = sc.transform(X_te)

            if rank > 0:
                max_r = max(1, min(rank, Xtr_sc.shape[1], Xtr_sc.shape[0] - 1))
                svd_t = TruncatedSVD(n_components=max_r, random_state=random_state)
                Xtr_f = svd_t.fit_transform(Xtr_sc)
                Xte_f = svd_t.transform(Xte_sc)
            else:
                Xtr_f, Xte_f = Xtr_sc, Xte_sc

            clf = LogisticRegression(C=1.0, max_iter=2000, random_state=random_state)
            clf.fit(Xtr_f, y_tr)
            sc_te = clf.decision_function(Xte_f)

            au = _auroc(sc_te, y_te)
            ba = _balanced_acc_from_scores(sc_te, y_te)
            sa = _stop_acc(sc_te, y_te, g_orig_te)

            if np.isfinite(au): auroc_vals.append(au)
            if np.isfinite(ba): bac_vals.append(ba)
            if np.isfinite(sa): stop_vals.append(sa)
        except Exception as exc:
            print(f"    [cv rank={rank}] fold error: {exc}")
            n_degen += 1

    return {
        "auroc_mean":  float(np.mean(auroc_vals)) if auroc_vals else float("nan"),
        "auroc_std":   float(np.std(auroc_vals))  if auroc_vals else float("nan"),
        "bac_mean":    float(np.mean(bac_vals))   if bac_vals   else float("nan"),
        "stop_mean":   float(np.mean(stop_vals))  if stop_vals  else float("nan"),
        "n_valid":     len(auroc_vals),
        "n_degen":     n_degen,
        "n_imputed":   n_imputed,
    }


# ─── main ablation ─────────────────────────────────────────────────────────────

def _run_main_ablation(
    domains_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    conditions: dict[str, list[int]],
    svd_rank: int,
    n_splits: int,
    random_state: int,
) -> list[dict]:
    """Run full ablation: 3 domains × 5 conditions × 2 modeling variants."""
    rows: list[dict] = []
    for domain in DOMAINS:
        X, y, groups = domains_data[domain]
        n_samples = X.shape[0]
        n_groups  = int(np.unique(groups).shape[0])
        print(f"\n[ABLATION] domain={domain}  n={n_samples}  groups={n_groups}")

        for cond_name, feat_idx in conditions.items():
            n_feat = len(feat_idx)
            for modeling_label, rank in [("no_svd", 0), (f"svd_r{svd_rank}", svd_rank)]:
                print(f"  {cond_name}/{modeling_label} ({n_feat} feat) ...")
                res = _cv_one_condition(
                    X, y, groups, feat_idx, rank, n_splits, random_state
                )
                rows.append({
                    "domain":              domain,
                    "condition":           cond_name,
                    "n_feat":              n_feat,
                    "modeling":            modeling_label,
                    "auroc_mean":          _fmt(res["auroc_mean"]),
                    "auroc_std":           _fmt(res["auroc_std"]),
                    "bac_mean":            _fmt(res["bac_mean"]),
                    "stop_mean":           _fmt(res["stop_mean"]),
                    "n_folds_valid":       res["n_valid"],
                    "n_folds_degenerate":  res["n_degen"],
                    "n_samples":           n_samples,
                    "n_groups":            n_groups,
                })
    return rows


# ─── unsupervised baseline ─────────────────────────────────────────────────────

def _run_unsupervised_baseline(
    domains_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> list[dict]:
    """Unsupervised MinAct / Medoid / kNN selection → StopAcc@100% per domain."""
    rows: list[dict] = []

    for domain in DOMAINS:
        X, y, groups = domains_data[domain]
        if X.shape[0] == 0:
            continue

        # Signal arrays from NAD features
        minact_scores     = -X[:, NAD_SCALAR_OFFSET + 0]        # lower count → better → negate
        medoid_scores     = -X[:, NAD_AGREE_OFFSET + 3]         # lower distance → better → negate
        knn_agree_scores  =  X[:, NAD_AGREE_OFFSET + 2]         # higher → better

        for signal_name, raw_scores in [
            ("MinAct",    minact_scores),
            ("Medoid",    medoid_scores),
            ("kNN_agree", knn_agree_scores),
        ]:
            # Impute NaN with 0 (neutral)
            scores = np.where(np.isnan(raw_scores), 0.0, raw_scores)
            sa = _stop_acc(scores, y, groups)
            rows.append({
                "domain":          domain,
                "selector":        signal_name,
                "stop_acc_at_100": _fmt(sa),
                "note":            "unsupervised; no training",
            })
            print(f"  [UNSUPERVISED] {domain}/{signal_name}: stop_acc={_fmt(sa)}")

    return rows


# ─── group sensitivity ────────────────────────────────────────────────────────

def _run_group_sensitivity(
    domains_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    conditions: dict[str, list[int]],
    svd_rank: int,
    n_splits: int,
    random_state: int,
) -> list[dict]:
    """Stratify by problem group size and compute performance per quintile."""
    rows: list[dict] = []
    # Only test canonical and canonical_plus_nad_agree (most informative comparison)
    selected_conds = {k: v for k, v in conditions.items()
                      if k in ("canonical", "canonical_plus_nad_agree")}

    for domain in DOMAINS:
        X, y, groups = domains_data[domain]
        if X.shape[0] == 0:
            continue

        unique_g = np.unique(groups)
        group_sizes = {g: int(np.sum(groups == g)) for g in unique_g}
        sizes_arr = np.array([group_sizes[g] for g in unique_g])

        # Compute quintile bins based on group sizes
        bins = np.percentile(sizes_arr, [0, 20, 40, 60, 80, 100])
        bins = np.unique(bins)
        if len(bins) < 2:
            continue

        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            if i == len(bins) - 2:
                in_bin = (sizes_arr >= lo) & (sizes_arr <= hi)
            else:
                in_bin = (sizes_arr >= lo) & (sizes_arr < hi)
            bin_groups = unique_g[in_bin]
            if len(bin_groups) < 2:
                continue

            mask = np.isin(groups, bin_groups)
            X_bin, y_bin, g_bin = X[mask], y[mask], groups[mask]
            bin_label = f"[{int(lo)}-{int(hi)}]"
            n_bin_groups = len(bin_groups)

            # Fraction with valid agreement (n_runs >= 2)
            agree_col = X_bin[:, NAD_AGREE_OFFSET]   # mean_jaccard_sim
            agree_valid_pct = float(100.0 * np.mean(~np.isnan(agree_col)))

            for cond_name, feat_idx in selected_conds.items():
                res = _cv_one_condition(
                    X_bin, y_bin, g_bin, feat_idx, 0, n_splits, random_state
                )
                rows.append({
                    "domain":              domain,
                    "condition":           cond_name,
                    "group_size_bin":      bin_label,
                    "n_groups":            n_bin_groups,
                    "auroc_mean":          _fmt(res["auroc_mean"]),
                    "stop_mean":           _fmt(res["stop_mean"]),
                    "agreement_valid_pct": f"{agree_valid_pct:.1f}",
                })

    return rows


# ─── summary builders ─────────────────────────────────────────────────────────

def _build_agreement_summary(
    domains_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> list[dict]:
    """Per-domain statistics for NAD agreement features."""
    rows: list[dict] = []
    agree_feat_cols = list(range(NAD_AGREE_OFFSET, NAD_AGREE_OFFSET + len(NAD_AGREE_NAMES)))
    scalar_feat_cols = list(range(NAD_SCALAR_OFFSET, NAD_SCALAR_OFFSET + len(NAD_SCALAR_NAMES)))

    all_features = list(zip(NAD_SCALAR_NAMES, scalar_feat_cols)) + \
                   list(zip(NAD_AGREE_NAMES, agree_feat_cols))

    for domain in DOMAINS:
        X, y, groups = domains_data[domain]
        if X.shape[0] == 0:
            continue
        for feat_name, col in all_features:
            vals = X[:, col]
            valid_mask = ~np.isnan(vals)
            pct_valid = float(100.0 * valid_mask.sum() / max(1, len(vals)))
            pct_nan   = 100.0 - pct_valid
            valid_vals = vals[valid_mask]
            rows.append({
                "domain":    domain,
                "feature":   feat_name,
                "mean":      _fmt(float(np.mean(valid_vals))) if valid_vals.size > 0 else "nan",
                "std":       _fmt(float(np.std(valid_vals)))  if valid_vals.size > 0 else "nan",
                "pct_valid": f"{pct_valid:.1f}",
                "pct_nan":   f"{pct_nan:.1f}",
                "note":      "",
            })
    return rows


def _build_coding_gain_summary(
    ablation_rows: list[dict],
    svd_rank: int,
) -> list[dict]:
    """Coding vs other domains gain analysis (C0 vs C3, all conditions)."""
    def _get(domain: str, cond: str, modeling: str) -> float:
        for r in ablation_rows:
            if r["domain"] == domain and r["condition"] == cond and r["modeling"] == modeling:
                try:
                    return float(r["auroc_mean"])
                except (ValueError, TypeError):
                    return float("nan")
        return float("nan")

    conditions_of_interest = [
        "canonical",
        "canonical_plus_nc",
        "canonical_plus_nad_scalar",
        "canonical_plus_nad_agree",
        "canonical_plus_all_nad",
    ]
    svd_label = f"svd_r{svd_rank}"

    rows: list[dict] = []
    for cond in conditions_of_interest:
        math_au    = _get("math",    cond, "no_svd")
        sci_au     = _get("science", cond, "no_svd")
        coding_au  = _get("coding",  cond, "no_svd")
        c0_coding  = _get("coding",  "canonical", "no_svd")
        delta      = coding_au - c0_coding if (np.isfinite(coding_au) and np.isfinite(c0_coding)) else float("nan")

        if np.isfinite(delta):
            if abs(delta) < 0.005:
                interp = "within noise"
            elif delta > 0.01:
                interp = "meaningful gain"
            elif delta > 0.005:
                interp = "marginal gain"
            elif delta < -0.005:
                interp = "degradation"
            else:
                interp = "negligible"
        else:
            interp = "nan"

        rows.append({
            "condition":              cond,
            "coding_auroc":           _fmt(coding_au),
            "math_auroc":             _fmt(math_au),
            "science_auroc":          _fmt(sci_au),
            "coding_delta_vs_c0":     _fmt(delta) if np.isfinite(delta) else "nan",
            "interpretation":         interp,
        })
    return rows


# ─── CSV output ───────────────────────────────────────────────────────────────

def _write_csvs(
    ablation_rows: list[dict],
    agree_summary_rows: list[dict],
    group_sens_rows: list[dict],
    coding_gain_rows: list[dict],
    unsup_rows: list[dict],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def _write(fname: str, fieldnames: list[str], rows: list[dict]) -> None:
        p = out_dir / fname
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"  wrote {p}  ({len(rows)} rows)")

    _write(
        "nad_lite_feature_ablation.csv",
        ["domain", "condition", "n_feat", "modeling",
         "auroc_mean", "auroc_std", "bac_mean", "stop_mean",
         "n_folds_valid", "n_folds_degenerate", "n_samples", "n_groups"],
        ablation_rows,
    )
    _write(
        "nad_lite_agreement_summary.csv",
        ["domain", "feature", "mean", "std", "pct_valid", "pct_nan", "note"],
        agree_summary_rows,
    )
    _write(
        "nad_lite_group_sensitivity.csv",
        ["domain", "condition", "group_size_bin", "n_groups",
         "auroc_mean", "stop_mean", "agreement_valid_pct"],
        group_sens_rows,
    )
    _write(
        "nad_lite_coding_gain_summary.csv",
        ["condition", "coding_auroc", "math_auroc", "science_auroc",
         "coding_delta_vs_c0", "interpretation"],
        coding_gain_rows,
    )
    if unsup_rows:
        _write(
            "nad_lite_unsupervised_baseline.csv",
            ["domain", "selector", "stop_acc_at_100", "note"],
            unsup_rows,
        )


# ─── Markdown doc ─────────────────────────────────────────────────────────────

def _write_doc(
    ablation_rows: list[dict],
    agree_summary_rows: list[dict],
    unsup_rows: list[dict],
    coding_gain_rows: list[dict],
    svd_rank: int,
    doc_path: Path,
) -> None:
    def _get(domain: str, cond: str, modeling: str) -> float:
        for r in ablation_rows:
            if r["domain"] == domain and r["condition"] == cond and r["modeling"] == modeling:
                try:
                    return float(r["auroc_mean"])
                except (ValueError, TypeError):
                    return float("nan")
        return float("nan")

    svd_label = f"svd_r{svd_rank}"

    # ── forced verdicts ───────────────────────────────────────────────────────
    # Q1: Are C3 (nad_agree) better than C2 (nad_scalar)?
    q1_deltas: dict[str, float] = {}
    for domain in DOMAINS:
        c2 = _get(domain, "canonical_plus_nad_scalar", "no_svd")
        c3 = _get(domain, "canonical_plus_nad_agree",  "no_svd")
        q1_deltas[domain] = c3 - c2 if (np.isfinite(c2) and np.isfinite(c3)) else float("nan")

    # Q2: Does C4 beat C1 on any domain by > 0.005?
    q2_beats: dict[str, float] = {}
    for domain in DOMAINS:
        c1 = _get(domain, "canonical_plus_nc",     "no_svd")
        c4 = _get(domain, "canonical_plus_all_nad","no_svd")
        q2_beats[domain] = c4 - c1 if (np.isfinite(c1) and np.isfinite(c4)) else float("nan")
    q2_any_domain = any(d > 0.005 for d in q2_beats.values() if np.isfinite(d))

    # Q3: Which domain benefits most from agreement features (C0 → C3)?
    q3_deltas: dict[str, float] = {}
    for domain in DOMAINS:
        c0 = _get(domain, "canonical",              "no_svd")
        c3 = _get(domain, "canonical_plus_nad_agree","no_svd")
        q3_deltas[domain] = c3 - c0 if (np.isfinite(c0) and np.isfinite(c3)) else float("nan")
    best_q3 = max(q3_deltas, key=lambda d: q3_deltas[d] if np.isfinite(q3_deltas[d]) else -1)

    # Q4: Do agreement features reduce coding instability vs C0?
    coding_c0 = _get("coding", "canonical",              "no_svd")
    coding_c3 = _get("coding", "canonical_plus_nad_agree","no_svd")
    coding_delta = coding_c3 - coding_c0 if (np.isfinite(coding_c0) and np.isfinite(coding_c3)) else float("nan")
    q4_positive = np.isfinite(coding_delta) and coding_delta > 0

    # Q5: Is unsupervised Medoid competitive with supervised C0 on math/science?
    medoid_math = medoid_sci = float("nan")
    c0_math     = _get("math",    "canonical", "no_svd")
    c0_sci      = _get("science", "canonical", "no_svd")
    for r in unsup_rows:
        if r["selector"] == "Medoid":
            if r["domain"] == "math":
                try: medoid_math = float(r["stop_acc_at_100"])
                except: pass
            if r["domain"] == "science":
                try: medoid_sci = float(r["stop_acc_at_100"])
                except: pass

    # Q6: Paper framing
    any_meaningful = any(d > 0.005 for d in q3_deltas.values() if np.isfinite(d))
    if any_meaningful:
        q6_framing = (f"inline note: agreement features provide marginal/meaningful gains "
                      f"in {best_q3}; report as supplementary features.")
    else:
        q6_framing = ("future work: NAD agreement features do not improve over canonical "
                      "at this threshold; revisit with rows/bank prefix-level Jaccard when available.")

    # ── build document ────────────────────────────────────────────────────────
    lines: list[str] = [
        "# 17: NAD-Lite Neuron Agreement Features",
        "",
        "**Date**: 2026-04-12  ",
        "**Status**: Analysis complete  ",
        "**Data**: cache_all_547b9060debe139e.pkl · 30 legacy features · position 1.0  ",
        "**ViewSpec**: `agg=MAX, cut=MASS(0.95), order=BY_KEY` — full-sequence neuron sets from base CSR  ",
        "**Note**: Prefix-level agreement features infeasible (rows/bank unavailable). "
        "All agreement operates at full-sequence position only.",
        "",
        "---",
        "",
        "## 1. Conditions",
        "",
        "| ID | Name | Features | N feat |",
        "|----|------|----------|--------|",
        "| C0 | `canonical` | TOKEN[0-10] + TRAJ[11-15] + AVAIL[19-24] | 22 |",
        "| C1 | `canonical_plus_nc` | C0 + nc_mean[16] + nc_slope[17] | 24 |",
        "| C2 | `canonical_plus_nad_scalar` | C0 + 4 NAD scalars [30-33] | 26 |",
        "| C3 | `canonical_plus_nad_agree` | C0 + 5 agreement features [34-38] | 27 |",
        "| C4 | `canonical_plus_all_nad` | C0 + 4 scalar + 5 agreement | 31 |",
        "",
        "**NAD Scalar features** (indices 30-33):",
        "- `activated_neuron_count`: unique neurons in full run (base CSR)",
        "- `neuron_density`: activated_neuron_count / max within problem group",
        "- `topk10_neuron_count`: neurons with weight ≥ 90th percentile of this run",
        "- `neuron_weight_entropy`: Shannon entropy of normalized weight distribution",
        "",
        "**NAD Agreement features** (indices 34-38, full-sequence Jaccard, n_peers ≥ 2):",
        "- `mean_jaccard_sim`: mean(1 - D[i,j≠i]) — mean similarity to all peers",
        "- `max_jaccard_sim`: max(1 - D[i,j≠i]) — most similar peer",
        "- `knn_agree_score`: mean of top-k similarities (k=5)",
        "- `medoid_distance`: mean(D[i,j≠i]) — lower = more central",
        "- `minact_rank`: rank of activated_neuron_count within group [0=smallest]",
        "",
        "Degenerate groups (n_runs < 2): agreement features = NaN → imputed with column mean.",
        "",
        "Modeling: `no_svd` (StandardScaler+LR) and `svd_r12` (Scaler+TruncatedSVD+LR).  ",
        "Representation: always `raw+rank`. Evaluation: 5-fold GroupKFold.",
        "",
        "---",
        "",
        "## 2. Main Ablation Results",
        "",
        "### AUROC by Domain and Condition (no_svd)",
        "",
        "| Domain | C0 | C1 | C2 | C3 | C4 |",
        "|--------|----|----|----|----|----|",
    ]
    cond_keys = ["canonical", "canonical_plus_nc", "canonical_plus_nad_scalar",
                 "canonical_plus_nad_agree", "canonical_plus_all_nad"]
    for domain in DOMAINS:
        vals = [_fmt(_get(domain, c, "no_svd")) for c in cond_keys]
        lines.append(f"| {domain} | " + " | ".join(vals) + " |")

    lines += [
        "",
        f"### AUROC by Domain and Condition ({svd_label})",
        "",
        "| Domain | C0 | C1 | C2 | C3 | C4 |",
        "|--------|----|----|----|----|----|",
    ]
    for domain in DOMAINS:
        vals = [_fmt(_get(domain, c, svd_label)) for c in cond_keys]
        lines.append(f"| {domain} | " + " | ".join(vals) + " |")

    lines += [
        "",
        "---",
        "",
        "## 3. NAD Feature Statistics",
        "",
        "| Domain | Feature | Mean | Std | Pct valid |",
        "|--------|---------|------|-----|-----------|",
    ]
    for r in agree_summary_rows:
        lines.append(
            f"| {r['domain']} | {r['feature']} | {r['mean']} | {r['std']} | {r['pct_valid']}% |"
        )

    if unsup_rows:
        lines += [
            "",
            "---",
            "",
            "## 4. Unsupervised Baselines (StopAcc@100%)",
            "",
            "| Domain | Selector | StopAcc@100% |",
            "|--------|----------|-------------|",
        ]
        for r in unsup_rows:
            lines.append(f"| {r['domain']} | {r['selector']} | {r['stop_acc_at_100']} |")

    lines += [
        "",
        "---",
        "",
        "## 5. Coding Domain Analysis",
        "",
        "| Condition | Coding AUROC | Math AUROC | Science AUROC | Coding Δ vs C0 | Interpretation |",
        "|-----------|-------------|-----------|--------------|----------------|----------------|",
    ]
    for r in coding_gain_rows:
        lines.append(
            f"| {r['condition']} | {r['coding_auroc']} | {r['math_auroc']} | "
            f"{r['science_auroc']} | {r['coding_delta_vs_c0']} | {r['interpretation']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 6. Forced Verdict",
        "",
        "**Q1 — Are NAD-lite agreement features (C3) better than NAD-lite scalars (C2)?**  ",
        "AUROC delta C3 − C2 (no_svd):",
    ]
    for domain in DOMAINS:
        d = q1_deltas.get(domain, float("nan"))
        d_str = f"{d:+.4f}" if np.isfinite(d) else "nan"
        verdict = "YES" if (np.isfinite(d) and d > 0.005) else ("MARGINAL" if (np.isfinite(d) and d > 0) else "NO")
        lines.append(f"- {domain}: Δ = {d_str} → {verdict}")

    lines += [
        "",
        f"**Q2 — Does C4 (all NAD) beat C1 (nc only) on any domain by > 0.005 AUROC?**  ",
        f"→ {'YES' if q2_any_domain else 'NO'}",
    ]
    for domain in DOMAINS:
        d = q2_beats.get(domain, float("nan"))
        d_str = f"{d:+.4f}" if np.isfinite(d) else "nan"
        lines.append(f"- {domain}: Δ = {d_str}")

    lines += [
        "",
        "**Q3 — Which domain benefits most from agreement features (C0 → C3)?**  ",
    ]
    for domain in DOMAINS:
        d = q3_deltas.get(domain, float("nan"))
        d_str = f"{d:+.4f}" if np.isfinite(d) else "nan"
        mark = " ← BEST" if domain == best_q3 else ""
        lines.append(f"- {domain}: Δ = {d_str}{mark}")

    lines += [
        "",
        f"**Q4 — Do agreement features (C3) reduce coding instability vs C0?**  ",
        f"C0 coding AUROC: {_fmt(coding_c0)}  ",
        f"C3 coding AUROC: {_fmt(coding_c3)}  ",
        f"Delta: {_fmt(coding_delta) if np.isfinite(coding_delta) else 'nan'}  ",
        f"→ {'POSITIVE' if q4_positive else 'NO IMPROVEMENT'}",
        "",
        "**Q5 — Is unsupervised Medoid competitive with C0 supervised on math/science?**  ",
        f"Medoid StopAcc (math): {_fmt(medoid_math)}  vs  C0 AUROC (math): {_fmt(c0_math)}  ",
        f"Medoid StopAcc (science): {_fmt(medoid_sci)}  vs  C0 AUROC (science): {_fmt(c0_sci)}  ",
        "→ (Note: StopAcc and AUROC are different metrics; direct comparison is approximate)",
        "",
        f"**Q6 — Paper framing**:  ",
        f"→ {q6_framing}",
        "",
        "---",
        "",
        "## 7. Limitations",
        "",
        "- Agreement features operate at **full-sequence** position only.",
        "  Prefix-level agreement (e.g. Jaccard at 50%) requires rows/bank keys, which",
        "  are unavailable (`has_rows_bank=0` in all current caches).",
        "- Singleton groups (n_runs < 2) have NaN agreement features, imputed with column mean.",
        "- Full-sequence Jaccard may conflate domain-level neuron patterns with run-level signals.",
        "- The roaring bitmap backend is not installed; NumPy backend is used (slower for large sets).",
        "",
        "---",
        "",
        "*Generated by SVDomain/run_nad_lite_agreement.py*",
    ]

    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {doc_path}")


# ─── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="NAD-Lite Neuron Agreement Features Ablation")
    ap.add_argument("--store-path",       default=DEFAULT_STORE_PATH,
                    help="Path to feature store pkl")
    ap.add_argument("--cache-root",       default=DEFAULT_CACHE_ROOT,
                    help="Cache root (MUI_HUB/cache or absolute path)")
    ap.add_argument("--nad-cache-path",   default=DEFAULT_NAD_CACHE,
                    help="Intermediate NAD feature cache pkl")
    ap.add_argument("--out-dir",          default=DEFAULT_OUT_DIR)
    ap.add_argument("--doc-path",         default=DEFAULT_DOC_PATH)
    ap.add_argument("--svd-rank",         type=int,   default=12)
    ap.add_argument("--n-splits",         type=int,   default=5)
    ap.add_argument("--random-state",     type=int,   default=42)
    ap.add_argument("--mass-cut",         type=float, default=0.95)
    ap.add_argument("--knn-k",            type=int,   default=5)
    ap.add_argument("--skip-unsupervised",action="store_true",
                    help="Skip unsupervised selector baseline")
    ap.add_argument("--skip-group-sensitivity", action="store_true",
                    help="Skip group-size sensitivity analysis (faster smoke test)")
    ap.add_argument("--force-extract",    action="store_true",
                    help="Recompute NAD features even if cache exists")
    args = ap.parse_args()

    store_path    = Path(REPO_ROOT / args.store_path)
    cache_root    = Path(REPO_ROOT / args.cache_root)
    nad_cache_path= Path(REPO_ROOT / args.nad_cache_path)
    out_dir       = Path(REPO_ROOT / args.out_dir)
    doc_path      = Path(REPO_ROOT / args.doc_path)
    nad_cache_key = (args.mass_cut, args.knn_k)

    # ── load feature store ─────────────────────────────────────────────────────
    print(f"Loading feature store: {store_path}")
    with store_path.open("rb") as f:
        store_data = pickle.load(f)
    for it in store_data["feature_store"]:
        import numpy as _np
        t = _np.asarray(it["tensor"])
        lbl = _np.asarray(it["labels"])
        print(f"  domain={it['domain']}  cache={it['base_cache_key']}  "
              f"n={t.shape[0]}  pos={int((_np.asarray(lbl)==1).sum())}")

    # ── extract or load NAD features ───────────────────────────────────────────
    nad_features: dict[str, dict] = {}

    if nad_cache_path.exists() and not args.force_extract:
        print(f"\nLoading NAD features from cache: {nad_cache_path}")
        with nad_cache_path.open("rb") as f:
            cache_store = pickle.load(f)
        if nad_cache_key in cache_store:
            nad_features = cache_store[nad_cache_key]
            print(f"  Loaded {len(nad_features)} cache entries for key={nad_cache_key}")
        else:
            print(f"  Key {nad_cache_key} not in cache; re-extracting ...")
            args.force_extract = True

    if not nad_features:
        print(f"\nExtracting NAD-lite features from live caches at: {cache_root}")
        nad_features = _extract_nad_lite_features(
            store_data, cache_root, args.mass_cut, args.knn_k
        )
        # Save to intermediate cache
        nad_cache_path.parent.mkdir(parents=True, exist_ok=True)
        existing: dict = {}
        if nad_cache_path.exists():
            with nad_cache_path.open("rb") as f:
                existing = pickle.load(f)
        existing[nad_cache_key] = nad_features
        with nad_cache_path.open("wb") as f:
            pickle.dump(existing, f, protocol=4)
        print(f"  Saved NAD features to {nad_cache_path}")

    # ── assemble domain data ───────────────────────────────────────────────────
    print("\nAssembling domain data (legacy + NAD) ...")
    domains_data: dict[str, tuple] = {}
    for domain in DOMAINS:
        X, y, groups = _assemble_domain_data(store_data, nad_features, domain)
        domains_data[domain] = (X, y, groups)
        n_nan = int(np.any(np.isnan(X), axis=1).sum())
        n_g = int(np.unique(groups).shape[0]) if X.shape[0] > 0 else 0
        print(f"  {domain}: n={X.shape[0]}  groups={n_g}  "
              f"pos={int((y==1).sum())}  nan_rows={n_nan}  feat_cols={X.shape[1]}")

    conditions = _get_conditions()
    print("\nConditions:")
    for name, idx in conditions.items():
        print(f"  {name}: {len(idx)} features")

    # ── main ablation ──────────────────────────────────────────────────────────
    ablation_rows = _run_main_ablation(
        domains_data, conditions,
        svd_rank=args.svd_rank,
        n_splits=args.n_splits,
        random_state=args.random_state,
    )

    # ── unsupervised baseline ──────────────────────────────────────────────────
    unsup_rows: list[dict] = []
    if not args.skip_unsupervised:
        print("\n[UNSUPERVISED] Running unsupervised baselines ...")
        unsup_rows = _run_unsupervised_baseline(domains_data)

    # ── group sensitivity ──────────────────────────────────────────────────────
    group_sens_rows: list[dict] = []
    if not args.skip_group_sensitivity:
        print("\n[SENSITIVITY] Running group-size sensitivity analysis ...")
        group_sens_rows = _run_group_sensitivity(
            domains_data, conditions,
            svd_rank=args.svd_rank,
            n_splits=args.n_splits,
            random_state=args.random_state,
        )

    # ── summary tables ─────────────────────────────────────────────────────────
    print("\n[SUMMARY] Building summary tables ...")
    agree_summary_rows = _build_agreement_summary(domains_data)
    coding_gain_rows   = _build_coding_gain_summary(ablation_rows, args.svd_rank)

    # ── write outputs ──────────────────────────────────────────────────────────
    print("\n[OUTPUT] Writing CSVs ...")
    _write_csvs(ablation_rows, agree_summary_rows, group_sens_rows,
                coding_gain_rows, unsup_rows, out_dir)

    print("\n[OUTPUT] Writing markdown doc ...")
    _write_doc(ablation_rows, agree_summary_rows, unsup_rows,
               coding_gain_rows, args.svd_rank, doc_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
