#!/usr/bin/env python3
"""Neuron-Augmented Ablation: do nc_mean/nc_slope add value beyond the 22-feature canonical bank?

Conditions tested across math / science / coding:
  A0  canonical              — TOKEN + TRAJ + AVAIL (22 feat)
  A1  canonical_plus_nc      — canonical + nc_mean + nc_slope (24 feat)
  A2  nc_only                — nc_mean + nc_slope + AVAIL flags (8 feat)
  A3  canonical_plus_prefix  — canonical + PREFIX_LOCAL (27 feat)
  A4  all_available          — A1 + A3 combined (29 feat; self_similarity excluded)

Modeling variants per condition: no_svd (ScalerLR) and svd_r12 (Scaler+TruncSVD+LR).
Representation: always raw+rank.

Outputs:
  results/tables/neuron_augmented_ablation.csv
  results/tables/neuron_added_value_by_domain.csv
  results/tables/neuron_vs_legacy_summary.csv
  results/tables/lowrank_interaction_summary.csv
  results/tables/neuron_feature_inventory.csv
  docs/15_NEURON_AUGMENTED_ABLATION.md
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.ops.earlystop_svd import (
    AVAILABILITY_FEATURES,
    LEGACY_FULL_FEATURE_NAMES,
    META_FEATURES,
    PREFIX_LOCAL_FEATURES,
    TRAJ_FEATURES,
    TOKEN_FEATURES,
    _auroc,
    _build_representation,
    _group_folds,
    _rank_transform_matrix,
)

DEFAULT_STORE_PATH = "results/cache/es_svd_ms_rr_r1/cache_all_547b9060debe139e.pkl"
DEFAULT_OUT_DIR = "results/tables"
DEFAULT_DOC_PATH = "docs/15_NEURON_AUGMENTED_ABLATION.md"
POS_INDEX = 11  # position 1.0 in the 12-position schema
DOMAINS = ["math", "science", "coding"]


# ─── feature index helpers ─────────────────────────────────────────────────────

def _name_to_idx() -> dict[str, int]:
    return {n: i for i, n in enumerate(LEGACY_FULL_FEATURE_NAMES)}


def _get_conditions() -> dict[str, list[int]]:
    """Return {condition_name: feature_indices} for all 5 ablation conditions."""
    n2i = _name_to_idx()
    token_idx  = [n2i[n] for n in TOKEN_FEATURES]       # 0-10
    traj_idx   = [n2i[n] for n in TRAJ_FEATURES]        # 11-15
    avail_idx  = [n2i[n] for n in AVAILABILITY_FEATURES] # 19-24
    prefix_idx = [n2i[n] for n in PREFIX_LOCAL_FEATURES] # 25-29
    nc_idx     = [n2i["nc_mean"], n2i["nc_slope"]]       # 16, 17

    canonical           = sorted(token_idx + traj_idx + avail_idx)           # 22 feat
    canonical_plus_nc   = sorted(canonical + nc_idx)                         # 24 feat
    nc_only             = sorted(nc_idx + avail_idx)                         # 8 feat
    canonical_plus_pref = sorted(canonical + prefix_idx)                     # 27 feat
    all_available       = sorted(canonical_plus_nc + prefix_idx)             # 29 feat

    return {
        "canonical":              canonical,
        "canonical_plus_nc":      canonical_plus_nc,
        "nc_only":                nc_only,
        "canonical_plus_prefix":  canonical_plus_pref,
        "all_available":          all_available,
    }


# ─── data loading ─────────────────────────────────────────────────────────────

def _load_domain_data(
    store_path: str | Path,
    domain: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load rows at position 1.0 for a given domain from cache_all pkl.

    Returns (X_raw [n,30], y [n,], groups [n,]).
    """
    with Path(store_path).open("rb") as f:
        data = pickle.load(f)

    items = [it for it in data["feature_store"] if it["domain"] == domain]
    X_parts, y_parts, g_parts = [], [], []
    for item in items:
        tensor = np.asarray(item["tensor"], dtype=np.float64)  # (n, 12, 30)
        labels = np.asarray(item["labels"], dtype=np.int32)
        gkeys  = np.asarray(item["group_keys"], dtype=object)
        X_parts.append(tensor[:, POS_INDEX, :])
        y_parts.append(labels)
        g_parts.append(gkeys)

    if not X_parts:
        return np.zeros((0, 30)), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=object)

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
    """GroupKFold CV; rank=0 → no_svd (plain LR). Returns aggregate stats."""
    x_rank = _rank_transform_matrix(X)
    X_rep  = _build_representation(X, x_rank, feature_indices, "raw+rank")
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
            sc = StandardScaler()
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
        "auroc_mean": float(np.mean(auroc_vals)) if auroc_vals else float("nan"),
        "auroc_std":  float(np.std(auroc_vals))  if auroc_vals else float("nan"),
        "bac_mean":   float(np.mean(bac_vals))   if bac_vals   else float("nan"),
        "stop_mean":  float(np.mean(stop_vals))  if stop_vals  else float("nan"),
        "n_valid":    len(auroc_vals),
        "n_degen":    n_degen,
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
        print(f"\n[ABLATION] domain={domain}  n={n_samples}  g={n_groups}")

        for cond_name, feat_idx in conditions.items():
            n_feat = len(feat_idx)
            for modeling_label, rank in [("no_svd", 0), (f"svd_r{svd_rank}", svd_rank)]:
                print(f"  {cond_name}/{modeling_label} ({n_feat} feat) ...")
                res = _cv_one_condition(
                    X, y, groups, feat_idx, rank, n_splits, random_state
                )
                for metric, mean_val, std_val in [
                    ("auroc",        res["auroc_mean"], res["auroc_std"]),
                    ("balanced_acc", res["bac_mean"],   0.0),
                    ("stop_acc",     res["stop_mean"],  0.0),
                ]:
                    rows.append({
                        "domain":              domain,
                        "condition":           cond_name,
                        "n_feat":              n_feat,
                        "modeling":            modeling_label,
                        "metric":              metric,
                        "mean":                _fmt(mean_val),
                        "std":                 _fmt(std_val),
                        "n_folds_valid":       res["n_valid"],
                        "n_folds_degenerate":  res["n_degen"],
                        "n_samples":           n_samples,
                        "n_groups":            n_groups,
                    })
    return rows


# ─── coding instability ────────────────────────────────────────────────────────

def _run_coding_instability(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    conditions_subset: dict[str, list[int]],
    n_seeds: int,
    n_bootstrap: int,
    n_splits: int,
    random_state: int,
) -> list[dict]:
    """Repeated GKF + bootstrap OOB for A0 and A1 on coding domain."""
    rows: list[dict] = []
    unique_g = np.unique(groups)
    n_groups  = len(unique_g)
    rng = np.random.default_rng(random_state)

    for cond_name, feat_idx in conditions_subset.items():
        x_rank = _rank_transform_matrix(X)
        X_rep  = _build_representation(X, x_rank, feat_idx, "raw+rank")

        # Protocol 1: Repeated GroupKFold with permuted group labels
        print(f"  [instability/{cond_name}] Protocol 1: Repeated GroupKFold ...")
        for seed in range(n_seeds):
            perm     = rng.permutation(n_groups)
            perm_map = {g: unique_g[perm[i]] for i, g in enumerate(unique_g)}
            groups_p = np.array([perm_map[g] for g in groups], dtype=object)
            folds    = _group_folds(groups_p, n_splits)

            for fold_i, (train_idx, test_idx) in enumerate(folds):
                Xtr, Xte = X_rep[train_idx], X_rep[test_idx]
                ytr, yte = y[train_idx], y[test_idx]
                single   = np.unique(yte).shape[0] < 2 or np.unique(ytr).shape[0] < 2

                au = float("nan")
                if not single:
                    try:
                        sc   = StandardScaler()
                        Xsc  = sc.fit_transform(Xtr)
                        Xtsc = sc.transform(Xte)
                        clf  = LogisticRegression(C=1.0, max_iter=2000, random_state=random_state)
                        clf.fit(Xsc, ytr)
                        au   = _auroc(clf.decision_function(Xtsc), yte)
                    except Exception as exc:
                        print(f"    [P1 seed={seed} fold={fold_i}] {exc}")

                rows.append({
                    "condition":    cond_name,
                    "protocol":     "repeated_gkf",
                    "seed_or_iter": seed,
                    "fold":         fold_i,
                    "is_degenerate": single,
                    "auroc":        _fmt(au),
                })

        # Protocol 2: Group bootstrap (out-of-bag)
        print(f"  [instability/{cond_name}] Protocol 2: Bootstrap OOB ...")
        for boot_i in range(n_bootstrap):
            boot_g  = rng.choice(unique_g, size=n_groups, replace=True)
            in_bag  = set(boot_g.tolist())
            oob     = set(unique_g.tolist()) - in_bag
            if not oob:
                continue

            train_mask = np.array([g in in_bag for g in groups])
            test_mask  = np.array([g in oob    for g in groups])
            Xtr, Xte   = X_rep[train_mask], X_rep[test_mask]
            ytr, yte   = y[train_mask], y[test_mask]
            single      = np.unique(yte).shape[0] < 2 or np.unique(ytr).shape[0] < 2

            au = float("nan")
            if not single:
                try:
                    sc  = StandardScaler()
                    Xsc = sc.fit_transform(Xtr)
                    Xts = sc.transform(Xte)
                    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=random_state)
                    clf.fit(Xsc, ytr)
                    au  = _auroc(clf.decision_function(Xts), yte)
                except Exception as exc:
                    print(f"    [P2 iter={boot_i}] {exc}")

            rows.append({
                "condition":    cond_name,
                "protocol":     "bootstrap",
                "seed_or_iter": boot_i,
                "fold":         "",
                "is_degenerate": single,
                "auroc":        _fmt(au),
            })

    return rows


# ─── neuron feature inventory ──────────────────────────────────────────────────

def _compute_neuron_inventory(
    domains_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> list[dict]:
    """Per-domain statistics for nc_mean (idx=16), nc_slope (idx=17) at position 1.0."""
    n2i = _name_to_idx()
    nc_mean_idx  = n2i["nc_mean"]
    nc_slope_idx = n2i["nc_slope"]
    ss_idx       = n2i["self_similarity"]

    rows: list[dict] = []
    for domain in DOMAINS:
        X, y, _ = domains_data[domain]
        if X.shape[0] == 0:
            continue

        for feat_name, col in [("nc_mean", nc_mean_idx), ("nc_slope", nc_slope_idx),
                                ("self_similarity", ss_idx)]:
            vals = X[:, col]
            n_nonzero = int((vals != 0).sum())
            rows.append({
                "domain":         domain,
                "feature":        feat_name,
                "feature_index":  col,
                "available":      "YES" if feat_name != "self_similarity" else "NO (all-zeros)",
                "n_samples":      X.shape[0],
                "n_nonzero":      n_nonzero,
                "pct_nonzero":    f"{100.0 * n_nonzero / max(1, X.shape[0]):.1f}",
                "mean":           _fmt(float(np.mean(vals))),
                "std":            _fmt(float(np.std(vals))),
                "min":            _fmt(float(np.min(vals))),
                "max":            _fmt(float(np.max(vals))),
                "note": (
                    "rows/bank not in cache → excluded from all conditions"
                    if feat_name == "self_similarity" else ""
                ),
            })
    return rows


# ─── summary table builders ────────────────────────────────────────────────────

def _build_summary_csvs(
    ablation_rows: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Derive added_value, vs_legacy, lowrank_interaction tables."""

    # Index ablation rows for fast lookup
    idx: dict[tuple, float] = {}
    for r in ablation_rows:
        key = (r["domain"], r["condition"], r["modeling"], r["metric"])
        try:
            idx[key] = float(r["mean"])
        except (ValueError, TypeError):
            idx[key] = float("nan")

    MODELINGS = ["no_svd"]  # also add svd entry dynamically
    svd_labels = sorted({r["modeling"] for r in ablation_rows if r["modeling"] != "no_svd"})
    all_modelings = ["no_svd"] + svd_labels

    # ── added_value: A1 vs A0 ─────────────────────────────────────────────────
    added_rows: list[dict] = []
    for domain in DOMAINS:
        for metric in ["auroc", "balanced_acc", "stop_acc"]:
            for modeling in all_modelings:
                a0 = idx.get((domain, "canonical",         modeling, metric), float("nan"))
                a1 = idx.get((domain, "canonical_plus_nc", modeling, metric), float("nan"))
                delta = a1 - a0 if (np.isfinite(a0) and np.isfinite(a1)) else float("nan")
                pct = f"{100.0 * delta / max(abs(a0), 1e-9):+.1f}%" if np.isfinite(delta) else "nan"
                if abs(delta) < 0.005 if np.isfinite(delta) else False:
                    interp = "within noise"
                elif delta > 0.01 if np.isfinite(delta) else False:
                    interp = "meaningful gain"
                elif delta < -0.01 if np.isfinite(delta) else False:
                    interp = "degradation"
                else:
                    interp = "negligible"
                added_rows.append({
                    "domain":          domain,
                    "metric":          metric,
                    "modeling":        modeling,
                    "a0_mean":         _fmt(a0),
                    "a1_mean":         _fmt(a1),
                    "delta":           _fmt(delta) if np.isfinite(delta) else "nan",
                    "a1_minus_a0_pct": pct,
                    "interpretation":  interp,
                })

    # ── vs_legacy: nc_only vs canonical vs canonical_plus_nc (AUROC) ──────────
    vs_rows: list[dict] = []
    for domain in DOMAINS:
        for modeling in all_modelings:
            nc  = idx.get((domain, "nc_only",         modeling, "auroc"), float("nan"))
            can = idx.get((domain, "canonical",        modeling, "auroc"), float("nan"))
            cn  = idx.get((domain, "canonical_plus_nc",modeling, "auroc"), float("nan"))
            vals = {"nc_only": nc, "canonical": can, "canonical_plus_nc": cn}
            winner = max(vals, key=lambda k: vals[k] if np.isfinite(vals[k]) else -1)
            vs_rows.append({
                "domain":               domain,
                "modeling":             modeling,
                "nc_only_auroc":        _fmt(nc),
                "canonical_auroc":      _fmt(can),
                "canonical_plus_nc_auroc": _fmt(cn),
                "winner":               winner,
            })

    # ── lowrank_interaction ───────────────────────────────────────────────────
    lr_rows: list[dict] = []
    for domain in DOMAINS:
        for cond_name in ["canonical", "canonical_plus_nc", "nc_only",
                          "canonical_plus_prefix", "all_available"]:
            no_svd_au = idx.get((domain, cond_name, "no_svd", "auroc"), float("nan"))
            for svd_label in svd_labels:
                svd_au = idx.get((domain, cond_name, svd_label, "auroc"), float("nan"))
                gain   = svd_au - no_svd_au if (np.isfinite(svd_au) and np.isfinite(no_svd_au)) else float("nan")
                interp = ""
                if np.isfinite(gain):
                    if abs(gain) < 0.005:
                        interp = "svd≈no_svd"
                    elif gain > 0:
                        interp = "svd helps"
                    else:
                        interp = "svd hurts"
                lr_rows.append({
                    "domain":           domain,
                    "condition":        cond_name,
                    "no_svd_auroc":     _fmt(no_svd_au),
                    "svd_auroc":        _fmt(svd_au),
                    "svd_gain":         _fmt(gain),
                    "interpretation":   interp,
                })

    return added_rows, vs_rows, lr_rows


# ─── CSV output ───────────────────────────────────────────────────────────────

def _write_csvs(
    ablation_rows: list[dict],
    added_rows: list[dict],
    vs_rows: list[dict],
    lr_rows: list[dict],
    inventory_rows: list[dict],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def _write(fname: str, fieldnames: list[str], rows: list[dict]) -> Path:
        p = out_dir / fname
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"  wrote {p}  ({len(rows)} rows)")
        return p

    _write(
        "neuron_augmented_ablation.csv",
        ["domain", "condition", "n_feat", "modeling", "metric", "mean", "std",
         "n_folds_valid", "n_folds_degenerate", "n_samples", "n_groups"],
        ablation_rows,
    )
    _write(
        "neuron_added_value_by_domain.csv",
        ["domain", "metric", "modeling", "a0_mean", "a1_mean", "delta",
         "a1_minus_a0_pct", "interpretation"],
        added_rows,
    )
    _write(
        "neuron_vs_legacy_summary.csv",
        ["domain", "modeling", "nc_only_auroc", "canonical_auroc",
         "canonical_plus_nc_auroc", "winner"],
        vs_rows,
    )
    _write(
        "lowrank_interaction_summary.csv",
        ["domain", "condition", "no_svd_auroc", "svd_auroc", "svd_gain", "interpretation"],
        lr_rows,
    )
    _write(
        "neuron_feature_inventory.csv",
        ["domain", "feature", "feature_index", "available", "n_samples", "n_nonzero",
         "pct_nonzero", "mean", "std", "min", "max", "note"],
        inventory_rows,
    )


# ─── Markdown doc ─────────────────────────────────────────────────────────────

def _instability_summary(
    rows: list[dict], cond: str
) -> tuple[float, float, float, float]:
    """Return (rgkf_mean, rgkf_std, boot_ci_lo, boot_ci_hi) for one condition."""
    rgkf = [float(r["auroc"]) for r in rows
            if r["condition"] == cond and r["protocol"] == "repeated_gkf"
            and r["auroc"] not in ("nan", "")]
    boot = [float(r["auroc"]) for r in rows
            if r["condition"] == cond and r["protocol"] == "bootstrap"
            and r["auroc"] not in ("nan", "")]
    m  = float(np.mean(rgkf)) if rgkf else float("nan")
    s  = float(np.std(rgkf))  if rgkf else float("nan")
    lo = float(np.percentile(boot, 2.5))  if boot else float("nan")
    hi = float(np.percentile(boot, 97.5)) if boot else float("nan")
    return m, s, lo, hi


def _write_doc(
    ablation_rows: list[dict],
    added_rows: list[dict],
    vs_rows: list[dict],
    lr_rows: list[dict],
    inventory_rows: list[dict],
    instability_rows: list[dict],
    doc_path: Path,
) -> None:
    # ── helpers ───────────────────────────────────────────────────────────────
    def _idx(domain: str, cond: str, modeling: str, metric: str) -> float:
        for r in ablation_rows:
            if (r["domain"] == domain and r["condition"] == cond
                    and r["modeling"] == modeling and r["metric"] == metric):
                try:
                    return float(r["mean"])
                except (ValueError, TypeError):
                    return float("nan")
        return float("nan")

    svd_label = next(
        (r["modeling"] for r in ablation_rows if r["modeling"] != "no_svd"), "svd_r12"
    )

    # ── forced verdicts ───────────────────────────────────────────────────────
    # Q1: Did A1 beat A0 on AUROC (any domain)?
    a1_beats_a0 = False
    domain_deltas: dict[str, float] = {}
    for domain in DOMAINS:
        a0 = _idx(domain, "canonical",         "no_svd", "auroc")
        a1 = _idx(domain, "canonical_plus_nc", "no_svd", "auroc")
        d  = a1 - a0 if (np.isfinite(a0) and np.isfinite(a1)) else float("nan")
        domain_deltas[domain] = d
        if np.isfinite(d) and d > 0.005:
            a1_beats_a0 = True

    # Q2: Does nc_only (A2) beat random (>0.55) on any domain?
    nc_above_chance = {d: _idx(d, "nc_only", "no_svd", "auroc") > 0.55 for d in DOMAINS}

    # Q3: SVD interaction — does nc augmentation change SVD gain differently?
    canonical_svd_gain: dict[str, float] = {}
    nc_svd_gain: dict[str, float] = {}
    for domain in DOMAINS:
        ns_can = _idx(domain, "canonical",         "no_svd",  "auroc")
        sv_can = _idx(domain, "canonical",          svd_label, "auroc")
        ns_nc  = _idx(domain, "canonical_plus_nc", "no_svd",  "auroc")
        sv_nc  = _idx(domain, "canonical_plus_nc",  svd_label, "auroc")
        canonical_svd_gain[domain] = (sv_can - ns_can) if (np.isfinite(sv_can) and np.isfinite(ns_can)) else float("nan")
        nc_svd_gain[domain]        = (sv_nc  - ns_nc)  if (np.isfinite(sv_nc)  and np.isfinite(ns_nc))  else float("nan")

    # Q4: Coding instability — does A1 reduce bootstrap CI width?
    if instability_rows:
        m0, s0, lo0, hi0 = _instability_summary(instability_rows, "canonical")
        m1, s1, lo1, hi1 = _instability_summary(instability_rows, "canonical_plus_nc")
        ci_narrowed = (np.isfinite(hi0) and np.isfinite(hi1) and (hi1 - lo1) < (hi0 - lo0))
    else:
        s0, lo0, hi0 = float("nan"), float("nan"), float("nan")
        s1, lo1, hi1 = float("nan"), float("nan"), float("nan")
        ci_narrowed  = False

    # Q5: Recommended framing
    any_meaningful = any(
        d > 0.005 for d in domain_deltas.values() if np.isfinite(d)
    )
    if any_meaningful:
        framing = "inline note: nc_mean/nc_slope provide marginal gains in [domain]; report as supplementary."
    else:
        framing = "future work: rows/bank (self_similarity) unavailable in current cache; re-evaluate when bank is rebuilt."

    # ── build document ────────────────────────────────────────────────────────
    lines: list[str] = [
        "# 15: Neuron-Augmented Ablation",
        "",
        "**Date**: 2026-04-12  ",
        "**Status**: Analysis complete  ",
        "**Data**: cache_all_547b9060debe139e.pkl · 30 legacy features · position 1.0  ",
        "**Note on self_similarity (index 18)**: all-zeros in this cache (rows/bank unavailable). Excluded from all conditions. See inventory CSV.",
        "",
        "---",
        "",
        "## 1. Conditions",
        "",
        "| ID | Name | Feature indices | N feat | Notes |",
        "|----|------|-----------------|--------|-------|",
        "| A0 | `canonical` | TOKEN[0-10] + TRAJ[11-15] + AVAIL[19-24] | 22 | Current production |",
        "| A1 | `canonical_plus_nc` | A0 + nc_mean[16] + nc_slope[17] | 24 | Neuron augmented |",
        "| A2 | `nc_only` | nc_mean[16] + nc_slope[17] + AVAIL[19-24] | 8 | Neuron-only baseline |",
        "| A3 | `canonical_plus_prefix` | A0 + PREFIX_LOCAL[25-29] | 27 | Upper-bound proxy |",
        "| A4 | `all_available` | A1 + PREFIX_LOCAL[25-29] | 29 | All non-zero features |",
        "",
        "Modeling: `no_svd` (StandardScaler+LR) and `svd_r12` (Scaler+TruncatedSVD(r=12)+LR).  ",
        "Representation: always `raw+rank`. Evaluation: 5-fold GroupKFold.",
        "",
        "---",
        "",
        "## 2. Main Ablation Results",
        "",
        "### AUROC by Domain and Condition",
        "",
        "| Domain | Condition | no_svd AUROC | svd_r12 AUROC | SVD gain |",
        "|--------|-----------|-------------|--------------|---------|",
    ]
    for domain in DOMAINS:
        for cond in ["canonical", "canonical_plus_nc", "nc_only",
                     "canonical_plus_prefix", "all_available"]:
            ns = _idx(domain, cond, "no_svd",  "auroc")
            sv = _idx(domain, cond, svd_label,  "auroc")
            g  = sv - ns if (np.isfinite(sv) and np.isfinite(ns)) else float("nan")
            g_str = f"{g:+.4f}" if np.isfinite(g) else "nan"
            lines.append(f"| {domain} | {cond} | {_fmt(ns)} | {_fmt(sv)} | {g_str} |")

    lines += [
        "",
        "---",
        "",
        "## 3. Neuron Feature Inventory",
        "",
        "| Domain | Feature | Pct nonzero | Mean | Std | Available |",
        "|--------|---------|------------|------|-----|-----------|",
    ]
    for r in inventory_rows:
        lines.append(
            f"| {r['domain']} | {r['feature']} | {r['pct_nonzero']}% "
            f"| {r['mean']} | {r['std']} | {r['available']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 4. Added Value of Neuron Features (A1 vs A0)",
        "",
        "AUROC delta (A1 − A0) for `no_svd`:",
        "",
        "| Domain | A0 AUROC | A1 AUROC | Δ | Interpretation |",
        "|--------|---------|---------|---|----------------|",
    ]
    for domain in DOMAINS:
        a0 = _idx(domain, "canonical",         "no_svd", "auroc")
        a1 = _idx(domain, "canonical_plus_nc", "no_svd", "auroc")
        d  = domain_deltas.get(domain, float("nan"))
        interp = "negligible"
        if np.isfinite(d):
            if d > 0.01:   interp = "meaningful gain"
            elif d > 0.005: interp = "marginal gain"
            elif d < -0.005: interp = "degradation"
        d_str = f"{d:+.4f}" if np.isfinite(d) else "nan"
        lines.append(f"| {domain} | {_fmt(a0)} | {_fmt(a1)} | {d_str} | {interp} |")

    if instability_rows:
        lines += [
            "",
            "---",
            "",
            "## 5. Coding Instability — A0 vs A1",
            "",
            f"**A0 (canonical)**: repeated-GKF std={_fmt(s0)}, bootstrap 95% CI=[{_fmt(lo0)}, {_fmt(hi0)}]  ",
            f"**A1 (canonical_plus_nc)**: repeated-GKF std={_fmt(s1)}, bootstrap 95% CI=[{_fmt(lo1)}, {_fmt(hi1)}]  ",
            f"**CI width narrowed by nc features**: {ci_narrowed}  ",
        ]

    lines += [
        "",
        "---",
        "",
        "## 6. Forced Verdict",
        "",
        f"**Q1 — Did A1 beat A0 on AUROC (Δ > 0.005) across any domain?**  ",
        f"→ {'YES' if a1_beats_a0 else 'NO'}  ",
    ]
    for domain in DOMAINS:
        d = domain_deltas.get(domain, float("nan"))
        d_str = f"{d:+.4f}" if np.isfinite(d) else "nan"
        lines.append(f"  - {domain}: Δ = {d_str}")

    lines += [
        "",
        f"**Q2 — Does nc_only (A2) beat chance (AUROC > 0.55) on any domain?**  ",
        f"→ {'YES' if any(nc_above_chance.values()) else 'NO'}  ",
    ]
    for domain in DOMAINS:
        v = _idx(domain, "nc_only", "no_svd", "auroc")
        lines.append(f"  - {domain}: {_fmt(v)}")

    lines += [
        "",
        "**Q3 — Does nc augmentation interact differently with SVD (different gain vs no_svd)?**  ",
    ]
    for domain in DOMAINS:
        g0 = canonical_svd_gain.get(domain, float("nan"))
        g1 = nc_svd_gain.get(domain, float("nan"))
        diff = g1 - g0 if (np.isfinite(g0) and np.isfinite(g1)) else float("nan")
        if np.isfinite(diff):
            lines.append(f"  - {domain}: canonical SVD gain={g0:+.4f}, nc SVD gain={g1:+.4f}, diff={diff:+.4f}")
        else:
            lines.append(f"  - {domain}: canonical SVD gain={_fmt(g0)}, nc SVD gain={_fmt(g1)}")

    if instability_rows:
        lines += [
            "",
            f"**Q4 — Did nc features reduce bootstrap CI width for coding?**  ",
            f"→ {'YES' if ci_narrowed else 'NO'}  ",
            f"  - A0 CI width: {_fmt(hi0 - lo0) if np.isfinite(hi0) and np.isfinite(lo0) else 'nan'}",
            f"  - A1 CI width: {_fmt(hi1 - lo1) if np.isfinite(hi1) and np.isfinite(lo1) else 'nan'}",
        ]

    lines += [
        "",
        f"**Q5 — Recommended paper framing**:  ",
        f"→ {framing}",
        "",
        "---",
        "",
        "## 7. Limitations",
        "",
        "- `self_similarity` (index 18, META_FEATURES) is all-zeros in this cache because the rows/bank",
        "  was not populated during feature extraction. This feature is excluded from all conditions.",
        "- `has_rows_bank` (index 24, AVAILABILITY_FEATURES) is also all-zeros for the same reason.",
        "- The neuron inventory therefore covers only `nc_mean` and `nc_slope` as usable neuron-derived signals.",
        "- PREFIX_LOCAL features (A3, A4) serve as the best proxy upper-bound since neuron_meta_plus",
        "  (unique_activated_neuron_count etc.) are not in the 30-feature legacy schema.",
        "",
        "---",
        "",
        "*Generated by SVDomain/run_neuron_augmented_ablation.py*",
    ]

    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {doc_path}")


# ─── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Neuron-Augmented Ablation")
    ap.add_argument("--store-path",      default=DEFAULT_STORE_PATH)
    ap.add_argument("--out-dir",         default=DEFAULT_OUT_DIR)
    ap.add_argument("--doc-path",        default=DEFAULT_DOC_PATH)
    ap.add_argument("--svd-rank",        type=int,   default=12)
    ap.add_argument("--n-splits",        type=int,   default=5)
    ap.add_argument("--n-seeds",         type=int,   default=10)
    ap.add_argument("--n-bootstrap",     type=int,   default=100)
    ap.add_argument("--random-state",    type=int,   default=42)
    ap.add_argument("--skip-instability", action="store_true",
                    help="Skip coding repeated-GKF instability analysis (faster smoke test)")
    args = ap.parse_args()

    store_path = Path(REPO_ROOT / args.store_path)
    out_dir    = Path(REPO_ROOT / args.out_dir)
    doc_path   = Path(REPO_ROOT / args.doc_path)

    print(f"Loading feature store: {store_path}")
    domains_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for domain in DOMAINS:
        X, y, groups = _load_domain_data(store_path, domain)
        domains_data[domain] = (X, y, groups)
        n_g = int(np.unique(groups).shape[0]) if X.shape[0] > 0 else 0
        print(f"  {domain}: n={X.shape[0]}  groups={n_g}  pos={int((y==1).sum())}")

    conditions = _get_conditions()
    print("\nConditions:")
    for name, idx in conditions.items():
        print(f"  {name}: {len(idx)} features  indices={idx[:5]}...")

    # ── neuron feature inventory ───────────────────────────────────────────────
    print("\n[INVENTORY] Computing neuron feature stats ...")
    inventory_rows = _compute_neuron_inventory(domains_data)

    # ── main ablation ──────────────────────────────────────────────────────────
    ablation_rows = _run_main_ablation(
        domains_data, conditions,
        svd_rank=args.svd_rank,
        n_splits=args.n_splits,
        random_state=args.random_state,
    )

    # ── coding instability ─────────────────────────────────────────────────────
    instability_rows: list[dict] = []
    if not args.skip_instability:
        print("\n[INSTABILITY] Coding domain: A0 vs A1 ...")
        X_c, y_c, g_c = domains_data["coding"]
        instability_rows = _run_coding_instability(
            X_c, y_c, g_c,
            conditions_subset={
                "canonical":         conditions["canonical"],
                "canonical_plus_nc": conditions["canonical_plus_nc"],
            },
            n_seeds=args.n_seeds,
            n_bootstrap=args.n_bootstrap,
            n_splits=args.n_splits,
            random_state=args.random_state,
        )

    # ── summary tables ─────────────────────────────────────────────────────────
    print("\n[SUMMARY] Building summary tables ...")
    added_rows, vs_rows, lr_rows = _build_summary_csvs(ablation_rows)

    # ── write outputs ──────────────────────────────────────────────────────────
    print("\n[OUTPUT] Writing CSVs ...")
    _write_csvs(ablation_rows, added_rows, vs_rows, lr_rows, inventory_rows, out_dir)

    if instability_rows:
        out_dir.mkdir(parents=True, exist_ok=True)
        inst_path = out_dir / "neuron_coding_instability.csv"
        with inst_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "condition", "protocol", "seed_or_iter", "fold", "is_degenerate", "auroc"
            ])
            w.writeheader()
            w.writerows(instability_rows)
        print(f"  wrote {inst_path}  ({len(instability_rows)} rows)")

    print("\n[OUTPUT] Writing markdown doc ...")
    _write_doc(
        ablation_rows, added_rows, vs_rows, lr_rows,
        inventory_rows, instability_rows, doc_path,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
