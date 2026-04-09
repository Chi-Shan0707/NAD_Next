from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from nad.core.selectors.trajectory_impl import (
    DEFAULT_REFLECTION_THRESHOLD,
    _compute_trajectory_scores_for_prefix_counts,
    _compute_trajectory_scores,
    _extract_slice_keysets,
)
from nad.core.views.reader import CacheReader
from nad.ops.accuracy import load_correctness_map
from nad.ops.earlystop import (
    CacheEntry,
    EARLY_STOP_POSITIONS,
    N_POSITIONS,
    build_problem_groups,
    discover_cache_entries,
    _problem_sort_key,
)


MATH_DATASETS = {"aime24", "aime25", "brumo25", "hmmt25"}
SCIENCE_DATASETS = {"gpqa"}
CODING_DATASETS = {"livecodebench_v5", "lcb_v5"}

TOKEN_FEATURES = [
    "tok_conf_prefix",
    "tok_conf_recency",
    "tok_gini_prefix",
    "tok_gini_tail",
    "tok_gini_slope",
    "tok_neg_entropy_prefix",
    "tok_neg_entropy_recency",
    "tok_selfcert_prefix",
    "tok_selfcert_recency",
    "tok_logprob_prefix",
    "tok_logprob_recency",
]

TRAJ_FEATURES = [
    "traj_continuity",
    "traj_reflection_count",
    "traj_novelty",
    "traj_max_reflection",
    "traj_late_convergence",
]

META_FEATURES = [
    "nc_mean",
    "nc_slope",
    "self_similarity",
]

AVAILABILITY_FEATURES = [
    "has_tok_conf",
    "has_tok_gini",
    "has_tok_neg_entropy",
    "has_tok_selfcert",
    "has_tok_logprob",
    "has_rows_bank",
]

FULL_FEATURE_NAMES = TOKEN_FEATURES + TRAJ_FEATURES + META_FEATURES + AVAILABILITY_FEATURES

BASELINE_SIGNAL_NAMES = TOKEN_FEATURES + TRAJ_FEATURES + META_FEATURES

FEATURE_FAMILY_MAP = {
    "token_only": TOKEN_FEATURES + [
        "has_tok_conf",
        "has_tok_gini",
        "has_tok_neg_entropy",
        "has_tok_selfcert",
        "has_tok_logprob",
    ],
    "token_plus_traj": TOKEN_FEATURES + TRAJ_FEATURES + [
        "has_tok_conf",
        "has_tok_gini",
        "has_tok_neg_entropy",
        "has_tok_selfcert",
        "has_tok_logprob",
        "has_rows_bank",
    ],
    "all": FULL_FEATURE_NAMES,
}

REPRESENTATIONS = ("raw", "rank", "raw+rank")


@dataclass(frozen=True)
class SVDSearchConfig:
    n_splits: int = 5
    family_names: tuple[str, ...] = ("token_only", "token_plus_traj", "all")
    representations: tuple[str, ...] = REPRESENTATIONS
    ranks: tuple[int, ...] = (2, 4, 6, 8, 12, 16)
    c_values: tuple[float, ...] = (0.1, 1.0, 3.0, 10.0)
    whiten_options: tuple[bool, ...] = (False, True)
    class_weight_options: tuple[str, ...] = ("none", "balanced")
    random_state: int = 42
    max_problems_per_cache: int = 0


def get_domain(dataset_name: str) -> str:
    if dataset_name in MATH_DATASETS:
        return "math"
    if dataset_name in SCIENCE_DATASETS:
        return "science"
    if dataset_name in CODING_DATASETS:
        return "coding"
    return "math"


def _prefix_mean(arr: Optional[np.ndarray], p: float) -> float:
    if arr is None:
        return 0.0
    t = int(arr.shape[0])
    if t == 0:
        return 0.0
    cut = max(1, int(p * t))
    return float(np.mean(arr[:cut]))


def _prefix_recency(arr: Optional[np.ndarray], p: float, lam: float = 0.3) -> float:
    if arr is None:
        return 0.0
    t = int(arr.shape[0])
    if t == 0:
        return 0.0
    cut = max(1, int(p * t))
    seg = arr[:cut]
    w = np.exp(lam * np.arange(cut, dtype=np.float64) / max(1, cut))
    return float(np.average(seg, weights=w))


def _prefix_tail_mean(arr: Optional[np.ndarray], p: float, tail_frac: float = 0.1) -> float:
    if arr is None:
        return 0.0
    t = int(arr.shape[0])
    if t == 0:
        return 0.0
    cut = max(1, int(p * t))
    tail_w = max(1, int(tail_frac * cut))
    return float(np.mean(arr[max(0, cut - tail_w):cut]))


def _prefix_half_slope(arr: Optional[np.ndarray], p: float) -> float:
    if arr is None:
        return 0.0
    t = int(arr.shape[0])
    if t < 2:
        return 0.0
    cut = max(1, int(p * t))
    if cut < 2:
        return 0.0
    half = cut // 2
    return float(np.mean(arr[half:cut]) - np.mean(arr[:half]))


def _prefix_count_slope(arr: Optional[np.ndarray], k: int) -> float:
    if arr is None:
        return 0.0
    if int(k) < 2:
        return 0.0
    cut = min(int(k), int(len(arr)))
    if cut < 2:
        return 0.0
    half = cut // 2
    if half <= 0 or half >= cut:
        return 0.0
    return float(np.mean(arr[half:cut]) - np.mean(arr[:half]))


def _empty_signal_map() -> dict[str, list[float]]:
    return {name: [0.0] * N_POSITIONS for name in FULL_FEATURE_NAMES}


def extract_earlystop_signals_for_sample(
    reader: CacheReader,
    run_id: int,
    required_features: Optional[set[str]] = None,
) -> dict[str, list[float]]:
    """Extract domain-agnostic early-stop features for one run.

    All continuous features follow "higher is better" orientation where possible.
    For example, reflection counts are negated.
    """
    req = set(FULL_FEATURE_NAMES) if required_features is None else set(required_features)
    tv = reader.get_token_view(int(run_id))
    if tv is None:
        return _empty_signal_map()

    need_tok_conf = bool(req & {"tok_conf_prefix", "tok_conf_recency", "has_tok_conf"})
    need_tok_gini = bool(req & {"tok_gini_prefix", "tok_gini_tail", "tok_gini_slope", "has_tok_gini"})
    need_tok_neg_entropy = bool(req & {"tok_neg_entropy_prefix", "tok_neg_entropy_recency", "has_tok_neg_entropy"})
    need_tok_selfcert = bool(req & {"tok_selfcert_prefix", "tok_selfcert_recency", "has_tok_selfcert"})
    need_tok_logprob = bool(req & {"tok_logprob_prefix", "tok_logprob_recency", "has_tok_logprob"})
    need_traj = bool(req & set(TRAJ_FEATURES))
    need_self_similarity = "self_similarity" in req
    need_ncount = bool(req & {"nc_mean", "nc_slope"})
    need_has_rows = "has_rows_bank" in req

    tok_conf_arr = None
    tok_gini_arr = None
    tok_neg_entropy_arr = None
    tok_selfcert_arr = None
    tok_logprob_arr = None
    if need_tok_conf and tv.tok_conf is not None:
        tok_conf_arr = np.asarray(tv.tok_conf, dtype=np.float64)
    if need_tok_gini and tv.tok_gini is not None:
        tok_gini_arr = np.asarray(tv.tok_gini, dtype=np.float64)
    if need_tok_neg_entropy and tv.tok_neg_entropy is not None:
        tok_neg_entropy_arr = np.asarray(tv.tok_neg_entropy, dtype=np.float64)
    if need_tok_selfcert and tv.tok_selfcert is not None:
        tok_selfcert_arr = np.asarray(tv.tok_selfcert, dtype=np.float64)
    if need_tok_logprob and tv.tok_logprob is not None:
        tok_logprob_arr = np.asarray(tv.tok_logprob, dtype=np.float64)

    rows_srp = reader.rows_sample_row_ptr
    rows_rp = reader.rows_row_ptr
    row_start = -1
    row_end = -1
    if rows_srp is not None and int(run_id) < len(rows_srp) - 1:
        row_start = int(rows_srp[int(run_id)])
        row_end = int(rows_srp[int(run_id) + 1])

    slices: list[Any] = []
    n_slices = 0
    if need_traj or need_self_similarity:
        slices = _extract_slice_keysets(reader, int(run_id))
        n_slices = len(slices)
    elif row_start >= 0 and row_end >= 0:
        n_slices = max(0, row_end - row_start)

    has_rows_bank = 1.0 if n_slices > 0 else 0.0

    nc_all: Optional[np.ndarray] = None
    if need_ncount and rows_rp is not None and row_start >= 0 and row_end > row_start:
        try:
            rp_seg = np.asarray(rows_rp[row_start:row_end + 1], dtype=np.int64)
            nc_all = np.diff(rp_seg).astype(np.float64)
        except Exception:
            nc_all = None

    self_similarity = 0.0
    if need_self_similarity and n_slices > 1:
        half = n_slices // 2
        try:
            first = set()
            for s in slices[:half]:
                first.update(int(k) for k in s)
            second = set()
            for s in slices[half:]:
                second.update(int(k) for k in s)
            inter = len(first & second)
            union = len(first | second)
            self_similarity = float(inter / union) if union > 0 else 0.0
        except Exception:
            self_similarity = 0.0

    traj_by_cut: dict[int, dict[str, float]] = {}
    if need_traj and n_slices > 0:
        traj_by_cut = _compute_trajectory_scores_for_prefix_counts(
            slices,
            [max(1, int(p * n_slices)) for p in EARLY_STOP_POSITIONS],
            reflection_threshold=DEFAULT_REFLECTION_THRESHOLD,
        )

    signals = _empty_signal_map()

    has_tok_conf = 1.0 if tok_conf_arr is not None and len(tok_conf_arr) > 0 else 0.0
    has_tok_gini = 1.0 if tok_gini_arr is not None and len(tok_gini_arr) > 0 else 0.0
    has_tok_neg_entropy = 1.0 if tok_neg_entropy_arr is not None and len(tok_neg_entropy_arr) > 0 else 0.0
    has_tok_selfcert = 1.0 if tok_selfcert_arr is not None and len(tok_selfcert_arr) > 0 else 0.0
    has_tok_logprob = 1.0 if tok_logprob_arr is not None and len(tok_logprob_arr) > 0 else 0.0

    for pos_i, p in enumerate(EARLY_STOP_POSITIONS):
        if "tok_conf_prefix" in req:
            signals["tok_conf_prefix"][pos_i] = _prefix_mean(tok_conf_arr, p)
        if "tok_conf_recency" in req:
            signals["tok_conf_recency"][pos_i] = _prefix_recency(tok_conf_arr, p)

        if "tok_gini_prefix" in req:
            signals["tok_gini_prefix"][pos_i] = _prefix_mean(tok_gini_arr, p)
        if "tok_gini_tail" in req:
            signals["tok_gini_tail"][pos_i] = _prefix_tail_mean(tok_gini_arr, p)
        if "tok_gini_slope" in req:
            signals["tok_gini_slope"][pos_i] = _prefix_half_slope(tok_gini_arr, p)

        if "tok_neg_entropy_prefix" in req:
            signals["tok_neg_entropy_prefix"][pos_i] = _prefix_mean(tok_neg_entropy_arr, p)
        if "tok_neg_entropy_recency" in req:
            signals["tok_neg_entropy_recency"][pos_i] = _prefix_recency(tok_neg_entropy_arr, p)

        if "tok_selfcert_prefix" in req:
            signals["tok_selfcert_prefix"][pos_i] = _prefix_mean(tok_selfcert_arr, p)
        if "tok_selfcert_recency" in req:
            signals["tok_selfcert_recency"][pos_i] = _prefix_recency(tok_selfcert_arr, p)

        if "tok_logprob_prefix" in req:
            signals["tok_logprob_prefix"][pos_i] = _prefix_mean(tok_logprob_arr, p)
        if "tok_logprob_recency" in req:
            signals["tok_logprob_recency"][pos_i] = _prefix_recency(tok_logprob_arr, p)

        if need_traj and n_slices > 0:
            k = max(1, int(p * n_slices))
            traj = traj_by_cut[k]
            if "traj_continuity" in req:
                signals["traj_continuity"][pos_i] = float(traj["mean_continuity"])
            if "traj_reflection_count" in req:
                signals["traj_reflection_count"][pos_i] = -float(traj["reflection_count"])
            if "traj_novelty" in req:
                signals["traj_novelty"][pos_i] = float(traj["mean_novelty"])
            if "traj_max_reflection" in req:
                signals["traj_max_reflection"][pos_i] = -float(traj["max_reflection"])
            if "traj_late_convergence" in req:
                signals["traj_late_convergence"][pos_i] = float(traj["late_convergence"])

        if need_ncount and nc_all is not None and len(nc_all) > 0:
            k = max(1, int(p * len(nc_all)))
            if "nc_mean" in req:
                signals["nc_mean"][pos_i] = float(np.mean(nc_all[:k]))
            if "nc_slope" in req:
                signals["nc_slope"][pos_i] = _prefix_count_slope(nc_all, k)

        if "self_similarity" in req:
            signals["self_similarity"][pos_i] = self_similarity

        if "has_tok_conf" in req:
            signals["has_tok_conf"][pos_i] = has_tok_conf
        if "has_tok_gini" in req:
            signals["has_tok_gini"][pos_i] = has_tok_gini
        if "has_tok_neg_entropy" in req:
            signals["has_tok_neg_entropy"][pos_i] = has_tok_neg_entropy
        if "has_tok_selfcert" in req:
            signals["has_tok_selfcert"][pos_i] = has_tok_selfcert
        if "has_tok_logprob" in req:
            signals["has_tok_logprob"][pos_i] = has_tok_logprob
        if need_has_rows:
            signals["has_rows_bank"][pos_i] = has_rows_bank

    return signals


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    try:
        from scipy.stats import mannwhitneyu

        stat, _ = mannwhitneyu(scores[pos], scores[neg], alternative="greater")
        return float(stat) / float(n_pos * n_neg)
    except Exception:
        ranks = np.argsort(np.argsort(scores)) + 1
        u = float(ranks[pos].sum()) - n_pos * (n_pos + 1) / 2.0
        return u / float(n_pos * n_neg)


def _group_folds(groups: np.ndarray, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return []
    splits = min(int(n_splits), int(len(unique_groups)))
    if splits < 2:
        return []
    gkf = GroupKFold(n_splits=splits)
    dummy_x = np.zeros((len(groups), 1), dtype=np.float64)
    return list(gkf.split(dummy_x, groups=groups))


def _rank_transform_matrix(x: np.ndarray) -> np.ndarray:
    n, m = x.shape
    out = np.zeros_like(x, dtype=np.float64)
    if n <= 1:
        return out
    for col_i in range(m):
        col = x[:, col_i]
        order = np.argsort(col, kind="mergesort")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n, dtype=np.float64)
        out[:, col_i] = ranks / float(n - 1)
    return out


def _fit_svd_lr_model(
    x: np.ndarray,
    y: np.ndarray,
    rank: int,
    c_value: float,
    whiten: bool,
    class_weight_name: str,
    random_state: int,
) -> Optional[dict[str, Any]]:
    if x.shape[0] < 4 or x.shape[1] < 1:
        return None
    if np.unique(y).shape[0] < 2:
        return None

    scaler = StandardScaler(with_mean=True, with_std=True)
    x_scaled = scaler.fit_transform(x)

    max_rank = min(int(rank), int(x_scaled.shape[1]), int(x_scaled.shape[0] - 1))
    if max_rank < 1:
        return None

    svd = TruncatedSVD(n_components=max_rank, random_state=int(random_state))
    z = svd.fit_transform(x_scaled)

    if whiten:
        s = np.asarray(svd.singular_values_, dtype=np.float64)
        s = np.where(np.abs(s) < 1e-8, 1.0, s)
        z = z / s

    class_weight = None if class_weight_name == "none" else "balanced"

    clf = LogisticRegression(
        C=float(c_value),
        class_weight=class_weight,
        max_iter=2000,
        random_state=int(random_state),
    )
    clf.fit(z, y)

    return {
        "scaler": scaler,
        "svd": svd,
        "lr": clf,
        "whiten": bool(whiten),
    }


def _predict_svd_lr(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    scaler = model["scaler"]
    svd = model["svd"]
    clf = model["lr"]
    z = svd.transform(scaler.transform(x))
    if bool(model.get("whiten", False)):
        s = np.asarray(svd.singular_values_, dtype=np.float64)
        s = np.where(np.abs(s) < 1e-8, 1.0, s)
        z = z / s
    return np.asarray(clf.decision_function(z), dtype=np.float64)


def _cv_auroc_baseline(
    scores: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
) -> tuple[float, int]:
    folds = _group_folds(groups, n_splits=n_splits)
    if not folds:
        return float("nan"), 0
    vals: list[float] = []
    for _, test_idx in folds:
        y_test = y[test_idx]
        if np.unique(y_test).shape[0] < 2:
            continue
        v = _auroc(scores[test_idx], y_test)
        if np.isfinite(v):
            vals.append(float(v))
    if not vals:
        return float("nan"), 0
    return float(np.mean(vals)), len(vals)


def _cv_auroc_svd(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    rank: int,
    c_value: float,
    whiten: bool,
    class_weight_name: str,
    random_state: int,
) -> tuple[float, int]:
    folds = _group_folds(groups, n_splits=n_splits)
    if not folds:
        return float("nan"), 0
    vals: list[float] = []
    for train_idx, test_idx in folds:
        y_train = y[train_idx]
        y_test = y[test_idx]
        if np.unique(y_train).shape[0] < 2:
            continue
        if np.unique(y_test).shape[0] < 2:
            continue

        model = _fit_svd_lr_model(
            x=x[train_idx],
            y=y_train,
            rank=rank,
            c_value=c_value,
            whiten=whiten,
            class_weight_name=class_weight_name,
            random_state=random_state,
        )
        if model is None:
            continue

        test_scores = _predict_svd_lr(model, x[test_idx])
        v = _auroc(test_scores, y_test)
        if np.isfinite(v):
            vals.append(float(v))

    if not vals:
        return float("nan"), 0
    return float(np.mean(vals)), len(vals)


def _build_representation(
    x_raw: np.ndarray,
    x_rank: np.ndarray,
    feature_indices: list[int],
    representation: str,
) -> np.ndarray:
    if representation == "raw":
        return x_raw[:, feature_indices]
    if representation == "rank":
        return x_rank[:, feature_indices]
    if representation == "raw+rank":
        return np.concatenate([
            x_raw[:, feature_indices],
            x_rank[:, feature_indices],
        ], axis=1)
    raise ValueError(f"Unknown representation: {representation}")


def _parse_meta_groups(entry: CacheEntry) -> list[tuple[str, list[int]]]:
    meta = json.loads((entry.cache_root / "meta.json").read_text(encoding="utf-8"))
    groups = build_problem_groups(meta)
    return sorted(groups.items(), key=lambda kv: _problem_sort_key(kv[0]))


def _build_domain_training_tables(
    cache_root: str | Path,
    required_feature_names: Optional[set[str]] = None,
    max_problems_per_cache: Optional[int] = None,
) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    entries = discover_cache_entries(cache_root)

    rows: dict[str, dict[int, list[np.ndarray]]] = {
        "math": {i: [] for i in range(N_POSITIONS)},
        "science": {i: [] for i in range(N_POSITIONS)},
        "coding": {i: [] for i in range(N_POSITIONS)},
    }
    labels: dict[str, dict[int, list[int]]] = {
        "math": {i: [] for i in range(N_POSITIONS)},
        "science": {i: [] for i in range(N_POSITIONS)},
        "coding": {i: [] for i in range(N_POSITIONS)},
    }
    group_keys: dict[str, dict[int, list[str]]] = {
        "math": {i: [] for i in range(N_POSITIONS)},
        "science": {i: [] for i in range(N_POSITIONS)},
        "coding": {i: [] for i in range(N_POSITIONS)},
    }

    for entry in entries:
        domain = get_domain(entry.dataset_name)
        correctness = load_correctness_map(str(entry.cache_root))
        reader = CacheReader(str(entry.cache_root))

        for problem_idx, (problem_id, sample_ids) in enumerate(_parse_meta_groups(entry)):
            if max_problems_per_cache is not None and problem_idx >= int(max_problems_per_cache):
                break
            group_id = f"{entry.cache_key}::{problem_id}"
            for sample_id in sample_ids:
                y = int(bool(correctness.get(int(sample_id), False)))
                signal_map = extract_earlystop_signals_for_sample(
                    reader,
                    int(sample_id),
                    required_features=required_feature_names,
                )

                sample_mat = np.zeros((N_POSITIONS, len(FULL_FEATURE_NAMES)), dtype=np.float64)
                for f_i, f_name in enumerate(FULL_FEATURE_NAMES):
                    sample_mat[:, f_i] = np.asarray(signal_map[f_name], dtype=np.float64)

                for pos_i in range(N_POSITIONS):
                    rows[domain][pos_i].append(sample_mat[pos_i])
                    labels[domain][pos_i].append(y)
                    group_keys[domain][pos_i].append(group_id)

    out: dict[str, dict[int, dict[str, np.ndarray]]] = {
        "math": {},
        "science": {},
        "coding": {},
    }

    for domain in out.keys():
        for pos_i in range(N_POSITIONS):
            if rows[domain][pos_i]:
                x_raw = np.vstack(rows[domain][pos_i]).astype(np.float64, copy=False)
                y = np.asarray(labels[domain][pos_i], dtype=np.int32)
                groups = np.asarray(group_keys[domain][pos_i], dtype=object)
            else:
                x_raw = np.zeros((0, len(FULL_FEATURE_NAMES)), dtype=np.float64)
                y = np.zeros((0,), dtype=np.int32)
                groups = np.asarray([], dtype=object)

            x_rank = np.zeros_like(x_raw)
            if x_raw.shape[0] > 0:
                by_group: dict[Any, list[int]] = {}
                for idx, g in enumerate(groups.tolist()):
                    by_group.setdefault(g, []).append(idx)
                for idxs in by_group.values():
                    sub = x_raw[idxs]
                    x_rank[idxs] = _rank_transform_matrix(sub)

            out[domain][pos_i] = {
                "x_raw": x_raw,
                "x_rank": x_rank,
                "y": y,
                "groups": groups,
            }

    return out


def train_earlystop_svd_bundle(
    cache_root: str | Path,
    config: SVDSearchConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    required_feature_names: set[str] = set()
    for family_name in config.family_names:
        required_feature_names.update(FEATURE_FAMILY_MAP[family_name])
    baseline_candidates = [s for s in BASELINE_SIGNAL_NAMES if s in required_feature_names]
    if not baseline_candidates:
        baseline_candidates = ["tok_conf_prefix"]
        required_feature_names.add("tok_conf_prefix")

    max_problems_per_cache = None
    if int(config.max_problems_per_cache) > 0:
        max_problems_per_cache = int(config.max_problems_per_cache)

    tables = _build_domain_training_tables(
        cache_root,
        required_feature_names=required_feature_names,
        max_problems_per_cache=max_problems_per_cache,
    )

    feature_to_idx = {name: i for i, name in enumerate(FULL_FEATURE_NAMES)}

    bundle: dict[str, Any] = {
        "bundle_version": "earlystop_svd_lowrank_lr_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_names": list(FULL_FEATURE_NAMES),
        "positions": list(EARLY_STOP_POSITIONS),
        "domains": {},
    }

    summary: dict[str, Any] = {
        "bundle_version": "earlystop_svd_lowrank_lr_v1",
        "created_at_utc": bundle["created_at_utc"],
        "cache_root": str(cache_root),
        "config": {
            "n_splits": int(config.n_splits),
            "families": list(config.family_names),
            "representations": list(config.representations),
            "ranks": list(config.ranks),
            "c_values": list(config.c_values),
            "whiten_options": [bool(v) for v in config.whiten_options],
            "class_weight_options": list(config.class_weight_options),
            "random_state": int(config.random_state),
            "max_problems_per_cache": int(config.max_problems_per_cache),
        },
        "domains": {},
    }

    total_svd_slots = 0
    total_baseline_slots = 0

    for domain in ("math", "science", "coding"):
        domain_bundle = {
            "routes": [],
        }
        domain_summary: dict[str, Any] = {
            "positions": [],
        }

        for pos_i in range(N_POSITIONS):
            tbl = tables[domain][pos_i]
            x_raw = tbl["x_raw"]
            x_rank = tbl["x_rank"]
            y = tbl["y"]
            groups = tbl["groups"]

            pos_desc = f"{domain}@{EARLY_STOP_POSITIONS[pos_i]:.1f}"
            print(f"[train] {pos_desc} samples={x_raw.shape[0]} groups={len(np.unique(groups))}")

            if x_raw.shape[0] == 0 or np.unique(y).shape[0] < 2:
                fallback_signal = "tok_conf_prefix"
                route = {
                    "route_type": "baseline",
                    "signal_name": fallback_signal,
                    "cv_auroc": float("nan"),
                    "note": "insufficient labeled data",
                }
                domain_bundle["routes"].append(route)
                domain_summary["positions"].append({
                    "position": float(EARLY_STOP_POSITIONS[pos_i]),
                    "route": route,
                })
                total_baseline_slots += 1
                continue

            best_baseline = {
                "signal_name": None,
                "cv_auroc": float("-inf"),
                "n_valid_folds": 0,
            }

            for signal_name in baseline_candidates:
                signal_idx = feature_to_idx[signal_name]
                score_vec = x_raw[:, signal_idx]
                cv_auc, n_folds = _cv_auroc_baseline(
                    scores=score_vec,
                    y=y,
                    groups=groups,
                    n_splits=config.n_splits,
                )
                if np.isfinite(cv_auc) and cv_auc > float(best_baseline["cv_auroc"]):
                    best_baseline = {
                        "signal_name": signal_name,
                        "cv_auroc": float(cv_auc),
                        "n_valid_folds": int(n_folds),
                    }

            best_svd: dict[str, Any] = {
                "cv_auroc": float("-inf"),
            }

            for family_name in config.family_names:
                fam_features = FEATURE_FAMILY_MAP[family_name]
                feat_indices = [feature_to_idx[n] for n in fam_features]

                for rep in config.representations:
                    x_rep = _build_representation(
                        x_raw=x_raw,
                        x_rank=x_rank,
                        feature_indices=feat_indices,
                        representation=rep,
                    )

                    for rank in config.ranks:
                        for c_value in config.c_values:
                            for whiten in config.whiten_options:
                                for class_weight_name in config.class_weight_options:
                                    cv_auc, n_folds = _cv_auroc_svd(
                                        x=x_rep,
                                        y=y,
                                        groups=groups,
                                        n_splits=config.n_splits,
                                        rank=int(rank),
                                        c_value=float(c_value),
                                        whiten=bool(whiten),
                                        class_weight_name=str(class_weight_name),
                                        random_state=config.random_state,
                                    )
                                    if not np.isfinite(cv_auc):
                                        continue
                                    if cv_auc > float(best_svd["cv_auroc"]):
                                        best_svd = {
                                            "cv_auroc": float(cv_auc),
                                            "n_valid_folds": int(n_folds),
                                            "family_name": family_name,
                                            "representation": rep,
                                            "rank": int(rank),
                                            "c_value": float(c_value),
                                            "whiten": bool(whiten),
                                            "class_weight": str(class_weight_name),
                                            "feature_names": list(fam_features),
                                            "feature_indices": list(feat_indices),
                                        }

            baseline_auc = float(best_baseline["cv_auroc"]) if np.isfinite(best_baseline["cv_auroc"]) else float("-inf")
            svd_auc = float(best_svd["cv_auroc"]) if np.isfinite(best_svd["cv_auroc"]) else float("-inf")

            if svd_auc > baseline_auc and np.isfinite(svd_auc):
                x_rep_full = _build_representation(
                    x_raw=x_raw,
                    x_rank=x_rank,
                    feature_indices=best_svd["feature_indices"],
                    representation=str(best_svd["representation"]),
                )
                model = _fit_svd_lr_model(
                    x=x_rep_full,
                    y=y,
                    rank=int(best_svd["rank"]),
                    c_value=float(best_svd["c_value"]),
                    whiten=bool(best_svd["whiten"]),
                    class_weight_name=str(best_svd["class_weight"]),
                    random_state=config.random_state,
                )
                if model is None:
                    route = {
                        "route_type": "baseline",
                        "signal_name": best_baseline["signal_name"],
                        "cv_auroc": float(best_baseline["cv_auroc"]),
                        "n_valid_folds": int(best_baseline["n_valid_folds"]),
                        "fallback_reason": "svd_fit_failed_on_full_data",
                    }
                    total_baseline_slots += 1
                else:
                    route = {
                        "route_type": "svd",
                        "cv_auroc": float(best_svd["cv_auroc"]),
                        "n_valid_folds": int(best_svd["n_valid_folds"]),
                        "family_name": str(best_svd["family_name"]),
                        "representation": str(best_svd["representation"]),
                        "rank": int(best_svd["rank"]),
                        "c_value": float(best_svd["c_value"]),
                        "whiten": bool(best_svd["whiten"]),
                        "class_weight": str(best_svd["class_weight"]),
                        "feature_names": list(best_svd["feature_names"]),
                        "feature_indices": list(best_svd["feature_indices"]),
                        "baseline_signal_name": best_baseline["signal_name"],
                        "baseline_cv_auroc": float(best_baseline["cv_auroc"]),
                        "model": model,
                    }
                    total_svd_slots += 1
            else:
                route = {
                    "route_type": "baseline",
                    "signal_name": best_baseline["signal_name"],
                    "cv_auroc": float(best_baseline["cv_auroc"]),
                    "n_valid_folds": int(best_baseline["n_valid_folds"]),
                    "svd_best_cv_auroc": None if not np.isfinite(svd_auc) else float(svd_auc),
                }
                total_baseline_slots += 1

            if route["route_type"] == "svd":
                print(
                    f"  -> SVD win auc={route['cv_auroc']:.4f} "
                    f"(baseline={route['baseline_cv_auroc']:.4f}) family={route['family_name']} rep={route['representation']}"
                )
            else:
                print(
                    f"  -> baseline keep auc={route['cv_auroc']:.4f} signal={route['signal_name']}"
                )

            domain_bundle["routes"].append(route)
            domain_summary["positions"].append({
                "position": float(EARLY_STOP_POSITIONS[pos_i]),
                "route": {
                    k: v for k, v in route.items() if k != "model"
                },
                "num_samples": int(x_raw.shape[0]),
                "num_groups": int(len(np.unique(groups))),
                "positive_rate": float(np.mean(y)) if y.size else float("nan"),
            })

        bundle["domains"][domain] = domain_bundle
        summary["domains"][domain] = domain_summary

    summary["totals"] = {
        "svd_slots": int(total_svd_slots),
        "baseline_slots": int(total_baseline_slots),
        "total_slots": int(total_svd_slots + total_baseline_slots),
    }

    return bundle, summary


def save_earlystop_svd_bundle(bundle: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(bundle, f)


def load_earlystop_svd_bundle(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with p.open("rb") as f:
        bundle = pickle.load(f)
    return bundle


def _problem_tensor(
    reader: CacheReader,
    sample_ids: list[int],
    required_feature_names: Optional[set[str]] = None,
) -> np.ndarray:
    n_runs = len(sample_ids)
    tensor = np.zeros((n_runs, N_POSITIONS, len(FULL_FEATURE_NAMES)), dtype=np.float64)
    for row_i, sample_id in enumerate(sample_ids):
        signal_map = extract_earlystop_signals_for_sample(
            reader,
            int(sample_id),
            required_features=required_feature_names,
        )
        for f_i, f_name in enumerate(FULL_FEATURE_NAMES):
            tensor[row_i, :, f_i] = np.asarray(signal_map[f_name], dtype=np.float64)
    return tensor


def score_cache_entry_earlystop_svd(
    entry: CacheEntry,
    bundle: dict[str, Any],
    max_problems: int | None = None,
) -> dict[str, dict[str, list[float]]]:
    domain = get_domain(entry.dataset_name)
    domain_bundle = bundle["domains"][domain]

    feature_to_idx = {name: i for i, name in enumerate(bundle["feature_names"])}
    required_features: set[str] = set()
    for route in domain_bundle["routes"]:
        if route["route_type"] == "baseline":
            required_features.add(str(route["signal_name"]))
        elif route["route_type"] == "svd":
            required_features.update(str(v) for v in route["feature_names"])

    reader = CacheReader(str(entry.cache_root))
    groups = _parse_meta_groups(entry)

    out: dict[str, dict[str, list[float]]] = {}

    for problem_i, (problem_id, sample_ids_raw) in enumerate(groups):
        if max_problems is not None and problem_i >= max_problems:
            break

        sample_ids = [int(sid) for sid in sample_ids_raw]
        tensor = _problem_tensor(
            reader,
            sample_ids,
            required_feature_names=required_features,
        )

        run_scores = {str(sid): [0.0] * N_POSITIONS for sid in sample_ids}

        for pos_i in range(N_POSITIONS):
            route = domain_bundle["routes"][pos_i]
            x_raw = tensor[:, pos_i, :]
            x_rank = _rank_transform_matrix(x_raw)

            if route["route_type"] == "baseline":
                signal_name = str(route["signal_name"])
                score_col = feature_to_idx[signal_name]
                scores = x_raw[:, score_col]
            elif route["route_type"] == "svd":
                feat_indices = [int(v) for v in route["feature_indices"]]
                rep = str(route["representation"])
                x_rep = _build_representation(
                    x_raw=x_raw,
                    x_rank=x_rank,
                    feature_indices=feat_indices,
                    representation=rep,
                )
                scores = _predict_svd_lr(route["model"], x_rep)
            else:
                raise ValueError(f"Unknown route type: {route['route_type']}")

            for run_idx, sample_id in enumerate(sample_ids):
                run_scores[str(sample_id)][pos_i] = float(scores[run_idx])

        out[str(problem_id)] = run_scores

    return out
