from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from nad.ops.earlystop import _problem_sort_key
from nad.ops.earlystop_svd import _auroc, _group_folds, _rank_transform_matrix


def make_problem_key(dataset_name: str, problem_id: str) -> str:
    return f"{dataset_name}::{problem_id}"


def split_problem_key(problem_key: str) -> tuple[str, str]:
    dataset_name, problem_id = str(problem_key).split("::", 1)
    return dataset_name, problem_id


def sort_problem_key(problem_key: str) -> tuple[str, Any]:
    dataset_name, problem_id = split_problem_key(str(problem_key))
    return dataset_name, _problem_sort_key(problem_id)


@dataclass(frozen=True)
class AnchorTable:
    x_raw: np.ndarray
    x_rank: np.ndarray
    x_full: np.ndarray
    x_token: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    problem_keys: np.ndarray
    sample_ids: np.ndarray


@dataclass(frozen=True)
class LinearHead:
    scaler: StandardScaler
    clf: LogisticRegression
    best_c: float


@dataclass(frozen=True)
class BasisBundle:
    kind: str
    method: str
    n_components: int
    scaler: Optional[StandardScaler] = None
    scaler_a: Optional[StandardScaler] = None
    scaler_b: Optional[StandardScaler] = None
    weights: Optional[np.ndarray] = None
    weights_a: Optional[np.ndarray] = None
    weights_b: Optional[np.ndarray] = None
    shared_weights: Optional[np.ndarray] = None
    stats: Optional[dict[str, Any]] = None


def subset_payload_by_problem_keys(
    payload: dict[str, Any],
    selected_problem_keys: set[str],
) -> Optional[dict[str, Any]]:
    if not selected_problem_keys:
        return None

    tensor_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    sample_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    cv_group_parts: list[np.ndarray] = []
    problem_ids: list[str] = []
    problem_offsets = [0]
    total_samples = 0

    offsets = [int(v) for v in payload["problem_offsets"]]
    all_problem_ids = [str(v) for v in payload["problem_ids"]]
    group_keys_full = np.asarray(payload.get("group_keys", []), dtype=object)
    cv_group_keys_full = np.asarray(payload.get("cv_group_keys", []), dtype=object)

    for problem_idx, problem_id in enumerate(all_problem_ids):
        problem_key = make_problem_key(str(payload["dataset_name"]), problem_id)
        if problem_key not in selected_problem_keys:
            continue
        start = offsets[problem_idx]
        end = offsets[problem_idx + 1]
        width = max(0, end - start)
        if width <= 0:
            continue

        tensor_parts.append(np.asarray(payload["tensor"][start:end], dtype=np.float64))
        label_parts.append(np.asarray(payload["labels"][start:end], dtype=np.int32))
        sample_parts.append(np.asarray(payload["sample_ids"][start:end], dtype=np.int32))
        if group_keys_full.shape[0] >= end:
            group_parts.append(np.asarray(group_keys_full[start:end], dtype=object))
        else:
            group_parts.append(np.asarray([problem_key] * width, dtype=object))
        if cv_group_keys_full.shape[0] >= end:
            cv_group_parts.append(np.asarray(cv_group_keys_full[start:end], dtype=object))
        else:
            cv_group_parts.append(np.asarray([problem_key] * width, dtype=object))
        problem_ids.append(problem_id)
        total_samples += width
        problem_offsets.append(problem_offsets[-1] + width)

    if not tensor_parts:
        return None

    return {
        "cache_key": str(payload["cache_key"]),
        "base_cache_key": str(payload.get("base_cache_key", payload["cache_key"])),
        "source_name": str(payload.get("source_name", "")),
        "dataset_name": str(payload["dataset_name"]),
        "domain": str(payload["domain"]),
        "positions": list(payload["positions"]),
        "tensor": np.concatenate(tensor_parts, axis=0).astype(np.float64, copy=False),
        "labels": np.concatenate(label_parts).astype(np.int32, copy=False),
        "sample_ids": np.concatenate(sample_parts).astype(np.int32, copy=False),
        "group_keys": np.concatenate(group_parts).astype(object, copy=False),
        "cv_group_keys": np.concatenate(cv_group_parts).astype(object, copy=False),
        "problem_ids": problem_ids,
        "problem_offsets": problem_offsets,
        "samples": int(total_samples),
    }


def collect_problem_keys(
    payloads: list[dict[str, Any]],
    *,
    domain: Optional[str] = None,
) -> list[str]:
    keys: list[str] = []
    for payload in payloads:
        if domain is not None and str(payload["domain"]) != domain:
            continue
        dataset_name = str(payload["dataset_name"])
        for problem_id in payload["problem_ids"]:
            keys.append(make_problem_key(dataset_name, str(problem_id)))
    return sorted(set(keys), key=sort_problem_key)


def sample_problem_keys(
    payloads: list[dict[str, Any]],
    *,
    domain: str,
    fraction: float,
    seed: int,
) -> set[str]:
    available = collect_problem_keys(payloads, domain=domain)
    if not available:
        return set()
    n_select = max(1, int(round(float(fraction) * len(available))))
    n_select = min(n_select, len(available))
    rng = np.random.RandomState(int(seed))
    chosen = rng.choice(len(available), size=n_select, replace=False)
    return {available[int(idx)] for idx in chosen.tolist()}


def subset_store_by_problem_keys(
    payloads: list[dict[str, Any]],
    *,
    domain: Optional[str],
    problem_keys: set[str],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    if not problem_keys:
        return selected
    for payload in payloads:
        if domain is not None and str(payload["domain"]) != domain:
            continue
        sub = subset_payload_by_problem_keys(payload, problem_keys)
        if sub is not None and int(sub["samples"]) > 0:
            selected.append(sub)
    return selected


def build_anchor_tables(
    feature_store: list[dict[str, Any]],
    *,
    fixed_feature_indices: list[int],
    token_feature_indices: list[int],
    anchor_position_indices: list[int],
    domain: Optional[str] = None,
) -> dict[int, AnchorTable]:
    n_anchors = len(anchor_position_indices)
    raw_rows: list[list[np.ndarray]] = [[] for _ in range(n_anchors)]
    rank_rows: list[list[np.ndarray]] = [[] for _ in range(n_anchors)]
    full_rows: list[list[np.ndarray]] = [[] for _ in range(n_anchors)]
    token_rows: list[list[np.ndarray]] = [[] for _ in range(n_anchors)]
    ys: list[list[np.ndarray]] = [[] for _ in range(n_anchors)]
    groups: list[list[np.ndarray]] = [[] for _ in range(n_anchors)]
    problem_keys: list[list[np.ndarray]] = [[] for _ in range(n_anchors)]
    sample_ids: list[list[np.ndarray]] = [[] for _ in range(n_anchors)]

    token_idx_arr = np.asarray(token_feature_indices, dtype=np.int32)

    for payload in feature_store:
        if domain is not None and str(payload["domain"]) != domain:
            continue
        tensor = np.asarray(payload["tensor"], dtype=np.float64)
        n_samples = int(tensor.shape[0])
        if n_samples <= 0:
            continue

        y = np.asarray(payload["labels"], dtype=np.int32)
        row_problem_keys_list: list[str] = []
        offsets = [int(v) for v in payload.get("problem_offsets", [0])]
        local_problem_ids = [str(v) for v in payload.get("problem_ids", [])]
        if len(offsets) == len(local_problem_ids) + 1:
            for problem_idx, problem_id in enumerate(local_problem_ids):
                width = max(0, offsets[problem_idx + 1] - offsets[problem_idx])
                row_problem_keys_list.extend(
                    [make_problem_key(str(payload["dataset_name"]), problem_id)] * width
                )
        row_problem_keys = np.asarray(row_problem_keys_list, dtype=object)
        if row_problem_keys.shape[0] != n_samples:
            row_problem_keys = np.asarray(
                [make_problem_key(str(payload["dataset_name"]), str(payload["problem_ids"][0]))] * n_samples,
                dtype=object,
            ) if payload.get("problem_ids") else np.asarray(
                [str(payload["dataset_name"])] * n_samples,
                dtype=object,
            )

        group_key_rows = np.asarray(payload.get("group_keys", row_problem_keys), dtype=object)
        cv_groups = np.asarray(payload.get("cv_group_keys", row_problem_keys), dtype=object)
        if group_key_rows.shape[0] != n_samples:
            group_key_rows = np.asarray(row_problem_keys, dtype=object)
        if cv_groups.shape[0] != n_samples:
            cv_groups = np.asarray(row_problem_keys, dtype=object)

        sample_ids_arr = np.asarray(payload["sample_ids"], dtype=np.int32)

        for anchor_idx, pos_idx in enumerate(anchor_position_indices):
            x_raw = tensor[:, pos_idx, :][:, fixed_feature_indices]
            x_rank = np.zeros_like(x_raw)
            by_problem: dict[Any, list[int]] = {}
            for row_idx, group_key in enumerate(group_key_rows.tolist()):
                by_problem.setdefault(group_key, []).append(row_idx)
            for idx_list in by_problem.values():
                idx_arr = np.asarray(idx_list, dtype=np.int32)
                x_rank[idx_arr] = _rank_transform_matrix(x_raw[idx_arr])

            x_full = np.concatenate([x_raw, x_rank], axis=1)
            x_token = np.concatenate([x_raw[:, token_idx_arr], x_rank[:, token_idx_arr]], axis=1)

            raw_rows[anchor_idx].append(x_raw)
            rank_rows[anchor_idx].append(x_rank)
            full_rows[anchor_idx].append(x_full)
            token_rows[anchor_idx].append(x_token)
            ys[anchor_idx].append(y)
            groups[anchor_idx].append(np.asarray(cv_groups, dtype=object))
            problem_keys[anchor_idx].append(row_problem_keys)
            sample_ids[anchor_idx].append(sample_ids_arr)

    out: dict[int, AnchorTable] = {}
    full_width = 2 * len(fixed_feature_indices)
    token_width = 2 * len(token_feature_indices)
    raw_width = len(fixed_feature_indices)

    for anchor_idx in range(n_anchors):
        if raw_rows[anchor_idx]:
            out[anchor_idx] = AnchorTable(
                x_raw=np.vstack(raw_rows[anchor_idx]).astype(np.float64, copy=False),
                x_rank=np.vstack(rank_rows[anchor_idx]).astype(np.float64, copy=False),
                x_full=np.vstack(full_rows[anchor_idx]).astype(np.float64, copy=False),
                x_token=np.vstack(token_rows[anchor_idx]).astype(np.float64, copy=False),
                y=np.concatenate(ys[anchor_idx]).astype(np.int32, copy=False),
                groups=np.concatenate(groups[anchor_idx]).astype(object, copy=False),
                problem_keys=np.concatenate(problem_keys[anchor_idx]).astype(object, copy=False),
                sample_ids=np.concatenate(sample_ids[anchor_idx]).astype(np.int32, copy=False),
            )
        else:
            out[anchor_idx] = AnchorTable(
                x_raw=np.zeros((0, raw_width), dtype=np.float64),
                x_rank=np.zeros((0, raw_width), dtype=np.float64),
                x_full=np.zeros((0, full_width), dtype=np.float64),
                x_token=np.zeros((0, token_width), dtype=np.float64),
                y=np.zeros((0,), dtype=np.int32),
                groups=np.asarray([], dtype=object),
                problem_keys=np.asarray([], dtype=object),
                sample_ids=np.zeros((0,), dtype=np.int32),
            )
    return out


def orthonormalize_columns(weights: np.ndarray) -> np.ndarray:
    if weights.size == 0:
        return weights
    q, _ = np.linalg.qr(np.asarray(weights, dtype=np.float64))
    return q.astype(np.float64, copy=False)


def _stable_inv_sqrt(cov: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(np.asarray(eigvals, dtype=np.float64), float(eps))
    inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return inv_sqrt.astype(np.float64, copy=False)


def fit_regcca_basis(
    x_a: np.ndarray,
    x_b: np.ndarray,
    *,
    n_components: int,
    reg: float = 0.1,
    same_space: bool = False,
) -> BasisBundle:
    if x_a.shape[0] != x_b.shape[0]:
        raise ValueError("CCA inputs must have the same number of rows")
    if x_a.shape[0] < 4:
        raise ValueError("CCA needs at least 4 paired rows")

    n_components = max(1, min(int(n_components), int(x_a.shape[1]), int(x_b.shape[1])))
    if same_space:
        shared_scaler = StandardScaler(with_mean=True, with_std=True)
        shared_scaler.fit(np.vstack([x_a, x_b]))
        a_scaled = shared_scaler.transform(x_a)
        b_scaled = shared_scaler.transform(x_b)
        scaler_a = shared_scaler
        scaler_b = shared_scaler
    else:
        scaler_a = StandardScaler(with_mean=True, with_std=True)
        scaler_b = StandardScaler(with_mean=True, with_std=True)
        a_scaled = scaler_a.fit_transform(x_a)
        b_scaled = scaler_b.fit_transform(x_b)

    denom = max(1, int(a_scaled.shape[0] - 1))
    cov_aa = (a_scaled.T @ a_scaled) / denom + float(reg) * np.eye(a_scaled.shape[1], dtype=np.float64)
    cov_bb = (b_scaled.T @ b_scaled) / denom + float(reg) * np.eye(b_scaled.shape[1], dtype=np.float64)
    cov_ab = (a_scaled.T @ b_scaled) / denom

    inv_aa = _stable_inv_sqrt(cov_aa)
    inv_bb = _stable_inv_sqrt(cov_bb)
    mtx = inv_aa @ cov_ab @ inv_bb
    u, s, vt = np.linalg.svd(mtx, full_matrices=False)

    weights_a = inv_aa @ u[:, :n_components]
    weights_b = inv_bb @ vt.T[:, :n_components]
    weights_a = orthonormalize_columns(weights_a)
    weights_b = orthonormalize_columns(weights_b)

    shared_weights = None
    if same_space and weights_a.shape == weights_b.shape:
        shared_weights = orthonormalize_columns(0.5 * (weights_a + weights_b))

    return BasisBundle(
        kind="pair",
        method="cca_ridge",
        n_components=int(n_components),
        scaler=scaler_a if same_space else None,
        scaler_a=scaler_a,
        scaler_b=scaler_b,
        weights_a=weights_a,
        weights_b=weights_b,
        shared_weights=shared_weights,
        stats={"canonical_singular_values": np.asarray(s[:n_components], dtype=np.float64).tolist()},
    )


def fit_rrr_basis(
    x_a: np.ndarray,
    x_b: np.ndarray,
    *,
    n_components: int,
    same_space: bool = False,
) -> BasisBundle:
    if x_a.shape[0] != x_b.shape[0]:
        raise ValueError("RRR inputs must have the same number of rows")
    if x_a.shape[0] < 4:
        raise ValueError("RRR needs at least 4 paired rows")

    n_components = max(1, min(int(n_components), int(x_a.shape[1]), int(x_b.shape[1])))
    if same_space:
        shared_scaler = StandardScaler(with_mean=True, with_std=True)
        shared_scaler.fit(np.vstack([x_a, x_b]))
        a_scaled = shared_scaler.transform(x_a)
        b_scaled = shared_scaler.transform(x_b)
        scaler_a = shared_scaler
        scaler_b = shared_scaler
    else:
        scaler_a = StandardScaler(with_mean=True, with_std=True)
        scaler_b = StandardScaler(with_mean=True, with_std=True)
        a_scaled = scaler_a.fit_transform(x_a)
        b_scaled = scaler_b.fit_transform(x_b)

    denom = max(1, int(a_scaled.shape[0] - 1))
    cross = (a_scaled.T @ b_scaled) / denom
    u, s, vt = np.linalg.svd(cross, full_matrices=False)
    weights_a = orthonormalize_columns(u[:, :n_components])
    weights_b = orthonormalize_columns(vt.T[:, :n_components])

    shared_weights = None
    if same_space and weights_a.shape == weights_b.shape:
        shared_weights = orthonormalize_columns(0.5 * (weights_a + weights_b))

    return BasisBundle(
        kind="pair",
        method="rrr_cross_svd",
        n_components=int(n_components),
        scaler=scaler_a if same_space else None,
        scaler_a=scaler_a,
        scaler_b=scaler_b,
        weights_a=weights_a,
        weights_b=weights_b,
        shared_weights=shared_weights,
        stats={"cross_singular_values": np.asarray(s[:n_components], dtype=np.float64).tolist()},
    )


def fit_denoise_svd_basis(
    x: np.ndarray,
    *,
    n_components: int,
    mask_rate: float = 0.15,
    n_augments: int = 2,
    seed: int = 42,
) -> BasisBundle:
    if x.shape[0] < 4:
        raise ValueError("Denoise SVD needs at least 4 rows")
    n_components = max(1, min(int(n_components), int(x.shape[1]), int(x.shape[0] - 1)))

    scaler = StandardScaler(with_mean=True, with_std=True)
    x_scaled = scaler.fit_transform(x)
    rng = np.random.RandomState(int(seed))
    noisy_rows = [x_scaled]
    for _ in range(max(1, int(n_augments))):
        mask = rng.random_sample(x_scaled.shape) < float(mask_rate)
        noisy_rows.append(np.where(mask, 0.0, x_scaled))
    x_aug = np.vstack(noisy_rows)

    svd = TruncatedSVD(n_components=n_components, random_state=int(seed))
    svd.fit(x_aug)
    weights = orthonormalize_columns(svd.components_.T)

    return BasisBundle(
        kind="single",
        method="denoise_svd",
        n_components=int(n_components),
        scaler=scaler,
        weights=weights,
        stats={
            "explained_variance_ratio": np.asarray(
                svd.explained_variance_ratio_,
                dtype=np.float64,
            ).tolist(),
        },
    )


def transform_pair_bundle(bundle: BasisBundle, x_a: np.ndarray, x_b: np.ndarray) -> np.ndarray:
    if bundle.kind != "pair" or bundle.scaler_a is None or bundle.scaler_b is None:
        raise ValueError("Expected a paired basis bundle")
    if bundle.weights_a is None or bundle.weights_b is None:
        raise ValueError("Missing paired basis weights")
    z_a = bundle.scaler_a.transform(x_a) @ bundle.weights_a
    z_b = bundle.scaler_b.transform(x_b) @ bundle.weights_b
    return 0.5 * (z_a + z_b)


def transform_single_bundle(bundle: BasisBundle, x: np.ndarray) -> np.ndarray:
    if bundle.kind != "single":
        raise ValueError("Expected a single-view basis bundle")
    if bundle.scaler is None or bundle.weights is None:
        raise ValueError("Missing single-view basis parameters")
    return bundle.scaler.transform(x) @ bundle.weights


def transform_shared_pair_single_input(bundle: BasisBundle, x: np.ndarray) -> np.ndarray:
    if bundle.kind != "pair" or bundle.scaler is None or bundle.shared_weights is None:
        raise ValueError("Expected a same-space paired basis bundle with shared weights")
    return bundle.scaler.transform(x) @ bundle.shared_weights


def fit_linear_head(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    c_values: tuple[float, ...],
    random_state: int = 42,
    n_cv_splits: int = 3,
) -> Optional[LinearHead]:
    if x.shape[0] < 4 or np.unique(y).shape[0] < 2:
        return None

    scaler = StandardScaler(with_mean=True, with_std=True)
    x_scaled = scaler.fit_transform(x)
    best_c = float(c_values[len(c_values) // 2])

    folds = _group_folds(groups, n_splits=n_cv_splits)
    if len(folds) >= 2:
        best_score = float("-inf")
        for c_val in c_values:
            aucs: list[float] = []
            for train_idx, test_idx in folds:
                y_train = y[train_idx]
                y_test = y[test_idx]
                if np.unique(y_train).shape[0] < 2 or np.unique(y_test).shape[0] < 2:
                    continue
                try:
                    clf = LogisticRegression(
                        C=float(c_val),
                        max_iter=2000,
                        random_state=int(random_state),
                    )
                    clf.fit(x_scaled[train_idx], y_train)
                    auc = _auroc(clf.decision_function(x_scaled[test_idx]), y_test)
                except Exception:
                    continue
                if np.isfinite(auc):
                    aucs.append(float(auc))
            if aucs:
                mean_auc = float(np.mean(aucs))
                if mean_auc > best_score:
                    best_score = mean_auc
                    best_c = float(c_val)

    clf = LogisticRegression(C=best_c, max_iter=2000, random_state=int(random_state))
    try:
        clf.fit(x_scaled, y)
    except Exception:
        return None

    return LinearHead(scaler=scaler, clf=clf, best_c=best_c)


def linear_head_scores(head: LinearHead, x: np.ndarray) -> np.ndarray:
    return np.asarray(head.clf.decision_function(head.scaler.transform(x)), dtype=np.float64)


def sigmoid(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    clipped = np.clip(scores, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def mask_features(x: np.ndarray, *, mask_rate: float, seed: int) -> np.ndarray:
    rng = np.random.RandomState(int(seed))
    mask = rng.random_sample(x.shape) < float(mask_rate)
    return np.where(mask, 0.0, x)


def select_pseudo_labels(
    scores: np.ndarray,
    groups: np.ndarray,
    *,
    threshold: float,
    alt_scores: Optional[np.ndarray] = None,
    max_per_group: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs = sigmoid(scores)
    conf = np.maximum(probs, 1.0 - probs)
    pseudo = (probs >= 0.5).astype(np.int32)

    selected = conf >= float(threshold)
    if alt_scores is not None:
        alt_probs = sigmoid(alt_scores)
        alt_conf = np.maximum(alt_probs, 1.0 - alt_probs)
        alt_pseudo = (alt_probs >= 0.5).astype(np.int32)
        selected &= alt_conf >= float(threshold)
        selected &= alt_pseudo == pseudo

    if not np.any(selected):
        return np.asarray([], dtype=np.int32), np.asarray([], dtype=np.int32), np.asarray([], dtype=np.float64)

    candidate_idx = np.flatnonzero(selected)
    candidate_conf = conf[candidate_idx]
    candidate_groups = np.asarray(groups[candidate_idx], dtype=object)
    keep: list[int] = []

    by_group: dict[Any, list[int]] = {}
    for local_pos, group_key in enumerate(candidate_groups.tolist()):
        by_group.setdefault(group_key, []).append(local_pos)

    for local_positions in by_group.values():
        local_idx = np.asarray(local_positions, dtype=np.int32)
        order = np.argsort(-candidate_conf[local_idx], kind="mergesort")
        take = local_idx[order[:max(1, int(max_per_group))]]
        keep.extend(candidate_idx[take].tolist())

    keep_arr = np.asarray(sorted(set(int(v) for v in keep)), dtype=np.int32)
    return keep_arr, pseudo[keep_arr], conf[keep_arr]


def augment_with_pseudo_labels(
    x_labeled: np.ndarray,
    y_labeled: np.ndarray,
    groups_labeled: np.ndarray,
    x_unlabeled: np.ndarray,
    groups_unlabeled: np.ndarray,
    teacher_head: LinearHead,
    *,
    threshold: float,
    alt_unlabeled: Optional[np.ndarray] = None,
    max_per_group: int = 8,
) -> dict[str, Any]:
    main_scores = linear_head_scores(teacher_head, x_unlabeled)
    alt_scores = None if alt_unlabeled is None else linear_head_scores(teacher_head, alt_unlabeled)
    selected_idx, pseudo_y, pseudo_conf = select_pseudo_labels(
        main_scores,
        groups_unlabeled,
        threshold=float(threshold),
        alt_scores=alt_scores,
        max_per_group=int(max_per_group),
    )

    if selected_idx.size == 0:
        return {
            "x": x_labeled,
            "y": y_labeled,
            "groups": groups_labeled,
            "selected_idx": selected_idx,
            "pseudo_y": pseudo_y,
            "pseudo_conf": pseudo_conf,
        }

    x_aug = np.vstack([x_labeled, x_unlabeled[selected_idx]])
    y_aug = np.concatenate([y_labeled, pseudo_y]).astype(np.int32, copy=False)
    groups_aug = np.concatenate([groups_labeled, groups_unlabeled[selected_idx]]).astype(object, copy=False)
    return {
        "x": x_aug,
        "y": y_aug,
        "groups": groups_aug,
        "selected_idx": selected_idx,
        "pseudo_y": pseudo_y,
        "pseudo_conf": pseudo_conf,
    }


def extract_route_models(
    bundle: dict[str, Any],
    *,
    domain: str,
    early_stop_positions: tuple[float, ...],
    pos_to_anchor_idx: dict[int, int],
) -> dict[int, dict[str, Any]]:
    routes = bundle.get("domains", {}).get(domain, {}).get("routes", [])
    out: dict[int, dict[str, Any]] = {}
    for pos_idx, _ in enumerate(early_stop_positions):
        anchor_idx = int(pos_to_anchor_idx[pos_idx])
        if anchor_idx in out:
            continue
        if pos_idx >= len(routes):
            continue
        route = routes[pos_idx]
        if route.get("route_type") != "svd":
            continue
        model = route.get("model")
        if model is not None:
            out[anchor_idx] = model
    return out


def frozen_svd_transform(route_model: dict[str, Any], x_full: np.ndarray) -> np.ndarray:
    scaler: StandardScaler = route_model["scaler"]
    svd: TruncatedSVD = route_model["svd"]
    z = svd.transform(scaler.transform(x_full))
    if bool(route_model.get("whiten", False)):
        singular_values = np.asarray(svd.singular_values_, dtype=np.float64)
        singular_values = np.where(np.abs(singular_values) < 1e-8, 1.0, singular_values)
        z = z / singular_values
    return np.asarray(z, dtype=np.float64)


def shuffle_pairs_within_groups(
    x_a: np.ndarray,
    x_b: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(int(seed))
    selected_a: list[np.ndarray] = []
    selected_b: list[np.ndarray] = []
    by_group: dict[Any, list[int]] = {}
    for idx, group_key in enumerate(np.asarray(groups, dtype=object).tolist()):
        by_group.setdefault(group_key, []).append(idx)

    for idx_list in by_group.values():
        if len(idx_list) < 2:
            continue
        idx_arr = np.asarray(idx_list, dtype=np.int32)
        perm = rng.permutation(idx_arr.shape[0])
        if np.all(perm == np.arange(idx_arr.shape[0])):
            perm = np.roll(perm, 1)
        selected_a.append(x_a[idx_arr])
        selected_b.append(x_b[idx_arr[perm]])

    if not selected_a:
        return np.zeros((0, x_a.shape[1]), dtype=np.float64), np.zeros((0, x_b.shape[1]), dtype=np.float64)

    return (
        np.vstack(selected_a).astype(np.float64, copy=False),
        np.vstack(selected_b).astype(np.float64, copy=False),
    )


def make_identity_featurizer() -> Callable[[AnchorTable], np.ndarray]:
    return lambda table: table.x_full
