#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import pickle
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from nad.core.selectors.lambda_svm_core import LambdaSVMScorer
from nad.core.selectors.math_svm_impl import augment_math_svm_features, select_math_feature_family
from nad.ops.earlystop import _problem_sort_key
from nad.ops.earlystop_svd import _build_representation, _fit_svd_transform, _rank_transform_matrix
from scripts.export_earlystop_svd_submission import (
    _feature_cache_path as _export_feature_cache_path,
    _load_or_build_feature_store,
)
from scripts.run_earlystop_prefix10_svd_round1 import (
    ANCHOR_POSITIONS,
    DEFAULT_FEATURE_CHUNK_PROBLEMS,
    EXTRACTION_POSITION_INDEX,
    FEATURE_TO_INDEX,
    PREFIX_SAFE_FEATURE_FAMILY_MAP,
)
from scripts.run_math_svm_sweep import _extract_all_problems
from SVDomain.train_es_svd_ms_rr_r1 import _build_holdout_problem_map, _split_feature_store


BESTOFN_MATH_FEATURE_FAMILY = "all_aug"
EARLYSTOP_FEATURE_FAMILY_NAME = "token_plus_traj_fixed"
EARLYSTOP_FEATURE_NAMES = tuple(PREFIX_SAFE_FEATURE_FAMILY_MAP["token_plus_traj"])
EARLYSTOP_REPRESENTATION = "raw+rank"
EARLYSTOP_REQUIRED_FEATURES = set(EARLYSTOP_FEATURE_NAMES)
EARLYSTOP_MATH_SCIENCE_POSITIONS = tuple(float(v) for v in ANCHOR_POSITIONS)
EARLYSTOP_CODING_POSITIONS = (0.70, 0.80, 0.90, 1.00)
EARLYSTOP_ALL_POSITIONS = tuple(
    sorted({float(v) for v in EARLYSTOP_MATH_SCIENCE_POSITIONS + EARLYSTOP_CODING_POSITIONS})
)
EARLYSTOP_POSITION_INDEX = {float(position): idx for idx, position in enumerate(EARLYSTOP_ALL_POSITIONS)}
PAIRWISE_LOGISTIC_PAIRING = (
    "within-problem positive-vs-negative pairs; mirrored diffs; no cross-problem pairs"
)
TREE_RANKING_LOGIC = "within-problem query groups with relevance labels {0,1}; no cross-problem mixing"
INFORMATIVE_GROUP_FILTER = (
    "drop within-problem groups that do not contain both positive and non-positive labels"
)


@dataclass(frozen=True)
class RankingSlice:
    task_key: str
    task_family: str
    domain: str
    dataset_scope: str
    anchor_pct: Optional[int]
    feature_family: str
    representation: str
    x: np.ndarray
    y: np.ndarray
    rank_groups: np.ndarray
    cv_groups: np.ndarray


@dataclass(frozen=True)
class CandidateSpec:
    model_name: str
    objective: str
    basis_mode: str
    pairing_logic: str
    params: dict[str, Any]


class IdentityBasis:
    def fit(self, x: np.ndarray) -> "IdentityBasis":
        _ = np.asarray(x, dtype=np.float64)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64)

    def meta(self) -> dict[str, Any]:
        return {"basis_mode": "raw_features", "basis_rank": None, "basis_whiten": None}


class FrozenZBasis:
    def __init__(self, *, rank: int, whiten: bool, random_state: int) -> None:
        self.rank = int(rank)
        self.whiten = bool(whiten)
        self.random_state = int(random_state)
        self.scaler: Optional[StandardScaler] = None
        self.svd: Any = None
        self.effective_rank: Optional[int] = None

    def fit(self, x: np.ndarray) -> "FrozenZBasis":
        transform = _fit_svd_transform(
            np.asarray(x, dtype=np.float64),
            rank=int(self.rank),
            whiten=bool(self.whiten),
            random_state=int(self.random_state),
        )
        if transform is None:
            raise ValueError(
                f"FrozenZBasis cannot fit rank={self.rank} on matrix shape={np.asarray(x).shape}"
            )
        self.scaler = transform["scaler"]
        self.svd = transform["svd"]
        self.effective_rank = int(transform["rank"])
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.scaler is None or self.svd is None:
            raise RuntimeError("FrozenZBasis.fit() must be called before transform()")
        x_arr = np.asarray(x, dtype=np.float64)
        z = self.svd.transform(self.scaler.transform(x_arr))
        if self.whiten:
            singular_values = np.asarray(self.svd.singular_values_, dtype=np.float64)
            singular_values = np.where(np.abs(singular_values) < 1e-8, 1.0, singular_values)
            z = z / singular_values
        return np.asarray(z, dtype=np.float64)

    def meta(self) -> dict[str, Any]:
        return {
            "basis_mode": "frozen_z",
            "basis_rank": None if self.effective_rank is None else int(self.effective_rank),
            "basis_whiten": bool(self.whiten),
        }


class PointwiseLogisticRanker:
    def __init__(
        self,
        *,
        c_value: float,
        class_weight: str | None,
        basis: IdentityBasis | FrozenZBasis,
        random_state: int,
    ) -> None:
        self.c_value = float(c_value)
        self.class_weight = None if class_weight in (None, "", "none", "None") else str(class_weight)
        self.basis = basis
        self.random_state = int(random_state)
        self.pipeline: Optional[Pipeline] = None

    def fit(self, x: np.ndarray, y: np.ndarray, rank_groups: np.ndarray) -> None:
        _ = rank_groups
        z = self.basis.fit(x).transform(x)
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=self.c_value,
                    class_weight=self.class_weight,
                    fit_intercept=True,
                    max_iter=4000,
                    random_state=self.random_state,
                    solver="lbfgs",
                ),
            ),
        ])
        self.pipeline.fit(z, np.asarray(y, dtype=np.int32))

    def score(self, x: np.ndarray) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("PointwiseLogisticRanker.fit() must be called before score()")
        z = self.basis.transform(x)
        if hasattr(self.pipeline, "decision_function"):
            return np.asarray(self.pipeline.decision_function(z), dtype=np.float64).reshape(-1)
        return np.asarray(self.pipeline.predict_proba(z)[:, 1], dtype=np.float64).reshape(-1)

    def meta(self) -> dict[str, Any]:
        return {"c_value": float(self.c_value), "class_weight": self.class_weight, **self.basis.meta()}


class PairwiseLogisticRanker:
    def __init__(
        self,
        *,
        c_value: float,
        basis: IdentityBasis | FrozenZBasis,
        random_state: int,
    ) -> None:
        self.c_value = float(c_value)
        self.basis = basis
        self.random_state = int(random_state)
        self.pipeline: Optional[Pipeline] = None

    def fit(self, x: np.ndarray, y: np.ndarray, rank_groups: np.ndarray) -> None:
        z = self.basis.fit(x).transform(x)
        x_pairs, y_pairs = _build_pairwise_logistic_examples(z, y, rank_groups)
        if x_pairs.shape[0] <= 0:
            raise ValueError("PairwiseLogisticRanker.fit() found no valid positive-vs-negative pairs")
        self.pipeline = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            (
                "clf",
                LogisticRegression(
                    C=self.c_value,
                    fit_intercept=False,
                    max_iter=4000,
                    random_state=self.random_state,
                    solver="lbfgs",
                ),
            ),
        ])
        self.pipeline.fit(x_pairs, y_pairs)

    def score(self, x: np.ndarray) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("PairwiseLogisticRanker.fit() must be called before score()")
        z = self.basis.transform(x)
        return np.asarray(self.pipeline.decision_function(z), dtype=np.float64).reshape(-1)

    def meta(self) -> dict[str, Any]:
        return {"c_value": float(self.c_value), **self.basis.meta()}


class PairwiseLinearSVMRanker:
    def __init__(
        self,
        *,
        c_value: float,
        loss: str,
        basis: IdentityBasis | FrozenZBasis,
        random_state: int,
    ) -> None:
        self.c_value = float(c_value)
        self.loss = str(loss)
        self.basis = basis
        self.random_state = int(random_state)
        self.scorer: Optional[LambdaSVMScorer] = None

    def fit(self, x: np.ndarray, y: np.ndarray, rank_groups: np.ndarray) -> None:
        _ = self.random_state
        z = self.basis.fit(x).transform(x)
        x_pairs, y_pairs = _build_pairwise_hinge_examples(z, y, rank_groups)
        if x_pairs.shape[0] <= 0:
            raise ValueError("PairwiseLinearSVMRanker.fit() found no valid positive-vs-negative pairs")
        scorer = LambdaSVMScorer(
            C=float(self.c_value),
            loss=str(self.loss),
            fit_intercept=False,
            backend="utility",
            dual="auto",
            max_iter=100000,
            tol=1e-4,
        )
        scorer.fit(x_pairs, y_pairs)
        self.scorer = scorer

    def score(self, x: np.ndarray) -> np.ndarray:
        if self.scorer is None:
            raise RuntimeError("PairwiseLinearSVMRanker.fit() must be called before score()")
        z = self.basis.transform(x)
        return np.asarray(self.scorer.score_group(z), dtype=np.float64).reshape(-1)

    def meta(self) -> dict[str, Any]:
        return {"c_value": float(self.c_value), "loss": str(self.loss), **self.basis.meta()}


class XGBRankerWrapper:
    def __init__(self, *, params: dict[str, Any], random_state: int) -> None:
        self.params = dict(params)
        self.random_state = int(random_state)
        self.model: Any = None

    def fit(self, x: np.ndarray, y: np.ndarray, rank_groups: np.ndarray) -> None:
        xgb = _import_xgboost()
        ordered = _order_rows_by_group(rank_groups)
        x_ord = np.asarray(x[ordered], dtype=np.float32)
        y_ord = np.asarray(y[ordered], dtype=np.int32)
        g_ord = np.asarray(rank_groups[ordered], dtype=object)
        group_sizes = _group_sizes(g_ord)
        if not group_sizes:
            raise ValueError("XGBRankerWrapper.fit() requires at least one non-empty group")
        model = xgb.XGBRanker(
            objective="rank:ndcg",
            eval_metric="ndcg@3",
            tree_method="hist",
            learning_rate=float(self.params["learning_rate"]),
            max_depth=int(self.params["max_depth"]),
            n_estimators=int(self.params["n_estimators"]),
            min_child_weight=float(self.params["min_child_weight"]),
            subsample=float(self.params["subsample"]),
            colsample_bytree=float(self.params["colsample_bytree"]),
            reg_lambda=float(self.params["reg_lambda"]),
            random_state=int(self.random_state),
            n_jobs=1,
        )
        model.fit(x_ord, y_ord, group=group_sizes, verbose=False)
        self.model = model

    def score(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("XGBRankerWrapper.fit() must be called before score()")
        return np.asarray(self.model.predict(np.asarray(x, dtype=np.float32)), dtype=np.float64).reshape(-1)

    def meta(self) -> dict[str, Any]:
        return {
            "learning_rate": float(self.params["learning_rate"]),
            "max_depth": int(self.params["max_depth"]),
            "n_estimators": int(self.params["n_estimators"]),
            "min_child_weight": float(self.params["min_child_weight"]),
            "subsample": float(self.params["subsample"]),
            "colsample_bytree": float(self.params["colsample_bytree"]),
            "reg_lambda": float(self.params["reg_lambda"]),
            "basis_mode": "raw_features",
            "basis_rank": None,
            "basis_whiten": None,
        }


def _stable_hash(text: str) -> int:
    digest = hashlib.sha1(str(text).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _coerce_metric(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _metric_key(value: Any) -> float:
    numeric = _coerce_metric(value)
    return float("-inf") if numeric is None else float(numeric)


def _safe_mean(values: list[float]) -> Optional[float]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return None
    return float(np.mean(np.asarray(finite, dtype=np.float64)))


def _group_folds(groups: np.ndarray, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_groups = np.unique(np.asarray(groups, dtype=object))
    if unique_groups.shape[0] < 2:
        return []
    splits = min(int(n_splits), int(unique_groups.shape[0]))
    if splits < 2:
        return []
    gkf = GroupKFold(n_splits=splits)
    dummy = np.zeros((len(groups), 1), dtype=np.float64)
    return list(gkf.split(dummy, groups=np.asarray(groups, dtype=object)))


def _order_rows_by_group(groups: np.ndarray) -> np.ndarray:
    groups_arr = np.asarray(groups, dtype=object)
    by_group: dict[str, list[int]] = {}
    order: list[str] = []
    for idx, group in enumerate(groups_arr.tolist()):
        key = str(group)
        if key not in by_group:
            by_group[key] = []
            order.append(key)
        by_group[key].append(int(idx))
    return np.asarray([row for key in order for row in by_group[key]], dtype=np.int64)


def _group_sizes(groups: np.ndarray) -> list[int]:
    groups_arr = np.asarray(groups, dtype=object)
    if groups_arr.size <= 0:
        return []
    sizes: list[int] = []
    last = str(groups_arr[0])
    count = 0
    for group in groups_arr.tolist():
        key = str(group)
        if key != last:
            sizes.append(int(count))
            last = key
            count = 1
        else:
            count += 1
    if count > 0:
        sizes.append(int(count))
    return sizes


def _build_pairwise_logistic_examples(
    x: np.ndarray,
    y: np.ndarray,
    rank_groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.int32).reshape(-1)
    g_arr = np.asarray(rank_groups, dtype=object).reshape(-1)
    n_feat = x_arr.shape[1] if x_arr.ndim == 2 else 0
    if x_arr.ndim != 2:
        raise ValueError(f"Expected 2D x matrix, got shape={x_arr.shape}")
    if x_arr.shape[0] != y_arr.shape[0] or y_arr.shape[0] != g_arr.shape[0]:
        raise ValueError("x, y, rank_groups must have matching row counts")

    pair_x_parts: list[np.ndarray] = []
    pair_y_parts: list[np.ndarray] = []
    by_group: dict[str, list[int]] = {}
    for idx, group in enumerate(g_arr.tolist()):
        by_group.setdefault(str(group), []).append(int(idx))

    for idxs in by_group.values():
        x_g = x_arr[idxs]
        y_g = y_arr[idxs]
        pos_idx = np.where(y_g > 0)[0]
        neg_idx = np.where(y_g <= 0)[0]
        if pos_idx.size == 0 or neg_idx.size == 0:
            continue
        diffs_pos = (x_g[pos_idx][:, None, :] - x_g[neg_idx][None, :, :]).reshape(-1, n_feat)
        diffs_neg = (x_g[neg_idx][None, :, :] - x_g[pos_idx][:, None, :]).reshape(-1, n_feat)
        pair_x_parts.append(np.concatenate([diffs_pos, diffs_neg], axis=0))
        pair_y_parts.append(
            np.concatenate([
                np.ones(diffs_pos.shape[0], dtype=np.int32),
                np.zeros(diffs_neg.shape[0], dtype=np.int32),
            ])
        )

    if not pair_x_parts:
        return np.zeros((0, n_feat), dtype=np.float64), np.zeros((0,), dtype=np.int32)
    return (
        np.concatenate(pair_x_parts, axis=0).astype(np.float64, copy=False),
        np.concatenate(pair_y_parts, axis=0).astype(np.int32, copy=False),
    )


def _build_pairwise_hinge_examples(
    x: np.ndarray,
    y: np.ndarray,
    rank_groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.int32).reshape(-1)
    g_arr = np.asarray(rank_groups, dtype=object).reshape(-1)
    n_feat = x_arr.shape[1] if x_arr.ndim == 2 else 0
    pair_x_parts: list[np.ndarray] = []
    pair_y_parts: list[np.ndarray] = []
    by_group: dict[str, list[int]] = {}
    for idx, group in enumerate(g_arr.tolist()):
        by_group.setdefault(str(group), []).append(int(idx))
    for idxs in by_group.values():
        x_g = x_arr[idxs]
        y_g = y_arr[idxs]
        pos_idx = np.where(y_g > 0)[0]
        neg_idx = np.where(y_g <= 0)[0]
        if pos_idx.size == 0 or neg_idx.size == 0:
            continue
        diffs_pos = (x_g[pos_idx][:, None, :] - x_g[neg_idx][None, :, :]).reshape(-1, n_feat)
        diffs_neg = (x_g[neg_idx][None, :, :] - x_g[pos_idx][:, None, :]).reshape(-1, n_feat)
        pair_x_parts.append(np.concatenate([diffs_pos, diffs_neg], axis=0))
        pair_y_parts.append(
            np.concatenate([
                np.ones(diffs_pos.shape[0], dtype=np.int32),
                -np.ones(diffs_neg.shape[0], dtype=np.int32),
            ])
        )
    if not pair_x_parts:
        return np.zeros((0, n_feat), dtype=np.float64), np.zeros((0,), dtype=np.int32)
    return (
        np.concatenate(pair_x_parts, axis=0).astype(np.float64, copy=False),
        np.concatenate(pair_y_parts, axis=0).astype(np.int32, copy=False),
    )


def _retain_informative_groups(slc: RankingSlice) -> RankingSlice:
    keep_mask = np.zeros(slc.y.shape[0], dtype=bool)
    groups_arr = np.asarray(slc.rank_groups, dtype=object).reshape(-1)
    labels_arr = np.asarray(slc.y, dtype=np.int32).reshape(-1)
    by_group: dict[str, list[int]] = {}
    for idx, group in enumerate(groups_arr.tolist()):
        by_group.setdefault(str(group), []).append(int(idx))
    for idxs in by_group.values():
        labels_g = labels_arr[idxs]
        has_pos = bool(np.any(labels_g > 0))
        has_nonpos = bool(np.any(labels_g <= 0))
        if has_pos and has_nonpos:
            keep_mask[np.asarray(idxs, dtype=np.int64)] = True
    if not np.any(keep_mask):
        raise ValueError(
            f"{slc.task_key} has no informative within-problem groups after applying filter: "
            f"{INFORMATIVE_GROUP_FILTER}"
        )
    return RankingSlice(
        task_key=str(slc.task_key),
        task_family=str(slc.task_family),
        domain=str(slc.domain),
        dataset_scope=str(slc.dataset_scope),
        anchor_pct=slc.anchor_pct,
        feature_family=str(slc.feature_family),
        representation=str(slc.representation),
        x=np.asarray(slc.x[keep_mask], dtype=np.float64),
        y=np.asarray(labels_arr[keep_mask], dtype=np.int32),
        rank_groups=np.asarray(groups_arr[keep_mask], dtype=object),
        cv_groups=np.asarray(slc.cv_groups[keep_mask], dtype=object),
    )


def _dcg_at_k(labels: np.ndarray, k: int) -> float:
    labels_arr = np.asarray(labels, dtype=np.float64).reshape(-1)
    if labels_arr.size <= 0:
        return 0.0
    gains = labels_arr[:k]
    discounts = np.log2(np.arange(gains.size, dtype=np.float64) + 2.0)
    return float(np.sum(gains / discounts))


def _ranking_metrics(
    *,
    scores: np.ndarray,
    labels: np.ndarray,
    rank_groups: np.ndarray,
) -> dict[str, Any]:
    scores_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels_arr = np.asarray(labels, dtype=np.int32).reshape(-1)
    groups_arr = np.asarray(rank_groups, dtype=object).reshape(-1)
    if scores_arr.shape[0] != labels_arr.shape[0] or labels_arr.shape[0] != groups_arr.shape[0]:
        raise ValueError("scores, labels, rank_groups must have matching row counts")

    by_group: dict[str, list[int]] = {}
    group_order: list[str] = []
    for idx, group in enumerate(groups_arr.tolist()):
        key = str(group)
        if key not in by_group:
            by_group[key] = []
            group_order.append(key)
        by_group[key].append(int(idx))

    hit1_total = 0.0
    hit3_total = 0.0
    mrr_total = 0.0
    ndcg3_total = 0.0
    pairwise_num = 0.0
    pairwise_den = 0.0
    all_scores: list[float] = []
    all_labels: list[int] = []

    for group in group_order:
        idxs = np.asarray(by_group[group], dtype=np.int64)
        scores_g = scores_arr[idxs]
        labels_g = labels_arr[idxs]
        order = np.argsort(-scores_g, kind="mergesort")
        ranked_labels = labels_g[order]
        hit1_total += float(ranked_labels[0] > 0) if ranked_labels.size > 0 else 0.0
        hit3_total += float(np.any(ranked_labels[: min(3, ranked_labels.size)] > 0)) if ranked_labels.size > 0 else 0.0
        pos_rank = np.where(ranked_labels > 0)[0]
        if pos_rank.size > 0:
            mrr_total += 1.0 / float(int(pos_rank[0]) + 1)
        ideal = np.sort(labels_g)[::-1]
        dcg = _dcg_at_k(ranked_labels, 3)
        idcg = _dcg_at_k(ideal, 3)
        ndcg3_total += float(dcg / idcg) if idcg > 0.0 else 0.0

        pos_scores = scores_g[labels_g > 0]
        neg_scores = scores_g[labels_g <= 0]
        if pos_scores.size > 0 and neg_scores.size > 0:
            diff = pos_scores[:, None] - neg_scores[None, :]
            pairwise_num += float((diff > 0.0).sum()) + 0.5 * float((diff == 0.0).sum())
            pairwise_den += float(diff.size)

        all_scores.extend(scores_g.tolist())
        all_labels.extend(labels_g.tolist())

    all_scores_arr = np.asarray(all_scores, dtype=np.float64)
    all_labels_arr = np.asarray(all_labels, dtype=np.int32)
    auroc: Optional[float] = None
    if all_labels_arr.size > 0 and np.unique(all_labels_arr).size >= 2:
        try:
            from sklearn.metrics import roc_auc_score

            auroc = float(roc_auc_score(all_labels_arr, all_scores_arr))
        except Exception:
            auroc = None

    selacc10 = None
    if all_labels_arr.size > 0:
        topk = max(1, int(math.ceil(0.10 * all_labels_arr.size)))
        order = np.argsort(-all_scores_arr, kind="mergesort")
        selacc10 = float(np.mean(all_labels_arr[order[:topk]]))

    n_groups = len(group_order)
    return {
        "pairwise_acc": None if pairwise_den <= 0 else float(pairwise_num / pairwise_den),
        "hit@1": float(hit1_total / n_groups) if n_groups else None,
        "hit@3": float(hit3_total / n_groups) if n_groups else None,
        "mrr": float(mrr_total / n_groups) if n_groups else None,
        "ndcg@3": float(ndcg3_total / n_groups) if n_groups else None,
        "auroc": auroc,
        "selacc@10%": selacc10,
        "n_problems": int(n_groups),
        "n_samples": int(all_labels_arr.size),
    }


def _mean_metrics(metrics_list: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("pairwise_acc", "hit@1", "hit@3", "mrr", "ndcg@3", "auroc", "selacc@10%")
    out: dict[str, Any] = {key: _safe_mean([float(item[key]) for item in metrics_list if _coerce_metric(item.get(key)) is not None]) for key in keys}
    out["n_valid_folds"] = int(len(metrics_list))
    return out


def _valid_svd_ranks(x_dim: int) -> tuple[int, ...]:
    max_rank = max(1, int(x_dim) - 1)
    candidates = [8, 16]
    ranks = sorted({rank for rank in candidates if int(rank) <= max_rank})
    if not ranks:
        return (int(max_rank),)
    return tuple(int(rank) for rank in ranks)


def _candidate_specs_for_slice(
    slc: RankingSlice,
    *,
    enable_tree_ranker: bool,
) -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []

    for c_value in (0.20, 1.0):
        for class_weight in ("balanced", "none"):
            specs.append(
                CandidateSpec(
                    model_name="pointwise_logistic",
                    objective="pointwise_logistic",
                    basis_mode="raw_features",
                    pairing_logic="pointwise labels only",
                    params={"c_value": float(c_value), "class_weight": str(class_weight)},
                )
            )

    for c_value in (0.30, 1.0):
        specs.append(
            CandidateSpec(
                model_name="pairwise_logistic",
                objective="pairwise_logistic_diff",
                basis_mode="raw_features",
                pairing_logic=PAIRWISE_LOGISTIC_PAIRING,
                params={"c_value": float(c_value)},
            )
        )
        specs.append(
            CandidateSpec(
                model_name="pairwise_linear_svm",
                objective="pairwise_linear_svm",
                basis_mode="raw_features",
                pairing_logic=PAIRWISE_LOGISTIC_PAIRING,
                params={"c_value": float(c_value), "loss": "squared_hinge"},
            )
        )

    if enable_tree_ranker:
        specs.append(
            CandidateSpec(
                model_name="xgboost_rank_ndcg",
                objective="rank:ndcg",
                basis_mode="raw_features",
                pairing_logic=TREE_RANKING_LOGIC,
                params={
                    "learning_rate": 0.05,
                    "max_depth": 4,
                    "n_estimators": 160,
                    "min_child_weight": 2.0,
                    "subsample": 0.9,
                    "colsample_bytree": 0.8,
                    "reg_lambda": 2.0,
                },
            )
        )

    for rank in _valid_svd_ranks(int(slc.x.shape[1])):
        for whiten in (False, True):
            for c_value in (1.0,):
                specs.append(
                    CandidateSpec(
                        model_name="pointwise_logistic_frozen_z",
                        objective="pointwise_logistic",
                        basis_mode="frozen_z",
                        pairing_logic="pointwise labels only",
                        params={
                            "rank": int(rank),
                            "whiten": bool(whiten),
                            "c_value": float(c_value),
                            "class_weight": "balanced",
                        },
                    )
                )
                specs.append(
                    CandidateSpec(
                        model_name="pairwise_logistic_frozen_z",
                        objective="pairwise_logistic_diff",
                        basis_mode="frozen_z",
                        pairing_logic=PAIRWISE_LOGISTIC_PAIRING,
                        params={"rank": int(rank), "whiten": bool(whiten), "c_value": float(c_value)},
                    )
                )

    return specs


def _make_model(spec: CandidateSpec, *, random_state: int):
    basis: IdentityBasis | FrozenZBasis
    if spec.basis_mode == "frozen_z":
        basis = FrozenZBasis(
            rank=int(spec.params["rank"]),
            whiten=bool(spec.params["whiten"]),
            random_state=int(random_state),
        )
    else:
        basis = IdentityBasis()

    if spec.model_name in {"pointwise_logistic", "pointwise_logistic_frozen_z"}:
        return PointwiseLogisticRanker(
            c_value=float(spec.params["c_value"]),
            class_weight=str(spec.params.get("class_weight", "balanced")),
            basis=basis,
            random_state=int(random_state),
        )
    if spec.model_name in {"pairwise_logistic", "pairwise_logistic_frozen_z"}:
        return PairwiseLogisticRanker(
            c_value=float(spec.params["c_value"]),
            basis=basis,
            random_state=int(random_state),
        )
    if spec.model_name == "pairwise_linear_svm":
        return PairwiseLinearSVMRanker(
            c_value=float(spec.params["c_value"]),
            loss=str(spec.params["loss"]),
            basis=basis,
            random_state=int(random_state),
        )
    if spec.model_name == "xgboost_rank_ndcg":
        return XGBRankerWrapper(params=spec.params, random_state=int(random_state))
    raise ValueError(f"Unsupported candidate model: {spec.model_name}")


def _evaluate_candidate_cv(
    slc: RankingSlice,
    spec: CandidateSpec,
    *,
    n_splits: int,
    random_state: int,
) -> Optional[dict[str, Any]]:
    folds = _group_folds(slc.cv_groups, n_splits=int(n_splits))
    if not folds:
        return None
    fold_metrics: list[dict[str, Any]] = []
    for train_idx, test_idx in folds:
        y_train = np.asarray(slc.y[train_idx], dtype=np.int32)
        y_test = np.asarray(slc.y[test_idx], dtype=np.int32)
        if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
            continue
        model = _make_model(spec, random_state=int(random_state))
        try:
            model.fit(slc.x[train_idx], y_train, slc.rank_groups[train_idx])
            scores = model.score(slc.x[test_idx])
        except Exception:
            continue
        fold_metrics.append(
            _ranking_metrics(scores=scores, labels=y_test, rank_groups=slc.rank_groups[test_idx])
        )
    if not fold_metrics:
        return None
    return _mean_metrics(fold_metrics)


def _fit_and_evaluate_holdout(
    train_slice: RankingSlice,
    holdout_slice: RankingSlice,
    spec: CandidateSpec,
    *,
    random_state: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model = _make_model(spec, random_state=int(random_state))
    model.fit(train_slice.x, train_slice.y, train_slice.rank_groups)
    holdout_scores = model.score(holdout_slice.x)
    holdout_metrics = _ranking_metrics(
        scores=holdout_scores,
        labels=holdout_slice.y,
        rank_groups=holdout_slice.rank_groups,
    )
    holdout_metrics["stop_acc"] = (
        holdout_metrics["hit@1"] if train_slice.task_family == "earlystop" else None
    )
    return model.meta(), holdout_metrics


def _select_best_candidate(
    candidates: list[tuple[CandidateSpec, dict[str, Any]]],
) -> tuple[CandidateSpec, dict[str, Any]]:
    if not candidates:
        raise ValueError("No valid candidate rows were produced")
    ordered = sorted(
        candidates,
        key=lambda item: (
            _metric_key(item[1].get("pairwise_acc")),
            _metric_key(item[1].get("hit@1")),
            _metric_key(item[1].get("ndcg@3")),
            _metric_key(item[1].get("mrr")),
        ),
        reverse=True,
    )
    return ordered[0]


def _qualify_feature_store(feature_store: list[dict[str, Any]], source_name: str) -> list[dict[str, Any]]:
    qualified: list[dict[str, Any]] = []
    for payload in feature_store:
        item = dict(payload)
        item["source_name"] = str(source_name)
        item["base_cache_key"] = str(payload["cache_key"])
        item["cache_key"] = f"{source_name}/{payload['cache_key']}"
        qualified.append(item)
    return qualified


def _build_earlystop_position_slice(
    *,
    task_key: str,
    domain: str,
    dataset_scope: str,
    anchor_pct: int,
    feature_store: list[dict[str, Any]],
    feature_indices: list[int],
) -> RankingSlice:
    pos_value = float(anchor_pct) / 100.0
    pos_idx = EARLYSTOP_POSITION_INDEX[pos_value]
    x_raw_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    rank_group_parts: list[np.ndarray] = []
    cv_group_parts: list[np.ndarray] = []

    for payload in feature_store:
        if str(payload["domain"]) != str(domain):
            continue
        position_index = {
            float(position): idx for idx, position in enumerate(payload.get("positions", EARLYSTOP_ALL_POSITIONS))
        }
        if pos_value not in position_index:
            raise ValueError(
                f"Payload {payload['cache_key']} does not provide anchor {pos_value:.2f}; "
                f"available={sorted(position_index.keys())}"
            )
        pos_idx = int(position_index[pos_value])
        tensor = np.asarray(payload["tensor"], dtype=np.float64)
        if tensor.shape[0] <= 0:
            continue
        x_raw_parts.append(np.asarray(tensor[:, pos_idx, :][:, feature_indices], dtype=np.float64))
        y_parts.append(np.asarray(payload["labels"], dtype=np.int32))
        rank_group_parts.append(np.asarray(payload["group_keys"], dtype=object))
        cv_group_parts.append(np.asarray(payload["cv_group_keys"], dtype=object))

    if not x_raw_parts:
        raise ValueError(f"No rows found for earlystop domain={domain} anchor={anchor_pct}%")

    x_raw = np.vstack(x_raw_parts).astype(np.float64, copy=False)
    y = np.concatenate(y_parts).astype(np.int32, copy=False)
    rank_groups = np.concatenate(rank_group_parts).astype(object, copy=False)
    cv_groups = np.concatenate(cv_group_parts).astype(object, copy=False)

    x_rank = np.zeros_like(x_raw)
    by_rank_group: dict[str, list[int]] = {}
    for row_idx, group_key in enumerate(rank_groups.tolist()):
        by_rank_group.setdefault(str(group_key), []).append(int(row_idx))
    for idxs in by_rank_group.values():
        x_rank[idxs] = _rank_transform_matrix(x_raw[idxs])

    x_rep = _build_representation(
        x_raw=x_raw,
        x_rank=x_rank,
        feature_indices=list(range(x_raw.shape[1])),
        representation=EARLYSTOP_REPRESENTATION,
    )
    return RankingSlice(
        task_key=str(task_key),
        task_family="earlystop",
        domain=str(domain),
        dataset_scope=str(dataset_scope),
        anchor_pct=int(anchor_pct),
        feature_family=EARLYSTOP_FEATURE_FAMILY_NAME,
        representation=EARLYSTOP_REPRESENTATION,
        x=np.asarray(x_rep, dtype=np.float64),
        y=y,
        rank_groups=rank_groups,
        cv_groups=cv_groups,
    )


def _split_bestofn_math_holdout(
    problems_by_source: list[tuple[str, list[Any]]],
    *,
    holdout_split: float,
    split_seed: int,
) -> tuple[list[tuple[str, Any]], list[tuple[str, Any]], dict[str, Any]]:
    by_dataset: dict[str, set[str]] = {}
    for _, problems in problems_by_source:
        for problem in problems:
            raw_pid = str(problem.problem_id).split(":", 1)[1]
            by_dataset.setdefault(str(problem.dataset), set()).add(str(raw_pid))

    holdout_map: dict[str, set[str]] = {}
    holdout_summary: dict[str, Any] = {}
    for dataset_name in sorted(by_dataset.keys()):
        ordered_problem_ids = sorted(by_dataset[dataset_name], key=_problem_sort_key)
        if len(ordered_problem_ids) < 2:
            holdout_ids: set[str] = set()
        else:
            rng = np.random.RandomState(int(split_seed) + _stable_hash(dataset_name))
            order = rng.permutation(len(ordered_problem_ids))
            n_holdout = int(round(len(ordered_problem_ids) * float(holdout_split)))
            n_holdout = max(1, n_holdout)
            n_holdout = min(len(ordered_problem_ids) - 1, n_holdout)
            holdout_ids = {ordered_problem_ids[int(idx)] for idx in order[:n_holdout].tolist()}
        holdout_map[dataset_name] = holdout_ids
        holdout_summary[dataset_name] = {
            "total_unique_problem_ids": int(len(ordered_problem_ids)),
            "holdout_unique_problem_ids": int(len(holdout_ids)),
            "train_unique_problem_ids": int(len(ordered_problem_ids) - len(holdout_ids)),
            "holdout_problem_ids": sorted(holdout_ids, key=_problem_sort_key),
        }

    train_rows: list[tuple[str, Any]] = []
    holdout_rows: list[tuple[str, Any]] = []
    for source_name, problems in problems_by_source:
        for problem in problems:
            raw_pid = str(problem.problem_id).split(":", 1)[1]
            if raw_pid in holdout_map.get(str(problem.dataset), set()):
                holdout_rows.append((source_name, problem))
            else:
                train_rows.append((source_name, problem))
    return train_rows, holdout_rows, holdout_summary


def _build_bestofn_math_slice(
    rows: list[tuple[str, Any]],
    *,
    task_key: str,
) -> RankingSlice:
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    rank_group_parts: list[np.ndarray] = []
    cv_group_parts: list[np.ndarray] = []
    for source_name, problem in rows:
        x_sel = select_math_feature_family(
            augment_math_svm_features(np.asarray(problem.X_all, dtype=np.float64)),
            BESTOFN_MATH_FEATURE_FAMILY,
        )
        n = int(x_sel.shape[0])
        x_parts.append(np.asarray(x_sel, dtype=np.float64))
        y_parts.append(np.asarray(problem.labels, dtype=np.int32))
        rank_group_parts.append(np.asarray([f"{source_name}::{problem.problem_id}"] * n, dtype=object))
        cv_group_parts.append(np.asarray([str(problem.problem_id)] * n, dtype=object))

    if not x_parts:
        raise ValueError("No best-of-N math rows were provided")

    return RankingSlice(
        task_key=str(task_key),
        task_family="bestofn",
        domain="math",
        dataset_scope="bestofn_math",
        anchor_pct=None,
        feature_family=BESTOFN_MATH_FEATURE_FAMILY,
        representation="raw_features",
        x=np.vstack(x_parts).astype(np.float64, copy=False),
        y=np.concatenate(y_parts).astype(np.int32, copy=False),
        rank_groups=np.concatenate(rank_group_parts).astype(object, copy=False),
        cv_groups=np.concatenate(cv_group_parts).astype(object, copy=False),
    )


def _load_earlystop_feature_store(
    *,
    main_cache_root: str,
    extra_cache_root: str,
    feature_cache_dir: Path,
    feature_workers: int,
    feature_chunk_problems: int,
    refresh_feature_cache: bool,
    max_problems_per_cache: int,
) -> list[dict[str, Any]]:
    stores: list[dict[str, Any]] = []
    roots = [
        ("main", str(main_cache_root)),
        ("extra", str(extra_cache_root)),
    ]
    for source_name, cache_root in roots:
        root_path = Path(cache_root)
        if not root_path.exists():
            continue
        desired_cache_path = _export_feature_cache_path(
            cache_dir=feature_cache_dir / source_name,
            cache_root=str(cache_root),
            positions=EARLYSTOP_ALL_POSITIONS,
            required_feature_names=set(EARLYSTOP_REQUIRED_FEATURES),
            max_problems=None if int(max_problems_per_cache) <= 0 else int(max_problems_per_cache),
            reflection_threshold=0.30,
        )
        if desired_cache_path.exists() and not refresh_feature_cache:
            feature_store, _, _ = _load_or_build_feature_store(
                cache_root=str(cache_root),
                positions=EARLYSTOP_ALL_POSITIONS,
                required_feature_names=set(EARLYSTOP_REQUIRED_FEATURES),
                max_problems=None if int(max_problems_per_cache) <= 0 else int(max_problems_per_cache),
                reflection_threshold=0.30,
                workers=max(1, int(feature_workers)),
                feature_chunk_problems=max(1, int(feature_chunk_problems)),
                feature_cache_dir=feature_cache_dir / source_name,
                refresh_feature_cache=bool(refresh_feature_cache),
                include_cache_keys=None,
                exclude_cache_keys=None,
            )
            stores.extend(_qualify_feature_store(feature_store, source_name))
            continue
        prebuilt = _load_prebuilt_earlystop_feature_store(source_name=source_name)
        if prebuilt is not None and int(max_problems_per_cache) <= 0 and not refresh_feature_cache:
            print(f"[load] Reusing prebuilt EarlyStop feature store source={source_name}", flush=True)
            stores.extend(prebuilt)
            continue
        feature_store, _, _ = _load_or_build_feature_store(
            cache_root=str(cache_root),
            positions=EARLYSTOP_ALL_POSITIONS,
            required_feature_names=set(EARLYSTOP_REQUIRED_FEATURES),
            max_problems=None if int(max_problems_per_cache) <= 0 else int(max_problems_per_cache),
            reflection_threshold=0.30,
            workers=max(1, int(feature_workers)),
            feature_chunk_problems=max(1, int(feature_chunk_problems)),
            feature_cache_dir=feature_cache_dir / source_name,
            refresh_feature_cache=bool(refresh_feature_cache),
            include_cache_keys=None,
            exclude_cache_keys=None,
        )
        stores.extend(_qualify_feature_store(feature_store, source_name))
    if not stores:
        raise ValueError("No earlystop feature stores were loaded")
    return stores


def _candidate_prebuilt_feature_store_paths(source_name: str) -> list[Path]:
    if source_name == "main":
        patterns = [
            "results/cache/rule_and_family_baselines/cache_all_ref030_*.pkl",
        ]
    else:
        patterns = [
            "results/cache/rule_and_family_baselines/cache_train_all_ref030_*.pkl",
        ]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(sorted(REPO_ROOT.glob(pattern)))
    return candidates


def _load_prebuilt_earlystop_feature_store(*, source_name: str) -> Optional[list[dict[str, Any]]]:
    for path in _candidate_prebuilt_feature_store_paths(source_name):
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        feature_store = payload.get("feature_store")
        if not isinstance(feature_store, list) or not feature_store:
            continue
        required_positions = set(float(v) for v in EARLYSTOP_ALL_POSITIONS)
        normalized: list[dict[str, Any]] = []
        ok = True
        for item in feature_store:
            item_positions = {float(v) for v in item.get("positions", payload.get("positions", []))}
            if not required_positions.issubset(item_positions):
                ok = False
                break
            normalized.append(
                {
                    **dict(item),
                    "source_name": str(source_name),
                    "base_cache_key": str(item.get("base_cache_key", item["cache_key"])),
                    "cache_key": str(item["cache_key"]),
                }
            )
        if ok and normalized:
            return normalized
    return None


def _load_bestofn_math_problems(
    *,
    distance_threads: int,
    max_problems_per_profile: int,
) -> list[tuple[str, list[Any]]]:
    problems_by_source: list[tuple[str, list[Any]]] = []
    for source_name in ("main", "train"):
        problems = _extract_all_problems(
            source_name,
            distance_threads=int(distance_threads),
            max_problems=int(max_problems_per_profile),
        )
        if problems:
            problems_by_source.append((source_name, list(problems)))
    if not problems_by_source:
        raise ValueError("No Best-of-N math problems were extracted")
    return problems_by_source


def _summarize_task_family(
    rows: list[dict[str, Any]],
    *,
    task_family: str,
    domain: Optional[str] = None,
    model_names: Optional[set[str]] = None,
) -> dict[str, float]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if str(row["task_family"]) != str(task_family):
            continue
        if domain is not None and str(row["domain"]) != str(domain):
            continue
        if model_names is not None and str(row["model_name"]) not in model_names:
            continue
        selected.append(row)

    total_weight = float(sum(int(row["holdout_n_problems"]) for row in selected))
    if total_weight <= 0.0:
        return {}

    metrics = {}
    for key in ("holdout_pairwise_acc", "holdout_hit@1", "holdout_ndcg@3", "holdout_mrr", "holdout_stop_acc"):
        num = 0.0
        den = 0.0
        for row in selected:
            value = _coerce_metric(row.get(key))
            weight = float(int(row["holdout_n_problems"]))
            if value is None or weight <= 0.0:
                continue
            num += float(value) * weight
            den += weight
        if den > 0.0:
            metrics[key] = float(num / den)
    metrics["n_rows"] = int(len(selected))
    metrics["n_problems"] = int(total_weight)
    return metrics


def _weighted_metric_from_rows(rows: list[dict[str, Any]], key: str) -> Optional[float]:
    num = 0.0
    den = 0.0
    for row in rows:
        value = _coerce_metric(row.get(key))
        weight = float(int(row.get("holdout_n_problems", 0)))
        if value is None or weight <= 0.0:
            continue
        num += float(value) * weight
        den += weight
    if den <= 0.0:
        return None
    return float(num / den)


def _aggregate_selected_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = {
        key: _weighted_metric_from_rows(rows, key)
        for key in (
            "holdout_pairwise_acc",
            "holdout_hit@1",
            "holdout_hit@3",
            "holdout_mrr",
            "holdout_ndcg@3",
            "holdout_auroc",
            "holdout_selacc@10%",
            "holdout_stop_acc",
        )
    }
    metrics["n_rows"] = int(len(rows))
    metrics["n_problems"] = int(sum(int(row.get("holdout_n_problems", 0)) for row in rows))
    return metrics


def _row_cv_selection_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        _metric_key(row.get("cv_pairwise_acc")),
        _metric_key(row.get("cv_hit@1")),
        _metric_key(row.get("cv_ndcg@3")),
        _metric_key(row.get("cv_mrr")),
    )


def _aggregate_family_comparison(
    rows: list[dict[str, Any]],
    *,
    task_keys: list[str],
    pointwise_model: str,
    ranking_models: set[str],
) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(str(row["task_key"]), []).append(row)

    pointwise_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    ranking_winners: dict[str, str] = {}
    for task_key in task_keys:
        items = by_task.get(str(task_key), [])
        point_row = next((row for row in items if str(row["model_name"]) == str(pointwise_model)), None)
        rank_candidates = [row for row in items if str(row["model_name"]) in ranking_models]
        if point_row is None or not rank_candidates:
            continue
        rank_row = sorted(rank_candidates, key=_row_cv_selection_key, reverse=True)[0]
        pointwise_rows.append(point_row)
        ranking_rows.append(rank_row)
        ranking_winners[str(task_key)] = str(rank_row["model_name"])

    return {
        "pointwise": _aggregate_selected_rows(pointwise_rows),
        "ranking": _aggregate_selected_rows(ranking_rows),
        "ranking_winners": ranking_winners,
    }


def _write_markdown_note(
    *,
    rows: list[dict[str, Any]],
    out_path: Path,
    summary_json_path: Path,
) -> None:
    ranking_models = {"pairwise_logistic", "pairwise_linear_svm", "xgboost_rank_ndcg"}
    pointwise_models = {"pointwise_logistic"}
    frozen_pointwise_models = {"pointwise_logistic_frozen_z"}
    frozen_pairwise_models = {"pairwise_logistic_frozen_z"}

    bestofn_comparison = _aggregate_family_comparison(
        rows,
        task_keys=["bestofn_math"],
        pointwise_model="pointwise_logistic",
        ranking_models=ranking_models,
    )
    early_math_comparison = _aggregate_family_comparison(
        rows,
        task_keys=[f"earlystop_math_{anchor}" for anchor in (10, 40, 70, 100)],
        pointwise_model="pointwise_logistic",
        ranking_models=ranking_models,
    )
    early_science_comparison = _aggregate_family_comparison(
        rows,
        task_keys=[f"earlystop_science_{anchor}" for anchor in (10, 40, 70, 100)],
        pointwise_model="pointwise_logistic",
        ranking_models=ranking_models,
    )
    coding_comparison = _aggregate_family_comparison(
        rows,
        task_keys=[f"earlystop_coding_{anchor}" for anchor in (70, 80, 90, 100)],
        pointwise_model="pointwise_logistic",
        ranking_models=ranking_models,
    )

    bestofn_pointwise = bestofn_comparison["pointwise"]
    bestofn_ranking = bestofn_comparison["ranking"]
    early_math_pointwise = early_math_comparison["pointwise"]
    early_math_ranking = early_math_comparison["ranking"]
    early_science_pointwise = early_science_comparison["pointwise"]
    early_science_ranking = early_science_comparison["ranking"]
    coding_pointwise = coding_comparison["pointwise"]
    coding_ranking = coding_comparison["ranking"]

    frozen_pointwise = _summarize_task_family(rows, task_family="earlystop", model_names=frozen_pointwise_models)
    frozen_pairwise = _summarize_task_family(rows, task_family="earlystop", model_names=frozen_pairwise_models)

    coding_delta = (
        _coerce_metric(coding_ranking.get("holdout_pairwise_acc"))
        or 0.0
    ) - (
        _coerce_metric(coding_pointwise.get("holdout_pairwise_acc"))
        or 0.0
    )
    noncoding_delta_values = []
    for lhs, rhs in (
        (early_math_ranking, early_math_pointwise),
        (early_science_ranking, early_science_pointwise),
    ):
        if _coerce_metric(lhs.get("holdout_pairwise_acc")) is not None and _coerce_metric(rhs.get("holdout_pairwise_acc")) is not None:
            noncoding_delta_values.append(
                float(lhs["holdout_pairwise_acc"]) - float(rhs["holdout_pairwise_acc"])
            )
    noncoding_delta = float(np.mean(noncoding_delta_values)) if noncoding_delta_values else 0.0

    overall_pointwise_pairwise = [
        float(bestofn_pointwise.get("holdout_pairwise_acc", 0.0)),
        float(early_math_pointwise.get("holdout_pairwise_acc", 0.0)),
        float(early_science_pointwise.get("holdout_pairwise_acc", 0.0)),
        float(coding_pointwise.get("holdout_pairwise_acc", 0.0)),
    ]
    overall_ranking_pairwise = [
        float(bestofn_ranking.get("holdout_pairwise_acc", 0.0)),
        float(early_math_ranking.get("holdout_pairwise_acc", 0.0)),
        float(early_science_ranking.get("holdout_pairwise_acc", 0.0)),
        float(coding_ranking.get("holdout_pairwise_acc", 0.0)),
    ]
    mean_pointwise = float(np.mean(overall_pointwise_pairwise))
    mean_ranking = float(np.mean(overall_ranking_pairwise))

    if mean_ranking > mean_pointwise + 1e-2:
        q1_answer = "Yes."
    elif mean_ranking > mean_pointwise + 1e-4:
        q1_answer = "Slightly yes."
    elif math.isclose(mean_ranking, mean_pointwise, rel_tol=0.0, abs_tol=1e-4):
        q1_answer = "They are effectively tied on this split."
    else:
        q1_answer = "No on average for this split."

    if coding_delta > noncoding_delta + 1e-4:
        q2_answer = "Yes; coding gains more from ranking loss than math/science in this run."
    elif coding_delta < noncoding_delta - 1e-4:
        q2_answer = "No; coding does not gain more than math/science in this run."
    else:
        q2_answer = "Not clearly; coding and math/science gain by similar amounts."

    frozen_delta = (
        _coerce_metric(frozen_pairwise.get("holdout_pairwise_acc")) or 0.0
    ) - (
        _coerce_metric(frozen_pointwise.get("holdout_pairwise_acc")) or 0.0
    )
    if frozen_delta > 1e-4:
        q3_answer = "Yes."
    elif frozen_delta < -1e-4:
        q3_answer = "No."
    else:
        q3_answer = "They are effectively tied."

    if coding_delta > 0.0 and frozen_delta > 0.0:
        q4_answer = "Yes, the evidence leans toward a ranking-head mismatch rather than a total representation failure."
    elif coding_delta > 0.0:
        q4_answer = "Partially; the raw representation still contains ranking signal, but frozen-z does not fully confirm it."
    else:
        q4_answer = "Not yet; the current results do not support a strong ranking-head-mismatch interpretation."

    lines = [
        "# Pairwise / Listwise Ranking Baselines",
        "",
        "## Protocol",
        "",
        "- Repository: `work/NAD_Next` only.",
        "- Grouping unit for ranking: within-problem runs only; no cross-problem pairs.",
        f"- Informative-group filter: {INFORMATIVE_GROUP_FILTER}.",
        "- Holdout unit: `dataset + problem_id`, matched across cache roots, `holdout_split=0.15`, `split_seed=42`.",
        "- Optimization target for model selection: grouped CV `pairwise_acc`, tie-broken by `hit@1`, `ndcg@3`, then `mrr`.",
        "- For the pointwise-vs-ranking comparison below, the ranking-family row uses the strongest raw ranking baseline per task slice after CV selection among `pairwise_logistic`, `pairwise_linear_svm`, and `xgboost_rank_ndcg`.",
        "- Reported metrics: holdout `pairwise_acc`, `hit@1`, `hit@3`, `mrr`, `ndcg@3`, `auroc`, `selacc@10%`; for EarlyStop single-anchor eval, `stop_acc = hit@1`.",
        f"- Full machine-readable summary: `{summary_json_path}`.",
        "",
        "## Aggregate Holdout Comparison",
        "",
        "| Slice | Pointwise `pairwise_acc` | Ranking-family `pairwise_acc` | Pointwise `hit@1` | Ranking-family `hit@1` |",
        "| --- | ---: | ---: | ---: | ---: |",
        "| Best-of-N math | {p1:.4f} | {r1:.4f} | {p2:.4f} | {r2:.4f} |".format(
            p1=float(bestofn_pointwise.get("holdout_pairwise_acc", float("nan"))),
            r1=float(bestofn_ranking.get("holdout_pairwise_acc", float("nan"))),
            p2=float(bestofn_pointwise.get("holdout_hit@1", float("nan"))),
            r2=float(bestofn_ranking.get("holdout_hit@1", float("nan"))),
        ),
        "| EarlyStop math | {p1:.4f} | {r1:.4f} | {p2:.4f} | {r2:.4f} |".format(
            p1=float(early_math_pointwise.get("holdout_pairwise_acc", float("nan"))),
            r1=float(early_math_ranking.get("holdout_pairwise_acc", float("nan"))),
            p2=float(early_math_pointwise.get("holdout_hit@1", float("nan"))),
            r2=float(early_math_ranking.get("holdout_hit@1", float("nan"))),
        ),
        "| EarlyStop science | {p1:.4f} | {r1:.4f} | {p2:.4f} | {r2:.4f} |".format(
            p1=float(early_science_pointwise.get("holdout_pairwise_acc", float("nan"))),
            r1=float(early_science_ranking.get("holdout_pairwise_acc", float("nan"))),
            p2=float(early_science_pointwise.get("holdout_hit@1", float("nan"))),
            r2=float(early_science_ranking.get("holdout_hit@1", float("nan"))),
        ),
        "| EarlyStop coding (late anchors) | {p1:.4f} | {r1:.4f} | {p2:.4f} | {r2:.4f} |".format(
            p1=float(coding_pointwise.get("holdout_pairwise_acc", float("nan"))),
            r1=float(coding_ranking.get("holdout_pairwise_acc", float("nan"))),
            p2=float(coding_pointwise.get("holdout_hit@1", float("nan"))),
            r2=float(coding_ranking.get("holdout_hit@1", float("nan"))),
        ),
        "",
        "## Explicit Answers",
        "",
        "### 1) Are pairwise objectives better than pointwise objectives?",
        "",
        f"{q1_answer} The mean holdout `pairwise_acc` across the four summary slices is `{mean_pointwise:.4f}` for pointwise logistic versus `{mean_ranking:.4f}` for the best CV-selected ranking-family baseline per slice.",
        "",
        "### 2) Does ranking loss help coding more than math/science?",
        "",
        f"{q2_answer} Coding changes by `{coding_delta:+.4f}` `pairwise_acc` versus the pointwise control, while the mean noncoding EarlyStop delta is `{noncoding_delta:+.4f}`.",
        "",
        "### 3) Does pairwise-on-frozen-z beat pointwise-on-frozen-z?",
        "",
        (
            f"{q3_answer} Across EarlyStop slices, frozen-z pointwise logistic reaches "
            f"`{float(frozen_pointwise.get('holdout_pairwise_acc', float('nan'))):.4f}` holdout `pairwise_acc`, "
            f"while frozen-z pairwise logistic reaches "
            f"`{float(frozen_pairwise.get('holdout_pairwise_acc', float('nan'))):.4f}`."
        ),
        "",
        "### 4) Should the paper treat coding as a “ranking-head mismatch” rather than a total representation failure?",
        "",
        q4_answer,
        "",
        "The concrete evidence used here is the same-representation comparison on coding late anchors: raw-feature pointwise logistic versus pairwise/listwise objectives, plus the frozen-z pointwise versus frozen-z pairwise comparison.",
        "",
        "## Notes",
        "",
        "- Best-of-N math uses the existing augmented math feature family `all_aug`.",
        f"- EarlyStop math/science/coding use the shared feature family `{EARLYSTOP_FEATURE_FAMILY_NAME}` with representation `{EARLYSTOP_REPRESENTATION}`.",
        "- The CSV contains one row per task/domain/anchor/model family winner after grouped-CV selection, final refit on the train split, and grouped holdout evaluation.",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _import_xgboost():
    try:
        return importlib.import_module("xgboost")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "xgboost is not installed. Re-run with `--install-tree-backend` or install it in the active environment."
        ) from exc


def _ensure_xgboost(*, install_tree_backend: bool) -> Optional[str]:
    try:
        module = importlib.import_module("xgboost")
        return getattr(module, "__version__", "unknown")
    except ModuleNotFoundError:
        if not install_tree_backend:
            return None
    print("[deps] Installing xgboost into the active interpreter environment...", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
    module = importlib.import_module("xgboost")
    return getattr(module, "__version__", "unknown")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train pairwise/listwise ranking baselines for SVDomain inside NAD_Next")
    ap.add_argument("--main-cache-root", default="MUI_HUB/cache")
    ap.add_argument("--extra-cache-root", default="MUI_HUB/cache_train")
    ap.add_argument("--holdout-split", type=float, default=0.15)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--cv-splits", type=int, default=3)
    ap.add_argument("--feature-workers", type=int, default=4)
    ap.add_argument("--feature-chunk-problems", type=int, default=DEFAULT_FEATURE_CHUNK_PROBLEMS)
    ap.add_argument("--distance-threads", type=int, default=4)
    ap.add_argument("--max-problems-per-cache", type=int, default=0)
    ap.add_argument("--max-math-problems-per-profile", type=int, default=0)
    ap.add_argument("--refresh-feature-cache", action="store_true")
    ap.add_argument("--install-tree-backend", action="store_true")
    ap.add_argument("--feature-cache-dir", default="results/cache/pairwise_listwise_rankers")
    ap.add_argument("--out-csv", default="results/tables/pairwise_listwise_rankers.csv")
    ap.add_argument("--out-summary-json", default="results/scans/pairwise_listwise_rankers/summary.json")
    ap.add_argument("--out-md", default="docs/PAIRWISE_LISTWISE_BASELINES.md")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    feature_cache_dir = REPO_ROOT / str(args.feature_cache_dir)
    out_csv = REPO_ROOT / str(args.out_csv)
    out_md = REPO_ROOT / str(args.out_md)
    out_summary_json = REPO_ROOT / str(args.out_summary_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_summary_json.parent.mkdir(parents=True, exist_ok=True)

    xgboost_version = _ensure_xgboost(install_tree_backend=bool(args.install_tree_backend))
    enable_tree_ranker = xgboost_version is not None

    print("[load] Best-of-N math problems", flush=True)
    math_problems_by_source = _load_bestofn_math_problems(
        distance_threads=int(args.distance_threads),
        max_problems_per_profile=int(args.max_math_problems_per_profile),
    )
    math_train_rows, math_holdout_rows, math_holdout_summary = _split_bestofn_math_holdout(
        math_problems_by_source,
        holdout_split=float(args.holdout_split),
        split_seed=int(args.split_seed),
    )
    math_train_slice = _build_bestofn_math_slice(math_train_rows, task_key="bestofn_math")
    math_holdout_slice = _build_bestofn_math_slice(math_holdout_rows, task_key="bestofn_math")

    print("[load] EarlyStop feature stores", flush=True)
    qualified_feature_store = _load_earlystop_feature_store(
        main_cache_root=str(args.main_cache_root),
        extra_cache_root=str(args.extra_cache_root),
        feature_cache_dir=feature_cache_dir,
        feature_workers=int(args.feature_workers),
        feature_chunk_problems=int(args.feature_chunk_problems),
        refresh_feature_cache=bool(args.refresh_feature_cache),
        max_problems_per_cache=int(args.max_problems_per_cache),
    )
    holdout_problem_map, holdout_problem_summary = _build_holdout_problem_map(
        qualified_feature_store,
        holdout_split=float(args.holdout_split),
        split_seed=int(args.split_seed),
    )
    train_store, holdout_store, _ = _split_feature_store(
        qualified_feature_store,
        holdout_problem_map=holdout_problem_map,
    )

    earlystop_feature_indices = [FEATURE_TO_INDEX[name] for name in EARLYSTOP_FEATURE_NAMES]
    task_pairs: list[tuple[RankingSlice, RankingSlice]] = [
        (math_train_slice, math_holdout_slice),
    ]
    for anchor_pct in (10, 40, 70, 100):
        task_pairs.append(
            (
                _build_earlystop_position_slice(
                    task_key=f"earlystop_math_{anchor_pct}",
                    domain="math",
                    dataset_scope="earlystop_math",
                    anchor_pct=int(anchor_pct),
                    feature_store=train_store,
                    feature_indices=earlystop_feature_indices,
                ),
                _build_earlystop_position_slice(
                    task_key=f"earlystop_math_{anchor_pct}",
                    domain="math",
                    dataset_scope="earlystop_math",
                    anchor_pct=int(anchor_pct),
                    feature_store=holdout_store,
                    feature_indices=earlystop_feature_indices,
                ),
            )
        )
        task_pairs.append(
            (
                _build_earlystop_position_slice(
                    task_key=f"earlystop_science_{anchor_pct}",
                    domain="science",
                    dataset_scope="earlystop_science",
                    anchor_pct=int(anchor_pct),
                    feature_store=train_store,
                    feature_indices=earlystop_feature_indices,
                ),
                _build_earlystop_position_slice(
                    task_key=f"earlystop_science_{anchor_pct}",
                    domain="science",
                    dataset_scope="earlystop_science",
                    anchor_pct=int(anchor_pct),
                    feature_store=holdout_store,
                    feature_indices=earlystop_feature_indices,
                ),
            )
        )
    for anchor_pct in (70, 80, 90, 100):
        task_pairs.append(
            (
                _build_earlystop_position_slice(
                    task_key=f"earlystop_coding_{anchor_pct}",
                    domain="coding",
                    dataset_scope="earlystop_coding_late",
                    anchor_pct=int(anchor_pct),
                    feature_store=train_store,
                    feature_indices=earlystop_feature_indices,
                ),
                _build_earlystop_position_slice(
                    task_key=f"earlystop_coding_{anchor_pct}",
                    domain="coding",
                    dataset_scope="earlystop_coding_late",
                    anchor_pct=int(anchor_pct),
                    feature_store=holdout_store,
                    feature_indices=earlystop_feature_indices,
                ),
            )
        )

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "protocol": {
            "holdout_split": float(args.holdout_split),
            "split_seed": int(args.split_seed),
            "cv_splits": int(args.cv_splits),
            "math_feature_family": BESTOFN_MATH_FEATURE_FAMILY,
            "earlystop_feature_family": EARLYSTOP_FEATURE_FAMILY_NAME,
            "earlystop_representation": EARLYSTOP_REPRESENTATION,
            "pairing_logic": PAIRWISE_LOGISTIC_PAIRING,
            "tree_ranking_logic": TREE_RANKING_LOGIC,
            "group_filtering_logic": INFORMATIVE_GROUP_FILTER,
            "xgboost_version": xgboost_version,
            "python_executable": sys.executable,
        },
        "bestofn_math_holdout_summary": math_holdout_summary,
        "earlystop_holdout_summary": holdout_problem_summary,
        "task_rows": [],
    }

    for raw_train_slice, raw_holdout_slice in task_pairs:
        try:
            train_slice = _retain_informative_groups(raw_train_slice)
            holdout_slice = _retain_informative_groups(raw_holdout_slice)
        except ValueError as exc:
            print(f"[skip] {raw_train_slice.task_key}: {exc}", flush=True)
            summary["task_rows"].append(
                {
                    "task_key": raw_train_slice.task_key,
                    "skipped": True,
                    "reason": str(exc),
                }
            )
            continue
        print(
            f"[task] {train_slice.task_key} train_problems={len(np.unique(train_slice.cv_groups))} "
            f"holdout_problems={len(np.unique(holdout_slice.cv_groups))}",
            flush=True,
        )
        candidate_specs = _candidate_specs_for_slice(
            train_slice,
            enable_tree_ranker=bool(enable_tree_ranker),
        )
        cv_candidates: list[tuple[CandidateSpec, dict[str, Any]]] = []
        for spec in candidate_specs:
            cv_metrics = _evaluate_candidate_cv(
                train_slice,
                spec,
                n_splits=int(args.cv_splits),
                random_state=int(args.split_seed),
            )
            if cv_metrics is None:
                continue
            cv_candidates.append((spec, cv_metrics))

        if not cv_candidates:
            reason = (
                f"{train_slice.task_key} has no valid candidate models after grouped CV; "
                f"likely too few informative train groups for cv_splits={int(args.cv_splits)}"
            )
            print(f"[skip] {reason}", flush=True)
            summary["task_rows"].append(
                {
                    "task_key": train_slice.task_key,
                    "skipped": True,
                    "reason": reason,
                }
            )
            continue

        winners_by_model: dict[str, tuple[CandidateSpec, dict[str, Any]]] = {}
        for spec, cv_metrics in cv_candidates:
            current = winners_by_model.get(spec.model_name)
            if current is None:
                winners_by_model[spec.model_name] = (spec, cv_metrics)
                continue
            better = _select_best_candidate([current, (spec, cv_metrics)])
            winners_by_model[spec.model_name] = better

        task_summary_rows: list[dict[str, Any]] = []
        for model_name in sorted(winners_by_model.keys()):
            spec, cv_metrics = winners_by_model[model_name]
            model_meta, holdout_metrics = _fit_and_evaluate_holdout(
                train_slice,
                holdout_slice,
                spec,
                random_state=int(args.split_seed),
            )
            row = {
                "task_key": str(train_slice.task_key),
                "task_family": str(train_slice.task_family),
                "domain": str(train_slice.domain),
                "dataset_scope": str(train_slice.dataset_scope),
                "anchor_pct": "" if train_slice.anchor_pct is None else int(train_slice.anchor_pct),
                "feature_family": str(train_slice.feature_family),
                "representation": str(train_slice.representation),
                "model_name": str(spec.model_name),
                "objective": str(spec.objective),
                "basis_mode": str(model_meta.get("basis_mode")),
                "basis_rank": "" if model_meta.get("basis_rank") is None else int(model_meta["basis_rank"]),
                "basis_whiten": "" if model_meta.get("basis_whiten") is None else int(bool(model_meta["basis_whiten"])),
                "pairing_logic": str(spec.pairing_logic),
                "group_filtering_logic": INFORMATIVE_GROUP_FILTER,
                "optimized_metric": "cv_pairwise_acc",
                "cv_pairwise_acc": cv_metrics.get("pairwise_acc"),
                "cv_hit@1": cv_metrics.get("hit@1"),
                "cv_hit@3": cv_metrics.get("hit@3"),
                "cv_mrr": cv_metrics.get("mrr"),
                "cv_ndcg@3": cv_metrics.get("ndcg@3"),
                "cv_auroc": cv_metrics.get("auroc"),
                "cv_selacc@10%": cv_metrics.get("selacc@10%"),
                "cv_n_valid_folds": int(cv_metrics.get("n_valid_folds", 0)),
                "holdout_pairwise_acc": holdout_metrics.get("pairwise_acc"),
                "holdout_hit@1": holdout_metrics.get("hit@1"),
                "holdout_hit@3": holdout_metrics.get("hit@3"),
                "holdout_mrr": holdout_metrics.get("mrr"),
                "holdout_ndcg@3": holdout_metrics.get("ndcg@3"),
                "holdout_auroc": holdout_metrics.get("auroc"),
                "holdout_selacc@10%": holdout_metrics.get("selacc@10%"),
                "holdout_stop_acc": holdout_metrics.get("stop_acc"),
                "train_n_problems": int(len(np.unique(train_slice.rank_groups))),
                "train_n_samples": int(train_slice.y.shape[0]),
                "holdout_n_problems": int(holdout_metrics.get("n_problems", 0)),
                "holdout_n_samples": int(holdout_metrics.get("n_samples", 0)),
                "split_seed": int(args.split_seed),
                "holdout_split": float(args.holdout_split),
            }
            for meta_key, meta_value in model_meta.items():
                if meta_key in {"basis_mode", "basis_rank", "basis_whiten"}:
                    continue
                row[f"model__{meta_key}"] = meta_value
            rows.append(row)
            task_summary_rows.append(row)
            print(
                f"  [winner] {model_name} cv_pairwise={_coerce_metric(row['cv_pairwise_acc'])} "
                f"holdout_pairwise={_coerce_metric(row['holdout_pairwise_acc'])}",
                flush=True,
            )
        summary["task_rows"].append({"task_key": train_slice.task_key, "rows": task_summary_rows})

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary["csv_path"] = str(out_csv)
    summary["markdown_path"] = str(out_md)
    out_summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_markdown_note(rows=rows, out_path=out_md, summary_json_path=out_summary_json)
    print(f"[done] csv={out_csv}", flush=True)
    print(f"[done] summary={out_summary_json}", flush=True)
    print(f"[done] markdown={out_md}", flush=True)


if __name__ == "__main__":
    main()
