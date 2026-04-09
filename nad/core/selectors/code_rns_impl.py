from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .base import SelectorContext
from .code_dynamic_impl import (
    DEFAULT_CODE_DYNAMIC_REFLECTION_LOOKBACK,
    DEFAULT_CODE_DYNAMIC_REFLECTION_THRESHOLD,
    extract_code_dynamic_raw_from_state,
    order_code_dynamic_group_indices,
    prepare_code_dynamic_run_state,
)
from .code_v2_impl import (
    CODE_V2_FEATURE_NAMES,
    DEFAULT_CODE_V2_WEIGHTS,
    build_code_v2_rank_features_from_raw,
    compute_code_v2_primary_scores_from_raw,
    extract_code_v2_raw_from_state,
)

CODE_RNS_FEATURE_NAMES = list(CODE_V2_FEATURE_NAMES)
CODE_RNS_NUISANCE_NAMES = [
    "log_length",
    "mean_tok_conf",
    "reflection_density",
]
CODE_RNS_TAIL_FEATURE_INDICES = (2, 3, 4)


def default_code_rns_model_path() -> Path:
    models_dir = Path(__file__).resolve().parent.parent.parent.parent / "models" / "ml_selectors"
    v1_path = models_dir / "code_rns_v1.pkl"
    if v1_path.exists():
        return v1_path
    return models_dir / "code_rns_round1.pkl"


@dataclass(frozen=True)
class CodeRNSConfig:
    shortlist_size: int = 10
    knn_k: int = 5
    lambda_weight: float = 0.15
    use_cf_negatives: bool = True
    max_positive_anchors: int = 4096
    max_negative_anchors: int = 4096
    max_cf_anchors: int = 2048
    length_num_bins: int = 4
    conf_num_bins: int = 4
    reflection_num_bins: int = 3
    reflection_threshold: float = DEFAULT_CODE_DYNAMIC_REFLECTION_THRESHOLD
    reflection_lookback_slices: int = DEFAULT_CODE_DYNAMIC_REFLECTION_LOOKBACK
    prefix_fraction: float = 0.30
    prefix_window_tokens: int = 128
    random_seed: int = 42

    def validate(self) -> "CodeRNSConfig":
        if int(self.shortlist_size) <= 0:
            raise ValueError("shortlist_size must be positive")
        if int(self.knn_k) <= 0:
            raise ValueError("knn_k must be positive")
        if float(self.lambda_weight) < 0.0:
            raise ValueError("lambda_weight must be non-negative")
        if int(self.max_positive_anchors) <= 0:
            raise ValueError("max_positive_anchors must be positive")
        if int(self.max_negative_anchors) <= 0:
            raise ValueError("max_negative_anchors must be positive")
        if int(self.max_cf_anchors) < 0:
            raise ValueError("max_cf_anchors must be non-negative")
        if int(self.length_num_bins) <= 0 or int(self.conf_num_bins) <= 0 or int(self.reflection_num_bins) <= 0:
            raise ValueError("bucket counts must be positive")
        if float(self.prefix_fraction) <= 0.0:
            raise ValueError("prefix_fraction must be positive")
        if int(self.prefix_window_tokens) <= 0:
            raise ValueError("prefix_window_tokens must be positive")
        return self

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CodeRNSBundle:
    config: CodeRNSConfig
    positive_anchors: np.ndarray
    hard_negative_anchors: np.ndarray
    cf_negative_anchors: np.ndarray
    distance_weights: np.ndarray
    feature_names: list[str] = field(default_factory=lambda: list(CODE_RNS_FEATURE_NAMES))
    nuisance_names: list[str] = field(default_factory=lambda: list(CODE_RNS_NUISANCE_NAMES))
    training_summary: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        import joblib

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, out_path)

    @classmethod
    def load(cls, path: str | Path) -> "CodeRNSBundle":
        import joblib

        return joblib.load(path)


@dataclass(frozen=True)
class CodeRNSDecision:
    config: CodeRNSConfig
    baseline_order: np.ndarray
    final_order: np.ndarray
    baseline_scores: np.ndarray
    final_scores: np.ndarray
    shortlist_indices: np.ndarray
    shortlist_negative_fraction: np.ndarray
    shortlist_adjusted_scores: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.as_dict(),
            "baseline_order": self.baseline_order.tolist(),
            "final_order": self.final_order.tolist(),
            "baseline_scores": self.baseline_scores.tolist(),
            "final_scores": self.final_scores.tolist(),
            "shortlist_indices": self.shortlist_indices.tolist(),
            "shortlist_negative_fraction": self.shortlist_negative_fraction.tolist(),
            "shortlist_adjusted_scores": self.shortlist_adjusted_scores.tolist(),
        }


def _safe_mean_tok_conf(run_state: dict[str, Any]) -> float:
    tok_conf = np.asarray(run_state.get("tok_conf", np.zeros(0, dtype=np.float64)), dtype=np.float64)
    if tok_conf.size <= 0:
        return 0.0
    return float(np.mean(tok_conf))


def _prepare_nuisance_matrix(nuisance: np.ndarray | None, n_rows: int) -> np.ndarray:
    if nuisance is None:
        return np.zeros((int(n_rows), len(CODE_RNS_NUISANCE_NAMES)), dtype=np.float64)
    arr = np.asarray(nuisance, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != len(CODE_RNS_NUISANCE_NAMES):
        raise ValueError(
            f"nuisance must have shape (N, {len(CODE_RNS_NUISANCE_NAMES)}), got {arr.shape}"
        )
    if arr.shape[0] != int(n_rows):
        raise ValueError(f"nuisance row mismatch: expected {n_rows}, got {arr.shape[0]}")
    out = np.asarray(arr, dtype=np.float64).copy()
    for col_idx in range(out.shape[1]):
        col = out[:, col_idx]
        valid = np.isfinite(col)
        fill = float(np.median(col[valid])) if valid.any() else 0.0
        out[:, col_idx] = np.where(valid, col, fill)
    return out


def _deterministic_subsample_rows(arr: np.ndarray, limit: int) -> np.ndarray:
    rows = int(arr.shape[0])
    keep = int(max(0, limit))
    if rows <= keep or keep <= 0:
        return np.asarray(arr, dtype=np.float64)
    idx = np.linspace(0, rows - 1, num=keep, dtype=np.int64)
    return np.asarray(arr[idx], dtype=np.float64)


def _quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    valid = np.isfinite(arr)
    if not valid.any():
        return np.zeros(0, dtype=np.float64)
    clean = arr[valid]
    if clean.size <= 1 or int(n_bins) <= 1:
        return np.zeros(0, dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, num=int(n_bins) + 1, dtype=np.float64)[1:-1]
    if quantiles.size <= 0:
        return np.zeros(0, dtype=np.float64)
    edges = np.unique(np.quantile(clean, quantiles))
    return np.asarray(edges, dtype=np.float64)


def _bucket_id(row: np.ndarray, edges_by_col: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[int, int, int]:
    return (
        int(np.searchsorted(edges_by_col[0], float(row[0]), side="right")),
        int(np.searchsorted(edges_by_col[1], float(row[1]), side="right")),
        int(np.searchsorted(edges_by_col[2], float(row[2]), side="right")),
    )


def _fallback_bucket_ids(bucket: tuple[int, int, int]) -> list[tuple[int, int, int] | tuple[str]]:
    length_bin, conf_bin, reflection_bin = bucket
    return [
        (length_bin, conf_bin, reflection_bin),
        (length_bin, conf_bin, -1),
        (length_bin, -1, -1),
        ("global",),
    ]


def _build_hard_negative_lookup(
    nuisance_neg: np.ndarray,
    *,
    config: CodeRNSConfig,
) -> dict[tuple[int, ...] | tuple[str], np.ndarray]:
    if nuisance_neg.size <= 0:
        return {("global",): np.zeros(0, dtype=np.int64)}
    edges_by_col = (
        _quantile_edges(nuisance_neg[:, 0], int(config.length_num_bins)),
        _quantile_edges(nuisance_neg[:, 1], int(config.conf_num_bins)),
        _quantile_edges(nuisance_neg[:, 2], int(config.reflection_num_bins)),
    )
    buckets: dict[tuple[int, ...] | tuple[str], list[int]] = {("global",): list(range(int(nuisance_neg.shape[0])))}
    for idx, row in enumerate(np.asarray(nuisance_neg, dtype=np.float64)):
        exact = _bucket_id(row, edges_by_col)
        buckets.setdefault(exact, []).append(int(idx))
        buckets.setdefault((exact[0], exact[1], -1), []).append(int(idx))
        buckets.setdefault((exact[0], -1, -1), []).append(int(idx))
    return {key: np.asarray(sorted(set(values)), dtype=np.int64) for key, values in buckets.items()}


def _build_cf_negative_anchors(
    positive_anchors: np.ndarray,
    hard_negative_anchors: np.ndarray,
    nuisance_pos: np.ndarray,
    nuisance_neg: np.ndarray,
    *,
    config: CodeRNSConfig,
) -> np.ndarray:
    if not bool(config.use_cf_negatives):
        return np.zeros((0, len(CODE_RNS_FEATURE_NAMES)), dtype=np.float64)
    if positive_anchors.size <= 0 or hard_negative_anchors.size <= 0:
        return np.zeros((0, len(CODE_RNS_FEATURE_NAMES)), dtype=np.float64)

    neg_lookup = _build_hard_negative_lookup(nuisance_neg, config=config)
    edges_by_col = (
        _quantile_edges(nuisance_neg[:, 0], int(config.length_num_bins)),
        _quantile_edges(nuisance_neg[:, 1], int(config.conf_num_bins)),
        _quantile_edges(nuisance_neg[:, 2], int(config.reflection_num_bins)),
    )
    global_tail = np.median(
        np.asarray(hard_negative_anchors[:, CODE_RNS_TAIL_FEATURE_INDICES], dtype=np.float64),
        axis=0,
    )

    cf_rows: list[np.ndarray] = []
    for pos_row, nuisance_row in zip(np.asarray(positive_anchors, dtype=np.float64), np.asarray(nuisance_pos, dtype=np.float64)):
        bucket = _bucket_id(nuisance_row, edges_by_col)
        matched_idx = np.zeros(0, dtype=np.int64)
        for fallback_key in _fallback_bucket_ids(bucket):
            matched_idx = np.asarray(neg_lookup.get(fallback_key, np.zeros(0, dtype=np.int64)), dtype=np.int64)
            if matched_idx.size > 0:
                break
        tail_template = global_tail
        if matched_idx.size > 0:
            tail_template = np.median(
                np.asarray(hard_negative_anchors[matched_idx][:, CODE_RNS_TAIL_FEATURE_INDICES], dtype=np.float64),
                axis=0,
            )
        cf_row = np.asarray(pos_row, dtype=np.float64).copy()
        cf_row[list(CODE_RNS_TAIL_FEATURE_INDICES)] = np.asarray(tail_template, dtype=np.float64)
        cf_rows.append(cf_row)

    if not cf_rows:
        return np.zeros((0, len(CODE_RNS_FEATURE_NAMES)), dtype=np.float64)
    cf_arr = np.asarray(cf_rows, dtype=np.float64)
    return _deterministic_subsample_rows(cf_arr, int(config.max_cf_anchors))


def extract_code_rns_context_payload(
    context: SelectorContext,
    *,
    config: CodeRNSConfig | None = None,
) -> dict[str, Any]:
    cfg = (config or CodeRNSConfig()).validate()
    n = len(context.run_ids)
    code_v2_raw = {
        "prefix_best_window_quality": np.full(n, np.nan, dtype=np.float64),
        "head_tail_gap": np.full(n, np.nan, dtype=np.float64),
        "tail_variance": np.full(n, np.nan, dtype=np.float64),
        "post_reflection_recovery": np.full(n, np.nan, dtype=np.float64),
        "last_block_instability": np.full(n, np.nan, dtype=np.float64),
    }
    nuisance = {
        "log_length": np.zeros(n, dtype=np.float64),
        "mean_tok_conf": np.zeros(n, dtype=np.float64),
        "reflection_density": np.zeros(n, dtype=np.float64),
    }

    for idx, run_id in enumerate(context.run_ids):
        token_view = context.cache.get_token_view(int(run_id))
        run_state = prepare_code_dynamic_run_state(
            context.cache,
            int(run_id),
            token_view=token_view,
        )
        code_v2_row = extract_code_v2_raw_from_state(
            run_state,
            reflection_threshold=float(cfg.reflection_threshold),
            reflection_lookback_slices=int(cfg.reflection_lookback_slices),
            prefix_fraction=float(cfg.prefix_fraction),
            prefix_window_tokens=int(cfg.prefix_window_tokens),
        )
        dynamic_row = extract_code_dynamic_raw_from_state(
            run_state,
            reflection_threshold=float(cfg.reflection_threshold),
            reflection_lookback_slices=int(cfg.reflection_lookback_slices),
            prefix_fraction=float(cfg.prefix_fraction),
            prefix_window_tokens=int(cfg.prefix_window_tokens),
        )
        for key in code_v2_raw:
            code_v2_raw[key][idx] = float(code_v2_row[key])
        nuisance["log_length"][idx] = float(math.log(max(float(dynamic_row.get("num_tokens", 0.0)), 1.0)))
        nuisance["mean_tok_conf"][idx] = float(_safe_mean_tok_conf(run_state))
        nuisance["reflection_density"][idx] = float(dynamic_row.get("reflection_density", 0.0))

    features = build_code_v2_rank_features_from_raw(code_v2_raw)
    baseline_scores, _ = compute_code_v2_primary_scores_from_raw(code_v2_raw)
    nuisance_matrix = np.column_stack(
        [
            np.asarray(nuisance["log_length"], dtype=np.float64),
            np.asarray(nuisance["mean_tok_conf"], dtype=np.float64),
            np.asarray(nuisance["reflection_density"], dtype=np.float64),
        ]
    ).astype(np.float64)
    return {
        "features": np.asarray(features, dtype=np.float64),
        "baseline_scores": np.asarray(baseline_scores, dtype=np.float64),
        "code_v2_raw": code_v2_raw,
        "nuisance": nuisance_matrix,
    }


def _order_to_desc_scores(order: np.ndarray, n: int, *, hi: float = 1.0, lo: float = 0.0) -> np.ndarray:
    out = np.zeros(int(n), dtype=np.float64)
    if int(n) <= 0:
        return out
    if int(n) == 1:
        out[int(order[0])] = float(hi)
        return out
    values = np.linspace(float(hi), float(lo), num=int(n), dtype=np.float64)
    for rank_pos, group_idx in enumerate(np.asarray(order, dtype=np.int64).tolist()):
        out[int(group_idx)] = float(values[int(rank_pos)])
    return out


def _weighted_distance_matrix(queries: np.ndarray, anchors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    q = np.asarray(queries, dtype=np.float64)
    a = np.asarray(anchors, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(1, 1, -1)
    if q.ndim != 2 or a.ndim != 2:
        raise ValueError("queries and anchors must be 2D")
    if q.shape[1] != a.shape[1]:
        raise ValueError(f"Feature mismatch: {q.shape[1]} vs {a.shape[1]}")
    diffs = q[:, None, :] - a[None, :, :]
    dist_sq = np.sum(np.square(diffs) * w, axis=2)
    return np.sqrt(np.maximum(dist_sq, 0.0))


class CodeRNSScorer:
    def __init__(
        self,
        *,
        bundle: CodeRNSBundle | None = None,
        config: CodeRNSConfig | None = None,
    ) -> None:
        self.bundle = bundle
        self.config = (config or (bundle.config if bundle is not None else CodeRNSConfig())).validate()

    def fit_anchor_bank(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        *,
        nuisance: np.ndarray | None = None,
    ) -> None:
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(labels, dtype=np.int32).reshape(-1)
        if X_arr.ndim != 2:
            raise ValueError(f"X must have shape (N, F), got {X_arr.shape}")
        if y_arr.ndim != 1:
            raise ValueError(f"labels must have shape (N,), got {y_arr.shape}")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(f"X/labels mismatch: {X_arr.shape[0]} vs {y_arr.shape[0]}")
        if X_arr.shape[1] != len(CODE_RNS_FEATURE_NAMES):
            raise ValueError(
                f"X must have {len(CODE_RNS_FEATURE_NAMES)} features, got {X_arr.shape[1]}"
            )

        nuisance_arr = _prepare_nuisance_matrix(nuisance, X_arr.shape[0])
        pos_mask = y_arr > 0
        neg_mask = ~pos_mask

        positive_all = np.asarray(X_arr[pos_mask], dtype=np.float64)
        hard_negative_all = np.asarray(X_arr[neg_mask], dtype=np.float64)
        nuisance_pos_all = np.asarray(nuisance_arr[pos_mask], dtype=np.float64)
        nuisance_neg_all = np.asarray(nuisance_arr[neg_mask], dtype=np.float64)

        positive_anchors = _deterministic_subsample_rows(positive_all, int(self.config.max_positive_anchors))
        hard_negative_anchors = _deterministic_subsample_rows(hard_negative_all, int(self.config.max_negative_anchors))
        nuisance_pos = _deterministic_subsample_rows(nuisance_pos_all, int(self.config.max_positive_anchors))
        nuisance_neg = _deterministic_subsample_rows(nuisance_neg_all, int(self.config.max_negative_anchors))
        cf_negative_anchors = _build_cf_negative_anchors(
            positive_anchors,
            hard_negative_anchors,
            nuisance_pos,
            nuisance_neg,
            config=self.config,
        )

        distance_weights = np.asarray(
            [float(DEFAULT_CODE_V2_WEIGHTS[name]) for name in DEFAULT_CODE_V2_WEIGHTS],
            dtype=np.float64,
        )
        weight_sum = float(np.sum(distance_weights))
        if weight_sum <= 0.0:
            distance_weights = np.full(len(CODE_RNS_FEATURE_NAMES), 1.0 / len(CODE_RNS_FEATURE_NAMES), dtype=np.float64)
        else:
            distance_weights = distance_weights / weight_sum

        self.bundle = CodeRNSBundle(
            config=self.config,
            positive_anchors=positive_anchors,
            hard_negative_anchors=hard_negative_anchors,
            cf_negative_anchors=cf_negative_anchors,
            distance_weights=distance_weights,
            training_summary={
                "n_train_rows": int(X_arr.shape[0]),
                "n_positive_rows": int(positive_all.shape[0]),
                "n_negative_rows": int(hard_negative_all.shape[0]),
                "n_positive_anchors": int(positive_anchors.shape[0]),
                "n_hard_negative_anchors": int(hard_negative_anchors.shape[0]),
                "n_cf_negative_anchors": int(cf_negative_anchors.shape[0]),
                "use_cf_negatives": bool(self.config.use_cf_negatives),
                "distance_weights": distance_weights.tolist(),
            },
        )

    def _require_bundle(self) -> CodeRNSBundle:
        if self.bundle is None:
            raise RuntimeError("CodeRNSScorer requires a fitted or loaded bundle")
        return self.bundle

    def _negative_fraction(self, X_query: np.ndarray) -> np.ndarray:
        bundle = self._require_bundle()
        Xq = np.asarray(X_query, dtype=np.float64)
        positive = np.asarray(bundle.positive_anchors, dtype=np.float64)
        hard_negative = np.asarray(bundle.hard_negative_anchors, dtype=np.float64)
        cf_negative = np.asarray(bundle.cf_negative_anchors, dtype=np.float64)

        anchor_blocks: list[np.ndarray] = []
        neg_labels: list[np.ndarray] = []
        if positive.size > 0:
            anchor_blocks.append(positive)
            neg_labels.append(np.zeros(positive.shape[0], dtype=np.float64))
        if hard_negative.size > 0:
            anchor_blocks.append(hard_negative)
            neg_labels.append(np.ones(hard_negative.shape[0], dtype=np.float64))
        if cf_negative.size > 0:
            anchor_blocks.append(cf_negative)
            neg_labels.append(np.ones(cf_negative.shape[0], dtype=np.float64))

        if not anchor_blocks:
            return np.full(Xq.shape[0], 0.5, dtype=np.float64)

        anchors = np.concatenate(anchor_blocks, axis=0)
        neg_mask = np.concatenate(neg_labels, axis=0)
        if float(np.sum(neg_mask)) <= 0.0:
            return np.zeros(Xq.shape[0], dtype=np.float64)
        if float(np.sum(1.0 - neg_mask)) <= 0.0:
            return np.ones(Xq.shape[0], dtype=np.float64)

        k = max(1, min(int(bundle.config.knn_k), int(anchors.shape[0])))
        distances = _weighted_distance_matrix(Xq, anchors, bundle.distance_weights)
        topk_idx = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        return np.asarray(np.mean(neg_mask[topk_idx], axis=1), dtype=np.float64)

    def decision_from_group_features(
        self,
        X: np.ndarray,
        baseline_scores: np.ndarray,
        D: np.ndarray,
        *,
        run_ids: list[int] | None = None,
    ) -> CodeRNSDecision:
        bundle = self._require_bundle()
        X_arr = np.asarray(X, dtype=np.float64)
        scores_arr = np.asarray(baseline_scores, dtype=np.float64)
        dist_arr = np.asarray(D, dtype=np.float64)
        n = int(scores_arr.shape[0])

        baseline_order = order_code_dynamic_group_indices(
            scores_arr,
            dist_arr,
            run_ids=run_ids,
        )
        shortlist_size = max(1, min(int(bundle.config.shortlist_size), n))
        shortlist = np.asarray(baseline_order[:shortlist_size], dtype=np.int64)
        if shortlist.size <= 1:
            return CodeRNSDecision(
                config=bundle.config,
                baseline_order=np.asarray(baseline_order, dtype=np.int64),
                final_order=np.asarray(baseline_order, dtype=np.int64),
                baseline_scores=np.asarray(scores_arr, dtype=np.float64),
                final_scores=_order_to_desc_scores(np.asarray(baseline_order, dtype=np.int64), n),
                shortlist_indices=shortlist,
                shortlist_negative_fraction=np.zeros(shortlist.size, dtype=np.float64),
                shortlist_adjusted_scores=np.asarray(scores_arr[shortlist], dtype=np.float64),
            )

        shortlist_neg_frac = self._negative_fraction(X_arr[shortlist])
        shortlist_adjusted_scores = np.asarray(
            scores_arr[shortlist] - float(bundle.config.lambda_weight) * shortlist_neg_frac,
            dtype=np.float64,
        )
        shortlist_run_ids = None
        if run_ids is not None:
            shortlist_run_ids = [int(run_ids[int(idx)]) for idx in shortlist.tolist()]
        shortlist_order_local = order_code_dynamic_group_indices(
            shortlist_adjusted_scores,
            dist_arr[np.ix_(shortlist, shortlist)],
            run_ids=shortlist_run_ids,
        )
        shortlist_reordered = np.asarray(shortlist[shortlist_order_local], dtype=np.int64)
        shortlist_set = {int(idx) for idx in shortlist.tolist()}
        rest = np.asarray([int(idx) for idx in baseline_order.tolist() if int(idx) not in shortlist_set], dtype=np.int64)
        final_order = np.concatenate([shortlist_reordered, rest], axis=0)
        final_scores = _order_to_desc_scores(final_order, n)
        return CodeRNSDecision(
            config=bundle.config,
            baseline_order=np.asarray(baseline_order, dtype=np.int64),
            final_order=np.asarray(final_order, dtype=np.int64),
            baseline_scores=np.asarray(scores_arr, dtype=np.float64),
            final_scores=np.asarray(final_scores, dtype=np.float64),
            shortlist_indices=np.asarray(shortlist, dtype=np.int64),
            shortlist_negative_fraction=np.asarray(shortlist_neg_frac, dtype=np.float64),
            shortlist_adjusted_scores=np.asarray(shortlist_adjusted_scores, dtype=np.float64),
        )

    def decision_for_context(self, context: SelectorContext, D: np.ndarray) -> CodeRNSDecision:
        payload = extract_code_rns_context_payload(context, config=self.config)
        return self.decision_from_group_features(
            payload["features"],
            payload["baseline_scores"],
            D,
            run_ids=list(map(int, context.run_ids)),
        )

    def score_context(self, context: SelectorContext, D: np.ndarray) -> np.ndarray:
        return np.asarray(self.decision_for_context(context, D).final_scores, dtype=np.float64)

    def select_best_index(self, context: SelectorContext, D: np.ndarray) -> int:
        decision = self.decision_for_context(context, D)
        if decision.final_order.size <= 0:
            return 0
        return int(decision.final_order[0])

    def save(self, path: str | Path) -> None:
        bundle = self._require_bundle()
        bundle.save(path)

    @classmethod
    def load(cls, path: str | Path) -> "CodeRNSScorer":
        bundle = CodeRNSBundle.load(path)
        return cls(bundle=bundle, config=bundle.config)
