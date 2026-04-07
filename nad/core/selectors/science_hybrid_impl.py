from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .base import SelectorContext
from .code_dynamic_impl import order_code_dynamic_group_indices
from .gpqa_pairwise_impl import (
    GPQAPairwiseScorer,
    build_gpqa_pairwise_features_configurable,
    extract_gpqa_pairwise_raw,
)
from .science_dynamic_impl import (
    DEFAULT_SCIENCE_DYNAMIC_WEIGHTS,
    compute_science_dynamic_primary_scores,
    compute_science_dynamic_primary_scores_from_raw,
)

SCIENCE_HYBRID_FAMILIES = (
    "margin_fallback",
    "shortlist_blend",
    "hard_override",
)

SCIENCE_HYBRID_PAIRWISE_BACKENDS = (
    "mean",
    "softmax_mean",
    "win_count",
    "copeland_margin",
)

DEFAULT_SCIENCE_HYBRID_FAMILY = "hard_override"
DEFAULT_SCIENCE_HYBRID_BACKEND = "mean"
DEFAULT_SCIENCE_HYBRID_TAU = 0.031746031746031744
DEFAULT_SCIENCE_HYBRID_K = 3
DEFAULT_SCIENCE_HYBRID_ALPHA = 0.50
DEFAULT_SCIENCE_HYBRID_M = 0.02
DEFAULT_SCIENCE_HYBRID_TEMPERATURE = 0.75


def default_gpqa_pairwise_model_path() -> Path:
    models_dir = Path(__file__).resolve().parent.parent.parent.parent / "models" / "ml_selectors"
    v1_path = models_dir / "gpqa_pairwise_v1.pkl"
    if v1_path.exists():
        return v1_path
    return models_dir / "gpqa_pairwise_round1.pkl"


@dataclass(frozen=True)
class ScienceHybridConfig:
    family: str = DEFAULT_SCIENCE_HYBRID_FAMILY
    backend: str = DEFAULT_SCIENCE_HYBRID_BACKEND
    tau: float = DEFAULT_SCIENCE_HYBRID_TAU
    k: int = DEFAULT_SCIENCE_HYBRID_K
    alpha: float = DEFAULT_SCIENCE_HYBRID_ALPHA
    m: float = DEFAULT_SCIENCE_HYBRID_M
    temperature: float = DEFAULT_SCIENCE_HYBRID_TEMPERATURE

    def validate(self) -> "ScienceHybridConfig":
        if self.family not in SCIENCE_HYBRID_FAMILIES:
            raise ValueError(f"Unknown science hybrid family: {self.family}")
        if self.backend not in SCIENCE_HYBRID_PAIRWISE_BACKENDS:
            raise ValueError(f"Unknown science hybrid pairwise backend: {self.backend}")
        if int(self.k) <= 0:
            raise ValueError("k must be positive")
        if not (0.0 <= float(self.alpha) <= 1.0):
            raise ValueError("alpha must be in [0, 1]")
        if float(self.temperature) <= 0.0:
            raise ValueError("temperature must be positive")
        return self

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScienceHybridDecision:
    config: ScienceHybridConfig
    baseline_order: np.ndarray
    pairwise_order: np.ndarray
    hybrid_order: np.ndarray
    baseline_scores: np.ndarray
    pairwise_scores: np.ndarray
    hybrid_scores: np.ndarray
    baseline_gap: float
    pairwise_margin_vs_baseline: float
    triggered: bool
    overridden: bool
    shortlist_indices: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.as_dict(),
            "baseline_order": self.baseline_order.tolist(),
            "pairwise_order": self.pairwise_order.tolist(),
            "hybrid_order": self.hybrid_order.tolist(),
            "baseline_scores": self.baseline_scores.tolist(),
            "pairwise_scores": self.pairwise_scores.tolist(),
            "hybrid_scores": self.hybrid_scores.tolist(),
            "baseline_gap": float(self.baseline_gap),
            "pairwise_margin_vs_baseline": float(self.pairwise_margin_vs_baseline),
            "triggered": bool(self.triggered),
            "overridden": bool(self.overridden),
            "shortlist_indices": self.shortlist_indices.tolist(),
        }


def _order_to_desc_scores(
    order: np.ndarray,
    n: int,
    *,
    hi: float = 1.0,
    lo: float = 0.0,
) -> np.ndarray:
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


def compute_pairwise_probability_matrix(
    scorer: GPQAPairwiseScorer,
    X: np.ndarray,
) -> np.ndarray:
    if scorer.pipeline is None:
        raise RuntimeError("GPQAPairwiseScorer.fit() must be called before probability extraction")
    X_arr = np.asarray(X, dtype=np.float64)
    n = int(X_arr.shape[0])
    probs = np.full((n, n), 0.5, dtype=np.float64)
    if n <= 1:
        return probs
    diffs = X_arr[:, None, :] - X_arr[None, :, :]
    mask = ~np.eye(n, dtype=bool)
    off_diag = diffs[mask]
    off_probs = scorer.pipeline.predict_proba(off_diag)[:, 1]
    probs[mask] = np.asarray(off_probs, dtype=np.float64)
    return probs


def compute_pairwise_backend_scores_from_matrix(
    prob_matrix: np.ndarray,
    *,
    backend: str = DEFAULT_SCIENCE_HYBRID_BACKEND,
    temperature: float = DEFAULT_SCIENCE_HYBRID_TEMPERATURE,
) -> np.ndarray:
    backend = str(backend)
    if backend not in SCIENCE_HYBRID_PAIRWISE_BACKENDS:
        raise ValueError(f"Unknown science hybrid pairwise backend: {backend}")

    probs = np.asarray(prob_matrix, dtype=np.float64)
    n = int(probs.shape[0])
    if n <= 1:
        return np.zeros(n, dtype=np.float64)

    out = np.zeros(n, dtype=np.float64)
    for idx in range(n):
        row = np.asarray(np.delete(probs[idx], idx), dtype=np.float64)
        if row.size == 0:
            out[idx] = 0.0
            continue
        if backend == "mean":
            out[idx] = float(np.mean(row))
        elif backend == "softmax_mean":
            centered = row - float(np.max(row))
            weights = np.exp(centered / max(float(temperature), 1e-6))
            out[idx] = float(np.average(row, weights=weights))
        elif backend == "win_count":
            out[idx] = float(np.mean(row > 0.5))
        elif backend == "copeland_margin":
            out[idx] = float(np.mean(np.where(row > 0.5, 1.0, np.where(row < 0.5, -1.0, 0.0))))
        else:
            raise ValueError(f"Unknown science hybrid pairwise backend: {backend}")
    return np.asarray(out, dtype=np.float64)


def compute_pairwise_backend_scores(
    scorer: GPQAPairwiseScorer,
    X: np.ndarray,
    *,
    backend: str = DEFAULT_SCIENCE_HYBRID_BACKEND,
    temperature: float = DEFAULT_SCIENCE_HYBRID_TEMPERATURE,
) -> tuple[np.ndarray, np.ndarray]:
    prob_matrix = compute_pairwise_probability_matrix(scorer, X)
    scores = compute_pairwise_backend_scores_from_matrix(
        prob_matrix,
        backend=backend,
        temperature=temperature,
    )
    return scores, prob_matrix


def shortlist_from_order(order: np.ndarray, k: int) -> np.ndarray:
    order_arr = np.asarray(order, dtype=np.int64)
    if order_arr.size <= 0:
        return np.zeros(0, dtype=np.int64)
    keep = max(1, min(int(k), int(order_arr.size)))
    return np.asarray(order_arr[:keep], dtype=np.int64)


def baseline_gap_from_scores(
    baseline_scores: np.ndarray,
    D: np.ndarray,
    *,
    run_ids: list[int] | None = None,
    baseline_gate_scores: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    order = order_code_dynamic_group_indices(
        baseline_scores,
        D,
        run_ids=run_ids,
    )
    if order.size <= 1:
        return 0.0, order
    gap_scores = np.asarray(
        baseline_scores if baseline_gate_scores is None else baseline_gate_scores,
        dtype=np.float64,
    )
    top1 = float(gap_scores[int(order[0])])
    top2 = float(gap_scores[int(order[1])])
    return float(top1 - top2), order


def compute_science_hybrid_decision(
    baseline_scores: np.ndarray,
    pairwise_prob_matrix: np.ndarray,
    D: np.ndarray,
    *,
    run_ids: list[int] | None = None,
    baseline_gate_scores: np.ndarray | None = None,
    config: ScienceHybridConfig | None = None,
) -> ScienceHybridDecision:
    cfg = (config or ScienceHybridConfig()).validate()

    baseline_scores_arr = np.asarray(baseline_scores, dtype=np.float64)
    pairwise_prob_matrix_arr = np.asarray(pairwise_prob_matrix, dtype=np.float64)
    n = int(baseline_scores_arr.size)

    baseline_gap, baseline_order = baseline_gap_from_scores(
        baseline_scores_arr,
        D,
        run_ids=run_ids,
        baseline_gate_scores=baseline_gate_scores,
    )
    pairwise_scores = compute_pairwise_backend_scores_from_matrix(
        pairwise_prob_matrix_arr,
        backend=cfg.backend,
        temperature=cfg.temperature,
    )
    pairwise_order = order_code_dynamic_group_indices(
        pairwise_scores,
        D,
        run_ids=run_ids,
    )
    hybrid_order = np.asarray(baseline_order, dtype=np.int64)
    triggered = False
    overridden = False
    shortlist = np.zeros(0, dtype=np.int64)

    if cfg.family == "margin_fallback":
        shortlist = shortlist_from_order(baseline_order, cfg.k)
        if float(baseline_gap) < float(cfg.tau) and shortlist.size > 1:
            triggered = True
            short_prob = pairwise_prob_matrix_arr[np.ix_(shortlist, shortlist)]
            short_scores = compute_pairwise_backend_scores_from_matrix(
                short_prob,
                backend=cfg.backend,
                temperature=cfg.temperature,
            )
            short_order = order_code_dynamic_group_indices(
                short_scores,
                D[np.ix_(shortlist, shortlist)],
                run_ids=None if run_ids is None else [int(run_ids[int(idx)]) for idx in shortlist.tolist()],
            )
            reordered = shortlist[short_order]
            mask = np.ones(n, dtype=bool)
            mask[reordered] = False
            rest = np.asarray([idx for idx in baseline_order.tolist() if mask[int(idx)]], dtype=np.int64)
            hybrid_order = np.concatenate([reordered, rest], axis=0)
    elif cfg.family == "shortlist_blend":
        shortlist = shortlist_from_order(baseline_order, cfg.k)
        if shortlist.size > 1:
            triggered = True
            baseline_short_scores = _order_to_desc_scores(np.arange(shortlist.size, dtype=np.int64), shortlist.size)
            short_prob = pairwise_prob_matrix_arr[np.ix_(shortlist, shortlist)]
            pairwise_short_scores = compute_pairwise_backend_scores_from_matrix(
                short_prob,
                backend=cfg.backend,
                temperature=cfg.temperature,
            )
            pairwise_short_order = order_code_dynamic_group_indices(
                pairwise_short_scores,
                D[np.ix_(shortlist, shortlist)],
                run_ids=None if run_ids is None else [int(run_ids[int(idx)]) for idx in shortlist.tolist()],
            )
            pairwise_short_rank = _order_to_desc_scores(pairwise_short_order, shortlist.size)
            blend_scores = float(cfg.alpha) * baseline_short_scores + (1.0 - float(cfg.alpha)) * pairwise_short_rank
            blend_order = order_code_dynamic_group_indices(
                blend_scores,
                D[np.ix_(shortlist, shortlist)],
                run_ids=None if run_ids is None else [int(run_ids[int(idx)]) for idx in shortlist.tolist()],
            )
            reordered = shortlist[blend_order]
            mask = np.ones(n, dtype=bool)
            mask[reordered] = False
            rest = np.asarray([idx for idx in baseline_order.tolist() if mask[int(idx)]], dtype=np.int64)
            hybrid_order = np.concatenate([reordered, rest], axis=0)
    elif cfg.family == "hard_override":
        pairwise_best_idx = int(pairwise_order[0]) if pairwise_order.size else 0
        baseline_best_idx = int(baseline_order[0]) if baseline_order.size else 0
        pairwise_margin = float(pairwise_scores[pairwise_best_idx] - pairwise_scores[baseline_best_idx])
        shortlist = np.asarray([pairwise_best_idx, baseline_best_idx], dtype=np.int64)
        if (
            pairwise_best_idx != baseline_best_idx
            and float(baseline_gap) < float(cfg.tau)
            and float(pairwise_margin) > float(cfg.m)
        ):
            triggered = True
            overridden = True
            hybrid_order = np.asarray(
                [pairwise_best_idx] + [idx for idx in baseline_order.tolist() if int(idx) != pairwise_best_idx],
                dtype=np.int64,
            )
    else:
        raise ValueError(f"Unknown science hybrid family: {cfg.family}")

    baseline_best = int(baseline_order[0]) if baseline_order.size else 0
    pairwise_best = int(pairwise_order[0]) if pairwise_order.size else 0
    pairwise_margin_vs_baseline = float(pairwise_scores[pairwise_best] - pairwise_scores[baseline_best])
    hybrid_scores = _order_to_desc_scores(hybrid_order, n)

    return ScienceHybridDecision(
        config=cfg,
        baseline_order=np.asarray(baseline_order, dtype=np.int64),
        pairwise_order=np.asarray(pairwise_order, dtype=np.int64),
        hybrid_order=np.asarray(hybrid_order, dtype=np.int64),
        baseline_scores=np.asarray(baseline_scores_arr, dtype=np.float64),
        pairwise_scores=np.asarray(pairwise_scores, dtype=np.float64),
        hybrid_scores=np.asarray(hybrid_scores, dtype=np.float64),
        baseline_gap=float(baseline_gap),
        pairwise_margin_vs_baseline=float(pairwise_margin_vs_baseline),
        triggered=bool(triggered),
        overridden=bool(overridden),
        shortlist_indices=np.asarray(shortlist, dtype=np.int64),
    )


def compute_science_hybrid_decision_from_feature_matrix(
    baseline_scores: np.ndarray,
    X: np.ndarray,
    scorer: GPQAPairwiseScorer,
    D: np.ndarray,
    *,
    run_ids: list[int] | None = None,
    baseline_gate_scores: np.ndarray | None = None,
    config: ScienceHybridConfig | None = None,
) -> ScienceHybridDecision:
    prob_matrix = compute_pairwise_probability_matrix(scorer, X)
    return compute_science_hybrid_decision(
        baseline_scores,
        prob_matrix,
        D,
        run_ids=run_ids,
        baseline_gate_scores=baseline_gate_scores,
        config=config,
    )


def compute_science_hybrid_decision_for_context(
    context: SelectorContext,
    D: np.ndarray,
    scorer: GPQAPairwiseScorer,
    *,
    config: ScienceHybridConfig | None = None,
) -> ScienceHybridDecision:
    baseline_scores, _, _ = compute_science_dynamic_primary_scores(
        context,
        weights=DEFAULT_SCIENCE_DYNAMIC_WEIGHTS,
    )
    raw = extract_gpqa_pairwise_raw(context)
    X = build_gpqa_pairwise_features_configurable(
        raw,
        include_margin=bool(getattr(scorer, "include_margin", False)),
        include_dominance=bool(getattr(scorer, "include_dominance", False)),
    )
    return compute_science_hybrid_decision_from_feature_matrix(
        baseline_scores,
        X,
        scorer,
        D,
        run_ids=list(map(int, context.run_ids)),
        baseline_gate_scores=np.asarray(raw["recency_conf_mean"], dtype=np.float64),
        config=config,
    )


def compute_science_baseline_scores_from_context(
    context: SelectorContext,
) -> np.ndarray:
    scores, _, _ = compute_science_dynamic_primary_scores(
        context,
        weights=DEFAULT_SCIENCE_DYNAMIC_WEIGHTS,
    )
    return np.asarray(scores, dtype=np.float64)


def compute_science_baseline_scores_from_raw(
    raw: dict[str, np.ndarray],
) -> np.ndarray:
    scores, _ = compute_science_dynamic_primary_scores_from_raw(
        raw,
        weights=DEFAULT_SCIENCE_DYNAMIC_WEIGHTS,
    )
    return np.asarray(scores, dtype=np.float64)
