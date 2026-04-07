"""
GPQA Group Pairwise Selector — Core Implementation

Bradley-Terry logistic regression on exhaustive pairwise feature differences.
This model is architecturally distinct from Extreme8/9/10 (tuple-sampling):
- Inference: scans all N*(N-1) ordered pairs deterministically (O(N²))
- Training: learns from (correct - wrong) feature differences

Feature vector (6-dim), all group-normalised before differencing:
  0  dc_z               z-score of DeepConf quality
  1  dc_r               rank01 of DeepConf quality
  2  reflection_count_r rank01 of reflection count
  3  prefix_conf_mean_r rank01 of prefix confidence mean
  4  recency_conf_mean_r rank01 of recency-weighted confidence mean
  5  late_recovery_r    rank01 of late-window recovery score
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .base import SelectorContext
from .extreme8_impl import extract_extreme8_raw_values
from .ml_features import _rank01, _zscore
from .science_dynamic_impl import extract_science_dynamic_raw_matrix

GPQA_PAIRWISE_FEATURE_NAMES: list[str] = [
    "dc_z",
    "dc_r",
    "reflection_count_r",
    "prefix_conf_mean_r",
    "recency_conf_mean_r",
    "late_recovery_r",
]
N_GPQA_PAIRWISE_FEATURES = len(GPQA_PAIRWISE_FEATURE_NAMES)

_DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "models"
    / "ml_selectors"
    / "gpqa_pairwise_round1.pkl"
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _safe_rank01(values: np.ndarray) -> np.ndarray:
    """rank01 with NaN imputation: fill non-finite with finite mean, or 0.5 if all NaN."""
    arr = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(arr)
    if not valid.any():
        return np.full(arr.shape, 0.5, dtype=np.float64)
    fill = float(np.mean(arr[valid]))
    filled = np.where(valid, arr, fill)
    return _rank01(filled)


def _stable_rank_feature(values: np.ndarray, *, higher_is_better: bool) -> np.ndarray:
    """Rank-normalize with neutral output for constant / degenerate arrays."""
    arr = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(arr)
    if not valid.any() or valid.sum() <= 1:
        return np.full(arr.shape, 0.5, dtype=np.float64)
    fill = float(np.mean(arr[valid]))
    filled = np.where(valid, arr, fill)
    if np.allclose(filled, filled[0], atol=1e-12, rtol=0.0):
        return np.full(arr.shape, 0.5, dtype=np.float64)
    if not higher_is_better:
        filled = -filled
    return np.asarray(_rank01(filled), dtype=np.float64)


# ── public API ────────────────────────────────────────────────────────────────

def extract_gpqa_pairwise_raw(context: SelectorContext) -> dict[str, np.ndarray]:
    """Extract raw per-run values needed by the GPQA pairwise model.

    Merges results from extract_extreme8_raw_values() and
    extract_science_dynamic_raw_matrix().  The returned dict contains all
    keys needed by build_gpqa_pairwise_features() plus late_worst_window
    (kept for downstream baseline evaluation, not used by the pairwise model).

    Returns:
        dict with keys:
            "dc_raw"           – (N,) DeepConf quality (imputed)
            "reflection_count" – (N,) reflection count
            "prefix_conf_mean" – (N,) prefix confidence mean (may contain NaN)
            "recency_conf_mean"– (N,) recency-weighted confidence mean
            "late_worst_window"– (N,) worst sliding-window mean in tail
            "late_recovery"    – (N,) final_window - worst_window
    """
    ext8 = extract_extreme8_raw_values(context)
    sci = extract_science_dynamic_raw_matrix(context)
    return {
        "dc_raw": ext8["dc_raw"],
        "reflection_count": ext8["reflection_count"],
        "prefix_conf_mean": sci["prefix_conf_mean"],
        "recency_conf_mean": sci["recency_conf_mean"],
        "late_worst_window": sci["late_worst_window"],
        "late_recovery": sci["late_recovery"],
    }


def build_gpqa_pairwise_margin_feature(recency_conf_mean: np.ndarray) -> np.ndarray:
    """Leader-margin feature from recency rank; smaller gap to leader is better."""
    recency_rank = _safe_rank01(np.asarray(recency_conf_mean, dtype=np.float64))
    margin = float(np.max(recency_rank)) - recency_rank
    return _stable_rank_feature(margin, higher_is_better=False)


def build_gpqa_pairwise_dominance_feature(recency_conf_mean: np.ndarray) -> np.ndarray:
    """Count strict recency wins vs other runs, then rank-normalize."""
    recency = np.asarray(recency_conf_mean, dtype=np.float64)
    dominance = (recency[:, None] > recency[None, :]).sum(axis=1).astype(np.float64)
    return _stable_rank_feature(dominance, higher_is_better=True)


def get_gpqa_pairwise_feature_names(
    *,
    include_margin: bool = False,
    include_dominance: bool = False,
) -> list[str]:
    names = list(GPQA_PAIRWISE_FEATURE_NAMES)
    if include_margin:
        names.append("recency_margin_r")
    if include_dominance:
        names.append("recency_dominance_r")
    return names


def build_gpqa_pairwise_features(raw: dict[str, np.ndarray]) -> np.ndarray:
    """Build the 6-dim group-normalised feature matrix from raw values.

    All features are computed within the group (group-normalised), so the
    model generalises across problems of different difficulty.

    Args:
        raw: dict returned by extract_gpqa_pairwise_raw().

    Returns:
        np.ndarray of shape (N, 6), dtype float64.
        Columns correspond to GPQA_PAIRWISE_FEATURE_NAMES.
    """
    return build_gpqa_pairwise_features_configurable(
        raw,
        include_margin=False,
        include_dominance=False,
    )


def build_gpqa_pairwise_features_configurable(
    raw: dict[str, np.ndarray],
    *,
    include_margin: bool = False,
    include_dominance: bool = False,
) -> np.ndarray:
    dc_raw = np.asarray(raw["dc_raw"], dtype=np.float64)
    reflection_count = np.asarray(raw["reflection_count"], dtype=np.float64)
    prefix_conf_mean = np.asarray(raw["prefix_conf_mean"], dtype=np.float64)
    recency_conf_mean = np.asarray(raw["recency_conf_mean"], dtype=np.float64)
    late_recovery = np.asarray(raw["late_recovery"], dtype=np.float64)

    cols: list[np.ndarray] = [
        _zscore(dc_raw),                    # 0  dc_z
        _rank01(dc_raw),                    # 1  dc_r
        _rank01(reflection_count),          # 2  reflection_count_r
        _safe_rank01(prefix_conf_mean),     # 3  prefix_conf_mean_r
        _safe_rank01(recency_conf_mean),    # 4  recency_conf_mean_r
        _safe_rank01(late_recovery),        # 5  late_recovery_r
    ]
    if include_margin:
        cols.append(build_gpqa_pairwise_margin_feature(recency_conf_mean))
    if include_dominance:
        cols.append(build_gpqa_pairwise_dominance_feature(recency_conf_mean))
    return np.column_stack(cols).astype(np.float64)


def build_pairwise_training_pairs(
    X: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build Bradley-Terry training pairs from group features and labels.

    For each (correct run i, wrong run j) pair, emits:
      X[i] - X[j]  →  label 1  (correct beats wrong)
      X[j] - X[i]  →  label 0  (wrong vs correct, anti-pair)

    Args:
        X:      (N, F) group feature matrix (group-normalised).
        labels: (N,) int array; > 0 means correct run.

    Returns:
        (X_pairs, y_pairs):
            X_pairs: (2 * n_pos * n_neg, F) float64
            y_pairs: (2 * n_pos * n_neg,) int32
        Both empty (shape (0, F) and (0,)) when no mixed-label pairs exist.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    n_feat = X.shape[1] if X.ndim == 2 else N_GPQA_PAIRWISE_FEATURES

    correct_idx = np.where(labels > 0)[0]
    wrong_idx = np.where(labels <= 0)[0]

    if correct_idx.size == 0 or wrong_idx.size == 0:
        return np.zeros((0, n_feat), dtype=np.float64), np.zeros(0, dtype=np.int32)

    X_correct = X[correct_idx]  # (n_pos, F)
    X_wrong = X[wrong_idx]      # (n_neg, F)
    n_pairs = correct_idx.size * wrong_idx.size

    # Vectorised: (n_pos, n_neg, F) then flatten to (n_pos*n_neg, F)
    diffs_pos = (X_correct[:, None, :] - X_wrong[None, :, :]).reshape(n_pairs, n_feat)
    diffs_neg = (X_wrong[None, :, :] - X_correct[:, None, :]).reshape(n_pairs, n_feat)

    X_pairs = np.concatenate([diffs_pos, diffs_neg], axis=0)
    y_pairs = np.concatenate([
        np.ones(n_pairs, dtype=np.int32),
        np.zeros(n_pairs, dtype=np.int32),
    ], axis=0)

    return X_pairs, y_pairs


class GPQAPairwiseScorer:
    """Bradley-Terry logistic regression scorer over full pairwise group differences.

    Training:
        scorer.fit(X_pairs, y_pairs)

    Inference (O(N²) deterministic, vs Extreme8's O(T·k) stochastic):
        scores = scorer.score_group(X)   # (N,) — argmax selects best run

    Persistence:
        scorer.save(path)
        scorer = GPQAPairwiseScorer.load(path)
    """

    def __init__(
        self,
        *,
        C: float = 1.0,
        include_margin: bool = False,
        include_dominance: bool = False,
    ) -> None:
        self.pipeline: Optional[object] = None
        self.C = float(C)
        self.include_margin = bool(include_margin)
        self.include_dominance = bool(include_dominance)

    def fit(self, X_pairs: np.ndarray, y_pairs: np.ndarray) -> None:
        """Fit StandardScaler + LogisticRegression on pairwise difference features.

        Args:
            X_pairs: (M, F) array of feature differences.
            y_pairs: (M,) binary labels (1 = first run wins, 0 = loses).
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=self.C, max_iter=1000, random_state=42)),
        ])
        self.pipeline.fit(
            np.asarray(X_pairs, dtype=np.float64),
            np.asarray(y_pairs, dtype=np.int32),
        )

    def score_group(self, X: np.ndarray) -> np.ndarray:
        """Score all runs in a group via exhaustive O(N²) pairwise scan.

        For each run i, computes mean P(i beats j) over all j ≠ i.

        Args:
            X: (N, F) group feature matrix (group-normalised).

        Returns:
            (N,) float64 — higher score → better run → select argmax.
        """
        if self.pipeline is None:
            raise RuntimeError(
                "GPQAPairwiseScorer.fit() must be called before score_group()"
            )

        X = np.asarray(X, dtype=np.float64)
        N = X.shape[0]

        if N <= 1:
            return np.zeros(N, dtype=np.float64)

        # (N, N, F) pairwise difference tensor
        diffs = X[:, None, :] - X[None, :, :]

        # Off-diagonal mask — row-major, so diffs[mask] has shape (N*(N-1), F)
        # Row i of the reshaped (N, N-1) block = P(i beats j) for all j != i
        mask = ~np.eye(N, dtype=bool)
        off_diag = diffs[mask]  # (N*(N-1), F)

        probs = self.pipeline.predict_proba(off_diag)[:, 1]  # (N*(N-1),)
        scores = probs.reshape(N, N - 1).mean(axis=1)        # (N,)

        return np.asarray(scores, dtype=np.float64)

    def save(self, path: str | Path) -> None:
        """Save scorer to disk via joblib."""
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "GPQAPairwiseScorer":
        """Load scorer from disk via joblib."""
        import joblib

        return joblib.load(path)
