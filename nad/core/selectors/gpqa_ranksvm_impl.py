"""
GPQA RankSVM Selector — Core Implementation

This is a narrow GPQA-only baseline that keeps the existing 6-dim
group-normalised GPQA pairwise features but swaps the Bradley-Terry
logistic objective for a linear pairwise hinge objective.

Training:
  learn on mirrored pairwise feature differences:
    X[i] - X[j]  -> +1
    X[j] - X[i]  -> -1

Inference:
  score each run directly with a decomposable utility:
    u_i = w · x_i

The scorer intentionally uses:
  - StandardScaler(with_mean=False)
  - LinearSVC(loss="hinge", fit_intercept=False)

so that the pairwise margin remains exactly decomposable into per-run
utilities after fitting on pairwise differences.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import warnings

from .gpqa_pairwise_impl import build_pairwise_training_pairs

GPQA_RANKSVM_BACKENDS: tuple[str, ...] = (
    "utility",
    "mean_margin",
    "win_count",
)


def build_pairwise_hinge_training_examples(
    X: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build mirrored pairwise-difference examples for hinge-loss training.

    Reuses the existing GPQA pairwise data builder and converts the binary
    Bradley-Terry labels to signed hinge labels:
      1 -> +1
      0 -> -1
    """
    X_pairs, y_pairs = build_pairwise_training_pairs(X, labels)
    if y_pairs.size <= 0:
        return X_pairs, np.zeros(0, dtype=np.int32)
    y_signed = np.where(np.asarray(y_pairs, dtype=np.int32) > 0, 1, -1).astype(np.int32)
    return np.asarray(X_pairs, dtype=np.float64), y_signed


class GPQARankSVMScorer:
    """Linear pairwise-hinge scorer with direct per-run utility inference."""

    def __init__(
        self,
        *,
        C: float = 1.0,
        loss: str = "hinge",
        fit_intercept: bool = False,
        backend: str = "utility",
        dual: str | bool = "auto",
        max_iter: int = 100000,
        tol: float = 1e-4,
    ) -> None:
        self.pipeline: Optional[object] = None
        self.C = float(C)
        self.loss = str(loss)
        self.fit_intercept = bool(fit_intercept)
        self.backend = str(backend)
        self.dual = dual
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.fit_warnings_: list[str] = []
        self.converged_: bool = True

    def fit(self, X_pairs: np.ndarray, y_pairs: np.ndarray) -> None:
        """Fit StandardScaler(with_mean=False) + LinearSVC on pairwise diffs."""
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC

        X_arr = np.asarray(X_pairs, dtype=np.float64)
        y_arr = np.asarray(y_pairs, dtype=np.int32)

        if X_arr.ndim != 2:
            raise ValueError(f"X_pairs must have shape (M, F), got {X_arr.shape}")
        if y_arr.ndim != 1:
            raise ValueError(f"y_pairs must have shape (M,), got {y_arr.shape}")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"X_pairs/y_pairs length mismatch: {X_arr.shape[0]} vs {y_arr.shape[0]}"
            )
        if y_arr.size <= 0:
            raise ValueError("GPQARankSVMScorer.fit() requires at least one training pair")
        if not np.all(np.isin(y_arr, (-1, 1))):
            raise ValueError("GPQARankSVMScorer expects hinge labels in {-1, +1}")
        if self.loss not in ("hinge", "squared_hinge"):
            raise ValueError(f"Unsupported RankSVM loss: {self.loss}")
        if self.backend not in GPQA_RANKSVM_BACKENDS:
            raise ValueError(f"Unsupported RankSVM backend: {self.backend}")
        if self.backend == "utility" and self.fit_intercept:
            raise ValueError("backend='utility' requires fit_intercept=False")

        self.pipeline = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            (
                "clf",
                LinearSVC(
                    C=self.C,
                    loss=self.loss,
                    penalty="l2",
                    dual=self.dual,
                    fit_intercept=self.fit_intercept,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    random_state=42,
                ),
            ),
        ])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            self.pipeline.fit(X_arr, y_arr)
        self.fit_warnings_ = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
        self.converged_ = len(self.fit_warnings_) == 0

    def pairwise_margin_matrix(self, X: np.ndarray) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError(
                "GPQARankSVMScorer.fit() must be called before pairwise margin extraction"
            )

        X_arr = np.asarray(X, dtype=np.float64)
        n = int(X_arr.shape[0])
        margins = np.zeros((n, n), dtype=np.float64)
        if n <= 1:
            return margins

        diffs = X_arr[:, None, :] - X_arr[None, :, :]
        mask = ~np.eye(n, dtype=bool)
        off_diag = diffs[mask]
        off_margins = self.pipeline.decision_function(off_diag)
        margins[mask] = np.asarray(off_margins, dtype=np.float64)
        return margins

    def _pairwise_margin_matrix(self, X: np.ndarray) -> np.ndarray:
        return self.pairwise_margin_matrix(X)

    def score_group(self, X: np.ndarray) -> np.ndarray:
        """Score all runs in a group via the configured RankSVM backend."""
        if self.pipeline is None:
            raise RuntimeError(
                "GPQARankSVMScorer.fit() must be called before score_group()"
            )

        X_arr = np.asarray(X, dtype=np.float64)
        n = int(X_arr.shape[0])
        if n <= 1:
            return np.zeros(n, dtype=np.float64)

        if self.backend == "utility":
            clf = self.pipeline.named_steps["clf"]
            intercept = np.asarray(getattr(clf, "intercept_", np.zeros(1)), dtype=np.float64)
            if intercept.size > 0 and not np.allclose(intercept, 0.0, atol=1e-12, rtol=0.0):
                raise RuntimeError(
                    "backend='utility' requires a zero-intercept classifier"
                )
            scores = self.pipeline.decision_function(X_arr)
            return np.asarray(scores, dtype=np.float64).reshape(n)

        margins = self.pairwise_margin_matrix(X_arr)
        off_diag = margins[~np.eye(n, dtype=bool)].reshape(n, n - 1)

        if self.backend == "mean_margin":
            return np.asarray(off_diag.mean(axis=1), dtype=np.float64)
        if self.backend == "win_count":
            wins = (off_diag > 0.0).astype(np.float64)
            ties = (off_diag == 0.0).astype(np.float64)
            return np.asarray(wins.mean(axis=1) + 0.5 * ties.mean(axis=1), dtype=np.float64)

        raise ValueError(f"Unsupported RankSVM backend: {self.backend}")

    def save(self, path: str | Path) -> None:
        """Save scorer to disk via joblib."""
        import joblib

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, out_path)

    @classmethod
    def load(cls, path: str | Path) -> "GPQARankSVMScorer":
        """Load scorer from disk via joblib."""
        import joblib

        return joblib.load(path)
