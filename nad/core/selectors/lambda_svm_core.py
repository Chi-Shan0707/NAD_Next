from __future__ import annotations

from pathlib import Path
from typing import Optional
import warnings

import numpy as np

LAMBDA_SVM_BACKENDS: tuple[str, ...] = (
    "utility",
    "mean_margin",
    "win_count",
)

PAIR_WEIGHT_MODES: tuple[str, ...] = (
    "uniform",
    "dcg_delta",
)


def _rank_positions_from_scores(reference_scores: np.ndarray | None, labels: np.ndarray) -> np.ndarray:
    labels_arr = np.asarray(labels, dtype=np.int32).reshape(-1)
    n = int(labels_arr.shape[0])
    if reference_scores is None:
        order = np.argsort(-labels_arr, kind="mergesort")
    else:
        scores = np.asarray(reference_scores, dtype=np.float64).reshape(-1)
        if scores.shape[0] != n:
            raise ValueError(f"reference_scores length mismatch: expected {n}, got {scores.shape[0]}")
        order = np.lexsort((np.arange(n, dtype=np.int32), -scores))
    positions = np.empty(n, dtype=np.int32)
    positions[order] = np.arange(n, dtype=np.int32)
    return positions


def build_weighted_pairwise_training_examples(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    reference_scores: np.ndarray | None = None,
    pair_weight_mode: str = "uniform",
    min_weight: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_arr = np.asarray(X, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int32).reshape(-1)
    n_feat = X_arr.shape[1] if X_arr.ndim == 2 else 0

    if X_arr.ndim != 2:
        raise ValueError(f"X must have shape (N, F), got {X_arr.shape}")
    if labels_arr.ndim != 1:
        raise ValueError(f"labels must have shape (N,), got {labels_arr.shape}")
    if X_arr.shape[0] != labels_arr.shape[0]:
        raise ValueError(f"X/labels length mismatch: {X_arr.shape[0]} vs {labels_arr.shape[0]}")

    correct_idx = np.where(labels_arr > 0)[0]
    wrong_idx = np.where(labels_arr <= 0)[0]
    if correct_idx.size == 0 or wrong_idx.size == 0:
        return (
            np.zeros((0, n_feat), dtype=np.float64),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.float64),
        )

    X_correct = X_arr[correct_idx]
    X_wrong = X_arr[wrong_idx]
    n_pairs = int(correct_idx.size * wrong_idx.size)

    diffs_pos = (X_correct[:, None, :] - X_wrong[None, :, :]).reshape(n_pairs, n_feat)
    diffs_neg = (X_wrong[None, :, :] - X_correct[:, None, :]).reshape(n_pairs, n_feat)
    X_pairs = np.concatenate([diffs_pos, diffs_neg], axis=0)
    y_pairs = np.concatenate([
        np.ones(n_pairs, dtype=np.int32),
        -np.ones(n_pairs, dtype=np.int32),
    ], axis=0)

    mode = str(pair_weight_mode)
    if mode not in PAIR_WEIGHT_MODES:
        raise ValueError(f"Unsupported pair_weight_mode: {mode}")
    if mode == "uniform":
        base_weights = np.ones(n_pairs, dtype=np.float64)
    else:
        positions = _rank_positions_from_scores(reference_scores, labels_arr)
        discounts = 1.0 / np.log2(positions.astype(np.float64) + 2.0)
        delta = np.abs(discounts[correct_idx][:, None] - discounts[wrong_idx][None, :]).reshape(-1)
        base_weights = np.maximum(delta, float(min_weight)).astype(np.float64, copy=False)
        weight_mean = float(np.mean(base_weights))
        if weight_mean > 0.0:
            base_weights = base_weights / weight_mean

    sample_weight = np.concatenate([base_weights, base_weights], axis=0)
    return (
        np.asarray(X_pairs, dtype=np.float64),
        np.asarray(y_pairs, dtype=np.int32),
        np.asarray(sample_weight, dtype=np.float64),
    )


class LambdaSVMScorer:
    def __init__(
        self,
        *,
        C: float = 1.0,
        loss: str = "squared_hinge",
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

    def fit(
        self,
        X_pairs: np.ndarray,
        y_pairs: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> None:
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
            raise ValueError(f"X_pairs/y_pairs length mismatch: {X_arr.shape[0]} vs {y_arr.shape[0]}")
        if y_arr.size <= 0:
            raise ValueError("LambdaSVMScorer.fit() requires at least one training pair")
        if not np.all(np.isin(y_arr, (-1, 1))):
            raise ValueError("LambdaSVMScorer expects hinge labels in {-1, +1}")
        if self.loss not in ("hinge", "squared_hinge"):
            raise ValueError(f"Unsupported LambdaSVM loss: {self.loss}")
        if self.backend not in LAMBDA_SVM_BACKENDS:
            raise ValueError(f"Unsupported LambdaSVM backend: {self.backend}")
        if self.backend == "utility" and self.fit_intercept:
            raise ValueError("backend='utility' requires fit_intercept=False")

        fit_kwargs: dict[str, np.ndarray] = {}
        if sample_weight is not None:
            weight_arr = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
            if weight_arr.shape[0] != X_arr.shape[0]:
                raise ValueError(
                    f"sample_weight length mismatch: expected {X_arr.shape[0]}, got {weight_arr.shape[0]}"
                )
            fit_kwargs["clf__sample_weight"] = weight_arr

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
            self.pipeline.fit(X_arr, y_arr, **fit_kwargs)
        self.fit_warnings_ = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
        self.converged_ = len(self.fit_warnings_) == 0

    def pairwise_margin_matrix(self, X: np.ndarray) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("LambdaSVMScorer.fit() must be called before pairwise margin extraction")

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

    def score_group(self, X: np.ndarray) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("LambdaSVMScorer.fit() must be called before score_group()")

        X_arr = np.asarray(X, dtype=np.float64)
        n = int(X_arr.shape[0])
        if n <= 1:
            return np.zeros(n, dtype=np.float64)

        if self.backend == "utility":
            clf = self.pipeline.named_steps["clf"]
            intercept = np.asarray(getattr(clf, "intercept_", np.zeros(1)), dtype=np.float64)
            if intercept.size > 0 and not np.allclose(intercept, 0.0, atol=1e-12, rtol=0.0):
                raise RuntimeError("backend='utility' requires a zero-intercept classifier")
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
        raise ValueError(f"Unsupported LambdaSVM backend: {self.backend}")

    def save(self, path: str | Path) -> None:
        import joblib

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, out_path)

    @classmethod
    def load(cls, path: str | Path) -> "LambdaSVMScorer":
        import joblib

        return joblib.load(path)
