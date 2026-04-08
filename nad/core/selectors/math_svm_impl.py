from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import warnings

BASE_MATH_SVM_FEATURE_DIM = 12

MATH_SVM_FEATURE_FAMILIES: tuple[str, ...] = (
    "all",
    "rank_only",
    "z_only",
    "no_logs",
    "structural",
    "confidence",
    "distance_confidence",
    "copeland_confidence",
    "all_aug",
    "rank_aug",
    "consensus_aug",
)

_MATH_SVM_FEATURE_FAMILY_INDICES: dict[str, tuple[int, ...]] = {
    "all": tuple(range(12)),
    "rank_only": (1, 3, 5, 7, 9, 10, 11),
    "z_only": (0, 2, 4, 6, 8, 10, 11),
    "no_logs": tuple(range(10)),
    "structural": (0, 1, 2, 3, 4, 5, 8, 9, 10, 11),
    "confidence": (6, 7, 10, 11),
    "distance_confidence": (0, 1, 2, 3, 6, 7, 8, 9, 10, 11),
    "copeland_confidence": (6, 7, 8, 9, 10, 11),
    "all_aug": tuple(range(20)),
    "rank_aug": (1, 3, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19),
    "consensus_aug": (1, 3, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19),
}

MATH_RANKSVM_BACKENDS: tuple[str, ...] = (
    "utility",
    "mean_margin",
    "win_count",
)


def math_feature_family_indices(feature_family: str) -> tuple[int, ...]:
    family = str(feature_family)
    if family not in _MATH_SVM_FEATURE_FAMILY_INDICES:
        raise ValueError(f"Unknown math SVM feature family: {family}")
    return _MATH_SVM_FEATURE_FAMILY_INDICES[family]


def select_math_feature_family(
    X: np.ndarray,
    feature_family: str,
) -> np.ndarray:
    idx = math_feature_family_indices(feature_family)
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {X_arr.shape}")
    return np.asarray(X_arr[:, idx], dtype=np.float64)


def augment_math_svm_features(X: np.ndarray) -> np.ndarray:
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {X_arr.shape}")
    if X_arr.shape[1] < BASE_MATH_SVM_FEATURE_DIM:
        raise ValueError(
            f"Expected at least {BASE_MATH_SVM_FEATURE_DIM} base features, got {X_arr.shape[1]}"
        )

    rank_cols = X_arr[:, [1, 3, 5, 7, 9]]
    struct_rank = X_arr[:, [1, 3, 9]]
    rank_consensus_mean = np.mean(rank_cols, axis=1, keepdims=True)
    rank_consensus_std = np.std(rank_cols, axis=1, keepdims=True)
    structural_rank_mean = np.mean(struct_rank, axis=1, keepdims=True)
    confidence_minus_structural = (X_arr[:, [7]] - structural_rank_mean)
    length_minus_consensus = (X_arr[:, [5]] - rank_consensus_mean)
    agree_cols = X_arr[:, [1, 3, 7, 9]]
    agreement_score = (1.0 - np.std(agree_cols, axis=1, keepdims=True))
    topness_fraction = np.mean(rank_cols >= 0.75, axis=1, keepdims=True)
    z_consensus_mean = np.mean(X_arr[:, [0, 2, 4, 6, 8]], axis=1, keepdims=True)

    extra = np.concatenate(
        [
            rank_consensus_mean,
            rank_consensus_std,
            structural_rank_mean,
            confidence_minus_structural,
            length_minus_consensus,
            agreement_score,
            topness_fraction,
            z_consensus_mean,
        ],
        axis=1,
    )
    return np.concatenate([X_arr, np.asarray(extra, dtype=np.float64)], axis=1)


def build_math_pairwise_hinge_training_examples(
    X: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    X_arr = np.asarray(X, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int32)
    n_feat = X_arr.shape[1] if X_arr.ndim == 2 else 0

    correct_idx = np.where(labels_arr > 0)[0]
    wrong_idx = np.where(labels_arr <= 0)[0]

    if correct_idx.size == 0 or wrong_idx.size == 0:
        return np.zeros((0, n_feat), dtype=np.float64), np.zeros(0, dtype=np.int32)

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
    return np.asarray(X_pairs, dtype=np.float64), np.asarray(y_pairs, dtype=np.int32)


class MathLinearSVMScorer:
    def __init__(
        self,
        *,
        C: float = 1.0,
        loss: str = "squared_hinge",
        fit_intercept: bool = True,
        class_weight: str | None = None,
        dual: str | bool = "auto",
        max_iter: int = 100000,
        tol: float = 1e-4,
    ) -> None:
        self.pipeline: Optional[object] = None
        self.C = float(C)
        self.loss = str(loss)
        self.fit_intercept = bool(fit_intercept)
        self.class_weight = None if class_weight in (None, "", "none", "None") else str(class_weight)
        self.dual = dual
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.fit_warnings_: list[str] = []
        self.converged_: bool = True

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC

        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int32)
        if X_arr.ndim != 2:
            raise ValueError(f"X must have shape (N, F), got {X_arr.shape}")
        if y_arr.ndim != 1:
            raise ValueError(f"y must have shape (N,), got {y_arr.shape}")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(f"X/y length mismatch: {X_arr.shape[0]} vs {y_arr.shape[0]}")
        if y_arr.size <= 0:
            raise ValueError("MathLinearSVMScorer.fit() requires at least one sample")
        if np.unique(y_arr).size < 2:
            raise ValueError("MathLinearSVMScorer.fit() requires both positive and negative labels")
        if self.loss not in ("hinge", "squared_hinge"):
            raise ValueError(f"Unsupported LinearSVC loss: {self.loss}")

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            (
                "clf",
                LinearSVC(
                    C=self.C,
                    loss=self.loss,
                    penalty="l2",
                    fit_intercept=self.fit_intercept,
                    class_weight=self.class_weight,
                    dual=self.dual,
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

    def score_group(self, X: np.ndarray) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("MathLinearSVMScorer.fit() must be called before score_group()")
        scores = self.pipeline.decision_function(np.asarray(X, dtype=np.float64))
        return np.asarray(scores, dtype=np.float64).reshape(-1)

    def save(self, path: str | Path) -> None:
        import joblib

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, out_path)

    @classmethod
    def load(cls, path: str | Path) -> "MathLinearSVMScorer":
        import joblib

        return joblib.load(path)


class MathRankSVMScorer:
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

    def fit(self, X_pairs: np.ndarray, y_pairs: np.ndarray) -> None:
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
            raise ValueError("MathRankSVMScorer.fit() requires at least one training pair")
        if not np.all(np.isin(y_arr, (-1, 1))):
            raise ValueError("MathRankSVMScorer expects hinge labels in {-1, +1}")
        if self.loss not in ("hinge", "squared_hinge"):
            raise ValueError(f"Unsupported RankSVM loss: {self.loss}")
        if self.backend not in MATH_RANKSVM_BACKENDS:
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
            raise RuntimeError("MathRankSVMScorer.fit() must be called before margin extraction")

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
            raise RuntimeError("MathRankSVMScorer.fit() must be called before score_group()")

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
        raise ValueError(f"Unsupported RankSVM backend: {self.backend}")

    def save(self, path: str | Path) -> None:
        import joblib

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, out_path)

    @classmethod
    def load(cls, path: str | Path) -> "MathRankSVMScorer":
        import joblib

        return joblib.load(path)
