from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import SelectorContext
from .lambda_svm_core import LambdaSVMScorer
from .math_svm_impl import MATH_SVM_FEATURE_FAMILIES, augment_math_svm_features, select_math_feature_family
from .ml_features import FEATURE_NAMES, extract_run_features

MATH_LAMBDASVM_EXTRA_FEATURE_NAMES = [
    "rank_consensus_mean",
    "rank_consensus_std",
    "structural_rank_mean",
    "confidence_minus_structural",
    "length_minus_consensus",
    "agreement_score",
    "topness_fraction",
    "z_consensus_mean",
]
MATH_LAMBDASVM_FEATURE_NAMES = list(FEATURE_NAMES) + list(MATH_LAMBDASVM_EXTRA_FEATURE_NAMES)


def default_math_lambdasvm_model_path() -> Path:
    models_dir = Path(__file__).resolve().parent.parent.parent.parent / "models" / "ml_selectors"
    return models_dir / "math_lambdasvm_round1.pkl"


def build_math_lambdasvm_features(
    D: np.ndarray,
    run_stats: dict,
    *,
    context: SelectorContext | None = None,
) -> np.ndarray:
    base = extract_run_features(D, run_stats, context=context)
    return augment_math_svm_features(base)


class MathLambdaSVMScorer(LambdaSVMScorer):
    def __init__(self, *, feature_family: str = "all_aug", **kwargs: object) -> None:
        if str(feature_family) not in MATH_SVM_FEATURE_FAMILIES:
            raise ValueError(f"Unsupported math feature_family: {feature_family}")
        super().__init__(**kwargs)
        self.feature_family = str(feature_family)

    def score_context(
        self,
        D: np.ndarray,
        run_stats: dict,
        *,
        context: SelectorContext | None = None,
    ) -> np.ndarray:
        X_full = build_math_lambdasvm_features(D, run_stats, context=context)
        X = select_math_feature_family(X_full, self.feature_family)
        return self.score_group(X)

