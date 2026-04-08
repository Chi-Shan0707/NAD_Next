from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import SelectorContext
from .deepsets_core import DeepSetsConfig, DeepSetsScorer
from .math_svm_impl import augment_math_svm_features
from .ml_features import FEATURE_NAMES, extract_run_features

MATH_DEEPSETS_EXTRA_FEATURE_NAMES = [
    "rank_consensus_mean",
    "rank_consensus_std",
    "structural_rank_mean",
    "confidence_minus_structural",
    "length_minus_consensus",
    "agreement_score",
    "topness_fraction",
    "z_consensus_mean",
]

MATH_DEEPSETS_FEATURE_NAMES = list(FEATURE_NAMES) + list(MATH_DEEPSETS_EXTRA_FEATURE_NAMES)


def default_math_deepsets_model_path() -> Path:
    models_dir = Path(__file__).resolve().parent.parent.parent.parent / "models" / "ml_selectors"
    v1_path = models_dir / "math_deepsets_v1.pkl"
    if v1_path.exists():
        return v1_path
    return models_dir / "math_deepsets_round1.pkl"


def build_math_deepsets_features(
    D: np.ndarray,
    run_stats: dict,
    *,
    context: SelectorContext | None = None,
) -> np.ndarray:
    base = extract_run_features(D, run_stats, context=context)
    return augment_math_svm_features(base)


class MathDeepSetsScorer(DeepSetsScorer):
    def __init__(self, *, config: DeepSetsConfig | None = None) -> None:
        super().__init__(config=config, feature_names=MATH_DEEPSETS_FEATURE_NAMES)

    @classmethod
    def load(cls, path: str | Path) -> "MathDeepSetsScorer":
        loaded = DeepSetsScorer.load(path)
        obj = cls(config=loaded.config)
        obj.model = loaded.model
        obj.feature_mean = loaded.feature_mean
        obj.feature_std = loaded.feature_std
        obj.training_summary = dict(loaded.training_summary)
        return obj
