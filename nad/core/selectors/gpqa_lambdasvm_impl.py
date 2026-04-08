from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import SelectorContext
from .gpqa_pairwise_impl import build_gpqa_pairwise_features, extract_gpqa_pairwise_raw
from .lambda_svm_core import LambdaSVMScorer

GPQA_LAMBDASVM_FEATURE_NAMES = [
    "dc_z",
    "dc_r",
    "reflection_count_r",
    "prefix_conf_mean_r",
    "recency_conf_mean_r",
    "late_recovery_r",
]


def default_gpqa_lambdasvm_model_path() -> Path:
    models_dir = Path(__file__).resolve().parent.parent.parent.parent / "models" / "ml_selectors"
    return models_dir / "gpqa_lambdasvm_round1.pkl"


def extract_gpqa_lambdasvm_raw(context: SelectorContext) -> dict[str, np.ndarray]:
    return extract_gpqa_pairwise_raw(context)


def build_gpqa_lambdasvm_features(raw: dict[str, np.ndarray]) -> np.ndarray:
    return build_gpqa_pairwise_features(raw)


class GPQALambdaSVMScorer(LambdaSVMScorer):
    def score_context(self, context: SelectorContext) -> np.ndarray:
        raw = extract_gpqa_lambdasvm_raw(context)
        X = build_gpqa_lambdasvm_features(raw)
        return self.score_group(X)

