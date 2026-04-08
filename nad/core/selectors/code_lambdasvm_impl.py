from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from .base import SelectorContext
from .code_v2_impl import (
    CODE_V2_FEATURE_NAMES,
    build_code_v2_rank_features_from_raw,
    extract_code_v2_raw_matrix,
    select_code_v2_best_index,
)
from .lambda_svm_core import LambdaSVMScorer

CODE_LAMBDASVM_FEATURE_NAMES = list(CODE_V2_FEATURE_NAMES)


def default_code_lambdasvm_model_path() -> Path:
    models_dir = Path(__file__).resolve().parent.parent.parent.parent / "models" / "ml_selectors"
    return models_dir / "code_lambdasvm_round1.pkl"


def extract_code_lambdasvm_raw(context: SelectorContext) -> dict[str, np.ndarray]:
    return extract_code_v2_raw_matrix(context)


def build_code_lambdasvm_features(raw: dict[str, np.ndarray]) -> np.ndarray:
    return build_code_v2_rank_features_from_raw(raw)


class CodeLambdaSVMScorer(LambdaSVMScorer):
    def score_context(self, context: SelectorContext) -> np.ndarray:
        raw = extract_code_lambdasvm_raw(context)
        X = build_code_lambdasvm_features(raw)
        return self.score_group(X)

    def select_best_index(self, context: SelectorContext, D: np.ndarray, *, run_ids: Iterable[int] | None = None) -> int:
        scores = self.score_context(context)
        return select_code_v2_best_index(scores, D, run_ids=run_ids or context.run_ids)

