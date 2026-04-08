from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import SelectorContext
from .code_v2_impl import (
    CODE_V2_FEATURE_NAMES,
    build_code_v2_rank_features_from_raw,
    extract_code_v2_raw_matrix,
    select_code_v2_best_index,
)
from .deepsets_core import DeepSetsConfig, DeepSetsScorer

CODE_DEEPSETS_FEATURE_NAMES = list(CODE_V2_FEATURE_NAMES)


def default_code_deepsets_model_path() -> Path:
    models_dir = Path(__file__).resolve().parent.parent.parent.parent / "models" / "ml_selectors"
    v1_path = models_dir / "code_deepsets_v1.pkl"
    if v1_path.exists():
        return v1_path
    return models_dir / "code_deepsets_round1.pkl"


def extract_code_deepsets_raw(context: SelectorContext) -> dict[str, np.ndarray]:
    return extract_code_v2_raw_matrix(context)


def build_code_deepsets_features(raw: dict[str, np.ndarray]) -> np.ndarray:
    return build_code_v2_rank_features_from_raw(raw)


class CodeDeepSetsScorer(DeepSetsScorer):
    def __init__(self, *, config: DeepSetsConfig | None = None) -> None:
        super().__init__(config=config, feature_names=CODE_DEEPSETS_FEATURE_NAMES)

    def score_context(self, context: SelectorContext) -> np.ndarray:
        raw = extract_code_deepsets_raw(context)
        X = build_code_deepsets_features(raw)
        return self.score_group(X)

    def select_best_index(self, context: SelectorContext, D: np.ndarray) -> int:
        scores = self.score_context(context)
        return select_code_v2_best_index(scores, D, run_ids=context.run_ids)

    @classmethod
    def load(cls, path: str | Path) -> "CodeDeepSetsScorer":
        loaded = DeepSetsScorer.load(path)
        obj = cls(config=loaded.config)
        obj.model = loaded.model
        obj.feature_mean = loaded.feature_mean
        obj.feature_std = loaded.feature_std
        obj.training_summary = dict(loaded.training_summary)
        return obj
