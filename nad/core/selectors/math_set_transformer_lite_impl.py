from __future__ import annotations

from pathlib import Path

from .base import SelectorContext
from .math_deepsets_impl import MATH_DEEPSETS_FEATURE_NAMES, build_math_deepsets_features
from .set_transformer_lite_core import SetTransformerLiteConfig, SetTransformerLiteScorer


def default_math_set_transformer_lite_model_path() -> Path:
    models_dir = Path(__file__).resolve().parent.parent.parent.parent / "models" / "ml_selectors"
    return models_dir / "math_set_transformer_lite_round1.pkl"


class MathSetTransformerLiteScorer(SetTransformerLiteScorer):
    def __init__(self, *, config: SetTransformerLiteConfig | None = None) -> None:
        super().__init__(config=config, feature_names=MATH_DEEPSETS_FEATURE_NAMES)

    def score_context(self, D, run_stats: dict, *, context: SelectorContext | None = None):
        X = build_math_deepsets_features(D, run_stats, context=context)
        return self.score_group(X)

    @classmethod
    def load(cls, path: str | Path) -> "MathSetTransformerLiteScorer":
        loaded = SetTransformerLiteScorer.load(path)
        obj = cls(config=loaded.config)
        obj.model = loaded.model
        obj.feature_mean = loaded.feature_mean
        obj.feature_std = loaded.feature_std
        obj.training_summary = dict(loaded.training_summary)
        return obj

