from __future__ import annotations

from pathlib import Path

from .base import SelectorContext
from .code_deepsets_impl import (
    CODE_DEEPSETS_FEATURE_NAMES,
    build_code_deepsets_features,
    extract_code_deepsets_raw,
)
from .code_v2_impl import select_code_v2_best_index
from .set_transformer_lite_core import SetTransformerLiteConfig, SetTransformerLiteScorer


def default_code_set_transformer_lite_model_path() -> Path:
    models_dir = Path(__file__).resolve().parent.parent.parent.parent / "models" / "ml_selectors"
    return models_dir / "code_set_transformer_lite_round1.pkl"


class CodeSetTransformerLiteScorer(SetTransformerLiteScorer):
    def __init__(self, *, config: SetTransformerLiteConfig | None = None) -> None:
        super().__init__(config=config, feature_names=CODE_DEEPSETS_FEATURE_NAMES)

    def score_context(self, context: SelectorContext):
        raw = extract_code_deepsets_raw(context)
        X = build_code_deepsets_features(raw)
        return self.score_group(X)

    def select_best_index(self, context: SelectorContext, D):
        scores = self.score_context(context)
        return select_code_v2_best_index(scores, D, run_ids=context.run_ids)

    @classmethod
    def load(cls, path: str | Path) -> "CodeSetTransformerLiteScorer":
        loaded = SetTransformerLiteScorer.load(path)
        obj = cls(config=loaded.config)
        obj.model = loaded.model
        obj.feature_mean = loaded.feature_mean
        obj.feature_std = loaded.feature_std
        obj.training_summary = dict(loaded.training_summary)
        return obj
